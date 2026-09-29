"""
Background task manager for ingestion operations.

Provides ``TaskManager`` — an in-memory, single-process task runner that
executes ingestion pipelines in background threads.  Designed for a
single-user portfolio project running on a GTX 1650 (4 GB VRAM):

    - **One GPU task at a time** — a ``threading.Semaphore(1)`` gates
      execution; additional tasks queue in FIFO order.
    - **Cancel via ``threading.Event``** — checked between pipeline steps;
      partial data is rolled back on cancellation.
    - **Task cleanup** — completed/failed/cancelled tasks are persisted to
      SQLite task history and pruned from memory after 24 hours by a
      background timer; lookups fall back to the history.
    - **Progress callback** — the pipeline's ``progress_callback`` feeds
      directly into the task's ``TaskProgress`` snapshot.

No Redis, no Celery — task state lives in a plain ``dict``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import (
    DatabaseError,
    FetchError,
    SECSemanticSearchError,
    get_logger,
)
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry, delete_filings_batch
from sec_semantic_search.ingest import (
    STEP_TOTAL,
    IngestCancelled,
    IngestObserver,
    plan_work,
    run_ingest,
)
from sec_semantic_search.pipeline import PipelineOrchestrator, ProcessedFiling
from sec_semantic_search.pipeline.fetch import FilingFetcher, FilingInfo

logger = get_logger(__name__)

_T = TypeVar("_T")

# In-memory TTL — tasks are persisted to SQLite before pruning.
_TASK_TTL_SECONDS = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------


class TaskState(StrEnum):
    """Lifecycle states for an ingestion task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Mutable progress snapshot updated by the worker thread."""

    current_ticker: str | None = None
    current_form_type: str | None = None
    step_label: str = ""
    step_index: int = 0
    step_total: int = 5
    filings_done: int = 0
    filings_total: int = 0
    filings_skipped: int = 0
    filings_failed: int = 0


@dataclass
class FilingResult:
    """Per-filing outcome stored after a successful ingest."""

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    segment_count: int
    chunk_count: int
    duration_seconds: float

    def to_dict(self) -> dict:
        """Serialise to a dict for WebSocket messages."""
        return {
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "segments": self.segment_count,
            "chunks": self.chunk_count,
            "time": round(self.duration_seconds, 1),
        }

    def to_history_dict(self) -> dict:
        """Serialise to a dict for task history persistence."""
        return {
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "segment_count": self.segment_count,
            "chunk_count": self.chunk_count,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TaskInfo:
    """
    Full state for a single ingestion task.

    Mutated by the worker thread; read by route handlers and WebSocket.
    Access to individual scalar/list fields is inherently thread-safe in
    CPython (GIL), but we avoid structural mutations to ``results`` from
    multiple threads.
    """

    task_id: str
    tickers: list[str]
    form_types: list[str]
    count_mode: str = "latest"
    count: int | None = None
    year: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    state: TaskState = TaskState.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    results: list[FilingResult] = field(default_factory=list)
    error: str | None = None

    # Per-session EDGAR credentials (name, email) — set by the route
    # handler when the user provides credentials via HTTP headers.
    # Never logged or persisted.
    edgar_name: str | None = None
    edgar_email: str | None = None

    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # GPU time limit timer — set by TaskManager when
    # MAX_TASK_DURATION_MINUTES > 0.  Cancelled on normal completion.
    _duration_timer: threading.Timer | None = field(default=None, repr=False)

    # Accession numbers this task has stored — all rolled back if the task
    # is cancelled (AD#16).
    _stored_accessions: list[str] = field(default_factory=list)

    # WebSocket message queue — worker thread pushes typed dicts via
    # call_soon_threadsafe; WebSocket handler awaits them directly.
    _message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Sequence number of the last pushed message. A counter update and the
    # message reporting it happen under ``_event_lock``, and the WebSocket
    # snapshot is built under it too, so ``snapshot.seq`` says exactly
    # which queued messages the snapshot already reflects (a reconnecting
    # client must not count them twice).
    _seq: int = field(default=0, repr=False)
    _event_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


# ---------------------------------------------------------------------------
# Ingest events → task state and WebSocket messages
# ---------------------------------------------------------------------------


class _TaskObserver(IngestObserver):
    """Feeds the shared ingest loop's events into one task."""

    def __init__(self, manager: TaskManager, info: TaskInfo) -> None:
        self._manager = manager
        self._info = info

    def listing(self, ticker: str, form_type: str | None) -> None:
        progress = self._info.progress
        progress.current_ticker = ticker
        if form_type is not None:
            progress.current_form_type = form_type
        progress.step_label = "Fetching"
        progress.step_index = 0

    def listing_failed(self, ticker: str, form_type: str, error: FetchError) -> None:
        logger.warning(
            "Task %s: fetch failed for %s %s — %s",
            self._info.task_id[:8],
            ticker,
            form_type,
            error.message,
        )

    def filing_started(self, position: int, total: int, filing: FilingInfo) -> None:
        progress = self._info.progress
        progress.current_ticker = filing.ticker
        progress.current_form_type = filing.form_type
        progress.step_label = "Checking duplicate"

    def step(self, filing: FilingInfo, label: str, index: int) -> None:
        info = self._info
        with info._event_lock:
            info.progress.current_ticker = filing.ticker
            info.progress.current_form_type = filing.form_type
            info.progress.step_label = label
            info.progress.step_index = index
            info.progress.step_total = STEP_TOTAL
            self._manager._push(
                info,
                {
                    "type": "step",
                    "ticker": filing.ticker,
                    "form_type": filing.form_type,
                    "step": label,
                    "step_number": index,
                    "total_steps": STEP_TOTAL,
                },
            )

    def skipped(self, filing: FilingInfo, reason: str) -> None:
        self._manager._record_outcome(
            self._info,
            {
                "type": "filing_skipped",
                "ticker": filing.ticker,
                "form_type": filing.form_type,
                "accession_number": filing.accession_number,
                "reason": reason,
            },
        )
        logger.info(
            "Task %s: skipped duplicate %s", self._info.task_id[:8], filing.accession_number
        )

    def failed(self, filing: FilingInfo, stage: str, error: SECSemanticSearchError) -> None:
        self._manager._record_outcome(
            self._info,
            {
                "type": "filing_failed",
                "ticker": filing.ticker,
                "form_type": filing.form_type,
                "accession_number": filing.accession_number,
                "error": error.message,
            },
        )
        logger.warning(
            "Task %s: %s failed for %s — %s",
            self._info.task_id[:8],
            stage,
            filing.accession_number,
            error.message,
        )

    def done(self, filing: FilingInfo, result: ProcessedFiling) -> None:
        info = self._info
        filing_id = result.filing_id
        stats = result.ingest_result
        # Recorded first: a cancel from here on rolls this filing back.
        info._stored_accessions.append(filing_id.accession_number)
        self._manager._record_outcome(
            info,
            {
                "type": "filing_done",
                "ticker": filing_id.ticker,
                "form_type": filing_id.form_type,
                "filing_date": filing_id.date_str,
                "accession_number": filing_id.accession_number,
                "segments": stats.segment_count,
                "chunks": stats.chunk_count,
                "time": round(stats.duration_seconds, 1),
            },
            FilingResult(
                ticker=filing_id.ticker,
                form_type=filing_id.form_type,
                filing_date=filing_id.date_str,
                accession_number=filing_id.accession_number,
                segment_count=stats.segment_count,
                chunk_count=stats.chunk_count,
                duration_seconds=stats.duration_seconds,
            ),
        )
        logger.info(
            "Task %s: ingested %s %s (%s) — %d chunks in %.1fs",
            info.task_id[:8],
            filing_id.ticker,
            filing_id.form_type,
            filing_id.date_str,
            stats.chunk_count,
            stats.duration_seconds,
        )


# ---------------------------------------------------------------------------
# Task manager
# ---------------------------------------------------------------------------


class TaskManager:
    """
    In-memory manager for background ingestion tasks.

    Usage (from route handlers)::

        manager = TaskManager(registry, chroma, fetcher, orchestrator)
        task_id = manager.create_task(request)
        info = manager.get_task(task_id)
        manager.cancel_task(task_id)

    The manager is stored on ``app.state`` (singleton per process).
    """

    def __init__(
        self,
        registry: MetadataRegistry,
        chroma: ChromaDBClient,
        fetcher: FilingFetcher,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        self._registry = registry
        self._chroma = chroma
        self._fetcher = fetcher
        self._orchestrator = orchestrator

        self._tasks: dict[str, TaskInfo] = {}
        self._gpu_semaphore = threading.Semaphore(1)
        self._edgar_lock = threading.Lock()
        self._lock = threading.Lock()  # protects _tasks dict mutations

        # Event loop reference — set via set_event_loop() during lifespan
        # startup.  Used by _push() to bridge sync worker → async queue.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Cleanup timer reference — stored so it can be cancelled on shutdown.
        self._cleanup_timer: threading.Timer | None = None
        self._shutdown_event = threading.Event()

        # Start the cleanup timer.
        self._start_cleanup_timer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_task(
        self,
        *,
        tickers: list[str],
        form_types: list[str],
        count_mode: str = "latest",
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        edgar_name: str | None = None,
        edgar_email: str | None = None,
    ) -> str:
        """
        Create a new ingestion task and start it in a background thread.

        Returns the task ID (UUID4 hex string).

        Raises:
            TaskQueueFullError: If the active task queue is at capacity.
        """
        # Guard against unbounded task queue (GPU semaphore starvation).
        max_active = get_settings().api.max_task_queue_size
        with self._lock:
            active_count = sum(
                1 for t in self._tasks.values() if t.state in (TaskState.PENDING, TaskState.RUNNING)
            )
            if active_count >= max_active:
                raise TaskQueueFullError(
                    f"Task queue is full ({active_count} active). "
                    "Wait for existing tasks to complete before submitting new ones."
                )

        task_id = uuid.uuid4().hex
        info = TaskInfo(
            task_id=task_id,
            tickers=tickers,
            form_types=form_types,
            count_mode=count_mode,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
            edgar_name=edgar_name,
            edgar_email=edgar_email,
        )

        with self._lock:
            self._tasks[task_id] = info

        thread = threading.Thread(
            target=self._run_task,
            args=(info,),
            name=f"ingest-{task_id[:8]}",
            daemon=True,
        )
        thread.start()

        logger.info(
            "Created task %s: tickers=%s, forms=%s, mode=%s",
            task_id[:8],
            tickers,
            form_types,
            count_mode,
        )
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Return task info, falling back to SQLite history if pruned."""
        info = self._tasks.get(task_id)
        if info is not None:
            return info

        # Check persisted history.
        try:
            history = self._registry.get_task_history(task_id)
        except Exception:
            logger.debug("Task history lookup failed for %s", task_id[:8])
            return None

        if history is None:
            return None

        return self._reconstruct_task_info(history)

    @staticmethod
    def _reconstruct_task_info(history: dict) -> TaskInfo:
        """Build a read-only ``TaskInfo`` from persisted history data."""
        info = TaskInfo(
            task_id=history["task_id"],
            tickers=history["tickers"],
            form_types=history["form_types"],
        )
        info.state = TaskState(history["status"])
        info.error = history["error"]
        info.progress = TaskProgress(
            filings_done=history["filings_done"],
            filings_skipped=history["filings_skipped"],
            filings_failed=history["filings_failed"],
        )
        info.results = [
            FilingResult(
                ticker=r["ticker"],
                form_type=r["form_type"],
                filing_date=r["filing_date"],
                accession_number=r["accession_number"],
                segment_count=r["segment_count"],
                chunk_count=r["chunk_count"],
                duration_seconds=r["duration_seconds"],
            )
            for r in history["results"]
        ]
        if history["started_at"]:
            info.started_at = datetime.fromisoformat(history["started_at"])
        if history["completed_at"]:
            info.completed_at = datetime.fromisoformat(history["completed_at"])
        return info

    def list_tasks(self) -> list[TaskInfo]:
        """Return all tasks (active and recent)."""
        return list(self._tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """
        Request cancellation of a running or pending task.

        Returns True if the cancel signal was sent, False if the task
        was not found or already finished.
        """
        info = self._tasks.get(task_id)
        if info is None:
            return False
        if info.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            return False
        info.cancel_event.set()
        logger.info("Cancel requested for task %s", task_id[:8])
        return True

    def has_active_task(self) -> bool:
        """Return True if any task is pending or running."""
        return any(t.state in (TaskState.PENDING, TaskState.RUNNING) for t in self._tasks.values())

    def shutdown(self) -> None:
        """
        Cancel the cleanup timer and prevent further rescheduling.

        Call this during application shutdown (lifespan teardown) to
        stop the recurring cleanup thread cleanly.
        """
        self._shutdown_event.set()
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
        logger.info("TaskManager shut down")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Store a reference to the running asyncio event loop.

        Called during lifespan startup so that ``_push()`` can use
        ``call_soon_threadsafe`` to bridge messages from the sync
        worker thread into the async ``asyncio.Queue``.
        """
        self._loop = loop

    # ------------------------------------------------------------------
    # GPU time limit
    # ------------------------------------------------------------------

    @staticmethod
    def _timeout_task(info: TaskInfo) -> None:
        """Cancel a task that has exceeded ``MAX_TASK_DURATION_MINUTES``.

        Called by ``threading.Timer`` from a daemon thread.  Sets the
        cancel event — the worker thread checks this between pipeline
        steps and performs a clean rollback, exactly like a user-initiated
        cancel.
        """
        if info.state == TaskState.RUNNING:
            info.cancel_event.set()
            logger.warning(
                "Task %s auto-cancelled: exceeded GPU time limit",
                info.task_id[:8],
            )

    # ------------------------------------------------------------------
    # WebSocket message helpers
    # ------------------------------------------------------------------

    def _push(self, info: TaskInfo, message: dict) -> None:
        """
        Push a WebSocket message onto the task's async queue.

        When called from a worker thread (the normal case), uses
        ``call_soon_threadsafe`` to schedule the ``put_nowait`` on the
        event loop thread — required because ``asyncio.Queue`` is not
        thread-safe.  Falls back to a direct ``put_nowait`` when no
        event loop is available (e.g. in unit tests).

        Every message gets the task's next ``seq``. Scheduling happens
        under ``_event_lock`` so queue order matches ``seq`` order even
        when two threads push.
        """
        with info._event_lock:
            info._seq += 1
            message = {**message, "seq": info._seq}
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(info._message_queue.put_nowait, message)
            else:
                # Fallback: direct put (safe when no coroutine is awaiting
                # queue.get(), e.g. in unit tests).
                info._message_queue.put_nowait(message)

    def _record_outcome(
        self,
        info: TaskInfo,
        message: dict,
        result: FilingResult | None = None,
    ) -> None:
        """
        Count one filing's outcome and push the message that reports it.

        ``filing_done`` appends ``result``; ``filing_skipped`` and
        ``filing_failed`` bump their counters. Both steps happen under
        ``_event_lock``, so a WebSocket snapshot sees either neither or
        both — never the count without the ``seq`` that reported it.
        """
        kind = message["type"]
        with info._event_lock:
            if kind == "filing_done":
                if result is None:
                    raise ValueError("filing_done needs a result")
                info.results.append(result)
            elif kind == "filing_skipped":
                info.progress.filings_skipped += 1
            elif kind == "filing_failed":
                info.progress.filings_failed += 1
            else:
                raise ValueError(f"Not a filing outcome: {kind!r}")
            info.progress.filings_done += 1
            self._push(info, message)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run_with_edgar_identity(
        self,
        info: TaskInfo,
        operation: Callable[..., _T],
        *args,
        **kwargs,
    ) -> _T:
        """Run an EDGAR-bound operation under the correct effective identity.

        ``edgar.set_identity()`` mutates process-global state, so the identity
        must be re-applied immediately before every EDGAR request cluster.
        A dedicated lock keeps this invariant correct even if task concurrency
        changes in the future.
        """
        with self._edgar_lock:
            self._fetcher.apply_identity(info.edgar_name, info.edgar_email)
            return operation(*args, **kwargs)

    def _run_task(self, info: TaskInfo) -> None:
        """
        Execute the ingestion task.  Runs in a background thread.

        Acquires the GPU semaphore (blocking — FIFO queue), then
        iterates over tickers × form_types running the two-phase ingest
        pipeline.
        """
        # Wait for GPU slot (blocks if another task is running).
        logger.info("Task %s waiting for GPU slot...", info.task_id[:8])
        self._gpu_semaphore.acquire()

        try:
            # Check for cancellation while queued.
            if info.cancel_event.is_set():
                info.state = TaskState.CANCELLED
                info.completed_at = datetime.now(UTC)
                self._push(info, {"type": "cancelled"})
                return

            info.state = TaskState.RUNNING
            info.started_at = datetime.now(UTC)

            # Start GPU time limit timer if configured.
            max_minutes = get_settings().api.max_task_duration_minutes
            if max_minutes > 0:
                timer = threading.Timer(
                    max_minutes * 60,
                    self._timeout_task,
                    args=(info,),
                )
                timer.daemon = True
                timer.start()
                info._duration_timer = timer

            self._execute(info)

        except Exception as exc:
            info.state = TaskState.FAILED
            info.error = str(exc)
            info.completed_at = datetime.now(UTC)
            self._push(
                info,
                {
                    "type": "failed",
                    "error": str(exc),
                    "details": None,
                },
            )
            logger.exception("Task %s failed unexpectedly", info.task_id[:8])
        finally:
            # Cancel the duration timer if it hasn't fired yet.
            if info._duration_timer is not None:
                info._duration_timer.cancel()
                info._duration_timer = None
            self._gpu_semaphore.release()

    def _execute(self, info: TaskInfo) -> None:
        """
        Run the shared two-phase ingest (``sec_semantic_search.ingest``).

        Lists the task's filings (metadata only), then for each one:
        duplicate check → filing limit (demo mode evicts the oldest
        filings instead of failing) → fetch, one filing ahead → parse →
        chunk → embed → store (SQLite first, then ChromaDB). Every EDGAR
        call runs under the task's own identity; progress and outcomes go
        to the task state and its WebSocket queue through ``_TaskObserver``.
        """
        observer = _TaskObserver(self, info)
        work = self._run_with_edgar_identity(info, self._build_work_list, info, observer)
        info.progress.filings_total = len(work)

        settings = get_settings()
        try:
            summary = run_ingest(
                work,
                fetch=lambda fi: self._run_with_edgar_identity(
                    info, self._fetcher.fetch_filing_content, fi
                ),
                orchestrator=self._orchestrator,
                registry=self._registry,
                chroma=self._chroma,
                max_filings=settings.database.max_filings,
                observer=observer,
                make_room=(lambda n: self._maybe_evict(info, n))
                if settings.api.demo_mode
                else None,
                cancel_event=info.cancel_event,
                prefetch_thread_name=f"prefetch-{info.task_id[:8]}",
            )
        except IngestCancelled:
            self._rollback(info)
            info.state = TaskState.CANCELLED
            info.completed_at = datetime.now(UTC)
            self._push(info, {"type": "cancelled"})
            logger.info("Task %s cancelled", info.task_id[:8])
            return

        if summary.limit_error is not None:
            exc = summary.limit_error
            info.state = TaskState.FAILED
            info.error = exc.message
            info.completed_at = datetime.now(UTC)
            self._push(info, {"type": "failed", "error": exc.message, "details": exc.details})
            return

        # All filings processed — mark complete.
        if info.state == TaskState.RUNNING:
            info.state = TaskState.COMPLETED
            info.completed_at = datetime.now(UTC)
            info.progress.step_label = "Complete"
            self._push(
                info,
                {
                    "type": "completed",
                    "results": [r.to_dict() for r in info.results],
                    "summary": {
                        "total": len(info.results)
                        + info.progress.filings_skipped
                        + info.progress.filings_failed,
                        "succeeded": len(info.results),
                        "skipped": info.progress.filings_skipped,
                        "failed": info.progress.filings_failed,
                    },
                },
            )
            logger.info(
                "Task %s completed: %d ingested, %d skipped, %d failed",
                info.task_id[:8],
                len(info.results),
                info.progress.filings_skipped,
                info.progress.filings_failed,
            )

    def _build_work_list(
        self,
        info: TaskInfo,
        observer: IngestObserver | None = None,
    ) -> list[FilingInfo]:
        """List the task's filings (metadata only; HTML is fetched later)."""
        return plan_work(
            self._fetcher,
            info.tickers,
            info.form_types,
            count_mode=info.count_mode,
            count=info.count,
            year=info.year,
            start_date=info.start_date,
            end_date=info.end_date,
            observer=observer or _TaskObserver(self, info),
            cancel_event=info.cancel_event,
        )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def _rollback(self, info: TaskInfo) -> None:
        """
        Roll back any filings stored during the current task.

        Called on cancellation to maintain dual-store consistency.
        Deletes from ChromaDB first, then SQLite (the delete order, AD#3).
        """
        if not info._stored_accessions:
            return

        logger.info(
            "Task %s: rolling back %d filing(s)",
            info.task_id[:8],
            len(info._stored_accessions),
        )

        for accession in info._stored_accessions:
            try:
                self._chroma.delete_filing(accession)
                self._registry.remove_filing(accession)
            except DatabaseError as exc:
                logger.error(
                    "Task %s: rollback failed for %s — %s",
                    info.task_id[:8],
                    accession,
                    exc.message,
                )

        info._stored_accessions.clear()

    # ------------------------------------------------------------------
    # FIFO eviction (demo mode)
    # ------------------------------------------------------------------

    def _maybe_evict(self, info: TaskInfo, new_filings: int) -> None:
        """
        Evict the oldest filings when demo mode is active and the
        database would exceed its capacity with the incoming batch.

        Uses ``list_oldest_filings()`` + ``delete_filings_batch()`` to
        remove the oldest ``slots_needed + DEMO_EVICTION_BUFFER`` filings
        from both stores.  Sends a WebSocket ``eviction`` message so the
        frontend can display a toast notification.

        No-op when there is enough space or when the eviction count
        computes to zero.
        """
        settings = get_settings()
        max_filings = settings.database.max_filings
        current_count = self._registry.count()
        available = max_filings - current_count

        if new_filings <= available:
            return  # enough room — no eviction needed

        slots_needed = new_filings - available
        eviction_count = slots_needed + settings.api.demo_eviction_buffer

        # Don't try to evict more than what's stored.
        eviction_count = min(eviction_count, current_count)

        if eviction_count <= 0:
            return

        oldest = self._registry.list_oldest_filings(eviction_count)
        if not oldest:
            return

        evicted_tickers = sorted({f.ticker for f in oldest})
        chunks_deleted = delete_filings_batch(
            oldest,
            chroma=self._chroma,
            registry=self._registry,
        )

        logger.info(
            "Task %s: FIFO eviction — deleted %d filing(s) (%d chunks) "
            "to make room for %d new filing(s)",
            info.task_id[:8],
            len(oldest),
            chunks_deleted,
            new_filings,
        )

        # Notify WebSocket clients of the eviction.
        self._push(
            info,
            {
                "type": "eviction",
                "filings_evicted": len(oldest),
                "chunks_evicted": chunks_deleted,
                "tickers_affected": evicted_tickers,
            },
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _start_cleanup_timer(self) -> None:
        """Schedule periodic pruning of stale tasks."""
        if self._shutdown_event.is_set():
            return
        timer = threading.Timer(60.0, self._cleanup_loop)
        timer.daemon = True
        timer.start()
        self._cleanup_timer = timer

    def _cleanup_loop(self) -> None:
        """Prune finished tasks older than TTL, then reschedule."""
        if self._shutdown_event.is_set():
            return
        try:
            self._prune_stale_tasks()
        except Exception:
            logger.exception("Task cleanup error")
        finally:
            self._start_cleanup_timer()

    def _prune_stale_tasks(self) -> None:
        """Persist and remove completed/failed/cancelled tasks older than the TTL."""
        now = time.time()
        terminal = (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)
        to_remove: list[str] = []

        for task_id, info in self._tasks.items():
            if info.state not in terminal:
                continue
            if info.completed_at is None:
                continue
            age = now - info.completed_at.timestamp()
            if age > _TASK_TTL_SECONDS:
                to_remove.append(task_id)

        if to_remove:
            # Persist to SQLite before removing from memory.
            for task_id in to_remove:
                info = self._tasks[task_id]
                try:
                    self._registry.save_task_history(
                        task_id,
                        status=info.state.value,
                        tickers=info.tickers,
                        form_types=info.form_types,
                        results=[r.to_history_dict() for r in info.results],
                        error=info.error,
                        started_at=(info.started_at.isoformat() if info.started_at else None),
                        completed_at=(info.completed_at.isoformat() if info.completed_at else None),
                        filings_done=info.progress.filings_done,
                        filings_skipped=info.progress.filings_skipped,
                        filings_failed=info.progress.filings_failed,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist task %s to history",
                        task_id[:8],
                    )

            with self._lock:
                for task_id in to_remove:
                    del self._tasks[task_id]
            logger.info("Pruned %d stale task(s)", len(to_remove))

            # Prune old history entries based on TASK_HISTORY_RETENTION_DAYS.
            # When 0 (the default), pruning is skipped (kept indefinitely).
            try:
                self._registry.prune_task_history()
            except Exception:
                logger.exception("Failed to prune task history")


class TaskQueueFullError(Exception):
    """Raised when the active task queue exceeds the maximum allowed size."""
