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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import (
    DatabaseError,
    FetchError,
    FilingIdentifier,
    FilingLimitExceededError,
    SECSemanticSearchError,
    get_logger,
)
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry, delete_filings_batch
from sec_semantic_search.pipeline import PipelineOrchestrator
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

    # Accession numbers stored so far in the *current* filing — used for
    # partial rollback on cancellation.
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
# One-ahead fetch
# ---------------------------------------------------------------------------


class _FilingPrefetcher:
    """
    Fetch filing HTML one filing ahead of the ingest worker.

    ``take()`` returns the HTML for the next filing in ``order`` and then
    starts fetching the one after it, so the EDGAR round-trip for filing
    *i+1* runs while filing *i* is parsed, embedded and stored. One worker
    thread keeps fetches sequential and holds at most one filing's HTML
    besides the one being processed.

    ``take()`` must be called for every filing in ``order``, in order.
    Whatever ``fetch`` raises, ``take()`` re-raises in the caller's thread.
    """

    def __init__(
        self,
        fetch: Callable[[FilingInfo], tuple[FilingIdentifier, str]],
        order: list[FilingInfo],
        *,
        cancel_event: threading.Event,
        thread_name: str,
    ) -> None:
        self._fetch = fetch
        self._order = order
        self._position = 0
        self._cancel_event = cancel_event
        self._in_flight: tuple[FilingInfo, Future] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)

    def __enter__(self) -> _FilingPrefetcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def take(self, filing_info: FilingInfo) -> tuple[FilingIdentifier, str]:
        """Wait for ``filing_info``'s HTML, then start fetching the next filing.

        Raises:
            _CancelledError: If the task was cancelled before the fetch began.
            RuntimeError: If called out of order.
        """
        if self._in_flight is None:
            self._submit_next()
        if self._in_flight is None or self._in_flight[0] is not filing_info:
            raise RuntimeError("Prefetch order does not match the ingest loop.")
        _, future = self._in_flight
        self._in_flight = None
        try:
            return future.result()
        finally:
            # Start the next fetch whether or not this one succeeded.
            self._submit_next()

    def close(self) -> None:
        """Drop queued fetches without waiting for one already running.

        A running fetch finishes in the background (edgartools bounds each
        request with a timeout) and its result is discarded, so a cancel or
        an early failure never waits on the network.
        """
        self._in_flight = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _submit_next(self) -> None:
        if self._position >= len(self._order):
            return
        filing_info = self._order[self._position]
        self._position += 1
        self._in_flight = (
            filing_info,
            self._pool.submit(self._fetch_unless_cancelled, filing_info),
        )

    def _fetch_unless_cancelled(self, filing_info: FilingInfo) -> tuple[FilingIdentifier, str]:
        # Checked when the fetch starts, not when it was queued: a cancel
        # during the previous filing skips this network call entirely.
        if self._cancel_event.is_set():
            raise _CancelledError
        return self._fetch(filing_info)


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
        Core ingestion logic — mirrors the CLI two-phase ingest.

        Steps per filing:
            1. Fetch metadata (cheap — ``list_available``)
            2. Duplicate check
            3. Fetch HTML content (one filing ahead — see ``_FilingPrefetcher``)
            4. Process (parse → chunk → embed — expensive GPU)
            5. Store (ChromaDB first, then SQLite)

        The next filing's HTML downloads while the current one is
        processed, so at most two filings' HTML are in memory at a time.
        """
        # Build the flat work list of filings to ingest (metadata only).
        work = self._run_with_edgar_identity(info, self._build_work_list, info)

        info.progress.filings_total = len(work)

        # Batch duplicate check — one SQL query for the whole work list.
        all_accessions = [fi.accession_number for fi in work]
        existing = self._registry.get_existing_accessions(all_accessions)

        # --- FIFO eviction (demo mode only) ------------------------------
        # When DEMO_MODE is enabled and the incoming batch would exceed the
        # filing limit, automatically evict the oldest filings to make room
        # instead of failing with FilingLimitExceededError.
        settings = get_settings()
        if settings.api.demo_mode:
            new_count = sum(1 for fi in work if fi.accession_number not in existing)
            self._maybe_evict(info, new_count)

        # Cache the filing count to avoid N separate COUNT(*) queries.
        # The GPU semaphore ensures single-task execution, so the count
        # only changes when *this* task stores a filing or evicts.
        cached_count = self._registry.count()
        max_filings = settings.database.max_filings

        # Fetch HTML one filing ahead: the next filing downloads while this
        # one is parsed, embedded and stored. Duplicates are never fetched,
        # and every fetch still goes through the EDGAR identity guard.
        to_fetch = [fi for fi in work if fi.accession_number not in existing]
        prefetcher = _FilingPrefetcher(
            lambda fi: self._run_with_edgar_identity(info, self._fetcher.fetch_filing_content, fi),
            to_fetch,
            cancel_event=info.cancel_event,
            thread_name=f"prefetch-{info.task_id[:8]}",
        )

        with prefetcher:
            for filing_info in work:
                filing_id = filing_info.to_identifier()

                # --- Cancellation check (between filings) --------------------
                if info.cancel_event.is_set():
                    self._rollback(info)
                    info.state = TaskState.CANCELLED
                    info.completed_at = datetime.now(UTC)
                    self._push(info, {"type": "cancelled"})
                    logger.info("Task %s cancelled", info.task_id[:8])
                    return

                ticker = filing_id.ticker
                form_type = filing_id.form_type

                info.progress.current_ticker = ticker
                info.progress.current_form_type = form_type
                info.progress.step_label = "Checking duplicate"
                info.progress.step_index = 1

                # --- Duplicate check -----------------------------------------
                if filing_id.accession_number in existing:
                    self._record_outcome(
                        info,
                        {
                            "type": "filing_skipped",
                            "ticker": ticker,
                            "form_type": form_type,
                            "accession_number": filing_id.accession_number,
                            "reason": "duplicate",
                        },
                    )
                    logger.info(
                        "Task %s: skipped duplicate %s",
                        info.task_id[:8],
                        filing_id.accession_number,
                    )
                    continue

                # --- Filing limit check (cached) ---------------------------------
                if cached_count >= max_filings:
                    if settings.api.demo_mode:
                        self._maybe_evict(info, 1)
                        # Re-read count after eviction.
                        cached_count = self._registry.count()
                        if cached_count >= max_filings:
                            exc = FilingLimitExceededError(cached_count, max_filings)
                            info.state = TaskState.FAILED
                            info.error = exc.message
                            info.completed_at = datetime.now(UTC)
                            self._push(
                                info,
                                {
                                    "type": "failed",
                                    "error": exc.message,
                                    "details": exc.details,
                                },
                            )
                            return
                    else:
                        exc = FilingLimitExceededError(cached_count, max_filings)
                        info.state = TaskState.FAILED
                        info.error = exc.message
                        info.completed_at = datetime.now(UTC)
                        self._push(
                            info,
                            {
                                "type": "failed",
                                "error": exc.message,
                                "details": exc.details,
                            },
                        )
                        return

                # --- Fetch HTML content (prefetched) --------------------------
                info.progress.step_label = "Fetching"
                info.progress.step_index = 0

                try:
                    _, html_content = prefetcher.take(filing_info)
                except _CancelledError:
                    # Cancelled before this filing's fetch began.
                    self._rollback(info)
                    info.state = TaskState.CANCELLED
                    info.completed_at = datetime.now(UTC)
                    self._push(info, {"type": "cancelled"})
                    logger.info("Task %s cancelled", info.task_id[:8])
                    return
                except FetchError as exc:
                    self._record_outcome(
                        info,
                        {
                            "type": "filing_failed",
                            "ticker": ticker,
                            "form_type": form_type,
                            "accession_number": filing_id.accession_number,
                            "error": exc.message,
                        },
                    )
                    logger.warning(
                        "Task %s: fetch failed for %s — %s",
                        info.task_id[:8],
                        filing_id.accession_number,
                        exc.message,
                    )
                    continue

                # --- Process (parse → chunk → embed) -------------------------
                def _progress_cb(
                    step: str,
                    current: int,
                    total: int,
                    _self: TaskManager = self,
                    _info: TaskInfo = info,
                    _ticker: str = ticker,
                    _form: str = form_type,
                ) -> None:
                    """Feed pipeline progress into task state."""
                    # Pipeline reports steps 1–4 (parse, chunk, embed, complete).
                    # With fetching as 0 and storing as 4 these are the
                    # 0-based stepper indices: 0=fetch, 1=parse, 2=chunk,
                    # 3=embed, 4=store. ``step_number`` carries the same index.
                    with _info._event_lock:
                        _info.progress.current_ticker = _ticker
                        _info.progress.current_form_type = _form
                        _info.progress.step_label = step
                        _info.progress.step_index = current
                        _info.progress.step_total = 5
                        _self._push(
                            _info,
                            {
                                "type": "step",
                                "ticker": _ticker,
                                "form_type": _form,
                                "step": step,
                                "step_number": current,
                                "total_steps": 5,
                            },
                        )

                    # Check cancellation between pipeline steps.
                    if _info.cancel_event.is_set():
                        raise _CancelledError

                info.progress.step_label = "Processing"
                info.progress.step_index = 1

                try:
                    result = self._orchestrator.process_filing(
                        filing_id,
                        html_content,
                        progress_callback=_progress_cb,
                    )
                except _CancelledError:
                    self._rollback(info)
                    info.state = TaskState.CANCELLED
                    info.completed_at = datetime.now(UTC)
                    self._push(info, {"type": "cancelled"})
                    logger.info("Task %s cancelled during processing", info.task_id[:8])
                    return
                except SECSemanticSearchError as exc:
                    self._record_outcome(
                        info,
                        {
                            "type": "filing_failed",
                            "ticker": ticker,
                            "form_type": form_type,
                            "accession_number": filing_id.accession_number,
                            "error": exc.message,
                        },
                    )
                    logger.warning(
                        "Task %s: processing failed for %s — %s",
                        info.task_id[:8],
                        filing_id.accession_number,
                        exc.message,
                    )
                    continue

                # --- Store (ChromaDB first, then SQLite) ---------------------
                info.progress.step_label = "Storing"
                info.progress.step_index = 4

                if info.cancel_event.is_set():
                    self._rollback(info)
                    info.state = TaskState.CANCELLED
                    info.completed_at = datetime.now(UTC)
                    self._push(info, {"type": "cancelled"})
                    return

                try:
                    # Atomic check-then-insert: holds the SQLite lock across
                    # both the duplicate check and the INSERT, preventing the
                    # race window where two threads both pass the batch
                    # duplicate check and then both register the same filing.
                    # SQLite registration is done first so that a late
                    # duplicate is caught before writing to ChromaDB.
                    registered = self._registry.register_filing_if_new(
                        result.filing_id,
                        result.ingest_result.chunk_count,
                        segments=result.segments,
                    )
                    if not registered:
                        # Another thread registered this filing between the
                        # batch duplicate check and now — treat as a skip.
                        self._record_outcome(
                            info,
                            {
                                "type": "filing_skipped",
                                "ticker": ticker,
                                "form_type": form_type,
                                "accession_number": filing_id.accession_number,
                                "reason": "duplicate",
                            },
                        )
                        logger.info(
                            "Task %s: skipped late duplicate %s",
                            info.task_id[:8],
                            filing_id.accession_number,
                        )
                        continue

                    try:
                        self._chroma.store_filing(result)
                    except DatabaseError:
                        # ChromaDB store failed after SQLite succeeded —
                        # roll back the SQLite entry to maintain consistency.
                        self._registry.remove_filing(filing_id.accession_number)
                        raise
                except DatabaseError as exc:
                    self._record_outcome(
                        info,
                        {
                            "type": "filing_failed",
                            "ticker": ticker,
                            "form_type": form_type,
                            "accession_number": filing_id.accession_number,
                            "error": exc.message,
                        },
                    )
                    logger.warning(
                        "Task %s: storage failed for %s — %s",
                        info.task_id[:8],
                        filing_id.accession_number,
                        exc.message,
                    )
                    continue

                # Record success and update the cached filing count.
                cached_count += 1
                info._stored_accessions.append(filing_id.accession_number)
                self._record_outcome(
                    info,
                    {
                        "type": "filing_done",
                        "ticker": filing_id.ticker,
                        "form_type": filing_id.form_type,
                        "filing_date": filing_id.date_str,
                        "accession_number": filing_id.accession_number,
                        "segments": result.ingest_result.segment_count,
                        "chunks": result.ingest_result.chunk_count,
                        "time": round(result.ingest_result.duration_seconds, 1),
                    },
                    FilingResult(
                        ticker=filing_id.ticker,
                        form_type=filing_id.form_type,
                        filing_date=filing_id.date_str,
                        accession_number=filing_id.accession_number,
                        segment_count=result.ingest_result.segment_count,
                        chunk_count=result.ingest_result.chunk_count,
                        duration_seconds=result.ingest_result.duration_seconds,
                    ),
                )

                logger.info(
                    "Task %s: ingested %s %s (%s) — %d chunks in %.1fs",
                    info.task_id[:8],
                    filing_id.ticker,
                    filing_id.form_type,
                    filing_id.date_str,
                    result.ingest_result.chunk_count,
                    result.ingest_result.duration_seconds,
                )

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

    # ------------------------------------------------------------------
    # Work list builder
    # ------------------------------------------------------------------

    def _build_work_list(
        self,
        info: TaskInfo,
    ) -> list[FilingInfo]:
        """
        Build a flat list of ``FilingInfo`` metadata objects.

        Only fetches lightweight metadata (no HTML content). HTML is
        fetched in ``_execute()`` one filing ahead of processing, so at
        most two filings' HTML are in memory at a time.
        """
        work: list[FilingInfo] = []

        for ticker in info.tickers:
            if info.cancel_event.is_set():
                break

            info.progress.current_ticker = ticker
            info.progress.step_label = "Fetching"
            info.progress.step_index = 0

            if info.count_mode == "total" and info.count is not None:
                # Cross-form mode: list available across forms, pick
                # the newest `count`.
                filings = self._fetcher.list_available_across_forms(
                    ticker,
                    tuple(info.form_types),
                    count=info.count,
                    year=info.year,
                    start_date=info.start_date,
                    end_date=info.end_date,
                )
                work.extend(filings)
            else:
                # Per-form mode: list available filings (metadata only).
                for form_type in info.form_types:
                    if info.cancel_event.is_set():
                        break

                    info.progress.current_form_type = form_type
                    effective_count = self._effective_count(info)

                    try:
                        available = self._fetcher.list_available(
                            ticker,
                            form_type,
                            count=effective_count,
                            year=info.year,
                            start_date=info.start_date,
                            end_date=info.end_date,
                        )
                        work.extend(available)
                    except FetchError as exc:
                        logger.warning(
                            "Task %s: fetch failed for %s %s — %s",
                            info.task_id[:8],
                            ticker,
                            form_type,
                            exc.message,
                        )

        return work

    @staticmethod
    def _effective_count(info: TaskInfo) -> int | None:
        """
        Determine the number of filings to fetch per form type.

        Mirrors the CLI's filter-aware default count logic.
        """
        if info.count_mode == "per_form" and info.count is not None:
            return info.count
        has_filters = (
            info.year is not None or info.start_date is not None or info.end_date is not None
        )
        if has_filters and info.count is None:
            return None  # all matching within filters
        if info.count is not None:
            return info.count
        return 1  # default: latest only

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def _rollback(self, info: TaskInfo) -> None:
        """
        Roll back any filings stored during the current task.

        Called on cancellation to maintain dual-store consistency.
        Deletes from ChromaDB first, then SQLite (matching store order).
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


# ---------------------------------------------------------------------------
# Internal sentinel exception for cancellation during pipeline
# ---------------------------------------------------------------------------


class _CancelledError(Exception):
    """Raised inside a progress callback to abort the pipeline."""


class TaskQueueFullError(Exception):
    """Raised when the active task queue exceeds the maximum allowed size."""
