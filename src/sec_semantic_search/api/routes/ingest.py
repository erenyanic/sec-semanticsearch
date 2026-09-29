"""
Ingestion endpoints for starting, monitoring, and cancelling filing ingests.

Provides five routes:
    - ``POST /``           — start single-ticker ingestion
    - ``POST /batch``      — start multi-ticker ingestion
    - ``GET /tasks``       — list all tasks (active + recent)
    - ``GET /tasks/{id}``  — get task status and progress
    - ``DELETE /tasks/{id}`` — cancel a running task

All business logic lives in ``tasks.TaskManager``; these routes are a
thin HTTP layer that validates input, delegates to the manager, and
converts internal dataclasses into Pydantic response schemas.

``get_task`` and ``cancel_task`` are plain ``def``: a task pruned from memory
is looked up in SQLite task history, which blocks. The rest only touch
in-memory state and stay async.
"""

from __future__ import annotations

import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from sec_semantic_search.api.dependencies import EdgarIdentity, get_edgar_identity, get_task_manager
from sec_semantic_search.api.schemas import (
    ErrorResponse,
    IngestRequest,
    IngestResultSchema,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
)
from sec_semantic_search.api.schemas import (
    TaskProgress as TaskProgressSchema,
)
from sec_semantic_search.api.tasks import TaskInfo, TaskManager, TaskQueueFullError
from sec_semantic_search.config import get_settings
from sec_semantic_search.core import audit_log, get_logger, redact_for_log

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Per-IP ingest cooldown tracking
# ---------------------------------------------------------------------------

# In-memory dict: IP → monotonic timestamp of last ingest request.
# SECURITY: requires --workers 1 (in-memory state is per-process).
_cooldown_lock = threading.Lock()
_last_ingest: dict[str, float] = {}
# How often to purge stale entries (seconds).
_COOLDOWN_PRUNE_INTERVAL = 600
_MAX_COOLDOWN_ENTRIES = 100_000
_last_cooldown_prune: float = 0.0


def _check_cooldown(client_ip: str) -> None:
    """Enforce ``INGEST_COOLDOWN_SECONDS`` for the given client IP.

    Raises ``HTTPException(429)`` if the client has ingested too recently.
    When ``INGEST_COOLDOWN_SECONDS`` is ``0``, the check is skipped entirely.
    """
    global _last_cooldown_prune  # noqa: PLW0603

    cooldown = get_settings().api.ingest_cooldown_seconds
    if cooldown <= 0:
        return

    now = time.monotonic()

    with _cooldown_lock:
        # Lazy prune: remove entries older than 2× the cooldown window
        # to prevent unbounded memory growth from stale IPs.
        if now - _last_cooldown_prune > _COOLDOWN_PRUNE_INTERVAL:
            cutoff = now - (cooldown * 2)
            stale = [ip for ip, ts in _last_ingest.items() if ts < cutoff]
            for ip in stale:
                del _last_ingest[ip]
            _last_cooldown_prune = now

        # Emergency prune: if the map still reaches the configured hard cap,
        # evict the oldest half before recording the next request.
        if len(_last_ingest) >= _MAX_COOLDOWN_ENTRIES:
            oldest_entries = sorted(
                _last_ingest.items(),
                key=lambda item: item[1],
            )
            prune_count = max(len(oldest_entries) // 2, 1)
            for ip, _ in oldest_entries[:prune_count]:
                del _last_ingest[ip]
            logger.warning(
                "Cooldown tracker reached %s entries; pruned %s oldest records.",
                len(oldest_entries),
                prune_count,
            )

        last = _last_ingest.get(client_ip)
        if last is not None:
            elapsed = now - last
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed) + 1
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "cooldown",
                        "message": (
                            f"Please wait {remaining}s before submitting another ingest request."
                        ),
                        "details": None,
                        "hint": (f"Ingest cooldown is {cooldown}s between requests."),
                    },
                )

        # Record this request timestamp.
        _last_ingest[client_ip] = now


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _task_info_to_status(info: TaskInfo) -> TaskStatus:
    """Convert an internal ``TaskInfo`` dataclass to the API ``TaskStatus`` schema."""
    return TaskStatus(
        task_id=info.task_id,
        status=info.state.value,
        tickers=info.tickers,
        form_types=info.form_types,
        progress=TaskProgressSchema(
            current_ticker=info.progress.current_ticker,
            current_form_type=info.progress.current_form_type,
            step_label=info.progress.step_label,
            step_index=info.progress.step_index,
            step_total=info.progress.step_total,
            filings_done=info.progress.filings_done,
            filings_total=info.progress.filings_total,
            filings_skipped=info.progress.filings_skipped,
            filings_failed=info.progress.filings_failed,
        ),
        results=[
            IngestResultSchema(
                ticker=r.ticker,
                form_type=r.form_type,
                filing_date=r.filing_date,
                accession_number=r.accession_number,
                segment_count=r.segment_count,
                chunk_count=r.chunk_count,
                duration_seconds=r.duration_seconds,
            )
            for r in info.results
        ],
        error=info.error,
        started_at=info.started_at,
        completed_at=info.completed_at,
    )


def _create_task(
    body: IngestRequest,
    manager: TaskManager,
    identity: EdgarIdentity,
    *,
    client_ip: str = "unknown",
) -> TaskResponse:
    """Shared logic for both add and batch endpoints."""
    settings = get_settings()

    # --- Abuse prevention: request caps --------------------------------
    max_tickers = settings.api.max_tickers_per_request
    if max_tickers > 0 and len(body.tickers) > max_tickers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "request_cap_exceeded",
                "message": (
                    f"Too many tickers: {len(body.tickers)} (maximum {max_tickers} per request)."
                ),
                "details": None,
                "hint": f"Submit at most {max_tickers} tickers per request.",
            },
        )

    max_filings = settings.api.max_filings_per_request
    if max_filings > 0 and body.count is not None and body.count > max_filings:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "request_cap_exceeded",
                "message": (
                    f"Too many filings requested: {body.count} (maximum {max_filings} per request)."
                ),
                "details": None,
                "hint": f"Request at most {max_filings} filings per request.",
            },
        )

    try:
        task_id = manager.create_task(
            tickers=body.tickers,
            form_types=body.form_types,
            count_mode=body.count_mode,
            count=body.count,
            year=body.year,
            start_date=body.start_date,
            end_date=body.end_date,
            edgar_name=identity.name,
            edgar_email=identity.email,
        )
    except TaskQueueFullError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_full",
                "message": str(exc),
                "details": None,
                "hint": "Wait for existing tasks to complete before submitting new ones.",
            },
        ) from exc

    logger.info(
        "Ingest task %s created: tickers=%s, forms=%s, mode=%s",
        task_id[:8],
        [redact_for_log(t) for t in body.tickers],
        body.form_types,
        body.count_mode,
    )

    audit_log(
        "ingest_task_created",
        client_ip=client_ip,
        endpoint="POST /api/ingest",
        detail=(
            f"task_id={task_id[:8]}, "
            f"tickers={[redact_for_log(t) for t in body.tickers]}, "
            f"forms={body.form_types}"
        ),
    )

    return TaskResponse(
        task_id=task_id,
        status="pending",
        websocket_url=f"/ws/ingest/{task_id}",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/add",
    response_model=TaskResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}},
    summary="Ingest filings for a single ticker",
)
async def ingest_add(
    request: Request,
    body: IngestRequest,
    manager: TaskManager = Depends(get_task_manager),
    identity: EdgarIdentity = Depends(get_edgar_identity),
) -> TaskResponse:
    """
    Start an ingestion task for a single ticker symbol.

    The task runs in the background — use the returned ``task_id`` to
    poll status via ``GET /tasks/{task_id}`` or connect to the WebSocket
    at the returned ``websocket_url`` for real-time progress.

    Only one ticker is accepted.  For multiple tickers, use
    ``POST /batch`` instead.
    """
    if len(body.tickers) != 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": "The /add endpoint accepts exactly one ticker.",
                "details": f"Received {len(body.tickers)} tickers: {body.tickers}",
                "hint": "Use POST /api/ingest/batch for multi-ticker ingestion.",
            },
        )

    client_ip = request.client.host if request.client else "unknown"
    _check_cooldown(client_ip)
    return _create_task(body, manager, identity, client_ip=client_ip)


@router.post(
    "/batch",
    response_model=TaskResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}},
    summary="Ingest filings for multiple tickers",
)
async def ingest_batch(
    request: Request,
    body: IngestRequest,
    manager: TaskManager = Depends(get_task_manager),
    identity: EdgarIdentity = Depends(get_edgar_identity),
) -> TaskResponse:
    """
    Start an ingestion task for one or more ticker symbols.

    Equivalent to ``sec-search ingest batch``.  The task runs in the
    background; poll status or connect via WebSocket.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_cooldown(client_ip)
    return _create_task(body, manager, identity, client_ip=client_ip)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List all ingestion tasks",
)
async def list_tasks(
    manager: TaskManager = Depends(get_task_manager),
) -> TaskListResponse:
    """
    Return all ingestion tasks, including active, completed, failed,
    and cancelled tasks that have not yet been pruned (24-hour TTL).
    """
    tasks = manager.list_tasks()
    statuses = [_task_info_to_status(t) for t in tasks]

    return TaskListResponse(tasks=statuses, total=len(statuses))


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatus,
    responses={404: {"model": ErrorResponse}},
    summary="Get ingestion task status",
)
def get_task(
    task_id: str,
    manager: TaskManager = Depends(get_task_manager),
) -> TaskStatus:
    """
    Return the current status and progress of a specific ingestion task.

    Includes per-filing results as they complete, progress counters,
    and any error messages.
    """
    info = manager.get_task(task_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Task '{task_id}' not found.",
                "details": None,
                "hint": "The task may have been pruned after completion (24-hour TTL).",
            },
        )

    return _task_info_to_status(info)


@router.delete(
    "/tasks/{task_id}",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    summary="Cancel a running ingestion task",
)
def cancel_task(
    request: Request,
    task_id: str,
    manager: TaskManager = Depends(get_task_manager),
) -> dict[str, str]:
    """
    Request cancellation of a running or pending ingestion task.

    Cancellation is cooperative — the worker thread checks the cancel
    flag between pipeline steps and rolls back any partially stored
    filings.
    """
    info = manager.get_task(task_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Task '{task_id}' not found.",
                "details": None,
                "hint": "The task may have been pruned after completion (24-hour TTL).",
            },
        )

    cancelled = manager.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "conflict",
                "message": f"Task '{task_id}' has already finished ({info.state.value}).",
                "details": None,
                "hint": "Only pending or running tasks can be cancelled.",
            },
        )

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "cancel_task",
        client_ip=client_ip,
        endpoint=f"DELETE /api/ingest/tasks/{task_id[:8]}",
        detail=f"task_id={task_id}",
    )
    return {"task_id": task_id, "status": "cancelling"}
