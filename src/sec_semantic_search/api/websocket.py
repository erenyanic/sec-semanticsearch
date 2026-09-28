"""
WebSocket endpoint for real-time ingestion progress streaming.

Provides a single WebSocket route at ``/ws/ingest/{task_id}`` that
pushes JSON messages as the background ingestion worker progresses
through the pipeline.

Message types (server → client):
    - ``snapshot``       — current state on connect (for reconnection)
    - ``step``           — pipeline step progress
    - ``filing_done``    — filing successfully ingested
    - ``filing_skipped`` — duplicate detected
    - ``filing_failed``  — processing or storage error
    - ``eviction``       — FIFO eviction occurred (demo mode only)
    - ``completed``      — task finished
    - ``failed``         — task failed
    - ``cancelled``      — task cancelled

The client does not send messages after the initial connection.
Disconnecting is handled gracefully — the task continues running
server-side.  Reconnecting sends a fresh snapshot so the client
can catch up.
"""

from __future__ import annotations

import asyncio
import hmac

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from sec_semantic_search.api.tasks import TaskInfo, TaskState
from sec_semantic_search.config import get_settings
from sec_semantic_search.core import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Terminal message types — close the WebSocket after sending one of these.
_TERMINAL_TYPES = frozenset({"completed", "failed", "cancelled"})
_AUTH_TIMEOUT_SECONDS = 5.0


def _build_snapshot(info: TaskInfo) -> dict:
    """
    Build a snapshot message from the current task state.

    Sent immediately on WebSocket connect so a reconnecting client
    can catch up on progress made while it was disconnected.
    """
    return {
        "type": "snapshot",
        "task_id": info.task_id,
        "status": info.state.value,
        "progress": {
            "current_ticker": info.progress.current_ticker,
            "current_form_type": info.progress.current_form_type,
            "step_label": info.progress.step_label,
            "step_index": info.progress.step_index,
            "step_total": info.progress.step_total,
            "filings_done": info.progress.filings_done,
            "filings_total": info.progress.filings_total,
            "filings_skipped": info.progress.filings_skipped,
            "filings_failed": info.progress.filings_failed,
        },
        "results": [r.to_dict() for r in info.results],
    }


@router.websocket("/ws/ingest/{task_id}")
async def ingest_progress(websocket: WebSocket, task_id: str) -> None:
    """
    Stream real-time ingestion progress for a specific task.

    On connect, validates the ``Origin`` header against allowed CORS
    origins (WebSocket upgrades are not protected by CORS). When API
    key auth is enabled, the client must authenticate immediately after
    the handshake with an ``{"type": "auth", "api_key": "..."}``
    message unless it already supplied ``X-API-Key`` during the upgrade.
    Once authenticated, the server sends a ``snapshot`` message with the
    current state and continuously forwards messages from the task's
    internal queue until the task reaches a terminal state or the client
    disconnects.
    """
    # --- Origin validation (browser-only endpoint; reject missing Origin) --
    origin = websocket.headers.get("origin")
    allowed_origins = get_settings().api.cors_origins
    if origin is None or origin not in allowed_origins:
        await websocket.close(code=4003, reason="Origin not allowed")
        return

    await websocket.accept()

    # --- API key validation ------------------------------------------------
    if not await _authenticate_websocket(websocket):
        return

    # Retrieve the TaskManager from app state. A task pruned from memory is
    # read back from SQLite task history, so keep the lookup off the loop.
    task_manager = websocket.app.state.task_manager
    info = await run_in_threadpool(task_manager.get_task, task_id)

    if info is None:
        await websocket.send_json(
            {
                "type": "error",
                "error": f"Task '{task_id}' not found.",
            }
        )
        await websocket.close(code=4404, reason="Task not found")
        return

    # Send current state snapshot (supports reconnection).
    await websocket.send_json(_build_snapshot(info))

    # If the task has already reached a terminal state, send the
    # terminal message and close — no need to stream further.
    if info.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
        terminal_msg = _drain_terminal_message(info)
        if terminal_msg is None:
            # Queue was empty — message consumed by prior connection or
            # not yet delivered via call_soon_threadsafe.  Reconstruct
            # from authoritative TaskInfo state.
            terminal_msg = _build_terminal_from_state(info)
        await websocket.send_json(terminal_msg)
        await websocket.close()
        return

    # Stream messages from the task's async queue until terminal or
    # disconnect.  Uses asyncio.wait_for with a 5s safety timeout —
    # normal messages arrive near-instantly via call_soon_threadsafe;
    # the timeout only fires during idle periods (e.g. PENDING state
    # waiting for the GPU semaphore).
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    info._message_queue.get(),
                    timeout=5.0,
                )
            except TimeoutError:
                # No message within timeout — check if the task ended
                # while the queue was empty (e.g. terminal message was
                # already consumed by a previous WebSocket connection,
                # or hasn't been delivered yet via call_soon_threadsafe).
                if info.state in (
                    TaskState.COMPLETED,
                    TaskState.FAILED,
                    TaskState.CANCELLED,
                ):
                    # Yield to the event loop so any pending
                    # call_soon_threadsafe callbacks can execute.
                    await asyncio.sleep(0)
                    sent_terminal = await _drain_and_send(
                        websocket,
                        info._message_queue,
                    )
                    if not sent_terminal:
                        # Queue didn't contain a terminal message —
                        # synthesise one from authoritative TaskInfo.
                        await websocket.send_json(_build_terminal_from_state(info))
                    break
                continue

            await websocket.send_json(message)

            if message.get("type") in _TERMINAL_TYPES:
                break

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected from task %s (task continues)",
            task_id[:8],
        )

    except Exception:
        logger.exception("WebSocket error for task %s", task_id[:8])

    finally:
        # Graceful close — task continues running regardless.
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001 — already closing
            pass


async def _authenticate_websocket(websocket: WebSocket) -> bool:
    """Authenticate a WebSocket without exposing the API key in the URL."""
    expected = get_settings().api.key
    if expected is None:
        return True

    header_key = websocket.headers.get("x-api-key")
    if header_key is not None and hmac.compare_digest(header_key, expected):
        return True

    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=_AUTH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        await websocket.close(code=4001, reason="Authentication timed out")
        return False
    except WebSocketDisconnect:
        return False
    except Exception:  # noqa: BLE001 — malformed or unexpected payload
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return False

    if not isinstance(message, dict) or message.get("type") != "auth":
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return False

    provided = message.get("api_key")
    if not isinstance(provided, str) or not hmac.compare_digest(provided, expected):
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return False

    return True


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------


async def _drain_and_send(websocket: WebSocket, q: asyncio.Queue) -> bool:
    """Drain all remaining messages from the queue and send them.

    Returns ``True`` if a terminal message was among those sent.
    """
    sent_terminal = False
    while not q.empty():
        message = q.get_nowait()
        await websocket.send_json(message)
        if message.get("type") in _TERMINAL_TYPES:
            sent_terminal = True
    return sent_terminal


def _drain_terminal_message(info: TaskInfo) -> dict | None:
    """
    Try to find a terminal message in the queue for an already-finished task.

    Returns the message or None if the queue is empty (the terminal
    message was already consumed by a previous WebSocket connection).
    """
    while not info._message_queue.empty():
        message = info._message_queue.get_nowait()
        if message.get("type") in _TERMINAL_TYPES:
            return message
    return None


def _build_terminal_from_state(info: TaskInfo) -> dict:
    """Construct a terminal message from the current ``TaskInfo`` state.

    This is the authoritative fallback when the terminal queue message
    was consumed by a prior WebSocket connection or hasn't arrived yet
    due to the ``call_soon_threadsafe`` scheduling delay (race window
    between ``info.state`` assignment and ``asyncio.Queue.put_nowait``).
    """
    if info.state == TaskState.COMPLETED:
        return {
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
        }
    if info.state == TaskState.FAILED:
        return {
            "type": "failed",
            "error": info.error or "Unknown error",
            "details": None,
        }
    # CANCELLED
    return {"type": "cancelled"}
