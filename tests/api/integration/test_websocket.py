"""
Integration tests for the WebSocket progress endpoint.

Uses FastAPI TestClient's ``websocket_connect()`` with task state
injected directly onto ``app.state.task_manager``.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sec_semantic_search.api.app import app
from sec_semantic_search.api.tasks import FilingResult, TaskManager, TaskState
from sec_semantic_search.api.websocket import _build_snapshot
from tests.helpers import make_task_info

_WS_HEADERS = {"origin": "http://localhost:3000"}


def _make_client_with_task(task_info=None):
    """Build a TestClient with a task manager that returns the given task."""
    manager = MagicMock()
    if task_info is not None:
        manager.get_task.return_value = task_info
    else:
        manager.get_task.return_value = None

    # Inject directly onto app.state (WebSocket reads from app.state,
    # not from dependency overrides).
    app.state.task_manager = manager
    return TestClient(app)


# -----------------------------------------------------------------------
# Connection handling
# -----------------------------------------------------------------------


class TestWebSocketConnect:
    """WebSocket connection and error handling."""

    def test_nonexistent_task(self):
        client = _make_client_with_task(task_info=None)
        with client.websocket_connect("/ws/ingest/nonexistent", headers=_WS_HEADERS) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not found" in msg["error"].lower()


class TestWebSocketSnapshot:
    """Snapshot message is sent on connect."""

    def test_pending_task_receives_snapshot(self):
        info = make_task_info(state=TaskState.PENDING)
        client = _make_client_with_task(task_info=info)

        # Put a cancelled message to terminate the loop.
        info._message_queue.put_nowait({"type": "cancelled"})

        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["task_id"] == info.task_id
            assert snapshot["status"] == "pending"
            assert "progress" in snapshot


class TestWebSocketCompleted:
    """Already-completed tasks send snapshot + terminal message."""

    def test_completed_task(self):
        info = make_task_info(state=TaskState.COMPLETED)
        # Push a terminal message into the queue.
        info._message_queue.put_nowait(
            {
                "type": "completed",
                "results": [],
                "summary": {"total": 0},
            }
        )

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"

            terminal = ws.receive_json()
            assert terminal["type"] == "completed"


class TestWebSocketStreaming:
    """Messages streamed in order during an active task."""

    def test_step_then_terminal(self):
        info = make_task_info(state=TaskState.RUNNING)
        info._message_queue.put_nowait(
            {
                "type": "step",
                "ticker": "AAPL",
                "form_type": "10-K",
                "step": "Embedding",
                "step_number": 4,
                "total_steps": 5,
            }
        )
        info._message_queue.put_nowait(
            {
                "type": "completed",
                "results": [],
                "summary": {"total": 0},
            }
        )

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"

            step = ws.receive_json()
            assert step["type"] == "step"
            assert step["step"] == "Embedding"

            completed = ws.receive_json()
            assert completed["type"] == "completed"

    def test_filing_done_message(self):
        info = make_task_info(state=TaskState.RUNNING)
        info._message_queue.put_nowait(
            {
                "type": "filing_done",
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "accession_number": "acc-1",
                "segments": 100,
                "chunks": 110,
                "time": 5.3,
            }
        )
        info._message_queue.put_nowait({"type": "completed", "results": [], "summary": {}})

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            ws.receive_json()  # snapshot
            filing_done = ws.receive_json()
            assert filing_done["type"] == "filing_done"
            assert filing_done["ticker"] == "AAPL"
            assert filing_done["chunks"] == 110

    def test_cancelled_message(self):
        info = make_task_info(state=TaskState.RUNNING)
        info._message_queue.put_nowait({"type": "cancelled"})

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            ws.receive_json()  # snapshot
            msg = ws.receive_json()
            assert msg["type"] == "cancelled"


# -----------------------------------------------------------------------
# Reconnect with a non-empty queue
# -----------------------------------------------------------------------


def _manager() -> TaskManager:
    with patch.object(TaskManager, "_start_cleanup_timer"):
        return TaskManager(
            registry=MagicMock(),
            chroma=MagicMock(),
            fetcher=MagicMock(),
            orchestrator=MagicMock(),
        )


def _done(manager: TaskManager, info, n: int) -> None:
    accession = f"0000320193-24-{n:06d}"
    manager._record_outcome(
        info,
        {
            "type": "filing_done",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": accession,
            "segments": 1,
            "chunks": 1,
            "time": 0.1,
        },
        FilingResult("AAPL", "10-K", "2024-11-01", accession, 1, 1, 0.1),
    )


class TestReconnectReplay:
    """A reconnecting client can tell which queued messages the snapshot covers.

    Repro from the audit: a 5-filing ingest, the tab closes after filing 2,
    filings 3–4 finish with no client connected, the tab reopens.
    """

    def test_snapshot_seq_covers_messages_queued_while_disconnected(self):
        manager = _manager()
        info = make_task_info(state=TaskState.RUNNING)
        info.progress.filings_total = 5
        for n in range(4):
            _done(manager, info, n)
        # Filings 1–2 were delivered to the first connection.
        info._message_queue.get_nowait()
        info._message_queue.get_nowait()

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            snapshot = ws.receive_json()

            def finish():
                # Filing 5 finishes after the reconnect, then the task completes.
                _done(manager, info, 4)
                info.state = TaskState.COMPLETED
                manager._push(info, {"type": "completed", "results": [], "summary": {}})

            # Run on the event loop thread: asyncio.Queue is not thread-safe.
            ws.portal.call(finish)
            streamed = [ws.receive_json() for _ in range(4)]

        assert snapshot["progress"]["filings_done"] == 4
        assert len(snapshot["results"]) == 4
        assert snapshot["seq"] == 4

        # Filings 3–4 are replays: already in the snapshot, seq <= snapshot.seq.
        assert [m["type"] for m in streamed] == [
            "filing_done",
            "filing_done",
            "filing_done",
            "completed",
        ]
        assert [m["seq"] for m in streamed] == [3, 4, 5, 6]
        replays = [m for m in streamed if m["seq"] <= snapshot["seq"]]
        fresh = [m for m in streamed if m["seq"] > snapshot["seq"]]
        assert {m["accession_number"] for m in replays} == {
            r["accession_number"] for r in snapshot["results"][2:]
        }
        # Applying only the fresh counters to the snapshot gives the truth.
        fresh_done = sum(1 for m in fresh if m["type"] == "filing_done")
        assert snapshot["progress"]["filings_done"] + fresh_done == info.progress.filings_done == 5

    def test_every_message_carries_seq(self):
        manager = _manager()
        info = make_task_info(state=TaskState.RUNNING)
        manager._record_outcome(
            info,
            {"type": "filing_skipped", "ticker": "A", "form_type": "10-K", "reason": "duplicate"},
        )
        manager._record_outcome(
            info,
            {"type": "filing_failed", "ticker": "A", "form_type": "10-K", "error": "x"},
        )
        manager._push(info, {"type": "cancelled"})

        client = _make_client_with_task(task_info=info)
        with client.websocket_connect(f"/ws/ingest/{info.task_id}", headers=_WS_HEADERS) as ws:
            snapshot = ws.receive_json()
            messages = [ws.receive_json() for _ in range(3)]

        assert snapshot["seq"] == 3
        assert snapshot["progress"]["filings_skipped"] == 1
        assert snapshot["progress"]["filings_failed"] == 1
        assert snapshot["progress"]["filings_done"] == 2
        assert [m["seq"] for m in messages] == [1, 2, 3]

    def test_snapshot_counts_always_match_its_seq(self):
        """Counter update and push are atomic with respect to the snapshot."""
        manager = _manager()
        info = make_task_info(state=TaskState.RUNNING)
        stop = threading.Event()

        def worker():
            n = 0
            while not stop.is_set() and n < 20_000:
                manager._record_outcome(
                    info,
                    {"type": "filing_skipped", "ticker": "A", "form_type": "10-K"},
                )
                n += 1

        thread = threading.Thread(target=worker)
        thread.start()
        mismatches = 0
        try:
            for _ in range(2_000):
                snap = _build_snapshot(info)
                # Every push here is an outcome that adds one to filings_done.
                if snap["progress"]["filings_done"] != snap["seq"]:
                    mismatches += 1
        finally:
            stop.set()
            thread.join()
        assert mismatches == 0

    def test_record_outcome_rejects_other_messages(self):
        manager = _manager()
        info = make_task_info(state=TaskState.RUNNING)
        with pytest.raises(ValueError):
            manager._record_outcome(info, {"type": "step"})
        with pytest.raises(ValueError):
            manager._record_outcome(info, {"type": "filing_done"})  # no result
        assert info.progress.filings_done == 0
        assert info._message_queue.empty()
