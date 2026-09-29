"""
Unit tests for TaskManager worker internals.

Covers the previously untested static/internal methods:
    - per_form_count() — the listing count per form
    - _rollback() — success and error tolerance
    - _push() — WebSocket message queuing
"""

from unittest.mock import MagicMock, patch

import pytest

from sec_semantic_search.api.tasks import TaskManager, TaskState
from sec_semantic_search.core.exceptions import DatabaseError
from sec_semantic_search.ingest import per_form_count
from tests.helpers import make_task_info

# -----------------------------------------------------------------------
# per_form_count() — shared by the CLI and the API
# -----------------------------------------------------------------------


class TestPerFormCount:
    """per_form_count() determines how many filings to list per form."""

    def test_per_form_with_count(self):
        assert per_form_count("per_form", 3, has_filters=False) == 3

    def test_per_form_without_count(self):
        """per_form mode without count falls through to default (1)."""
        assert per_form_count("per_form", None, has_filters=False) == 1

    def test_latest_with_filters_no_count(self):
        """With date filters active and no explicit count, list all matching."""
        assert per_form_count("latest", None, has_filters=True) is None

    def test_latest_with_explicit_count(self):
        """Explicit count should be used even in 'latest' mode."""
        assert per_form_count("latest", 5, has_filters=False) == 5
        assert per_form_count("latest", 5, has_filters=True) == 5

    def test_default_returns_one(self):
        """No filters, no count, 'latest' mode → default to 1."""
        assert per_form_count("latest", None, has_filters=False) == 1


# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# _rollback()
# -----------------------------------------------------------------------


@pytest.fixture
def manager():
    """TaskManager with all dependencies mocked."""
    with patch.object(TaskManager, "_start_cleanup_timer"):
        mgr = TaskManager(
            registry=MagicMock(),
            chroma=MagicMock(),
            fetcher=MagicMock(),
            orchestrator=MagicMock(),
        )
    return mgr


class TestRollback:
    """_rollback() cleans up partially stored filings on cancel."""

    def test_rollback_deletes_stored_accessions(self, manager):
        info = make_task_info(state=TaskState.RUNNING)
        info._stored_accessions = ["ACC-001", "ACC-002"]

        manager._rollback(info)

        assert manager._chroma.delete_filing.call_count == 2
        assert manager._registry.remove_filing.call_count == 2
        manager._chroma.delete_filing.assert_any_call("ACC-001")
        manager._chroma.delete_filing.assert_any_call("ACC-002")
        manager._registry.remove_filing.assert_any_call("ACC-001")
        manager._registry.remove_filing.assert_any_call("ACC-002")

    def test_rollback_clears_accessions_list(self, manager):
        info = make_task_info(state=TaskState.RUNNING)
        info._stored_accessions = ["ACC-001"]

        manager._rollback(info)

        assert info._stored_accessions == []

    def test_rollback_empty_list_is_noop(self, manager):
        info = make_task_info(state=TaskState.RUNNING)
        info._stored_accessions = []

        manager._rollback(info)

        manager._chroma.delete_filing.assert_not_called()
        manager._registry.remove_filing.assert_not_called()

    def test_rollback_tolerates_database_error(self, manager):
        """If one rollback fails, it should still attempt the rest."""
        info = make_task_info(state=TaskState.RUNNING)
        info._stored_accessions = ["ACC-001", "ACC-002"]

        manager._chroma.delete_filing.side_effect = [
            DatabaseError("fail"),
            None,
        ]

        # Should not raise — errors are logged, not propagated.
        manager._rollback(info)
        assert manager._chroma.delete_filing.call_count == 2

    def test_rollback_chromadb_first_then_sqlite(self, manager):
        """Rollback must follow the same order as store: ChromaDB then SQLite."""
        info = make_task_info(state=TaskState.RUNNING)
        info._stored_accessions = ["ACC-001"]

        call_order = []
        manager._chroma.delete_filing.side_effect = lambda acc: call_order.append(("chroma", acc))
        manager._registry.remove_filing.side_effect = lambda acc: call_order.append(
            ("registry", acc)
        )

        manager._rollback(info)

        assert call_order == [("chroma", "ACC-001"), ("registry", "ACC-001")]


# -----------------------------------------------------------------------
# _push()
# -----------------------------------------------------------------------


class TestPush:
    """_push() puts messages on the task's async queue for WebSocket streaming."""

    def test_message_placed_on_queue(self, manager):
        info = make_task_info()
        message = {"type": "step", "step": "Parsing"}

        manager._push(info, message)

        assert not info._message_queue.empty()
        assert info._message_queue.get_nowait() == {**message, "seq": 1}
        assert message == {"type": "step", "step": "Parsing"}  # caller's dict untouched

    def test_seq_increments_per_task(self, manager):
        first, second = make_task_info(task_id="a" * 12), make_task_info(task_id="b" * 12)
        manager._push(first, {"type": "step"})
        manager._push(first, {"type": "step"})
        manager._push(second, {"type": "step"})

        assert [first._message_queue.get_nowait()["seq"] for _ in range(2)] == [1, 2]
        assert second._message_queue.get_nowait()["seq"] == 1
        assert first._seq == 2

    def test_multiple_messages_fifo(self, manager):
        info = make_task_info()
        manager._push(info, {"type": "step", "step": "Parsing"})
        manager._push(info, {"type": "step", "step": "Embedding"})
        manager._push(info, {"type": "completed", "results": []})

        msgs = []
        while not info._message_queue.empty():
            msgs.append(info._message_queue.get_nowait())

        assert [m["type"] for m in msgs] == ["step", "step", "completed"]
