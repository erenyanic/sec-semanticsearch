"""
Tests for the one-ahead HTML fetch in ``TaskManager._execute()`` (F-04).

Covers:
    - The next filing is fetched while the current one is processed
    - Fetches stay one ahead (at most two filings' HTML in memory)
    - Duplicates are never fetched; order is preserved
    - ``FetchError`` still marks one filing failed; other errors still escape
    - Every prefetch goes through the EDGAR identity guard, off the worker thread
    - Cancellation skips a fetch that has not started and never waits on one that has
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from sec_semantic_search.api.tasks import (
    TaskManager,
    TaskState,
    _CancelledError,
    _FilingPrefetcher,
)
from sec_semantic_search.core.exceptions import FetchError
from tests.helpers import make_task_info

# Generous bound for event waits; only reached when a test is failing.
_WAIT = 5.0


# -----------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------


@pytest.fixture
def manager():
    """TaskManager with all dependencies mocked and storage succeeding."""
    registry = MagicMock()
    registry.get_existing_accessions.return_value = set()
    registry.count.return_value = 0
    registry.register_filing_if_new.return_value = True
    with patch.object(TaskManager, "_start_cleanup_timer"):
        mgr = TaskManager(
            registry=registry,
            chroma=MagicMock(),
            fetcher=MagicMock(),
            orchestrator=MagicMock(),
        )
    return mgr


@pytest.fixture(autouse=True)
def _settings():
    with patch("sec_semantic_search.api.tasks.get_settings") as mock_settings:
        mock_settings.return_value.api.demo_mode = False
        mock_settings.return_value.database.max_filings = 100
        yield


def _filing(index: int) -> MagicMock:
    """A ``FilingInfo`` stand-in with a matching identifier."""
    accession = f"0000000000-24-{index:06d}"
    fi = MagicMock(name=f"filing-{index}", accession_number=accession)
    fi.to_identifier.return_value = MagicMock(
        ticker="AAPL",
        form_type="10-K",
        accession_number=accession,
        date_str="2024-01-01",
    )
    return fi


def _processed(filing_id) -> MagicMock:
    result = MagicMock()
    result.filing_id = filing_id
    result.ingest_result.segment_count = 1
    result.ingest_result.chunk_count = 1
    result.ingest_result.duration_seconds = 0.1
    return result


def _messages(info) -> list[dict]:
    out = []
    while not info._message_queue.empty():
        out.append(info._message_queue.get_nowait())
    return out


# -----------------------------------------------------------------------
# Overlap and ordering
# -----------------------------------------------------------------------


class TestOverlap:
    """The next filing downloads while the current one is processed."""

    def test_next_fetch_runs_during_processing(self, manager):
        work = [_filing(0), _filing(1)]
        manager._build_work_list = MagicMock(return_value=work)
        second_fetch_started = threading.Event()
        overlapped: list[bool] = []

        def fetch(fi):
            if fi is work[1]:
                second_fetch_started.set()
            return fi.to_identifier(), "<html></html>"

        def process(filing_id, html, progress_callback=None):
            if filing_id is work[0].to_identifier():
                # Sequential code only fetches filing 1 after this returns.
                overlapped.append(second_fetch_started.wait(_WAIT))
            return _processed(filing_id)

        manager._fetcher.fetch_filing_content.side_effect = fetch
        manager._orchestrator.process_filing.side_effect = process
        info = make_task_info(state=TaskState.RUNNING)

        manager._execute(info)

        assert overlapped == [True]
        assert info.state == TaskState.COMPLETED
        assert len(info.results) == 2

    def test_stays_one_filing_ahead(self, manager):
        """While filing 0 is processed, filing 2 must not be fetched yet."""
        work = [_filing(i) for i in range(3)]
        manager._build_work_list = MagicMock(return_value=work)
        started = {i: threading.Event() for i in range(3)}
        third_started_early: list[bool] = []

        def fetch(fi):
            started[work.index(fi)].set()
            return fi.to_identifier(), "<html></html>"

        def process(filing_id, html, progress_callback=None):
            if filing_id is work[0].to_identifier():
                assert started[1].wait(_WAIT)
                third_started_early.append(started[2].wait(0.3))
            return _processed(filing_id)

        manager._fetcher.fetch_filing_content.side_effect = fetch
        manager._orchestrator.process_filing.side_effect = process

        manager._execute(make_task_info(state=TaskState.RUNNING))

        assert third_started_early == [False]
        assert manager._fetcher.fetch_filing_content.call_count == 3

    def test_duplicates_are_never_fetched(self, manager):
        work = [_filing(0), _filing(1), _filing(2)]
        manager._build_work_list = MagicMock(return_value=work)
        manager._registry.get_existing_accessions.return_value = {work[1].accession_number}
        manager._fetcher.fetch_filing_content.side_effect = lambda fi: (
            fi.to_identifier(),
            "<html></html>",
        )
        manager._orchestrator.process_filing.side_effect = lambda fid, html, **kw: _processed(fid)
        info = make_task_info(state=TaskState.RUNNING)

        manager._execute(info)

        fetched = [c.args[0] for c in manager._fetcher.fetch_filing_content.call_args_list]
        assert fetched == [work[0], work[2]]
        assert info.progress.filings_skipped == 1
        types = [m["type"] for m in _messages(info)]
        assert types == ["filing_done", "filing_skipped", "filing_done", "completed"]


# -----------------------------------------------------------------------
# EDGAR identity
# -----------------------------------------------------------------------


class TestEdgarIdentity:
    """Prefetches run off the worker thread but inside the identity guard."""

    def test_each_fetch_applies_task_identity_in_prefetch_thread(self, manager):
        work = [_filing(0), _filing(1)]
        manager._build_work_list = MagicMock(return_value=work)
        threads: list[str] = []
        guarded: list[bool] = []

        def fetch(fi):
            threads.append(threading.current_thread().name)
            guarded.append(manager._edgar_lock.locked())
            return fi.to_identifier(), "<html></html>"

        manager._fetcher.fetch_filing_content.side_effect = fetch
        manager._orchestrator.process_filing.side_effect = lambda fid, html, **kw: _processed(fid)
        info = make_task_info(state=TaskState.RUNNING)
        info.edgar_name = "Jane Analyst"
        info.edgar_email = "jane@example.com"

        manager._execute(info)

        assert all(name.startswith("prefetch-") for name in threads)
        assert guarded == [True, True]
        # Once for the work list, then once per fetch.
        assert manager._fetcher.apply_identity.call_count == 3
        manager._fetcher.apply_identity.assert_called_with("Jane Analyst", "jane@example.com")


# -----------------------------------------------------------------------
# Error semantics
# -----------------------------------------------------------------------


class TestErrors:
    """Fetch failures behave exactly as before the prefetch."""

    def test_fetch_error_fails_one_filing_and_continues(self, manager):
        work = [_filing(0), _filing(1)]
        manager._build_work_list = MagicMock(return_value=work)

        def fetch(fi):
            if fi is work[0]:
                raise FetchError("EDGAR unavailable")
            return fi.to_identifier(), "<html></html>"

        manager._fetcher.fetch_filing_content.side_effect = fetch
        manager._orchestrator.process_filing.side_effect = lambda fid, html, **kw: _processed(fid)
        info = make_task_info(state=TaskState.RUNNING)

        manager._execute(info)

        assert info.state == TaskState.COMPLETED
        assert info.progress.filings_failed == 1
        assert [r.accession_number for r in info.results] == [work[1].accession_number]
        failed = [m for m in _messages(info) if m["type"] == "filing_failed"]
        assert failed[0]["error"] == "EDGAR unavailable"

    def test_unexpected_fetch_error_still_escapes(self, manager):
        manager._build_work_list = MagicMock(return_value=[_filing(0)])
        manager._fetcher.fetch_filing_content.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            manager._execute(make_task_info(state=TaskState.RUNNING))

        manager._orchestrator.process_filing.assert_not_called()


# -----------------------------------------------------------------------
# Cancellation
# -----------------------------------------------------------------------


class TestCancellation:
    """A cancel never starts a new fetch and never waits on a running one."""

    def test_cancel_during_fetch_skips_the_next_fetch(self, manager):
        work = [_filing(0), _filing(1)]
        manager._build_work_list = MagicMock(return_value=work)
        info = make_task_info(state=TaskState.RUNNING)

        def fetch(fi):
            info.cancel_event.set()  # user cancels while filing 0 downloads
            return fi.to_identifier(), "<html></html>"

        manager._fetcher.fetch_filing_content.side_effect = fetch
        manager._orchestrator.process_filing.side_effect = lambda fid, html, **kw: _processed(fid)

        manager._execute(info)

        assert info.state == TaskState.CANCELLED
        assert manager._fetcher.fetch_filing_content.call_count == 1
        manager._registry.register_filing_if_new.assert_not_called()

    def test_cancel_raised_by_take_rolls_back(self, manager):
        manager._build_work_list = MagicMock(return_value=[_filing(0)])
        manager._rollback = MagicMock()
        info = make_task_info(state=TaskState.RUNNING)

        with patch.object(_FilingPrefetcher, "take", side_effect=_CancelledError):
            manager._execute(info)

        assert info.state == TaskState.CANCELLED
        manager._rollback.assert_called_once_with(info)
        assert _messages(info)[-1] == {"type": "cancelled"}
        manager._orchestrator.process_filing.assert_not_called()


# -----------------------------------------------------------------------
# _FilingPrefetcher on its own
# -----------------------------------------------------------------------


class TestPrefetcher:
    """Direct tests of the prefetch helper."""

    def test_take_out_of_order_raises(self):
        a, b = _filing(0), _filing(1)
        with _FilingPrefetcher(
            lambda fi: (None, ""), [a, b], cancel_event=threading.Event(), thread_name="t"
        ) as prefetcher:
            with pytest.raises(RuntimeError, match="order"):
                prefetcher.take(b)

    def test_cancelled_before_start_never_calls_fetch(self):
        fetch = MagicMock()
        cancel = threading.Event()
        cancel.set()
        with _FilingPrefetcher(fetch, [_filing(0)], cancel_event=cancel, thread_name="t") as p:
            with pytest.raises(_CancelledError):
                p.take(p._order[0])
        fetch.assert_not_called()

    def test_close_does_not_wait_for_running_fetch(self):
        release = threading.Event()
        running = threading.Event()
        order = [_filing(0), _filing(1)]

        def fetch(fi):
            if fi is order[1]:
                running.set()
                release.wait(_WAIT)
            return None, "<html></html>"

        prefetcher = _FilingPrefetcher(
            fetch, order, cancel_event=threading.Event(), thread_name="t"
        )
        try:
            prefetcher.take(order[0])
            assert running.wait(_WAIT)
            start = time.monotonic()
            prefetcher.close()
            assert time.monotonic() - start < 1.0
        finally:
            release.set()
