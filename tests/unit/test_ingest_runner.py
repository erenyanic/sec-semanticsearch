"""
Tests for the shared two-phase ingest (``sec_semantic_search.ingest``).

The CLI and the API worker both run ``plan_work()`` then ``run_ingest()``;
these tests pin the loop's behaviour once, independent of either caller:

    - Planning: cross-form and per-form listing, counts, listing failures
    - Duplicates are skipped and never fetched; late duplicates write nothing
    - Fetch, processing and storage failures fail one filing and continue
    - Store order: SQLite (atomic) first, then ChromaDB, rollback on failure
    - Filing limit: stop with ``limit_error``, or ``make_room`` and continue
    - Cancellation between filings, between steps and before storing
    - Observer events and stepper positions
"""

import threading
from datetime import date
from unittest.mock import MagicMock, call

import pytest

from sec_semantic_search.core.exceptions import (
    DatabaseError,
    EmbeddingError,
    FetchError,
    FilingLimitExceededError,
)
from sec_semantic_search.core.types import FilingIdentifier
from sec_semantic_search.ingest import (
    IngestCancelled,
    IngestObserver,
    plan_work,
    run_ingest,
)
from sec_semantic_search.pipeline.fetch import FilingInfo


def _info(n: int, form: str = "10-K") -> FilingInfo:
    return FilingInfo(
        ticker="AAPL",
        form_type=form,
        filing_date=date(2024, 1, 1 + n),
        accession_number=f"0000320193-24-{n:06d}",
        company_name="Apple Inc.",
    )


def _fetch(fi: FilingInfo) -> tuple[FilingIdentifier, str]:
    return (
        FilingIdentifier(fi.ticker, fi.form_type, fi.filing_date, fi.accession_number),
        f"<html>{fi.accession_number}</html>",
    )


def _processed(filing_id, html, progress_callback=None):
    for step, n in (("Parsing", 1), ("Chunking", 2), ("Embedding", 3), ("Complete", 4)):
        if progress_callback:
            progress_callback(step, n, 4)
    result = MagicMock()
    result.filing_id = filing_id
    result.ingest_result.chunk_count = 7
    return result


class _Recorder(IngestObserver):
    def __init__(self):
        self.events: list[tuple] = []

    def listing(self, ticker, form_type):
        self.events.append(("listing", ticker, form_type))

    def listing_failed(self, ticker, form_type, error):
        self.events.append(("listing_failed", ticker, form_type))

    def filing_started(self, position, total, filing):
        self.events.append(("started", position, total, filing.accession_number[-1]))

    def step(self, filing, label, index):
        self.events.append(("step", label, index))

    def skipped(self, filing, reason):
        self.events.append(("skipped", filing.accession_number[-1], reason))

    def failed(self, filing, stage, error):
        self.events.append(("failed", filing.accession_number[-1], stage))

    def done(self, filing, result):
        self.events.append(("done", filing.accession_number[-1]))


@pytest.fixture
def deps():
    registry = MagicMock()
    registry.get_existing_accessions.return_value = set()
    registry.count.return_value = 0
    registry.register_filing_if_new.return_value = True
    chroma = MagicMock()
    orchestrator = MagicMock()
    orchestrator.process_filing.side_effect = _processed
    return registry, chroma, orchestrator


def _run(work, deps, **kwargs):
    registry, chroma, orchestrator = deps
    kwargs.setdefault("fetch", _fetch)
    kwargs.setdefault("max_filings", 100)
    return run_ingest(
        work,
        orchestrator=orchestrator,
        registry=registry,
        chroma=chroma,
        **kwargs,
    )


# -----------------------------------------------------------------------
# plan_work
# -----------------------------------------------------------------------


class TestPlanWork:
    def test_total_mode_lists_across_forms_per_ticker(self):
        fetcher = MagicMock()
        fetcher.list_available_across_forms.side_effect = lambda t, *a, **k: [_info(len(t))]
        work = plan_work(fetcher, ["AAPL", "MSFT"], ["10-K", "10-Q"], count_mode="total", count=3)

        assert len(work) == 2
        assert fetcher.list_available_across_forms.call_args_list == [
            call("AAPL", ("10-K", "10-Q"), count=3, year=None, start_date=None, end_date=None),
            call("MSFT", ("10-K", "10-Q"), count=3, year=None, start_date=None, end_date=None),
        ]
        fetcher.list_available.assert_not_called()

    @pytest.mark.parametrize(
        ("count_mode", "count", "year", "expected"),
        [
            ("latest", None, None, 1),  # default: latest only
            ("per_form", 2, None, 2),
            ("latest", None, 2023, None),  # filters, no count: all matching
            ("latest", 4, 2023, 4),
        ],
    )
    def test_per_form_counts(self, count_mode, count, year, expected):
        fetcher = MagicMock()
        fetcher.list_available.return_value = []
        plan_work(
            fetcher, ["AAPL"], ["10-K", "10-Q"], count_mode=count_mode, count=count, year=year
        )

        assert [c.args for c in fetcher.list_available.call_args_list] == [
            ("AAPL", "10-K"),
            ("AAPL", "10-Q"),
        ]
        assert {c.kwargs["count"] for c in fetcher.list_available.call_args_list} == {expected}

    def test_listing_failure_is_reported_and_skipped(self):
        fetcher = MagicMock()
        fetcher.list_available.side_effect = [FetchError("down"), [_info(1, "10-Q")]]
        observer = _Recorder()
        work = plan_work(fetcher, ["AAPL"], ["10-K", "10-Q"], observer=observer)

        assert [fi.form_type for fi in work] == ["10-Q"]
        assert ("listing_failed", "AAPL", "10-K") in observer.events

    def test_cancel_stops_planning(self):
        fetcher = MagicMock()
        cancel = threading.Event()
        cancel.set()
        assert plan_work(fetcher, ["AAPL"], ["10-K"], cancel_event=cancel) == []
        fetcher.list_available.assert_not_called()


# -----------------------------------------------------------------------
# run_ingest — happy path and events
# -----------------------------------------------------------------------


class TestRunIngest:
    def test_ingests_and_reports_steps(self, deps):
        registry, chroma, _ = deps
        observer = _Recorder()
        summary = _run([_info(1)], deps, observer=observer)

        assert (summary.succeeded, summary.skipped, summary.failed) == (1, 0, 0)
        assert observer.events == [
            ("started", 0, 1, "1"),
            ("step", "Fetching", 0),
            ("step", "Parsing", 1),
            ("step", "Chunking", 2),
            ("step", "Embedding", 3),
            ("step", "Storing", 4),
            ("done", "1"),
        ]

    def test_store_order_is_sqlite_then_chromadb(self, deps):
        registry, chroma, _ = deps
        order: list[str] = []
        registry.register_filing_if_new.side_effect = lambda *a, **k: order.append("sqlite") or True
        chroma.store_filing.side_effect = lambda r: order.append("chroma")

        _run([_info(1)], deps)

        assert order == ["sqlite", "chroma"]
        registry.register_filing.assert_not_called()

    def test_empty_work_touches_nothing(self, deps):
        registry, _, orchestrator = deps
        summary = _run([], deps)
        assert (summary.succeeded, summary.skipped, summary.failed) == (0, 0, 0)
        registry.get_existing_accessions.assert_not_called()
        orchestrator.process_filing.assert_not_called()


class TestDuplicates:
    def test_duplicates_skipped_and_never_fetched(self, deps):
        registry, _, _ = deps
        work = [_info(1), _info(2), _info(3)]
        registry.get_existing_accessions.return_value = {work[1].accession_number}
        fetched: list[str] = []

        def fetch(fi):
            fetched.append(fi.accession_number)
            return _fetch(fi)

        observer = _Recorder()
        summary = _run(work, deps, fetch=fetch, observer=observer)

        assert fetched == [work[0].accession_number, work[2].accession_number]
        assert (summary.succeeded, summary.skipped) == (2, 1)
        assert ("skipped", "2", "duplicate") in observer.events
        registry.get_existing_accessions.assert_called_once()

    def test_late_duplicate_is_a_skip_and_writes_nothing(self, deps):
        registry, chroma, _ = deps
        registry.register_filing_if_new.return_value = False
        summary = _run([_info(1)], deps)

        assert (summary.succeeded, summary.skipped) == (0, 1)
        chroma.store_filing.assert_not_called()


class TestFailures:
    def test_fetch_failure_fails_one_filing(self, deps):
        def fetch(fi):
            if fi.accession_number.endswith("1"):
                raise FetchError("timeout")
            return _fetch(fi)

        observer = _Recorder()
        summary = _run([_info(1), _info(2)], deps, fetch=fetch, observer=observer)

        assert (summary.succeeded, summary.failed) == (1, 1)
        assert ("failed", "1", "fetch") in observer.events

    def test_processing_failure_fails_one_filing(self, deps):
        _, chroma, orchestrator = deps
        orchestrator.process_filing.side_effect = [
            EmbeddingError("OOM"),
            _processed(*_fetch(_info(2))),
        ]
        observer = _Recorder()
        summary = _run([_info(1), _info(2)], deps, observer=observer)

        assert (summary.succeeded, summary.failed) == (1, 1)
        assert ("failed", "1", "processing") in observer.events
        assert chroma.store_filing.call_count == 1

    def test_storage_failure_rolls_back_sqlite(self, deps):
        registry, chroma, _ = deps
        chroma.store_filing.side_effect = DatabaseError("disk full")
        observer = _Recorder()
        summary = _run([_info(1)], deps, observer=observer)

        assert summary.failed == 1
        registry.remove_filing.assert_called_once_with(_info(1).accession_number)
        assert ("failed", "1", "storage") in observer.events

    def test_other_errors_escape(self, deps):
        _, _, orchestrator = deps
        orchestrator.process_filing.side_effect = RuntimeError("bug")
        with pytest.raises(RuntimeError, match="bug"):
            _run([_info(1)], deps)


class TestFilingLimit:
    def test_stops_with_limit_error(self, deps):
        registry, _, orchestrator = deps
        registry.count.return_value = 1
        summary = _run([_info(1), _info(2), _info(3)], deps, max_filings=2)

        assert summary.succeeded == 1
        assert isinstance(summary.limit_error, FilingLimitExceededError)
        assert orchestrator.process_filing.call_count == 1

    def test_duplicates_need_no_capacity(self, deps):
        registry, _, _ = deps
        registry.count.return_value = 5
        registry.get_existing_accessions.return_value = {_info(1).accession_number}
        summary = _run([_info(1)], deps, max_filings=5)

        assert summary.skipped == 1
        assert summary.limit_error is None

    def test_make_room_before_the_batch_and_when_full(self, deps):
        registry, _, _ = deps
        registry.get_existing_accessions.return_value = {_info(3).accession_number}
        # Full at start; each in-loop make_room frees one slot.
        counts = iter([2, 1, 1])
        registry.count.side_effect = lambda: next(counts)
        make_room = MagicMock()

        summary = _run([_info(1), _info(2), _info(3)], deps, max_filings=2, make_room=make_room)

        # The whole batch of new filings first, then one slot each time the
        # store is found full.
        assert make_room.call_args_list == [call(2), call(1), call(1)]
        assert summary.succeeded == 2
        assert summary.limit_error is None


class TestCancellation:
    def test_cancel_before_start(self, deps):
        _, _, orchestrator = deps
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(IngestCancelled):
            _run([_info(1)], deps, cancel_event=cancel)
        orchestrator.process_filing.assert_not_called()

    def test_cancel_between_pipeline_steps(self, deps):
        _, chroma, orchestrator = deps
        cancel = threading.Event()

        def process(filing_id, html, progress_callback=None):
            progress_callback("Parsing", 1, 4)
            cancel.set()
            progress_callback("Chunking", 2, 4)
            raise AssertionError("not reached")

        orchestrator.process_filing.side_effect = process
        with pytest.raises(IngestCancelled):
            _run([_info(1)], deps, cancel_event=cancel)
        chroma.store_filing.assert_not_called()

    def test_cancel_after_processing_skips_the_store(self, deps):
        registry, chroma, orchestrator = deps
        cancel = threading.Event()

        def process(filing_id, html, progress_callback=None):
            cancel.set()  # set after the last step check
            return _processed(filing_id, html)

        orchestrator.process_filing.side_effect = process
        with pytest.raises(IngestCancelled):
            _run([_info(1)], deps, cancel_event=cancel)
        registry.register_filing_if_new.assert_not_called()
        chroma.store_filing.assert_not_called()

    def test_done_reported_before_a_later_cancel(self, deps):
        """What ``done`` reported is what a caller must roll back."""
        cancel = threading.Event()
        observer = _Recorder()
        original_done = observer.done

        def done(filing, result):
            original_done(filing, result)
            cancel.set()

        observer.done = done
        with pytest.raises(IngestCancelled):
            _run([_info(1), _info(2)], deps, cancel_event=cancel, observer=observer)
        assert [e for e in observer.events if e[0] == "done"] == [("done", "1")]
