"""
Two-phase ingest shared by the CLI and the API worker.

``plan_work()`` lists filing metadata — no HTML — for a request.
``run_ingest()`` then, for each listed filing:

    duplicate check → filing limit → fetch (one filing ahead) → parse
    → chunk → embed → store

The caller supplies the fetch callable (the API wraps it in its EDGAR
identity guard) and an ``IngestObserver`` for progress output: Rich
progress bars in the CLI, WebSocket messages in the API.

Store order (AD#3): SQLite first, through the atomic
``register_filing_if_new()``, which also catches a duplicate registered
after the batch check; then ChromaDB. When ChromaDB fails the SQLite row
is removed again, so neither store keeps a filing the other lacks.
Deletes stay ChromaDB first (``delete_filings_batch``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from sec_semantic_search.core import (
    DatabaseError,
    FetchError,
    FilingIdentifier,
    FilingLimitExceededError,
    SECSemanticSearchError,
    get_logger,
)
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry
from sec_semantic_search.pipeline import PipelineOrchestrator, ProcessedFiling
from sec_semantic_search.pipeline.fetch import FilingFetcher, FilingInfo

logger = get_logger(__name__)

# Stepper positions, shared with the frontend's five-step tracker.
STEP_FETCHING = 0
STEP_STORING = 4
STEP_TOTAL = 5

FetchFn = Callable[[FilingInfo], tuple[FilingIdentifier, str]]


class IngestCancelled(Exception):
    """The cancel event was set; the caller rolls back and stops."""


@dataclass
class IngestSummary:
    """Outcome counts for one ``run_ingest()`` call."""

    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    # Set when the run stopped early because the filing limit was reached
    # and ``make_room`` (if any) could not free a slot.
    limit_error: FilingLimitExceededError | None = None


class IngestObserver:
    """Receives ingest events. Every hook is a no-op; override what you need.

    ``position`` counts from 0 over the whole work list, duplicates
    included. Hooks run on the ingest thread, never the prefetch thread.
    """

    def listing(self, ticker: str, form_type: str | None) -> None:
        """Filing metadata is being listed (``form_type`` is None across forms)."""

    def listing_failed(self, ticker: str, form_type: str, error: FetchError) -> None:
        """Listing one ticker's filings of one form failed; planning continues."""

    def filing_started(self, position: int, total: int, filing: FilingInfo) -> None:
        """A filing from the work list is up next."""

    def step(self, filing: FilingInfo, label: str, index: int) -> None:
        """The filing reached stepper position ``index`` (0 Fetching … 4 Storing)."""

    def skipped(self, filing: FilingInfo, reason: str) -> None:
        """The filing was already stored (``reason`` is ``"duplicate"``)."""

    def failed(self, filing: FilingInfo, stage: str, error: SECSemanticSearchError) -> None:
        """``stage`` is ``"fetch"``, ``"processing"`` or ``"storage"``."""

    def done(self, filing: FilingInfo, result: ProcessedFiling) -> None:
        """The filing is in both stores."""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def per_form_count(
    count_mode: str,
    count: int | None,
    *,
    has_filters: bool,
) -> int | None:
    """How many filings to list per form type (``None`` = all matching).

    ``per_form`` with a count takes it; with filters and no count, every
    filing matching them (``list_available`` caps at ``max_filings``);
    otherwise the explicit count or the latest one.
    """
    if count_mode == "per_form" and count is not None:
        return count
    if has_filters and count is None:
        return None
    if count is not None:
        return count
    return 1


def plan_work(
    fetcher: FilingFetcher,
    tickers: Sequence[str],
    form_types: Sequence[str],
    *,
    count_mode: str = "latest",
    count: int | None = None,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    observer: IngestObserver | None = None,
    cancel_event: threading.Event | None = None,
) -> list[FilingInfo]:
    """
    List the filings a request covers, as metadata only.

    ``count_mode="total"`` takes the newest ``count`` filings per ticker
    across ``form_types``; otherwise each ticker × form type is listed
    with ``per_form_count()``. A listing that fails for one form type is
    reported to ``observer.listing_failed`` and skipped.
    """
    observer = observer or IngestObserver()
    has_filters = year is not None or start_date is not None or end_date is not None
    work: list[FilingInfo] = []

    for ticker in tickers:
        if cancel_event is not None and cancel_event.is_set():
            break

        if count_mode == "total" and count is not None:
            observer.listing(ticker, None)
            work.extend(
                fetcher.list_available_across_forms(
                    ticker,
                    tuple(form_types),
                    count=count,
                    year=year,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            continue

        for form_type in form_types:
            if cancel_event is not None and cancel_event.is_set():
                break
            observer.listing(ticker, form_type)
            try:
                work.extend(
                    fetcher.list_available(
                        ticker,
                        form_type,
                        count=per_form_count(count_mode, count, has_filters=has_filters),
                        year=year,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            except FetchError as exc:
                observer.listing_failed(ticker, form_type, exc)

    return work


# ---------------------------------------------------------------------------
# One-ahead fetch
# ---------------------------------------------------------------------------


class FilingPrefetcher:
    """
    Fetch filing HTML one filing ahead of the ingest loop.

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
        fetch: FetchFn,
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

    def __enter__(self) -> FilingPrefetcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def take(self, filing_info: FilingInfo) -> tuple[FilingIdentifier, str]:
        """Wait for ``filing_info``'s HTML, then start fetching the next filing.

        Raises:
            IngestCancelled: If the run was cancelled before the fetch began.
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
            raise IngestCancelled
        return self._fetch(filing_info)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def store_processed_filing(
    result: ProcessedFiling,
    *,
    registry: MetadataRegistry,
    chroma: ChromaDBClient,
) -> bool:
    """
    Store one processed filing in both stores (AD#3 store order).

    Returns ``False`` without writing anything when the filing is already
    registered (a duplicate that appeared after the batch check).

    Raises:
        DatabaseError: If either store fails; SQLite is rolled back first
            when ChromaDB is the one that failed.
    """
    if not registry.register_filing_if_new(
        result.filing_id,
        result.ingest_result.chunk_count,
        segments=result.segments,
    ):
        return False
    try:
        chroma.store_filing(result)
    except Exception:
        try:
            registry.remove_filing(result.filing_id.accession_number)
        except DatabaseError as rollback_error:
            # Keep the original error; the orphaned row is logged for repair.
            logger.error(
                "Rollback of %s failed after a ChromaDB error — %s",
                result.filing_id.accession_number,
                rollback_error.message,
            )
        raise
    return True


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_ingest(
    work: Sequence[FilingInfo],
    *,
    fetch: FetchFn,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    chroma: ChromaDBClient,
    max_filings: int,
    observer: IngestObserver | None = None,
    make_room: Callable[[int], None] | None = None,
    cancel_event: threading.Event | None = None,
    prefetch_thread_name: str = "prefetch",
) -> IngestSummary:
    """
    Ingest every filing in ``work``, skipping those already stored.

    Args:
        work: Filing metadata from ``plan_work()``.
        fetch: Returns a filing's identifier and HTML. Runs on a prefetch
            thread, one filing ahead; wrap it in whatever guard the
            caller needs (the API applies its per-task EDGAR identity).
        max_filings: The registry's capacity.
        observer: Progress and outcome hooks.
        make_room: Called with the number of slots needed when the batch
            would exceed ``max_filings`` (before the loop, for the whole
            batch) or when a filing finds the store full. The API's demo
            mode evicts the oldest filings here; without it the run stops.
        cancel_event: Checked between filings, between pipeline steps and
            before storing. When set, ``IngestCancelled`` is raised and
            the caller rolls back what ``observer.done`` reported.

    Returns:
        The outcome counts; ``limit_error`` is set if the store filled up.

    Raises:
        IngestCancelled: When ``cancel_event`` is set.
    """
    observer = observer or IngestObserver()
    cancel = cancel_event or threading.Event()
    summary = IngestSummary()
    if not work:
        return summary

    # One SQL query for the whole work list.
    existing = registry.get_existing_accessions([fi.accession_number for fi in work])
    to_fetch = [fi for fi in work if fi.accession_number not in existing]
    if make_room is not None and to_fetch:
        make_room(len(to_fetch))
    # Counted once and then kept current here: only this run stores, and
    # ``make_room`` is followed by a fresh count.
    stored = registry.count()

    def check_cancel() -> None:
        if cancel.is_set():
            raise IngestCancelled

    # Duplicates are never fetched; the next filing downloads while this
    # one is parsed, embedded and stored.
    with FilingPrefetcher(
        fetch, to_fetch, cancel_event=cancel, thread_name=prefetch_thread_name
    ) as prefetcher:
        for position, filing in enumerate(work):
            check_cancel()
            observer.filing_started(position, len(work), filing)

            if filing.accession_number in existing:
                summary.skipped += 1
                observer.skipped(filing, "duplicate")
                continue

            if stored >= max_filings and make_room is not None:
                make_room(1)
                stored = registry.count()
            if stored >= max_filings:
                summary.limit_error = FilingLimitExceededError(stored, max_filings)
                break

            observer.step(filing, "Fetching", STEP_FETCHING)
            try:
                filing_id, html_content = prefetcher.take(filing)
            except FetchError as exc:
                summary.failed += 1
                observer.failed(filing, "fetch", exc)
                continue

            def on_progress(step: str, current: int, _total: int, _filing=filing) -> None:
                # "Complete" (4) is followed at once by "Storing" (4).
                if step != "Complete":
                    observer.step(_filing, step, current)
                check_cancel()

            try:
                result = orchestrator.process_filing(
                    filing_id,
                    html_content,
                    progress_callback=on_progress,
                )
            except SECSemanticSearchError as exc:
                summary.failed += 1
                observer.failed(filing, "processing", exc)
                continue

            check_cancel()
            observer.step(filing, "Storing", STEP_STORING)
            try:
                registered = store_processed_filing(result, registry=registry, chroma=chroma)
            except DatabaseError as exc:
                summary.failed += 1
                observer.failed(filing, "storage", exc)
                continue

            if not registered:
                summary.skipped += 1
                observer.skipped(filing, "duplicate")
                continue

            stored += 1
            summary.succeeded += 1
            observer.done(filing, result)

    return summary
