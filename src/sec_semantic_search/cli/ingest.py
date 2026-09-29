"""Ingest subcommands for adding SEC filings to the database."""

from collections.abc import Iterator
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sec_semantic_search.config import DEFAULT_FORM_TYPES, parse_form_types
from sec_semantic_search.core import (
    DatabaseError,
    FetchError,
    FilingIdentifier,
    FilingLimitExceededError,
    SECSemanticSearchError,
)
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry
from sec_semantic_search.pipeline import PipelineOrchestrator
from sec_semantic_search.pipeline.fetch import FilingFetcher

console = Console()

ingest_app = typer.Typer(no_args_is_help=True)

# Step labels used in the progress display for ingestion.
_STEPS = ["Fetching", "Parsing", "Chunking", "Embedding", "Storing"]


def _print_error(
    label: str,
    message: str,
    *,
    details: str | None = None,
    hint: str | None = None,
) -> None:
    """Print a consistently formatted error with optional details and hint."""
    console.print(f"[red]{label}:[/red] {message}")
    if details:
        console.print(f"  [dim]{details}[/dim]")
    if hint:
        console.print(f"  [dim italic]Hint: {hint}[/dim italic]")


def _make_progress() -> Progress:
    """Create a Rich Progress instance for ingestion steps."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total} steps"),
        TimeElapsedColumn(),
        console=console,
    )


def _validate_date(value: str | None, param_name: str) -> str | None:
    """
    Validate a date string in YYYY-MM-DD format.

    Returns the value unchanged if valid, or raises ``typer.BadParameter``.
    """
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise typer.BadParameter(
            f"Invalid date format for {param_name}: '{value}'. Expected YYYY-MM-DD."
        ) from None
    return value


def _fetch_filings(
    fetcher: FilingFetcher,
    ticker: str,
    form_type: str,
    *,
    count: int | None = 1,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[tuple[FilingIdentifier, str]]:
    """
    Fetch filing(s) using the appropriate FilingFetcher method.

    Yields ``(FilingIdentifier, html_content)`` tuples.  For *count=1* with
    no filters the fast-path ``fetch_latest()`` is used; for *count=1* with
    filters ``fetch_one()`` is used; for *count > 1* or *count=None* the
    ``fetch()`` generator is used (``None`` means all available within
    filters, capped by ``max_filings``).
    """
    has_filters = year is not None or start_date is not None or end_date is not None

    if count == 1 and not has_filters:
        yield fetcher.fetch_latest(ticker, form_type)
    elif count == 1 and has_filters:
        yield fetcher.fetch_one(
            ticker,
            form_type,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        yield from fetcher.fetch(
            ticker,
            form_type,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )


def _ingest_one_form(
    ticker: str,
    form_type: str,
    *,
    count: int | None = 1,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FilingFetcher,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    chroma: ChromaDBClient,
    progress: Progress,
    step_task_id: int,
    filing_task_id: int | None = None,
    form_label: str = "",
) -> tuple[int, int, int]:
    """
    Ingest filing(s) for one ticker and one form type.

    Runs the full pipeline per filing: fetch → duplicate check → process →
    store.  When *count* is 1 (default) the behaviour is identical to the
    previous single-filing flow.  When *count* is ``None`` all available
    filings matching the filters are fetched (capped by ``max_filings``).

    Args:
        ticker: Uppercased stock ticker symbol.
        form_type: Single validated form type (e.g. "10-K").
        count: Number of filings to ingest, or ``None`` for all matching.
        year: Optional filing-year filter.
        start_date: Optional start-date filter (YYYY-MM-DD).
        end_date: Optional end-date filter (YYYY-MM-DD).
        registry: MetadataRegistry instance.
        chroma: ChromaDBClient instance.
        progress: Active Rich Progress instance.
        step_task_id: Progress task ID for the 5-step pipeline bar.
        filing_task_id: Optional outer progress task ID (multi-filing only).
        form_label: Optional suffix for progress descriptions (e.g. " (1/2)").

    Returns:
        Tuple of (succeeded, skipped, failed) counts.
    """
    multi = count is None or count > 1

    # --- Fetch ---------------------------------------------------------------
    progress.update(
        step_task_id,
        description=f"Fetching {ticker} {form_type}{form_label}...",
    )
    try:
        filings_iter = _fetch_filings(
            fetcher,
            ticker,
            form_type,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
        # Materialise in advance so FetchError surfaces here, not mid-loop.
        filings = list(filings_iter)
    except FetchError as e:
        progress.stop()
        _print_error(
            "Fetch failed",
            e.message,
            details=e.details,
            hint="Check the ticker symbol is valid and you have an internet connection.",
        )
        return 0, 0, 1

    if not filings:
        progress.stop()
        console.print(
            f"[yellow]No filings found[/yellow] for {ticker} {form_type} with the given filters."
        )
        return 0, 0, 0

    # Update filing-level bar total now that we know the actual count.
    if filing_task_id is not None:
        progress.update(filing_task_id, total=len(filings))

    progress.advance(step_task_id)

    succeeded = 0
    skipped = 0
    failed = 0

    # Batch duplicate check — one SQL query for the whole work list.
    existing = registry.get_existing_accessions([fid.accession_number for fid, _ in filings])

    for filing_idx, (filing_id, html_content) in enumerate(filings):
        filing_num = f" [{filing_idx + 1}/{len(filings)}]" if multi else ""

        # Filing-limit check before each filing (important for count > 1).
        if filing_idx > 0:
            try:
                registry.check_filing_limit()
            except FilingLimitExceededError:
                progress.stop()
                console.print(
                    f"[yellow]Filing limit reached[/yellow] after "
                    f"{succeeded} ingestion(s) — stopping."
                )
                break

        # Reset the step bar for subsequent filings.
        if filing_idx > 0:
            progress.update(step_task_id, completed=1)  # fetch step already done

        # --- Duplicate check -------------------------------------------------
        if filing_id.accession_number in existing:
            if multi:
                progress.console.print(
                    f"  [yellow]Already ingested{filing_num}:[/yellow] "
                    f"{ticker} {form_type} ({filing_id.date_str})"
                )
            else:
                progress.stop()
                console.print(
                    f"[yellow]Already ingested:[/yellow] {ticker} {form_type} "
                    f"({filing_id.date_str}, {filing_id.accession_number})"
                )
            skipped += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        # --- Process: parse → chunk → embed ----------------------------------
        def _on_progress(
            step: str,
            _current: int,
            _total: int,
            _fnum: str = filing_num,
        ) -> None:
            if step != "Complete":
                progress.update(
                    step_task_id,
                    description=f"{step} {ticker} {form_type}{form_label}{_fnum}...",
                )
                progress.advance(step_task_id)

        try:
            result = orchestrator.process_filing(
                filing_id,
                html_content,
                progress_callback=_on_progress,
            )
        except SECSemanticSearchError as e:
            if multi:
                progress.console.print(f"  [red]Processing failed{filing_num}:[/red] {e.message}")
            else:
                progress.stop()
                _print_error(
                    "Processing failed",
                    e.message,
                    details=e.details,
                    hint="If this is a memory error, try lowering EMBEDDING_BATCH_SIZE in .env.",
                )
            failed += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        # --- Store: ChromaDB first, then SQLite ------------------------------
        progress.update(
            step_task_id,
            description=f"Storing {ticker} {form_type}{form_label}{filing_num}...",
        )
        try:
            chroma.store_filing(result)
            registry.register_filing(
                result.filing_id,
                result.ingest_result.chunk_count,
                segments=result.segments,
            )
        except DatabaseError as e:
            if multi:
                progress.console.print(f"  [red]Storage failed{filing_num}:[/red] {e.message}")
            else:
                progress.stop()
                _print_error(
                    "Storage failed",
                    e.message,
                    hint="Check disk space and that the data directory is writable.",
                )
            failed += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        progress.advance(step_task_id)

        # --- Per-filing summary ----------------------------------------------
        stats = result.ingest_result
        if multi:
            progress.console.print(
                f"  [green]Ingested{filing_num}:[/green] {ticker} {form_type} "
                f"({filing_id.date_str})  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
        else:
            progress.stop()
            console.print(
                f"[green]Ingested:[/green] {ticker} {form_type} ({filing_id.date_str})\n"
                f"  Segments: {stats.segment_count}  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
        succeeded += 1
        if filing_task_id is not None:
            progress.advance(filing_task_id)

    return succeeded, skipped, failed


def _ingest_across_forms(
    ticker: str,
    form_types: tuple[str, ...],
    *,
    count: int,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FilingFetcher,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    chroma: ChromaDBClient,
) -> tuple[int, int, int]:
    """
    Ingest the *count* most recent filings across all *form_types*.

    Uses ``list_available()`` to preview filings across form types, merges
    them by date, selects the newest *count*, then fetches, processes, and
    stores each one.

    Returns:
        Tuple of (succeeded, skipped, failed) counts.
    """
    # --- List available filings across form types ----------------------------
    console.print(f"Listing available {ticker} filings across {', '.join(form_types)}...")
    selected = fetcher.list_available_across_forms(
        ticker,
        form_types,
        count=count,
        year=year,
        start_date=start_date,
        end_date=end_date,
    )

    if not selected:
        console.print(f"[yellow]No filings found[/yellow] for {ticker} with the given filters.")
        return 0, 0, 0

    console.print(
        f"Found {len(selected)} filing(s): "
        + ", ".join(f"{fi.form_type} ({fi.filing_date})" for fi in selected)
    )

    succeeded = 0
    skipped = 0
    failed = 0

    # Batch duplicate check — one SQL query for the whole work list.
    existing = registry.get_existing_accessions([fi.accession_number for fi in selected])

    with _make_progress() as progress:
        filing_task = progress.add_task(
            f"{ticker}: 0/{len(selected)} filings",
            total=len(selected),
        )
        step_task = progress.add_task(
            "Fetching...",
            total=len(_STEPS),
        )

        for filing_idx, fi in enumerate(selected):
            filing_num = f" [{filing_idx + 1}/{len(selected)}]"
            label = f"{ticker} {fi.form_type}"

            # Filing-limit check.
            try:
                registry.check_filing_limit()
            except FilingLimitExceededError:
                progress.stop()
                console.print(
                    f"[yellow]Filing limit reached[/yellow] after "
                    f"{succeeded} ingestion(s) — stopping."
                )
                break

            # Reset step bar for each filing.
            progress.update(
                step_task,
                completed=0,
                description=f"Fetching {label}{filing_num}...",
            )

            # Duplicate check (before expensive fetch).
            if fi.accession_number in existing:
                progress.console.print(
                    f"  [yellow]Already ingested{filing_num}:[/yellow] {label} ({fi.filing_date})"
                )
                skipped += 1
                progress.advance(filing_task)
                continue

            # Fetch HTML content for this specific filing.
            try:
                filing_id, html_content = fetcher.fetch_by_accession(
                    fi.ticker,
                    fi.form_type,
                    fi.accession_number,
                )
            except FetchError as e:
                progress.console.print(f"  [red]Fetch failed{filing_num}:[/red] {e.message}")
                failed += 1
                progress.advance(filing_task)
                continue

            progress.advance(step_task)

            # Process: parse → chunk → embed.
            def _on_progress(
                step: str,
                _current: int,
                _total: int,
                _label: str = label,
                _fnum: str = filing_num,
            ) -> None:
                if step != "Complete":
                    progress.update(
                        step_task,
                        description=f"{step} {_label}{_fnum}...",
                    )
                    progress.advance(step_task)

            try:
                result = orchestrator.process_filing(
                    filing_id,
                    html_content,
                    progress_callback=_on_progress,
                )
            except SECSemanticSearchError as e:
                progress.console.print(f"  [red]Processing failed{filing_num}:[/red] {e.message}")
                failed += 1
                progress.advance(filing_task)
                continue

            # Store: ChromaDB first, then SQLite.
            progress.update(
                step_task,
                description=f"Storing {label}{filing_num}...",
            )
            try:
                chroma.store_filing(result)
                registry.register_filing(
                    result.filing_id,
                    result.ingest_result.chunk_count,
                    segments=result.segments,
                )
            except DatabaseError as e:
                progress.console.print(f"  [red]Storage failed{filing_num}:[/red] {e.message}")
                failed += 1
                progress.advance(filing_task)
                continue

            progress.advance(step_task)

            stats = result.ingest_result
            progress.console.print(
                f"  [green]Ingested{filing_num}:[/green] {label} "
                f"({filing_id.date_str})  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
            succeeded += 1
            progress.advance(filing_task)

    return succeeded, skipped, failed


@ingest_app.command("add")
def add(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL).")],
    form: Annotated[
        str,
        typer.Option(
            "--form",
            "-f",
            help="SEC form type(s), comma-separated (e.g. 8-K, 10-K, 10-Q).",
        ),
    ] = DEFAULT_FORM_TYPES,
    total: Annotated[
        int | None,
        typer.Option(
            "--total",
            "-t",
            help="Total number of filings to ingest (across all form types, newest first).",
            min=1,
        ),
    ] = None,
    number: Annotated[
        int | None,
        typer.Option(
            "--number",
            "-n",
            help="Number of filings to ingest per form type.",
            min=1,
        ),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option("--year", "-y", help="Filter by filing year (e.g. 2023)."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Start date filter (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="End date filter (YYYY-MM-DD)."),
    ] = None,
) -> None:
    """
    Fetch and ingest SEC filing(s) for a company.

    Examples:

        sec-search ingest add AAPL

        sec-search ingest add AAPL -f 10-K

        sec-search ingest add AAPL -t 3

        sec-search ingest add AAPL -n 2 -f 10-K

        sec-search ingest add AAPL -y 2023

        sec-search ingest add AAPL --start-date 2022-01-01 --end-date 2023-12-31
    """
    ticker = ticker.upper()

    if total is not None and number is not None:
        console.print("[red]--total and --number are mutually exclusive.[/red]")
        raise typer.Exit(code=1)

    try:
        form_types = parse_form_types(form)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    registry = MetadataRegistry()
    chroma = ChromaDBClient()
    fetcher = FilingFetcher()
    orchestrator = PipelineOrchestrator()

    # --- Cross-form mode: -t (total across form types) -----------------------
    if total is not None:
        succeeded, skipped, failed = _ingest_across_forms(
            ticker,
            form_types,
            count=total,
            year=year,
            start_date=start_date,
            end_date=end_date,
            fetcher=fetcher,
            orchestrator=orchestrator,
            registry=registry,
            chroma=chroma,
        )

        if total > 1:
            console.print(
                f"\n[bold]Summary:[/bold] "
                f"[green]{succeeded} ingested[/green], "
                f"[yellow]{skipped} skipped[/yellow], "
                f"[red]{failed} failed[/red]"
            )

        if failed > 0 and succeeded == 0 and skipped == 0:
            raise typer.Exit(code=1)
        return

    # --- Per-form mode: -n or default -----------------------------------------
    # When filters narrow the results and no explicit count is given, fetch all
    # matching filings (capped by max_filings) rather than just the latest one.
    has_filters = year is not None or start_date is not None or end_date is not None
    if number is not None:
        effective_per_form: int | None = number
    elif has_filters:
        effective_per_form = None  # all matching within filters
    else:
        effective_per_form = 1  # default: latest only

    succeeded = 0
    skipped = 0
    failed = 0

    for idx, form_type in enumerate(form_types):
        # Check filing limit before each form type.
        try:
            registry.check_filing_limit()
        except FilingLimitExceededError as e:
            _print_error(
                "Filing limit reached",
                e.message,
                hint="Remove filings with 'sec-search manage remove' or raise the limit via DB_MAX_FILINGS.",
            )
            raise typer.Exit(code=1) from None

        form_label = f" ({idx + 1}/{len(form_types)})" if len(form_types) > 1 else ""

        with _make_progress() as progress:
            if effective_per_form == 1:
                # Single filing: one 5-step bar (unchanged default UX).
                step_task = progress.add_task(
                    f"Fetching {ticker} {form_type}{form_label}...",
                    total=len(_STEPS),
                )
                s, sk, f = _ingest_one_form(
                    ticker,
                    form_type,
                    count=1,
                    year=year,
                    start_date=start_date,
                    end_date=end_date,
                    fetcher=fetcher,
                    orchestrator=orchestrator,
                    registry=registry,
                    chroma=chroma,
                    progress=progress,
                    step_task_id=step_task,
                    form_label=form_label,
                )
            else:
                # Multiple filings (or all matching): dual bars.
                # When count is None the total is unknown until fetch;
                # _ingest_one_form updates it after materialisation.
                estimated = effective_per_form or 0
                filing_task = progress.add_task(
                    f"{ticker} {form_type}{form_label}: filings",
                    total=estimated or None,
                )
                step_task = progress.add_task(
                    f"Fetching {ticker} {form_type}{form_label}...",
                    total=len(_STEPS),
                )
                s, sk, f = _ingest_one_form(
                    ticker,
                    form_type,
                    count=effective_per_form,
                    year=year,
                    start_date=start_date,
                    end_date=end_date,
                    fetcher=fetcher,
                    orchestrator=orchestrator,
                    registry=registry,
                    chroma=chroma,
                    progress=progress,
                    step_task_id=step_task,
                    filing_task_id=filing_task,
                    form_label=form_label,
                )

        succeeded += s
        skipped += sk
        failed += f

    # Show a combined summary when multiple form types or filings requested.
    if len(form_types) > 1 or effective_per_form != 1:
        console.print(
            f"\n[bold]Summary:[/bold] "
            f"[green]{succeeded} ingested[/green], "
            f"[yellow]{skipped} skipped[/yellow], "
            f"[red]{failed} failed[/red]"
        )

    if failed > 0 and succeeded == 0 and skipped == 0:
        raise typer.Exit(code=1)


@ingest_app.command("batch")
def batch(
    tickers: Annotated[
        list[str],
        typer.Argument(help="Stock ticker symbols (e.g. AAPL MSFT GOOGL)."),
    ],
    form: Annotated[
        str,
        typer.Option(
            "--form",
            "-f",
            help="SEC form type(s), comma-separated (e.g. 8-K, 10-K, 10-Q).",
        ),
    ] = DEFAULT_FORM_TYPES,
    total: Annotated[
        int | None,
        typer.Option(
            "--total",
            "-t",
            help="Total filings per ticker (across form types, newest first).",
            min=1,
        ),
    ] = None,
    number: Annotated[
        int | None,
        typer.Option(
            "--number",
            "-n",
            help="Number of filings per ticker per form type.",
            min=1,
        ),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option("--year", "-y", help="Filter by filing year (e.g. 2023)."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Start date filter (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="End date filter (YYYY-MM-DD)."),
    ] = None,
) -> None:
    """
    Fetch and ingest filings for multiple companies.

    Examples:

        sec-search ingest batch AAPL MSFT GOOGL

        sec-search ingest batch AAPL MSFT -f 10-K

        sec-search ingest batch AAPL MSFT -t 3

        sec-search ingest batch AAPL MSFT GOOGL -n 2 -y 2023
    """
    tickers = [t.upper() for t in tickers]

    if total is not None and number is not None:
        console.print("[red]--total and --number are mutually exclusive.[/red]")
        raise typer.Exit(code=1)

    try:
        form_types = parse_form_types(form)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    registry = MetadataRegistry()
    chroma = ChromaDBClient()
    fetcher = FilingFetcher()
    orchestrator = PipelineOrchestrator()

    total_succeeded = 0
    total_skipped = 0
    total_failed = 0

    # --- Cross-form mode: -t (total per ticker across form types) ------------
    if total is not None:
        for ticker in tickers:
            console.print(f"\n[bold]{ticker}[/bold]")
            s, sk, f = _ingest_across_forms(
                ticker,
                form_types,
                count=total,
                year=year,
                start_date=start_date,
                end_date=end_date,
                fetcher=fetcher,
                orchestrator=orchestrator,
                registry=registry,
                chroma=chroma,
            )
            total_succeeded += s
            total_skipped += sk
            total_failed += f

        console.print(
            f"\n[bold]Batch complete:[/bold] "
            f"[green]{total_succeeded} ingested[/green], "
            f"[yellow]{total_skipped} skipped[/yellow], "
            f"[red]{total_failed} failed[/red]"
        )
        return

    # --- Per-form mode: -n or default -----------------------------------------
    has_filters = year is not None or start_date is not None or end_date is not None
    if number is not None:
        effective_per_form: int | None = number
    elif has_filters:
        effective_per_form = None  # all matching within filters
    else:
        effective_per_form = 1  # default: latest only

    # Build a flat work list of (ticker, form_type) pairs.
    work_items = [(t, f) for t in tickers for f in form_types]

    with _make_progress() as progress:
        overall = progress.add_task(f"Batch: 0/{len(work_items)}", total=len(work_items))
        step_task = progress.add_task("Waiting...", total=len(_STEPS), visible=False)

        for i, (ticker, form_type) in enumerate(work_items):
            label = f"{ticker} {form_type}"

            # Check filing limit before each work item.
            try:
                registry.check_filing_limit()
            except FilingLimitExceededError as e:
                progress.stop()
                _print_error(
                    "Filing limit reached",
                    e.message,
                    hint="Remove filings with 'sec-search manage remove' or raise the limit via DB_MAX_FILINGS.",
                )
                break

            progress.update(overall, description=f"Batch: {i + 1}/{len(work_items)} — {label}")
            progress.update(
                step_task,
                description=f"Fetching {label}...",
                completed=0,
                visible=True,
            )

            # Fetch all filings for this work item.
            try:
                filings = list(
                    _fetch_filings(
                        fetcher,
                        ticker,
                        form_type,
                        count=effective_per_form,
                        year=year,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            except FetchError as e:
                progress.console.print(f"  [red]{label}: Fetch failed —[/red] {e.message}")
                total_failed += 1
                progress.advance(overall)
                continue

            if not filings:
                progress.console.print(
                    f"  [yellow]{label}: No filings found with the given filters[/yellow]"
                )
                progress.advance(overall)
                continue

            progress.advance(step_task)

            # Batch duplicate check for this work item's filings.
            existing = registry.get_existing_accessions(
                [fid.accession_number for fid, _ in filings]
            )

            # Process each filing within this work item.
            for filing_idx, (filing_id, html_content) in enumerate(filings):
                multi = len(filings) > 1
                filing_num = f" [{filing_idx + 1}/{len(filings)}]" if multi else ""

                # Filing-limit check before each filing (for per_form > 1).
                if filing_idx > 0:
                    try:
                        registry.check_filing_limit()
                    except FilingLimitExceededError:
                        progress.console.print(
                            f"  [yellow]{label}: Filing limit reached[/yellow] "
                            f"after {filing_idx} filing(s)"
                        )
                        break

                # Reset step bar for subsequent filings.
                if filing_idx > 0:
                    progress.update(step_task, completed=1)

                # Duplicate check.
                if filing_id.accession_number in existing:
                    progress.console.print(
                        f"  [yellow]{label}{filing_num}: Already ingested[/yellow] "
                        f"({filing_id.date_str})"
                    )
                    total_skipped += 1
                    continue

                # Process — wire orchestrator callback to progress bar.
                def _on_progress(
                    step: str,
                    _current: int,
                    _total: int,
                    _label: str = label,
                    _filing_num: str = filing_num,
                ) -> None:
                    if step != "Complete":
                        progress.update(
                            step_task,
                            description=f"{step} {_label}{_filing_num}...",
                        )
                        progress.advance(step_task)

                try:
                    result = orchestrator.process_filing(
                        filing_id,
                        html_content,
                        progress_callback=_on_progress,
                    )
                except SECSemanticSearchError as e:
                    progress.console.print(
                        f"  [red]{label}{filing_num}: Processing failed —[/red] {e.message}"
                    )
                    total_failed += 1
                    continue

                # Store.
                progress.update(step_task, description=f"Storing {label}{filing_num}...")
                try:
                    chroma.store_filing(result)
                    registry.register_filing(
                        result.filing_id,
                        result.ingest_result.chunk_count,
                        segments=result.segments,
                    )
                except DatabaseError as e:
                    progress.console.print(
                        f"  [red]{label}{filing_num}: Storage failed —[/red] {e.message}"
                    )
                    total_failed += 1
                    continue

                progress.advance(step_task)

                stats = result.ingest_result
                progress.console.print(
                    f"  [green]{label}{filing_num}:[/green] {filing_id.date_str}  |  "
                    f"Chunks: {stats.chunk_count}  |  "
                    f"Time: {stats.duration_seconds:.1f}s"
                )
                total_succeeded += 1

            progress.advance(overall)

        # Hide the per-filing step bar at the end.
        progress.update(step_task, visible=False)

    # Summary.
    console.print(
        f"\n[bold]Batch complete:[/bold] "
        f"[green]{total_succeeded} ingested[/green], "
        f"[yellow]{total_skipped} skipped[/yellow], "
        f"[red]{total_failed} failed[/red]"
    )
