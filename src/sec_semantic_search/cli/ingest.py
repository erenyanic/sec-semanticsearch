"""Ingest subcommands for adding SEC filings to the database.

Both commands run the shared two-phase ingest (``sec_semantic_search.ingest``)
that the API worker uses: list filing metadata, skip duplicates, fetch one
filing ahead, parse → chunk → embed, store SQLite first then ChromaDB.
This module only maps options to a request and renders progress.
"""

from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from sec_semantic_search.config import DEFAULT_FORM_TYPES, get_settings, parse_form_types
from sec_semantic_search.core import FetchError, SECSemanticSearchError
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry
from sec_semantic_search.ingest import (
    STEP_TOTAL,
    IngestObserver,
    IngestSummary,
    plan_work,
    run_ingest,
)
from sec_semantic_search.pipeline import PipelineOrchestrator, ProcessedFiling
from sec_semantic_search.pipeline.fetch import FilingFetcher, FilingInfo

console = Console()

ingest_app = typer.Typer(no_args_is_help=True)

# Printed after a failure, by stage.
_FAILURE_HINTS = {
    "fetch": "Check the ticker symbol is valid and you have an internet connection.",
    "processing": "If this is a memory error, try lowering EMBEDDING_BATCH_SIZE in .env.",
    "storage": "Check disk space and that the data directory is writable.",
}


def _make_progress() -> Progress:
    """Create a Rich Progress instance for ingestion steps."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
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


class _CliReporter(IngestObserver):
    """Renders ingest events as Rich progress bars and one line per filing."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._filings: TaskID = progress.add_task("Listing filings...", total=None)
        self._steps: TaskID = progress.add_task("", total=STEP_TOTAL, visible=False)
        self._tag = ""
        self._total = 0
        self.listing_failures = 0

    @staticmethod
    def _label(filing: FilingInfo) -> str:
        # Escaped: labels go into Rich markup (console lines and progress
        # descriptions), and EDGAR supplies the form type.
        return escape(f"{filing.ticker} {filing.form_type}")

    def listing(self, ticker: str, form_type: str | None) -> None:
        forms = form_type or "all forms"
        self._progress.update(self._filings, description=escape(f"Listing {ticker} {forms}..."))

    def listing_failed(self, ticker: str, form_type: str, error: FetchError) -> None:
        self.listing_failures += 1
        self._progress.console.print(
            f"  [red]{escape(ticker)} {escape(form_type)}: listing failed —[/red] "
            f"{escape(error.message)}"
        )
        self._progress.console.print(f"  [dim italic]Hint: {_FAILURE_HINTS['fetch']}[/dim italic]")

    def filing_started(self, position: int, total: int, filing: FilingInfo) -> None:
        self._tag = f" [{position + 1}/{total}]" if total > 1 else ""
        self._total = total
        self._progress.update(
            self._filings,
            total=total,
            completed=position,
            description=f"Filings: {self._label(filing)}{self._tag}",
        )
        self._progress.update(self._steps, completed=0, visible=True)

    def step(self, filing: FilingInfo, label: str, index: int) -> None:
        self._progress.update(
            self._steps,
            completed=index,
            description=f"{label} {self._label(filing)}{self._tag}...",
        )

    def skipped(self, filing: FilingInfo, reason: str) -> None:
        self._progress.console.print(
            f"  [yellow]Already ingested{self._tag}:[/yellow] {self._label(filing)} "
            f"({filing.filing_date.isoformat()}, {escape(filing.accession_number)})"
        )

    def failed(self, filing: FilingInfo, stage: str, error: SECSemanticSearchError) -> None:
        self._progress.console.print(
            f"  [red]{stage.capitalize()} failed{self._tag}:[/red] {self._label(filing)} — "
            f"{escape(error.message)}"
        )
        if error.details:
            self._progress.console.print(f"    [dim]{escape(error.details)}[/dim]")
        self._progress.console.print(f"    [dim italic]Hint: {_FAILURE_HINTS[stage]}[/dim italic]")

    def done(self, filing: FilingInfo, result: ProcessedFiling) -> None:
        stats = result.ingest_result
        self._progress.update(self._steps, completed=STEP_TOTAL)
        self._progress.console.print(
            f"  [green]Ingested{self._tag}:[/green] {self._label(filing)} "
            f"({result.filing_id.date_str})  |  Segments: {stats.segment_count}  |  "
            f"Chunks: {stats.chunk_count}  |  Time: {stats.duration_seconds:.1f}s"
        )

    def finish(self) -> None:
        self._progress.update(self._filings, completed=self._total)
        self._progress.update(self._steps, visible=False)


def _run(
    tickers: list[str],
    form: str,
    *,
    total: int | None,
    number: int | None,
    year: int | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[IngestSummary, int]:
    """Validate options, then plan and run the ingest.

    Returns the summary and the number of listings that failed.
    """
    if total is not None and number is not None:
        console.print("[red]--total and --number are mutually exclusive.[/red]")
        raise typer.Exit(code=1)

    try:
        form_types = parse_form_types(form)
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(code=1) from None

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    # -t: newest N per ticker across forms; -n: N per form; default: the
    # latest per form, or everything matching when a date filter is given.
    if total is not None:
        count_mode, count = "total", total
    elif number is not None:
        count_mode, count = "per_form", number
    else:
        count_mode, count = "latest", None

    registry = MetadataRegistry()
    chroma = ChromaDBClient()
    fetcher = FilingFetcher()
    orchestrator = PipelineOrchestrator()

    with _make_progress() as progress:
        reporter = _CliReporter(progress)
        work = plan_work(
            fetcher,
            tickers,
            form_types,
            count_mode=count_mode,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
            observer=reporter,
        )
        if not work:
            progress.stop()
            console.print(
                f"[yellow]No filings found[/yellow] for {escape(', '.join(tickers))} "
                f"({', '.join(form_types)}) with the given filters."
            )
            return IngestSummary(), reporter.listing_failures

        summary = run_ingest(
            work,
            fetch=fetcher.fetch_filing_content,
            orchestrator=orchestrator,
            registry=registry,
            chroma=chroma,
            max_filings=get_settings().database.max_filings,
            observer=reporter,
        )
        reporter.finish()

    if summary.limit_error is not None:
        console.print(
            f"[yellow]Filing limit reached[/yellow] after {summary.succeeded} "
            f"ingestion(s) — stopping. {escape(summary.limit_error.message)}"
        )
        console.print(
            "  [dim italic]Hint: Remove filings with 'sec-search manage remove' or raise "
            "the limit via DB_MAX_FILINGS.[/dim italic]"
        )
    return summary, reporter.listing_failures


def _print_summary(title: str, summary: IngestSummary, failed: int) -> None:
    console.print(
        f"\n[bold]{title}:[/bold] "
        f"[green]{summary.succeeded} ingested[/green], "
        f"[yellow]{summary.skipped} skipped[/yellow], "
        f"[red]{failed} failed[/red]"
    )


def _exit_code(summary: IngestSummary, failed: int) -> int:
    """1 when nothing was ingested or already present and something failed."""
    nothing_done = summary.succeeded == 0 and summary.skipped == 0
    return 1 if nothing_done and (failed > 0 or summary.limit_error is not None) else 0


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
    summary, listing_failures = _run(
        [ticker.upper()],
        form,
        total=total,
        number=number,
        year=year,
        start_date=start_date,
        end_date=end_date,
    )
    failed = summary.failed + listing_failures
    if summary.succeeded + summary.skipped + failed > 1:
        _print_summary("Summary", summary, failed)
    code = _exit_code(summary, failed)
    if code:
        raise typer.Exit(code=code)


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
    summary, listing_failures = _run(
        [t.upper() for t in tickers],
        form,
        total=total,
        number=number,
        year=year,
        start_date=start_date,
        end_date=end_date,
    )
    _print_summary("Batch complete", summary, summary.failed + listing_failures)
