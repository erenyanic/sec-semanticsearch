"""Search command for querying ingested SEC filings."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from sec_semantic_search.core import DatabaseError, SearchError
from sec_semantic_search.search import SearchEngine

console = Console()

# Maximum characters to display per result in the table.
_CONTENT_PREVIEW_LIMIT = 1000
_SECTION_PATH_LIMIT = 500


def _similarity_text(similarity: float) -> Text:
    """
    Return a colour-coded similarity percentage.

    Green for >= 40%, yellow for >= 25%, dim otherwise.
    These thresholds reflect typical cosine similarity ranges
    from the embedding model where raw scores are relatively low.
    """
    pct = f"{similarity:.1%}"
    if similarity >= 0.40:
        return Text(pct, style="bold green")
    if similarity >= 0.25:
        return Text(pct, style="yellow")
    return Text(pct, style="dim")


def search(
    query: Annotated[str, typer.Argument(help="Natural language search query.")],
    top: Annotated[
        int | None,
        typer.Option("--top", "-t", help="Number of results to return."),
    ] = None,
    ticker: Annotated[
        list[str] | None,
        typer.Option("--ticker", "-k", help="Filter by ticker symbol(s). Repeat for multiple."),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option("--form", "-f", help="Filter by form type(s). Repeat for multiple."),
    ] = None,
    accession: Annotated[
        list[str] | None,
        typer.Option(
            "--accession",
            "-a",
            help="Restrict search to specific filing(s) by accession number. Repeat for multiple.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date", help="Filter results to filings on or after this date (YYYY-MM-DD)."
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date", help="Filter results to filings on or before this date (YYYY-MM-DD)."
        ),
    ] = None,
) -> None:
    """
    Search ingested SEC filings with a natural language query.

    Examples:

        sec-search search "risk factors related to supply chain"

        sec-search search "revenue recognition" -t 10 -k AAPL

        sec-search search "revenue" -k AAPL -k MSFT

        sec-search search "liquidity" -f 10-Q

        sec-search search "debt covenants" -a 0000320193-23-000106

        sec-search search "revenue" --start-date 2023-01-01 --end-date 2023-12-31
    """
    with console.status("Searching..."):
        try:
            # Build a registry instance so the engine can resolve each
            # chunk back to its parent segment for display. The CLI does
            # not need to share the registry with any other singleton,
            # so we create a short-lived one here.
            from sec_semantic_search.database import MetadataRegistry

            engine = SearchEngine(registry=MetadataRegistry())
            # Normalise ticker(s) to uppercase; pass list or None.
            ticker_filter: list[str] | None = [t.upper() for t in ticker] if ticker else None
            form_filter: list[str] | None = [f.upper() for f in form] if form else None
            results = engine.search(
                query=query,
                top_k=top,
                ticker=ticker_filter,
                form_type=form_filter,
                accession_number=accession,
                start_date=start_date,
                end_date=end_date,
            )
        except DatabaseError as e:
            console.print(f"[red]Search failed:[/red] {e.message}")
            console.print(
                "  [dim italic]Hint: Check that the metadata database is readable. "
                "If DB_ENCRYPTION_KEY is set, install the encryption extra with "
                "'pip install sec-semantic-search[encryption]'.[/dim italic]"
            )
            raise typer.Exit(code=1) from None
        except SearchError as e:
            console.print(f"[red]Search failed:[/red] {e.message}")
            if e.details:
                console.print(f"  [dim]{e.details}[/dim]")
            console.print(
                "  [dim italic]Hint: Ensure filings have been ingested with "
                "'sec-search ingest add'.[/dim italic]"
            )
            raise typer.Exit(code=1) from None

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        console.print(
            "[dim italic]Hint: Try a broader query, or check that filings are ingested "
            "with 'sec-search manage status'.[/dim italic]"
        )
        return

    console.print(f"\n[bold]Found {len(results)} result(s)[/bold] for: [italic]{query}[/italic]\n")

    table = Table(show_lines=True, expand=True, border_style="dim")
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Similarity", width=10, justify="right")
    table.add_column("Source", style="cyan", width=20, no_wrap=True)
    table.add_column("Section", style="dim", max_width=30)
    table.add_column("Content")

    for i, result in enumerate(results, 1):
        # Source: ticker, form type, and date in one compact column.
        source = f"{result.ticker} {result.form_type}"
        if result.filing_date:
            source += f"\n{result.filing_date}"

        # Prefer the parent segment text when available so the analyst
        # sees the broader paragraph; highlight the embedded chunk
        # inside it. Falls back to the chunk text for legacy data.
        display_source = result.parent_content or result.content
        if len(display_source) > _CONTENT_PREVIEW_LIMIT:
            display_source = display_source[:_CONTENT_PREVIEW_LIMIT] + "..."
        content_text = Text(display_source)
        if result.parent_content and result.content in result.parent_content:
            content_text.highlight_words([result.content], style="black on yellow")

        # Truncate long section paths for display.
        section = result.path
        if len(section) > _SECTION_PATH_LIMIT:
            section = section[:_SECTION_PATH_LIMIT] + "..."

        table.add_row(
            str(i),
            _similarity_text(result.similarity),
            source,
            section,
            content_text,
        )

    console.print(table)
