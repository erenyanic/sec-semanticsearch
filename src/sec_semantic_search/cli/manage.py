"""Database management subcommands (status, list, remove)."""

from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import DatabaseError
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry, delete_filings_batch

console = Console()

manage_app = typer.Typer(no_args_is_help=True)


@manage_app.command("status")
def status() -> None:
    """
    Show database status and filing statistics.

    Examples:

        sec-search manage status
    """
    registry = MetadataRegistry()
    chroma = ChromaDBClient()
    settings = get_settings()

    stats = registry.get_statistics()
    chunk_count = chroma.collection_count()
    max_filings = settings.database.max_filings

    # Build a key-value table for the status metrics.
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    # Filings count with capacity indicator.
    filing_style = "green" if stats.filing_count > 0 else "dim"
    table.add_row("Filings", Text(f"{stats.filing_count}/{max_filings}", style=filing_style))

    # Chunk count.
    chunk_style = "green" if chunk_count > 0 else "dim"
    table.add_row("Chunks", Text(str(chunk_count), style=chunk_style))

    if stats.filing_count > 0:
        ticker_list = ", ".join(stats.tickers)
        table.add_row(
            "Tickers",
            Text(f"{len(stats.tickers)} ({ticker_list})", style="cyan"),
        )

        breakdown = "  |  ".join(f"{form}: {count}" for form, count in stats.form_breakdown.items())
        table.add_row("Forms", Text(breakdown))
    else:
        table.add_row("Tickers", Text("—", style="dim"))
        table.add_row("Forms", Text("—", style="dim"))

    console.print(Panel(table, title="[bold]Database Status[/bold]", expand=False))


@manage_app.command("list")
def list_filings(
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-k", help="Filter by ticker symbol."),
    ] = None,
    form: Annotated[
        str | None,
        typer.Option("--form", "-f", help="Filter by form type."),
    ] = None,
) -> None:
    """
    List all ingested filings.

    Examples:

        sec-search manage list

        sec-search manage list -k AAPL

        sec-search manage list -f 10-K
    """
    registry = MetadataRegistry()
    filings = registry.list_filings(
        ticker=ticker.upper() if ticker else None,
        form_type=form.upper() if form else None,
    )

    if not filings:
        console.print("[yellow]No filings found.[/yellow]")
        return

    table = Table(
        title="[bold]Ingested Filings[/bold]",
        border_style="dim",
        header_style="bold",
    )
    table.add_column("Ticker", style="cyan")
    table.add_column("Form", style="green")
    table.add_column("Filing Date")
    table.add_column("Accession Number", style="dim")
    table.add_column("Chunks", justify="right", style="bold")
    table.add_column("Ingested At", style="dim")

    # Text cells: stored values are data, never Rich markup.
    for f in filings:
        table.add_row(
            Text(f.ticker),
            Text(f.form_type),
            Text(f.filing_date),
            Text(f.accession_number),
            str(f.chunk_count),
            Text(f.ingested_at),
        )

    console.print(table)


@manage_app.command("remove")
def remove(
    accession_number: Annotated[
        str | None,
        typer.Argument(help="Accession number of the filing to remove."),
    ] = None,
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-k", help="Remove all filings for this ticker."),
    ] = None,
    form: Annotated[
        str | None,
        typer.Option("--form", "-f", help="Remove all filings of this form type."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """
    Remove filing(s) from the database.

    Remove a single filing by accession number, or remove multiple filings
    matching --ticker and/or --form filters.

    Examples:

        sec-search manage remove 0000320193-24-000123

        sec-search manage remove --ticker AAPL

        sec-search manage remove --form 10-K

        sec-search manage remove --ticker AAPL --form 10-K

        sec-search manage remove --ticker MSFT -y
    """
    has_filters = ticker is not None or form is not None

    if accession_number is None and not has_filters:
        console.print(
            "[red]Provide an accession number or use --ticker/--form to select filings.[/red]"
        )
        raise typer.Exit(code=1)

    if accession_number is not None and has_filters:
        console.print("[red]Cannot combine an accession number with --ticker/--form filters.[/red]")
        raise typer.Exit(code=1)

    registry = MetadataRegistry()

    # --- Single filing by accession number -----------------------------------
    if accession_number is not None:
        filing = registry.get_filing(accession_number)
        if filing is None:
            console.print(f"[red]Filing not found:[/red] {escape(accession_number)}")
            console.print(
                "  [dim italic]Hint: Run 'sec-search manage list' to see available "
                "accession numbers.[/dim italic]"
            )
            raise typer.Exit(code=1)

        detail = Table(show_header=False, box=None, padding=(0, 2))
        detail.add_column("Key", style="bold")
        detail.add_column("Value")
        detail.add_row("Filing", Text(f"{filing.ticker} {filing.form_type}", style="cyan"))
        detail.add_row("Date", Text(filing.filing_date))
        detail.add_row("Chunks", Text(str(filing.chunk_count), style="bold"))
        detail.add_row("Accession", Text(filing.accession_number, style="dim"))
        console.print(Panel(detail, title="[bold yellow]Remove Filing[/bold yellow]", expand=False))

        if not yes:
            confirmed = typer.confirm("Remove this filing?")
            if not confirmed:
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(code=0)

        try:
            chroma = ChromaDBClient()
            delete_filings_batch([filing], chroma=chroma, registry=registry)
        except DatabaseError as e:
            console.print(f"[red]Removal failed:[/red] {escape(e.message)}")
            console.print(
                "  [dim italic]Hint: Check that the data directory is writable.[/dim italic]"
            )
            raise typer.Exit(code=1) from None

        console.print(
            f"[green]Removed:[/green] {escape(f'{filing.ticker} {filing.form_type}')} "
            f"({escape(filing.filing_date)}) — {filing.chunk_count} chunks deleted"
        )
        return

    # --- Bulk removal by --ticker and/or --form ------------------------------
    filings = registry.list_filings(
        ticker=ticker.upper() if ticker else None,
        form_type=form.upper() if form else None,
    )

    if not filings:
        filter_desc = " and ".join(
            part
            for part in [
                f"ticker={ticker.upper()}" if ticker else None,
                f"form={form.upper()}" if form else None,
            ]
            if part
        )
        console.print(f"[yellow]No filings found matching {escape(filter_desc)}.[/yellow]")
        return

    # Show what will be deleted.
    total_chunks = sum(f.chunk_count for f in filings)
    filter_parts: list[str] = []
    if ticker:
        filter_parts.append(f"ticker=[cyan]{escape(ticker.upper())}[/cyan]")
    if form:
        filter_parts.append(f"form=[green]{escape(form.upper())}[/green]")
    filter_desc = ", ".join(filter_parts)

    console.print(
        f"\n[bold yellow]Bulk Remove[/bold yellow]  ({filter_desc})\n"
        f"  {len(filings)} filing(s), {total_chunks} chunks total\n"
    )

    for f in filings:
        console.print(
            f"  [dim]•[/dim] {escape(f'{f.ticker} {f.form_type} ({f.filing_date})')} "
            f"— {f.chunk_count} chunks"
        )

    console.print()

    if not yes:
        confirmed = typer.confirm(f"{len(filings)} filing(s) will be deleted. Are you sure?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    try:
        chroma = ChromaDBClient()
        total_deleted = delete_filings_batch(
            filings,
            registry=registry,
            chroma=chroma,
        )
    except DatabaseError as e:
        console.print(f"[red]Removal failed:[/red] {escape(e.message)}")
        console.print("  [dim italic]Hint: Check that the data directory is writable.[/dim italic]")
        raise typer.Exit(code=1) from None

    console.print(
        f"\n[green]Done:[/green] {len(filings)} filing(s) removed, {total_deleted} chunks deleted"
    )


@manage_app.command("clear")
def clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """
    One command to rule them all — delete every filing from the database.

    Examples:

        sec-search manage clear

        sec-search manage clear -y
    """
    registry = MetadataRegistry()
    filings = registry.list_filings()

    if not filings:
        console.print("[yellow]Database is already empty.[/yellow]")
        return

    total_chunks = sum(f.chunk_count for f in filings)
    unique_tickers = sorted({f.ticker for f in filings})

    console.print(
        f"\n[bold red]Clear Database[/bold red]\n"
        f"  {len(filings)} filing(s), {total_chunks} chunks, "
        f"{len(unique_tickers)} ticker(s): {escape(', '.join(unique_tickers))}\n"
    )

    if not yes:
        confirmed = typer.confirm(f"ALL {len(filings)} filing(s) will be deleted. Are you sure?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    try:
        chroma = ChromaDBClient()
        total_deleted = delete_filings_batch(
            filings,
            registry=registry,
            chroma=chroma,
        )
    except DatabaseError as e:
        console.print(f"[red]Clear failed:[/red] {escape(e.message)}")
        console.print("  [dim italic]Hint: Check that the data directory is writable.[/dim italic]")
        raise typer.Exit(code=1) from None

    console.print(
        f"\n[green]Database cleared:[/green] {len(filings)} filing(s) removed, "
        f"{total_deleted} chunks deleted"
    )
