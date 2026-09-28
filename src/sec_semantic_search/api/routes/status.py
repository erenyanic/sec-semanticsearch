"""
Status endpoint — database overview and statistics.

Provides ``GET /api/status/`` returning filing count, chunk count,
per-ticker and per-form breakdowns.  Mirrors the CLI ``manage status``
command output.

The handler is plain ``def``: the registry statistics query and the ChromaDB
count block, so FastAPI runs it in the threadpool rather than on the event loop.
"""

from fastapi import APIRouter, Depends, Request

from sec_semantic_search.api.dependencies import get_chroma, get_registry, is_admin_request
from sec_semantic_search.api.schemas import StatusResponse, TickerBreakdown
from sec_semantic_search.config import get_settings
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry

router = APIRouter()


@router.get(
    "/",
    response_model=StatusResponse,
    summary="Database overview",
)
def status(
    request: Request,
    registry: MetadataRegistry = Depends(get_registry),
    chroma: ChromaDBClient = Depends(get_chroma),
) -> StatusResponse:
    """
    Return a full overview of database contents and capacity.

    Includes filing count, chunk count, unique tickers, form type
    breakdown, and per-ticker statistics.
    """
    settings = get_settings()
    stats = registry.get_statistics()
    chunk_count = chroma.collection_count()

    ticker_breakdown = [
        TickerBreakdown(
            ticker=ts.ticker,
            filings=ts.filings,
            chunks=ts.chunks,
            forms=ts.forms,
        )
        for ts in stats.ticker_breakdown
    ]

    # The frontend needs to know whether to show the Welcome screen.
    # Session is required when the server has no EDGAR identity AND
    # the setting explicitly mandates per-session credentials.
    has_server_identity = bool(settings.edgar.identity_name and settings.edgar.identity_email)
    edgar_session_required = settings.api.edgar_session_required and not has_server_identity

    return StatusResponse(
        filing_count=stats.filing_count,
        max_filings=settings.database.max_filings,
        chunk_count=chunk_count,
        tickers=stats.tickers,
        form_breakdown=stats.form_breakdown,
        ticker_breakdown=ticker_breakdown,
        edgar_session_required=edgar_session_required,
        demo_mode=settings.api.demo_mode,
        is_admin=is_admin_request(request),
    )
