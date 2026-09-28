"""
Filing management endpoints — list, retrieve, and delete ingested filings.

Provides full CRUD (minus create — that's ingest) for the filing registry:
    - ``GET    /api/filings/``            — list filings with optional filters
    - ``GET    /api/filings/{accession}`` — get a single filing by accession number
    - ``DELETE /api/filings/{accession}`` — delete a single filing (ChromaDB first, then SQLite)
    - ``POST   /api/filings/bulk-delete`` — bulk delete by ticker/form_type filter
    - ``DELETE /api/filings/``            — clear all filings (requires ``confirm=true``)

Every handler is plain ``def`` so FastAPI runs it in the threadpool. SQLite
reads and ChromaDB deletes block; on the event loop they would stall health
probes, WebSocket delivery and every other request.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from sec_semantic_search.api.dependencies import get_chroma, get_registry, verify_admin_key
from sec_semantic_search.api.schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    ClearAllResponse,
    DeleteByIdsRequest,
    DeleteByIdsResponse,
    DeleteResponse,
    ErrorResponse,
    FilingListResponse,
    FilingSchema,
)
from sec_semantic_search.config import get_settings
from sec_semantic_search.core import DatabaseError, audit_log, get_logger
from sec_semantic_search.database import (
    ChromaDBClient,
    MetadataRegistry,
    clear_all_filings,
    delete_filings_batch,
)
from sec_semantic_search.database.metadata import FilingRecord

logger = get_logger(__name__)

router = APIRouter()


def _record_to_schema(record: FilingRecord) -> FilingSchema:
    """Convert a database ``FilingRecord`` to an API ``FilingSchema``."""
    return FilingSchema(
        ticker=record.ticker,
        form_type=record.form_type,
        filing_date=record.filing_date,
        accession_number=record.accession_number,
        chunk_count=record.chunk_count,
        ingested_at=record.ingested_at,
    )


@router.get(
    "/",
    response_model=FilingListResponse,
    summary="List ingested filings",
)
def list_filings(
    registry: MetadataRegistry = Depends(get_registry),
    ticker: str | None = Query(None, description="Filter by ticker symbol"),
    form_type: str | None = Query(None, description="Filter by form type (8-K, 10-K, or 10-Q)"),
    sort_by: Literal["filing_date", "ticker", "form_type", "chunk_count", "ingested_at"] = Query(
        "filing_date", description="Column to sort by"
    ),
    order: Literal["asc", "desc"] = Query("desc", description="Sort order"),
) -> FilingListResponse:
    """
    List all ingested filings with optional filters and sorting.

    Results are returned in the order specified by ``sort_by`` and ``order``.
    The underlying registry always returns filings ordered by filing_date
    descending; additional sorting is applied in-memory.
    """
    records = registry.list_filings(
        ticker=ticker.upper() if ticker else None,
        form_type=form_type.upper() if form_type else None,
    )

    # Apply sorting.  The registry returns filing_date DESC by default,
    # so we only need to re-sort when the caller requests something else.
    if sort_by != "filing_date" or order != "desc":
        reverse = order == "desc"
        records.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)

    schemas = [_record_to_schema(r) for r in records]
    return FilingListResponse(filings=schemas, total=len(schemas))


@router.get(
    "/{accession}",
    response_model=FilingSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get a single filing",
)
def get_filing(
    accession: str = Path(..., max_length=20, pattern=r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$"),
    registry: MetadataRegistry = Depends(get_registry),
) -> FilingSchema:
    """
    Retrieve a single filing record by accession number.

    Returns 404 if the filing is not found in the registry.
    """
    record = registry.get_filing(accession)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Filing not found: {accession}",
                "hint": "Use GET /api/filings/ to list available accession numbers.",
            },
        )
    return _record_to_schema(record)


# ---------------------------------------------------------------------------
# Delete endpoints (W1.3)
# ---------------------------------------------------------------------------


@router.delete(
    "/{accession}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Delete a single filing",
)
def delete_filing(
    request: Request,
    accession: str = Path(..., max_length=20, pattern=r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$"),
    registry: MetadataRegistry = Depends(get_registry),
    chroma: ChromaDBClient = Depends(get_chroma),
) -> DeleteResponse:
    """
    Delete a single filing by accession number from both stores.

    Removes vector data from ChromaDB first, then the metadata row from
    SQLite.  Returns the number of chunks deleted.
    """
    record = registry.get_filing(accession)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Filing not found: {accession}",
                "hint": "Use GET /api/filings/ to list available accession numbers.",
            },
        )

    try:
        chroma.delete_filing(accession)
        registry.remove_filing(accession)
    except DatabaseError as exc:
        logger.error("Delete filing %s failed: %s", accession, exc.details)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "database_error",
                "message": "Database operation failed. Check server logs.",
                "details": None,
                "hint": "Check that the data directory is writable.",
            },
        ) from exc

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "delete_filing",
        client_ip=client_ip,
        endpoint="DELETE /api/filings/{accession}",
        detail=f"accession={accession} ticker={record.ticker} form={record.form_type} chunks={record.chunk_count}",
    )
    return DeleteResponse(
        accession_number=accession,
        chunks_deleted=record.chunk_count,
    )


@router.post(
    "/delete-by-ids",
    response_model=DeleteByIdsResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Delete filings by accession numbers",
)
def delete_by_ids(
    request: Request,
    body: DeleteByIdsRequest,
    registry: MetadataRegistry = Depends(get_registry),
    chroma: ChromaDBClient = Depends(get_chroma),
) -> DeleteByIdsResponse:
    """
    Delete specific filings by their accession numbers in a single request.

    Looks up each accession number in the registry, deletes those that exist
    from both stores, and reports any that were not found.  This is more
    efficient than making N sequential ``DELETE /api/filings/{accession}``
    calls from the frontend.
    """
    # Batch lookup: single SQL query instead of N individual get_filing() calls.
    found = registry.get_filings_by_accessions(body.accession_numbers)
    found_accessions = {r.accession_number for r in found}
    not_found = [a for a in body.accession_numbers if a not in found_accessions]

    if not found:
        return DeleteByIdsResponse(
            filings_deleted=0,
            chunks_deleted=0,
            not_found=not_found,
        )

    try:
        total_chunks = delete_filings_batch(
            found,
            chroma=chroma,
            registry=registry,
        )
    except DatabaseError as exc:
        logger.error("Delete by IDs failed: %s", exc.details)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "database_error",
                "message": "Database operation failed. Check server logs.",
                "details": None,
                "hint": "Check that the data directory is writable.",
            },
        ) from exc

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "delete_by_ids",
        client_ip=client_ip,
        endpoint="POST /api/filings/delete-by-ids",
        detail=f"deleted={len(found)} chunks={total_chunks} not_found={len(not_found)}",
    )
    return DeleteByIdsResponse(
        filings_deleted=len(found),
        chunks_deleted=total_chunks,
        not_found=not_found,
    )


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Bulk delete filings by filter",
    dependencies=[Depends(verify_admin_key)],
)
def bulk_delete(
    request: Request,
    body: BulkDeleteRequest,
    registry: MetadataRegistry = Depends(get_registry),
    chroma: ChromaDBClient = Depends(get_chroma),
) -> BulkDeleteResponse:
    """
    Delete all filings matching the given ticker and/or form_type filter.

    At least one filter (``ticker`` or ``form_type``) must be provided.
    Returns the number of filings and chunks deleted.
    """
    if body.ticker is None and body.form_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": "At least one filter is required.",
                "hint": (
                    "Provide 'ticker' and/or 'form_type'. "
                    "To delete everything, use DELETE /api/filings/?confirm=true instead."
                ),
            },
        )

    filings = registry.list_filings(
        ticker=body.ticker.upper() if body.ticker else None,
        form_type=body.form_type,  # Already uppercased by validator
    )

    if not filings:
        return BulkDeleteResponse(
            filings_deleted=0,
            chunks_deleted=0,
            tickers_affected=[],
        )

    try:
        total_chunks = delete_filings_batch(
            filings,
            chroma=chroma,
            registry=registry,
        )
    except DatabaseError as exc:
        logger.error("Bulk delete failed: %s", exc.details)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "database_error",
                "message": "Database operation failed. Check server logs.",
                "details": None,
                "hint": "Check that the data directory is writable.",
            },
        ) from exc

    tickers_affected = sorted({f.ticker for f in filings})

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "bulk_delete",
        client_ip=client_ip,
        endpoint="POST /api/filings/bulk-delete",
        detail=f"filings={len(filings)} chunks={total_chunks} tickers={tickers_affected}",
    )
    return BulkDeleteResponse(
        filings_deleted=len(filings),
        chunks_deleted=total_chunks,
        tickers_affected=tickers_affected,
    )


@router.delete(
    "/",
    response_model=ClearAllResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Clear all filings",
    dependencies=[Depends(verify_admin_key)],
)
def clear_all(
    request: Request,
    confirm: bool = Query(False, description="Safety flag — must be true to proceed"),
    registry: MetadataRegistry = Depends(get_registry),
    chroma: ChromaDBClient = Depends(get_chroma),
) -> ClearAllResponse:
    """
    Delete every filing from both stores.

    Requires ``confirm=true`` as a safety measure to prevent accidental
    data loss.  Disabled entirely when ``DEMO_MODE=true`` — returns 403
    for everyone, including admins.  Demo data is cleared by the
    deployment instead: on Cloud Run it lives in an in-memory volume
    that is discarded with the instance; elsewhere
    ``scripts/demo-reset.sh`` can run on a schedule.
    """
    if get_settings().api.demo_mode:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "demo_mode",
                "message": "Clear all is disabled in demo mode.",
                "details": None,
                "hint": "Demo data is temporary and is cleared automatically.",
            },
        )

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "confirmation_required",
                "message": "This will delete ALL filings. Pass ?confirm=true to proceed.",
                "hint": "Add '?confirm=true' to the request URL.",
            },
        )

    try:
        filings_deleted, chunks_deleted = clear_all_filings(
            chroma=chroma,
            registry=registry,
        )
    except DatabaseError as exc:
        logger.error("Clear all failed: %s", exc.details)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "database_error",
                "message": "Database operation failed. Check server logs.",
                "details": None,
                "hint": "Check that the data directory is writable.",
            },
        ) from exc

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "clear_all",
        client_ip=client_ip,
        endpoint="DELETE /api/filings/?confirm=true",
        detail=f"filings={filings_deleted} chunks={chunks_deleted}",
    )
    return ClearAllResponse(
        filings_deleted=filings_deleted,
        chunks_deleted=chunks_deleted,
    )
