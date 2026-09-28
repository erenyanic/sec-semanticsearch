"""
Search endpoint for semantic search over ingested SEC filings.

Provides a single route:
    - ``POST /api/search/`` — run a semantic search query with optional filters
"""

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from sec_semantic_search.api.dependencies import get_search_engine
from sec_semantic_search.api.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultSchema,
)
from sec_semantic_search.core import EmbeddingBusyError, SearchError, get_logger, redact_for_log
from sec_semantic_search.search import SearchEngine

logger = get_logger(__name__)

router = APIRouter()

# How long a search waits for the embedding model while an ingest batch or
# a model load holds it. Past this the caller gets a retryable 503 instead
# of pinning a threadpool worker for the length of a whole filing's encode.
_EMBED_WAIT_SECONDS = 30.0


# Plain ``def``: FastAPI runs it in the threadpool. Embedding, the ChromaDB
# query and the SQLite join all block, and on the event loop they would
# stall health probes and WebSocket delivery for the length of the search.
@router.post(
    "/",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Semantic search over filings",
)
def search(
    body: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine),
) -> SearchResponse:
    """
    Search ingested SEC filings using a natural language query.

    The query is embedded with the same model used during ingestion
    and matched against stored chunks via cosine similarity.  Results
    are returned ranked by similarity (highest first).

    Accepts optional filters for ticker, form type, minimum similarity
    threshold, and accession number (filing-specific search).
    """
    start = time.perf_counter()

    try:
        results = engine.search(
            query=body.query,
            top_k=body.top_k,
            ticker=body.ticker,
            form_type=body.form_type,
            min_similarity=body.min_similarity,
            accession_number=body.accession_number,
            start_date=body.start_date,
            end_date=body.end_date,
            embed_timeout=_EMBED_WAIT_SECONDS,
        )
    except EmbeddingBusyError as exc:
        logger.warning("Search rejected: %s", exc.details)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_busy",
                "message": "The embedding model is busy. Try again shortly.",
                "details": None,
                "hint": "An ingest or model load is in progress.",
            },
            headers={"Retry-After": "5"},
        ) from exc
    except SearchError as exc:
        # Empty query is a validation error (400); everything else is 500.
        if "empty" in exc.message.lower():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "validation_error",
                    "message": exc.message,
                    "details": None,
                    "hint": "Provide a non-empty search query.",
                },
            ) from exc

        logger.error("Search failed: %s — %s", exc.message, exc.details)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "search_error",
                "message": "Search operation failed. Check server logs.",
                "details": None,
                "hint": "Ensure filings have been ingested and the database is accessible.",
            },
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000

    result_schemas = [
        SearchResultSchema(
            content=r.content,
            path=r.path,
            content_type=r.content_type.value,
            ticker=r.ticker,
            form_type=r.form_type,
            similarity=r.similarity,
            filing_date=r.filing_date,
            accession_number=r.accession_number,
            chunk_id=r.chunk_id,
            segment_index=r.segment_index,
            parent_content=r.parent_content,
        )
        for r in results
    ]

    logger.info(
        "Search '%s' returned %d result(s) in %.1f ms",
        redact_for_log(body.query[:80]),
        len(result_schemas),
        elapsed_ms,
    )

    payload = SearchResponse(
        results=result_schemas,
        total_results=len(result_schemas),
        search_time_ms=round(elapsed_ms, 1),
    )
    return JSONResponse(
        content=payload.model_dump(),
        headers={"Cache-Control": "no-store"},
    )
