"""
Search endpoint for semantic search over ingested SEC filings.

Provides a single route:
    - ``POST /api/search/`` — run a semantic search query with optional filters
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Response

from sec_semantic_search.api.dependencies import get_search_engine
from sec_semantic_search.api.schemas import (
    ErrorResponse,
    ParentSegmentSchema,
    SearchRequest,
    SearchResponse,
    SearchResultSchema,
)
from sec_semantic_search.core import (
    EmbeddingBusyError,
    SearchError,
    SearchResult,
    get_logger,
    redact_for_log,
)
from sec_semantic_search.search import SearchEngine

logger = get_logger(__name__)

router = APIRouter()

# How long a search waits for the embedding model while an ingest batch or
# a model load holds it. Past this the caller gets a retryable 503 instead
# of pinning a threadpool worker for the length of a whole filing's encode.
_EMBED_WAIT_SECONDS = 30.0

# Longest parent text returned per segment. Filing tables can run to tens
# of kilobytes; with ``top_k`` up to 100 an uncapped response could reach
# megabytes. Longer segments are sent as an excerpt around the matched
# chunks.
_PARENT_MAX_CHARS = 16_000


def _parent_excerpt(parent: str, chunks: list[str]) -> ParentSegmentSchema:
    """Return ``parent``, or a window of it that keeps the matched chunks.

    ``chunks`` are the referencing results' chunk texts, best match first.
    The window covers all of them when they fit, otherwise the best one.
    A chunk not found verbatim is ignored; with none found the excerpt
    starts at the beginning of the segment.
    """
    if len(parent) <= _PARENT_MAX_CHARS:
        return ParentSegmentSchema(content=parent)

    spans = [(i, i + len(c)) for c in chunks if (i := parent.find(c)) >= 0]
    start = 0
    if spans:
        low, high = min(s for s, _ in spans), max(e for _, e in spans)
        if high - low > _PARENT_MAX_CHARS:
            low, high = spans[0]
        centre = (low + high) // 2
        start = max(0, min(centre - _PARENT_MAX_CHARS // 2, len(parent) - _PARENT_MAX_CHARS))
    end = start + _PARENT_MAX_CHARS
    return ParentSegmentSchema(
        content=parent[start:end],
        truncated_start=start > 0,
        truncated_end=end < len(parent),
    )


def _build_parents(
    results: list[SearchResult],
) -> list[tuple[str | None, ParentSegmentSchema | None]]:
    """Return each result's ``(parent_key, parent)``.

    A segment's text goes with the first result that cites it; later
    results get the key alone. Placing it there, next to the chunk it
    contains, also lets gzip encode the chunk as a back-reference. A
    result gets no key when its parent is missing or identical to its
    chunk, so single-chunk segments are not sent twice.
    """
    chunks_by_key: dict[str, list[str]] = {}
    for r in results:
        if r.parent_content and r.parent_content != r.content:
            key = f"{r.accession_number}:{r.segment_index}"
            chunks_by_key.setdefault(key, []).append(r.content)

    out: list[tuple[str | None, ParentSegmentSchema | None]] = []
    sent: set[str] = set()
    for r in results:
        if not r.parent_content or r.parent_content == r.content:
            out.append((None, None))
            continue
        key = f"{r.accession_number}:{r.segment_index}"
        if key in sent:
            out.append((key, None))
        else:
            sent.add(key)
            out.append((key, _parent_excerpt(r.parent_content, chunks_by_key[key])))
    return out


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
    response: Response,
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
            parent_key=key,
            parent=parent,
        )
        for r, (key, parent) in zip(results, _build_parents(results), strict=True)
    ]

    logger.info(
        "Search '%s' returned %d result(s) in %.1f ms",
        redact_for_log(body.query[:80]),
        len(result_schemas),
        elapsed_ms,
    )

    # Results derive from the private query; never let a cache keep them.
    response.headers["Cache-Control"] = "no-store"
    # Returning the model (not a JSONResponse) lets FastAPI serialize it
    # straight to JSON bytes in pydantic-core.
    return SearchResponse(
        results=result_schemas,
        total_results=len(result_schemas),
        search_time_ms=round(elapsed_ms, 1),
    )
