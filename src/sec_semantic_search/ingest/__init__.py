"""
Ingest module — the two-phase ingest loop shared by the CLI and the API.

    - plan_work: List the filings a request covers (metadata only)
    - run_ingest: Duplicate check, filing limit, fetch one ahead, process, store
    - IngestObserver: Progress and outcome hooks (Rich in the CLI, WebSocket in the API)
    - store_processed_filing: SQLite first (atomic), then ChromaDB, with rollback

Usage:
    from sec_semantic_search.ingest import plan_work, run_ingest

    work = plan_work(fetcher, ["AAPL"], ["10-K"])
    summary = run_ingest(
        work,
        fetch=fetcher.fetch_filing_content,
        orchestrator=orchestrator,
        registry=registry,
        chroma=chroma,
        max_filings=settings.database.max_filings,
    )
"""

from sec_semantic_search.ingest.runner import (
    STEP_TOTAL,
    FilingPrefetcher,
    IngestCancelled,
    IngestObserver,
    IngestSummary,
    per_form_count,
    plan_work,
    run_ingest,
    store_processed_filing,
)

__all__ = [
    "STEP_TOTAL",
    "FilingPrefetcher",
    "IngestCancelled",
    "IngestObserver",
    "IngestSummary",
    "per_form_count",
    "plan_work",
    "run_ingest",
    "store_processed_filing",
]
