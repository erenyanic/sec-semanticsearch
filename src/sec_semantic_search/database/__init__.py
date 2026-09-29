"""
Database module — ChromaDB vector storage and SQLite metadata registry.

This module provides the storage layer for ingested SEC filings:
    - ChromaDBClient: Vector storage for chunk embeddings and similarity search
    - MetadataRegistry: SQLite registry for filing metadata and management
    - FilingRecord: Dataclass representing a filing registry entry
    - delete_filings_batch: Shared helper to delete filings from both stores

Usage:
    from sec_semantic_search.database import (
        ChromaDBClient,
        MetadataRegistry,
        delete_filings_batch,
    )

    # Store a processed filing
    client = ChromaDBClient()
    registry = MetadataRegistry()

    registry.check_filing_limit()
    client.store_filing(processed_filing)
    registry.register_filing(processed_filing.filing_id, chunk_count=59)

    # Delete filings from both stores
    filings = registry.list_filings(ticker="AAPL")
    total_chunks = delete_filings_batch(filings, chroma=client, registry=registry)
"""

import logging

from sec_semantic_search.core import get_logger
from sec_semantic_search.database.client import ChromaDBClient
from sec_semantic_search.database.metadata import (
    DatabaseStatistics,
    FilingRecord,
    MetadataRegistry,
    TickerStatistics,
)

logger = get_logger(__name__)


def delete_filings_batch(
    filings: list[FilingRecord],
    *,
    chroma: ChromaDBClient,
    registry: MetadataRegistry,
) -> int:
    """
    Delete a list of filings from both stores (ChromaDB first, then SQLite).

    This is the single source of truth for dual-store deletion logic,
    used by both the CLI and API layers.  Uses batched operations on
    both stores to reduce round-trips from O(N) to O(1).

    Args:
        filings: Filing records to delete.
        chroma: ChromaDB client instance.
        registry: Metadata registry instance.

    Returns:
        Total number of chunks deleted across all filings.

    Raises:
        DatabaseError: If any deletion fails (propagated
            from ChromaDBClient or MetadataRegistry).
    """
    if not filings:
        return 0

    accession_numbers = [f.accession_number for f in filings]
    total_chunks = sum(f.chunk_count for f in filings)

    # ChromaDB first (store order convention), then SQLite.
    chroma.delete_filings_batch(accession_numbers)
    registry.remove_filings_batch(accession_numbers)

    # One summary line: a demo-mode eviction deletes 500+ filings at a time,
    # and a line per filing would add nothing an operator acts on.
    logger.info(
        "Deleted %d filing(s) across %d ticker(s) — %d chunks",
        len(filings),
        len({f.ticker for f in filings}),
        total_chunks,
    )
    if logger.isEnabledFor(logging.DEBUG):
        for filing in filings:
            logger.debug(
                "Deleted %s %s (%s) — %d chunks",
                filing.ticker,
                filing.form_type,
                filing.filing_date,
                filing.chunk_count,
            )

    return total_chunks


def clear_all_filings(
    *,
    chroma: ChromaDBClient,
    registry: MetadataRegistry,
) -> tuple[int, int]:
    """
    Delete every filing from both stores efficiently.

    Uses ``ChromaDBClient.clear_collection()`` and
    ``MetadataRegistry.clear_all()`` to avoid loading all records
    into memory — O(1) memory instead of O(N).

    Args:
        chroma: ChromaDB client instance.
        registry: Metadata registry instance.

    Returns:
        Tuple of ``(filings_deleted, chunks_deleted)``.

    Raises:
        DatabaseError: If any deletion fails.
    """
    # ChromaDB first (store order convention), then SQLite.
    chunks_deleted = chroma.clear_collection()
    filings_deleted = registry.clear_all()

    logger.info(
        "Cleared all data: %d filing(s), %d chunk(s)",
        filings_deleted,
        chunks_deleted,
    )
    return filings_deleted, chunks_deleted


__all__ = [
    # Main classes
    "ChromaDBClient",
    "MetadataRegistry",
    # Supporting types
    "DatabaseStatistics",
    "FilingRecord",
    "TickerStatistics",
    # Helpers
    "clear_all_filings",
    "delete_filings_batch",
]
