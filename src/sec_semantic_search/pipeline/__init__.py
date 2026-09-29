"""
Pipeline module — fetch, parse, chunk, embed, and orchestrate.

This module provides the complete ingestion pipeline for SEC filings:
    - FilingFetcher: Fetch filings from SEC EDGAR
    - FilingParser: Parse HTML into semantic segments
    - TextChunker: Split segments into embedding-ready chunks
    - EmbeddingGenerator: Generate vector embeddings
    - PipelineOrchestrator: Parse, chunk and embed one fetched filing

Usage:
    from sec_semantic_search.pipeline import (
        FilingFetcher,
        FilingParser,
        TextChunker,
        EmbeddingGenerator,
        PipelineOrchestrator,
    )

    # Fetch, then parse → chunk → embed (storage is the database module's job)
    filing_id, html = FilingFetcher().fetch_latest("AAPL", "10-K")
    result = PipelineOrchestrator().process_filing(filing_id, html)
"""

from sec_semantic_search.pipeline.chunk import TextChunker
from sec_semantic_search.pipeline.embed import EmbeddingGenerator
from sec_semantic_search.pipeline.fetch import FilingFetcher, FilingInfo
from sec_semantic_search.pipeline.orchestrator import (
    PipelineOrchestrator,
    ProcessedFiling,
    ProgressCallback,
)
from sec_semantic_search.pipeline.parse import FilingParser

__all__ = [
    # Main classes
    "FilingFetcher",
    "FilingParser",
    "TextChunker",
    "EmbeddingGenerator",
    "PipelineOrchestrator",
    # Supporting types
    "FilingInfo",
    "ProcessedFiling",
    "ProgressCallback",
]
