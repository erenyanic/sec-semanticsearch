"""
Pipeline orchestrator for SEC filing ingestion.

This module coordinates the processing half of the ingestion pipeline:
    Parse → Chunk → Embed

Fetching is the caller's job (the CLI and the API ingest worker fetch
with ``FilingFetcher`` and pass the HTML in), and so is storage.

Usage:
    from sec_semantic_search.pipeline import FilingFetcher, PipelineOrchestrator

    filing_id, html_content = FilingFetcher().fetch_latest("AAPL", "10-K")
    result = PipelineOrchestrator().process_filing(filing_id, html_content)
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sec_semantic_search.core import (
    Chunk,
    FilingIdentifier,
    IngestResult,
    Segment,
    get_logger,
)
from sec_semantic_search.pipeline.chunk import TextChunker
from sec_semantic_search.pipeline.embed import EmbeddingGenerator
from sec_semantic_search.pipeline.parse import FilingParser

logger = get_logger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[[str, int, int], None]


@dataclass
class ProcessedFiling:
    """
    Result of processing a single filing through the pipeline.

    This contains all the data needed for storage in the database.

    Attributes:
        filing_id: Identifier for the filing
        chunks: Chunked text ready for storage
        embeddings: Vector embeddings for each chunk
        ingest_result: Statistics about the ingestion
        segments: Parsed segments retained so the storage layer can persist
            the parent context alongside the chunk vectors. The database
            layer joins these back onto search results so the UI can display
            the broader paragraph that produced each chunk.
    """

    filing_id: FilingIdentifier
    chunks: list[Chunk]
    embeddings: np.ndarray
    ingest_result: IngestResult
    segments: list[Segment]


class PipelineOrchestrator:
    """
    Coordinates the SEC filing ingestion pipeline.

    This class ties together the parser, chunker, and embedding generator
    to turn one filing's HTML into chunks and vectors, reporting progress
    through an optional callback.

    Note:
        The orchestrator neither fetches nor stores. It takes HTML the
        caller fetched and returns a ProcessedFiling containing chunks
        and embeddings that the database layer can store.

    Example:
        >>> orchestrator = PipelineOrchestrator()
        >>> result = orchestrator.process_filing(filing_id, html_content)
        >>> print(f"Processed {result.ingest_result.chunk_count} chunks")
    """

    def __init__(
        self,
        parser: FilingParser | None = None,
        chunker: TextChunker | None = None,
        embedder: EmbeddingGenerator | None = None,
    ) -> None:
        """
        Initialise the orchestrator with pipeline components.

        Components are created with defaults if not provided, allowing
        dependency injection for testing.

        Args:
            parser: FilingParser instance (optional)
            chunker: TextChunker instance (optional)
            embedder: EmbeddingGenerator instance (optional)
        """
        self.parser = parser or FilingParser()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or EmbeddingGenerator()

        logger.debug("PipelineOrchestrator initialised")

    def process_filing(
        self,
        filing_id: FilingIdentifier,
        html_content: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessedFiling:
        """
        Process a single filing through the pipeline.

        This method runs the full pipeline on HTML content that has
        already been fetched. Use this when you have the HTML content
        available (e.g., from a previous fetch or cache).

        Pipeline steps:
            1. Parse HTML → Segments
            2. Chunk segments → Chunks
            3. Generate embeddings

        Args:
            filing_id: Identifier for the filing
            html_content: Raw HTML content
            progress_callback: Optional callback(step_name, current, total)

        Returns:
            ProcessedFiling containing all processed data

        Example:
            >>> result = orchestrator.process_filing(filing_id, html)
            >>> print(f"Created {len(result.chunks)} chunks")
        """
        start_time = time.time()

        def report_progress(step: str, current: int, total: int) -> None:
            if progress_callback:
                progress_callback(step, current, total)

        logger.info(
            "Processing %s %s (%s)",
            filing_id.ticker,
            filing_id.form_type,
            filing_id.date_str,
        )

        # Step 1: Parse
        report_progress("Parsing", 1, 4)
        segments = self.parser.parse(html_content, filing_id)

        # Step 2: Chunk
        report_progress("Chunking", 2, 4)
        chunks = self.chunker.chunk_segments(segments)

        # Step 3: Embed
        report_progress("Embedding", 3, 4)
        embeddings = self.embedder.embed_chunks(chunks, show_progress=False)

        # Complete
        report_progress("Complete", 4, 4)
        duration = time.time() - start_time

        ingest_result = IngestResult(
            filing_id=filing_id,
            segment_count=len(segments),
            chunk_count=len(chunks),
            duration_seconds=duration,
        )

        logger.info(
            "Processed %s %s: %d segments → %d chunks in %.1fs",
            filing_id.ticker,
            filing_id.form_type,
            len(segments),
            len(chunks),
            duration,
        )

        return ProcessedFiling(
            filing_id=filing_id,
            chunks=chunks,
            embeddings=embeddings,
            ingest_result=ingest_result,
            segments=segments,
        )
