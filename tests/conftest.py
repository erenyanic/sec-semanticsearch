"""
Shared pytest fixtures for SEC-SemanticSearch tests.

This module provides reusable test data and temporary resources used
across both unit and integration tests:

    - sample_filing_id: A realistic FilingIdentifier
    - sample_segments: Segments covering all three ContentType variants
    - sample_chunks: Pre-built Chunk objects with sequential indices
    - tmp_db_path / tmp_chroma_path: Isolated temporary database paths
    - sample_html: Minimal HTML that doc2dict can parse
"""

from datetime import date

import pytest

from sec_semantic_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    Segment,
)

# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_filing_id() -> FilingIdentifier:
    """
    A realistic, reusable filing identifier.

    Uses AAPL 10-K to match the project's existing verification data,
    but with a synthetic accession number to avoid collision with real
    ingested filings.
    """
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        accession_number="0000320193-24-000001",
    )


@pytest.fixture
def sample_segments(sample_filing_id: FilingIdentifier) -> list[Segment]:
    """
    A small list covering all three ContentType variants.

    Three segments are enough to exercise chunking, database storage,
    and search without being slow. Each has realistic but short content
    so token counts are predictable. Indices are assigned as
    ``FilingParser.parse()`` assigns them.
    """
    return [
        Segment(
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            content=(
                "The Company designs, manufactures and markets smartphones. "
                "The Company's products include iPhone, Mac, iPad and wearables. "
                "The Company sells its products worldwide through retail and online stores."
            ),
            filing_id=sample_filing_id,
            segment_index=0,
        ),
        Segment(
            path="Part I > Item 1A > Risk Factors",
            content_type=ContentType.TEXTSMALL,
            content="See also the risk factors described in our annual report.",
            filing_id=sample_filing_id,
            segment_index=1,
        ),
        Segment(
            path="Part II > Item 8 > Financial Statements",
            content_type=ContentType.TABLE,
            content="Revenue | 394,328 | 383,285\nNet Income | 93,736 | 96,995",
            filing_id=sample_filing_id,
            segment_index=2,
        ),
    ]


@pytest.fixture
def sample_chunks(sample_filing_id: FilingIdentifier) -> list[Chunk]:
    """
    Pre-built chunks with sequential indices.

    These mirror what TextChunker would produce from sample_segments —
    useful for database and search tests that don't need to run the
    chunker themselves.
    """
    return [
        Chunk(
            content=(
                "The Company designs, manufactures and markets smartphones. "
                "The Company's products include iPhone, Mac, iPad and wearables. "
                "The Company sells its products worldwide through retail and online stores."
            ),
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            filing_id=sample_filing_id,
            chunk_index=0,
            segment_index=0,
        ),
        Chunk(
            content="See also the risk factors described in our annual report.",
            path="Part I > Item 1A > Risk Factors",
            content_type=ContentType.TEXTSMALL,
            filing_id=sample_filing_id,
            chunk_index=1,
            segment_index=1,
        ),
        Chunk(
            content="Revenue | 394,328 | 383,285\nNet Income | 93,736 | 96,995",
            path="Part II > Item 8 > Financial Statements",
            content_type=ContentType.TABLE,
            filing_id=sample_filing_id,
            chunk_index=2,
            segment_index=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Temporary database paths
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """
    Isolated SQLite database path inside pytest's tmp directory.

    Each test receives a unique temporary directory, so databases never
    collide or persist between runs.
    """
    return str(tmp_path / "test_metadata.sqlite")


@pytest.fixture
def tmp_chroma_path(tmp_path) -> str:
    """
    Isolated ChromaDB storage directory.

    ChromaDB's PersistentClient writes to this directory; pytest
    cleans it up automatically after the test session.
    """
    return str(tmp_path / "test_chroma_db")


# ---------------------------------------------------------------------------
# Sample HTML for parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_html() -> str:
    """
    Minimal HTML that doc2dict can parse into segments.

    This is intentionally small — just enough structure to produce
    a few segments with titles, text, and a table. Real SEC filings
    are megabytes; test HTML should be a few hundred bytes.
    """
    return """
    <html>
    <body>
    <div>
        <h1>Part I</h1>
        <div>
            <h2>Item 1. Business</h2>
            <p>The Company designs and sells consumer electronics.</p>
            <p>Products include smartphones, tablets, and computers.</p>
        </div>
        <div>
            <h2>Item 1A. Risk Factors</h2>
            <p>The Company faces significant competition in all markets.</p>
        </div>
    </div>
    </body>
    </html>
    """
