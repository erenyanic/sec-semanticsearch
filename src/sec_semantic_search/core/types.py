"""
Core data types for SEC-SemanticSearch.

This module defines the domain objects used throughout the pipeline:
    - FilingIdentifier: Unique identifier for an SEC filing
    - Segment: Parsed content unit from a filing
    - Chunk: Embedding-ready text unit
    - SearchResult: Query result with similarity score

Design notes:
    - Dataclasses are used for simplicity and performance (no runtime validation)
    - FilingIdentifier is frozen (immutable) as it serves as an identifier
    - ContentType enum ensures type-safe content classification
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class ContentType(Enum):
    """
    Content types extracted from SEC filings via doc2dict.

    Values:
        TEXT: Regular paragraph text
        TEXTSMALL: Smaller text elements (footnotes, captions)
        TABLE: Tabular data converted to text representation
    """

    TEXT = "text"
    TEXTSMALL = "textsmall"
    TABLE = "table"


@dataclass(frozen=True)
class FilingIdentifier:
    """
    Unique identifier for an SEC filing.

    This immutable identifier is used to track filings throughout the pipeline
    and serves as the primary key in the metadata registry.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        form_type: SEC form type (e.g., "10-K", "10-Q")
        filing_date: Date the filing was submitted to SEC
        accession_number: SEC-assigned unique identifier (e.g., "0000320193-23-000077")

    Example:
        >>> filing_id = FilingIdentifier(
        ...     ticker="AAPL",
        ...     form_type="10-K",
        ...     filing_date=date(2023, 11, 3),
        ...     accession_number="0000320193-23-000077",
        ... )
    """

    ticker: str
    form_type: str
    filing_date: date
    accession_number: str

    def __post_init__(self) -> None:
        """Validate and normalise field values."""
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "ticker", self.ticker.upper())
        object.__setattr__(self, "form_type", self.form_type.upper())

    @property
    def date_str(self) -> str:
        """Return filing date as ISO format string (YYYY-MM-DD)."""
        return self.filing_date.isoformat()


@dataclass
class Segment:
    """
    A semantically meaningful unit of content extracted from a filing.

    Segments are created by the parser from doc2dict output. Each segment
    represents a coherent piece of content (paragraph, table, footnote)
    with its hierarchical location in the document.

    Attributes:
        path: Hierarchical path (e.g., "Part I > Item 1A > Risk Factors")
        content_type: Type of content (text, textsmall, table)
        content: The actual text content
        filing_id: Reference to the source filing
        segment_index: Zero-based position of this segment within the filing.
            Assigned by the parser and inherited by every chunk derived from
            this segment so the search layer can resolve the chunk back to
            its parent context for display. ``None`` until assigned — never
            ``0``, which is a real index (the filing's first segment).

    Example:
        >>> segment = Segment(
        ...     path="Part I > Item 1A > Risk Factors",
        ...     content_type=ContentType.TEXT,
        ...     content="Our business is subject to...",
        ...     filing_id=filing_id,
        ... )
    """

    path: str
    content_type: ContentType
    content: str
    filing_id: FilingIdentifier
    segment_index: int | None = field(default=None)


@dataclass
class Chunk:
    """
    An embedding-ready text unit derived from a segment.

    Chunks are created by splitting long segments at sentence boundaries.
    Each chunk inherits metadata from its source segment and is assigned
    a unique index for ChromaDB storage.

    Attributes:
        content: The text content (respects token limit)
        path: Inherited hierarchical path from source segment
        content_type: Inherited content type from source segment
        filing_id: Reference to the source filing
        chunk_index: Zero-based index within the filing's chunks
        segment_index: Index of the parent segment, copied from it by the
            chunker. ``None`` when unknown: the chunk is then stored without
            the key and shown without parent context, rather than with the
            filing's first segment.

    The chunk_id property generates the ChromaDB document ID in the format:
        {TICKER}_{FORM_TYPE}_{DATE}_{INDEX}
    """

    content: str
    path: str
    content_type: ContentType
    filing_id: FilingIdentifier
    chunk_index: int = field(default=0)
    token_count: int = field(default=0)
    segment_index: int | None = field(default=None)

    @property
    def chunk_id(self) -> str:
        """
        Generate unique ChromaDB document ID.

        Format: {TICKER}_{FORM_TYPE}_{DATE}_{INDEX}
        Example: AAPL_10-K_2023-11-03_042
        """
        return (
            f"{self.filing_id.ticker}_"
            f"{self.filing_id.form_type}_"
            f"{self.filing_id.date_str}_"
            f"{self.chunk_index:03d}"
        )

    def to_metadata(self) -> dict:
        """
        Convert chunk metadata to ChromaDB-compatible dict.

        Returns:
            Dictionary suitable for ChromaDB metadata. Includes both
            ``filing_date`` (ISO string for display) and ``filing_date_int``
            (``YYYYMMDD`` integer for range queries with ``$gte``/``$lte``).
            ``segment_index`` is present only when known.
        """
        metadata = {
            "path": self.path,
            "content_type": self.content_type.value,
            "ticker": self.filing_id.ticker,
            "form_type": self.filing_id.form_type,
            "filing_date": self.filing_id.date_str,
            "filing_date_int": int(self.filing_id.date_str.replace("-", "")),
            "accession_number": self.filing_id.accession_number,
        }
        if self.segment_index is not None:
            metadata["segment_index"] = self.segment_index
        return metadata


@dataclass
class SearchResult:
    """
    A single result from a semantic search query.

    Search results are returned by the search engine, ranked by similarity.
    Each result contains the matched chunk content along with its metadata
    and relevance score.

    Attributes:
        content: The matched chunk text (the small, embedded slice)
        path: Hierarchical path in the source document
        content_type: Type of content (text, textsmall, table)
        ticker: Stock ticker of the source filing
        form_type: SEC form type of the source filing
        similarity: Cosine similarity score (0.0 to 1.0, higher is better)
        filing_date: Date of the source filing (optional)
        accession_number: SEC accession number (optional)
        chunk_id: ChromaDB document ID (optional)
        segment_index: Index of the parent segment within the filing
            (optional — present when the chunk metadata carries it).
        parent_content: Full text of the parent segment, attached by the
            search engine after the vector match. Decouples the small
            embedded chunk from the broader context shown to the user.
            ``None`` when the parent could not be resolved (e.g. a chunk
            ingested before the segments table was populated).
    """

    content: str
    path: str
    content_type: ContentType
    ticker: str
    form_type: str
    similarity: float
    filing_date: str | None = None
    accession_number: str | None = None
    chunk_id: str | None = None
    segment_index: int | None = None
    parent_content: str | None = None

    @classmethod
    def from_chromadb_result(
        cls,
        document: str,
        metadata: dict,
        distance: float,
        chunk_id: str | None = None,
    ) -> "SearchResult":
        """
        Create SearchResult from ChromaDB query output.

        ChromaDB returns cosine distance; this method converts it to
        similarity (1 - distance).

        Args:
            document: The chunk text content
            metadata: ChromaDB metadata dictionary
            distance: Cosine distance from ChromaDB (0.0 to 2.0)
            chunk_id: Optional document ID

        Returns:
            SearchResult instance with similarity score
        """
        raw_index = metadata.get("segment_index")
        # ChromaDB returns metadata values as the type they were stored as,
        # but be defensive: legacy rows ingested before the parent-context
        # change may omit segment_index entirely.
        segment_index = int(raw_index) if isinstance(raw_index, (int, float)) else None

        return cls(
            content=document,
            path=metadata.get("path", "(unknown)"),
            content_type=ContentType(metadata.get("content_type", "text")),
            ticker=metadata.get("ticker", ""),
            form_type=metadata.get("form_type", ""),
            similarity=1.0 - distance,
            filing_date=metadata.get("filing_date"),
            accession_number=metadata.get("accession_number"),
            chunk_id=chunk_id,
            segment_index=segment_index,
        )


@dataclass
class IngestResult:
    """
    Result of a filing ingestion operation.

    Returned by the pipeline orchestrator after successfully ingesting
    a filing, providing statistics for CLI output and logging.

    Attributes:
        filing_id: Identifier of the ingested filing
        segment_count: Number of segments extracted from HTML
        chunk_count: Number of chunks after splitting
        duration_seconds: Time taken for the full pipeline
    """

    filing_id: FilingIdentifier
    segment_count: int
    chunk_count: int
    duration_seconds: float
