"""
Custom exception hierarchy for SEC-SemanticSearch.

All exceptions inherit from SECSemanticSearchError, allowing callers to catch
all project-specific errors with a single except clause when desired.

Exception hierarchy:
    SECSemanticSearchError (base)
    ├── FetchError — SEC EDGAR API or network failures
    ├── ParseError — HTML parsing failures (doc2dict)
    ├── ChunkingError — Text chunking failures
    ├── EmbeddingError — Embedding generation failures
    │   └── EmbeddingBusyError — Model held by another caller past the wait bound
    ├── DatabaseError — ChromaDB or SQLite failures
    │   └── FilingLimitExceededError — Maximum filing count reached
    └── SearchError — Search operation failures
"""


class SECSemanticSearchError(Exception):
    """
    Base exception for all SEC-SemanticSearch errors.

    Args:
        message: Human-readable error description.
        details: Optional additional context for debugging.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


class FetchError(SECSemanticSearchError):
    """
    Raised when fetching SEC filings fails.

    Examples:
        - Network connectivity issues
        - SEC EDGAR API rate limiting
        - Invalid ticker symbol
        - Filing not found for specified form type
    """

    pass


class ParseError(SECSemanticSearchError):
    """
    Raised when parsing filing HTML fails.

    Examples:
        - Malformed HTML content
        - Unexpected document structure
        - doc2dict library errors
    """

    pass


class ChunkingError(SECSemanticSearchError):
    """
    Raised when text chunking fails.

    Examples:
        - Empty content after parsing
        - Chunking algorithm failures
    """

    pass


class EmbeddingError(SECSemanticSearchError):
    """
    Raised when embedding generation fails.

    Examples:
        - Model loading failures
        - GPU memory exhaustion
        - Invalid input to embedding model
    """

    pass


class EmbeddingBusyError(EmbeddingError):
    """
    Raised when the embedding model could not be acquired in time.

    Only callers that pass a ``lock_timeout`` to ``EmbeddingGenerator``
    can see this. The API maps it to ``503 Service Unavailable`` so a
    search queued behind an ingest batch or a model load fails fast and
    retryably instead of holding a worker thread indefinitely.
    """

    pass


class DatabaseError(SECSemanticSearchError):
    """
    Raised when database operations fail.

    Examples:
        - ChromaDB connection failures
        - SQLite write errors
        - Collection not found
    """

    pass


class FilingLimitExceededError(DatabaseError):
    """
    Raised when the maximum filing limit is reached.

    This is a soft limit for portfolio project scope, configurable via
    DB_MAX_FILINGS environment variable.
    """

    def __init__(
        self,
        current_count: int,
        max_filings: int,
        details: str | None = None,
    ) -> None:
        self.current_count = current_count
        self.max_filings = max_filings
        message = (
            f"Filing limit exceeded: {current_count}/{max_filings} filings stored. "
            f"Remove existing filings or increase DB_MAX_FILINGS."
        )
        super().__init__(message, details)


class SearchError(SECSemanticSearchError):
    """
    Raised when search operations fail.

    Examples:
        - Empty query string
        - No filings ingested
        - ChromaDB query failures
    """

    pass
