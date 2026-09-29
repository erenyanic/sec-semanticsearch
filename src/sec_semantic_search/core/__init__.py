"""
Core module — types, exceptions, and logging.

This module provides the foundational components used throughout the package:
    - Data types (FilingIdentifier, Segment, Chunk, SearchResult, IngestResult)
    - Exception hierarchy (SECSemanticSearchError and subclasses)
    - Logging utilities (get_logger, configure_logging)

Usage:
    from sec_semantic_search.core import (
        FilingIdentifier,
        Segment,
        Chunk,
        SearchResult,
        FetchError,
        get_logger,
    )
"""

from sec_semantic_search.core.exceptions import (
    ChunkingError,
    DatabaseError,
    EmbeddingBusyError,
    EmbeddingError,
    FetchError,
    FilingLimitExceededError,
    ParseError,
    SearchError,
    SECSemanticSearchError,
)
from sec_semantic_search.core.logging import (
    audit_log,
    configure_logging,
    get_logger,
    redact_for_log,
    suppress_third_party_loggers,
)
from sec_semantic_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    IngestResult,
    SearchResult,
    Segment,
)

__all__ = [
    # Types
    "ContentType",
    "FilingIdentifier",
    "Segment",
    "Chunk",
    "SearchResult",
    "IngestResult",
    # Exceptions
    "SECSemanticSearchError",
    "FetchError",
    "ParseError",
    "ChunkingError",
    "EmbeddingError",
    "EmbeddingBusyError",
    "DatabaseError",
    "FilingLimitExceededError",
    "SearchError",
    # Logging
    "audit_log",
    "get_logger",
    "configure_logging",
    "redact_for_log",
    "suppress_third_party_loggers",
]
