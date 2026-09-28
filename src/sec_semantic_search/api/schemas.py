"""
Pydantic v2 request and response schemas for the SEC Semantic Search API.

Schemas are separate from the core dataclasses in ``sec_semantic_search.core``
to provide a stable, explicit API contract.  Internal representations may
change without affecting the API surface.

Naming convention:
    - Request schemas:  ``<Resource>Request`` (e.g. ``SearchRequest``)
    - Response schemas: ``<Resource>Response`` or ``<Resource>Schema``
    - Nested schemas:   plain names without suffix (e.g. ``TickerBreakdown``)

All datetime strings are ISO 8601.  Similarity scores are floats in [0.0, 1.0].
"""

from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

from sec_semantic_search.config.constants import SUPPORTED_FORMS

# SEC ticker symbols: 1–5 uppercase letters, optionally with dots (e.g. BRK.B).
_TICKER_RE = re.compile(r"^[A-Z][A-Z.]{0,4}$")

# SEC accession number format: NNNNNNNNNN-YY-NNNNNN
_ACCESSION_RE = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
_DELETE_BY_IDS_MAX = 50


# ---------------------------------------------------------------------------
# Shared / error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """
    Structured error response returned for all 4xx and 5xx responses.

    Matches the CLI error format (error type, human message, optional hint).
    """

    error: str = Field(..., description="Machine-readable error type")
    message: str = Field(..., description="Human-readable error description")
    details: str | None = Field(None, description="Additional technical context")
    hint: str | None = Field(None, description="Suggested remediation action")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TickerBreakdown(BaseModel):
    """Per-ticker statistics for the status response."""

    ticker: str
    filings: int = Field(..., ge=0)
    chunks: int = Field(..., ge=0)
    forms: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """
    Response for ``GET /api/status/``.

    Provides a full overview of database contents and capacity, plus
    deployment flags that the frontend needs to adapt its UI.
    """

    filing_count: int = Field(..., ge=0, description="Total ingested filings")
    max_filings: int = Field(..., ge=1, description="Configured maximum filing count")
    chunk_count: int = Field(..., ge=0, description="Total chunks in vector store")
    tickers: list[str] = Field(default_factory=list, description="Unique ticker symbols")
    form_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Filing count per form type (e.g. {'10-K': 5, '10-Q': 12})",
    )
    ticker_breakdown: list[TickerBreakdown] = Field(
        default_factory=list,
        description="Per-ticker filing and chunk statistics",
    )
    edgar_session_required: bool = Field(
        False,
        description=(
            "True when server-side EDGAR credentials are unset and "
            "EDGAR_SESSION_REQUIRED=true — frontend must show Welcome screen"
        ),
    )
    demo_mode: bool = Field(
        False,
        description="True when DEMO_MODE is enabled — frontend shows banner",
    )
    is_admin: bool = Field(
        False,
        description=(
            "True when the request carries a valid admin key, or when "
            "ADMIN_API_KEY is not configured (Scenario A)"
        ),
    )


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------


class FilingSchema(BaseModel):
    """
    A single filing record as returned by listing endpoints.

    Mirrors ``database.metadata.FilingRecord`` without the internal ``id``.
    """

    ticker: str
    form_type: str
    filing_date: str = Field(..., description="ISO date (YYYY-MM-DD)")
    accession_number: str
    chunk_count: int = Field(..., ge=0)
    ingested_at: str = Field(..., description="ISO 8601 UTC timestamp")


class FilingListResponse(BaseModel):
    """Response for ``GET /api/filings/``."""

    filings: list[FilingSchema]
    total: int = Field(..., ge=0)


class DeleteResponse(BaseModel):
    """Response for ``DELETE /api/filings/{accession}``."""

    accession_number: str
    chunks_deleted: int = Field(..., ge=0)


class BulkDeleteRequest(BaseModel):
    """Request body for ``POST /api/filings/bulk-delete``."""

    ticker: str | None = Field(None, description="Filter by ticker symbol")
    form_type: str | None = Field(None, description="Filter by form type (8-K, 10-K, or 10-Q)")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str | None) -> str | None:
        """Normalise and validate ticker format."""
        if v is None:
            return None
        upper = v.upper().strip()
        if not _TICKER_RE.match(upper):
            msg = f"Invalid ticker symbol: '{v}'. Expected 1–5 uppercase letters (e.g. AAPL, BRK.B)"
            raise ValueError(msg)
        return upper

    @field_validator("form_type")
    @classmethod
    def validate_form_type(cls, v: str | None) -> str | None:
        """Normalise and validate form_type to uppercase."""
        if v is None:
            return None
        upper = v.upper()
        if upper not in SUPPORTED_FORMS:
            allowed = ", ".join(SUPPORTED_FORMS)
            msg = f"form_type must be one of: {allowed}"
            raise ValueError(msg)
        return upper


class BulkDeleteResponse(BaseModel):
    """Response for ``POST /api/filings/bulk-delete``."""

    filings_deleted: int = Field(..., ge=0)
    chunks_deleted: int = Field(..., ge=0)
    tickers_affected: list[str] = Field(default_factory=list)


class DeleteByIdsRequest(BaseModel):
    """Request body for ``POST /api/filings/delete-by-ids``."""

    accession_numbers: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Accession numbers to delete. Limited to 50 per request; "
            "use admin bulk-delete or clear-all for larger operations"
        ),
    )

    @field_validator("accession_numbers")
    @classmethod
    def validate_accession_numbers(cls, v: list[str]) -> list[str]:
        """Validate all accession numbers match SEC format."""
        if len(v) > _DELETE_BY_IDS_MAX:
            msg = (
                f"At most {_DELETE_BY_IDS_MAX} accession numbers are allowed per request. "
                "Use bulk-delete or clear-all for larger admin operations."
            )
            raise ValueError(msg)
        invalid = [a for a in v if not _ACCESSION_RE.match(a)]
        if invalid:
            msg = f"Invalid accession number format: {invalid[:3]}. Expected NNNNNNNNNN-YY-NNNNNN"
            raise ValueError(msg)
        return v


class DeleteByIdsResponse(BaseModel):
    """Response for ``POST /api/filings/delete-by-ids``."""

    filings_deleted: int = Field(..., ge=0)
    chunks_deleted: int = Field(..., ge=0)
    not_found: list[str] = Field(
        default_factory=list,
        description="Accession numbers that were not found in the registry",
    )


class ClearAllResponse(BaseModel):
    """Response for ``DELETE /api/filings/?confirm=true``."""

    filings_deleted: int = Field(..., ge=0)
    chunks_deleted: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class ParentSegmentSchema(BaseModel):
    """
    Parent segment text shared by every result that references it.

    Segments longer than the API's cap are sent as an excerpt around the
    matched chunks; the two flags say which ends were cut.
    """

    content: str
    truncated_start: bool = Field(False, description="Text before the excerpt was omitted")
    truncated_end: bool = Field(False, description="Text after the excerpt was omitted")


class SearchResultSchema(BaseModel):
    """
    A single result from a semantic search query.

    Uses ``content_type`` as a plain string (not the ``ContentType`` enum)
    to keep the API contract stable and framework-agnostic.

    ``content`` is the short embedded chunk (what the vector match was
    scored against). ``parent`` is the broader segment the chunk was
    carved out of — clients render it and highlight ``content`` inside
    it. Each segment is sent once, with the first result that cites it;
    later results citing the same segment carry only ``parent_key``.
    ``parent_key`` is ``None`` when the parent adds nothing to
    ``content`` (the segment is a single chunk) or is unavailable
    (legacy data with no segment row); clients then display ``content``.
    """

    content: str
    path: str
    content_type: str = Field(..., description="One of: 'text', 'textsmall', 'table'")
    ticker: str
    form_type: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    filing_date: str | None = Field(None, description="ISO date (YYYY-MM-DD)")
    accession_number: str | None = None
    chunk_id: str | None = None
    segment_index: int | None = Field(
        None,
        description="Zero-based position of the parent segment within the filing",
    )
    parent_key: str | None = Field(
        None,
        description="Parent segment key ('{accession}:{segment_index}')",
    )
    parent: ParentSegmentSchema | None = Field(
        None,
        description="Parent segment text; only on the first result citing parent_key",
    )


class SearchRequest(BaseModel):
    """
    Request body for ``POST /api/search/``.

    Filter fields (``ticker``, ``form_type``, ``accession_number``) accept
    either a single string or a list of strings.  A single string is
    normalised to a one-element list for uniform downstream handling.
    """

    query: str = Field(
        ..., min_length=1, max_length=2000, description="Natural language search query"
    )
    top_k: int = Field(5, ge=1, le=100, description="Maximum number of results")
    ticker: list[str] | None = Field(None, description="Filter to specific ticker(s)")
    form_type: list[str] | None = Field(
        None, description="Filter to form type(s) (e.g. '8-K', '10-K', '10-Q')"
    )
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    accession_number: list[str] | None = Field(
        None, description="Restrict search to specific filing(s) by accession number"
    )
    start_date: str | None = Field(
        None, description="Lower bound for filing date (inclusive, YYYY-MM-DD)"
    )
    end_date: str | None = Field(
        None, description="Upper bound for filing date (inclusive, YYYY-MM-DD)"
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validate that date strings are well-formed ``YYYY-MM-DD``."""
        if v is None:
            return None
        try:
            date.fromisoformat(v)
        except ValueError:
            msg = f"Invalid date format: '{v}'. Expected YYYY-MM-DD"
            raise ValueError(msg) from None
        return v

    @field_validator("accession_number", mode="before")
    @classmethod
    def coerce_accession_number(cls, v: str | list[str] | None) -> list[str] | None:
        """Wrap a single string in a list, then validate format."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        invalid = [a for a in v if not _ACCESSION_RE.match(a)]
        if invalid:
            msg = f"Invalid accession number format: {invalid[:3]}. Expected NNNNNNNNNN-YY-NNNNNN"
            raise ValueError(msg)
        return v

    @field_validator("form_type", mode="before")
    @classmethod
    def coerce_form_type(cls, v: str | list[str] | None) -> list[str] | None:
        """Wrap a single string in a list, then normalise and validate."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        normalised = [f.upper() for f in v]
        invalid = [f for f in normalised if f not in SUPPORTED_FORMS]
        if invalid:
            allowed = ", ".join(SUPPORTED_FORMS)
            msg = f"form_type must be one of: {allowed}; got: {invalid}"
            raise ValueError(msg)
        return normalised

    @field_validator("ticker", mode="before")
    @classmethod
    def coerce_ticker(cls, v: str | list[str] | None) -> list[str] | None:
        """Wrap a single string in a list, then normalise and validate."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        result = [t.upper().strip() for t in v]
        invalid = [t for t in result if not _TICKER_RE.match(t)]
        if invalid:
            msg = f"Invalid ticker symbol(s): {invalid}. Expected 1–5 uppercase letters (e.g. AAPL, BRK.B)"
            raise ValueError(msg)
        return result


class SearchResponse(BaseModel):
    """Response for ``POST /api/search/``.

    The query is intentionally omitted from the response to avoid
    echoing sensitive user input back over the wire (see
    SECURITY VULNERABILITIES.md §F4).  The client already holds the
    query in local state.
    """

    results: list[SearchResultSchema]
    total_results: int = Field(..., ge=0)
    search_time_ms: float = Field(..., ge=0.0, description="Wall-clock search time in ms")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TaskProgress(BaseModel):
    """Progress snapshot for a running ingestion task."""

    current_ticker: str | None = None
    current_form_type: str | None = None
    step_label: str = Field("", description="Human-readable step description")
    step_index: int = Field(0, ge=0, description="Current pipeline step (0-based)")
    step_total: int = Field(5, ge=1, description="Total pipeline steps")
    filings_done: int = Field(0, ge=0)
    filings_total: int = Field(0, ge=0)
    filings_skipped: int = Field(0, ge=0)
    filings_failed: int = Field(0, ge=0)


class IngestResultSchema(BaseModel):
    """Result for a single filing that was successfully ingested."""

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    segment_count: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0.0)


class TaskStatus(BaseModel):
    """
    Full status of an ingestion task.

    Returned by ``GET /api/ingest/tasks/{task_id}`` and pushed via
    WebSocket as the task progresses.
    """

    task_id: str
    status: str = Field(
        ..., description="One of: 'pending', 'running', 'completed', 'failed', 'cancelled'"
    )
    tickers: list[str]
    form_types: list[str]
    progress: TaskProgress
    results: list[IngestResultSchema] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IngestRequest(BaseModel):
    """
    Request body for ``POST /api/ingest/add`` and ``POST /api/ingest/batch``.

    Mirrors the CLI ``ingest add`` flags with explicit typing.
    """

    tickers: list[str] = Field(..., min_length=1, description="Ticker symbols to ingest")
    form_types: list[str] = Field(
        default=["10-K", "10-Q"],
        description="SEC form types to ingest (8-K, 10-K, 10-Q)",
    )
    count_mode: str = Field(
        "latest",
        description=(
            "'latest': 1 filing per form per ticker; "
            "'total': count filings shared across forms; "
            "'per_form': count filings per form type"
        ),
    )
    count: int | None = Field(
        None,
        ge=1,
        description="Number of filings (None = all matching when date filters active)",
    )
    year: int | None = Field(None, ge=1993, description="Filter to a specific filing year")
    start_date: str | None = Field(None, description="ISO date (YYYY-MM-DD)")
    end_date: str | None = Field(None, description="ISO date (YYYY-MM-DD)")

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        """Normalise tickers to uppercase, strip whitespace, and validate format."""
        result = [t.upper().strip() for t in v if t.strip()]
        invalid = [t for t in result if not _TICKER_RE.match(t)]
        if invalid:
            msg = f"Invalid ticker symbol(s): {invalid}. Expected 1–5 uppercase letters (e.g. AAPL, BRK.B)"
            raise ValueError(msg)
        return result

    @field_validator("form_types")
    @classmethod
    def validate_form_types(cls, v: list[str]) -> list[str]:
        """Normalise and validate form types."""
        normalised = [f.upper().strip() for f in v]
        invalid = [f for f in normalised if f not in SUPPORTED_FORMS]
        if invalid:
            allowed = ", ".join(SUPPORTED_FORMS)
            msg = f"Unsupported form types: {invalid}. Allowed: {allowed}"
            raise ValueError(msg)
        return normalised

    @field_validator("count_mode")
    @classmethod
    def validate_count_mode(cls, v: str) -> str:
        """Validate count mode."""
        allowed = ("latest", "total", "per_form")
        if v not in allowed:
            msg = f"count_mode must be one of {allowed}"
            raise ValueError(msg)
        return v


class TaskResponse(BaseModel):
    """
    Immediate response when an ingest task is created.

    The caller uses ``task_id`` to poll status or connect via WebSocket.
    """

    task_id: str
    status: str = "pending"
    websocket_url: str = Field(..., description="WebSocket URL for real-time progress")


class TaskListResponse(BaseModel):
    """Response for ``GET /api/ingest/tasks``."""

    tasks: list[TaskStatus]
    total: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# GPU / resources
# ---------------------------------------------------------------------------


class GPUStatusResponse(BaseModel):
    """
    Response for ``GET /api/resources/gpu``.

    Reports whether the embedding model is loaded and on which device.
    """

    model_loaded: bool
    device: str | None = Field(None, description="'cuda' or 'cpu' when model is loaded")
    model_name: str
    approximate_vram_mb: int | None = Field(
        None,
        description="Approximate VRAM usage in MB (None if not loaded or CPU)",
    )


class GPUUnloadResponse(BaseModel):
    """Response for ``DELETE /api/resources/gpu``."""

    status: str = Field(..., description="'unloaded' or 'already_unloaded'")
