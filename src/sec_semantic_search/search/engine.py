"""
Search engine for semantic search over SEC filings.

This module provides the high-level search interface that coordinates
query embedding and ChromaDB similarity search. It serves as the
primary entry point for the CLI and any future web interface.

Usage:
    from sec_semantic_search.search import SearchEngine

    engine = SearchEngine()
    results = engine.search("revenue and financial performance")
"""

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import (
    EmbeddingBusyError,
    SearchError,
    SearchResult,
    get_logger,
    redact_for_log,
)
from sec_semantic_search.database import ChromaDBClient, MetadataRegistry
from sec_semantic_search.pipeline import EmbeddingGenerator

logger = get_logger(__name__)


def _redact_filter(value: str | list[str] | None) -> str | list[str]:
    """Return a ticker filter as it may appear in a log line."""
    if not value:
        return "any"
    if isinstance(value, str):
        return redact_for_log(value)
    return [redact_for_log(v) for v in value]


class SearchEngine:
    """
    Facade for semantic search over ingested SEC filings.

    This class coordinates query embedding and vector similarity search,
    providing a single ``search()`` method that accepts a plain text query
    and returns ranked results. It reads defaults from ``SearchSettings``
    (``SEARCH_TOP_K``, ``SEARCH_MIN_SIMILARITY``) so callers can search
    with minimal arguments.

    The engine accepts optional pre-built dependencies so that the CLI
    layer can share an ``EmbeddingGenerator`` instance between ingestion
    and search (avoiding loading the model twice).

    Example:
        >>> engine = SearchEngine()
        >>> results = engine.search("risk factors related to supply chain")
        >>> for r in results:
        ...     print(f"[{r.similarity:.3f}] {r.path}")
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator | None = None,
        chroma_client: ChromaDBClient | None = None,
        registry: MetadataRegistry | None = None,
    ) -> None:
        """
        Initialise the search engine.

        Args:
            embedder: Pre-built embedding generator. If None, a new
                      instance is created (model loads lazily on first query).
            chroma_client: Pre-built ChromaDB client. If None, a new
                           instance is created using settings.
            registry: Pre-built metadata registry. Used to resolve each
                ChromaDB match back to its full parent segment for
                display. ``None`` disables parent-context lookup —
                ``SearchResult.parent_content`` will stay ``None`` and
                callers fall back to the chunk text. The API wires a
                singleton registry; CLI search builds one on demand.
        """
        self._embedder = embedder or EmbeddingGenerator()
        self._chroma_client = chroma_client or ChromaDBClient()
        self._registry = registry

        settings = get_settings()
        self._default_top_k = settings.search.top_k
        self._default_min_similarity = settings.search.min_similarity

        logger.debug(
            "SearchEngine initialised: top_k=%d, min_similarity=%.2f",
            self._default_top_k,
            self._default_min_similarity,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        ticker: str | list[str] | None = None,
        form_type: str | list[str] | None = None,
        min_similarity: float | None = None,
        accession_number: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        embed_timeout: float | None = None,
    ) -> list[SearchResult]:
        """
        Search ingested filings for chunks relevant to the query.

        The query is embedded using the same model used during ingestion,
        then matched against stored chunks via cosine similarity. Results
        below ``min_similarity`` are filtered out.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return. Defaults to
                   ``SEARCH_TOP_K`` from settings.
            ticker: Optional filter — only search filings from these
                ticker(s). Single string or list of strings.
            form_type: Optional filter — only search these form type(s)
                (e.g. "10-K", "10-Q"). Single string or list of strings.
            min_similarity: Minimum similarity threshold (0.0–1.0).
                            Defaults to ``SEARCH_MIN_SIMILARITY`` from settings.
            accession_number: Optional filter — restrict search to specific
                filing(s) by accession number. Single string or list.
            start_date: Optional lower bound for filing date (inclusive,
                ``YYYY-MM-DD``).
            end_date: Optional upper bound for filing date (inclusive,
                ``YYYY-MM-DD``).
            embed_timeout: Seconds to wait for the embedding model when
                another encode or a model load holds it. ``None`` waits
                without limit.

        Returns:
            List of ``SearchResult`` objects ordered by similarity
            (highest first), filtered by the minimum similarity threshold.

        Raises:
            EmbeddingBusyError: If ``embed_timeout`` elapses first.
            SearchError: If the query is empty or the search operation fails.
        """
        if not query or not query.strip():
            raise SearchError(
                "Empty search query",
                details="Cannot search with an empty or whitespace-only query.",
            )

        effective_top_k = top_k if top_k is not None else self._default_top_k
        effective_min_sim = (
            min_similarity if min_similarity is not None else self._default_min_similarity
        )

        # Never the query text: queries are not persisted (AD#29), and a log
        # file or Cloud Logging would keep it. The API route logs it once,
        # through ``redact_for_log``. Ticker filters reveal research interest
        # too, so they follow the same redaction setting.
        logger.info(
            "Searching (top_k=%d, min_similarity=%.2f, ticker=%s, form_type=%s)",
            effective_top_k,
            effective_min_sim,
            _redact_filter(ticker),
            form_type if form_type else "any",
        )

        try:
            query_embeddings = self._embedder.embed_query_for_chromadb(
                query, lock_timeout=embed_timeout
            )

            results = self._chroma_client.query(
                query_embeddings=query_embeddings,
                n_results=effective_top_k,
                ticker=ticker,
                form_type=form_type,
                accession_number=accession_number,
                start_date=start_date,
                end_date=end_date,
            )
        except (SearchError, EmbeddingBusyError):
            raise
        except Exception as e:
            raise SearchError(
                "Search failed",
                details=str(e),
            ) from e

        # Filter by minimum similarity threshold
        if effective_min_sim > 0.0:
            before_count = len(results)
            results = [r for r in results if r.similarity >= effective_min_sim]
            filtered_count = before_count - len(results)
            if filtered_count > 0:
                logger.debug(
                    "Filtered %d results below similarity threshold %.2f",
                    filtered_count,
                    effective_min_sim,
                )

        # Attach parent-segment text so the UI can show the broader
        # paragraph while still highlighting the short chunk that
        # produced the vector match. One batched lookup, not N.
        self._attach_parent_content(results)

        logger.info("Search returned %d results", len(results))
        return results

    def _attach_parent_content(self, results: list[SearchResult]) -> None:
        """Populate ``SearchResult.parent_content`` in place.

        Resolves each result back to its parent ``Segment`` via the
        metadata registry using one batched SQL query. Missing parents
        (legacy chunks without ``segment_index``, or rows lost to a
        previous partial rollback) silently fall through — the API still
        returns the chunk text via ``content``.
        """
        if self._registry is None or not results:
            return

        pairs: list[tuple[str, int]] = []
        for r in results:
            if r.accession_number and r.segment_index is not None:
                pairs.append((r.accession_number, r.segment_index))

        if not pairs:
            return

        try:
            parent_map = self._registry.get_parent_segments(pairs)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully on lookup failure
            logger.warning(
                "Parent-context lookup failed; returning chunk text only: %s",
                exc,
            )
            return

        for r in results:
            if r.accession_number and r.segment_index is not None:
                r.parent_content = parent_map.get((r.accession_number, r.segment_index))
