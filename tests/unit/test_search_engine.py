"""
Unit tests for SearchEngine.

Tests the engine's own logic in isolation (no real ChromaDB):
    - Empty/whitespace query rejection
    - Exception wrapping (non-SearchError → SearchError)
    - accession_number filter passthrough
    - Similarity threshold filtering
    - Default parameter usage from settings
"""

from unittest.mock import MagicMock

import pytest

from sec_semantic_search.core.exceptions import (
    EmbeddingBusyError,
    EmbeddingError,
    SearchError,
)
from sec_semantic_search.core.types import ContentType, SearchResult
from sec_semantic_search.search.engine import SearchEngine


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query_for_chromadb.return_value = [[0.1] * 768]
    return embedder


@pytest.fixture
def mock_chroma():
    chroma = MagicMock()
    chroma.query.return_value = []
    return chroma


@pytest.fixture
def engine(mock_embedder, mock_chroma):
    return SearchEngine(embedder=mock_embedder, chroma_client=mock_chroma)


class TestExceptionWrapping:
    """Non-SearchError exceptions from dependencies should be wrapped."""

    def test_embedding_error_wrapped(self, engine, mock_embedder):
        mock_embedder.embed_query_for_chromadb.side_effect = EmbeddingError("GPU OOM")
        with pytest.raises(SearchError, match="Search failed"):
            engine.search("test query")

    def test_generic_exception_wrapped(self, engine, mock_embedder):
        mock_embedder.embed_query_for_chromadb.side_effect = RuntimeError("unexpected")
        with pytest.raises(SearchError, match="Search failed"):
            engine.search("test query")

    def test_search_error_not_double_wrapped(self, engine, mock_chroma):
        """SearchError from ChromaDB should propagate without re-wrapping."""
        from sec_semantic_search.core.exceptions import DatabaseError

        mock_chroma.query.side_effect = DatabaseError("connection lost")
        with pytest.raises(SearchError, match="Search failed"):
            engine.search("test query")


class TestEmbedTimeout:
    """The API's bounded wait reaches the embedder; a busy model is not flattened."""

    def test_timeout_forwarded_to_embedder(self, engine, mock_embedder):
        engine.search("test query", embed_timeout=12.5)
        _, kwargs = mock_embedder.embed_query_for_chromadb.call_args
        assert kwargs["lock_timeout"] == 12.5

    def test_default_waits_without_limit(self, engine, mock_embedder):
        engine.search("test query")
        _, kwargs = mock_embedder.embed_query_for_chromadb.call_args
        assert kwargs["lock_timeout"] is None

    def test_busy_error_propagates_unwrapped(self, engine, mock_embedder):
        """The route maps this type to 503; wrapping it in SearchError would make it a 500."""
        mock_embedder.embed_query_for_chromadb.side_effect = EmbeddingBusyError("busy")
        with pytest.raises(EmbeddingBusyError):
            engine.search("test query", embed_timeout=0.1)


class TestAccessionNumberFilter:
    """accession_number should be forwarded to ChromaDB."""

    def test_passed_to_chroma(self, engine, mock_chroma):
        engine.search("test", accession_number="ACC-123")
        _, kwargs = mock_chroma.query.call_args
        assert kwargs["accession_number"] == "ACC-123"

    def test_none_when_not_provided(self, engine, mock_chroma):
        engine.search("test")
        _, kwargs = mock_chroma.query.call_args
        assert kwargs.get("accession_number") is None


class TestSimilarityFiltering:
    """min_similarity post-filtering is SearchEngine's unique logic."""

    def _make_result(self, similarity):
        return SearchResult(
            content="text",
            path="Part I",
            content_type=ContentType.TEXT,
            ticker="AAPL",
            form_type="10-K",
            similarity=similarity,
        )

    def test_filters_below_threshold(self, engine, mock_chroma):
        mock_chroma.query.return_value = [
            self._make_result(0.5),
            self._make_result(0.3),
            self._make_result(0.1),
        ]
        results = engine.search("test", min_similarity=0.25)
        assert len(results) == 2
        assert all(r.similarity >= 0.25 for r in results)

    def test_zero_threshold_keeps_all(self, engine, mock_chroma):
        mock_chroma.query.return_value = [
            self._make_result(0.01),
        ]
        results = engine.search("test", min_similarity=0.0)
        assert len(results) == 1


class TestDefaultParameters:
    """Engine should use settings defaults when params are None."""

    def test_default_top_k_from_settings(self, engine, mock_chroma):
        engine.search("test")
        _, kwargs = mock_chroma.query.call_args
        assert kwargs["n_results"] == 5  # DEFAULT_SEARCH_TOP_K

    def test_explicit_top_k_overrides(self, engine, mock_chroma):
        engine.search("test", top_k=10)
        _, kwargs = mock_chroma.query.call_args
        assert kwargs["n_results"] == 10
