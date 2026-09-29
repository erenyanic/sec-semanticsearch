"""
Tests for the PipelineOrchestrator.

The orchestrator coordinates parser → chunker → embedder. We inject
mock components via its dependency injection constructor to test the
orchestration logic without real HTML parsing or GPU embedding.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from sec_semantic_search.config.constants import EMBEDDING_DIMENSION
from sec_semantic_search.core.types import ContentType, Segment
from sec_semantic_search.pipeline.orchestrator import PipelineOrchestrator, ProcessedFiling


@pytest.fixture
def mock_parser(sample_filing_id):
    """A mock parser that returns two segments."""
    parser = MagicMock()
    parser.parse.return_value = [
        Segment(
            path="Part I",
            content_type=ContentType.TEXT,
            content="Segment one content.",
            filing_id=sample_filing_id,
        ),
        Segment(
            path="Part II",
            content_type=ContentType.TEXT,
            content="Segment two content.",
            filing_id=sample_filing_id,
        ),
    ]
    return parser


@pytest.fixture
def mock_chunker(sample_chunks):
    """A mock chunker that returns pre-built sample chunks."""
    chunker = MagicMock()
    chunker.chunk_segments.return_value = sample_chunks
    return chunker


@pytest.fixture
def mock_embedder(sample_chunks):
    """A mock embedder that returns correctly shaped random arrays."""
    embedder = MagicMock()
    embedder.embed_chunks.return_value = np.random.default_rng(42).random(
        (len(sample_chunks), EMBEDDING_DIMENSION), dtype=np.float32
    )
    return embedder


@pytest.fixture
def orchestrator(mock_parser, mock_chunker, mock_embedder):
    """An orchestrator with all components mocked."""
    return PipelineOrchestrator(
        parser=mock_parser,
        chunker=mock_chunker,
        embedder=mock_embedder,
    )


# -----------------------------------------------------------------------
# process_filing
# -----------------------------------------------------------------------


class TestProcessFiling:
    """process_filing() runs parse → chunk → embed on provided HTML."""

    def test_returns_processed_filing(self, orchestrator, sample_filing_id):
        result = orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        assert isinstance(result, ProcessedFiling)

    def test_contains_chunks(self, orchestrator, sample_filing_id, sample_chunks):
        result = orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        assert result.chunks == sample_chunks

    def test_contains_embeddings(self, orchestrator, sample_filing_id, sample_chunks):
        result = orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        assert result.embeddings.shape == (len(sample_chunks), EMBEDDING_DIMENSION)

    def test_ingest_result_statistics(self, orchestrator, sample_filing_id, sample_chunks):
        result = orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        stats = result.ingest_result
        assert stats.filing_id is sample_filing_id
        assert stats.segment_count == 2  # mock_parser returns 2 segments
        assert stats.chunk_count == len(sample_chunks)
        assert stats.duration_seconds >= 0.0

    def test_calls_parser_with_html(self, orchestrator, sample_filing_id, mock_parser):
        orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        mock_parser.parse.assert_called_once_with("<html>test</html>", sample_filing_id)

    def test_calls_chunker_with_segments(
        self, orchestrator, sample_filing_id, mock_chunker, mock_parser
    ):
        orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        mock_chunker.chunk_segments.assert_called_once_with(mock_parser.parse.return_value)

    def test_calls_embedder_with_chunks(
        self, orchestrator, sample_filing_id, mock_embedder, sample_chunks
    ):
        orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        mock_embedder.embed_chunks.assert_called_once_with(sample_chunks, show_progress=False)


# -----------------------------------------------------------------------
# Progress callback
# -----------------------------------------------------------------------


class TestProgressCallback:
    """process_filing() should invoke the progress callback at each stage."""

    def test_callback_called(self, orchestrator, sample_filing_id):
        callback = MagicMock()
        orchestrator.process_filing(
            sample_filing_id, "<html>test</html>", progress_callback=callback
        )
        # Should be called for: Parsing, Chunking, Embedding, Complete
        assert callback.call_count == 4

    def test_callback_receives_step_names(self, orchestrator, sample_filing_id):
        steps = []

        def capture(step, current, total):
            steps.append(step)

        orchestrator.process_filing(
            sample_filing_id, "<html>test</html>", progress_callback=capture
        )
        assert "Parsing" in steps
        assert "Chunking" in steps
        assert "Embedding" in steps
        assert "Complete" in steps

    def test_no_callback_ok(self, orchestrator, sample_filing_id):
        """Should work without a callback (default None)."""
        result = orchestrator.process_filing(sample_filing_id, "<html>test</html>")
        assert result is not None
