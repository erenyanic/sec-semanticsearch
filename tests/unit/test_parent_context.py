"""
Tests for the parent-context separation feature.

The vector chunk stored in ChromaDB and the parent segment text shown
to the user are now decoupled: chunks carry a ``segment_index`` in
their ChromaDB metadata, parent segments live in a dedicated SQLite
table, and the search engine performs a single batched join to attach
``parent_content`` to each result.

These tests exercise the contract at each layer in isolation:

    1. Parser assigns sequential ``segment_index``.
    2. Chunker propagates ``segment_index`` to every derived chunk and
       includes it in ``Chunk.to_metadata()``.
    3. ``MetadataRegistry`` persists segments alongside their filing,
       bulk-fetches them by (accession, segment_index), and cascades on
       delete / clear_all.
    4. ``SearchEngine`` joins ChromaDB results against the registry and
       degrades gracefully when the parent cannot be resolved.
    5. ``SearchResult.from_chromadb_result`` correctly reads
       ``segment_index`` from metadata, including legacy rows that omit
       the field entirely.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from sec_semantic_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    SearchResult,
    Segment,
)
from sec_semantic_search.database.metadata import MetadataRegistry
from sec_semantic_search.pipeline.chunk import TextChunker
from sec_semantic_search.search.engine import SearchEngine
from tests.helpers import count_segments

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        accession_number="0000320193-24-000999",
    )


@pytest.fixture
def long_segments(filing_id) -> list[Segment]:
    """Two long segments — at least one will split into multiple chunks
    so we can verify all derived chunks share the same parent index."""
    text_long = (
        "Revenue increased by ten percent year over year. "
        "Gross margin improved due to cost reductions. "
        "Operating expenses remained flat compared to prior year. "
        "Net income grew substantially across all segments. "
        "Free cash flow exceeded expectations driven by working capital efficiency. "
        "Capital expenditure focused on data centre infrastructure. "
        "Share repurchases accelerated in the second half. "
        "Dividend payments remained consistent with prior policy."
    )
    return [
        Segment(
            path="Part II > Item 7 > MD&A",
            content_type=ContentType.TEXT,
            content=text_long,
            filing_id=filing_id,
            segment_index=0,
        ),
        Segment(
            path="Part I > Item 1A > Risk Factors",
            content_type=ContentType.TEXTSMALL,
            content="Operating in regulated markets presents compliance challenges.",
            filing_id=filing_id,
            segment_index=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Chunker → segment_index propagation
# ---------------------------------------------------------------------------


class TestChunkerSegmentIndexPropagation:
    def test_all_chunks_carry_source_segment_index(self, filing_id, long_segments):
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=5)
        chunks = chunker.chunk_segments(long_segments)

        # Each chunk must report the segment_index of whichever source
        # segment it was carved from. Both segments must produce chunks.
        seen_indices = {c.segment_index for c in chunks}
        assert seen_indices == {0, 1}

    def test_segment_index_in_chunk_metadata(self, filing_id):
        seg = Segment(
            path="Root",
            content_type=ContentType.TEXT,
            content="Short single-sentence segment.",
            filing_id=filing_id,
            segment_index=7,
        )
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=0)
        chunks = chunker.chunk_segment(seg)
        assert len(chunks) == 1
        meta = chunks[0].to_metadata()
        assert meta["segment_index"] == 7

    def test_long_segment_multiple_chunks_share_index(self, filing_id, long_segments):
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=5)
        # First segment of long_segments is the multi-chunk one
        chunks = chunker.chunk_segment(long_segments[0])
        assert len(chunks) >= 2
        assert {c.segment_index for c in chunks} == {0}


# ---------------------------------------------------------------------------
# Registry: segments table CRUD + cascade
# ---------------------------------------------------------------------------


class TestRegistrySegmentsTable:
    def test_register_filing_persists_segments(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(
            filing_id,
            chunk_count=5,
            segments=long_segments,
        )
        assert count_segments(registry, filing_id.accession_number) == len(long_segments)

    def test_register_filing_if_new_persists_segments(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        inserted = registry.register_filing_if_new(
            filing_id,
            chunk_count=5,
            segments=long_segments,
        )
        assert inserted is True
        assert count_segments(registry, filing_id.accession_number) == len(long_segments)

    def test_register_filing_if_new_duplicate_does_not_double_insert(
        self, tmp_db_path, filing_id, long_segments
    ):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing_if_new(filing_id, chunk_count=5, segments=long_segments)
        inserted = registry.register_filing_if_new(filing_id, chunk_count=5, segments=long_segments)
        assert inserted is False
        # Count should still match the first insert — no duplication.
        assert count_segments(registry, filing_id.accession_number) == len(long_segments)

    def test_get_parent_segments_returns_text_by_pair(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(filing_id, chunk_count=5, segments=long_segments)

        pairs = [(filing_id.accession_number, 0), (filing_id.accession_number, 1)]
        parents = registry.get_parent_segments(pairs)
        assert parents[(filing_id.accession_number, 0)] == long_segments[0].content
        assert parents[(filing_id.accession_number, 1)] == long_segments[1].content

    def test_get_parent_segments_skips_missing(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(filing_id, chunk_count=5, segments=long_segments)

        # 99 was never inserted — must simply be absent, not raise.
        parents = registry.get_parent_segments(
            [(filing_id.accession_number, 0), (filing_id.accession_number, 99)]
        )
        assert (filing_id.accession_number, 0) in parents
        assert (filing_id.accession_number, 99) not in parents

    def test_get_parent_segments_empty_input(self, tmp_db_path):
        registry = MetadataRegistry(db_path=tmp_db_path)
        assert registry.get_parent_segments([]) == {}

    def test_remove_filing_cascades_segments(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(filing_id, chunk_count=5, segments=long_segments)
        assert count_segments(registry, filing_id.accession_number) == len(long_segments)

        registry.remove_filing(filing_id.accession_number)
        assert count_segments(registry, filing_id.accession_number) == 0

    def test_remove_filings_batch_cascades_segments(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(filing_id, chunk_count=5, segments=long_segments)

        registry.remove_filings_batch([filing_id.accession_number])
        assert count_segments(registry, filing_id.accession_number) == 0

    def test_clear_all_cascades_segments(self, tmp_db_path, filing_id, long_segments):
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(filing_id, chunk_count=5, segments=long_segments)
        assert count_segments(registry) == len(long_segments)

        registry.clear_all()
        assert count_segments(registry) == 0


# ---------------------------------------------------------------------------
# SearchEngine: parent-context attachment
# ---------------------------------------------------------------------------


def _make_result(
    accession: str = "0000320193-24-000999",
    segment_index: int | None = 0,
    content: str = "chunk text",
) -> SearchResult:
    return SearchResult(
        content=content,
        path="Part II > Item 7",
        content_type=ContentType.TEXT,
        ticker="AAPL",
        form_type="10-K",
        similarity=0.5,
        accession_number=accession,
        segment_index=segment_index,
    )


class TestSearchEngineParentContext:
    def test_parent_content_attached_when_registry_provided(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query_for_chromadb.return_value = [[0.1] * 768]

        mock_chroma = MagicMock()
        mock_chroma.query.return_value = [
            _make_result(content="chunk text"),
        ]

        mock_registry = MagicMock()
        mock_registry.get_parent_segments.return_value = {
            (
                "0000320193-24-000999",
                0,
            ): "Broader paragraph containing chunk text and surrounding sentences.",
        }

        engine = SearchEngine(
            embedder=mock_embedder, chroma_client=mock_chroma, registry=mock_registry
        )
        results = engine.search("anything")
        assert len(results) == 1
        assert results[0].parent_content == (
            "Broader paragraph containing chunk text and surrounding sentences."
        )

    def test_parent_content_none_when_registry_omitted(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query_for_chromadb.return_value = [[0.1] * 768]
        mock_chroma = MagicMock()
        mock_chroma.query.return_value = [_make_result()]

        # No registry passed — parent_content stays None and the call
        # never reaches a non-existent dependency.
        engine = SearchEngine(embedder=mock_embedder, chroma_client=mock_chroma)
        results = engine.search("anything")
        assert results[0].parent_content is None

    def test_parent_content_lookup_failure_degrades_gracefully(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query_for_chromadb.return_value = [[0.1] * 768]
        mock_chroma = MagicMock()
        mock_chroma.query.return_value = [_make_result()]

        mock_registry = MagicMock()
        # A failing registry must not break search — fall back to chunk text.
        mock_registry.get_parent_segments.side_effect = RuntimeError("db down")

        engine = SearchEngine(
            embedder=mock_embedder, chroma_client=mock_chroma, registry=mock_registry
        )
        results = engine.search("anything")
        assert results[0].parent_content is None
        assert results[0].content == "chunk text"

    def test_results_without_segment_index_skipped(self):
        """Legacy chunks (no segment_index in metadata) get no parent lookup."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query_for_chromadb.return_value = [[0.1] * 768]
        mock_chroma = MagicMock()
        mock_chroma.query.return_value = [
            _make_result(segment_index=None),
        ]
        mock_registry = MagicMock()

        engine = SearchEngine(
            embedder=mock_embedder, chroma_client=mock_chroma, registry=mock_registry
        )
        engine.search("anything")
        # No segment_index → no pair gathered → no batched call.
        mock_registry.get_parent_segments.assert_not_called()


# ---------------------------------------------------------------------------
# SearchResult.from_chromadb_result: metadata round-trip
# ---------------------------------------------------------------------------


class TestSearchResultMetadataRoundTrip:
    def test_segment_index_read_from_metadata(self):
        meta = {
            "path": "Part I > Item 1",
            "content_type": "text",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000001",
            "segment_index": 42,
        }
        result = SearchResult.from_chromadb_result(
            document="chunk",
            metadata=meta,
            distance=0.1,
            chunk_id="cid",
        )
        assert result.segment_index == 42

    def test_segment_index_missing_yields_none(self):
        """Legacy rows ingested before this feature have no segment_index."""
        meta = {
            "path": "Part I",
            "content_type": "text",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000001",
        }
        result = SearchResult.from_chromadb_result(
            document="chunk",
            metadata=meta,
            distance=0.1,
        )
        assert result.segment_index is None

    def test_chunk_to_metadata_round_trips(self):
        """to_metadata() output should reconstruct the segment_index."""
        chunk = Chunk(
            content="text",
            path="Part I",
            content_type=ContentType.TEXT,
            filing_id=FilingIdentifier(
                ticker="AAPL",
                form_type="10-K",
                filing_date=date(2024, 1, 1),
                accession_number="0000320193-24-000002",
            ),
            chunk_index=3,
            token_count=2,
            segment_index=11,
        )
        meta = chunk.to_metadata()
        result = SearchResult.from_chromadb_result(
            document=chunk.content, metadata=meta, distance=0.2
        )
        assert result.segment_index == 11


# ---------------------------------------------------------------------------
# Parser → segment_index assignment
# ---------------------------------------------------------------------------


class TestParserSegmentIndex:
    def test_parser_assigns_sequential_indices(self, sample_html, filing_id):
        from sec_semantic_search.pipeline.parse import FilingParser

        segments = FilingParser().parse(sample_html, filing_id)
        # Indices must be 0..N-1 with no gaps and in extraction order.
        assert [s.segment_index for s in segments] == list(range(len(segments)))
