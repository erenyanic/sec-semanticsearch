"""
Integration tests for the ingestion pipeline and database layer.

These tests verify that modules compose correctly:
    - Parser output feeds into the chunker
    - Chunker output feeds into the database
    - MetadataRegistry CRUD operations work end-to-end
    - ChromaDB stores and retrieves chunks with correct metadata
    - Dual-store (ChromaDB + SQLite) operations stay consistent

The embedding step is skipped — loading the 300M-parameter model
takes ~10 seconds and requires CUDA. We use random numpy arrays of
the correct dimension (768) instead. The model itself is a third-party
dependency; what we verify here is that *our wiring* is correct.
"""

import numpy as np
import pytest

from sec_semantic_search.config.constants import EMBEDDING_DIMENSION
from sec_semantic_search.core.exceptions import (
    DatabaseError,
    FilingLimitExceededError,
)
from sec_semantic_search.database.client import ChromaDBClient
from sec_semantic_search.database.metadata import MetadataRegistry
from sec_semantic_search.pipeline.chunk import TextChunker
from sec_semantic_search.pipeline.orchestrator import ProcessedFiling
from sec_semantic_search.pipeline.parse import FilingParser

# -----------------------------------------------------------------------
# Parse → Chunk pipeline
# -----------------------------------------------------------------------


class TestParseChunkPipeline:
    """Verify that parser output is valid chunker input."""

    def test_parse_then_chunk(self, sample_html, sample_filing_id):
        """HTML → Parser → Chunker should produce chunks with metadata."""
        parser = FilingParser()
        chunker = TextChunker(token_limit=500, tolerance=50)

        segments = parser.parse(sample_html, sample_filing_id)
        chunks = chunker.chunk_segments(segments)

        assert len(chunks) > 0
        # Every chunk should have a valid chunk_id
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.filing_id is sample_filing_id

    def test_sequential_indices_across_segments(self, sample_html, sample_filing_id):
        """Chunk indices should be sequential starting from 0."""
        parser = FilingParser()
        chunker = TextChunker(token_limit=500, tolerance=50)

        segments = parser.parse(sample_html, sample_filing_id)
        chunks = chunker.chunk_segments(segments)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# -----------------------------------------------------------------------
# MetadataRegistry (SQLite)
# -----------------------------------------------------------------------


class TestMetadataRegistry:
    """CRUD operations on the SQLite metadata registry."""

    def test_register_and_retrieve(self, tmp_db_path, sample_filing_id):
        """Register a filing, then retrieve it by accession number."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=10)

        record = registry.get_filing(sample_filing_id.accession_number)
        assert record is not None
        assert record.ticker == "AAPL"
        assert record.form_type == "10-K"
        assert record.chunk_count == 10

    def test_count(self, tmp_db_path, sample_filing_id):
        """count() should reflect the number of registered filings."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        assert registry.count() == 0

        registry.register_filing(sample_filing_id, chunk_count=5)
        assert registry.count() == 1

    def test_count_with_filters(self, tmp_db_path, sample_filing_id):
        """count(ticker=...) should filter correctly."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=5)

        assert registry.count(ticker="AAPL") == 1
        assert registry.count(ticker="MSFT") == 0
        assert registry.count(form_type="10-K") == 1
        assert registry.count(form_type="10-Q") == 0

    def test_list_filings(self, tmp_db_path, sample_filing_id):
        """list_filings() should return FilingRecord objects."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=10)

        filings = registry.list_filings()
        assert len(filings) == 1
        assert filings[0].accession_number == sample_filing_id.accession_number

    def test_list_filings_with_filter(self, tmp_db_path, sample_filing_id):
        """Filtering by non-matching ticker should return empty list."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=10)

        assert registry.list_filings(ticker="MSFT") == []

    def test_existing_accessions(self, tmp_db_path, sample_filing_id):
        """get_existing_accessions() should detect already-registered filings."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        assert registry.get_existing_accessions([sample_filing_id.accession_number]) == set()

        registry.register_filing(sample_filing_id, chunk_count=5)
        assert registry.get_existing_accessions([sample_filing_id.accession_number]) == {
            sample_filing_id.accession_number
        }

    def test_remove_filing(self, tmp_db_path, sample_filing_id):
        """remove_filing() should delete and return True; second call returns False."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=5)

        assert registry.remove_filing(sample_filing_id.accession_number) is True
        assert registry.count() == 0
        assert registry.remove_filing(sample_filing_id.accession_number) is False

    def test_duplicate_registration_raises(self, tmp_db_path, sample_filing_id):
        """Registering the same filing twice should raise DatabaseError."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.register_filing(sample_filing_id, chunk_count=5)

        with pytest.raises(DatabaseError, match="already exists"):
            registry.register_filing(sample_filing_id, chunk_count=5)

    def test_filing_limit(self, tmp_db_path, sample_filing_id):
        """
        check_filing_limit() should raise when limit is reached.

        We create a registry and manually set _max_filings=1, then
        register one filing to trigger the limit.
        """
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry._max_filings = 1

        registry.register_filing(sample_filing_id, chunk_count=5)

        with pytest.raises(FilingLimitExceededError):
            registry.check_filing_limit()


# -----------------------------------------------------------------------
# ChromaDBClient
# -----------------------------------------------------------------------


def _make_processed_filing(chunks, filing_id):
    """Helper: create a ProcessedFiling with random embeddings."""
    embeddings = np.random.default_rng(42).random(
        (len(chunks), EMBEDDING_DIMENSION), dtype=np.float32
    )
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=None,  # Not needed for storage
        segments=[],  # Storage tests don't exercise the parent-context path
    )


class TestChromaDBClient:
    """ChromaDB store, query, and delete operations."""

    def test_store_and_count(self, tmp_chroma_path, sample_chunks, sample_filing_id):
        """Storing a filing should increase the collection count."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        assert client.collection_count() == 0

        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        client.store_filing(pf)
        assert client.collection_count() == len(sample_chunks)

    def test_query_returns_results(self, tmp_chroma_path, sample_chunks, sample_filing_id):
        """After storing chunks, querying should return SearchResult objects."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        client.store_filing(pf)

        # Query with a random embedding (results won't be semantically
        # meaningful, but we verify the query pipeline works)
        query_emb = (
            np.random.default_rng(99).random((1, EMBEDDING_DIMENSION), dtype=np.float32).tolist()
        )
        results = client.query(query_emb, n_results=2)

        assert len(results) > 0
        assert results[0].ticker == "AAPL"
        assert results[0].form_type == "10-K"

    def test_query_with_ticker_filter(self, tmp_chroma_path, sample_chunks, sample_filing_id):
        """Filtering by a non-matching ticker should return no results."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        client.store_filing(pf)

        query_emb = (
            np.random.default_rng(99).random((1, EMBEDDING_DIMENSION), dtype=np.float32).tolist()
        )
        results = client.query(query_emb, n_results=5, ticker="MSFT")
        assert len(results) == 0

    def test_delete_filing(self, tmp_chroma_path, sample_chunks, sample_filing_id):
        """Deleting a filing should remove all its chunks."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        client.store_filing(pf)

        client.delete_filing(sample_filing_id.accession_number)
        assert client.collection_count() == 0

    def test_delete_nonexistent_filing(self, tmp_chroma_path):
        """Deleting a non-existent filing should be a no-op."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        client.delete_filing("NONEXISTENT-ACC")  # Should not raise


# -----------------------------------------------------------------------
# filing_date_int migration (BF-012)
# -----------------------------------------------------------------------


class TestFilingDateIntMigration:
    """Auto-migration should backfill filing_date_int for legacy chunks."""

    def test_migration_backfills_missing_field(
        self, tmp_chroma_path, sample_chunks, sample_filing_id
    ):
        """Chunks stored without filing_date_int get it added on next init."""
        import chromadb

        from sec_semantic_search.config.constants import COLLECTION_NAME

        # Step 1: Store chunks directly via raw ChromaDB (no filing_date_int)
        raw_client = chromadb.PersistentClient(path=tmp_chroma_path)
        collection = raw_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = np.random.default_rng(42).random(
            (len(sample_chunks), EMBEDDING_DIMENSION), dtype=np.float32
        )
        ids = [c.chunk_id for c in sample_chunks]
        documents = [c.content for c in sample_chunks]
        # Deliberately omit filing_date_int (simulates pre-BF-012 data)
        metadatas = [
            {
                "path": c.path,
                "content_type": c.content_type.value,
                "ticker": c.filing_id.ticker,
                "form_type": c.filing_id.form_type,
                "filing_date": c.filing_id.date_str,
                "accession_number": c.filing_id.accession_number,
            }
            for c in sample_chunks
        ]
        collection.add(
            ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas
        )

        # Verify filing_date_int is NOT present before migration
        pre = collection.get(include=["metadatas"])
        assert all("filing_date_int" not in m for m in pre["metadatas"])

        # Step 2: Create a ChromaDBClient — migration should run automatically
        client = ChromaDBClient(chroma_path=tmp_chroma_path)

        # Step 3: Verify filing_date_int is now present
        post = client._collection.get(include=["metadatas"])
        for meta in post["metadatas"]:
            assert "filing_date_int" in meta
            assert meta["filing_date_int"] == int(meta["filing_date"].replace("-", ""))

    def test_migration_skips_already_migrated(
        self, tmp_chroma_path, sample_chunks, sample_filing_id
    ):
        """Chunks that already have filing_date_int are left unchanged."""
        # Store normally (includes filing_date_int)
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        client.store_filing(pf)

        # Reinitialise — migration should be a no-op
        client2 = ChromaDBClient(chroma_path=tmp_chroma_path)
        post = client2._collection.get(include=["metadatas"])
        for meta in post["metadatas"]:
            assert meta["filing_date_int"] == 20241101

    def test_migration_empty_collection(self, tmp_chroma_path):
        """Migration on an empty collection should be a no-op (no error)."""
        client = ChromaDBClient(chroma_path=tmp_chroma_path)
        assert client.collection_count() == 0

    def test_date_filter_works_after_migration(
        self, tmp_chroma_path, sample_chunks, sample_filing_id
    ):
        """After migration, date-range queries should match backfilled data."""
        import chromadb

        from sec_semantic_search.config.constants import COLLECTION_NAME

        # Store without filing_date_int (pre-BF-012)
        raw_client = chromadb.PersistentClient(path=tmp_chroma_path)
        collection = raw_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = np.random.default_rng(42).random(
            (len(sample_chunks), EMBEDDING_DIMENSION), dtype=np.float32
        )
        ids = [c.chunk_id for c in sample_chunks]
        documents = [c.content for c in sample_chunks]
        metadatas = [
            {
                "path": c.path,
                "content_type": c.content_type.value,
                "ticker": c.filing_id.ticker,
                "form_type": c.filing_id.form_type,
                "filing_date": c.filing_id.date_str,
                "accession_number": c.filing_id.accession_number,
            }
            for c in sample_chunks
        ]
        collection.add(
            ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas
        )

        # Init triggers migration
        client = ChromaDBClient(chroma_path=tmp_chroma_path)

        # Query with date range covering 2024-11-01
        query_emb = (
            np.random.default_rng(99).random((1, EMBEDDING_DIMENSION), dtype=np.float32).tolist()
        )
        results = client.query(
            query_emb,
            n_results=5,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert len(results) > 0

        # Query with date range NOT covering 2024-11-01
        results_empty = client.query(
            query_emb,
            n_results=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        assert len(results_empty) == 0


# -----------------------------------------------------------------------
# Dual-store consistency
# -----------------------------------------------------------------------


class TestDualStoreConsistency:
    """ChromaDB + SQLite should stay in sync during store/delete."""

    def test_store_both_delete_both(
        self,
        tmp_chroma_path,
        tmp_db_path,
        sample_chunks,
        sample_filing_id,
    ):
        """Full lifecycle: store in both → verify → delete from both → verify empty."""
        chroma = ChromaDBClient(chroma_path=tmp_chroma_path)
        registry = MetadataRegistry(db_path=tmp_db_path)

        # Store (ChromaDB first, then SQLite — matches production order)
        pf = _make_processed_filing(sample_chunks, sample_filing_id)
        chroma.store_filing(pf)
        registry.register_filing(sample_filing_id, chunk_count=len(sample_chunks))

        # Verify both populated
        assert chroma.collection_count() == len(sample_chunks)
        assert registry.count() == 1

        # Delete (ChromaDB first, then SQLite — matches production order)
        chroma.delete_filing(sample_filing_id.accession_number)
        registry.remove_filing(sample_filing_id.accession_number)

        # Verify both empty
        assert chroma.collection_count() == 0
        assert registry.count() == 0
