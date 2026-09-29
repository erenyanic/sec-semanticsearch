"""
Tests for ``scripts/measure_store.py``.

Builds a small real store (ChromaDB + SQLite registry) and checks the
counts, the disk split and that the recall measurement both works and
leaves the collection's ``ef_search`` as it found it.
"""

import importlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from sec_semantic_search.core.types import ContentType, FilingIdentifier, Segment

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_store.py"


@pytest.fixture(scope="module")
def measure_store():
    # Imported by name: the script runs its store access in spawned child
    # processes, which import it again from sys.path.
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        yield importlib.import_module("measure_store")
    finally:
        sys.path.remove(str(SCRIPT.parent))
        sys.modules.pop("measure_store", None)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import chromadb

    from sec_semantic_search.config.constants import COLLECTION_NAME
    from sec_semantic_search.database.metadata import MetadataRegistry

    root = tmp_path_factory.mktemp("store")
    registry = MetadataRegistry(db_path=str(root / "metadata.sqlite"), encryption_key="")
    collection = chromadb.PersistentClient(path=str(root / "chroma")).get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    rng = np.random.default_rng(0)
    for f in range(3):
        fid = FilingIdentifier(
            "AAPL", "10-K", date(2020, 1, 1) + timedelta(days=f), f"0000320193-2{f}-000001"
        )
        segments = [
            Segment("Item 1", ContentType.TEXT, f"Segment {i} text.", fid, i) for i in range(40)
        ]
        registry.register_filing(fid, 40, segments=segments)
        vectors = rng.standard_normal((40, 32)).astype(np.float32)
        collection.add(
            ids=[f"{fid.accession_number}_{i}" for i in range(40)],
            embeddings=vectors,
            documents=[s.content for s in segments],
        )
    registry.close()
    return root


@pytest.fixture(scope="module")
def report(measure_store, store):
    # One run with recall: each measure() spawns a few child processes.
    return measure_store.measure(
        store / "chroma", store / "metadata.sqlite", sample=20, k=5, ef_search_values=[50, 200]
    )


class TestMeasureStore:
    def test_counts_and_disk_split(self, report):
        assert (report["filings"], report["segments"], report["chunks"]) == (3, 120, 120)
        assert report["chunks_per_filing"] == 40.0
        disk = report["bytes"]
        assert disk["chroma_total"] == disk["chroma_hnsw"] + disk["chroma_sqlite_and_other"]
        assert disk["chroma_total"] > 0 and disk["registry"] > 0
        assert report["hnsw_config"] == {
            "max_neighbors": 16,
            "ef_construction": 100,
            "ef_search": 100,
        }

    def test_index_memory(self, report):
        if sys.platform != "linux":
            pytest.skip("reads /proc/self/status")
        assert report["index_rss_bytes"] > 0
        assert report["index_rss_bytes_per_filing"] == round(report["index_rss_bytes"] / 3)

    def test_recall_per_ef_search(self, report):
        # 120 vectors: HNSW is effectively exact at this size.
        assert report["recall_at_5"] == {50: 1.0, 200: 1.0}

    def test_ef_search_restored(self, measure_store, store, report):
        after = measure_store.measure(store / "chroma", store / "metadata.sqlite", sample=0)
        assert after["hnsw_config"]["ef_search"] == 100
        assert "recall_at_10" not in after

    def test_key_without_sqlcipher_reads_plain_sqlite(self, measure_store, store, monkeypatch):
        """Like MetadataRegistry: a key is ignored when pysqlcipher3 is absent."""
        monkeypatch.setitem(sys.modules, "pysqlcipher3", None)
        counts = measure_store.registry_counts(store / "metadata.sqlite", key="k")
        assert counts == {"filings": 3, "segments": 120}

    def test_report_holds_no_identifiers(self, report):
        text = repr(report)
        assert "AAPL" not in text and "0000320193" not in text and "Segment" not in text
