#!/usr/bin/env python
"""
Measure a local store: disk and memory per filing, HNSW recall.

Reports, for the configured (or given) ChromaDB directory and SQLite
registry:

    - filings, chunks and chunks per filing
    - bytes on disk: the HNSW index, ChromaDB's SQLite (documents, metadata
      and the full-text index ChromaDB always keeps), the registry (filings,
      parent segments, task history), per filing and per chunk
    - resident memory added by loading the index for a first query (Linux)
    - recall@k of the HNSW index against exact cosine search, for one or
      more ``ef_search`` values, using stored chunk vectors as queries

Recall only means something on real embeddings. Uniformly random vectors
have no neighbourhoods (every point is about as far from a query as its
true nearest neighbours are), so HNSW scores far lower on them than on
text embeddings, which cluster: on a synthetic store of 402,489 random
768-d vectors recall@10 was 0.05 at ``ef_search=100``, and on 100,000
clustered ones 1.0.

``ef_search`` can be changed on an existing collection, and takes effect
the next time a process loads it (restart the API); ``max_neighbors`` (M)
and ``ef_construction`` are fixed when the collection is created, so
changing them means a wipe and re-ingest. The script restores the
collection's ``ef_search`` when it finishes.

Run it against a stopped API or a copy of ``data/``: two processes must
not open the same ChromaDB directory (opening a store while another
process was writing to it lost 227 entries from its HNSW index in
testing). It prints aggregates
only — no tickers, accession numbers or text.

Usage:
    python scripts/measure_store.py
    python scripts/measure_store.py --chroma-path data/chroma_db \\
        --metadata-db data/metadata.sqlite --sample 300 --ef-search 100 200 400
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

_HNSW_FILES = ("data_level0.bin", "link_lists.bin", "header.bin", "length.bin")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def disk_usage(chroma_path: Path, metadata_db: Path) -> dict[str, int]:
    """Bytes on disk, split by what they hold."""
    hnsw = sum(
        p.stat().st_size for p in chroma_path.rglob("*") if p.is_file() and p.name in _HNSW_FILES
    )
    registry = sum(
        _size(p)
        for p in (metadata_db, Path(f"{metadata_db}-wal"), Path(f"{metadata_db}-shm"))
        if p.exists()
    )
    chroma_total = _size(chroma_path)
    return {
        "chroma_total": chroma_total,
        "chroma_hnsw": hnsw,
        "chroma_sqlite_and_other": chroma_total - hnsw,
        "registry": registry,
    }


def registry_counts(metadata_db: Path, key: str | None = None) -> dict[str, int]:
    """Filing and segment counts, read without the application stack.

    ``key`` is the SQLCipher key for an encrypted registry. As in
    ``MetadataRegistry``, it is used only when pysqlcipher3 is installed.
    """
    sqlcipher = None
    if key:
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore[import-untyped]
        except ImportError:
            pass
    if sqlcipher is not None:
        conn = sqlcipher.connect(str(metadata_db))
        conn.execute(f"PRAGMA key = \"x'{key.encode().hex()}'\"")
    else:
        conn = sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True)
    try:
        filings = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    finally:
        conn.close()
    return {"filings": filings, "segments": segments}


# ChromaDB 1.5 (local mode) reads a collection's ``ef_search`` when the
# index is loaded; changing it later in the same process has no effect.
# So every step that opens the store runs in a fresh child process, one at
# a time (never two processes on one ChromaDB directory), and the recall
# for each ``ef_search`` is measured in its own process.


def _in_child(func, *args):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(func, args)


def _open(chroma_path: str):
    import chromadb

    from sec_semantic_search.config.constants import COLLECTION_NAME

    return chromadb.PersistentClient(path=chroma_path).get_collection(COLLECTION_NAME)


def _snapshot(chroma_path: str, vectors_file: str | None, batch: int = 5000) -> dict:
    """Count and HNSW settings; optionally dump every vector, L2-normalized."""
    collection = _open(chroma_path)
    info = {
        "chunks": collection.count(),
        "hnsw": {
            name: collection.configuration_json["hnsw"][name]
            for name in ("max_neighbors", "ef_construction", "ef_search")
        },
    }
    if vectors_file is None:
        return info
    ids: list[str] = []
    parts: list[np.ndarray] = []
    offset = 0
    while True:
        page = collection.get(include=["embeddings"], limit=batch, offset=offset)
        if not page["ids"]:
            break
        ids.extend(page["ids"])
        parts.append(np.asarray(page["embeddings"], dtype=np.float32))
        offset += len(page["ids"])
    matrix = np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.savez(vectors_file, ids=np.array(ids), matrix=matrix / np.where(norms == 0, 1, norms))
    return info


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _memory(chroma_path: str) -> int | None:
    """Resident bytes added by opening the collection and running one query."""
    import chromadb  # noqa: F401  (imported before the baseline reading)

    before = _rss_bytes()
    collection = _open(chroma_path)
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
    if len(sample) == 0:
        return None
    collection.query(query_embeddings=[list(sample[0])], n_results=1, include=[])
    after = _rss_bytes()
    return None if before is None or after is None else after - before


def _query(
    chroma_path: str, ef_search: int, queries_file: str, k: int, restore: int | None
) -> list[list[str]]:
    collection = _open(chroma_path)
    collection.modify(configuration={"hnsw": {"ef_search": ef_search}})
    try:
        data = np.load(queries_file)
        return [
            collection.query(query_embeddings=[vector.tolist()], n_results=k + 1, include=[])[
                "ids"
            ][0]
            for vector in data["vectors"]
        ]
    finally:
        if restore is not None:
            collection.modify(configuration={"hnsw": {"ef_search": restore}})


def _recall(
    chroma_path: Path,
    vectors_file: str,
    original_ef_search: int,
    *,
    sample: int,
    k: int,
    ef_search_values: list[int],
    seed: int = 0,
) -> dict[int, float]:
    """Mean recall@k of HNSW against exact cosine search, per ``ef_search``.

    Each sampled stored vector is used as a query; the vector itself is
    excluded from both result lists. The last child restores the
    collection's original ``ef_search``.
    """
    data = np.load(vectors_file)
    ids, matrix = [str(i) for i in data["ids"]], data["matrix"]
    if len(ids) <= k:
        return {}
    queries = random.Random(seed).sample(range(len(ids)), min(sample, len(ids)))

    exact: list[set[str]] = []
    for q in queries:
        scores = matrix @ matrix[q]
        scores[q] = -np.inf
        exact.append({ids[i] for i in np.argpartition(-scores, k)[:k]})

    queries_file = str(Path(vectors_file).with_name("queries.npz"))
    np.savez(queries_file, vectors=matrix[queries])
    del matrix, data

    results: dict[int, float] = {}
    for n, ef in enumerate(ef_search_values):
        last = n == len(ef_search_values) - 1
        found = _in_child(
            _query, str(chroma_path), ef, queries_file, k, original_ef_search if last else None
        )
        hits = sum(
            len(truth & {i for i in got if i != ids[q]})
            for q, truth, got in zip(queries, exact, found, strict=True)
        )
        results[ef] = round(hits / (k * len(queries)), 4)
    return results


def measure(
    chroma_path: Path,
    metadata_db: Path,
    *,
    sample: int = 200,
    k: int = 10,
    ef_search_values: list[int] | None = None,
    key: str | None = None,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        vectors_file = os.path.join(tmp, "vectors.npz") if sample > 0 else None
        info = _in_child(_snapshot, str(chroma_path), vectors_file)
        chunks = info["chunks"]
        index_rss = _in_child(_memory, str(chroma_path))
        counts = registry_counts(metadata_db, key)
        disk = disk_usage(chroma_path, metadata_db)
        filings = counts["filings"] or 1
        report = {
            "filings": counts["filings"],
            "segments": counts["segments"],
            "chunks": chunks,
            "chunks_per_filing": round(chunks / filings, 1),
            "hnsw_config": info["hnsw"],
            "bytes": disk,
            "bytes_per_filing": {name: round(v / filings) for name, v in disk.items()},
            "bytes_per_chunk": {name: round(v / max(chunks, 1)) for name, v in disk.items()},
            "index_rss_bytes": index_rss,
            "index_rss_bytes_per_filing": None if index_rss is None else round(index_rss / filings),
        }
        if vectors_file is not None:
            report[f"recall_at_{k}"] = _recall(
                chroma_path,
                vectors_file,
                info["hnsw"]["ef_search"],
                sample=sample,
                k=k,
                ef_search_values=ef_search_values or [100],
            )
    return report


def main() -> None:
    from sec_semantic_search.config import get_settings

    database = get_settings().database
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chroma-path", type=Path, default=Path(database.chroma_path))
    parser.add_argument("--metadata-db", type=Path, default=Path(database.metadata_db_path))
    parser.add_argument("--sample", type=int, default=200, help="queries for recall; 0 skips it")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[100, 200])
    args = parser.parse_args()
    report = measure(
        args.chroma_path,
        args.metadata_db,
        sample=args.sample,
        k=args.k,
        ef_search_values=args.ef_search,
        key=database.encryption_key,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
