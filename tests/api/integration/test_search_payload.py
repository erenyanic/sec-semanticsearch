"""
Tests for the search response payload (F-05).

Covers:
    - gzip compression above 1 KB, with security and cache headers intact
    - Each parent segment sent once, with the first result citing it
    - Parents identical to their chunk are not sent at all
    - Long parents capped as an excerpt that keeps the matched chunk
    - The route returns the model, so FastAPI serializes it without stdlib json
"""

import gzip
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sec_semantic_search.api.app import app
from sec_semantic_search.api.dependencies import get_search_engine
from sec_semantic_search.api.routes.search import _PARENT_MAX_CHARS, _parent_excerpt
from sec_semantic_search.core.types import ContentType, SearchResult

_ACCESSION = "0000320193-24-000001"


def _result(content: str, parent: str | None = None, segment_index: int | None = None, **kw):
    fields = {
        "content": content,
        "path": "Part II > Item 7",
        "content_type": ContentType.TEXT,
        "ticker": "AAPL",
        "form_type": "10-K",
        "similarity": 0.5,
        "filing_date": "2024-11-01",
        "accession_number": _ACCESSION,
        "chunk_id": f"AAPL_10-K_2024-11-01_{segment_index}",
        "segment_index": segment_index,
        "parent_content": parent,
    }
    fields.update(kw)
    return SearchResult(**fields)


@pytest.fixture
def client_for():
    def _make(results):
        engine = MagicMock()
        engine.search.return_value = results
        app.dependency_overrides[get_search_engine] = lambda: engine
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------
# Compression
# -----------------------------------------------------------------------


class TestCompression:
    """Large API responses are gzip-encoded when the client accepts it."""

    def _large_results(self):
        paragraph = "Revenue increased due to higher services net sales. " * 60
        return [_result(paragraph, segment_index=i) for i in range(5)]

    def test_search_response_is_gzipped(self, client_for):
        client = client_for(self._large_results())
        resp = client.post(
            "/api/search/", json={"query": "revenue"}, headers={"Accept-Encoding": "gzip"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in resp.headers["vary"].lower()
        assert int(resp.headers["content-length"]) < len(resp.content) / 3
        assert resp.json()["total_results"] == 5

    def test_gzipped_response_keeps_security_and_cache_headers(self, client_for):
        client = client_for(self._large_results())
        resp = client.post(
            "/api/search/", json={"query": "revenue"}, headers={"Accept-Encoding": "gzip"}
        )
        assert resp.headers["content-encoding"] == "gzip"
        assert resp.headers["cache-control"] == "no-store"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]

    def test_compressed_body_never_contains_the_query(self, client_for):
        client = client_for(self._large_results())
        query = "confidential acquisition target"
        with client.stream(
            "POST", "/api/search/", json={"query": query}, headers={"Accept-Encoding": "gzip"}
        ) as resp:
            raw = b"".join(resp.iter_raw())
        body = gzip.decompress(raw).decode()
        assert query not in body
        assert "query" not in json.loads(body)

    def test_small_response_not_compressed(self):
        resp = TestClient(app).get("/api/health", headers={"Accept-Encoding": "gzip"})
        assert "content-encoding" not in resp.headers

    def test_not_compressed_without_accept_encoding(self, client_for):
        client = client_for(self._large_results())
        resp = client.post(
            "/api/search/", json={"query": "revenue"}, headers={"Accept-Encoding": "identity"}
        )
        assert "content-encoding" not in resp.headers
        assert resp.headers["cache-control"] == "no-store"


# -----------------------------------------------------------------------
# Parent deduplication
# -----------------------------------------------------------------------


class TestParentDeduplication:
    """Each parent segment appears once, however many results cite it."""

    def test_shared_parent_sent_once_with_first_result(self, client_for):
        parent = "First sentence. Second sentence. Third sentence."
        results = [
            _result("Second sentence.", parent, segment_index=3),
            _result("First sentence.", parent, segment_index=3),
            _result("Third sentence.", parent, segment_index=3),
        ]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()

        assert [r["parent_key"] for r in data["results"]] == [f"{_ACCESSION}:3"] * 3
        assert data["results"][0]["parent"]["content"] == parent
        assert data["results"][1]["parent"] is None
        assert data["results"][2]["parent"] is None

    def test_distinct_parents_each_sent(self, client_for):
        results = [
            _result("Alpha.", "Alpha. More alpha.", segment_index=1),
            _result("Beta.", "Beta. More beta.", segment_index=2),
            _result("More alpha.", "Alpha. More alpha.", segment_index=1),
        ]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()
        assert [r["parent_key"] for r in data["results"]] == [
            f"{_ACCESSION}:1",
            f"{_ACCESSION}:2",
            f"{_ACCESSION}:1",
        ]
        sent = [r["parent"]["content"] if r["parent"] else None for r in data["results"]]
        assert sent == ["Alpha. More alpha.", "Beta. More beta.", None]

    def test_same_segment_index_in_other_filing_is_a_different_parent(self, client_for):
        results = [
            _result("Alpha.", "Alpha. More alpha.", segment_index=1),
            _result(
                "Gamma.",
                "Gamma. More gamma.",
                segment_index=1,
                accession_number="1111111111-24-000001",
            ),
        ]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()
        assert data["results"][1]["parent_key"] == "1111111111-24-000001:1"
        assert data["results"][1]["parent"]["content"] == "Gamma. More gamma."

    def test_parent_identical_to_chunk_not_sent(self, client_for):
        """A single-chunk segment would otherwise travel twice."""
        results = [_result("Whole segment.", "Whole segment.", segment_index=0)]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()
        assert data["results"][0]["parent_key"] is None
        assert data["results"][0]["parent"] is None
        assert data["results"][0]["content"] == "Whole segment."


# -----------------------------------------------------------------------
# Parent cap
# -----------------------------------------------------------------------


def _long_parent(chunk: str, position: int, length: int) -> str:
    filler = "x" * length
    return filler[:position] + chunk + filler[position : length - len(chunk)]


class TestParentExcerpt:
    """Parents longer than the cap become an excerpt around the chunks."""

    def test_short_parent_returned_whole(self):
        excerpt = _parent_excerpt("Short parent.", ["Short"])
        assert excerpt.content == "Short parent."
        assert not excerpt.truncated_start and not excerpt.truncated_end

    def test_excerpt_keeps_a_chunk_deep_inside_the_parent(self):
        chunk = "MATCHED CHUNK TEXT"
        parent = _long_parent(chunk, position=30_000, length=50_000)
        excerpt = _parent_excerpt(parent, [chunk])
        assert len(excerpt.content) == _PARENT_MAX_CHARS
        assert chunk in excerpt.content
        assert excerpt.truncated_start and excerpt.truncated_end

    def test_excerpt_at_the_end_is_not_marked_truncated_at_the_end(self):
        chunk = "TAIL CHUNK"
        parent = _long_parent(chunk, position=49_990, length=50_000)
        excerpt = _parent_excerpt(parent, [chunk])
        assert excerpt.content.endswith(chunk)
        assert excerpt.truncated_start and not excerpt.truncated_end

    def test_chunk_not_found_falls_back_to_head(self):
        parent = "y" * 40_000
        excerpt = _parent_excerpt(parent, ["absent"])
        assert excerpt.content == parent[:_PARENT_MAX_CHARS]
        assert not excerpt.truncated_start and excerpt.truncated_end

    def test_window_covers_all_chunks_when_they_fit(self):
        first, second = "FIRST CHUNK", "SECOND CHUNK"
        parent = "z" * 60_000
        parent = parent[:20_000] + first + parent[20_000:28_000] + second + parent[28_000:]
        excerpt = _parent_excerpt(parent, [first, second])
        assert first in excerpt.content and second in excerpt.content

    def test_window_prefers_best_chunk_when_chunks_are_far_apart(self):
        best, other = "BEST CHUNK", "OTHER CHUNK"
        parent = "z" * 80_000
        parent = parent[:5_000] + other + parent[5_000:70_000] + best + parent[70_000:]
        excerpt = _parent_excerpt(parent, [best, other])
        assert best in excerpt.content
        assert other not in excerpt.content

    def test_excerpt_covers_every_result_citing_the_parent(self, client_for):
        """The window is chosen from all citing chunks, not just the first."""
        first, second = "FIRST CHUNK", "SECOND CHUNK"
        parent = "z" * 60_000
        parent = parent[:20_000] + first + parent[20_000:28_000] + second + parent[28_000:]
        results = [
            _result(first, parent, segment_index=2),
            _result(second, parent, segment_index=2),
        ]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()
        sent = data["results"][0]["parent"]["content"]
        assert first in sent and second in sent

    def test_route_caps_long_parent(self, client_for):
        chunk = "Net sales by category."
        parent = _long_parent(chunk, position=25_000, length=60_000)
        results = [_result(chunk, parent, segment_index=9)]
        data = client_for(results).post("/api/search/", json={"query": "q"}).json()
        sent = data["results"][0]["parent"]
        assert len(sent["content"]) <= _PARENT_MAX_CHARS
        assert chunk in sent["content"]
        assert sent["truncated_start"] and sent["truncated_end"]


# -----------------------------------------------------------------------
# Serialization path
# -----------------------------------------------------------------------


class TestSerialization:
    """The route returns the model so FastAPI serializes it in pydantic-core."""

    def test_search_response_skips_stdlib_json(self, client_for):
        client = client_for([_result("Alpha.", "Alpha. More alpha.", segment_index=1)])
        body = json.dumps({"query": "q"}).encode()
        with patch("json.dumps", side_effect=AssertionError("stdlib json.dumps used")):
            resp = client.post(
                "/api/search/", content=body, headers={"Content-Type": "application/json"}
            )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["parent"]["content"] == "Alpha. More alpha."
