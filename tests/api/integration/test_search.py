"""
Integration tests for the ``POST /api/search/`` endpoint.

The ``SearchEngine`` is fully mocked — these tests exercise the route
handler's input validation, error mapping, and response formatting.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from sec_semantic_search.api.app import app
from sec_semantic_search.api.dependencies import get_search_engine
from sec_semantic_search.core.exceptions import EmbeddingBusyError, SearchError
from sec_semantic_search.core.types import ContentType, SearchResult


def _make_client(search_results=None, search_error=None):
    """Build a TestClient with a mocked SearchEngine."""
    engine = MagicMock()
    if search_error:
        engine.search.side_effect = search_error
    else:
        engine.search.return_value = search_results or []
    app.dependency_overrides[get_search_engine] = lambda: engine
    return TestClient(app, raise_server_exceptions=False), engine


def _make_result(**overrides):
    """Create a minimal SearchResult for testing."""
    defaults = {
        "content": "Sample content",
        "path": "Part I > Item 1",
        "content_type": ContentType.TEXT,
        "ticker": "AAPL",
        "form_type": "10-K",
        "similarity": 0.45,
        "filing_date": "2024-11-01",
        "accession_number": "0000320193-24-000001",
        "chunk_id": "AAPL_10-K_2024-11-01_0",
    }
    defaults.update(overrides)
    return SearchResult(**defaults)


class TestSearchEndpoint:
    """POST /api/search/ — semantic search."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_valid_query_with_results(self):
        results = [_make_result()]
        client, _ = _make_client(search_results=results)
        resp = client.post("/api/search/", json={"query": "revenue"})
        assert resp.status_code == 200
        data = resp.json()
        assert "query" not in data  # §F4: query not echoed in response
        assert data["total_results"] == 1
        assert data["search_time_ms"] >= 0
        assert data["results"][0]["ticker"] == "AAPL"
        assert data["results"][0]["content_type"] == "text"

    def test_parent_context_fields_in_response(self):
        """The parent segment travels with the result and is referenced by key.

        The frontend renders the parent and highlights the matched chunk
        inside it. Verified against a mocked engine so the test stays fast
        and deterministic.
        """
        results = [
            _make_result(
                content="cash and equivalents decreased",
                parent_content="Operating cash flow improved despite the fact that "
                "cash and equivalents decreased by twelve percent year over year.",
                segment_index=4,
            )
        ]
        client, _ = _make_client(search_results=results)
        resp = client.post("/api/search/", json={"query": "cash"})
        assert resp.status_code == 200
        payload = resp.json()["results"][0]
        assert payload["segment_index"] == 4
        assert "parent_content" not in payload
        assert payload["parent_key"] == "0000320193-24-000001:4"
        parent = payload["parent"]
        assert parent["content"].startswith("Operating cash flow")
        assert payload["content"] in parent["content"]
        assert parent["truncated_start"] is False
        assert parent["truncated_end"] is False

    def test_parent_context_fields_default_to_none(self):
        """Legacy results without parent context must still serialise."""
        results = [_make_result()]
        client, _ = _make_client(search_results=results)
        resp = client.post("/api/search/", json={"query": "revenue"})
        payload = resp.json()["results"][0]
        assert payload["segment_index"] is None
        assert payload["parent_key"] is None
        assert payload["parent"] is None

    def test_valid_query_no_results(self):
        client, _ = _make_client(search_results=[])
        resp = client.post("/api/search/", json={"query": "obscure query"})
        data = resp.json()
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_empty_query_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": ""})
        assert resp.status_code == 422  # Pydantic min_length=1

    def test_search_error_empty_returns_400(self):
        error = SearchError("Empty search query")
        client, _ = _make_client(search_error=error)
        resp = client.post("/api/search/", json={"query": "x"})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "validation_error"

    def test_search_error_other_returns_500(self):
        error = SearchError("Database unreachable", details="timeout")
        client, _ = _make_client(search_error=error)
        resp = client.post("/api/search/", json={"query": "x"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "search_error"

    def test_model_busy_returns_503(self):
        """A search that cannot get the embedder in time fails fast and retryably."""
        client, _ = _make_client(search_error=EmbeddingBusyError("Embedding model is busy"))
        resp = client.post("/api/search/", json={"query": "test"})
        assert resp.status_code == 503
        assert resp.headers["retry-after"] == "5"
        detail = resp.json()["detail"]
        assert detail["error"] == "model_busy"
        assert detail["details"] is None

    def test_embed_wait_is_bounded(self):
        """The route never lets a search wait on the embedder without limit."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test"})
        _, kwargs = engine.search.call_args
        assert kwargs["embed_timeout"] is not None
        assert 0 < kwargs["embed_timeout"] <= 60

    def test_ticker_filter_passed(self):
        """Single-string ticker is coerced to a one-element list."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "ticker": "aapl"})
        _, kwargs = engine.search.call_args
        assert kwargs["ticker"] == ["AAPL"]

    def test_ticker_list_filter_passed(self):
        """Multiple tickers are passed as a list."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "ticker": ["aapl", "msft"]})
        _, kwargs = engine.search.call_args
        assert kwargs["ticker"] == ["AAPL", "MSFT"]

    def test_form_type_filter_passed(self):
        """Single-string form_type is coerced to a one-element list."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "form_type": "10-q"})
        _, kwargs = engine.search.call_args
        assert kwargs["form_type"] == ["10-Q"]

    def test_form_type_list_filter_passed(self):
        """Multiple form types are passed as a list."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "form_type": ["10-K", "10-Q"]})
        _, kwargs = engine.search.call_args
        assert kwargs["form_type"] == ["10-K", "10-Q"]

    def test_valid_8k_form_type_accepted(self):
        client, engine = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "form_type": "8-K"})
        assert resp.status_code == 200
        _, kwargs = engine.search.call_args
        assert kwargs["form_type"] == ["8-K"]

    def test_invalid_form_type_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "form_type": "20-F"})
        assert resp.status_code == 422

    def test_invalid_form_type_in_list_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "form_type": ["10-K", "20-F"]})
        assert resp.status_code == 422

    def test_top_k_out_of_range_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "top_k": 101})
        assert resp.status_code == 422

    def test_accession_number_passed(self):
        """Single-string accession is coerced to a one-element list."""
        client, engine = _make_client()
        client.post(
            "/api/search/", json={"query": "test", "accession_number": "0000320193-24-000001"}
        )
        _, kwargs = engine.search.call_args
        assert kwargs["accession_number"] == ["0000320193-24-000001"]

    def test_accession_number_list_passed(self):
        """Multiple accession numbers are passed as a list."""
        client, engine = _make_client()
        client.post(
            "/api/search/",
            json={
                "query": "test",
                "accession_number": ["0000320193-24-000001", "0000320193-24-000002"],
            },
        )
        _, kwargs = engine.search.call_args
        assert kwargs["accession_number"] == ["0000320193-24-000001", "0000320193-24-000002"]

    def test_invalid_accession_in_list_returns_422(self):
        client, _ = _make_client()
        resp = client.post(
            "/api/search/",
            json={"query": "test", "accession_number": ["0000320193-24-000001", "invalid"]},
        )
        assert resp.status_code == 422

    def test_start_date_passed(self):
        """start_date is forwarded to the search engine."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "start_date": "2023-01-01"})
        _, kwargs = engine.search.call_args
        assert kwargs["start_date"] == "2023-01-01"

    def test_end_date_passed(self):
        """end_date is forwarded to the search engine."""
        client, engine = _make_client()
        client.post("/api/search/", json={"query": "test", "end_date": "2023-12-31"})
        _, kwargs = engine.search.call_args
        assert kwargs["end_date"] == "2023-12-31"

    def test_both_dates_passed(self):
        """start_date and end_date are both forwarded."""
        client, engine = _make_client()
        client.post(
            "/api/search/",
            json={"query": "test", "start_date": "2023-01-01", "end_date": "2023-12-31"},
        )
        _, kwargs = engine.search.call_args
        assert kwargs["start_date"] == "2023-01-01"
        assert kwargs["end_date"] == "2023-12-31"

    def test_invalid_start_date_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "start_date": "bad"})
        assert resp.status_code == 422

    def test_invalid_end_date_returns_422(self):
        client, _ = _make_client()
        resp = client.post("/api/search/", json={"query": "test", "end_date": "2023-13-01"})
        assert resp.status_code == 422
