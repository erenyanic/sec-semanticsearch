"""
Integration tests for the filing management endpoints.

Covers listing, retrieval, single delete, bulk delete, and clear all.
Dependencies are mocked via ``app.dependency_overrides``.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from sec_semantic_search.api.app import app
from sec_semantic_search.api.dependencies import get_chroma, get_registry
from sec_semantic_search.core.exceptions import DatabaseError
from tests.helpers import make_filing_record


def _make_client(filings=None, chunk_count=0, get_filing_result=None):
    """Build a TestClient with mocked registry and chroma."""
    registry = MagicMock()
    registry.list_filings.return_value = filings or []
    registry.list_filings_page.return_value = (filings or [], len(filings or []))
    registry.get_filing.return_value = get_filing_result

    chroma = MagicMock()
    chroma.collection_count.return_value = chunk_count
    chroma.delete_filing.return_value = None

    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_chroma] = lambda: chroma
    return TestClient(app, raise_server_exceptions=False), registry, chroma


# -----------------------------------------------------------------------
# GET /api/filings/
# -----------------------------------------------------------------------


class TestListFilings:
    """List filings with optional filters and sorting."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_empty(self):
        client, *_ = _make_client()
        resp = client.get("/api/filings/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings"] == []
        assert data["total"] == 0

    def test_with_filings(self):
        filings = [make_filing_record()]
        client, *_ = _make_client(filings=filings)
        data = client.get("/api/filings/").json()
        assert data["total"] == 1
        assert data["filings"][0]["ticker"] == "AAPL"

    def test_filter_by_ticker(self):
        client, registry, _ = _make_client()
        client.get("/api/filings/?ticker=aapl")
        assert registry.list_filings_page.call_args.kwargs["ticker"] == "AAPL"
        assert registry.list_filings_page.call_args.kwargs["form_type"] is None

    def test_filter_by_form_type(self):
        client, registry, _ = _make_client()
        client.get("/api/filings/?form_type=10-q")
        assert registry.list_filings_page.call_args.kwargs["ticker"] is None
        assert registry.list_filings_page.call_args.kwargs["form_type"] == "10-Q"

    def test_defaults_request_first_page_by_filing_date_desc(self):
        client, registry, _ = _make_client()
        client.get("/api/filings/")
        registry.list_filings_page.assert_called_once_with(
            ticker=None,
            form_type=None,
            sort_by="filing_date",
            order="desc",
            limit=25,
            offset=0,
        )

    def test_sort_and_page_passed_to_registry(self):
        client, registry, _ = _make_client()
        client.get("/api/filings/?sort_by=ticker&order=asc&limit=10&offset=20")
        kwargs = registry.list_filings_page.call_args.kwargs
        assert (kwargs["sort_by"], kwargs["order"]) == ("ticker", "asc")
        assert (kwargs["limit"], kwargs["offset"]) == (10, 20)

    def test_total_counts_all_pages(self):
        client, registry, _ = _make_client()
        registry.list_filings_page.return_value = ([make_filing_record()], 137)
        data = client.get("/api/filings/?limit=1").json()
        assert len(data["filings"]) == 1
        assert data["total"] == 137

    @pytest.mark.parametrize(
        "query",
        [
            "limit=0",
            "limit=201",
            "offset=-1",
            "offset=1000001",
            "offset=99999999999999999999999",
            "sort_by=id",
            "sort_by=filing_date%3B%20DROP%20TABLE%20filings",
            "order=sideways",
        ],
    )
    def test_invalid_paging_or_sort_returns_422(self, query):
        client, registry, _ = _make_client()
        resp = client.get(f"/api/filings/?{query}")
        assert resp.status_code == 422
        registry.list_filings_page.assert_not_called()


# -----------------------------------------------------------------------
# GET /api/filings/{accession}
# -----------------------------------------------------------------------


class TestGetFiling:
    """Retrieve a single filing by accession number."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_existing(self):
        record = make_filing_record()
        client, *_ = _make_client(get_filing_result=record)
        resp = client.get("/api/filings/0000320193-24-000001")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"

    def test_not_found(self):
        client, *_ = _make_client(get_filing_result=None)
        resp = client.get("/api/filings/9999999999-99-999999")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"


# -----------------------------------------------------------------------
# DELETE /api/filings/{accession}
# -----------------------------------------------------------------------


class TestDeleteFiling:
    """Delete a single filing."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_existing(self):
        record = make_filing_record(chunk_count=50)
        client, *_ = _make_client(get_filing_result=record)
        resp = client.delete("/api/filings/0000320193-24-000001")
        assert resp.status_code == 200
        assert resp.json()["chunks_deleted"] == 50  # from FilingRecord.chunk_count

    def test_not_found(self):
        client, *_ = _make_client(get_filing_result=None)
        resp = client.delete("/api/filings/9999999999-99-999999")
        assert resp.status_code == 404

    def test_database_error(self):
        record = make_filing_record()
        client, _, chroma = _make_client(get_filing_result=record)
        chroma.delete_filing.side_effect = DatabaseError("disk full", details="ENOSPC")
        resp = client.delete("/api/filings/0000320193-24-000001")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "database_error"


# -----------------------------------------------------------------------
# POST /api/filings/delete-by-ids
# -----------------------------------------------------------------------


class TestDeleteByIds:
    """Delete specific filings by accession numbers."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_all_found(self):
        rec1 = make_filing_record(id=1, accession_number="0000000001-24-000001", chunk_count=50)
        rec2 = make_filing_record(
            id=2, accession_number="0000000002-24-000002", filing_date="2024-06-01", chunk_count=30
        )
        client, registry, _ = _make_client()
        registry.get_filings_by_accessions.return_value = [rec1, rec2]
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["0000000001-24-000001", "0000000002-24-000002"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings_deleted"] == 2
        assert data["chunks_deleted"] == 80  # 50 + 30
        assert data["not_found"] == []

    def test_some_not_found(self):
        rec1 = make_filing_record(id=1, accession_number="0000000001-24-000001", chunk_count=40)
        client, registry, _ = _make_client()
        registry.get_filings_by_accessions.return_value = [rec1]
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["0000000001-24-000001", "9999999999-99-999999"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings_deleted"] == 1
        assert data["chunks_deleted"] == 40
        assert data["not_found"] == ["9999999999-99-999999"]

    def test_all_not_found(self):
        client, registry, _ = _make_client()
        registry.get_filings_by_accessions.return_value = []
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["9999999991-99-999991", "9999999992-99-999992"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings_deleted"] == 0
        assert data["chunks_deleted"] == 0
        assert set(data["not_found"]) == {"9999999991-99-999991", "9999999992-99-999992"}

    def test_empty_list_returns_422(self):
        client, *_ = _make_client()
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": []},
        )
        assert resp.status_code == 422  # Pydantic min_length=1

    def test_too_many_ids_returns_422(self):
        client, *_ = _make_client()
        accession_numbers = [f"{i:010d}-24-000001" for i in range(51)]
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": accession_numbers},
        )
        assert resp.status_code == 422
        assert "At most 50 accession numbers" in str(resp.json())

    def test_database_error(self):
        rec = make_filing_record(accession_number="0000000001-24-000001")
        client, registry, chroma = _make_client()
        registry.get_filings_by_accessions.return_value = [rec]
        chroma.delete_filings_batch.side_effect = DatabaseError("disk full", details="ENOSPC")
        resp = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["0000000001-24-000001"]},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "database_error"


# -----------------------------------------------------------------------
# POST /api/filings/bulk-delete
# -----------------------------------------------------------------------


class TestBulkDelete:
    """Bulk delete filings by filter."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_by_ticker(self):
        filings = [make_filing_record()]  # chunk_count=100 by default
        client, registry, _ = _make_client()
        registry.list_filings.return_value = filings
        resp = client.post("/api/filings/bulk-delete", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings_deleted"] == 1
        assert data["chunks_deleted"] == 100  # from FilingRecord.chunk_count
        assert data["tickers_affected"] == ["AAPL"]

    def test_no_filters_returns_400(self):
        client, *_ = _make_client()
        resp = client.post("/api/filings/bulk-delete", json={})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "validation_error"

    def test_no_matching_filings(self):
        client, *_ = _make_client()
        resp = client.post("/api/filings/bulk-delete", json={"ticker": "XYZ"})
        assert resp.status_code == 200
        assert resp.json()["filings_deleted"] == 0

    def test_database_error(self):
        filings = [make_filing_record()]
        client, registry, chroma = _make_client()
        registry.list_filings.return_value = filings
        chroma.delete_filings_batch.side_effect = DatabaseError("fail")
        resp = client.post("/api/filings/bulk-delete", json={"ticker": "AAPL"})
        assert resp.status_code == 500


# -----------------------------------------------------------------------
# DELETE /api/filings/?confirm=true
# -----------------------------------------------------------------------


class TestClearAll:
    """Clear all filings from the database."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_without_confirm(self):
        client, *_ = _make_client()
        resp = client.delete("/api/filings/")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "confirmation_required"

    def test_confirm_empty_database(self):
        client, registry, chroma = _make_client()
        chroma.clear_collection.return_value = 0
        registry.clear_all.return_value = 0
        resp = client.delete("/api/filings/?confirm=true")
        assert resp.status_code == 200
        assert resp.json()["filings_deleted"] == 0

    def test_confirm_with_filings(self):
        client, registry, chroma = _make_client()
        chroma.clear_collection.return_value = 100
        registry.clear_all.return_value = 2
        resp = client.delete("/api/filings/?confirm=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filings_deleted"] == 2
        assert data["chunks_deleted"] == 100

    def test_database_error(self):
        client, registry, chroma = _make_client()
        chroma.clear_collection.side_effect = DatabaseError("fail")
        resp = client.delete("/api/filings/?confirm=true")
        assert resp.status_code == 500
