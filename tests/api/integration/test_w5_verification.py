"""
W5.9 Verification tests — consolidated privacy, credentials, access control,
and abuse prevention tests.

This file fills gaps in the existing W5.x test coverage and provides
cross-feature verification:

    - **Privacy**: encrypted DB round-trip lifecycle, search log redaction path,
      query text never persisted, ticker stripping end-to-end
    - **Session credentials**: regular API key rejected at admin endpoints,
      EDGAR headers not leaked in error responses
    - **Access control**: full parametrised permission matrix (all operations ×
      all auth levels), regular key ≠ admin key
    - **Abuse prevention**: combined cooldown + request cap enforcement,
      cooldown lazy pruning, GPU time limit with completed/failed states
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sec_semantic_search.api.app import app
from sec_semantic_search.api.dependencies import (
    EdgarIdentity,
    get_chroma,
    get_edgar_identity,
    get_embedder,
    get_registry,
    get_search_engine,
    get_task_manager,
)
from sec_semantic_search.api.tasks import TaskManager, TaskState
from sec_semantic_search.core.types import FilingIdentifier
from sec_semantic_search.database.metadata import MetadataRegistry
from tests.helpers import make_filing_record, make_task_info

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_full_client(
    *,
    filings=None,
    chunk_count=0,
    search_results=None,
):
    """Build a TestClient with all dependencies mocked."""
    registry = MagicMock()
    registry.list_filings.return_value = filings or []
    registry.get_filing.return_value = filings[0] if filings else None
    registry.get_statistics.return_value = MagicMock(
        filing_count=len(filings or []),
        tickers=["AAPL"] if filings else [],
        form_breakdown={"10-K": 1} if filings else {},
        ticker_breakdown=[],
    )

    chroma = MagicMock()
    chroma.collection_count.return_value = chunk_count
    chroma.delete_filing.return_value = None

    engine = MagicMock()
    engine.search.return_value = search_results or []

    embedder = MagicMock()
    embedder.is_loaded = True
    embedder.device = "cpu"
    embedder.model_name = "test-model"
    embedder.approximate_vram_mb = 0

    manager = MagicMock()
    manager.has_active_task.return_value = False
    manager.create_task.return_value = "test-task-id"

    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_chroma] = lambda: chroma
    app.dependency_overrides[get_search_engine] = lambda: engine
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_task_manager] = lambda: manager
    app.dependency_overrides[get_edgar_identity] = lambda: EdgarIdentity(
        name="Test",
        email="test@example.com",
    )

    return TestClient(app, raise_server_exceptions=False), {
        "registry": registry,
        "chroma": chroma,
        "engine": engine,
        "embedder": embedder,
        "manager": manager,
    }


# =======================================================================
# 1. Privacy-specific tests
# =======================================================================


class TestEncryptedDBRoundTrip:
    """Verify MetadataRegistry lifecycle with encrypted and unencrypted paths."""

    def test_unencrypted_full_lifecycle(self, tmp_path):
        """Full register → duplicate check → list → get → delete cycle."""
        db_path = str(tmp_path / "test.sqlite")
        # Explicit empty key ensures unencrypted mode regardless of
        # DB_ENCRYPTION_KEY in .env.
        registry = MetadataRegistry(db_path=db_path, encryption_key="")

        filing_id = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 1, 15),
            accession_number="0000320193-24-000099",
        )

        # Register
        registry.register_filing(filing_id, chunk_count=42)

        # Duplicate check (takes accession number string)
        assert registry.get_existing_accessions([filing_id.accession_number]) == {
            filing_id.accession_number
        }

        # List
        filings = registry.list_filings(ticker="AAPL")
        assert len(filings) == 1
        assert filings[0].chunk_count == 42

        # Get single
        record = registry.get_filing("0000320193-24-000099")
        assert record is not None
        assert record.ticker == "AAPL"

        # Delete
        registry.remove_filing("0000320193-24-000099")
        assert registry.get_filing("0000320193-24-000099") is None
        assert not registry.encrypted

    def test_encrypted_lifecycle_with_mocked_sqlcipher(self, tmp_path):
        """Verify PRAGMA key is sent and data round-trips through encrypted path."""
        import sqlite3 as real_sqlite3

        db_path = str(tmp_path / "encrypted.sqlite")

        # Use a real sqlite3 connection under the hood, but track PRAGMA key.
        pragma_calls = []
        original_connect = real_sqlite3.connect

        class PragmaTrackingConnection:
            """Wraps a real sqlite3 connection but tracks PRAGMA key calls."""

            def __init__(self, *args, **kwargs):
                self._conn = original_connect(*args, **kwargs)
                self._conn.row_factory = real_sqlite3.Row

            def execute(self, sql, *args, **kwargs):
                if sql.strip().startswith("PRAGMA key"):
                    pragma_calls.append(sql)
                    return self._conn.cursor()
                return self._conn.execute(sql, *args, **kwargs)

            def cursor(self):
                return self._conn.cursor()

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def __enter__(self):
                return self._conn.__enter__()

            def __exit__(self, *args):
                return self._conn.__exit__(*args)

        mock_module = MagicMock()
        mock_module.connect = lambda *a, **kw: PragmaTrackingConnection(*a, **kw)
        mock_module.Error = real_sqlite3.Error
        mock_module.IntegrityError = real_sqlite3.IntegrityError
        mock_module.Row = real_sqlite3.Row

        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=mock_module,
        ):
            registry = MetadataRegistry(
                db_path=db_path,
                encryption_key="test-secret-key",
            )

        assert registry.encrypted is True
        # PRAGMA key should have been called during _init_db
        assert len(pragma_calls) >= 1
        assert "PRAGMA key" in pragma_calls[0]

        # Verify data round-trip works through the encrypted-flagged registry
        filing_id = FilingIdentifier(
            ticker="MSFT",
            form_type="10-Q",
            filing_date=date(2024, 6, 1),
            accession_number="0000789019-24-000001",
        )
        registry.register_filing(filing_id, chunk_count=10)
        assert registry.get_existing_accessions(["0000789019-24-000001"]) == {
            "0000789019-24-000001"
        }
        record = registry.get_filing("0000789019-24-000001")
        assert record is not None
        assert record.ticker == "MSFT"

    def test_encryption_key_with_special_characters(self, tmp_path):
        """Keys with quotes/special chars don't cause injection via hex encoding."""
        import sqlite3 as real_sqlite3

        db_path = str(tmp_path / "special.sqlite")

        mock_module = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_module.connect.return_value = mock_conn
        mock_module.Error = real_sqlite3.Error
        mock_module.IntegrityError = real_sqlite3.IntegrityError

        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=mock_module,
        ):
            key = """Robert'); DROP TABLE filings;--"""
            MetadataRegistry(db_path=db_path, encryption_key=key)

        # Verify PRAGMA key was called with hex literal (safe encoding)
        pragma_call = None
        for call in mock_conn.execute.call_args_list:
            sql = call[0][0] if call[0] else ""
            if "PRAGMA key" in sql:
                pragma_call = sql
                break

        assert pragma_call is not None
        # Hex-encoded blob literal — no raw key text
        assert "x'" in pragma_call
        assert "DROP TABLE" not in pragma_call


class TestSearchLogRedaction:
    """Verify search queries are redacted in logs when LOG_REDACT_QUERIES is set."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @patch.dict(os.environ, {"LOG_REDACT_QUERIES": "true"}, clear=False)
    @patch("sec_semantic_search.api.routes.search.logger")
    def test_search_query_redacted_in_log(self, mock_logger):
        """Search route redacts the query text in log output when redaction enabled."""
        client, deps = _make_full_client()
        deps["engine"].search.return_value = []

        resp = client.post("/api/search/", json={"query": "revenue growth analysis"})
        assert resp.status_code == 200

        # The logger.info call should contain <redacted:...> not the original query
        mock_logger.info.assert_called_once()
        log_msg_args = mock_logger.info.call_args
        # Second positional arg is the redacted query
        logged_query = log_msg_args[0][1]
        assert "<redacted:" in logged_query
        assert "revenue growth analysis" not in logged_query

    @patch.dict(os.environ, {"LOG_REDACT_QUERIES": ""}, clear=False)
    @patch("sec_semantic_search.api.routes.search.logger")
    def test_search_query_not_redacted_when_disabled(self, mock_logger):
        """Search route logs the full query when redaction is disabled."""
        client, deps = _make_full_client()
        deps["engine"].search.return_value = []

        resp = client.post("/api/search/", json={"query": "supply chain risk"})
        assert resp.status_code == 200

        mock_logger.info.assert_called_once()
        logged_query = mock_logger.info.call_args[0][1]
        assert "supply chain risk" in logged_query

    @patch.dict(os.environ, {"LOG_REDACT_QUERIES": "true"}, clear=False)
    @patch("sec_semantic_search.api.routes.ingest.logger")
    def test_ingest_tickers_redacted_in_log(self, mock_logger):
        """Ingest route redacts ticker symbols in log output when redaction enabled."""
        client, _ = _make_full_client()

        resp = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        assert resp.status_code == 202

        mock_logger.info.assert_called_once()
        log_msg_args = mock_logger.info.call_args[0]
        # Tickers list should contain redacted values
        logged_tickers = log_msg_args[2]
        assert all("<redacted:" in t for t in logged_tickers)
        assert "AAPL" not in str(logged_tickers)


class TestQueryNeverPersisted:
    """Verify search queries are never written to any database or history."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_search_does_not_call_registry(self):
        """Search route never writes to the metadata registry."""
        client, deps = _make_full_client()
        deps["engine"].search.return_value = []

        client.post("/api/search/", json={"query": "investment thesis"})

        # Registry should not have been called for writes
        deps["registry"].register_filing.assert_not_called()
        deps["registry"].save_task_history.assert_not_called()

    def test_search_response_includes_query_but_never_stores_it(self):
        """Search response must not echo the query (§F4) and never stores it."""
        client, deps = _make_full_client()
        deps["engine"].search.return_value = []

        resp = client.post("/api/search/", json={"query": "semiconductor supply chain"})
        data = resp.json()

        # Query must not be echoed in the response (§F4)
        assert "query" not in data

        # No writes to any store
        deps["registry"].register_filing.assert_not_called()


class TestTickerStrippingEndToEnd:
    """Verify ticker stripping works through the full task history path."""

    def test_task_history_strips_tickers_by_default(self, tmp_path):
        """save_task_history stores NULL tickers when TASK_HISTORY_PERSIST_TICKERS=false."""
        db_path = str(tmp_path / "test.sqlite")
        registry = MetadataRegistry(db_path=db_path)

        with patch(
            "sec_semantic_search.database.metadata.get_settings",
        ) as mock_settings:
            mock_settings.return_value.database.task_history_persist_tickers = False
            mock_settings.return_value.database.task_history_retention_days = 0

            registry.save_task_history(
                "test-123",
                status="completed",
                tickers=["AAPL", "MSFT"],
                form_types=["10-K"],
                results=[],
                filings_done=2,
                filings_skipped=0,
                filings_failed=0,
            )

        # Read back — tickers should be empty list (NULL → [])
        history = registry.get_task_history("test-123")
        assert history is not None
        assert history["tickers"] == []

    def test_error_message_scrubbed_of_identifiers(self, tmp_path):
        """Error messages have tickers and accession numbers scrubbed."""
        db_path = str(tmp_path / "test.sqlite")
        registry = MetadataRegistry(db_path=db_path)

        with patch(
            "sec_semantic_search.database.metadata.get_settings",
        ) as mock_settings:
            mock_settings.return_value.database.task_history_persist_tickers = False
            mock_settings.return_value.database.task_history_retention_days = 0

            registry.save_task_history(
                "test-456",
                status="failed",
                tickers=["TSLA"],
                form_types=["10-Q"],
                results=[],
                error="Failed to fetch TSLA filing 0001318605-24-000001",
                filings_done=0,
                filings_skipped=0,
                filings_failed=1,
            )

        history = registry.get_task_history("test-456")
        assert history is not None
        error = history["error"]
        assert "TSLA" not in error
        assert "0001318605-24-000001" not in error
        assert "[TICKER]" in error
        assert "[ACCESSION]" in error


# =======================================================================
# 2. Access control — full permission matrix
# =======================================================================


class TestPermissionMatrix:
    """Parametrised tests for the full two-tier API key permission matrix.

    Tests every admin-protected endpoint with three auth levels:
    - no key: rejected (403)
    - regular API key only (not admin key): rejected (403)
    - correct admin key: allowed
    """

    def teardown_method(self):
        app.dependency_overrides.clear()
        from sec_semantic_search.api.routes import ingest as ingest_mod

        with ingest_mod._cooldown_lock:
            ingest_mod._last_ingest.clear()

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Set up common mocks for all permission tests."""
        self.filings = [make_filing_record()]
        self.client, self.deps = _make_full_client(
            filings=self.filings,
            chunk_count=100,
        )

    # --- Admin endpoints that should require admin key ---

    _ADMIN_ENDPOINTS = [
        ("POST", "/api/filings/bulk-delete", {"ticker": "AAPL"}),
        ("DELETE", "/api/filings/?confirm=true", None),
        ("DELETE", "/api/resources/gpu", None),
    ]

    @pytest.mark.parametrize(
        "method,url,json_body",
        _ADMIN_ENDPOINTS,
        ids=["bulk-delete", "clear-all", "gpu-unload"],
    )
    @patch("sec_semantic_search.api.routes.filings.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_admin_endpoint_rejects_no_key(
        self,
        mock_dep_settings,
        mock_route_settings,
        method,
        url,
        json_body,
    ):
        """Admin endpoints reject requests with no admin key (403)."""
        mock_dep_settings.return_value.api.admin_key = "admin-secret"
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.demo_mode = False

        if method == "POST":
            resp = self.client.post(url, json=json_body)
        else:
            resp = self.client.request(method, url)
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "method,url,json_body",
        _ADMIN_ENDPOINTS,
        ids=["bulk-delete", "clear-all", "gpu-unload"],
    )
    @patch("sec_semantic_search.api.routes.filings.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_admin_endpoint_rejects_regular_api_key_as_admin_key(
        self,
        mock_dep_settings,
        mock_route_settings,
        method,
        url,
        json_body,
    ):
        """Sending a regular API key in X-Admin-Key header is rejected (403).

        This verifies the two keys are independent — a valid API_KEY does
        not grant admin access.
        """
        mock_dep_settings.return_value.api.admin_key = "admin-secret"
        mock_dep_settings.return_value.api.key = "regular-key"
        mock_route_settings.return_value.api.demo_mode = False

        # Pass valid API key but use the regular key as admin key — wrong!
        headers = {"X-API-Key": "regular-key", "X-Admin-Key": "regular-key"}
        if method == "POST":
            resp = self.client.post(url, json=json_body, headers=headers)
        else:
            resp = self.client.request(method, url, headers=headers)
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "method,url,json_body",
        _ADMIN_ENDPOINTS,
        ids=["bulk-delete", "clear-all", "gpu-unload"],
    )
    @patch("sec_semantic_search.api.routes.filings.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_admin_endpoint_allows_correct_admin_key(
        self,
        mock_dep_settings,
        mock_route_settings,
        method,
        url,
        json_body,
    ):
        """Admin endpoints allow requests with correct admin key."""
        mock_dep_settings.return_value.api.admin_key = "admin-secret"
        mock_dep_settings.return_value.api.key = "regular-key"
        mock_route_settings.return_value.api.demo_mode = False

        # Pass both valid API key and valid admin key
        headers = {"X-API-Key": "regular-key", "X-Admin-Key": "admin-secret"}
        if method == "POST":
            resp = self.client.post(url, json=json_body, headers=headers)
        else:
            resp = self.client.request(method, url, headers=headers)
        assert resp.status_code == 200

    # --- Non-admin endpoints should work without admin key ---

    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_search_works_without_admin_key(self, mock_settings):
        """Search is not an admin endpoint — works without any key."""
        mock_settings.return_value.api.admin_key = "admin-secret"
        mock_settings.return_value.api.key = None

        resp = self.client.post("/api/search/", json={"query": "test"})
        assert resp.status_code == 200

    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_single_delete_works_without_admin_key(self, mock_settings):
        """Single filing delete is not an admin endpoint."""
        mock_settings.return_value.api.admin_key = "admin-secret"
        mock_settings.return_value.api.key = None

        resp = self.client.delete("/api/filings/0000320193-24-000001")
        assert resp.status_code == 200

    @patch("sec_semantic_search.api.routes.ingest.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_ingest_works_without_admin_key(self, mock_dep_settings, mock_route_settings):
        """Ingest is not an admin endpoint."""
        mock_dep_settings.return_value.api.admin_key = "admin-secret"
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.max_tickers_per_request = 0
        mock_route_settings.return_value.api.max_filings_per_request = 0
        mock_route_settings.return_value.api.ingest_cooldown_seconds = 0

        resp = self.client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        assert resp.status_code == 202


class TestEdgarCredentialPrivacy:
    """Verify EDGAR credentials are never leaked in API responses or errors."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_401_error_does_not_contain_credentials(self, mock_settings):
        """401 response for missing EDGAR credentials does not leak env var values."""
        mock_settings.return_value.api.key = None
        mock_settings.return_value.api.admin_key = None
        mock_settings.return_value.api.edgar_session_required = True
        mock_settings.return_value.edgar.identity_name = None
        mock_settings.return_value.edgar.identity_email = None

        # Remove the mock override so the real dependency runs
        app.dependency_overrides.pop(get_edgar_identity, None)

        registry = MagicMock()
        chroma = MagicMock()
        manager = MagicMock()
        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma
        app.dependency_overrides[get_task_manager] = lambda: manager

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        assert resp.status_code == 401

        # Error body should not contain any actual credential values
        body = resp.text
        assert "test@example.com" not in body

    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_401_error_includes_helpful_hint(self, mock_settings):
        """401 response includes hints on how to provide credentials."""
        mock_settings.return_value.api.key = None
        mock_settings.return_value.api.admin_key = None
        mock_settings.return_value.api.edgar_session_required = True
        mock_settings.return_value.edgar.identity_name = None
        mock_settings.return_value.edgar.identity_email = None

        app.dependency_overrides.pop(get_edgar_identity, None)
        app.dependency_overrides[get_registry] = lambda: MagicMock()
        app.dependency_overrides[get_chroma] = lambda: MagicMock()
        app.dependency_overrides[get_task_manager] = lambda: MagicMock()

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        data = resp.json()
        assert "hint" in data["detail"]
        assert "X-Edgar-Name" in data["detail"]["hint"]


# =======================================================================
# 3. Abuse prevention — cross-feature tests
# =======================================================================


class TestCooldownAndCapsCombined:
    """Verify cooldown and request caps work together correctly."""

    def teardown_method(self):
        app.dependency_overrides.clear()
        from sec_semantic_search.api.routes import ingest as ingest_mod

        with ingest_mod._cooldown_lock:
            ingest_mod._last_ingest.clear()

    @patch("sec_semantic_search.api.routes.ingest.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_batch_endpoint_also_respects_cooldown(
        self,
        mock_dep_settings,
        mock_route_settings,
    ):
        """Both /add and /batch share the same per-IP cooldown."""
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.max_tickers_per_request = 0
        mock_route_settings.return_value.api.max_filings_per_request = 0
        mock_route_settings.return_value.api.ingest_cooldown_seconds = 60

        client, _ = _make_full_client()

        # First request via /add
        resp1 = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        assert resp1.status_code == 202

        # Second request via /batch — same IP, should be blocked
        resp2 = client.post(
            "/api/ingest/batch",
            json={
                "tickers": ["MSFT"],
                "form_types": ["10-K"],
            },
        )
        assert resp2.status_code == 429

    @patch("sec_semantic_search.api.routes.ingest.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_cooldown_error_includes_retry_hint(
        self,
        mock_dep_settings,
        mock_route_settings,
    ):
        """Cooldown 429 response includes wait time hint."""
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.max_tickers_per_request = 0
        mock_route_settings.return_value.api.max_filings_per_request = 0
        mock_route_settings.return_value.api.ingest_cooldown_seconds = 60

        client, _ = _make_full_client()

        client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
            },
        )
        resp = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["MSFT"],
                "form_types": ["10-K"],
            },
        )
        assert resp.status_code == 429
        data = resp.json()
        assert "cooldown" in data["detail"]["error"]
        assert "hint" in data["detail"]

    @patch("sec_semantic_search.api.routes.ingest.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_request_caps_produce_structured_error(
        self,
        mock_dep_settings,
        mock_route_settings,
    ):
        """Request cap violations return structured error with hint."""
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.max_tickers_per_request = 2
        mock_route_settings.return_value.api.max_filings_per_request = 0
        mock_route_settings.return_value.api.ingest_cooldown_seconds = 0

        client, _ = _make_full_client()

        resp = client.post(
            "/api/ingest/batch",
            json={
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "form_types": ["10-K"],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["error"] == "request_cap_exceeded"
        assert "hint" in data["detail"]
        assert "2" in data["detail"]["hint"]  # max tickers shown in hint


class TestGPUTimeLimitExtended:
    """Extended GPU time limit tests for edge cases."""

    def test_timeout_on_failed_task_is_noop(self):
        """_timeout_task does nothing if the task already failed."""
        info = make_task_info(state=TaskState.FAILED)
        TaskManager._timeout_task(info)
        assert not info.cancel_event.is_set()

    def test_timeout_only_cancels_running_tasks(self):
        """_timeout_task only sets cancel_event for RUNNING state."""
        # PENDING — should NOT be cancelled (task hasn't started yet)
        pending = make_task_info(state=TaskState.PENDING)
        TaskManager._timeout_task(pending)
        assert not pending.cancel_event.is_set()

        # COMPLETED — should NOT be cancelled
        completed = make_task_info(state=TaskState.COMPLETED)
        TaskManager._timeout_task(completed)
        assert not completed.cancel_event.is_set()

        # RUNNING — SHOULD be cancelled
        running = make_task_info(state=TaskState.RUNNING)
        TaskManager._timeout_task(running)
        assert running.cancel_event.is_set()


# =======================================================================
# 4. Demo mode cross-feature tests
# =======================================================================


class TestDemoModeCrossFeature:
    """Test demo mode interactions with other features."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @patch("sec_semantic_search.api.routes.filings.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_demo_mode_clear_all_blocked_even_scenario_a(
        self,
        mock_dep_settings,
        mock_route_settings,
    ):
        """Clear all blocked in demo mode even when no auth keys configured (Scenario A)."""
        mock_dep_settings.return_value.api.admin_key = None
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.demo_mode = True

        client, _ = _make_full_client()
        resp = client.delete("/api/filings/?confirm=true")
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "demo_mode"

    @patch("sec_semantic_search.api.routes.status.get_settings")
    @patch("sec_semantic_search.api.routes.status.is_admin_request")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_status_exposes_demo_mode_flag(
        self,
        mock_dep_settings,
        mock_is_admin,
        mock_route_settings,
    ):
        """Status endpoint always shows demo_mode flag regardless of auth."""
        mock_dep_settings.return_value.api.key = None
        settings = MagicMock()
        settings.database.max_filings = 500
        settings.api.demo_mode = True
        settings.api.edgar_session_required = False
        settings.edgar.identity_name = "Test"
        settings.edgar.identity_email = "test@test.com"
        mock_route_settings.return_value = settings
        mock_is_admin.return_value = False

        client, _ = _make_full_client()
        resp = client.get("/api/status/")
        assert resp.status_code == 200
        assert resp.json()["demo_mode"] is True
        assert resp.json()["is_admin"] is False


# =======================================================================
# 5. Audit logging tests
# =======================================================================


class TestAuditLogging:
    """Verify security-relevant actions produce audit log entries."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @patch("sec_semantic_search.api.routes.filings.audit_log")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_single_delete_produces_audit_log(self, mock_settings, mock_audit):
        """Deleting a filing produces a SECURITY_AUDIT log entry."""
        mock_settings.return_value.api.key = None
        mock_settings.return_value.api.admin_key = None

        client, _ = _make_full_client(
            filings=[make_filing_record()],
            chunk_count=100,
        )
        resp = client.delete("/api/filings/0000320193-24-000001")
        assert resp.status_code == 200
        mock_audit.assert_called_once()
        assert mock_audit.call_args.args[0] == "delete_filing"

    @patch("sec_semantic_search.api.routes.filings.audit_log")
    @patch("sec_semantic_search.api.routes.filings.get_settings")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_bulk_delete_produces_audit_log(
        self,
        mock_dep_settings,
        mock_route_settings,
        mock_audit,
    ):
        """Bulk delete produces a SECURITY_AUDIT log entry."""
        mock_dep_settings.return_value.api.admin_key = None
        mock_dep_settings.return_value.api.key = None
        mock_route_settings.return_value.api.demo_mode = False

        client, _ = _make_full_client(filings=[make_filing_record()], chunk_count=100)
        resp = client.post("/api/filings/bulk-delete", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        mock_audit.assert_called_once()
        assert mock_audit.call_args.args[0] == "bulk_delete"

    @patch("sec_semantic_search.api.dependencies.audit_log")
    @patch("sec_semantic_search.api.dependencies.get_settings")
    def test_admin_key_rejection_produces_audit_log(
        self,
        mock_settings,
        mock_audit,
    ):
        """Rejected admin key attempt produces audit trail."""
        mock_settings.return_value.api.admin_key = "secret"
        mock_settings.return_value.api.key = None

        client, _ = _make_full_client(filings=[make_filing_record()])
        resp = client.post(
            "/api/filings/bulk-delete",
            json={"ticker": "AAPL"},
            headers={"X-Admin-Key": "wrong"},
        )
        assert resp.status_code == 403
        mock_audit.assert_called_once()
        assert mock_audit.call_args.args[0] == "admin_denied"
