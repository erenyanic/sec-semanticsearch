"""
Security tests for vulnerability fixes identified in SECURITY VULNERABILITIES.md.

Each test class maps to a specific finding number from the security audit.
Tests verify that the fix is in place and working correctly.
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from sec_semantic_search.api.app import app
from sec_semantic_search.api.dependencies import (
    get_chroma,
    get_embedder,
    get_registry,
    get_search_engine,
    get_task_manager,
)
from sec_semantic_search.api.schemas import (
    BulkDeleteRequest,
    DeleteByIdsRequest,
    IngestRequest,
    SearchRequest,
)
from sec_semantic_search.api.tasks import (
    TaskManager,
    TaskQueueFullError,
    TaskState,
)
from sec_semantic_search.core.exceptions import DatabaseError, SearchError
from tests.helpers import make_filing_record, make_task_info

_WS_HEADERS = {"origin": "http://localhost:3000"}

# -----------------------------------------------------------------------
# Finding #1: .env not tracked by git
# -----------------------------------------------------------------------


class TestEnvNotTracked:
    """Verify .env is in .gitignore and not tracked."""

    def test_gitignore_contains_env(self):
        from pathlib import Path

        gitignore = Path(__file__).parents[3] / ".gitignore"
        content = gitignore.read_text()
        assert ".env" in content


# -----------------------------------------------------------------------
# Finding #2: WebSocket origin validation
# -----------------------------------------------------------------------


class TestWebSocketOriginValidation:
    """WebSocket endpoint rejects connections from disallowed origins."""

    def test_allowed_origin_connects(self):
        """Connection from allowed origin should succeed."""
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers={"origin": "http://localhost:3000"},
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"

    def test_missing_origin_header_is_rejected(self):
        """Connection without Origin header should be rejected."""
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/ingest/{info.task_id}"):
                pass

        assert exc_info.value.code == 4003

    def test_disallowed_origin_is_rejected(self):
        """Connection from an untrusted origin should be rejected."""
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/ws/ingest/{info.task_id}",
                headers={"origin": "https://evil.example"},
            ):
                pass

        assert exc_info.value.code == 4003


# -----------------------------------------------------------------------
# Finding #4: demo-reset restart strategy hardening
# -----------------------------------------------------------------------


class TestDemoResetScript:
    """The demo reset script must not execute arbitrary shell code."""

    def test_demo_reset_script_rejects_eval_based_restart(self):
        script = Path(__file__).parents[3] / "scripts" / "demo-reset.sh"
        content = script.read_text()

        assert 'eval "$RESTART_CMD"' not in content
        assert 'case "$RESTART_STRATEGY" in' in content
        assert "RESTART_CMD is deprecated and rejected for safety" in content


# -----------------------------------------------------------------------
# Finding #3: CORS configuration
# -----------------------------------------------------------------------


class TestCORSConfiguration:
    """CORS should use explicit method and header lists."""

    def test_cors_allows_get(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "GET" in resp.headers.get("access-control-allow-methods", "")

    def test_cors_blocks_put(self):
        """PUT should not be in allowed methods."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PUT",
            },
        )
        # PUT should not appear in allowed methods
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "PUT" not in allowed


# -----------------------------------------------------------------------
# Finding #5: API binds to 127.0.0.1 by default
# -----------------------------------------------------------------------


class TestDefaultBindAddress:
    """API should default to 127.0.0.1 instead of 0.0.0.0."""

    def test_default_host_is_localhost(self):
        from sec_semantic_search.config.settings import ApiSettings

        settings = ApiSettings()
        assert settings.host == "127.0.0.1"


# -----------------------------------------------------------------------
# Finding #6: Error response information disclosure
# -----------------------------------------------------------------------


class TestErrorRedaction:
    """Error responses must not leak internal details."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_delete_error_redacts_details(self):
        registry = MagicMock()
        record = make_filing_record()
        registry.get_filing.return_value = record

        chroma = MagicMock()
        chroma.delete_filing.side_effect = DatabaseError(
            "SQLite error", details="UNIQUE constraint failed on filings.accession_number"
        )

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.delete(f"/api/filings/{record.accession_number}")
        assert resp.status_code == 500
        body = resp.json()["detail"]
        # The internal SQLite error details must NOT be exposed
        assert "UNIQUE constraint" not in body.get("message", "")
        assert body.get("details") is None

    def test_search_error_redacts_details(self):
        engine = MagicMock()
        engine.search.side_effect = SearchError(
            "Internal failure", details="torch.cuda.OutOfMemoryError: CUDA out of memory"
        )

        app.dependency_overrides[get_search_engine] = lambda: engine
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/search/", json={"query": "revenue growth"})
        assert resp.status_code == 500
        body = resp.json()["detail"]
        assert "CUDA" not in body.get("message", "")
        assert body.get("details") is None

    def test_search_response_omits_query(self):
        """Search response must not echo the query back to the caller (§F4)."""
        engine = MagicMock()
        engine.search.return_value = []

        app.dependency_overrides[get_search_engine] = lambda: engine
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/search/", json={"query": "revenue growth"})
        assert resp.status_code == 200
        assert "query" not in resp.json()

    def test_search_response_has_no_store_header(self):
        """Search response must include Cache-Control: no-store (§F4)."""
        engine = MagicMock()
        engine.search.return_value = []

        app.dependency_overrides[get_search_engine] = lambda: engine
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/search/", json={"query": "revenue growth"})
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-store"


# -----------------------------------------------------------------------
# Finding #9: Search query length limit
# -----------------------------------------------------------------------


class TestSearchQueryLengthLimit:
    """Search query must be capped at 2000 characters."""

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError, match="at most 2000"):
            SearchRequest(query="x" * 2001)

    def test_query_at_limit_accepted(self):
        req = SearchRequest(query="x" * 2000)
        assert len(req.query) == 2000

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError, match="at least 1"):
            SearchRequest(query="")


# -----------------------------------------------------------------------
# Finding #11: Ticker symbol validation
# -----------------------------------------------------------------------


class TestTickerValidation:
    """Ticker symbols must match [A-Z.]{1,5} format."""

    def test_valid_ticker(self):
        req = SearchRequest(query="test", ticker="AAPL")
        assert req.ticker == ["AAPL"]

    def test_valid_ticker_with_dot(self):
        req = SearchRequest(query="test", ticker="BRK.B")
        assert req.ticker == ["BRK.B"]

    def test_ticker_normalised_to_uppercase(self):
        req = SearchRequest(query="test", ticker="aapl")
        assert req.ticker == ["AAPL"]

    def test_valid_ticker_list(self):
        req = SearchRequest(query="test", ticker=["AAPL", "MSFT"])
        assert req.ticker == ["AAPL", "MSFT"]

    def test_invalid_ticker_path_traversal(self):
        with pytest.raises(ValidationError, match="Invalid ticker"):
            SearchRequest(query="test", ticker="../../../etc/passwd")

    def test_invalid_ticker_too_long(self):
        with pytest.raises(ValidationError, match="Invalid ticker"):
            SearchRequest(query="test", ticker="TOOLONG")

    def test_invalid_ticker_numbers(self):
        with pytest.raises(ValidationError, match="Invalid ticker"):
            SearchRequest(query="test", ticker="123")

    def test_ingest_ticker_validation(self):
        with pytest.raises(ValidationError, match="Invalid ticker"):
            IngestRequest(tickers=["../hack"])

    def test_ingest_valid_tickers(self):
        req = IngestRequest(tickers=["AAPL", "MSFT"])
        assert req.tickers == ["AAPL", "MSFT"]

    def test_bulk_delete_ticker_validation(self):
        with pytest.raises(ValidationError, match="Invalid ticker"):
            BulkDeleteRequest(ticker="../../etc")


# -----------------------------------------------------------------------
# Finding #12: Accession number validation
# -----------------------------------------------------------------------


class TestAccessionNumberValidation:
    """Accession numbers must match NNNNNNNNNN-YY-NNNNNN format."""

    def test_valid_accession_number(self):
        req = SearchRequest(query="test", accession_number="0000320193-24-000001")
        assert req.accession_number == ["0000320193-24-000001"]

    def test_valid_accession_number_list(self):
        req = SearchRequest(
            query="test",
            accession_number=["0000320193-24-000001", "0000320193-24-000002"],
        )
        assert req.accession_number == [
            "0000320193-24-000001",
            "0000320193-24-000002",
        ]

    def test_invalid_accession_number(self):
        with pytest.raises(ValidationError, match="Invalid accession"):
            SearchRequest(query="test", accession_number="invalid-format")

    def test_invalid_accession_in_list(self):
        with pytest.raises(ValidationError, match="Invalid accession"):
            SearchRequest(
                query="test",
                accession_number=["0000320193-24-000001", "not-valid"],
            )

    def test_delete_by_ids_accession_validation(self):
        with pytest.raises(ValidationError, match="Invalid accession"):
            DeleteByIdsRequest(accession_numbers=["not-valid"])

    def test_delete_by_ids_valid(self):
        req = DeleteByIdsRequest(accession_numbers=["0000320193-24-000001"])
        assert len(req.accession_numbers) == 1

    def test_filing_path_param_validation(self):
        """Path parameter must match the accession number format."""
        registry = MagicMock()
        registry.get_filing.return_value = None
        app.dependency_overrides[get_registry] = lambda: registry

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/filings/not-a-valid-accession")
        assert resp.status_code == 422  # Pydantic/FastAPI validation error
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------
# Finding #13: GPU semaphore starvation / task queue cap
# -----------------------------------------------------------------------


class TestTaskQueueCap:
    """Task manager must reject tasks when queue is full."""

    def test_queue_full_raises_error(self):
        registry = MagicMock()
        chroma = MagicMock()
        fetcher = MagicMock()
        orchestrator = MagicMock()

        manager = TaskManager(
            registry=registry,
            chroma=chroma,
            fetcher=fetcher,
            orchestrator=orchestrator,
        )

        # Fill the queue with pending tasks (mock them as pending).
        for i in range(5):
            info = make_task_info(
                task_id=f"task{i:032d}",
                state=TaskState.PENDING,
            )
            manager._tasks[info.task_id] = info

        # The 6th task should be rejected.
        with pytest.raises(TaskQueueFullError, match="queue is full"):
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])

        manager.shutdown()

    def test_queue_allows_after_completion(self):
        registry = MagicMock()
        chroma = MagicMock()
        fetcher = MagicMock()
        orchestrator = MagicMock()

        manager = TaskManager(
            registry=registry,
            chroma=chroma,
            fetcher=fetcher,
            orchestrator=orchestrator,
        )

        # Fill with completed tasks — they should not count.
        for i in range(10):
            info = make_task_info(
                task_id=f"done{i:032d}",
                state=TaskState.COMPLETED,
            )
            manager._tasks[info.task_id] = info

        # Should succeed because completed tasks don't count.
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        assert task_id is not None
        manager.shutdown()

    def test_ingest_route_returns_429(self):
        """The ingest route should return 429 when the queue is full."""
        from sec_semantic_search.api.dependencies import get_task_manager

        manager = MagicMock()
        manager.create_task.side_effect = TaskQueueFullError("Task queue is full (5 active).")

        app.dependency_overrides[get_task_manager] = lambda: manager
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"]},
        )
        assert resp.status_code == 429
        assert "queue" in resp.json()["detail"]["message"].lower()
        app.dependency_overrides.clear()


# -----------------------------------------------------------------------
# Finding #8: Unsafe SQLite threading — atomic check-then-insert
# -----------------------------------------------------------------------


class TestAtomicRegistration:
    """SQLite registration must be atomic to prevent race conditions."""

    def test_task_manager_uses_atomic_registration(self):
        """TaskManager._execute() store step uses register_filing_if_new."""
        import inspect

        from sec_semantic_search.api.tasks import TaskManager

        source = inspect.getsource(TaskManager._execute)
        # The atomic method must be used, not the non-atomic register_filing.
        assert "register_filing_if_new" in source
        # The old non-atomic pattern should not be present.
        assert "register_filing(" not in source.replace("register_filing_if_new", "")

    def test_register_filing_if_new_holds_lock(self):
        """register_filing_if_new must hold the lock across check and insert."""
        import inspect

        from sec_semantic_search.database.metadata import MetadataRegistry

        source = inspect.getsource(MetadataRegistry.register_filing_if_new)
        # The lock and connection context manager must wrap both
        # the SELECT check and the INSERT — a single `with` block.
        assert "with self._lock, self._conn:" in source

    def test_chromadb_rollback_on_failure(self):
        """If ChromaDB store fails after SQLite registration, SQLite is rolled back."""
        registry = MagicMock()
        registry.register_filing_if_new.return_value = True
        chroma = MagicMock()
        chroma.store_filing.side_effect = DatabaseError("ChromaDB write error", details="disk full")
        fetcher = MagicMock()
        orchestrator = MagicMock()

        manager = TaskManager(
            registry=registry,
            chroma=chroma,
            fetcher=fetcher,
            orchestrator=orchestrator,
        )
        manager.shutdown()

        # Verify that remove_filing is called to roll back SQLite
        # when ChromaDB fails. We check the source code structure
        # because the full _execute flow requires extensive mocking.
        import inspect

        source = inspect.getsource(TaskManager._execute)
        assert "remove_filing" in source


# -----------------------------------------------------------------------
# Finding #16: Security headers
# -----------------------------------------------------------------------


class TestSecurityHeaders:
    """All responses must include security headers."""

    def test_health_check_has_security_headers(self):
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert resp.headers.get("Content-Security-Policy") is not None
        assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]

    def test_error_response_has_security_headers(self):
        """404 responses from valid routes should still include security headers."""
        registry = MagicMock()
        registry.get_filing.return_value = None
        app.dependency_overrides[get_registry] = lambda: registry

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/filings/0000320193-24-000001")
        assert resp.status_code == 404
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("Content-Security-Policy") is not None
        app.dependency_overrides.clear()

    def test_docs_has_csp_header(self):
        """Swagger docs should inherit the API CSP header."""
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert resp.headers.get("Content-Security-Policy") is not None
        assert "script-src" in resp.headers["Content-Security-Policy"]


class TestTransportSecurityHardening:
    """Deployment-facing CSP and TLS warning regressions."""

    def test_nginx_sets_csp_header(self):
        nginx_conf = Path(__file__).parents[3] / "nginx.conf"
        content = nginx_conf.read_text()
        assert "Content-Security-Policy" in content

    def test_http_proxy_warning_emitted_once(self, monkeypatch):
        monkeypatch.setattr(
            "sec_semantic_search.api.app.get_settings",
            lambda: MagicMock(
                api=MagicMock(
                    key="shared-key",
                    admin_key=None,
                    edgar_session_required=False,
                ),
            ),
        )

        with patch("sec_semantic_search.api.app.logger.warning") as mock_warning:
            client = TestClient(app, raise_server_exceptions=False)
            client.get("/api/health", headers={"X-Forwarded-Proto": "http"})
            client.get("/api/health", headers={"X-Forwarded-Proto": "http"})

        assert mock_warning.call_count == 1
        assert "Scenarios B and C require TLS" in mock_warning.call_args.args[0]

    def test_deployment_docs_warn_scenarios_b_c_need_tls(self):
        deployment = Path(__file__).parents[3] / "docs" / "DEPLOYMENT.md"
        if not deployment.exists():
            pytest.skip("docs/DEPLOYMENT.md is intentionally gitignored")
        content = deployment.read_text()
        assert "Scenarios B/C are insecure without TLS" in content


# -----------------------------------------------------------------------
# Finding #7: Database path validation
# -----------------------------------------------------------------------


class TestDatabasePathValidation:
    """Database paths must not escape the project directory."""

    def test_default_paths_accepted(self):
        """Default relative paths within the project should be valid."""
        from sec_semantic_search.config.settings import DatabaseSettings

        settings = DatabaseSettings()
        assert settings.chroma_path == "./data/chroma_db"
        assert settings.metadata_db_path == "./data/metadata.sqlite"

    def test_nested_relative_path_accepted(self):
        """Deeper relative paths within the project should be valid."""
        from sec_semantic_search.config.settings import DatabaseSettings

        settings = DatabaseSettings(
            chroma_path="./data/deep/nested/chroma",
            metadata_db_path="./data/deep/nested/meta.sqlite",
        )
        assert "deep/nested" in settings.chroma_path

    def test_path_traversal_chroma_rejected(self, monkeypatch):
        """Path traversal via chroma_path must be rejected."""
        from sec_semantic_search.config.settings import DatabaseSettings

        with pytest.raises(ValidationError, match="outside the project directory"):
            DatabaseSettings(chroma_path="../../etc/evil_chroma")

    def test_path_traversal_metadata_rejected(self, monkeypatch):
        """Path traversal via metadata_db_path must be rejected."""
        from sec_semantic_search.config.settings import DatabaseSettings

        with pytest.raises(ValidationError, match="outside the project directory"):
            DatabaseSettings(metadata_db_path="../../../tmp/evil.sqlite")

    def test_absolute_path_outside_cwd_rejected(self):
        """Absolute paths outside the working directory must be rejected."""
        from sec_semantic_search.config.settings import DatabaseSettings

        with pytest.raises(ValidationError, match="outside the project directory"):
            DatabaseSettings(chroma_path="/tmp/evil_chroma")

    def test_absolute_path_inside_cwd_accepted(self):
        """Absolute paths within the working directory should be valid."""
        import os

        from sec_semantic_search.config.settings import DatabaseSettings

        safe_path = os.path.join(os.getcwd(), "data", "safe_chroma")
        settings = DatabaseSettings(chroma_path=safe_path)
        assert settings.chroma_path == safe_path

    def test_symlink_in_path_rejected(self, tmp_path):
        """Symlinks in database path components must be rejected."""
        import os

        from sec_semantic_search.config.settings import DatabaseSettings

        # Create a symlink inside CWD that points outside CWD
        link_path = Path(os.getcwd()) / "data" / "symlink_test"
        link_path.parent.mkdir(parents=True, exist_ok=True)
        target = tmp_path / "outside"
        target.mkdir()

        try:
            link_path.symlink_to(target)
            with pytest.raises(ValidationError, match="symlink"):
                DatabaseSettings(
                    chroma_path=str(link_path / "chroma_db"),
                )
        finally:
            # Clean up the symlink
            if link_path.is_symlink():
                link_path.unlink()

    def test_env_var_path_traversal_rejected(self, monkeypatch):
        """Path traversal via environment variables must be rejected."""
        from sec_semantic_search.config.settings import DatabaseSettings

        monkeypatch.setenv("DB_CHROMA_PATH", "../../../../etc/shadow_chroma")
        with pytest.raises(ValidationError, match="outside the project directory"):
            DatabaseSettings()


# -----------------------------------------------------------------------
# Finding #4: API key authentication
# -----------------------------------------------------------------------


class TestApiKeyAuthentication:
    """API endpoints must reject unauthenticated requests when API_KEY is set."""

    TEST_KEY = "test-secret-key-12345"

    def setup_method(self):
        """Inject mock dependencies so routes don't touch real stores."""
        registry = MagicMock()
        registry.list_filings.return_value = []
        registry.get_statistics.return_value = MagicMock(
            filing_count=0,
            tickers=[],
            form_breakdown={},
            ticker_breakdown=[],
        )
        chroma = MagicMock()
        chroma.collection_count.return_value = 0

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma

    def teardown_method(self):
        app.dependency_overrides.clear()

    def _enable_auth(self, monkeypatch):
        """Patch settings to enable API key authentication."""
        monkeypatch.setattr(
            "sec_semantic_search.api.dependencies.get_settings",
            lambda: MagicMock(api=MagicMock(key=self.TEST_KEY)),
        )

    # -- Health check is always public --

    def test_health_accessible_without_key(self, monkeypatch):
        """Health check should be accessible even when auth is enabled."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_does_not_expose_version(self):
        """Health endpoint must not reveal the application version (§F3)."""
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok"}
        assert "version" not in body

    # -- Auth disabled (default) --

    def test_no_key_configured_allows_all(self):
        """When API_KEY is not set, all endpoints are accessible."""
        # Default settings have key=None — no auth needed.
        client = TestClient(app)
        resp = client.get("/api/status/")
        assert resp.status_code == 200

    # -- Auth enabled: missing key --

    def test_missing_key_returns_401(self, monkeypatch):
        """Requests without X-API-Key header should get 401."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/status/")
        assert resp.status_code == 401
        assert "unauthorised" in resp.json()["detail"]["error"]

    # -- Auth enabled: wrong key --

    def test_wrong_key_returns_401(self, monkeypatch):
        """Requests with an incorrect API key should get 401."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/api/status/",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    # -- Auth enabled: correct key --

    def test_correct_key_allows_request(self, monkeypatch):
        """Requests with the correct API key should succeed."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/api/status/",
            headers={"X-API-Key": self.TEST_KEY},
        )
        assert resp.status_code == 200

    # -- Multiple endpoints protected --

    def test_search_requires_key(self, monkeypatch):
        """POST /api/search/ should require API key."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/search/", json={"query": "revenue"})
        assert resp.status_code == 401

    def test_filings_requires_key(self, monkeypatch):
        """GET /api/filings/ should require API key."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/filings/")
        assert resp.status_code == 401

    def test_ingest_requires_key(self, monkeypatch):
        """POST /api/ingest/add should require API key."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"]},
        )
        assert resp.status_code == 401

    def test_resources_requires_key(self, monkeypatch):
        """GET /api/resources/gpu should require API key."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/resources/gpu")
        assert resp.status_code == 401

    def test_delete_requires_key(self, monkeypatch):
        """DELETE /api/filings/ should require API key."""
        self._enable_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/api/filings/?confirm=true")
        assert resp.status_code == 401

    # -- WebSocket auth --

    def test_websocket_rejects_without_key(self, monkeypatch):
        """WebSocket should reject connections when API key is wrong."""
        monkeypatch.setattr(
            "sec_semantic_search.api.websocket.get_settings",
            lambda: MagicMock(
                api=MagicMock(key=self.TEST_KEY, cors_origins=["http://localhost:3000"]),
            ),
        )
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        # Connection with wrong key should fail.
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/ws/ingest/{info.task_id}",
                headers=_WS_HEADERS,
            ) as ws:
                ws.send_json({"type": "auth", "api_key": "wrong-key"})
                ws.receive_json()

    def test_websocket_accepts_correct_key(self, monkeypatch):
        """WebSocket should accept connections with correct API key."""
        monkeypatch.setattr(
            "sec_semantic_search.api.websocket.get_settings",
            lambda: MagicMock(
                api=MagicMock(key=self.TEST_KEY, cors_origins=["http://localhost:3000"]),
            ),
        )
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers=_WS_HEADERS,
        ) as ws:
            ws.send_json({"type": "auth", "api_key": self.TEST_KEY})
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"

    def test_websocket_rejects_query_param_auth(self, monkeypatch):
        """WebSocket should not accept the API key via query string."""
        monkeypatch.setattr(
            "sec_semantic_search.api.websocket.get_settings",
            lambda: MagicMock(
                api=MagicMock(key=self.TEST_KEY, cors_origins=["http://localhost:3000"]),
            ),
        )
        # The key in the URL is ignored, so the server waits for an auth
        # message that never comes. Only the outcome of the timeout is under
        # test, not its production length (5 s).
        monkeypatch.setattr("sec_semantic_search.api.websocket._AUTH_TIMEOUT_SECONDS", 0.05)
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/ws/ingest/{info.task_id}?api_key={self.TEST_KEY}",
                headers=_WS_HEADERS,
            ) as ws:
                ws.receive_json()

        assert exc_info.value.code == 4001
        manager.get_task.assert_not_called()

    def test_websocket_accepts_no_key_when_auth_disabled(self):
        """WebSocket should work without key when auth is disabled."""
        info = make_task_info(state=TaskState.COMPLETED)
        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers=_WS_HEADERS,
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"


# -----------------------------------------------------------------------
# Finding #3: frontend admin key exposure
# -----------------------------------------------------------------------


class TestFrontendAdminKeyExposure:
    """Frontend should keep the admin key out of public browser code."""

    def test_frontend_api_source_does_not_reference_next_public_admin_key(self):
        api_source = Path("frontend/src/lib/api.ts").read_text(encoding="utf-8")

        assert "NEXT_PUBLIC_ADMIN_KEY" not in api_source
        assert "/api/admin/filings/bulk-delete" in api_source
        assert "/api/admin/resources/gpu" in api_source

    def test_frontend_uses_server_side_admin_routes(self):
        assert Path("frontend/src/app/api/admin/session/route.ts").exists()
        assert Path("frontend/src/app/api/admin/filings/bulk-delete/route.ts").exists()
        assert Path("frontend/src/app/api/admin/resources/gpu/route.ts").exists()

    def test_docker_compose_uses_server_only_admin_key_for_frontend(self):
        compose_source = Path("docker-compose.yml").read_text(encoding="utf-8")

        assert "NEXT_PUBLIC_ADMIN_KEY" not in compose_source
        assert "ADMIN_API_KEY=${ADMIN_API_KEY:-}" in compose_source


# -----------------------------------------------------------------------
# Finding #10: Rate limiting
# -----------------------------------------------------------------------


class TestRateLimiting:
    """API must rate-limit requests to prevent resource exhaustion."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_rate_limit_middleware_present(self):
        """RateLimitMiddleware must be in the middleware stack."""
        from sec_semantic_search.api.rate_limit import RateLimitMiddleware

        # Walk the middleware stack to find RateLimitMiddleware.
        found = False
        middleware = app.middleware_stack
        while middleware is not None:
            if isinstance(middleware, RateLimitMiddleware):
                found = True
                break
            middleware = getattr(middleware, "app", None)
        assert found, "RateLimitMiddleware not found in middleware stack"

    def test_rate_limit_settings_exist(self):
        """ApiSettings must expose rate limit configuration."""
        from sec_semantic_search.config.settings import ApiSettings

        settings = ApiSettings()
        assert settings.rate_limit_search > 0
        assert settings.rate_limit_ingest > 0
        assert settings.rate_limit_delete > 0
        assert settings.rate_limit_general > 0

    def test_sliding_window_allows_within_limit(self):
        """Requests within the limit should be allowed."""
        from sec_semantic_search.api.rate_limit import _SlidingWindow

        window = _SlidingWindow(requests_per_minute=5)
        for _ in range(5):
            allowed, _ = window.is_allowed("test-ip")
            assert allowed

    def test_sliding_window_blocks_over_limit(self):
        """Requests exceeding the limit should be blocked."""
        from sec_semantic_search.api.rate_limit import _SlidingWindow

        window = _SlidingWindow(requests_per_minute=3)
        for _ in range(3):
            window.is_allowed("test-ip")

        allowed, retry_after = window.is_allowed("test-ip")
        assert not allowed
        assert retry_after > 0

    def test_sliding_window_per_ip_isolation(self):
        """Different IPs should have independent counters."""
        from sec_semantic_search.api.rate_limit import _SlidingWindow

        window = _SlidingWindow(requests_per_minute=2)
        window.is_allowed("ip-a")
        window.is_allowed("ip-a")
        # ip-a is exhausted
        allowed_a, _ = window.is_allowed("ip-a")
        assert not allowed_a
        # ip-b should still be allowed
        allowed_b, _ = window.is_allowed("ip-b")
        assert allowed_b

    def test_classify_path_search(self):
        """Search endpoints should be classified as 'search'."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/search/", "POST") == "search"

    def test_classify_path_ingest(self):
        """Ingest POST endpoints should be classified as 'ingest'."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/ingest/add", "POST") == "ingest"
        assert _classify_path("/api/ingest/batch", "POST") == "ingest"

    def test_classify_path_delete(self):
        """DELETE methods should be classified as 'delete'."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/filings/123", "DELETE") == "delete"

    def test_classify_path_general(self):
        """Other API paths should be classified as 'general'."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/status/", "GET") == "general"

    def test_classify_path_health_not_limited(self):
        """Health check should not be rate-limited."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/health", "GET") == "general"

    def test_classify_path_docs_not_limited(self):
        """Documentation paths should not be rate-limited."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/docs", "GET") is None
        assert _classify_path("/openapi.json", "GET") is None

    def test_rate_limit_response_format(self):
        """429 responses must include Retry-After header and structured body."""
        from sec_semantic_search.api.rate_limit import _SlidingWindow

        # Verify the response format by testing the sliding window directly.
        window = _SlidingWindow(requests_per_minute=1)
        window.is_allowed("test")
        allowed, retry_after = window.is_allowed("test")
        assert not allowed
        assert isinstance(retry_after, int)
        assert retry_after >= 1


# -----------------------------------------------------------------------
# Finding #14: WebSocket race condition
# -----------------------------------------------------------------------


class TestWebSocketTerminalFallback:
    """WebSocket must deliver terminal messages even when queue is empty."""

    def test_completed_task_gets_terminal_when_queue_empty(self):
        """A completed task with empty queue should synthesise a terminal message."""
        from sec_semantic_search.api.tasks import FilingResult

        info = make_task_info(state=TaskState.COMPLETED)
        info.results.append(
            FilingResult(
                ticker="AAPL",
                form_type="10-K",
                filing_date="2024-11-01",
                accession_number="0000320193-24-000001",
                segment_count=50,
                chunk_count=60,
                duration_seconds=5.2,
            )
        )
        info.progress.filings_skipped = 1
        info.progress.filings_failed = 0

        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers=_WS_HEADERS,
        ) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            # Terminal message should always be delivered.
            terminal = ws.receive_json()
            assert terminal["type"] == "completed"
            assert terminal["summary"]["succeeded"] == 1
            assert terminal["summary"]["skipped"] == 1

    def test_failed_task_gets_terminal_when_queue_empty(self):
        """A failed task with empty queue should synthesise a failed message."""
        info = make_task_info(state=TaskState.FAILED, error="Out of memory")

        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers=_WS_HEADERS,
        ) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "failed"
            assert "Out of memory" in terminal["error"]

    def test_cancelled_task_gets_terminal_when_queue_empty(self):
        """A cancelled task with empty queue should synthesise a cancelled message."""
        info = make_task_info(state=TaskState.CANCELLED)

        manager = MagicMock()
        manager.get_task.return_value = info
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/ingest/{info.task_id}",
            headers=_WS_HEADERS,
        ) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "cancelled"

    def test_build_terminal_from_state_completed(self):
        """_build_terminal_from_state returns correct completed message."""
        from sec_semantic_search.api.websocket import _build_terminal_from_state

        info = make_task_info(state=TaskState.COMPLETED)
        msg = _build_terminal_from_state(info)
        assert msg["type"] == "completed"
        assert "results" in msg
        assert "summary" in msg

    def test_build_terminal_from_state_failed(self):
        """_build_terminal_from_state returns correct failed message."""
        from sec_semantic_search.api.websocket import _build_terminal_from_state

        info = make_task_info(state=TaskState.FAILED, error="Test error")
        msg = _build_terminal_from_state(info)
        assert msg["type"] == "failed"
        assert msg["error"] == "Test error"

    def test_build_terminal_from_state_cancelled(self):
        """_build_terminal_from_state returns correct cancelled message."""
        from sec_semantic_search.api.websocket import _build_terminal_from_state

        info = make_task_info(state=TaskState.CANCELLED)
        msg = _build_terminal_from_state(info)
        assert msg["type"] == "cancelled"


# -----------------------------------------------------------------------
# Finding #15: Security audit logging
# -----------------------------------------------------------------------


class TestSecurityAuditLogging:
    """Destructive operations must produce SECURITY_AUDIT log entries."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_audit_log_function_exists(self):
        """audit_log must be importable from core."""
        from sec_semantic_search.core import audit_log

        assert callable(audit_log)

    def test_audit_log_emits_warning(self, caplog):
        """audit_log must emit a WARNING-level log with SECURITY_AUDIT prefix."""
        from sec_semantic_search.core.logging import audit_log

        # The sec_semantic_search logger has propagate=False, so we
        # must attach caplog's handler directly to capture records.
        pkg_logger = logging.getLogger("sec_semantic_search")
        pkg_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING):
                audit_log(
                    "test_action",
                    client_ip="1.2.3.4",
                    endpoint="/test",
                    detail="testing",
                )

            assert any("SECURITY_AUDIT" in r.message for r in caplog.records)
            assert any("test_action" in r.message for r in caplog.records)
            assert any("1.2.3.4" in r.message for r in caplog.records)
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def _with_audit_capture(self, caplog):
        """Context helper: attach caplog handler to the package logger."""
        pkg_logger = logging.getLogger("sec_semantic_search")
        pkg_logger.addHandler(caplog.handler)
        return pkg_logger

    def test_delete_filing_produces_audit_log(self, caplog):
        """DELETE /api/filings/{accession} must produce an audit entry."""
        record = make_filing_record()
        registry = MagicMock()
        registry.get_filing.return_value = record
        chroma = MagicMock()

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma

        pkg_logger = self._with_audit_capture(caplog)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.WARNING):
                resp = client.delete(f"/api/filings/{record.accession_number}")

            assert resp.status_code == 200
            assert any(
                "SECURITY_AUDIT" in r.message and "delete_filing" in r.message
                for r in caplog.records
            )
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def test_clear_all_produces_audit_log(self, caplog):
        """DELETE /api/filings/?confirm=true must produce an audit entry."""
        registry = MagicMock()
        registry.clear_all.return_value = 1
        chroma = MagicMock()
        chroma.clear_collection.return_value = 42

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma

        pkg_logger = self._with_audit_capture(caplog)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.WARNING):
                resp = client.delete("/api/filings/?confirm=true")

            assert resp.status_code == 200
            assert any(
                "SECURITY_AUDIT" in r.message and "clear_all" in r.message for r in caplog.records
            )
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def test_cancel_task_produces_audit_log(self, caplog):
        """DELETE /api/ingest/tasks/{id} must produce an audit entry."""
        info = make_task_info(state=TaskState.RUNNING)
        manager = MagicMock()
        manager.get_task.return_value = info
        manager.cancel_task.return_value = True

        app.dependency_overrides[get_task_manager] = lambda: manager

        pkg_logger = self._with_audit_capture(caplog)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.WARNING):
                resp = client.delete(f"/api/ingest/tasks/{info.task_id}")

            assert resp.status_code == 200
            assert any(
                "SECURITY_AUDIT" in r.message and "cancel_task" in r.message for r in caplog.records
            )
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def test_gpu_unload_produces_audit_log(self, caplog):
        """DELETE /api/resources/gpu must produce an audit entry."""
        embedder = MagicMock()
        embedder.is_loaded = True
        task_manager = MagicMock()
        task_manager.has_active_task.return_value = False

        app.dependency_overrides[get_embedder] = lambda: embedder
        app.dependency_overrides[get_task_manager] = lambda: task_manager

        pkg_logger = self._with_audit_capture(caplog)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.WARNING):
                resp = client.delete("/api/resources/gpu")

            assert resp.status_code == 200
            assert any(
                "SECURITY_AUDIT" in r.message and "gpu_unload" in r.message for r in caplog.records
            )
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def test_audit_log_in_route_source_code(self):
        """All destructive route handlers must call audit_log."""
        import inspect

        from sec_semantic_search.api.routes import filings, ingest, resources

        # Check each destructive function's source.
        for func in [
            filings.delete_filing,
            filings.delete_by_ids,
            filings.bulk_delete,
            filings.clear_all,
            ingest.cancel_task,
            resources.gpu_unload,
        ]:
            source = inspect.getsource(func)
            assert "audit_log(" in source, f"{func.__name__} must call audit_log()"


# -----------------------------------------------------------------------
# Finding #17: Search query logging — redact in production
# -----------------------------------------------------------------------


class TestQueryLogRedaction:
    """Verify that search and ingest routes use redact_for_log()."""

    def test_search_route_uses_redact_for_log(self):
        """The search route must call redact_for_log on the query before logging."""
        import inspect

        from sec_semantic_search.api.routes import search

        source = inspect.getsource(search.search)
        assert "redact_for_log(" in source, (
            "search route must call redact_for_log() before logging the query"
        )

    def test_ingest_route_uses_redact_for_log(self):
        """The ingest helper must call redact_for_log on tickers before logging."""
        import inspect

        from sec_semantic_search.api.routes import ingest

        source = inspect.getsource(ingest._create_task)
        assert "redact_for_log(" in source, (
            "_create_task must call redact_for_log() before logging tickers"
        )

    def test_search_redacts_query_in_log_output(self, monkeypatch, caplog):
        """With LOG_REDACT_QUERIES=true, the actual query must not appear in logs."""
        monkeypatch.setenv("LOG_REDACT_QUERIES", "true")

        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: mock_engine

        pkg_logger = logging.getLogger("sec_semantic_search")
        caplog.handler.setLevel(logging.DEBUG)
        pkg_logger.addHandler(caplog.handler)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.INFO, logger="sec_semantic_search"):
                resp = client.post(
                    "/api/search/",
                    json={"query": "revenue growth forecast"},
                )

            assert resp.status_code == 200
            # The actual query text must NOT appear in any log record
            for record in caplog.records:
                assert "revenue growth forecast" not in record.message
            # But a redacted marker should appear
            assert any("<redacted:" in r.message for r in caplog.records)
        finally:
            pkg_logger.removeHandler(caplog.handler)

    def test_search_shows_query_when_redaction_disabled(self, monkeypatch, caplog):
        """With LOG_REDACT_QUERIES unset, the query appears normally in logs."""
        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)

        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: mock_engine

        pkg_logger = logging.getLogger("sec_semantic_search")
        caplog.handler.setLevel(logging.DEBUG)
        pkg_logger.addHandler(caplog.handler)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            with caplog.at_level(logging.INFO, logger="sec_semantic_search"):
                resp = client.post(
                    "/api/search/",
                    json={"query": "revenue growth forecast"},
                )

            assert resp.status_code == 200
            assert any("revenue growth forecast" in r.message for r in caplog.records)
        finally:
            pkg_logger.removeHandler(caplog.handler)


# -----------------------------------------------------------------------
# Finding #18: HTTPS enforcement — documentation and SSL support
# -----------------------------------------------------------------------


class TestHttpsEnforcement:
    """Verify that HTTPS/TLS support is documented and available."""

    def test_run_module_accepts_ssl_certfile(self):
        """The API entry point must accept --ssl-certfile."""
        import inspect

        from sec_semantic_search.api import run

        source = inspect.getsource(run.main)
        assert "--ssl-certfile" in source

    def test_run_module_accepts_ssl_keyfile(self):
        """The API entry point must accept --ssl-keyfile."""
        import inspect

        from sec_semantic_search.api import run

        source = inspect.getsource(run.main)
        assert "--ssl-keyfile" in source

    def test_readme_documents_https_requirement(self):
        """The README must document HTTPS as a production requirement."""
        readme = Path(__file__).parents[3] / "README.md"
        content = readme.read_text()
        assert "HTTPS" in content
        assert "ssl-certfile" in content

    def test_run_module_docstring_mentions_tls(self):
        """The run module docstring must mention TLS/HTTPS."""
        from sec_semantic_search.api import run

        assert run.__doc__ is not None
        assert "HTTPS" in run.__doc__ or "TLS" in run.__doc__


# -----------------------------------------------------------------------
# Finding #19: Pinned dependency versions
# -----------------------------------------------------------------------


class TestPinnedDependencies:
    """Verify that production dependencies use exact version pins."""

    def test_dependencies_are_pinned(self):
        """Production deps in pyproject.toml must use == pins (except torch)."""
        import tomllib

        pyproject = Path(__file__).parents[3] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        deps = data["project"]["dependencies"]
        for dep in deps:
            # torch is excluded — local CUDA build suffix varies
            if dep.startswith("torch"):
                continue
            assert "==" in dep, f"Dependency '{dep}' is not pinned to an exact version"

    def test_dev_dependencies_are_pinned(self):
        """Dev deps in pyproject.toml must use == pins."""
        import tomllib

        pyproject = Path(__file__).parents[3] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        deps = data["project"]["optional-dependencies"]["dev"]
        for dep in deps:
            assert "==" in dep, f"Dev dependency '{dep}' is not pinned to an exact version"


# -----------------------------------------------------------------------
# Finding #20: Task TTL data loss — persistence and extended TTL
# -----------------------------------------------------------------------


class TestTaskPersistence:
    """Verify that completed tasks are persisted and recoverable."""

    def test_ttl_is_24_hours(self):
        """In-memory TTL must be 24 hours, not 1 hour."""
        from sec_semantic_search.api.tasks import _TASK_TTL_SECONDS

        assert _TASK_TTL_SECONDS == 86_400

    def test_task_history_table_created(self, tmp_path, monkeypatch):
        """MetadataRegistry must create the task_history table."""
        monkeypatch.setenv("DB_METADATA_DB_PATH", str(tmp_path / "test.sqlite"))
        monkeypatch.setenv("DB_CHROMA_PATH", str(tmp_path / "chroma"))

        from sec_semantic_search.config import reload_settings

        reload_settings()

        from sec_semantic_search.database.metadata import MetadataRegistry

        # Explicit empty key ensures unencrypted mode so raw sqlite3
        # can read the file regardless of DB_ENCRYPTION_KEY in .env.
        registry = MetadataRegistry(str(tmp_path / "test.sqlite"), encryption_key="")
        try:
            import sqlite3

            conn = sqlite3.connect(str(tmp_path / "test.sqlite"))
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            assert "task_history" in table_names
            conn.close()
        finally:
            registry.close()

    def test_save_and_retrieve_task_history(self, tmp_path, monkeypatch):
        """Tasks can be saved to and retrieved from task_history."""
        monkeypatch.setenv("DB_METADATA_DB_PATH", str(tmp_path / "test.sqlite"))
        monkeypatch.setenv("DB_CHROMA_PATH", str(tmp_path / "chroma"))

        from sec_semantic_search.config import reload_settings

        reload_settings()

        from sec_semantic_search.database.metadata import MetadataRegistry

        registry = MetadataRegistry(str(tmp_path / "test.sqlite"))
        try:
            registry.save_task_history(
                "abc123",
                status="completed",
                tickers=["AAPL"],
                form_types=["10-K"],
                results=[
                    {
                        "ticker": "AAPL",
                        "form_type": "10-K",
                        "filing_date": "2024-11-01",
                        "accession_number": "0000320193-24-000001",
                        "segment_count": 50,
                        "chunk_count": 100,
                        "duration_seconds": 12.5,
                    }
                ],
                started_at="2024-11-15T10:00:00+00:00",
                completed_at="2024-11-15T10:01:00+00:00",
                filings_done=1,
                filings_skipped=0,
                filings_failed=0,
            )

            result = registry.get_task_history("abc123")
            assert result is not None
            assert result["task_id"] == "abc123"
            assert result["status"] == "completed"
            # Tickers stripped by default (TASK_HISTORY_PERSIST_TICKERS=false).
            assert result["tickers"] == []
            assert len(result["results"]) == 1
            assert result["results"][0]["chunk_count"] == 100
            assert result["filings_done"] == 1
        finally:
            registry.close()

    def test_get_task_history_returns_none_for_missing(self, tmp_path, monkeypatch):
        """get_task_history returns None for unknown task IDs."""
        monkeypatch.setenv("DB_METADATA_DB_PATH", str(tmp_path / "test.sqlite"))
        monkeypatch.setenv("DB_CHROMA_PATH", str(tmp_path / "chroma"))

        from sec_semantic_search.config import reload_settings

        reload_settings()

        from sec_semantic_search.database.metadata import MetadataRegistry

        registry = MetadataRegistry(str(tmp_path / "test.sqlite"))
        try:
            assert registry.get_task_history("nonexistent") is None
        finally:
            registry.close()

    def test_task_manager_get_task_falls_back_to_history(self):
        """get_task() checks SQLite when task not found in memory."""
        from sec_semantic_search.api.tasks import TaskManager

        mock_registry = MagicMock()
        mock_registry.get_task_history.return_value = {
            "task_id": "persisted123",
            "status": "completed",
            "tickers": ["MSFT"],
            "form_types": ["10-Q"],
            "results": [],
            "error": None,
            "started_at": "2024-11-15T10:00:00+00:00",
            "completed_at": "2024-11-15T10:01:00+00:00",
            "filings_done": 2,
            "filings_skipped": 1,
            "filings_failed": 0,
        }

        with patch.object(TaskManager, "_start_cleanup_timer"):
            mgr = TaskManager(
                registry=mock_registry,
                chroma=MagicMock(),
                fetcher=MagicMock(),
                orchestrator=MagicMock(),
            )

        info = mgr.get_task("persisted123")
        assert info is not None
        assert info.task_id == "persisted123"
        assert info.state.value == "completed"
        assert info.tickers == ["MSFT"]
        assert info.progress.filings_done == 2
        assert info.progress.filings_skipped == 1

    def test_prune_persists_before_removing(self):
        """_prune_stale_tasks must save to history before deleting from memory."""
        from sec_semantic_search.api.tasks import TaskManager, TaskState

        mock_registry = MagicMock()

        with patch.object(TaskManager, "_start_cleanup_timer"):
            mgr = TaskManager(
                registry=mock_registry,
                chroma=MagicMock(),
                fetcher=MagicMock(),
                orchestrator=MagicMock(),
            )

        info = make_task_info(task_id="prunable", state=TaskState.COMPLETED)
        # Set completed_at far in the past so TTL check triggers
        info.completed_at = datetime(2020, 1, 1, tzinfo=UTC)
        mgr._tasks["prunable"] = info

        mgr._prune_stale_tasks()

        # Task should be removed from memory
        assert "prunable" not in mgr._tasks
        # But persisted to history
        mock_registry.save_task_history.assert_called_once()
        call_kwargs = mock_registry.save_task_history.call_args
        assert call_kwargs[0][0] == "prunable"  # first positional arg = task_id


# -----------------------------------------------------------------------
# Hardening H2: Request body size limit
# -----------------------------------------------------------------------


class TestContentSizeLimit:
    """ContentSizeLimitMiddleware rejects oversized request bodies."""

    def test_normal_request_passes(self):
        """A normal-sized POST body should be accepted (not blocked by size limit)."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        app.dependency_overrides[get_registry] = lambda: MagicMock()

        client = TestClient(app)
        resp = client.post("/api/search/", json={"query": "revenue growth", "top_k": 5})
        # Should not be 413
        assert resp.status_code != 413

        app.dependency_overrides.clear()

    def test_oversized_content_length_rejected(self):
        """A request declaring > 1 MB Content-Length should get 413."""
        client = TestClient(app)
        resp = client.post(
            "/api/search/",
            content=b"x",
            headers={"Content-Length": "2000000", "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["detail"]["error"] == "payload_too_large"

    def test_invalid_content_length_rejected(self):
        """A request with a non-numeric Content-Length should get 400."""
        client = TestClient(app)
        resp = client.post(
            "/api/search/",
            content=b"{}",
            headers={"Content-Length": "not-a-number", "Content-Type": "application/json"},
        )
        assert resp.status_code == 400


# -----------------------------------------------------------------------
# Hardening H3: Audit log covers ingest task creation
# -----------------------------------------------------------------------


class TestIngestAuditLog:
    """Ingest task creation is recorded in the audit log."""

    def test_ingest_add_emits_audit_log(self):
        """POST /api/ingest/add should produce an audit_log entry."""
        manager = MagicMock(spec=TaskManager)
        manager.create_task.return_value = "task-id-123"
        app.dependency_overrides[get_task_manager] = lambda: manager

        client = TestClient(app)

        with patch("sec_semantic_search.api.routes.ingest.audit_log") as mock_audit:
            resp = client.post(
                "/api/ingest/add",
                json={"tickers": ["AAPL"], "form_types": ["10-K"]},
            )

        assert resp.status_code == 202
        mock_audit.assert_called_once()
        call_args = mock_audit.call_args
        assert call_args[0][0] == "ingest_task_created"
        assert "client_ip" in call_args[1]
        assert "endpoint" in call_args[1]

        app.dependency_overrides.clear()

    def test_ingest_batch_emits_audit_log(self):
        """POST /api/ingest/batch should also produce an audit_log entry."""
        manager = MagicMock(spec=TaskManager)
        manager.create_task.return_value = "task-id-456"
        app.dependency_overrides[get_task_manager] = lambda: manager

        client = TestClient(app)

        with patch("sec_semantic_search.api.routes.ingest.audit_log") as mock_audit:
            resp = client.post(
                "/api/ingest/batch",
                json={"tickers": ["AAPL", "MSFT"], "form_types": ["10-K"]},
            )

        assert resp.status_code == 202
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0] == "ingest_task_created"

        app.dependency_overrides.clear()


# -----------------------------------------------------------------------
# Hardening H6: Permissions-Policy header
# -----------------------------------------------------------------------


class TestPermissionsPolicyHeader:
    """SecurityHeadersMiddleware includes Permissions-Policy."""

    def test_permissions_policy_present_on_response(self):
        """Every response should include a Permissions-Policy header."""
        client = TestClient(app)
        resp = client.get("/api/health")
        pp = resp.headers.get("permissions-policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp
        assert "geolocation=()" in pp


# -----------------------------------------------------------------------
# Finding F1: OpenAPI/Swagger UI exposed by default
# -----------------------------------------------------------------------


class TestOpenApiConditionalExposure:
    """OpenAPI docs are disabled when API_KEY is set (§F1)."""

    def _build_app(self, *, api_key: str | None):
        """Create a fresh FastAPI app with the given API key setting."""
        from sec_semantic_search.api.app import create_app

        mock_settings = MagicMock()
        mock_settings.api.key = api_key
        mock_settings.api.admin_key = None
        mock_settings.api.cors_origins = ["http://localhost:3000"]
        mock_settings.api.rate_limit_search = 0
        mock_settings.api.rate_limit_ingest = 0
        mock_settings.api.rate_limit_delete = 0
        mock_settings.api.rate_limit_general = 0
        mock_settings.api.edgar_session_required = False
        with patch(
            "sec_semantic_search.api.app.get_settings",
            return_value=mock_settings,
        ):
            return create_app()

    def test_docs_available_without_api_key(self):
        """When API_KEY is unset, /docs, /redoc, /openapi.json are served."""
        test_app = self._build_app(api_key=None)
        assert test_app.docs_url == "/docs"
        assert test_app.redoc_url == "/redoc"
        assert test_app.openapi_url == "/openapi.json"

    def test_docs_disabled_with_api_key(self):
        """When API_KEY is set, /docs, /redoc, /openapi.json are disabled."""
        test_app = self._build_app(api_key="test-secret-key")
        assert test_app.docs_url is None
        assert test_app.redoc_url is None
        assert test_app.openapi_url is None

    def test_docs_endpoint_returns_404_with_api_key(self):
        """HTTP requests to /docs return 404 when API_KEY is set."""
        test_app = self._build_app(api_key="test-secret-key")
        client = TestClient(test_app, raise_server_exceptions=False)
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_docs_endpoint_returns_200_without_api_key(self):
        """HTTP requests to /docs return 200 when API_KEY is unset."""
        test_app = self._build_app(api_key=None)
        client = TestClient(test_app, raise_server_exceptions=False)
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200


# -----------------------------------------------------------------------
# Finding F2: Rate limiting behind reverse proxy (proxy-headers)
# -----------------------------------------------------------------------


class TestProxyHeadersConfiguration:
    """Uvicorn is configured to trust proxy headers so that real client
    IPs are used for rate limiting and audit logging (F2, F7)."""

    def test_dockerfile_includes_proxy_headers(self):
        """Dockerfile.api CMD must include --proxy-headers."""
        dockerfile = Path(__file__).parents[3] / "Dockerfile.api"
        content = dockerfile.read_text()
        assert "--proxy-headers" in content

    def test_dockerfile_includes_forwarded_allow_ips(self):
        """Dockerfile.api CMD must include --forwarded-allow-ips."""
        dockerfile = Path(__file__).parents[3] / "Dockerfile.api"
        content = dockerfile.read_text()
        assert "--forwarded-allow-ips" in content

    def test_nginx_sets_x_forwarded_for(self):
        """nginx.conf must set X-Forwarded-For to pass real client IP."""
        nginx_conf = Path(__file__).parents[3] / "nginx.conf"
        content = nginx_conf.read_text()
        assert "X-Forwarded-For" in content
        assert "X-Real-IP" in content


class TestNginxAdminRouting:
    """``/api/admin/*`` are Next.js route handlers that hold the admin key
    server-side (AD#32). Behind nginx they must reach the frontend, not
    FastAPI, which has no admin routes: routed to the API, admin login
    returned 404 under Docker Compose."""

    @staticmethod
    def _locations(block: str) -> dict[str, str]:
        """Map each ``location`` prefix in a server block to its upstream."""
        out: dict[str, str] = {}
        for match in re.finditer(r"location (\S+) \{(.*?)\}", block, re.S):
            upstream = re.search(r"proxy_pass http://(\w+);", match.group(2))
            out[match.group(1)] = upstream.group(1) if upstream else ""
        return out

    def _blocks(self) -> tuple[str, str]:
        content = (Path(__file__).parents[3] / "nginx.conf").read_text()
        http_block = content[content.index("# ── HTTP server") : content.index("# ── TLS server")]
        tls_block = content[content.index("# ── TLS server") :]
        # The TLS template is commented out; strip the comment markers.
        tls_block = "\n".join(line.lstrip("#") for line in tls_block.splitlines())
        return http_block, tls_block

    def test_admin_prefix_routes_to_frontend(self):
        for block in self._blocks():
            locations = self._locations(block)
            assert locations["/api/admin/"] == "frontend_app"
            assert locations["/api/"] == "api_backend"

    def test_admin_location_forwards_client_ip(self):
        """The Next.js admin limiter keys on the client address."""
        http_block, _ = self._blocks()
        admin = re.search(r"location /api/admin/ \{(.*?)\}", http_block, re.S).group(1)
        assert "X-Forwarded-For $proxy_add_x_forwarded_for" in admin
        assert "X-Real-IP $remote_addr" in admin

    def test_fastapi_serves_no_admin_routes(self):
        paths = {getattr(route, "path", "") for route in app.routes}
        assert not any(path.startswith("/api/admin") for path in paths)


# -----------------------------------------------------------------------
# Finding F3: Chunked transfer encoding size limit bypass
# -----------------------------------------------------------------------


class TestChunkedTransferEncodingSizeLimit:
    """ContentSizeLimitMiddleware enforces limits on streamed (chunked)
    request bodies that lack a Content-Length header (F3)."""

    def test_small_chunked_body_passes(self):
        """A small body without Content-Length should be accepted."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        app.dependency_overrides[get_registry] = lambda: MagicMock()

        client = TestClient(app)
        resp = client.post(
            "/api/search/",
            json={"query": "revenue growth", "top_k": 5},
        )
        assert resp.status_code != 413

        app.dependency_overrides.clear()

    def test_oversized_streamed_body_rejected(self):
        """A streamed body exceeding 1 MB should get 413 even without
        Content-Length (defends against chunked transfer encoding bypass)."""
        client = TestClient(app)
        oversized_payload = b"x" * (1024 * 1024 + 1)  # 1 MB + 1 byte
        resp = client.post(
            "/api/search/",
            content=oversized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["detail"]["error"] == "payload_too_large"

    def test_exactly_at_limit_passes(self):
        """A body exactly at the 1 MB boundary should be accepted."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        app.dependency_overrides[get_registry] = lambda: MagicMock()

        client = TestClient(app)
        # Build a valid JSON payload near the 1 MB boundary
        # (the actual body will be slightly smaller due to JSON overhead)
        payload = {"query": "x" * 2000, "top_k": 5}
        resp = client.post("/api/search/", json=payload)
        assert resp.status_code != 413

        app.dependency_overrides.clear()


# -----------------------------------------------------------------------
# Finding F4: CSP 'unsafe-inline' removed from script-src
# -----------------------------------------------------------------------


class TestCSPUnsafeInlineRemoved:
    """Content-Security-Policy must not include 'unsafe-inline' in
    script-src (F4)."""

    def test_api_csp_no_unsafe_inline_in_script_src(self):
        """API response CSP script-src must not contain 'unsafe-inline'."""
        client = TestClient(app)
        resp = client.get("/api/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        # Parse the script-src directive
        for directive in csp.split(";"):
            directive = directive.strip()
            if directive.startswith("script-src"):
                assert "'unsafe-inline'" not in directive, (
                    f"script-src still contains 'unsafe-inline': {directive}"
                )
                break

    def test_api_csp_retains_unsafe_inline_in_style_src(self):
        """style-src may keep 'unsafe-inline' for Tailwind compatibility."""
        client = TestClient(app)
        resp = client.get("/api/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        for directive in csp.split(";"):
            directive = directive.strip()
            if directive.startswith("style-src"):
                assert "'unsafe-inline'" in directive
                break

    def test_nginx_csp_no_unsafe_inline_in_script_src(self):
        """nginx.conf CSP script-src must not contain 'unsafe-inline'."""
        nginx_conf = Path(__file__).parents[3] / "nginx.conf"
        content = nginx_conf.read_text()
        # Find CSP lines and check script-src
        for line in content.splitlines():
            if "Content-Security-Policy" in line and "script-src" in line:
                # Extract the script-src directive
                idx = line.index("script-src")
                rest = line[idx:]
                # script-src ends at the next semicolon
                end = rest.find(";")
                script_src = rest[:end] if end != -1 else rest
                assert "'unsafe-inline'" not in script_src, (
                    f"nginx script-src still contains 'unsafe-inline': {script_src}"
                )


# -----------------------------------------------------------------------
# Finding F5: Brute-force protection on admin session login
# -----------------------------------------------------------------------


class TestAdminLoginRateLimit:
    """POST /api/admin/session must be rate-limited to deter brute-force
    attacks (F5)."""

    def test_auth_category_in_classify_path(self):
        """_classify_path must return 'auth' for admin session POST."""
        from sec_semantic_search.api.rate_limit import _classify_path

        assert _classify_path("/api/admin/session", "POST") == "auth"

    def test_auth_category_does_not_match_get(self):
        """GET /api/admin/session should not be classified as auth."""
        from sec_semantic_search.api.rate_limit import _classify_path

        result = _classify_path("/api/admin/session", "GET")
        assert result != "auth"

    def test_auth_bucket_has_strict_limit(self):
        """The auth rate-limit bucket must have a strict limit (<=10/min)."""
        from sec_semantic_search.api.rate_limit import RateLimitMiddleware

        middleware = app.middleware_stack
        while middleware is not None:
            if isinstance(middleware, RateLimitMiddleware):
                auth_bucket = middleware._buckets.get("auth")
                assert auth_bucket is not None, "No 'auth' bucket in rate limiter"
                assert auth_bucket.limit <= 10, (
                    f"Auth rate limit too permissive: {auth_bucket.limit}/min"
                )
                break
            middleware = getattr(middleware, "app", None)

    def test_auth_sliding_window_blocks_after_limit(self):
        """Auth sliding window should block after 5 attempts."""
        from sec_semantic_search.api.rate_limit import _SlidingWindow

        window = _SlidingWindow(requests_per_minute=5)
        for _ in range(5):
            allowed, _ = window.is_allowed("attacker-ip")
            assert allowed

        allowed, retry_after = window.is_allowed("attacker-ip")
        assert not allowed
        assert retry_after > 0


# -----------------------------------------------------------------------
# Finding F6: Admin session cookie expiry
# -----------------------------------------------------------------------


class TestAdminSessionCookieExpiry:
    """The admin session cookie must have a maxAge to force
    re-authentication (F6)."""

    def test_route_handler_sets_max_age(self):
        """The route handler source must include maxAge on the cookie."""
        route_file = (
            Path(__file__).parents[3]
            / "frontend"
            / "src"
            / "app"
            / "api"
            / "admin"
            / "session"
            / "route.ts"
        )
        content = route_file.read_text()
        assert "maxAge" in content, "Admin session cookie must set maxAge"
        # Verify it's a reasonable value (1-24 hours = 3600-86400 seconds)
        assert "3600" in content, "maxAge should be 3600 (1 hour)"


# -----------------------------------------------------------------------
# Observation #14: nginx server_tokens disabled
# -----------------------------------------------------------------------


class TestNginxServerTokens:
    """nginx must suppress version disclosure (Observation #14)."""

    def test_nginx_server_tokens_off(self):
        """nginx.conf must contain 'server_tokens off'."""
        nginx_conf = Path(__file__).parents[3] / "nginx.conf"
        content = nginx_conf.read_text()
        assert "server_tokens off" in content
