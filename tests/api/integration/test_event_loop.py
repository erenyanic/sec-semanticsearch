"""
Blocking work must stay off the asyncio event loop (OPTIMIZATIONS.md F-01).

Routes whose bodies call SQLite, ChromaDB or the embedding model are plain
``def`` so FastAPI runs them in the threadpool; the two remaining
``BaseHTTPMiddleware`` classes are pure ASGI. The mocks below record
whether they were called on the event-loop thread — ``asyncio.get_running_loop()``
only succeeds there — which makes the checks deterministic rather than
timing-based. One timing test covers the user-visible property: the health
probe answers while a search is still running.
"""

import asyncio
import inspect
import threading
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from sec_semantic_search.api.app import InsecureTransportWarningMiddleware, app
from sec_semantic_search.api.dependencies import (
    get_chroma,
    get_embedder,
    get_registry,
    get_search_engine,
    get_task_manager,
)
from sec_semantic_search.api.rate_limit import RateLimitMiddleware
from sec_semantic_search.api.routes import filings, ingest, resources, search, status
from sec_semantic_search.database.metadata import DatabaseStatistics, FilingRecord

_WS_HEADERS = {"origin": "http://localhost:3000"}


def _on_event_loop() -> bool:
    """Return True when called from the thread running the event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _recording(calls: list[bool], return_value=None):
    """Side effect that records the calling thread and returns a value."""

    def _side_effect(*args, **kwargs):
        calls.append(_on_event_loop())
        return return_value

    return _side_effect


def _record(accession: str = "0000320193-24-000001") -> FilingRecord:
    return FilingRecord(
        id=1,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2024-11-01",
        accession_number=accession,
        chunk_count=10,
        ingested_at="2024-11-02T00:00:00",
    )


class TestBlockingHandlersAreSync:
    """Structural guard: these handlers must not be coroutine functions."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            search.search,
            filings.list_filings,
            filings.get_filing,
            filings.delete_filing,
            filings.delete_by_ids,
            filings.bulk_delete,
            filings.clear_all,
            status.status,
            resources.gpu_unload,
            ingest.get_task,
            ingest.cancel_task,
        ],
        ids=lambda f: f"{f.__module__.rsplit('.', 1)[-1]}.{f.__name__}",
    )
    def test_handler_is_plain_def(self, endpoint):
        assert not inspect.iscoroutinefunction(endpoint)


class TestHandlersRunInThreadpool:
    """Behavioural guard: the blocking calls happen off the event loop."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_search_runs_off_loop(self):
        calls: list[bool] = []
        engine = MagicMock()
        engine.search.side_effect = _recording(calls, return_value=[])
        app.dependency_overrides[get_search_engine] = lambda: engine

        resp = TestClient(app).post("/api/search/", json={"query": "revenue"})

        assert resp.status_code == 200
        assert calls == [False]

    def test_list_filings_runs_off_loop(self):
        calls: list[bool] = []
        registry = MagicMock()
        registry.list_filings_page.side_effect = _recording(calls, return_value=([_record()], 1))
        app.dependency_overrides[get_registry] = lambda: registry

        resp = TestClient(app).get("/api/filings/")

        assert resp.status_code == 200
        assert calls == [False]

    def test_delete_filing_runs_off_loop(self):
        calls: list[bool] = []
        registry = MagicMock()
        registry.get_filing.return_value = _record()
        registry.remove_filing.side_effect = _recording(calls)
        chroma = MagicMock()
        chroma.delete_filing.side_effect = _recording(calls)
        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma

        resp = TestClient(app).delete("/api/filings/0000320193-24-000001")

        assert resp.status_code == 200
        assert calls == [False, False]

    def test_status_runs_off_loop(self):
        calls: list[bool] = []
        registry = MagicMock()
        registry.get_statistics.side_effect = _recording(
            calls,
            return_value=DatabaseStatistics(
                filing_count=0,
                tickers=[],
                form_breakdown={},
                ticker_breakdown=[],
            ),
        )
        chroma = MagicMock()
        chroma.collection_count.side_effect = _recording(calls, return_value=0)
        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_chroma] = lambda: chroma

        resp = TestClient(app).get("/api/status/")

        assert resp.status_code == 200
        assert calls == [False, False]

    def test_gpu_unload_runs_off_loop(self):
        """Unload waits on the embedder lock, so it must never block the loop."""
        calls: list[bool] = []
        embedder = MagicMock()
        embedder.is_loaded = True
        embedder.unload.side_effect = _recording(calls)
        manager = MagicMock()
        manager.has_active_task.return_value = False
        app.dependency_overrides[get_embedder] = lambda: embedder
        app.dependency_overrides[get_task_manager] = lambda: manager

        resp = TestClient(app).delete("/api/resources/gpu")

        assert resp.status_code == 200
        assert calls == [False]

    def test_get_task_runs_off_loop(self):
        """A pruned task falls back to SQLite history."""
        calls: list[bool] = []
        manager = MagicMock()
        manager.get_task.side_effect = _recording(calls, return_value=None)
        app.dependency_overrides[get_task_manager] = lambda: manager

        resp = TestClient(app).get("/api/ingest/tasks/abc123")

        assert resp.status_code == 404
        assert calls == [False]

    def test_websocket_task_lookup_runs_off_loop(self):
        calls: list[bool] = []
        manager = MagicMock()
        manager.get_task.side_effect = _recording(calls, return_value=None)
        app.state.task_manager = manager

        client = TestClient(app)
        with client.websocket_connect("/ws/ingest/abc123", headers=_WS_HEADERS) as ws:
            assert ws.receive_json()["type"] == "error"

        assert calls == [False]


class TestHealthDuringSearch:
    """The liveness probe must answer while a search is in progress."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @pytest.mark.anyio
    async def test_health_responds_while_search_blocked(self):
        started = threading.Event()
        release = threading.Event()

        def slow_search(**kwargs):
            started.set()
            # Bounded so a regression fails the test instead of hanging it.
            release.wait(timeout=5)
            return []

        engine = MagicMock()
        engine.search.side_effect = slow_search
        app.dependency_overrides[get_search_engine] = lambda: engine

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            search_task = asyncio.create_task(client.post("/api/search/", json={"query": "q"}))
            assert await asyncio.to_thread(started.wait, 5)

            health = await asyncio.wait_for(client.get("/api/health"), timeout=2)

            assert health.status_code == 200
            assert not search_task.done(), "search finished before release — loop was blocked"

            release.set()
            resp = await search_task

        assert resp.status_code == 200


class TestPureAsgiMiddleware:
    """No middleware left on the ``BaseHTTPMiddleware`` slow path."""

    @pytest.mark.parametrize(
        "middleware_cls",
        [RateLimitMiddleware, InsecureTransportWarningMiddleware],
    )
    def test_not_base_http_middleware(self, middleware_cls):
        assert not issubclass(middleware_cls, BaseHTTPMiddleware)

    @pytest.mark.anyio
    async def test_rate_limit_passes_websocket_scope_through(self):
        inner_calls: list[str] = []

        async def inner(scope, receive, send):
            inner_calls.append(scope["type"])

        middleware = RateLimitMiddleware(inner, general_rpm=1)
        scope = {"type": "websocket", "path": "/ws/ingest/x", "client": ("1.2.3.4", 1)}
        for _ in range(3):
            await middleware(scope, None, None)

        assert inner_calls == ["websocket"] * 3

    @pytest.mark.anyio
    async def test_rate_limit_keys_on_scope_client(self):
        """The bucket key is the ASGI client host (rewritten by --proxy-headers)."""
        sent: list[dict] = []

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(message):
            sent.append(message)

        middleware = RateLimitMiddleware(inner, general_rpm=1)

        def scope(host: str) -> dict:
            return {
                "type": "http",
                "method": "GET",
                "path": "/api/status/",
                "headers": [],
                "query_string": b"",
                "client": (host, 1),
            }

        await middleware(scope("10.0.0.1"), None, send)
        await middleware(scope("10.0.0.1"), None, send)
        await middleware(scope("10.0.0.2"), None, send)

        statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
        assert statuses == [200, 429, 200]


class TestRateLimitResponseThroughStack:
    """A 429 from the pure-ASGI limiter still passes through the header middleware."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_429_carries_retry_after_and_security_headers(self):
        middleware = app.middleware_stack
        while middleware is not None and not isinstance(middleware, RateLimitMiddleware):
            middleware = getattr(middleware, "app", None)
        if middleware is None:
            TestClient(app).get("/api/health")  # builds the middleware stack
            middleware = app.middleware_stack
            while not isinstance(middleware, RateLimitMiddleware):
                middleware = middleware.app
        if "search" not in middleware._buckets:
            pytest.skip("search rate limit disabled in this environment")

        engine = MagicMock()
        engine.search.return_value = []
        app.dependency_overrides[get_search_engine] = lambda: engine
        client = TestClient(app)

        resp = None
        for _ in range(middleware._buckets["search"].limit + 1):
            resp = client.post("/api/search/", json={"query": "q"})
            if resp.status_code == 429:
                break

        assert resp is not None and resp.status_code == 429
        assert int(resp.headers["retry-after"]) >= 1
        assert resp.json()["detail"]["error"] == "rate_limited"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in resp.headers
