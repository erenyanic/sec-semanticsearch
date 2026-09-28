"""
FastAPI application factory for SEC-SemanticSearch.

The public symbol is ``app`` — the ASGI application object used by
uvicorn and by the test client.

Architecture:
    - Singletons (ChromaDBClient, MetadataRegistry, SearchEngine,
      FilingFetcher) are initialised once in the lifespan context
      manager and stored on ``app.state``.
    - Route modules access them through dependency functions in
      ``dependencies.py`` (which read from ``request.app.state``).
    - No business logic lives here — this is pure wiring.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from sec_semantic_search import __version__
from sec_semantic_search.api.dependencies import verify_api_key
from sec_semantic_search.api.rate_limit import RateLimitMiddleware
from sec_semantic_search.config import get_settings
from sec_semantic_search.core import get_logger

logger = get_logger(__name__)

_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data: blob: https:",
        "font-src 'self' data: https:",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com",
        "connect-src 'self' ws: wss:",
    ]
)

_PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()"
)

_SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"x-xss-protection", b"1; mode=block"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"content-security-policy", _CONTENT_SECURITY_POLICY.encode()),
    (b"permissions-policy", _PERMISSIONS_POLICY.encode()),
]


# ---------------------------------------------------------------------------
# Security headers middleware (pure ASGI — no threadpool overhead)
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response.

    Pure ASGI implementation — intercepts the ``http.response.start``
    message and appends headers directly, avoiding the per-request
    ``run_in_threadpool`` overhead of ``BaseHTTPMiddleware``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(_SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class InsecureTransportWarningMiddleware:
    """Log a one-time warning when protected traffic arrives over HTTP.

    Pure ASGI implementation — reads headers from the scope and never
    wraps the response, unlike ``BaseHTTPMiddleware``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._warned = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not self._warned:
            self._maybe_warn(Headers(scope=scope))
        await self.app(scope, receive, send)

    def _maybe_warn(self, headers: Headers) -> None:
        settings = get_settings()
        if not (settings.api.key or settings.api.admin_key or settings.api.edgar_session_required):
            return

        forwarded_proto = headers.get("x-forwarded-proto")
        if forwarded_proto is None:
            return

        proto = forwarded_proto.split(",", 1)[0].strip().lower()
        if proto != "http":
            return

        self._warned = True
        logger.warning(
            "Insecure transport detected (X-Forwarded-Proto=http) while "
            "authentication or per-session EDGAR credentials are enabled. "
            "Scenarios B and C require TLS; enable HTTPS at the reverse proxy "
            "or launch the API with --ssl-certfile/--ssl-keyfile.",
        )


# ---------------------------------------------------------------------------
# Request body size limit middleware (pure ASGI — no threadpool overhead)
# ---------------------------------------------------------------------------

# 1 MB — matches nginx client_max_body_size; defence in depth for
# direct-to-uvicorn access (local dev without reverse proxy).
_MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class ContentSizeLimitMiddleware:
    """Reject requests whose body exceeds the allowed limit.

    Pure ASGI implementation with two layers of defence:

    1. **Content-Length check** — short-circuits before reading the body
       when the declared size exceeds the limit.
    2. **Stream-counting wrapper** — intercepts ``http.request`` messages
       and tracks bytes received so far.  Rejects mid-stream if the
       cumulative total exceeds the limit, protecting against chunked
       transfer encoding (which omits ``Content-Length``).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")

        if content_length_raw is not None:
            try:
                length = int(content_length_raw)
            except (ValueError, UnicodeDecodeError):
                await self._send_json(
                    send,
                    status=400,
                    body={"detail": "Invalid Content-Length header."},
                )
                return

            if length > _MAX_CONTENT_LENGTH:
                await self._send_json(
                    send,
                    status=413,
                    body={
                        "detail": {
                            "error": "payload_too_large",
                            "message": (
                                f"Request body too large ({length:,} bytes). "
                                f"Maximum allowed: {_MAX_CONTENT_LENGTH:,} bytes."
                            ),
                            "details": None,
                            "hint": "Reduce the request payload size.",
                        },
                    },
                )
                return

        # Stream-counting wrapper — enforces the limit during body reads
        # even when Content-Length is absent (chunked transfer encoding).
        bytes_received = 0
        rejected = False

        async def receive_with_limit() -> dict:
            nonlocal bytes_received, rejected
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > _MAX_CONTENT_LENGTH:
                    rejected = True
                    # Return an empty body with more_body=False to signal
                    # end of stream; the 413 is sent via send_with_guard.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def send_with_guard(message: dict) -> None:
            if rejected and message["type"] == "http.response.start":
                # Suppress the app's response — we send our own 413.
                return
            if rejected and message["type"] == "http.response.body":
                return
            await send(message)

        await self.app(scope, receive_with_limit, send_with_guard)

        if rejected:
            await self._send_json(
                send,
                status=413,
                body={
                    "detail": {
                        "error": "payload_too_large",
                        "message": (
                            f"Request body too large (>{_MAX_CONTENT_LENGTH:,} bytes). "
                            f"Maximum allowed: {_MAX_CONTENT_LENGTH:,} bytes."
                        ),
                        "details": None,
                        "hint": "Reduce the request payload size.",
                    },
                },
            )

    @staticmethod
    async def _send_json(send: Send, *, status: int, body: dict) -> None:
        """Send a complete JSON response via raw ASGI messages."""
        payload = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(payload)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload,
            }
        )


# ---------------------------------------------------------------------------
# Lifespan — initialise singletons, store on app.state
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Initialise and clean up application-level singletons.

    Heavy imports happen here (not at module top level) to keep
    ``import sec_semantic_search.api.app`` lightweight and avoid
    pulling torch/chromadb at test collection time.

    Startup order:
        1. MetadataRegistry (SQLite — fast)
        2. ChromaDBClient (ChromaDB — fast; model not loaded yet)
        3. EmbeddingGenerator (lazy — model loads on first use)
        4. SearchEngine (wraps embedder + chroma — fast)
        5. FilingFetcher (sets EDGAR identity — fast)
        6. PipelineOrchestrator (wraps fetcher + embedder — fast)
        7. TaskManager (wraps registry + chroma + fetcher + orchestrator)
    """
    from sec_semantic_search.api.tasks import TaskManager
    from sec_semantic_search.database import ChromaDBClient, MetadataRegistry
    from sec_semantic_search.pipeline import (
        EmbeddingGenerator,
        FilingFetcher,
        PipelineOrchestrator,
    )
    from sec_semantic_search.search import SearchEngine

    logger.info("SEC Semantic Search API starting up (v%s)", __version__)

    settings = get_settings()

    registry = MetadataRegistry()
    chroma = ChromaDBClient()
    embedder = EmbeddingGenerator()
    search_engine = SearchEngine(embedder=embedder, chroma_client=chroma, registry=registry)
    fetcher = FilingFetcher()
    orchestrator = PipelineOrchestrator(fetcher=fetcher, embedder=embedder)
    task_manager = TaskManager(
        registry=registry,
        chroma=chroma,
        fetcher=fetcher,
        orchestrator=orchestrator,
    )

    # Give the TaskManager a reference to the running event loop so
    # worker threads can bridge messages into the asyncio.Queue via
    # call_soon_threadsafe.
    task_manager.set_event_loop(asyncio.get_running_loop())

    app.state.registry = registry
    app.state.chroma = chroma
    app.state.embedder = embedder
    app.state.search_engine = search_engine
    app.state.fetcher = fetcher
    app.state.orchestrator = orchestrator
    app.state.task_manager = task_manager
    app.state.settings = settings

    logger.info("All singletons initialised. API ready.")
    yield
    logger.info("SEC Semantic Search API shutting down.")
    task_manager.shutdown()
    registry.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully configured ASGI application with CORS middleware,
    lifespan management, and a health-check endpoint. Route modules
    are included as they are implemented in W1.2–W1.8.
    """
    settings = get_settings()

    # Disable OpenAPI/Swagger UI when authentication is enabled (Scenarios
    # B/C).  In dev mode (no API_KEY) the docs remain available for API
    # discoverability.  See SECURITY VULNERABILITIES.md §F1.
    is_protected = bool(settings.api.key)

    application = FastAPI(
        title="SEC Semantic Search API",
        description=(
            "REST API for semantic search over ingested SEC filings "
            "(8-K, 10-K, 10-Q). Wraps the sec-semantic-search Python package "
            "over HTTP."
        ),
        version=__version__,
        docs_url=None if is_protected else "/docs",
        redoc_url=None if is_protected else "/redoc",
        openapi_url=None if is_protected else "/openapi.json",
        lifespan=lifespan,
    )

    # -- Response compression -----------------------------------------------
    # Innermost, so only application responses are compressed and every
    # outer middleware (security headers included) sees the final headers.
    # Filing text compresses several-fold; responses under 1 KB are left
    # alone. No response echoes user input next to a secret, which keeps
    # BREACH-style length attacks out of reach. Level 6 rather than
    # Starlette's default 9: compression runs on the event loop.
    application.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

    # -- Request body size limit --------------------------------------------
    application.add_middleware(ContentSizeLimitMiddleware)

    # -- Rate limiting ------------------------------------------------------
    application.add_middleware(
        RateLimitMiddleware,
        search_rpm=settings.api.rate_limit_search,
        ingest_rpm=settings.api.rate_limit_ingest,
        delete_rpm=settings.api.rate_limit_delete,
        general_rpm=settings.api.rate_limit_general,
    )

    # -- Security headers ---------------------------------------------------
    application.add_middleware(SecurityHeadersMiddleware)

    # -- Transport security warning -----------------------------------------
    application.add_middleware(InsecureTransportWarningMiddleware)

    # -- CORS ---------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Admin-Key",
            "X-Edgar-Name",
            "X-Edgar-Email",
        ],
    )

    # -- Routers (uncommented as implemented in W1.2–W1.8) ------------------
    from sec_semantic_search.api.routes.filings import router as filings_router
    from sec_semantic_search.api.routes.ingest import router as ingest_router
    from sec_semantic_search.api.routes.resources import router as resources_router
    from sec_semantic_search.api.routes.search import router as search_router
    from sec_semantic_search.api.routes.status import router as status_router
    from sec_semantic_search.api.websocket import router as ws_router

    auth = [Depends(verify_api_key)]
    application.include_router(
        status_router, prefix="/api/status", tags=["status"], dependencies=auth
    )
    application.include_router(
        filings_router, prefix="/api/filings", tags=["filings"], dependencies=auth
    )
    application.include_router(
        search_router, prefix="/api/search", tags=["search"], dependencies=auth
    )
    application.include_router(
        ingest_router, prefix="/api/ingest", tags=["ingest"], dependencies=auth
    )
    application.include_router(ws_router, tags=["websocket"])  # WS validates key in handler
    application.include_router(
        resources_router, prefix="/api/resources", tags=["resources"], dependencies=auth
    )

    # -- Health check -------------------------------------------------------
    @application.get("/api/health", tags=["meta"], summary="Health check")
    async def health() -> dict[str, str]:
        """Return API liveness status.

        The version is intentionally omitted from this unauthenticated
        endpoint to avoid disclosing it to anonymous clients (see
        SECURITY VULNERABILITIES.md §F3).  Authenticated users can
        obtain the version via ``GET /api/status/``.
        """
        return {"status": "ok"}

    return application


app = create_app()
