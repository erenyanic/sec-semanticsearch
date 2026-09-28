"""
In-memory sliding-window rate limiter for the FastAPI application.

Provides per-IP rate limiting with configurable limits for different
endpoint categories.  The limiter uses a simple sliding-window counter
that tracks request timestamps per client IP.

Design choices:
    - **No external dependency** — pure Python with ``threading.Lock``
      for thread safety.  Appropriate for a single-process application.
    - **Per-IP tracking** — uses the ASGI ``client`` host as the key
      (the real client IP once uvicorn's ``--proxy-headers`` rewrites it).
    - **Category-based limits** — different limits for search (GPU),
      ingest (GPU), delete (destructive), and general endpoints.
    - **Automatic cleanup** — stale entries are pruned periodically
      to prevent unbounded memory growth.
    - **Disableable** — set all limits to ``0`` to disable.

.. warning::
    **Single-worker requirement** — all rate-limit state lives in
    process memory.  Running uvicorn with ``--workers > 1`` creates
    independent copies of the counters, effectively multiplying
    the allowed rate by the number of workers and resetting state
    on every restart.  The Dockerfile enforces ``--workers 1``; do
    not override this when rate limiting or ingest cooldown is active.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from sec_semantic_search.core import get_logger

logger = get_logger(__name__)

# How often to prune stale entries (seconds).
_CLEANUP_INTERVAL = 300  # 5 minutes


class _SlidingWindow:
    """Thread-safe sliding-window counter for a single rate limit bucket."""

    __slots__ = ("_limit", "_window", "_requests", "_lock", "_last_cleanup")

    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    @property
    def limit(self) -> int:
        return self._limit

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check whether *key* may make another request.

        Returns ``(allowed, retry_after_seconds)``.
        """
        now = time.monotonic()

        with self._lock:
            # Periodic cleanup of stale keys.
            if now - self._last_cleanup > _CLEANUP_INTERVAL:
                self._prune(now)
                self._last_cleanup = now

            cutoff = now - self._window
            timestamps = self._requests[key]
            # Pop expired timestamps from the front (deque is ordered).
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._limit:
                retry_after = int(self._window - (now - timestamps[0])) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0

    def reset(self) -> None:
        """Clear all tracked requests (useful for testing)."""
        with self._lock:
            self._requests.clear()

    def _prune(self, now: float) -> None:
        """Remove keys whose most recent request is older than the window."""
        cutoff = now - self._window
        stale_keys = [k for k, v in self._requests.items() if not v or v[-1] <= cutoff]
        for k in stale_keys:
            del self._requests[k]


# ---------------------------------------------------------------------------
# Endpoint category classification
# ---------------------------------------------------------------------------


def _classify_path(path: str, method: str) -> str | None:
    """Map a request path + method to a rate-limit category.

    Returns ``None`` for paths that should not be rate-limited (e.g.
    health check, docs).
    """
    # Auth endpoints get a strict limit to deter brute-force (F5).
    if path.startswith("/api/admin/session") and method == "POST":
        return "auth"
    if path.startswith("/api/search"):
        return "search"
    if path.startswith("/api/ingest") and method == "POST":
        return "ingest"
    if method == "DELETE":
        return "delete"
    if path.startswith("/api/"):
        return "general"
    # Non-API paths (docs, health, openapi.json) — no limit.
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware:
    """Per-IP sliding-window rate limiter.

    Parameters correspond to requests-per-minute for each category.
    A value of ``0`` disables limiting for that category.

    Pure ASGI implementation — reads path, method and client straight
    from the scope, avoiding the per-request response wrapping of
    ``BaseHTTPMiddleware``. WebSocket and lifespan scopes pass through.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        search_rpm: int = 30,
        ingest_rpm: int = 5,
        delete_rpm: int = 10,
        general_rpm: int = 60,
        auth_rpm: int = 5,
    ) -> None:
        self.app = app
        self._buckets: dict[str, _SlidingWindow] = {}
        for category, rpm in [
            ("search", search_rpm),
            ("ingest", ingest_rpm),
            ("delete", delete_rpm),
            ("general", general_rpm),
            ("auth", auth_rpm),
        ]:
            if rpm > 0:
                self._buckets[category] = _SlidingWindow(rpm)

    def reset(self) -> None:
        """Reset all rate-limit counters (useful for testing)."""
        for bucket in self._buckets.values():
            bucket.reset()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        method: str = scope["method"]
        category = _classify_path(path, method)

        if category is None or category not in self._buckets:
            await self.app(scope, receive, send)
            return

        bucket = self._buckets[category]
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        allowed, retry_after = bucket.is_allowed(client_ip)
        if not allowed:
            logger.warning(
                "Rate limit exceeded: %s from %s on %s %s (limit: %d/min)",
                category,
                client_ip,
                method,
                path,
                bucket.limit,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error": "rate_limited",
                        "message": f"Rate limit exceeded ({category}). Try again in {retry_after}s.",
                        "details": None,
                        "hint": f"Maximum {bucket.limit} {category} requests per minute.",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
