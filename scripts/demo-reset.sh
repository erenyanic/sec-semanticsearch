#!/usr/bin/env bash
#
# demo-reset.sh — Nightly reset script for SEC Semantic Search demo mode.
#
# Clears all ingested data (ChromaDB + SQLite) so the demo starts fresh.
# Designed to be run via cron, a systemd timer, or container entrypoint on
# self-hosted (Docker Compose) demo deployments. Cloud Run does not use it:
# there the data lives on an in-memory volume and is cleared with the instance.
#
# Usage:
#   ./scripts/demo-reset.sh                  # Uses default data paths
#   CHROMA_PATH=/data/chroma ./scripts/demo-reset.sh   # Custom paths
#
# Scheduling examples:
#   cron:            0 0 * * * /path/to/demo-reset.sh >> /var/log/demo-reset.log 2>&1
#   Docker:          Add to container entrypoint with crond
#
# Environment variables (all optional — defaults match .env.example):
#   CHROMA_PATH        — ChromaDB data directory (default: ./data/chroma_db)
#   SQLITE_PATH        — SQLite database file  (default: ./data/metadata.sqlite)
#   API_URL            — API base URL for health check (default: http://localhost:8000)
#   RESTART_STRATEGY   — Restart strategy: none, docker-compose, systemd
#   RESTART_TARGET     — Service/unit name for the chosen strategy
#   RESTART_CMD        — Deprecated and rejected for safety

set -euo pipefail

CHROMA_PATH="${CHROMA_PATH:-./data/chroma_db}"
SQLITE_PATH="${SQLITE_PATH:-./data/metadata.sqlite}"
API_URL="${API_URL:-http://localhost:8000}"
RESTART_CMD="${RESTART_CMD:-}"
RESTART_STRATEGY="${RESTART_STRATEGY:-none}"
RESTART_TARGET="${RESTART_TARGET:-api}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

restart_api() {
    case "$RESTART_STRATEGY" in
        none|"")
            echo "[$(timestamp)] No restart strategy configured; assuming the API auto-restarts"
            ;;
        docker-compose)
            echo "[$(timestamp)] Restarting Docker Compose service: $RESTART_TARGET"
            docker compose restart "$RESTART_TARGET"
            ;;
        systemd)
            echo "[$(timestamp)] Restarting systemd unit: $RESTART_TARGET"
            systemctl restart "$RESTART_TARGET"
            ;;
        *)
            echo "[$(timestamp)] ERROR: Unsupported RESTART_STRATEGY: $RESTART_STRATEGY" >&2
            exit 1
            ;;
    esac
}

echo "[$(timestamp)] Demo reset starting..."

# --- Clear ChromaDB data ---
if [ -d "$CHROMA_PATH" ]; then
    rm -rf "$CHROMA_PATH"
    echo "[$(timestamp)] Cleared ChromaDB: $CHROMA_PATH"
else
    echo "[$(timestamp)] ChromaDB directory not found (already clean): $CHROMA_PATH"
fi

# --- Clear SQLite metadata ---
if [ -f "$SQLITE_PATH" ]; then
    rm -f "$SQLITE_PATH"
    echo "[$(timestamp)] Cleared SQLite: $SQLITE_PATH"
else
    echo "[$(timestamp)] SQLite file not found (already clean): $SQLITE_PATH"
fi

# --- Restart API if configured ---
if [ -n "$RESTART_CMD" ]; then
    echo "[$(timestamp)] ERROR: RESTART_CMD is deprecated and rejected for safety. Use RESTART_STRATEGY and RESTART_TARGET instead." >&2
    exit 1
fi

restart_api

# --- Health check (best-effort) ---
echo "[$(timestamp)] Waiting for API to become healthy..."
for i in $(seq 1 30); do
    if curl -sf "${API_URL}/api/health" > /dev/null 2>&1; then
        echo "[$(timestamp)] API healthy after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[$(timestamp)] WARNING: API did not respond within 30s"
    fi
    sleep 1
done

echo "[$(timestamp)] Demo reset complete."
