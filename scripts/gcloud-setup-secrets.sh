#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# gcloud-setup-secrets.sh — Create Secret Manager secrets for
#                           SEC Semantic Search Cloud Run deployment.
#
# Usage:
#   ./scripts/gcloud-setup-secrets.sh
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - Project set (gcloud config set project PROJECT_ID)
#   - Secret Manager API enabled
#   - Service account created (sec-search-sa)
#
# The script creates four secrets:
#   1. sec-search-db-encryption-key — SQLCipher encryption key
#   2. sec-search-api-key           — General API access key
#   3. sec-search-admin-key         — Admin-only destructive operations
#   4. sec-search-hf-token          — Hugging Face read token (build time)
#
# The first three are granted to the runtime service account with the
# secretmanager.secretAccessor role. The Hugging Face token is granted to
# the Cloud Build service account only: it bakes the gated embedding model
# into the API image, and the running service never needs it.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID environment variable}"
SERVICE_ACCOUNT="sec-search-sa@${PROJECT_ID}.iam.gserviceaccount.com"
REGION="${REGION:-us-central1}"

# Service account that runs `gcloud builds submit`. Projects whose first
# build ran after mid-2024 use the Compute Engine default service account;
# older projects use PROJECT_NUMBER@cloudbuild.gserviceaccount.com. Set
# BUILD_SERVICE_ACCOUNT to override.
BUILD_SERVICE_ACCOUNT="${BUILD_SERVICE_ACCOUNT:-}"

# ── Helper functions ─────────────────────────────────────────────────
timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

create_secret() {
    local name="$1"
    local description="$2"

    if gcloud secrets describe "$name" --project="$PROJECT_ID" > /dev/null 2>&1; then
        echo "[$(timestamp)] Secret '$name' already exists — skipping creation."
    else
        echo "[$(timestamp)] Creating secret: $name"
        gcloud secrets create "$name" \
            --project="$PROJECT_ID" \
            --replication-policy="automatic" \
            --labels="app=sec-semantic-search"
        echo "[$(timestamp)] Created: $name"
    fi

    # Prompt for the secret value.
    echo ""
    read -rsp "  Enter value for '$name' ($description): " secret_value
    echo ""

    if [ -z "$secret_value" ]; then
        echo "[$(timestamp)] WARNING: Empty value — skipping version creation for '$name'."
        return
    fi

    echo -n "$secret_value" | gcloud secrets versions add "$name" \
        --project="$PROJECT_ID" \
        --data-file=-

    echo "[$(timestamp)] Secret version added for '$name'."
}

grant_access() {
    local name="$1"
    local member="${2:-$SERVICE_ACCOUNT}"

    echo "[$(timestamp)] Granting secretAccessor to ${member} for '$name'"
    gcloud secrets add-iam-policy-binding "$name" \
        --project="$PROJECT_ID" \
        --member="serviceAccount:${member}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
}

# ── Enable Secret Manager API ───────────────────────────────────────
echo "[$(timestamp)] Ensuring Secret Manager API is enabled..."
gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID"

if [ -z "$BUILD_SERVICE_ACCOUNT" ]; then
    PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

# ── Create secrets ───────────────────────────────────────────────────
echo ""
echo "=== SEC Semantic Search — Secret Manager Setup ==="
echo "Project: $PROJECT_ID"
echo "Service Account: $SERVICE_ACCOUNT"
echo "Build Service Account: $BUILD_SERVICE_ACCOUNT"
echo ""

create_secret "sec-search-db-encryption-key" "SQLCipher encryption key"
create_secret "sec-search-api-key"           "API key for general access"
create_secret "sec-search-admin-key"         "Admin key for destructive operations"
create_secret "sec-search-hf-token"          "Hugging Face read token with access to the embedding model"

# ── Grant access to service account ──────────────────────────────────
echo ""
echo "[$(timestamp)] Granting service account access to secrets..."
grant_access "sec-search-db-encryption-key"
grant_access "sec-search-api-key"
grant_access "sec-search-admin-key"
# Build-time only — deliberately not granted to the runtime service account.
grant_access "sec-search-hf-token" "$BUILD_SERVICE_ACCOUNT"

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "[$(timestamp)] Verification:"
for secret in sec-search-db-encryption-key sec-search-api-key sec-search-admin-key sec-search-hf-token; do
    version=$(gcloud secrets versions list "$secret" \
        --project="$PROJECT_ID" \
        --format="value(name)" \
        --limit=1 2>/dev/null || echo "NONE")
    echo "  $secret: latest version = $version"
done

echo ""
echo "[$(timestamp)] Secret Manager setup complete."
echo ""
echo "Next steps:"
echo "  1. Deploy API service:      gcloud run services replace cloud/api-service.yaml --region=$REGION"
echo "  2. Deploy frontend service: gcloud run services replace cloud/frontend-service.yaml --region=$REGION"
