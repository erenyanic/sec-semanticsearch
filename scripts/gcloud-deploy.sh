#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# gcloud-deploy.sh — Deploy SEC Semantic Search to Google Cloud Run
#
# Usage:
#   # Full deployment (infrastructure + build + deploy)
#   ./scripts/gcloud-deploy.sh
#
#   # Individual steps
#   ./scripts/gcloud-deploy.sh setup      # Create infrastructure only
#   ./scripts/gcloud-deploy.sh build      # Build and push images only
#   ./scripts/gcloud-deploy.sh deploy     # Deploy services only
#   ./scripts/gcloud-deploy.sh status     # Show deployment status
#   ./scripts/gcloud-deploy.sh teardown   # Remove all resources
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - Docker installed and running
#   - PROJECT_ID and REGION environment variables set
#   - Secrets created (see scripts/gcloud-setup-secrets.sh)
#   - HUGGING_FACE_TOKEN set for the `build` step (bakes the gated model)
#
# See docs/DEPLOYMENT.md for full deployment guide.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID environment variable}"
REGION="${REGION:-us-central1}"
REPO_NAME="sec-search"
SERVICE_ACCOUNT_NAME="sec-search-sa"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
# The frontend service and Cloud Build each run as their own account, so
# neither shares an identity (or its secrets) with the other or with the
# Compute Engine default account.
FRONTEND_SERVICE_ACCOUNT_NAME="sec-search-frontend"
FRONTEND_SERVICE_ACCOUNT="${FRONTEND_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SERVICE_ACCOUNT_NAME="sec-search-build"
BUILD_SERVICE_ACCOUNT="${BUILD_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/api:latest"
FRONTEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend:latest"

# CUDA PyTorch wheel for GPU-enabled API image.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# ── Helpers ──────────────────────────────────────────────────────────
timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
}

ensure_service_account() {
    local name="$1"
    local display_name="$2"
    local email="${name}@${PROJECT_ID}.iam.gserviceaccount.com"

    if gcloud iam service-accounts describe "$email" --project="$PROJECT_ID" > /dev/null 2>&1; then
        log "Service account '$name' already exists."
    else
        log "Creating service account: $name"
        gcloud iam service-accounts create "$name" \
            --project="$PROJECT_ID" \
            --display-name="$display_name"
    fi
}

sed_replace() {
    # Replace placeholder tokens in YAML files with actual values.
    # Works on both GNU and BSD sed.
    local file="$1"
    local temp_file
    temp_file=$(mktemp)

    sed \
        -e "s|PROJECT_ID|${PROJECT_ID}|g" \
        -e "s|REGION|${REGION}|g" \
        "$file" > "$temp_file"

    cat "$temp_file"
    rm -f "$temp_file"
}

# ── Step 1: Infrastructure setup ────────────────────────────────────
do_setup() {
    log "=== Infrastructure Setup ==="

    # Enable required APIs.
    log "Enabling required APIs..."
    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        storage.googleapis.com \
        --project="$PROJECT_ID"

    # Create service accounts: the API's runtime identity, the frontend's,
    # and the one Cloud Build runs as (deploy.yml).
    ensure_service_account "$SERVICE_ACCOUNT_NAME" "SEC Semantic Search Service Account"
    ensure_service_account "$FRONTEND_SERVICE_ACCOUNT_NAME" "SEC Semantic Search frontend"
    ensure_service_account "$BUILD_SERVICE_ACCOUNT_NAME" "SEC Semantic Search Cloud Build"

    # Grant IAM roles to the API service account. No storage role: the API
    # keeps its data on an in-memory volume, not in Cloud Storage. No
    # Secret Manager role either: gcloud-setup-secrets.sh grants access
    # per secret, and project-wide access would include the build-time
    # Hugging Face token.
    local roles=(
        "roles/run.invoker"
        "roles/logging.logWriter"
    )
    for role in "${roles[@]}"; do
        log "Granting ${role}..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="$role" \
            --quiet > /dev/null
    done

    # Deployments set up before per-secret access hold it project-wide.
    local api_roles
    api_roles=$(gcloud projects get-iam-policy "$PROJECT_ID" \
        --flatten="bindings[].members" \
        --filter="bindings.members=\"serviceAccount:${SERVICE_ACCOUNT}\"" \
        --format="value(bindings.role)")
    if grep -qxF "roles/secretmanager.secretAccessor" <<< "$api_roles"; then
        log "Removing project-wide roles/secretmanager.secretAccessor from ${SERVICE_ACCOUNT_NAME}..."
        gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="roles/secretmanager.secretAccessor" \
            --all \
            --quiet > /dev/null
    fi

    # Cloud Build writes its logs to Cloud Logging (CLOUD_LOGGING_ONLY in
    # deploy.yml). The frontend account needs no project role.
    log "Granting roles/logging.logWriter to ${BUILD_SERVICE_ACCOUNT_NAME}..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
        --role="roles/logging.logWriter" \
        --quiet > /dev/null

    # Create Artifact Registry repository.
    if gcloud artifacts repositories describe "$REPO_NAME" \
        --location="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
        log "Artifact Registry repository '$REPO_NAME' already exists."
    else
        log "Creating Artifact Registry repository: $REPO_NAME"
        gcloud artifacts repositories create "$REPO_NAME" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --description="SEC Semantic Search container images"
    fi

    # Cloud Build pushes the images and reads the layer cache back.
    log "Granting roles/artifactregistry.writer on '$REPO_NAME' to ${BUILD_SERVICE_ACCOUNT_NAME}..."
    gcloud artifacts repositories add-iam-policy-binding "$REPO_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --member="serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
        --role="roles/artifactregistry.writer" > /dev/null

    log "Infrastructure setup complete."
}

# ── Step 2: Build and push container images ──────────────────────────
do_build() {
    log "=== Building Container Images ==="

    # Configure Docker for Artifact Registry.
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    # Build API image (CUDA-enabled for GPU) with the embedding model baked
    # in. The Cloud Run manifest sets HF_HUB_OFFLINE=1, so the build must
    # have the weights; the token travels as a BuildKit secret only.
    : "${HUGGING_FACE_TOKEN:?Set HUGGING_FACE_TOKEN to bake the embedding model into the API image}"
    export HUGGING_FACE_TOKEN
    log "Building API image (CUDA-enabled, model baked in)..."
    DOCKER_BUILDKIT=1 docker build \
        -f Dockerfile.api \
        --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
        --build-arg REQUIRE_BAKED_MODEL=1 \
        --secret id=hf_token,env=HUGGING_FACE_TOKEN \
        -t "$API_IMAGE" \
        .

    # Build frontend image.
    log "Building frontend image..."
    docker build \
        -f Dockerfile.frontend \
        -t "$FRONTEND_IMAGE" \
        .

    # Push images.
    log "Pushing API image..."
    docker push "$API_IMAGE"

    log "Pushing frontend image..."
    docker push "$FRONTEND_IMAGE"

    log "Images pushed to Artifact Registry."
}

# ── Step 3: Deploy services ──────────────────────────────────────────
do_deploy() {
    log "=== Deploying Services ==="

    # Deploy API service.
    log "Deploying API service..."
    sed_replace cloud/api-service.yaml | \
        gcloud run services replace - \
            --region="$REGION" \
            --project="$PROJECT_ID"

    # Allow unauthenticated access to the API (API key handles auth).
    gcloud run services add-iam-policy-binding sec-search-api \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --member="allUsers" \
        --role="roles/run.invoker" \
        --quiet > /dev/null

    # Get API URL for frontend configuration.
    API_URL=$(gcloud run services describe sec-search-api \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    log "API deployed at: $API_URL"

    # Deploy frontend service.
    log "Deploying frontend service..."
    sed_replace cloud/frontend-service.yaml | \
        gcloud run services replace - \
            --region="$REGION" \
            --project="$PROJECT_ID"

    gcloud run services add-iam-policy-binding sec-search-frontend \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --member="allUsers" \
        --role="roles/run.invoker" \
        --quiet > /dev/null

    FRONTEND_URL=$(gcloud run services describe sec-search-frontend \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    log "Frontend deployed at: $FRONTEND_URL"

    # Update API CORS with the actual frontend URL.
    log "Updating API CORS origins with frontend URL..."
    gcloud run services update sec-search-api \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --update-env-vars="API_CORS_ORIGINS=[\"${FRONTEND_URL}\"]" \
        --quiet

    # Update frontend with the actual API URL.
    log "Updating frontend with API URL..."
    gcloud run services update sec-search-frontend \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --update-env-vars="INTERNAL_API_BASE_URL=${API_URL}" \
        --quiet

    log "All services deployed."
    echo ""
    echo "  API:      $API_URL"
    echo "  Frontend: $FRONTEND_URL"
    echo ""
}

# ── Status ───────────────────────────────────────────────────────────
do_status() {
    log "=== Deployment Status ==="
    echo ""

    echo "Services:"
    gcloud run services list \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --filter="metadata.labels.app=sec-semantic-search" \
        --format="table(metadata.name, status.url, status.conditions[0].status)" \
        2>/dev/null || echo "  No services found."
}

# ── Teardown ─────────────────────────────────────────────────────────
do_teardown() {
    log "=== Teardown ==="
    echo ""
    echo "This will delete ALL Cloud Run resources for SEC Semantic Search."
    echo ""
    read -rp "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Teardown cancelled."
        exit 0
    fi

    # Deployments created before data became ephemeral also have a demo
    # reset job and scheduler; both deletes are no-ops when absent.
    log "Deleting legacy demo reset scheduler and job (if present)..."
    gcloud scheduler jobs delete sec-search-demo-reset \
        --location="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true
    gcloud run jobs delete sec-search-demo-reset \
        --region="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true

    log "Deleting frontend service..."
    gcloud run services delete sec-search-frontend \
        --region="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true

    log "Deleting API service..."
    gcloud run services delete sec-search-api \
        --region="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true

    log "Teardown complete."
    echo ""
    echo "Remaining resources (manual cleanup if needed):"
    echo "  - Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
    echo "  - Secrets: sec-search-db-encryption-key, sec-search-api-key, sec-search-admin-key, sec-search-hf-token"
    echo "  - Legacy data bucket, if created by an older version: gs://${PROJECT_ID}-sec-search-data"
    echo "  - Service accounts: ${SERVICE_ACCOUNT}, ${FRONTEND_SERVICE_ACCOUNT}, ${BUILD_SERVICE_ACCOUNT}"
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-all}" in
    setup)     do_setup ;;
    build)     do_build ;;
    deploy)    do_deploy ;;
    status)    do_status ;;
    teardown)  do_teardown ;;
    all)
        do_setup
        echo ""
        do_build
        echo ""
        do_deploy
        echo ""
        do_status
        ;;
    *)
        echo "Usage: $0 {setup|build|deploy|status|teardown|all}"
        exit 1
        ;;
esac
