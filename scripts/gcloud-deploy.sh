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
#   ./scripts/gcloud-deploy.sh scheduler  # Set up Cloud Scheduler only
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
BUCKET_NAME="${PROJECT_ID}-sec-search-data"
SERVICE_ACCOUNT_NAME="sec-search-sa"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

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
        cloudscheduler.googleapis.com \
        storage.googleapis.com \
        --project="$PROJECT_ID"

    # Create service account.
    if gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" > /dev/null 2>&1; then
        log "Service account '$SERVICE_ACCOUNT_NAME' already exists."
    else
        log "Creating service account: $SERVICE_ACCOUNT_NAME"
        gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
            --project="$PROJECT_ID" \
            --display-name="SEC Semantic Search Service Account"
    fi

    # Grant IAM roles to the service account.
    local roles=(
        "roles/run.invoker"
        "roles/storage.objectAdmin"
        "roles/secretmanager.secretAccessor"
        "roles/logging.logWriter"
    )
    for role in "${roles[@]}"; do
        log "Granting ${role}..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="$role" \
            --quiet > /dev/null
    done

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

    # Create GCS bucket for persistent data.
    if gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="$PROJECT_ID" > /dev/null 2>&1; then
        log "GCS bucket '$BUCKET_NAME' already exists."
    else
        log "Creating GCS bucket: $BUCKET_NAME"
        gcloud storage buckets create "gs://${BUCKET_NAME}" \
            --project="$PROJECT_ID" \
            --location="$REGION" \
            --uniform-bucket-level-access \
            --public-access-prevention
    fi

    # Grant bucket access to the service account.
    log "Granting bucket access to service account..."
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/storage.objectAdmin" \
        --quiet > /dev/null

    # Create initial directory structure in GCS.
    log "Ensuring data directory structure..."
    echo -n "" | gcloud storage cp - "gs://${BUCKET_NAME}/chroma_db/.keep" --quiet 2>/dev/null || true
    echo -n "" | gcloud storage cp - "gs://${BUCKET_NAME}/sqlite/.keep" --quiet 2>/dev/null || true

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

    # Deploy demo reset job.
    log "Deploying demo reset job..."
    sed_replace cloud/demo-reset-job.yaml | \
        gcloud run jobs replace - \
            --region="$REGION" \
            --project="$PROJECT_ID"

    log "All services deployed."
    echo ""
    echo "  API:      $API_URL"
    echo "  Frontend: $FRONTEND_URL"
    echo ""
}

# ── Step 4: Cloud Scheduler ──────────────────────────────────────────
do_scheduler() {
    log "=== Setting Up Cloud Scheduler ==="

    SCHEDULER_NAME="sec-search-demo-reset"
    JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${SCHEDULER_NAME}:run"

    if gcloud scheduler jobs describe "$SCHEDULER_NAME" \
        --location="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
        log "Scheduler job '$SCHEDULER_NAME' already exists — updating..."
        gcloud scheduler jobs update http "$SCHEDULER_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --schedule="0 0 * * *" \
            --time-zone="UTC" \
            --uri="$JOB_URI" \
            --http-method=POST \
            --oauth-service-account-email="$SERVICE_ACCOUNT"
    else
        log "Creating scheduler job: $SCHEDULER_NAME"
        gcloud scheduler jobs create http "$SCHEDULER_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --schedule="0 0 * * *" \
            --time-zone="UTC" \
            --uri="$JOB_URI" \
            --http-method=POST \
            --oauth-service-account-email="$SERVICE_ACCOUNT" \
            --description="Nightly demo data reset for SEC Semantic Search"
    fi

    log "Cloud Scheduler configured (midnight UTC daily)."
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

    echo ""
    echo "Jobs:"
    gcloud run jobs list \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --filter="metadata.labels.app=sec-semantic-search" \
        --format="table(metadata.name, status.conditions[0].status)" \
        2>/dev/null || echo "  No jobs found."

    echo ""
    echo "Scheduler:"
    gcloud scheduler jobs list \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --filter="description~'SEC Semantic Search'" \
        --format="table(name, schedule, state)" \
        2>/dev/null || echo "  No scheduler jobs found."

    echo ""
    echo "GCS Bucket:"
    gcloud storage ls "gs://${BUCKET_NAME}/" 2>/dev/null || echo "  Bucket not found."
}

# ── Teardown ─────────────────────────────────────────────────────────
do_teardown() {
    log "=== Teardown ==="
    echo ""
    echo "This will delete ALL Cloud Run resources for SEC Semantic Search."
    echo "Data in the GCS bucket will NOT be deleted automatically."
    echo ""
    read -rp "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Teardown cancelled."
        exit 0
    fi

    log "Deleting Cloud Scheduler job..."
    gcloud scheduler jobs delete sec-search-demo-reset \
        --location="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true

    log "Deleting Cloud Run job..."
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
    echo "  - GCS bucket: gs://${BUCKET_NAME}"
    echo "  - Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
    echo "  - Secrets: sec-search-db-encryption-key, sec-search-api-key, sec-search-admin-key"
    echo "  - Service account: ${SERVICE_ACCOUNT}"
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-all}" in
    setup)     do_setup ;;
    build)     do_build ;;
    deploy)    do_deploy ;;
    scheduler) do_scheduler ;;
    status)    do_status ;;
    teardown)  do_teardown ;;
    all)
        do_setup
        echo ""
        do_build
        echo ""
        do_deploy
        echo ""
        do_scheduler
        echo ""
        do_status
        ;;
    *)
        echo "Usage: $0 {setup|build|deploy|scheduler|status|teardown|all}"
        exit 1
        ;;
esac
