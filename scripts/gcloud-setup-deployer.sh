#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# gcloud-setup-deployer.sh — Workload Identity Federation and a
#                            least-privilege deployer service account
#                            for .github/workflows/deploy.yml
#
# Usage:
#   export PROJECT_ID=your-gcp-project-id
#   export REGION=us-central1
#   export GITHUB_REPOSITORY=Owner/Repo   # exactly as ${{ github.repository }}
#   ./scripts/gcloud-setup-deployer.sh
#
# Optional:
#   WIF_POOL_ID       Workload Identity pool    (default: github)
#   WIF_PROVIDER_ID   OIDC provider in the pool (default: sec-search-deploy)
#   DEPLOYER_NAME     deployer account name     (default: sec-search-deployer)
#
# Prerequisites:
#   - gcloud CLI authenticated as a project owner (gcloud auth login)
#   - ./scripts/gcloud-deploy.sh setup has run (runtime service account
#     and Artifact Registry repository exist)
#
# Idempotent: re-running re-applies the provider condition and removes
# the broad project roles that earlier hand-made setups granted.
#
# Token exchange. The OIDC provider accepts a GitHub token only when it
# was issued to deploy.yml running at a v* tag of GITHUB_REPOSITORY. A
# token minted by any other workflow, branch, pull request or fork of the
# repository is rejected by Google before the deployer can be
# impersonated.
#
# Deployer roles (nothing else):
#   project     roles/run.admin                          deploy services, set invoker policy
#   project     roles/cloudbuild.builds.editor           submit and follow builds
#   project     roles/serviceusage.serviceUsageConsumer  call APIs billed to the project
#   project     roles/storage.bucketViewer               bucket metadata only (gcloud checks the
#                                                        staging bucket belongs to the project)
#   bucket      roles/storage.objectAdmin                gs://PROJECT_cloudbuild — build source upload
#   repository  roles/artifactregistry.writer            sec-search images — tag pushed images
#   account     roles/iam.serviceAccountUser             only the accounts the services and builds run as
#
# No Secret Manager role: secrets are created by gcloud-setup-secrets.sh,
# and each service reads its own secrets under its own identity.
#
# Output: the values for the GCP_WORKLOAD_IDENTITY_PROVIDER and
# GCP_SERVICE_ACCOUNT repository secrets.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID environment variable}"
REGION="${REGION:-us-central1}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:?Set GITHUB_REPOSITORY to owner/repo, exactly as GitHub spells it}"
POOL_ID="${WIF_POOL_ID:-github}"
PROVIDER_ID="${WIF_PROVIDER_ID:-sec-search-deploy}"
DEPLOYER_NAME="${DEPLOYER_NAME:-sec-search-deployer}"
DEPLOYER_SA="${DEPLOYER_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="sec-search-sa@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_NAME="sec-search"

# The repository name is written into a CEL expression below; anything
# but GitHub's own owner/name characters could change its meaning.
if [[ ! "$GITHUB_REPOSITORY" =~ ^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "GITHUB_REPOSITORY must be owner/repo (got: ${GITHUB_REPOSITORY})" >&2
    exit 1
fi

# GitHub OIDC claims compare case-sensitively. workflow_ref is
# "owner/repo/.github/workflows/deploy.yml@refs/tags/v1.2.3" for a tag
# push and for a manual run dispatched from a tag.
ATTRIBUTE_MAPPING="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref,attribute.workflow_ref=assertion.workflow_ref"
ATTRIBUTE_CONDITION="assertion.repository == '${GITHUB_REPOSITORY}' && assertion.workflow_ref.startsWith('${GITHUB_REPOSITORY}/.github/workflows/deploy.yml@refs/tags/v')"

# Project roles the deployer must not hold. Earlier setups granted the
# first two outright and the others project-wide instead of per resource.
BROAD_ROLES=(
    "roles/secretmanager.admin"
    "roles/storage.admin"
    "roles/storage.objectAdmin"
    "roles/iam.serviceAccountUser"
    "roles/artifactregistry.writer"
)

# ── Helpers ──────────────────────────────────────────────────────────
timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
}

project_roles_of() {
    # Roles bound to a principal directly on the project.
    gcloud projects get-iam-policy "$PROJECT_ID" \
        --flatten="bindings[].members" \
        --filter="bindings.members=\"$1\"" \
        --format="value(bindings.role)"
}

# ── Resolve project values ───────────────────────────────────────────
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
POOL_NAME="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}"

# gcloud builds submit stages source in gs://<project>_cloudbuild, with
# ':' and '.' replaced by '_' and 'google' by 'elgoog'.
STAGING_BUCKET="${PROJECT_ID//:/_}"
STAGING_BUCKET="${STAGING_BUCKET//./_}"
STAGING_BUCKET="${STAGING_BUCKET//google/elgoog}_cloudbuild"

# Accounts the deployer may act as: the API runs as the runtime account;
# the frontend service and Cloud Build run as the Compute Engine default
# account.
ACT_AS_ACCOUNTS=(
    "$RUNTIME_SA"
    "$COMPUTE_SA"
)

echo ""
echo "=== SEC Semantic Search — GitHub deployer setup ==="
echo "Project:    $PROJECT_ID ($PROJECT_NUMBER)"
echo "Repository: $GITHUB_REPOSITORY"
echo "Deployer:   $DEPLOYER_SA"
echo ""

log "Enabling required APIs..."
gcloud services enable \
    iam.googleapis.com \
    iamcredentials.googleapis.com \
    sts.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

# ── Workload Identity pool and provider ──────────────────────────────
if gcloud iam workload-identity-pools describe "$POOL_ID" \
    --project="$PROJECT_ID" --location=global > /dev/null 2>&1; then
    log "Workload Identity pool '$POOL_ID' already exists."
else
    log "Creating Workload Identity pool: $POOL_ID"
    gcloud iam workload-identity-pools create "$POOL_ID" \
        --project="$PROJECT_ID" \
        --location=global \
        --display-name="GitHub Actions"
fi

if gcloud iam workload-identity-pools providers describe "$PROVIDER_ID" \
    --project="$PROJECT_ID" --location=global \
    --workload-identity-pool="$POOL_ID" > /dev/null 2>&1; then
    log "Updating provider '$PROVIDER_ID' (mapping and condition)..."
    gcloud iam workload-identity-pools providers update-oidc "$PROVIDER_ID" \
        --project="$PROJECT_ID" \
        --location=global \
        --workload-identity-pool="$POOL_ID" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --attribute-mapping="$ATTRIBUTE_MAPPING" \
        --attribute-condition="$ATTRIBUTE_CONDITION"
else
    log "Creating provider: $PROVIDER_ID"
    gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
        --project="$PROJECT_ID" \
        --location=global \
        --workload-identity-pool="$POOL_ID" \
        --display-name="SEC Search deploy workflow" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --attribute-mapping="$ATTRIBUTE_MAPPING" \
        --attribute-condition="$ATTRIBUTE_CONDITION"
fi

# ── Deployer service account ─────────────────────────────────────────
if gcloud iam service-accounts describe "$DEPLOYER_SA" --project="$PROJECT_ID" > /dev/null 2>&1; then
    log "Service account '$DEPLOYER_NAME' already exists."
else
    log "Creating service account: $DEPLOYER_NAME"
    gcloud iam service-accounts create "$DEPLOYER_NAME" \
        --project="$PROJECT_ID" \
        --display-name="SEC Semantic Search GitHub deployer"
fi

log "Allowing $GITHUB_REPOSITORY to impersonate the deployer..."
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" \
    --project="$PROJECT_ID" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/${POOL_NAME}/attribute.repository/${GITHUB_REPOSITORY}" \
    --quiet > /dev/null

# ── Project roles ────────────────────────────────────────────────────
for role in \
    "roles/run.admin" \
    "roles/cloudbuild.builds.editor" \
    "roles/serviceusage.serviceUsageConsumer" \
    "roles/storage.bucketViewer"; do
    log "Granting ${role}..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${DEPLOYER_SA}" \
        --role="$role" \
        --condition=None \
        --quiet > /dev/null
done

current_roles=$(project_roles_of "serviceAccount:${DEPLOYER_SA}")
for role in "${BROAD_ROLES[@]}"; do
    if grep -qxF "$role" <<< "$current_roles"; then
        log "Removing project-wide ${role}..."
        gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${DEPLOYER_SA}" \
            --role="$role" \
            --all \
            --quiet > /dev/null
    fi
done

# ── Resource-scoped roles ────────────────────────────────────────────
# The staging bucket must belong to this project. If another project
# owns the name, creation fails here and nothing is granted on it.
project_buckets=$(gcloud storage buckets list --project="$PROJECT_ID" --format="value(name)")
if grep -qxF "$STAGING_BUCKET" <<< "$project_buckets"; then
    log "Staging bucket gs://${STAGING_BUCKET} already exists."
else
    log "Creating staging bucket: gs://${STAGING_BUCKET}"
    gcloud storage buckets create "gs://${STAGING_BUCKET}" \
        --project="$PROJECT_ID" \
        --uniform-bucket-level-access
fi

log "Granting roles/storage.objectAdmin on gs://${STAGING_BUCKET}..."
gcloud storage buckets add-iam-policy-binding "gs://${STAGING_BUCKET}" \
    --member="serviceAccount:${DEPLOYER_SA}" \
    --role="roles/storage.objectAdmin" > /dev/null

log "Granting roles/artifactregistry.writer on repository '$REPO_NAME'..."
gcloud artifacts repositories add-iam-policy-binding "$REPO_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --member="serviceAccount:${DEPLOYER_SA}" \
    --role="roles/artifactregistry.writer" > /dev/null

for account in "${ACT_AS_ACCOUNTS[@]}"; do
    log "Granting roles/iam.serviceAccountUser on ${account}..."
    gcloud iam service-accounts add-iam-policy-binding "$account" \
        --project="$PROJECT_ID" \
        --member="serviceAccount:${DEPLOYER_SA}" \
        --role="roles/iam.serviceAccountUser" \
        --quiet > /dev/null
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
log "Deployer setup complete. Set these repository secrets:"
echo "  GCP_WORKLOAD_IDENTITY_PROVIDER = ${POOL_NAME}/providers/${PROVIDER_ID}"
echo "  GCP_SERVICE_ACCOUNT            = ${DEPLOYER_SA}"
echo ""
echo "Review any other principals bound to this pool or to the deployer:"
echo "  gcloud iam service-accounts get-iam-policy ${DEPLOYER_SA} --project=${PROJECT_ID}"
