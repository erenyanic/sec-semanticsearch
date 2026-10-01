#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# github-protect-deploy.sh — Restrict who can start a production deploy
#
# Usage:
#   export GITHUB_REPOSITORY=Owner/Repo
#   export DEPLOY_REVIEWERS=login1,login2   # optional; default: the gh user
#   ./scripts/github-protect-deploy.sh
#
# Prerequisites:
#   - gh CLI authenticated as a repository admin (gh auth login)
#
# A v* tag push deploys to Cloud Run (.github/workflows/deploy.yml), so by
# default anyone with write access could deploy. This script applies, and
# re-applies on every run:
#
#   1. Tag ruleset "Release tags (v*)": only repository admins may create,
#      move or delete tags starting with v, including names with '/'
#      (rulesets match with fnmatch, where '*' stops at '/').
#   2. Environment `production`: deployments from v* tags only, each one
#      waiting for a DEPLOY_REVIEWERS approval that admins cannot bypass.
#
# The Workload Identity provider (scripts/gcloud-setup-deployer.sh) also
# refuses Google credentials to anything but deploy.yml at a v* tag.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REPO="${GITHUB_REPOSITORY:?Set GITHUB_REPOSITORY to owner/repo}"
ENVIRONMENT="production"
RULESET_NAME="Release tags (v*)"
# Built-in repository role id for "Admin".
ADMIN_ROLE_ID=5

if [[ ! "$REPO" =~ ^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "GITHUB_REPOSITORY must be owner/repo (got: ${REPO})" >&2
    exit 1
fi

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*"
}

# ── Reviewers ────────────────────────────────────────────────────────
if [ -z "${DEPLOY_REVIEWERS:-}" ]; then
    DEPLOY_REVIEWERS=$(gh api user --jq .login)
fi

reviewers_json=""
IFS=',' read -ra reviewer_logins <<< "$DEPLOY_REVIEWERS"
if [ "${#reviewer_logins[@]}" -gt 6 ]; then
    echo "GitHub allows at most 6 required reviewers per environment." >&2
    exit 1
fi
for login in "${reviewer_logins[@]}"; do
    if [[ ! "$login" =~ ^[A-Za-z0-9-]+$ ]]; then
        echo "Invalid GitHub login in DEPLOY_REVIEWERS: ${login}" >&2
        exit 1
    fi
    user_id=$(gh api "users/${login}" --jq .id)
    if [[ ! "$user_id" =~ ^[0-9]+$ ]]; then
        echo "Could not resolve GitHub user: ${login}" >&2
        exit 1
    fi
    reviewers_json+="${reviewers_json:+,}{\"type\":\"User\",\"id\":${user_id}}"
done

echo ""
echo "=== SEC Semantic Search — deploy protection ==="
echo "Repository: $REPO"
echo "Reviewers:  $DEPLOY_REVIEWERS"
echo ""

# ── 1. Tag ruleset ───────────────────────────────────────────────────
ruleset_json=$(cat <<EOF
{
  "name": "${RULESET_NAME}",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [
    {"actor_id": ${ADMIN_ROLE_ID}, "actor_type": "RepositoryRole", "bypass_mode": "always"}
  ],
  "conditions": {"ref_name": {"include": ["refs/tags/v*", "refs/tags/v*/**/*"], "exclude": []}},
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}]
}
EOF
)

ruleset_id=$(gh api --paginate "repos/${REPO}/rulesets" \
    --jq ".[] | select(.name == \"${RULESET_NAME}\") | .id")
if [ -n "$ruleset_id" ]; then
    log "Updating ruleset '${RULESET_NAME}' (${ruleset_id})..."
    gh api --method PUT "repos/${REPO}/rulesets/${ruleset_id}" --input - <<< "$ruleset_json" > /dev/null
else
    log "Creating ruleset '${RULESET_NAME}'..."
    gh api --method POST "repos/${REPO}/rulesets" --input - <<< "$ruleset_json" > /dev/null
fi

# ── 2. Production environment ────────────────────────────────────────
environment_json=$(cat <<EOF
{
  "reviewers": [${reviewers_json}],
  "can_admins_bypass": false,
  "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}
}
EOF
)

log "Configuring environment '${ENVIRONMENT}' (reviewers, custom deployment policy)..."
gh api --method PUT "repos/${REPO}/environments/${ENVIRONMENT}" --input - <<< "$environment_json" > /dev/null

policies=$(gh api --paginate "repos/${REPO}/environments/${ENVIRONMENT}/deployment-branch-policies" \
    --jq '.branch_policies[] | "\(.type // "branch") \(.name)"')
if grep -qxF "tag v*" <<< "$policies"; then
    log "Deployment policy 'tag v*' already present."
else
    log "Allowing deployments from v* tags..."
    gh api --method POST "repos/${REPO}/environments/${ENVIRONMENT}/deployment-branch-policies" \
        -f name='v*' -f type=tag > /dev/null
fi

others=$(grep -vxF "tag v*" <<< "$policies" | grep -v '^$' || true)
if [ -n "$others" ]; then
    echo ""
    echo "WARNING: '${ENVIRONMENT}' also accepts deployments from:"
    while IFS= read -r policy; do
        echo "  - ${policy}"
    done <<< "$others"
    echo "Remove them in Settings → Environments → ${ENVIRONMENT} unless intended."
fi

echo ""
log "Deploy protection applied."
