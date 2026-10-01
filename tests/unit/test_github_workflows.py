"""
Tests for GitHub Actions workflow configuration files.

Validates `.github/workflows/ci.yml` and `.github/workflows/deploy.yml`
for structural correctness, least-privilege permissions, pinned action
versions, and Phase 4.2 CI/CD design invariants.

These tests do not touch GitHub — they only inspect the YAML files. A
regression in workflow structure (e.g. accidentally broadening the
deploy trigger to every push on `main`, or embedding a secret literal)
is caught here before the change reaches a pull request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"
DEPLOY_WORKFLOW = WORKFLOWS_DIR / "deploy.yml"


# ── Helpers ──────────────────────────────────────────────────────────


def _load_workflow(path: Path) -> dict:
    assert path.exists(), f"{path} not found — Phase 4.2 CI/CD config missing"
    return yaml.safe_load(path.read_text())


def _workflow_on(workflow: dict) -> dict:
    """PyYAML parses the bare word `on:` as boolean True — handle both."""
    return workflow.get("on") or workflow.get(True) or {}


def _steps_of(workflow: dict, job_name: str) -> list[dict]:
    return workflow["jobs"][job_name]["steps"]


def _collect_uses(workflow: dict) -> list[str]:
    """Return every `uses:` reference across all jobs and steps."""
    refs: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                refs.append(step["uses"])
    return refs


def _collect_run_blocks(workflow: dict) -> list[str]:
    """Return every `run:` script across all jobs and steps."""
    scripts: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                scripts.append(step["run"])
    return scripts


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ci_workflow() -> dict:
    return _load_workflow(CI_WORKFLOW)


@pytest.fixture
def deploy_workflow() -> dict:
    return _load_workflow(DEPLOY_WORKFLOW)


# ── Presence & basic structure ───────────────────────────────────────


class TestWorkflowFilesExist:
    """Phase 4.2 requires two workflow files."""

    def test_workflows_directory_exists(self):
        assert WORKFLOWS_DIR.is_dir(), ".github/workflows not found"

    def test_ci_workflow_exists(self):
        assert CI_WORKFLOW.exists(), "ci.yml missing"

    def test_deploy_workflow_exists(self):
        assert DEPLOY_WORKFLOW.exists(), "deploy.yml missing"

    def test_ci_parses_as_yaml(self, ci_workflow):
        assert ci_workflow is not None
        assert "jobs" in ci_workflow

    def test_deploy_parses_as_yaml(self, deploy_workflow):
        assert deploy_workflow is not None
        assert "jobs" in deploy_workflow


# ── CI workflow structure ────────────────────────────────────────────


class TestCiWorkflowStructure:
    """ci.yml must run backend, frontend, and lint on push/PR to main."""

    def test_name(self, ci_workflow):
        assert ci_workflow.get("name") == "CI"

    def test_triggers_on_push_to_main(self, ci_workflow):
        on = _workflow_on(ci_workflow)
        assert "push" in on
        assert "main" in on["push"]["branches"]

    def test_triggers_on_pull_request(self, ci_workflow):
        on = _workflow_on(ci_workflow)
        assert "pull_request" in on
        assert "main" in on["pull_request"]["branches"]

    def test_manual_dispatch_supported(self, ci_workflow):
        on = _workflow_on(ci_workflow)
        assert "workflow_dispatch" in on

    def test_least_privilege_permissions(self, ci_workflow):
        """CI must not request broader than read access to repo contents."""
        perms = ci_workflow.get("permissions", {})
        assert perms == {"contents": "read"}, "CI workflow must use minimal read-only permissions"

    def test_cancels_stale_runs(self, ci_workflow):
        """Stale CI runs should be cancelled to conserve Actions minutes."""
        concurrency = ci_workflow.get("concurrency", {})
        assert concurrency.get("cancel-in-progress") is True

    def test_concurrency_group_keyed_on_ref(self, ci_workflow):
        group = ci_workflow["concurrency"]["group"]
        assert "${{ github.ref }}" in group

    def test_has_three_independent_jobs(self, ci_workflow):
        jobs = ci_workflow["jobs"]
        assert set(jobs.keys()) == {"backend-tests", "frontend-tests", "lint"}

    def test_jobs_run_in_parallel(self, ci_workflow):
        """None of the three CI jobs should depend on another."""
        for job_name, job in ci_workflow["jobs"].items():
            assert "needs" not in job, f"CI job {job_name!r} should run independently"


class TestCiBackendJob:
    """Backend job must install CPU torch and run pytest."""

    @pytest.fixture
    def backend_job(self, ci_workflow) -> dict:
        return ci_workflow["jobs"]["backend-tests"]

    def test_runs_on_ubuntu(self, backend_job):
        assert backend_job["runs-on"] == "ubuntu-latest"

    def test_has_timeout(self, backend_job):
        assert "timeout-minutes" in backend_job

    def test_python_3_12(self, backend_job):
        """Project targets Python 3.12 (matches Dockerfile.api)."""
        matrix = backend_job["strategy"]["matrix"]["python-version"]
        assert "3.12" in matrix

    def test_installs_cpu_torch_wheel(self, ci_workflow):
        """CI must pull the CPU torch wheel — not the full CUDA build."""
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "backend-tests")
        )
        assert "download.pytorch.org/whl/cpu" in run_blocks
        assert "download.pytorch.org/whl/cu" not in run_blocks

    def test_installs_dev_extra(self, ci_workflow):
        """The dev extra provides pytest, ruff, mypy, httpx."""
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "backend-tests")
        )
        assert ".[dev]" in run_blocks

    def test_does_not_install_encryption_extra(self, ci_workflow):
        """The pysqlcipher3 build is expensive and unnecessary for CI.

        Encryption tests mock `pysqlcipher3` via sys.modules, so they
        work without the real library. Installing the encryption extra
        would require apt-get install libsqlcipher-dev, slowing builds.
        """
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "backend-tests")
        )
        assert ".[encryption]" not in run_blocks
        assert ".[dev,encryption]" not in run_blocks

    def test_runs_pytest(self, ci_workflow):
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "backend-tests")
        )
        assert "pytest" in run_blocks

    def test_edgar_identity_env_set(self, ci_workflow):
        """Tests import settings — EDGAR identity must be populated."""
        pytest_step = next(
            step
            for step in _steps_of(ci_workflow, "backend-tests")
            if "pytest" in step.get("run", "")
        )
        assert "EDGAR_IDENTITY_NAME" in pytest_step.get("env", {})
        assert "EDGAR_IDENTITY_EMAIL" in pytest_step.get("env", {})


class TestCiFrontendJob:
    """Frontend job must run Vitest, ESLint, and the production build."""

    @pytest.fixture
    def frontend_job(self, ci_workflow) -> dict:
        return ci_workflow["jobs"]["frontend-tests"]

    def test_node_22(self, frontend_job):
        """Matches Dockerfile.frontend base image (node:22.22.1-alpine)."""
        matrix = frontend_job["strategy"]["matrix"]["node-version"]
        assert "22" in matrix

    def test_working_directory_is_frontend(self, frontend_job):
        assert frontend_job["defaults"]["run"]["working-directory"] == "frontend"

    def test_runs_npm_ci(self, ci_workflow):
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "frontend-tests")
        )
        assert "npm ci" in run_blocks

    def test_runs_eslint(self, ci_workflow):
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "frontend-tests")
        )
        assert "npm run lint" in run_blocks

    def test_runs_vitest(self, ci_workflow):
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "frontend-tests")
        )
        assert "npm run test" in run_blocks

    def test_builds_production_bundle(self, ci_workflow):
        """npm run build verifies the Next.js standalone output works."""
        run_blocks = "\n".join(
            step.get("run", "") for step in _steps_of(ci_workflow, "frontend-tests")
        )
        assert "npm run build" in run_blocks


class TestCiLintJob:
    """Lint job must run blocking Ruff check and format verification."""

    def test_installs_pinned_ruff(self, ci_workflow):
        """Ruff version must match the dev extra pin for reproducibility."""
        run_blocks = "\n".join(step.get("run", "") for step in _steps_of(ci_workflow, "lint"))
        assert "ruff==0.15.5" in run_blocks

    def test_runs_ruff_check(self, ci_workflow):
        run_blocks = "\n".join(step.get("run", "") for step in _steps_of(ci_workflow, "lint"))
        assert re.search(r"\bruff check\b", run_blocks)
        assert "ruff check src tests" in run_blocks

    def test_runs_ruff_format_check(self, ci_workflow):
        run_blocks = "\n".join(step.get("run", "") for step in _steps_of(ci_workflow, "lint"))
        assert "ruff format --check src tests" in run_blocks

    def test_lint_is_blocking(self, ci_workflow):
        """Lint is a required check once the backlog in TODO.md §4.3 is cleared."""
        lint_job = ci_workflow["jobs"]["lint"]
        assert lint_job.get("continue-on-error") is not True, (
            "lint job must block merges once the Ruff backlog is cleared"
        )


# ── Deploy workflow structure ────────────────────────────────────────


class TestDeployWorkflowTriggers:
    """Deploy must only run on tag push or manual dispatch — never on main push."""

    def test_name(self, deploy_workflow):
        assert deploy_workflow.get("name") == "Deploy to Cloud Run"

    def test_does_not_trigger_on_branch_push(self, deploy_workflow):
        """Pushing to main must NOT deploy — only tagged releases do."""
        on = _workflow_on(deploy_workflow)
        push_config = on.get("push", {})
        # `push.branches` must not be defined (or must be empty). Only
        # `push.tags` is acceptable.
        assert "branches" not in push_config, (
            "Deploy workflow must not trigger on branch pushes — only tag pushes (v*) are allowed"
        )

    def test_triggers_on_version_tags(self, deploy_workflow):
        on = _workflow_on(deploy_workflow)
        tags = on["push"]["tags"]
        assert any(pattern.startswith("v") for pattern in tags), (
            "Deploy should trigger on semver-style tags (v*)"
        )

    def test_manual_dispatch_supported(self, deploy_workflow):
        on = _workflow_on(deploy_workflow)
        assert "workflow_dispatch" in on

    def test_dispatch_has_environment_input(self, deploy_workflow):
        on = _workflow_on(deploy_workflow)
        inputs = on["workflow_dispatch"]["inputs"]
        assert "environment" in inputs


class TestDeployWorkflowSecurity:
    """Deploy must use WIF, least privilege, and avoid hardcoded secrets."""

    @staticmethod
    def _authenticating_jobs(workflow: dict) -> set[str]:
        return {
            name
            for name, job in workflow["jobs"].items()
            if any(
                step.get("uses", "").startswith("google-github-actions/auth")
                for step in job.get("steps", [])
            )
        }

    def test_workflow_default_is_read_only(self, deploy_workflow):
        """No job may inherit id-token: write from the workflow level.

        Any code that runs in a job holding id-token: write can mint a
        GitHub OIDC token for this repository (security audit 2026-09-16).
        """
        assert deploy_workflow.get("permissions") == {"contents": "read"}

    def test_every_job_declares_permissions(self, deploy_workflow):
        for name, job in deploy_workflow["jobs"].items():
            assert "permissions" in job, f"deploy job {name!r} must declare its permissions"

    def test_id_token_only_where_google_auth_runs(self, deploy_workflow):
        """id-token: write exactly on the jobs that exchange it for Google credentials."""
        with_id_token = {
            name
            for name, job in deploy_workflow["jobs"].items()
            if job.get("permissions", {}).get("id-token") == "write"
        }
        assert with_id_token == self._authenticating_jobs(deploy_workflow)
        assert with_id_token == {"build", "deploy", "smoke-test"}

    def test_no_job_writes_to_the_repository(self, deploy_workflow):
        """id-token is the only write permission any deploy job holds."""
        for name, job in deploy_workflow["jobs"].items():
            for scope, level in job["permissions"].items():
                assert level == "read" or scope == "id-token", (
                    f"deploy job {name!r} holds {scope}: {level}"
                )

    def test_ci_gate_reads_checks_only(self, deploy_workflow):
        assert deploy_workflow["jobs"]["await-ci"]["permissions"] == {
            "contents": "read",
            "checks": "read",
        }

    def test_ci_gate_runs_no_action_code(self, deploy_workflow):
        """The gate must not run an action whose dependencies a SHA pin cannot freeze.

        lewagon/wait-on-check-action is a composite action that installed
        ruby/setup-ruby and actions/cache by mutable tag and ran
        ``bundle install`` at job time.
        """
        steps = deploy_workflow["jobs"]["await-ci"]["steps"]
        assert all("uses" not in step for step in steps)
        assert "lewagon/" not in DEPLOY_WORKFLOW.read_text()

    def test_ci_gate_waits_for_every_ci_job(self, deploy_workflow, ci_workflow):
        """Each CI job's rendered check name must be in the gate's list."""
        gate = deploy_workflow["jobs"]["await-ci"]["steps"][0]["run"]
        for job in ci_workflow["jobs"].values():
            name = job["name"]
            for key, values in job.get("strategy", {}).get("matrix", {}).items():
                name = name.replace(f"${{{{ matrix.{key} }}}}", str(values[0]))
            assert f'"{name}"' in gate, f"deploy gate does not wait for {name!r}"

    def test_ci_gate_counts_github_actions_checks_only(self, deploy_workflow):
        """A check of the same name from another installed app must not pass the gate."""
        gate = deploy_workflow["jobs"]["await-ci"]["steps"][0]["run"]
        assert "app_id=15368" in gate
        assert "filter=latest" in gate
        assert '"completed success"' in gate

    def test_ci_gate_reads_context_through_env(self, deploy_workflow):
        step = deploy_workflow["jobs"]["await-ci"]["steps"][0]
        assert "${{" not in step["run"]
        assert step["env"]["SHA"] == "${{ github.sha }}"
        assert step["env"]["REPO"] == "${{ github.repository }}"

    def test_uses_workload_identity_federation(self, deploy_workflow):
        """google-github-actions/auth must be called with WIF, not a JSON key."""
        uses = _collect_uses(deploy_workflow)
        auth_refs = [u for u in uses if u.startswith("google-github-actions/auth")]
        assert len(auth_refs) >= 1, "Deploy must authenticate to GCP"

        # Walk every step and confirm `credentials_json` is never used.
        for job in deploy_workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("google-github-actions/auth"):
                    with_block = step.get("with", {})
                    assert "credentials_json" not in with_block, (
                        "Deploy must use workload_identity_provider — "
                        "long-lived JSON keys are forbidden"
                    )
                    assert "workload_identity_provider" in with_block
                    assert "service_account" in with_block

    def test_secrets_referenced_not_embedded(self, deploy_workflow):
        """Required GCP secrets must be pulled from ${{ secrets.* }}."""
        text = DEPLOY_WORKFLOW.read_text()
        for secret in [
            "GCP_WORKLOAD_IDENTITY_PROVIDER",
            "GCP_SERVICE_ACCOUNT",
            "GCP_PROJECT_ID",
            "GCP_REGION",
        ]:
            assert f"secrets.{secret}" in text, f"Deploy workflow must reference secrets.{secret}"

    def test_no_hardcoded_api_keys_or_encryption_keys(self, deploy_workflow):
        """Never embed API_KEY, ADMIN_API_KEY, or encryption key values."""
        text = DEPLOY_WORKFLOW.read_text()
        # These env var names are fine to reference — we only forbid
        # literal assignments like `API_KEY: somevalue` where the value
        # is a plain string (not a ${{ secrets.X }} reference).
        forbidden_patterns = [
            r'API_KEY:\s*"[^$][^"]*"',
            r'ADMIN_API_KEY:\s*"[^$][^"]*"',
            r'DB_ENCRYPTION_KEY:\s*"[^$][^"]*"',
        ]
        for pattern in forbidden_patterns:
            assert not re.search(pattern, text), (
                f"Deploy workflow appears to hardcode a secret: {pattern}"
            )

    def test_concurrency_does_not_cancel_in_progress(self, deploy_workflow):
        """Deploys must serialise — never cancel an in-flight deploy."""
        concurrency = deploy_workflow.get("concurrency", {})
        assert concurrency.get("cancel-in-progress") is False


class TestDeployWorkflowJobs:
    """The deploy workflow must sequence: await-ci → build → deploy → smoke-test."""

    def test_has_await_ci_gate(self, deploy_workflow):
        assert "await-ci" in deploy_workflow["jobs"]

    def test_build_job_depends_on_await_ci(self, deploy_workflow):
        build = deploy_workflow["jobs"]["build"]
        assert "await-ci" in build.get("needs", [])

    def test_deploy_job_depends_on_build(self, deploy_workflow):
        deploy = deploy_workflow["jobs"]["deploy"]
        assert "build" in deploy.get("needs", [])

    def test_smoke_test_depends_on_deploy(self, deploy_workflow):
        smoke = deploy_workflow["jobs"]["smoke-test"]
        assert "deploy" in smoke.get("needs", [])

    def test_deploy_uses_environment_protection(self, deploy_workflow):
        """GitHub environments can require reviewer approval before deploy."""
        deploy = deploy_workflow["jobs"]["deploy"]
        env = deploy.get("environment", {})
        # Can be a string or an object.
        if isinstance(env, dict):
            assert "name" in env
        else:
            assert env

    def test_smoke_test_hits_api_health(self, deploy_workflow):
        """Post-deploy verification must call the unauthenticated health endpoint."""
        runs = _collect_run_blocks(deploy_workflow)
        combined = "\n".join(runs)
        assert "/api/health" in combined

    def test_build_sets_cuda_torch_index(self, deploy_workflow):
        """API image must be built with the CUDA torch wheel for L4 GPU."""
        build = deploy_workflow["jobs"]["build"]
        env = build.get("env", {})
        assert env.get("TORCH_INDEX_URL") == "https://download.pytorch.org/whl/cu124"

    def test_builds_run_as_the_build_account(self, deploy_workflow):
        """Only sec-search-build can read the Hugging Face token (audit 2026-09-28).

        Without --service-account, Cloud Build runs as the Compute Engine
        default account, which the frontend service used to run as too.
        """
        build = deploy_workflow["jobs"]["build"]
        assert build["env"]["BUILD_SERVICE_ACCOUNT_NAME"] == "sec-search-build"
        submits = [
            run for run in _collect_run_blocks({"jobs": {"build": build}}) if "builds submit" in run
        ]
        assert len(submits) == 2
        for run in submits:
            assert (
                '--service-account="projects/${PROJECT_ID}/serviceAccounts/'
                '${BUILD_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"'
            ) in run
            # A user-specified build account requires an explicit logging mode.
            assert "logging: CLOUD_LOGGING_ONLY" in run

    def test_deploy_reuses_gcloud_deploy_script(self, deploy_workflow):
        """Single source of truth: reuse scripts/gcloud-deploy.sh."""
        runs = _collect_run_blocks(deploy_workflow)
        combined = "\n".join(runs)
        assert "scripts/gcloud-deploy.sh deploy" in combined

    def test_no_demo_reset_scheduler(self, deploy_workflow):
        """Demo data is ephemeral on Cloud Run; there is no reset job to schedule (F-03)."""
        combined = "\n".join(_collect_run_blocks(deploy_workflow))
        assert "gcloud-deploy.sh scheduler" not in combined


# ── Action version pinning (supply chain) ────────────────────────────


class TestActionPinning:
    """Third-party actions must be pinned to full commit SHAs."""

    _FORBIDDEN_REFS = {"@main", "@master", "@latest"}
    _SHA_RE = re.compile(r"@[0-9a-f]{40}\b")

    @pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, DEPLOY_WORKFLOW])
    def test_no_floating_action_refs(self, workflow_path):
        workflow = _load_workflow(workflow_path)
        uses = _collect_uses(workflow)
        for ref in uses:
            for forbidden in self._FORBIDDEN_REFS:
                assert forbidden not in ref, (
                    f"{workflow_path.name}: action {ref!r} uses a floating "
                    f"tag ({forbidden}) — pin to a specific version"
                )

    @pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, DEPLOY_WORKFLOW])
    def test_actions_have_explicit_version(self, workflow_path):
        """Every `uses:` must have a @version suffix."""
        workflow = _load_workflow(workflow_path)
        uses = _collect_uses(workflow)
        for ref in uses:
            # Local actions (./path/to/action) are exempt.
            if ref.startswith("./"):
                continue
            assert "@" in ref, f"{workflow_path.name}: action {ref!r} has no version pin"

    @pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, DEPLOY_WORKFLOW])
    def test_actions_pinned_to_full_sha(self, workflow_path):
        """Every third-party action must be pinned to a full 40-char SHA.

        Tag-only pins (e.g. ``@v4``) are mutable — a compromised upstream
        can silently change what they point to. SHA pinning is the only
        immutable reference. Version comments (``# v4.2.2``) are allowed
        alongside the SHA for readability.
        """
        workflow = _load_workflow(workflow_path)
        uses = _collect_uses(workflow)
        for ref in uses:
            if ref.startswith("./"):
                continue
            assert self._SHA_RE.search(ref), (
                f"{workflow_path.name}: action {ref!r} is not pinned to a "
                f"full commit SHA — use owner/repo@<sha> # <tag> format"
            )


# ── Script-injection protection ──────────────────────────────────────


class TestScriptInjection:
    """Guard against command injection via attacker-controlled GitHub context.

    GitHub's security advisory on script injection warns that inlining
    `${{ github.event.pull_request.title }}` (or similar) into a `run:`
    block lets an attacker embed shell metacharacters in a PR title. The
    safe pattern is to pass untrusted data through `env:` instead.
    """

    # github context fields that are attacker-controlled.
    _UNSAFE_CONTEXT_FIELDS = [
        "github.event.issue.title",
        "github.event.issue.body",
        "github.event.pull_request.title",
        "github.event.pull_request.body",
        "github.event.pull_request.head.ref",
        "github.event.pull_request.head.label",
        "github.event.comment.body",
        "github.event.review.body",
        "github.event.review_comment.body",
        "github.event.pages",
        "github.event.commits",
        "github.head_ref",
    ]

    @pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, DEPLOY_WORKFLOW])
    def test_no_untrusted_context_in_run_blocks(self, workflow_path):
        """Untrusted github.* fields must never appear inside `run:`."""
        workflow = _load_workflow(workflow_path)
        for job_name, job in workflow["jobs"].items():
            for i, step in enumerate(job.get("steps", [])):
                run_script = step.get("run", "")
                for unsafe in self._UNSAFE_CONTEXT_FIELDS:
                    assert unsafe not in run_script, (
                        f"{workflow_path.name}::{job_name}::step[{i}] "
                        f"inlines {unsafe} into a run: block. "
                        "Pass it through `env:` instead."
                    )

    @pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, DEPLOY_WORKFLOW])
    def test_no_step_outputs_in_run_blocks(self, workflow_path):
        """Step outputs must be passed via env:, not inlined in run: scripts.

        Even when the source is trusted (e.g. gcloud output), inlining
        ``${{ steps.*.outputs.* }}`` in ``run:`` blocks is fragile — values
        containing shell metacharacters can break the script. The ``env:``
        indirection is both safer and consistent with the project's
        script-injection guard (AGENT.md rule #41).
        """
        workflow = _load_workflow(workflow_path)
        step_output_re = re.compile(r"\$\{\{\s*steps\.[^}]+\.outputs\.")
        for job_name, job in workflow["jobs"].items():
            for i, step in enumerate(job.get("steps", [])):
                run_script = step.get("run", "")
                assert not step_output_re.search(run_script), (
                    f"{workflow_path.name}::{job_name}::step[{i}] "
                    f"inlines steps.*.outputs.* into a run: block. "
                    "Pass it through `env:` instead."
                )


# ── Cross-workflow consistency ───────────────────────────────────────


class TestWorkflowConsistency:
    """Deploy invariants that depend on the rest of the repo."""

    def test_gcloud_deploy_script_is_executable(self):
        """Deploy workflow depends on scripts/gcloud-deploy.sh being runnable."""
        script = PROJECT_ROOT / "scripts" / "gcloud-deploy.sh"
        assert script.exists()
        assert script.stat().st_mode & 0o111

    def test_dockerfile_api_exists(self):
        """Build step references Dockerfile.api."""
        assert (PROJECT_ROOT / "Dockerfile.api").exists()

    def test_dockerfile_frontend_exists(self):
        """Build step references Dockerfile.frontend."""
        assert (PROJECT_ROOT / "Dockerfile.frontend").exists()

    def test_node_version_matches_dockerfile(self, ci_workflow):
        """CI Node major version must match Dockerfile.frontend base image."""
        dockerfile = (PROJECT_ROOT / "Dockerfile.frontend").read_text()
        match = re.search(r"node:(\d+)\.", dockerfile)
        assert match is not None, "Cannot parse Node version from Dockerfile"
        dockerfile_major = match.group(1)

        ci_node = ci_workflow["jobs"]["frontend-tests"]["strategy"]["matrix"]["node-version"]
        assert str(dockerfile_major) in [str(v) for v in ci_node], (
            f"CI Node version {ci_node} does not match Dockerfile "
            f"base image (node:{dockerfile_major}.x)"
        )

    def test_python_version_matches_pyproject(self, ci_workflow):
        """CI Python must satisfy pyproject.toml's requires-python."""
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert 'requires-python = ">=3.12"' in pyproject

        ci_python = ci_workflow["jobs"]["backend-tests"]["strategy"]["matrix"]["python-version"]
        # At least one version must be 3.12 or newer.
        assert any(float(str(v)) >= 3.12 for v in ci_python), (
            f"CI Python versions {ci_python} must include >=3.12"
        )
