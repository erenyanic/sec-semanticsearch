"""
Tests for Cloud Run deployment configuration files.

Validates YAML structure, placeholder consistency, script syntax, and
configuration alignment between service definitions. These tests run
without a GCP account — they only check the files themselves.
"""

import subprocess
from pathlib import Path

import pytest
import yaml

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOUD_DIR = PROJECT_ROOT / "cloud"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def api_service():
    """Load and parse the API service YAML."""
    path = CLOUD_DIR / "api-service.yaml"
    assert path.exists(), f"{path} not found"
    return yaml.safe_load(path.read_text())


@pytest.fixture
def frontend_service():
    """Load and parse the frontend service YAML."""
    path = CLOUD_DIR / "frontend-service.yaml"
    assert path.exists(), f"{path} not found"
    return yaml.safe_load(path.read_text())


# ── Helpers ──────────────────────────────────────────────────────────


def _get_env_dict(container: dict) -> dict[str, str | dict]:
    """Convert a container's env list to a name→value/valueFrom dict."""
    result = {}
    for entry in container.get("env", []):
        if "value" in entry:
            result[entry["name"]] = entry["value"]
        elif "valueFrom" in entry:
            result[entry["name"]] = entry["valueFrom"]
    return result


def _get_container(service: dict, name: str) -> dict:
    """Extract a named container from a service spec."""
    containers = service["spec"]["template"]["spec"]["containers"]
    for c in containers:
        if c.get("name") == name:
            return c
    raise ValueError(f"Container '{name}' not found")


# ── YAML structure tests ─────────────────────────────────────────────


class TestApiServiceYaml:
    """Validate API service YAML structure and required fields."""

    def test_valid_yaml(self, api_service):
        """api-service.yaml parses as valid YAML."""
        assert api_service is not None

    def test_api_version(self, api_service):
        assert api_service["apiVersion"] == "serving.knative.dev/v1"

    def test_kind_is_service(self, api_service):
        assert api_service["kind"] == "Service"

    def test_service_name(self, api_service):
        assert api_service["metadata"]["name"] == "sec-search-api"

    def test_labels(self, api_service):
        labels = api_service["metadata"]["labels"]
        assert labels["app"] == "sec-semantic-search"
        assert labels["component"] == "api"

    def test_gen2_execution_environment(self, api_service):
        annotations = api_service["spec"]["template"]["metadata"]["annotations"]
        assert annotations["run.googleapis.com/execution-environment"] == "gen2"

    def test_gpu_type_nvidia_l4(self, api_service):
        annotations = api_service["spec"]["template"]["metadata"]["annotations"]
        assert annotations["run.googleapis.com/gpu-type"] == "nvidia-l4"

    def test_single_instance_max_scale(self, api_service):
        """Max instances must be 1 (in-memory state, AD#22)."""
        annotations = api_service["spec"]["template"]["metadata"]["annotations"]
        assert annotations["autoscaling.knative.dev/maxScale"] == "1"

    def test_scale_to_zero(self, api_service):
        annotations = api_service["spec"]["template"]["metadata"]["annotations"]
        assert annotations["autoscaling.knative.dev/minScale"] == "0"

    def test_container_concurrency_bounded(self, api_service):
        """The GPU serializes embedding, so a deep per-instance queue
        only lengthens waits; Cloud Run should shed load with 429 instead."""
        concurrency = api_service["spec"]["template"]["spec"]["containerConcurrency"]
        assert 8 <= concurrency <= 16

    def test_container_concurrency_fits_progress_streams(self, api_service):
        """Every open WebSocket holds a slot: leave room for one progress
        stream per queued ingest plus searches and page loads."""
        spec = api_service["spec"]["template"]["spec"]
        env = _get_env_dict(_get_container(api_service, "api"))
        queue = int(env["API_MAX_TASK_QUEUE_SIZE"])
        assert spec["containerConcurrency"] >= queue + 8

    def test_timeout_sufficient_for_ingest(self, api_service):
        """Timeout must be >= 3600s for long-running ingest tasks."""
        timeout = api_service["spec"]["template"]["spec"]["timeoutSeconds"]
        assert timeout >= 3600

    def test_container_port(self, api_service):
        container = _get_container(api_service, "api")
        ports = container["ports"]
        assert any(p["containerPort"] == 8000 for p in ports)

    def test_gpu_resource_limit(self, api_service):
        container = _get_container(api_service, "api")
        limits = container["resources"]["limits"]
        assert limits["nvidia.com/gpu"] == "1"

    def test_memory_sufficient_for_gpu(self, api_service):
        """GPU instances require adequate memory for model + runtime."""
        container = _get_container(api_service, "api")
        memory = container["resources"]["limits"]["memory"]
        # Parse GiB value.
        value = int(memory.replace("Gi", ""))
        assert value >= 16

    def test_startup_probe_targets_health(self, api_service):
        container = _get_container(api_service, "api")
        probe = container["startupProbe"]
        assert probe["httpGet"]["path"] == "/api/health"
        assert probe["httpGet"]["port"] == 8000

    def test_liveness_probe_targets_health(self, api_service):
        container = _get_container(api_service, "api")
        probe = container["livenessProbe"]
        assert probe["httpGet"]["path"] == "/api/health"

    def test_embedding_device_cuda(self, api_service):
        env = _get_env_dict(_get_container(api_service, "api"))
        assert env["EMBEDDING_DEVICE"] == "cuda"

    def test_embedding_batch_size_leverages_gpu(self, api_service):
        """Batch size should be >= 32 to utilise L4 GPU effectively."""
        env = _get_env_dict(_get_container(api_service, "api"))
        assert int(env["EMBEDDING_BATCH_SIZE"]) >= 32

    def test_encryption_key_file_based(self, api_service):
        """Encryption key must use file-based loading (not env var)."""
        env = _get_env_dict(_get_container(api_service, "api"))
        assert "DB_ENCRYPTION_KEY_FILE" in env
        assert "DB_ENCRYPTION_KEY" not in env

    def test_api_keys_from_secret_manager(self, api_service):
        """API keys must be injected from Secret Manager, not hardcoded."""
        env = _get_env_dict(_get_container(api_service, "api"))
        assert isinstance(env["API_KEY"], dict)
        assert "secretKeyRef" in env["API_KEY"]
        assert isinstance(env["API_ADMIN_KEY"], dict)
        assert "secretKeyRef" in env["API_ADMIN_KEY"]

    def test_edgar_session_required(self, api_service):
        env = _get_env_dict(_get_container(api_service, "api"))
        assert env["API_EDGAR_SESSION_REQUIRED"] == "true"

    def test_log_redaction_enabled(self, api_service):
        env = _get_env_dict(_get_container(api_service, "api"))
        assert env["LOG_REDACT_QUERIES"] == "true"

    def test_ticker_persistence_disabled(self, api_service):
        env = _get_env_dict(_get_container(api_service, "api"))
        assert env["DB_TASK_HISTORY_PERSIST_TICKERS"] == "false"

    def test_demo_mode_enabled(self, api_service):
        """Default YAML targets Scenario C (public demo)."""
        env = _get_env_dict(_get_container(api_service, "api"))
        assert env["API_DEMO_MODE"] == "true"

    def test_abuse_prevention_caps_set(self, api_service):
        """Scenario C must have non-zero abuse prevention caps."""
        env = _get_env_dict(_get_container(api_service, "api"))
        assert int(env["API_MAX_TICKERS_PER_REQUEST"]) > 0
        assert int(env["API_MAX_FILINGS_PER_REQUEST"]) > 0
        assert int(env["API_INGEST_COOLDOWN_SECONDS"]) > 0
        assert int(env["API_MAX_TASK_DURATION_MINUTES"]) > 0

    def test_data_volume_in_memory(self, api_service):
        """SQLite and ChromaDB need local-filesystem semantics, not GCS FUSE (F-03)."""
        volumes = api_service["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in volumes if v["name"] == "data-volume")
        assert "csi" not in data_vol
        assert data_vol["emptyDir"]["medium"] == "Memory"

    def test_no_gcs_fuse_volume(self, api_service):
        for volume in api_service["spec"]["template"]["spec"]["volumes"]:
            assert volume.get("csi", {}).get("driver") != "gcsfuse.run.googleapis.com"

    def test_data_volume_size_limit_below_memory_limit(self, api_service):
        """Unbounded, the volume defaults to half the container memory (Cloud Run docs)."""
        volumes = api_service["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in volumes if v["name"] == "data-volume")
        size_gi = int(data_vol["emptyDir"]["sizeLimit"].replace("Gi", ""))
        container = _get_container(api_service, "api")
        memory_gi = int(container["resources"]["limits"]["memory"].replace("Gi", ""))
        assert 0 < size_gi <= memory_gi // 2

    def test_filing_limit_fits_data_volume(self, api_service):
        """DB_MAX_FILINGS must fit the in-memory volume at ~5 MB per filing.

        The 2026-09-16 audit measured 3.3 MB of ChromaDB files per filing
        before chunk overlap and the SQLite ``segments`` table; 5 MB leaves
        room for both. Re-measure after a post-v2 ingest.
        """
        volumes = api_service["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in volumes if v["name"] == "data-volume")
        size_mb = int(data_vol["emptyDir"]["sizeLimit"].replace("Gi", "")) * 1024
        env = _get_env_dict(_get_container(api_service, "api"))
        assert int(env["DB_MAX_FILINGS"]) * 5 <= size_mb

    def test_secret_volume_defined(self, api_service):
        volumes = api_service["spec"]["template"]["spec"]["volumes"]
        secret_vol = next(v for v in volumes if v["name"] == "db-encryption-key")
        assert secret_vol["secret"]["secretName"] == "sec-search-db-encryption-key"

    def test_data_volume_mounted(self, api_service):
        container = _get_container(api_service, "api")
        mounts = {m["name"]: m for m in container["volumeMounts"]}
        assert "data-volume" in mounts
        assert mounts["data-volume"]["mountPath"] == "/app/data"

    def test_secret_volume_mounted_readonly(self, api_service):
        container = _get_container(api_service, "api")
        mounts = {m["name"]: m for m in container["volumeMounts"]}
        assert "db-encryption-key" in mounts
        assert mounts["db-encryption-key"]["readOnly"] is True


class TestFrontendServiceYaml:
    """Validate frontend service YAML structure and required fields."""

    def test_valid_yaml(self, frontend_service):
        assert frontend_service is not None

    def test_api_version(self, frontend_service):
        assert frontend_service["apiVersion"] == "serving.knative.dev/v1"

    def test_kind_is_service(self, frontend_service):
        assert frontend_service["kind"] == "Service"

    def test_service_name(self, frontend_service):
        assert frontend_service["metadata"]["name"] == "sec-search-frontend"

    def test_labels(self, frontend_service):
        labels = frontend_service["metadata"]["labels"]
        assert labels["app"] == "sec-semantic-search"
        assert labels["component"] == "frontend"

    def test_can_scale_multiple_instances(self, frontend_service):
        """Frontend is stateless — should allow multiple instances."""
        annotations = frontend_service["spec"]["template"]["metadata"]["annotations"]
        max_scale = int(annotations["autoscaling.knative.dev/maxScale"])
        assert max_scale > 1

    def test_container_port(self, frontend_service):
        container = _get_container(frontend_service, "frontend")
        ports = container["ports"]
        assert any(p["containerPort"] == 3000 for p in ports)

    def test_node_env_production(self, frontend_service):
        env = _get_env_dict(_get_container(frontend_service, "frontend"))
        assert env["NODE_ENV"] == "production"

    def test_internal_api_url_configured(self, frontend_service):
        env = _get_env_dict(_get_container(frontend_service, "frontend"))
        assert "INTERNAL_API_BASE_URL" in env

    def test_admin_key_from_secret_manager(self, frontend_service):
        """Admin key must come from Secret Manager (AD#49)."""
        env = _get_env_dict(_get_container(frontend_service, "frontend"))
        assert isinstance(env["ADMIN_API_KEY"], dict)
        assert "secretKeyRef" in env["ADMIN_API_KEY"]

    def test_no_gpu_resources(self, frontend_service):
        """Frontend does not need GPU resources."""
        container = _get_container(frontend_service, "frontend")
        limits = container["resources"]["limits"]
        assert "nvidia.com/gpu" not in limits

    def test_startup_probe(self, frontend_service):
        container = _get_container(frontend_service, "frontend")
        probe = container["startupProbe"]
        assert probe["httpGet"]["path"] == "/"
        assert probe["httpGet"]["port"] == 3000


class TestEphemeralDemoData:
    """Demo data lives only as long as the API instance (F-03).

    The nightly reset job, its Cloud Scheduler trigger and the GCS bucket
    existed only to wipe the FUSE-mounted stores. With in-memory storage
    the data is cleared whenever the instance scales to zero or a new
    revision deploys, so none of them may come back.
    """

    def test_reset_job_manifest_removed(self):
        assert not (CLOUD_DIR / "demo-reset-job.yaml").exists()

    def test_deploy_script_has_no_bucket_or_scheduler(self):
        script = (SCRIPTS_DIR / "gcloud-deploy.sh").read_text()
        assert "gcloud storage buckets create" not in script
        assert "gcloud scheduler jobs create" not in script
        assert "demo-reset-job.yaml" not in script

    def test_runtime_account_has_no_storage_role(self):
        """Least privilege: the API no longer touches Cloud Storage."""
        script = (SCRIPTS_DIR / "gcloud-deploy.sh").read_text()
        setup = script.split("do_setup() {", 1)[1].split("\n}", 1)[0]
        assert "roles/storage" not in setup

    def test_entrypoint_creates_data_dirs(self):
        """The in-memory volume hides the image's /app/data subdirectories."""
        entrypoint = (PROJECT_ROOT / "docker-entrypoint.sh").read_text()
        assert "mkdir -p /app/data/chroma_db /app/data/sqlite /app/logs" in entrypoint
        assert entrypoint.index("mkdir -p /app/data") < entrypoint.index("chown app:app")

    def test_demo_banner_does_not_promise_nightly_reset(self):
        banner = PROJECT_ROOT / "frontend" / "src" / "components" / "layout" / "DemoBanner.tsx"
        assert "nightly" not in banner.read_text().lower()


# ── Cross-service consistency tests ──────────────────────────────────


class TestServiceConsistency:
    """Verify consistency between Cloud Run service definitions."""

    def test_admin_key_secret_name_matches(self, api_service, frontend_service):
        """Both services must reference the same admin key secret."""
        api_env = _get_env_dict(_get_container(api_service, "api"))
        fe_env = _get_env_dict(_get_container(frontend_service, "frontend"))

        api_secret = api_env["API_ADMIN_KEY"]["secretKeyRef"]["name"]
        fe_secret = fe_env["ADMIN_API_KEY"]["secretKeyRef"]["name"]
        assert api_secret == fe_secret

    def test_all_services_share_app_label(self, api_service, frontend_service):
        """All resources should share the 'sec-semantic-search' app label."""
        assert api_service["metadata"]["labels"]["app"] == "sec-semantic-search"
        assert frontend_service["metadata"]["labels"]["app"] == "sec-semantic-search"


# ── Placeholder tests ────────────────────────────────────────────────


class TestPlaceholders:
    """Verify placeholder tokens are consistent and replaceable."""

    @pytest.fixture(
        params=[
            "cloud/api-service.yaml",
            "cloud/frontend-service.yaml",
        ]
    )
    def yaml_content(self, request):
        path = PROJECT_ROOT / request.param
        return path.read_text(), request.param

    def test_uses_project_id_placeholder(self, yaml_content):
        """All YAML files must use PROJECT_ID as the placeholder."""
        content, filename = yaml_content
        assert "PROJECT_ID" in content, f"{filename} does not contain PROJECT_ID placeholder"

    def test_no_hardcoded_project_ids(self, yaml_content):
        """YAML files must not contain actual GCP project IDs."""
        content, filename = yaml_content
        # A real project ID is lowercase alphanumeric with hyphens, 6-30 chars.
        # PROJECT_ID (all caps) is the placeholder — that's fine.
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Skip comments and the placeholder token itself.
            stripped = line.strip()
            if stripped.startswith("#") or "PROJECT_ID" in line:
                continue
            # Flag if a line looks like it has a real project ID in
            # a resource reference (e.g. gcr.io/my-real-project/).
            assert "gcr.io/" not in stripped, (
                f"{filename}:{i} may contain a hardcoded project reference"
            )


# ── Shell script syntax tests ────────────────────────────────────────


class TestShellScripts:
    """Validate shell script syntax without executing them."""

    @pytest.fixture(
        params=[
            "scripts/gcloud-deploy.sh",
            "scripts/gcloud-setup-secrets.sh",
            "scripts/demo-reset.sh",
        ]
    )
    def script_path(self, request):
        return PROJECT_ROOT / request.param

    def test_script_exists(self, script_path):
        assert script_path.exists(), f"{script_path} not found"

    def test_script_is_executable(self, script_path):
        assert script_path.stat().st_mode & 0o111, f"{script_path} is not executable"

    def test_script_has_shebang(self, script_path):
        first_line = script_path.read_text().split("\n")[0]
        assert first_line.startswith("#!"), f"{script_path} missing shebang line"

    def test_bash_syntax_valid(self, script_path):
        """Run bash -n to check syntax without executing the script."""
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error in {script_path.name}: {result.stderr}"

    def test_uses_set_euo_pipefail(self, script_path):
        """Scripts should use strict mode for safety."""
        content = script_path.read_text()
        assert "set -e" in content or "set -eu" in content, (
            f"{script_path.name} does not use strict error handling"
        )
