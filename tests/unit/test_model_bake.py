"""
Tests for baking the embedding model into the API image (OPTIMIZATIONS.md F-02).

The Cloud Run service runs with ``HF_HUB_OFFLINE=1`` against weights baked
in at build time, so every link in that chain is checked here: the
Dockerfile bakes through a BuildKit secret and ships only the Hub cache,
the deploy paths make the bake mandatory, the token never reaches a build
arg, an image layer or the runtime environment, and Compose keeps a
runtime-downloaded model across container re-creation.
"""

import re
from pathlib import Path

import pytest
import yaml

from sec_semantic_search.config.settings import EmbeddingSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile.api"


def _instructions(stage_text: str) -> list[str]:
    """Join backslash continuations and drop comments and blank lines."""
    lines: list[str] = []
    current = ""
    for raw in stage_text.splitlines():
        stripped = raw.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        lines.append(current + stripped)
        current = ""
    return lines


def _stages() -> dict[str, list[str]]:
    """Map stage name (``builder``/``runtime``) to its instructions."""
    text = DOCKERFILE.read_text()
    stages: dict[str, list[str]] = {}
    for match in re.finditer(r"^FROM \S+ AS (\w+)\n(.*?)(?=^FROM |\Z)", text, re.M | re.S):
        stages[match.group(1)] = _instructions(match.group(2))
    return stages


def _env(container: dict) -> dict:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container.get("env", [])}


@pytest.fixture(scope="module")
def stages():
    result = _stages()
    assert {"builder", "runtime"} <= result.keys()
    return result


@pytest.fixture(scope="module")
def api_env():
    service = yaml.safe_load((PROJECT_ROOT / "cloud" / "api-service.yaml").read_text())
    containers = service["spec"]["template"]["spec"]["containers"]
    return _env(next(c for c in containers if c["name"] == "api"))


@pytest.fixture(scope="module")
def api_cloudbuild():
    """The inline Cloud Build config the deploy workflow writes for the API image."""
    workflow = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text())
    step = next(
        s
        for s in workflow["jobs"]["build"]["steps"]
        if s.get("name", "").startswith("Build API image")
    )
    config = step["run"].split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    return yaml.safe_load(config)


class TestDockerfileBake:
    def test_bake_uses_buildkit_secret(self, stages):
        runs = [i for i in stages["builder"] if i.startswith("RUN ")]
        assert any("--mount=type=secret,id=hf_token" in r for r in runs)

    def test_token_never_in_arg_or_env(self):
        """ARG and ENV values persist in image metadata and ``docker history``."""
        for instruction in _instructions(DOCKERFILE.read_text()):
            if instruction.startswith(("ARG ", "ENV ")):
                assert "TOKEN" not in instruction.upper(), instruction

    def test_bake_verifies_offline_load(self, stages):
        bake = next(i for i in stages["builder"] if "id=hf_token" in i)
        assert "HF_HUB_OFFLINE=1 python" in bake

    def test_required_bake_fails_without_secret(self, stages):
        assert "ARG REQUIRE_BAKED_MODEL=0" in stages["builder"]
        bake = next(i for i in stages["builder"] if "id=hf_token" in i)
        assert '[ "$REQUIRE_BAKED_MODEL" = "1" ]' in bake
        assert "exit 1" in bake

    def test_runtime_copies_only_hub_cache(self, stages):
        """Token files and the hf_xet chunk cache live outside hub/ and stay behind."""
        copies = [i for i in stages["runtime"] if i.startswith("COPY --from=builder")]
        hf_copies = [c for c in copies if "/opt/hf" in c]
        assert hf_copies == ["COPY --from=builder --chown=app:app /opt/hf/hub /opt/hf/hub"]

    def test_runtime_sets_hf_home(self, stages):
        env_lines = " ".join(i for i in stages["runtime"] if i.startswith("ENV "))
        assert "HF_HOME=/opt/hf" in env_lines

    def test_runtime_does_not_force_offline(self, stages):
        """Offline mode belongs to the Cloud Run manifest; unbaked images must still download."""
        env_lines = " ".join(i for i in stages["runtime"] if i.startswith("ENV "))
        assert "HF_HUB_OFFLINE" not in env_lines

    def test_hf_cache_writable_by_app(self, stages):
        run = next(i for i in stages["runtime"] if "mkdir -p" in i and "/opt/hf" in i)
        assert "chown -R app:app" in run and "/opt/hf" in run.split("chown", 1)[1]

    def test_baked_model_matches_runtime_model(self, stages, api_env):
        """A mismatch would make every offline cold start fail to load the model."""
        arg = next(i for i in stages["builder"] if i.startswith("ARG EMBEDDING_MODEL_NAME="))
        baked = arg.split("=", 1)[1]
        assert baked == EmbeddingSettings.model_fields["model_name"].default
        assert baked == api_env["EMBEDDING_MODEL_NAME"]


class TestCloudRunOffline:
    def test_hub_offline(self, api_env):
        assert api_env["HF_HUB_OFFLINE"] == "1"

    def test_no_hub_token_at_runtime(self, api_env):
        assert "HUGGING_FACE_TOKEN" not in api_env
        assert "HF_TOKEN" not in api_env


class TestDeployBuildsBakedImage:
    def test_cloud_build_passes_secret_and_requires_bake(self, api_cloudbuild):
        step = api_cloudbuild["steps"][0]
        command = " ".join(step["args"])
        assert "--secret id=hf_token,env=HF_TOKEN" in command
        assert "REQUIRE_BAKED_MODEL=1" in command
        assert "DOCKER_BUILDKIT=1" in step.get("env", [])
        assert step["secretEnv"] == ["HF_TOKEN"]

    def test_token_not_expanded_on_command_line(self, api_cloudbuild):
        """``$$HF_TOKEN`` in args would put the token in the process list and build logs."""
        command = " ".join(api_cloudbuild["steps"][0]["args"])
        assert "$HF_TOKEN" not in command
        assert "--build-arg HF_TOKEN" not in command

    def test_token_read_from_secret_manager(self, api_cloudbuild):
        secrets = api_cloudbuild["availableSecrets"]["secretManager"]
        assert secrets == [
            {
                "versionName": "projects/${PROJECT_ID}/secrets/sec-search-hf-token/versions/latest",
                "env": "HF_TOKEN",
            }
        ]

    def test_manual_build_script_bakes(self):
        script = (PROJECT_ROOT / "scripts" / "gcloud-deploy.sh").read_text()
        build = script.split("do_build() {", 1)[1].split("\n}", 1)[0]
        assert "--secret id=hf_token,env=HUGGING_FACE_TOKEN" in build
        assert "REQUIRE_BAKED_MODEL=1" in build
        assert "--build-arg HUGGING_FACE_TOKEN" not in build
        assert "--build-arg HF_TOKEN" not in build

    def test_hf_token_granted_to_build_account_only(self):
        script = (PROJECT_ROOT / "scripts" / "gcloud-setup-secrets.sh").read_text()
        grants = re.findall(r'^grant_access "sec-search-hf-token"(.*)$', script, re.M)
        assert grants == [' "$BUILD_SERVICE_ACCOUNT"']


class TestComposeModelCache:
    def test_hf_cache_volume_mounted_at_hf_home(self):
        compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
        assert "hf_cache:/opt/hf" in compose["services"]["api"]["volumes"]
        assert "hf_cache" in compose["volumes"]

    def test_entrypoint_chowns_hf_cache(self):
        entrypoint = (PROJECT_ROOT / "docker-entrypoint.sh").read_text()
        assert 'HF_CACHE_DIR="${HF_HOME:-/opt/hf}"' in entrypoint
        assert 'mkdir -p "$HF_CACHE_DIR"' in entrypoint
        chown = next(line for line in entrypoint.splitlines() if line.startswith("chown "))
        assert '"$HF_CACHE_DIR"' in chown
