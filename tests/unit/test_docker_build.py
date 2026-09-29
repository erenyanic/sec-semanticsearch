"""
Tests for the API image's layer order and build cache.

A source edit must reuse every dependency layer: PyTorch, the PyPI
dependencies and the baked model come before ``COPY src/``, and the project
is installed as a wheel in the runtime stage's last layer so the multi-GB
venv layer stays identical. PyTorch comes from its own index only
(dependency confusion), and the Cloud Build step caches the ``builder``
stage, which an image's inline cache would not cover.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from tests.unit.test_model_bake import _stages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile.api"


@pytest.fixture(scope="module")
def stages():
    return _stages()


def _index(instructions: list[str], predicate) -> int:
    return next(i for i, line in enumerate(instructions) if predicate(line))


class TestLayerOrder:
    def test_dependencies_and_model_come_before_the_source(self, stages):
        builder = stages["builder"]
        torch = _index(builder, lambda i: "torch>=" in i)
        deps = _index(builder, lambda i: "-r /tmp/requirements.txt" in i)
        bake = _index(builder, lambda i: "id=hf_token" in i)
        source = _index(builder, lambda i: i.startswith("COPY src/"))
        assert torch < deps < bake < source

    def test_source_is_built_as_a_wheel_not_installed_in_the_builder_venv(self, stages):
        builder = stages["builder"]
        after_source = builder[_index(builder, lambda i: i.startswith("COPY src/")) :]
        assert any("pip wheel" in i and "--no-deps" in i for i in after_source)
        assert not any(i.startswith("RUN pip install") for i in after_source)

    def test_runtime_installs_the_wheel_last(self, stages):
        runtime = stages["runtime"]
        venv = _index(runtime, lambda i: i.startswith("COPY --from=builder /opt/venv"))
        wheel = _index(runtime, lambda i: "pip install" in i and "/tmp/wheels" in i)
        runs = [n for n, i in enumerate(runtime) if i.startswith(("RUN ", "COPY "))]
        assert venv < wheel
        assert wheel == runs[-1]
        assert "--no-deps" in runtime[wheel]

    def test_requirements_come_from_pyproject(self, stages):
        """The dependency layer lists pyproject.toml's deps, torch excepted."""
        builder = stages["builder"]
        extract = next(i for i in builder if "tomllib" in i)
        code = re.search(r'python -c "(.*?)" > /tmp/requirements.txt', extract).group(1)
        pyproject = PROJECT_ROOT / "pyproject.toml"
        namespace = {"open": lambda *a, **k: pyproject.open("rb")}
        printed: list[str] = []
        namespace["print"] = printed.append
        exec(code, namespace)  # the command the build runs, against the real file
        requirements = printed[0].splitlines()

        project = tomllib.loads(pyproject.read_text())["project"]
        expected = [d for d in project["dependencies"] if not d.startswith("torch")]
        assert requirements == expected + project["optional-dependencies"]["encryption"]
        assert not any(r.startswith("torch") for r in requirements)


class TestPackageIndexes:
    def test_torch_only_from_its_own_index(self, stages):
        torch = next(i for i in stages["builder"] if "torch>=" in i)
        assert '--index-url "$TORCH_INDEX_URL"' in torch
        assert '"torch>=2.10.0"' in torch  # quoted: unquoted >= is a shell redirect

    def test_no_extra_index_url_anywhere(self):
        instructions = [i for stage in _stages().values() for i in stage]
        assert not any("--extra-index-url" in i for i in instructions)
        ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        install = ci.split("Install backend dependencies", 1)[1].split("- name:", 1)[0]
        assert "--extra-index-url" not in install
        assert "--index-url https://download.pytorch.org/whl/cpu" in install

    def test_no_debian_python_headers(self, stages):
        """The base image ships its interpreter's headers; python3-dev adds another Python's."""
        apt = next(i for i in stages["builder"] if "apt-get install" in i)
        assert "python3-dev" not in apt


class TestDockerignore:
    def test_readme_assets_are_not_sent(self):
        ignored = (PROJECT_ROOT / ".dockerignore").read_text().splitlines()
        assert "assets/" in ignored


class TestCloudBuildCache:
    @pytest.fixture(scope="class")
    def cloudbuild(self):
        workflow = yaml.safe_load(
            (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        )
        step = next(
            s
            for s in workflow["jobs"]["build"]["steps"]
            if s.get("name", "").startswith("Build API image")
        )
        config = step["run"].split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
        return step, yaml.safe_load(config)

    def test_builder_stage_is_built_and_cached_first(self, cloudbuild):
        _, config = cloudbuild
        command = config["steps"][0]["args"][1]
        builder, final = command.split(" && ")
        assert "--target builder" in builder
        assert "--cache-from ${_BUILD_CACHE}" in builder
        assert "-t ${_BUILD_CACHE}" in builder
        assert "--cache-from ${_BUILD_CACHE}" in final
        assert "--cache-from ${_IMAGE}" in final
        assert "-t ${_IMAGE}" in final
        for build in (builder, final):
            assert "--build-arg BUILDKIT_INLINE_CACHE=1" in build
            # Both builds bake the same way; the cache never holds a no-secret layer.
            assert "--build-arg REQUIRE_BAKED_MODEL=1" in build
            assert "--secret id=hf_token,env=HF_TOKEN" in build

    def test_cache_image_is_pushed(self, cloudbuild):
        _, config = cloudbuild
        assert config["images"] == ["${_IMAGE}", "${_BUILD_CACHE}"]

    def test_cache_substitution_is_passed(self, cloudbuild):
        step, _ = cloudbuild
        assert '_BUILD_CACHE="${API_BUILD_CACHE}"' in step["run"]
        assert step["env"]["API_BUILD_CACHE"] == "${{ steps.image-tags.outputs.api_build_cache }}"
