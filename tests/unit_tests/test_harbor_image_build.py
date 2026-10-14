# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dockerfile builds (image mode three): planning, building through a tool, recording, and the loader."""

import json
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.image_build import (
    BuildError,
    BuildPlanEntry,
    BuildResult,
    ImageBuilder,
    build_images,
    detect_tool,
    needs_build,
    normalize_inspect,
    plan_builds,
    record_built_image,
)
from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, HarborTaskError, load_task


TOML = """
schema_version = "1.4"

[task]
name = "org/hello"
version = "1.0.0"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
cpus = 1
memory_mb = 2048
"""

BUILD_DOCKERFILE = "FROM ubuntu:24.04\nCOPY tools/ /opt/tools/\nRUN /opt/tools/setup.sh\nWORKDIR /app\nUSER app\n"
OVERLAY_DOCKERFILE = "FROM ubuntu:24.04\nRUN apt-get install -y jq\nWORKDIR /app\n"
PULL_DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
REPO = "registry.example/team/repo"


def write_task(root: Path, *, dockerfile: str = BUILD_DOCKERFILE, toml: str = TOML, extra: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Do it.\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    for name, text in (extra or {"tools/setup.sh": "#!/bin/sh\necho ok\n"}).items():
        target = root / "environment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


UBUNTU_RECORD = {"os": "linux", "architecture": "amd64", "image": "ubuntu:24.04", "config": {"Env": ["HOME=/root"]}}


def write_base_record(folder: Path, image: str = "ubuntu:24.04") -> None:
    """Record a base image's configuration next to the task folders, as prepare does before planning builds."""
    path = folder / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {**UBUNTU_RECORD, "image": image}
    path.write_text(json.dumps(recorded))


BUILDAH_INSPECT = {
    "FromImageID": "abc123",
    "OCIv1": {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
            "WorkingDir": "/app",
            "User": "app",
            "Cmd": ["/bin/sh"],
            "Labels": {"ignored": "yes"},
        },
    },
}
ARM_BUILDAH_INSPECT = {"FromImageID": "arm001", "OCIv1": {"architecture": "arm64", "os": "linux", "config": {}}}
DOCKER_INSPECT = [
    {
        "Id": "sha256:def456",
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {"Env": ["PATH=/usr/bin"], "WorkingDir": "/srv", "User": "", "Entrypoint": ["/entry"], "Cmd": None},
        "RepoDigests": ["registry.example/team/repo@sha256:feedface"],
    }
]


class FakeTool:
    """Records every command and answers build, push and inspect the way the real tool would."""

    def __init__(self, tool: str, *, fail: str | None = None, inspect: dict | list | None = None) -> None:
        self.tool = tool
        self.fail = fail
        self.inspect = inspect
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        verb = argv[1] if self.tool != "buildah" else next(a for a in argv[1:] if a in ("bud", "push", "inspect"))
        if self.fail and self.fail in argv:
            return subprocess.CompletedProcess(argv, 1, "", f"{verb}: boom\n")
        if verb in ("inspect", "image"):
            if self.inspect is not None:
                data = self.inspect
            else:
                data = BUILDAH_INSPECT if self.tool == "buildah" else DOCKER_INSPECT
            return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")
        if verb == "push":
            if "--digestfile" in argv:
                Path(argv[argv.index("--digestfile") + 1]).write_text("sha256:pushed01\n")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "latest: digest: sha256:pushed02 size: 1234\n", "")
        return subprocess.CompletedProcess(argv, 0, "built\n", "")


def local_docker(taskset: str = "ds") -> tuple[ImageBuilder, FakeTool]:
    tool = FakeTool("docker")
    return ImageBuilder(tool="docker", taskset=taskset, run=tool), tool


def tagged_buildah(tmp_path: Path, **kwargs) -> tuple[ImageBuilder, FakeTool]:
    """A buildah builder that tags for a repository but pushes nothing, so the image stays local."""
    tool = FakeTool("buildah", **kwargs)
    builder = ImageBuilder(tool="buildah", repository=REPO, push=False, storage_root=tmp_path / "store", run=tool)
    return builder, tool


class TestContextDigest:
    def test_stable_and_sensitive_to_content_and_names(self, tmp_path):
        a = write_task(tmp_path / "a")
        b = write_task(tmp_path / "b")
        assert context_digest(a / "environment") == context_digest(b / "environment")
        (b / "environment" / "tools" / "setup.sh").write_text("#!/bin/sh\necho changed\n")
        assert context_digest(a / "environment") != context_digest(b / "environment")
        c = write_task(tmp_path / "c", extra={"tools/other.sh": "#!/bin/sh\necho ok\n"})
        assert context_digest(a / "environment") != context_digest(c / "environment")

    def test_the_executable_bit_is_part_of_the_context(self, tmp_path):
        a = write_task(tmp_path / "a")
        before = context_digest(a / "environment")
        script = a / "environment" / "tools" / "setup.sh"
        script.chmod(script.stat().st_mode | 0o111)
        assert context_digest(a / "environment") != before

    def test_symlinks_hash_their_target_text(self, tmp_path):
        a = write_task(tmp_path / "a")
        (a / "environment" / "link").symlink_to("tools/setup.sh")
        first = context_digest(a / "environment")
        (a / "environment" / "link").unlink()
        (a / "environment" / "link").symlink_to("tools/elsewhere.sh")
        assert context_digest(a / "environment") != first

    def test_requires_a_folder(self, tmp_path):
        with pytest.raises(ValueError, match="Not a build context"):
            context_digest(tmp_path / "missing")


class TestPlan:
    def test_needs_build_classifies_the_three_modes(self):
        records = {"ubuntu:24.04": UBUNTU_RECORD}
        assert needs_build(BUILD_DOCKERFILE, records)
        assert not needs_build(PULL_DOCKERFILE)
        assert needs_build("FROM ubuntu\nRUN <<EOF\ntrue\nEOF\n")
        # An overlay file resolves only against its base image's record: with it, overlay; without, a build.
        assert not needs_build(OVERLAY_DOCKERFILE, records)
        assert needs_build(OVERLAY_DOCKERFILE)
        expanding = "FROM ubuntu:24.04\nRUN x\nWORKDIR $HOME/app\n"
        assert not needs_build(expanding, records) and needs_build(expanding)

    def test_groups_tasks_by_context_and_skips_prebuilt_and_overlay(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")  # same context as a
        write_task(folder / "c", extra={"tools/setup.sh": "#!/bin/sh\necho other\n"})
        write_task(folder / "d", dockerfile=OVERLAY_DOCKERFILE)
        write_task(folder / "e", toml=TOML + 'docker_image = "org/prebuilt:1"\n')
        write_base_record(folder)
        plan = plan_builds(folder)
        assert [entry.tasks for entry in plan] == [("a", "b"), ("c",)]
        assert plan[0].context == (folder / "a" / "environment").resolve()
        assert len(plan[0].short_digest) == 12

    def test_select_and_dataset_overrides_apply(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")
        (folder / "dataset.toml").write_text('[gym.tasks."b".environment]\ndocker_image = "org/pinned:2"\n')
        assert [entry.tasks for entry in plan_builds(folder)] == [("a",)]
        assert plan_builds(folder, select=lambda name: name == "b") == []

    def test_single_task_folder(self, tmp_path):
        task = write_task(tmp_path / "solo")
        (plan,) = plan_builds(task)
        assert plan.tasks == ("solo",)

    def test_detect_tool_needs_docker_without_a_repository(self, monkeypatch):
        monkeypatch.setattr(
            shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("buildah", "podman") else None
        )
        assert detect_tool() is None and detect_tool(REPO) == "buildah"
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name == "docker" else None)
        assert detect_tool() == "docker" and detect_tool(REPO) == "docker"


class TestImageBuilder:
    def test_docker_builds_inspects_and_records_locally(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("docker", inspect=[{**DOCKER_INSPECT[0], "Config": BUILDAH_INSPECT["OCIv1"]["config"]}])
        builder = ImageBuilder(tool="docker", taskset="ds", run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.image == result.tag == f"localhost/nemo-gym-harbor/ds:{result.digest[:12]}"
        assert result.pushed is False and result.tool == "docker"
        build_argv, inspect_argv = tool.calls
        assert build_argv[:4] == ["docker", "build", "--platform", "linux/amd64"]
        assert build_argv[-3:] == [
            "-f",
            str(folder / "a" / "environment" / "Dockerfile"),
            str(folder / "a" / "environment"),
        ]
        assert inspect_argv == ["docker", "image", "inspect", result.tag]
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: {"image": result.tag, "tasks": ["a"], "tool": "docker", "pushed": False}}
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert (
            configs[result.tag]["config"]["User"] == "app" and configs[result.tag]["config_digest"] == "sha256:def456"
        )

    def test_buildah_builds_and_pushes_with_its_storage_flags(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah")
        builder = ImageBuilder(
            tool="buildah", repository=REPO, storage_root=tmp_path / "store", taskset="ds", run=tool
        )
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.tag == f"{REPO}:{result.digest[:12]}"
        assert result.image == f"{REPO}@sha256:pushed01" and result.pushed and result.tool == "buildah"
        build_argv, inspect_argv, push_argv = tool.calls
        assert build_argv[:7] == [
            "buildah",
            "--storage-driver",
            "vfs",
            "--root",
            str(tmp_path / "store" / "storage"),
            "--runroot",
            str(tmp_path / "store" / "run"),
        ]
        assert build_argv[7:12] == ["bud", "--isolation", "chroot", "--platform", "linux/amd64"]
        assert build_argv[-3:] == [
            "-f",
            str(folder / "a" / "environment" / "Dockerfile"),
            str(folder / "a" / "environment"),
        ]
        assert inspect_argv[-3:] == ["--type", "image", result.tag]
        assert push_argv[-2:] == [result.tag, f"docker://{result.tag}"] and "--digestfile" in push_argv
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: {"image": result.image, "tasks": ["a"], "tool": "buildah", "pushed": True}}
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert configs[result.image] == {
            "architecture": "amd64",
            "config": {
                "Cmd": ["/bin/sh"],
                "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
                "User": "app",
                "WorkingDir": "/app",
            },
            "config_digest": "sha256:abc123",
            "image": result.image,
            "os": "linux",
        }

    def test_podman_pushes_with_a_digest_file_and_records_the_pinned_image(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("podman")
        builder = ImageBuilder(tool="podman", repository=REPO, run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.pushed and result.tag == f"{REPO}:{result.digest[:12]}"
        assert result.image == f"{REPO}@sha256:pushed01"
        assert [call[1] for call in tool.calls] == ["build", "image", "push"]
        assert tool.calls[0][:5] == ["podman", "build", "--platform", "linux/amd64", "-t"]
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built[result.digest]["image"] == result.image and built[result.digest]["pushed"] is True
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]["config"]["WorkingDir"] == "/srv"

    def test_docker_push_digest_comes_from_the_push_output(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("docker")
        builder = ImageBuilder(tool="docker", repository=REPO, platform="linux/amd64", run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.image == f"{REPO}@sha256:pushed02"
        assert tool.calls[0][:5] == ["docker", "build", "--platform", "linux/amd64", "-t"]

    def test_an_image_of_another_platform_is_refused_before_push_and_record(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah", inspect=ARM_BUILDAH_INSPECT)
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "s", run=tool)
        with pytest.raises(BuildError, match="came out as linux/arm64, not linux/amd64"):
            build_images(folder, builder=builder, log=lambda _: None)
        assert not any("push" in call for call in tool.calls)
        assert not (folder / BUILT_IMAGES_FILE).exists()
        # Asking for that platform accepts the image.
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64", storage_root=tmp_path / "s", run=tool
        )
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]["architecture"] == "arm64"

    def test_a_recorded_image_is_rebuilt_when_it_does_not_fit_the_run(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        local, _ = local_docker()
        (first,) = build_images(folder, builder=local, log=lambda _: None)
        # A local image cannot serve a run that pushes: it is rebuilt and the record replaced.
        lines: list[str] = []
        pushing = ImageBuilder(tool="podman", repository=REPO, run=FakeTool("podman"))
        (second,) = build_images(folder, builder=pushing, log=lines.append)
        assert lines[0] == f"Rebuilding {first.image} for a: it was built locally and this run pushes to a repository"
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built[first.digest] == {"image": second.image, "tasks": ["a"], "tool": "podman", "pushed": True}
        # A pushed image serves a local run too.
        assert build_images(folder, builder=local, log=lines.append) == []
        assert lines[-1] == f"Built image recorded for a: {second.image}"
        # A recorded image of another platform is rebuilt as well, and its record replaced.
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        configs[second.image]["architecture"] = "arm64"
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps(configs))
        build_images(folder, builder=pushing, log=lines.append)
        assert lines[-2].endswith(": it is linux/arm64, not linux/amd64")
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[second.image]["architecture"] == "amd64"

    def test_rebuild_drops_the_planned_records_first_and_replaces_the_image_config(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        # The rebuilt image carries a different configuration under the same local tag.
        tool.inspect = [{**DOCKER_INSPECT[0], "Id": "sha256:rebuilt", "Config": {"WorkingDir": "/rebuilt"}}]
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, rebuild=True, log=lines.append)
        assert lines[0] == f"Dropped the recorded build {first.digest[:12]} (--rebuild)"
        assert again.digest == first.digest and len([c for c in tool.calls if c[1] == "build"]) == 2
        assert again.image == first.image
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[again.image] == again.record
        assert again.record["config_digest"] == "sha256:rebuilt"

    def test_platform_variants_compare_on_os_and_arch(self, tmp_path):
        from nemo_gym.tasks.harbor.image_build import platform_key, recorded_image_mismatch

        assert platform_key("linux/arm64/v8") == "linux/arm64" and platform_key(None) == ""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah", inspect=ARM_BUILDAH_INSPECT)
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64/v8", storage_root=tmp_path / "s", run=tool
        )
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[result.digest]
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/arm64/v8") is None
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/amd64") is not None

    def test_pushing_needs_a_repository_and_unknown_tools_are_refused(self, tmp_path):
        with pytest.raises(BuildError, match="needs --build-repository"):
            ImageBuilder(tool="docker", push=True, run=FakeTool("docker"))
        with pytest.raises(BuildError, match="Unknown build tool"):
            ImageBuilder(tool="kaniko", run=FakeTool("docker"))
        # buildah and podman keep images where no sandbox provider looks; without a push target they are refused.
        for tool in ("buildah", "podman"):
            with pytest.raises(BuildError, match=f"{tool} keeps built images.*--build-repository"):
                ImageBuilder(tool=tool, run=FakeTool(tool))
        # A repository without a push is allowed: the image is tagged for it and kept local.
        assert ImageBuilder(tool="buildah", repository=REPO, push=False, run=FakeTool("buildah")).push is False

    def test_no_tool_on_path_is_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(BuildError, match=r"No image builder found: install docker \(buildah and podman need"):
            ImageBuilder()
        with pytest.raises(BuildError, match="No image builder found: install buildah, podman, docker"):
            ImageBuilder(repository=REPO)

    def test_a_failed_build_records_nothing_and_names_the_tasks(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b", extra={"tools/setup.sh": "#!/bin/sh\necho b\n"})
        tool = FakeTool("docker", fail=str(folder / "b" / "environment"))
        builder = ImageBuilder(tool="docker", run=tool, taskset="ds")
        with pytest.raises(BuildError, match=r"Building .* for task\(s\) b failed \(exit 1\)") as info:
            build_images(folder, builder=builder, log=lambda _: None)
        assert "boom" in str(info.value)
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        # The first context was recorded before the second failed; the failed one left no trace.
        assert [entry["tasks"] for entry in built.values()] == [["a"]]

    def test_recorded_contexts_are_not_rebuilt(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        build_images(folder, builder=builder, log=lambda _: None)
        lines: list[str] = []
        assert build_images(folder, builder=builder, log=lines.append) == []
        expected = f"Built image recorded for a: localhost/nemo-gym-harbor/ds:{plan_builds(folder)[0].short_digest}"
        assert len(tool.calls) == 2 and lines == [expected]

    def test_record_keeps_existing_entries(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps({"other:1": {"os": "linux"}}))
        result = BuildResult(
            digest="d" * 64, tasks=("a",), tag="t:1", image="t:1", tool="docker", pushed=False, record={"os": "linux"}
        )
        record_built_image(folder, result)
        later = BuildResult(
            digest="d" * 64, tasks=("z",), tag="t:9", image="t:9", tool="docker", pushed=False, record={}
        )
        record_built_image(folder, later)
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())["d" * 64]["image"] == "t:1"
        assert set(json.loads((folder / IMAGE_CONFIGS_FILE).read_text())) == {"other:1", "t:1"}
        # Overwriting replaces both the build entry and the image's record.
        record_built_image(folder, later, overwrite=True)
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())["d" * 64]["image"] == "t:9"
        assert "t:9" in json.loads((folder / IMAGE_CONFIGS_FILE).read_text())

    def test_normalize_inspect_rejects_unexpected_shapes_and_warns_off_platform(self, caplog):
        with pytest.raises(BuildError, match="no OCIv1"):
            normalize_inspect("buildah", {"nope": 1})
        with pytest.raises(BuildError, match="no Config"):
            normalize_inspect("docker", [])
        arm = [{"Id": "sha256:1", "Architecture": "arm64", "Os": "linux", "Config": {}}]
        with caplog.at_level("WARNING"):
            record = normalize_inspect("podman", arm)
        assert record["architecture"] == "arm64" and "not linux/amd64" in caplog.text

    def test_build_argv_for_buildah_uses_a_platform_when_given(self, tmp_path):
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64", storage_root=tmp_path, run=FakeTool("buildah")
        )
        entry = BuildPlanEntry("a" * 64, tmp_path / "ctx", ("t",))
        argv = builder.build_argv(entry, "x:1")
        assert argv[argv.index("--platform") + 1] == "linux/arm64"


class TestLoaderBuildMode:
    def test_a_built_image_is_picked_up_in_pull_mode(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, _ = tagged_buildah(tmp_path)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        task = load_task(folder / "a")
        assert task.image == result.tag
        assert task.overlay == ()
        # WORKDIR, ENV and USER come from the built image's recorded configuration.
        assert task.workdir == "/app" and task.user == "app" and task.sandbox_user == "app"
        assert task.env == {"PATH": "/usr/bin:/bin", "APP_HOME": "/app"}
        assert task.needs_sandbox

    def test_task_toml_settings_win_over_the_built_image(self, tmp_path):
        folder = tmp_path / "ds"
        toml = TOML + 'workdir = "/work"\n\n[environment.env]\nAPP_HOME = "/elsewhere"\n'
        write_task(folder / "a", toml=toml)
        builder, _ = tagged_buildah(tmp_path)
        build_images(folder, builder=builder, log=lambda _: None)
        task = load_task(folder / "a")
        assert task.workdir == "/work" and task.env["APP_HOME"] == "/elsewhere"

    def test_without_a_recorded_build_the_error_says_how_to_build(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        with pytest.raises(HarborTaskError, match=r"needs a build \(COPY\).*built-images.json.*--build-tool") as info:
            load_task(folder / "a")
        assert "--no-build" in str(info.value) and "docker_image" in str(info.value)

    def test_a_changed_context_needs_a_new_build(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, _ = tagged_buildah(tmp_path)
        build_images(folder, builder=builder, log=lambda _: None)
        (folder / "a" / "environment" / "tools" / "setup.sh").write_text("#!/bin/sh\necho changed\n")
        with pytest.raises(HarborTaskError, match="needs a build"):
            load_task(folder / "a")


@pytest.mark.skipif(shutil.which("buildah") is None, reason="buildah is not installed")
def test_buildah_builds_a_real_image_rootless(tmp_path):
    folder = tmp_path / "ds"
    write_task(
        folder / "a",
        dockerfile="FROM docker.io/library/alpine:3.20\nCOPY note.txt /note.txt\nWORKDIR /work\nENV GREETING=hi\n",
        extra={"note.txt": "hello\n"},
    )
    # The host's own platform, so the build needs no emulation; the tagged repository is never pushed.
    host = "linux/arm64" if platform.machine() in ("aarch64", "arm64") else "linux/amd64"
    builder = ImageBuilder(
        tool="buildah",
        repository="localhost/nemo-gym-test",
        push=False,
        platform=host,
        storage_root=tmp_path / "store",
    )
    (result,) = build_images(folder, builder=builder, log=lambda _: None)
    assert result.tag.startswith("localhost/nemo-gym-test:") and result.pushed is False
    record = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]
    assert f"{record['os']}/{record['architecture']}" == host
    assert record["config"]["WorkingDir"] == "/work"
    assert "GREETING=hi" in record["config"]["Env"]
    assert record["config_digest"].startswith("sha256:")
    task = load_task(folder / "a")
    assert task.image == result.image and task.workdir == "/work" and task.env["GREETING"] == "hi"


class TestPrepareBuilds:
    """``prepare_target`` builds what needs building before the tasks load."""

    @pytest.fixture
    def fake_builder(self, monkeypatch):
        import nemo_gym.tasks.harbor.cli as harbor_cli

        made: list[dict] = []

        def factory(**kwargs):
            made.append(dict(kwargs))
            tool = kwargs.pop("tool", None) or "docker"
            return ImageBuilder(tool=tool, run=FakeTool(tool), **kwargs)

        monkeypatch.setattr(harbor_cli, "ImageBuilder", factory)
        return made

    def test_prepare_builds_once_and_the_tasks_load_in_pull_mode(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.cli import BuildOptions, prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )
        assert fake_builder == [{"tool": "podman", "repository": "reg.example/r", "platform": None, "taskset": "ds"}]
        assert [t.task_id for t in prepared.tasks] == ["a", "b", "c"]
        built = next(t for t in prepared.tasks if t.task_id == "a")
        assert built.image == "reg.example/r@sha256:pushed01" and built.overlay == ()
        assert prepared.images == ("reg.example/r@sha256:pushed01", "ubuntu:24.04")
        assert json.loads((tmp_path / "out" / "ds" / "images.json").read_text()) == list(prepared.images)
        out = capsys.readouterr().out
        assert "Built reg.example/r@sha256:pushed01 (pushed, podman) for task(s) a, b" in out
        assert "Built 1 image(s) with podman, pushed 1" in out
        assert "Images: 2 distinct" in out
        # A second prepare finds the build recorded and builds nothing.
        fake_builder.clear()
        prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )
        assert "Built image recorded for a, b" in capsys.readouterr().out

    def test_no_build_leaves_the_tasks_out_with_the_loader_message(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.cli import BuildOptions, prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert fake_builder == []
        assert [t.task_id for t in prepared.tasks] == ["c"]
        assert "needs a build (COPY)" in prepared.skipped["a"]
        assert (
            "Skipping Dockerfile builds (--no-build); tasks that need one are left out: a" in capsys.readouterr().out
        )
        # Once the build is recorded, --no-build says nothing and the task runs from the recorded image.
        build_images(folder, builder=local_docker()[0], log=lambda _: None)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["a", "c"]
        assert "--no-build" not in capsys.readouterr().out

    def test_a_failed_build_stops_prepare_naming_the_tasks(self, tmp_path, monkeypatch):
        import nemo_gym.tasks.harbor.cli as harbor_cli
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        monkeypatch.setattr(
            harbor_cli,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="docker", run=FakeTool("docker", fail=str(folder / "a" / "environment"))),
        )
        with pytest.raises(ValueError, match=r"Could not build the Dockerfile image\(s\) for task\(s\) a: Building"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_prepare_passes_rebuild_through(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.cli import BuildOptions, prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="docker"))
        prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="docker", rebuild=True))
        out = capsys.readouterr().out
        assert "Dropped the recorded build" in out and out.count("Built localhost/nemo-gym-harbor/ds:") == 2

    def test_build_options_from_args_and_environment(self, monkeypatch):
        import argparse

        from nemo_gym.tasks.harbor.cli import BuildOptions

        monkeypatch.setenv("NEMO_GYM_BUILD_REPOSITORY", "env.example/repo")
        options = BuildOptions.from_args(argparse.Namespace(no_build=False, build_tool=None, build_repository=None))
        assert options == BuildOptions(enabled=True, tool=None, repository="env.example/repo", platform=None)
        options = BuildOptions.from_args(
            argparse.Namespace(
                no_build=True,
                build_tool="buildah",
                build_repository="flag.example/r",
                build_platform="linux/amd64",
                rebuild=True,
            )
        )
        assert options == BuildOptions(
            enabled=False, tool="buildah", repository="flag.example/r", platform="linux/amd64", rebuild=True
        )
