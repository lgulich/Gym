# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import tarfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.materialize import materialize_task
from resources_servers.harbor.app import HarborResourcesServer, resolve_env_templates
from resources_servers.harbor.task_data import TaskData
from resources_servers.harbor.tests.test_app import TASK_TOML, make_server, seed_body, verify_body


def request() -> Request:
    return Request({"type": "http", "session": {}})


def refresh_task(server, task):
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    return task


@pytest.mark.parametrize("section", ["environment", "verifier"])
def test_missing_env_fails_before_provisioning(tmp_path, monkeypatch, section):
    monkeypatch.delenv("HARBOR_TEST_REQUIRED", raising=False)
    toml = TASK_TOML + f'\n[{section}.env]\nTOKEN = "${{HARBOR_TEST_REQUIRED}}"\n'
    server, task, _, created = make_server(tmp_path, monkeypatch, toml=toml)
    response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
    assert response.status_code == 422 and "HARBOR_TEST_REQUIRED" in response.text
    assert created == []


def test_templates_resolve_at_use_without_changing_materialized_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "agent-secret")
    monkeypatch.setenv("HARBOR_TEST_EMPTY", "")
    monkeypatch.delenv("HARBOR_TEST_UNSET", raising=False)
    template = {
        "token": "${HARBOR_TEST_TOKEN}",
        "empty": "${HARBOR_TEST_EMPTY:-fallback}",
        "default": "${HARBOR_TEST_UNSET:-fallback}",
        "literal": "prefix-${HARBOR_TEST_TOKEN}",
    }
    assert resolve_env_templates(template) == {
        "token": "agent-secret",
        "empty": "",
        "default": "fallback",
        "literal": "prefix-${HARBOR_TEST_TOKEN}",
    }
    toml = (
        TASK_TOML
        + '\n[environment.env]\nTOKEN = "${HARBOR_TEST_TOKEN}"\n[verifier.env]\nTOKEN = "${HARBOR_TEST_TOKEN}"\n'
    )
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
    monkeypatch.setattr(
        "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    assert server._sandbox_spec(task, "/app").env == {"TOKEN": "agent-secret"}
    client = TestClient(server.setup_webserver())
    response = client.post("/seed_session", json=seed_body(task))
    assert response.status_code == 200 and "agent-secret" not in response.text
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "verifier-secret")
    response = client.post("/verify", json=verify_body())
    assert response.json()["reward"] == 1
    run = next(call for call in sandbox.execs if "test.sh" in call["command"])
    assert run["env"] == {"TOKEN": "verifier-secret"}
    assert task.env["TOKEN"] == "${HARBOR_TEST_TOKEN}"
    assert "agent-secret" not in materialize_task(task, "ds").model_dump_json()
    assert "verifier-secret" not in (server._sessions["rs-1"].task.path / "task.toml").read_text()


@pytest.mark.parametrize("variant", ["agent_user", "dockerfile_user"])
def test_unsupported_semantics_fail_before_provisioning(tmp_path, monkeypatch, variant):
    # Separate verification is supported since M10; agent user overrides arrive with overlay images.
    toml = TASK_TOML
    dockerfile = "FROM ubuntu:24.04\nWORKDIR /app"
    if variant == "agent_user":
        toml = toml.replace("[agent]", '[agent]\nuser = "nobody"')
    else:
        dockerfile += "\nUSER nobody"
    server, task, _, created = make_server(tmp_path, monkeypatch, toml=toml, dockerfile=dockerfile)
    response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
    assert response.status_code == 422 and "not supported" in response.text
    assert created == []


# A Compose file makes the task a Compose group (M13), covered by TestComposeAndRouting.
@pytest.mark.parametrize("build_file", [None, "Dockerfile"])
def test_prebuilt_environment_files_only_uploaded_without_build_definition(tmp_path, monkeypatch, build_file):
    toml = TASK_TOML.replace("[environment]", '[environment]\ndocker_image = "ubuntu:24.04"')
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
    env = task.path / "environment"
    (env / "Dockerfile").unlink()
    if build_file:
        (env / build_file).write_text("FROM ubuntu:24.04" if build_file == "Dockerfile" else "services: {}")
    (env / "input.txt").write_text("task input")
    task = refresh_task(server, task)
    response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
    assert response.status_code == 200, response.text
    assert len(sandbox.uploaded_archives) == (0 if build_file else 1)
    if not build_file:
        with tarfile.open(fileobj=io.BytesIO(sandbox.uploaded_archives[0])) as archive:
            assert archive.extractfile("input.txt").read() == b"task input"


def test_live_folder_edits_do_not_change_seeded_verifier(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    original = (task.path / "tests" / "test.sh").read_bytes()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_body(task)).status_code == 200
    (task.path / "tests" / "test.sh").write_text("tampered verifier")
    assert client.post("/seed_session", json=seed_body(task, session="rs-2")).status_code == 200
    assert client.post("/verify", json=verify_body()).json()["reward"] == 1
    with tarfile.open(fileobj=io.BytesIO(sandbox.uploaded_archives[-1])) as archive:
        script = next(name for name in archive.getnames() if name.lstrip("./") == "test.sh")
        assert archive.extractfile(script).read() == original
    assert server._sessions["rs-1"].task.path == server._sessions["rs-2"].task.path


@pytest.mark.asyncio
async def test_concurrent_first_seeds_share_snapshot_even_if_one_is_cancelled(tmp_path, monkeypatch):
    server, task, _, created = make_server(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    copies = []
    original = server._snapshot_task

    def snapshot(folder, expected):
        copies.append(folder)
        entered.set()
        assert release.wait(10)
        return original(folder, expected)

    monkeypatch.setattr(server, "_snapshot_task", snapshot)
    first = asyncio.create_task(server.seed_session(request(), ResourcesSeedSessionRequest(**seed_body(task))))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(
            server.seed_session(request(), ResourcesSeedSessionRequest(**seed_body(task, session="rs-2")))
        )
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        release.set()
    assert (await second).resources_session_id == "rs-2"
    assert copies == [task.path] and len(created) == 1
    await server.shutdown()
    assert not server._tasks and not server._locks


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["start", "prepare", "serialize"])
async def test_cancelled_seed_stops_owned_sandbox(tmp_path, monkeypatch, stage):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    entered = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    if stage == "start":
        monkeypatch.setattr("resources_servers.harbor.app.AsyncSandbox", lambda config: sandbox)
        monkeypatch.setattr("resources_servers.harbor.app.resolve_provider_config", lambda *args: {})
        monkeypatch.setattr("resources_servers.harbor.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr(server, "_sandbox_spec", lambda *args: None)
        monkeypatch.setattr(sandbox, "start", block, raising=False)
        monkeypatch.setattr(server, "_create_sandbox", HarborResourcesServer._create_sandbox.__get__(server))
    elif stage == "prepare":
        monkeypatch.setattr(server, "_prepare_workdir", block)
    else:
        monkeypatch.setattr(sandbox, "serialize", block)
    seed = asyncio.create_task(server.seed_session(request(), ResourcesSeedSessionRequest(**seed_body(task))))
    await asyncio.wait_for(entered.wait(), 5)
    seed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await seed
    assert sandbox.stopped and not server._sandboxes and not server._sessions and not server._locks
    await server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["close", "shutdown"])
async def test_failed_cleanup_keeps_ownership_until_success(tmp_path, monkeypatch, retry, caplog):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    attempts = []

    async def fail_setup(*args):
        raise RuntimeError("setup failed")

    async def stop():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("stop temporarily unavailable")
        sandbox.stopped = True

    monkeypatch.setattr(server, "_prepare_workdir", fail_setup)
    monkeypatch.setattr(sandbox, "stop", stop)
    with pytest.raises(HTTPException, match="setup failed") as error:
        await server.seed_session(request(), ResourcesSeedSessionRequest(**seed_body(task)))
    assert error.value.status_code == 503
    assert server._sandboxes["rs-1"] is sandbox and not server._sessions
    assert "Failed to stop Harbor sandbox" in caplog.text
    if retry == "close":
        await server.close_resources_session(
            request(),
            ResourcesCloseSessionRequest(resources_session_id="rs-1", episode_id={"rollout_id": "r1", "attempt": 0}),
        )
    await server.shutdown()
    assert sandbox.stopped and len(attempts) == 2 and not server._sandboxes


@pytest.mark.asyncio
async def test_close_waits_for_seed_and_keeps_single_lock(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    original = server._prepare_workdir

    async def prepare(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(server, "_prepare_workdir", prepare)
    seed = asyncio.create_task(server.seed_session(request(), ResourcesSeedSessionRequest(**seed_body(task))))
    await asyncio.wait_for(entered.wait(), 5)
    close = asyncio.create_task(
        server.close_resources_session(
            request(),
            ResourcesCloseSessionRequest(resources_session_id="rs-1", episode_id={"rollout_id": "r1", "attempt": 0}),
        )
    )
    await asyncio.sleep(0)
    assert not close.done() and not sandbox.stopped
    release.set()
    await asyncio.gather(seed, close)
    assert sandbox.stopped and not server._locks and not server._sessions
    await server.shutdown()


def test_closed_history_is_bounded_and_unknown_requests_leave_no_locks(tmp_path, monkeypatch):
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    server.config.closed_session_cache_size = 3
    client = TestClient(server.setup_webserver())
    for i in range(12):
        assert (
            client.post("/seed_session", json=seed_body(task, session=f"bad-{i}", task_id="missing")).status_code
            == 404
        )
        assert client.post("/verify", json=verify_body()).status_code == 404
        assert (
            client.post(
                "/close_session",
                json={"resources_session_id": str(i), "episode_id": {"rollout_id": "r1", "attempt": 0}},
            ).status_code
            == 200
        )
        assert not server._locks
    assert list(server._closed) == ["9", "10", "11"]
    assert client.post("/seed_session", json=seed_body(task, session="11")).status_code == 409


@pytest.mark.parametrize("artifacts", [None, Path("custom/verifier")])
def test_logs_use_global_results_root_across_working_directories(tmp_path, monkeypatch, artifacts):
    old, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    results = tmp_path / "caller" / "results"
    component = tmp_path / "component"
    component.mkdir()
    monkeypatch.setattr("resources_servers.harbor.app.get_global_config_dict", lambda: {"results_dir": str(results)})
    monkeypatch.chdir(component)
    config = old.config.model_copy(update={"artifacts_dir": artifacts})
    server = HarborResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_create_sandbox", old._create_sandbox)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_body(task)).status_code == 200
    payload = client.post("/verify", json=verify_body()).json()
    expected = results / (artifacts or "harbor/verifier")
    assert Path(payload["verifier_logs_dir"]).parent == expected
    monkeypatch.chdir(tmp_path)
    assert (Path(payload["verifier_logs_dir"]) / "reward.txt").read_text() == "1\n"
    assert not (component / "results").exists()


def test_task_data_requires_digest_and_accepts_materialized_row(tmp_path, monkeypatch):
    _, task, _, _ = make_server(tmp_path, monkeypatch)
    row = materialize_task(task, "ds").model_dump(mode="json")
    data = TaskData.model_validate(row["task_input"]["task_data"])
    assert data.ng_digest == task.digest
    assert DIGEST_KEY in TaskData.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        TaskData.model_validate({})
