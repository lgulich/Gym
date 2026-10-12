# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nemo_gym.sandbox.providers.base import SandboxExecResult
from resources_servers.harbor import app as harbor
from resources_servers.harbor.tests.test_app import FakeSandbox, make_server, seed_body, verify_body


class LocalSandbox:
    """Run production transfer and verifier commands on a pytest-owned filesystem tree."""

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload(self, source, target):
        shutil.copyfile(source, target)

    async def download(self, source, target):
        shutil.copyfile(source, target)


@pytest.fixture
def verifier_tree(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    root.mkdir()
    logs, tests = root / "logs", root / "tests"
    monkeypatch.setattr(harbor, "TESTS_DIR", str(tests))
    monkeypatch.setattr(harbor, "VERIFIER_LOGS_DIR", str(logs / "verifier"))
    monkeypatch.setattr(harbor, "AGENT_LOGS_DIR", str(logs / "agent"))
    monkeypatch.setattr(harbor, "VERIFIER_STDOUT", str(logs / "verifier" / "test-stdout.txt"))
    return root, logs, tests


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("linked", ["none", "tests", "verifier", "logs"])
async def test_linked_verifier_roots_cannot_preserve_planted_success(tmp_path, monkeypatch, verifier_tree, linked):
    root, logs, tests = verifier_tree
    planted = root / "planted"
    planted.mkdir()
    (planted / "reward.json").write_text('{"reward": 1}')
    (planted / "helper.sh").write_text("echo 1")
    if linked == "logs":
        (planted / "verifier").mkdir()
        (planted / "verifier" / "reward.json").write_text('{"reward": 1}')
        logs.symlink_to(planted, target_is_directory=True)
    else:
        logs.mkdir()
        if linked != "none":
            target = tests if linked == "tests" else logs / "verifier"
            target.symlink_to(planted, target_is_directory=True)
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\nif test -f {shlex.quote(str(tests / 'helper.sh'))}; then\n"
        f"  bash {shlex.quote(str(tests / 'helper.sh'))}\nelse\n  echo 0\nfi "
        f"> {shlex.quote(str(logs / 'verifier' / 'reward.txt'))}\n"
    )
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)), linked)
    assert outcome["reward"] == 0
    assert not outcome.get("mask_sample", False)
    assert not tests.is_symlink() and not logs.is_symlink() and not (logs / "verifier").is_symlink()
    assert not (tests / "helper.sh").exists()
    assert (planted / "reward.json").read_text() == '{"reward": 1}'
    await server.shutdown()


@pytest.mark.parametrize("sign", ["", "-"])
def test_large_numeric_reward_is_cached_measured_zero(tmp_path, monkeypatch, sign):
    sandbox = FakeSandbox(verifier_files={"reward.json": '{"reward": ' + sign + "9" * 400 + "}"})
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        first = client.post("/verify", json=verify_body())
        second = client.post("/verify", json=verify_body())
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.json()["reward"] == 0 and first.json()["mask_sample"] is False
        assert first.json()["failure_kind"] == "harbor:invalid_reward"
        assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_diagnostic_link_does_not_mask_valid_reward(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "private"
    outside.write_text("must not be copied")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\necho 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/diagnostic-link'))}\n"
        "echo useful-output\n"
    )
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)), "logs")
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    artifacts = Path(outcome["verifier_logs_dir"])
    assert not (artifacts / "diagnostic-link").exists()
    assert "useful-output" in (artifacts / "test-stdout.txt").read_text()
    assert all("must not be copied" not in path.read_text() for path in artifacts.rglob("*") if path.is_file())
    await server.shutdown()


def test_timeout_retains_logs_and_caches_measured_zero(tmp_path, monkeypatch):
    sandbox = FakeSandbox(
        verifier_files={"test-stdout.txt": "verifier reached checkpoint"},
        test_result=SandboxExecResult(stdout=None, stderr="timeout", return_code=124, error_type="timeout"),
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        first = client.post("/verify", json=verify_body()).json()
        assert first["reward"] == 0 and first["mask_sample"] is False
        assert first["failure_kind"] == "harbor:verifier_timeout"
        assert first["verifier_return_code"] == 124
        assert "checkpoint" in (Path(first["verifier_logs_dir"]) / "test-stdout.txt").read_text()
        assert client.post("/verify", json=verify_body()).json() == first
        assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1


def test_sandbox_lifetime_covers_episode_and_cleanup(tmp_path, monkeypatch):
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    monkeypatch.setattr(harbor, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    spec = server._sandbox_spec(task, "/app")
    assert spec.ttl_s >= 21600 + 180


@pytest.mark.asyncio
async def test_optional_diagnostic_transfer_failure_preserves_score(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    download = sandbox.download
    calls = 0

    async def fail_diagnostics(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic transfer failed")
        await download(source, target)

    monkeypatch.setattr(sandbox, "download", fail_diagnostics)
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=sandbox, workdir="/app"), "diagnostics")
    assert calls == 2
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    await server.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_reward_link_still_masks_transfer_failure(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "untrusted-reward"
    outside.write_text("1")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests/test.sh").write_text(
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
    )
    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)), "reward"
    )
    assert outcome["reward"] == 0 and outcome["mask_sample"] is True
    assert not (server._artifacts_dir / "reward/reward.txt").exists()
    await server.shutdown()
