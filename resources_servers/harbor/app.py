# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources server for Harbor-format tasks.

One server instance serves one or more tasksets. Each taskset maps to a folder of
task folders in Gym's shared datasets folder plus the digest every task had when its
rows were materialized (see ``nemo_gym.tasks.harbor``).

- ``/seed_session`` checks the row's digest against the mapping and the folder,
  starts the task's image as a sandbox, creates the working directory, and hands the
  agent a ``SandboxAccess``.
- ``/verify`` runs ``tests/test.sh`` inside that same sandbox (Harbor's shared
  verifier mode), then reads ``/logs/verifier/reward.json`` or ``reward.txt``.
- ``/close_session`` stops the sandbox.

Reward rule: when ``test.sh`` ran, the sample is measured. A missing or invalid reward
file scores 0 with a ``failure_kind``. Only a Gym-side failure (sandbox lost, transfer
failed) masks the sample.
"""

import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import RESULTS_DIR, failure_kinds
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.task import HarborTaskError
from resources_servers.harbor.sandbox_io import _exec_as_root, download_dir, upload_dir


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"
VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"

# Namespaced failure kinds for outcomes the shared vocabulary does not name.
VERIFIER_TIMEOUT_KIND = "harbor:verifier_timeout"
MISSING_REWARD_KIND = "harbor:missing_reward"
INVALID_REWARD_KIND = "harbor:invalid_reward"


class HarborVerifyRequest(BaseVerifyRequest):
    """The flat verify body the environment server posts: the row's ``task_data`` keys beside the params
    and the response. The session cookie identifies the episode; the digest, when present, must match it.
    """

    model_config = ConfigDict(populate_by_name=True)

    ng_digest: str | None = Field(default=None, alias=DIGEST_KEY)


class HarborVerifyResponse(BaseVerifyResponse):
    """The verify response plus what the Harbor verifier produced.

    ``verifier_rewards`` is the parsed ``reward.json`` (``reward.txt`` becomes ``{"reward": value}``).
    The ``verifier_*`` fields are ``None`` when ``test.sh`` never ran.
    """

    verifier_rewards: dict[str, float] | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None


class HarborTasksetConfig(BaseModel):
    """Where one taskset's folders live and which digest each task was materialized with."""

    model_config = ConfigDict(extra="forbid")

    folder: Path
    tasks: dict[str, str]


class HarborResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    num_workers: Literal[1] = 1
    tasksets: dict[str, HarborTasksetConfig] = Field(default_factory=dict)
    # Name of the top-level sandbox provider block in the merged config.
    sandbox_provider: str = "sandbox"
    sandbox_ready_timeout_s: float = Field(default=900, gt=0)
    # Cover the owning environment's full episode (including setup/queueing) and final cleanup.
    episode_timeout_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    cleanup_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    sandbox_ttl_slack_s: float = Field(default=900, ge=0)
    sandbox_provider_options: dict[str, Any] = Field(default_factory=dict)
    sandbox_metadata: dict[str, str] = Field(default_factory=dict)
    # Replaces the task's declared cpus, memory_mb and storage_mb when set (keys: cpu, memory_mib, disk_gib).
    # Used to run a benchmark at fixed resources, for example to compare against another server.
    sandbox_resources_override: dict[str, Any] | None = None
    # Export CPU-count env vars (OMP_NUM_THREADS and friends) matching the sandbox CPU limit.
    derive_cpu_env: bool = True
    # Operator environment for every sandbox; a task's own `[environment.env]` wins.
    sandbox_env: dict[str, str] = Field(default_factory=dict)
    # Shell commands run as root in every new sandbox before the agent sees it, for repairs the
    # operator owns rather than the task (for example pointing an end-of-life distro at an archive).
    # A failing command fails the seed with a retryable 503.
    sandbox_setup_commands: list[str] = Field(default_factory=list)
    sandbox_setup_timeout_s: float = Field(default=600, gt=0)
    # Extra seconds granted to `test.sh` beyond `[verifier].timeout_sec` before the in-container `timeout` kills it.
    verifier_grace_s: float = Field(default=30, ge=0)
    # Verifier logs are downloaded here, one folder per resources session.
    artifacts_dir: Path | None = None
    # Retain recent close tombstones while releasing idle session locks immediately.
    closed_session_cache_size: int = Field(default=10000, ge=1)


@dataclass
class HarborSession:
    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str
    # The first verify's outcome; a retried /verify replays it instead of re-running test.sh.
    verify_outcome: dict[str, Any] | None = None
    verify_error: tuple[int, str] | None = None


@dataclass
class _SessionLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    """Resolve Harbor's whole-value host templates without changing the stored task."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif match.group(1) in os.environ:
            resolved[key] = os.environ[match.group(1)]
        elif match.group(2) is not None:
            resolved[key] = match.group(2)
        else:
            raise HTTPException(422, f"Environment variable {match.group(1)!r} is required by the task")
    return resolved


def parse_reward_file(directory: Path) -> tuple[dict[str, float] | None, str | None]:
    """Read reward.json when present; only a missing JSON file falls back to reward.txt."""
    for path in (directory / "reward.json", directory / "reward.txt"):
        if path.is_file():
            return _parse_one_reward_file(path)
    return None, "no reward.json or reward.txt was written"


def _parse_one_reward_file(path: Path) -> tuple[dict[str, float] | None, str | None]:
    # The verifier wrote these bytes; never let a stray non-UTF-8 byte escape as an exception.
    text = path.read_text(errors="replace").strip()
    if not text:
        return None, f"{path.name} is empty"
    try:
        raw = json.loads(text) if path.suffix == ".json" else {"reward": float(text)}
    except ValueError as exc:
        return None, f"{path.name} is not valid: {exc}"
    if not isinstance(raw, dict) or not raw:
        return None, f"{path.name} must hold a non-empty JSON object"
    rewards: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            return None, f"{path.name}: {key!r} is not a finite number"
        try:
            value = float(value)
        except OverflowError:
            return None, f"{path.name}: {key!r} is not a finite number"
        if not math.isfinite(value):
            return None, f"{path.name}: {key!r} is not a finite number"
        rewards[str(key)] = value
    return rewards, None


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


def _sandbox_resources(task: HarborTask) -> dict[str, Any]:
    environment = task.config.environment
    resources: dict[str, Any] = {}
    if environment.cpus is not None:
        resources["cpu"] = environment.cpus
    if environment.memory_mb is not None:
        resources["memory_mib"] = environment.memory_mb
    if environment.storage_mb is not None:
        resources["disk_gib"] = max(1, math.ceil(environment.storage_mb / 1024))
    if environment.gpus:
        resources["gpu"] = environment.gpus
        if environment.gpu_types:
            resources["gpu_type"] = environment.gpu_types[0]
    return resources


class HarborResourcesServer(SimpleResourcesServer):
    config: HarborResourcesServerConfig
    ray_enabled = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: dict[str, HarborSession] = {}
        self._locks: dict[str, _SessionLock] = {}
        # Ownership starts before sandbox.start(), and ends only after stop() succeeds.
        self._sandboxes: dict[str, AsyncSandbox] = {}
        self._tasks: dict[tuple[Path, str], asyncio.Task[HarborTask]] = {}
        self._snapshots = tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-tasks-")
        self._closed: OrderedDict[str, None] = OrderedDict()
        self._shutting_down = False
        artifacts = self.config.artifacts_dir
        if artifacts is None or not artifacts.is_absolute():
            results_root = Path(get_global_config_dict().get("results_dir", RESULTS_DIR)).resolve()
            artifacts = results_root / (artifacts if artifacts is not None else "harbor/verifier")
        self._artifacts_dir = artifacts.resolve()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    @asynccontextmanager
    async def _session_lock(self, session_id: str) -> AsyncIterator[None]:
        entry = self._locks.setdefault(session_id, _SessionLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            # Waiters already hold this entry; never replace their lock with a new one.
            if entry.users == 0:
                del self._locks[session_id]

    async def _stop_sandbox(self, session_id: str) -> None:
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            await sandbox.stop()
            del self._sandboxes[session_id]
        self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        self._shutting_down = True
        # Include seeds still loading a task, before they have acquired a sandbox.
        for session_id in list(self._sandboxes.keys() | self._locks.keys()):
            async with self._session_lock(session_id):
                try:
                    await self._stop_sandbox(session_id)
                except Exception:
                    LOGGER.exception("Failed to stop abandoned Harbor sandbox")
        pending = [task for task in self._tasks.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not self._sandboxes:
            await asyncio.to_thread(self._snapshots.cleanup)
            self._tasks.clear()

    def _snapshot_task(self, folder: Path, expected: str) -> HarborTask:
        snapshot_root = Path(tempfile.mkdtemp(dir=self._snapshots.name))
        try:
            snapshot = snapshot_root / folder.name
            shutil.copytree(folder, snapshot, symlinks=True)
            task = load_task(snapshot)
            if task.digest != expected:
                raise HTTPException(
                    409, f"Task folder {folder} changed since materialization; re-materialize the taskset"
                )
            return task
        except (OSError, HarborTaskError) as exc:
            shutil.rmtree(snapshot_root)
            raise HTTPException(422, f"Could not snapshot task {folder}: {exc}") from exc
        except BaseException:
            shutil.rmtree(snapshot_root)
            raise

    # -- task resolution -------------------------------------------------------------------

    async def _resolve_task(self, task_id: TaskId, task_data: dict[str, Any]) -> HarborTask:
        taskset = self.config.tasksets.get(task_id.taskset)
        if taskset is None:
            raise HTTPException(404, f"Taskset {task_id.taskset!r} is not served by this resources server")
        expected = taskset.tasks.get(task_id.task_id)
        if expected is None:
            raise HTTPException(404, f"Task {task_id.task_id!r} is not in taskset {task_id.taskset!r}")
        digest = task_data.get(DIGEST_KEY)
        if digest != expected:
            raise HTTPException(
                422,
                f"Row digest for {task_id.task_id!r} does not match the taskset mapping; re-materialize the taskset",
            )
        folder = Path(taskset.folder) / task_id.task_id
        key = (folder, expected)
        pending = self._tasks.get(key)
        if pending is None:
            # Concurrent first seeds share the same copy/hash, including if a caller is cancelled.
            pending = asyncio.create_task(asyncio.to_thread(self._snapshot_task, folder, expected))
            self._tasks[key] = pending

            def remove_failed_load(done: asyncio.Task[HarborTask]) -> None:
                if done.cancelled() or done.exception() is not None:
                    self._tasks.pop(key, None)

            pending.add_done_callback(remove_failed_load)
        return pending.result() if pending.done() else await asyncio.shield(pending)

    # -- sandbox ---------------------------------------------------------------------------

    def _sandbox_spec(self, task: HarborTask, workdir: str | None) -> SandboxSpec:
        global_config_dict = get_global_config_dict()
        metadata = (
            resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_resources_server": self.config.name, "harbor_task": task.task_id[:63]}
        )
        ttl = (
            self.config.episode_timeout_seconds + self.config.cleanup_timeout_seconds + self.config.sandbox_ttl_slack_s
        )
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(task) | (self.config.sandbox_resources_override or {})
        env = dict(self.config.sandbox_env) | resolve_env_templates(task.env)
        if self.config.derive_cpu_env:
            env = cpu_cap_env(resources.get("cpu")) | env
        return SandboxSpec(
            image=task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=env,
            metadata=metadata,
            resources=resources,
            provider_options=dict(self.config.sandbox_provider_options),
        )

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, session_id: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[session_id] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _prepare_workdir(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str | None) -> str:
        """Create the working directory, or resolve the image's own WORKDIR when the task sets none."""
        if workdir is None:
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.return_code != 0 or not (result.stdout or "").strip():
                raise RuntimeError(f"Could not resolve the image working directory: {result.stderr or result.stdout}")
            workdir = (result.stdout or "").strip().splitlines()[-1]
        commands = [f"mkdir -p {shlex.quote(workdir)}"]
        if task.user:
            commands.append(f"chown {shlex.quote(task.user)} {shlex.quote(workdir)}")
        result = await _exec_as_root(sandbox, " && ".join(commands), timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")
        environment_dir = task.path / "environment"
        if (
            task.config.environment.docker_image
            and environment_dir.is_dir()
            and not any(
                (environment_dir / name).exists()
                for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
            )
        ):
            await upload_dir(sandbox, environment_dir, workdir)
        return workdir

    async def _run_setup_commands(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        for command in self.config.sandbox_setup_commands:
            # Root first, like every other setup step; sandboxes that cannot switch users fall back.
            result = await _exec_as_root(sandbox, command, timeout_s=self.config.sandbox_setup_timeout_s)
            if result.return_code != 0:
                raise RuntimeError(
                    f"Sandbox setup command failed for {task.task_id!r} (exit {result.return_code}): "
                    f"{(result.stderr or result.stdout or '')[-500:]}"
                )

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=self.config.sandbox_provider,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
        )

    # -- protocol --------------------------------------------------------------------------

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        if self._shutting_down:
            raise HTTPException(503, "Resources server is shutting down")
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._session_lock(session_id):
            if self._shutting_down:
                raise HTTPException(503, "Resources server is shutting down")
            if session_id in self._closed:
                raise HTTPException(409, f"Resources session is already closed: {session_id}")
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.identity != (body.episode_id, body.task_id):
                    raise HTTPException(409, "resources_session_id is already bound to another episode or task")
                return ResourcesSeedSessionResponse(
                    resources_session_id=session_id, sandbox_access=await self._sandbox_access(existing)
                )

            task = await self._resolve_task(body.task_id, body.task_data)
            if not task.needs_sandbox:
                raise HTTPException(
                    422, f"Task {task.task_id!r} declares no image; sandbox-less tasks are not supported yet"
                )
            if not task.config.is_shared_verifier:
                raise HTTPException(422, "Separate verification is not supported yet")
            if task.user is not None:
                raise HTTPException(
                    422, "Agent user overrides ([agent].user or Dockerfile USER) are not supported yet"
                )
            # Fail before provisioning; retain templates in the snapshot and resolve at use time.
            resolve_env_templates(task.env)
            resolve_env_templates(task.config.verifier.env)
            if self._shutting_down:
                raise HTTPException(503, "Resources server is shutting down")
            if session_id in self._sandboxes:
                raise HTTPException(503, "Previous sandbox cleanup failed; close this session before retrying")
            try:
                sandbox = await self._create_sandbox(task, task.workdir, session_id=session_id)
                self._sandboxes[session_id] = sandbox
                workdir = await self._prepare_workdir(sandbox, task, task.workdir)
                await self._run_setup_commands(sandbox, task)
                session = HarborSession(
                    task=task,
                    taskset=body.task_id.taskset,
                    identity=(body.episode_id, body.task_id),
                    sandbox=sandbox,
                    workdir=workdir,
                )
                access = await self._sandbox_access(session)
            except BaseException as exc:
                try:
                    await self._stop_sandbox(session_id)
                except Exception:
                    LOGGER.exception("Failed to stop Harbor sandbox after failed seed")
                if not isinstance(exc, Exception):
                    raise
                LOGGER.exception("Sandbox setup failed for %s", task.task_id)
                raise HTTPException(503, f"Could not set up sandbox for {task.task_id!r}: {exc}") from exc
            self._sessions[session_id] = session
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=access)

    def _session_for(self, request: Request, digest: str | None = None) -> HarborSession:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise HTTPException(404, "Unknown Harbor resources session; seed it first")
        if digest is not None and digest != session.task.digest:
            raise HTTPException(409, "Verification digest does not match the seeded task")
        return session

    async def verify(self, request: Request, body: HarborVerifyRequest) -> HarborVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            raise HTTPException(404, "Unknown Harbor resources session; seed it first")
        async with self._session_lock(session_id):
            session = self._session_for(request, body.ng_digest)
            if session.verify_error is not None:
                raise HTTPException(*session.verify_error)
            if session.verify_outcome is None:
                try:
                    session.verify_outcome = await self._run_verifier(session)
                except HTTPException as exc:
                    # Cache only the error data, not a traceback retaining request/sandbox state.
                    session.verify_error = (exc.status_code, str(exc.detail))
                    raise
            outcome = session.verify_outcome
        return HarborVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            **outcome,
        )

    async def _run_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote."""
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        started = asyncio.get_running_loop().time()
        try:
            # Protocol IDs are opaque caller input, never host paths. Reserve a fresh directory
            # so neither traversal nor stale output from a reused ID can supply verifier files.
            if self._artifacts_dir.resolve() != self._artifacts_dir:
                raise RuntimeError("Verifier artifact root was redirected after server startup")
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = self._artifacts_dir / f"session-{uuid4().hex}"
            # Exclusive mkdir keeps the directory fresh while honoring the process umask and parent ACLs.
            logs_dir.mkdir()
            logs_dir = logs_dir.resolve()
            if not logs_dir.is_relative_to(self._artifacts_dir):
                raise RuntimeError("Verifier log directory escapes the configured artifact root")
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        try:
            # Replace agent-planted directory links before cleanup: find does not follow its starting
            # symlink, while tar does. Unlink /logs first without touching its former target.
            roots = (str(Path(VERIFIER_LOGS_DIR).parent), TESTS_DIR, VERIFIER_LOGS_DIR, AGENT_LOGS_DIR)
            unlink_roots = [f"if test -L {shlex.quote(path)}; then rm -- {shlex.quote(path)}; fi" for path in roots]
            prepare = await _exec_as_root(
                sandbox,
                " && ".join(
                    unlink_roots
                    + [
                        f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                        f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete",
                        f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    ]
                ),
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(sandbox, task.path / "tests", TESTS_DIR)
            # One plain exec for the whole run. The provider keeps it alive for as long as
            # `timeout_s` says (OpenSandbox polls a background command, Docker streams), and
            # `timeout` inside the container is the cap that stops the tests themselves.
            # Commands must not be left running after the exec returns: OpenSandbox reaps
            # them when the command completes, so a background launch never finishes.
            budget = int(settings.timeout_sec)
            result = await sandbox.exec(
                f"timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                cwd=session.workdir,
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            # `timeout` exits 137 when it had to kill test.sh; a provider that gave up first reports "timeout".
            timed_out = result.error_type == "timeout" or result.return_code == 137
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            if not timed_out:
                # Reward transfer is required; optional diagnostics cannot invalidate a measured score.
                await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt"))
                rewards, problem = parse_reward_file(logs_dir)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")

        # Absolute, so the row names a folder that exists from wherever the run was launched; the server
        # process runs from its own folder, where a relative artifacts_dir would otherwise resolve.
        extras = {
            "verifier_return_code": result.return_code,
            "verifier_logs_dir": str(Path(logs_dir).resolve()),
            "verifier_seconds": round(asyncio.get_running_loop().time() - started, 1),
        }
        # Keep whatever the verifier wrote, including on a timeout, so a slow test.sh can be diagnosed.
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        if timed_out:
            return (
                self._measured(
                    0.0,
                    VERIFIER_TIMEOUT_KIND,
                    f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}; tail: {_stdout_tail(logs_dir)}",
                )
                | extras
            )
        if rewards is None:
            kind = MISSING_REWARD_KIND if "written" in (problem or "") else INVALID_REWARD_KIND
            return self._measured(0.0, kind, f"{problem}; tail: {_stdout_tail(logs_dir)}") | extras
        reward = select_reward(rewards)
        if reward is None:
            raise HTTPException(
                422,
                f"reward.json for {task.task_id!r} has several keys and none named 'reward': {sorted(rewards)}",
            )
        return {"reward": reward, "verifier_rewards": rewards} | extras

    @staticmethod
    def _measured(reward: float, kind: str, reason: str) -> dict[str, Any]:
        return {"reward": reward, "mask_sample": False, "failure_kind": kind, "failure_reason": reason[:2000]}

    @staticmethod
    def _masked(kind: str, reason: str) -> dict[str, Any]:
        return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": reason[:2000]}

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._session_lock(session_id):
            session = self._sessions.get(session_id)
            if session is not None and session.identity[0] != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded resources session")
            await self._stop_sandbox(session_id)
            self._closed[session_id] = None
            self._closed.move_to_end(session_id)
            while len(self._closed) > self.config.closed_session_cache_size:
                self._closed.popitem(last=False)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


if __name__ == "__main__":
    HarborResourcesServer.run_webserver()
