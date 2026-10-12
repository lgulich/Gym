# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. The supported subset is listed below.

## What the server does

- `/seed_session`: checks the row's digest against the taskset mapping, snapshots and checks the task
  files once per digest, starts the task image, and returns sandbox access for the agent harness.
  Concurrent seeds share the snapshot. Later source edits cannot change an already cached task;
  re-materialize and restart the server to use edited tasks.
- `/verify`: clears `/tests` and `/logs/verifier`, uploads the snapshot's `tests/`, runs
  `bash /tests/test.sh`, and downloads the verifier logs. Retries replay the first outcome, including
  an ambiguous-reward error, without running the verifier again.
- `/close_session`: stops the sandbox. Setup cancellation also attempts cleanup; failed stops remain
  owned by the server for a later close or shutdown retry. Idle session locks are released, and only
  the most recent `closed_session_cache_size` closed IDs are retained (default 10,000).

When `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a `failure_kind`.
Only a Gym-side failure (sandbox lost, transfer failed) sets `mask_sample`. An existing `reward.json`
takes precedence even if malformed; `reward.txt` is read only when JSON is absent. Boolean JSON rewards
are numeric, as in Harbor. Gym selects the `reward` key or the only key; multiple keys without `reward`
are an authoring error (422) because Gym requires one scalar score.

## Supported task behavior

| Setting or feature | Behavior in this layer |
| --- | --- |
| Image | Prebuilt `environment.docker_image`, or a base-image-only Dockerfile (`FROM`, literal absolute `WORKDIR`, `ENV`, `LABEL`). Dockerfile builds are deferred. |
| Verification | Single-step, shared-sandbox verification. Explicit `verifier.environment_mode = "separate"` or a separate verifier environment is rejected at seed. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values return 422 before provisioning. Resolved values are not added to dataset rows, task snapshots, or server diagnostics. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. Artifact `exclude` rules and shared-mode collect hooks are not implemented in this layer. |
| Image ENTRYPOINT | No Harbor-compatible entrypoint lifecycle is guaranteed; startup follows the selected Gym sandbox provider. Tasks must not rely on an entrypoint running. |
| Other features | Multi-step tasks, in-sandbox MCP tools, Windows and TPU environments are unsupported. |

Clearing `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's
single-step shared mode, which merges tests. It prevents agent-planted reward files or test helpers
from being used. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction.

## Configuration and artifacts

See [`configs/harbor.yaml`](configs/harbor.yaml) for this layer's supported settings. Verifier logs default
to `${results_dir}/harbor/verifier/<resources_session_id>`, using Gym's absolute results root. An explicit
relative `artifacts_dir` is relative to that root; an absolute path is used directly. Responses always
include an absolute `verifier_logs_dir` when logs were downloaded.

Resource settings come from the task. `sandbox_resources_override`, `gpu_sandbox_provider`,
`request_gpu_type` and `verifier_grace_s` belong to later stack layers and are not configuration options
in this PR.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). A local task folder, or a folder of task
folders, works the same way. Running model episodes additionally requires a compatible agent harness
that borrows the sandbox access returned by this server.
