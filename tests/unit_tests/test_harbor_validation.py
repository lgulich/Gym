# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
import yaml

import nemo_gym.cli.main as main_module
import nemo_gym.global_config as global_config
from nemo_gym.path_utils import failures_path_for
from nemo_gym.tasks.harbor.cli import build_run, prepare_target, resolve_agent, summarize_validation
from tests.unit_tests.test_harbor_tasks import write_task


def result(task="a", *, index=0, repeat=0, **fields):
    return {
        "_ng_task_id": {"taskset": "ds", "task_id": task},
        "_ng_task_index": index,
        "_ng_rollout_index": repeat,
        "reward": 1.0,
        **fields,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("missing", [result("b", index=1), result(repeat=1)])
def test_summary_requires_each_selected_task_and_repeat(tmp_path, missing):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result()])
    report = summarize_validation(path, [], expected=[result(), missing])
    assert not report.ok and "missing result" in report.text


def test_success_supersedes_only_failures_for_the_same_task_repeat(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(_ng_attempt_index=1)])
    write_jsonl(failures_path_for(path), [result(_ng_attempt_index=0, _ng_failure_message="old failure")])
    report = summarize_validation(path, [], expected=[result()])
    assert report.ok and "old failure" not in report.text
    write_jsonl(failures_path_for(path), [result(repeat=1, _ng_failure_message="unresolved repeat")])
    report = summarize_validation(path, [], expected=[result(), result(repeat=1)])
    assert not report.ok and "unresolved repeat" in report.text


def test_failure_sidecar_is_reported_without_a_results_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(failures_path_for(path), [result(_ng_failure_message="seed failed")])
    report = summarize_validation(path, [], expected=[result()])
    assert not report.ok and "seed failed" in report.text


def test_results_from_other_tasksets_do_not_cover_selected_tasks(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(_ng_task_id={"taskset": "other", "task_id": "a"})])
    report = summarize_validation(path, [], expected=[result()])
    assert not report.ok and "missing result" in report.text


@pytest.mark.parametrize("reward", [None, 0.0, float("nan"), float("inf")])
def test_invalid_or_failing_reward_never_passes(tmp_path, reward):
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result(reward=reward)])
    assert not summarize_validation(path, [], expected=[result()]).ok


@pytest.fixture
def validation_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)

    def invoke(*flags):
        monkeypatch.setattr(
            sys, "argv", ["gym", "dataset", "validate", str(tmp_path / "ds"), "--sandbox", "docker", *flags]
        )
        main_module.main()

    return invoke


@pytest.mark.parametrize("source", ["cli", "file"])
def test_validate_reads_effective_output_and_preserves_evaluation_artifacts(
    tmp_path, monkeypatch, validation_cli, source
):
    write_task(tmp_path / "ds" / "a")
    evaluated = prepare_target(str(tmp_path / "ds"))
    config_path, _ = build_run(evaluated, resolve_agent("hermes"), sandbox="docker", overrides=[])
    original_config = config_path.read_bytes()
    stale = evaluated.output_dir / "rollouts.jsonl"
    write_jsonl(stale, [result()])
    custom = tmp_path / "custom output.jsonl"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"output_jsonl_fpath": str(custom)}))
    seen = []

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        seen.append(target)
        config = global_config.get_global_config_dict()
        assert Path(config.output_jsonl_fpath) == custom
        write_jsonl(custom, [result(reward=0.0)])

    monkeypatch.setattr(main_module, "dispatch", collect)
    flags = [f"+output_jsonl_fpath={json.dumps(str(custom))}"] if source == "cli" else ["--config", str(overlay)]
    with pytest.raises(SystemExit) as exc:
        validation_cli(*flags)
    assert exc.value.code == 1
    assert seen == ["nemo_gym.cli.eval:e2e_rollout_collection"]
    assert config_path.read_bytes() == original_config
    assert json.loads(stale.read_text())["reward"] == 1.0
    assert (tmp_path / "results/harbor/validation/ds/run_config.yaml").is_file()


@pytest.mark.parametrize("missing_repeat", [False, True])
def test_validate_filters_solutionless_before_limit_and_checks_repeats(
    tmp_path, monkeypatch, capsys, validation_cli, missing_repeat
):
    for name in ("a", "b", "c"):
        write_task(tmp_path / "ds" / name)
    (tmp_path / "ds/a/solution/solve.sh").unlink()

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        config = global_config.get_global_config_dict()
        resources = config.harbor_ds_resources_server.resources_servers.harbor
        rows = [json.loads(line) for line in Path(resources.datasets[0].jsonl_fpath).read_text().splitlines()]
        assert [row["task_id"]["task_id"] for row in rows] == ["b", "c"]
        repeats = [result("b", repeat=0)] if missing_repeat else [result("b", repeat=i) for i in range(2)]
        write_jsonl(Path(config.output_jsonl_fpath), repeats)

    monkeypatch.setattr(main_module, "dispatch", collect)
    if missing_repeat:
        with pytest.raises(SystemExit) as exc:
            validation_cli("--limit", "1", "+num_repeats=2")
        assert exc.value.code == 1
    else:
        validation_cli("--limit", "1", "+num_repeats=2")
    output = capsys.readouterr().out
    assert "a\tunvalidated" in output
    assert "c [repeat" not in output
    assert ("missing result" in output) is missing_repeat


@pytest.mark.parametrize("with_solution", [False, True])
def test_validate_fails_for_skipped_tasks_even_with_no_solutions(
    tmp_path, monkeypatch, capsys, validation_cli, with_solution
):
    write_task(tmp_path / "ds" / "a")
    write_task(tmp_path / "ds" / "bad", toml="invalid toml")
    if not with_solution:
        (tmp_path / "ds/a/solution/solve.sh").unlink()

    def collect(target, tokens):
        monkeypatch.setattr(sys, "argv", ["e2e", *tokens])
        assert with_solution
        write_jsonl(Path(global_config.get_global_config_dict().output_jsonl_fpath), [result()])

    monkeypatch.setattr(main_module, "dispatch", collect)
    with pytest.raises(SystemExit) as exc:
        validation_cli()
    assert exc.value.code == 1
    assert "bad\tskipped:" in capsys.readouterr().out


def test_all_solutionless_tasks_are_reported_without_dispatch(tmp_path, monkeypatch, capsys, validation_cli):
    write_task(tmp_path / "ds" / "a")
    (tmp_path / "ds/a/solution/solve.sh").unlink()
    monkeypatch.setattr(main_module, "dispatch", lambda *args: pytest.fail("must not dispatch solutionless tasks"))
    validation_cli()
    assert "a\tunvalidated" in capsys.readouterr().out


@pytest.mark.parametrize("cli_override", [False, True])
def test_network_default_is_below_explicit_config_and_cli(tmp_path, monkeypatch, cli_override):
    write_task(tmp_path / "ds" / "a")
    prepared = prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "out")
    overlay = tmp_path / "network.yaml"
    overlay.write_text("use_absolute_ip: false\n")
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths

    _, tokens = build_run(
        prepared,
        resolve_agent("hermes"),
        sandbox="docker",
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'},{overlay}]",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
            "+policy_model_name=test",
            *(["+use_absolute_ip=true"] if cli_override else []),
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    assert global_config.get_global_config_dict().use_absolute_ip is cli_override
