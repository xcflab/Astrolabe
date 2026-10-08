from __future__ import annotations

import sys
from pathlib import Path

import pytest

from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner


def _make_runner(tmp_path: Path, *, cache_dir: Path, num_samples: int = 4) -> DPLMRunner:
    repo_dir = tmp_path / "dplm_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    return DPLMRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_dir),
            "daemon": True,
            "num_samples": int(num_samples),
            "work_dir": str(work_dir),
            "response_cache": True,
            "response_cache_dir": str(cache_dir),
        }
    )


def test_dplm_runner_consumes_cached_response_before_new_infer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _make_runner(tmp_path, cache_dir=tmp_path / "cache", num_samples=3)
    infer_calls: list[list[dict[str, object]]] = []

    def _fake_infer(task_list):
        infer_calls.append([dict(t) for t in task_list])
        outputs = {}
        call_idx = len(infer_calls)
        for task in task_list:
            name = str(task["name"])
            n = int(task.get("num_samples", 1))
            outputs[name] = [f"{name}_SEQ_{call_idx}_{i}" for i in range(n)]
        return outputs

    monkeypatch.setattr(runner, "_infer_tasks_no_cache", _fake_infer)

    task = {"name": "A", "masked_sequence": "AA<mask>AA", "num_samples": 1}
    first = runner.design_sequences([task])
    second = runner.design_sequences([task])
    third = runner.design_sequences([task])
    fourth = runner.design_sequences([task])

    assert first["A"] == ["A_SEQ_1_0"]
    assert second["A"] == ["A_SEQ_1_1"]
    assert third["A"] == ["A_SEQ_1_2"]
    assert fourth["A"] == ["A_SEQ_2_0"]
    assert len(infer_calls) == 2
    assert infer_calls[0][0]["num_samples"] == 3
    assert infer_calls[1][0]["num_samples"] == 3


def test_dplm_runner_response_cache_persists_between_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    task = {"name": "A", "masked_sequence": "AB<mask>CD", "num_samples": 1}

    runner1 = _make_runner(tmp_path / "r1", cache_dir=cache_dir, num_samples=3)

    def _fake_infer(task_list):
        outputs = {}
        for task_i in task_list:
            name = str(task_i["name"])
            n = int(task_i.get("num_samples", 1))
            outputs[name] = [f"{name}_CACHED_{i}" for i in range(n)]
        return outputs

    monkeypatch.setattr(runner1, "_infer_tasks_no_cache", _fake_infer)
    first = runner1.design_sequences([task])
    assert first["A"] == ["A_CACHED_0"]

    runner2 = _make_runner(tmp_path / "r2", cache_dir=cache_dir, num_samples=3)
    monkeypatch.setattr(
        runner2,
        "_infer_tasks_no_cache",
        lambda _tasks: pytest.fail("Runner should have used persisted cache instead of inferring again."),
    )
    second = runner2.design_sequences([task])
    assert second["A"] == ["A_CACHED_1"]

    key = runner2._cache_key(task)
    assert runner2._load_cached_samples(key) == ["A_CACHED_2"]
