from __future__ import annotations

import gc
import importlib
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner, _terminate_process


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True


def _make_runner(tmp_path: Path) -> DPLMRunner:
    repo_dir = tmp_path / "dplm_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    return DPLMRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_dir),
            "daemon": True,
            "work_dir": str(tmp_path / "work"),
            "response_cache_dir": str(tmp_path / "cache"),
        }
    )


def test_dplm_runner_finalizer_does_not_keep_runner_alive(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    proc = _FakeProc()
    runner._proc = proc
    runner._proc_finalizer = weakref.finalize(runner, _terminate_process, proc)
    runner_ref = weakref.ref(runner)

    del runner
    gc.collect()

    assert runner_ref() is None
    assert proc.terminated


def test_dplm_runner_close_releases_daemon_and_prevents_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _make_runner(tmp_path)
    proc = _FakeProc()
    runner._proc = proc
    runner._proc_finalizer = weakref.finalize(runner, _terminate_process, proc)
    monkeypatch.setattr(runner, "_daemon_request", lambda _payload: {"type": "ok"})

    runner.close()

    assert proc.terminated
    assert runner._proc is None
    with pytest.raises(RuntimeError, match="closed"):
        runner.design_sequences([])


def test_dplm_runner_lock_is_reentrant_for_daemon_restart(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    assert runner._lock.acquire(blocking=False)
    assert runner._lock.acquire(blocking=False)
    runner._lock.release()
    runner._lock.release()


def test_dplm_runner_host_subprocess_env_drops_container_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _make_runner(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/container/py312")
    monkeypatch.setenv("APPTAINERENV_PYTHONPATH", "/host/.gh_envs/py312_site")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/stale/triton")

    env = runner._build_host_subprocess_env()

    assert "PYTHONPATH" not in env
    assert "APPTAINERENV_PYTHONPATH" not in env
    assert env["TRITON_CACHE_DIR"] == str(runner.triton_cache_dir)


def test_run_pipeline_shutdowns_mutators_before_raw_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_module = importlib.import_module("run")
    events: list[str] = []

    class _FakeMutator:
        def shutdown(self) -> None:
            events.append("shutdown")

    args = SimpleNamespace(
        out_dir=str(tmp_path / "out"),
        logging=None,
        task_name="lifecycle_smoke",
        _raw_model_results_dir_auto_managed=False,
    )

    def _fake_initialize_pipeline_components(_args):
        global_config.init_const("mutators", {"dplm": _FakeMutator()})
        return object(), SimpleNamespace(current_evaluator="fake_evaluator"), {}

    global_config.reset()
    monkeypatch.setattr(run_module, "parse_arguments", lambda _cfg: args)
    monkeypatch.setattr(run_module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_module,
        "initialize_pipeline_components",
        _fake_initialize_pipeline_components,
    )
    monkeypatch.setattr(
        run_module,
        "load_or_evaluate_initial",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_module,
        "run_optimization_loop",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module,
        "_cleanup_auto_managed_raw_dir",
        lambda *_args, **_kwargs: events.append("cleanup"),
    )

    try:
        run_module._run_pipeline({})
    finally:
        global_config.reset()

    assert events == ["shutdown", "cleanup"]
