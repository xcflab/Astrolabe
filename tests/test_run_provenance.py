"""Focused tests for the minimal `provenance.json` written by every real run.

Covers the machine-readable metadata collected at run start (version, Git, config
digest, seed, Python/dependency versions, hardware, explicit input file identities)
and the started/completed/failed/interrupted state contract, without running the
pipeline.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys

import pytest
from omegaconf import OmegaConf

from astrolabe import app
from astrolabe import run_config_lifecycle as lifecycle
from astrolabe.argparser import parse_arguments
from astrolabe.core import global_config

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _base_args(tmp_path: pathlib.Path, **overrides) -> dict:
    """Minimal arg set accepted by `parse_arguments` (same shape as argparser tests)."""
    args = {
        "task_name": "prov",
        "out_dir": str(tmp_path),
        "raw_model_results_dir": str(tmp_path / "raw"),
        "seed": 7,
        "debug": False,
        "named_files": None,
        "seqs": {"A": "AAA"},
        "types": None,
        "loss": {"loss_terms": {}},
        "stages": {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
                "evaluator": "E_default",
            }
        },
        "mutator": {
            "mutator_terms": {"MCMC": {"_type": "MCMC"}},
            "MCMC": {"mutation_method": "random", "select_positions": "random"},
            "selector_expression": None,
        },
        "evaluator": {"evaluator_terms": {"E_default": {"_type": "dummy"}}},
    }
    args.update(overrides)
    return args


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_provenance(run_dir: pathlib.Path) -> dict:
    return json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))


def test_parse_arguments_writes_started_provenance_with_metadata(
    tmp_path: pathlib.Path,
) -> None:
    global_config.reset()
    named = tmp_path / "motif.pdb"
    named.write_text("ATOM      1  N   ALA A   1\n", encoding="utf-8")
    homology = tmp_path / "homology.json"
    homology.write_text('{"A": "MSA"}\n', encoding="utf-8")

    resolved = parse_arguments(
        OmegaConf.create(
            _base_args(
                tmp_path,
                named_files={"motif": str(named)},
                homologies={"A": f"{homology}:A", "B": None},
            )
        )
    )

    run_dir = pathlib.Path(resolved.out_dir)
    provenance = _read_provenance(run_dir)

    # State contract: a run that has just started is not terminal.
    assert provenance["schema_version"] == 1
    assert provenance["state"] == "started"
    assert provenance["terminal"] is False
    assert provenance["ended_at"] is None
    assert provenance["error"] is None
    assert provenance["status_history"] == [
        {"state": "started", "at": provenance["started_at"]}
    ]

    # Run identity: validated (actual) seed and the real run directory.
    assert provenance["run"]["task_name"] == "prov"
    assert provenance["run"]["run_dir"] == str(run_dir)
    assert provenance["run"]["seed"] == 7

    # Version plus Python/dependency versions (metadata only, no GPU library import).
    assert provenance["version"]["astrolabe"]
    assert provenance["version"]["source"] in {
        "package attribute",
        "installed distribution metadata",
    }
    assert provenance["python"]["version"] == platform.python_version()
    assert provenance["dependencies"]["omegaconf"] == importlib.metadata.version(
        "omegaconf"
    )

    # Git provenance comes from the source root checkout.
    git = provenance["git"]
    assert git["available"] is True
    assert len(git["commit"]) == 40
    assert isinstance(git["dirty"], bool)
    assert isinstance(git["dirty_entries"], int)

    # Hardware/device visibility.
    hardware = provenance["hardware"]
    assert hardware["cpu_count"] >= 1
    assert set(hardware["device_visibility_env"]) == {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    }
    assert "available" in hardware["nvidia_smi"]

    # Persisted config digest matches the config.yaml this run directory holds.
    assert provenance["config"]["algorithm"] == "sha256"
    assert provenance["config"]["sha256"] == _sha256(run_dir / "config.yaml")

    # Explicit input file identities, streamed and tolerant of a `:CHAIN` suffix.
    motif_entry = provenance["input_files"]["named_files"]["motif"]
    assert motif_entry["sha256"] == _sha256(named)
    assert motif_entry["size_bytes"] == named.stat().st_size

    homology_entry = provenance["input_files"]["homologies"]["A"]
    assert homology_entry["sha256"] == _sha256(homology)
    assert pathlib.Path(homology_entry["path"]) == homology
    assert provenance["input_files"]["homologies"]["B"] is None


def test_terminal_states_never_report_completion(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "001_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("seed: 3\n", encoding="utf-8")
    args = OmegaConf.create({"task_name": "t", "out_dir": str(run_dir), "seed": 3})

    assert lifecycle.start_run_provenance(args) == run_dir / "provenance.json"

    lifecycle.finish_run_provenance(args, "failed", error=ValueError("boom"))
    failed = _read_provenance(run_dir)
    assert failed["state"] == "failed"
    assert failed["terminal"] is True
    assert failed["error"] == {"type": "ValueError", "message": "boom"}
    assert failed["ended_at"] == failed["updated_at"]
    assert failed["started_at"] <= failed["ended_at"]
    assert [entry["state"] for entry in failed["status_history"]] == [
        "started",
        "failed",
    ]

    lifecycle.finish_run_provenance(args, "interrupted", error=KeyboardInterrupt())
    interrupted = _read_provenance(run_dir)
    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal"] is True
    assert interrupted["error"]["type"] == "KeyboardInterrupt"
    assert "completed" not in [entry["state"] for entry in interrupted["status_history"]]

    # The record is replaced atomically: no temp file is left behind.
    assert sorted(entry.name for entry in run_dir.iterdir()) == [
        "config.yaml",
        "provenance.json",
    ]


def test_provenance_failures_are_not_raised(tmp_path: pathlib.Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    args = OmegaConf.create({"task_name": "t", "out_dir": str(missing_dir)})

    assert lifecycle.start_run_provenance(args) is None
    assert lifecycle.finish_run_provenance(args, "failed") is None
    assert not (missing_dir / "provenance.json").exists()


def test_git_provenance_does_not_infer_from_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cwd is a real checkout here, but an installed-wheel-like source root has no .git.
    monkeypatch.chdir(_REPO_ROOT)
    record = lifecycle.collect_git_provenance(
        tmp_path / "site-packages" / "astrolabe"
    )

    assert record["available"] is False
    assert record["commit"] is None
    assert record["dirty"] is None
    assert ".git" in record["reason"]


def test_metadata_collection_avoids_gpu_imports_and_version_fallback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = OmegaConf.create({"task_name": "t", "out_dir": str(tmp_path), "seed": 11})

    before = set(sys.modules)
    payload = lifecycle.build_run_provenance(args)
    newly_imported = {name.split(".")[0] for name in set(sys.modules) - before}

    assert not newly_imported & {"torch", "jax", "protenix", "esm", "alphafold"}
    assert payload["hardware"]["device_visibility_env"]["CUDA_VISIBLE_DEVICES"] is None
    for distribution, version in payload["dependencies"].items():
        assert version == importlib.metadata.version(distribution)

    # Distribution metadata is used when the package attribute is absent.
    monkeypatch.delattr(sys.modules["astrolabe"], "__version__", raising=False)
    fallback = lifecycle.collect_version_provenance()
    assert fallback["source"] in {"installed distribution metadata", "unavailable"}


def test_run_pipeline_records_completed_failed_and_interrupted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = app._load_run_module()
    monkeypatch.setattr(run, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_shutdown_pipeline_resources", lambda: None)

    class _StageManager:
        current_evaluator = "eval"

    def _run_once(run_dir: pathlib.Path, failure: BaseException | None) -> dict:
        run_dir.mkdir()
        args = OmegaConf.create(
            {"task_name": run_dir.name, "out_dir": str(run_dir), "seed": 5}
        )

        def _parse(cfg):
            # Mirrors the real parse_arguments: started provenance is written first.
            lifecycle.start_run_provenance(args)
            return args

        def _initialize(args):
            if failure is not None:
                raise failure
            return (None, _StageManager(), None)

        monkeypatch.setattr(run, "parse_arguments", _parse)
        monkeypatch.setattr(run, "initialize_pipeline_components", _initialize)
        monkeypatch.setattr(run, "load_or_evaluate_initial", lambda *a, **k: "initial")
        monkeypatch.setattr(run, "run_optimization_loop", lambda *a, **k: None)

        if failure is None:
            run._run_pipeline(OmegaConf.create({}))
        else:
            with pytest.raises(type(failure)):
                run._run_pipeline(OmegaConf.create({}))
        return _read_provenance(run_dir)

    completed = _run_once(tmp_path / "completed", None)
    assert completed["state"] == "completed"
    assert completed["terminal"] is True
    assert [entry["state"] for entry in completed["status_history"]] == [
        "started",
        "completed",
    ]

    failed = _run_once(tmp_path / "failed", RuntimeError("boom"))
    assert failed["state"] == "failed"
    assert failed["error"] == {"type": "RuntimeError", "message": "boom"}

    interrupted = _run_once(tmp_path / "interrupted", KeyboardInterrupt())
    assert interrupted["state"] == "interrupted"
    assert interrupted["state"] != "completed"
    assert "completed" not in [entry["state"] for entry in interrupted["status_history"]]
