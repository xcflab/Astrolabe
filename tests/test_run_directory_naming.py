from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astrolabe.argparser import create_run_directory
from astrolabe.posthoc.io_utils import scan_run_dirs


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = (
        repo_root / "examples" / "hcar1" / "run_constraint_compliance_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_constraint_compliance_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scores(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores.csv").write_text("step,weighted_score\n0,1.0\n", encoding="utf-8")


def test_create_run_directory_uses_minimum_three_digit_width(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, "demo")
    second = create_run_directory(tmp_path, "demo")

    assert first.name == "001_run"
    assert second.name == "002_run"


def test_create_run_directory_continues_past_999(tmp_path: Path) -> None:
    task_root = tmp_path / "demo"
    (task_root / "999_run").mkdir(parents=True)

    thousand = create_run_directory(tmp_path, "demo")
    thousand_one = create_run_directory(tmp_path, "demo")

    assert thousand.name == "1000_run"
    assert thousand_one.name == "1001_run"


def test_scan_run_dirs_sorts_by_numeric_run_index(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    _write_scores(out_root / "task_a" / "010_run")
    _write_scores(out_root / "task_a" / "999_run")
    _write_scores(out_root / "task_a" / "1000_run")

    run_names = [path.name for path in scan_run_dirs(out_root)]

    assert run_names == ["010_run", "999_run", "1000_run"]


def test_constraint_benchmark_lists_runs_by_numeric_run_index(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    task_root = tmp_path / "outputs" / "task_a"
    for name in ("010_run", "999_run", "1000_run"):
        (task_root / name).mkdir(parents=True)

    run_names = [path.name for path in module._list_run_dirs(tmp_path / "outputs", "task_a")]

    assert run_names == ["010_run", "999_run", "1000_run"]


def test_constraint_benchmark_defaults_to_portable_env_group(monkeypatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.delenv("GH_CONFIG_ENV", raising=False)

    args = module._build_parser().parse_args(["--target-homology", "/tmp/hcar1.json:A"])

    assert args.config_env == "env"


def test_constraint_benchmark_gpu_inherits_cuda_visible_devices(monkeypatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    args = module._build_parser().parse_args(["--target-homology", "/tmp/hcar1.json:A"])

    assert args.gpu == "3"


def test_constraint_benchmark_dry_run_command_includes_env_group(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_benchmark_module()
    manifest = tmp_path / "manifest.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_constraint_compliance_benchmark.py",
            "--dry-run",
            "--target-homology",
            "/tmp/hcar1.json:A",
            "--out-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(manifest),
            "--evaluators",
            "protenix",
            "--modes",
            "contact",
            "--session-tag",
            "pytest",
        ],
    )

    assert module.main() == 0

    out = capsys.readouterr().out
    assert "env=env" in out
    assert "homologies.B=/tmp/hcar1.json:A" in out
    assert manifest.exists()
