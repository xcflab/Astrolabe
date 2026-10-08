from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import astrolabe.cli as cli


CONFIG_DIR = Path(cli.__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _patch_default_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (config_dir / "env").mkdir(exist_ok=True)
    (config_dir / "env" / "env_local.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_default_config_dir", lambda: config_dir)


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def test_cli_design_multitarget_binder_x_init_dry_run(capsys):
    rc = _run(
        [
            "design",
            "multitarget-binder",
            "--task-name",
            "t_multitarget_x",
            "--out-dir",
            "outputs",
            "--variant",
            "x-init-direct-dual-protenix",
            "--pair-name",
            "il4_il13_shared_il4ra",
            "--shared-hotspot-manifest",
            "inputs/il4_il13_shared_il4ra.hotspots.yaml",
            "--binder-len",
            "80-120",
            "--target-b-seq",
            "HKCDITLQEI",
            "--target-b-hotspots",
            "5,6,9",
            "--target-c-seq",
            "PGPVPPSTAL",
            "--target-c-hotspots",
            "8,9,12",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pair_name: il4_il13_shared_il4ra" in out
    assert "shared_hotspot_manifest: inputs/il4_il13_shared_il4ra.hotspots.yaml" in out
    assert "mode: mask_mix" in out
    assert "mask_fraction: 0.99" in out
    assert "B: +HKCDITLQEI" in out
    assert "C: +PGPVPPSTAL" in out
    assert "target: B:5,6,9" in out
    assert "target: C:8,9,12" in out
    assert "LigandMPNN_itf" in out
    assert "LigandMPNN_noitf" in out
    assert "- dplm" not in out
    assert "dual_protenix:" in out
    assert "dual_af3:" not in out


def test_cli_design_multitarget_binder_dplm_dry_run(capsys):
    rc = _run(
        [
            "design",
            "multitarget-binder",
            "--task-name",
            "t_multitarget_dplm",
            "--out-dir",
            "outputs",
            "--variant",
            "dplm-protenix",
            "--pair-name",
            "il4_il13_shared_il4ra",
            "--shared-hotspot-manifest",
            "inputs/il4_il13_shared_il4ra.hotspots.yaml",
            "--binder-len",
            "80-120",
            "--target-b-seq",
            "HKCDITLQEI",
            "--target-b-hotspots",
            "5,6,9",
            "--target-c-seq",
            "PGPVPPSTAL",
            "--target-c-hotspots",
            "8,9,12",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "initializer:" in out
    assert "mode: dplm" in out
    assert "reinitialize_random_segments_with_dplm" not in out
    assert "pair_name: il4_il13_shared_il4ra" in out
    assert "target: B:5,6,9" in out
    assert "target: C:8,9,12" in out
    assert "evaluator: Protenix" in out
    assert "dual_protenix:" in out
    assert "dual_af3:" not in out


def test_cli_design_multitarget_binder_accepts_initial_binder_expr(capsys):
    rc = _run(
        [
            "design",
            "multitarget-binder",
            "--task-name",
            "t_multitarget_expr",
            "--out-dir",
            "outputs",
            "--variant",
            "dplm-protenix",
            "--pair-name",
            "il4_il13_shared_il4ra",
            "--shared-hotspot-manifest",
            "inputs/il4_il13_shared_il4ra.hotspots.yaml",
            "--initial-binder-expr",
            "ACDEFG 20-30",
            "--target-b-seq",
            "HKCDITLQEI",
            "--target-b-hotspots",
            "5,6,9",
            "--target-c-seq",
            "PGPVPPSTAL",
            "--target-c-hotspots",
            "8,9,12",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "A: ACDEFG 20-30" in out
