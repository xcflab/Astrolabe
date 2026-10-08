from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from astrolabe.posthoc.db import open_db, upsert_records
from astrolabe.posthoc.refold import export_refold_manifest


def test_refold_from_db_propagates_resource_group_summary(monkeypatch, tmp_path: Path) -> None:
    import astrolabe.posthoc.refold as refold_mod

    def _export_refold_manifest(**kwargs):
        Path(kwargs["out_manifest"]).write_text("", encoding="utf-8")
        return {
            "db_path": str(kwargs["db_path"]),
            "selector_kind": "preset",
            "selector": "",
            "preset": "preset",
            "filter_expr": "",
            "written": 2,
        }

    def _run_refold(**_kwargs):
        return {
            "processed": 2,
            "skipped": 1,
            "failed": 0,
            "ran_af3": 1,
            "ran_protenix": 1,
            "reused_af3": 1,
            "reused_protenix": 0,
            "groups": 2,
            "resource_groups": 1,
            "evaluator_cache_hits": 1,
            "evaluator_cache_misses": 1,
            "failure_report": None,
        }

    monkeypatch.setattr(refold_mod, "export_refold_manifest", _export_refold_manifest)
    monkeypatch.setattr(refold_mod, "run_refold", _run_refold)

    summary = refold_mod.refold_from_db(
        db_path=tmp_path / "scores.sqlite",
        preset_path=tmp_path / "preset.yaml",
        dst_dir=tmp_path / "refold",
        dry_run=True,
    )

    assert summary["groups"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["reused_af3"] == 1


def _seed_record_with_config(*, db_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "homologies:",
                "  A: null",
                "  B: null",
                "types:",
                "  A: protein",
                "  B: protein",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": run_dir,
                    "task": "task_a",
                    "run": "001_run",
                    "step": 0,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": {
                        "stage": "default",
                        "weighted_score": 1.0,
                        "pA_sequence": "AAAA",
                        "pB_sequence": "BBBB",
                    },
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
    finally:
        conn.close()


def test_export_refold_manifest_includes_rows_blocked_only_by_missing_preset_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_record_with_config(db_path=db_path, run_dir=run_dir)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())


def test_export_refold_manifest_includes_rows_blocked_only_by_missing_filter_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_record_with_config(db_path=db_path, run_dir=run_dir)

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        filter_expr="protenix_iptm > 0.7",
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["selector_kind"] == "dsl"
    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())


def _seed_refold_record(
    *,
    db_path: Path,
    run_dir: Path,
    metrics: dict[str, object],
    stage_alias: str = "StrictPX",
    stage_type: str = "protenix",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "homologies:",
                "  A: null",
                "  B: null",
                "types:",
                "  A: protein",
                "  B: protein",
                "stages:",
                f"  default:\n    evaluator: {stage_alias}",
                "evaluator:",
                "  evaluator_terms:",
                f"    {stage_alias}:\n      _type: {stage_type}",
                "    AF3Alias:\n      _type: alphafold3",
                "  protenix:\n    repo_path: /tmp/protenix",
                "  alphafold3:\n    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": run_dir,
                    "task": "task_a",
                    "run": "001_run",
                    "step": 0,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": metrics,
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
    finally:
        conn.close()


def test_export_refold_manifest_skips_rows_that_already_pass_selector(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
            "protenix_iptm": 0.91,
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["records_passed"] == 1
    assert summary["written"] == 0
    assert manifest_path.read_text(encoding="utf-8") == ""


def test_export_refold_manifest_marks_needed_evaluators_from_missing_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
    )

    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["needed_evaluators"] == ["protenix"]


def test_run_refold_only_requires_backend_configs_that_manifest_needs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["af3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3", "protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 1
    assert summary["ran_protenix"] == 0


def test_run_refold_restores_cyclization_and_nonstandard_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.core import global_config
    from astrolabe.posthoc.refold import run_refold

    source_dir = tmp_path / "source" / "task_a" / "001_run"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "A",
                "src_run_dir": str(source_dir),
                "src_config": str(source_dir / "config.yaml"),
                "seqs": {"A": "A[S5]AAAA[R8]A"},
                "types": {"A": "protein"},
                "homologies": {"A": None},
                "needed_evaluators": ["af3"],
                "nonstandard_residues": {
                    "aliases": {
                        "S5": {
                            "ccd_id": "AS5",
                            "parent_aa": "A",
                            "atom_names": [
                                "SC5",
                                "SC4",
                                "SC3",
                                "SC2",
                                "SC1",
                                "CA",
                                "N",
                                "CB",
                                "C",
                                "O",
                                "OXT",
                            ],
                            "source": {
                                "format": "smiles",
                                "value": "C=CCCC[C@](N)(C)C(=O)O",
                            },
                        },
                        "R8": {
                            "ccd_id": "VR8",
                            "parent_aa": "A",
                            "atom_names": [
                                "SC8",
                                "SC7",
                                "SC6",
                                "SC5",
                                "SC4",
                                "SC3",
                                "SC2",
                                "SC1",
                                "CA",
                                "N",
                                "CB",
                                "C",
                                "O",
                                "OXT",
                            ],
                            "source": {
                                "format": "smiles",
                                "value": "C=CCCCCCC[C@@](N)(C)C(=O)O",
                            },
                        },
                    }
                },
                "cyclization": {
                    "bonds": [
                        {
                            "a": {
                                "proto": "A",
                                "pos": 2,
                                "aa": "A",
                                "atom": "SC5",
                                "residue": "S5",
                            },
                            "b": {
                                "proto": "A",
                                "pos": 7,
                                "aa": "A",
                                "atom": "SC8",
                                "residue": "R8",
                            },
                            "order": "single",
                            "target_distance": 1.54,
                            "signal_mode": "both",
                        }
                    ],
                    "encoding": {"scope": "all_bonds"},
                    "signal_mode": "bond_only",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _construct_standard_input(oligo_names, seed):
        from astrolabe.utils.base import construct_standard_input

        inputs = construct_standard_input(oligo_names, seed)
        fold_input = inputs["A"]
        captured["bonded_atom_pairs"] = fold_input.bonded_atom_pairs
        captured["cyclic_relative_patches"] = fold_input.cyclic_relative_patches
        captured["user_ccd"] = fold_input.user_ccd
        captured["ptm_dict"] = dict(global_config.protomers["A"].ptm_dict)
        return inputs

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        _construct_standard_input,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"A": [0.81]},
                ptm={"A": [0.72]},
                plddt={"A": [0.9]},
            )

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_evaluators",
        lambda _cfg, _out_root, oligo_names: {"Alphafold3": _FakeAF3Evaluator()},
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 1
    assert captured["bonded_atom_pairs"] == [(("A", 2, "SC5"), ("A", 7, "SC8"))]
    assert captured["cyclic_relative_patches"]
    assert "data_AS5" in str(captured["user_ccd"])
    assert captured["ptm_dict"] == {1: "AS5", 6: "VR8"}


def test_run_refold_reuses_source_af3_outputs_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    af3_dir = src_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB"
    af3_dir.mkdir(parents=True, exist_ok=True)
    (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "structures").mkdir(parents=True, exist_ok=True)
    (src_run_dir / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["af3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("construct_evaluators should not be called when source outputs can be reused")

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _boom)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    dst_run_dir = tmp_path / "refold_runs" / "task_a" / "001_run"
    assert summary["failed"] == 0
    assert (dst_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB" / "step_000_AB_summary_confidences.json").exists()
    assert (dst_run_dir / "structures" / "step_000_AB_model.cif").exists()


def test_run_refold_does_not_reuse_source_outputs_when_external_config_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    af3_dir = src_run_dir / "raw_model_results" / "alphafold3" / "step_000_AB"
    af3_dir.mkdir(parents=True, exist_ok=True)
    (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "structures").mkdir(parents=True, exist_ok=True)
    (src_run_dir / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3_a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    external_cfg = tmp_path / "external.yaml"
    external_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                "    model_dir: /tmp/af3_b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["af3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        config_yaml=external_cfg,
    )

    assert summary["failed"] == 0
    assert summary["reused_af3"] == 0
    assert summary["ran_af3"] == 1
    assert calls["construct"] == 1


def test_run_refold_protenix_propagates_binding_site_from_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    captured: dict[str, object] = {}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        from astrolabe.posthoc import refold as refold_module

        assert oligo_names == ["AB"]
        captured["binding_site"] = refold_module.global_config.binding_site
        captured["protenix_constraint"] = refold_module.global_config.protenix_constraint
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 1
    assert captured["binding_site"] == {
        "AB": {"binder": "A", "target": "B:5,6", "mode": "pocket"}
    }
    assert captured["protenix_constraint"] is None


def test_run_refold_does_not_reuse_source_protenix_outputs_when_binding_site_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    pred_dir = (
        src_run_dir
        / "raw_model_results"
        / "protenix"
        / "step_000"
        / "AB"
        / "seed_1"
        / "predictions"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "AB_seed_1_summary_confidence_sample_0.json").write_text("{}\n", encoding="utf-8")
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    external_cfg = tmp_path / "external.yaml"
    external_cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
        config_yaml=external_cfg,
    )

    assert summary["failed"] == 0
    assert summary["reused_protenix"] == 0
    assert summary["ran_protenix"] == 1
    assert calls["construct"] == 1


def test_run_refold_protenix_passes_constraint_weight_path_into_evaluator_conf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from omegaconf import OmegaConf

    from astrolabe.posthoc.refold import run_refold

    src_run_dir = tmp_path / "source" / "task_a" / "001_run"
    src_run_dir.mkdir(parents=True, exist_ok=True)
    (src_run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "weight_path:",
                "  protenix_constraint: /tmp/protenix_constraint.pt",
                "binding_site:",
                "  AB:",
                "    binder: A",
                "    target: B:5,6",
                "    mode: pocket",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                "    load_checkpoint_path: /tmp/protenix_default.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "src_run_dir": str(src_run_dir),
                "src_config": str(src_run_dir / "config.yaml"),
                "seqs": {"A": "AAAA", "B": "BBBB"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "needed_evaluators": ["protenix"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.construct_standard_input",
        lambda *_args, **_kwargs: {"AB": "input"},
    )
    monkeypatch.setattr(
        "astrolabe.posthoc.refold._merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )

    captured: dict[str, object] = {}

    class _FakeProtenixEvaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return None

    def _construct_evaluators(cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        captured["constraint_ckpt"] = OmegaConf.select(cfg, "weight_path.protenix_constraint")
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 1
    assert captured["constraint_ckpt"] == "/tmp/protenix_constraint.pt"


def test_run_refold_accepts_workers_for_parallel_group_queue(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(tmp_path / "source_a"),
                        "src_config": str(tmp_path / "source_a" / "config.yaml"),
                        "seqs": {"A": "AAAA", "B": "BBBB"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
                json.dumps(
                    {
                        "task": "task_b",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(tmp_path / "source_b"),
                        "src_config": str(tmp_path / "source_b" / "config.yaml"),
                        "seqs": {"A": "CCCC", "B": "DDDD"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
        dry_run=True,
    )

    assert summary["failed"] == 0
    assert summary["processed"] == 2
    assert summary["workers_requested"] == 2
    assert summary["workers"] == 1


def test_run_refold_separates_outputs_for_distinct_source_runs_with_same_task_and_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source" / "seed_1" / "task_a" / "001_run"
    source_b = tmp_path / "source" / "seed_2" / "task_a" / "001_run"
    for source in (source_a, source_b):
        source.mkdir(parents=True, exist_ok=True)
        (source / "config.yaml").write_text(
            "\n".join(
                [
                    "seed: 1",
                    "evaluator:",
                    "  evaluator_terms:",
                    "    AF3Alias:",
                    "      _type: alphafold3",
                    "  alphafold3:",
                    "    model_dir: /tmp/af3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_a),
                        "src_config": str(source_a / "config.yaml"),
                        "seqs": {"A": "AAAA", "B": "BBBB"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_b),
                        "src_config": str(source_b / "config.yaml"),
                        "seqs": {"A": "CCCC", "B": "DDDD"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    class _FakeAF3Evaluator:
        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    scores_paths = sorted((tmp_path / "refold_runs" / "task_a").rglob("scores.csv"))
    assert summary["failed"] == 0
    assert summary["processed"] == 2
    assert len(scores_paths) == 2

    observed_sources = set()
    for scores_path in scores_paths:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        observed_sources.add(rows[0]["src_run_dir"])

    assert observed_sources == {str(source_a), str(source_b)}


def test_run_refold_parallel_separates_outputs_for_distinct_source_runs_with_same_task_and_run(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source" / "seed_1" / "task_a" / "001_run"
    source_b = tmp_path / "source" / "seed_2" / "task_a" / "001_run"
    for source in (source_a, source_b):
        source.mkdir(parents=True, exist_ok=True)
        af3_dir = source / "raw_model_results" / "alphafold3" / "step_000_AB"
        af3_dir.mkdir(parents=True, exist_ok=True)
        (af3_dir / "step_000_AB_summary_confidences.json").write_text("{}\n", encoding="utf-8")
        (source / "structures").mkdir(parents=True, exist_ok=True)
        (source / "structures" / "step_000_AB_model.cif").write_text("data\n", encoding="utf-8")
        (source / "config.yaml").write_text(
            "\n".join(
                [
                    "seed: 1",
                    "evaluator:",
                    "  evaluator_terms:",
                    "    AF3Alias:",
                    "      _type: alphafold3",
                    "  alphafold3:",
                    "    model_dir: /tmp/af3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_a),
                        "src_config": str(source_a / "config.yaml"),
                        "seqs": {"A": "AAAA", "B": "BBBB"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
                json.dumps(
                    {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 0,
                        "oligo": "AB",
                        "src_run_dir": str(source_b),
                        "src_config": str(source_b / "config.yaml"),
                        "seqs": {"A": "CCCC", "B": "DDDD"},
                        "types": {"A": "protein", "B": "protein"},
                        "homologies": {"A": None, "B": None},
                        "needed_evaluators": ["af3"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        workers=2,
    )

    scores_paths = sorted((tmp_path / "refold_runs" / "task_a").rglob("scores.csv"))
    run_dir_names = {path.parent.name for path in scores_paths}
    assert summary["failed"] == 0
    assert summary["reused_af3"] == 2
    assert summary["workers"] == 2
    assert len(scores_paths) == 2
    assert len(run_dir_names) == 2
    assert any("seed_1" in name for name in run_dir_names)
    assert any("seed_2" in name for name in run_dir_names)

    observed_sources = set()
    for scores_path in scores_paths:
        with scores_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        observed_sources.add(rows[0]["src_run_dir"])

    assert observed_sources == {str(source_a), str(source_b)}


def _write_refold_af3_config(path: Path, *, model_dir: str = "/tmp/af3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    AF3Alias:",
                "      _type: alphafold3",
                "  alphafold3:",
                f"    model_dir: {model_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _refold_manifest_item(
    *,
    source: Path,
    task: str,
    sequence_a: str = "AAAA",
    needed_evaluators: list[str] | None = None,
) -> dict[str, object]:
    return {
        "task": task,
        "run": "001_run",
        "step": 0,
        "oligo": "AB",
        "src_run_dir": str(source),
        "src_config": str(source / "config.yaml"),
        "seqs": {"A": sequence_a, "B": "BBBB"},
        "types": {"A": "protein", "B": "protein"},
        "homologies": {"A": None, "B": None},
        "needed_evaluators": needed_evaluators or ["af3"],
    }


def _write_refold_protenix_config(path: Path, *, checkpoint: str = "/tmp/protenix.pt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed: 1",
                "evaluator:",
                "  evaluator_terms:",
                "    Protenix:",
                "      _type: protenix",
                "  protenix:",
                f"    load_checkpoint_path: {checkpoint}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_refold_reuses_evaluator_init_across_output_groups_with_same_resource(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml")
    _write_refold_af3_config(source_b / "config.yaml")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(_refold_manifest_item(source=source_a, task="task_a")),
                json.dumps(_refold_manifest_item(source=source_b, task="task_b", sequence_a="CCCC")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}
    observed_output_dirs: list[str] = []

    class _FakeAF3Evaluator:
        output_dir = None

        def evaluate(self, _inputs, runtime_config=None):
            observed_output_dirs.append(str(self.output_dir))
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 2
    assert summary["groups"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert calls["construct"] == 1
    assert len(set(observed_output_dirs)) == 2
    assert all(path.endswith("raw_model_results/alphafold3") for path in observed_output_dirs)


def test_run_refold_retargets_cached_protenix_runner_across_output_groups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_protenix_config(source_a / "config.yaml")
    _write_refold_protenix_config(source_b / "config.yaml")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    _refold_manifest_item(
                        source=source_a,
                        task="task_a",
                        needed_evaluators=["protenix"],
                    )
                ),
                json.dumps(
                    _refold_manifest_item(
                        source=source_b,
                        task="task_b",
                        sequence_a="CCCC",
                        needed_evaluators=["protenix"],
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr("astrolabe.posthoc.refold._merge_backend_metrics_into_rows", lambda **_kwargs: None)
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}
    observed_dump_dirs: list[str] = []
    observed_dumper_dirs: list[str] = []

    class _FakeProtenixEvaluator:
        def __init__(self):
            self.model_runner = SimpleNamespace(
                configs=SimpleNamespace(dump_dir=None),
                dumper=SimpleNamespace(base_dir=None),
            )

        def evaluate(self, _inputs, runtime_config=None):
            observed_dump_dirs.append(str(self.model_runner.configs.dump_dir))
            observed_dumper_dirs.append(str(self.model_runner.dumper.base_dir))
            return None

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Protenix": _FakeProtenixEvaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["protenix"],
    )

    assert summary["failed"] == 0
    assert summary["ran_protenix"] == 2
    assert summary["resource_groups"] == 1
    assert summary["evaluator_cache_misses"] == 1
    assert summary["evaluator_cache_hits"] == 1
    assert calls["construct"] == 1
    assert observed_dump_dirs == observed_dumper_dirs
    assert len(set(observed_dump_dirs)) == 2
    assert all(path.endswith("raw_model_results/protenix") for path in observed_dump_dirs)


def test_run_refold_keeps_evaluator_init_separate_for_different_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from astrolabe.posthoc.refold import run_refold

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml", model_dir="/tmp/af3_a")
    _write_refold_af3_config(source_b / "config.yaml", model_dir="/tmp/af3_b")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(_refold_manifest_item(source=source_a, task="task_a")),
                json.dumps(_refold_manifest_item(source=source_b, task="task_b", sequence_a="CCCC")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("astrolabe.posthoc.refold._init_refold_global_config", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold._build_protomers", lambda **_: None)
    monkeypatch.setattr("astrolabe.posthoc.refold.construct_standard_input", lambda *_args, **_kwargs: {"AB": "input"})
    monkeypatch.setattr(
        "astrolabe.posthoc.refold.global_config",
        SimpleNamespace(set_variable=lambda *_args, **_kwargs: None),
    )

    calls = {"construct": 0}

    class _FakeAF3Evaluator:
        output_dir = None

        def evaluate(self, _inputs, runtime_config=None):
            return SimpleNamespace(
                iptm={"AB": [0.81]},
                ptm={"AB": [0.72]},
                plddt={"AB": [0.9]},
            )

    def _construct_evaluators(_cfg, _out_root, oligo_names):
        calls["construct"] += 1
        assert oligo_names == ["AB"]
        return {"Alphafold3": _FakeAF3Evaluator()}

    monkeypatch.setattr("astrolabe.posthoc.refold.construct_evaluators", _construct_evaluators)

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
    )

    assert summary["failed"] == 0
    assert summary["ran_af3"] == 2
    assert summary["resource_groups"] == 2
    assert summary["evaluator_cache_misses"] == 2
    assert summary["evaluator_cache_hits"] == 0
    assert calls["construct"] == 2


def test_refold_parallel_shards_keep_same_resource_groups_together(tmp_path: Path) -> None:
    from astrolabe.posthoc import refold_plan

    source_a = tmp_path / "source_a" / "task_a" / "001_run"
    source_b = tmp_path / "source_b" / "task_b" / "001_run"
    source_c = tmp_path / "source_c" / "task_c" / "001_run"
    _write_refold_af3_config(source_a / "config.yaml", model_dir="/tmp/af3_shared")
    _write_refold_af3_config(source_b / "config.yaml", model_dir="/tmp/af3_shared")
    _write_refold_af3_config(source_c / "config.yaml", model_dir="/tmp/af3_other")

    item_a = _refold_manifest_item(source=source_a, task="task_a")
    item_b = _refold_manifest_item(source=source_b, task="task_b")
    item_c = _refold_manifest_item(source=source_c, task="task_c")
    plan = refold_plan.build_refold_plan(
        items=[item_a, item_b, item_c],
        requested_backends={"af3"},
        config_yaml=None,
        load_runtime_signature=refold_plan_runtime_signature,
    )

    shards, resource_groups = refold_plan.assign_refold_plan_to_worker_shards(
        plan=plan,
        worker_count=2,
    )

    locations = {
        str(item["task"]): shard_idx
        for shard_idx, shard in enumerate(shards)
        for item in shard
    }
    assert resource_groups == 2
    assert locations["task_a"] == locations["task_b"]
    assert locations["task_c"] != locations["task_a"]


def refold_plan_runtime_signature(path: Path, backend: str) -> dict[str, str]:
    if backend == "af3":
        text = Path(path).read_text(encoding="utf-8")
        model_dir = next(
            (line.split(":", 1)[1].strip() for line in text.splitlines() if "model_dir:" in line),
            "",
        )
        return {"backend": backend, "model_dir": model_dir}
    return {"backend": backend, "path": str(path)}


def test_run_refold_dry_run_accepts_manifest_without_src_config(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "dummy_task",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "seqs": {"A": "ACDE", "B": "FGHI"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "src_run_dir": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_refold(
        manifest_jsonl=manifest_path,
        out_root=tmp_path / "refold_runs",
        evaluators=["af3"],
        dry_run=True,
        workers=3,
    )

    assert summary["failed"] == 0
    assert summary["processed"] == 1
    assert summary["workers_requested"] == 3
    assert summary["workers"] == 1
    assert (tmp_path / "refold_runs" / "dummy_task" / "001_run" / "scores.csv").exists()


def test_refold_backend_selection_rejects_unknown_backend() -> None:
    from astrolabe.posthoc import refold_backends

    with pytest.raises(ValueError, match="Unsupported refold backend"):
        refold_backends.refold_backend_names(["af3", "esmfold"])


def test_run_refold_rejects_unknown_backend_before_creating_output(
    tmp_path: Path,
) -> None:
    from astrolabe.posthoc.refold import run_refold

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "dummy_task",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "seqs": {"A": "ACDE", "B": "FGHI"},
                "types": {"A": "protein", "B": "protein"},
                "homologies": {"A": None, "B": None},
                "src_run_dir": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "refold_runs"

    with pytest.raises(ValueError, match="Unsupported refold backend"):
        run_refold(
            manifest_jsonl=manifest_path,
            out_root=out_root,
            evaluators=["esmfold"],
            dry_run=True,
        )

    assert not out_root.exists()


def test_export_refold_manifest_includes_passing_rows_when_passed_only_disabled(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    _seed_refold_record(
        db_path=db_path,
        run_dir=run_dir,
        metrics={
            "stage": "default",
            "weighted_score": 1.0,
            "pA_sequence": "AAAA",
            "pB_sequence": "BBBB",
            "protenix_iptm": 0.91,
        },
    )
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_protenix",
                "rules:",
                "  - key: protenix_iptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.jsonl"
    summary = export_refold_manifest(
        db_path=db_path,
        preset_path=preset_path,
        out_manifest=manifest_path,
        run_selection="representative-per-run",
        passed_only=False,
    )

    assert summary["records_passed"] == 1
    assert summary["written"] == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["src_run_dir"] == str(run_dir.resolve())
    assert payload["needed_evaluators"] == []
