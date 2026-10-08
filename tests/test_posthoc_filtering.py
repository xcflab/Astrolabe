from __future__ import annotations

import csv
from pathlib import Path

import pytest

from astrolabe.posthoc.collect import collect_scores
from astrolabe.posthoc.db import get_meta, infer_db_out_root, iter_records, open_db, upsert_records
from astrolabe.posthoc.filtering import (
    FilterExpressionRuntimeError,
    MissingRequiredMetricsError,
    filter_structures,
    load_metrics_json,
    prune_outputs,
    preview_preset,
)


def _write_structure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data\n", encoding="utf-8")


def _seed_records(db_path: Path, records: list[dict]) -> None:
    conn = open_db(db_path)
    try:
        upsert_records(conn, records)
    finally:
        conn.close()


def _read_report_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_scores(run_dir: Path, rows: list[dict[str, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows cannot be empty")
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(h, "")) for h in headers))
    (run_dir / "scores.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_filter_structures_supports_filter_expression(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_ok = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_bad = tmp_path / "outputs" / "task_b" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_ok)
    _write_structure(structure_bad)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_ok,
                "metrics": {"weighted_score": 1.2, "binder_ptm": 0.82},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_b" / "001_run",
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_bad,
                "metrics": {"weighted_score": -0.1, "binder_ptm": 0.40},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0 and binder_ptm >= 0.7",
        dst_dir=dst_dir,
    )

    assert summary["selector_kind"] == "dsl"
    assert summary["filter_expr"] == "weighted_score > 0 and binder_ptm >= 0.7"
    assert summary["selected"] == 1
    assert summary["linked"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["passed"] == "True"
    assert (dst_dir / "task_a" / "task_a__001_run__step_000__AB_model.cif").exists()


def test_filter_report_uses_observed_oligo_and_provider_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_path = (
        tmp_path
        / "outputs"
        / "task_a"
        / "001_run"
        / "structures"
        / "step_000_ABCD_model.cif"
    )
    _write_structure(structure_path)
    _seed_records(
        db_path,
        [
            {
                "run_dir": structure_path.parents[1],
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "ABCD",
                "structure_path": structure_path,
                "metrics": {
                    "weighted_score": 1.2,
                    "ABCD.iptm": 0.76,
                    "ABCD.binder_iptm": 0.64,
                    "Alphafold3.iptm": 0.76,
                    "AB.iptm": 0.12,
                },
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0",
        dst_dir=dst_dir,
        prefer_oligo="ABCD",
    )

    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    fields = set(report_rows[0])
    assert "ABCD.iptm" in fields
    assert "ABCD.binder_iptm" in fields
    assert "Alphafold3.iptm" in fields
    assert "AB.iptm" not in fields
    assert "protenix_iptm" not in fields
    assert "boltz_iptm" not in fields


def test_filter_structures_writes_default_filtered_db_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    structure_ok = run_dir / "structures" / "step_000_AB_model.cif"
    structure_bad = tmp_path / "outputs" / "task_b" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_ok)
    _write_structure(structure_bad)

    _seed_records(
        db_path,
        [
            {
                "run_dir": run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_ok,
                "metrics": {"weighted_score": 1.2, "binder_ptm": 0.82},
                "scores_mtime_ns": 11,
                "scores_size": 101,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_b" / "001_run",
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_bad,
                "metrics": {"weighted_score": -0.1, "binder_ptm": 0.40},
                "scores_mtime_ns": 12,
                "scores_size": 102,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0 and binder_ptm >= 0.7",
        dst_dir=dst_dir,
    )

    out_db = dst_dir / "filtered.sqlite"
    assert summary["out_db"] == str(out_db.resolve())
    assert out_db.exists()

    conn = open_db(out_db)
    try:
        rows = list(iter_records(conn, oligo="AB"))
        assert len(rows) == 1
        row = rows[0]
        metrics = load_metrics_json(row)
        assert row["run_dir"] == str(run_dir)
        assert row["structure_path"] == str(dst_dir / "task_a" / "task_a__001_run__step_000__AB_model.cif")
        assert metrics["weighted_score"] == 1.2
        assert metrics["filtered_source_db"] == str(db_path.resolve())
        assert metrics["filtered_source_run_dir"] == str(run_dir)
        assert metrics["filtered_source_structure"] == str(structure_ok)
        assert metrics["filtered_dst_structure"] == row["structure_path"]
        assert get_meta(conn, key="dataset_kind") == "posthoc_filtered"
        assert get_meta(conn, key="filtered_filter_expr") == "weighted_score > 0 and binder_ptm >= 0.7"
        assert infer_db_out_root(conn) == str(dst_dir.resolve())
    finally:
        conn.close()


def test_filter_structures_respects_explicit_out_db_path(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    explicit_db = tmp_path / "datasets" / "custom.sqlite"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"binder_ptm": 0.82},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.8",
        dst_dir=dst_dir,
        out_db=explicit_db,
    )

    assert summary["out_db"] == str(explicit_db.resolve())
    assert explicit_db.exists()
    assert not (dst_dir / "filtered.sqlite").exists()


def test_prune_outputs_dry_run_and_delete_raw_model_results(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    structure_path = run_dir / "structures" / "step_000_AB_model.cif"
    raw_dir = run_dir / "raw_model_results"
    _write_structure(structure_path)
    (raw_dir / "alphafold3").mkdir(parents=True, exist_ok=True)
    ((raw_dir / "alphafold3") / "payload.txt").write_text("payload\n", encoding="utf-8")

    _seed_records(
        db_path,
        [
            {
                "run_dir": run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"binder_ptm": 0.82},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    dry_summary = prune_outputs(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.8",
        delete_scope="raw_model_results",
        dry_run=True,
        report_csv=tmp_path / "dry_report.csv",
    )
    assert dry_summary["selected"] == 1
    assert dry_summary["deleted_paths"] == 0
    assert raw_dir.exists()

    real_summary = prune_outputs(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.8",
        delete_scope="raw_model_results",
        dry_run=False,
        report_csv=tmp_path / "real_report.csv",
    )
    assert real_summary["selected"] == 1
    assert real_summary["deleted_paths"] == 1
    assert not raw_dir.exists()


def test_prune_outputs_expression_missing_metric_fails_fast_without_deleting(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    structure_path = run_dir / "structures" / "step_000_AB_model.cif"
    raw_dir = run_dir / "raw_model_results"
    _write_structure(structure_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "payload.txt").write_text("payload\n", encoding="utf-8")

    _seed_records(
        db_path,
        [
            {
                "run_dir": run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    with pytest.raises(MissingRequiredMetricsError, match="binder_ptm"):
        prune_outputs(
            db_path=db_path,
            filter_expr="weighted_score > 0 and binder_ptm >= 0.7",
            delete_scope="raw_model_results",
            dry_run=False,
            report_csv=tmp_path / "prune_report.csv",
        )

    assert raw_dir.exists()
    assert not (tmp_path / "prune_report.csv").exists()


def test_filter_structures_supports_evaluator_alias_metrics_from_collect(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "outputs"
    run_dir = out_root / "task_a" / "001_run"
    structure_path = run_dir / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "stages:",
                "  default:",
                "    evaluator: StrictPX",
                "evaluator:",
                "  evaluator_terms:",
                "    StrictPX:",
                "      _type: Protenix",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_scores(
        run_dir,
        [
            {
                "step": "0",
                "stage": "default",
                "weighted_score": "1.0",
                "binder_ptm": "0.82",
            }
        ],
    )
    collect_scores(
        out_root=out_root,
        db_path=tmp_path / "collect.db",
        workers=1,
        fail_on_missing_required=False,
    )

    summary = filter_structures(
        db_path=tmp_path / "collect.db",
        filter_expr="StrictPX.binder_ptm >= 0.8",
        dst_dir=tmp_path / "filtered",
    )

    assert summary["selected"] == 1
    report_rows = _read_report_rows(tmp_path / "filtered" / "filter_report.csv")
    assert report_rows[0]["StrictPX.binder_ptm"] == "0.82"


def test_filter_structures_supports_dotted_metric_names_in_expression(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2, "AB.iptm": 0.76},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0 and AB.iptm >= 0.7",
        dst_dir=dst_dir,
    )

    assert summary["selected"] == 1
    assert summary["linked"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert report_rows[0]["passed"] == "True"
    assert report_rows[0]["AB.iptm"] == "0.76"


def test_filter_structures_supports_row_metadata_columns_in_expression(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    run_dir = tmp_path / "outputs" / "03_runs" / "task_a" / "001_run"
    structure_path = run_dir / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr=(
            f'run_dir == "{run_dir}" and structure_path == "{structure_path}" and weighted_score > 0'
        ),
        dst_dir=dst_dir,
    )

    assert summary["selected"] == 1
    assert summary["linked"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert report_rows[0]["passed"] == "True"
    assert report_rows[0]["structure_path"] == str(structure_path)


def test_filter_structures_infers_db_out_root_from_legacy_db(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    out_root = tmp_path / "outputs"
    run_dir = out_root / "03_runs" / "task_a" / "001_run"
    structure_path = run_dir / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0",
        dst_dir=dst_dir,
    )

    assert summary["db_out_root"] == str(out_root)


def test_collect_scores_persists_db_out_root_for_filter_summary(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    run_dir = out_root / "task_a" / "001_run"
    _write_scores(
        run_dir,
        [{"step": "0", "weighted_score": "1.0", "stage": "default"}],
    )

    collect_scores(
        out_root=out_root,
        db_path=tmp_path / "collect.db",
        workers=1,
    )

    summary = filter_structures(
        db_path=tmp_path / "collect.db",
        filter_expr="weighted_score > 0",
        dst_dir=tmp_path / "filtered",
    )

    assert summary["db_out_root"] == str(out_root)


def test_filter_structures_representative_per_run_selects_best_from_last_stage(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_a = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_b = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_001_AB_model.cif"
    structure_c = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_002_AB_model.cif"
    _write_structure(structure_a)
    _write_structure(structure_b)
    _write_structure(structure_c)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_a,
                "metrics": {"stage": "explore", "weighted_score": 5.0, "binder_ptm": 0.91},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "structure_path": structure_b,
                "metrics": {"stage": "refine", "weighted_score": 0.4, "binder_ptm": 0.93},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 2,
                "oligo": "AB",
                "structure_path": structure_c,
                "metrics": {"stage": "refine", "weighted_score": 0.9, "binder_ptm": 0.95},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.9",
        dst_dir=dst_dir,
        run_selection="representative-per-run",
    )

    assert summary["selected"] == 1
    assert summary["linked"] == 1
    assert summary["records_considered"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["step"] == "2"
    assert report_rows[0]["stage"] == "refine"
    assert (dst_dir / "task_a" / "task_a__001_run__step_002__AB_model.cif").exists()


def test_filter_structures_representative_per_run_filters_representative_only(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_a = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_b = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_001_AB_model.cif"
    structure_c = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_002_AB_model.cif"
    _write_structure(structure_a)
    _write_structure(structure_b)
    _write_structure(structure_c)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_a,
                "metrics": {"stage": "explore", "weighted_score": 5.0, "binder_ptm": 0.95},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "structure_path": structure_b,
                "metrics": {"stage": "refine", "weighted_score": 0.4, "binder_ptm": 0.92},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 2,
                "oligo": "AB",
                "structure_path": structure_c,
                "metrics": {"stage": "refine", "weighted_score": 0.9, "binder_ptm": 0.10},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    all_summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.9",
        dst_dir=dst_dir / "all",
        run_selection="all-passing",
    )
    representative_summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.9",
        dst_dir=dst_dir / "representative",
        run_selection="representative-per-run",
    )

    assert all_summary["selected"] == 2
    assert all_summary["linked"] == 2
    assert representative_summary["selected"] == 0
    assert representative_summary["linked"] == 0
    assert representative_summary["records_considered"] == 1


def test_filter_structures_representative_per_run_uses_src_run_dir_as_logical_run(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    src_run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    refold_run_dir = tmp_path / "refold_runs" / "task_a" / "001_run"
    structure_src = src_run_dir / "structures" / "step_000_AB_model.cif"
    structure_refold = refold_run_dir / "structures" / "step_001_AB_model.cif"
    _write_structure(structure_src)
    _write_structure(structure_refold)

    _seed_records(
        db_path,
        [
            {
                "run_dir": src_run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_src,
                "metrics": {"stage": "default", "weighted_score": 0.2, "binder_ptm": 0.65},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": refold_run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 1,
                "oligo": "AB",
                "structure_path": structure_refold,
                "metrics": {
                    "stage": "refold",
                    "weighted_score": 0.9,
                    "binder_ptm": 0.95,
                    "src_run_dir": str(src_run_dir),
                },
                "scores_mtime_ns": 2,
                "scores_size": 20,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.9",
        dst_dir=dst_dir,
        run_selection="representative-per-run",
    )

    assert summary["records_considered"] == 1
    assert summary["selected"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["run_dir"] == str(refold_run_dir)
    assert report_rows[0]["step"] == "1"
    assert report_rows[0]["stage"] == "refold"


def test_filter_structures_representative_per_run_prefers_refold_stage_even_when_original_has_later_step(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    src_run_dir = tmp_path / "outputs" / "task_a" / "001_run"
    refold_run_dir = tmp_path / "refold_runs" / "task_a" / "001_run"
    structure_src = src_run_dir / "structures" / "step_050_AB_model.cif"
    structure_refold = refold_run_dir / "structures" / "step_018_AB_model.cif"
    _write_structure(structure_src)
    _write_structure(structure_refold)

    _seed_records(
        db_path,
        [
            {
                "run_dir": src_run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 50,
                "oligo": "AB",
                "structure_path": structure_src,
                "metrics": {"stage": "af3_final", "weighted_score": 1.5},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": refold_run_dir,
                "task": "task_a",
                "run": "001_run",
                "step": 18,
                "oligo": "AB",
                "structure_path": structure_refold,
                "metrics": {
                    "stage": "refold",
                    "weighted_score": 0.9,
                    "protenix_iptm": 0.8,
                    "src_run_dir": str(src_run_dir),
                },
                "scores_mtime_ns": 2,
                "scores_size": 20,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="protenix_iptm > 0.7",
        dst_dir=dst_dir,
        run_selection="representative-per-run",
    )

    assert summary["records_considered"] == 1
    assert summary["selected"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["run_dir"] == str(refold_run_dir)
    assert report_rows[0]["step"] == "18"
    assert report_rows[0]["stage"] == "refold"


def test_filter_structures_expression_missing_metric_fails_fast(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    with pytest.raises(MissingRequiredMetricsError, match="binder_ptm"):
        filter_structures(
            db_path=db_path,
            filter_expr="weighted_score > 0 and binder_ptm >= 0.7",
            dst_dir=dst_dir,
        )

    assert (dst_dir / "filter_report.csv").exists()
    assert not (dst_dir / "filtered.sqlite").exists()


def test_filter_structures_preset_missing_metric_suggests_selector_driven_refold(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    preset_path = tmp_path / "preset.yaml"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)
    preset_path.write_text(
        "\n".join(
            [
                "name: needs_refold",
                "rules:",
                "  - key: binder_ptm",
                "    op: ge",
                "    value: 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    with pytest.raises(MissingRequiredMetricsError, match=r"gh posthoc refold --db"):
        filter_structures(
            db_path=db_path,
            preset_path=preset_path,
            dst_dir=dst_dir,
        )


def test_filter_structures_expression_missing_guard_allows_filtering(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_missing = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_fail = tmp_path / "outputs" / "task_b" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_missing)
    _write_structure(structure_fail)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_missing,
                "metrics": {"accepted": True, "weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_b" / "001_run",
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_fail,
                "metrics": {"accepted": True, "weighted_score": 1.2, "A.A_loop_frac": 0.4},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="accepted == True and weighted_score > 0 and (missing(A.A_loop_frac) or A.A_loop_frac < 0.25)",
        dst_dir=dst_dir,
        prefer_oligo="AB",
    )

    assert summary["selected"] == 1
    assert summary["linked"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["passed"] == "True"
    assert report_rows[0]["missing"] == "A.A_loop_frac"


def test_filter_structures_expression_runtime_error_fails_fast(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    with pytest.raises(FilterExpressionRuntimeError, match=r"runtime error"):
        filter_structures(
            db_path=db_path,
            filter_expr="missing(weighted_score, weighted_score)",
            dst_dir=dst_dir,
        )


def test_filter_structures_expression_invalid_metric_value_fails_fast(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_path,
                "metrics": {"weighted_score": 1.2, "binder_ptm": ""},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            }
        ],
    )

    with pytest.raises(MissingRequiredMetricsError, match=r"invalid_required: binder_ptm"):
        filter_structures(
            db_path=db_path,
            filter_expr="weighted_score > 0 and binder_ptm >= 0.7",
            dst_dir=dst_dir,
        )


def test_filter_structures_counts_passed_rows_even_when_structure_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_missing = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_fail = tmp_path / "outputs" / "task_b" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_fail)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_missing,
                "metrics": {"binder_iptm": 0.82},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_b" / "001_run",
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_fail,
                "metrics": {"binder_iptm": 0.12},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_iptm > 0.7",
        dst_dir=dst_dir,
    )

    assert summary["selected"] == 1
    assert summary["linked"] == 0
    assert summary["structure_missing"] == 1
    report_rows = _read_report_rows(dst_dir / "filter_report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["task"] == "task_a"
    assert report_rows[0]["passed"] == "True"
    assert report_rows[0]["dst_structure"] == ""
    assert report_rows[0]["link_issue"] == "structure_missing"


def test_filter_structures_dry_run_counts_without_writing_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    dst_dir = tmp_path / "filtered"
    structure_ok = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    structure_missing = tmp_path / "outputs" / "task_b" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_ok)

    _seed_records(
        db_path,
        [
            {
                "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                "task": "task_a",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_ok,
                "metrics": {"binder_ptm": 0.82},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
            {
                "run_dir": tmp_path / "outputs" / "task_b" / "001_run",
                "task": "task_b",
                "run": "001_run",
                "step": 0,
                "oligo": "AB",
                "structure_path": structure_missing,
                "metrics": {"binder_ptm": 0.91},
                "scores_mtime_ns": 1,
                "scores_size": 10,
            },
        ],
    )

    summary = filter_structures(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.8",
        dst_dir=dst_dir,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["selected"] == 2
    assert summary["linkable"] == 1
    assert summary["structure_missing"] == 1
    assert summary["out_db"] == ""
    assert not summary["report_csv"]
    assert not dst_dir.exists()
    assert not (dst_dir / "filter_report.csv").exists()
    assert not (dst_dir / "filtered.sqlite").exists()


def test_preview_preset_renders_rule_summary(tmp_path: Path) -> None:
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "name: demo_preset",
                "rules:",
                "  - key: weighted_score",
                "    op: gt",
                "    value: 0",
                "  - key: AB.iptm",
                "    op: ge",
                "    value: 0.7",
                "    allow_missing: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    preview = preview_preset(preset_path)

    assert preview["preset"] == "demo_preset"
    assert preview["rule_count"] == 2
    assert preview["required_metrics"] == ["weighted_score"]
    assert "weighted_score > 0" in preview["preview_filter_expr"]
    assert "missing(AB.iptm) or AB.iptm >= 0.7" in preview["preview_filter_expr"]


def test_collect_scores_merges_out_roots_metadata_across_repeated_collects(tmp_path: Path) -> None:
    out_root_a = tmp_path / "outputs_a"
    out_root_b = tmp_path / "outputs_b"
    _write_scores(
        out_root_a / "task_a" / "001_run",
        [{"step": "0", "weighted_score": "1.0", "stage": "default"}],
    )
    _write_scores(
        out_root_b / "task_b" / "001_run",
        [{"step": "0", "weighted_score": "2.0", "stage": "refold", "src_run_dir": str(out_root_a / "task_a" / "001_run")}],
    )

    db_path = tmp_path / "collect.db"
    collect_scores(out_root=out_root_a, db_path=db_path, workers=1)
    collect_scores(out_root=out_root_b, db_path=db_path, workers=1)

    conn = open_db(db_path)
    try:
        out_roots_json = get_meta(conn, key="out_roots_json")
    finally:
        conn.close()

    assert out_roots_json is not None
    assert str(out_root_a.resolve()) in out_roots_json
    assert str(out_root_b.resolve()) in out_roots_json


def test_filter_structures_omits_db_out_root_when_multiple_roots_only_share_filesystem_root(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    structure_path = tmp_path / "outputs" / "task_a" / "001_run" / "structures" / "step_000_AB_model.cif"
    _write_structure(structure_path)

    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                    "task": "task_a",
                    "run": "001_run",
                    "step": 0,
                    "oligo": "AB",
                    "structure_path": structure_path,
                    "metrics": {"weighted_score": 1.0},
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
        from astrolabe.posthoc.db import set_out_roots

        set_out_roots(
            conn,
            roots=[
                str((tmp_path / "outputs").resolve()),
                "/var/tmp/other-root",
            ],
        )
    finally:
        conn.close()

    summary = filter_structures(
        db_path=db_path,
        filter_expr="weighted_score > 0",
        dst_dir=tmp_path / "filtered",
    )

    assert summary["db_out_root"] == ""
