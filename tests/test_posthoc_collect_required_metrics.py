from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from astrolabe.posthoc.collect import (
    MissingRequiredMetricsError,
    _augment_with_scalar_defaults,
    collect_scores,
)
from astrolabe.posthoc.db import iter_records, open_db


def _write_scores(run_dir: Path, rows: list[dict[str, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows cannot be empty")
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(h, "")) for h in headers))
    (run_dir / "scores.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_config(run_dir: Path, text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(text, encoding="utf-8")


def test_scalar_defaults_do_not_mislabel_complex_scores_as_binder_scores() -> None:
    metrics = {
        "ABCD.iptm": 0.71,
        "ABCD.ptm": 0.82,
        "AB.iptm": 0.69,
        "AB.ptm": 0.80,
    }

    _augment_with_scalar_defaults(metrics=metrics, oligo="ABCD")

    assert "binder_iptm" not in metrics
    assert "binder_ptm" not in metrics


def test_scalar_defaults_promote_only_explicit_binder_scores() -> None:
    metrics = {
        "ABCD.binder_iptm": 0.61,
        "ABCD.binder_ptm": 0.72,
    }

    _augment_with_scalar_defaults(metrics=metrics, oligo="ABCD")

    assert metrics["binder_iptm"] == 0.61
    assert metrics["binder_iptm_provenance"] == "scores:ABCD.binder_iptm"
    assert metrics["binder_ptm"] == 0.72
    assert metrics["binder_ptm_provenance"] == "scores:ABCD.binder_ptm"


def test_collect_scores_fails_fast_when_required_metric_missing(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    _write_scores(
        out_root / "task_ok" / "001_run",
        [{"step": "0", "AB.ptm": "0.72"}],
    )
    _write_scores(
        out_root / "task_bad" / "001_run",
        [{"step": "0", "weighted_score": "1.0"}],
    )

    with pytest.raises(MissingRequiredMetricsError, match="required_metrics"):
        collect_scores(
            out_root=out_root,
            db_path=tmp_path / "collect.db",
            workers=1,
            required_metrics=["binder_ptm"],
        )


def test_collect_scores_allows_missing_required_when_opted_out(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    _write_scores(
        out_root / "task_bad" / "001_run",
        [{"step": "0", "weighted_score": "1.0"}],
    )

    summary = collect_scores(
        out_root=out_root,
        db_path=tmp_path / "collect.db",
        workers=1,
        required_metrics=["binder_ptm"],
        fail_on_missing_required=False,
    )

    assert summary["required_metrics"] == ["binder_ptm"]
    assert summary["required_failed_records_total"] == 1
    assert summary["missing_required_counts"]["binder_ptm"] == 1


def test_collect_scores_supports_multiple_out_roots(tmp_path: Path) -> None:
    out_root_a = tmp_path / "outputs_a"
    out_root_b = tmp_path / "outputs_b"
    _write_scores(
        out_root_a / "task_a" / "001_run",
        [{"step": "0", "weighted_score": "1.0"}],
    )
    _write_scores(
        out_root_b / "task_b" / "001_run",
        [{"step": "0", "weighted_score": "2.0"}],
    )

    db_path = tmp_path / "collect.db"
    summary = collect_scores(
        out_roots=[out_root_a, out_root_b],
        db_path=db_path,
        workers=1,
        fail_on_missing_required=False,
    )

    assert summary["runs_found"] == 2
    assert summary["runs_processed"] == 2
    assert summary["out_roots"] == [str(out_root_a.resolve()), str(out_root_b.resolve())]

    conn = open_db(db_path)
    try:
        tasks = sorted({str(row["task"]) for row in iter_records(conn)})
    finally:
        conn.close()

    assert tasks == ["task_a", "task_b"]


def test_collect_scores_reprocesses_when_chain_context_changes(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "outputs"
    run_dir = out_root / "task_a" / "001_run"
    _write_scores(run_dir, [{"step": "0", "weighted_score": "1.0"}])
    db_path = tmp_path / "collect.db"

    first = collect_scores(
        out_root=out_root,
        db_path=db_path,
        workers=1,
        binder_chain="A",
        target_chain="B",
    )
    unchanged = collect_scores(
        out_root=out_root,
        db_path=db_path,
        workers=1,
        binder_chain="A",
        target_chain="B",
    )
    regrouped = collect_scores(
        out_root=out_root,
        db_path=db_path,
        workers=1,
        binder_chain="A",
        target_chain="B,C,D",
    )

    assert first["runs_processed"] == 1
    assert unchanged["runs_processed"] == 0
    assert unchanged["runs_skipped"] == 1
    assert regrouped["runs_processed"] == 1
    assert regrouped["runs_skipped"] == 0

    conn = open_db(db_path)
    try:
        row = next(iter_records(conn))
        metadata_json = conn.execute(
            "select metadata_json from runs where run_dir = ?",
            (str(run_dir.resolve()),),
        ).fetchone()[0]
    finally:
        conn.close()
    metrics = json.loads(str(row["metrics_json"]))
    metadata = json.loads(str(metadata_json))
    assert metrics["binder_chain"] == "A"
    assert metrics["target_chain"] == "B,C,D"
    assert metadata["posthoc_collection_context"] == {
        "derivation_version": 3,
        "binder_chain": "A",
        "target_chain": "B,C,D",
    }


def test_collect_scores_persists_evaluator_alias_metadata_and_alias_metrics(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "outputs"
    run_dir = out_root / "task_a" / "001_run"
    _write_run_config(
        run_dir,
        "\n".join(
            [
                "stages:",
                "  default:",
                "    evaluator: StrictPX",
                "  af3_final:",
                "    evaluator: FastAF3",
                "evaluator:",
                "  evaluator_terms:",
                "    StrictPX:",
                "      _type: Protenix",
                "    FastAF3:",
                "      _type: Alphafold3",
            ]
        )
        + "\n",
    )
    _write_scores(
        run_dir,
        [
            {
                "step": "0",
                "stage": "default",
                "weighted_score": "1.0",
                "binder_ptm": "0.82",
                "iptm": "0.71",
                "contact_molecular_surface": "612.0",
                "binder_rog": "11.2",
            }
        ],
    )

    collect_scores(
        out_root=out_root,
        db_path=tmp_path / "collect.db",
        workers=1,
        fail_on_missing_required=False,
    )

    conn = open_db(tmp_path / "collect.db")
    try:
        row = next(iter_records(conn))
        metadata_json = conn.execute(
            "select metadata_json from runs where run_dir = ?",
            (str(run_dir.resolve()),),
        ).fetchone()[0]
    finally:
        conn.close()

    assert '"StrictPX"' in str(metadata_json)
    assert '"FastAF3"' in str(metadata_json)
    metrics_json = json.loads(str(row["metrics_json"]))
    assert metrics_json["StrictPX.binder_ptm"] == 0.82
    assert metrics_json["StrictPX.iptm"] == 0.71
    assert "StrictPX.contact_molecular_surface" not in metrics_json
    assert "StrictPX.binder_rog" not in metrics_json


def test_collect_scores_can_archive_raw_model_results(tmp_path: Path) -> None:
    out_root = tmp_path / "outputs"
    run_dir = out_root / "task_a" / "001_run"
    _write_scores(
        run_dir,
        [{"step": "0", "weighted_score": "1.0"}],
    )
    raw_dir = run_dir / "raw_model_results" / "alphafold3"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "payload.txt").write_text("payload\n", encoding="utf-8")

    summary = collect_scores(
        out_root=out_root,
        db_path=tmp_path / "collect.db",
        workers=1,
        archive_dir=tmp_path / "archives",
        archive_workers=1,
        fail_on_missing_required=False,
    )

    assert summary["archive_dir"] == str((tmp_path / "archives").resolve())
    assert summary["archived_created"] == 1
    archives = sorted((tmp_path / "archives").rglob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as tf:
        names = tf.getnames()
    assert "raw_model_results/alphafold3/payload.txt" in names


def test_collect_scores_uses_source_run_alias_context_for_refold_rows(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source_outputs" / "task_a" / "001_run"
    _write_run_config(
        source_run_dir,
        "\n".join(
            [
                "stages:",
                "  default:",
                "    evaluator: StrictPX",
                "  af3_final:",
                "    evaluator: FastAF3",
                "evaluator:",
                "  evaluator_terms:",
                "    StrictPX:",
                "      _type: protenix",
                "    FastAF3:",
                "      _type: alphafold3",
            ]
        )
        + "\n",
    )

    refold_root = tmp_path / "refold_outputs"
    refold_run_dir = refold_root / "task_a" / "001_run"
    _write_scores(
        refold_run_dir,
        [
            {
                "step": "1",
                "stage": "refold",
                "src_run_dir": str(source_run_dir.resolve()),
                "refold_backends": "protenix",
                "weighted_score": "",
            }
        ],
    )
    summary_dir = (
        refold_run_dir
        / "raw_model_results"
        / "protenix"
        / "step_001"
        / "AB"
        / "seed_1"
        / "predictions"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "AB_seed_1_summary_confidence_sample_0.json").write_text(
        json.dumps(
            {
                "iptm": 0.73,
                "ptm": 0.66,
                "plddt": 0.84,
                "ranking_score": 0.73,
                "chain_ptm": [0.61, 0.66],
                "chain_iptm": [0.12, 0.73],
            }
        ),
        encoding="utf-8",
    )

    collect_scores(
        out_root=refold_root,
        db_path=tmp_path / "collect.db",
        workers=1,
        fail_on_missing_required=False,
    )

    conn = open_db(tmp_path / "collect.db")
    try:
        row = next(iter_records(conn))
    finally:
        conn.close()

    metrics_json = json.loads(str(row["metrics_json"]))
    assert metrics_json["StrictPX.protenix_iptm"] == 0.73
