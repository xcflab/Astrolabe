from __future__ import annotations

import csv
from pathlib import Path

from astrolabe.posthoc.db import open_db, upsert_records
from astrolabe.posthoc.report import generate_posthoc_report


def _seed_records(db_path: Path, records: list[dict]) -> None:
    conn = open_db(db_path)
    try:
        upsert_records(conn, records)
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_posthoc_report_distinguishes_all_designs_vs_representative_success_rate(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    records = []
    for run_name in ["001_run", "002_run"]:
        for step, score, metric in [
            (0, 1.0, 0.95),
            (1, 0.2, 0.10),
        ]:
            records.append(
                {
                    "run_dir": tmp_path / "outputs" / "task_a" / run_name,
                    "task": "task_a",
                    "run": run_name,
                    "step": step,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": {
                        "stage": "default" if step == 0 else "refine",
                        "weighted_score": score,
                        "binder_ptm": metric,
                    },
                    "scores_mtime_ns": step + 1,
                    "scores_size": 10,
                }
            )
    _seed_records(db_path, records)

    summary = generate_posthoc_report(
        db_path=db_path,
        filter_expr="binder_ptm >= 0.8",
        prefer_oligo="AB",
        dst_dir=tmp_path / "report",
        trend_metrics=["binder_ptm", "weighted_score"],
    )

    assert summary["selector_kind"] == "dsl"
    assert (tmp_path / "report" / "posthoc_summary.csv").exists()
    assert (tmp_path / "report" / "posthoc_per_step.csv").exists()
    assert (tmp_path / "report" / "posthoc_report.md").exists()
    assert (tmp_path / "report" / "step_trends.png").exists()
    assert (tmp_path / "report" / "success_rate.png").exists()

    summary_rows = _read_csv(tmp_path / "report" / "posthoc_summary.csv")
    by_scope = {row["scope"]: row for row in summary_rows}
    assert by_scope["all_designs"]["success_rate"] == "0.5"
    assert by_scope["representative_per_run"]["success_rate"] == "0.0"

    per_step_rows = _read_csv(tmp_path / "report" / "posthoc_per_step.csv")
    rep_rows = [row for row in per_step_rows if row["scope"] == "representative_per_run"]
    assert any(row["step"] == "0" and row["success_rate"] == "1.0" for row in rep_rows)
    assert any(row["step"] == "1" and row["success_rate"] == "0.0" for row in rep_rows)
