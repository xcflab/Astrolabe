from __future__ import annotations

from pathlib import Path

from astrolabe.posthoc.db import open_db, upsert_records, upsert_run_signature


def test_upsert_records_compacts_metrics_json_and_drops_duplicate_identity_keys(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "scores.db"
    conn = open_db(db_path)
    try:
        upsert_records(
            conn,
            [
                {
                    "run_dir": tmp_path / "outputs" / "task_a" / "001_run",
                    "task": "task_a",
                    "run": "001_run",
                    "step": 7,
                    "oligo": "AB",
                    "structure_path": "",
                    "metrics": {
                        "task": "task_a",
                        "run": "001_run",
                        "step": 7,
                        "oligo": "AB",
                        "weighted_score": 1.25,
                        "binder_chain": "B",
                    },
                    "scores_mtime_ns": 1,
                    "scores_size": 10,
                }
            ],
        )
        row = conn.execute("SELECT metrics_json FROM records").fetchone()
    finally:
        conn.close()

    payload = str(row["metrics_json"])
    assert payload == '{"binder_chain":"B","weighted_score":1.25}'


def test_upsert_run_signature_writes_compact_metadata_json(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    conn = open_db(db_path)
    try:
        upsert_run_signature(
            conn,
            run_dir=str(tmp_path / "outputs" / "task_a" / "001_run"),
            task="task_a",
            run="001_run",
            metadata={"stage_evaluator_aliases": {"default": "AF3Alias"}},
            scores_mtime_ns=1,
            scores_size=10,
        )
        row = conn.execute("SELECT metadata_json FROM runs").fetchone()
    finally:
        conn.close()

    assert str(row["metadata_json"]) == '{"stage_evaluator_aliases":{"default":"AF3Alias"}}'
