from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

import astrolabe.cli as gh_cli
import astrolabe.experimental.candidate_batch_runner as candidate_batch_runner
from astrolabe.experimental.candidate_batch_runner import (
    CandidateDecisionRun,
    CandidateExecutionError,
    CandidatePoolConfig,
    CandidatePoolScheduler,
    CandidateResult,
    CandidateTask,
    ProcessCandidateBatchAdapters,
    ProcessCandidateOutputReader,
    ProcessCandidateRunnerConfig,
    _best_score_row,
    _candidate_seed_offset,
    _extract_sequences,
    run_process_candidate_batch,
)


def _task(candidate_id: str, score: float, delay_s: float = 0.0) -> CandidateTask:
    return CandidateTask(
        candidate_id=candidate_id,
        generation=1,
        payload={"score": score, "delay_s": delay_s},
    )


def _fake_worker(task: CandidateTask) -> CandidateResult:
    payload = dict(task.payload or {})
    delay_s = float(payload.get("delay_s", 0.0) or 0.0)
    if delay_s:
        time.sleep(delay_s)
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return CandidateResult(
        task=task,
        status="completed",
        score=float(payload["score"]),
        payload={"candidate_id": task.candidate_id},
    )


def test_candidate_pool_collects_min_completed_and_carries_straggler() -> None:
    cfg = CandidatePoolConfig(num_candidates=3, min_completed=2, max_wait_s=1.0)
    with CandidatePoolScheduler(cfg, _fake_worker) as scheduler:
        scheduler.submit_generation(
            1,
            [
                _task("fast_best", 0.8, 0.01),
                _task("slow_straggler", 0.9, 0.2),
                _task("fast_worse", 0.5, 0.01),
            ],
        )
        first = scheduler.collect_until_ready()

        assert first is not None
        assert first.trigger == "min_completed"
        assert first.selected is not None
        assert first.selected.candidate_id == "fast_best"
        assert first.carried_forward_candidate_ids == ("slow_straggler",)

        second = scheduler.collect_until_ready()

    assert second is not None
    assert second.trigger == "drain"
    assert second.selected is not None
    assert second.selected.candidate_id == "slow_straggler"


def test_candidate_pool_max_wait_s_allows_partial_collection() -> None:
    cfg = CandidatePoolConfig(num_candidates=2, min_completed=2, max_wait_s=0.05)
    with CandidatePoolScheduler(cfg, _fake_worker) as scheduler:
        scheduler.submit_generation(
            1,
            [
                _task("fast", 0.7, 0.01),
                _task("slow", 0.9, 0.2),
            ],
        )
        first = scheduler.collect_until_ready()

    assert first is not None
    assert first.trigger == "max_wait_s"
    assert first.selected is not None
    assert first.selected.candidate_id == "fast"
    assert first.carried_forward_candidate_ids == ("slow",)


def test_candidate_pool_max_wait_s_fails_when_no_candidate_finishes() -> None:
    cfg = CandidatePoolConfig(num_candidates=1, min_completed=1, max_wait_s=0.01)
    with pytest.raises(CandidateExecutionError, match="No candidate finished"):
        with CandidatePoolScheduler(cfg, _fake_worker) as scheduler:
            scheduler.submit_generation(1, [_task("slow", 0.9, 0.2)])
            scheduler.collect_until_ready()


def test_candidate_pool_records_failed_candidates_without_hiding_them() -> None:
    cfg = CandidatePoolConfig(num_candidates=2, min_completed=2, max_wait_s=1.0)
    failing = CandidateTask(
        candidate_id="failed",
        generation=1,
        payload={"score": 0.0, "error": "model failed"},
    )
    with CandidatePoolScheduler(cfg, _fake_worker) as scheduler:
        scheduler.submit_generation(1, [_task("ok", 0.6, 0.01), failing])
        collection = scheduler.collect_until_ready()

    assert collection is not None
    assert collection.selected is not None
    assert collection.selected.candidate_id == "ok"
    assert [result.candidate_id for result in collection.failed] == ["failed"]
    assert "model failed" in str(collection.failed[0].error)
    row = collection.as_row()
    failed_rows = row["failed_candidates"]
    assert isinstance(failed_rows, list)
    assert failed_rows[0]["candidate_id"] == "failed"
    assert failed_rows[0]["error"] == "model failed"
    assert failed_rows[0]["duration_s"] == pytest.approx(
        collection.failed[0].duration_s
    )
    assert failed_rows[0]["payload"] is None


def test_candidate_pool_fail_fast_raises_on_failed_candidate() -> None:
    cfg = CandidatePoolConfig(
        num_candidates=1,
        min_completed=1,
        max_wait_s=1.0,
        fail_policy="fail-fast",
    )
    failing = CandidateTask(
        candidate_id="failed",
        generation=1,
        payload={"score": 0.0, "error": "model failed"},
    )
    with CandidatePoolScheduler(cfg, _fake_worker) as scheduler:
        scheduler.submit_generation(1, [failing])
        with pytest.raises(CandidateExecutionError, match="model failed"):
            scheduler.collect_until_ready()


def test_best_score_row_prefers_accepted_rows(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    scores_csv.write_text(
        "\n".join(
            [
                "step,weighted_score,accepted,pA_sequence",
                "0,0.95,false,AAAA",
                "1,0.50,true,AAAC",
                "2,0.70,true,AAAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = _best_score_row(scores_csv)

    assert row["step"] == "2"
    assert _extract_sequences(row) == {"A": "AAAG"}


def test_process_candidate_output_reader_preserves_candidate_score_rule(
    tmp_path: Path,
) -> None:
    scores_csv = tmp_path / "out" / "candidate_task" / "001_run" / "scores.csv"
    scores_csv.parent.mkdir(parents=True)
    scores_csv.write_text(
        "\n".join(
            [
                "step,weighted_score,accepted,pA_sequence",
                "0,0.95,false,AAAA",
                "1,0.50,true,AAAC",
                "2,0.70,true,AAAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = ProcessCandidateOutputReader(tmp_path / "out").read_best_score(
        "candidate_task"
    )

    assert result.weighted_score == pytest.approx(0.70)
    assert result.sequences == {"A": "AAAG"}


def test_process_candidate_output_reader_uses_sorted_last_scores_csv(
    tmp_path: Path,
) -> None:
    older = tmp_path / "out" / "candidate_task" / "001_run" / "scores.csv"
    newer = tmp_path / "out" / "candidate_task" / "002_run" / "scores.csv"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text(
        "step,weighted_score,accepted,pA_sequence\n0,0.95,true,OLDER\n",
        encoding="utf-8",
    )
    newer.write_text(
        "step,weighted_score,accepted,pA_sequence\n0,0.50,true,NEWER\n",
        encoding="utf-8",
    )

    result = ProcessCandidateOutputReader(tmp_path / "out").read_best_score(
        "candidate_task"
    )

    assert result.weighted_score == pytest.approx(0.50)
    assert result.sequences == {"A": "NEWER"}


def test_process_candidate_batch_dry_run_writes_candidate_configs(tmp_path: Path) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
                "seqs:",
                "  A: 10-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=2,
            max_wait_s=1.0,
            decision_rounds=1,
            candidate_steps=2,
            dry_run=True,
        )
    )

    assert summary["dry_run"] is True
    assert summary["collections"][0]["selected_candidate_id"] == "cand_0001"
    generated = sorted((tmp_path / "out" / "candidate_pool").rglob("config.yaml"))
    assert len(generated) == 2
    assert "steps: 2" in generated[0].read_text(encoding="utf-8")


def test_process_candidate_batch_dry_run_uses_stable_candidate_seeds(
    tmp_path: Path,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 100",
                "steps: 5",
                "seqs:",
                "  A: 10-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=2,
            max_wait_s=1.0,
            decision_rounds=1,
            candidate_steps=2,
            dry_run=True,
        )
    )

    configs = {
        path.parent.name: path.read_text(encoding="utf-8")
        for path in (tmp_path / "out" / "candidate_pool").rglob("config.yaml")
    }
    assert "seed: 101" in configs["cand_0001"]
    assert "seed: 102" in configs["cand_0002"]
    assert _candidate_seed_offset("custom_candidate") == _candidate_seed_offset(
        "custom_candidate"
    )


def test_process_candidate_batch_raises_when_decision_round_has_no_success(
    tmp_path: Path,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _failed_worker(_config, _base_cfg):
        def _run(task: CandidateTask) -> CandidateResult:
            return CandidateResult(task=task, status="failed", error="candidate failed")

        return _run

    with pytest.raises(CandidateExecutionError, match="No successful candidate"):
        run_process_candidate_batch(
            ProcessCandidateRunnerConfig(
                config_yaml=config_yaml,
                out_root=tmp_path / "out",
                runner=Path("/tmp/fake-gh-dev"),
                num_candidates=2,
                min_completed=2,
                max_wait_s=1.0,
                decision_rounds=1,
                candidate_steps=2,
            ),
            adapters=ProcessCandidateBatchAdapters(worker_factory=_failed_worker),
        )


def test_process_candidate_batch_reports_drain_selection_separately(
    tmp_path: Path,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _worker(_config, _base_cfg):
        def _run(task: CandidateTask) -> CandidateResult:
            if task.candidate_id == "cand_0002":
                time.sleep(0.1)
                score = 0.9
            else:
                score = 0.5
            return CandidateResult(
                task=task,
                status="completed",
                score=score,
                payload=candidate_batch_runner.ProcessCandidatePayload(
                    config_yaml="",
                    out_dir="",
                    task_name="",
                    stdout_log="",
                    stderr_log="",
                    sequences={},
                ),
            )

        return _run

    summary = run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=1,
            max_wait_s=1.0,
            decision_rounds=1,
            candidate_steps=2,
        ),
        adapters=ProcessCandidateBatchAdapters(worker_factory=_worker),
    )

    assert summary["selected_candidate_ids"] == ["cand_0001"]
    assert summary["drain_selected_candidate_ids"] == ["cand_0002"]


def test_process_candidate_batch_late_previous_generation_drains_not_decides(
    tmp_path: Path,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    delays = {
        "cand_0001": 0.01,
        "cand_0002": 0.08,
        "cand_0003": 0.15,
        "cand_0004": 0.16,
    }
    scores = {
        "cand_0001": 0.5,
        "cand_0002": 0.99,
        "cand_0003": 0.7,
        "cand_0004": 0.6,
    }
    observed_generation2_sequences: list[dict[str, str]] = []

    def _worker(_config, _base_cfg):
        def _run(task: CandidateTask) -> CandidateResult:
            if task.generation == 2:
                observed_generation2_sequences.append(
                    dict((task.payload or {}).get("sequences") or {})
                )
            time.sleep(delays[task.candidate_id])
            return CandidateResult(
                task=task,
                status="completed",
                score=scores[task.candidate_id],
                payload=candidate_batch_runner.ProcessCandidatePayload(
                    config_yaml="",
                    out_dir="",
                    task_name="",
                    stdout_log="",
                    stderr_log="",
                    sequences={"A": task.candidate_id},
                ),
            )

        return _run

    summary = run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=1,
            max_wait_s=1.0,
            decision_rounds=2,
            candidate_steps=2,
        ),
        adapters=ProcessCandidateBatchAdapters(worker_factory=_worker),
    )

    assert summary["selected_candidate_ids"] == ["cand_0001", "cand_0003"]
    assert "cand_0002" in summary["drain_selected_candidate_ids"]
    assert observed_generation2_sequences == [{"A": "cand_0001"}] * 2


@pytest.mark.parametrize("decision_rounds", [0, -1])
def test_process_candidate_batch_non_positive_decision_rounds_only_drains(
    tmp_path: Path,
    decision_rounds: int,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=2,
            max_wait_s=1.0,
            decision_rounds=decision_rounds,
            candidate_steps=2,
            dry_run=True,
        )
    )

    assert summary["selected_candidate_ids"] == []
    assert summary["drain_selected_candidate_ids"] == ["cand_0001"]
    assert summary["collections"][0]["selected_candidate_id"] == "cand_0001"


def test_candidate_batch_main_dry_run_json_output_preserves_summary_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = candidate_batch_runner.main(
        [
            "--config-yaml",
            str(config_yaml),
            "--out-root",
            str(tmp_path / "out"),
            "--runner",
            "/tmp/fake-gh-dev",
            "--num-candidates",
            "2",
            "--min-completed",
            "2",
            "--decision-rounds",
            "1",
            "--candidate-steps",
            "2",
            "--dry-run",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert rc == 0
    assert captured.err == ""
    assert summary["selected_candidate_ids"] == ["cand_0001"]
    assert summary["drain_selected_candidate_ids"] == []
    assert summary["failed_candidates"] == []
    assert set(summary) == {
        "config_yaml",
        "out_root",
        "runner",
        "dry_run",
        "num_candidates",
        "min_completed",
        "max_wait_s",
        "max_in_flight",
        "candidate_timeout_s",
        "decision_rounds",
        "collections",
        "selected_candidate_ids",
        "drain_selected_candidate_ids",
        "failed_candidates",
    }


def test_candidate_batch_main_dry_run_text_output_includes_collection_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = candidate_batch_runner.main(
        [
            "--config-yaml",
            str(config_yaml),
            "--out-root",
            str(tmp_path / "out"),
            "--runner",
            "/tmp/fake-gh-dev",
            "--num-candidates",
            "2",
            "--min-completed",
            "2",
            "--decision-rounds",
            "1",
            "--candidate-steps",
            "2",
            "--dry-run",
            "--format",
            "text",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert "config_yaml:" in captured.out
    assert "pool: num_candidates=2 min_completed=2" in captured.out
    assert "collection: index=1 trigger=min_completed selected=cand_0001" in captured.out


def test_candidate_batch_cli_command_dry_run_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parser = gh_cli._build_parser()
    ns = parser.parse_args(
        [
            "experimental",
            "candidate-batch",
            "--config-yaml",
            str(config_yaml),
            "--out-root",
            str(tmp_path / "out"),
            "--runner",
            "/tmp/fake-gh-dev",
            "--num-candidates",
            "2",
            "--min-completed",
            "2",
            "--dry-run",
            "--format",
            "json",
        ]
    )

    rc = int(ns._handler(ns))

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert rc == 0
    assert captured.err == ""
    assert summary["selected_candidate_ids"] == ["cand_0001"]


def test_candidate_batch_cli_missing_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = gh_cli._build_parser()
    ns = parser.parse_args(
        [
            "experimental",
            "candidate-batch",
            "--config-yaml",
            str(tmp_path / "missing.yaml"),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        ns._handler(ns)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "error: candidate batch config not found:" in captured.err
    assert "Traceback" not in captured.err


def test_candidate_batch_cli_invalid_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_yaml = tmp_path / "bad.yaml"
    config_yaml.write_text("task_name: [\n", encoding="utf-8")
    parser = gh_cli._build_parser()
    ns = parser.parse_args(
        [
            "experimental",
            "candidate-batch",
            "--config-yaml",
            str(config_yaml),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        ns._handler(ns)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "error: failed to load candidate batch config" in captured.err
    assert str(config_yaml) in captured.err
    assert "Traceback" not in captured.err


def test_candidate_decision_run_uses_selected_sequences_for_next_generation() -> None:
    observed_generation2_sequences: list[dict[str, str]] = []

    def _worker(task: CandidateTask) -> CandidateResult:
        if task.candidate_id == "cand_0001":
            time.sleep(0.01)
            sequence = "selected_generation_1"
            score = 0.5
        elif task.candidate_id == "cand_0002":
            time.sleep(0.15)
            sequence = "late_generation_1"
            score = 0.99
        else:
            observed_generation2_sequences.append(
                dict((task.payload or {}).get("sequences") or {})
            )
            sequence = f"generation_2_{task.candidate_id}"
            score = 0.7 if task.candidate_id == "cand_0003" else 0.6
        return CandidateResult(
            task=task,
            status="completed",
            score=score,
            payload=candidate_batch_runner.ProcessCandidatePayload(
                config_yaml="",
                out_dir="",
                task_name="",
                stdout_log="",
                stderr_log="",
                sequences={"A": sequence},
            ),
        )

    run = CandidateDecisionRun(
        pool_config=CandidatePoolConfig(
            num_candidates=2,
            min_completed=1,
            max_wait_s=1.0,
        ),
        worker=_worker,
        decision_rounds=2,
    ).run()

    assert [c.selected.candidate_id for c in run.decision_collections if c.selected] == [
        "cand_0001",
        "cand_0003",
    ]
    assert [c.selected.candidate_id for c in run.drain_collections if c.selected] == [
        "cand_0002"
    ]
    assert observed_generation2_sequences
    assert observed_generation2_sequences == [{"A": "selected_generation_1"}] * 2


def test_process_candidate_batch_uses_selector_adapter(tmp_path: Path) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scores = {"cand_0001": 0.9, "cand_0002": 0.4}

    def _worker(_config, _base_cfg):
        def _run(task: CandidateTask) -> CandidateResult:
            return CandidateResult(
                task=task,
                status="completed",
                score=scores[task.candidate_id],
                payload=candidate_batch_runner.ProcessCandidatePayload(
                    config_yaml="",
                    out_dir="",
                    task_name="",
                    stdout_log="",
                    stderr_log="",
                    sequences={},
                ),
            )

        return _run

    def _select_lowest_score(results):
        completed = [result for result in results if result.status == "completed"]
        return min(completed, key=lambda result: result.score)

    summary = run_process_candidate_batch(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            num_candidates=2,
            min_completed=2,
            max_wait_s=1.0,
            decision_rounds=1,
            candidate_steps=2,
        ),
        adapters=ProcessCandidateBatchAdapters(
            worker_factory=_worker,
            selector=_select_lowest_score,
        ),
    )

    assert summary["selected_candidate_ids"] == ["cand_0002"]


def test_candidate_pool_worker_returned_failed_result_logs_error_and_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = CandidatePoolConfig(num_candidates=1, min_completed=1, max_wait_s=1.0)
    logger = candidate_batch_runner.logging.getLogger("test_candidate_pool_failed_result")

    def _worker(task: CandidateTask) -> CandidateResult:
        time.sleep(0.01)
        return CandidateResult(task=task, status="failed", error="subprocess failed")

    with caplog.at_level(candidate_batch_runner.logging.INFO, logger=logger.name):
        with CandidatePoolScheduler(cfg, _worker, logger=logger) as scheduler:
            scheduler.submit_generation(1, [CandidateTask(candidate_id="failed", generation=1)])
            collection = scheduler.collect_until_ready()

    assert collection is not None
    assert collection.failed[0].duration_s > 0
    error_events = [
        record.getMessage()
        for record in caplog.records
        if "event=span_error" in record.getMessage()
    ]
    assert error_events
    assert "status=failed" in error_events[-1]
    assert "subprocess failed" in error_events[-1]


def test_process_candidate_worker_does_not_use_collection_wait_as_subprocess_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_cfg = candidate_batch_runner._load_base_cfg(config_yaml)
    calls: list[float | None] = []

    def _fake_run(*_args, **kwargs):
        calls.append(kwargs.get("timeout"))
        out_root = tmp_path / "out"
        task_name = "base__g001__cand_0001"
        scores = out_root / task_name / "scores.csv"
        scores.parent.mkdir(parents=True)
        scores.write_text(
            "step,weighted_score,accepted,pA_sequence\n0,0.7,true,AAAA\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=kwargs, returncode=0)

    monkeypatch.setattr(candidate_batch_runner.subprocess, "run", _fake_run)
    worker = candidate_batch_runner.ProcessCandidateWorker(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            max_wait_s=0.01,
            candidate_steps=2,
            candidate_timeout_s=None,
        ),
        base_cfg,
    )

    result = worker(CandidateTask(candidate_id="cand_0001", generation=1))

    assert result.status == "completed"
    assert calls == [None]


def test_process_candidate_worker_uses_explicit_candidate_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_yaml = tmp_path / "base.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'runs'}",
                "seed: 10",
                "steps: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_cfg = candidate_batch_runner._load_base_cfg(config_yaml)
    calls: list[float | None] = []

    def _fake_run(*_args, **kwargs):
        calls.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args=kwargs, returncode=1)

    monkeypatch.setattr(candidate_batch_runner.subprocess, "run", _fake_run)
    worker = candidate_batch_runner.ProcessCandidateWorker(
        ProcessCandidateRunnerConfig(
            config_yaml=config_yaml,
            out_root=tmp_path / "out",
            runner=Path("/tmp/fake-gh-dev"),
            max_wait_s=0.01,
            candidate_steps=2,
            candidate_timeout_s=5.0,
        ),
        base_cfg,
    )

    result = worker(CandidateTask(candidate_id="cand_0001", generation=1))

    assert result.status == "failed"
    assert calls == [5.0]
    assert "stderr.log" in str(result.error)
    assert isinstance(result.payload, candidate_batch_runner.ProcessCandidatePayload)
