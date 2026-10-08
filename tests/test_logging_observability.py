from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run as run_mod
from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.utils import StandardScores
from astrolabe.initializers.base import InitializerResult
from astrolabe.losses.base import LossEngine
from astrolabe.structure_processors.base import StructureProcessor
from astrolabe.stage_transition import TransitionConsistencyGateResult
from astrolabe.mutators.wrappers.dplm_quality_filter import (
    DplmQualityFilterConfig,
    filter_dplm_sequences,
)
from astrolabe.utils.base import log_mutator_sequence_changes
from astrolabe.utils.log_events import (
    build_event_message,
    build_step_start_message,
    event_span,
    summarize_event_items,
)
from astrolabe.utils.logging_utils import should_strip_console_ansi, strip_ansi


def _flush_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_setup_logging_strips_ansi_in_plain_file_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg = {
        "console_level": "INFO",
        "info_file_level": "INFO",
        "debug_file_level": "DEBUG",
        "color": {"console": "never", "file": "strip", "dump": "strip"},
    }

    run_mod.setup_logging(out_dir, logging_cfg)
    logging.info("plain \033[31mred\033[0m text")
    _flush_handlers()

    info_log = (out_dir / "logs" / "info.log").read_text(encoding="utf-8")
    assert "plain red text" in info_log
    assert "\033[" not in info_log


def test_setup_logging_preserves_ansi_in_file_ansi_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg = {
        "console_level": "INFO",
        "info_file_level": "INFO",
        "debug_file_level": "DEBUG",
        "color": {"console": "never", "file": "ansi", "dump": "strip"},
    }

    run_mod.setup_logging(out_dir, logging_cfg)
    logging.info("plain \033[32mgreen\033[0m text")
    _flush_handlers()

    info_log = (out_dir / "logs" / "info.log").read_text(encoding="utf-8")
    assert "\033[" in info_log


def test_console_auto_color_mode_is_tty_sensitive() -> None:
    class _TTY:
        def isatty(self):
            return True

    class _NotTTY:
        def isatty(self):
            return False

    assert should_strip_console_ansi("auto", _TTY()) is False
    assert should_strip_console_ansi("auto", _NotTTY()) is True
    assert should_strip_console_ansi("never", _TTY()) is True
    assert should_strip_console_ansi("always", _NotTTY()) is False


def test_event_message_renders_stable_single_line_fields() -> None:
    message = build_event_message(
        event="span_end",
        span_id="0123456789abcdef",
        component="predocker",
        alias="dock",
        stage="default",
        step=3,
        status="success",
        duration_ms=1.23456,
        artifact=Path("out with space.pdb"),
        result="structure_ready",
        detail_level="summary",
        detail="finished",
    )

    assert message == (
        'event=span_end span_id=0123456789abcdef component=predocker alias=dock '
        'stage=default step=3 '
        'status=success duration_ms=1.235 artifact="out with space.pdb" '
        "result=structure_ready detail_level=summary detail=finished"
    )


def test_event_span_correlates_start_and_end_with_one_span_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        with event_span(component="test_component", alias="unit"):
            pass

    messages = [
        record.getMessage()
        for record in caplog.records
        if "component=test_component" in record.getMessage()
    ]
    start = next(message for message in messages if "event=span_start" in message)
    end = next(message for message in messages if "event=span_end" in message)
    start_id = re.search(r"\bspan_id=([0-9a-f]{32})\b", start)
    end_id = re.search(r"\bspan_id=([0-9a-f]{32})\b", end)

    assert start_id is not None
    assert end_id is not None
    assert start_id.group(1) == end_id.group(1)


def test_event_span_logs_error_and_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="span failed"):
        with event_span(
            component="test_component",
            alias="unit",
            stage="test_stage",
            step=7,
        ):
            raise RuntimeError("span failed")

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "event=span_error" in message
        and "component=test_component" in message
        and "alias=unit" in message
        and "stage=test_stage" in message
        and "step=7" in message
        and "error=RuntimeError" in message
        and '"span failed"' in message
        and "duration_ms=" in message
        for message in messages
    )


def test_event_item_summary_is_bounded() -> None:
    assert summarize_event_items(["A", "B", "C", "D", "E", "F", "G"]) == (
        "A,B,C,D,E,+2"
    )


def test_apply_initializer_logs_initializer_unit_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    structure_path = tmp_path / "initialized.pdb"

    class _Initializer:
        def apply(self):
            return InitializerResult(
                mode="structure",
                sequences={name: "AAAA" for name in "ABCDEFG"},
                structure_path=structure_path,
            )

    monkeypatch.setattr(run_mod, "_resolve_initializer_mode", lambda _args: "structure")
    monkeypatch.setattr(
        run_mod,
        "build_initializer",
        lambda _args, *, mutators=None: _Initializer(),
    )

    with caplog.at_level(logging.INFO):
        result = run_mod.apply_initializer(SimpleNamespace())

    messages = [record.getMessage() for record in caplog.records]
    assert result.mode == "structure"
    assert any(
        "event=span_start" in message
        and "component=initializer" in message
        and "alias=structure" in message
        and "stage=initialization" in message
        and "step=0" in message
        for message in messages
    )
    assert any(
        "event=span_end" in message
        and "component=initializer" in message
        and "result=structure" in message
        and f"artifact={structure_path}" in message
        and "sequence_count=7" in message
        and "sequence_names=A,B,C,D,E,+2" in message
        and "duration_ms=" in message
        for message in messages
    )


class _LoggingPredocker(StructureProcessor):
    def _run_impl(self, input_file):
        output = Path(self.debug_dir) / "logged_output.pdb"
        output.write_text("MODEL\nEND\n", encoding="utf-8")
        return str(output)


class _SkippingPredocker(_LoggingPredocker):
    def eval_condition(self) -> bool:
        return False


def _init_logging_predocker_globals(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="dock_stage"))
    global_config.set_variable("step", 5)


def test_predocker_logs_success_span_with_artifact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _init_logging_predocker_globals(tmp_path)
    input_path = tmp_path / "input.pdb"
    input_path.write_text("MODEL\nEND\n", encoding="utf-8")
    predocker = _LoggingPredocker(cache_enabled=False)
    predocker.alias = "dock_alias"

    with caplog.at_level(logging.DEBUG):
        output = predocker.run(str(input_path))

    messages = [record.getMessage() for record in caplog.records]
    assert Path(output).exists()
    assert any(
        "event=span_start" in msg
        and "component=predocker" in msg
        and "alias=dock_alias" in msg
        and "stage=dock_stage" in msg
        and "step=5" in msg
        for msg in messages
    )
    assert any(
        "event=span_end" in msg
        and "status=success" in msg
        and "artifact=" in msg
        and "logged_output.pdb" in msg
        for msg in messages
    )
    assert any("event=predocker_preprocess" in msg for msg in messages)


def test_predocker_logs_condition_skip_without_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _init_logging_predocker_globals(tmp_path)
    input_path = tmp_path / "input.pdb"
    input_path.write_text("MODEL\nEND\n", encoding="utf-8")
    predocker = _SkippingPredocker(cache_enabled=False, _condition="plddt > 0.0")
    predocker.alias = "conditional"

    with caplog.at_level(logging.DEBUG):
        output = predocker.run(str(input_path))

    messages = [record.getMessage() for record in caplog.records]
    assert output == str(input_path)
    assert any(
        "event=predocker_condition" in msg
        and "alias=conditional" in msg
        and "status=skip" in msg
        for msg in messages
    )
    assert any(
        "event=span_skip" in msg
        and "status=skip" in msg
        and "detail=condition_false" in msg
        and str(input_path) in msg
        for msg in messages
    )
    assert not any("logged_output.pdb" in msg for msg in messages)


def test_perform_evaluation_logs_evaluator_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="eval_stage"))
    global_config.init_const("out_dir", tmp_path)
    global_config.set_variable("step", 3)

    class _Evaluator:
        alias = "eval_alias"

        def evaluate(self, inputs, runtime_config):
            name = next(iter(inputs))
            path = tmp_path / f"{name}.pdb"
            path.write_text("MODEL\nEND\n", encoding="utf-8")
            return StandardScores(
                names=[name],
                plddt={name: [0.9]},
                ptm={name: [None, 0.8]},
                iptm={name: [0.7]},
                complex_paths={name: path},
            )

    monkeypatch.setattr(
        run_mod,
        "construct_standard_input",
        lambda oligos, seed: {oligo: object() for oligo in oligos},
    )

    args = SimpleNamespace(
        seed=1,
        post_eval_hooks=None,
        enable_pyrosetta=False,
        skip_complex_poses=True,
        per_residue_scores=None,
    )

    with caplog.at_level(logging.INFO):
        result = run_mod.perform_evaluation(args, _Evaluator(), {}, 3, ["AB"])

    messages = [record.getMessage() for record in caplog.records]
    assert result.names == ["AB"]
    assert any(
        "event=span_start" in msg
        and "component=evaluator" in msg
        and "alias=eval_alias" in msg
        and "stage=eval_stage" in msg
        and "step=3" in msg
        and "oligo=AB" in msg
        for msg in messages
    )
    assert any(
        "event=span_end" in msg
        and "component=evaluator" in msg
        and "status=success" in msg
        and "result=standard_scores" in msg
        and "oligomer_count=1" in msg
        and "plddt_mean=0.900" in msg
        and "ptm_mean=0.800" in msg
        and "iptm_mean=0.700" in msg
        and "AB.pdb" in msg
        for msg in messages
    )


def test_log_mutator_sequence_changes_shows_diff_lines_in_info(caplog: pytest.LogCaptureFixture) -> None:
    before = {"A": "AAAA"}
    after = {"A": "AAAT"}
    telemetry = {
        "metrics": {"entropy_norm": 0.42},
    }

    with caplog.at_level(logging.DEBUG):
        log_mutator_sequence_changes("DPLM", before, after, telemetry=telemetry)

    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]

    assert any("changed_positions=1" in msg for msg in info_messages)
    assert any("entropy_norm=0.420" in msg for msg in info_messages)
    assert any("selection_score=N/A" in msg for msg in info_messages)
    assert any("Before(score↑):" in msg for msg in info_messages)
    assert any("Mutate(ss):" in msg for msg in info_messages)
    assert any("After(entropy↓):" in msg for msg in info_messages)
    assert all("Before(score↑):" not in msg for msg in debug_messages)


def test_execute_mutation_step_logs_mutator_span(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="mut_stage"))
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                mutated_sequence="AAAA",
                components=[],
                components_slices=[],
            )
        },
    )
    global_config.init_const(
        "mutable_protomers",
        {"A": SimpleNamespace(mutated_sequence="AAAA")},
    )
    global_config.init_const("segments", {})

    class _Mutator:
        name = "DummyMutator"
        alias = "mut_alias"
        last_mutation_telemetry = {"metrics": {}}

        def mutate(self, _mutable_protomers, _evaluation_result, _runtime_config):
            global_config.protomers["A"].mutated_sequence = "AAAT"

    original_log_mutator_sequence_changes = run_mod.log_mutator_sequence_changes

    def _log_mutator_sequence_changes(*args, **kwargs):
        assert any(
            "event=span_end" in record.getMessage()
            and "component=mutator" in record.getMessage()
            for record in caplog.records
        )
        return original_log_mutator_sequence_changes(*args, **kwargs)

    monkeypatch.setattr(
        run_mod,
        "log_mutator_sequence_changes",
        _log_mutator_sequence_changes,
    )

    with caplog.at_level(logging.INFO):
        run_mod.execute_mutation_step(
            [_Mutator()],
            SimpleNamespace(),
            {},
            step=5,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "event=span_start" in msg
        and "component=mutator" in msg
        and "alias=mut_alias" in msg
        and "stage=mut_stage" in msg
        and "step=5" in msg
        for msg in messages
    )
    assert any(
        "event=span_end" in msg
        and "component=mutator" in msg
        and "alias=mut_alias" in msg
        and "status=success" in msg
        and "result=changed" in msg
        and "changed_positions=1" in msg
        and "changed_protomers=1" in msg
        and "duration_ms=" in msg
        for msg in messages
    )


def test_loss_term_logs_timing_context_and_result_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = LossEngine.__new__(LossEngine)
    engine.runtime_context = SimpleNamespace(
        step=7,
        stage_manager=SimpleNamespace(current_stage="loss_stage"),
    )
    eval_result = SimpleNamespace(names=["AB"])

    with caplog.at_level(logging.INFO):
        per_oligo, mean_scores = engine._extract_scores_from_specs(
            eval_result,
            {
                "confidence": (
                    "plddt",
                    lambda _eval_result: {"AB": 0.25},
                    {},
                )
            },
        )

    messages = [record.getMessage() for record in caplog.records]
    assert per_oligo["AB"]["confidence"] == pytest.approx(0.25)
    assert mean_scores["confidence"] == pytest.approx(0.25)
    assert any(
        "event=span_start" in message
        and "component=loss" in message
        and "alias=confidence" in message
        and "stage=loss_stage" in message
        and "step=7" in message
        and "oligo=AB" in message
        and "detail=plddt" in message
        for message in messages
    )
    assert any(
        "event=span_end" in message
        and "component=loss" in message
        and "alias=confidence" in message
        and "result=per_oligomer" in message
        and "result_count=1" in message
        and "result_mean=0.250" in message
        and "duration_ms=" in message
        for message in messages
    )


def test_loss_engine_logs_aggregate_result_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = LossEngine.__new__(LossEngine)
    engine.runtime_context = SimpleNamespace(
        step=9,
        stage_manager=SimpleNamespace(current_stage="loss_stage"),
    )
    monkeypatch.setattr(
        engine,
        "_compute_loss",
        lambda _eval_result, _runtime_config: (
            1.25,
            {"confidence": 0.25, "iteration": 9.0},
        ),
        raising=False,
    )

    with caplog.at_level(logging.INFO):
        weighted_score, score_context = engine.compute_loss(object(), {})

    messages = [record.getMessage() for record in caplog.records]
    assert weighted_score == pytest.approx(1.25)
    assert score_context["confidence"] == pytest.approx(0.25)
    assert any(
        "event=span_end" in message
        and "component=loss_engine" in message
        and "alias=LossEngine" in message
        and "stage=loss_stage" in message
        and "step=9" in message
        and "result=aggregate" in message
        and "score_terms=2" in message
        and "weighted_score=1.250" in message
        and "duration_ms=" in message
        for message in messages
    )


def test_decision_logs_score_component_deltas(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(apply_mutations=lambda: None, clear_mutations=lambda: None),
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), mutated_sequence="AAAT")},
    )
    global_config.init_const("out_dir", tmp_path)

    decisions = Decisions(
        sequences={0: {"pA": "AAAA"}},
        scores={0: 1.0},
        evaluation_results={0: object()},
        scores_detail={0: {"plddt": 0.5, "iptm": 0.2, "iteration": 0.0, "B_length": 16.0}},
        accepted_steps=[0],
        last_accepted=0,
        stage_history={0: "default"},
    )

    class _Loss:
        def compute_loss(self, _eval_result, _runtime_config):
            return 1.4, {"plddt": 0.6, "iptm": 0.25, "iteration": 1.0, "B_length": 16.0}

        def is_early_stopping(self, _scores_detail):
            return False

        def is_soft_accepted(self, _scores_detail):
            return False

    with caplog.at_level(logging.INFO):
        decisions.make_mutation_decision(
            step=1,
            eval_result=object(),
            loss_computer=_Loss(),
            runtime_config={},
        )

    messages = [strip_ansi(record.getMessage()) for record in caplog.records]
    assert any("step=1 score" in msg and "delta=" in msg for msg in messages)
    assert any("component=plddt" in msg for msg in messages)
    assert any("delta=" in msg and "component=iptm" in msg for msg in messages)
    assert all("component=iteration" not in msg for msg in messages)
    assert all("component=B_length" not in msg for msg in messages)
    assert all("score_component" not in msg for msg in messages)
    assert any("delta=+0.050" in msg and "component=iptm" in msg for msg in messages)
    assert any(
        "event=span_end" in msg
        and "component=decision_maker" in msg
        and "alias=Decisions" in msg
        and "stage=default" in msg
        and "step=1" in msg
        and "result=accept" in msg
        and "accepted=true" in msg
        and "early_stop=false" in msg
        and "weighted_score=1.400" in msg
        and "duration_ms=" in msg
        for msg in messages
    )


def test_decision_logs_minimum_component_rows_when_changes_are_few(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(apply_mutations=lambda: None, clear_mutations=lambda: None),
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), mutated_sequence="AAAT")},
    )
    global_config.init_const("out_dir", tmp_path)

    decisions = Decisions(
        sequences={0: {"pA": "AAAA"}},
        scores={0: 1.0},
        evaluation_results={0: object()},
        scores_detail={
            0: {
                "plddt": 0.500,
                "iptm": 0.200,
                "ptm": 0.700,
                "binder_rog": 8.000,
                "iteration": 0.0,
                "B_length": 16.0,
            }
        },
        accepted_steps=[0],
        last_accepted=0,
        stage_history={0: "default"},
    )

    class _Loss:
        def compute_loss(self, _eval_result, _runtime_config):
            return 1.2, {
                "plddt": 0.510,
                "iptm": 0.200,
                "ptm": 0.700,
                "binder_rog": 8.000,
                "iteration": 1.0,
                "B_length": 16.0,
            }

        def is_early_stopping(self, _scores_detail):
            return False

        def is_soft_accepted(self, _scores_detail):
            return False

    with caplog.at_level(logging.INFO):
        decisions.make_mutation_decision(
            step=1,
            eval_result=object(),
            loss_computer=_Loss(),
            runtime_config={},
        )

    messages = [strip_ansi(record.getMessage()) for record in caplog.records]
    component_lines = [msg for msg in messages if msg.startswith("component=")]
    assert len(component_lines) >= 4
    assert any("component=plddt" in msg and "delta=+0.010" in msg for msg in component_lines)
    assert any("component=iptm" in msg and "delta=+0.000" in msg for msg in component_lines)
    assert all("component=iteration" not in msg for msg in messages)
    assert all("component=B_length" not in msg for msg in messages)


def test_recompute_last_score_preserves_stage_change_rows_in_scores_csv(
    tmp_path: Path,
) -> None:
    global_config.reset()
    stage_manager = SimpleNamespace(current_stage="target_protenix")
    global_config.init_const("stage_manager", stage_manager)
    global_config.init_const("out_dir", tmp_path)

    decisions = Decisions(
        sequences={
            0: {"pA": "AAAA"},
            1: {"pA": "AAAT"},
        },
        scores={0: 1.0, 1: 2.5},
        evaluation_results={0: object(), 1: object()},
        scores_detail={
            0: {"iptm": 0.2},
            1: {
                "iptm": 0.71,
                "hotspot_support": 0.33,
                "iteration": 1.0,
            },
        },
        accepted_steps=[0, 1],
        last_accepted=1,
        stage_history={0: "default", 1: "target_protenix"},
    )
    decisions.dump_results(tmp_path, incremental=False)

    stage_manager.current_stage = "af3_final"

    class _Loss:
        def compute_loss(self, _eval_result, _runtime_config):
            return 0.8, {
                "iptm": 0.075,
                "hotspot_support": 0.0,
                "iteration": 1.0,
            }

    decisions.recompute_last_score(_Loss(), {}, eval_result=object())

    with (tmp_path / "scores.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    step_rows = [r for r in rows if r["step"] == "1"]
    assert len(step_rows) == 2

    enter_stage_row, reevaluate_row = step_rows

    assert enter_stage_row["stage"] == "target_protenix"
    assert enter_stage_row["note"] == "enter stage"
    assert float(enter_stage_row["weighted_score"]) == pytest.approx(2.5)
    assert float(enter_stage_row["iptm"]) == pytest.approx(0.71)
    assert float(enter_stage_row["hotspot_support"]) == pytest.approx(0.33)

    assert reevaluate_row["stage"] == "af3_final"
    assert reevaluate_row["note"] == "re-evaluate due to stage change"
    assert float(reevaluate_row["weighted_score"]) == pytest.approx(0.8)
    assert float(reevaluate_row["iptm"]) == pytest.approx(0.075)
    assert float(reevaluate_row["hotspot_support"]) == pytest.approx(0.0)


def test_dump_results_flattens_nested_score_details_into_dotted_columns(
    tmp_path: Path,
) -> None:
    decisions = Decisions(
        sequences={0: {"pA": "AAAA"}},
        scores={0: 1.0},
        evaluation_results={0: object()},
        scores_detail={
            0: {
                "iptm": 0.2,
                "AB": {
                    "iptm": 0.81,
                    "lis": 0.44,
                },
            }
        },
        accepted_steps=[0],
        last_accepted=0,
        stage_history={0: "default"},
    )

    decisions.dump_results(tmp_path, incremental=False)

    with (tmp_path / "scores.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["AB.iptm"] == "0.81"
    assert rows[0]["AB.lis"] == "0.44"


def test_decisions_create_includes_default_confidence_snapshot(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), sequence="AA")},
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )

    structure_path = tmp_path / "ab.pdb"
    structure_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CA  GLY B   1       5.000   0.000   0.000  1.00 85.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = Decisions.create(
        StandardScores(
            names=["AB"],
            plddt={"AB": [0.81]},
            ptm={"AB": [0.72]},
            iptm={"AB": [0.63]},
            pae={"AB": [np.array([[0.0, 5.0], [5.0, 0.0]], dtype=float)]},
            chain_pair_scores={
                "AB": {
                    "chain_pair_lis": [
                        np.array([[0.0, 0.44], [0.44, 0.0]], dtype=float)
                    ]
                }
            },
            complex_paths={"AB": structure_path},
        ),
        runtime_config={},
        score_func=lambda _eval_result, _runtime_config: (1.0, {}),
    )

    detail = decisions.scores_detail[0]
    assert detail["plddt"] == pytest.approx(0.81)
    assert detail["ptm"] == pytest.approx(0.72)
    assert detail["iptm"] == pytest.approx(0.63)
    assert detail["AB"]["ptm"] == pytest.approx(0.72)
    assert detail["AB"]["iptm"] == pytest.approx(0.63)
    assert detail["ipae"] == pytest.approx(5.0)
    assert detail["AB"]["ipae"] == pytest.approx(5.0)
    assert detail["ipae_min"] == pytest.approx(5.0)
    assert detail["AB"]["ipae_min"] == pytest.approx(5.0)
    assert detail["ipae_minimal"] == pytest.approx(5.0)
    assert detail["AB"]["ipae_minimal"] == pytest.approx(5.0)
    assert detail["ipae_max"] == pytest.approx(5.0)
    assert detail["AB"]["ipae_max"] == pytest.approx(5.0)
    assert detail["lis"] == pytest.approx(0.44)
    assert detail["AB"]["lis"] == pytest.approx(0.44)
    assert detail["AB"]["ipsae"] == pytest.approx(1.0 / 26.0)


def test_decisions_create_ipae_uses_interface_pairs_not_all_cross_chain_pairs(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), sequence="AAAA")},
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )

    structure_path = tmp_path / "ab_interface.pdb"
    structure_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CA  GLY A   2     100.000   0.000   0.000  1.00 85.00           C",
                "ATOM      3  CA  SER B   1       5.000   0.000   0.000  1.00 88.00           C",
                "ATOM      4  CA  THR B   2     200.000   0.000   0.000  1.00 87.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pae = np.array(
        [
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 20.0, 30.0],
            [1.0, 20.0, 0.0, 0.0],
            [10.0, 30.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    decisions = Decisions.create(
        StandardScores(
            names=["AB"],
            plddt={"AB": [0.81]},
            ptm={"AB": [0.72]},
            iptm={"AB": [0.63]},
            pae={"AB": [pae]},
            complex_paths={"AB": structure_path},
        ),
        runtime_config={},
        score_func=lambda _eval_result, _runtime_config: (1.0, {}),
    )

    detail = decisions.scores_detail[0]
    assert detail["ipae"] == pytest.approx(1.0)
    assert detail["AB"]["ipae"] == pytest.approx(1.0)


def test_decisions_keep_ipae_distinct_from_pae_min_with_af3_confidence_fallback(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), sequence="A")},
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.set_variable("step", 0)

    structure_path = tmp_path / "ab_fallback.pdb"
    structure_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CA  SER B   1       5.000   0.000   0.000  1.00 88.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    step_dir = tmp_path / "raw_model_results" / "alphafold3" / "step_000_AB"
    step_dir.mkdir(parents=True)
    (step_dir / "step_000_AB_confidences.json").write_text(
        json.dumps(
            {
                "token_chain_ids": ["A", "B"],
                "token_res_ids": [1, 1],
                "pae": [[0.0, 5.0], [3.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )

    decisions = Decisions.create(
        StandardScores(
            names=["AB"],
            plddt={"AB": [0.81]},
            ptm={"AB": [0.72]},
            iptm={"AB": [0.63]},
            pae={"AB": [np.array([8.0, 9.0], dtype=float)]},
            chain_pair_scores={
                "AB": {
                    "chain_pair_pae_min": [
                        np.array([[0.0, 7.0], [9.0, 0.0]], dtype=float)
                    ]
                }
            },
            complex_paths={"AB": structure_path},
        ),
        runtime_config={},
        score_func=lambda _eval_result, _runtime_config: (1.0, {}),
    )

    detail = decisions.scores_detail[0]
    assert detail["ipae"] == pytest.approx(4.0)
    assert detail["AB"]["ipae"] == pytest.approx(4.0)
    assert detail["ipae_min"] == pytest.approx(3.0)
    assert detail["AB"]["ipae_min"] == pytest.approx(3.0)
    assert detail["ipae_minimal"] == pytest.approx(3.0)
    assert detail["AB"]["ipae_minimal"] == pytest.approx(3.0)
    assert detail["ipae_max"] == pytest.approx(5.0)
    assert detail["AB"]["ipae_max"] == pytest.approx(5.0)
    assert detail["pae_min"] == pytest.approx(7.0)
    assert detail["AB"]["pae_min"] == pytest.approx(7.0)
    assert detail["ipsae"] == pytest.approx(0.1)
    assert detail["AB"]["ipsae"] == pytest.approx(0.1)


def test_filter_dplm_sequences_reports_entropy_statistics() -> None:
    cfg = DplmQualityFilterConfig(entropy_threshold=0.1)
    kept, stats = filter_dplm_sequences(["ACDEFG", "ACDEFG"], cfg=cfg, expected_length=6)

    assert kept == ["ACDEFG"]
    assert "entropy_norm_mean" in stats
    assert stats["entropy_norm_mean"] > 0


def test_step_start_message_contains_stage() -> None:
    message = build_step_start_message(step=3, total_steps=100, stage_name="ligand")
    assert "step=3/100" in message
    assert "stage=ligand" in message


def test_log_mutator_sequence_changes_shows_site_level_scores(
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()
    global_config.init_const(
        "segments",
        {
            "segA": SimpleNamespace(
                mutations=[(3, "A", "T", np.float32(1.234), np.float32(0.456))],
            )
        },
    )
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                components=["segA"],
                components_slices=[slice(0, 4)],
            )
        },
    )

    telemetry = {
        "metrics": {"entropy_norm": 0.42},
        "secondary_structure_by_proto": {"A": "HHHL"},
    }
    with caplog.at_level(logging.INFO):
        log_mutator_sequence_changes("DPLM", {"A": "AAAA"}, {"A": "AAAT"}, telemetry=telemetry)

    messages = [strip_ansi(record.getMessage()) for record in caplog.records]
    assert any("sites A: A4:A->T(score=0.456 entropy=1.234 ss=L)" in msg for msg in messages)


def test_log_mutator_sequence_changes_uses_gradient_and_ss_colors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()
    global_config.init_const(
        "segments",
        {
            "segA": SimpleNamespace(
                mutations=[
                    (0, "A", "V", 0.900, 0.100),
                    (1, "A", "T", 0.500, 0.500),
                    (2, "A", "Q", 0.100, 0.900),
                ],
            )
        },
    )
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                components=["segA"],
                components_slices=[slice(0, 4)],
            )
        },
    )

    telemetry = {
        "metrics": {"entropy_norm": 0.42},
        "secondary_structure_by_proto": {"A": "HELX"},
    }
    with caplog.at_level(logging.DEBUG):
        log_mutator_sequence_changes("DPLM", {"A": "AAAA"}, {"A": "VTQA"}, telemetry=telemetry)

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    before_line = next(msg for msg in info_messages if "] A    Before(score↑):" in msg)
    mutate_line = next(msg for msg in info_messages if "] A    Mutate(ss):" in msg)
    after_line = next(msg for msg in info_messages if "] A    After(entropy↓):" in msg)

    assert "\x1b[38;5;" in before_line
    assert "\x1b[38;5;" in after_line
    assert "\x1b[35m|\x1b[0m" in mutate_line
    assert "\x1b[36m|\x1b[0m" in mutate_line
    assert "\x1b[33m|\x1b[0m" in mutate_line


def test_log_mutator_sequence_changes_uses_snapshot_metrics_before_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()
    global_config.init_const(
        "segments",
        {
            "segA": SimpleNamespace(
                mutations=[],
            )
        },
    )
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                components=["segA"],
                components_slices=[slice(0, 4)],
            )
        },
    )

    telemetry = {
        "metrics": {"entropy_norm": 0.42},
        "secondary_structure_by_proto": {"A": "HHHL"},
        "site_metrics_by_proto": {
            "A": {
                3: {
                    "to": "T",
                    "entropy": np.float32(0.123),
                    "score": np.float32(0.987),
                },
            }
        },
    }
    with caplog.at_level(logging.INFO):
        log_mutator_sequence_changes("LigandMPNN", {"A": "AAAA"}, {"A": "AAAT"}, telemetry=telemetry)

    messages = [strip_ansi(record.getMessage()) for record in caplog.records]
    assert any("A4:A->T(score=0.987 entropy=0.123 ss=L)" in msg for msg in messages)


def test_recompute_last_score_preserves_transition_consistency_gate_fields(
    tmp_path: Path,
) -> None:
    global_config.reset()
    stage_manager = SimpleNamespace(
        current_stage="af3_final",
        transition_consistency_gate_attempts=0,
        transition_consistency_gate_context=SimpleNamespace(
            source_stage="target_protenix",
            target_stage="af3_final",
        ),
    )
    global_config.init_const("stage_manager", stage_manager)
    global_config.init_const("out_dir", tmp_path)

    decisions = Decisions(
        sequences={
            1: {"pA": "AAAT"},
        },
        scores={1: 2.5},
        evaluation_results={1: object()},
        scores_detail={
            1: {
                "iptm": 0.71,
                "hotspot_support": 0.33,
                "iteration": 1.0,
            },
        },
        accepted_steps=[1],
        last_accepted=1,
        stage_history={1: "target_protenix"},
    )

    class _Loss:
        def compute_loss(self, _eval_result, _runtime_config):
            return 3.25, {
                "iptm": 0.91,
                "hotspot_support": 0.41,
                "iteration": 1.0,
            }

    decisions.recompute_last_score(
        _Loss(),
        {
            "transition_consistency_gate_result": TransitionConsistencyGateResult(
                gate_active=True,
                score=0.87,
                threshold=0.8,
                passed=True,
                comparisons={"AB~AB": 0.87},
            ),
            "transition_consistency_gate_metadata": {
                "source_stage": "target_protenix",
                "target_stage": "af3_final",
                "attempt": 1,
            },
        },
        eval_result=object(),
    )

    rows = list(csv.DictReader((tmp_path / "scores.csv").open("r", encoding="utf-8")))
    reevaluate_row = rows[-1]
    assert reevaluate_row["note"] == "re-evaluate due to stage change"
    assert reevaluate_row["stage"] == "af3_final"
    assert reevaluate_row["transition_consistency_gate_score"] == "0.87"
    assert reevaluate_row["transition_consistency_gate_threshold"] == "0.8"
    assert reevaluate_row["transition_consistency_gate_passed"] == "1.0"
    assert reevaluate_row["transition_consistency_gate_source_stage"] == "target_protenix"
    assert reevaluate_row["transition_consistency_gate_target_stage"] == "af3_final"
    assert float(reevaluate_row["transition_consistency_gate_attempt"]) == pytest.approx(1.0)
