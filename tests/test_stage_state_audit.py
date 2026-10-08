from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import run as run_mod
from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


def _reset_runtime(tmp_path: Path | None = None) -> LossEngine:
    global_config.reset()
    if tmp_path is not None:
        global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(apply_mutations=lambda: None, clear_mutations=lambda: None),
    )
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    return loss_computer


def _scores_at(
    tmp_path: Path,
    *names: str,
    weighted: float = 0.0,
    stop_now: float = 0.0,
) -> StandardScores:
    complex_paths = {}
    for name in names:
        structure_path = tmp_path / f"{name.lower()}_{weighted}_{stop_now}.pdb"
        structure_path.write_text(f"{name}\n", encoding="utf-8")
        complex_paths[name] = structure_path
    scores = StandardScores(
        names=list(names),
        plddt={name: [1.0] for name in names},
        complex_paths=complex_paths,
    )
    scores.weighted = weighted
    scores.stop_now = stop_now
    return scores


def test_no_entrance_keeps_default_stage_and_runtime(tmp_path: Path) -> None:
    loss_computer = _reset_runtime(tmp_path)
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Eval0",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 10.0",
                "evaluator": "Eval1",
                "mutators": ["TargetMut"],
            },
        }
    )
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)

    entered = stage_manager.decide_entrance(
        _scores_at(tmp_path, "A"),
        loss_computer,
    )

    assert entered is None
    assert stage_manager.current_stage == "default"
    assert stage_manager.execution_setting is stage_manager.default_setting
    assert stage_manager.transition_consistency_gate_context is None
    assert stage_manager.current_transition_input_scores is None


def test_transition_gate_active_uses_target_runtime_without_changing_official_stage(
    tmp_path: Path,
) -> None:
    loss_computer = _reset_runtime(tmp_path)
    source_mutator = SimpleNamespace(name="source")
    target_mutator = SimpleNamespace(name="target")
    gate_evaluator = SimpleNamespace(name="gate")
    global_config.init_const(
        "mutators",
        {"SourceMut": source_mutator, "TargetMut": target_mutator},
    )
    global_config.init_const(
        "evaluators",
        {
            "SourceEval": SimpleNamespace(name="source"),
            "TargetEval": SimpleNamespace(name="target"),
            "GateEval": gate_evaluator,
        },
    )
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "SourceEval",
                "mutators": ["SourceMut"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "TargetEval",
                "mutators": ["TargetMut"],
                "transition_consistency_gate": {
                    "evaluator": "GateEval",
                    "consistency_threshold": 0.8,
                    "max_attempts": 2,
                },
            },
        }
    )
    stage_manager = StageManager(
        stage_args,
        loss_computer=loss_computer,
        runtime_context=global_config.build_runtime_context(),
    )
    source_scores = _scores_at(tmp_path, "A", "AB")
    gate_seed = _scores_at(tmp_path, "AB")

    stage_manager.activate_transition_consistency_gate(
        target_stage="stage_1",
        previous_stage_result=source_scores,
        previous_stage_oligo=["A"],
        gate_seed_scores=gate_seed,
    )

    assert stage_manager.current_stage == "default"
    assert stage_manager.execution_setting is stage_manager.stage_args["stage_1"]
    assert stage_manager.current_oligo == ["AB"]
    assert stage_manager.current_evaluator is gate_evaluator
    assert stage_manager.current_mutators == [target_mutator]
    assert stage_manager.current_transition_input_scores is gate_seed


def test_early_stop_records_current_stage_before_loop_break(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _reset_runtime(tmp_path)
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object()})

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A", weighted=0.0)
    early_stop_eval = _scores_at(tmp_path, "A", weighted=1.0, stop_now=1.0)

    class _EarlyStopLoss:
        def compute_loss(self, eval_result, _runtime_config):
            details = {
                "weighted": float(eval_result.weighted),
                "stop_now": float(eval_result.stop_now),
            }
            return float(details["weighted"]), details

        def is_early_stopping(self, scores_detail):
            return bool(scores_detail.get("stop_now"))

        def is_soft_accepted(self, _scores_detail):
            return False

        def passes_acceptance_gate(self, _scores_detail):
            return True

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "perform_evaluation",
        lambda *args, **kwargs: early_stop_eval,
    )

    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=_EarlyStopLoss(),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config={},
    )

    rows = list(csv.DictReader((tmp_path / "scores.csv").open(encoding="utf-8")))
    step_one_rows = [row for row in rows if row["step"] == "1"]

    assert result is early_stop_eval
    assert len(step_one_rows) == 1
    assert step_one_rows[0]["stage"] == "default"
    assert step_one_rows[0]["accepted"] == "True"
    assert float(step_one_rows[0]["stop_now"]) == pytest.approx(1.0)
