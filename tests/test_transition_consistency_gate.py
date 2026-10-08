from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.utils import StandardScores
from astrolabe.stage_transition import TransitionConsistencyGateResult


def _scores(name: str = "A") -> StandardScores:
    return StandardScores(
        names=[name],
        plddt={name: [1.0]},
        complex_paths={name: Path(f"{name.lower()}.pdb")},
    )


def _init_runtime(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(_init_runtime, "_applied", True),
            clear_mutations=lambda: setattr(_init_runtime, "_cleared", True),
        ),
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(
            current_stage="stage_1",
            complete_transition_consistency_gate=lambda: setattr(
                _init_runtime, "_completed_gate", True
            )
            or "stage_2",
        ),
    )
    global_config.init_const("use_metropolis_acceptance", False)
    _init_runtime._applied = False
    _init_runtime._cleared = False
    _init_runtime._completed_gate = False


def _loss(score: float):
    return SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (score, {}),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
    )


def _gate_runtime(*, score: float, threshold: float = 0.9, passed: bool) -> dict:
    return {
        "transition_consistency_gate_result": TransitionConsistencyGateResult(
            gate_active=True,
            score=score,
            threshold=threshold,
            passed=passed,
            comparisons={},
        ),
        "transition_consistency_gate_metadata": {
            "source_stage": "stage_1",
            "target_stage": "stage_2",
            "attempt": 1,
        },
    }


def test_gate_accepts_even_when_weighted_score_is_worse(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = Decisions.create(
        _scores(),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (0.0, {}),
    )

    early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        step=1,
        eval_result=_scores(),
        loss_computer=_loss(10.0),
        runtime_config=_gate_runtime(score=0.95, passed=True),
    )

    assert early_stop is False
    assert accepted is True
    assert accepted_eval is decisions.evaluation_results[1]
    assert _init_runtime._applied is True
    assert _init_runtime._cleared is False
    assert _init_runtime._completed_gate is False


def test_acceptance_gate_can_read_transition_consistency_score(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = Decisions.create(
        _scores(),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (0.0, {}),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            10.0,
            {"plddt": 0.9},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: (
            scores_detail["transition_consistency_gate_score"] >= 0.8
            and scores_detail["plddt"] > 0.5
        ),
    )

    _early_stop, _accepted_eval, accepted = decisions.make_mutation_decision(
        step=1,
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config=_gate_runtime(score=0.95, passed=True),
    )

    assert accepted is True
    assert decisions.row_history[-1]["transition_consistency_gate_score"] == 0.95
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 1.0


def test_acceptance_gate_blocks_transition_consistency_accept(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = Decisions.create(
        _scores(),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (0.0, {}),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            10.0,
            {"pair_hotspot_support": 0.0},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: scores_detail[
            "pair_hotspot_support"
        ]
        >= 0.14,
    )

    early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        step=1,
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config=_gate_runtime(score=0.95, passed=True),
    )

    assert early_stop is False
    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert _init_runtime._applied is False
    assert _init_runtime._cleared is True
    assert _init_runtime._completed_gate is False
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 0.0


def test_gate_rejects_even_when_weighted_score_is_better(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = Decisions.create(
        _scores(),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (10.0, {}),
    )

    early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        step=1,
        eval_result=_scores(),
        loss_computer=_loss(0.0),
        runtime_config=_gate_runtime(score=0.4, passed=False),
    )

    assert early_stop is False
    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert _init_runtime._applied is False
    assert _init_runtime._cleared is True
    assert _init_runtime._completed_gate is False


def test_force_accept_cannot_bypass_gate(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = Decisions.create(
        _scores(),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (10.0, {}),
    )

    with pytest.raises(ValueError, match="active transition-consistency gate"):
        decisions.make_mutation_decision(
            step=1,
            eval_result=_scores(),
            loss_computer=_loss(0.0),
            runtime_config=_gate_runtime(score=0.4, passed=False),
            force_accept=True,
        )
