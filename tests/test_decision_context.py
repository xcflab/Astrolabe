from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.decision_context import (
    build_decision_score_context,
    passes_stage_entry_acceptance_gate,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.stage_transition import TransitionConsistencyGateResult


def _scores() -> StandardScores:
    return StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        complex_paths={"A": Path("a.pdb")},
    )


def test_decision_score_context_builds_previous_delta_and_gate_fields() -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    loss = SimpleNamespace(
        compute_loss=lambda _eval_result, _runtime_config: (
            2.0,
            {"plddt": 0.8, "pair_hotspot_support": 0.4},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: scores_detail["delta_plddt"] >= 0.1,
        passes_acceptance_gate=lambda scores_detail: (
            scores_detail["transition_consistency_gate_score"] >= 0.9
            and scores_detail["stage_entry"] == 0.0
        ),
    )

    context = build_decision_score_context(
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config={
            "transition_consistency_gate_result": TransitionConsistencyGateResult(
                gate_active=True,
                score=0.95,
                threshold=0.9,
                passed=True,
                comparisons={},
            ),
            "transition_consistency_gate_metadata": {
                "source_stage": "default",
                "target_stage": "ligand",
                "attempt": 2,
            },
        },
        previous_weighted=1.5,
        previous_scores={"plddt": 0.7, "pair_hotspot_support": 0.2},
        stage_entry=0.0,
        include_previous_delta=True,
    )

    assert context.weighted_score == 2.0
    assert context.hard_gate_passed is True
    assert context.soft_accepted is True
    assert context.scores_detail["previous_weighted_score"] == 1.5
    assert context.scores_detail["delta_weighted_score"] == 0.5
    assert context.scores_detail["previous_plddt"] == 0.7
    assert context.scores_detail["delta_plddt"] == pytest.approx(0.1)
    assert context.scores_detail["transition_consistency_gate_score"] == 0.95
    assert context.scores_detail["transition_consistency_gate_source_stage"] == "default"
    assert context.scores_detail["transition_consistency_gate_target_stage"] == "ligand"
    assert context.scores_detail["transition_consistency_gate_attempt"] == 2
    assert context.scores_detail["acceptance_gate_passed"] == 1.0


def test_active_transition_gate_requires_explicit_runtime_metadata() -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(
            transition_consistency_gate_context=SimpleNamespace(
                source_stage="global_source",
                target_stage="global_target",
            ),
            transition_consistency_gate_attempts=7,
        ),
    )
    loss = SimpleNamespace(
        compute_loss=lambda _eval_result, _runtime_config: (1.0, {"plddt": 0.8}),
    )

    with pytest.raises(ValueError, match="transition_consistency_gate_metadata"):
        build_decision_score_context(
            eval_result=_scores(),
            loss_computer=loss,
            runtime_config={
                "transition_consistency_gate_result": TransitionConsistencyGateResult(
                    gate_active=True,
                    score=0.95,
                    threshold=0.9,
                    passed=True,
                    comparisons={},
                ),
            },
        )


def test_inactive_transition_gate_does_not_require_metadata() -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    loss = SimpleNamespace(
        compute_loss=lambda _eval_result, _runtime_config: (1.0, {"plddt": 0.8}),
    )

    context = build_decision_score_context(
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config={
            "transition_consistency_gate_result": TransitionConsistencyGateResult(
                gate_active=False,
                score=0.0,
                threshold=0.9,
                passed=False,
                comparisons={},
            ),
        },
    )

    assert "transition_consistency_gate_score" not in context.scores_detail


def test_decision_score_context_builds_stage_entry_self_delta_context() -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    loss = SimpleNamespace(
        compute_loss=lambda _eval_result, _runtime_config: (
            3.0,
            {"cross_contact_lis": 0.3},
        ),
        passes_acceptance_gate=lambda scores_detail: (
            scores_detail["stage_entry"] == 1.0
            and scores_detail["previous_weighted_score"] == 3.0
            and scores_detail["delta_cross_contact_lis"] == 0.0
        ),
    )

    context = build_decision_score_context(
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config={},
        stage_entry=1.0,
        include_self_delta=True,
        evaluate_decision_flags=False,
    )

    assert context.hard_gate_passed is True
    assert context.scores_detail["stage_entry"] == 1.0
    assert context.scores_detail["previous_weighted_score"] == 3.0
    assert context.scores_detail["delta_weighted_score"] == 0.0
    assert context.scores_detail["previous_cross_contact_lis"] == 0.3
    assert context.scores_detail["delta_cross_contact_lis"] == 0.0


def test_stage_entry_acceptance_helper_uses_gate_and_self_delta_context() -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    seen = {}

    def _passes(scores_detail):
        seen.update(scores_detail)
        return (
            scores_detail["stage_entry"] == 1.0
            and scores_detail["transition_consistency_gate_score"] >= 0.8
            and scores_detail["previous_plddt"] == scores_detail["plddt"]
            and scores_detail["delta_plddt"] == 0.0
        )

    loss = SimpleNamespace(
        compute_loss=lambda _eval_result, _runtime_config: (
            1.0,
            {"plddt": 0.9},
        ),
        passes_acceptance_gate=_passes,
    )

    assert passes_stage_entry_acceptance_gate(
        eval_result=_scores(),
        loss_computer=loss,
        runtime_config={
            "transition_consistency_gate_result": TransitionConsistencyGateResult(
                gate_active=True,
                score=0.91,
                threshold=0.8,
                passed=True,
                comparisons={},
            ),
            "transition_consistency_gate_metadata": {
                "source_stage": "default",
                "target_stage": "stage_1",
                "attempt": 1,
            },
        },
    )

    assert seen["stage_entry"] == 1.0
    assert seen["transition_consistency_gate_score"] == 0.91
    assert seen["delta_plddt"] == 0.0
