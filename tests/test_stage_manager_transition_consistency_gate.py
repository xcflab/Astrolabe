from __future__ import annotations

import json

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager
from astrolabe.stage_transition import (
    TransitionConsistencyGateResult,
    snapshot_standard_scores,
)


def _init_loss() -> LossEngine:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})
    return loss_computer


def _scores(*names: str) -> StandardScores:
    return StandardScores(
        names=list(names),
        plddt={name: [1.0] for name in names},
        complex_paths={name: Path(f"{name.lower()}.pdb") for name in names},
    )


def test_default_stage_rejects_transition_consistency_gate() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                },
            },
            "stage_1": {"oligo": ["AB"], "_entrance": "plddt > 0.0"},
        }
    )

    with pytest.raises(
        ValueError, match="Default stage cannot define 'transition_consistency_gate'"
    ):
        StageManager(stage_args, loss_computer=loss_computer)


def test_decide_entrance_keeps_source_stage_for_gated_transition() -> None:
    loss_computer = _init_loss()
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
                "_entrance": "plddt > 0.0",
                "evaluator": "Eval1",
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    entered = stage_manager.decide_entrance(_scores("A"), loss_computer)

    assert entered == "stage_1"
    assert stage_manager.current_stage == "default"


def test_stage_manager_tracks_transition_consistency_gate_state() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A", "AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Eval0",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "Eval1",
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    accepted_previous = _scores("A", "AB")
    gate_seed = _scores("AB")

    stage_manager.activate_transition_consistency_gate(
        target_stage="stage_1",
        previous_stage_result=accepted_previous,
        previous_stage_oligo=["A"],
        gate_seed_scores=gate_seed,
    )

    assert stage_manager.current_stage == "default"
    assert stage_manager.current_transition_input_scores is gate_seed
    assert stage_manager.transition_consistency_gate_context is not None
    assert (
        stage_manager.transition_consistency_gate_context.reference_scores.names == ["A"]
    )


def test_stage_manager_defaults_transition_consistency_gate_weight_to_one() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A", "AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Eval0",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "Eval1",
                "transition_consistency_gate": {
                    "consistency_threshold": 0.8,
                    "max_attempts": 2,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)

    assert stage_manager.stage_args["stage_1"].transition_consistency_gate is not None
    assert (
        stage_manager.stage_args["stage_1"].transition_consistency_gate.weight
        == pytest.approx(1.0)
    )


def test_stage_manager_stores_transition_consistency_gate_evaluator() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "SourceEval",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "TargetEval",
                "transition_consistency_gate": {
                    "evaluator": "GateEval",
                    "consistency_threshold": 0.8,
                    "max_attempts": 2,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)

    gate = stage_manager.stage_args["stage_1"].transition_consistency_gate
    assert gate is not None
    assert gate.evaluator == "GateEval"
    assert (
        stage_manager.resolve_transition_consistency_gate_evaluator("stage_1")
        == "GateEval"
    )


def test_stage_manager_aborts_transition_consistency_gate_after_max_attempts() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "SourceEval",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "TargetEval",
                "transition_consistency_gate": {
                    "weight": 0.5,
                    "consistency_threshold": 0.8,
                    "max_attempts": 2,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    source_scores = _scores("A")
    gate_seed = _scores("AB")
    stage_manager.activate_transition_consistency_gate(
        target_stage="stage_1",
        previous_stage_result=source_scores,
        previous_stage_oligo=["A"],
        gate_seed_scores=gate_seed,
    )

    assert stage_manager.record_transition_consistency_gate_failure() is False
    assert stage_manager.transition_consistency_gate_attempts == 1
    assert stage_manager.record_transition_consistency_gate_failure() is True

    rollback_scores = stage_manager.abort_transition_consistency_gate()
    assert stage_manager.current_stage == "default"
    assert stage_manager.transition_consistency_gate_context is None
    assert rollback_scores is source_scores


def test_stage_manager_writes_transition_gate_artifacts_for_seed_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("out_dir", tmp_path)
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Eval0",
                "mutators": ["MCMC"],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
                "evaluator": "Eval1",
                "transition_consistency_gate": {
                    "consistency_threshold": 0.8,
                    "max_attempts": 3,
                },
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)

    source_live = tmp_path / "live_source.cif"
    seed_live = tmp_path / "live_seed.cif"
    attempt_live = tmp_path / "live_attempt.cif"
    source_live.write_text("SOURCE\n", encoding="utf-8")
    seed_live.write_text("SEED\n", encoding="utf-8")
    attempt_live.write_text("ATTEMPT\n", encoding="utf-8")

    source_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": source_live},
    )
    preserved_source = snapshot_standard_scores(
        source_scores,
        output_dir=tmp_path / "transition_gate" / "step_007_default_to_stage_1" / "source_stage",
    )
    seed_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": seed_live},
    )
    attempt_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": attempt_live},
    )
    artifact_root = tmp_path / "transition_gate" / "step_007_default_to_stage_1"

    monkeypatch.setattr(
        'astrolabe.stage_manager.compute_reference_consistency',
        lambda **kwargs: TransitionConsistencyGateResult(
            gate_active=True,
            score=0.5,
            threshold=0.8,
            passed=False,
            comparisons={"AB~AB": 0.5},
        ),
    )

    stage_manager.activate_transition_consistency_gate(
        target_stage="stage_1",
        previous_stage_result=preserved_source,
        previous_stage_oligo=["AB"],
        gate_seed_scores=seed_scores,
        artifact_root=artifact_root,
        activation_step=7,
    )
    global_config.set_variable("step", 7)
    stage_manager.evaluate_transition_consistency(seed_scores)
    global_config.set_variable("step", 8)
    stage_manager.evaluate_transition_consistency(attempt_scores)

    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    labels = [item["label"] for item in manifest["evaluations"]]

    assert (artifact_root / "source_stage" / "AB_model.cif").exists()
    assert (artifact_root / "seed" / "AB_model.cif").exists()
    assert (artifact_root / "attempt_01" / "AB_model.cif").exists()
    assert labels == ["seed", "attempt_01"]
