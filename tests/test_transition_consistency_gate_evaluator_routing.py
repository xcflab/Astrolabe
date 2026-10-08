from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


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
        complex_paths={name: f"{name.lower()}.pdb" for name in names},
    )


def test_transition_consistency_gate_uses_gate_evaluator_while_active() -> None:
    loss_computer = _init_loss()
    global_config.init_const("evaluators", {
        "SourceEval": SimpleNamespace(name="source"),
        "TargetEval": SimpleNamespace(name="target"),
        "GateEval": SimpleNamespace(name="gate"),
    })
    runtime_context = global_config.build_runtime_context()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
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
    stage_manager = StageManager(
        stage_args,
        loss_computer=loss_computer,
        runtime_context=runtime_context,
    )

    assert stage_manager.current_evaluator.name == "source"

    stage_manager.activate_transition_consistency_gate(
        target_stage="stage_1",
        previous_stage_result=_scores("AB"),
        previous_stage_oligo=["AB"],
        gate_seed_scores=_scores("AB"),
    )

    assert stage_manager.current_evaluator.name == "gate"

    stage_manager.complete_transition_consistency_gate()

    assert stage_manager.current_evaluator.name == "target"
