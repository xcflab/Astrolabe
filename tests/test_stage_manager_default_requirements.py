from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from astrolabe.core import global_config
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


def _init_loss() -> LossEngine:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})
    return loss_computer


def test_stage_manager_requires_default_evaluator() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {"oligo": ["AB"], "_entrance": "plddt > 0.0"},
        }
    )
    with pytest.raises(ValueError, match="Default stage must define 'evaluator'"):
        StageManager(stage_args, loss_computer=loss_computer)


def test_stage_manager_requires_default_weighted_score_expression() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {"oligo": ["AB"], "_entrance": "plddt > 0.0"},
        }
    )
    with pytest.raises(
        ValueError, match="Default stage must define 'weighted_score_expression'"
    ):
        StageManager(stage_args, loss_computer=loss_computer)


def test_stage_manager_requires_default_mutators() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "evaluator": "Dummy",
                "weighted_score_expression": "plddt",
            },
            "stage_1": {"oligo": ["AB"], "_entrance": "plddt > 0.0"},
        }
    )
    with pytest.raises(ValueError, match="Default stage must define 'mutators'"):
        StageManager(stage_args, loss_computer=loss_computer)


def test_stage_manager_current_evaluator_errors_on_unknown_alias() -> None:
    loss_computer = _init_loss()
    global_config.init_const("evaluators", {})

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "evaluator": "Missing",
                "weighted_score_expression": "plddt",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {"oligo": ["AB"], "_entrance": "plddt > 0.0"},
        }
    )
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    with pytest.raises(ValueError, match="unknown/disabled evaluator alias"):
        _ = stage_manager.current_evaluator


def test_nondefault_stage_does_not_inherit_default_acceptance_threshold() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "evaluator": "Dummy",
                "weighted_score_expression": "plddt",
                "acceptance_threshold": "plddt > 0.5",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    stage_manager.set_stage("stage_1")

    assert stage_manager.current_acceptance_threshold is None


def test_nondefault_stage_does_not_inherit_default_acceptance_gate() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "evaluator": "Dummy",
                "weighted_score_expression": "plddt",
                "acceptance_gate": "plddt > 0.5",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    stage_manager.set_stage("stage_1")

    assert stage_manager.current_acceptance_gate is None


def test_stage_manager_accepts_prefix_early_stopping_stage_field() -> None:
    loss_computer = _init_loss()
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AC"],
                "mutation_num": 1,
                "evaluator": "Dummy",
                "weighted_score_expression": "plddt",
                "prefix_early_stopping": "prefix_steps == 8 and best.AC.binder_ptm <= 0.71",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {
                "oligo": ["AC"],
                "_entrance": "plddt > 0.0",
                "prefix_early_stopping": "prefix_steps == 12 and best.AC.binder_ptm <= 0.75",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    assert stage_manager.current_prefix_early_stopping == (
        "prefix_steps == 8 and best.AC.binder_ptm <= 0.71"
    )
    assert loss_computer.prefix_early_stopping == (
        "prefix_steps == 8 and best.AC.binder_ptm <= 0.71"
    )

    stage_manager.set_stage("stage_1")

    assert stage_manager.current_prefix_early_stopping == (
        "prefix_steps == 12 and best.AC.binder_ptm <= 0.75"
    )
    assert loss_computer.prefix_early_stopping == (
        "prefix_steps == 12 and best.AC.binder_ptm <= 0.75"
    )
