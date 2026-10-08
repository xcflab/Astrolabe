from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


def test_decide_entrance_skips_future_stage_when_oligo_vars_missing() -> None:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "dual_protenix": {
                "oligo": ["AB", "AC"],
                "_entrance": "plddt > 2.0",
            },
            "dual_af3": {
                "oligo": ["AB", "AC"],
                "_entrance": "min(AB.iptm, AC.iptm) > 0.5",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)
    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": Path("a.pdb")},
    )

    entered = stage_manager.decide_entrance(eval_result, loss_computer)
    assert entered is None


def test_decide_entrance_skips_future_stage_when_expression_term_needs_missing_oligos() -> None:
    global_config.reset()
    loss_computer = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "pair_min_iptm": "min(AB.iptm, AC.iptm)",
            },
        },
        validate_config=False,
    )
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "dual_af3": {
                "oligo": ["AB", "AC"],
                "_entrance": "pair_min_iptm > 0.5",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)
    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": Path("a.pdb")},
    )

    entered = stage_manager.decide_entrance(eval_result, loss_computer)
    assert entered is None


def test_decide_entrance_can_use_loss_term_not_required_by_current_stage() -> None:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}}, validate_config=False)
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "iteration",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "monomer_gate": {
                "oligo": ["A"],
                "_entrance": "plddt > 0.5",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)
    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        complex_paths={"A": Path("a.pdb")},
    )

    entered = stage_manager.decide_entrance(eval_result, loss_computer)
    assert entered == "monomer_gate"
