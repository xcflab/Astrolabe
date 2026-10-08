from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager, StageUnit


def _find_pair_with_set_order_mismatch(candidates: list[str]) -> tuple[str, str]:
    """Pick (first, second) where set iteration order starts with second."""
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            ordered = [first, second]
            if list(set(ordered)) != ordered:
                return first, second
    raise RuntimeError("Could not find a pair with set-order mismatch; expand candidates.")


def _find_triple_with_set_order_mismatch(candidates: list[str]) -> tuple[str, str, str]:
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates[i + 1 :], start=i + 1):
            for c in candidates[j + 1 :]:
                ordered = [a, b, c]
                if list(set(ordered)) != ordered:
                    return a, b, c
    raise RuntimeError("Could not find a triple with set-order mismatch; expand candidates.")


def test_stage_oligo_markers_distinguish_negative_design_from_ignore() -> None:
    unit = StageUnit(oligo=["AB", "-AC", "~AD"])

    assert unit.oligo == ["AB", "AC", "AD"]
    assert unit.negative_design_complexes == ["AC"]
    assert unit.ignored_complexes == ["AD"]


def test_stage_manager_publishes_negative_and_ignored_complexes() -> None:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})
    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB", "-AC", "~AD"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            }
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)

    assert stage_manager.current_oligo == ["AB", "AC", "AD"]
    assert global_config.negative_design_complexes == ["AC"]
    assert global_config.ignored_complexes == ["AD"]


def test_stage_manager_decide_entrance_respects_config_order() -> None:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})

    stage1, stage2 = _find_pair_with_set_order_mismatch(
        [
            "stage_A",
            "stage_B",
            "stage_C",
            "stage_D",
            "stage_E",
            "stage_F",
            "stage_G",
        ]
    )

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            stage1: {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
            },
            stage2: {
                "oligo": ["AB"],
                "_entrance": "plddt > 0.0",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)
    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": Path("ab.pdb")},
    )

    entered = stage_manager.decide_entrance(eval_result, loss_computer)
    assert entered == stage1


def test_stage_manager_all_oligos_preserves_first_seen_order() -> None:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})

    o1, o2, o3 = _find_triple_with_set_order_mismatch(
        [
            "AB",
            "CD",
            "EF",
            "GH",
            "IJ",
            "KL",
            "MN",
            "OP",
            "QR",
            "ST",
        ]
    )

    stage_args = OmegaConf.create(
        {
            "default": {
                "oligo": [o1],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Dummy",
                "mutators": ["MCMC"],
                "mutator_weights": [1.0],
            },
            "stage_1": {"oligo": [o2], "_entrance": "plddt > 0.0"},
            "stage_2": {"oligo": [o3], "_entrance": "plddt > 0.0"},
        }
    )
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    assert stage_manager.all_oligos == [o1, o2, o3]


def test_stage_manager_does_not_reenter_prior_stage() -> None:
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
            "monomer_chai": {
                "oligo": ["A"],
                "_entrance": "monomer_ready",
            },
            "dual_chai": {
                "oligo": ["AB", "AC"],
                "_entrance": "dual_ready",
            },
        }
    )

    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    loss_computer.score_context = lambda _eval_result: {  # type: ignore[attr-defined]
        "plddt": 1.0,
        "monomer_ready": True,
        "dual_ready": True,
    }

    eval_result = SimpleNamespace(names=["A", "AB", "AC"], complex_paths={"A": Path("a.pdb"), "AB": Path("ab.pdb"), "AC": Path("ac.pdb")})

    assert stage_manager.decide_entrance(eval_result, loss_computer) == "monomer_chai"
    assert stage_manager.current_stage == "monomer_chai"

    assert stage_manager.decide_entrance(eval_result, loss_computer) == "dual_chai"
    assert stage_manager.current_stage == "dual_chai"

    assert stage_manager.decide_entrance(eval_result, loss_computer) is None
    assert stage_manager.current_stage == "dual_chai"
