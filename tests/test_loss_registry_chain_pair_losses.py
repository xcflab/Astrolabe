from __future__ import annotations

from pathlib import Path

import numpy as np

import astrolabe.losses.confidence  # noqa: F401  (registers losses)
from astrolabe.losses.base import loss_registry
from astrolabe.evaluators.utils import StandardScores


def _scores_with_chain_pair_mats() -> StandardScores:
    return StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": Path("ab.pdb")},
        chain_pair_scores={
            "AB": {
                "chain_pair_contact_lis": [np.array([[1.0, 0.0], [0.0, 1.0]])],
                "chain_pair_iptm": [np.array([[2.0, 0.5], [0.5, 2.0]])],
                "chain_pair_lis": [np.array([[3.0, 1.0], [1.0, 3.0]])],
                "chain_pair_pae_min": [np.array([[4.0, 2.0], [2.0, 4.0]])],
            }
        },
    )


def test_intra_chain_losses_use_correct_metric_key() -> None:
    scores = _scores_with_chain_pair_mats()
    expected = {
        "intra_contact_lis": 1.0,
        "intra_iptm": 2.0,
        "intra_lis": 3.0,
        "intra_pae_min": 4.0,
    }
    for name, expected_value in expected.items():
        func = loss_registry.get_function(name)
        result = func(scores)
        assert result == {"AB": expected_value}


def test_cross_chain_losses_use_correct_metric_key() -> None:
    scores = _scores_with_chain_pair_mats()
    expected = {
        "cross_contact_lis": 0.0,
        "cross_iptm": 0.5,
        "cross_lis": 1.0,
        "cross_pae_min": 2.0,
    }
    for name, expected_value in expected.items():
        func = loss_registry.get_function(name)
        result = func(scores)
        assert result == {"AB": expected_value}


def test_chain_pair_losses_return_zero_when_scores_missing() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": Path("ab.pdb")},
        chain_pair_scores={"AB": None},
    )

    for name in [
        "intra_contact_lis",
        "intra_iptm",
        "intra_lis",
        "intra_pae_min",
        "cross_contact_lis",
        "cross_iptm",
        "cross_lis",
        "cross_pae_min",
    ]:
        func = loss_registry.get_function(name)
        assert func(scores) == {"AB": 0.0}
