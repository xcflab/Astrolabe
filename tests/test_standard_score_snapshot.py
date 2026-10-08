from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.score_snapshot import (
    build_standard_score_snapshot,
    merge_score_details,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine


def test_standard_score_snapshot_exposes_direct_and_chain_pair_outputs() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        ptm={"AB": [0.8]},
        iptm={"AB": [0.7]},
        pde={"AB": [np.array([[2.0, 4.0], [4.0, 2.0]], dtype=float)]},
        complex_paths={"AB": Path("ab.pdb")},
        chain_pair_scores={
            "AB": {
                "chain_ptm": [np.array([0.3, 0.9], dtype=float)],
                "chain_pair_iptm": [
                    np.array([[1.0, 0.4], [0.6, 0.8]], dtype=float)
                ],
                "chain_pair_lis": [
                    np.array([[0.0, 0.2], [0.6, 0.0]], dtype=float)
                ],
            },
        },
    )

    snapshot = build_standard_score_snapshot(scores)

    assert snapshot["plddt"] == 0.9
    assert snapshot["ptm"] == 0.8
    assert snapshot["iptm"] == 0.7
    assert snapshot["pde"] == 0.15
    assert snapshot["chain_ptm"] == 0.6
    assert snapshot["chain_pair_iptm"] == 0.7
    assert snapshot["cross_iptm"] == 0.5
    assert snapshot["intra_iptm"] == 0.9
    assert snapshot["chain_pair_lis"] == 0.2
    assert snapshot["cross_lis"] == 0.4
    assert snapshot["intra_lis"] == 0.0
    assert snapshot["lis"] == 0.4
    assert snapshot["AB"]["pde"] == 0.15
    assert snapshot["AB"]["chain_ptm"] == 0.6
    assert snapshot["AB"]["cross_iptm"] == 0.5


def test_merge_score_details_preserves_explicit_loss_values() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        ptm={"AB": [0.8]},
        complex_paths={"AB": Path("ab.pdb")},
    )

    merged = merge_score_details(
        scores,
        {
            "plddt": -1.0,
            "AB": {"plddt": -2.0},
            "custom_loss": 0.25,
        },
    )

    assert merged["plddt"] == -1.0
    assert merged["AB"]["plddt"] == -2.0
    assert merged["AB"]["ptm"] == 0.8
    assert merged["custom_loss"] == 0.25


def test_merge_score_details_preserves_attribute_access_for_expression_context() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {"pair_min_iptm": "min(AB.iptm, AC.iptm)"},
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="pair_min_iptm",
        early_stopping="pair_min_iptm > 0.5",
        acceptance_threshold=None,
        validate_config=False,
    )
    scores = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.9], "AC": [0.8]},
        iptm={"AB": [0.4], "AC": [0.6]},
        complex_paths={"AB": Path("ab.pdb"), "AC": Path("ac.pdb")},
    )

    _, detail = engine.compute_loss(scores, runtime_config={})
    merged = merge_score_details(scores, detail)

    assert merged["AB"].iptm == 0.4
    assert merged["AC"].iptm == 0.6
    assert engine.is_early_stopping(merged) is False


def test_decisions_create_uses_standard_score_snapshot_for_default_details(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), sequence="AA")},
    )
    global_config.init_const(
        "stage_manager",
        SimpleNamespace(current_stage="default"),
    )

    decisions = Decisions.create(
        StandardScores(
            names=["AB"],
            plddt={"AB": [0.9]},
            pde={"AB": [np.array([[2.0, 4.0], [4.0, 2.0]], dtype=float)]},
            complex_paths={"AB": tmp_path / "ab.pdb"},
            chain_pair_scores={
                "AB": {
                    "chain_ptm": [np.array([0.3, 0.9], dtype=float)],
                    "chain_pair_iptm": [
                        np.array([[1.0, 0.4], [0.6, 0.8]], dtype=float)
                    ],
                },
            },
        ),
        runtime_config={},
        score_func=lambda _eval_result, _runtime_config: (1.0, {}),
    )

    detail = decisions.scores_detail[0]
    row = decisions.row_history[0]
    assert detail["pde"] == 0.15
    assert detail["AB"]["chain_ptm"] == 0.6
    assert detail["AB"]["cross_iptm"] == 0.5
    assert row["pde"] == 0.15
    assert row["AB.chain_ptm"] == 0.6
    assert row["AB.cross_iptm"] == 0.5
    global_config.reset()
