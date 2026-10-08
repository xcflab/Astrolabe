from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from astrolabe.evaluators.utils import StandardScores


def test_standard_scores_wraps_scalar_metrics_in_lists() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": 0.5},
        ptm={"AB": 0.1},
        iptm={"AB": 0.2},
        complex_paths={"AB": Path("ab.pdb")},
    )
    assert scores.plddt["AB"] == [0.5]
    assert scores.ptm["AB"] == [0.1]
    assert scores.iptm["AB"] == [0.2]


def test_standard_scores_wraps_per_proto_scores_in_lists() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.5]},
        complex_paths={"AB": Path("ab.pdb")},
        per_proto_scores={
            "A": np.array([0.9, 0.8], dtype=float),
            "B": [np.array([0.1, 0.2], dtype=float)],
        },
    )
    assert isinstance(scores.per_proto_scores["A"], list)
    assert np.allclose(scores.per_proto_scores["A"][0], np.array([0.9, 0.8]))
    assert isinstance(scores.per_proto_scores["B"], list)
    assert np.allclose(scores.per_proto_scores["B"][0], np.array([0.1, 0.2]))


def test_standard_scores_wraps_named_per_proto_score_terms_in_lists() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.5]},
        complex_paths={"AB": Path("ab.pdb")},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([90.0, 80.0], dtype=float),
                "sap": np.array([3.0, 1.5], dtype=float),
            }
        },
    )

    assert isinstance(scores.per_proto_score_terms["A"]["plddt"], list)
    assert np.allclose(
        scores.per_proto_score_terms["A"]["plddt"][0],
        np.array([0.9, 0.8], dtype=float),
    )
    assert isinstance(scores.per_proto_score_terms["A"]["sap"], list)
    assert np.allclose(
        scores.per_proto_score_terms["A"]["sap"][0],
        np.array([3.0, 1.5], dtype=float),
    )


def test_standard_scores_wraps_chain_pair_scores_in_lists() -> None:
    mat = np.eye(2, dtype=float)
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.5]},
        complex_paths={"AB": Path("ab.pdb")},
        chain_pair_scores={
            "AB": {
                "chain_pair_iptm": mat,
            }
        },
    )
    assert isinstance(scores.chain_pair_scores["AB"]["chain_pair_iptm"], list)
    assert np.allclose(scores.chain_pair_scores["AB"]["chain_pair_iptm"][0], mat)


def test_standard_scores_normalizes_plddt_scales_to_unit_interval() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [np.array([50.0, 90.0], dtype=float)]},
        complex_paths={"AB": Path("ab.pdb")},
        per_proto_scores={"A": [np.array([10.0, 20.0], dtype=float)]},
    )
    assert np.allclose(scores.plddt["AB"][0], np.array([0.5, 0.9]))
    assert np.allclose(scores.per_proto_scores["A"][0], np.array([0.1, 0.2]))


def test_standard_scores_scaling_does_not_double_scale_mixed_lists() -> None:
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.5, 50.0]},
        complex_paths={"AB": Path("ab.pdb")},
        per_proto_scores={
            "A": [np.array([0.1, 0.2], dtype=float), np.array([10.0, 20.0], dtype=float)]
        },
    )
    assert scores.plddt["AB"] == [0.5, 0.5]
    assert np.allclose(scores.per_proto_scores["A"][0], np.array([0.1, 0.2]))
    assert np.allclose(scores.per_proto_scores["A"][1], np.array([0.1, 0.2]))


def test_filter_by_oligomer_keeps_empty_complex_pose_container() -> None:
    @dataclass
    class _ComplexPoses:
        poses: dict[str, object]

    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": Path("a.pdb")},
        complex_poses=_ComplexPoses({"A": object()}),
    )

    filtered = scores.filter_by_oligomer(["AB"])

    assert filtered.names == []
    assert filtered.complex_poses is not None
    assert filtered.complex_poses.poses == {}


def test_filter_by_oligomer_keeps_duck_typed_empty_complex_pose_container() -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": Path("a.pdb")},
        complex_poses=SimpleNamespace(poses={"A": object()}),
    )

    filtered = scores.filter_by_oligomer(["AB"])

    assert filtered.names == []
    assert filtered.complex_poses is not None
    assert filtered.complex_poses.poses == {}
