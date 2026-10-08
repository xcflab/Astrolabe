from __future__ import annotations

from pathlib import Path

import pytest

from astrolabe.evaluators.utils import StandardScores
from astrolabe.stage_transition import (
    _map_reference_occurrences,
    compute_reference_consistency,
)


def _scores(*names: str) -> StandardScores:
    return StandardScores(
        names=list(names),
        plddt={name: [1.0] for name in names},
        complex_paths={name: Path(f"{name.lower()}.pdb") for name in names},
    )


def test_map_reference_occurrences_uses_protomer_occurrence_order() -> None:
    assert _map_reference_occurrences("AA", "AAA") == [0, 1]
    assert _map_reference_occurrences("AA", "ABA") == [0, 2]
    assert _map_reference_occurrences("AB", "AC") is None


def test_compute_reference_consistency_uses_min_across_current_matches() -> None:
    def _compare(**kwargs):
        return 0.7 if kwargs["current_oligo"] == "AB" else 0.9

    result = compute_reference_consistency(
        current_scores=_scores("AB", "AC"),
        reference_scores=_scores("A"),
        threshold=0.8,
        compare_fn=_compare,
    )

    assert result.gate_active is True
    assert result.passed is False
    assert result.score == pytest.approx(0.7)
    assert result.comparisons["AB~A"] == pytest.approx(0.7)
    assert result.comparisons["AC~A"] == pytest.approx(0.9)
    assert result.comparisons["min~A"] == pytest.approx(0.7)


def test_compute_reference_consistency_fails_when_reference_has_no_match() -> None:
    result = compute_reference_consistency(
        current_scores=_scores("BC"),
        reference_scores=_scores("A"),
        threshold=0.8,
        compare_fn=lambda **kwargs: 1.0,
    )

    assert result.gate_active is True
    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert result.comparisons["missing~A"] == pytest.approx(0.0)
