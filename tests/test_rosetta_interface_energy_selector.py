from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.selectors.rosetta_interface_energy import (
    RosettaInterfaceEnergyPositionSelector,
)
from astrolabe.selectors.utils import construct_position_selector


class _Protomer:
    def __init__(self, length: int, mutable_mask=None) -> None:
        self.length = int(length)
        if mutable_mask is None:
            mutable_mask = np.ones(self.length, dtype=bool)
        self.mutable_mask = np.asarray(mutable_mask, dtype=bool)


def _scores(per_proto_score_terms):
    return SimpleNamespace(per_proto_score_terms=per_proto_score_terms)


def test_rosetta_interface_energy_selects_global_top_k_with_filters_and_stable_ties() -> None:
    evaluation_result = _scores(
        {
            "A": {
                "rosetta_interface_dg": [
                    np.array([5.0, np.nan, 5.0, 1.0], dtype=float)
                ]
            },
            "B": {
                "rosetta_interface_dg": [
                    np.array([5.0, 6.0, 4.0], dtype=float)
                ]
            },
        }
    )
    protomers = {
        "A": _Protomer(4),
        "B": _Protomer(3, [True, False, True]),
    }

    sites = construct_position_selector(
        "rosetta_interface_energy(top_k=4)",
        evaluation_result,
        protomers,
    )

    assert np.array_equal(sites["A"], np.array([True, False, True, False]))
    assert np.array_equal(sites["B"], np.array([True, False, True]))
    assert not hasattr(evaluation_result, "complex_paths")
    assert not hasattr(evaluation_result, "complex_poses")


def test_rosetta_interface_energy_returns_all_candidates_when_fewer_than_top_k() -> None:
    selector = RosettaInterfaceEnergyPositionSelector(
        _scores(
            {
                "A": {
                    "rosetta_interface_dg": [
                        np.array([np.nan, -1.0, 2.0], dtype=float)
                    ]
                }
            }
        ),
        {"A": _Protomer(3, [True, False, True])},
        top_k=5,
    )

    assert np.array_equal(selector.sites["A"], np.array([False, False, True]))


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rosetta_interface_energy_rejects_invalid_top_k(top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        RosettaInterfaceEnergyPositionSelector(
            _scores({"A": {"rosetta_interface_dg": [np.array([1.0])]}}),
            {"A": _Protomer(1)},
            top_k=top_k,
        )


def test_rosetta_interface_energy_rejects_missing_term() -> None:
    with pytest.raises(KeyError, match="rosetta_interface_dg"):
        RosettaInterfaceEnergyPositionSelector(
            _scores({"A": {"sap": [np.array([1.0])]}}),
            {"A": _Protomer(1)},
            top_k=1,
        )


def test_rosetta_interface_energy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        RosettaInterfaceEnergyPositionSelector(
            _scores({"A": {"rosetta_interface_dg": [np.array([1.0, 2.0])]}}),
            {"A": _Protomer(3)},
            top_k=1,
        )


def test_rosetta_interface_energy_rejects_empty_candidate_set() -> None:
    with pytest.raises(ValueError, match="no finite mutable candidates"):
        RosettaInterfaceEnergyPositionSelector(
            _scores({"A": {"rosetta_interface_dg": [np.array([np.nan])]}}),
            {"A": _Protomer(1)},
            top_k=1,
        )
