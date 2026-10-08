from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.losses import cyclic as cyclic_losses


def _coords(rows):
    return np.asarray(rows, dtype=object)


def test_separation_std_is_zero_for_planar_regular_square(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cyclic_losses,
        "get_coord_CA",
        lambda _path: _coords(
            [
                ["A", 1, 1.0, 0.0, 0.0],
                ["B", 1, 0.0, 1.0, 0.0],
                ["C", 1, -1.0, 0.0, 0.0],
                ["D", 1, 0.0, -1.0, 0.0],
            ]
        ),
    )

    value = cyclic_losses.calculate_separation_std_single(
        "ABCD",
        SimpleNamespace(complex_paths={"ABCD": Path("good.cif")}),
    )

    assert value == pytest.approx(0.0)


def test_separation_std_preserves_legacy_radial_std_for_nonplanar_same_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cyclic_losses,
        "get_coord_CA",
        lambda _path: _coords(
            [
                ["A", 1, 1.0, 0.0, 1.0],
                ["B", 1, 0.0, 1.0, -1.0],
                ["C", 1, -1.0, 0.0, 1.0],
                ["D", 1, 0.0, -1.0, -1.0],
            ]
        ),
    )

    value = cyclic_losses.calculate_separation_std_single(
        "ABCD",
        SimpleNamespace(complex_paths={"ABCD": Path("bad.cif")}),
    )

    assert value == pytest.approx(0.0)


def test_cyclic_symmetry_composite_penalizes_nonplanar_same_radius_arrangement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cyclic_losses,
        "get_coord_CA",
        lambda _path: _coords(
            [
                ["A", 1, 1.0, 0.0, 1.0],
                ["B", 1, 0.0, 1.0, -1.0],
                ["C", 1, -1.0, 0.0, 1.0],
                ["D", 1, 0.0, -1.0, -1.0],
            ]
        ),
    )

    value = cyclic_losses.calculate_cyclic_symmetry_composite_single(
        "ABCD",
        SimpleNamespace(complex_paths={"ABCD": Path("bad.cif")}),
    )

    assert value > 0.2
