from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.losses.interaction import hotspot_buried_ratio


class _Protomer:
    def __init__(self, selector_values):
        self.selector_values = selector_values

    def select_residues_by_alias(self, expression: str):
        return self.selector_values.copy()


def _eval_result(names=None):
    names = names or ["AB"]
    poses = {name: object() for name in names}
    return SimpleNamespace(
        names=names,
        complex_poses=SimpleNamespace(poses=poses),
    )


def test_hotspot_buried_ratio_uses_selected_positions(monkeypatch) -> None:
    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction.global_config",
        SimpleNamespace(
            protomers={
                "A": _Protomer([0.0, 0.0]),
                "B": _Protomer([1.0, 1.0, 0.0]),
            }
        ),
    )

    def fake_sums(pose, selector, *, sidechain_only=True):
        assert list(selector) == [0.0, 0.0, 1.0, 1.0, 0.0]
        assert sidechain_only is True
        return 24.0, 60.0

    monkeypatch.setattr(
        "astrolabe.losses.interaction._selected_bound_and_unbound_exposed_sasa",
        fake_sums,
    )

    out = hotspot_buried_ratio(
        _eval_result(),
        select_expression="B_1:35,39",
    )

    assert out["AB"] == pytest.approx(0.6)


def test_hotspot_buried_ratio_is_neutral_when_unbound_exposure_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction.global_config",
        SimpleNamespace(
            protomers={
                "A": _Protomer([0.0, 0.0]),
                "B": _Protomer([1.0, 0.0]),
            }
        ),
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction._selected_bound_and_unbound_exposed_sasa",
        lambda pose, selector, sidechain_only=True: (0.0, 0.0),
    )

    out = hotspot_buried_ratio(
        _eval_result(),
        select_expression="B_1:35",
    )

    assert out["AB"] == pytest.approx(0.0)


def test_hotspot_buried_ratio_is_neutral_for_single_chain(monkeypatch) -> None:
    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction.global_config",
        SimpleNamespace(protomers={"A": _Protomer([1.0, 1.0])}),
    )

    out = hotspot_buried_ratio(
        _eval_result(["A"]),
        select_expression="A_1:1-2",
    )

    assert out["A"] == pytest.approx(0.0)
