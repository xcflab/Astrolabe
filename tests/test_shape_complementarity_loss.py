from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.losses import interaction


def test_compute_shape_complementarity_between_chains_returns_filter_value(
    monkeypatch,
) -> None:
    fake_sub_pose = SimpleNamespace(
        total_residue=lambda: 10,
        num_chains=lambda: 2,
    )

    class FakeShapeComplementarityFilter:
        def jump_id(self, jump_id: int) -> None:
            assert jump_id == 1

        def use_rosetta_radii(self, enabled: bool) -> None:
            assert enabled is True

        def report_sm(self, pose) -> float:
            assert pose is fake_sub_pose
            return 0.42

    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)
    monkeypatch.setattr(
        interaction,
        "_isolate_chains",
        lambda pose, keep_chains: fake_sub_pose,
    )
    monkeypatch.setattr(
        interaction,
        "ShapeComplementarityFilter",
        FakeShapeComplementarityFilter,
    )

    out = interaction._compute_shape_complementarity_between_chains(
        object(),
        "A",
        "B",
    )

    assert out == pytest.approx(0.42)
