from __future__ import annotations

import pathlib

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.protein_quality import (
    BuriedUnsatPositionSelector,
    CavityNeighborPositionSelector,
    SurfaceHighSAPPositionSelector,
    _residue_subset_to_chain_masks,
)
from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized


def _try_pose():
    try:
        import pyrosetta  # type: ignore
    except ImportError:
        pytest.skip("PyRosetta not available")
    if not pyrosetta_was_initialized(pyrosetta):
        dalpha = pathlib.Path("astrolabe/assets/DAlphaBall.gcc").resolve()
        pyrosetta.init(f"-mute all -holes:dalphaball {dalpha}")
    return pyrosetta.pose_from_sequence("ACDEFGHIK")


def _protomer(length: int):
    return type(
        "Protomer",
        (),
        {
            "length": length,
            "type": ProtomerTypes.PROTEIN,
            "mutable_mask": np.ones(length, dtype=bool),
        },
    )()


def setup_function() -> None:
    global_config.reset()


def teardown_function() -> None:
    global_config.reset()


def test_surface_high_sap_selector_marks_high_sap_surface_residues() -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    selector = SurfaceHighSAPPositionSelector(
        ComplexPoses({"A": pose}),
        threshold=3.0,
    )

    assert selector.sites["A"].any()
    assert selector.sites["A"][4] or selector.sites["A"][7]


def test_surface_high_sap_selector_has_dalphaball_proxy(monkeypatch) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    monkeypatch.setattr(
        "astrolabe.selectors.protein_quality._per_residue_sap_scores",
        lambda _pose: (_ for _ in ()).throw(RuntimeError("DAlphaBall unavailable")),
    )

    selector = SurfaceHighSAPPositionSelector(
        ComplexPoses({"A": pose}),
        threshold=2.5,
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)


def test_residue_subset_to_chain_masks_merges_repeated_protomers() -> None:
    class _Pose:
        def num_chains(self) -> int:
            return 2

        def chain_begin(self, chain_index: int) -> int:
            return {1: 1, 2: 4}[chain_index]

        def chain_end(self, chain_index: int) -> int:
            return {1: 3, 2: 6}[chain_index]

    masks = _residue_subset_to_chain_masks(
        _Pose(),
        [True, False, False, False, True, False],
        {"A": "A", "B": "A"},
    )

    assert masks["A"].tolist() == [True, True, False]


def test_buried_unsat_selector_uses_per_residue_unsat_filter(monkeypatch) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    def _fake_unsat_score(self, _pose, idx):
        return 1.0 if idx == pose.total_residue() else 0.0

    monkeypatch.setattr(
        BuriedUnsatPositionSelector,
        "_residue_unsat_score",
        _fake_unsat_score,
    )

    selector = BuriedUnsatPositionSelector(
        ComplexPoses({"A": pose}),
        threshold=0.5,
        ignore_surface_res=False,
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)
    assert selector.sites["A"][-1]


def test_buried_unsat_selector_has_dalphaball_proxy(monkeypatch) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    monkeypatch.setattr(
        BuriedUnsatPositionSelector,
        "_residue_unsat_score",
        lambda self, _pose, _idx: (_ for _ in ()).throw(
            RuntimeError("libgfortran DAlphaBall unavailable")
        ),
    )

    selector = BuriedUnsatPositionSelector(
        ComplexPoses({"A": pose}),
        threshold=0.5,
        ignore_surface_res=False,
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)


def test_buried_unsat_selector_has_atom_type_backend_proxy(monkeypatch) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    monkeypatch.setattr(
        BuriedUnsatPositionSelector,
        "_residue_unsat_score",
        lambda self, _pose, _idx: (_ for _ in ()).throw(
            RuntimeError("unrecognized atom_type_name 'CS'")
        ),
    )

    selector = BuriedUnsatPositionSelector(
        ComplexPoses({"A": pose}),
        threshold=0.5,
        ignore_surface_res=False,
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)


def test_buried_unsat_selector_explicit_proxy_skips_per_residue_filter(
    monkeypatch,
) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    def _unexpected_filter(self, _pose, _idx):
        raise AssertionError("explicit proxy mode should not call per-residue filter")

    monkeypatch.setattr(
        BuriedUnsatPositionSelector,
        "_residue_unsat_score",
        _unexpected_filter,
    )

    selector = BuriedUnsatPositionSelector(
        ComplexPoses({"A": pose}),
        method="buried_polar_proxy",
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)


def test_cavity_neighbor_selector_expands_high_holes_residues() -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _protomer(pose.total_residue())})

    selector = CavityNeighborPositionSelector(
        ComplexPoses({"A": pose}),
        holes_threshold=100.0,
        neighbor_distance=4.0,
    )

    assert selector.sites["A"].shape == (pose.total_residue(),)
    assert selector.sites["A"].any()


def test_cavity_neighbor_auto_uses_packstat_proxy_for_large_poses(monkeypatch) -> None:
    class _Residue:
        def is_protein(self) -> bool:
            return True

    class _Pose:
        def total_residue(self) -> int:
            return 4

        def residue(self, _idx):
            return _Residue()

        def chain(self, _idx) -> int:
            return 1

    class _Scores:
        _values = {1: 0.20, 2: 0.26, 3: 0.60, 4: 0.10}

        def __getitem__(self, idx):
            return self._values[idx]

    global_config.init_const("protomers", {"A": _protomer(4)})
    monkeypatch.setattr(
        "astrolabe.selectors.protein_quality._compute_residue_packstat_scores",
        lambda _pose: _Scores(),
    )

    selector = CavityNeighborPositionSelector.__new__(CavityNeighborPositionSelector)
    selector.holes_threshold = 40.0
    selector.packstat_cutoff = None
    selector.method = "auto"
    selector.max_holes_filter_residues = 2

    def _unexpected_holes_filter(_pose, _chain2proto):
        raise AssertionError("large auto mode should not call per-residue HolesFilter")

    monkeypatch.setattr(
        selector,
        "_holes_filter_focus_indices",
        _unexpected_holes_filter,
    )

    focus = selector._focus_indices("A", _Pose(), {"A": "A"})

    assert focus == ["1", "4"]
