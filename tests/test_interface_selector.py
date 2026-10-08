from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.evaluators.utils import ComplexPoses
import astrolabe.selectors.interface as interface_module
from astrolabe.selectors.interface import InterfacePositionSelector
from astrolabe.selectors.utils import construct_position_selector


class _Residue:
    def __init__(
        self,
        atoms: dict[str, tuple[float, float, float]],
        *,
        protein: bool = True,
    ):
        self._atoms = {
            name: np.array(coord, dtype=float)
            for name, coord in atoms.items()
        }
        self._names = list(self._atoms)
        self._protein = protein

    def is_protein(self) -> bool:
        return self._protein

    def has(self, atom_name: str) -> bool:
        return atom_name in self._atoms

    def xyz(self, atom) -> np.ndarray:
        if isinstance(atom, str):
            return self._atoms[atom]
        return self._atoms[self._names[atom - 1]]

    def natoms(self) -> int:
        return len(self._names)

    def atom_name(self, index: int) -> str:
        return self._names[index - 1]


class _Pose:
    def __init__(self):
        self._chains = {1: (1, 3), 2: (4, 5)}
        self._residues = {
            1: _Residue({"CA": (0.0, 0.0, 0.0), "CB": (0.5, 0.0, 0.0)}),
            2: _Residue({"CA": (20.0, 0.0, 0.0), "CB": (9.0, 0.0, 0.0)}),
            3: _Residue({"CA": (50.0, 0.0, 0.0), "CB": (50.5, 0.0, 0.0)}),
            4: _Residue(
                {"CA": (5.0, 0.0, 0.0), "CB": (5.5, 0.0, 0.0), "H": (9.0, 0.0, 0.0)}
            ),
            5: _Residue(
                {"CA": (42.0, 0.0, 0.0), "CB": (42.5, 0.0, 0.0), "H": (50.5, 0.0, 0.0)}
            ),
        }

    def num_chains(self) -> int:
        return len(self._chains)

    def chain_begin(self, chain_index: int) -> int:
        return self._chains[chain_index][0]

    def chain_end(self, chain_index: int) -> int:
        return self._chains[chain_index][1]

    def residue(self, residue_index: int) -> _Residue:
        return self._residues[residue_index]

    def total_residue(self) -> int:
        return len(self._residues)


class _SidechainPose(_Pose):
    def __init__(self):
        self._chains = {1: (1, 3), 2: (4, 5)}
        self._residues = {
            1: _Residue({"CA": (0.0, 0.0, 0.0), "CB": (10.0, 0.0, 0.0)}),
            2: _Residue({"CA": (3.0, 0.0, 0.0), "CB": (30.0, 0.0, 0.0)}),
            3: _Residue({"CA": (50.0, 0.0, 0.0), "CB": (50.0, 0.0, 0.0)}),
            4: _Residue({"CA": (100.0, 0.0, 0.0), "CB": (13.5, 0.0, 0.0)}),
            5: _Residue({"CA": (4.0, 0.0, 0.0), "CB": (60.0, 0.0, 0.0)}),
        }


def _protomer(length: int):
    return SimpleNamespace(length=length)


def _chain_name(chain_index: int, _) -> str:
    return {1: "A", 2: "B"}[chain_index]


def test_interface_selector_default_uses_heavy_atom_distances(monkeypatch) -> None:
    def fail_rosetta_selector(*_):
        pytest.fail("default mode should not call the PyRosetta interface selector")

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fail_rosetta_selector,
    )

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": _Pose()}),
        interface_distance=4.0,
    )

    np.testing.assert_array_equal(selector.sites["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([True, False]))


def test_interface_selector_rosetta_mode_preserves_pyrosetta_selector(monkeypatch) -> None:
    pose = _Pose()
    calls = []

    def fake_select_interface_residues(pose_arg, interface, cutoff):
        calls.append((pose_arg, interface, cutoff))
        return [True, False, False, True, False]

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fake_select_interface_residues,
    )

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": pose}),
        interface_distance=8.6,
        mode="rosetta",
    )

    assert calls == [(pose, "A_B", 9)]
    np.testing.assert_array_equal(selector.sites["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([True, False]))


def test_interface_selector_ca_mode_uses_ca_distances(monkeypatch) -> None:
    def fail_rosetta_selector(*_):
        pytest.fail("CA mode should not call the PyRosetta interface selector")

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fail_rosetta_selector,
    )

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": _Pose()}),
        interface_distance=6.0,
        mode="ca",
    )

    np.testing.assert_array_equal(selector.sites["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([True, False]))


def test_interface_selector_buried_sasa_mode_uses_delta_sasa_cutoff(monkeypatch) -> None:
    def fail_rosetta_selector(*_):
        pytest.fail("buried_sasa mode should not call the PyRosetta interface selector")

    calls = []

    def fake_pair_buried_sasa_values(cls, pose, chain_i_index, chain_j_index):
        calls.append((pose, chain_i_index, chain_j_index))
        return np.array([0.5, 2.0, 1.0]), np.array([1.1, 0.0])

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fail_rosetta_selector,
    )
    monkeypatch.setattr(
        InterfacePositionSelector,
        "_pair_buried_sasa_values",
        classmethod(fake_pair_buried_sasa_values),
    )
    pose = _Pose()

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": pose}),
        mode="buried_sasa",
        buried_sasa_cutoff=1.0,
    )

    assert calls == [(pose, 1, 2)]
    np.testing.assert_array_equal(selector.sites["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([True, False]))


def test_interface_selector_buried_sasa_mode_defaults_to_10a2_cutoff(monkeypatch) -> None:
    def fake_pair_buried_sasa_values(cls, pose, chain_i_index, chain_j_index):
        return np.array([9.9, 10.1, 20.0]), np.array([10.0, 10.01])

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        InterfacePositionSelector,
        "_pair_buried_sasa_values",
        classmethod(fake_pair_buried_sasa_values),
    )

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": _Pose()}),
        mode="buried_sasa",
    )

    np.testing.assert_array_equal(selector.sites["A"], np.array([False, True, True]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([False, True]))


def test_construct_position_selector_uses_heavy_atom_default(monkeypatch) -> None:
    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    protomers = {"A": _protomer(3), "B": _protomer(2)}
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        "interface(interface_distance=4.0)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([True, False]))


def test_construct_position_selector_accepts_interface_buried_sasa_mode(monkeypatch) -> None:
    def fake_pair_buried_sasa_values(cls, pose, chain_i_index, chain_j_index):
        return np.array([1.5, 2.5, 0.0]), np.array([0.9, 3.0])

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        InterfacePositionSelector,
        "_pair_buried_sasa_values",
        classmethod(fake_pair_buried_sasa_values),
    )
    protomers = {"A": _protomer(3), "B": _protomer(2)}
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        "interface(mode='buried_sasa', buried_sasa_cutoff=2.0)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([False, True]))


def test_construct_position_selector_accepts_interface_ca_mode(monkeypatch) -> None:
    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    protomers = {"A": _protomer(3), "B": _protomer(2)}
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        "interface(mode='ca', interface_distance=6.0)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selection["B"], np.array([True, False]))


def test_interface_selector_sidechain_heavy_mode_uses_sidechain_distances(
    monkeypatch,
) -> None:
    def fail_rosetta_selector(*_):
        pytest.fail("sidechain_heavy mode should not call the PyRosetta selector")

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fail_rosetta_selector,
    )

    selector = InterfacePositionSelector(
        ComplexPoses({"AB": _SidechainPose()}),
        interface_distance=4.0,
        mode="sidechain_heavy",
    )

    np.testing.assert_array_equal(selector.sites["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([True, False]))


def test_construct_position_selector_accepts_interface_sidechain_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    protomers = {"A": _protomer(3), "B": _protomer(2)}
    evaluation_result = SimpleNamespace(
        complex_poses=ComplexPoses({"AB": _SidechainPose()})
    )

    selection = construct_position_selector(
        "interface(mode='sidechain_heavy', interface_distance=4.0)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selection["B"], np.array([True, False]))


def test_interface_selector_sasa_mode_uses_relative_sasa_cutoff(monkeypatch) -> None:
    def fail_rosetta_selector(*_):
        pytest.fail("sasa mode should not call the PyRosetta interface selector")

    calls = []

    def fake_bound_and_unbound_sasa(pose, *, sidechain_only: bool):
        calls.append((pose, sidechain_only))
        return (
            np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            np.array([0.22, 0.25, 0.41, 0.46, 0.75]),
        )

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        interface_module,
        "select_interface_residues",
        fail_rosetta_selector,
    )
    monkeypatch.setattr(
        InterfacePositionSelector,
        "_bound_and_unbound_residue_sasa",
        staticmethod(fake_bound_and_unbound_sasa),
    )

    pose = _Pose()
    selector = InterfacePositionSelector(
        ComplexPoses({"AB": pose}),
        mode="sasa",
        sasa_cutoff=0.1,
    )

    assert calls == [(pose, True)]
    np.testing.assert_array_equal(selector.sites["A"], np.array([True, False, True]))
    np.testing.assert_array_equal(selector.sites["B"], np.array([False, True]))


def test_construct_position_selector_accepts_interface_sasa_mode(
    monkeypatch,
) -> None:
    def fake_bound_and_unbound_sasa(pose, *, sidechain_only: bool):
        assert sidechain_only is False
        return (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([10.9, 25.0, 30.1, 50.0, 50.2]),
        )

    monkeypatch.setattr(interface_module, "get_chain_from_chain_id", _chain_name)
    monkeypatch.setattr(
        InterfacePositionSelector,
        "_bound_and_unbound_residue_sasa",
        staticmethod(fake_bound_and_unbound_sasa),
    )
    protomers = {"A": _protomer(3), "B": _protomer(2)}
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        "interface(mode='sasa', sasa_cutoff=0.1, sasa_sidechain_only=False)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([True, False]))


def test_interface_selector_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        InterfacePositionSelector(ComplexPoses({}), mode="cb")


def test_interface_selector_rejects_relative_sasa_cutoff_outside_fraction() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        InterfacePositionSelector(ComplexPoses({}), mode="sasa", sasa_cutoff=4.0)
