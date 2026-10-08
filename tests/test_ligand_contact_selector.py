from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes, SegmentManager
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.ligand import LigandContactPositionSelector
from astrolabe.selectors.protein_quality import (
    BuriedUnsatPositionSelector,
    CavityNeighborPositionSelector,
    SurfaceHighSAPPositionSelector,
)
from astrolabe.selectors.utils import construct_position_selector


class _AtomType:
    def __init__(self, element: str):
        self._element = element

    def element(self) -> str:
        return self._element


class _Residue:
    def __init__(
        self,
        coords: list[tuple[float, float, float]],
        *,
        element: str = "C",
        protein: bool = True,
        ligand: bool = False,
    ):
        self._coords = [np.array(c, dtype=float) for c in coords]
        self._types = [_AtomType(element) for _ in coords]
        self._protein = protein
        self._ligand = ligand

    def natoms(self) -> int:
        return len(self._coords)

    def atom_type(self, index: int) -> _AtomType:
        return self._types[index - 1]

    def xyz(self, index: int) -> np.ndarray:
        return self._coords[index - 1]

    def is_protein(self) -> bool:
        return self._protein

    def is_ligand(self) -> bool:
        return self._ligand


class _Pose:
    def __init__(self):
        self._residues = {
            1: _Residue([(0.0, 0.0, 0.0)], protein=True),
            2: _Residue([(10.0, 0.0, 0.0)], protein=True),
            3: _Residue([(20.0, 0.0, 0.0)], protein=True),
            4: _Residue([(10.8, 0.0, 0.0)], protein=False, ligand=True),
        }
        self._chains = {1: (1, 3), 2: (4, 4)}

    def num_chains(self) -> int:
        return len(self._chains)

    def chain_begin(self, chain_index: int) -> int:
        return self._chains[chain_index][0]

    def chain_end(self, chain_index: int) -> int:
        return self._chains[chain_index][1]

    def residue(self, residue_index: int) -> _Residue:
        return self._residues[residue_index]

    def chain(self, residue_index: int) -> int:
        for chain_index, (begin, end) in self._chains.items():
            if begin <= residue_index <= end:
                return chain_index
        raise KeyError(residue_index)


def _protomer(length: int, proto_type: ProtomerTypes):
    return SimpleNamespace(
        length=length,
        type=proto_type,
        mutable_mask=np.ones(length, dtype=bool),
        aggregate_bool=lambda mask: mask,
    )


def setup_function() -> None:
    global_config.reset()


def teardown_function() -> None:
    global_config.reset()


def test_ligand_contact_selector_marks_current_ligand_neighbors() -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    pose = _Pose()
    selector = LigandContactPositionSelector(
        complex_poses=ComplexPoses({"AB": pose}),
        distance_max=2.0,
    )

    selection = selector.select_positions()

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))


def test_construct_position_selector_accepts_ligand_contact_expression() -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        "ligand_contact(distance_max=2.0)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))


def test_construct_position_selector_accepts_hotspot_alias_expression() -> None:
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    protomers = {
        "A": Protomer("A", "AC@shared_1 DE@itf_A_1 FG@shared_2", "protein"),
        "B": Protomer("B", "shared_1 HI@itf_B_1 shared_2", "protein"),
    }
    global_config.init_const("protomers", protomers)
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({}))

    selection = construct_position_selector(
        "hotspot('shared_1') or hotspot('itf_B_1')",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(
        selection["A"], np.array([True, True, False, False, False, False])
    )
    np.testing.assert_array_equal(
        selection["B"], np.array([True, True, True, True, False, False])
    )


def test_construct_position_selector_accepts_protein_quality_selectors(monkeypatch) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.PROTEIN),
    }
    global_config.init_const("protomers", protomers)
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({}))

    monkeypatch.setattr(
        SurfaceHighSAPPositionSelector,
        "select_positions",
        lambda self, **_: setattr(
            self,
            "sites",
            {
                "A": np.array([True, False, False]),
                "B": np.array([False]),
            },
        )
        or self.sites,
    )
    monkeypatch.setattr(
        BuriedUnsatPositionSelector,
        "select_positions",
        lambda self, **_: setattr(
            self,
            "sites",
            {
                "A": np.array([False, True, False]),
                "B": np.array([False]),
            },
        )
        or self.sites,
    )
    monkeypatch.setattr(
        CavityNeighborPositionSelector,
        "select_positions",
        lambda self, **_: setattr(
            self,
            "sites",
            {
                "A": np.array([False, False, True]),
                "B": np.array([True]),
            },
        )
        or self.sites,
    )

    selection = construct_position_selector(
        "surface_high_sap(threshold=2.0) or buried_unsat(threshold=1.0) or cavity_neighbor(holes_threshold=30)",
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([True, True, True]))
    np.testing.assert_array_equal(selection["B"], np.array([True]))


def test_construct_position_selector_builds_complex_poses_for_quality_selectors(
    monkeypatch,
) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.PROTEIN),
    }
    global_config.init_const("protomers", protomers)
    constructed = ComplexPoses({"AB": _Pose()})
    calls = []

    def _construct(eval_result):
        calls.append(eval_result.complex_paths)
        return constructed

    monkeypatch.setattr(ComplexPoses, "construct_complex_poses", _construct)
    monkeypatch.setattr(
        SurfaceHighSAPPositionSelector,
        "select_positions",
        lambda self, **_: setattr(
            self,
            "sites",
            {
                "A": np.array([True, False, False]),
                "B": np.array([False]),
            },
        )
        or self.sites,
    )
    evaluation_result = SimpleNamespace(
        complex_paths={"AB": "pred.cif"},
        complex_poses=None,
    )

    selection = construct_position_selector(
        "surface_high_sap(threshold=2.0)",
        evaluation_result,
        protomers,
    )

    assert calls == [{"AB": "pred.cif"}]
    assert evaluation_result.complex_poses is constructed
    np.testing.assert_array_equal(selection["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))
