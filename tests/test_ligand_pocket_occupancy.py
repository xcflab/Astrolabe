from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses import interaction
from astrolabe.losses.base import LossEngine
from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue


class _AtomType:
    def __init__(self, element: str):
        self._element = element

    def element(self):
        return self._element


class _XYZ:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _Residue:
    def __init__(self, coords):
        self._coords = list(coords)

    def natoms(self):
        return len(self._coords)

    def atom_type(self, atom_idx):
        return _AtomType("C")

    def xyz(self, atom_idx):
        return _XYZ(*self._coords[atom_idx - 1])


class _PdbInfo:
    def __init__(self, chain_by_idx, number_by_idx):
        self._chain_by_idx = dict(chain_by_idx)
        self._number_by_idx = dict(number_by_idx)

    def chain(self, idx):
        return self._chain_by_idx[idx]

    def number(self, idx):
        return self._number_by_idx[idx]


class _Pose:
    def __init__(self, residues, chain_by_idx, number_by_idx):
        self._residues = list(residues)
        self._pdb_info = _PdbInfo(chain_by_idx, number_by_idx)

    def total_residue(self):
        return len(self._residues)

    def residue(self, idx):
        return self._residues[idx - 1]

    def pdb_info(self):
        return self._pdb_info


def _eval_result(ligand_coords):
    pose = _Pose(
        residues=[
            _Residue([(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)]),
            _Residue([(3.0, 3.0, 3.0), (5.0, 5.0, 5.0)]),
            _Residue(ligand_coords),
        ],
        chain_by_idx={1: "A", 2: "A", 3: "B"},
        number_by_idx={1: 1, 2: 2, 3: 1},
    )
    return SimpleNamespace(
        names=["AB"],
        complex_poses=SimpleNamespace(poses={"AB": pose}),
    )


def _eval_result_multichain_pocket(ligand_coords):
    pose = _Pose(
        residues=[
            _Residue([(0.0, 0.0, 0.0)]),
            _Residue([(30.0, 30.0, 30.0)]),
            _Residue([(2.0, 2.0, 2.0)]),
            _Residue([(40.0, 40.0, 40.0)]),
            _Residue(ligand_coords),
        ],
        chain_by_idx={1: "A", 2: "A", 3: "B", 4: "B", 5: "C"},
        number_by_idx={1: 10, 2: 20, 3: 20, 4: 30, 5: 1},
    )
    return SimpleNamespace(
        names=["ABC"],
        complex_poses=SimpleNamespace(poses={"ABC": pose}),
    )


def _init_protomers(*, dualvhh: bool = False):
    global_config.reset()
    b_type = ProtomerTypes.PROTEIN if dualvhh else ProtomerTypes.SMILES
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "B": SimpleNamespace(type=b_type),
            "C": SimpleNamespace(type=ProtomerTypes.SMILES),
        },
    )


def test_ligand_pocket_occupancy_scores_vina_style_inside_pose(monkeypatch):
    _init_protomers()
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)

    score = interaction.ligand_pocket_occupancy(
        _eval_result([(1.0, 1.0, 1.0), (1.5, 1.5, 1.5)]),
        pocket_chain="A",
        pocket_residues=[1, 2],
        pocket_padding=0.0,
    )

    assert score == {"AB": 1.0}


def test_ligand_pocket_occupancy_scores_outside_pose_low(monkeypatch):
    _init_protomers()
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)

    score = interaction.ligand_pocket_occupancy(
        _eval_result([(9.0, 9.0, 9.0), (10.0, 10.0, 10.0)]),
        pocket_chain="A",
        pocket_residues=[1, 2],
        pocket_padding=0.0,
    )

    assert score == {"AB": 0.0}


def test_ligand_pocket_occupancy_accepts_chain_qualified_pocket_residue_map(monkeypatch):
    _init_protomers(dualvhh=True)
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)

    score = interaction.ligand_pocket_occupancy(
        _eval_result_multichain_pocket([(1.0, 1.0, 1.0), (1.5, 1.5, 1.5)]),
        pocket_chain="A,B",
        pocket_residues_by_chain={"A": [10], "B": [20]},
        pocket_padding=0.0,
    )

    assert score == {"ABC": 1.0}


def test_ligand_pocket_occupancy_chain_qualified_map_does_not_mix_equal_numbers(monkeypatch):
    _init_protomers(dualvhh=True)
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)

    score = interaction.ligand_pocket_occupancy(
        _eval_result_multichain_pocket([(10.0, 10.0, 10.0)]),
        pocket_chain="A,B",
        pocket_residues_by_chain={"A": [10], "B": [20]},
        pocket_padding=0.0,
    )

    assert score == {"ABC": 0.0}


def test_ligand_pocket_occupancy_schema_accepts_chain_qualified_map_without_legacy_residues():
    loss = LossEngine(
        {
            "loss_terms": {
                "ligand_in_pocket": {
                    "_type": "ligand_pocket_occupancy",
                    "pocket_chain": "A,B",
                    "pocket_residues_by_chain": {"A": [10], "B": [20]},
                }
            },
            "expression_terms": {},
        }
    )

    loss.setup(
        weighted_score_expression="ligand_in_pocket",
        early_stopping=None,
        acceptance_threshold=None,
    )

    assert loss.loss_specs["ligand_in_pocket"][2] == {
        "pocket_chain": "A,B",
        "pocket_residues_by_chain": {"A": [10], "B": [20]},
    }


def test_autodock_vina_uses_configured_ligand_chain_for_dualvhh_oligo(tmp_path):
    _init_protomers(dualvhh=True)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )

    assert predocker._configured_ligand_chain_id_for_oligo("ABC") == "C"


def test_autodock_vina_requires_one_configured_ligand_chain(tmp_path):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        },
    )
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )

    with pytest.raises(ValueError, match="Expected exactly one ligand protomer"):
        predocker._configured_ligand_chain_id_for_oligo("AB")


def test_autodock_vina_rank_vina_order_selects_first_pose(tmp_path):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    first = tmp_path / "pose_0.pdb"
    second = tmp_path / "pose_1.pdb"
    first.write_text("END\n", encoding="utf-8")
    second.write_text("END\n", encoding="utf-8")
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_vina_order=True,
    )

    selected = predocker._filter_ranking(
        {
            0: (str(first), "C"),
            1: (str(second), "C"),
        }
    )

    assert selected == str(first)
    assert predocker._last_pose_candidates == [str(first), str(second)]
