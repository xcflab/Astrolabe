from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses import interaction
from astrolabe.losses.base import LossEngine


def _init_protomers(*, interface: bool = False):
    global_config.reset()
    protomers = {
        "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
        "B": SimpleNamespace(type=ProtomerTypes.SMILES),
    }
    if interface:
        protomers["E"] = SimpleNamespace(type=ProtomerTypes.PROTEIN)
    global_config.init_const("protomers", protomers)


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    coord,
    *,
    record: str = "ATOM",
    element: str = "C",
) -> str:
    x, y, z = coord
    return (
        f"{record:<6}{serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def _write_protein_ligand_pdb(path, chains: dict[str, np.ndarray], ligand_chain: str, ligand_coords):
    serial = 1
    lines = []
    for chain_id, coords in chains.items():
        for resseq, coord in enumerate(coords, start=1):
            lines.append(
                _pdb_atom_line(serial, "CA", "ALA", chain_id, resseq, coord, element="C")
            )
            serial += 1
    for atom_idx, coord in enumerate(ligand_coords, start=1):
        lines.append(
            _pdb_atom_line(
                serial,
                f"C{atom_idx}",
                "LIG",
                ligand_chain,
                1,
                coord,
                record="HETATM",
                element="C",
            )
        )
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_cb_pocket_ligand_pdb(
    path,
    chains: dict[str, list[tuple[str, np.ndarray]]],
    ligand_chain: str,
    ligand_coords,
    *,
    omit_cb_positions_by_chain: dict[str, set[int]] | None = None,
):
    omit_cb_positions_by_chain = omit_cb_positions_by_chain or {}
    serial = 1
    lines = []
    for chain_id, residues in chains.items():
        for resseq, (resname, cb_coord) in enumerate(residues, start=1):
            cb_coord = np.asarray(cb_coord, dtype=float)
            ca_coord = cb_coord + np.asarray([0.0, 0.0, -0.1], dtype=float)
            lines.append(
                _pdb_atom_line(serial, "CA", resname, chain_id, resseq, ca_coord, element="C")
            )
            serial += 1
            if resname != "GLY" and resseq not in omit_cb_positions_by_chain.get(
                chain_id,
                set(),
            ):
                lines.append(
                    _pdb_atom_line(serial, "CB", resname, chain_id, resseq, cb_coord, element="C")
                )
                serial += 1
    for atom_idx, coord in enumerate(ligand_coords, start=1):
        lines.append(
            _pdb_atom_line(
                serial,
                f"C{atom_idx}",
                "LIG",
                ligand_chain,
                1,
                coord,
                record="HETATM",
                element="C",
            )
        )
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_ligand_pdb(path, coords):
    lines = [
        _pdb_atom_line(
            atom_idx,
            f"C{atom_idx}",
            "LIG",
            "B",
            1,
            coord,
            record="HETATM",
            element="C",
        )
        for atom_idx, coord in enumerate(coords, start=1)
    ]
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _eval_result(oligo_name: str, structure_path):
    class _EvalResult:
        pass

    result = _EvalResult()
    result.names = [oligo_name]
    result.complex_paths = {oligo_name: str(structure_path)}
    return result


def _init_dual_protein_ligand_protomers():
    global_config.reset()
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "C": SimpleNamespace(type=ProtomerTypes.SMILES),
        },
    )


def test_ligand_aligned_reference_pose_distance_scores_target_pose_near_zero(tmp_path):
    _init_protomers()
    reference_ca = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    reference_ligand = np.asarray([[2.0, 0.2, 0.0], [2.4, 0.8, 0.0]], dtype=float)
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference.pdb",
        {"A": reference_ca},
        "B",
        reference_ligand,
    )
    reference_ligand_path = _write_ligand_pdb(
        tmp_path / "reference_ligand.pdb",
        reference_ligand,
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current.pdb",
        {"A": reference_ca + shift},
        "B",
        reference_ligand + shift,
    )

    score = interaction.ligand_aligned_reference_pose_distance(
        _eval_result("AB", current),
        reference_structure_path=str(reference),
        reference_ligand_path=str(reference_ligand_path),
        reference_structure_chain="A",
        current_structure_chain="A",
        min_ca_atoms=3,
    )

    assert score["AB"] < 1e-6


def test_ligand_aligned_reference_pose_distance_scores_wrong_pose_far(tmp_path):
    _init_protomers()
    reference_ca = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    reference_ligand = np.asarray([[2.0, 0.2, 0.0], [2.4, 0.8, 0.0]], dtype=float)
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)
    wrong_pose_offset = np.asarray([5.5, 0.0, 0.0], dtype=float)

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference.pdb",
        {"A": reference_ca},
        "B",
        reference_ligand,
    )
    reference_ligand_path = _write_ligand_pdb(
        tmp_path / "reference_ligand.pdb",
        reference_ligand,
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current_wrong.pdb",
        {"A": reference_ca + shift},
        "B",
        reference_ligand + shift + wrong_pose_offset,
    )

    score = interaction.ligand_aligned_reference_pose_distance(
        _eval_result("AB", current),
        reference_structure_path=str(reference),
        reference_ligand_path=str(reference_ligand_path),
        reference_structure_chain="A",
        current_structure_chain="A",
        min_ca_atoms=3,
    )

    assert score["AB"] > 4.0


def test_ligand_aligned_reference_pose_distance_supports_interface_chain_mapping(tmp_path):
    _init_protomers(interface=True)
    reference_a = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    reference_e = np.asarray(
        [[5.0, 0.0, 0.0], [6.5, 0.0, 0.0], [5.0, 1.5, 0.0]], dtype=float
    )
    reference_ligand = np.asarray([[2.6, 0.4, 0.0], [3.0, 0.9, 0.0]], dtype=float)
    shift = np.asarray([-4.0, 8.0, 1.5], dtype=float)

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference_interface.pdb",
        {"A": reference_a, "E": reference_e},
        "B",
        reference_ligand,
    )
    reference_ligand_path = _write_ligand_pdb(
        tmp_path / "reference_ligand.pdb",
        reference_ligand,
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current_interface.pdb",
        {
            "A": reference_a + shift,
            "B": reference_e + shift,
        },
        "C",
        reference_ligand + shift,
    )

    score = interaction.ligand_aligned_reference_pose_distance(
        _eval_result("AEB", current),
        reference_structure_path=str(reference),
        reference_ligand_path=str(reference_ligand_path),
        reference_structure_chain={"A": "A", "E": "E"},
        current_structure_chain={"A": "A", "E": "B"},
        min_ca_atoms=3,
    )

    assert score["AEB"] < 1e-6


def test_ligand_reference_pose_distance_is_available_to_loss_conditions(tmp_path):
    _init_protomers()
    reference_ca = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    reference_ligand = np.asarray([[2.0, 0.2, 0.0], [2.4, 0.8, 0.0]], dtype=float)
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference.pdb",
        {"A": reference_ca},
        "B",
        reference_ligand,
    )
    reference_ligand_path = _write_ligand_pdb(
        tmp_path / "reference_ligand.pdb",
        reference_ligand,
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current_wrong.pdb",
        {"A": reference_ca + shift},
        "B",
        reference_ligand + shift + np.asarray([5.5, 0.0, 0.0], dtype=float),
    )

    loss = LossEngine(
        {
            "loss_terms": {
                "target_ligand_pose_dist": {
                    "_type": "ligand_aligned_reference_pose_distance",
                    "reference_structure_path": str(reference),
                    "reference_ligand_path": str(reference_ligand_path),
                    "reference_structure_chain": "A",
                    "current_structure_chain": "A",
                    "min_ca_atoms": 3,
                }
            },
            "expression_terms": {},
        }
    )
    loss.setup(
        weighted_score_expression="target_ligand_pose_dist",
        early_stopping=None,
        acceptance_threshold=None,
    )

    env = loss.score_context(_eval_result("AB", current))

    assert env["target_ligand_pose_dist"] > 4.0
    assert loss.eval_condition(env, "target_ligand_pose_dist > 4.0")


def test_ligand_aligned_reference_pocket_box_occupancy_counts_heavy_atom_fraction(tmp_path):
    _init_protomers()
    reference_ca = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)
    ligand_coords = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [8.0, 0.5, 0.0],
        ],
        dtype=float,
    )

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference.pdb",
        {"A": reference_ca},
        "B",
        np.asarray([[0.5, 0.5, 0.0]], dtype=float),
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current.pdb",
        {"A": reference_ca + shift},
        "B",
        ligand_coords + shift,
    )

    score = interaction.ligand_aligned_reference_pocket_box_occupancy(
        _eval_result("AB", current),
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3],
        reference_box_padding=0.0,
        min_box_size=5.0,
    )

    assert score["AB"] == 0.75


def test_ligand_aligned_reference_pocket_box_occupancy_is_available_to_loss_conditions(
    tmp_path,
):
    _init_protomers()
    reference_ca = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)
    ligand_coords = np.asarray(
        [[0.5, 0.5, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [8.0, 0.5, 0.0]],
        dtype=float,
    )

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference.pdb",
        {"A": reference_ca},
        "B",
        np.asarray([[0.5, 0.5, 0.0]], dtype=float),
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current.pdb",
        {"A": reference_ca + shift},
        "B",
        ligand_coords + shift,
    )

    loss = LossEngine(
        {
            "loss_terms": {
                "target_ligand_in_target_pocket": {
                    "_type": "ligand_aligned_reference_pocket_box_occupancy",
                    "reference_structure_path": str(reference),
                    "reference_structure_chain": "A",
                    "current_structure_chain": "A",
                    "reference_pocket_positions": [1, 2, 3],
                    "reference_box_padding": 0.0,
                    "min_box_size": 5.0,
                }
            },
            "expression_terms": {},
        }
    )
    loss.setup(
        weighted_score_expression="target_ligand_in_target_pocket",
        early_stopping=None,
        acceptance_threshold=None,
    )

    env = loss.score_context(_eval_result("AB", current))

    assert env["target_ligand_in_target_pocket"] == 0.75
    assert loss.eval_condition(env, "target_ligand_in_target_pocket < 0.8")


def test_ligand_aligned_reference_pocket_box_occupancy_supports_interface_chain_mapping(
    tmp_path,
):
    _init_protomers(interface=True)
    reference_a = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=float
    )
    reference_e = np.asarray(
        [[5.0, 0.0, 0.0], [6.5, 0.0, 0.0], [5.0, 1.5, 0.0]], dtype=float
    )
    shift = np.asarray([-4.0, 8.0, 1.5], dtype=float)

    reference = _write_protein_ligand_pdb(
        tmp_path / "reference_interface.pdb",
        {"A": reference_a, "E": reference_e},
        "B",
        np.asarray([[3.2, 0.5, 0.0]], dtype=float),
    )
    current = _write_protein_ligand_pdb(
        tmp_path / "current_interface.pdb",
        {
            "A": reference_a + shift,
            "B": reference_e + shift,
        },
        "C",
        np.asarray([[2.7, 0.2, 0.0], [4.2, 0.4, 0.0], [3.4, 1.2, 0.0]], dtype=float)
        + shift,
    )

    score = interaction.ligand_aligned_reference_pocket_box_occupancy(
        _eval_result("AEB", current),
        reference_structure_path=str(reference),
        reference_structure_chain={"A": "A", "E": "E"},
        current_structure_chain={"A": "A", "E": "B"},
        reference_pocket_positions_by_chain={"A": [2], "E": [1]},
        reference_box_padding=0.0,
        min_box_size=5.0,
    )

    assert score["AEB"] == 1.0


def test_ligand_aligned_reference_pocket_cb_alpha_shape_counts_fraction(tmp_path):
    _init_protomers()
    pocket_cb = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    shift = np.asarray([10.0, -3.0, 2.0], dtype=float)
    ligand_coords = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )

    reference = _write_cb_pocket_ligand_pdb(
        tmp_path / "reference_alpha.pdb",
        {"A": pocket_cb},
        "B",
        np.asarray([[0.5, 0.5, 0.5]], dtype=float),
    )
    current = _write_cb_pocket_ligand_pdb(
        tmp_path / "current_alpha.pdb",
        {"A": [(resname, coord + shift) for resname, coord in pocket_cb]},
        "B",
        ligand_coords + shift,
    )

    score = interaction.ligand_aligned_reference_pocket_cb_alpha_shape_occupancy(
        _eval_result("AB", current),
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        alpha_radius=10.0,
    )

    assert score["AB"] == 0.75


def test_ligand_aligned_reference_pocket_cb_alpha_shape_falls_back_to_ca_when_cb_missing(
    tmp_path,
):
    _init_protomers()
    pocket_cb = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("THR", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    ligand_coords = np.asarray([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=float)
    reference = _write_cb_pocket_ligand_pdb(
        tmp_path / "reference_alpha_missing_cb.pdb",
        {"A": pocket_cb},
        "B",
        ligand_coords,
    )
    current = _write_cb_pocket_ligand_pdb(
        tmp_path / "current_alpha_missing_cb.pdb",
        {"A": pocket_cb},
        "B",
        ligand_coords,
        omit_cb_positions_by_chain={"A": {1}},
    )

    score = interaction.ligand_aligned_reference_pocket_cb_alpha_shape_occupancy(
        _eval_result("AB", current),
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        alpha_radius=10.0,
    )

    assert score["AB"] == 1.0


def test_ligand_aligned_reference_pocket_cb_alpha_shape_rejects_invalid_radius():
    ligand_coords = np.asarray([[0.5, 0.5, 0.5]], dtype=float)
    pocket_cb = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="alpha_radius must be a positive finite value"):
        interaction._fraction_inside_cb_alpha_shape(
            ligand_coords,
            pocket_cb,
            alpha_radius=0.0,
        )

    with pytest.raises(ValueError, match="alpha_max_radius must be greater than"):
        interaction._fraction_inside_cb_alpha_shape(
            ligand_coords,
            pocket_cb,
            alpha_min_radius=6.0,
            alpha_max_radius=3.0,
        )


def test_ligand_aligned_reference_pocket_cb_alpha_shape_uses_qhull_fallback(monkeypatch):
    ligand_coords = np.asarray([[0.5, 0.5, 0.5]], dtype=float)
    pocket_cb = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    real_delaunay = interaction.Delaunay
    calls = []

    def flaky_delaunay(points, *args, **kwargs):
        calls.append(kwargs.get("qhull_options"))
        if len(calls) == 1:
            raise interaction.QhullError("forced initial failure")
        return real_delaunay(points, *args, **kwargs)

    monkeypatch.setattr(interaction, "Delaunay", flaky_delaunay)

    score = interaction._fraction_inside_cb_alpha_shape(
        ligand_coords,
        pocket_cb,
        alpha_radius=10.0,
    )

    assert score == 1.0
    assert calls == [None, "QJ"]


def test_ligand_aligned_reference_pocket_cb_alpha_shape_is_available_to_conditions(
    tmp_path,
):
    _init_protomers()
    pocket_cb = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    ligand_coords = np.asarray(
        [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [3.0, 3.0, 3.0]],
        dtype=float,
    )
    reference = _write_cb_pocket_ligand_pdb(
        tmp_path / "reference_alpha.pdb",
        {"A": pocket_cb},
        "B",
        np.asarray([[0.5, 0.5, 0.5]], dtype=float),
    )
    current = _write_cb_pocket_ligand_pdb(
        tmp_path / "current_alpha.pdb",
        {"A": pocket_cb},
        "B",
        ligand_coords,
    )

    loss = LossEngine(
        {
            "loss_terms": {
                "target_ligand_in_target_pocket": {
                    "_type": "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy",
                    "reference_structure_path": str(reference),
                    "reference_structure_chain": "A",
                    "current_structure_chain": "A",
                    "reference_pocket_positions": [1, 2, 3, 4],
                    "alpha_radius": 10.0,
                }
            },
            "expression_terms": {},
        }
    )
    loss.setup(
        weighted_score_expression="target_ligand_in_target_pocket",
        early_stopping=None,
        acceptance_threshold=None,
    )

    env = loss.score_context(_eval_result("AB", current))

    assert env["target_ligand_in_target_pocket"] == 0.75
    assert loss.eval_condition(env, "target_ligand_in_target_pocket < 0.8")


def test_ligand_aligned_reference_pocket_cb_alpha_shape_supports_interface_mapping(
    tmp_path,
):
    _init_protomers(interface=True)
    reference_a = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
    ]
    reference_e = [
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    shift = np.asarray([-4.0, 8.0, 1.5], dtype=float)
    ligand_coords = np.asarray(
        [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        dtype=float,
    )

    reference = _write_cb_pocket_ligand_pdb(
        tmp_path / "reference_interface_alpha.pdb",
        {"A": reference_a, "E": reference_e},
        "B",
        np.asarray([[0.5, 0.5, 0.5]], dtype=float),
    )
    current = _write_cb_pocket_ligand_pdb(
        tmp_path / "current_interface_alpha.pdb",
        {
            "A": [(resname, coord + shift) for resname, coord in reference_a],
            "B": [(resname, coord + shift) for resname, coord in reference_e],
        },
        "C",
        ligand_coords + shift,
    )

    score = interaction.ligand_aligned_reference_pocket_cb_alpha_shape_occupancy(
        _eval_result("AEB", current),
        reference_structure_path=str(reference),
        reference_structure_chain={"A": "A", "E": "E"},
        current_structure_chain={"A": "A", "E": "B"},
        reference_pocket_positions_by_chain={"A": [1, 2], "E": [1, 2]},
        alpha_radius=10.0,
    )

    assert score["AEB"] == 1.0


def test_ligand_protein_interface_occupancy_scores_dynamic_two_chain_interface(
    tmp_path,
):
    _init_dual_protein_ligand_protomers()
    interface_a = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
    ]
    interface_b = [
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    inside = _write_cb_pocket_ligand_pdb(
        tmp_path / "dynamic_interface_inside.pdb",
        {"A": interface_a, "B": interface_b},
        "C",
        np.asarray([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=float),
    )
    outside = _write_cb_pocket_ligand_pdb(
        tmp_path / "dynamic_interface_outside.pdb",
        {"A": interface_a, "B": interface_b},
        "C",
        np.asarray([[12.0, 12.0, 12.0]], dtype=float),
    )

    inside_score = interaction.ligand_protein_interface_occupancy(
        _eval_result("ABC", inside),
        interface_chains=["A", "B"],
        ligand_targets="C",
        interface_distance=8.0,
        interface_padding=0.0,
        min_box_size=5.0,
    )
    outside_score = interaction.ligand_protein_interface_occupancy(
        _eval_result("ABC", outside),
        interface_chains=["A", "B"],
        ligand_targets="C",
        interface_distance=8.0,
        interface_padding=0.0,
        min_box_size=5.0,
    )

    assert inside_score == {"ABC": 1.0}
    assert outside_score == {"ABC": 0.0}


def test_ligand_protein_interface_cb_alpha_shape_scores_dynamic_interface(
    tmp_path,
):
    _init_dual_protein_ligand_protomers()
    interface_a = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
    ]
    interface_b = [
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    inside = _write_cb_pocket_ligand_pdb(
        tmp_path / "dynamic_interface_alpha_inside.pdb",
        {"A": interface_a, "B": interface_b},
        "C",
        np.asarray([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=float),
    )
    outside = _write_cb_pocket_ligand_pdb(
        tmp_path / "dynamic_interface_alpha_outside.pdb",
        {"A": interface_a, "B": interface_b},
        "C",
        np.asarray([[12.0, 12.0, 12.0]], dtype=float),
    )

    inside_score = interaction.ligand_protein_interface_cb_alpha_shape_occupancy(
        _eval_result("ABC", inside),
        interface_chains=["A", "B"],
        ligand_targets="C",
        interface_distance=8.0,
        alpha_radius=10.0,
    )
    outside_score = interaction.ligand_protein_interface_cb_alpha_shape_occupancy(
        _eval_result("ABC", outside),
        interface_chains=["A", "B"],
        ligand_targets="C",
        interface_distance=8.0,
        alpha_radius=10.0,
    )

    assert inside_score == {"ABC": 1.0}
    assert outside_score == {"ABC": 0.0}


def test_dynamic_interface_occupancy_terms_are_available_to_loss_conditions(
    tmp_path,
):
    _init_dual_protein_ligand_protomers()
    interface_a = [
        ("ALA", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("VAL", np.asarray([4.0, 0.0, 0.0], dtype=float)),
    ]
    interface_b = [
        ("GLY", np.asarray([0.0, 4.0, 0.0], dtype=float)),
        ("SER", np.asarray([0.0, 0.0, 4.0], dtype=float)),
    ]
    current = _write_cb_pocket_ligand_pdb(
        tmp_path / "dynamic_interface_condition.pdb",
        {"A": interface_a, "B": interface_b},
        "C",
        np.asarray([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=float),
    )

    loss = LossEngine(
        {
            "loss_terms": {
                "ligand_in_interface": {
                    "_type": "ligand_protein_interface_occupancy",
                    "interface_chains": ["A", "B"],
                    "ligand_targets": "C",
                    "interface_distance": 8.0,
                    "interface_padding": 0.0,
                    "min_box_size": 5.0,
                },
                "ligand_in_interface_alpha": {
                    "_type": "ligand_protein_interface_cb_alpha_shape_occupancy",
                    "interface_chains": ["A", "B"],
                    "ligand_targets": "C",
                    "interface_distance": 8.0,
                    "alpha_radius": 10.0,
                },
            },
            "expression_terms": {},
        }
    )
    loss.setup(
        weighted_score_expression="ligand_in_interface + ligand_in_interface_alpha",
        early_stopping=None,
        acceptance_threshold=None,
    )

    env = loss.score_context(_eval_result("ABC", current))

    assert env["ligand_in_interface"] == 1.0
    assert env["ligand_in_interface_alpha"] == 1.0
    assert loss.eval_condition(env, "ligand_in_interface >= 1.0")
