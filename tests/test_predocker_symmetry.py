from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli
import run as run_mod
from astrolabe.core import global_config
from astrolabe.core.repeat_symmetry import (
    apply_transform,
    build_internal_repeat_movable_groups,
    build_internal_repeat_residue_groups,
    canonicalize_exact_cyclic_cn,
    collect_atom_bundle,
    dihedral_group_transforms_canonical,
    harmonize_atom_bundles,
    kabsch_transform,
    load_structure,
    protein_residues,
    rotation_angle_degrees,
    rotation_axis,
)
from astrolabe.mutators.wrappers.ligandmpnn_utils import parse_PDB
from astrolabe.structure_processors.base import initialize_predockers, run_nested_predocker
from astrolabe.structure_processors import symmetry as symmetry_mod
from astrolabe.structure_processors.symmetry import (
    SymmetryRescueProcessor,
    _dihedral_pair_contact_selection_score,
    _project_cn_bundle_layout,
)


def _setup_protomers(tmp_path: Path, seqs: dict[str, str]) -> None:
    global_config.reset()
    args = SimpleNamespace(
        exclude_AA=[],
        initial_temperature=0.05,
        out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw",
        use_metropolis_acceptance=False,
        enable_pyrosetta=False,
        skip_complex_poses=True,
        protenix_constraint=None,
        binding_site=None,
        cyclization=None,
        ligand_resnames=None,
        seqs=seqs,
        types={proto: "protein" for proto in seqs},
        homologies={proto: None for proto in seqs},
    )
    run_mod.initialize_global_config(args)
    run_mod.setup_protomers(args)


def _write_ca_pdb(path: Path, chain_coords: dict[str, np.ndarray]) -> Path:
    lines: list[str] = []
    serial = 1
    for chain_id, coords in chain_coords.items():
        for res_i, xyz in enumerate(np.asarray(coords, dtype=float), start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_backbone_pdb(
    path: Path,
    chain_coords: dict[str, np.ndarray],
    *,
    add_oxt_last: bool = False,
    oxt_chains: set[str] | None = None,
) -> Path:
    lines: list[str] = []
    serial = 1
    atom_offsets = {
        "N": np.array([-1.2, 0.0, 0.0], dtype=float),
        "CA": np.array([0.0, 0.0, 0.0], dtype=float),
        "C": np.array([1.2, 0.0, 0.0], dtype=float),
        "O": np.array([1.9, 0.4, 0.0], dtype=float),
    }
    for chain_id, coords in chain_coords.items():
        coord_array = np.asarray(coords, dtype=float)
        for res_i, ca_xyz in enumerate(coord_array, start=1):
            for atom_name, offset in atom_offsets.items():
                xyz = ca_xyz + offset
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} ALA {chain_id}{res_i:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           {element:>2s}"
                )
                serial += 1
            if add_oxt_last and res_i == len(coord_array) and (
                oxt_chains is None or chain_id in oxt_chains
            ):
                xyz = ca_xyz + np.array([2.1, -0.4, 0.0], dtype=float)
                lines.append(
                    f"ATOM  {serial:5d} {'OXT':>4s} ALA {chain_id}{res_i:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           {'O':>2s}"
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_chain_coords(path: Path, chain_id: str) -> np.ndarray:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("predocker", str(path))
    coords: list[np.ndarray] = []
    for model in structure:
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                coords.append(np.asarray(residue["CA"].coord, dtype=float))
        break
    return np.asarray(coords, dtype=float)


def _load_atom_coord(path: Path, chain_id: str, residue_number: int, atom_name: str) -> np.ndarray:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("predocker", str(path))
    residue = next(structure.get_models())[chain_id][(" ", int(residue_number), " ")]
    return np.asarray(residue[atom_name].coord, dtype=float)


def _load_case_manifest(case_dir: Path) -> dict:
    return json.loads((case_dir / "03_manifest.json").read_text(encoding="utf-8"))


def _expected_atom_from_group_transform(
    input_path: Path,
    output_path: Path,
    *,
    chain_id: str,
    groups: list[np.ndarray],
    group_index: int,
    residue_number: int,
    atom_name: str,
) -> np.ndarray:
    input_residues = protein_residues(load_structure(input_path), chain_id)
    output_residues = protein_residues(load_structure(output_path), chain_id)
    input_common = harmonize_atom_bundles(
        [collect_atom_bundle(input_residues, group) for group in groups]
    )
    output_common = harmonize_atom_bundles(
        [collect_atom_bundle(output_residues, group) for group in groups]
    )
    corr_R, corr_t = kabsch_transform(
        input_common[group_index].coords,
        output_common[group_index].coords,
    )
    input_atom = _load_atom_coord(
        input_path,
        chain_id,
        residue_number=residue_number,
        atom_name=atom_name,
    )
    return apply_transform(np.asarray([input_atom], dtype=float), corr_R, corr_t)[0]


def _expected_atom_from_cyclic_chain_transform(
    input_path: Path,
    output_path: Path,
    *,
    chain_ids: list[str],
    chain_index: int,
    residue_number: int,
    atom_name: str,
) -> np.ndarray:
    input_structure = load_structure(input_path)
    output_structure = load_structure(output_path)
    input_common = harmonize_atom_bundles(
        [
            collect_atom_bundle(
                protein_residues(input_structure, chain_id),
                np.arange(
                    len(protein_residues(input_structure, chain_id)),
                    dtype=int,
                ),
            )
            for chain_id in chain_ids
        ]
    )
    output_common = harmonize_atom_bundles(
        [
            collect_atom_bundle(
                protein_residues(output_structure, chain_id),
                np.arange(
                    len(protein_residues(output_structure, chain_id)),
                    dtype=int,
                ),
            )
            for chain_id in chain_ids
        ]
    )
    corr_R, corr_t = kabsch_transform(
        input_common[chain_index].coords,
        output_common[chain_index].coords,
    )
    input_atom = _load_atom_coord(
        input_path,
        chain_ids[chain_index],
        residue_number=residue_number,
        atom_name=atom_name,
    )
    return apply_transform(np.asarray([input_atom], dtype=float), corr_R, corr_t)[0]


def _align_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P0 = np.asarray(P, dtype=float)
    Q0 = np.asarray(Q, dtype=float)
    R, t = kabsch_transform(P0, Q0)
    aligned = apply_transform(P0, R, t)
    return float(np.sqrt(np.mean(np.sum((aligned - Q0) ** 2, axis=1))))


def _rotation_z(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _assert_consecutive_cn(coord_sets: list[np.ndarray], target_rotation_deg: float) -> None:
    for idx in range(len(coord_sets)):
        left = coord_sets[idx]
        right = coord_sets[(idx + 1) % len(coord_sets)]
        R, t = kabsch_transform(left, right)
        axis = rotation_axis(R, t)
        assert abs(rotation_angle_degrees(R) - target_rotation_deg) < 5e-2
        assert np.linalg.norm(np.cross(axis, np.array([0.0, 0.0, 1.0], dtype=float))) < 5e-3
        assert abs(float(np.dot(np.asarray(t, dtype=float), axis))) < 5e-3


def _assert_d3_group_layout(coord_sets: list[np.ndarray], atol: float = 2e-2) -> None:
    transforms = dihedral_group_transforms_canonical(3)
    reference = coord_sets[0]
    for coords, (R, t) in zip(coord_sets, transforms, strict=True):
        expected = apply_transform(reference, R, t)
        assert float(np.sqrt(np.mean(np.sum((expected - coords) ** 2, axis=1)))) < atol


def test_tandem_repeat_canonicalizer_makes_repeat_units_identical(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "GGG@cap AAA@u u TTT@tail"})
    cap = np.array([[-3.0, 0.0, 0.0], [-2.1, 0.3, 0.0], [-1.2, 0.1, 0.0]], dtype=float)
    unit = np.array([[0.0, 0.0, 0.0], [0.9, 0.4, 0.0], [1.9, 0.1, 0.0]], dtype=float)
    rotation = _rotation_z(25.0)
    offset = np.array([4.5, 1.2, 0.0], dtype=float)
    chain_a = np.vstack([cap, unit, unit @ rotation + offset, cap @ rotation + offset])
    input_pdb = _write_ca_pdb(tmp_path / "repeat_input.pdb", {"A": chain_a})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="tandem_repeat",
        oligo="A",
        repeat_canonicalizer_mode="translation",
    )

    output_pdb = Path(predocker.run(str(input_pdb)))
    coords = _load_chain_coords(output_pdb, "A")

    assert _align_rmsd(chain_a[:6], coords[:6]) < 1e-3
    assert _align_rmsd(chain_a[6:], coords[6:]) < 1e-3
    assert _align_rmsd(coords[3:6], coords[6:9]) < 1e-3


def test_heterooligomer_shared_core_canonicalizer_applies_rigid_transform_to_whole_protomer(
    tmp_path: Path,
) -> None:
    _setup_protomers(
        tmp_path,
        {
            "A": "AAA@core_1 CCC@itf_a DDD@core_2",
            "B": "core_1 EEE@itf_b core_2",
        },
    )
    chain_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
            [4.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    chain_b = np.array(
        [
            [20.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [25.0, 9.0, 3.0],
            [26.0, 8.0, 4.0],
            [27.0, 9.0, 4.0],
            [14.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    input_pdb = _write_ca_pdb(tmp_path / "hetero_input.pdb", {"A": chain_a, "B": chain_b})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="heterooligomer_shared_core",
        oligo="AB",
    )

    output_pdb = Path(predocker.run(str(input_pdb)))
    out_a = _load_chain_coords(output_pdb, "A")
    out_b = _load_chain_coords(output_pdb, "B")

    assert _align_rmsd(chain_a, out_a) < 2e-2
    assert _align_rmsd(chain_b, out_b) < 2e-2
    assert _align_rmsd(
        np.vstack([out_a[:3], out_a[6:9]]),
        np.vstack([out_b[:3], out_b[6:9]]),
    ) < 2e-2


def test_pseudocycle_canonicalizer_accepts_terminal_oxt_in_last_repeat(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.6, 0.4, 0.0],
            [3.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    chain_a = np.vstack([base_unit @ _rotation_z(90.0 * i) for i in range(4)])
    input_pdb = _write_backbone_pdb(
        tmp_path / "pseudo_oxt_input.pdb",
        {"A": chain_a},
        add_oxt_last=True,
    )
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
    )

    output_path = Path(predocker.run(str(input_pdb)))

    assert output_path.exists()
    assert _load_chain_coords(output_path, "A").shape == chain_a.shape


def test_symmetry_canonicalizer_returns_existing_path_when_cache_moves_output(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u"})
    chain_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.4, 0.0],
            [1.9, 0.1, 0.0],
            [4.5, 1.2, 0.0],
            [5.147, 1.943, 0.0],
            [6.263, 2.097, 0.0],
        ],
        dtype=float,
    )
    input_pdb = _write_backbone_pdb(tmp_path / "cache_move_input.pdb", {"A": chain_a})
    predocker = SymmetryRescueProcessor(
        mode="tandem_repeat",
        oligo="A",
        repeat_canonicalizer_mode="translation",
    )

    output_path = Path(predocker.run(str(input_pdb)))

    assert output_path.exists()


def test_pseudocycle_rfdiffusion_backend_writes_case_artifacts_and_exact_cycle(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.3],
            [2.7, 0.4, 0.1],
            [3.3, -0.1, -0.2],
        ],
        dtype=float,
    )
    chain_a = np.vstack(
        [
            base_unit @ _rotation_z(87.0),
            base_unit @ _rotation_z(181.0) + np.array([0.4, -0.2, 0.5], dtype=float),
            base_unit @ _rotation_z(268.0) + np.array([-0.2, 0.3, -0.4], dtype=float),
            base_unit @ _rotation_z(357.0) + np.array([0.1, -0.4, 0.2], dtype=float),
        ]
    )
    input_pdb = _write_backbone_pdb(tmp_path / "pseudo_rfdiff_input.pdb", {"A": chain_a})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rfdiffusion",
    )

    output_path = Path(predocker.run(str(input_pdb)))

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    assert output_path == case_dir / "02_redock_result.pdb"
    assert (case_dir / "00_initial_structure.pdb").exists()
    assert (case_dir / "01_average_asu_structure.pdb").exists()
    assert (case_dir / "02_redock_result.pdb").exists()
    manifest = _load_case_manifest(case_dir)
    assert manifest["backend"] == "rfdiffusion"
    assert manifest["mode"] == "pseudocycle"
    assert manifest["order"] == 4
    assert manifest["asu_kind"] == "repeat_unit"

    coords = _load_chain_coords(output_path, "A")
    coord_sets = [coords[i * 3 : (i + 1) * 3] for i in range(4)]
    _assert_consecutive_cn(coord_sets, 90.0)
    avg_coords = _load_chain_coords(case_dir / "01_average_asu_structure.pdb", "A")
    assert avg_coords.shape == (3, 3)


def test_pseudocycle_rfdiffusion_backend_moves_non_common_terminal_atoms_with_repeat_transform(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.3],
            [2.7, 0.4, 0.1],
            [3.3, -0.1, -0.2],
        ],
        dtype=float,
    )
    chain_a = np.vstack(
        [
            base_unit @ _rotation_z(87.0),
            base_unit @ _rotation_z(181.0) + np.array([0.4, -0.2, 0.5], dtype=float),
            base_unit @ _rotation_z(268.0) + np.array([-0.2, 0.3, -0.4], dtype=float),
            base_unit @ _rotation_z(357.0) + np.array([0.1, -0.4, 0.2], dtype=float),
        ]
    )
    input_pdb = _write_backbone_pdb(
        tmp_path / "pseudo_rfdiff_oxt_input.pdb",
        {"A": chain_a},
        add_oxt_last=True,
    )
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rfdiffusion",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    movable_groups = build_internal_repeat_movable_groups("A")
    input_oxt = _load_atom_coord(input_pdb, "A", residue_number=len(chain_a), atom_name="OXT")
    output_oxt = _load_atom_coord(output_path, "A", residue_number=len(chain_a), atom_name="OXT")
    expected_oxt = _expected_atom_from_group_transform(
        input_pdb,
        output_path,
        chain_id="A",
        groups=movable_groups,
        group_index=-1,
        residue_number=len(chain_a),
        atom_name="OXT",
    )

    assert np.linalg.norm(output_oxt - input_oxt) > 1e-2
    assert np.allclose(output_oxt, expected_oxt, atol=2e-2)


@pytest.mark.parametrize("backend", ["rfdiffusion", "rpxdock"])
def test_exact_pseudocycle_backends_reject_translation_repeat_mode(
    tmp_path: Path,
    backend: str,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u"})
    with pytest.raises(
        ValueError,
        match="repeat_canonicalizer_mode='translation' is not supported for exact pseudocycle backends",
    ):
        SymmetryRescueProcessor(
            cache_enabled=False,
            mode="pseudocycle",
            oligo="A",
            included_protomer="A",
            backend=backend,
            repeat_canonicalizer_mode="translation",
        )


def test_cyclic_oligomer_rfdiffusion_backend_handles_repeated_protomer_names(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "B": protomer @ _rotation_z(92.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "C": protomer @ _rotation_z(181.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "D": protomer @ _rotation_z(269.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_input.pdb", chain_coords)
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rfdiffusion",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    out_sets = [_load_chain_coords(output_path, chain_id) for chain_id in ["A", "B", "C", "D"]]
    _assert_consecutive_cn(out_sets, 90.0)

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    manifest = _load_case_manifest(case_dirs[0])
    assert manifest["mode"] == "cyclic_oligomer"
    assert manifest["asu_kind"] == "protomer"


def test_cyclic_oligomer_rfdiffusion_backend_uses_structure_chain_order(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [4.8, 0.0, 0.0],
            [5.5, 0.5, 0.0],
            [6.2, 0.1, 0.0],
            [7.0, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "X": protomer @ _rotation_z(5.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "Y": protomer @ _rotation_z(95.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "Z": protomer @ _rotation_z(185.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "W": protomer @ _rotation_z(275.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_nondefault_chain_input.pdb", chain_coords)
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rfdiffusion",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    out_sets = [_load_chain_coords(output_path, chain_id) for chain_id in ["X", "Y", "Z", "W"]]
    _assert_consecutive_cn(out_sets, 90.0)


def test_dihedral_oligomer_rfdiffusion_backend_handles_d3_layout(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [6.0, 0.0, 2.0],
            [6.8, 0.4, 2.2],
            [7.5, -0.2, 1.8],
            [8.1, 0.1, 2.1],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    perturb = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.1, -0.1, 0.1], dtype=float),
        np.array([-0.1, 0.1, -0.1], dtype=float),
        np.array([0.2, 0.0, -0.1], dtype=float),
        np.array([-0.2, 0.1, 0.0], dtype=float),
        np.array([0.0, -0.2, 0.1], dtype=float),
    ]
    chain_coords = {
        chain_id: apply_transform(base, R, t) + perturb[idx]
        for idx, (chain_id, (R, t)) in enumerate(zip("ABCDEF", transforms, strict=True))
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_input.pdb", chain_coords)
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rfdiffusion",
        order=3,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    out_sets = [_load_chain_coords(output_path, chain_id) for chain_id in "ABCDEF"]
    _assert_d3_group_layout(out_sets)
    manifest = _load_case_manifest(output_path.parent)
    assert manifest["mode"] == "dihedral_oligomer"
    assert manifest["point_group"] == "D3"
    assert manifest["subunit_count"] == 6
    assert manifest["asu_kind"] == "protomer"


def test_rpxdock_backend_requires_vendored_dependency(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array([[2.0, 0.0, 0.0], [2.6, 0.4, 0.0], [3.1, 0.0, 0.0]], dtype=float)
    chain_a = np.vstack([base_unit @ _rotation_z(90.0 * i) for i in range(4)])
    input_pdb = _write_backbone_pdb(tmp_path / "pseudo_rpxdock_input.pdb", {"A": chain_a})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        rpxdock_vendor_repo=tmp_path / "missing_rpxdock_vendor",
        rpxdock_failure_fallback="raise",
    )

    with pytest.raises((ImportError, RuntimeError, FileNotFoundError)):
        predocker.run(str(input_pdb))


def test_exact_backends_reject_tandem_repeat_mode(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u"})
    chain_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.4, 0.0],
            [1.9, 0.1, 0.0],
            [4.5, 1.2, 0.0],
            [5.1, 1.9, 0.0],
            [6.2, 2.1, 0.0],
        ],
        dtype=float,
    )
    input_pdb = _write_backbone_pdb(tmp_path / "tandem_exact_backend_input.pdb", {"A": chain_a})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="tandem_repeat",
        oligo="A",
        backend="rfdiffusion",
    )

    with pytest.raises(ValueError):
        predocker.run(str(input_pdb))


def test_exact_backend_cache_key_includes_backend_and_order(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    input_pdb = _write_backbone_pdb(
        tmp_path / "cache_key_input.pdb",
        {"A": np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]], dtype=float)},
    )
    predocker_a = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rfdiffusion",
        order=4,
    )
    predocker_b = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        order=4,
    )

    key_a = predocker_a._generate_cache_key(str(input_pdb))
    key_b = predocker_b._generate_cache_key(str(input_pdb))

    assert key_a != key_b


def test_rpxdock_cache_key_and_case_dir_include_runtime_parameters(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    input_pdb = _write_backbone_pdb(
        tmp_path / "rpxdock_cache_key_input.pdb",
        {"A": np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]], dtype=float)},
    )
    predocker_a = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
        rpxdock_python_bin="/envs/rpxdock_a/bin/python",
        rpxdock_vendor_repo="/vendor/rpxdock_a",
        rpxdock_timeout_s=300,
        rpxdock_extra_args=("--beam_size", "5000"),
    )
    predocker_b = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
        rpxdock_python_bin="/envs/rpxdock_b/bin/python",
        rpxdock_vendor_repo="/vendor/rpxdock_b",
        rpxdock_timeout_s=900,
        rpxdock_extra_args=("--beam_size", "9000"),
    )

    key_a = predocker_a._generate_cache_key(str(input_pdb))
    key_b = predocker_b._generate_cache_key(str(input_pdb))
    case_dir_a = predocker_a._case_dir(str(input_pdb))
    case_dir_b = predocker_b._case_dir(str(input_pdb))

    assert key_a != key_b
    assert case_dir_a != case_dir_b


def test_project_cn_bundle_layout_aligns_before_averaging() -> None:
    base = np.array(
        [
            [4.0, 0.0, 0.0],
            [4.8, 0.4, 0.0],
            [5.5, -0.2, 0.0],
        ],
        dtype=float,
    )
    bundles = []
    ca_sets = []
    perturb_rots = [_rotation_z(0.0), _rotation_z(4.0), _rotation_z(-3.0), _rotation_z(2.0)]
    perturb_shifts = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.15, -0.08, 0.0], dtype=float),
        np.array([-0.10, 0.05, 0.0], dtype=float),
        np.array([0.06, 0.07, 0.0], dtype=float),
    ]
    for idx, (Rpert, tpert) in enumerate(zip(perturb_rots, perturb_shifts, strict=True)):
        unit = (base @ Rpert) + tpert
        placed = unit @ _rotation_z(90.0 * idx)
        bundles.append(placed)
        ca_sets.append(placed)

    naive = np.mean(
        np.stack(
            [coords @ _rotation_z(-90.0 * idx) for idx, coords in enumerate(bundles)],
            axis=0,
        ),
        axis=0,
    )
    aligned_avg, _, _, _ = _project_cn_bundle_layout(ca_sets, bundles)
    naive_err = np.mean(
        [_align_rmsd(coords @ _rotation_z(-90.0 * idx), naive) for idx, coords in enumerate(bundles)]
    )
    aligned_err = np.mean(
        [_align_rmsd(coords @ _rotation_z(-90.0 * idx), aligned_avg) for idx, coords in enumerate(bundles)]
    )

    assert aligned_err < naive_err


def test_canonicalize_exact_cyclic_cn_places_reference_centroid_on_positive_x_axis() -> None:
    base = np.array(
        [
            [5.0, 0.0, 1.0],
            [5.7, 0.5, 1.2],
            [6.3, 0.1, 0.8],
        ],
        dtype=float,
    )
    coord_sets = []
    for angle_deg, shift in [
        (4.0, np.array([0.3, -0.2, 0.4], dtype=float)),
        (92.0, np.array([0.2, -0.1, -0.3], dtype=float)),
        (181.0, np.array([-0.3, 0.2, 0.4], dtype=float)),
        (269.0, np.array([0.1, 0.3, -0.2], dtype=float)),
    ]:
        coord_sets.append(base @ _rotation_z(angle_deg) + shift)

    canonical = canonicalize_exact_cyclic_cn(coord_sets)
    first_centroid = np.mean(coord_sets[0], axis=0, keepdims=True)
    first_centroid_can = apply_transform(
        first_centroid,
        canonical.to_canonical_rotation,
        canonical.to_canonical_translation,
    )[0]

    assert first_centroid_can[0] > 0.0
    assert abs(first_centroid_can[1]) < 1e-6


def test_cyclic_oligomer_rpxdock_backend_recenters_dock_input_to_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "B": protomer @ _rotation_z(92.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "C": protomer @ _rotation_z(181.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "D": protomer @ _rotation_z(269.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        dummy_output = tmp_path / "fake_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([6.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake.stdout.log",
            stderr_path=tmp_path / "fake.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
    )

    predocker.run(str(input_pdb))

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    avg_coords = _load_chain_coords(case_dir / "01_average_asu_structure.pdb", "A")
    dock_input_coords = _load_chain_coords(case_dir / "01_dock_input_structure.pdb", "A")

    assert np.linalg.norm(np.mean(avg_coords, axis=0)[:2]) > 1.0
    assert np.linalg.norm(np.mean(dock_input_coords, axis=0)) < 1e-6


def test_dihedral_oligomer_rpxdock_backend_preserves_radius_height_and_uses_dihedral_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [6.0, 0.0, 2.0],
            [6.8, 0.4, 2.2],
            [7.5, -0.2, 1.8],
            [8.1, 0.1, 2.1],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    chain_coords = {
        chain_id: apply_transform(base, R, t)
        for chain_id, (R, t) in zip("ABCDEF", transforms, strict=True)
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_rpxdock_input.pdb", chain_coords)
    seen: dict[str, object] = {}

    def _fake_run_local_cyclic_dock(monomer_path, config):
        seen["monomer_path"] = Path(monomer_path)
        seen["architecture"] = config.architecture
        seen["cart_bounds"] = config.cart_bounds
        seen["limit_rotation_to_z"] = config.limit_rotation_to_z
        dummy_output = tmp_path / "fake_d3_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.zeros(3, dtype=float),
            command=("fake-rpxdock", "--architecture", str(config.architecture)),
            stdout_path=tmp_path / "fake_d3.stdout.log",
            stderr_path=tmp_path / "fake_d3.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rpxdock",
        order=3,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    dock_input_coords = _load_chain_coords(case_dir / "01_dock_input_structure.pdb", "A")
    dock_input_centroid = np.mean(dock_input_coords, axis=0)
    dock_input_principal = symmetry_mod._principal_axis(dock_input_coords)
    out_sets = [_load_chain_coords(output_path, chain_id) for chain_id in "ABCDEF"]
    manifest = _load_case_manifest(case_dir)

    assert seen["architecture"] == "D3_3"
    assert seen["limit_rotation_to_z"] is False
    assert seen["monomer_path"] == case_dir / "01_dock_input_structure.pdb"
    assert seen["cart_bounds"] == (0.0, 100.0)
    assert np.linalg.norm(dock_input_centroid[:2]) > 1.0
    assert abs(float(dock_input_centroid[2])) > 1.0
    assert abs(float(dock_input_principal[2])) > 0.99
    _assert_d3_group_layout(out_sets, atol=3e-2)
    assert manifest["rpxdock_architecture"] == "D3_3"
    assert manifest["rpxdock_axis"]["axis_source"] == "monomer_pca"
    assert abs(manifest["rpxdock_axis"]["canonical_centroid"][1]) < 1e-6
    assert manifest["rpxdock_fit"]["fitted_radius"] > 1.0
    assert manifest["rpxdock_fit"]["fitted_z_height"] > 1.0


def test_dihedral_oligomer_rpxdock_backend_keeps_accepted_projected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [6.0, 0.0, 2.0],
            [6.8, 0.4, 2.2],
            [7.5, -0.2, 1.8],
            [8.1, 0.1, 2.1],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    chain_coords = {
        chain_id: apply_transform(base, R, t)
        for chain_id, (R, t) in zip("ABCDEF", transforms, strict=True)
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_rpxdock_dispersed_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(monomer_path, config):
        dummy_output = tmp_path / "fake_dispersed_d3_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        dock_input_center = np.mean(_load_chain_coords(Path(monomer_path), "A"), axis=0)
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([2.0, 0.0, 5.0], dtype=float) - dock_input_center,
            command=("fake-rpxdock", "--architecture", str(config.architecture)),
            stdout_path=tmp_path / "fake_dispersed_d3.stdout.log",
            stderr_path=tmp_path / "fake_dispersed_d3.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rpxdock",
        order=3,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    out_sets = [_load_chain_coords(output_path, chain_id) for chain_id in "ABCDEF"]
    manifest = _load_case_manifest(output_path.parent)
    postprocess = manifest["rpxdock_postprocess"]["dihedral_radial_compaction"][0]

    _assert_d3_group_layout(out_sets, atol=3e-2)
    assert postprocess["applied"] is False
    assert postprocess["failed"] is False
    assert postprocess["scale"] == 1.0
    assert postprocess["after"]["heavy_contact_count"] > 0.0
    assert postprocess["after"] == postprocess["before"]
    assert postprocess["after"]["by_pair_class"]["cn_same_layer"]["heavy_contact_count"] > 0.0
    assert postprocess["after"]["by_pair_class"]["c2_same_index"]["heavy_contact_count"] > 0.0
    assert 2.6 <= postprocess["after"]["min_heavy_distance"] <= 5.5


def test_dihedral_oligomer_rpxdock_backend_leaves_candidate_when_compaction_has_no_safe_solution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [6.0, 0.0, 20.0],
            [6.8, 0.4, 20.2],
            [7.5, -0.2, 19.8],
            [8.1, 0.1, 20.1],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    chain_coords = {
        chain_id: apply_transform(base, R, t)
        for chain_id, (R, t) in zip("ABCDEF", transforms, strict=True)
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_rpxdock_no_safe_compaction_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(monomer_path, config):
        dummy_output = tmp_path / "fake_no_safe_d3_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([24.0, 0.0, 20.0], dtype=float),
            command=("fake-rpxdock", "--architecture", str(config.architecture)),
            stdout_path=tmp_path / "fake_no_safe_d3.stdout.log",
            stderr_path=tmp_path / "fake_no_safe_d3.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rpxdock",
        order=3,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    out_chain_a = _load_chain_coords(output_path, "A")
    dock_input_chain_a = _load_chain_coords(
        output_path.parent / "01_dock_input_structure.pdb",
        "A",
    )
    manifest = _load_case_manifest(output_path.parent)
    postprocess = manifest["rpxdock_postprocess"]["dihedral_radial_compaction"][0]

    assert postprocess["applied"] is False
    assert postprocess["failed"] is True
    assert postprocess["scale"] == 1.0
    assert postprocess["z_scale"] == 1.0
    assert postprocess["after"] == postprocess["before"]
    assert not np.allclose(out_chain_a, _load_chain_coords(input_pdb, "A"))
    assert np.allclose(
        np.mean(out_chain_a, axis=0),
        np.mean(dock_input_chain_a, axis=0) + np.array([24.0, 0.0, 20.0], dtype=float),
        atol=1e-6,
    )


def test_dihedral_oligomer_rpxdock_backend_selects_primary_by_c2_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [6.0, 0.0, 2.0],
            [6.8, 0.4, 2.2],
            [7.5, -0.2, 1.8],
            [8.1, 0.1, 2.1],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    chain_coords = {
        chain_id: apply_transform(base, R, t)
        for chain_id, (R, t) in zip("ABCDEF", transforms, strict=True)
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_rpxdock_select_c2_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(monomer_path, config):
        dummy_rank0 = tmp_path / "fake_select_rank0.pdb"
        dummy_rank1 = tmp_path / "fake_select_rank1.pdb"
        dummy_rank0.write_text("END\n", encoding="utf-8")
        dummy_rank1.write_text("END\n", encoding="utf-8")
        dock_input_center = np.mean(_load_chain_coords(Path(monomer_path), "A"), axis=0)
        rank0_translation = np.array([24.0, 0.0, 20.0], dtype=float) - dock_input_center
        rank1_translation = np.array([2.0, 0.0, 2.0], dtype=float) - dock_input_center
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_rank0,
            rotation=np.eye(3, dtype=float),
            translation=rank0_translation,
            command=("fake-rpxdock", "--architecture", str(config.architecture)),
            stdout_path=tmp_path / "fake_select.stdout.log",
            stderr_path=tmp_path / "fake_select.stderr.log",
            candidates=(
                rpxdock_wrapper.RpxDockCandidate(
                    rank=0,
                    docked_structure_path=dummy_rank0,
                    rotation=np.eye(3, dtype=float),
                    translation=rank0_translation,
                ),
                rpxdock_wrapper.RpxDockCandidate(
                    rank=1,
                    docked_structure_path=dummy_rank1,
                    rotation=np.eye(3, dtype=float),
                    translation=rank1_translation,
                ),
            ),
        )

    compactions = iter(
        [
            {
                "applied": False,
                "failed": True,
                "scale": 1.0,
                "z_scale": 1.0,
                "before": {
                    "min_heavy_distance": 20.0,
                    "heavy_contact_count": 30.0,
                    "by_pair_class": {
                        "cn_same_layer": {"min_heavy_distance": 4.0, "heavy_contact_count": 30.0},
                        "c2_same_index": {"min_heavy_distance": 20.0, "heavy_contact_count": 0.0},
                        "cross_other": {"min_heavy_distance": 20.0, "heavy_contact_count": 0.0},
                    },
                },
                "after": {
                    "min_heavy_distance": 20.0,
                    "heavy_contact_count": 30.0,
                    "by_pair_class": {
                        "cn_same_layer": {"min_heavy_distance": 4.0, "heavy_contact_count": 30.0},
                        "c2_same_index": {"min_heavy_distance": 20.0, "heavy_contact_count": 0.0},
                        "cross_other": {"min_heavy_distance": 20.0, "heavy_contact_count": 0.0},
                    },
                },
            },
            {
                "applied": True,
                "failed": False,
                "scale": 0.9,
                "z_scale": 0.9,
                "before": {
                    "min_heavy_distance": 8.0,
                    "heavy_contact_count": 40.0,
                    "by_pair_class": {
                        "cn_same_layer": {"min_heavy_distance": 4.0, "heavy_contact_count": 40.0},
                        "c2_same_index": {"min_heavy_distance": 8.0, "heavy_contact_count": 0.0},
                        "cross_other": {"min_heavy_distance": 8.0, "heavy_contact_count": 0.0},
                    },
                },
                "after": {
                    "min_heavy_distance": 3.2,
                    "heavy_contact_count": 52.0,
                    "by_pair_class": {
                        "cn_same_layer": {"min_heavy_distance": 3.2, "heavy_contact_count": 40.0},
                        "c2_same_index": {"min_heavy_distance": 3.8, "heavy_contact_count": 12.0},
                        "cross_other": {"min_heavy_distance": 8.0, "heavy_contact_count": 0.0},
                    },
                },
            },
        ]
    )

    def _fake_compaction(*_args, **_kwargs):
        return next(compactions)

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)
    monkeypatch.setattr(symmetry_mod, "_compact_projected_bundles_if_dispersed", _fake_compaction)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="dihedral_oligomer",
        oligo="AAAAAA",
        backend="rpxdock",
        order=3,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    out_chain_a = _load_chain_coords(output_path, "A")
    manifest = _load_case_manifest(output_path.parent)
    postprocess = manifest["rpxdock_postprocess"]

    assert postprocess["selected_primary_rank"] == 1
    assert postprocess["selected_primary_candidate_index"] == 1
    assert postprocess["selected_primary_contact_summary"]["c2_same_index_contacts"] == 12
    assert postprocess["selected_primary_contact_summary"]["cn_same_layer_contacts"] == 40
    assert [item["rank"] for item in postprocess["dihedral_radial_compaction"]] == [0, 1]
    assert Path(manifest["output_structure"]).name == "02_redock_result.pdb"
    assert Path(manifest["output_candidates"][1]).name == "02_redock_result__top0.pdb"
    assert np.allclose(
        np.mean(out_chain_a, axis=0),
        np.array([2.0, 0.0, 2.0], dtype=float),
        atol=1e-3,
    )


def test_dihedral_rpxdock_primary_selection_prioritizes_c2_over_cn_only() -> None:
    c2_score = _dihedral_pair_contact_selection_score(
        {
            "c2_same_index_contacts": 4,
            "cn_same_layer_contacts": 0,
            "heavy_contact_count": 4,
            "failed": False,
            "contact_target": 12,
        },
        rank=1,
        candidate_index=1,
    )
    cn_only_score = _dihedral_pair_contact_selection_score(
        {
            "c2_same_index_contacts": 0,
            "cn_same_layer_contacts": 40,
            "heavy_contact_count": 40,
            "failed": False,
            "contact_target": 12,
        },
        rank=0,
        candidate_index=0,
    )

    assert c2_score > cn_only_score


def test_dihedral_chain_metrics_count_contacts_by_pair_class(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAAAA"})
    base = np.array(
        [
            [5.0, 0.0, 1.6],
            [5.8, 0.3, 1.7],
            [6.5, -0.1, 1.5],
        ],
        dtype=float,
    )
    transforms = dihedral_group_transforms_canonical(3)
    compacted_base = symmetry_mod._radially_scaled_coords(base, 0.85, z_scale=0.75)
    chain_coords = {
        chain_id: apply_transform(compacted_base, R, t)
        for chain_id, (R, t) in zip("ABCDEF", transforms, strict=True)
    }
    input_pdb = _write_backbone_pdb(tmp_path / "d3_pair_metrics_input.pdb", chain_coords)
    structure = load_structure(input_pdb)
    bundles = []
    for chain_id in "ABCDEF":
        residues = protein_residues(structure, chain_id)
        bundles.append(collect_atom_bundle(residues, np.arange(len(residues), dtype=int)))

    metrics = symmetry_mod._bundle_chain_metrics(
        bundles,
        contact_cutoff=5.0,
        dihedral_order=3,
    )

    assert metrics["heavy_contact_count"] > 0.0
    assert metrics["by_pair_class"]["cn_same_layer"]["heavy_contact_count"] > 0.0
    assert metrics["by_pair_class"]["c2_same_index"]["heavy_contact_count"] > 0.0
    assert metrics["by_pair_class"]["cross_other"]["heavy_contact_count"] == 0.0


def test_cyclic_oligomer_defaults_to_rfdiffusion_backend(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
    )

    assert predocker.backend == "rfdiffusion"


def test_cyclic_oligomer_rpxdock_failure_defaults_to_rfdiffusion_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0),
        "B": protomer @ _rotation_z(92.0),
        "C": protomer @ _rotation_z(181.0),
        "D": protomer @ _rotation_z(269.0),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_nohit_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        raise rpxdock_wrapper.RpxDockOutputError("no docked models")

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    manifest = _load_case_manifest(output_path.parent)
    out_a = _load_chain_coords(output_path, "A")
    out_b = _load_chain_coords(output_path, "B")

    assert manifest["backend"] == "rfdiffusion"
    assert manifest["fallback_from_backend"] == "rpxdock"
    assert manifest["fallback_policy"] == "rfdiffusion"
    assert "RpxDockOutputError" in manifest["fallback_reason"]
    assert np.linalg.norm(np.mean(out_a, axis=0) - np.mean(out_b, axis=0)) > 1.0


def test_cyclic_oligomer_rpxdock_failure_can_return_input_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    chain_coords = {
        "A": np.array([[5.0, 0.0, 0.0], [5.7, 0.5, 0.0], [6.3, 0.1, 0.0]], dtype=float),
        "B": np.array([[0.0, 5.0, 0.0], [-0.5, 5.7, 0.0], [-0.1, 6.3, 0.0]], dtype=float),
        "C": np.array([[-5.0, 0.0, 0.0], [-5.7, -0.5, 0.0], [-6.3, -0.1, 0.0]], dtype=float),
        "D": np.array([[0.0, -5.0, 0.0], [0.5, -5.7, 0.0], [0.1, -6.3, 0.0]], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_input_fallback.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        raise rpxdock_wrapper.RpxDockOutputError("no docked models")

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
        rpxdock_failure_fallback="input",
    )

    output_path = Path(predocker.run(str(input_pdb)))
    manifest = _load_case_manifest(output_path.parent)

    assert manifest["backend"] == "input"
    assert manifest["fallback_policy"] == "input"
    np.testing.assert_allclose(_load_chain_coords(output_path, "A"), chain_coords["A"], atol=1e-3)


def test_cyclic_oligomer_rpxdock_failure_can_raise_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    chain_coords = {
        "A": np.array([[5.0, 0.0, 0.0], [5.7, 0.5, 0.0], [6.3, 0.1, 0.0]], dtype=float),
        "B": np.array([[0.0, 5.0, 0.0], [-0.5, 5.7, 0.0], [-0.1, 6.3, 0.0]], dtype=float),
        "C": np.array([[-5.0, 0.0, 0.0], [-5.7, -0.5, 0.0], [-6.3, -0.1, 0.0]], dtype=float),
        "D": np.array([[0.0, -5.0, 0.0], [0.5, -5.7, 0.0], [0.1, -6.3, 0.0]], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_raise_input.pdb", chain_coords)

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        raise rpxdock_wrapper.RpxDockOutputError("no docked models")

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
        rpxdock_failure_fallback="raise",
    )

    with pytest.raises(rpxdock_wrapper.RpxDockOutputError, match="no docked models"):
        predocker.run(str(input_pdb))


def test_rpxdock_failure_fallback_rejects_unknown_policy(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})

    with pytest.raises(ValueError, match="rpxdock_failure_fallback"):
        SymmetryRescueProcessor(
            cache_enabled=False,
            mode="cyclic_oligomer",
            oligo="AAAA",
            backend="rpxdock",
            rpxdock_failure_fallback="silent",
        )


def test_pseudocycle_rpxdock_backend_recenters_dock_input_to_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.6, 0.4, 0.0],
            [3.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    chain_a = np.vstack(
        [
            base_unit @ _rotation_z(3.0),
            base_unit @ _rotation_z(91.0) + np.array([0.1, -0.2, 0.0], dtype=float),
            base_unit @ _rotation_z(182.0) + np.array([-0.2, 0.1, 0.0], dtype=float),
            base_unit @ _rotation_z(269.0) + np.array([0.05, 0.15, 0.0], dtype=float),
        ]
    )
    input_pdb = _write_backbone_pdb(tmp_path / "pseudo_rpxdock_input.pdb", {"A": chain_a})

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        dummy_output = tmp_path / "fake_pseudo_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([3.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_pseudo.stdout.log",
            stderr_path=tmp_path / "fake_pseudo.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    manifest = _load_case_manifest(case_dir)
    avg_coords = _load_chain_coords(case_dir / "01_average_asu_structure.pdb", "A")
    dock_input_coords = _load_chain_coords(case_dir / "01_dock_input_structure.pdb", "A")
    out_coords = _load_chain_coords(output_path, "A")

    assert manifest["mode"] == "pseudocycle"
    assert manifest["asu_kind"] == "repeat_unit"
    assert np.linalg.norm(np.mean(avg_coords, axis=0)[:2]) > 1.0
    assert np.linalg.norm(np.mean(dock_input_coords, axis=0)) < 1e-3
    assert out_coords.shape == chain_a.shape


def test_pseudocycle_rpxdock_backend_applies_docked_fit_transform_to_larger_movable_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "GGG@cap AAA@u u TTT@tail"})
    cap = np.array([[-3.0, 0.0, 0.0], [-2.1, 0.3, 0.0], [-1.2, 0.1, 0.0]], dtype=float)
    unit = np.array([[0.0, 0.0, 0.0], [0.9, 0.4, 0.0], [1.9, 0.1, 0.0]], dtype=float)
    tail = np.array([[3.0, 0.0, 0.0], [4.1, -0.2, 0.0], [5.1, 0.1, 0.0]], dtype=float)
    rotation = _rotation_z(175.0)
    offset = np.array([0.15, -0.05, 0.0], dtype=float)
    chain_a = np.vstack([cap, unit, unit @ rotation + offset, tail @ rotation + offset])
    input_pdb = _write_backbone_pdb(tmp_path / "pseudo_cap_tail_input.pdb", {"A": chain_a})

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        dummy_output = tmp_path / "fake_pseudo_cap_tail_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([2.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_cap_tail.stdout.log",
            stderr_path=tmp_path / "fake_cap_tail.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        order=2,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    out_coords = _load_chain_coords(output_path, "A")
    fit_groups = build_internal_repeat_residue_groups("A")
    movable_groups = build_internal_repeat_movable_groups("A")

    assert out_coords.shape == chain_a.shape
    for fit_group, movable_group in zip(
        fit_groups,
        movable_groups,
        strict=True,
    ):
        corr_R, corr_t = kabsch_transform(chain_a[fit_group], out_coords[fit_group])
        expected_movable = apply_transform(chain_a[movable_group], corr_R, corr_t)
        assert np.allclose(out_coords[movable_group], expected_movable, atol=1e-3)


def test_pseudocycle_rpxdock_backend_moves_non_common_terminal_atoms_with_repeat_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.6, 0.4, 0.0],
            [3.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    chain_a = np.vstack(
        [
            base_unit @ _rotation_z(3.0),
            base_unit @ _rotation_z(91.0) + np.array([0.1, -0.2, 0.0], dtype=float),
            base_unit @ _rotation_z(182.0) + np.array([-0.2, 0.1, 0.0], dtype=float),
            base_unit @ _rotation_z(269.0) + np.array([0.05, 0.15, 0.0], dtype=float),
        ]
    )
    input_pdb = _write_backbone_pdb(
        tmp_path / "pseudo_rpxdock_oxt_input.pdb",
        {"A": chain_a},
        add_oxt_last=True,
    )

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        dummy_output = tmp_path / "fake_pseudo_oxt_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([3.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_pseudo_oxt.stdout.log",
            stderr_path=tmp_path / "fake_pseudo_oxt.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    movable_groups = build_internal_repeat_movable_groups("A")
    input_oxt = _load_atom_coord(input_pdb, "A", residue_number=len(chain_a), atom_name="OXT")
    output_oxt = _load_atom_coord(output_path, "A", residue_number=len(chain_a), atom_name="OXT")
    expected_oxt = _expected_atom_from_group_transform(
        input_pdb,
        output_path,
        chain_id="A",
        groups=movable_groups,
        group_index=-1,
        residue_number=len(chain_a),
        atom_name="OXT",
    )

    assert np.linalg.norm(output_oxt - input_oxt) > 1e-2
    assert np.allclose(output_oxt, expected_oxt, atol=2e-2)


@pytest.mark.parametrize("reference_chain_index", [0, 5])
def test_cyclic_oligomer_reference_chain_index_must_be_in_range(
    tmp_path: Path,
    reference_chain_index: int,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "B": protomer @ _rotation_z(92.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "C": protomer @ _rotation_z(181.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "D": protomer @ _rotation_z(269.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_invalid_reference_input.pdb", chain_coords)
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rfdiffusion",
        order=4,
        reference_chain_index=reference_chain_index,
    )

    with pytest.raises(ValueError):
        predocker.run(str(input_pdb))


def test_cyclic_oligomer_reference_chain_index_selects_requested_chain(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [4.8, 0.0, 0.0],
            [5.5, 0.5, 0.0],
            [6.2, 0.1, 0.0],
            [7.0, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "X": protomer @ _rotation_z(5.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "Y": protomer @ _rotation_z(95.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "Z": protomer @ _rotation_z(185.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "W": protomer @ _rotation_z(275.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_reference_chain_input.pdb", chain_coords)
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rfdiffusion",
        order=4,
        reference_chain_index=2,
    )

    output_path = Path(predocker.run(str(input_pdb)))

    case_dir = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))[0]
    manifest = _load_case_manifest(case_dir)
    avg_coords = _load_chain_coords(case_dir / "01_average_asu_structure.pdb", "Y")
    out_centroids = {
        chain_id: np.mean(_load_chain_coords(output_path, chain_id), axis=0)
        for chain_id in ["X", "Y", "Z", "W"]
    }
    anchor_chain = max(out_centroids, key=lambda chain_id: float(out_centroids[chain_id][0]))

    assert manifest["reference_chain_index"] == 2
    assert avg_coords.shape == protomer.shape
    assert anchor_chain == "Y"
    assert abs(float(out_centroids["Y"][1])) < 1e-2


def test_cyclic_oligomer_rfdiffusion_backend_moves_non_common_terminal_atoms_with_chain_transform(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "B": protomer @ _rotation_z(92.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "C": protomer @ _rotation_z(181.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "D": protomer @ _rotation_z(269.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(
        tmp_path / "c4_oxt_input.pdb",
        chain_coords,
        add_oxt_last=True,
        oxt_chains={"D"},
    )
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rfdiffusion",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    input_oxt = _load_atom_coord(input_pdb, "D", residue_number=len(protomer), atom_name="OXT")
    output_oxt = _load_atom_coord(output_path, "D", residue_number=len(protomer), atom_name="OXT")
    expected_oxt = _expected_atom_from_cyclic_chain_transform(
        input_pdb,
        output_path,
        chain_ids=["A", "B", "C", "D"],
        chain_index=3,
        residue_number=len(protomer),
        atom_name="OXT",
    )

    assert np.linalg.norm(output_oxt - input_oxt) > 1e-2
    assert np.allclose(output_oxt, expected_oxt, atol=2e-2)


def test_cyclic_oligomer_rpxdock_backend_moves_non_common_terminal_atoms_with_chain_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0) + np.array([0.0, 0.0, 0.2], dtype=float),
        "B": protomer @ _rotation_z(92.0) + np.array([0.2, -0.1, -0.3], dtype=float),
        "C": protomer @ _rotation_z(181.0) + np.array([-0.3, 0.2, 0.4], dtype=float),
        "D": protomer @ _rotation_z(269.0) + np.array([0.1, 0.3, -0.2], dtype=float),
    }
    input_pdb = _write_backbone_pdb(
        tmp_path / "c4_rpxdock_oxt_input.pdb",
        chain_coords,
        add_oxt_last=True,
        oxt_chains={"D"},
    )

    def _fake_run_local_cyclic_dock(*_args, **_kwargs):
        dummy_output = tmp_path / "fake_c4_oxt_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([6.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_c4_oxt.stdout.log",
            stderr_path=tmp_path / "fake_c4_oxt.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
    )

    output_path = Path(predocker.run(str(input_pdb)))
    input_oxt = _load_atom_coord(input_pdb, "D", residue_number=len(protomer), atom_name="OXT")
    output_oxt = _load_atom_coord(output_path, "D", residue_number=len(protomer), atom_name="OXT")
    expected_oxt = _expected_atom_from_cyclic_chain_transform(
        input_pdb,
        output_path,
        chain_ids=["A", "B", "C", "D"],
        chain_index=3,
        residue_number=len(protomer),
        atom_name="OXT",
    )

    assert np.linalg.norm(output_oxt - input_oxt) > 1e-2
    assert np.allclose(output_oxt, expected_oxt, atol=2e-2)


def test_initialize_predockers_alias_level_rpxdock_overrides_win(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    initialize_predockers(
        OmegaConf.create(
            {
                "predocker_terms": {
                    "symmetry": {
                        "_type": "symmetry_canonicalizer",
                        "mode": "pseudocycle",
                        "oligo": "A",
                        "backend": "rpxdock",
                        "rpxdock_python_bin": "/alias/python",
                        "rpxdock_vendor_repo": "/alias/vendor",
                        "rpxdock_timeout_s": 321,
                    }
                },
                "symmetry_canonicalizer": {
                    "rpxdock_python_bin": "/default/python",
                    "rpxdock_vendor_repo": "/default/vendor",
                    "rpxdock_timeout_s": 999,
                },
            }
        )
    )

    predocker = global_config.predockers["symmetry"]

    assert predocker.rpxdock_python_bin == "/alias/python"
    assert predocker.rpxdock_vendor_repo == "/alias/vendor"
    assert predocker.rpxdock_timeout_s == 321


def test_run_nested_predocker_passes_runtime_oligo_without_mutating_registered_predocker(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        backend="rfdiffusion",
        oligo="",
    )
    seen: list[str] = []

    def _fake_run(input_file):
        seen.append(predocker.oligo)
        return input_file

    runtime_clone = predocker.runtime_clone(oligo="AA")
    runtime_clone.run = lambda input_file: (seen.append(runtime_clone.oligo), input_file)[1]  # type: ignore[method-assign]
    predocker.runtime_clone = lambda **kwargs: runtime_clone  # type: ignore[method-assign]
    global_config.init_const("predockers", {"symmetry": predocker})

    output = run_nested_predocker(["symmetry"], str(tmp_path / "input.pdb"), runtime_oligo="AA")

    assert output == str(tmp_path / "input.pdb")
    assert seen == ["AA"]
    assert predocker.oligo == ""


def test_exact_symmetry_backend_uses_unified_post_run_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u"})
    input_pdb = _write_backbone_pdb(
        tmp_path / "unified_post_run_input.pdb",
        {"A": np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]], dtype=float)},
    )
    predocker = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rfdiffusion",
    )
    case_dir = predocker.debug_dir / "symmetry_cases" / "fake_case"
    result_path = case_dir / "02_redock_result.pdb"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("END\n", encoding="utf-8")

    seen: list[tuple[str, Path]] = []

    monkeypatch.setattr(predocker, "_run_impl", lambda _input_file: str(result_path))

    def _fake_after_cache_store(cache_key, input_file, result):
        seen.append(("after", Path(result)))

    def _fake_take_cache(cache_key):
        cached = Path(predocker._cache[cache_key])
        seen.append(("take", cached))
        return str(cached)

    monkeypatch.setattr(predocker, "_after_cache_store", _fake_after_cache_store)
    monkeypatch.setattr(predocker, "take_cache", _fake_take_cache)

    first = predocker.run(str(input_pdb))
    second = predocker.run(str(input_pdb))

    assert first == str(result_path)
    assert second == str(result_path)
    assert seen == [
        ("after", result_path),
        ("take", result_path),
        ("take", result_path),
    ]


def test_ligandmpnn_parse_pdb_accepts_mmcif_paths(tmp_path: Path) -> None:
    pdb_path = _write_backbone_pdb(
        tmp_path / "ligandmpnn_input.pdb",
        {"A": np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=float)},
    )
    cif_path = tmp_path / "ligandmpnn_input.cif"
    cif_path.write_text(cli._single_chain_mmcif_string(pdb_path, "A"), encoding="utf-8")

    protein_dict, *_ = parse_PDB(str(cif_path), device="cpu")

    assert protein_dict["X"].shape[0] == 2


def test_rpxdock_symmetry_cache_returns_cached_candidates_without_rerun(
    tmp_path: Path,
) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u"})
    input_pdb = _write_backbone_pdb(
        tmp_path / "rpxdock_cache_sampling_input.pdb",
        {"A": np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]], dtype=float)},
    )
    predocker = SymmetryRescueProcessor(
        cache_enabled=True,
        mode="cyclic_oligomer",
        oligo="A",
        backend="rpxdock",
    )
    case_dir = predocker.debug_dir / "symmetry_cases" / "fake_cache_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    primary = case_dir / "02_redock_result.pdb"
    alt = case_dir / "02_redock_result__top1.pdb"
    primary.write_text("END\n", encoding="utf-8")
    alt.write_text("END\n", encoding="utf-8")
    (case_dir / "03_manifest.json").write_text(
        json.dumps(
            {
                "output_structure": str(primary),
                "output_candidates": [str(primary), str(alt)],
            }
        ) + "\n",
        encoding="utf-8",
    )

    class _CycleChoice:
        def __init__(self):
            self.calls = 0

        def choice(self, items):
            value = items[self.calls % len(items)]
            self.calls += 1
            return value

    predocker._cache_rng = _CycleChoice()
    run_calls = {"count": 0}

    def _fake_run_impl(_input_file):
        run_calls["count"] += 1
        return str(primary)

    predocker._run_impl = _fake_run_impl  # type: ignore[method-assign]

    first = predocker.run(str(input_pdb))
    second = predocker.run(str(input_pdb))
    third = predocker.run(str(input_pdb))

    assert run_calls["count"] == 1
    assert first == str(primary)
    assert second == str(alt)
    assert third == str(primary)
    assert predocker.get_cache_stats()["candidate_bank_sizes"]


def test_cyclic_oligomer_rpxdock_backend_writes_pdb_dock_input_for_mmcif_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Bio.PDB import MMCIFIO
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    chain_coords = {
        "A": np.array([[9.0, 0.0, 0.0], [10.0, 0.5, 0.0], [11.0, 0.0, 0.0]], dtype=float),
        "B": np.array([[0.0, 9.0, 0.0], [-0.5, 10.0, 0.0], [0.0, 11.0, 0.0]], dtype=float),
        "C": np.array([[-9.0, 0.0, 0.0], [-10.0, -0.5, 0.0], [-11.0, 0.0, 0.0]], dtype=float),
        "D": np.array([[0.0, -9.0, 0.0], [0.5, -10.0, 0.0], [0.0, -11.0, 0.0]], dtype=float),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_input_for_cif.pdb", chain_coords)
    from Bio.PDB import PDBParser

    cif_path = tmp_path / "c4_rpxdock_input_for_cif.cif"
    structure = PDBParser(QUIET=True).get_structure("c4_cif", str(input_pdb))
    mmcif = MMCIFIO()
    mmcif.set_structure(structure)
    mmcif.save(str(cif_path))

    def _fake_run_local_cyclic_dock(monomer_path, config):
        dummy_output = tmp_path / "fake_cif_rpxdock_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        assert Path(monomer_path).suffix.lower() == ".pdb"
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([6.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_cif.stdout.log",
            stderr_path=tmp_path / "fake_cif.stderr.log",
            candidates=(
                rpxdock_wrapper.RpxDockCandidate(
                    rank=0,
                    docked_structure_path=dummy_output,
                    rotation=np.eye(3, dtype=float),
                    translation=np.array([6.0, 0.0, 0.0], dtype=float),
                ),
            ),
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
    )

    predocker.run(str(cif_path))

    case_dirs = sorted((tmp_path / "predocker" / "symmetry_cases").glob("*"))
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    assert (case_dir / "01_dock_input_structure.pdb").exists()
    assert not (case_dir / "01_dock_input_structure.cif").exists()


def test_cyclic_oligomer_rpxdock_backend_does_not_pass_inferred_longaxis_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAAA"})
    protomer = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.7, 0.5, 0.0],
            [6.3, 0.1, 0.0],
            [7.1, -0.2, 0.0],
        ],
        dtype=float,
    )
    chain_coords = {
        "A": protomer @ _rotation_z(4.0),
        "B": protomer @ _rotation_z(92.0),
        "C": protomer @ _rotation_z(181.0),
        "D": protomer @ _rotation_z(269.0),
    }
    input_pdb = _write_backbone_pdb(tmp_path / "c4_rpxdock_longaxis_input.pdb", chain_coords)
    seen = {}

    def _fake_run_local_cyclic_dock(monomer_path, config):
        seen['max_longaxis_dot_z'] = config.max_longaxis_dot_z
        dummy_output = tmp_path / "fake_longaxis_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([6.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_longaxis.stdout.log",
            stderr_path=tmp_path / "fake_longaxis.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="cyclic_oligomer",
        oligo="AAAA",
        backend="rpxdock",
        order=4,
    )

    predocker.run(str(input_pdb))

    assert seen['max_longaxis_dot_z'] is None


def test_pseudocycle_rpxdock_backend_does_not_pass_inferred_longaxis_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.structure_processors.wrappers import rpxdock as rpxdock_wrapper

    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})
    base_unit = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.6, 0.4, 0.0],
            [3.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    chain_a = np.vstack(
        [
            base_unit @ _rotation_z(3.0),
            base_unit @ _rotation_z(91.0),
            base_unit @ _rotation_z(182.0),
            base_unit @ _rotation_z(269.0),
        ]
    )
    input_pdb = _write_backbone_pdb(tmp_path / "pseudo_rpxdock_longaxis_input.pdb", {"A": chain_a})
    seen = {}

    def _fake_run_local_cyclic_dock(monomer_path, config):
        seen['max_longaxis_dot_z'] = config.max_longaxis_dot_z
        dummy_output = tmp_path / "fake_pseudo_longaxis_output.pdb"
        dummy_output.write_text("END\n", encoding="utf-8")
        return rpxdock_wrapper.RpxDockRunResult(
            docked_structure_path=dummy_output,
            rotation=np.eye(3, dtype=float),
            translation=np.array([3.0, 0.0, 0.0], dtype=float),
            command=("fake-rpxdock",),
            stdout_path=tmp_path / "fake_pseudo_longaxis.stdout.log",
            stderr_path=tmp_path / "fake_pseudo_longaxis.stderr.log",
        )

    monkeypatch.setattr(rpxdock_wrapper, "run_local_cyclic_dock", _fake_run_local_cyclic_dock)

    predocker = SymmetryRescueProcessor(
        cache_enabled=False,
        mode="pseudocycle",
        oligo="A",
        included_protomer="A",
        backend="rpxdock",
        order=4,
    )

    predocker.run(str(input_pdb))

    assert seen['max_longaxis_dot_z'] is None
