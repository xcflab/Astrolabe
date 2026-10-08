from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import ComplexPoses
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

    def dump_pdb(self, _path: str) -> None:
        raise AssertionError("HoleSurfacePositionSelector should reuse existing channel_hole artifacts in this test.")


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


def _write_scan_summary(path: Path) -> None:
    rows = [
        {
            "scan_id": "x03_y03__dx+0_dy+0",
            "parent_scan_id": "x03_y03",
            "ix": 3,
            "iy": 4,
            "cx": 10.0,
            "cy": 0.0,
            "cz": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
            "status": "ok",
            "profile_length": 50.0,
            "effective_path_len": 45.0,
            "effective_start_s": -20.0,
            "effective_end_s": 25.0,
            "n_profile_points": 2,
            "min_radius": 1.5,
            "min_radius_profile": 0.0,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_channel_artifacts(root: Path) -> tuple[Path, Path]:
    step_dir = root / "channel_hole" / "step_000_AB"
    _write_scan_summary(step_dir / "scan_summary.tsv")
    aligned_structure = step_dir / "step_000_AB_model__channel_hole_aligned.pdb"
    aligned_structure.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2      13.350   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A   3      15.200   0.000   0.000  1.00 20.00           C",
                "HETATM    4  C1  LIG B   1      10.800   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spheres_path = step_dir / "x03_y03__dx+0_dy+0" / "x03_y03__dx+0_dy+0_spheres.pdb"
    spheres_path.parent.mkdir(parents=True, exist_ok=True)
    spheres_path.write_text(
        "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.50  0.00           Q\nEND\n",
        encoding="utf-8",
    )
    reference_structure = root / "reference_structure.pdb"
    reference_structure.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n", encoding="utf-8")
    reference_spheres = root / "reference_spheres.pdb"
    reference_spheres.write_text(
        "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.50  0.00           Q\nEND\n",
        encoding="utf-8",
    )
    return reference_structure, reference_spheres


def _write_bottleneck_artifacts(root: Path) -> tuple[Path, Path]:
    step_dir = root / "channel_hole" / "step_000_AB"
    _write_scan_summary(step_dir / "scan_summary.tsv")
    aligned_structure = step_dir / "step_000_AB_model__channel_hole_aligned.pdb"
    aligned_structure.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CB  ALA A   1      20.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  TYR A   2      13.200   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CB  TYR A   2      13.300   0.000   0.000  1.00 20.00           C",
                "ATOM      5  CA  PHE A   3      13.800   0.000   0.000  1.00 20.00           C",
                "ATOM      6  CB  PHE A   3      13.900   0.000   0.000  1.00 20.00           C",
                "HETATM    7  C1  LIG B   1      10.800   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spheres_path = step_dir / "x03_y03__dx+0_dy+0" / "x03_y03__dx+0_dy+0_spheres.pdb"
    spheres_path.parent.mkdir(parents=True, exist_ok=True)
    spheres_path.write_text(
        "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.50  0.00           Q\nEND\n",
        encoding="utf-8",
    )
    reference_structure = root / "reference_structure.pdb"
    reference_structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
        encoding="utf-8",
    )
    reference_spheres = root / "reference_spheres.pdb"
    reference_spheres.write_text(
        "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.50  0.00           Q\nEND\n",
        encoding="utf-8",
    )
    return reference_structure, reference_spheres


def _write_multi_bottleneck_artifacts(root: Path) -> tuple[Path, Path]:
    step_dir = root / "channel_hole" / "step_000_AB"
    _write_scan_summary(step_dir / "scan_summary.tsv")
    aligned_structure = step_dir / "step_000_AB_model__channel_hole_aligned.pdb"
    aligned_structure.write_text(
        "\n".join(
            [
                "ATOM      1  CA  TYR A   1      13.200   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CB  TYR A   1      13.300   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  PHE A   2      13.450   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CB  PHE A   2      13.550   0.000   0.000  1.00 20.00           C",
                "ATOM      5  CA  LEU A   3      33.200   0.000   0.000  1.00 20.00           C",
                "ATOM      6  CB  LEU A   3      33.300   0.000   0.000  1.00 20.00           C",
                "HETATM    7  C1  LIG B   1      10.800   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spheres_path = step_dir / "x03_y03__dx+0_dy+0" / "x03_y03__dx+0_dy+0_spheres.pdb"
    spheres_path.parent.mkdir(parents=True, exist_ok=True)
    spheres_path.write_text(
        "\n".join(
            [
                "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.00  0.00           Q",
                "ATOM      2  QSS SPH A   2      20.000   0.000   0.000  2.50  0.00           Q",
                "ATOM      3  QSS SPH A   3      30.000   0.000   0.000  1.18  0.00           Q",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reference_structure = root / "reference_structure.pdb"
    reference_structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
        encoding="utf-8",
    )
    reference_spheres = root / "reference_spheres.pdb"
    reference_spheres.write_text(
        "ATOM      1  QSS SPH A   1      10.000   0.000   0.000  1.50  0.00           Q\nEND\n",
        encoding="utf-8",
    )
    return reference_structure, reference_spheres


def test_construct_position_selector_accepts_hole_surface_expression(tmp_path: Path) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.set_variable("step", 0)
    reference_structure, reference_spheres = _write_channel_artifacts(tmp_path / "raw_model_results")
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        (
            "hole_surface("
            f"reference_structure_path='{str(reference_structure)}', "
            f"reference_spheres_path='{str(reference_spheres)}', "
            "surface_gap_max=2.0"
            ") and not ligand_contact(distance_max=2.0)"
        ),
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, False, True]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))


def test_construct_position_selector_accepts_pore_bottleneck_expression(tmp_path: Path) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.set_variable("step", 0)
    reference_structure, reference_spheres = _write_bottleneck_artifacts(
        tmp_path / "raw_model_results"
    )
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        (
            "pore_bottleneck("
            f"reference_structure_path='{str(reference_structure)}', "
            f"reference_spheres_path='{str(reference_spheres)}', "
            "top_n=1, surface_gap_max=2.0"
            ")"
        ),
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([False, True, False]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))


def test_pore_bottleneck_selects_blockers_from_multiple_target_limiting_regions(tmp_path: Path) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.set_variable("step", 0)
    reference_structure, reference_spheres = _write_multi_bottleneck_artifacts(
        tmp_path / "raw_model_results"
    )
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))

    selection = construct_position_selector(
        (
            "pore_bottleneck("
            f"reference_structure_path='{str(reference_structure)}', "
            f"reference_spheres_path='{str(reference_spheres)}', "
            "top_n=1, surface_gap_max=2.0, min_radius_window=0.05, "
            "target_min_radius=1.2, bottleneck_region_gap=0"
            ")"
        ),
        evaluation_result,
        protomers,
    )

    np.testing.assert_array_equal(selection["A"], np.array([True, False, True]))
    np.testing.assert_array_equal(selection["B"], np.array([False]))


def test_pore_bottleneck_can_stage_regions_by_activation_index(tmp_path: Path) -> None:
    protomers = {
        "A": _protomer(3, ProtomerTypes.PROTEIN),
        "B": _protomer(1, ProtomerTypes.CCD),
    }
    global_config.init_const("protomers", protomers)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.set_variable("step", 0)
    reference_structure, reference_spheres = _write_multi_bottleneck_artifacts(
        tmp_path / "raw_model_results"
    )
    evaluation_result = SimpleNamespace(complex_poses=ComplexPoses({"AB": _Pose()}))
    expression = (
        "pore_bottleneck("
        f"reference_structure_path='{str(reference_structure)}', "
        f"reference_spheres_path='{str(reference_spheres)}', "
        "top_n=1, surface_gap_max=2.0, min_radius_window=0.05, "
        "target_min_radius=1.2, bottleneck_region_gap=0, "
        "region_stride=2, region_offset_source='activation'"
        ")"
    )

    global_config.set_variable("pore_bottleneck_activation_index", 0)
    selection_0 = construct_position_selector(expression, evaluation_result, protomers)
    global_config.set_variable("pore_bottleneck_activation_index", 1)
    selection_1 = construct_position_selector(expression, evaluation_result, protomers)

    np.testing.assert_array_equal(selection_0["A"], np.array([True, False, False]))
    np.testing.assert_array_equal(selection_1["A"], np.array([False, False, True]))
    np.testing.assert_array_equal(selection_0["B"], np.array([False]))
    np.testing.assert_array_equal(selection_1["B"], np.array([False]))
