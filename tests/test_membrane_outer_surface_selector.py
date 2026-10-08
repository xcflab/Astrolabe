from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.selectors.utils import construct_position_selector


def _protomer(length: int):
    return SimpleNamespace(
        type=ProtomerTypes.PROTEIN,
        length=length,
        mutable_mask=np.ones(length, dtype=bool),
        tied_positions=[(idx,) for idx in range(length)],
        aggregate_bool=lambda mask: mask,
    )


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


def _rotation_z(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def test_membrane_outer_surface_selects_lipid_facing_slab_residues(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("protomers", {"A": _protomer(5)})
    global_config.init_const("out_dir", str(tmp_path / "out"))
    global_config.init_const(
        "raw_model_results_dir",
        tmp_path / "out" / "raw_model_results",
    )
    global_config.set_variable("step", 0)

    base = np.array(
        [
            [10.0, 0.0, -4.0],
            [4.0, 0.0, -2.0],
            [11.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [10.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    chain_coords = {}
    for chain_id, angle in zip(["A", "B", "C", "D"], [0.0, 90.0, 180.0, 270.0], strict=True):
        chain_coords[chain_id] = base @ _rotation_z(angle)
    pdb_path = _write_ca_pdb(tmp_path / "c4.pdb", chain_coords)

    scores = StandardScores(
        names=["AAAA"],
        plddt={"AAAA": [1.0]},
        complex_paths={"AAAA": pdb_path},
    )
    selected = construct_position_selector(
        "membrane_outer_surface(membrane_thickness=10.0, outer_surface_depth=2.0, axial_bin_size=4.0)",
        scores,
        global_config.protomers,
    )

    assert selected["A"].tolist() == [True, False, True, False, True]


def test_membrane_outer_surface_defaults_to_membrane_core_slab(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("protomers", {"A": _protomer(3)})
    global_config.init_const("out_dir", str(tmp_path / "out"))
    global_config.init_const(
        "raw_model_results_dir",
        tmp_path / "out" / "raw_model_results",
    )
    global_config.set_variable("step", 0)

    base = np.array(
        [
            [10.0, 0.0, -14.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 16.0],
        ],
        dtype=float,
    )
    chain_coords = {}
    for chain_id, angle in zip(["A", "B", "C", "D"], [0.0, 90.0, 180.0, 270.0], strict=True):
        chain_coords[chain_id] = base @ _rotation_z(angle)
    pdb_path = _write_ca_pdb(tmp_path / "c4_default_slab.pdb", chain_coords)

    scores = StandardScores(
        names=["AAAA"],
        plddt={"AAAA": [1.0]},
        complex_paths={"AAAA": pdb_path},
    )
    selected_default = construct_position_selector(
        "membrane_outer_surface(outer_surface_depth=2.0, axial_bin_size=4.0)",
        scores,
        global_config.protomers,
    )
    selected_explicit = construct_position_selector(
        "membrane_outer_surface(membrane_thickness=30.0, outer_surface_depth=2.0, axial_bin_size=4.0)",
        scores,
        global_config.protomers,
    )

    assert selected_default["A"].tolist() == selected_explicit["A"].tolist()
