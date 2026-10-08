from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run as run_mod
from astrolabe.core import global_config
from astrolabe.core.repeat_symmetry import (
    apply_transform,
    build_internal_repeat_residue_groups,
    build_shared_core_residue_groups,
    dihedral_group_transforms_canonical,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import _VALID_LOSSES
from astrolabe.losses.repeat_symmetry import (
    _ananas_group_arg,
    _normalize_good,
    _normalize_bounded_good,
    _parse_ananas_rmsd,
    ananas_rmsd_norm,
    ananas_rmsd_raw,
    ananas_rmsd_wrapper,
    dihedral_pair_contact_fraction_norm,
    dihedral_pair_contact_fraction_raw,
    dihedral_pair_contact_count_norm,
    dihedral_pair_contact_count_raw,
    dihedral_pair_min_distance_norm,
    dihedral_pair_min_distance_raw,
    dihedral_pair_sep_cv_raw,
    dihedral_ring_sep_cv_raw,
    point_group_orbit_rmsd_ratio,
    point_group_orbit_rmsd_raw,
    point_group_rotation_raw,
    point_group_symmetry_norm,
    pseudocycle_closure_raw,
    protomer_rmsd_norm,
    protomer_rmsd_raw,
    pseudocycle_closure_norm,
    pseudocycle_rise_raw,
    pseudocycle_rise_norm,
    pseudocycle_rotation_raw,
    pseudocycle_rotation_norm,
    repeat_unit_rmsd_norm,
    repeat_unit_rmsd_raw,
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


def _write_backbone_pdb(path: Path, chain_coords: dict[str, np.ndarray]) -> Path:
    lines: list[str] = []
    serial = 1
    offsets = {
        "N": np.array([-0.45, 0.00, 0.00], dtype=float),
        "CA": np.array([0.00, 0.00, 0.00], dtype=float),
        "C": np.array([0.45, 0.00, 0.00], dtype=float),
        "O": np.array([0.70, 0.25, 0.00], dtype=float),
    }
    for chain_id, coords in chain_coords.items():
        for res_i, xyz in enumerate(np.asarray(coords, dtype=float), start=1):
            for atom_name, offset in offsets.items():
                pos = np.asarray(xyz, dtype=float) + offset
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} ALA {chain_id}{res_i:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00           {atom_name[0]}"
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


def test_build_internal_repeat_residue_groups_supports_multi_segment_units(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u CCC@v u v"})

    groups = build_internal_repeat_residue_groups("A")

    assert len(groups) == 2
    assert np.array_equal(groups[0], np.arange(0, 6))
    assert np.array_equal(groups[1], np.arange(6, 12))


def test_segment_manager_registers_all_repeat_component_names(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAA@u u u u"})

    assert {"A_1", "A_2", "A_3", "A_4"}.issubset(global_config.segment_manager.segments)


def test_build_shared_core_residue_groups_ignore_unique_interface_segments(tmp_path: Path) -> None:
    _setup_protomers(
        tmp_path,
        {
            "A": "AAA@core_1 CCC@itf_a DDD@core_2",
            "B": "core_1 EEE@itf_b core_2",
            "C": "core_1 FFF@itf_c core_2",
        },
    )

    groups = build_shared_core_residue_groups("ABC")

    assert set(groups.keys()) == {"A", "B", "C"}
    assert np.array_equal(groups["A"], np.array([0, 1, 2, 6, 7, 8]))
    assert np.array_equal(groups["B"], np.array([0, 1, 2, 6, 7, 8]))
    assert np.array_equal(groups["C"], np.array([0, 1, 2, 6, 7, 8]))


def test_bounded_normalization_maps_best_and_worst_exactly() -> None:
    assert _normalize_bounded_good(0.0, best=0.0, worst=10.0) == pytest.approx(1.0)
    assert _normalize_bounded_good(10.0, best=0.0, worst=10.0) == pytest.approx(0.0)
    assert _normalize_bounded_good(5.0, best=0.0, worst=10.0) == pytest.approx(0.5)
    assert _normalize_bounded_good(-1.0, best=0.0, worst=10.0) == pytest.approx(1.0)
    assert _normalize_bounded_good(12.0, best=0.0, worst=10.0) == pytest.approx(0.0)


def test_repeat_and_pseudocycle_losses_score_ideal_internal_repeat_geometry(tmp_path: Path) -> None:
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
    pdb_path = _write_ca_pdb(tmp_path / "pseudo_cycle.pdb", {"A": chain_a})
    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [0.9]},
        ptm={"A": [0.8]},
        complex_paths={"A": pdb_path},
    )

    repeat_raw = repeat_unit_rmsd_raw(eval_result)["A"]
    repeat_norm = repeat_unit_rmsd_norm(eval_result)["A"]
    rot_raw = pseudocycle_rotation_raw(eval_result)["A"]
    rot_norm = pseudocycle_rotation_norm(eval_result)["A"]
    rise_raw = pseudocycle_rise_raw(eval_result)["A"]
    rise_norm = pseudocycle_rise_norm(eval_result)["A"]
    closure_raw = pseudocycle_closure_raw(eval_result)["A"]
    closure_norm = pseudocycle_closure_norm(eval_result)["A"]

    assert repeat_raw < 1e-6
    assert repeat_norm == pytest.approx(1.0)
    assert rot_raw < 1e-6
    assert rot_norm == pytest.approx(1.0)
    assert rise_raw < 1e-6
    assert rise_norm == pytest.approx(1.0)
    assert closure_raw == pytest.approx(0.0)
    assert closure_norm == pytest.approx(1.0)


def test_point_group_losses_score_ideal_dihedral_geometry(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
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
    pdb_path = _write_ca_pdb(tmp_path / "d3_ideal.pdb", chain_coords)
    eval_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        ptm={"AAAAAA": [0.8]},
        complex_paths={"AAAAAA": pdb_path},
    )

    assert point_group_orbit_rmsd_raw(eval_result, point_group="D3")["AAAAAA"] < 0.15
    assert point_group_orbit_rmsd_ratio(eval_result, point_group="D3")["AAAAAA"] < 0.02
    assert point_group_rotation_raw(eval_result, point_group="D3")["AAAAAA"] < 1.0
    assert point_group_symmetry_norm(eval_result, point_group="D3")["AAAAAA"] > 0.95
    assert dihedral_ring_sep_cv_raw(eval_result, order=3)["AAAAAA"] < 1e-5
    assert dihedral_pair_sep_cv_raw(eval_result, order=3)["AAAAAA"] < 1e-5


def test_dihedral_pair_contact_losses_reward_c2_pair_contacts(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
    top_ring = {
        "A": np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float),
        "B": np.array([[20.0, 0.0, 0.0], [21.5, 0.0, 0.0]], dtype=float),
        "C": np.array([[40.0, 0.0, 0.0], [41.5, 0.0, 0.0]], dtype=float),
    }
    close_pairs = {
        **top_ring,
        "D": top_ring["A"] + np.array([0.0, 3.0, 0.0], dtype=float),
        "E": top_ring["B"] + np.array([0.0, 3.0, 0.0], dtype=float),
        "F": top_ring["C"] + np.array([0.0, 3.0, 0.0], dtype=float),
    }
    far_pairs = {
        **top_ring,
        "D": top_ring["A"] + np.array([0.0, 20.0, 0.0], dtype=float),
        "E": top_ring["B"] + np.array([0.0, 20.0, 0.0], dtype=float),
        "F": top_ring["C"] + np.array([0.0, 20.0, 0.0], dtype=float),
    }
    partial_pairs = {
        **top_ring,
        "D": top_ring["A"] + np.array([0.0, 3.0, 0.0], dtype=float),
        "E": top_ring["B"] + np.array([0.0, 20.0, 0.0], dtype=float),
        "F": top_ring["C"] + np.array([0.0, 20.0, 0.0], dtype=float),
    }
    close_path = _write_backbone_pdb(tmp_path / "d3_close_pairs.pdb", close_pairs)
    far_path = _write_backbone_pdb(tmp_path / "d3_far_pairs.pdb", far_pairs)
    partial_path = _write_backbone_pdb(tmp_path / "d3_partial_pairs.pdb", partial_pairs)
    close_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": close_path},
    )
    far_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": far_path},
    )
    partial_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": partial_path},
    )

    close_count = dihedral_pair_contact_count_raw(
        close_result,
        order=3,
        contact_cutoff=5.0,
    )["AAAAAA"]
    far_count = dihedral_pair_contact_count_raw(
        far_result,
        order=3,
        contact_cutoff=5.0,
    )["AAAAAA"]
    close_dist = dihedral_pair_min_distance_raw(close_result, order=3)["AAAAAA"]
    far_dist = dihedral_pair_min_distance_raw(far_result, order=3)["AAAAAA"]

    assert close_count > 0
    assert far_count == 0
    assert close_dist < 5.0
    assert far_dist > 10.0
    assert dihedral_pair_contact_count_norm(close_result, order=3, target=2.0)["AAAAAA"] == pytest.approx(1.0)
    assert dihedral_pair_contact_count_norm(far_result, order=3, target=2.0)["AAAAAA"] == pytest.approx(0.0)
    assert dihedral_pair_contact_fraction_raw(close_result, order=3, min_contacts=1)["AAAAAA"] == pytest.approx(1.0)
    assert dihedral_pair_contact_fraction_raw(partial_result, order=3, min_contacts=1)["AAAAAA"] == pytest.approx(1 / 3)
    assert dihedral_pair_contact_fraction_norm(close_result, order=3, min_contacts=1)["AAAAAA"] == pytest.approx(1.0)
    assert dihedral_pair_contact_fraction_norm(far_result, order=3, min_contacts=1)["AAAAAA"] == pytest.approx(0.0)
    assert dihedral_pair_min_distance_norm(close_result, order=3, best=3.0, worst=15.0)["AAAAAA"] > (
        dihedral_pair_min_distance_norm(far_result, order=3, best=3.0, worst=15.0)["AAAAAA"]
    )


def test_dihedral_pair_contact_losses_are_available_to_config_validation() -> None:
    assert "dihedral_pair_contact_count_raw" in _VALID_LOSSES
    assert "dihedral_pair_contact_count_norm" in _VALID_LOSSES
    assert "dihedral_pair_contact_fraction_raw" in _VALID_LOSSES
    assert "dihedral_pair_contact_fraction_norm" in _VALID_LOSSES
    assert "dihedral_pair_min_distance_raw" in _VALID_LOSSES
    assert "dihedral_pair_min_distance_norm" in _VALID_LOSSES


def test_point_group_losses_penalize_dihedral_orbit_and_orientation_errors(tmp_path: Path) -> None:
    _setup_protomers(tmp_path, {"A": "AAAA"})
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
    ideal_path = _write_ca_pdb(tmp_path / "d3_ideal_for_compare.pdb", chain_coords)
    perturbed = dict(chain_coords)
    perturbed["E"] = perturbed["E"] @ _rotation_z(18.0) + np.array([0.7, -0.3, 0.4], dtype=float)
    perturbed_path = _write_ca_pdb(tmp_path / "d3_perturbed.pdb", perturbed)
    ideal_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": ideal_path},
    )
    perturbed_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": perturbed_path},
    )

    assert point_group_orbit_rmsd_ratio(
        perturbed_result,
        point_group="D3",
    )["AAAAAA"] > point_group_orbit_rmsd_ratio(ideal_result, point_group="D3")["AAAAAA"]
    assert point_group_rotation_raw(
        perturbed_result,
        point_group="D3",
    )["AAAAAA"] > point_group_rotation_raw(ideal_result, point_group="D3")["AAAAAA"]
    assert point_group_symmetry_norm(
        perturbed_result,
        point_group="D3",
    )["AAAAAA"] < point_group_symmetry_norm(ideal_result, point_group="D3")["AAAAAA"]


def test_ananas_parser_prefers_average_rmsd_and_falls_back_to_table_rows() -> None:
    out_with_average = """
RMSD RMSD_R RMSD_T RMSD_Z RADGYR ORDER AXIS X Y Z P0_X P0_Y P0_Z
1.000 0.100 0.200 0.300 10.0 3 0.0 0.0 1.0 0.0 0.0 0.0
2.000 0.200 0.300 0.400 10.0 3 0.0 0.0 1.0 0.0 0.0 0.0
Average RMSD : 1.250
"""
    out_table_only = """
RMSD RMSD_R RMSD_T RMSD_Z RADGYR ORDER AXIS X Y Z P0_X P0_Y P0_Z
1.000 0.100 0.200 0.300 10.0 3 0.0 0.0 1.0 0.0 0.0 0.0
2.000 0.200 0.300 0.400 10.0 3 0.0 0.0 1.0 0.0 0.0 0.0
"""

    assert _parse_ananas_rmsd(out_with_average) == pytest.approx(1.25)
    assert _parse_ananas_rmsd(out_table_only) == pytest.approx(1.5)

    with pytest.raises(ValueError, match="Could not parse AnAnaS RMSD"):
        _parse_ananas_rmsd("no result")


def test_ananas_rmsd_loss_invokes_requested_dihedral_group_and_normalizes(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.txt"
    fake_ananas = tmp_path / "fake_ananas.sh"
    fake_ananas.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s %s\\n' \"$1\" \"$2\" >> \"$ANANAS_CAPTURE\"",
                "cat <<'EOF'",
                "RMSD RMSD_R RMSD_T RMSD_Z RADGYR ORDER AXIS X Y Z P0_X P0_Y P0_Z",
                "1.000 0.100 0.200 0.300 10.0 3 0.0 0.0 1.0 0.0 0.0 0.0",
                "Average RMSD : 1.000",
                "EOF",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_ananas.chmod(0o755)
    pdb_path = _write_ca_pdb(tmp_path / "d3.pdb", {"A": np.zeros((1, 3))})
    eval_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": pdb_path},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ANANAS_CAPTURE", str(capture_path))
        raw = ananas_rmsd_raw(
            eval_result,
            point_group="D",
            order=3,
            ananas_path=str(fake_ananas),
        )
        norm = ananas_rmsd_norm(
            eval_result,
            point_group="D3",
            ananas_path=str(fake_ananas),
            best=0.0,
            worst=2.0,
        )

    assert raw["AAAAAA"] == pytest.approx(1.0)
    assert norm["AAAAAA"] == pytest.approx(0.5)
    calls = capture_path.read_text(encoding="utf-8").strip().splitlines()
    assert calls == [f"{pdb_path} d3", f"{pdb_path} d3"]
    assert _ananas_group_arg("C", 3, chain_count=3) == "c3"
    assert _ananas_group_arg("D3", None, chain_count=6) == "d3"


def test_ananas_rmsd_loss_penalizes_no_detected_symmetry_without_crashing(tmp_path: Path) -> None:
    fake_ananas = tmp_path / "fake_ananas_no_symmetry.sh"
    fake_ananas.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "cat <<'EOF'",
                "t......... :",
                "============================== Timing : ==============================",
                "Detecting Symmetry.................................................... : 0.0001 s",
                "EOF",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_ananas.chmod(0o755)
    pdb_path = _write_ca_pdb(tmp_path / "d2_no_symmetry.pdb", {"A": np.zeros((1, 3))})
    eval_result = StandardScores(
        names=["AAAA"],
        plddt={"AAAA": [0.9]},
        complex_paths={"AAAA": pdb_path},
    )

    with pytest.warns(RuntimeWarning, match="AnAnaS did not report"):
        raw = ananas_rmsd_raw(
            eval_result,
            point_group="D",
            order=2,
            ananas_path=str(fake_ananas),
        )
    with pytest.warns(RuntimeWarning, match="AnAnaS did not report"):
        norm = ananas_rmsd_norm(
            eval_result,
            point_group="D",
            order=2,
            ananas_path=str(fake_ananas),
            best=0.0,
            worst=2.0,
        )

    assert math.isinf(raw["AAAA"])
    assert norm["AAAA"] == pytest.approx(0.0)


def test_ananas_path_resolves_from_environment(tmp_path: Path) -> None:
    fake_ananas = tmp_path / "fake_ananas.sh"
    fake_ananas.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "cat <<'EOF'",
                "Average RMSD : 0.250",
                "EOF",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_ananas.chmod(0o755)
    pdb_path = _write_ca_pdb(tmp_path / "d3.pdb", {"A": np.zeros((1, 3))})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("GH_ANANAS", str(fake_ananas))
        assert ananas_rmsd_wrapper(pdb_path, "d3") == pytest.approx(0.25)


def test_protomer_rmsd_losses_only_compare_shared_core(tmp_path: Path) -> None:
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
    rigid = chain_a[[0, 1, 2, 6, 7, 8]] @ _rotation_z(180.0) + np.array([20.0, 0.0, 0.0])
    chain_b = np.array(
        [
            rigid[0],
            rigid[1],
            rigid[2],
            [25.0, 9.0, 3.0],
            [26.0, 8.0, 4.0],
            [27.0, 9.0, 4.0],
            rigid[3],
            rigid[4],
            rigid[5],
        ],
        dtype=float,
    )
    pdb_path = _write_ca_pdb(tmp_path / "hetero_core.pdb", {"A": chain_a, "B": chain_b})
    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        iptm={"AB": [0.8]},
        ptm={"AB": [0.8]},
        complex_paths={"AB": pdb_path},
    )

    raw = protomer_rmsd_raw(eval_result)["AB"]
    norm = protomer_rmsd_norm(eval_result)["AB"]

    assert raw < 1e-6
    assert norm > 0.7


def test_protomer_rmsd_norm_uses_legacy_sigmoid_defaults_when_bounds_are_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": Path("/tmp/unused.pdb")},
    )
    monkeypatch.setattr(
        "astrolabe.losses.repeat_symmetry.protomer_rmsd_raw",
        lambda _eval_result: {"AB": 1.5},
    )

    norm = protomer_rmsd_norm(eval_result)["AB"]

    assert norm == pytest.approx(_normalize_good(1.5, m=2.0, s=2.0))
