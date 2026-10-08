from __future__ import annotations

from pathlib import Path

import astrolabe.cli as cli


REPO_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def _run(argv: list[str]) -> int:
    parser = cli._build_parser()
    ns = parser.parse_args(argv)
    handler = getattr(ns, "_handler", None)
    assert handler is not None
    return int(handler(ns))


def test_benchmark_binder_v1_4_x_mask_dry_run(capsys) -> None:
    rc = _run(
        [
            "--config-dir",
            str(REPO_CONFIG_ROOT),
            "run",
            "--task",
            "benchmark_binder_v1_4_x_mask",
            "--dry-run",
            "env=env",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "mode: x_mix" in out
    assert "use_metropolis_acceptance: false" in out
    assert "reinitialize_random_segments_with_dplm" not in out
    assert "_type: XMix" in out
    assert "residues: X" in out
    assert "redesign_no_x:" in out
    assert "mutators:\n    - x_mask" in out


def test_benchmark_binder_omni_optimize_refine_dry_run(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.cif"
    reference.write_text("data_reference\n#\n", encoding="utf-8")
    overlay = tmp_path / "cre_overlay.yaml"
    overlay.write_text(
        "\n".join(
            [
                "# @package _global_",
                "task_name: omni_refine_dry",
                f"out_dir: {tmp_path / 'out'}",
                "steps: 2",
                "seqs:",
                "  A: ACDEFGHIKLMNPQRSTVWY",
                "  B: +ACDEFGHIKLMNPQRSTVWY",
                "homologies:",
                "  B: null",
                "named_files:",
                f"  structure_start_ref: {reference}",
                "structure_start:",
                "  mode: repredict_gate",
                "  gate_timing: [step0, pre_accept]",
                "  aggregate: min",
                "  references:",
                "    AB:",
                "      structure_alias: structure_start_ref",
                "      chain_map:",
                "        A: A",
                "        B: B",
                "  reference_tmscore:",
                "    enabled: true",
                "    threshold: 0.87",
                "  hotspot_gate:",
                "    enabled: false",
                "    expression: pair_hotspot_support",
                "    floor: 0.25",
                "    reference_fraction: 0.8",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run(
        [
            "--config-dir",
            str(REPO_CONFIG_ROOT),
            "run",
            "--task",
            "benchmark_binder_omni_optimize_refine",
            "--overlay",
            str(overlay),
            "--dry-run",
            "env=env",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "allow_mmcif_pdb_fallback: true" in out
    assert "use_metropolis_acceptance: true" in out
    assert "stack_pre_update: true" in out
    assert "rosetta_minimize:" in out
    assert "_type: bindcraft_interface_dG" in out
    assert "surface_high_sap(threshold=2.5)" in out
    assert "sap_per_residue:" in out
    assert "reference_penalty:" in out
    assert "mutation_num: 50%" in out
    assert "acceptance_gate: null" in out
    assert "MCMC_reference_probe:" not in out
    assert "low_confidence_repair:" not in out


def test_benchmark_ligand_omni_optimize_refine_dry_run(tmp_path, capsys) -> None:
    overlay = tmp_path / "ligand_omni_overlay.yaml"
    overlay.write_text(
        "\n".join(
            [
                "# @package _global_",
                "task_name: ligand_omni_refine_dry",
                f"out_dir: {tmp_path / 'out'}",
                "steps: 2",
                "seqs:",
                "  A: ACDEFGHIKLMNPQRSTVWY",
                "  B: FAD",
                "types:",
                "  B: c",
                "homologies:",
                "  A: null",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run(
        [
            "--config-dir",
            str(REPO_CONFIG_ROOT),
            "run",
            "--task",
            "benchmark_ligand_omni_optimize_refine",
            "--overlay",
            str(overlay),
            "--dry-run",
            "env=env",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "allow_mmcif_pdb_fallback: true" in out
    assert "rosetta_minimize:" in out
    assert "stack_pre_update: false" in out
    assert "autodock_vina_box:" not in out
    assert "LigandMPNN_pocket_unsat_repair:" in out
    assert "LigandMPNN_quality_repair:" in out
    assert "LigandMPNN_surface_cleanup:" in out
    assert "model_type: soluble_mpnn" in out
    assert "ligand_unsat(ligand_targets=['B'], distance_min=3.5, distance_max=7.5)" in out
    assert "ligand_contact(ligand_targets=['B'], distance_max=4.5)" in out
    assert "surface_high_sap(threshold=2.5)" in out
    assert "buried_unsat(threshold=1.0, ignore_surface_res=True, method='buried_polar_proxy')" in out
    assert "cavity_neighbor(holes_threshold=40.0, neighbor_distance=5.0, method='packstat')" in out
    assert "sap_per_residue:" in out
    assert "binder_rg_ratio:" in out
    assert "pocket_quality:" in out
    assert "protein_quality_penalty:" in out
    assert "mutator_weights:" in out
    assert "    - 0.35" in out
    assert "    - 0.45" in out
    assert "    - 0.2" in out
    assert "mutation_num: 50%" in out
    assert "bindcraft_interface_dG" not in out


def test_benchmark_ligand_omni_optimize_refine_autodock_dry_run(tmp_path, capsys) -> None:
    overlay = tmp_path / "ligand_omni_autodock_overlay.yaml"
    overlay.write_text(
        "\n".join(
            [
                "# @package _global_",
                "task_name: ligand_omni_autodock_dry",
                f"out_dir: {tmp_path / 'out'}",
                "steps: 2",
                "seqs:",
                "  A: ACDEFGHIKLMNPQRSTVWY",
                "  B: FAD",
                "types:",
                "  B: c",
                "homologies:",
                "  A: null",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run(
        [
            "--config-dir",
            str(REPO_CONFIG_ROOT),
            "run",
            "--task",
            "benchmark_ligand_omni_optimize_refine_autodock",
            "--overlay",
            str(overlay),
            "--dry-run",
            "env=env",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "autodock_vina_box:" in out
    assert "_condition: ligand_burial < 0.7" in out
    assert "rank_alpha_shape_occupancy: false" in out
    assert "rank_alpha_shape_retry_attempts: 0" in out
    assert "LigandMPNN_pocket_unsat_repair:" in out
    assert "LigandMPNN_quality_repair:" in out
    assert "predocker:" in out
    assert "    - autodock_vina_box" in out
