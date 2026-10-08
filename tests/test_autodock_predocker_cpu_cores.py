from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes


def test_autodock_predocker_falls_back_when_sched_getaffinity_missing(
    tmp_path: Path, monkeypatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    from astrolabe.structure_processors.autodock import AutoDockGPUProcessor

    predocker = AutoDockGPUProcessor(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path=None,
        autogrid4_path=None,
    )

    assert isinstance(predocker.num_cpu_cores, int)
    assert predocker.num_cpu_cores == 4


def test_autodock_vina_uses_vina_result_output_name(
    tmp_path: Path, monkeypatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors import autodock as autodock_mod
    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    monkeypatch.setattr(
        autodock_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ok"),
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )
    out = predocker._run_vina_docking(
        str(tmp_path / "toy_protein.pdbqt"),
        str(tmp_path / "ligand.pdbqt"),
        {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
        },
    )

    assert out.endswith("toy_vina_result.pdbqt")


def test_autodock_vina_rejects_receptor_without_protein_suffix(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )

    with pytest.raises(ValueError, match="Expected '_protein' in receptor filename"):
        predocker._run_vina_docking(
            str(tmp_path / "toy_receptor.pdbqt"),
            str(tmp_path / "ligand.pdbqt"),
            {
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
            },
        )


def test_autodock_vina_workflow_accepts_path_cif_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    input_cif = tmp_path / "step_000_AB_model.cif"
    input_cif.write_text("data_mock\n", encoding="utf-8")
    converted_pdb = tmp_path / "step_000_AB_model.pdb"
    protein_pdb = tmp_path / "step_000_AB_model_protein.pdb"
    ligand_pdb = tmp_path / "step_000_AB_model_ligand.pdb"
    result_pdbqt = tmp_path / "step_000_AB_model_vina_result.pdbqt"
    final_pdb = tmp_path / "step_000_AB_model_vina_merged.pdb"

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )
    calls = {}

    def fake_convert(path):
        calls["convert_input"] = path
        return str(converted_pdb)

    def fake_separate(path):
        calls["separate_input"] = path
        return str(protein_pdb), str(ligand_pdb)

    monkeypatch.setattr(predocker, "_convert_cif_to_pdb", fake_convert)
    monkeypatch.setattr(predocker, "_separate_protein_ligand_with_prody", fake_separate)
    monkeypatch.setattr(
        predocker,
        "_determine_binding_site",
        lambda *_args, **_kwargs: {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
        },
    )
    monkeypatch.setattr(predocker, "_write_binding_site_metadata", lambda **_kwargs: None)
    monkeypatch.setattr(
        predocker,
        "_coordinate_transformed_docking_inputs",
        lambda protein_path, ligand_path: (protein_path, ligand_path),
    )
    monkeypatch.setattr(predocker, "_prepare_receptor_with_obabel", lambda *_args: None)
    monkeypatch.setattr(predocker, "_prepare_ligand_with_obabel", lambda *_args: None)
    monkeypatch.setattr(
        predocker, "_run_vina_docking", lambda *_args, **_kwargs: str(result_pdbqt)
    )
    monkeypatch.setattr(
        predocker,
        "_merge_ligand_with_receptor_with_retry",
        lambda *_args, **_kwargs: str(final_pdb),
    )
    monkeypatch.setattr(predocker, "_write_penalty_metadata", lambda **_kwargs: None)

    assert predocker._run_complete_vina_workflow(input_cif) == str(final_pdb)
    assert calls["convert_input"] == str(input_cif)
    assert calls["separate_input"] == str(converted_pdb)


def test_reference_pocket_alignment_binding_site_tracks_translated_current_structure(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference.pdb"
    cur_pdb = tmp_path / "current.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A   1       0.500   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A   2       4.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER A   3       8.000   0.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A   1      10.500  20.000  30.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A   2      14.000  20.000  30.000  1.00 10.00           C",
                "ATOM      4  CA  SER A   3      18.000  20.000  30.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain="A",
        reference_pocket_positions=[1, 2, 3],
        reference_box_padding=1.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(cur_pdb))
    assert box["center_x"] == pytest.approx(14.0)
    assert box["center_y"] == pytest.approx(20.0)
    assert box["center_z"] == pytest.approx(30.0)
    assert box["size_x"] == pytest.approx(10.0)
    assert box["size_y"] == pytest.approx(5.0)
    assert box["size_z"] == pytest.approx(5.0)


def test_reference_pocket_binding_site_uses_current_residue_box_without_alignment(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_bad.pdb"
    cur_pdb = tmp_path / "current_bad.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A   2       4.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  SER A   3       8.000   0.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A   2       0.000   4.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  SER A   3       0.000   9.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain="A",
        reference_pocket_positions=[1, 2, 3],
        current_structure_chain="A",
        reference_box_padding=1.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(cur_pdb))
    assert box["center_x"] == pytest.approx(0.0)
    assert box["center_y"] == pytest.approx(4.5)
    assert box["center_z"] == pytest.approx(0.0)
    assert box["size_x"] == pytest.approx(5.0)
    assert box["size_y"] == pytest.approx(11.0)
    assert box["size_z"] == pytest.approx(5.0)


def test_reference_pocket_binding_site_maps_reference_author_numbering_to_current_ordinal_positions(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_numbered.pdb"
    cur_pdb = tmp_path / "current_renumbered.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  38       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A  38       1.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A  39       4.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER A  40       8.000   0.000   0.000  1.00 10.00           C",
                "ATOM      5  CA  TYR A  41      12.000   0.000   0.000  1.00 10.00           C",
                "ATOM      6  CA  LEU A  42      16.000   0.000   0.000  1.00 10.00           C",
                "ATOM      7  CA  THR A  43      20.000   0.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      90.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A   1      91.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A   2      94.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER A   3      98.000   0.000   0.000  1.00 10.00           C",
                "ATOM      5  CA  TYR A   4      10.000  20.000  30.000  1.00 10.00           C",
                "ATOM      6  CA  LEU A   5      14.000  20.000  30.000  1.00 10.00           C",
                "ATOM      7  CA  THR A   6      18.000  20.000  30.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[41, 42, 43],
        reference_box_padding=1.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(cur_pdb))
    assert box["center_x"] == pytest.approx(14.0)
    assert box["center_y"] == pytest.approx(20.0)
    assert box["center_z"] == pytest.approx(30.0)
    assert box["size_x"] == pytest.approx(10.0)
    assert box["size_y"] == pytest.approx(5.0)
    assert box["size_z"] == pytest.approx(5.0)


def test_reference_pocket_binding_site_uses_design_sequence_for_inserted_domain_order(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(sequence="ACDHIKLMNPQRSTVWYEFG")},
    )

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_inserted_domain.pdb"
    cur_pdb = tmp_path / "current_all_ala_design_order.pdb"
    ref_residues = [
        (1, "ALA"),
        (2, "CYS"),
        (3, "ASP"),
        (4, "GLU"),
        (5, "PHE"),
        (6, "GLY"),
        (7, "HIS"),
        (8, "ILE"),
        (9, "LYS"),
        (10, "LEU"),
        (11, "MET"),
        (12, "ASN"),
        (13, "PRO"),
        (14, "GLN"),
        (15, "ARG"),
        (16, "SER"),
        (17, "THR"),
        (18, "VAL"),
        (19, "TRP"),
        (20, "TYR"),
    ]
    ref_pdb.write_text(
        "\n".join(
            f"ATOM  {serial:5d}  CA  {resname:>3s} A{seqid:4d}    {float(seqid):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           C"
            for serial, (seqid, resname) in enumerate(ref_residues, start=1)
        )
        + "\nEND\n",
        encoding="utf-8",
    )
    current_lines = []
    for seqid in range(1, 21):
        if seqid in {4, 5, 6}:
            x = 10.0 + float(seqid - 4) * 4.0
            y = 20.0
            z = 30.0
        else:
            x = 100.0 + float(seqid)
            y = 0.0
            z = 0.0
        current_lines.append(
            f"ATOM  {seqid:5d}  CA  ALA A{seqid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C"
        )
    cur_pdb.write_text("\n".join(current_lines) + "\nEND\n", encoding="utf-8")

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[7, 8, 9],
        reference_box_padding=1.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(cur_pdb))

    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["mapped_current_positions"] == [4, 5, 6]
    assert box["center_x"] == pytest.approx(14.0)
    assert box["center_y"] == pytest.approx(20.0)
    assert box["center_z"] == pytest.approx(30.0)


def test_reference_pocket_binding_site_metadata_records_reference_to_current_mapping(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.set_variable("step", 1)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_numbered_meta.pdb"
    cur_pdb = tmp_path / "current_renumbered_meta.pdb"
    input_pdb = tmp_path / "step_000_AB_model.pdb"
    input_pdb.write_text("END\n", encoding="utf-8")
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  38       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A  38       1.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A  39       4.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER A  40       8.000   0.000   0.000  1.00 10.00           C",
                "ATOM      5  CA  TYR A  41      12.000   0.000   0.000  1.00 10.00           C",
                "ATOM      6  CA  LEU A  42      16.000   0.000   0.000  1.00 10.00           C",
                "ATOM      7  CA  THR A  43      20.000   0.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      90.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CB  ALA A   1      91.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A   2      94.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER A   3      98.000   0.000   0.000  1.00 10.00           C",
                "ATOM      5  CA  TYR A   4      10.000  20.000  30.000  1.00 10.00           C",
                "ATOM      6  CA  LEU A   5      14.000  20.000  30.000  1.00 10.00           C",
                "ATOM      7  CA  THR A   6      18.000  20.000  30.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[41, 42, 43],
        reference_box_padding=1.0,
    )

    binding_site = predocker._calculate_binding_site_from_aligned_reference_pocket(
        str(cur_pdb)
    )
    predocker._write_binding_site_metadata(
        input_file_path=str(input_pdb),
        binding_site=binding_site,
    )

    metadata_path = tmp_path / "predocker" / "autodock_binding_site_step_001_AB.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    mappings = payload["details"]["residue_mappings"]
    assert [row["reference_author_position"] for row in mappings] == [41, 42, 43]
    assert [row["current_seqid"] for row in mappings] == [4, 5, 6]
    assert [row["current_ordinal_position"] for row in mappings] == [4, 5, 6]


def test_reference_pocket_binding_site_accepts_chain_qualified_positions(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(sequence="AG"),
            "E": SimpleNamespace(sequence="ST"),
        },
    )

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_interface.pdb"
    cur_pdb = tmp_path / "current_interface.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A  20       4.000   0.000   0.000  1.00 10.00           C",
                "TER",
                "ATOM      3  CA  SER E 101       0.000   8.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  THR E 102       4.000   8.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      10.000   2.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A   2      14.000   2.000   0.000  1.00 10.00           C",
                "TER",
                "ATOM      3  CA  SER B   1      10.000  12.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  THR B   2      14.000  12.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain={"A": "A", "E": "E"},
        current_structure_chain={"A": "A", "E": "B"},
        reference_pocket_positions_by_chain={"A": [10], "E": [102]},
        reference_box_padding=1.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(
        str(cur_pdb),
        oligo_name="AEB",
    )

    assert box["center_x"] == pytest.approx(12.0)
    assert box["center_y"] == pytest.approx(7.0)
    assert box["center_z"] == pytest.approx(0.0)
    assert box["size_x"] == pytest.approx(6.0)
    assert box["size_y"] == pytest.approx(12.0)
    assert box["size_z"] == pytest.approx(5.0)
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["mapped_current_positions_by_chain"] == {"A": [1], "E": [2]}
    mapping_summary = [
        (
            row["pocket_key"],
            row["reference_author_position"],
            row["current_chain_id"],
            row["current_seqid"],
        )
        for row in details["residue_mappings"]
    ]
    assert mapping_summary == [("A", 10, "A", 1), ("E", 102, "B", 2)]


def test_reference_pocket_binding_site_ordinal_mode_maps_mutated_interface_chains(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    ref_pdb = tmp_path / "reference_mutated_interface.pdb"
    cur_pdb = tmp_path / "current_mutated_interface.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A  20       4.000   0.000   0.000  1.00 10.00           C",
                "TER",
                "ATOM      3  CA  SER E 101       0.000   8.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  THR E 102       4.000   8.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cur_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  LEU A   1      10.000   2.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  ILE A   2      14.000   2.000   0.000  1.00 10.00           C",
                "TER",
                "ATOM      3  CA  ASN B   1      10.000  12.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  TYR B   2      14.000  12.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(ref_pdb),
        reference_structure_chain={"A": "A", "E": "E"},
        current_structure_chain={"A": "A", "E": "B"},
        reference_pocket_positions_by_chain={"A": [10], "E": [102]},
        reference_box_padding=1.0,
        reference_alignment_mode="ordinal",
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(cur_pdb))

    assert box["center_x"] == pytest.approx(12.0)
    assert box["center_y"] == pytest.approx(7.0)
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["reference_alignment_mode"] == "ordinal"
    assert details["sequence_alignment_mismatch_counts_by_chain"] == {"A": 2, "E": 2}
    assert details["mapped_current_positions_by_chain"] == {"A": [1], "E": [2]}


def test_autodock_vina_run_returns_penalty_artifact_and_preserves_input_on_soft_alignment_failure(
    tmp_path: Path, monkeypatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.set_variable("step", 1)

    from astrolabe.structure_processors.autodock import (
        AutoDockVinaLigandPoseRescue,
        DockingPenaltyError,
    )

    input_path = tmp_path / "step_000_AB_model.pdb"
    input_path.write_text("END\n", encoding="utf-8")
    protein_path = tmp_path / "step_000_AB_model_protein.pdb"
    protein_path.write_text("END\n", encoding="utf-8")
    ligand_path = tmp_path / "step_000_AB_model_ligand.pdb"
    ligand_path.write_text("END\n", encoding="utf-8")

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
    )

    monkeypatch.setattr(
        predocker,
        "_separate_protein_ligand_with_prody",
        lambda _path: (str(protein_path), str(ligand_path)),
    )
    monkeypatch.setattr(
        predocker,
        "_determine_binding_site",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DockingPenaltyError(
                "Local pocket alignment RMSD exceeds threshold: 14.699 > 3.000",
                reason="binding_site_unavailable",
            )
        ),
    )

    out = predocker.run(str(input_path))

    output_path = Path(out)
    assert output_path != input_path
    assert output_path.parent == tmp_path / "predocker"
    assert output_path.exists()
    assert input_path.exists()
    assert os.path.samefile(output_path, input_path)
    metadata_path = tmp_path / "predocker" / "autodock_penalty_step_001_AB.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["penalty_active"] is True
    assert payload["reason"] == "binding_site_unavailable"
    assert "RMSD exceeds threshold" in payload["message"]


def test_autodock_vina_penalty_artifact_copies_when_hardlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    input_path = tmp_path / "step_000_AB_model.pdb"
    input_path.write_text("MODEL\nEND\n", encoding="utf-8")
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
    )

    monkeypatch.setattr(
        os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(30, "Read-only file system")
        ),
    )

    output_path = Path(predocker._materialize_penalty_artifact(str(input_path)))

    assert output_path.exists()
    assert output_path != input_path
    assert output_path.read_text(encoding="utf-8") == input_path.read_text(
        encoding="utf-8"
    )


def test_autodock_vina_filter_ranking_without_metrics_samples_available_pose(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_com_diff=False,
        rank_buried_contrib=False,
    )

    inside_path = tmp_path / "inside_pose.pdb"
    inside_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "HETATM    2  C1  LIG Z   1       1.000   0.000   0.000  1.00 10.00           C",
                "HETATM    3  C2  LIG Z   1       2.000   0.000   0.000  1.00 10.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    selected = predocker._filter_ranking(
        {
            0: (str(inside_path), "Z"),
            1: (str(inside_path), "Z"),
        },
    )

    assert selected == str(inside_path)


def test_interface_binding_site_uses_dynamic_two_chain_interface_box(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import (
        DEFAULT_BINDING_SITE,
        AutoDockVinaLigandPoseRescue,
    )

    protein = tmp_path / "two_chain_interface.pdb"
    protein.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA B   1       4.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  ALA B   2       4.000   4.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  ALA A   2       0.000   4.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="interface",
        binding_site_padding=1.0,
        binding_site_coverage=100,
    )

    box = predocker._calculate_binding_site_from_protein_interface(
        str(protein),
        padding=1.0,
        coverage_percent=100,
    )

    assert box != DEFAULT_BINDING_SITE
    assert box["size_x"] >= 5.0
    assert box["size_y"] >= 5.0
    assert box["size_z"] >= 5.0
    assert predocker._last_binding_site_debug["chain_ids"] == ["A", "B"]
    assert predocker._last_binding_site_debug["interface_coordinate_count"] == 4


def test_interface_binding_site_strategy_passes_configured_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="interface",
        binding_site_padding=3.25,
        binding_site_coverage=77,
    )
    calls = {}

    def fake_interface_box(path, *, padding, coverage_percent):
        calls["path"] = path
        calls["padding"] = padding
        calls["coverage_percent"] = coverage_percent
        return {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "size_x": 5.0,
            "size_y": 5.0,
            "size_z": 5.0,
        }

    monkeypatch.setattr(
        predocker,
        "_calculate_binding_site_from_protein_interface",
        fake_interface_box,
    )

    box = predocker._determine_binding_site(str(tmp_path / "protein.pdb"))

    assert box["size_x"] == 5.0
    assert calls == {
        "path": str(tmp_path / "protein.pdb"),
        "padding": 3.25,
        "coverage_percent": 77.0,
    }


def test_interface_binding_site_unavailable_becomes_docking_penalty(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError

    protein = tmp_path / "one_chain.pdb"
    protein.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="interface",
    )

    with pytest.raises(DockingPenaltyError, match="at least two chains") as exc_info:
        predocker._calculate_binding_site_from_protein_interface(str(protein))

    assert exc_info.value.reason == "binding_site_unavailable"
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["strategy"] == "interface"
    assert details["reason"] == "binding_site_unavailable"
    assert details["chain_ids"] == ["A"]


def _write_alpha_shape_pocket_pose(
    path: Path,
    ligand_coords: list[tuple[float, float, float]],
    *,
    omit_cb_positions: set[int] | None = None,
) -> Path:
    omit_cb_positions = omit_cb_positions or set()
    residue_atoms = [
        (1, "CA", "ALA", (-1.0, -1.0, -1.0)),
        (1, "CB", "ALA", (0.0, 0.0, 0.0)),
        (1, "CG", "ALA", (25.0, 25.0, 25.0)),
        (2, "CA", "VAL", (3.0, -1.0, -1.0)),
        (2, "CB", "VAL", (4.0, 0.0, 0.0)),
        (3, "CA", "SER", (-1.0, 3.0, -1.0)),
        (3, "CB", "SER", (0.0, 4.0, 0.0)),
        (4, "CA", "THR", (-1.0, -1.0, 3.0)),
        (4, "CB", "THR", (0.0, 0.0, 4.0)),
    ]
    lines = []
    serial = 1
    for resseq, atom_name, resname, (x, y, z) in residue_atoms:
        if atom_name == "CB" and resseq in omit_cb_positions:
            continue
        lines.append(
            f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C"
        )
        serial += 1
    lines.append("TER")
    for idx, (x, y, z) in enumerate(ligand_coords, start=1):
        lines.append(
            f"HETATM{100 + idx:5d}  C{idx:<2d} LIG Z   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_rotated_cb_pocket_pose(path: Path) -> Path:
    c = math.sqrt(0.5)
    local_cb_coords = [
        (-6.0, -1.0, -0.5),
        (6.0, -1.0, 0.5),
        (-6.0, 1.0, 0.5),
        (6.0, 1.0, -0.5),
    ]
    resnames = ["ALA", "VAL", "SER", "THR"]
    lines = []
    serial = 1
    for resseq, (resname, (x_local, y_local, z_local)) in enumerate(
        zip(resnames, local_cb_coords),
        start=1,
    ):
        x = x_local * c - y_local * c + 10.0
        y = x_local * c + y_local * c - 5.0
        z = z_local + 2.0
        ca = (x - 0.4, y - 0.4, z - 0.4)
        for atom_name, (ax, ay, az) in [("CA", ca), ("CB", (x, y, z))]:
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A{resseq:4d}    "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00 10.00           C"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_rotated_cb_pocket_complex(path: Path) -> Path:
    _write_rotated_cb_pocket_pose(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.pop()
    lines.append(
        "HETATM  900  C1  LIG Z   1      10.000  -5.000   2.000  1.00 10.00           C"
    )
    lines.append(end)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_single_ligand_atom(path: Path, coord: tuple[float, float, float]) -> Path:
    x, y, z = coord
    path.write_text(
        "\n".join(
            [
                f"HETATM    1  C1  LIG Z   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _first_atom_coord(path: Path) -> tuple[float, float, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            return (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    raise AssertionError(f"no atom coordinates in {path}")


def _pdb_resnames(path: Path) -> list[str]:
    return [
        line[17:20].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def test_autodock_vina_collect_merged_results_uses_converted_pdb_and_ligand_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN, sequence="AA", name="A"),
            "B": SimpleNamespace(type=ProtomerTypes.CCD, sequence="B49", name="B"),
        },
    )
    global_config.init_const("ligand_resnames", {"B": "Z49", "B49": "Z49"})

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    receptor = tmp_path / "step_001_AB_model_protein.pdb"
    receptor.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result_pdbqt = tmp_path / "step_001_AB_model_vina_result.pdbqt"
    result_pdbqt.write_text("REMARK should not be parsed\n", encoding="utf-8")
    converted = tmp_path / "step_001_AB_model_vina_result.pdb"
    converted.write_text(
        "\n".join(
            [
                "MODEL        1",
                "MODEL 1",
                "HETATM    1  C4  B49 Z   1       1.000   0.000   0.000  1.00 10.00           C",
                "ENDMDL",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_vina_order=True,
    )
    monkeypatch.setattr(predocker, "convert_pdbqt_results_to_pdb", lambda path: str(converted))

    results = predocker._collect_merged_ligand_results(
        str(result_pdbqt),
        str(receptor),
    )

    merged_path, ligand_chain = results[0]
    assert ligand_chain == "B"
    assert Path(merged_path).name == "step_001_AB_model_vina_result_merged_0.pdb"
    assert _pdb_resnames(Path(merged_path)) == ["ALA", "Z49"]


def test_reference_pocket_binding_site_can_use_tight_cb_alpha_shape_box(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    reference = _write_alpha_shape_pocket_pose(tmp_path / "reference_alpha_box.pdb", [])
    current = _write_alpha_shape_pocket_pose(tmp_path / "current_alpha_box.pdb", [])
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        reference_box_geometry="cb_alpha_shape",
        reference_box_padding=0.5,
        reference_min_box_size=0.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(current))

    assert box["center_x"] == pytest.approx(2.0)
    assert box["center_y"] == pytest.approx(2.0)
    assert box["center_z"] == pytest.approx(2.0)
    assert box["size_x"] == pytest.approx(5.0)
    assert box["size_y"] == pytest.approx(5.0)
    assert box["size_z"] == pytest.approx(5.0)
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["reference_box_geometry"] == "cb_alpha_shape"
    assert details["current_box_coordinate_count"] == 4


def test_reference_pocket_binding_site_cb_alpha_shape_box_falls_back_to_ca_when_cb_missing(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    reference = _write_alpha_shape_pocket_pose(tmp_path / "reference_alpha_box.pdb", [])
    current = _write_alpha_shape_pocket_pose(
        tmp_path / "current_alpha_box_missing_cb.pdb",
        [],
        omit_cb_positions={1},
    )
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        reference_box_geometry="cb_alpha_shape",
        reference_box_padding=0.5,
        reference_min_box_size=0.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(current))

    assert box["size_x"] > 0.0
    assert box["size_y"] > 0.0
    assert box["size_z"] > 0.0
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["current_box_coordinate_count"] == 4
    assert details["reference_box_selected_volume"] <= details[
        "reference_box_axis_aligned_volume"
    ]


def test_reference_pocket_binding_site_ordinal_mode_maps_mutated_current_residues(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError

    reference = _write_alpha_shape_pocket_pose(tmp_path / "reference_mutation_map.pdb", [])
    current = _write_alpha_shape_pocket_pose(tmp_path / "current_mutation_map.pdb", [])
    current.write_text(
        current.read_text(encoding="utf-8")
        .replace(" ALA ", " LEU ")
        .replace(" VAL ", " ILE ")
        .replace(" SER ", " ASN ")
        .replace(" THR ", " TYR "),
        encoding="utf-8",
    )
    common_kwargs = dict(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        reference_box_geometry="cb_alpha_shape",
        reference_box_padding=0.5,
        reference_min_box_size=0.0,
    )

    strict_predocker = AutoDockVinaLigandPoseRescue(**common_kwargs)
    with pytest.raises(DockingPenaltyError, match="residue mismatches"):
        strict_predocker._calculate_binding_site_from_aligned_reference_pocket(str(current))

    predocker = AutoDockVinaLigandPoseRescue(
        **common_kwargs,
        reference_alignment_mode="ordinal",
    )
    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(current))

    assert box["size_x"] == pytest.approx(5.0)
    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["reference_alignment_mode"] == "ordinal"
    assert details["sequence_alignment_mismatch_count"] == 4
    assert details["mapped_current_positions"] == [1, 2, 3, 4]
    assert [row["current_resname"] for row in details["residue_mappings"]] == [
        "LEU",
        "ILE",
        "ASN",
        "TYR",
    ]


def test_reference_pocket_binding_site_rotates_cb_alpha_shape_box_when_tighter(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    reference = _write_rotated_cb_pocket_pose(tmp_path / "reference_rotated_box.pdb")
    current = _write_rotated_cb_pocket_pose(tmp_path / "current_rotated_box.pdb")
    ligand = _write_single_ligand_atom(tmp_path / "ligand.pdb", (10.0, -5.0, 2.0))
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        reference_box_geometry="cb_alpha_shape",
        reference_box_padding=0.0,
        reference_min_box_size=0.0,
    )

    box = predocker._calculate_binding_site_from_aligned_reference_pocket(str(current))

    details = predocker._last_binding_site_debug
    assert details is not None
    assert details["reference_box_orientation"] == "pca"
    assert details["reference_box_coordinate_frame"] == "pca_local"
    assert details["reference_box_selected_volume"] < (
        0.5 * details["reference_box_axis_aligned_volume"]
    )
    assert sorted([box["size_x"], box["size_y"], box["size_z"]]) == pytest.approx(
        [1.0, 2.0, 12.0],
        abs=0.15,
    )

    rotated_protein, rotated_ligand = predocker._coordinate_transformed_docking_inputs(
        str(current),
        str(ligand),
    )
    assert rotated_protein != str(current)
    assert rotated_ligand != str(ligand)
    roundtrip = tmp_path / "ligand_roundtrip.pdb"
    predocker._apply_coordinate_transform_to_pdb(
        rotated_ligand,
        str(roundtrip),
        predocker._last_binding_site_transform,
        inverse=True,
    )
    assert _first_atom_coord(roundtrip) == pytest.approx((10.0, -5.0, 2.0), abs=1e-3)


def test_autodock_vina_workflow_uses_rotated_cb_alpha_shape_box_and_restores_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    reference = _write_rotated_cb_pocket_pose(tmp_path / "reference_rotated_box.pdb")
    current = _write_rotated_cb_pocket_complex(tmp_path / "current_complex.pdb")
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        binding_site_strategy="aligned_reference_pocket",
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        reference_box_geometry="cb_alpha_shape",
        reference_box_padding=0.0,
        reference_min_box_size=0.0,
        rank_vina_order=True,
        rank_buried_fraction=False,
    )
    calls = {}
    monkeypatch.setattr(
        predocker,
        "_prepare_receptor_with_obabel",
        lambda protein_path, receptor_path: calls.setdefault("protein_path", protein_path),
    )
    monkeypatch.setattr(
        predocker,
        "_prepare_ligand_with_obabel",
        lambda ligand_path, ligand_pdbqt_path: calls.setdefault("ligand_path", ligand_path),
    )
    monkeypatch.setattr(predocker, "convert_pdbqt_results_to_pdb", lambda path: path)

    def fake_vina(receptor_pdbqt_path, ligand_pdbqt_path, binding_site):
        calls["receptor_pdbqt_path"] = receptor_pdbqt_path
        calls["ligand_pdbqt_path"] = ligand_pdbqt_path
        calls["binding_site"] = dict(binding_site)
        output = Path(receptor_pdbqt_path).with_name(
            Path(receptor_pdbqt_path).name.replace("_protein", "_vina_result")
        )
        output.write_text(
            "\n".join(
                [
                    "MODEL        1",
                    (
                        "HETATM    1  C1  LIG Z   1    "
                        f"{binding_site['center_x']:8.3f}"
                        f"{binding_site['center_y']:8.3f}"
                        f"{binding_site['center_z']:8.3f}"
                        "  1.00 10.00           C"
                    ),
                    "ENDMDL",
                    "END",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return str(output)

    monkeypatch.setattr(predocker, "_run_vina_docking", fake_vina)

    selected = Path(predocker._run_complete_vina_workflow(str(current)))

    assert "_oriented_box" in calls["protein_path"]
    assert "_oriented_box" in calls["ligand_path"]
    assert selected.name.endswith("_original_frame.pdb")
    assert _first_atom_coord(selected) == pytest.approx(
        _first_atom_coord(current),
        abs=1e-3,
    )
    assert predocker._last_binding_site_debug["reference_box_orientation"] == "pca"


def test_autodock_vina_alpha_shape_pose_filter_selects_first_passing_pose(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue

    reference = _write_alpha_shape_pocket_pose(tmp_path / "reference_alpha_filter.pdb", [])
    outside = _write_alpha_shape_pocket_pose(
        tmp_path / "outside_pose.pdb",
        [(8.0, 8.0, 8.0), (9.0, 8.0, 8.0)],
    )
    inside = _write_alpha_shape_pocket_pose(
        tmp_path / "inside_pose.pdb",
        [(0.5, 0.5, 0.5), (1.0, 0.5, 0.5)],
        omit_cb_positions={1},
    )
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_com_diff=False,
        rank_buried_contrib=False,
        rank_vina_order=True,
        rank_alpha_shape_occupancy=True,
        rank_alpha_shape_min_occupancy=0.8,
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        alpha_radius=10.0,
    )

    selected = predocker._filter_ranking(
        {
            0: (str(outside), "Z"),
            1: (str(inside), "Z"),
        }
    )

    assert selected == str(inside)
    assert predocker._last_pose_candidates == [str(inside)]


def test_autodock_vina_alpha_shape_pose_filter_penalizes_when_no_pose_passes(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError

    reference = _write_alpha_shape_pocket_pose(tmp_path / "reference_alpha_fail.pdb", [])
    outside = _write_alpha_shape_pocket_pose(
        tmp_path / "outside_pose_only.pdb",
        [(8.0, 8.0, 8.0), (9.0, 8.0, 8.0)],
    )
    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_com_diff=False,
        rank_buried_contrib=False,
        rank_alpha_shape_occupancy=True,
        rank_alpha_shape_min_occupancy=0.8,
        reference_structure_path=str(reference),
        reference_structure_chain="A",
        current_structure_chain="A",
        reference_pocket_positions=[1, 2, 3, 4],
        alpha_radius=10.0,
    )

    with pytest.raises(DockingPenaltyError, match="No Vina pose satisfied"):
        predocker._filter_ranking({0: (str(outside), "Z")})


def test_autodock_vina_retries_alpha_shape_filter_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError

    predocker = AutoDockVinaLigandPoseRescue(
        obabel_path="obabel",
        autodock_gpu_path=None,
        vina_path="vina",
        autogrid4_path=None,
        rank_buried_fraction=False,
        rank_com_diff=False,
        rank_buried_contrib=False,
        rank_alpha_shape_occupancy=True,
        rank_alpha_shape_retry_attempts=2,
        rank_alpha_shape_retry_exhaustiveness=16,
        rank_alpha_shape_retry_num_modes=20,
    )
    initial_pdbqt = str(tmp_path / "toy_vina_result.pdbqt")
    receptor_pdb = str(tmp_path / "toy_protein.pdb")
    receptor_pdbqt = str(tmp_path / "toy_protein.pdbqt")
    ligand_pdbqt = str(tmp_path / "toy_ligand.pdbqt")
    binding_site = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
    }
    retry_calls = []

    def fail_initial_merge(*_args, **_kwargs):
        raise DockingPenaltyError(
            "No Vina pose satisfied the alpha-shape pocket occupancy filter",
            reason="alpha_shape_pose_filter_failed",
            details={"alpha_shape_occupancy_by_pose": {"0": 0.7}},
        )

    def fake_vina(*_args, attempt=0, exhaustiveness=None, num_modes=None, **_kwargs):
        retry_calls.append(
            {
                "attempt": attempt,
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
            }
        )
        return str(tmp_path / f"toy_vina_result_retry{attempt}.pdbqt")

    def fake_collect(path, *_args, sample_id_offset=0, **_kwargs):
        stem = Path(path).stem
        return {
            sample_id_offset: (
                str(tmp_path / f"{stem}_merged_0.pdb"),
                "Z",
            )
        }

    def fake_filter(results):
        assert sorted(results) == [0, 1, 2]
        return results[2][0]

    monkeypatch.setattr(predocker, "_merge_ligand_with_receptor", fail_initial_merge)
    monkeypatch.setattr(predocker, "_run_vina_docking", fake_vina)
    monkeypatch.setattr(predocker, "_collect_merged_ligand_results", fake_collect)
    monkeypatch.setattr(predocker, "_filter_ranking", fake_filter)

    selected = predocker._merge_ligand_with_receptor_with_retry(
        initial_pdbqt,
        receptor_pdb,
        receptor_pdbqt_path=receptor_pdbqt,
        ligand_pdbqt_path=ligand_pdbqt,
        binding_site=binding_site,
    )

    assert selected.endswith("toy_vina_result_retry2_merged_0.pdb")
    assert retry_calls == [
        {"attempt": 1, "exhaustiveness": 16, "num_modes": 20},
        {"attempt": 2, "exhaustiveness": 16, "num_modes": 20},
    ]
