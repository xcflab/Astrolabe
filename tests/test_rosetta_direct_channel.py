from __future__ import annotations

from pathlib import Path

from astrolabe.posthoc.rosetta_direct_channel import (
    _template_link_lines,
    build_pocket_fixed_screen_plan,
    normalize_target_aas,
)


def _write_small_channel_input(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C",
                "ATOM      3  N   CYS A   2      12.000  10.000  10.000  1.00 20.00           N",
                "ATOM      4  CA  CYS A   2      13.000  10.000  10.000  1.00 20.00           C",
                "ATOM      5  SG  CYS A   2      14.000  10.000  10.000  1.00 20.00           S",
                "ATOM      6  N   GLY A   3      15.000  10.000  10.000  1.00 20.00           N",
                "ATOM      7  CA  GLY A   3      16.000  10.000  10.000  1.00 20.00           C",
                "ATOM      8  N   SER A   4      25.000  10.000  10.000  1.00 20.00           N",
                "ATOM      9  CA  SER A   4      26.000  10.000  10.000  1.00 20.00           C",
                "HETATM   10  C1  LIG L 101      14.500  11.200  10.000  1.00 20.00           C",
                "HETATM   11  O1  LIG L 101      15.500  11.200  10.000  1.00 20.00           O",
                "LINK         SG  CYS A   2                 C1  LIG L 101     1555   1555  1.39",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_normalize_target_aas_accepts_mixed_token_styles() -> None:
    assert normalize_target_aas("ala|thr|ser|gly") == ("A", "T", "S", "G")
    assert normalize_target_aas(["A", "SER", "gly", "A"]) == ("A", "S", "G")


def test_build_pocket_fixed_screen_plan_marks_ligand_contact_mutable_positions(
    tmp_path: Path,
) -> None:
    structure = _write_small_channel_input(tmp_path / "channel_input.pdb")
    mutable_positions = tmp_path / "mutable_positions.txt"
    mutable_positions.write_text("2-4\n", encoding="utf-8")

    plan = build_pocket_fixed_screen_plan(
        structure_path=structure,
        protein_chain_id="A",
        mutable_positions_path=mutable_positions,
        ligand_chain_id="L",
        ligand_resname="LIG",
        ligand_resseq=101,
        target_aas="ala|ser",
        contact_distance=5.0,
    )

    assert plan["mutable_positions"] == [2, 3, 4]
    assert plan["protected_positions"] == [2, 3]
    assert plan["design_positions"] == [4]
    assert plan["protected_residue_ids"] == ["A2", "A3"]
    assert [row["name"] for row in plan["variants"]] == [
        "pocket_fixed_all_ala",
        "pocket_fixed_all_ser",
    ]
    assert plan["variants"][0]["target_sequence"] == "ACGA"
    assert plan["variants"][1]["target_sequence"] == "ACGS"


def test_template_link_lines_preserve_covalent_connection_records(tmp_path: Path) -> None:
    structure = _write_small_channel_input(tmp_path / "channel_input.pdb")

    assert _template_link_lines(structure) == [
        "LINK         SG  CYS A   2                 C1  LIG L 101     1555   1555  1.39"
    ]
