from __future__ import annotations

from pathlib import Path

from astrolabe.core.structure_parser import create_structure_parser


def test_cif_parser_skips_nonpolymer_hetatm_in_chain_sequence(tmp_path: Path):
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_toy",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_asym_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.0 20.0 1 A 1",
                "ATOM 2 C CA . GLY A 1 2 ? 1.0 0.0 0.0 1.0 20.0 2 A 1",
                "HETATM 3 O O . HOH A 1 3 ? 2.0 0.0 0.0 1.0 20.0 3 A 1",
                "#",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parser = create_structure_parser(str(cif_path))
    parsed = parser.parse()

    assert parsed["A"][1] == "AG"
    assert parsed["A"][0] == [1, 2]
