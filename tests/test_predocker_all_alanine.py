from __future__ import annotations

from pathlib import Path

from astrolabe.core import global_config
from astrolabe.structure_processors.all_alanine import AllAlanineStructureProcessor


def test_all_alanine_preserves_hetero_ligand_atoms(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)

    input_pdb = tmp_path / "protein_ligand.pdb"
    input_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   TRP A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  TRP A   1       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   TRP A   1       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  O   TRP A   1       2.500   0.500   0.000  1.00 20.00           O",
                "ATOM      5  CB  TRP A   1       1.000   1.000   0.000  1.00 20.00           C",
                "ATOM      6  CG  TRP A   1       1.000   2.000   0.000  1.00 20.00           C",
                "HETATM    7  C1  LIG B   1       5.000   5.000   5.000  1.00 20.00           C",
                "HETATM    8  O1  LIG B   1       6.000   5.000   5.000  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_pdb = Path(AllAlanineStructureProcessor().run(str(input_pdb)))
    output_text = output_pdb.read_text(encoding="utf-8")

    assert " ALA A   1" in output_text
    assert " CG  " not in output_text
    assert output_text.count("HETATM") == 2
    assert " LIG B   1" in output_text
