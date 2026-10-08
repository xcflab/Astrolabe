from __future__ import annotations

from pathlib import Path

from astrolabe.core import global_config
from astrolabe.mutators.wrappers import ligandmpnn_utils


def _pdb_atom_line(
    *,
    record: str,
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resnum: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:>5} {name:^4} {resname:>3} {chain}{resnum:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{20.00:>6.2f}          {element:>2}\n"
    )


def _write_test_structure(tmp_path: Path, middle_resname: str) -> Path:
    lines = [
        _pdb_atom_line(
            record="ATOM",
            serial=1,
            name="N",
            resname="ALA",
            chain="A",
            resnum=1,
            x=1.0,
            y=1.0,
            z=1.0,
            element="N",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=2,
            name="CA",
            resname="ALA",
            chain="A",
            resnum=1,
            x=2.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=3,
            name="C",
            resname="ALA",
            chain="A",
            resnum=1,
            x=3.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=4,
            name="O",
            resname="ALA",
            chain="A",
            resnum=1,
            x=4.0,
            y=1.0,
            z=1.0,
            element="O",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=5,
            name="N",
            resname=middle_resname,
            chain="A",
            resnum=2,
            x=5.0,
            y=1.0,
            z=1.0,
            element="N",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=6,
            name="CA",
            resname=middle_resname,
            chain="A",
            resnum=2,
            x=6.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=7,
            name="C",
            resname=middle_resname,
            chain="A",
            resnum=2,
            x=7.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=8,
            name="O",
            resname=middle_resname,
            chain="A",
            resnum=2,
            x=8.0,
            y=1.0,
            z=1.0,
            element="O",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=9,
            name="CB",
            resname=middle_resname,
            chain="A",
            resnum=2,
            x=6.0,
            y=2.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=10,
            name="N",
            resname="GLY",
            chain="A",
            resnum=3,
            x=9.0,
            y=1.0,
            z=1.0,
            element="N",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=11,
            name="CA",
            resname="GLY",
            chain="A",
            resnum=3,
            x=10.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=12,
            name="C",
            resname="GLY",
            chain="A",
            resnum=3,
            x=11.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=13,
            name="O",
            resname="GLY",
            chain="A",
            resnum=3,
            x=12.0,
            y=1.0,
            z=1.0,
            element="O",
        ),
        "TER\n",
        "END\n",
    ]
    path = tmp_path / f"{middle_resname.lower()}.pdb"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_parse_pdb_keeps_peptide_linking_nonstandard_residue(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = _write_test_structure(tmp_path, "VAM")
    monkeypatch.setattr(
        ligandmpnn_utils,
        "cached_ccd",
        lambda *args, **kwargs: {
            "VAM": {
                "_chem_comp.type": ["L-PEPTIDE LINKING"],
                "_chem_comp.one_letter_code": ["E"],
            }
        },
        raising=False,
    )

    global_config.reset()
    global_config.init_const("user_ccd", None)

    parsed, *_ = ligandmpnn_utils.parse_PDB(
        str(path),
        device="cpu",
        parse_atoms_with_zero_occupancy=True,
    )

    assert parsed["R_idx"].tolist() == [1, 2, 3]
    assert parsed["mask"].tolist() == [1, 1, 1]


def test_parse_pdb_excludes_non_peptide_nonstandard_residue(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = _write_test_structure(tmp_path, "LIG")
    monkeypatch.setattr(
        ligandmpnn_utils,
        "cached_ccd",
        lambda *args, **kwargs: {
            "LIG": {
                "_chem_comp.type": ["NON-POLYMER"],
            }
        },
        raising=False,
    )

    global_config.reset()
    global_config.init_const("user_ccd", None)

    parsed, *_ = ligandmpnn_utils.parse_PDB(
        str(path),
        device="cpu",
        parse_atoms_with_zero_occupancy=True,
    )

    assert parsed["R_idx"].tolist() == [1, 3]
    assert parsed["mask"].tolist() == [1, 1]


def test_parse_pdb_does_not_load_ccd_for_non_backbone_ligand(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lines = [
        _pdb_atom_line(
            record="ATOM",
            serial=1,
            name="N",
            resname="ALA",
            chain="A",
            resnum=1,
            x=1.0,
            y=1.0,
            z=1.0,
            element="N",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=2,
            name="CA",
            resname="ALA",
            chain="A",
            resnum=1,
            x=2.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=3,
            name="C",
            resname="ALA",
            chain="A",
            resnum=1,
            x=3.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="ATOM",
            serial=4,
            name="O",
            resname="ALA",
            chain="A",
            resnum=1,
            x=4.0,
            y=1.0,
            z=1.0,
            element="O",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=5,
            name="C1",
            resname="LIG",
            chain="B",
            resnum=1,
            x=5.0,
            y=1.0,
            z=1.0,
            element="C",
        ),
        _pdb_atom_line(
            record="HETATM",
            serial=6,
            name="O1",
            resname="LIG",
            chain="B",
            resnum=1,
            x=6.0,
            y=1.0,
            z=1.0,
            element="O",
        ),
        "TER\n",
        "END\n",
    ]
    path = tmp_path / "lig.pdb"
    path.write_text("".join(lines), encoding="utf-8")

    def _raise_if_called(*_, **__):
        raise AssertionError("CCD lookup should be skipped for non-backbone ligands")

    monkeypatch.setattr(
        ligandmpnn_utils,
        "cached_ccd",
        _raise_if_called,
        raising=False,
    )

    global_config.reset()
    global_config.init_const("user_ccd", None)

    parsed, *_ = ligandmpnn_utils.parse_PDB(
        str(path),
        device="cpu",
        parse_atoms_with_zero_occupancy=True,
    )

    assert parsed["R_idx"].tolist() == [1]
    assert parsed["mask"].tolist() == [1]
