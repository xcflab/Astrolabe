from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
def _monkeypatch_global_const(monkeypatch, name: str, value) -> None:
    monkeypatch.setattr(global_config, f"_{name}", value, raising=True)
    monkeypatch.setitem(global_config._initialized_constants, name, True)


def _protomer(*, length: int) -> SimpleNamespace:
    return SimpleNamespace(length=length)


def test_dssp_wrapper_falls_back_to_biotite_geometry_when_mkdssp_is_unusable(
    monkeypatch,
    tmp_path,
) -> None:
    _monkeypatch_global_const(monkeypatch, "dssp_path", str(tmp_path / "mkdssp"))
    dssp_bin = tmp_path / "mkdssp"
    dssp_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dssp_bin.chmod(0o755)

    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       2.028   1.410   0.000  1.00 20.00           C",
                "ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00 20.00           O",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from astrolabe.losses import structure as structure_losses

    result = structure_losses.dssp_wrapper(str(pdb_path))

    assert result
    assert set(result) <= {"H", "E", "L"}


def test_dssp_wrapper_fallback_uses_protein_only_input_and_cleans_temp(
    monkeypatch,
    tmp_path,
) -> None:
    _monkeypatch_global_const(monkeypatch, "dssp_path", str(tmp_path / "mkdssp"))
    dssp_bin = tmp_path / "mkdssp"
    dssp_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dssp_bin.chmod(0o755)

    pdb_path = tmp_path / "with_ligand.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       2.028   1.410   0.000  1.00 20.00           C",
                "ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00 20.00           O",
                "HETATM    5  C1  LIG B   1       1.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from astrolabe.losses import structure as structure_losses

    captured: dict[str, object] = {}

    def fake_geometry(path):
        temp_path = Path(path)
        captured["path"] = temp_path
        captured["text"] = temp_path.read_text(encoding="utf-8")
        return ["L"]

    monkeypatch.setattr(
        structure_losses,
        "_dssp_list_from_biotite_geometry",
        fake_geometry,
    )

    assert structure_losses.dssp_wrapper(str(pdb_path)) == ["L"]
    assert "HETATM" not in str(captured["text"])
    assert not Path(captured["path"]).exists()


def test_protein_only_dssp_input_strips_ligand_hetatm(tmp_path: Path) -> None:
    pdb_path = tmp_path / "with_ligand.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "HETATM    2  C1  LIG B   1       1.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from astrolabe.losses import structure as structure_losses

    dssp_input, tmp_file = structure_losses._protein_only_pdb_for_dssp(str(pdb_path))

    try:
        assert tmp_file is not None
        text = Path(dssp_input).read_text(encoding="utf-8")
        assert "HETATM" not in text
        assert " ALA " in text
    finally:
        if tmp_file is not None:
            Path(tmp_file).unlink()


def test_mkdssp_compat_wrapper_uses_mkdssp_cli_detection() -> None:
    from astrolabe.losses import structure as structure_losses

    assert structure_losses._looks_like_mkdssp("/usr/local/bin/mkdssp_compat.sh")


def test_longest_secondary_structure_run_fraction_uses_max_run_across_copies(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protomer(length=6),
            "B": _protomer(length=4),
        },
    )

    from astrolabe.losses import structure as structure_losses

    monkeypatch.setattr(
        structure_losses,
        "_dssp_list_for_chain_subset",
        lambda _path, chain_ids: list("HHHLLL") if chain_ids == ["A"] else list("HHHHHL"),
    )

    scores = SimpleNamespace(
        names=["ABA"],
        complex_paths={"ABA": Path("aba.pdb")},
    )

    result = structure_losses.longest_secondary_structure_run_fraction(
        scores,
        included_protomers="A",
        sse_type="H",
    )

    assert result == {"ABA": pytest.approx(5.0 / 6.0)}


def test_longest_secondary_structure_run_fraction_collapses_loop_symbols(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protomer(length=8),
        },
    )

    from astrolabe.losses import structure as structure_losses

    monkeypatch.setattr(
        structure_losses,
        "_dssp_list_for_chain_subset",
        lambda _path, _chain_ids: list("HHHTTSC-"),
    )

    scores = SimpleNamespace(
        names=["A"],
        complex_paths={"A": Path("a.pdb")},
    )

    result = structure_losses.longest_secondary_structure_run_fraction(
        scores,
        included_protomers="A",
        sse_type="L",
    )

    assert result == {"A": pytest.approx(5.0 / 8.0)}


def test_longest_secondary_structure_run_fraction_rejects_invalid_sse_type(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protomer(length=4),
        },
    )

    from astrolabe.losses import structure as structure_losses

    scores = SimpleNamespace(
        names=["A"],
        complex_paths={"A": Path("a.pdb")},
    )

    with pytest.raises(ValueError, match="sse_type"):
        structure_losses.longest_secondary_structure_run_fraction(
            scores,
            included_protomers="A",
            sse_type="X",
        )


def test_secondary_structure_fraction_runs_dssp_on_included_protein_subset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protomer(length=3),
            "B": _protomer(length=1),
        },
    )

    cif_path = tmp_path / "ab.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_ab",
                "#",
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
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 N N . ALA A 1 1 ? 0 0 0 1.00 1.00 ? 1 ALA A N 1",
                "ATOM 2 C CA . ALA A 1 1 ? 1 0 0 1.00 1.00 ? 1 ALA A CA 1",
                "ATOM 3 C C . ALA A 1 1 ? 2 0 0 1.00 1.00 ? 1 ALA A C 1",
                "ATOM 4 O O . ALA A 1 1 ? 3 0 0 1.00 1.00 ? 1 ALA A O 1",
                "ATOM 5 N N . GLY A 1 2 ? 4 0 0 1.00 1.00 ? 2 GLY A N 1",
                "ATOM 6 C CA . GLY A 1 2 ? 5 0 0 1.00 1.00 ? 2 GLY A CA 1",
                "ATOM 7 C C . GLY A 1 2 ? 6 0 0 1.00 1.00 ? 2 GLY A C 1",
                "ATOM 8 O O . GLY A 1 2 ? 7 0 0 1.00 1.00 ? 2 GLY A O 1",
                "ATOM 9 N N . SER A 1 3 ? 8 0 0 1.00 1.00 ? 3 SER A N 1",
                "ATOM 10 C CA . SER A 1 3 ? 9 0 0 1.00 1.00 ? 3 SER A CA 1",
                "ATOM 11 C C . SER A 1 3 ? 10 0 0 1.00 1.00 ? 3 SER A C 1",
                "ATOM 12 O O . SER A 1 3 ? 11 0 0 1.00 1.00 ? 3 SER A O 1",
                "HETATM 13 C C1 . Z49 B 2 . ? 12 0 0 1.00 1.00 ? 1 Z49 B C1 1",
                "#",
            ]
        ),
        encoding="utf-8",
    )

    from astrolabe.losses import structure as structure_losses

    captured = {}

    real_subset_writer = structure_losses._write_protein_chain_subset_pdb

    def fake_subset(path: Path, chain_ids: list[str]):
        captured["chain_ids"] = chain_ids
        subset_path = tmp_path / "subset.pdb"
        real_subset_writer(path, chain_ids, subset_path)
        text = subset_path.read_text(encoding="utf-8")
        captured["subset_text"] = text
        return list("HLH")

    monkeypatch.setattr(structure_losses, "_dssp_list_for_chain_subset", fake_subset)

    scores = SimpleNamespace(
        names=["AB"],
        complex_paths={"AB": cif_path},
    )

    result = structure_losses.secondary_structure_fraction(
        scores,
        included_protomers="A",
        sse_type="L",
    )

    assert result == {"AB": pytest.approx(1.0 / 3.0)}
    assert captured["chain_ids"] == ["A"]
    assert " Z49 " not in captured["subset_text"]
    assert " ALA " in captured["subset_text"]
