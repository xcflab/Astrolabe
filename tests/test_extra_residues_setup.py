from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.utils import extra_residues


@pytest.fixture(autouse=True)
def _reset_global_config() -> None:
    global_config.reset()
    yield
    global_config.reset()


def _init_common_constants(
    tmp_path,
    *,
    molfile_to_params: str = "/tmp/not-used/molfile_to_params.py",
    ligand_resnames=None,
    nonstandard_aliases=None,
    user_ccd=None,
) -> None:
    global_config.init_const("protomers", {})
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("molfile_to_params", molfile_to_params)
    global_config.init_const("obabel_path", "obabel")
    global_config.init_const("ligand_resnames", ligand_resnames)
    global_config.init_const("nonstandard_aliases", nonstandard_aliases or {})
    global_config.init_const("user_ccd", user_ccd)


def _patch_ccd_ctor(monkeypatch) -> None:
    from alphafold3.constants import chemical_components

    monkeypatch.setattr(chemical_components, "Ccd", lambda user_ccd=None: {})


def _mock_polymer_ccd_entry(resname: str) -> dict[str, list[str]]:
    return {
        "_chem_comp.id": [resname],
        "_chem_comp_atom.atom_id": ["N", "CA", "C", "O", "OXT", "H", "H2", "HA"],
        "_chem_comp_atom.type_symbol": ["N", "C", "C", "O", "O", "H", "H", "H"],
        "_chem_comp_atom.charge": ["0"] * 8,
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.4", "2.7", "3.5", "2.7", "-0.6", "-0.6", "1.4"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "0.0", "0.8", "-0.9", "0.8", "-0.8", "-1.0"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0"] * 8,
        "_chem_comp_bond.atom_id_1": ["N", "N", "N", "CA", "CA", "C", "C"],
        "_chem_comp_bond.atom_id_2": ["CA", "H", "H2", "C", "HA", "O", "OXT"],
        "_chem_comp_bond.value_order": ["SING", "SING", "SING", "SING", "SING", "DOUB", "SING"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 7,
    }


def _mock_smiles_alias_ccd_with_two_n_hydrogens(resname: str) -> dict[str, list[str]]:
    return {
        "_chem_comp.id": [resname],
        "_chem_comp_atom.atom_id": [
            "SC5",
            "SC4",
            "SC3",
            "SC2",
            "SC1",
            "CA",
            "N",
            "CB",
            "C",
            "O",
            "OXT",
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
            "H9",
            "H10",
            "H11",
            "H12",
            "H13",
            "H14",
            "H15",
        ],
        "_chem_comp_atom.type_symbol": [
            "C", "C", "C", "C", "C", "C", "N", "C", "C", "O", "O",
            "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H",
        ],
        "_chem_comp_atom.charge": ["0"] * 26,
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": [
            "3.578", "2.965", "2.106", "0.800", "-0.135", "-1.473", "-1.199",
            "-2.303", "-2.261", "-2.441", "-2.760", "3.469", "4.200", "3.111",
            "2.680", "1.868", "0.288", "1.033", "0.393", "-0.344", "-2.102",
            "-0.656", "-3.284", "-1.767", "-2.490", "-3.276",
        ],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": [
            "-0.761", "0.232", "1.239", "0.626", "0.182", "-0.425", "-1.576",
            "-0.889", "0.627", "0.525", "1.738", "-0.879", "-1.464", "0.333",
            "1.667", "2.075", "1.395", "-0.235", "-0.575", "1.057", "-2.024",
            "-2.296", "-1.295", "-1.688", "-0.046", "2.457",
        ],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": [
            "0.091", "-0.559", "0.159", "0.685", "-0.459", "0.040", "0.914",
            "-1.175", "0.796", "2.040", "0.122", "1.164", "-0.455", "-1.631",
            "1.009", "-0.534", "1.302", "1.348", "-1.080", "-1.114", "1.196",
            "0.383", "-0.845", "-1.731", "-1.875", "0.615",
        ],
        "_chem_comp_bond.atom_id_1": [
            "SC5", "SC4", "SC3", "SC2", "SC1", "CA", "CA", "CA", "C", "C",
            "SC5", "SC5", "SC4", "SC3", "SC3", "SC2", "SC2", "SC1", "SC1",
            "N", "N", "CB", "CB", "CB", "OXT",
        ],
        "_chem_comp_bond.atom_id_2": [
            "SC4", "SC3", "SC2", "SC1", "CA", "N", "CB", "C", "O", "OXT",
            "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
            "H10", "H11", "H12", "H13", "H14", "H15",
        ],
        "_chem_comp_bond.value_order": [
            "DOUB", "SING", "SING", "SING", "SING", "SING", "SING", "SING", "DOUB", "SING",
            "SING", "SING", "SING", "SING", "SING", "SING", "SING", "SING", "SING",
            "SING", "SING", "SING", "SING", "SING", "SING",
        ],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 25,
    }


def test_setup_extra_residues_uses_af3_compatible_default_smiles_resname(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    captured = {}

    def fake_generate_mol2(smiles: str, output_dir: str, resname_full: str):
        captured["smiles"] = smiles
        captured["resname"] = resname_full
        return {resname_full: f"{output_dir}/{resname_full}.mol2"}

    monkeypatch.setattr(extra_residues, "generate_mol2_files_for_protomer", fake_generate_mol2)
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues()

    assert captured["smiles"] == "CCO"
    assert captured["resname"] == "L_B"
    assert set(rslt.keys()) == {"L_B"}


def test_setup_extra_residues_honors_custom_resname_for_smiles(tmp_path, monkeypatch) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    captured = {}

    def fake_generate_mol2(smiles: str, output_dir: str, resname_full: str):
        captured["resname"] = resname_full
        return {resname_full: f"{output_dir}/{resname_full}.mol2"}

    monkeypatch.setattr(extra_residues, "generate_mol2_files_for_protomer", fake_generate_mol2)
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues(ligand_resnames={"B": "LIG"})

    assert captured["resname"] == "LIG"
    assert set(rslt.keys()) == {"LIG"}


def test_get_rosetta_builtin_residue_names_reads_name_or_filename(tmp_path) -> None:
    residue_dir = tmp_path / "residue_types"
    residue_dir.mkdir()
    (residue_dir / "lig.params").write_text("NAME LIGA\nATOM\n", encoding="utf-8")
    (residue_dir / "fallback.params").write_text("# no name line\n", encoding="utf-8")

    extra_residues.get_rosetta_builtin_residue_names.cache_clear()
    names = extra_residues.get_rosetta_builtin_residue_names(str(residue_dir))

    assert names == {"FALLBACK", "LIGA"}


def test_setup_extra_residues_uses_structure_resname_for_single_smiles_ligand(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    structure_path = tmp_path / "step_000_AB_model.cif"
    structure_path.write_text("data_mock\n", encoding="utf-8")
    extracted_mol2 = tmp_path / "raw_model_results" / "l01.mol2"
    extracted_mol2.parent.mkdir(parents=True, exist_ok=True)
    extracted_mol2.write_text("@<TRIPOS>MOLECULE\nl01\n", encoding="utf-8")

    def unexpected_smiles_generation(*args, **kwargs):
        raise AssertionError("SMILES mol2 generation should use the structure residue name")

    captured = {}

    monkeypatch.setattr(
        extra_residues,
        "extract_hetero_residues_to_mol2",
        lambda *_args, **_kwargs: {"l01": str(extracted_mol2)},
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_mol2_files_for_protomer",
        unexpected_smiles_generation,
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_pyrosetta_params_commands",
        lambda mol2_files, output_dir: captured.update(
            {"mol2_files": dict(mol2_files), "output_dir": output_dir}
        )
        or [],
    )

    rslt = extra_residues.setup_extra_residues(
        structure_files={"AB": structure_path}
    )

    assert rslt == {"l01": str(extracted_mol2)}
    assert captured["mol2_files"] == {"l01": str(extracted_mol2)}


def test_setup_extra_residues_falls_back_to_source_smiles_for_fragmented_structure_mol2(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    structure_path = tmp_path / "step_018_AB_model.cif"
    structure_path.write_text("data_mock\n", encoding="utf-8")
    extracted_mol2 = tmp_path / "raw_model_results" / "l01.mol2"
    extracted_mol2.parent.mkdir(parents=True, exist_ok=True)
    extracted_mol2.write_text("@<TRIPOS>MOLECULE\nfragmented\n", encoding="utf-8")
    fallback_mol2 = tmp_path / "raw_model_results" / "l01_from_smiles.mol2"
    calls = []

    monkeypatch.setattr(
        extra_residues,
        "extract_hetero_residues_to_mol2",
        lambda *_args, **_kwargs: {"l01": str(extracted_mol2)},
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_mol2_files_for_protomer",
        lambda smiles, output_dir, resname_full: calls.append(
            ("fallback", smiles, output_dir, resname_full)
        )
        or {resname_full: str(fallback_mol2)},
    )

    def fake_generate_params(mol2_files, output_dir):
        calls.append(("params", dict(mol2_files), output_dir))
        if len([call for call in calls if call[0] == "params"]) == 1:
            raise RuntimeError("Fragment 3 has 2 atoms")
        return []

    monkeypatch.setattr(
        extra_residues,
        "generate_pyrosetta_params_commands",
        fake_generate_params,
    )

    rslt = extra_residues.setup_extra_residues(
        structure_files={"AB": structure_path}
    )

    assert rslt == {"l01": str(fallback_mol2)}
    assert calls[0] == ("params", {"l01": str(extracted_mol2)}, str(tmp_path / "raw_model_results"))
    assert calls[1] == ("fallback", "CCO", str(tmp_path / "raw_model_results"), "l01")
    assert calls[2] == ("params", {"l01": str(fallback_mol2)}, str(tmp_path / "raw_model_results"))


def test_setup_extra_residues_falls_back_for_named_smiles_resname(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path, ligand_resnames={"B": "OQO"})
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="OQO_SMILES",
        name="B",
    )

    structure_path = tmp_path / "step_006_AB_model.cif"
    structure_path.write_text("data_mock\n", encoding="utf-8")
    extracted_mol2 = tmp_path / "raw_model_results" / "OQO.mol2"
    extracted_mol2.parent.mkdir(parents=True, exist_ok=True)
    extracted_mol2.write_text("@<TRIPOS>MOLECULE\nfragmented\n", encoding="utf-8")
    fallback_mol2 = tmp_path / "raw_model_results" / "OQO_from_smiles.mol2"
    calls = []

    monkeypatch.setattr(
        extra_residues,
        "extract_hetero_residues_to_mol2",
        lambda *_args, **_kwargs: {"OQO": str(extracted_mol2)},
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_mol2_files_for_protomer",
        lambda smiles, output_dir, resname_full: calls.append(
            ("fallback", smiles, output_dir, resname_full)
        )
        or {resname_full: str(fallback_mol2)},
    )

    def fake_generate_params(mol2_files, output_dir):
        calls.append(("params", dict(mol2_files), output_dir))
        if len([call for call in calls if call[0] == "params"]) == 1:
            raise RuntimeError("No acceptable neighbor atoms in molecule")
        return []

    monkeypatch.setattr(
        extra_residues,
        "generate_pyrosetta_params_commands",
        fake_generate_params,
    )

    rslt = extra_residues.setup_extra_residues(
        structure_files={"AB": structure_path}
    )

    assert rslt == {"OQO": str(fallback_mol2)}
    assert calls[0] == ("params", {"OQO": str(extracted_mol2)}, str(tmp_path / "raw_model_results"))
    assert calls[1] == ("fallback", "OQO_SMILES", str(tmp_path / "raw_model_results"), "OQO")
    assert calls[2] == ("params", {"OQO": str(fallback_mol2)}, str(tmp_path / "raw_model_results"))


def test_setup_extra_residues_generates_all_nonstandard_alias_targets(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(
        tmp_path,
        nonstandard_aliases={
            "S5": {"ccd_id": "S5A", "parent_aa": "S"},
            "R8": {"ccd_id": "R8A", "parent_aa": "R"},
        },
    )
    from alphafold3.constants import chemical_components

    monkeypatch.setattr(
        chemical_components,
        "Ccd",
        lambda user_ccd=None: {
            "S5A": _mock_polymer_ccd_entry("S5A"),
            "R8A": _mock_polymer_ccd_entry("R8A"),
        },
    )

    captured = []

    monkeypatch.setattr(
        extra_residues,
        "generate_rosetta_polymer_params_for_ccd",
        lambda *, ccd_entry, resname, parent_aa, output_dir, properties: captured.append((ccd_entry, resname, parent_aa, tuple(properties)))
        or Path(output_dir) / f"{resname}.params",
    )
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues()

    assert set(rslt.keys()) == {"S5A", "R8A"}
    assert {resname for _ccd_entry, resname, _parent_aa, _props in captured} == {"S5A", "R8A"}


def test_setup_extra_residues_prefers_alias_polymer_params_without_builtin_smiles_lookup(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(
        tmp_path,
        nonstandard_aliases={
            "S5": {"ccd_id": "S5A", "parent_aa": "S", "smiles": "C=CC"},
        },
        user_ccd="data_S5A",
    )
    monkeypatch.setattr(
        extra_residues,
        "ccd_code_to_smiles",
        lambda _ccd, code: (_ for _ in ()).throw(AssertionError(f"unexpected CCD lookup: {code}")),
    )
    from alphafold3.constants import chemical_components

    monkeypatch.setattr(
        chemical_components,
        "Ccd",
        lambda user_ccd=None: {
            "S5A": _mock_polymer_ccd_entry("S5A")
        },
    )
    captured = {}

    def _fake_generate_polymer_params(*, ccd_entry, resname, parent_aa, output_dir, properties):
        captured["resname"] = resname
        captured["parent_aa"] = parent_aa
        captured["properties"] = list(properties)
        return Path(output_dir) / f"{resname}.params"

    monkeypatch.setattr(extra_residues, "generate_rosetta_polymer_params_for_ccd", _fake_generate_polymer_params)
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues()

    assert set(rslt.keys()) == {"S5A"}
    assert captured["resname"] == "S5A"
    assert captured["parent_aa"] == "S"
    assert "PROTEIN" in captured["properties"]


def test_setup_extra_residues_rejects_too_long_nonstandard_alias(tmp_path) -> None:
    _init_common_constants(
        tmp_path,
        nonstandard_aliases={
            "S5LONG": {"ccd_id": "S5A", "parent_aa": "S"},
        },
    )

    with pytest.raises(ValueError, match="too long"):
        extra_residues.setup_extra_residues()


def test_setup_extra_residues_converts_nondefault_ccd_with_ccd_codename(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.CCD,
        sequence="9TP",
        name="B",
    )

    from alphafold3.constants import chemical_components

    monkeypatch.setattr(
        chemical_components,
        "Ccd",
        lambda user_ccd=None: {"9TP": {"_chem_comp_atom.atom_id": ["C10", "N11", "O17"]}},
    )
    monkeypatch.setattr(
        extra_residues,
        "_get_runtime_rosetta_builtin_residue_names",
        lambda: {"ATP", "SO4"},
    )

    captured = {}

    def fake_generate_mol2(ccd_entry, output_dir: str, resname_full: str):
        captured["ccd_entry"] = ccd_entry
        captured["resname"] = resname_full
        return {resname_full: f"{output_dir}/{resname_full}.mol2"}

    monkeypatch.setattr(extra_residues, "generate_mol2_files_for_ccd", fake_generate_mol2)
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues()

    assert isinstance(captured["ccd_entry"], dict)
    assert captured["resname"] == "9TP"
    assert set(rslt.keys()) == {"9TP"}


def test_setup_extra_residues_prefers_ccd_driven_mol2_for_ccd(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.CCD,
        sequence="9TP",
        name="B",
    )

    from alphafold3.constants import chemical_components

    monkeypatch.setattr(
        chemical_components,
        "Ccd",
        lambda user_ccd=None: {"9TP": {"_chem_comp_atom.atom_id": ["C10", "N11", "O17"]}},
    )
    monkeypatch.setattr(
        extra_residues,
        "_get_runtime_rosetta_builtin_residue_names",
        lambda: set(),
    )
    captured = {}
    monkeypatch.setattr(
        extra_residues,
        "extract_hetero_residues_to_mol2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected structure-extracted mol2 fallback")
        ),
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_mol2_files_for_protomer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected SMILES->mol2 fallback")
        ),
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_mol2_files_for_ccd",
        lambda ccd_entry, output_dir, resname_full: captured.update(
            {"ccd_entry": ccd_entry, "output_dir": output_dir, "resname": resname_full}
        ) or {resname_full: f"{output_dir}/{resname_full}.mol2"},
    )
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues(
        structure_files={"AB": tmp_path / "ab.pdb"}
    )

    assert captured["resname"] == "9TP"
    assert isinstance(captured["ccd_entry"], dict)
    assert rslt == {"9TP": f"{tmp_path}/raw_model_results/9TP.mol2"}


def test_setup_extra_residues_generates_alias_params_for_builtin_ccd_collision(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(tmp_path, ligand_resnames={"B": "Z49", "B49": "Z49"})
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.CCD,
        sequence="B49",
        name="B",
    )

    from alphafold3.constants import chemical_components

    b49_entry = {
        "_chem_comp_atom.atom_id": ["C4", "C13", "N23"],
        "_chem_comp_atom.type_symbol": ["C", "C", "N"],
        "_chem_comp_atom.charge": ["0", "0", "0"],
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.0", "2.0"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "0.0"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0", "0.0", "0.0"],
        "_chem_comp_bond.atom_id_1": ["C4", "C13"],
        "_chem_comp_bond.atom_id_2": ["C13", "N23"],
        "_chem_comp_bond.value_order": ["SING", "SING"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N", "N"],
    }
    monkeypatch.setattr(chemical_components, "Ccd", lambda user_ccd=None: {"B49": b49_entry})
    monkeypatch.setattr(
        extra_residues,
        "_get_runtime_rosetta_builtin_residue_names",
        lambda: {"B49"},
    )
    monkeypatch.setattr(
        extra_residues,
        "_query_pyrosetta_builtin_residue_names",
        lambda codes: set(codes),
    )
    captured = {}

    def fake_generate_mol2(ccd_entry, output_dir: str, resname_full: str):
        captured["ccd_entry"] = ccd_entry
        captured["resname"] = resname_full
        return {resname_full: f"{output_dir}/{resname_full}.mol2"}

    monkeypatch.setattr(extra_residues, "generate_mol2_files_for_ccd", fake_generate_mol2)
    monkeypatch.setattr(extra_residues, "generate_pyrosetta_params_commands", lambda *args, **kwargs: [])

    rslt = extra_residues.setup_extra_residues()

    assert captured == {"ccd_entry": b49_entry, "resname": "Z49"}
    assert rslt == {"Z49": f"{tmp_path}/raw_model_results/Z49.mol2"}


def test_setup_extra_residues_fails_on_params_name_conflict(tmp_path) -> None:
    _init_common_constants(
        tmp_path,
        ligand_resnames={"B": "S5A"},
        nonstandard_aliases={
            "S5": {"ccd_id": "S5A", "parent_aa": "S"},
        },
    )
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    with pytest.raises(ValueError, match="Params resname conflict"):
        extra_residues.setup_extra_residues()


def test_setup_extra_residues_fails_when_mol2_generation_fails(tmp_path, monkeypatch) -> None:
    _init_common_constants(tmp_path)
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.SMILES,
        sequence="CCO",
        name="B",
    )

    monkeypatch.setattr(extra_residues, "generate_mol2_files_for_protomer", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="Failed to generate mol2"):
        extra_residues.setup_extra_residues()


def test_generate_pyrosetta_params_commands_requires_molfile_path(tmp_path) -> None:
    _init_common_constants(tmp_path, molfile_to_params=None)
    mol2_path = tmp_path / "LIG.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="molfile_to_params path is not configured"):
        extra_residues.generate_pyrosetta_params_commands({"LIG": str(mol2_path)}, str(tmp_path))


def test_generate_pyrosetta_params_commands_raises_on_failed_subprocess(
    tmp_path,
) -> None:
    script_path = tmp_path / "molfile_to_params.py"
    script_path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('boom')\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    _init_common_constants(tmp_path, molfile_to_params=str(script_path))
    mol2_path = tmp_path / "LIG.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to generate params"):
        extra_residues.generate_pyrosetta_params_commands({"LIG": str(mol2_path)}, str(tmp_path))


def test_generate_pyrosetta_params_commands_keeps_input_atom_names(
    tmp_path, monkeypatch
) -> None:
    script_path = tmp_path / "molfile_to_params.py"
    script_path.write_text("#!/usr/bin/env python2\n", encoding="utf-8")
    _init_common_constants(tmp_path, molfile_to_params=str(script_path))
    mol2_path = tmp_path / "LIG.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")

    recorded = {}

    def _fake_run(cmd, cwd, capture_output, text, check):
        recorded["cmd"] = list(cmd)
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(extra_residues.subprocess, "run", _fake_run)

    extra_residues.generate_pyrosetta_params_commands({"LIG": str(mol2_path)}, str(tmp_path))

    assert "--keep-names" in recorded["cmd"]
    assert "--clobber" in recorded["cmd"]

def test_generate_pyrosetta_params_commands_ignores_shebang_python_lookup(
    tmp_path,
) -> None:
    script_path = tmp_path / "molfile_to_params.py"
    script_path.write_text(
        "#!/usr/bin/env python\n"
        "from pathlib import Path\n"
        "import sys\n"
        "mol2 = Path(sys.argv[-1])\n"
        "resname = sys.argv[sys.argv.index('-n') + 1]\n"
        "Path(resname + '.params').write_text(mol2.read_text(encoding='utf-8'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    _init_common_constants(tmp_path, molfile_to_params=str(script_path))
    mol2_path = tmp_path / "LIG.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")

    commands = extra_residues.generate_pyrosetta_params_commands(
        {"LIG": str(mol2_path)},
        str(tmp_path),
    )

    assert commands
    assert (tmp_path / "LIG.params").exists()


def test_generate_pyrosetta_params_commands_reuses_existing_outputs(
    tmp_path, monkeypatch
) -> None:
    script_path = tmp_path / "molfile_to_params.py"
    script_path.write_text("#!/usr/bin/env python3\nraise SystemExit(99)\n", encoding="utf-8")
    script_path.chmod(0o755)
    _init_common_constants(tmp_path, molfile_to_params=str(script_path))
    mol2_path = tmp_path / "LIG.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")
    (tmp_path / "LIG.params").write_text("params", encoding="utf-8")
    (tmp_path / "LIG_0001.pdb").write_text("pdb", encoding="utf-8")

    def _fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(extra_residues.subprocess, "run", _fail_run)

    commands = extra_residues.generate_pyrosetta_params_commands(
        {"LIG": str(mol2_path)},
        str(tmp_path),
    )

    assert commands == [f"# reuse existing params for LIG: {tmp_path / 'LIG.params'}"]


def test_find_nondefault_rosetta_ccd_codes() -> None:
    codes = ["ATP", "9TP", "HEM", "ATP"]
    missing = extra_residues.find_nondefault_rosetta_ccd_codes(
        codes,
        builtin_residue_names={"ATP", "SO4"},
        builtin_residue_name_checker=lambda _code: False,
    )
    assert missing == ["9TP", "HEM"]


def test_find_nondefault_rosetta_ccd_codes_uses_runtime_checker() -> None:
    codes = ["B49", "9TP"]
    missing = extra_residues.find_nondefault_rosetta_ccd_codes(
        codes,
        builtin_residue_names=set(),
        builtin_residue_name_checker=lambda code: code == "B49",
    )
    assert missing == ["9TP"]


def test_generate_mol2_files_for_protomer_raises_when_all_converters_fail(
    tmp_path, monkeypatch
) -> None:
    def _primary(*args, **kwargs):
        raise extra_residues.ExtraResidueError("primary-fail")

    def _fallback(*args, **kwargs):
        raise extra_residues.ExtraResidueError("fallback-fail")

    monkeypatch.setattr(extra_residues, "_convert_smiles_to_mol2", _primary)
    monkeypatch.setattr(extra_residues, "_convert_smiles_to_mol2_pure_rdkit", _fallback)

    with pytest.raises(RuntimeError, match="primary=primary-fail; fallback=fallback-fail"):
        extra_residues.generate_mol2_files_for_protomer(
            smiles="CCO",
            output_dir=str(tmp_path),
            resname_full="LIG",
        )


def test_generate_mol2_files_for_ccd_preserves_ccd_atom_names(tmp_path) -> None:
    ccd_entry = {
        "_chem_comp_atom.atom_id": ["C10", "N11", "O17"],
        "_chem_comp_atom.type_symbol": ["C", "N", "O"],
        "_chem_comp_atom.charge": ["0", "0", "0"],
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.3", "2.4"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.1", "0.0"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0", "0.0", "0.0"],
        "_chem_comp_bond.atom_id_1": ["C10", "N11"],
        "_chem_comp_bond.atom_id_2": ["N11", "O17"],
        "_chem_comp_bond.value_order": ["SING", "DOUB"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N", "N"],
    }

    rslt = extra_residues.generate_mol2_files_for_ccd(
        ccd_entry,
        str(tmp_path),
        "9TP",
    )

    mol2_path = tmp_path / "9TP.mol2"
    assert rslt == {"9TP": str(mol2_path)}
    atom_names = []
    in_atom_section = False
    for line in mol2_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "@<TRIPOS>ATOM":
            in_atom_section = True
            continue
        if line.startswith("@<TRIPOS>") and in_atom_section:
            break
        if in_atom_section and line.strip():
            atom_names.append(line.split()[1])

    assert atom_names == ["C10", "N11", "O17"]


def test_write_rosetta_polymer_sdf_for_ccd_uses_free_residue_connection_placeholders(
    tmp_path,
) -> None:
    from alphafold3.constants import chemical_components

    ccd = chemical_components.Ccd()

    aib_sdf = extra_residues._write_rosetta_polymer_sdf_for_ccd(
        ccd["AIB"],
        output_dir=str(tmp_path),
        resname="AIB",
        parent_aa="A",
        properties=["PROTEIN", "ALPHA_AA", "ACHIRAL_BACKBONE", "ACHIRAL_SIDECHAIN"],
    )
    aba_sdf = extra_residues._write_rosetta_polymer_sdf_for_ccd(
        ccd["ABA"],
        output_dir=str(tmp_path),
        resname="ABA",
        parent_aa="A",
        properties=["PROTEIN", "ALPHA_AA", "L_AA"],
    )

    aib_text = aib_sdf.read_text(encoding="utf-8")
    aba_text = aba_sdf.read_text(encoding="utf-8")

    assert "M  ROOT 1" in aib_text
    assert "M  POLY_N_BB 1" in aib_text
    assert "M  POLY_CA_BB 2" in aib_text
    assert "M  POLY_C_BB 3" in aib_text
    assert "M  POLY_O_BB 4" in aib_text
    assert "M  POLY_UPPER 5" in aib_text
    assert "M  POLY_LOWER 9" in aib_text
    assert "M  POLY_IGNORE 10" in aib_text
    assert "M  POLY_PROPERTIES PROTEIN ALPHA_AA ACHIRAL_BACKBONE ACHIRAL_SIDECHAIN" in aib_text

    assert "M  ROOT 1" in aba_text
    assert "M  POLY_N_BB 1" in aba_text
    assert "M  POLY_CA_BB 2" in aba_text
    assert "M  POLY_C_BB 3" in aba_text
    assert "M  POLY_O_BB 4" in aba_text
    assert "M  POLY_UPPER 7" in aba_text
    assert "M  POLY_LOWER 9" in aba_text
    assert "M  POLY_IGNORE 16" in aba_text
    assert "M  POLY_PROPERTIES PROTEIN ALPHA_AA L_AA" in aba_text


def test_write_rosetta_polymer_sdf_for_ccd_accepts_internal_residue_fragment(
    tmp_path,
) -> None:
    ccd_entry = {
        "_chem_comp.id": ["S5A"],
        "_chem_comp_atom.atom_id": ["N", "CA", "C", "O", "CB", "HA"],
        "_chem_comp_atom.type_symbol": ["N", "C", "C", "O", "C", "H"],
        "_chem_comp_atom.charge": ["0"] * 6,
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.4", "2.7", "3.5", "1.4", "1.4"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "0.0", "0.8", "1.2", "-1.0"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0"] * 6,
        "_chem_comp_bond.atom_id_1": ["N", "CA", "CA", "CA", "C"],
        "_chem_comp_bond.atom_id_2": ["CA", "C", "CB", "HA", "O"],
        "_chem_comp_bond.value_order": ["SING", "SING", "SING", "SING", "DOUB"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 5,
    }

    sdf_path = extra_residues._write_rosetta_polymer_sdf_for_ccd(
        ccd_entry,
        output_dir=str(tmp_path),
        resname="S5A",
        parent_aa="S",
        properties=["PROTEIN", "ALPHA_AA", "L_AA"],
    )

    text = sdf_path.read_text(encoding="utf-8")
    assert "M  POLY_N_BB " in text
    assert "M  POLY_CA_BB " in text
    assert "M  POLY_C_BB " in text
    assert "M  POLY_O_BB " in text
    assert "M  POLY_UPPER " in text
    assert "M  POLY_LOWER " in text
    assert "M  POLY_PROPERTIES PROTEIN ALPHA_AA L_AA" in text


def test_write_rosetta_polymer_sdf_for_ccd_infers_renamed_backbone_atoms(
    tmp_path,
) -> None:
    ccd_entry = {
        "_chem_comp.id": ["S5A"],
        "_chem_comp_atom.atom_id": ["N1", "CA1", "C1", "O1", "OXT1", "H1", "H21", "HA1", "CB1"],
        "_chem_comp_atom.type_symbol": ["N", "C", "C", "O", "O", "H", "H", "H", "C"],
        "_chem_comp_atom.charge": ["0"] * 9,
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.4", "2.7", "3.5", "2.7", "-0.6", "-0.6", "1.4", "1.4"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "0.0", "0.8", "-0.9", "0.8", "-0.8", "-1.0", "1.2"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0"] * 9,
        "_chem_comp_bond.atom_id_1": ["N1", "N1", "N1", "CA1", "CA1", "CA1", "C1", "C1"],
        "_chem_comp_bond.atom_id_2": ["CA1", "H1", "H21", "C1", "HA1", "CB1", "O1", "OXT1"],
        "_chem_comp_bond.value_order": ["SING", "SING", "SING", "SING", "SING", "SING", "DOUB", "SING"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 8,
    }

    sdf_path = extra_residues._write_rosetta_polymer_sdf_for_ccd(
        ccd_entry,
        output_dir=str(tmp_path),
        resname="S5A",
        parent_aa="S",
        properties=["PROTEIN", "ALPHA_AA", "L_AA"],
    )

    text = sdf_path.read_text(encoding="utf-8")
    assert "M  POLY_N_BB 1" in text
    assert "M  POLY_CA_BB 2" in text
    assert "M  POLY_C_BB 3" in text
    assert "M  POLY_O_BB 4" in text
    assert "M  POLY_UPPER 5" in text
    assert "M  POLY_LOWER 7" in text


def test_infer_nonstandard_alias_rosetta_properties_handles_renamed_backbone_atoms() -> None:
    ccd_entry = {
        "_chem_comp_atom.atom_id": ["N1", "CA1", "C1", "O1", "OXT1", "H1", "H21", "HA1", "CB1"],
        "_chem_comp_atom.type_symbol": ["N", "C", "C", "O", "O", "H", "H", "H", "C"],
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.4", "2.7", "3.5", "2.7", "-0.6", "-0.6", "1.4", "1.4"],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "0.0", "0.8", "-0.9", "0.8", "-0.8", "-1.0", "1.2"],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0"] * 9,
        "_chem_comp_bond.atom_id_1": ["N1", "N1", "N1", "CA1", "CA1", "CA1", "C1", "C1"],
        "_chem_comp_bond.atom_id_2": ["CA1", "H1", "H21", "C1", "HA1", "CB1", "O1", "OXT1"],
        "_chem_comp_bond.value_order": ["SING", "SING", "SING", "SING", "SING", "SING", "DOUB", "SING"],
        "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 8,
    }

    properties = extra_residues._infer_nonstandard_alias_rosetta_properties({}, ccd_entry)
    assert properties == ["PROTEIN", "ALPHA_AA", "L_AA"]


def test_write_rosetta_polymer_sdf_for_ccd_accepts_multiple_n_hydrogens(
    tmp_path,
) -> None:
    ccd_entry = _mock_smiles_alias_ccd_with_two_n_hydrogens("AS5")

    sdf_path = extra_residues._write_rosetta_polymer_sdf_for_ccd(
        ccd_entry,
        output_dir=str(tmp_path),
        resname="AS5",
        parent_aa="A",
        properties=["PROTEIN", "ALPHA_AA", "ACHIRAL_BACKBONE", "ACHIRAL_SIDECHAIN"],
    )

    text = sdf_path.read_text(encoding="utf-8")
    assert "M  POLY_N_BB 7" in text
    assert "M  POLY_CA_BB 6" in text
    assert "M  POLY_C_BB 9" in text
    assert "M  POLY_O_BB 10" in text
    assert "M  POLY_UPPER 11" in text
    assert "M  POLY_LOWER 21" in text


def test_infer_nonstandard_alias_rosetta_properties_accepts_multiple_n_hydrogens() -> None:
    ccd_entry = _mock_smiles_alias_ccd_with_two_n_hydrogens("AS5")

    properties = extra_residues._infer_nonstandard_alias_rosetta_properties({}, ccd_entry)

    assert properties == ["PROTEIN", "ALPHA_AA", "ACHIRAL_BACKBONE", "ACHIRAL_SIDECHAIN"]


def test_postprocess_free_residue_polymer_params_renames_lower_placeholder() -> None:
    params_text = "\n".join(
        [
            "NAME AIB",
            "ATOM  N   Nbb  NH2  -0.496",
            "ATOM  H   HNbb H    0.284",
            "ATOM  H   Hpol H    0.404",
            "ICOOR_INTERNAL    H    -59.918506   73.265989    1.008511   N     CA    C  ",
            "ICOOR_INTERNAL    H     53.827785   73.302193    1.009488   N     CA    C  ",
            "",
        ]
    )

    out = extra_residues._postprocess_free_residue_polymer_params_text(params_text)
    lines = out.splitlines()

    assert sum(1 for line in lines if line.startswith("ATOM  H   ")) == 1
    assert any(line.startswith("ICOOR_INTERNAL LOWER") for line in lines)
    assert sum(1 for line in lines if line.startswith("ICOOR_INTERNAL    H")) == 1


def test_setup_extra_residues_uses_polymer_generator_for_nonstandard_alias(
    tmp_path, monkeypatch
) -> None:
    _init_common_constants(
        tmp_path,
        nonstandard_aliases={
            "ABA": {"ccd_id": "ABA", "parent_aa": "A"},
        },
    )
    from alphafold3.constants import chemical_components

    monkeypatch.setattr(
        chemical_components,
        "Ccd",
        lambda user_ccd=None: {
            "ABA": {
                "_chem_comp.id": ["ABA"],
                "_chem_comp_atom.atom_id": [
                    "N",
                    "CA",
                    "C",
                    "O",
                    "CB",
                    "CG",
                    "OXT",
                    "H",
                    "H2",
                    "HA",
                ],
                "_chem_comp_atom.type_symbol": [
                    "N",
                    "C",
                    "C",
                    "O",
                    "C",
                    "C",
                    "O",
                    "H",
                    "H",
                    "H",
                ],
                "_chem_comp_atom.charge": ["0"] * 10,
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal": [str(i) for i in range(10)],
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0"] * 10,
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0"] * 10,
                "_chem_comp_bond.atom_id_1": ["N", "N", "CA", "CA", "CA", "C", "C"],
                "_chem_comp_bond.atom_id_2": ["H", "H2", "C", "CB", "HA", "O", "OXT"],
                "_chem_comp_bond.value_order": ["SING", "SING", "SING", "SING", "SING", "DOUB", "SING"],
                "_chem_comp_bond.pdbx_aromatic_flag": ["N"] * 7,
            }
        },
    )
    monkeypatch.setattr(
        extra_residues,
        "generate_pyrosetta_params_commands",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nonstandard alias should not use generic ligand params path")
        ),
    )

    captured = {}

    def _fake_generate_rosetta_polymer_params(*, ccd_entry, resname, parent_aa, output_dir, properties):
        captured["resname"] = resname
        captured["parent_aa"] = parent_aa
        captured["properties"] = list(properties)
        captured["output_dir"] = output_dir
        return Path(output_dir) / f"{resname}.params"

    monkeypatch.setattr(
        extra_residues,
        "generate_rosetta_polymer_params_for_ccd",
        _fake_generate_rosetta_polymer_params,
    )

    rslt = extra_residues.setup_extra_residues()

    assert captured["resname"] == "ABA"
    assert captured["parent_aa"] == "A"
    assert "PROTEIN" in captured["properties"]
    assert "ALPHA_AA" in captured["properties"]
    assert rslt["ABA"].endswith("ABA.params")

def test_needs_molfile_to_params_for_nonstandard_pyrosetta() -> None:
    import run as gh_run

    global_config.reset()
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=ProtomerTypes.PROTEIN)},
    )
    global_config.init_const(
        "nonstandard_aliases",
        {"S5": {"ccd_id": "AS5", "parent_aa": "A"}},
    )

    args = SimpleNamespace(enable_pyrosetta=True)

    assert gh_run._needs_molfile_to_params(args) is True


def test_initialize_global_config_accepts_direct_user_ccd_path(tmp_path: Path) -> None:
    import run as gh_run

    ccd_text = "data_R13SB\n#\n_chem_comp.id R13SB\n#\n"
    ccd_path = tmp_path / "R13SB.cif"
    ccd_path.write_text(ccd_text, encoding="utf-8")

    args = SimpleNamespace(
        exclude_AA=[],
        covalent_bonds=None,
        initial_temperature=0.03,
        out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw_model_results",
        debug=False,
        use_metropolis_acceptance=True,
        enable_pyrosetta=True,
        skip_complex_poses=False,
        allow_mmcif_pdb_fallback=False,
        per_residue_scores=None,
        protenix_constraint=None,
        binding_site=None,
        dynamic_binding_site_enabled=False,
        chai_restraints=None,
        channel=None,
        cyclization=None,
        ligand_resnames={"B": "R13"},
        nonstandard_residues=None,
        user_ccd=str(ccd_path),
    )

    gh_run.initialize_global_config(args)

    assert global_config.user_ccd == ccd_text
    assert global_config.nonstandard_aliases == {}
    assert global_config.ligand_resnames == {"B": "R13"}
