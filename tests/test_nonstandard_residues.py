from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from astrolabe.core import nonstandard_residues


def test_build_nonstandard_aliases_empty_config() -> None:
    aliases, user_ccd = nonstandard_residues.build_nonstandard_aliases_and_user_ccd(None)
    assert aliases == {}
    assert user_ccd is None


def test_build_nonstandard_aliases_uses_upper_alias_and_concatenates_blocks(monkeypatch) -> None:
    def _fake_build_alias_entry(*, alias_name: str, alias_cfg):
        return {"ccd_id": f"{alias_name}X", "kind": "ptm", "parent_aa": "S"}, f"data_{alias_name}"

    monkeypatch.setattr(nonstandard_residues, "_build_alias_entry", _fake_build_alias_entry)
    aliases, user_ccd = nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
        {
            "aliases": {
                "s5": {"dummy": 1},
                "r8": {"dummy": 1},
            }
        }
    )
    assert sorted(aliases.keys()) == ["R8", "S5"]
    assert aliases["S5"]["ccd_id"] == "S5X"
    assert aliases["R8"]["ccd_id"] == "R8X"
    assert user_ccd is not None
    assert "data_S5" in user_ccd
    assert "data_R8" in user_ccd


def test_build_nonstandard_aliases_rejects_long_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        nonstandard_residues,
        "_build_alias_entry",
        lambda *, alias_name, alias_cfg: (
            {"ccd_id": "S5A", "kind": "ptm", "parent_aa": "S"},
            "data",
        ),
    )
    with pytest.raises(ValueError, match="too long"):
        nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
            {"aliases": {"S5LONG": {"dummy": 1}}}
        )


def test_build_nonstandard_aliases_keeps_smiles_for_runtime_params() -> None:
    aliases, _ = nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
        {
            "aliases": {
                "s5": {
                    "ccd_id": "QZ1",
                    "parent_aa": "S",
                    "source": {"format": "smiles", "value": "NCC(=O)O"},
                    "atom_names": ["N", "CA", "C", "O", "OXT"],
                }
            }
        }
    )
    assert aliases["S5"]["ccd_id"] == "QZ1"
    assert aliases["S5"]["kind"] == "ptm"
    assert aliases["S5"]["smiles"]


def test_build_nonstandard_aliases_allows_ligand_alias_without_parent_aa_or_atom_names() -> None:
    aliases, _ = nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
        {
            "aliases": {
                "qz1": {
                    "ccd_id": "QZ1",
                    "source": {"format": "smiles", "value": "C=CC"},
                }
            }
        }
    )
    assert aliases["QZ1"]["ccd_id"] == "QZ1"
    assert aliases["QZ1"]["kind"] == "ligand"
    assert "parent_aa" not in aliases["QZ1"]


def test_build_nonstandard_aliases_ptm_requires_backbone_atom_names() -> None:
    with pytest.raises(ValueError, match="backbone atom names"):
        nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
            {
                "aliases": {
                    "s5": {
                        "ccd_id": "S5A",
                        "parent_aa": "S",
                        "source": {"format": "smiles", "value": "C=CC"},
                        "atom_names": ["CB", "CG", "CD"],
                    }
                }
            }
        )


def test_resolve_nonstandard_alias_hit_and_miss() -> None:
    alias_map = {"S5": {"ccd_id": "S5A", "parent_aa": "S"}}
    assert nonstandard_residues.resolve_nonstandard_alias("s5", alias_map=alias_map) == (
        "S5A",
        "S",
    )
    assert nonstandard_residues.resolve_nonstandard_alias("MSE", alias_map=alias_map) == (
        "MSE",
        None,
    )


def test_resolve_nonstandard_alias_rejects_invalid_parent_aa() -> None:
    alias_map = {"S5": {"ccd_id": "S5A", "parent_aa": "ZZ"}}
    with pytest.raises(ValueError, match="invalid parent_aa"):
        nonstandard_residues.resolve_nonstandard_alias("S5", alias_map=alias_map)


def test_resolve_nonstandard_alias_rejects_ligand_in_ptm_context() -> None:
    alias_map = {"QZ1": {"ccd_id": "QZ1", "kind": "ligand"}}
    with pytest.raises(ValueError, match="ligand-only"):
        nonstandard_residues.resolve_nonstandard_alias(
            "QZ1",
            alias_map=alias_map,
            require_ptm=True,
        )


def test_set_rdkit_atom_names_accepts_omegaconf_listconfig() -> None:
    Chem = nonstandard_residues._require_rdkit()
    mol = Chem.MolFromSmiles("C=CC")
    assert mol is not None
    names = OmegaConf.create(["C1", "C2", "C3"])
    nonstandard_residues._set_rdkit_atom_names(mol, names)
    assigned = [mol.GetAtomWithIdx(i).GetProp("atom_name") for i in range(mol.GetNumAtoms())]
    assert assigned == ["C1", "C2", "C3"]


def test_set_mmcif_atom_names_accepts_omegaconf_listconfig() -> None:
    block = {
        "_chem_comp_atom.atom_id": ["N", "CA", "C", "O"],
        "_chem_comp_bond.atom_id_1": ["N", "CA", "C"],
        "_chem_comp_bond.atom_id_2": ["CA", "C", "O"],
    }
    names = OmegaConf.create(["N", "CA", "C", "O1"])
    atom_names = nonstandard_residues._set_mmcif_atom_names("TPX", block, names)
    assert "O1" in atom_names
    assert block["_chem_comp_atom.atom_id"][-1] == "O1"
    assert block["_chem_comp_bond.atom_id_2"][-1] == "O1"


def test_require_af3_cif_dict_imports_cpp_module() -> None:
    cif_dict = nonstandard_residues._require_af3_cif_dict()
    assert hasattr(cif_dict, "parse_multi_data_cif")


def test_ensure_mmcif_required_keys_fills_missing_ideal_coordinates() -> None:
    block = {
        "_chem_comp_atom.atom_id": ["N", "CA"],
        "_chem_comp_atom.type_symbol": ["N", "C"],
        "_chem_comp_bond.atom_id_1": ["N"],
        "_chem_comp_bond.atom_id_2": ["CA"],
    }
    out = nonstandard_residues._ensure_mmcif_required_keys(
        ccd_id="S5A",
        component_kind="ptm",
        parent_aa="S",
        block=block,
    )
    assert out["_chem_comp_atom.pdbx_model_Cartn_x_ideal"] == ["0.000", "0.000"]
    assert out["_chem_comp_atom.pdbx_model_Cartn_y_ideal"] == ["0.000", "0.000"]
    assert out["_chem_comp_atom.pdbx_model_Cartn_z_ideal"] == ["0.000", "0.000"]
    assert out["_chem_comp.mon_nstd_parent_comp_id"] == ["S"]


def test_ensure_mmcif_required_keys_for_ligand_defaults_nonpolymer_parent() -> None:
    block = {
        "_chem_comp_atom.atom_id": ["C1"],
        "_chem_comp_atom.type_symbol": ["C"],
        "_chem_comp_bond.atom_id_1": [],
        "_chem_comp_bond.atom_id_2": [],
    }
    out = nonstandard_residues._ensure_mmcif_required_keys(
        ccd_id="QZ1",
        component_kind="ligand",
        block=block,
    )
    assert out["_chem_comp.type"] == ["NON-POLYMER"]
    assert out["_chem_comp.mon_nstd_parent_comp_id"] == ["?"]

def test_build_nonstandard_aliases_from_smiles_emits_nonzero_ideal_coordinates() -> None:
    aliases, user_ccd = nonstandard_residues.build_nonstandard_aliases_and_user_ccd(
        {
            "aliases": {
                "s5": {
                    "ccd_id": "AS5",
                    "parent_aa": "A",
                    "source": {
                        "format": "smiles",
                        "value": "C=CCCC[C@](N)(C)C(=O)O",
                    },
                    "atom_names": [
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
                    ],
                }
            }
        }
    )
    assert aliases["S5"]["ccd_id"] == "AS5"
    assert user_ccd is not None

    _block_name, block = nonstandard_residues._parse_mmcif_block(user_ccd)
    xs = [float(v) for v in block["_chem_comp_atom.pdbx_model_Cartn_x_ideal"]]
    ys = [float(v) for v in block["_chem_comp_atom.pdbx_model_Cartn_y_ideal"]]
    zs = [float(v) for v in block["_chem_comp_atom.pdbx_model_Cartn_z_ideal"]]

    assert any(abs(v) > 1e-6 for v in xs + ys + zs)
