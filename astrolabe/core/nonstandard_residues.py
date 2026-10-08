from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_REQUIRED_USER_CCD_KEYS = (
    "_chem_comp.id",
    "_chem_comp.name",
    "_chem_comp.type",
    "_chem_comp.formula",
    "_chem_comp.mon_nstd_parent_comp_id",
    "_chem_comp.pdbx_synonyms",
    "_chem_comp.formula_weight",
    "_chem_comp_atom.comp_id",
    "_chem_comp_atom.atom_id",
    "_chem_comp_atom.type_symbol",
    "_chem_comp_atom.charge",
    "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
    "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
    "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    "_chem_comp_bond.atom_id_1",
    "_chem_comp_bond.atom_id_2",
    "_chem_comp_bond.value_order",
    "_chem_comp_bond.pdbx_aromatic_flag",
)

_AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
_ALIAS_KIND_PTM = "ptm"
_ALIAS_KIND_LIGAND = "ligand"
_PTM_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
_BOND_ORDER_TO_MMCIFF = {
    "single": "SING",
    "double": "DOUB",
    "triple": "TRIP",
    "aromatic": "AROM",
}
_BOND_ORDER_TO_RDKIT = {
    "single": "SINGLE",
    "double": "DOUBLE",
    "triple": "TRIPLE",
    "aromatic": "AROMATIC",
}


def _as_mapping(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "keys"):
        return {str(k): cfg[k] for k in cfg.keys()}
    raise TypeError(f"Unsupported nonstandard_residues config type: {type(cfg)}")


def _normalize_bond_order(order: Any) -> str:
    value = str(order or "").strip().lower()
    alias = {
        "s": "single",
        "single": "single",
        "d": "double",
        "double": "double",
        "t": "triple",
        "triple": "triple",
        "a": "aromatic",
        "aromatic": "aromatic",
    }
    out = alias.get(value, value)
    if out not in _BOND_ORDER_TO_MMCIFF:
        raise ValueError(
            f"Invalid bond order {order!r} (expected single|double|triple|aromatic)"
        )
    return out


def _require_alias_parent_aa(parent_aa: Any, *, alias_name: str) -> str:
    aa = str(parent_aa or "").strip().upper()
    if aa not in _AA_ALPHABET:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.parent_aa must be one-letter AA in {_AA_ALPHABET}, got {parent_aa!r}"
        )
    return aa


def _infer_alias_kind(alias_cfg: Mapping[str, Any]) -> str:
    """Infer alias kind from config.

    Auto rule (current default): aliases with parent_aa are treated as PTM;
    aliases without parent_aa are treated as ligand-like small molecules.
    """

    parent_aa = str(alias_cfg.get("parent_aa", "")).strip()
    return _ALIAS_KIND_PTM if parent_aa else _ALIAS_KIND_LIGAND


def _require_ptm_backbone_atom_names(alias_name: str, atom_names: set[str]) -> None:
    normalized = {str(name).strip().upper() for name in atom_names if str(name).strip()}
    missing = sorted(_PTM_BACKBONE_ATOMS.difference(normalized))
    if missing:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name} is PTM-like and must provide peptide "
            f"backbone atom names {sorted(_PTM_BACKBONE_ATOMS)}; missing: {missing}"
        )


def _source_from_alias_cfg(alias_name: str, cfg: Mapping[str, Any]) -> tuple[str, Any]:
    source = cfg.get("source")
    if source is None:
        for direct in ("smiles", "sdf", "mol", "mmcif", "pdb"):
            if cfg.get(direct) is not None:
                return direct, cfg.get(direct)
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name} missing source "
            "(expected source.format+source.value or one of smiles/sdf/mol/mmcif/pdb)"
        )

    if isinstance(source, str):
        # Shorthand: assume SMILES literal.
        return "smiles", source
    source_cfg = _as_mapping(source)
    fmt = str(source_cfg.get("format", "")).strip().lower()
    value = source_cfg.get("value", source_cfg.get("path"))
    if not fmt:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.source.format is required"
        )
    if value is None:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.source.value/path is required"
        )
    return fmt, value


def _load_text(value: Any) -> str:
    text = str(value)
    p = Path(text).expanduser()
    if p.exists():
        return p.read_text(encoding="utf-8")
    return text


def _require_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Nonstandard residue conversion requires RDKit when source is "
            "smiles/sdf/mol/pdb."
        ) from exc
    return Chem


def _require_af3_cif_dict():
    try:
        from alphafold3.cpp import cif_dict  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Nonstandard residue conversion requires vendored AlphaFold3 parsers."
        ) from exc
    return cif_dict


def _rdkit_atom_display_name(atom) -> str:
    if atom.HasProp("atom_name"):
        return str(atom.GetProp("atom_name")).strip()
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is not None:
        return str(pdb_info.GetName() or "").strip()
    return ""


def _resolve_atom_index_for_rdkit_mol(mol, ref: Any) -> int:
    if isinstance(ref, int):
        idx = int(ref)
    else:
        s = str(ref).strip()
        if s.isdigit():
            idx = int(s)
        else:
            idx = -1
            for atom in mol.GetAtoms():
                if _rdkit_atom_display_name(atom) == s:
                    idx = int(atom.GetIdx())
                    break
            if idx < 0:
                raise ValueError(f"Unknown atom reference {ref!r} for RDKit molecule")
    if idx < 0 or idx >= int(mol.GetNumAtoms()):
        raise ValueError(
            f"Atom index out of range: {idx} (num_atoms={mol.GetNumAtoms()})"
        )
    return idx


def _set_rdkit_atom_names(mol, atom_names_cfg: Any) -> set[str]:
    if (
        not isinstance(atom_names_cfg, (list, tuple))
        and not isinstance(atom_names_cfg, (str, bytes))
        and not hasattr(atom_names_cfg, "keys")
        and hasattr(atom_names_cfg, "__iter__")
    ):
        atom_names_cfg = list(atom_names_cfg)

    assignments: dict[int, str] = {}
    if isinstance(atom_names_cfg, (list, tuple)):
        if len(atom_names_cfg) != int(mol.GetNumAtoms()):
            raise ValueError(
                "atom_names list length must match number of atoms "
                f"({len(atom_names_cfg)} != {mol.GetNumAtoms()})"
            )
        for idx, name in enumerate(atom_names_cfg):
            assignments[int(idx)] = str(name).strip()
    else:
        mapping = _as_mapping(atom_names_cfg)
        for raw_key, raw_val in mapping.items():
            idx = _resolve_atom_index_for_rdkit_mol(mol, raw_key)
            assignments[int(idx)] = str(raw_val).strip()

    if len(assignments) != int(mol.GetNumAtoms()):
        raise ValueError(
            "atom_names must provide a name for every atom "
            f"(provided={len(assignments)} num_atoms={mol.GetNumAtoms()})"
        )

    seen: set[str] = set()
    for idx, name in sorted(assignments.items()):
        if not name:
            raise ValueError(f"atom_names[{idx}] is empty")
        if name in seen:
            raise ValueError(f"Duplicated atom name in atom_names: {name!r}")
        seen.add(name)
        mol.GetAtomWithIdx(int(idx)).SetProp("atom_name", name)
    return seen


def _apply_explicit_pdb_bonds(alias_name: str, mol, cfg: Mapping[str, Any]):
    bonds = cfg.get("bonds")
    if bonds is None or not list(bonds):
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.bonds is required for pdb source"
        )

    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdchem  # type: ignore

    rw = Chem.RWMol(mol)
    existing = list(rw.GetBonds())
    for bond in existing:
        rw.RemoveBond(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))

    for idx, bond in enumerate(list(bonds)):
        bcfg = _as_mapping(bond)
        atom1 = bcfg.get("atom1")
        atom2 = bcfg.get("atom2")
        if atom1 is None or atom2 is None:
            raise ValueError(
                f"nonstandard_residues.aliases.{alias_name}.bonds[{idx}] must provide atom1/atom2"
            )
        order = _normalize_bond_order(bcfg.get("order"))
        a_idx = _resolve_atom_index_for_rdkit_mol(rw, atom1)
        b_idx = _resolve_atom_index_for_rdkit_mol(rw, atom2)
        bond_type = getattr(rdchem.BondType, _BOND_ORDER_TO_RDKIT[order])
        rw.AddBond(int(a_idx), int(b_idx), bond_type)
        if order == "aromatic":
            b = rw.GetBondBetweenAtoms(int(a_idx), int(b_idx))
            if b is not None:
                b.SetIsAromatic(True)
            rw.GetAtomWithIdx(int(a_idx)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(b_idx)).SetIsAromatic(True)

    Chem.SanitizeMol(rw)
    return rw.GetMol()


def _parse_mmcif_block(text: str) -> tuple[str, dict[str, list[str]]]:
    cif_dict = _require_af3_cif_dict()
    blocks = cif_dict.parse_multi_data_cif(text)
    if not blocks:
        raise ValueError("No mmCIF data blocks found for nonstandard residue source")
    block_name = next(iter(blocks.keys()))
    block = blocks[block_name]
    return str(block_name), {str(k): list(v) for k, v in block.items()}


def _set_mmcif_atom_names(
    alias_name: str,
    block: dict[str, list[str]],
    atom_names_cfg: Any,
) -> set[str]:
    atom_ids = list(block.get("_chem_comp_atom.atom_id") or [])
    if not atom_ids:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name} mmCIF source missing _chem_comp_atom.atom_id"
        )

    updated = list(atom_ids)
    is_sequence_like = isinstance(atom_names_cfg, (list, tuple)) or (
        atom_names_cfg is not None
        and not isinstance(atom_names_cfg, (str, bytes, bytearray))
        and not hasattr(atom_names_cfg, "keys")
        and hasattr(atom_names_cfg, "__iter__")
    )
    if is_sequence_like:
        atom_names_seq = list(atom_names_cfg)
        if len(atom_names_seq) != len(atom_ids):
            raise ValueError(
                f"atom_names list length must match atom count ({len(atom_names_seq)} != {len(atom_ids)})"
            )
        for idx, name in enumerate(atom_names_seq):
            updated[idx] = str(name).strip()
    else:
        mapping = _as_mapping(atom_names_cfg)
        name_to_idx = {str(name): idx for idx, name in enumerate(atom_ids)}
        for raw_key, raw_val in mapping.items():
            key = str(raw_key).strip()
            if key.isdigit():
                idx = int(key)
            else:
                if key not in name_to_idx:
                    raise ValueError(
                        f"Unknown atom key {raw_key!r} in atom_names for alias {alias_name}"
                    )
                idx = int(name_to_idx[key])
            if idx < 0 or idx >= len(atom_ids):
                raise ValueError(
                    f"atom_names index out of range for alias {alias_name}: {idx}"
                )
            updated[idx] = str(raw_val).strip()

    if any((not n) for n in updated):
        raise ValueError(f"Empty atom name after remapping for alias {alias_name}")
    if len(set(updated)) != len(updated):
        raise ValueError(f"Duplicate atom names after remapping for alias {alias_name}")

    old_to_new = {old: new for old, new in zip(atom_ids, updated)}
    block["_chem_comp_atom.atom_id"] = list(updated)
    for bond_key in ("_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2"):
        if bond_key in block:
            block[bond_key] = [old_to_new.get(v, v) for v in block[bond_key]]
    return set(updated)


def _ensure_mmcif_required_keys(
    *,
    ccd_id: str,
    component_kind: str,
    parent_aa: str | None = None,
    block: dict[str, list[str]],
    formula: str = "?",
    formula_weight: str = "?",
) -> dict[str, list[str]]:
    if component_kind not in {_ALIAS_KIND_PTM, _ALIAS_KIND_LIGAND}:
        raise ValueError(f"Unknown nonstandard alias kind: {component_kind!r}")

    out = dict(block)
    atom_count = len(out.get("_chem_comp_atom.atom_id", []))
    bond_count = len(out.get("_chem_comp_bond.atom_id_1", []))

    out["data_"] = [str(ccd_id)]
    out["_chem_comp.id"] = [str(ccd_id)]
    out.setdefault("_chem_comp.name", [str(ccd_id)])
    if component_kind == _ALIAS_KIND_PTM:
        parent = str(parent_aa or "").strip().upper()
        if parent not in _AA_ALPHABET:
            raise ValueError(
                f"PTM-like alias {ccd_id!r} requires valid parent_aa in {_AA_ALPHABET}, got {parent_aa!r}"
            )
        out["_chem_comp.type"] = ["L-PEPTIDE LINKING"]
        out["_chem_comp.mon_nstd_parent_comp_id"] = [parent]
    else:
        out.setdefault("_chem_comp.type", ["NON-POLYMER"])
        out.setdefault("_chem_comp.mon_nstd_parent_comp_id", ["?"])
    out.setdefault("_chem_comp.formula", [str(formula or "?")])
    out.setdefault("_chem_comp.pdbx_synonyms", ["?"])
    out.setdefault("_chem_comp.formula_weight", [str(formula_weight or "?")])
    out.setdefault("_chem_comp_atom.comp_id", [str(ccd_id)] * atom_count)
    out.setdefault("_chem_comp_atom.charge", ["0"] * atom_count)
    for src_key, dst_key in (
        ("_chem_comp_atom.model_Cartn_x", "_chem_comp_atom.pdbx_model_Cartn_x_ideal"),
        ("_chem_comp_atom.model_Cartn_y", "_chem_comp_atom.pdbx_model_Cartn_y_ideal"),
        ("_chem_comp_atom.model_Cartn_z", "_chem_comp_atom.pdbx_model_Cartn_z_ideal"),
    ):
        if dst_key not in out:
            src_vals = list(out.get(src_key) or [])
            if len(src_vals) == atom_count:
                out[dst_key] = src_vals
            else:
                out[dst_key] = ["0.000"] * atom_count
    out.setdefault("_chem_comp_bond.comp_id", [str(ccd_id)] * bond_count)
    out.setdefault("_chem_comp_bond.value_order", ["SING"] * bond_count)
    out.setdefault("_chem_comp_bond.pdbx_aromatic_flag", ["N"] * bond_count)

    missing = [k for k in _REQUIRED_USER_CCD_KEYS if k not in out]
    if missing:
        raise ValueError(
            f"Generated mmCIF for CCD {ccd_id!r} is missing required keys: {missing}"
        )
    return out


def _rdkit_mol_has_non_degenerate_conformer(mol) -> bool:
    if mol.GetNumConformers() <= 0:
        return False
    conf = mol.GetConformer()
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        xs.append(float(pos.x))
        ys.append(float(pos.y))
        zs.append(float(pos.z))
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    return max(span_x, span_y, span_z) > 1e-3


def _ensure_rdkit_mol_has_non_degenerate_coordinates(mol, *, alias_name: str):
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    work = Chem.Mol(mol)
    if _rdkit_mol_has_non_degenerate_conformer(work):
        return work

    work = Chem.AddHs(work, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 100
    elif hasattr(params, "maxIterations"):
        params.maxIterations = 100

    result = AllChem.EmbedMolecule(work, params)
    if result == -1:
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(work, params)
    if result == -1:
        raise RuntimeError(
            f"Failed to generate non-degenerate 3D coordinates for nonstandard alias {alias_name}"
        )

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(work)
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(work, mmff_props, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(work, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(work, maxIters=500)
        except Exception:
            pass
    return work


def _rdkit_mol_to_user_ccd_block(
    *,
    mol,
    ccd_id: str,
    component_kind: str,
    parent_aa: str | None = None,
) -> str:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore

    try:
        from alphafold3.data.tools import rdkit_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Converting nonstandard residues to CCD requires vendored AlphaFold3 rdkit_utils."
        ) from exc

    mol = _ensure_rdkit_mol_has_non_degenerate_coordinates(
        mol,
        alias_name=str(ccd_id),
    )
    cif_dict = _require_af3_cif_dict()
    ccd_cif = rdkit_utils.mol_to_ccd_cif(
        mol,
        component_id=str(ccd_id),
        pdbx_smiles=Chem.MolToSmiles(mol),
        include_hydrogens=True,
    )
    block = {str(k): list(v) for k, v in ccd_cif.items()}
    block = _ensure_mmcif_required_keys(
        ccd_id=str(ccd_id),
        component_kind=component_kind,
        parent_aa=parent_aa,
        block=block,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        formula_weight=f"{Descriptors.MolWt(mol):.3f}",
    )
    return cif_dict.CifDict(block).to_string()


def _mmcif_source_to_user_ccd_block(
    *,
    alias_name: str,
    mmcif_text: str,
    ccd_id: str,
    component_kind: str,
    parent_aa: str | None = None,
    atom_names_cfg: Any = None,
) -> tuple[str, set[str]]:
    _, block = _parse_mmcif_block(mmcif_text)
    atom_names: set[str]
    if atom_names_cfg is None:
        atom_names = {
            str(v).strip()
            for v in list(block.get("_chem_comp_atom.atom_id") or [])
            if str(v).strip()
        }
        if not atom_names:
            raise ValueError(
                f"nonstandard_residues.aliases.{alias_name} mmCIF source missing _chem_comp_atom.atom_id"
            )
    else:
        atom_names = _set_mmcif_atom_names(alias_name, block, atom_names_cfg)
    block = _ensure_mmcif_required_keys(
        ccd_id=str(ccd_id),
        component_kind=component_kind,
        parent_aa=parent_aa,
        block=block,
    )
    cif_dict = _require_af3_cif_dict()
    return cif_dict.CifDict(block).to_string(), atom_names


def _extract_smiles_from_mmcif_block(block: Mapping[str, Any]) -> str:
    """Best-effort extraction of a SMILES descriptor from an mmCIF CCD block."""

    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        if hasattr(value, "__iter__"):
            return [str(v).strip() for v in value if str(v).strip()]
        v = str(value).strip()
        return [v] if v else []

    descriptor_types = _as_list(block.get("_pdbx_chem_comp_descriptor.type"))
    descriptors = _as_list(block.get("_pdbx_chem_comp_descriptor.descriptor"))
    invalid = {"?", ".", "NONE", "NULL"}

    if descriptor_types and len(descriptor_types) == len(descriptors):
        for typ, desc in zip(descriptor_types, descriptors, strict=True):
            if "SMILES_CANONICAL" in typ.upper() and desc.upper() not in invalid:
                return desc
        for typ, desc in zip(descriptor_types, descriptors, strict=True):
            if "SMILES" in typ.upper() and desc.upper() not in invalid:
                return desc
    for desc in descriptors:
        if desc.upper() not in invalid:
            return desc
    return ""


def _build_alias_entry(
    *,
    alias_name: str,
    alias_cfg: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    ccd_id = str(alias_cfg.get("ccd_id", alias_name)).strip().upper()
    if not ccd_id:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.ccd_id cannot be empty"
        )
    alias_kind = _infer_alias_kind(alias_cfg)
    if alias_kind == _ALIAS_KIND_PTM:
        parent_aa = _require_alias_parent_aa(
            alias_cfg.get("parent_aa"),
            alias_name=alias_name,
        )
    else:
        parent_aa = None
    atom_names_cfg = alias_cfg.get("atom_names")
    if alias_kind == _ALIAS_KIND_PTM and atom_names_cfg is None:
        raise ValueError(
            f"nonstandard_residues.aliases.{alias_name}.atom_names is required for PTM aliases"
        )
    source_fmt, source_value = _source_from_alias_cfg(alias_name, alias_cfg)
    source_fmt = source_fmt.lower()

    if source_fmt == "mmcif":
        mmcif_text = _load_text(source_value)
        _, raw_block = _parse_mmcif_block(mmcif_text)
        smiles = _extract_smiles_from_mmcif_block(raw_block)
        block, atom_names = _mmcif_source_to_user_ccd_block(
            alias_name=alias_name,
            mmcif_text=mmcif_text,
            ccd_id=ccd_id,
            component_kind=alias_kind,
            parent_aa=parent_aa,
            atom_names_cfg=atom_names_cfg,
        )
        if alias_kind == _ALIAS_KIND_PTM:
            _require_ptm_backbone_atom_names(alias_name, atom_names)
        alias_entry = {"ccd_id": ccd_id, "kind": alias_kind}
        if parent_aa:
            alias_entry["parent_aa"] = parent_aa
        if smiles:
            alias_entry["smiles"] = smiles
        return alias_entry, block

    Chem = _require_rdkit()
    if source_fmt == "smiles":
        mol = Chem.MolFromSmiles(str(source_value))
        if mol is None:
            raise ValueError(
                f"Failed to parse SMILES for nonstandard alias {alias_name}: {source_value!r}"
            )
    elif source_fmt == "sdf":
        source_path = Path(str(source_value)).expanduser()
        suppl = Chem.SDMolSupplier(str(source_path), removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            raise ValueError(
                f"Failed to parse SDF for nonstandard alias {alias_name}: {source_path}"
            )
    elif source_fmt == "mol":
        source_path = Path(str(source_value)).expanduser()
        mol = Chem.MolFromMolFile(str(source_path), removeHs=False)
        if mol is None:
            raise ValueError(
                f"Failed to parse MOL for nonstandard alias {alias_name}: {source_path}"
            )
    elif source_fmt == "pdb":
        source_path = Path(str(source_value)).expanduser()
        mol = Chem.MolFromPDBFile(
            str(source_path),
            sanitize=False,
            removeHs=False,
            proximityBonding=False,
        )
        if mol is None:
            raise ValueError(
                f"Failed to parse PDB for nonstandard alias {alias_name}: {source_path}"
            )
        mol = _apply_explicit_pdb_bonds(alias_name, mol, alias_cfg)
    else:
        raise ValueError(
            f"Unsupported nonstandard source format for alias {alias_name}: {source_fmt!r}"
        )

    atom_names: set[str] = set()
    if atom_names_cfg is not None:
        atom_names = _set_rdkit_atom_names(mol, atom_names_cfg)
    if alias_kind == _ALIAS_KIND_PTM:
        _require_ptm_backbone_atom_names(alias_name, atom_names)
    canonical_smiles = ""
    try:
        smiles_mol = Chem.Mol(mol)
        try:
            smiles_mol = Chem.RemoveHs(smiles_mol)
        except Exception:
            pass
        canonical_smiles = str(Chem.MolToSmiles(smiles_mol, canonical=True)).strip()
    except Exception:
        canonical_smiles = ""
    block = _rdkit_mol_to_user_ccd_block(
        mol=mol,
        ccd_id=ccd_id,
        component_kind=alias_kind,
        parent_aa=parent_aa,
    )
    alias_entry = {"ccd_id": ccd_id, "kind": alias_kind}
    if parent_aa:
        alias_entry["parent_aa"] = parent_aa
    if canonical_smiles:
        alias_entry["smiles"] = canonical_smiles
    return alias_entry, block


def build_nonstandard_aliases_and_user_ccd(
    raw_cfg: Any,
) -> tuple[dict[str, dict[str, str]], str | None]:
    cfg = _as_mapping(raw_cfg)
    if not cfg:
        return {}, None

    aliases_cfg = _as_mapping(cfg.get("aliases"))
    if not aliases_cfg:
        return {}, None

    alias_map: dict[str, dict[str, str]] = {}
    blocks: list[str] = []
    for raw_alias, raw_alias_cfg in aliases_cfg.items():
        alias = str(raw_alias).strip().upper()
        if not alias:
            raise ValueError("nonstandard_residues alias key cannot be empty")
        if len(alias) > 3:
            raise ValueError(
                f"nonstandard_residues alias {alias!r} is too long (max 3 characters)"
            )
        if alias in alias_map:
            raise ValueError(f"Duplicated nonstandard_residues alias: {alias}")
        alias_cfg = _as_mapping(raw_alias_cfg)
        alias_entry, block = _build_alias_entry(
            alias_name=alias,
            alias_cfg=alias_cfg,
        )
        alias_map[alias] = alias_entry
        blocks.append(str(block).strip() + "\n")

    if not blocks:
        return alias_map, None
    return alias_map, "\n".join(blocks)


def resolve_nonstandard_alias(
    token: str,
    *,
    alias_map: Mapping[str, Mapping[str, str]] | None,
    require_ptm: bool = False,
) -> tuple[str, str | None]:
    key = str(token or "").strip().upper()
    if not key:
        raise ValueError("Empty PTM/alias token")
    if not alias_map:
        return key, None
    entry = alias_map.get(key)
    if entry is None:
        return key, None
    ccd_id = str(entry.get("ccd_id", "")).strip().upper()
    parent_aa = str(entry.get("parent_aa", "")).strip().upper()
    kind = str(entry.get("kind", "")).strip().lower()
    if not ccd_id:
        raise ValueError(f"Alias {key!r} resolves to empty ccd_id")
    if not kind:
        kind = _ALIAS_KIND_PTM if parent_aa else _ALIAS_KIND_LIGAND
    if kind not in {_ALIAS_KIND_PTM, _ALIAS_KIND_LIGAND}:
        raise ValueError(f"Alias {key!r} has invalid kind: {kind!r}")
    if require_ptm and kind != _ALIAS_KIND_PTM:
        raise ValueError(
            f"Alias {key!r} is configured as ligand-only and cannot be used in protein PTM sequence."
        )
    if parent_aa and parent_aa not in _AA_ALPHABET:
        raise ValueError(f"Alias {key!r} has invalid parent_aa: {parent_aa!r}")
    return ccd_id, (parent_aa or None)
