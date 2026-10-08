import json
import logging
import pathlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Iterable, Mapping, Sequence, Set

try:
    import Bio.PDB  # type: ignore
    from Bio.PDB import PDBIO, Select  # type: ignore
    from Bio.PDB.MMCIFParser import MMCIFParser  # type: ignore

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    Bio = None  # type: ignore
    PDBIO = None  # type: ignore
    Select = object  # type: ignore
    MMCIFParser = None  # type: ignore

# RDKit imports for SMILES to 3D conformer generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. SMILES to mol2 conversion may not work properly.")

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes


_ONE_TO_THREE_AA = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


class ExtraResidueError(RuntimeError):
    """Raised when hetero-residue extraction or ligand conversion fails."""


class HeteroResidueSelector(Select):
    """选择器类，用于选择特定的非标准残基。"""

    def __init__(
        self, target_resname: str, target_resid: int = None, target_chain: str = None
    ):
        self.target_resname = target_resname
        self.target_resid = target_resid
        self.target_chain = target_chain

    def accept_residue(self, residue):
        """接受符合条件的残基。"""
        resname = residue.get_resname().strip()

        # 检查残基名称
        if resname != self.target_resname:
            return False

        # 如果指定了残基ID，检查是否匹配
        if self.target_resid is not None:
            if residue.get_id()[1] != self.target_resid:
                return False

        # 如果指定了链ID，检查是否匹配
        if self.target_chain is not None:
            if residue.get_parent().get_id() != self.target_chain:
                return False

        return True


def extract_hetero_residues_to_mol2(
    pdb_path: str,
    output_dir: str,
    exclude_water: bool = True,
    exclude_ions: bool = True,
) -> Dict[str, str]:
    """
    从PDB/CIF文件中提取非标准残基并保存为mol2文件。

    Args:
        pdb_path: PDB或CIF文件路径
        output_dir: 输出目录
        exclude_water: 是否排除水分子
        exclude_ions: 是否排除离子

    Returns:
        Dict mapping residue names to mol2 file paths
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError(
            "Biopython is required for extracting hetero residues (`pip install biopython`)."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析结构文件
    structure = _parse_structure_file(pdb_path)

    # 获取所有非标准残基
    hetero_residues = _get_hetero_residues(structure, exclude_water, exclude_ions)

    if not hetero_residues:
        logging.info(f"No hetero residues found in {pdb_path}")
        return {}

    # 按残基类型分组并保存
    mol2_files = {}

    for resname, residue_instances in hetero_residues.items():
        mol2_path = output_dir / f"{resname}.mol2"
        _save_residue_instances_to_mol2(
            structure, residue_instances, mol2_path, resname
        )
        mol2_files[resname] = str(mol2_path)
        logging.info(f"Saved {resname} to {mol2_path}")

    return mol2_files


def _parse_structure_file(pdb_path: str):
    """解析PDB或CIF文件。"""
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError(
            "Biopython is required for parsing PDB/CIF files (`pip install biopython`)."
        )
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = Bio.PDB.PDBParser(QUIET=True)

    return parser.get_structure("structure", pdb_path)


def _get_hetero_residues(
    structure, exclude_water: bool = True, exclude_ions: bool = True
) -> Dict[str, List]:
    """获取所有非标准残基，按类型分组。"""

    # 定义要排除的残基
    excluded_residues = set()

    if exclude_water:
        excluded_residues.update(["HOH", "WAT", "H2O", "TIP", "SOL"])

    if exclude_ions:
        common_ions = [
            "NA",
            "CL",
            "K",
            "CA",
            "MG",
            "ZN",
            "FE",
            "MN",
            "CU",
            "NI",
            "CO",
            "SO4",
            "PO4",
            "NO3",
            "BR",
            "I",
            "F",
            "LI",
            "RB",
            "CS",
            "BA",
            "SR",
        ]
        excluded_residues.update(common_ions)

    # 标准氨基酸残基
    standard_aa = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

    # 标准核酸残基
    standard_na = {"A", "T", "G", "C", "U", "DA", "DT", "DG", "DC", "DU"}

    hetero_residues = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()

                # 跳过标准残基和排除的残基
                if (
                    resname in standard_aa
                    or resname in standard_na
                    or resname in excluded_residues
                ):
                    continue

                # 跳过水分子（hetflag检查）
                hetflag, resseq, icode = residue.get_id()
                if hetflag == "W":  # 水分子
                    continue

                # 收集非标准残基信息
                residue_info = {
                    "residue": residue,
                    "chain_id": chain.get_id(),
                    "resseq": resseq,
                    "icode": icode,
                }

                if resname not in hetero_residues:
                    hetero_residues[resname] = []

                hetero_residues[resname].append(residue_info)

    return hetero_residues


def _save_residue_instances_to_mol2(
    structure, residue_instances: List[Dict], output_path: Path, resname: str
) -> None:
    """保存残基实例到mol2文件，保持完整的分子几何结构。"""
    if len(residue_instances) == 1:
        _save_single_residue_to_mol2(
            structure, residue_instances[0], output_path, resname
        )
        return

    success_count = 0
    failed_indices: list[str] = []
    for i, residue_info in enumerate(residue_instances):
        if i == 0:
            current_output = output_path
        else:
            current_output = output_path.with_name(f"{output_path.stem}_{i+1}.mol2")
        try:
            _save_single_residue_to_mol2(
                structure, residue_info, current_output, resname
            )
            success_count += 1
        except ExtraResidueError as exc:
            failed_indices.append(f"#{i+1}: {exc}")

    if success_count == 0:
        details = "; ".join(failed_indices) if failed_indices else "<unknown>"
        raise ExtraResidueError(
            f"Failed to save any residue instance for {resname}: {details}"
        )
    if failed_indices:
        logging.warning(
            "Saved %d/%d residue instances for %s; failures: %s",
            success_count,
            len(residue_instances),
            resname,
            "; ".join(failed_indices),
        )


def _save_single_residue_to_mol2(
    structure, residue_info: Dict, output_path: Path, resname: str
) -> None:
    """保存单个残基实例到mol2文件。"""
    temp_pdb = output_path.with_suffix(".temp.pdb")
    selector = HeteroResidueSelector(
        target_resname=resname,
        target_resid=residue_info["resseq"],
        target_chain=residue_info["chain_id"],
    )

    try:
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(temp_pdb), selector)
        _validate_temp_pdb(temp_pdb, resname)
        _convert_pdb_to_mol2_with_openbabel(temp_pdb, output_path, resname)
    except Exception as exc:
        raise ExtraResidueError(
            f"Failed to export residue {resname} (chain={residue_info['chain_id']}, resseq={residue_info['resseq']})"
        ) from exc
    finally:
        if temp_pdb.exists():
            try:
                temp_pdb.unlink()
            except OSError as exc:
                logging.warning("Failed to remove temporary file %s: %s", temp_pdb, exc)


def _validate_temp_pdb(pdb_path: Path, expected_resname: str) -> None:
    """验证临时PDB文件是否包含预期的残基。"""
    lines = pdb_path.read_text(encoding="utf-8").splitlines()
    atom_count = 0
    has_target_residue = False

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            atom_count += 1
            resname = line[17:20].strip()
            if resname == expected_resname:
                has_target_residue = True

    if atom_count == 0:
        raise ExtraResidueError(f"No atoms found in temporary PDB file: {pdb_path}")
    if not has_target_residue:
        raise ExtraResidueError(
            f"Target residue {expected_resname} not found in temporary PDB file: {pdb_path}"
        )


def _convert_pdb_to_mol2_with_openbabel(
    pdb_path: Path, mol2_path: Path, resname: str
) -> None:
    """使用OpenBabel将PDB转换为mol2格式。"""
    cmd = [
        global_config.obabel_path or "obabel",
        str(pdb_path),
        "-O",
        str(mol2_path),
        "-h",
        "--title",
        resname,
        "-p",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ExtraResidueError(
            f"obabel command failed for {resname}: {exc.stderr}"
        ) from exc
    except FileNotFoundError as exc:
        raise ExtraResidueError(
            "obabel command not found. Please install OpenBabel."
        ) from exc

    if not mol2_path.exists() or mol2_path.stat().st_size <= 0:
        raise ExtraResidueError(f"mol2 file was not created or is empty: {mol2_path}")
    _validate_mol2_file(mol2_path, resname)


def _validate_mol2_file(mol2_path: Path, expected_resname: str) -> None:
    """验证mol2文件是否正确生成。"""
    content = mol2_path.read_text(encoding="utf-8")
    required_sections = ["@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM"]
    for section in required_sections:
        if section not in content:
            raise ExtraResidueError(f"Missing required section {section} in {mol2_path}")

    lines = content.split("\n")
    atom_section = False
    atom_count = 0
    for line in lines:
        if line.strip() == "@<TRIPOS>ATOM":
            atom_section = True
            continue
        if line.strip().startswith("@<TRIPOS>") and atom_section:
            break
        if atom_section and line.strip():
            atom_count += 1

    if atom_count == 0:
        raise ExtraResidueError(f"No atoms found in mol2 file: {mol2_path}")
    logging.info(
        "Successfully validated mol2 file %s for %s with %d atoms",
        mol2_path,
        expected_resname,
        atom_count,
    )


def generate_pyrosetta_params_commands(
    mol2_files: Dict[str, str],
    output_dir: str,
) -> List[str]:
    """
    生成用于创建PyRosetta参数文件的命令列表。

    Args:
        mol2_files: 残基名称到mol2文件路径的映射
        output_dir: 参数文件输出目录

    Returns:
        List of command strings to generate params files
    """
    commands: List[str] = []
    script = getattr(global_config, "molfile_to_params", None)
    if not script:
        raise RuntimeError(
            "molfile_to_params path is not configured; cannot generate PyRosetta params."
        )
    script_path = Path(str(script)).expanduser()
    if not script_path.exists():
        raise FileNotFoundError(
            f"molfile_to_params script not found: {script_path}. "
            "Please mount Rosetta and set GH_MOLFILE_TO_PARAMS."
        )

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    for resname, mol2_path in mol2_files.items():
        mol2_file = Path(str(mol2_path)).expanduser()
        if not mol2_file.exists():
            raise FileNotFoundError(f"mol2 file not found for {resname}: {mol2_file}")
        params_path = output_path / f"{resname}.params"
        companion_pdb = output_path / f"{resname}_0001.pdb"
        if params_path.exists() and companion_pdb.exists():
            logging.info("Reusing existing params file for %s", resname)
            commands.append(f"# reuse existing params for {resname}: {params_path}")
            continue

        params_path = output_path / f"{resname}.params"
        conformer_pdb_path = output_path / f"{resname}_0001.pdb"
        if params_path.exists() and conformer_pdb_path.exists():
            logging.info(
                "Reusing existing params outputs for %s: %s and %s",
                resname,
                params_path,
                conformer_pdb_path,
            )
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--keep-names",
            "--clobber",
            "-n",
            str(resname),
            str(mol2_file),
        ]
        commands.append(" ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=str(output_path),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            raise RuntimeError(
                f"Failed to generate params for {resname} using {mol2_file}. "
                f"stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        logging.info("Generated params file for %s", resname)
    return commands


def _normalize_ccd_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        return [raw] if raw else []
    if isinstance(value, Sequence):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _infer_polymer_script_path() -> Path:
    script = getattr(global_config, "molfile_to_params", None)
    if not script:
        raise RuntimeError(
            "molfile_to_params path is not configured; cannot generate Rosetta polymer params."
        )
    script_path = Path(str(script)).expanduser()
    polymer_path = script_path.with_name("molfile_to_params_polymer.py")
    if not polymer_path.exists():
        raise FileNotFoundError(
            f"Rosetta polymer params script not found: {polymer_path}"
        )
    return polymer_path


def _build_ccd_polymer_graph(ccd_entry: Mapping[str, object]) -> dict[str, object]:
    atom_ids = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.atom_id"))
    atom_types = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.type_symbol"))
    xs = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_x_ideal"))
    ys = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_y_ideal"))
    zs = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_z_ideal"))
    bond_a1 = _normalize_ccd_field(ccd_entry.get("_chem_comp_bond.atom_id_1"))
    bond_a2 = _normalize_ccd_field(ccd_entry.get("_chem_comp_bond.atom_id_2"))
    bond_orders = _normalize_ccd_field(ccd_entry.get("_chem_comp_bond.value_order"))
    alt_atom_ids = _normalize_ccd_field(ccd_entry.get("_chem_comp_atom.alt_atom_id"))
    component_atom_ids = _normalize_ccd_field(
        ccd_entry.get("_chem_comp_atom.pdbx_component_atom_id")
    )
    backbone_flags = _normalize_ccd_field(
        ccd_entry.get("_chem_comp_atom.pdbx_backbone_atom_flag")
    )
    leaving_flags = _normalize_ccd_field(
        ccd_entry.get("_chem_comp_atom.pdbx_leaving_atom_flag")
    )

    atom_count = len(atom_ids)
    if not atom_count:
        raise ValueError("CCD entry is missing atom ids.")
    expected_atom_fields = {
        "_chem_comp_atom.type_symbol": atom_types,
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": xs,
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ys,
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": zs,
    }
    bad_atom_fields = {
        key: len(values)
        for key, values in expected_atom_fields.items()
        if len(values) != atom_count
    }
    if bad_atom_fields:
        raise ValueError(
            f"CCD entry has inconsistent atom fields: {bad_atom_fields}, expected {atom_count}"
        )

    for field_name, values in (
        ("_chem_comp_atom.alt_atom_id", alt_atom_ids),
        ("_chem_comp_atom.pdbx_component_atom_id", component_atom_ids),
        ("_chem_comp_atom.pdbx_backbone_atom_flag", backbone_flags),
        ("_chem_comp_atom.pdbx_leaving_atom_flag", leaving_flags),
    ):
        if values and len(values) != atom_count:
            raise ValueError(
                f"CCD entry field {field_name} length {len(values)} != atom count {atom_count}"
            )

    bond_count = len(bond_a1)
    if len(bond_a2) != bond_count:
        raise ValueError("CCD bond atom_id_1 and atom_id_2 lengths do not match.")
    if bond_orders and len(bond_orders) != bond_count:
        raise ValueError("CCD bond value_order length does not match bond count.")
    if not bond_orders:
        bond_orders = ["SING"] * bond_count

    name_to_index = {name: idx + 1 for idx, name in enumerate(atom_ids)}
    atom_elem = {
        name: atom_types[idx].strip().upper() for idx, name in enumerate(atom_ids)
    }
    coords = {
        name: (float(xs[idx]), float(ys[idx]), float(zs[idx]))
        for idx, name in enumerate(atom_ids)
    }
    neighbors: dict[str, set[str]] = {name: set() for name in atom_ids}
    bond_order_map: dict[frozenset[str], str] = {}
    metadata_names: dict[str, set[str]] = {}
    backbone_map = {
        atom_ids[idx]: backbone_flags[idx].strip().upper() == "Y"
        for idx in range(atom_count)
    } if backbone_flags else {name: False for name in atom_ids}
    leaving_map = {
        atom_ids[idx]: leaving_flags[idx].strip().upper() == "Y"
        for idx in range(atom_count)
    } if leaving_flags else {name: False for name in atom_ids}

    for idx, atom_name in enumerate(atom_ids):
        labels = {atom_name.strip().upper()}
        if alt_atom_ids:
            labels.add(alt_atom_ids[idx].strip().upper())
        if component_atom_ids:
            labels.add(component_atom_ids[idx].strip().upper())
        labels.discard("")
        metadata_names[atom_name] = labels

    for lhs, rhs, order in zip(bond_a1, bond_a2, bond_orders, strict=True):
        if lhs not in name_to_index or rhs not in name_to_index:
            raise ValueError(f"CCD bond references unknown atom: {lhs!r}, {rhs!r}")
        neighbors.setdefault(lhs, set()).add(rhs)
        neighbors.setdefault(rhs, set()).add(lhs)
        bond_order_map[frozenset((lhs, rhs))] = str(order).strip().upper()

    return {
        "atom_ids": atom_ids,
        "name_to_index": name_to_index,
        "atom_elem": atom_elem,
        "coords": coords,
        "neighbors": neighbors,
        "bond_order_map": bond_order_map,
        "metadata_names": metadata_names,
        "backbone_flags": backbone_map,
        "leaving_flags": leaving_map,
    }


def _graph_bond_order(graph: Mapping[str, object], lhs: str, rhs: str) -> str:
    bond_order_map = graph["bond_order_map"]  # type: ignore[index]
    return str(bond_order_map.get(frozenset((lhs, rhs)), ""))


def _graph_atom_has_label(graph: Mapping[str, object], atom_name: str, label: str) -> bool:
    labels = graph["metadata_names"]  # type: ignore[index]
    return str(label).strip().upper() in labels.get(atom_name, set())


def _graph_atom_has_label_prefix(
    graph: Mapping[str, object], atom_name: str, prefix: str
) -> bool:
    labels = graph["metadata_names"]  # type: ignore[index]
    needle = str(prefix).strip().upper()
    return any(label.startswith(needle) for label in labels.get(atom_name, set()))


def _natural_atom_name_key(name: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(name).strip().upper())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _infer_polymer_backbone_roles(graph: Mapping[str, object]) -> dict[str, str]:
    atom_ids = graph["atom_ids"]  # type: ignore[index]
    atom_elem = graph["atom_elem"]  # type: ignore[index]
    neighbors = graph["neighbors"]  # type: ignore[index]
    backbone_flags = graph["backbone_flags"]  # type: ignore[index]

    metadata_roles: dict[str, str] = {}
    for role, label in (("n", "N"), ("ca", "CA"), ("c", "C"), ("o", "O")):
        matches = [name for name in atom_ids if _graph_atom_has_label(graph, name, label)]
        if len(matches) == 1:
            metadata_roles[role] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f"CCD entry has ambiguous metadata backbone label {label!r}: {matches}")
    if len(metadata_roles) == 4:
        return metadata_roles

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for carbonyl_c in atom_ids:
        if atom_elem.get(carbonyl_c) != "C":
            continue
        oxygens = [
            nbr
            for nbr in neighbors.get(carbonyl_c, set())
            if atom_elem.get(nbr) == "O" and _graph_bond_order(graph, carbonyl_c, nbr) in {"DOUB", "DOUBLE"}
        ]
        if len(oxygens) != 1:
            continue
        carbonyl_o = oxygens[0]
        for alpha_c in neighbors.get(carbonyl_c, set()):
            if atom_elem.get(alpha_c) != "C":
                continue
            if _graph_bond_order(graph, carbonyl_c, alpha_c) not in {"SING", "SINGLE"}:
                continue
            nitrogens = [
                nbr
                for nbr in neighbors.get(alpha_c, set())
                if atom_elem.get(nbr) == "N" and _graph_bond_order(graph, alpha_c, nbr) in {"SING", "SINGLE"}
            ]
            for n_atom in nitrogens:
                key = (n_atom, alpha_c, carbonyl_c, carbonyl_o)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "n": n_atom,
                        "ca": alpha_c,
                        "c": carbonyl_c,
                        "o": carbonyl_o,
                    }
                )
    if not candidates:
        raise ValueError("Unable to infer polymer backbone roles from CCD bonding.")

    compatible = [
        candidate
        for candidate in candidates
        if all(candidate.get(role) == atom_name for role, atom_name in metadata_roles.items())
    ]
    if compatible:
        candidates = compatible
    if len(candidates) == 1:
        return candidates[0]

    scored: list[tuple[int, dict[str, str]]] = []
    for candidate in candidates:
        score = 0
        for role, atom_name in candidate.items():
            score += 3 if metadata_roles.get(role) == atom_name else 0
            score += 1 if backbone_flags.get(atom_name, False) else 0
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) == 1:
        return best[0]
    raise ValueError(f"Unable to disambiguate polymer backbone roles from CCD: {best}")


def _select_polymer_connection_atom(
    graph: Mapping[str, object],
    candidates: Sequence[str],
    *,
    preferred_prefixes: Sequence[str],
    context: str,
    allow_equivalent_hydrogens: bool = False,
) -> str | None:
    if not candidates:
        return None
    unique = list(dict.fromkeys(str(name) for name in candidates))
    for prefix in preferred_prefixes:
        matches = [
            name for name in unique if _graph_atom_has_label_prefix(graph, name, prefix)
        ]
        if len(matches) == 1:
            return matches[0]
    leaving_flags = graph["leaving_flags"]  # type: ignore[index]
    leaving = [name for name in unique if leaving_flags.get(name, False)]
    if len(leaving) == 1:
        return leaving[0]
    if len(unique) == 1:
        return unique[0]
    atom_elem = graph["atom_elem"]  # type: ignore[index]
    if allow_equivalent_hydrogens and all(atom_elem.get(name) == "H" for name in unique):
        return sorted(unique, key=_natural_atom_name_key)[0]
    raise ValueError(f"CCD entry has ambiguous {context}: {unique}")


def _infer_existing_polymer_connection_atoms(
    graph: Mapping[str, object], backbone: Mapping[str, str]
) -> tuple[str | None, str | None]:
    atom_elem = graph["atom_elem"]  # type: ignore[index]
    neighbors = graph["neighbors"]  # type: ignore[index]
    upper_candidates = [
        name
        for name in neighbors.get(backbone["c"], set())
        if atom_elem.get(name) == "O" and name != backbone["o"]
    ]
    lower_candidates = [
        name
        for name in neighbors.get(backbone["n"], set())
        if atom_elem.get(name) == "H"
    ]
    upper = _select_polymer_connection_atom(
        graph,
        upper_candidates,
        preferred_prefixes=("OXT",),
        context="upper-connection oxygen candidates",
    )
    lower = _select_polymer_connection_atom(
        graph,
        lower_candidates,
        preferred_prefixes=("H2",),
        context="lower-connection hydrogen candidates",
        allow_equivalent_hydrogens=True,
    )
    return upper, lower


def _vector_add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _vector_sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _vector_scale(vec: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (vec[0] * factor, vec[1] * factor, vec[2] * factor)


def _vector_norm(vec: tuple[float, float, float]) -> float:
    return float((vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5)


def _normalized_vector(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _vector_norm(vec)
    if norm <= 1e-6:
        return (1.0, 0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _make_unique_atom_name(existing: set[str], preferred: str) -> str:
    candidate = preferred
    if candidate not in existing:
        return candidate
    suffix = 1
    while f"{preferred}{suffix}" in existing:
        suffix += 1
    return f"{preferred}{suffix}"


def _append_ccd_atom(
    ccd_entry: dict[str, object],
    *,
    atom_name: str,
    element: str,
    coords: tuple[float, float, float],
    is_backbone: bool = False,
    is_leaving: bool = False,
) -> None:
    for key in (
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    ):
        ccd_entry.setdefault(key, [])
    ccd_entry["_chem_comp_atom.atom_id"].append(atom_name)  # type: ignore[index]
    ccd_entry["_chem_comp_atom.type_symbol"].append(element)  # type: ignore[index]
    ccd_entry["_chem_comp_atom.charge"].append("0")  # type: ignore[index]
    ccd_entry["_chem_comp_atom.pdbx_model_Cartn_x_ideal"].append(f"{coords[0]:.3f}")  # type: ignore[index]
    ccd_entry["_chem_comp_atom.pdbx_model_Cartn_y_ideal"].append(f"{coords[1]:.3f}")  # type: ignore[index]
    ccd_entry["_chem_comp_atom.pdbx_model_Cartn_z_ideal"].append(f"{coords[2]:.3f}")  # type: ignore[index]

    for key, value in (
        ("_chem_comp_atom.alt_atom_id", atom_name),
        ("_chem_comp_atom.pdbx_component_atom_id", atom_name),
        ("_chem_comp_atom.pdbx_backbone_atom_flag", "Y" if is_backbone else "N"),
        ("_chem_comp_atom.pdbx_leaving_atom_flag", "Y" if is_leaving else "N"),
    ):
        if key in ccd_entry:
            ccd_entry[key].append(value)  # type: ignore[index]


def _append_ccd_bond(
    ccd_entry: dict[str, object],
    *,
    lhs: str,
    rhs: str,
    order: str,
) -> None:
    for key in (
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order",
        "_chem_comp_bond.pdbx_aromatic_flag",
    ):
        ccd_entry.setdefault(key, [])
    ccd_entry["_chem_comp_bond.atom_id_1"].append(lhs)  # type: ignore[index]
    ccd_entry["_chem_comp_bond.atom_id_2"].append(rhs)  # type: ignore[index]
    ccd_entry["_chem_comp_bond.value_order"].append(order)  # type: ignore[index]
    ccd_entry["_chem_comp_bond.pdbx_aromatic_flag"].append("N")  # type: ignore[index]


def _prepare_rosetta_polymer_ccd_entry(
    ccd_entry: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    graph = _build_ccd_polymer_graph(ccd_entry)
    backbone = _infer_polymer_backbone_roles(graph)
    upper_name, lower_name = _infer_existing_polymer_connection_atoms(graph, backbone)

    augmented: dict[str, object] = {
        str(key): list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else value
        for key, value in ccd_entry.items()
    }
    coords = graph["coords"]  # type: ignore[index]
    neighbors = graph["neighbors"]  # type: ignore[index]
    existing_names = set(graph["atom_ids"])  # type: ignore[index]

    if upper_name is None:
        upper_name = _make_unique_atom_name(existing_names, "OXT")
        c_pos = coords[backbone["c"]]
        o_pos = coords[backbone["o"]]
        ca_pos = coords[backbone["ca"]]
        direction = _normalized_vector(
            _vector_add(_vector_sub(c_pos, o_pos), _vector_sub(c_pos, ca_pos))
        )
        upper_pos = _vector_add(c_pos, _vector_scale(direction, 1.25))
        _append_ccd_atom(
            augmented,
            atom_name=upper_name,
            element="O",
            coords=upper_pos,
            is_backbone=True,
            is_leaving=True,
        )
        _append_ccd_bond(augmented, lhs=backbone["c"], rhs=upper_name, order="SING")
        existing_names.add(upper_name)

    if lower_name is None:
        lower_name = _make_unique_atom_name(existing_names, "H2")
        n_pos = coords[backbone["n"]]
        ca_pos = coords[backbone["ca"]]
        direction = _vector_sub(n_pos, ca_pos)
        for hydrogen in (
            name
            for name in neighbors.get(backbone["n"], set())
            if graph["atom_elem"].get(name) == "H"  # type: ignore[index]
        ):
            direction = _vector_add(direction, _vector_sub(n_pos, coords[hydrogen]))
        lower_pos = _vector_add(n_pos, _vector_scale(_normalized_vector(direction), 1.0))
        _append_ccd_atom(
            augmented,
            atom_name=lower_name,
            element="H",
            coords=lower_pos,
            is_backbone=True,
            is_leaving=True,
        )
        _append_ccd_bond(augmented, lhs=backbone["n"], rhs=lower_name, order="SING")
        existing_names.add(lower_name)

    prepared_graph = _build_ccd_polymer_graph(augmented)
    name_to_index = prepared_graph["name_to_index"]  # type: ignore[index]
    upper_hydrogens = sorted(
        name_to_index[name]
        for name in prepared_graph["neighbors"].get(upper_name, set())  # type: ignore[index]
        if prepared_graph["atom_elem"].get(name) == "H"  # type: ignore[index]
    )

    tags = {
        "poly_n_bb": name_to_index[backbone["n"]],
        "poly_ca_bb": name_to_index[backbone["ca"]],
        "poly_c_bb": name_to_index[backbone["c"]],
        "poly_o_bb": name_to_index[backbone["o"]],
        "poly_upper": name_to_index[upper_name],
        "poly_lower": name_to_index[lower_name],
        "root": name_to_index[backbone["n"]],
        "poly_ignore": upper_hydrogens,
    }
    return augmented, tags


def _infer_rosetta_polymer_tags_from_ccd(
    ccd_entry: Mapping[str, object],
) -> dict[str, object]:
    _prepared_ccd, tags = _prepare_rosetta_polymer_ccd_entry(ccd_entry)

    return tags


def _infer_nonstandard_alias_rosetta_properties(
    alias_entry: Mapping[str, object],
    ccd_entry: Mapping[str, object],
) -> list[str]:
    override = alias_entry.get("rosetta_properties", alias_entry.get("polymer_properties"))
    if override is not None:
        if isinstance(override, str):
            return [token for token in override.split() if token]
        return [str(token).strip() for token in override if str(token).strip()]

    graph = _build_ccd_polymer_graph(ccd_entry)
    backbone = _infer_polymer_backbone_roles(graph)
    atom_ids = graph["atom_ids"]  # type: ignore[index]
    atom_elem = graph["atom_elem"]  # type: ignore[index]
    neighbors = graph["neighbors"]  # type: ignore[index]
    backbone_atoms = set(backbone.values())
    upper_candidates = [
        name
        for name in neighbors.get(backbone["c"], set())
        if atom_elem.get(name) == "O" and name != backbone["o"]
    ]
    for upper_name in upper_candidates:
        backbone_atoms.add(upper_name)

    ca_hydrogens = [
        name for name in neighbors.get(backbone["ca"], set()) if atom_elem.get(name) == "H"
    ]
    ca_sidechain_heavy_neighbors = [
        name
        for name in neighbors.get(backbone["ca"], set())
        if atom_elem.get(name) != "H" and name not in {backbone["n"], backbone["c"]}
    ]
    properties = ["PROTEIN", "ALPHA_AA"]
    if len(ca_hydrogens) == 1 and len(ca_sidechain_heavy_neighbors) == 1:
        properties.append("L_AA")
    else:
        properties.append("ACHIRAL_BACKBONE")
        sidechain_heavy = [
            name
            for name in atom_ids
            if atom_elem.get(name) != "H" and name not in backbone_atoms
        ]
        if len(sidechain_heavy) > 1:
            properties.append("ACHIRAL_SIDECHAIN")
    return properties


def _write_rosetta_polymer_sdf_for_ccd(
    ccd_entry: Mapping[str, object],
    *,
    output_dir: str,
    resname: str,
    parent_aa: str,
    properties: Sequence[str],
) -> Path:
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required to write Rosetta polymer SDF files.")
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    sdf_path = output_path / f"{resname}.polymer.sdf"
    prepared_ccd, tags = _prepare_rosetta_polymer_ccd_entry(ccd_entry)
    mol = _build_rdkit_mol_from_ccd_entry(prepared_ccd, str(resname))
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    footer_lines = [
        f"M  ROOT {tags['root']}",
        f"M  POLY_N_BB {tags['poly_n_bb']}",
        f"M  POLY_CA_BB {tags['poly_ca_bb']}",
        f"M  POLY_C_BB {tags['poly_c_bb']}",
        f"M  POLY_O_BB {tags['poly_o_bb']}",
        f"M  POLY_UPPER {tags['poly_upper']}",
        f"M  POLY_LOWER {tags['poly_lower']}",
        "M  POLY_PROPERTIES " + " ".join(str(token).strip() for token in properties if str(token).strip()),
        "$$$$",
    ]
    if tags["poly_ignore"]:
        footer_lines.insert(
            6,
            "M  POLY_IGNORE " + " ".join(str(i) for i in tags["poly_ignore"]),
        )
    text = sdf_path.read_text(encoding="utf-8")
    if "$$$$\n" not in text:
        raise RuntimeError(f"Unexpected SDF payload for {resname}: missing '$$$$' terminator.")
    sdf_path.write_text(text.replace("$$$$\n", "\n".join(footer_lines) + "\n"), encoding="utf-8")
    return sdf_path


def _postprocess_free_residue_polymer_params_text(params_text: str) -> str:
    lines = params_text.splitlines()
    filtered: list[str] = []
    hpol_removed = False
    lower_renamed = False
    icoor_h_indices: list[int] = []

    for idx, line in enumerate(lines):
        if line.startswith("ATOM") and " Hpol " in line and line.split()[1] == "H":
            hpol_removed = True
            continue
        if line.startswith("ICOOR_INTERNAL"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "H":
                icoor_h_indices.append(len(filtered))
        filtered.append(line)

    if len(icoor_h_indices) >= 2:
        lower_idx = icoor_h_indices[-1]
        filtered[lower_idx] = filtered[lower_idx].replace("ICOOR_INTERNAL    H", "ICOOR_INTERNAL LOWER", 1)
        lower_renamed = True

    if not lower_renamed:
        raise RuntimeError(
            "Failed to normalize Rosetta polymer params for free-residue LOWER placeholder."
        )
    return "\n".join(filtered) + ("\n" if params_text.endswith("\n") else "")


def generate_rosetta_polymer_params_for_ccd(
    *,
    ccd_entry: Mapping[str, object],
    resname: str,
    parent_aa: str,
    output_dir: str,
    properties: Sequence[str],
) -> Path:
    polymer_script = _infer_polymer_script_path()
    sdf_path = _write_rosetta_polymer_sdf_for_ccd(
        ccd_entry,
        output_dir=output_dir,
        resname=resname,
        parent_aa=parent_aa,
        properties=properties,
    )
    parent_three = _ONE_TO_THREE_AA.get(str(parent_aa).strip().upper())
    if parent_three is None:
        raise ValueError(f"Unsupported parent_aa for Rosetta polymer params: {parent_aa!r}")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(polymer_script),
        "-i",
        str(sdf_path.resolve()),
        "-n",
        str(resname),
        "--use-parent-rotamers",
        parent_three,
        "--no-pdb",
    ]
    try:
        subprocess.run(
            cmd,
            cwd=str(output_path),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to generate polymer params for {resname} using {sdf_path}. "
            f"stdout={stdout!r} stderr={stderr!r}"
        ) from exc

    params_path = output_path / f"{resname}.params"
    if not params_path.exists():
        raise RuntimeError(f"Rosetta polymer params generation produced no params file for {resname}.")
    params_path.write_text(
        _postprocess_free_residue_polymer_params_text(
            params_path.read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    return params_path


def _sanitize_resname(candidate: Optional[str], fallback: str = "LIG") -> str:
    """Sanitize protomer names so they become valid residue names."""

    if candidate:
        sanitized = "".join(ch for ch in candidate.upper() if ch.isalnum() or ch == "_")
    else:
        sanitized = ""

    if not sanitized:
        sanitized = fallback

    # Rosetta/pyrosetta expects <=3 characters for residue names.
    return sanitized[:3]


def _normalize_ligand_resnames(ligand_resnames: object) -> Dict[str, str]:
    """Normalize user-provided ligand resname mapping."""

    if ligand_resnames is None:
        return {}
    if isinstance(ligand_resnames, Mapping):
        return {
            str(k).strip(): str(v).strip()
            for k, v in ligand_resnames.items()
            if str(k).strip() and str(v).strip()
        }
    if hasattr(ligand_resnames, "items"):
        try:
            return {
                str(k).strip(): str(v).strip()
                for k, v in ligand_resnames.items()
                if str(k).strip() and str(v).strip()
            }
        except Exception:
            return {}
    return {}


def _resolve_ligand_resname(
    *,
    protomer_key: str,
    protomer_name: str,
    ligand_resnames: Mapping[str, str],
    ccd_code: Optional[str] = None,
) -> str:
    """Resolve params/residue name for a ligand entity."""

    ccd_code_norm = str(ccd_code or "").strip().upper()
    fallback = ccd_code_norm or f"L_{protomer_key}"

    keys_to_try = []
    if ccd_code_norm:
        keys_to_try.extend(
            [
                f"{protomer_key}:{ccd_code_norm}",
                f"{protomer_name}:{ccd_code_norm}",
                ccd_code_norm,
            ]
        )
    keys_to_try.extend([protomer_key, protomer_name])

    for key in keys_to_try:
        value = ligand_resnames.get(key)
        if value is not None:
            return _sanitize_resname(str(value), fallback=fallback)

    return _sanitize_resname(fallback, fallback="LIG")


def get_merged_ligand_resnames(
    ligand_resnames: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return merged ligand resname map from global config and call-site overrides."""

    try:
        configured_map = _normalize_ligand_resnames(
            getattr(global_config, "ligand_resnames", None)
        )
    except Exception:
        configured_map = {}
    override_map = _normalize_ligand_resnames(ligand_resnames)
    return {**configured_map, **override_map}


def resolve_ligand_resname_for_protomer(
    *,
    protomer_key: str,
    protomer_name: str,
    ligand_resnames: Optional[Mapping[str, str]] = None,
    ccd_code: Optional[str] = None,
) -> str:
    """Resolve ligand params name with merged global+override mapping."""

    merged_resname_map = get_merged_ligand_resnames(ligand_resnames)
    return _resolve_ligand_resname(
        protomer_key=protomer_key,
        protomer_name=protomer_name,
        ligand_resnames=merged_resname_map,
        ccd_code=ccd_code,
    )


def _remap_single_smiles_target_to_structure_resname(
    *,
    extracted: Mapping[str, str],
    structure_extract_resnames: Set[str],
    unresolved_resnames: Set[str],
    target_meta: Dict[str, Dict[str, str]],
    target_smiles: Dict[str, str],
    generated_files: Dict[str, str],
) -> Optional[str]:
    """Use the actual mmCIF ligand resname when a single SMILES target is renamed."""

    unmatched_smiles = [
        resname
        for resname in structure_extract_resnames
        if resname in target_smiles and resname in unresolved_resnames
    ]
    if len(unmatched_smiles) != 1:
        return None

    direct_target_names = set(target_meta)
    extracted_candidates = [
        resname
        for resname, mol2_path in extracted.items()
        if resname not in direct_target_names and mol2_path
    ]
    if len(extracted_candidates) != 1:
        return None

    source_resname = unmatched_smiles[0]
    structure_resname = extracted_candidates[0]
    if structure_resname in target_meta:
        return None

    target_meta[structure_resname] = target_meta.pop(source_resname)
    target_smiles[structure_resname] = target_smiles.pop(source_resname)
    generated_files[structure_resname] = str(extracted[structure_resname])
    unresolved_resnames.remove(source_resname)
    structure_extract_resnames.remove(source_resname)
    logging.info(
        "Using structure-derived ligand residue name %s for %s (%s).",
        structure_resname,
        source_resname,
        target_meta[structure_resname]["owner"],
    )
    return structure_resname


def _normalize_nonstandard_aliases(raw_aliases: object) -> Dict[str, Dict[str, object]]:
    """Normalize global nonstandard alias config into upper-case alias map."""

    if raw_aliases is None:
        return {}
    if isinstance(raw_aliases, Mapping):
        items = list(raw_aliases.items())
    elif hasattr(raw_aliases, "items"):
        items = list(raw_aliases.items())
    else:
        raise TypeError(
            f"Unsupported nonstandard_aliases type: {type(raw_aliases)}"
        )

    normalized: Dict[str, Dict[str, object]] = {}
    for raw_alias, raw_entry in items:
        alias = str(raw_alias).strip().upper()
        if not alias:
            raise ValueError("nonstandard_aliases contains an empty alias key.")
        if len(alias) > 3:
            raise ValueError(
                f"nonstandard alias {alias!r} is too long (max 3 characters)."
            )
        if alias in normalized:
            raise ValueError(f"Duplicated nonstandard alias: {alias}")
        if not isinstance(raw_entry, Mapping) and not hasattr(raw_entry, "get"):
            raise TypeError(
                f"nonstandard alias {alias!r} config must be a mapping, got {type(raw_entry)}"
            )
        ccd_id = str(raw_entry.get("ccd_id", "")).strip().upper()  # type: ignore[arg-type]
        if not ccd_id:
            raise ValueError(
                f"nonstandard alias {alias!r} has empty ccd_id; cannot generate params."
            )
        entry = {
            "ccd_id": ccd_id,
            "parent_aa": str(raw_entry.get("parent_aa", "")).strip().upper(),  # type: ignore[arg-type]
        }
        smiles = str(raw_entry.get("smiles", "")).strip()  # type: ignore[arg-type]
        if smiles:
            entry["smiles"] = smiles
        atom_names = raw_entry.get("atom_names")  # type: ignore[attr-defined]
        if atom_names is not None:
            if isinstance(atom_names, (list, tuple)):
                entry["atom_names"] = [str(v).strip() for v in atom_names]
            elif hasattr(atom_names, "__iter__") and not isinstance(atom_names, (str, bytes)):
                entry["atom_names"] = [str(v).strip() for v in atom_names]
            elif hasattr(atom_names, "items"):
                entry["atom_names"] = {
                    str(k).strip(): str(v).strip() for k, v in atom_names.items()
                }
            else:
                raise TypeError(
                    f"nonstandard alias {alias!r} atom_names must be list/iterable or mapping, got {type(atom_names)}"
                )
        normalized[alias] = entry
    return normalized


def _require_ccd_code_resname(ccd_code: str) -> str:
    """Return params name for CCD-backed PTM; must match CCD code exactly."""

    ccd = str(ccd_code).strip().upper()
    if not ccd:
        raise ValueError("CCD code cannot be empty.")
    sanitized = _sanitize_resname(ccd, fallback=ccd)
    if sanitized != ccd:
        raise ValueError(
            f"CCD code {ccd!r} cannot be used as params name directly. "
            "PTM params names must equal CCD code and stay within Rosetta naming limits."
        )
    return ccd


def _register_params_target(
    *,
    targets: Dict[str, Dict[str, str]],
    resname: str,
    source_id: str,
    owner: str,
) -> bool:
    """Register a params target by residue name.

    Returns True if new target should be generated, False if same target already exists.
    """

    existing = targets.get(resname)
    if existing is None:
        targets[resname] = {"source_id": source_id, "owner": owner}
        return True

    if existing["source_id"] != source_id:
        raise ValueError(
            f"Params resname conflict for {resname!r}: "
            f"{existing['owner']} ({existing['source_id']}) vs {owner} ({source_id})."
        )
    logging.info(
        "Skipping duplicated params target for %s (owner=%s, same source=%s).",
        resname,
        owner,
        source_id,
    )
    return False


def _normalize_ccd_codes(ccd_sequence: object) -> List[str]:
    """Normalize CCD sequence string into CCD code list."""

    if not isinstance(ccd_sequence, str):
        return []
    codes = []
    for token in ccd_sequence.split(","):
        code = token.strip().upper()
        if code:
            codes.append(code)
    return codes


def _normalize_descriptor_field(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        return [raw] if raw else []
    if isinstance(value, Sequence):
        normalized = []
        for item in value:
            s = str(item).strip()
            if s:
                normalized.append(s)
        return normalized
    return [str(value).strip()]


def ccd_code_to_smiles(ccd: Mapping[str, object], code: str) -> str:
    """Get a SMILES string from AF3 CCD entry."""

    code_norm = str(code).strip().upper()
    if not code_norm:
        raise ValueError("CCD code cannot be empty")
    if code_norm not in ccd:
        raise KeyError(f"CCD code not found in AF3 chemical components: {code_norm}")

    entry = ccd[code_norm]
    if not isinstance(entry, Mapping):
        raise ValueError(f"Unexpected CCD entry format for {code_norm}: {type(entry)}")

    descriptor_types = _normalize_descriptor_field(
        entry.get("_pdbx_chem_comp_descriptor.type")
    )
    descriptors = _normalize_descriptor_field(
        entry.get("_pdbx_chem_comp_descriptor.descriptor")
    )

    if descriptor_types and len(descriptor_types) == len(descriptors):
        for typ, desc in zip(descriptor_types, descriptors, strict=True):
            if "SMILES_CANONICAL" in typ.upper() and desc not in {"?", ".", "None"}:
                return desc
        for typ, desc in zip(descriptor_types, descriptors, strict=True):
            if "SMILES" in typ.upper() and desc not in {"?", ".", "None"}:
                return desc

    for desc in descriptors:
        if desc not in {"?", ".", "None"}:
            return desc

    raise ValueError(f"No SMILES descriptor found for CCD code: {code_norm}")


def _infer_rosetta_residue_types_dir_from_molfile_to_params(
    molfile_to_params_path: object,
) -> Optional[Path]:
    """Infer Rosetta residue_types directory from molfile_to_params.py path."""

    if molfile_to_params_path is None:
        return None
    script_path = Path(str(molfile_to_params_path)).expanduser()
    if not script_path.exists():
        return None

    subpath = Path("database/chemical/residue_type_sets/fa_standard/residue_types")
    for parent in [script_path.parent, *script_path.parents]:
        candidate = parent / subpath
        if candidate.is_dir():
            return candidate
    return None


@lru_cache(maxsize=8)
def get_rosetta_builtin_residue_names(residue_types_dir: str) -> Set[str]:
    """Parse Rosetta built-in residue names from *.params files."""

    residue_dir = Path(residue_types_dir)
    if not residue_dir.is_dir():
        return set()

    params_paths: List[Path]
    try:
        files_proc = subprocess.run(
            ["rg", "--files", str(residue_dir), "-g", "*.params"],
            check=True,
            capture_output=True,
            text=True,
        )
        params_paths = [
            Path(line)
            for line in files_proc.stdout.splitlines()
            if str(line).strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        params_paths = list(residue_dir.rglob("*.params"))

    name_by_path: Dict[Path, str] = {}
    if params_paths:
        try:
            name_proc = subprocess.run(
                ["rg", "-n", "-H", "^NAME\\s+", *[str(path) for path in params_paths]],
                check=False,
                capture_output=True,
                text=True,
            )
            for raw_line in name_proc.stdout.splitlines():
                path_text, _lineno, content = raw_line.split(":", 2)
                parts = content.strip().split()
                if len(parts) >= 2:
                    name_by_path[Path(path_text)] = parts[1].upper()
        except (FileNotFoundError, ValueError):
            name_by_path = {}

    names: Set[str] = set()
    for params_path in params_paths:
        found_name = name_by_path.get(params_path)
        if found_name is None:
            try:
                with params_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        striped = line.strip()
                        if not striped or striped.startswith("#"):
                            continue
                        if striped.startswith("NAME "):
                            parts = striped.split()
                            if len(parts) >= 2:
                                found_name = parts[1].upper()
                            break
            except OSError:
                continue
        if not found_name:
            found_name = params_path.stem.upper()
        names.add(found_name)
    return names


def _get_runtime_rosetta_builtin_residue_names() -> Optional[Set[str]]:
    """Try to load Rosetta built-in residue names using runtime config paths."""

    try:
        molfile_path = getattr(global_config, "molfile_to_params", None)
    except Exception:
        molfile_path = None

    residue_types_dir = _infer_rosetta_residue_types_dir_from_molfile_to_params(
        molfile_path
    )
    if residue_types_dir is None:
        return None
    return get_rosetta_builtin_residue_names(str(residue_types_dir))


@lru_cache(maxsize=128)
def _query_pyrosetta_builtin_residue_names(codes: tuple[str, ...]) -> Set[str]:
    """Query PyRosetta for built-in residue names without initializing this process."""

    normalized = tuple(
        sorted({str(code).strip().upper() for code in codes if str(code).strip()})
    )
    if not normalized:
        return set()

    try:
        import pyrosetta

        if hasattr(pyrosetta, "is_initialized") and pyrosetta.is_initialized():
            residue_type_set = (
                pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
                .residue_type_set("fa_standard")
            )
            return {
                code
                for code in normalized
                if _pyrosetta_residue_type_set_has_name(residue_type_set, code)
            }
    except Exception:
        pass

    query_script = r"""
import contextlib
import json
import os
import sys

codes = json.loads(sys.stdin.read() or "[]")
import pyrosetta
with open(os.devnull, "w") as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        pyrosetta.init("-mute all")
residue_type_set = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")
found = []
for code in codes:
    try:
        if residue_type_set.has_name(code):
            found.append(code)
    except Exception:
        pass
print(json.dumps(found))
"""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", query_script],
            input=json.dumps(list(normalized)),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return set(json.loads(completed.stdout.strip() or "[]"))
    except Exception as exc:
        logging.info(
            "Could not query PyRosetta built-in residue set in a subprocess; "
            "falling back to params files only: %s",
            exc,
        )
        return set()


def _pyrosetta_residue_type_set_has_name(residue_type_set, residue_name: str) -> bool:
    try:
        return bool(residue_type_set.has_name(residue_name))
    except Exception:
        return False


def _ccd_codes_requiring_params_for_protomer(
    *,
    proto_name: str,
    protomer_name: str,
    ccd_codes: Sequence[str],
    ligand_resnames: Mapping[str, str],
    builtin_residue_names: Set[str],
) -> List[str]:
    """Return CCD codes that need generated params after ligand resname aliases."""

    normalized_codes: List[str] = []
    seen: Set[str] = set()
    for raw_code in ccd_codes:
        ccd_code = str(raw_code).strip().upper()
        if not ccd_code or ccd_code in seen:
            continue
        seen.add(ccd_code)
        normalized_codes.append(ccd_code)

    resolved_by_code = {
        ccd_code: _resolve_ligand_resname(
            protomer_key=proto_name,
            protomer_name=protomer_name,
            ligand_resnames=ligand_resnames,
            ccd_code=ccd_code,
        )
        for ccd_code in normalized_codes
    }
    unaliased_query_codes = tuple(
        ccd_code
        for ccd_code in normalized_codes
        if resolved_by_code[ccd_code] == ccd_code
        and ccd_code not in builtin_residue_names
    )
    pyrosetta_builtin_names = _query_pyrosetta_builtin_residue_names(
        unaliased_query_codes
    )

    required: List[str] = []
    for ccd_code in normalized_codes:
        resolved_resname = resolved_by_code[ccd_code]
        if resolved_resname != ccd_code:
            required.append(ccd_code)
            continue
        if (
            ccd_code not in builtin_residue_names
            and ccd_code not in pyrosetta_builtin_names
        ):
            required.append(ccd_code)
    return required


def find_nondefault_rosetta_ccd_codes(
    ccd_codes: Sequence[str],
    *,
    builtin_residue_names: Optional[Set[str]] = None,
    residue_types_dir: Optional[str] = None,
    builtin_residue_name_checker=None,
) -> List[str]:
    """Return CCD codes that are not built into the Rosetta residue set."""

    if builtin_residue_names is None and residue_types_dir is not None:
        builtin_residue_names = get_rosetta_builtin_residue_names(residue_types_dir)
    if builtin_residue_names is None:
        builtin_residue_names = _get_runtime_rosetta_builtin_residue_names() or set()

    normalized_codes: List[str] = []
    seen: Set[str] = set()
    for raw in ccd_codes:
        code = str(raw).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        normalized_codes.append(code)

    if builtin_residue_name_checker is None:
        pyrosetta_builtin_residue_names = _query_pyrosetta_builtin_residue_names(
            tuple(code for code in normalized_codes if code not in builtin_residue_names)
        )
    else:
        pyrosetta_builtin_residue_names = set()

    missing: List[str] = []
    for code in normalized_codes:
        if code in builtin_residue_names:
            continue
        if builtin_residue_name_checker is not None and builtin_residue_name_checker(code):
            continue
        if code in pyrosetta_builtin_residue_names:
            continue
        missing.append(code)
    return missing


def _normalize_smiles_input(smiles: object) -> List[str]:
    """Normalize SMILES input into a simple list of strings."""

    if isinstance(smiles, str):
        smiles = smiles.strip()
        return [smiles] if smiles else []

    if isinstance(smiles, Iterable):
        normalized = []
        for value in smiles:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    normalized.append(stripped)
        return normalized

    logging.warning(f"Unsupported SMILES input type: {type(smiles)}")
    return []


def _convert_smiles_to_mol2(smiles: str, mol2_path: Path, resname: str) -> None:
    """Use RDKit to convert a SMILES string into a mol2 file with 3D conformer.
    
    This function uses RDKit to:
    1. Parse the SMILES string
    2. Add hydrogens
    3. Generate a 3D conformer using ETKDG algorithm
    4. Optimize the geometry using MMFF94 force field
    5. Save as mol2 format (via OpenBabel conversion from RDKit mol)
    """
    
    if not RDKIT_AVAILABLE:
        raise ExtraResidueError("RDKit is not available. Cannot convert SMILES to mol2.")
    
    mol2_path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_sdf_path: Optional[Path] = None
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ExtraResidueError(f"Failed to parse SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer using ETKDG (Experimental-Torsion Knowledge Distance Geometry)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # For reproducibility
        if hasattr(params, "maxAttempts"):
            params.maxAttempts = 100
        elif hasattr(params, "maxIterations"):
            params.maxIterations = 100
        
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # If ETKDG fails, try with random coordinates
            logging.warning(f"ETKDG failed for {resname}, trying with random coordinates")
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                raise ExtraResidueError(
                    f"Failed to generate 3D conformer for SMILES: {smiles}"
                )
        
        # Optimize geometry using MMFF94 force field
        try:
            mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            if mmff_result == -1:
                logging.warning(f"MMFF optimization failed for {resname}, trying UFF")
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception as e:
            logging.warning(f"Force field optimization failed for {resname}: {e}")
            # Continue without optimization - we still have a valid 3D structure
        
        # Set molecule name/title
        mol.SetProp("_Name", resname)
        
        # Save to temporary SDF file first (RDKit native format)
        tmp_sdf_path = mol2_path.with_suffix(".temp.sdf")
        writer = Chem.SDWriter(str(tmp_sdf_path))
        writer.write(mol)
        writer.close()
        
        # Convert SDF to mol2 using OpenBabel (for proper mol2 format with atom types)
        cmd = [
            global_config.obabel_path or "obabel",
            str(tmp_sdf_path),
            "-O",
            str(mol2_path),
            "--title",
            resname,
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not mol2_path.exists() or mol2_path.stat().st_size <= 0:
            raise ExtraResidueError(f"mol2 file was not created or is empty: {mol2_path}")
        _validate_mol2_file(mol2_path, resname)

    except subprocess.CalledProcessError as e:
        raise ExtraResidueError(
            f"obabel command failed when converting SDF to mol2: {e.stderr}"
        ) from e
    except FileNotFoundError as e:
        raise ExtraResidueError(
            "obabel command not found. Please install OpenBabel."
        ) from e
    except Exception as e:
        raise ExtraResidueError(
            f"SMILES to mol2 conversion failed for resname={resname}: {e}"
        ) from e
    finally:
        if tmp_sdf_path and tmp_sdf_path.exists():
            try:
                tmp_sdf_path.unlink()
            except OSError as exc:
                logging.warning("Failed to remove temporary file %s: %s", tmp_sdf_path, exc)


def _convert_smiles_to_mol2_pure_rdkit(smiles: str, mol2_path: Path, resname: str) -> None:
    """Pure RDKit implementation to convert SMILES to mol2 format.
    
    This is a fallback if OpenBabel is not available.
    Note: RDKit's mol2 writer has limitations with atom types.
    """
    
    if not RDKIT_AVAILABLE:
        raise ExtraResidueError("RDKit is not available. Cannot convert SMILES to mol2.")
    
    mol2_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ExtraResidueError(f"Failed to parse SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if hasattr(params, "maxAttempts"):
            params.maxAttempts = 100
        elif hasattr(params, "maxIterations"):
            params.maxIterations = 100
        
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                raise ExtraResidueError(
                    f"Failed to generate 3D conformer for SMILES: {smiles}"
                )
        
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass  # Continue with unoptimized structure
        
        # Set molecule name
        mol.SetProp("_Name", resname)
        
        # Write mol2 file using RDKit
        Chem.MolToMolFile(mol, str(mol2_path.with_suffix(".mol")))
        
        # Try to use RDKit's mol2 writer if available
        try:
            from rdkit.Chem import MolToMol2File
            MolToMol2File(mol, str(mol2_path))
            if mol2_path.exists() and mol2_path.stat().st_size > 0:
                return
        except (ImportError, AttributeError):
            # MolToMol2File not available in this RDKit version
            # Fall back to manual mol2 generation
            pass
        
        # Manual mol2 file generation
        _write_mol2_from_rdkit_mol(mol, mol2_path, resname)
        
    except Exception as e:
        raise ExtraResidueError(
            f"Pure RDKit SMILES to mol2 conversion failed for resname={resname}: {e}"
        ) from e


def _normalize_ccd_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _normalize_element_symbol(symbol: str) -> str:
    token = str(symbol or "").strip()
    if not token:
        raise ExtraResidueError("CCD atom entry has empty type_symbol")
    if len(token) == 1:
        return token.upper()
    return token[0].upper() + token[1:].lower()


def _build_rdkit_mol_from_ccd_entry(ccd_entry: Mapping[str, object], resname: str):
    if not RDKIT_AVAILABLE:
        raise ExtraResidueError("RDKit is not available. Cannot convert CCD to mol2.")

    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit.Geometry import Point3D

    atom_ids = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.atom_id"))
    type_symbols = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.type_symbol"))
    charges = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.charge"))
    xs = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_x_ideal"))
    ys = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_y_ideal"))
    zs = _normalize_ccd_list(ccd_entry.get("_chem_comp_atom.pdbx_model_Cartn_z_ideal"))

    atom_count = len(atom_ids)
    if atom_count == 0:
        raise ExtraResidueError(f"CCD entry for {resname} is missing atom ids")
    expected_keys = {
        "type_symbol": len(type_symbols),
        "charge": len(charges),
        "x": len(xs),
        "y": len(ys),
        "z": len(zs),
    }
    bad = {k: v for k, v in expected_keys.items() if v != atom_count}
    if bad:
        raise ExtraResidueError(
            f"CCD entry for {resname} has inconsistent atom fields: {bad}, expected {atom_count}"
        )

    rw = Chem.RWMol()
    atom_index: Dict[str, int] = {}
    for atom_id, symbol, charge in zip(atom_ids, type_symbols, charges, strict=True):
        atom = Chem.Atom(_normalize_element_symbol(symbol))
        try:
            atom.SetFormalCharge(int(float(str(charge).strip() or "0")))
        except Exception:
            atom.SetFormalCharge(0)
        idx = int(rw.AddAtom(atom))
        rw.GetAtomWithIdx(idx).SetProp("atom_name", str(atom_id).strip())
        atom_index[str(atom_id).strip()] = idx

    bond_a1 = _normalize_ccd_list(ccd_entry.get("_chem_comp_bond.atom_id_1"))
    bond_a2 = _normalize_ccd_list(ccd_entry.get("_chem_comp_bond.atom_id_2"))
    bond_orders = _normalize_ccd_list(ccd_entry.get("_chem_comp_bond.value_order"))
    aromatic_flags = _normalize_ccd_list(ccd_entry.get("_chem_comp_bond.pdbx_aromatic_flag"))
    bond_count = len(bond_a1)
    if not (len(bond_a2) == len(bond_orders) == len(aromatic_flags) == bond_count):
        raise ExtraResidueError(
            f"CCD entry for {resname} has inconsistent bond fields"
        )

    order_map = {
        "SING": rdchem.BondType.SINGLE,
        "SINGLE": rdchem.BondType.SINGLE,
        "DOUB": rdchem.BondType.DOUBLE,
        "DOUBLE": rdchem.BondType.DOUBLE,
        "TRIP": rdchem.BondType.TRIPLE,
        "TRIPLE": rdchem.BondType.TRIPLE,
        "AROM": rdchem.BondType.AROMATIC,
        "AROMATIC": rdchem.BondType.AROMATIC,
    }
    for a1, a2, order, aromatic in zip(
        bond_a1, bond_a2, bond_orders, aromatic_flags, strict=True
    ):
        lhs = atom_index[str(a1).strip()]
        rhs = atom_index[str(a2).strip()]
        normalized_order = str(order).strip().upper()
        bond_type = order_map.get(normalized_order)
        if bond_type is None:
            raise ExtraResidueError(
                f"Unsupported CCD bond order {order!r} for {resname}"
            )
        rw.AddBond(lhs, rhs, bond_type)
        if bond_type == rdchem.BondType.AROMATIC or str(aromatic).strip().upper() == "Y":
            bond = rw.GetBondBetweenAtoms(lhs, rhs)
            if bond is not None:
                bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(lhs).SetIsAromatic(True)
            rw.GetAtomWithIdx(rhs).SetIsAromatic(True)

    mol = rw.GetMol()
    conf = Chem.Conformer(atom_count)
    for idx, (x, y, z) in enumerate(zip(xs, ys, zs, strict=True)):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    mol.SetProp("_Name", resname)
    return mol


def _write_mol2_from_rdkit_mol(mol, mol2_path: Path, resname: str) -> None:
    """Manually write mol2 file from RDKit mol object."""
    
    try:
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        lines = []
        
        # MOLECULE section
        lines.append("@<TRIPOS>MOLECULE")
        lines.append(resname)
        lines.append(f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0")
        lines.append("SMALL")
        lines.append("NO_CHARGES")
        lines.append("")
        
        # ATOM section
        lines.append("@<TRIPOS>ATOM")
        for i, atom in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            atom_name = (
                atom.GetProp("atom_name").strip() if atom.HasProp("atom_name") else symbol
            )
            # Simple atom type assignment
            atom_type = symbol
            if symbol == "C":
                if atom.GetIsAromatic():
                    atom_type = "C.ar"
                elif atom.GetHybridization() == Chem.HybridizationType.SP3:
                    atom_type = "C.3"
                elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                    atom_type = "C.2"
                else:
                    atom_type = "C.3"
            elif symbol == "N":
                if atom.GetIsAromatic():
                    atom_type = "N.ar"
                elif atom.GetHybridization() == Chem.HybridizationType.SP3:
                    atom_type = "N.3"
                elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                    atom_type = "N.2"
                else:
                    atom_type = "N.3"
            elif symbol == "O":
                if atom.GetHybridization() == Chem.HybridizationType.SP3:
                    atom_type = "O.3"
                else:
                    atom_type = "O.2"
            elif symbol == "S":
                atom_type = "S.3"
            elif symbol == "P":
                atom_type = "P.3"
            elif symbol == "H":
                atom_type = "H"
            
            lines.append(f"{i+1:>7} {atom_name:<4} {pos.x:>10.4f} {pos.y:>10.4f} {pos.z:>10.4f} {atom_type:<6} 1 {resname:<8} 0.0000")
        
        # BOND section
        lines.append("@<TRIPOS>BOND")
        bond_type_map = {
            Chem.BondType.SINGLE: "1",
            Chem.BondType.DOUBLE: "2",
            Chem.BondType.TRIPLE: "3",
            Chem.BondType.AROMATIC: "ar",
        }
        for i, bond in enumerate(bonds):
            bt = bond_type_map.get(bond.GetBondType(), "1")
            lines.append(f"{i+1:>6} {bond.GetBeginAtomIdx()+1:>6} {bond.GetEndAtomIdx()+1:>6} {bt}")
        
        # SUBSTRUCTURE section
        lines.append("@<TRIPOS>SUBSTRUCTURE")
        lines.append(f"1 {resname} 1 RESIDUE 1 A {resname} 0 ROOT")
        
        # Write to file
        with open(mol2_path, "w") as f:
            f.write("\n".join(lines))
        
        if not mol2_path.exists() or mol2_path.stat().st_size <= 0:
            raise ExtraResidueError(f"Failed to write non-empty mol2 file: {mol2_path}")
        
    except Exception as e:
        raise ExtraResidueError(f"Failed to write mol2 file {mol2_path}: {e}") from e


def generate_mol2_files_for_ccd(
    ccd_entry: Mapping[str, object], output_dir: str, resname_full: Optional[str] = None
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mol2_path = output_path / f"{resname_full}.mol2"
    mol = _build_rdkit_mol_from_ccd_entry(ccd_entry, str(resname_full))
    _write_mol2_from_rdkit_mol(mol, mol2_path, str(resname_full))
    _validate_mol2_file(mol2_path, str(resname_full))
    return {str(resname_full): str(mol2_path)}


def generate_mol2_files_for_protomer(
    smiles: object, output_dir: str, resname_full: Optional[str] = None
) -> Dict[str, str]:
    """
    从SMILES输入生成mol2文件。

    Args:
        smiles_input: SMILES字符串或SMILES字符串列表
        output_dir: 输出文件
        base_resname: 基础残基名称（可选）

    Returns:
        Dict mapping residue names to mol2 file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mol2_files = {}


    mol2_path = output_path / f"{resname_full}.mol2"

    try:
        _convert_smiles_to_mol2(smiles, mol2_path, resname_full)
    except ExtraResidueError as primary_exc:
        logging.info(
            "Primary SMILES->mol2 conversion failed for %s (%s); trying pure RDKit fallback.",
            resname_full,
            primary_exc,
        )
        try:
            _convert_smiles_to_mol2_pure_rdkit(smiles, mol2_path, resname_full)
        except ExtraResidueError as fallback_exc:
            raise RuntimeError(
                f"Failed to generate mol2 file for resname={resname_full} smiles={smiles!r}. "
                f"primary={primary_exc}; fallback={fallback_exc}"
            ) from fallback_exc

    mol2_files[resname_full] = str(mol2_path)
    logging.info(f"Generated mol2 file for {resname_full}: {mol2_path}")

    return mol2_files


def setup_extra_residues(
    ligand_resnames: Optional[Mapping[str, str]] = None,
    structure_files: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """
    从评估结果中提取非标准残基并准备PyRosetta参数文件。

    这个函数替换原来的setup_extra_res函数。
    """
    protomers = global_config.protomers
    nonstandard_aliases = _normalize_nonstandard_aliases(
        getattr(global_config, "nonstandard_aliases", None)
    )
    has_ligand_protomers = any(
        v.type in (ProtomerTypes.SMILES, ProtomerTypes.CCD) for _, v in protomers.items()
    )
    if not has_ligand_protomers and not nonstandard_aliases:
        logging.info("Skipping extra residue setup; no ligand or nonstandard PTM targets.")
        return {}

    merged_resname_map = get_merged_ligand_resnames(ligand_resnames)
    output_dir = global_config.out_dir / "raw_model_results"
    all_mol2_files: Dict[str, str] = {}

    builtin_residue_names = _get_runtime_rosetta_builtin_residue_names()
    if builtin_residue_names is None:
        builtin_residue_names = set()
        logging.info(
            "Could not detect Rosetta built-in residue set from molfile_to_params path; "
            "treating all CCD protomer codes as non-default for params generation."
        )

    ccd_obj = None

    def _ensure_ccd():
        nonlocal ccd_obj
        if ccd_obj is None:
            from alphafold3.constants import chemical_components

            try:
                user_ccd = getattr(global_config, "user_ccd")
            except Exception:
                user_ccd = None
            ccd_obj = chemical_components.Ccd(user_ccd=user_ccd)
        return ccd_obj

    # resname -> owner/source info to enforce cross-scope uniqueness.
    target_meta: Dict[str, Dict[str, str]] = {}
    # resname -> smiles payload used for mol2 generation.
    target_smiles: Dict[str, str] = {}
    # resname -> CCD code resolved lazily only if structure-derived extraction is unavailable.
    target_ccd_codes: Dict[str, str] = {}
    # resname -> nonstandard alias metadata that must be materialized as polymer params.
    target_nonstandard_aliases: Dict[str, Dict[str, object]] = {}
    alias_smiles_by_ccd: Dict[str, str] = {}
    if nonstandard_aliases:
        for alias_entry in nonstandard_aliases.values():
            ccd_code = str(alias_entry.get("ccd_id", "")).strip().upper()
            alias_smiles = str(alias_entry.get("smiles", "")).strip()
            if not ccd_code or not alias_smiles:
                continue
            existing = alias_smiles_by_ccd.get(ccd_code)
            if existing and existing != alias_smiles:
                raise ValueError(
                    f"Conflicting nonstandard alias smiles for CCD code {ccd_code!r}."
                )
            alias_smiles_by_ccd[ccd_code] = alias_smiles

    for proto_name, protomer in protomers.items():
        if protomer.type == ProtomerTypes.SMILES:
            resname = _resolve_ligand_resname(
                protomer_key=proto_name,
                protomer_name=str(getattr(protomer, "name", proto_name)),
                ligand_resnames=merged_resname_map,
            )
            source_id = f"SMILES:{str(protomer.sequence).strip()}"
            owner = f"protomer:{proto_name}"
            if _register_params_target(
                targets=target_meta,
                resname=resname,
                source_id=source_id,
                owner=owner,
            ):
                target_smiles[resname] = str(protomer.sequence)
            continue

        if protomer.type != ProtomerTypes.CCD:
            continue

        ccd_codes = _normalize_ccd_codes(protomer.sequence)
        protomer_name = str(getattr(protomer, "name", proto_name))
        params_codes = _ccd_codes_requiring_params_for_protomer(
            proto_name=proto_name,
            protomer_name=protomer_name,
            ccd_codes=ccd_codes,
            ligand_resnames=merged_resname_map,
            builtin_residue_names=builtin_residue_names,
        )
        if not params_codes:
            logging.info(
                "Skipping params generation for protomer %s: all CCD codes are built into Rosetta (%s).",
                proto_name,
                ",".join(ccd_codes),
            )
            continue

        for ccd_code in params_codes:
            resname = _resolve_ligand_resname(
                protomer_key=proto_name,
                protomer_name=protomer_name,
                ligand_resnames=merged_resname_map,
                ccd_code=ccd_code,
            )
            owner = f"protomer:{proto_name}:ccd:{ccd_code}"
            source_id = f"CCD:{ccd_code}"
            if _register_params_target(
                targets=target_meta,
                resname=resname,
                source_id=source_id,
                owner=owner,
            ):
                alias_smiles = alias_smiles_by_ccd.get(ccd_code, "")
                if alias_smiles:
                    target_smiles[resname] = alias_smiles
                else:
                    target_ccd_codes[resname] = ccd_code

    if nonstandard_aliases:
        for alias, alias_entry in sorted(nonstandard_aliases.items()):
            ccd_code = str(alias_entry["ccd_id"]).strip().upper()
            if ccd_code in builtin_residue_names:
                logging.info(
                    "Skipping params generation for nonstandard alias %s: CCD code %s is built into Rosetta.",
                    alias,
                    ccd_code,
                )
                continue
            resname = _require_ccd_code_resname(ccd_code)
            owner = f"nonstandard_alias:{alias}"
            source_id = f"CCD:{ccd_code}"
            if not _register_params_target(
                targets=target_meta,
                resname=resname,
                source_id=source_id,
                owner=owner,
            ):
                continue

            alias_kind = str(alias_entry.get("kind", "ptm")).strip().lower()
            if alias_kind == "ptm" or alias_entry.get("parent_aa"):
                target_nonstandard_aliases[resname] = dict(alias_entry)
                continue

            alias_smiles = str(alias_entry.get("smiles", "")).strip()
            if alias_smiles:
                target_smiles[resname] = alias_smiles
            else:
                target_ccd_codes[resname] = ccd_code

    generated_files: Dict[str, str] = {}
    unresolved_resnames: Set[str] = set(target_meta)

    if target_nonstandard_aliases:
        ccd_data = _ensure_ccd()
        for resname, alias_entry in sorted(target_nonstandard_aliases.items()):
            ccd_code = str(alias_entry["ccd_id"]).strip().upper()
            parent_aa = str(alias_entry["parent_aa"]).strip().upper()
            ccd_entry = ccd_data[ccd_code]
            params_path = generate_rosetta_polymer_params_for_ccd(
                ccd_entry=ccd_entry,
                resname=resname,
                parent_aa=parent_aa,
                output_dir=str(output_dir),
                properties=_infer_nonstandard_alias_rosetta_properties(
                    alias_entry,
                    ccd_entry,
                ),
            )
            generated_files[resname] = str(params_path)
            unresolved_resnames.discard(resname)

    structure_extract_resnames: Set[str] = {
        resname for resname in unresolved_resnames if resname not in target_ccd_codes
    }

    if structure_files and structure_extract_resnames:
        for _oligo_name, structure_path in structure_files.items():
            extracted = extract_hetero_residues_to_mol2(
                str(structure_path),
                str(output_dir),
            )
            if not extracted:
                continue
            for resname in list(structure_extract_resnames):
                mol2_path = extracted.get(resname)
                if mol2_path:
                    generated_files[resname] = str(mol2_path)
                    unresolved_resnames.remove(resname)
                    structure_extract_resnames.remove(resname)
            _remap_single_smiles_target_to_structure_resname(
                extracted=extracted,
                structure_extract_resnames=structure_extract_resnames,
                unresolved_resnames=unresolved_resnames,
                target_meta=target_meta,
                target_smiles=target_smiles,
                generated_files=generated_files,
            )
            if not structure_extract_resnames:
                break

    for resname in list(unresolved_resnames):
        if resname in target_smiles:
            smiles = target_smiles[resname]
            mol2_files = generate_mol2_files_for_protomer(
                smiles,
                str(output_dir),
                resname,
            )
        elif resname in target_ccd_codes:
            ccd_data = _ensure_ccd()
            mol2_files = generate_mol2_files_for_ccd(
                ccd_data[target_ccd_codes[resname]],
                str(output_dir),
                resname,
            )
        else:
            raise RuntimeError(
                f"Missing ligand source for target {resname} ({target_meta[resname]['owner']})."
            )
        if resname not in mol2_files:
            raise RuntimeError(
                f"Failed to generate mol2 for target {resname} ({target_meta[resname]['owner']})."
            )
        generated_files.update(mol2_files)

    ligand_like_inputs = {
        resname: path
        for resname, path in generated_files.items()
        if str(path).endswith('.mol2')
    }
    if ligand_like_inputs:
        try:
            generate_pyrosetta_params_commands(
                ligand_like_inputs,
                str(output_dir),
            )
        except RuntimeError:
            fallback_inputs: Dict[str, str] = {}
            fallback_resnames = sorted(
                resname
                for resname in ligand_like_inputs
                if resname in target_smiles
            )
            for resname in fallback_resnames:
                if resname not in ligand_like_inputs or resname not in target_smiles:
                    continue
                logging.warning(
                    "Mol2 params generation failed for SMILES ligand %s; "
                    "regenerating mol2 from the source SMILES with the resolved residue name.",
                    resname,
                )
                fallback_mol2 = generate_mol2_files_for_protomer(
                    target_smiles[resname],
                    str(output_dir),
                    resname,
                )
                if resname not in fallback_mol2:
                    raise RuntimeError(
                        f"Failed to generate fallback mol2 for SMILES ligand {resname}."
                    )
                fallback_inputs[resname] = fallback_mol2[resname]
            if not fallback_inputs:
                raise
            ligand_like_inputs.update(fallback_inputs)
            generated_files.update(fallback_inputs)
            generate_pyrosetta_params_commands(
                ligand_like_inputs,
                str(output_dir),
            )

    return generated_files
