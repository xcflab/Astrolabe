from __future__ import annotations

import copy
import gzip
import io
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Structure import Structure as BioStructure
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.folding_input import DnaChain, Input, Ligand, ProteinChain, RnaChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.rf3 import RF3RunnerRequest, RF3RunnerResponse


AMINO_ACID_1_TO_3 = {
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
RF3_SMILES_RESNAME = "UNL"
PROTEIN_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _get_current_step(runtime_config: dict[str, Any] | None = None) -> int:
    if runtime_config and hasattr(runtime_config, "get"):
        value = runtime_config.get("step")
        if value is not None:
            return int(value)
    try:
        return int(getattr(global_config, "step", 0) or 0)
    except Exception:
        return 0


def _default_request_seed(input_obj: Input, runtime_config: dict[str, Any] | None = None) -> int | None:
    if runtime_config and hasattr(runtime_config, "get"):
        value = runtime_config.get("seed")
        if value is not None:
            return int(value)
    rng_seeds = list(getattr(input_obj, "rng_seeds", []) or [])
    if rng_seeds:
        return int(rng_seeds[0])
    return None


def _materialize_msa(
    *,
    chain_id: str,
    jobname: str,
    unpaired_msa: str | None,
    paired_msa: str | None,
) -> str | None:
    msa_text = str(unpaired_msa or "").strip() or str(paired_msa or "").strip()
    if not msa_text:
        return None

    msa_dir = Path(global_config.out_dir) / "sequences" / "rf3" / jobname
    msa_dir.mkdir(parents=True, exist_ok=True)
    msa_path = msa_dir / f"{chain_id}.a3m"
    msa_path.write_text(msa_text + ("\n" if not msa_text.endswith("\n") else ""), encoding="utf-8")
    return str(msa_path)


def _ligand_residue_name(chain: Ligand) -> str:
    if chain.ccd_ids:
        if len(chain.ccd_ids) != 1:
            raise ValueError(
                f"RF3 evaluator supports one CCD code per ligand chain, got {list(chain.ccd_ids)!r} for {chain.id!r}."
            )
        return str(chain.ccd_ids[0])
    if chain.smiles:
        return RF3_SMILES_RESNAME
    raise ValueError(f"Ligand chain {chain.id!r} requires either `smiles` or `ccd_ids`.")


def _rf3_ligand_residue_name(
    chain: Ligand,
    *,
    fixed_ligands: dict[str, Any] | None = None,
) -> str:
    ligand_cfg = _fixed_ligand_cfg_for_chain(chain, fixed_ligands=fixed_ligands or {})
    if ligand_cfg:
        ccd_id = str(ligand_cfg.get("ccd_id") or "").strip()
        if ccd_id:
            return ccd_id
    return _ligand_residue_name(chain)


def _rf3_output_ligand_residue_name(
    chain: Ligand,
    *,
    fixed_ligands: dict[str, Any] | None = None,
) -> str:
    ligand_cfg = _fixed_ligand_cfg_for_chain(chain, fixed_ligands=fixed_ligands or {})
    if ligand_cfg:
        output_resname = str(ligand_cfg.get("output_resname") or "").strip()
        if output_resname:
            return output_resname
    return _rf3_ligand_residue_name(chain, fixed_ligands=fixed_ligands)


def _normalize_rf3_conf_container(rf3_conf: Any) -> dict[str, Any]:
    if rf3_conf is None:
        return {}
    if isinstance(rf3_conf, dict):
        container = dict(rf3_conf)
    else:
        container = OmegaConf.to_container(rf3_conf, resolve=True)
    if not isinstance(container, dict):
        return {}
    return dict(container)


def _structure_parser(structure_path: Path):
    name = structure_path.name.lower()
    return MMCIFParser(QUIET=True) if any(ext in name for ext in (".cif", ".mmcif")) else PDBParser(QUIET=True)


def _format_pdb_atom_line(
    *,
    serial: int,
    record_name: str,
    atom_name: str,
    altloc: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
    element: str,
) -> str:
    return (
        f"{record_name:<6}{serial:5d} {atom_name[:4]:<4}{altloc[:1]:1}"
        f"{resname[:3]:>3} {chain_id[:1]:1}{resseq:4d}{icode[:1]:1}   "
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
        f"{float(occupancy):6.2f}{float(bfactor):6.2f}          "
        f"{element[:2]:>2}"
    )


def _protein_residues_from_template_mmcif(mmcif_text: str):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("rf3_protein_template", io.StringIO(mmcif_text))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError("Protein template mmCIF does not contain any models.")
    chain = next(model.get_chains(), None)
    if chain is None:
        raise ValueError("Protein template mmCIF does not contain any chains.")
    residues = []
    for residue in chain.get_residues():
        hetflag, _resseq, _icode = residue.get_id()
        if str(hetflag).strip():
            continue
        residues.append(residue)
    if not residues:
        raise ValueError("Protein template mmCIF does not contain protein residues.")
    return residues


def _select_protein_template(chain: ProteinChain):
    templates = list(getattr(chain, "templates", None) or [])
    for template in templates:
        mmcif_text = getattr(template, "mmcif", None)
        if isinstance(mmcif_text, str) and mmcif_text.strip():
            return template
    return None


def _materialize_protein_template_component(
    *,
    chain: ProteinChain,
    jobname: str,
    output_dir: Path,
    msa_path: str | None,
) -> dict[str, Any] | None:
    template = _select_protein_template(chain)
    if template is None:
        return None

    sequence = str(chain.sequence).strip().upper()
    if not sequence:
        raise ValueError(f"Protein chain {chain.id!r} has an empty sequence.")
    residues = _protein_residues_from_template_mmcif(str(template.mmcif))
    if len(residues) != len(sequence):
        raise ValueError(
            f"RF3 protein template for chain {chain.id!r} has {len(residues)} residues, "
            f"but the designed sequence has {len(sequence)} residues. "
            "RF3 structure-start currently requires a full-chain template."
        )

    template_path = output_dir / f"{jobname}__{chain.id}_protein_template.pdb"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for resseq, (residue, target_aa) in enumerate(zip(residues, sequence, strict=True), start=1):
        target_resname = AMINO_ACID_1_TO_3.get(target_aa)
        if target_resname is None:
            raise ValueError(
                f"Cannot materialize RF3 protein template for chain {chain.id!r}: "
                f"unsupported residue {target_aa!r} at position {resseq}."
            )
        source_resname = str(residue.get_resname()).strip().upper()
        keep_atoms = None
        if source_resname != target_resname:
            keep_atoms = set(PROTEIN_BACKBONE_ATOMS)
            if target_resname != "GLY":
                keep_atoms.add("CB")
        for atom in residue.get_atoms():
            atom_name = str(atom.get_name()).strip()
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H" or atom_name.upper().startswith("H"):
                continue
            if keep_atoms is not None and atom_name not in keep_atoms:
                continue
            coord = atom.get_coord()
            lines.append(
                _format_pdb_atom_line(
                    serial=serial,
                    record_name="ATOM",
                    atom_name=atom_name,
                    altloc=str(atom.get_altloc() or " ").replace(".", " "),
                    resname=target_resname,
                    chain_id=str(chain.id),
                    resseq=resseq,
                    icode="",
                    x=float(coord[0]),
                    y=float(coord[1]),
                    z=float(coord[2]),
                    occupancy=float(atom.get_occupancy() or 1.0),
                    bfactor=float(atom.get_bfactor() or 0.0),
                    element=element or atom_name[:1],
                )
            )
            serial += 1
    lines.append(f"TER   {serial:5d}")
    lines.append("END")
    template_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    component: dict[str, Any] = {"path": str(template_path)}
    if msa_path:
        component["msa_paths"] = {str(chain.id): msa_path}
    return component


def _extract_ligand_residue_to_cif(
    *,
    structure_path: Path,
    residue_chain_id: str,
    residue_resname: str,
    residue_resseq: int,
    residue_icode: str,
    output_chain_id: str,
    output_resname: str,
    output_resseq: int,
    output_icode: str,
    output_path: Path,
) -> Path:
    parser = _structure_parser(structure_path)
    handle = None
    try:
        if structure_path.name.lower().endswith(".gz"):
            handle = gzip.open(structure_path, "rt", encoding="utf-8", errors="replace")
            structure = parser.get_structure("rf3_ligand", handle)
        else:
            structure = parser.get_structure("rf3_ligand", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in ligand source structure: {structure_path}")

    match_residue = None
    for chain in model.get_chains():
        if str(chain.id).strip() != residue_chain_id:
            continue
        for residue in chain.get_residues():
            hetflag, resseq, icode = residue.get_id()
            if int(resseq) != int(residue_resseq):
                continue
            if str(icode or "").strip() != str(residue_icode or "").strip():
                continue
            if str(residue.get_resname()).strip() != residue_resname:
                continue
            match_residue = copy.deepcopy(residue)
            break
        if match_residue is not None:
            break
    if match_residue is None:
        raise ValueError(
            f"Ligand residue {residue_chain_id}:{residue_resname}:{residue_resseq}"
            f"{':' + residue_icode if residue_icode else ''} not found in {structure_path}"
        )

    return _write_residue_to_cif(
        residue=match_residue,
        output_chain_id=output_chain_id,
        output_resname=output_resname,
        output_resseq=output_resseq,
        output_icode=output_icode,
        output_path=output_path,
    )


def _write_residue_to_cif(
    *,
    residue,
    output_chain_id: str,
    output_resname: str,
    output_resseq: int,
    output_icode: str,
    output_path: Path,
) -> Path:
    out_struct = BioStructure("ligand")
    out_model = BioModel(0)
    out_chain = BioChain(output_chain_id)
    out_chain.add(copy.deepcopy(residue))
    out_model.add(out_chain)
    out_struct.add(out_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io_out = MMCIFIO()
    io_out.set_structure(out_struct)
    with output_path.open("w", encoding="utf-8") as handle_out:
        io_out.save(handle_out)
    _rewrite_atom_site_identity(
        output_path,
        chain_id=output_chain_id,
        resname=output_resname,
        resseq=output_resseq,
        icode=output_icode,
    )
    return output_path


def _matching_residue(
    *,
    structure_path: Path,
    residue_chain_id: str,
    residue_resname: str,
    residue_resseq: int,
    residue_icode: str,
):
    parser = _structure_parser(structure_path)
    handle = None
    try:
        if structure_path.name.lower().endswith(".gz"):
            handle = gzip.open(structure_path, "rt", encoding="utf-8", errors="replace")
            structure = parser.get_structure("rf3_ligand", handle)
        else:
            structure = parser.get_structure("rf3_ligand", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in ligand source structure: {structure_path}")

    for chain in model.get_chains():
        if str(chain.id).strip() != residue_chain_id:
            continue
        for residue in chain.get_residues():
            _hetflag, resseq, icode = residue.get_id()
            if int(resseq) != int(residue_resseq):
                continue
            if str(icode or "").strip() != str(residue_icode or "").strip():
                continue
            if str(residue.get_resname()).strip() != residue_resname:
                continue
            return residue
    return None


def _count_heavy_atoms_in_residue(residue) -> int:
    return sum(1 for atom in residue.get_atoms() if str(atom.element or "").strip().upper() != "H")


def _heavy_atom_names_in_residue(residue) -> list[str]:
    return [
        str(atom.get_name()).strip()
        for atom in residue.get_atoms()
        if str(atom.element or "").strip().upper() != "H"
    ]


def _extract_single_matching_residue(
    *,
    structure_path: Path,
    residue_resname: str,
):
    parser = _structure_parser(structure_path)
    handle = None
    try:
        if structure_path.name.lower().endswith(".gz"):
            handle = gzip.open(structure_path, "rt", encoding="utf-8", errors="replace")
            structure = parser.get_structure("rf3_ligand", handle)
        else:
            structure = parser.get_structure("rf3_ligand", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in ligand source structure: {structure_path}")

    matches = []
    for chain in model.get_chains():
        for residue in chain.get_residues():
            if str(residue.get_resname()).strip() != residue_resname:
                continue
            matches.append(residue)
    if not matches:
        raise ValueError(f"Residue {residue_resname!r} not found in {structure_path}")
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {residue_resname!r} residue in {structure_path}, found {len(matches)} matches."
        )
    return matches[0]


def _generated_standalone_conformer_path(ligand_name: str) -> Path | None:
    out_dir = getattr(global_config, "out_dir", None)
    if out_dir in (None, ""):
        return None
    raw_model_results_dir = Path(out_dir).expanduser() / "raw_model_results"
    candidates = [
        raw_model_results_dir / f"{ligand_name}_0001.pdb",
        raw_model_results_dir / f"{ligand_name}_0001.cif",
        raw_model_results_dir / f"{ligand_name}_0001.mmcif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _chemical_component_atom_names_from_cif(
    structure_path: Path,
    *,
    component_id: str,
) -> list[str] | None:
    suffixes = [suffix.lower() for suffix in structure_path.suffixes]
    if ".cif" not in suffixes and ".mmcif" not in suffixes:
        return None

    cif = MMCIF2Dict(str(structure_path))
    chem_comp_id = cif.get("_chem_comp.id")
    if isinstance(chem_comp_id, (list, tuple)):
        chem_comp_id = chem_comp_id[0] if chem_comp_id else ""
    chem_comp_id = str(chem_comp_id or "").strip()
    if chem_comp_id and chem_comp_id != component_id:
        raise ValueError(
            f"RF3 ligand component override {structure_path} has _chem_comp.id={chem_comp_id!r}, "
            f"but fixed_ligands expects {component_id!r}."
        )
    atom_ids = cif.get("_chem_comp_atom.atom_id")
    if atom_ids is None:
        return None
    if isinstance(atom_ids, str):
        atom_ids = [atom_ids]
    else:
        atom_ids = list(atom_ids)

    comp_ids = cif.get("_chem_comp_atom.comp_id")
    if isinstance(comp_ids, str):
        comp_ids = [comp_ids]
    elif comp_ids is not None:
        comp_ids = list(comp_ids)
    else:
        comp_ids = [component_id] * len(atom_ids)

    type_symbols = cif.get("_chem_comp_atom.type_symbol")
    if isinstance(type_symbols, str):
        type_symbols = [type_symbols]
    elif type_symbols is not None:
        type_symbols = list(type_symbols)
    else:
        type_symbols = [""] * len(atom_ids)

    names: list[str] = []
    for idx, atom_id in enumerate(atom_ids):
        comp_id = str(comp_ids[idx] if idx < len(comp_ids) else component_id).strip()
        if comp_id and comp_id != component_id:
            continue
        atom_name = str(atom_id).strip()
        if not atom_name:
            continue
        type_symbol = str(type_symbols[idx] if idx < len(type_symbols) else "").strip().upper()
        if type_symbol == "H" or atom_name.upper().startswith("H"):
            continue
        names.append(atom_name)
    if not names:
        raise ValueError(
            f"RF3 ligand component override {structure_path} contains no heavy atoms for {component_id!r}."
        )
    return names


def _allowed_atom_names_for_fixed_ligand(
    *,
    chain: Ligand,
    fixed_ligands: dict[str, Any],
) -> list[str] | None:
    ligand_name = _rf3_ligand_residue_name(chain, fixed_ligands=fixed_ligands)
    ligand_cfg = _fixed_ligand_cfg_for_chain(chain, fixed_ligands=fixed_ligands)
    if not ligand_cfg:
        return None

    residue_cfg = dict(ligand_cfg.get("residue") or {})
    residue_chain_id = str(residue_cfg.get("chain_id") or chain.id).strip()
    residue_resname = str(residue_cfg.get("resname") or ligand_name).strip()
    residue_resseq = int(residue_cfg.get("resseq") or 1)
    residue_icode = str(residue_cfg.get("icode") or "").strip()
    source = str(ligand_cfg.get("source") or "").strip()

    if source == "structure_residue":
        structure_path = Path(str(ligand_cfg.get("structure_path") or "")).expanduser().resolve()
        residue = _matching_residue(
            structure_path=structure_path,
            residue_chain_id=residue_chain_id,
            residue_resname=residue_resname,
            residue_resseq=residue_resseq,
            residue_icode=residue_icode,
        )
        if residue is None:
            raise ValueError(
                f"Ligand residue {residue_chain_id}:{residue_resname}:{residue_resseq}"
                f"{':' + residue_icode if residue_icode else ''} not found in {structure_path}"
            )
        return _heavy_atom_names_in_residue(residue)

    if source == "override":
        override_path = Path(str(ligand_cfg.get("path") or "")).expanduser().resolve()
        atom_names = _chemical_component_atom_names_from_cif(
            override_path,
            component_id=ligand_name,
        )
        if atom_names is not None:
            return atom_names
        residue = _extract_single_matching_residue(
            structure_path=override_path,
            residue_resname=ligand_name,
        )
        return _heavy_atom_names_in_residue(residue)

    return None


def _trim_structure_residue_to_atom_names(
    *,
    structure_path: Path,
    chain_id: str,
    resname: str,
    allowed_atom_names: list[str],
) -> None:
    allowed = {str(name).strip() for name in allowed_atom_names}
    parser = _structure_parser(structure_path)
    handle = None
    try:
        if structure_path.name.lower().endswith(".gz"):
            handle = gzip.open(structure_path, "rt", encoding="utf-8", errors="replace")
            structure = parser.get_structure("rf3_model", handle)
        else:
            structure = parser.get_structure("rf3_model", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    matched_residue = False
    for model in structure:
        for chain in model:
            if str(chain.id).strip() != chain_id:
                continue
            for residue in chain:
                if str(residue.get_resname()).strip() != resname:
                    continue
                matched_residue = True
                for atom in list(residue.get_atoms()):
                    atom_name = str(atom.get_name()).strip()
                    if atom_name not in allowed:
                        residue.detach_child(atom.id)
    if not matched_residue:
        raise ValueError(f"Residue {chain_id}/{resname} not found in predicted structure {structure_path}")

    if structure_path.suffix.lower() == ".cif":
        io_writer = MMCIFIO()
    else:
        io_writer = PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(str(structure_path))


def _rename_structure_residue(
    *,
    structure_path: Path,
    chain_id: str,
    source_resname: str,
    target_resname: str,
) -> None:
    if source_resname == target_resname:
        return
    parser = _structure_parser(structure_path)
    handle = None
    try:
        if structure_path.name.lower().endswith(".gz"):
            handle = gzip.open(structure_path, "rt", encoding="utf-8", errors="replace")
            structure = parser.get_structure("rf3_model", handle)
        else:
            structure = parser.get_structure("rf3_model", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    matched_residue = False
    for model in structure:
        for chain in model:
            if str(chain.id).strip() != chain_id:
                continue
            for residue in chain:
                if str(residue.get_resname()).strip() != source_resname:
                    continue
                matched_residue = True
                residue.resname = target_resname
    if not matched_residue:
        raise ValueError(
            f"Residue {chain_id}/{source_resname} not found in predicted structure {structure_path}"
        )

    io_writer = MMCIFIO() if structure_path.suffix.lower() == ".cif" else PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(str(structure_path))


def _rewrite_atom_site_identity(
    cif_path: Path,
    *,
    chain_id: str,
    resname: str,
    resseq: int,
    icode: str,
) -> None:
    lines = cif_path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    in_atom_site_loop = False
    data_started = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = False
            data_started = False
            header_lines = []
            continue
        if stripped.startswith("_atom_site."):
            in_atom_site_loop = True
            header_lines.append(stripped)
            continue
        if not in_atom_site_loop:
            continue
        if stripped.startswith(("ATOM", "HETATM")):
            data_started = True
            tokens = line.split()
            index_by_header = {
                header: header_lines.index(header)
                for header in header_lines
                if header in header_lines
            }
            replacements = {
                "_atom_site.label_comp_id": resname,
                "_atom_site.label_asym_id": chain_id,
                "_atom_site.label_seq_id": str(int(resseq)),
                "_atom_site.pdbx_PDB_ins_code": str(icode or "."),
                "_atom_site.auth_seq_id": str(int(resseq)),
                "_atom_site.auth_asym_id": chain_id,
            }
            for header, value in replacements.items():
                token_idx = index_by_header.get(header)
                if token_idx is not None and token_idx < len(tokens):
                    tokens[token_idx] = value
            lines[idx] = " ".join(tokens)
        elif data_started:
            break
    cif_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixed_ligand_cfg_for_chain(
    chain: Ligand,
    *,
    fixed_ligands: dict[str, Any],
) -> dict[str, Any] | None:
    cfg = fixed_ligands.get(str(chain.id))
    if not isinstance(cfg, dict):
        return None
    return dict(cfg)


def _as_selection_list(value: Any) -> list[str] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value) if str(item)]


def _dedupe_selections(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _ligand_component_and_selection(
    chain: Ligand,
    *,
    jobname: str,
    output_dir: Path,
    fixed_ligands: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    fixed_ligands = fixed_ligands or {}
    ligand_cfg = _fixed_ligand_cfg_for_chain(chain, fixed_ligands=fixed_ligands)
    ligand_name = _rf3_ligand_residue_name(chain, fixed_ligands=fixed_ligands)
    if not ligand_cfg:
        return {"ccd_code": ligand_name, "chain_id": str(chain.id)}, f"{chain.id}/{ligand_name}/1/*"

    residue_cfg = dict(ligand_cfg.get("residue") or {})
    residue_chain_id = str(residue_cfg.get("chain_id") or chain.id).strip()
    residue_resname = str(residue_cfg.get("resname") or ligand_name).strip()
    residue_resseq = int(residue_cfg.get("resseq") or 1)
    residue_icode = str(residue_cfg.get("icode") or "").strip()
    selection = f"{chain.id}/{ligand_name}/1/*"

    source = str(ligand_cfg.get("source") or "").strip()
    if source == "structure_residue":
        conformer_path = output_dir / f"{jobname}__{chain.id}_conformer.cif"
        structure_path = Path(str(ligand_cfg.get("structure_path") or "")).expanduser().resolve()
        _extract_ligand_residue_to_cif(
            structure_path=structure_path,
            residue_chain_id=residue_chain_id,
            residue_resname=residue_resname,
            residue_resseq=residue_resseq,
            residue_icode=residue_icode,
            output_chain_id=str(chain.id),
            output_resname=ligand_name,
            output_resseq=1,
            output_icode="",
            output_path=conformer_path,
        )
        return {"path": str(conformer_path)}, selection
    if source == "override":
        override_path = Path(str(ligand_cfg.get("path") or "")).expanduser().resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"RF3 ligand conformer override not found: {override_path}")
        return {"path": str(override_path)}, selection
    return {"ccd_code": ligand_name, "chain_id": str(chain.id)}, selection


def _component_from_chain(chain: Any, *, jobname: str, output_dir: Path) -> dict[str, Any]:
    if isinstance(chain, ProteinChain):
        if getattr(chain, "ptms", None):
            raise ValueError("RF3 evaluator does not yet support protein PTMs in prepared inputs.")
        msa_path = _materialize_msa(
            chain_id=str(chain.id),
            jobname=jobname,
            unpaired_msa=getattr(chain, "unpaired_msa", None),
            paired_msa=getattr(chain, "paired_msa", None),
        )
        template_component = _materialize_protein_template_component(
            chain=chain,
            jobname=jobname,
            output_dir=output_dir,
            msa_path=msa_path,
        )
        if template_component is not None:
            return template_component
        component = {
            "seq": str(chain.sequence),
            "chain_id": str(chain.id),
        }
        if msa_path:
            component["msa_path"] = msa_path
        return component
    if isinstance(chain, RnaChain):
        return {"seq": str(chain.sequence), "chain_id": str(chain.id), "chain_type": "rna"}
    if isinstance(chain, DnaChain):
        return {"seq": str(chain.sequence), "chain_id": str(chain.id), "chain_type": "dna"}
    if isinstance(chain, Ligand):
        component, _selection = _ligand_component_and_selection(
            chain,
            jobname=jobname,
            output_dir=output_dir,
        )
        return component
    raise TypeError(f"Unsupported RF3 chain type: {type(chain).__name__}")


def _resolve_bond_residue_name(
    chain: Any,
    residue_index: int,
    *,
    fixed_ligands: dict[str, Any] | None = None,
) -> str:
    if isinstance(chain, ProteinChain):
        sequence = str(chain.sequence)
        if residue_index < 1 or residue_index > len(sequence):
            raise ValueError(
                f"Residue index {residue_index} is out of range for chain {chain.id!r} with length {len(sequence)}."
            )
        residue_code = sequence[residue_index - 1].upper()
        if residue_code not in AMINO_ACID_1_TO_3:
            raise ValueError(
                f"Cannot derive RF3 bond residue name for chain {chain.id!r} residue {residue_index}: {residue_code!r}."
            )
        return AMINO_ACID_1_TO_3[residue_code]
    if isinstance(chain, Ligand):
        return _rf3_ligand_residue_name(chain, fixed_ligands=fixed_ligands)
    if isinstance(chain, RnaChain | DnaChain):
        return str(chain.sequence[residue_index - 1]).upper()
    raise TypeError(f"Unsupported RF3 bond chain type: {type(chain).__name__}")


def _build_bonds(
    input_obj: Input,
    *,
    fixed_ligands: dict[str, Any] | None = None,
) -> list[list[str]]:
    bonds: list[list[str]] = []
    chain_by_id = {str(getattr(chain, "id")): chain for chain in input_obj.chains}
    for atom1, atom2 in list(getattr(input_obj, "bonded_atom_pairs", None) or []):
        chain_id1, res_id1, atom_name1 = atom1
        chain_id2, res_id2, atom_name2 = atom2
        chain1 = chain_by_id[str(chain_id1)]
        chain2 = chain_by_id[str(chain_id2)]
        bonds.append(
            [
                f"{chain_id1}/{_resolve_bond_residue_name(chain1, int(res_id1), fixed_ligands=fixed_ligands)}/{int(res_id1)}/{atom_name1}",
                f"{chain_id2}/{_resolve_bond_residue_name(chain2, int(res_id2), fixed_ligands=fixed_ligands)}/{int(res_id2)}/{atom_name2}",
            ]
        )
    return bonds


def _extract_per_residue_plddt_from_structure(structure_path: Path | str) -> dict[str, Any]:
    structure_path = Path(structure_path)
    parser = MMCIFParser(QUIET=True) if structure_path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("rf3", str(structure_path))

    overall: list[float] = []
    by_chain: dict[str, list[float]] = defaultdict(list)
    for model in structure:
        for chain in model:
            for residue in chain:
                atom_bfactors = [
                    float(atom.get_bfactor())
                    for atom in residue.get_atoms()
                    if atom.get_bfactor() is not None
                ]
                if not atom_bfactors:
                    continue
                residue_plddt = float(np.mean(atom_bfactors))
                overall.append(residue_plddt)
                by_chain[str(chain.id)].append(residue_plddt)

    return {
        "overall": np.asarray(overall, dtype=float),
        "by_chain": {
            chain_id: np.asarray(values, dtype=float) for chain_id, values in by_chain.items()
        },
    }


def _coerce_chain_pair_scores(summary_confidences: dict[str, Any] | None) -> dict[str, list[np.ndarray]] | None:
    if not summary_confidences:
        return None

    chain_pair_scores: dict[str, list[np.ndarray]] = {}
    for field_name in (
        "chain_ptm",
        "chain_pair_pae",
        "chain_pair_pde",
        "chain_pair_pae_min",
        "chain_pair_pde_min",
    ):
        value = summary_confidences.get(field_name)
        if value is None:
            continue
        chain_pair_scores[field_name] = [np.asarray(value, dtype=float)]
    return chain_pair_scores or None


class RoseTTAFold3Evaluator(Evaluator):
    def __init__(self, model_runner, output_dir, rf3_conf=None):
        self.model_runner = model_runner
        self.output_dir = Path(output_dir)
        self.rf3_conf = rf3_conf or OmegaConf.create({})

    def prepare_input(self, inputs, runtime_config=None):
        runtime_config = runtime_config or {}
        step = _get_current_step(runtime_config)
        requests: dict[str, RF3RunnerRequest] = {}
        base_overrides = _normalize_rf3_conf_container(self.rf3_conf)
        base_overrides.pop("_type", None)
        fixed_ligands = dict(base_overrides.pop("fixed_ligands", {}) or {})
        template_selection_override = _as_selection_list(
            base_overrides.pop("template_selection", None)
        )
        conformer_selection_override = _as_selection_list(
            base_overrides.pop("ground_truth_conformer_selection", None)
        )
        for oligo_name, input_obj in inputs.items():
            jobname = f"step_{step:03d}_{oligo_name}"
            output_dir = self.output_dir / jobname
            output_dir.mkdir(parents=True, exist_ok=True)
            derived_template_selections: list[str] = []
            derived_conformer_selections: list[str] = []
            derived_protein_template_selections: list[str] = []
            components: list[dict[str, Any]] = []
            has_path_backed_ligand = False
            for chain in input_obj.chains:
                if isinstance(chain, Ligand):
                    component, selection = _ligand_component_and_selection(
                        chain,
                        jobname=jobname,
                        output_dir=output_dir,
                        fixed_ligands=fixed_ligands,
                    )
                    components.append(component)
                    has_path_backed_ligand = has_path_backed_ligand or ("path" in component)
                    derived_template_selections.append(selection)
                    derived_conformer_selections.append(selection)
                else:
                    component = _component_from_chain(
                        chain,
                        jobname=jobname,
                        output_dir=output_dir,
                    )
                    components.append(component)
                    if isinstance(chain, ProteinChain) and "path" in component:
                        selection = str(chain.id)
                        derived_template_selections.append(selection)
                        derived_protein_template_selections.append(selection)
            template_selection = (
                _dedupe_selections(
                    list(template_selection_override) + derived_protein_template_selections
                )
                if template_selection_override is not None
                else _dedupe_selections(derived_template_selections)
            )
            conformer_selection = (
                list(conformer_selection_override)
                if conformer_selection_override is not None
                else _dedupe_selections(derived_conformer_selections)
            )
            payload = {
                "name": jobname,
                "components": components,
                "bonds": _build_bonds(input_obj, fixed_ligands=fixed_ligands),
                "template_selection": template_selection,
                "ground_truth_conformer_selection": conformer_selection,
            }
            prepared_input_path = output_dir / f"{jobname}__input.json"
            prepared_input_path.write_text(
                json.dumps(payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            request_overrides = dict(base_overrides)
            if "seed" not in request_overrides:
                request_seed = _default_request_seed(input_obj, runtime_config)
                if request_seed is not None:
                    request_overrides["seed"] = request_seed
            if has_path_backed_ligand and "add_missing_atoms" not in request_overrides:
                request_overrides["add_missing_atoms"] = False
            requests[oligo_name] = RF3RunnerRequest(
                request_id=jobname,
                prepared_input_path=prepared_input_path,
                output_dir=output_dir,
                template_selection=list(payload["template_selection"]),
                ground_truth_conformer_selection=list(
                    payload["ground_truth_conformer_selection"]
                ),
                rf3_overrides=request_overrides,
            )
        return requests

    def run_function(self, input_feat):
        outputs: dict[str, RF3RunnerResponse] = {}
        for oligo_name, request in input_feat.items():
            response = self.model_runner.submit(request)
            if response.status != "success":
                raise RuntimeError(
                    f"RF3 request {request.request_id!r} failed: {response.error or 'unknown error'}"
                )
            outputs[oligo_name] = response
        return outputs

    def prepare_output(self, outputs, inputs, runtime_config):
        names = list(inputs.keys())
        plddt: dict[str, list[np.ndarray]] = {}
        ptm: dict[str, list[float]] = {}
        iptm: dict[str, list[float]] = {}
        complex_paths: dict[str, Path] = {}
        per_proto_scores: dict[str, list[np.ndarray]] = defaultdict(list)
        chain_pair_scores: dict[str, dict[str, list[np.ndarray]]] = {}
        structures_dir = Path(global_config.out_dir) / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)
        fixed_ligands = dict(_normalize_rf3_conf_container(self.rf3_conf).get("fixed_ligands", {}) or {})

        for oligo_name, response in outputs.items():
            if response.predicted_structure_path is None:
                raise FileNotFoundError(f"RF3 output for {oligo_name!r} does not include a predicted structure path.")

            for chain in inputs[oligo_name].chains:
                if not isinstance(chain, Ligand):
                    continue
                allowed_atom_names = _allowed_atom_names_for_fixed_ligand(
                    chain=chain,
                    fixed_ligands=fixed_ligands,
                )
                if not allowed_atom_names:
                    continue
                rf3_resname = _rf3_ligand_residue_name(
                    chain,
                    fixed_ligands=fixed_ligands,
                )
                _trim_structure_residue_to_atom_names(
                    structure_path=Path(response.predicted_structure_path),
                    chain_id=str(chain.id),
                    resname=rf3_resname,
                    allowed_atom_names=allowed_atom_names,
                )
                _rename_structure_residue(
                    structure_path=Path(response.predicted_structure_path),
                    chain_id=str(chain.id),
                    source_resname=rf3_resname,
                    target_resname=_rf3_output_ligand_residue_name(
                        chain,
                        fixed_ligands=fixed_ligands,
                    ),
                )

            summary = response.summary_confidences
            if summary is None and response.summary_confidence_path is not None:
                summary = json.loads(response.summary_confidence_path.read_text(encoding="utf-8"))
            if summary is None:
                raise FileNotFoundError(
                    f"RF3 output for {oligo_name!r} does not include summary confidences."
                )

            residue_scores = _extract_per_residue_plddt_from_structure(
                response.predicted_structure_path
            )
            plddt[oligo_name] = [np.asarray(residue_scores["overall"], dtype=float)]
            ptm_value = summary.get("ptm")
            iptm_value = summary.get("iptm")
            if ptm_value is not None:
                ptm[oligo_name] = [float(ptm_value)]
            if iptm_value is not None:
                iptm[oligo_name] = [float(iptm_value)]

            canonical_path = structures_dir / f"{response.request_id}_model{Path(response.predicted_structure_path).suffix}"
            shutil.copy(str(response.predicted_structure_path), str(canonical_path))
            complex_paths[oligo_name] = canonical_path
            chain_pair_score_payload = _coerce_chain_pair_scores(summary)
            if chain_pair_score_payload is not None:
                chain_pair_scores[oligo_name] = chain_pair_score_payload

            for chain in inputs[oligo_name].chains:
                if not isinstance(chain, ProteinChain):
                    continue
                chain_id = str(chain.id)
                if chain_id in residue_scores["by_chain"]:
                    per_proto_scores[chain_id].append(
                        np.asarray(residue_scores["by_chain"][chain_id], dtype=float)
                    )

        return StandardScores(
            names=names,
            plddt=plddt,
            ptm=ptm or None,
            iptm=iptm or None,
            per_proto_scores=dict(per_proto_scores) or None,
            chain_pair_scores=chain_pair_scores or None,
            complex_paths=complex_paths,
        )
