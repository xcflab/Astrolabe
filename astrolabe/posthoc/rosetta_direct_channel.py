from __future__ import annotations

import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


_ONE_TO_THREE = {
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
_THREE_TO_ONE = {value: key for key, value in _ONE_TO_THREE.items()}
_PROTEIN_CONTACT_ELEMENTS = frozenset({"C", "N", "O", "S", "P"})
_LIGAND_CONTACT_ELEMENTS = frozenset({"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"})


@dataclass(frozen=True)
class ChainResidue:
    chain_position: int
    pose_index: int
    pdb_resseq: int
    pdb_icode: str
    resname3: str
    resname1: str


@dataclass(frozen=True)
class PlannedMutation:
    chain_position: int
    pose_index: int
    pdb_resseq: int
    pdb_icode: str
    original_aa: str
    target_aa: str


def normalize_target_aas(raw_values: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw_values, str):
        tokens = [
            token.strip()
            for token in re.split(r"[\s,|;]+", str(raw_values or "").strip())
            if token.strip()
        ]
    else:
        tokens = [str(token or "").strip() for token in raw_values if str(token or "").strip()]

    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        upper = token.upper()
        if len(upper) == 1:
            aa = upper
        else:
            aa = _THREE_TO_ONE.get(upper)
        if aa is None or aa not in _ONE_TO_THREE:
            raise ValueError(f"Unsupported target amino acid token: {token!r}")
        if aa in seen:
            continue
        seen.add(aa)
        ordered.append(aa)
    if not ordered:
        raise ValueError("At least one target amino acid is required.")
    return tuple(ordered)


def _format_chain_residue_id(chain_id: str, pdb_resseq: int, pdb_icode: str = "") -> str:
    suffix = str(pdb_icode or "").strip()
    return f"{str(chain_id).strip()}{int(pdb_resseq)}{suffix}"


def _load_biopython_model(structure_path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    path = Path(structure_path).expanduser().resolve()
    name = path.name.lower()
    is_cif = any(ext in name for ext in (".cif", ".mmcif"))
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            structure = parser.get_structure(path.stem, handle)
    else:
        structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in structure: {path}")
    return model


def _find_chain(model, chain_id: str):
    target = str(chain_id or "").strip()
    for chain in model.get_chains():
        if str(chain.id).strip() == target:
            return chain
    raise ValueError(f"Chain {target!r} not found in structure model.")


def _find_structure_residue(model, *, chain_id: str, resname: str, resseq: int, icode: str = ""):
    chain = _find_chain(model, chain_id)
    want_icode = str(icode or "").strip()
    want_resname = str(resname or "").strip()
    for residue in chain.get_residues():
        _hetflag, current_resseq, current_icode = residue.get_id()
        if int(current_resseq) != int(resseq):
            continue
        if str(current_icode or "").strip() != want_icode:
            continue
        if str(residue.get_resname()).strip() != want_resname:
            continue
        return residue
    selector = f"{chain_id}:{resname}:{int(resseq)}"
    if want_icode:
        selector = f"{selector}:{want_icode}"
    raise ValueError(f"Residue {selector} not found in structure model.")


def _normalize_atom_element(atom) -> str:
    raw = str(getattr(atom, "element", "") or "").strip()
    if not raw:
        letters = "".join(ch for ch in str(atom.get_name() or "").strip() if ch.isalpha())
        if not letters:
            return ""
        if letters[:2].upper() in {"CL", "BR"}:
            raw = letters[:2]
        else:
            raw = letters[:1]
    if len(raw) == 1:
        return raw.upper()
    return raw[0].upper() + raw[1:].lower()


def _heavy_atom_coords(residue, *, allowed_elements: Iterable[str]) -> np.ndarray:
    allowed = {str(element) for element in allowed_elements}
    coords: list[np.ndarray] = []
    for atom in residue.get_atoms():
        element = _normalize_atom_element(atom)
        if not element or element == "H" or element not in allowed:
            continue
        coords.append(np.asarray(atom.get_coord(), dtype=float))
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(coords)


def _min_pairwise_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return float("inf")
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(np.min(distances))


def ligand_contact_chain_positions_from_structure(
    *,
    structure_path: str | Path,
    protein_chain_id: str,
    ligand_chain_id: str,
    ligand_resname: str,
    ligand_resseq: int,
    ligand_icode: str = "",
    chain_position_by_pdb_residue: dict[tuple[int, str], int],
    distance_min: float = 0.0,
    distance_max: float = 5.0,
    protein_elements: Iterable[str] = _PROTEIN_CONTACT_ELEMENTS,
    ligand_elements: Iterable[str] = _LIGAND_CONTACT_ELEMENTS,
) -> tuple[int, ...]:
    if distance_max <= 0:
        return ()
    if distance_min < 0:
        raise ValueError("distance_min must be >= 0")
    if distance_min > distance_max:
        raise ValueError("distance_min must be <= distance_max")

    model = _load_biopython_model(structure_path)
    protein_chain = _find_chain(model, protein_chain_id)
    ligand_residue = _find_structure_residue(
        model,
        chain_id=ligand_chain_id,
        resname=ligand_resname,
        resseq=ligand_resseq,
        icode=ligand_icode,
    )
    ligand_coords = _heavy_atom_coords(ligand_residue, allowed_elements=ligand_elements)
    if ligand_coords.size == 0:
        raise ValueError("Selected ligand residue has no heavy atoms for contact detection.")

    contact_positions: list[int] = []
    for residue in protein_chain.get_residues():
        hetflag, resseq, icode = residue.get_id()
        if str(hetflag).strip():
            continue
        key = (int(resseq), str(icode or "").strip())
        chain_position = chain_position_by_pdb_residue.get(key)
        if chain_position is None:
            continue
        residue_coords = _heavy_atom_coords(residue, allowed_elements=protein_elements)
        if residue_coords.size == 0:
            continue
        min_distance = _min_pairwise_distance(residue_coords, ligand_coords)
        if float(distance_min) <= min_distance <= float(distance_max):
            contact_positions.append(int(chain_position))
    return tuple(sorted(set(contact_positions)))


def build_pocket_fixed_screen_plan(
    *,
    structure_path: str | Path,
    protein_chain_id: str,
    mutable_positions_path: str | Path,
    ligand_chain_id: str,
    ligand_resname: str,
    ligand_resseq: int,
    ligand_icode: str = "",
    target_aas: str | Sequence[str] = ("A", "S", "T", "G"),
    contact_distance: float = 5.0,
) -> dict[str, Any]:
    from astrolabe.cli_channel import load_mutable_positions

    residues = protein_chain_residues_from_structure(structure_path, protein_chain_id)
    sequence = "".join(item.resname1 for item in residues)
    mutable_positions = tuple(
        load_mutable_positions(
            Path(mutable_positions_path).expanduser().resolve(),
            sequence_length=len(sequence),
        )
    )
    residue_by_chain_position = {item.chain_position: item for item in residues}
    chain_position_by_pdb_residue = {
        (item.pdb_resseq, item.pdb_icode): item.chain_position for item in residues
    }
    pocket_positions = ligand_contact_chain_positions_from_structure(
        structure_path=structure_path,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        ligand_resname=ligand_resname,
        ligand_resseq=ligand_resseq,
        ligand_icode=ligand_icode,
        chain_position_by_pdb_residue=chain_position_by_pdb_residue,
        distance_max=float(contact_distance),
    )
    protected_positions = tuple(pos for pos in mutable_positions if pos in set(pocket_positions))
    design_positions = tuple(pos for pos in mutable_positions if pos not in set(protected_positions))
    target_codes = normalize_target_aas(target_aas)

    variants: list[dict[str, Any]] = []
    for aa in target_codes:
        sequence_chars = list(sequence)
        for position in design_positions:
            sequence_chars[int(position) - 1] = aa
        variants.append(
            {
                "name": f"pocket_fixed_all_{_ONE_TO_THREE[aa].lower()}",
                "target_aa": aa,
                "target_resname": _ONE_TO_THREE[aa],
                "target_sequence": "".join(sequence_chars),
            }
        )

    return {
        "structure_path": str(Path(structure_path).expanduser().resolve()),
        "protein_chain_id": str(protein_chain_id),
        "mutable_positions_path": str(Path(mutable_positions_path).expanduser().resolve()),
        "ligand_residue": {
            "chain_id": str(ligand_chain_id),
            "resname": str(ligand_resname),
            "resseq": int(ligand_resseq),
            "icode": str(ligand_icode or "").strip(),
        },
        "contact_distance": float(contact_distance),
        "mutable_positions": list(mutable_positions),
        "mutable_residue_ids": [
            _format_chain_residue_id(
                protein_chain_id,
                residue_by_chain_position[position].pdb_resseq,
                residue_by_chain_position[position].pdb_icode,
            )
            for position in mutable_positions
        ],
        "protected_positions": list(protected_positions),
        "protected_residue_ids": [
            _format_chain_residue_id(
                protein_chain_id,
                residue_by_chain_position[position].pdb_resseq,
                residue_by_chain_position[position].pdb_icode,
            )
            for position in protected_positions
        ],
        "design_positions": list(design_positions),
        "design_residue_ids": [
            _format_chain_residue_id(
                protein_chain_id,
                residue_by_chain_position[position].pdb_resseq,
                residue_by_chain_position[position].pdb_icode,
            )
            for position in design_positions
        ],
        "variants": variants,
    }


def residue_heavy_atom_names_from_structure(
    *,
    structure_path: str | Path,
    chain_id: str,
    resname: str,
    resseq: int,
    icode: str = "",
) -> list[str]:
    residue = _find_structure_residue(
        _load_biopython_model(structure_path),
        chain_id=chain_id,
        resname=resname,
        resseq=resseq,
        icode=icode,
    )
    atoms: list[str] = []
    for atom in residue.get_atoms():
        element = _normalize_atom_element(atom)
        if element == "H":
            continue
        atoms.append(str(atom.get_name()).strip())
    return atoms


def run_pocket_fixed_screen(
    *,
    structure_path: str | Path,
    protein_chain_id: str,
    mutable_positions_path: str | Path,
    ligand_chain_id: str,
    ligand_resname: str,
    ligand_resseq: int,
    ligand_icode: str = "",
    reference_spheres_path: str | Path,
    output_root: str | Path,
    reference_structure_path: str | Path | None = None,
    alignment_chains: str | Sequence[str] | None = None,
    target_aas: str | Sequence[str] = ("A", "S", "T", "G"),
    contact_distance: float = 5.0,
    radius_gate_path_len_threshold: float = 40.0,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    prepare_dir = output_root / "01_prepare"
    runs_dir = output_root / "02_runs"
    collect_dir = output_root / "03_collect"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    collect_dir.mkdir(parents=True, exist_ok=True)

    reference_structure = Path(reference_structure_path or structure_path).expanduser().resolve()
    plan = build_pocket_fixed_screen_plan(
        structure_path=structure_path,
        protein_chain_id=protein_chain_id,
        mutable_positions_path=mutable_positions_path,
        ligand_chain_id=ligand_chain_id,
        ligand_resname=ligand_resname,
        ligand_resseq=ligand_resseq,
        ligand_icode=ligand_icode,
        target_aas=target_aas,
        contact_distance=contact_distance,
    )
    plan["reference_structure_path"] = str(reference_structure)
    plan["reference_spheres_path"] = str(Path(reference_spheres_path).expanduser().resolve())
    plan["alignment_chains"] = str(alignment_chains or protein_chain_id)
    plan_json = prepare_dir / "pocket_fixed_screen_plan.json"
    plan_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    template_ligand_atoms = residue_heavy_atom_names_from_structure(
        structure_path=structure_path,
        chain_id=ligand_chain_id,
        resname=ligand_resname,
        resseq=ligand_resseq,
        icode=ligand_icode,
    )

    rows: list[dict[str, Any]] = []
    for variant in plan["variants"]:
        variant_name = str(variant["name"])
        variant_dir = runs_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        mutated_structure_path = variant_dir / "mutated_structure.pdb"
        mutated_path, mutation_plan = mutate_structure_to_sequence(
            structure_path=structure_path,
            chain_id=protein_chain_id,
            target_sequence=str(variant["target_sequence"]),
            output_path=mutated_structure_path,
            template_structure_path=structure_path,
        )
        mutation_plan_path = variant_dir / "mutation_plan.json"
        mutation_plan_path.write_text(
            json.dumps(
                [
                    {
                        "chain_position": item.chain_position,
                        "pose_index": item.pose_index,
                        "pdb_resseq": item.pdb_resseq,
                        "pdb_icode": item.pdb_icode,
                        "original_aa": item.original_aa,
                        "target_aa": item.target_aa,
                    }
                    for item in mutation_plan
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics = score_mutated_structure(
            structure_path=mutated_path,
            output_dir=variant_dir / "channel_hole",
            reference_structure_path=reference_structure,
            reference_spheres_path=reference_spheres_path,
            alignment_chains=alignment_chains or protein_chain_id,
            radius_gate_path_len_threshold=radius_gate_path_len_threshold,
            hole_executable=hole_executable,
            hole_radius_file=hole_radius_file,
        )
        row = {
            "variant": variant_name,
            "target_aa": str(variant["target_aa"]),
            "target_resname": str(variant["target_resname"]),
            "contact_distance": float(contact_distance),
            "mutable_positions_count": len(plan["mutable_positions"]),
            "protected_positions_count": len(plan["protected_positions"]),
            "design_positions_count": len(plan["design_positions"]),
            "protected_residue_ids": ",".join(plan["protected_residue_ids"]),
            "design_residue_ids": ",".join(plan["design_residue_ids"]),
            "mutation_count": len(mutation_plan),
            "mutation_residue_ids": ",".join(
                _format_chain_residue_id(protein_chain_id, item.pdb_resseq, item.pdb_icode)
                for item in mutation_plan
            ),
            "mutated_structure_path": str(mutated_path),
            "mutation_plan_path": str(mutation_plan_path),
            "channel_hole_dir": str((variant_dir / "channel_hole").resolve()),
            "best_channel_min_radius": float(metrics["best_channel_min_radius"]),
            "best_channel_effective_path_len": float(metrics["best_channel_effective_path_len"]),
            "channel_hole": float(metrics["channel_hole"]),
            "best_channel_scan_id": str(metrics["best_channel_scan_id"]),
            "max_geometric_channel_min_radius": float(metrics["max_geometric_channel_min_radius"]),
            "max_geometric_channel_effective_path_len": float(
                metrics["max_geometric_channel_effective_path_len"]
            ),
            "max_geometric_channel_scan_id": str(metrics["max_geometric_channel_scan_id"]),
        }
        mutated_ligand_atoms = residue_heavy_atom_names_from_structure(
            structure_path=mutated_path,
            chain_id=ligand_chain_id,
            resname=ligand_resname,
            resseq=ligand_resseq,
            icode=ligand_icode,
        )
        row["ligand_atom_count"] = len(mutated_ligand_atoms)
        row["ligand_atom_names_match_template"] = mutated_ligand_atoms == template_ligand_atoms
        rows.append(row)

    best_variant = max(rows, key=lambda item: float(item["best_channel_min_radius"])) if rows else None
    summary_json = collect_dir / "pocket_fixed_screen_summary.json"
    summary_csv = collect_dir / "pocket_fixed_screen_summary.csv"
    summary_payload = {
        "output_root": str(output_root),
        "plan_json": str(plan_json),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "structure_path": plan["structure_path"],
        "reference_structure_path": plan["reference_structure_path"],
        "reference_spheres_path": plan["reference_spheres_path"],
        "protein_chain_id": plan["protein_chain_id"],
        "ligand_residue": plan["ligand_residue"],
        "contact_distance": float(contact_distance),
        "mutable_positions_count": len(plan["mutable_positions"]),
        "protected_positions_count": len(plan["protected_positions"]),
        "design_positions_count": len(plan["design_positions"]),
        "protected_residue_ids": plan["protected_residue_ids"],
        "design_residue_ids": plan["design_residue_ids"],
        "target_aas": [str(item["target_aa"]) for item in plan["variants"]],
        "best_variant": best_variant["variant"] if best_variant else None,
        "best_variant_min_radius": float(best_variant["best_channel_min_radius"]) if best_variant else None,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return summary_payload


def _import_pyrosetta():
    import pyrosetta  # type: ignore

    return pyrosetta


def init_pyrosetta_once(
    init_options: str = "-ignore_zero_occupancy false -mute all",
):
    pyrosetta = _import_pyrosetta()
    if hasattr(pyrosetta, "is_initialized") and pyrosetta.is_initialized():
        return pyrosetta
    pyrosetta.init(str(init_options))
    return pyrosetta


def load_pose(structure_path: str | Path, *, init_options: str = "-ignore_zero_occupancy false -mute all"):
    pyrosetta = init_pyrosetta_once(init_options)
    return pyrosetta.pose_from_file(str(Path(structure_path).expanduser().resolve()))


def protein_chain_residues(pose, chain_id: str) -> tuple[ChainResidue, ...]:
    info = pose.pdb_info()
    rows: list[ChainResidue] = []
    chain = str(chain_id or "").strip()
    if not chain:
        raise ValueError("chain_id must not be empty")
    chain_position = 0
    for pose_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(pose_index)
        if str(info.chain(pose_index)).strip() != chain or not residue.is_protein():
            continue
        chain_position += 1
        rows.append(
            ChainResidue(
                chain_position=chain_position,
                pose_index=pose_index,
                pdb_resseq=int(info.number(pose_index)),
                pdb_icode=str(info.icode(pose_index) or "").strip(),
                resname3=str(residue.name3()).strip(),
                resname1=str(residue.name1()).strip().upper(),
            )
        )
    if not rows:
        raise ValueError(f"No protein residues found for chain {chain!r}")
    return tuple(rows)


def protein_chain_residues_from_structure(
    structure_path: str | Path,
    chain_id: str,
) -> tuple[ChainResidue, ...]:
    chain = _find_chain(_load_biopython_model(structure_path), chain_id)
    rows: list[ChainResidue] = []
    chain_position = 0
    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.get_id()
        if str(hetflag).strip():
            continue
        resname3 = str(residue.get_resname()).strip().upper()
        resname1 = _THREE_TO_ONE.get(resname3)
        if resname1 is None:
            raise ValueError(
                f"Unsupported protein residue {resname3!r} in chain {chain_id!r} at {int(resseq)}{str(icode or '').strip()}"
            )
        chain_position += 1
        rows.append(
            ChainResidue(
                chain_position=chain_position,
                pose_index=chain_position,
                pdb_resseq=int(resseq),
                pdb_icode=str(icode or "").strip(),
                resname3=resname3,
                resname1=resname1,
            )
        )
    if not rows:
        raise ValueError(f"No protein residues found for chain {chain_id!r}")
    return tuple(rows)


def plan_chain_mutations_to_sequence(
    pose,
    *,
    chain_id: str,
    target_sequence: str,
) -> tuple[PlannedMutation, ...]:
    target = str(target_sequence or "").strip().upper()
    residues = protein_chain_residues(pose, chain_id)
    if len(target) != len(residues):
        raise ValueError(
            f"Target sequence length mismatch for chain {chain_id!r}: "
            f"{len(target)} vs {len(residues)}"
        )
    plan: list[PlannedMutation] = []
    for residue, target_aa in zip(residues, target):
        if target_aa not in _ONE_TO_THREE:
            raise ValueError(f"Unsupported target residue letter: {target_aa!r}")
        if residue.resname1 == target_aa:
            continue
        plan.append(
            PlannedMutation(
                chain_position=residue.chain_position,
                pose_index=residue.pose_index,
                pdb_resseq=residue.pdb_resseq,
                pdb_icode=residue.pdb_icode,
                original_aa=residue.resname1,
                target_aa=target_aa,
            )
        )
    return tuple(plan)


def mutate_pose_with_plan(
    pose,
    mutation_plan: Sequence[PlannedMutation],
    *,
    preserve_atom_coords: bool = True,
):
    pyrosetta = _import_pyrosetta()
    mutator_cls = pyrosetta.rosetta.protocols.simple_moves.MutateResidue
    mutated_pose = pose.clone()
    for item in mutation_plan:
        mutator = mutator_cls(int(item.pose_index), _ONE_TO_THREE[str(item.target_aa).upper()])
        if hasattr(mutator, "set_preserve_atom_coords"):
            mutator.set_preserve_atom_coords(bool(preserve_atom_coords))
        mutator.apply(mutated_pose)
    return mutated_pose


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


def _template_nonprotein_atom_lines(template_structure_path: str | Path) -> list[str]:
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    path = Path(template_structure_path).expanduser().resolve()
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    lines: list[str] = []
    serial = 1
    for chain in model.get_chains():
        for residue in chain.get_residues():
            hetflag, resseq, icode = residue.get_id()
            if str(hetflag).strip() == "":
                continue
            for atom in residue.get_atoms():
                element = str(getattr(atom, "element", "") or "").strip().upper()
                if element == "H" or str(atom.get_name()).strip().upper().startswith("H"):
                    continue
                coord = atom.get_coord()
                lines.append(
                    _format_pdb_atom_line(
                        serial=serial,
                        record_name="HETATM",
                        atom_name=str(atom.get_fullname() or atom.get_name()),
                        altloc=str(atom.get_altloc() or " ").replace(".", " "),
                        resname=str(residue.get_resname()).strip(),
                        chain_id=str(chain.id).strip() or "A",
                        resseq=int(resseq),
                        icode=str(icode or "").strip(),
                        x=float(coord[0]),
                        y=float(coord[1]),
                        z=float(coord[2]),
                        occupancy=float(atom.get_occupancy() or 1.0),
                        bfactor=float(atom.get_bfactor() or 0.0),
                        element=element or str(atom.get_name()).strip()[:1],
                    )
                )
                serial += 1
    return lines


def _template_link_lines(template_structure_path: str | Path) -> list[str]:
    path = Path(template_structure_path).expanduser().resolve()
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle if line.startswith("LINK")]


def export_pose_heavyatom_pdb(
    pose,
    *,
    output_path: str | Path,
    template_structure_path: str | Path | None = None,
) -> Path:
    info = pose.pdb_info()
    lines: list[str] = []
    if template_structure_path is not None:
        lines.extend(_template_link_lines(template_structure_path))
    serial = 1
    previous_chain = None
    for pose_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(pose_index)
        if not residue.is_protein():
            continue
        chain_id = str(info.chain(pose_index)).strip() or "A"
        if previous_chain is not None and chain_id != previous_chain:
            lines.append(f"TER   {serial:5d}")
            serial += 1
        previous_chain = chain_id
        for atom_index in range(1, residue.natoms() + 1):
            if residue.atom_is_hydrogen(atom_index):
                continue
            atom_name = str(residue.atom_name(atom_index))
            coords = residue.xyz(atom_index)
            element = str(residue.atom_type(atom_index).element() or "").strip().upper()
            if not element or element == "X":
                continue
            lines.append(
                _format_pdb_atom_line(
                    serial=serial,
                    record_name="ATOM",
                    atom_name=atom_name,
                    altloc=" ",
                    resname=str(residue.name3()).strip(),
                    chain_id=chain_id,
                    resseq=int(info.number(pose_index)),
                    icode=str(info.icode(pose_index) or "").strip(),
                    x=float(coords.x),
                    y=float(coords.y),
                    z=float(coords.z),
                    occupancy=1.0,
                    bfactor=0.0,
                    element=element or atom_name.strip()[:1],
                )
            )
            serial += 1
    if previous_chain is not None:
        lines.append(f"TER   {serial:5d}")
        serial += 1
    if template_structure_path is not None:
        for nonprotein_line in _template_nonprotein_atom_lines(template_structure_path):
            lines.append(nonprotein_line[:6] + f"{serial:5d}" + nonprotein_line[11:])
            serial += 1
    lines.append("END")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def mutate_structure_to_sequence(
    *,
    structure_path: str | Path,
    chain_id: str,
    target_sequence: str,
    output_path: str | Path,
    template_structure_path: str | Path | None = None,
    init_options: str = "-ignore_zero_occupancy false -mute all",
    preserve_atom_coords: bool = True,
) -> tuple[Path, tuple[PlannedMutation, ...]]:
    pose = load_pose(structure_path, init_options=init_options)
    mutation_plan = plan_chain_mutations_to_sequence(
        pose,
        chain_id=chain_id,
        target_sequence=target_sequence,
    )
    mutated_pose = mutate_pose_with_plan(
        pose,
        mutation_plan,
        preserve_atom_coords=preserve_atom_coords,
    )
    exported = export_pose_heavyatom_pdb(
        mutated_pose,
        output_path=output_path,
        template_structure_path=template_structure_path or structure_path,
    )
    return exported, mutation_plan


def score_mutated_structure(
    *,
    structure_path: str | Path,
    output_dir: str | Path,
    reference_structure_path: str | Path,
    reference_spheres_path: str | Path,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = 2.0,
    end_radius: float = 5.0,
    effective_radius_cutoff: float = 4.0,
    radius_gate_path_len_threshold: float = 40.0,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
) -> dict[str, object]:
    from astrolabe.losses.channel_hole import compute_channel_hole_metrics_for_structure

    return compute_channel_hole_metrics_for_structure(
        structure_path=structure_path,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        output_dir=output_dir,
        alignment_chains=alignment_chains,
        lateral_step=lateral_step,
        end_radius=end_radius,
        effective_radius_cutoff=effective_radius_cutoff,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        hole_executable=hole_executable,
        hole_radius_file=hole_radius_file,
    )
