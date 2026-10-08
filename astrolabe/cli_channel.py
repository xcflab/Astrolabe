from __future__ import annotations

import gzip
import io
import pathlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResidueRef:
    chain_id: str
    resname: str
    resseq: int
    icode: str = ""


def _open_text(path: pathlib.Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_ligand_residue_selector(raw: str) -> ResidueRef:
    text = str(raw or "").strip()
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in {3, 4}:
        raise ValueError(
            "--ligand-residue must be '<chain>:<resname>:<resseq>[:<icode>]'"
        )
    chain_id, resname, resseq_text = parts[:3]
    icode = parts[3] if len(parts) == 4 else ""
    if not chain_id:
        raise ValueError("--ligand-residue chain id must not be empty")
    if not resname:
        raise ValueError("--ligand-residue resname must not be empty")
    try:
        resseq = int(resseq_text)
    except ValueError as exc:
        raise ValueError(
            f"--ligand-residue resseq must be an integer, got {resseq_text!r}"
        ) from exc
    return ResidueRef(
        chain_id=chain_id,
        resname=resname,
        resseq=resseq,
        icode=icode,
    )


def load_mutable_positions(path: pathlib.Path, *, sequence_length: int) -> list[int]:
    text = path.read_text(encoding="utf-8")
    positions: list[int] = []
    seen: set[int] = set()

    def _parse_endpoint(raw_token: str) -> int:
        token = str(raw_token or "").strip()
        if not token:
            raise ValueError("Empty mutable position token.")
        if ":" in token:
            _chain_id, pos_text = token.split(":", 1)
        else:
            match = re.search(r"\d", token)
            if match is None:
                raise ValueError(f"Invalid mutable position token: {raw_token!r}")
            pos_text = token[match.start() :]
        if not pos_text.isdigit():
            raise ValueError(f"Invalid mutable position token: {raw_token!r}")
        value = int(pos_text)
        if value <= 0:
            raise ValueError(f"Invalid mutable position: {raw_token!r}")
        return value

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in [item for item in line.replace(",", " ").split() if item]:
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = _parse_endpoint(start_text)
                end = _parse_endpoint(end_text)
                if end < start:
                    raise ValueError(f"Invalid mutable position range: {token!r}")
                values = range(start, end + 1)
            else:
                start = _parse_endpoint(token)
                values = [start]
            for pos in values:
                if pos > sequence_length:
                    raise ValueError(
                        f"Mutable position {pos} is out of range for sequence length {sequence_length}"
                    )
                if pos not in seen:
                    seen.add(pos)
                    positions.append(pos)
    if not positions:
        raise ValueError(f"No mutable positions found in {path}")
    return positions


def build_mutable_sequence_statement(sequence: str, mutable_positions: list[int]) -> str:
    if not sequence:
        raise ValueError("Protein sequence must not be empty")
    mutable = set(int(pos) for pos in mutable_positions)
    parts: list[str] = []
    start = 0
    current_mutable = 1 in mutable
    for idx in range(2, len(sequence) + 1):
        is_mutable = idx in mutable
        if is_mutable == current_mutable:
            continue
        segment = sequence[start : idx - 1]
        parts.append(segment if current_mutable else f"+{segment}")
        start = idx - 1
        current_mutable = is_mutable
    tail = sequence[start:]
    parts.append(tail if current_mutable else f"+{tail}")
    return " ".join(part for part in parts if part)


def find_residue(structure_path: pathlib.Path, residue_ref: ResidueRef):
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    name = structure_path.name.lower()
    is_cif = any(ext in name for ext in (".cif", ".mmcif"))
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    if name.endswith(".gz"):
        with gzip.open(structure_path, "rt", encoding="utf-8", errors="replace") as handle:
            structure = parser.get_structure("input", handle)
    else:
        structure = parser.get_structure("input", str(structure_path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in structure: {structure_path}")
    for chain in model.get_chains():
        if str(chain.id).strip() != residue_ref.chain_id:
            continue
        for residue in chain.get_residues():
            hetflag, resseq, icode = residue.get_id()
            residue_icode = str(icode or "").strip()
            if int(resseq) != residue_ref.resseq:
                continue
            if residue_icode != str(residue_ref.icode or "").strip():
                continue
            resname = str(residue.get_resname()).strip()
            if resname != residue_ref.resname:
                raise ValueError(
                    f"Ligand residue selector expected resname {residue_ref.resname!r}, "
                    f"found {resname!r}"
                )
            return residue
    raise ValueError(
        f"Ligand residue {residue_ref.chain_id}:{residue_ref.resname}:{residue_ref.resseq}"
        f"{':' + residue_ref.icode if residue_ref.icode else ''} not found in {structure_path}"
    )


def _parse_pdb_atom_records(lines: list[str]) -> dict[int, dict[str, Any]]:
    atoms: dict[int, dict[str, Any]] = {}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11].strip())
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        atoms[serial] = {
            "record_type": line[:6].strip(),
            "chain_id": line[21:22].strip() or "A",
            "resname": line[17:20].strip(),
            "resseq": resseq,
            "icode": line[26:27].strip(),
            "atom_name": line[12:16].strip(),
        }
    return atoms


def _pair_to_bond(
    atom_a: dict[str, Any],
    atom_b: dict[str, Any],
    *,
    protein_chain_id: str,
    ligand_ref: ResidueRef,
) -> dict[str, Any] | None:
    ligand_match_a = (
        atom_a["chain_id"] == ligand_ref.chain_id
        and atom_a["resname"] == ligand_ref.resname
        and atom_a["resseq"] == ligand_ref.resseq
        and atom_a["icode"] == ligand_ref.icode
    )
    ligand_match_b = (
        atom_b["chain_id"] == ligand_ref.chain_id
        and atom_b["resname"] == ligand_ref.resname
        and atom_b["resseq"] == ligand_ref.resseq
        and atom_b["icode"] == ligand_ref.icode
    )
    protein_match_a = atom_a["chain_id"] == protein_chain_id and not ligand_match_a
    protein_match_b = atom_b["chain_id"] == protein_chain_id and not ligand_match_b
    if ligand_match_a and protein_match_b:
        ligand_atom = atom_a
        protein_atom = atom_b
    elif ligand_match_b and protein_match_a:
        ligand_atom = atom_b
        protein_atom = atom_a
    else:
        return None
    return {
        "protein_atom": {
            "chain_id": protein_atom["chain_id"],
            "resname": protein_atom["resname"],
            "resseq": protein_atom["resseq"],
            "icode": protein_atom["icode"],
            "atom_name": protein_atom["atom_name"],
        },
        "ligand_atom": {
            "chain_id": ligand_atom["chain_id"],
            "resname": ligand_atom["resname"],
            "resseq": ligand_atom["resseq"],
            "icode": ligand_atom["icode"],
            "atom_name": ligand_atom["atom_name"],
        },
    }


def extract_structure_covalent_bonds(
    structure_path: pathlib.Path,
    *,
    protein_chain_id: str,
    ligand_ref: ResidueRef,
) -> list[dict[str, Any]]:
    name = structure_path.name.lower()
    if any(ext in name for ext in (".cif", ".mmcif")):
        return []
    with _open_text(structure_path) as handle:
        lines = handle.read().splitlines()
    bonds: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, int, str]] = set()
    for line in lines:
        if not line.startswith("LINK"):
            continue
        try:
            atom_a = {
                "record_type": "LINK",
                "chain_id": line[21:22].strip() or "A",
                "resname": line[17:20].strip(),
                "resseq": int(line[22:26].strip()),
                "icode": line[26:27].strip(),
                "atom_name": line[12:16].strip(),
            }
            atom_b = {
                "record_type": "LINK",
                "chain_id": line[51:52].strip() or "A",
                "resname": line[47:50].strip(),
                "resseq": int(line[52:56].strip()),
                "icode": line[56:57].strip(),
                "atom_name": line[42:46].strip(),
            }
        except ValueError:
            continue
        bond = _pair_to_bond(
            atom_a,
            atom_b,
            protein_chain_id=protein_chain_id,
            ligand_ref=ligand_ref,
        )
        if bond is None:
            continue
        key = (
            str(bond["protein_atom"]["chain_id"]),
            int(bond["protein_atom"]["resseq"]),
            str(bond["protein_atom"]["atom_name"]),
            str(bond["ligand_atom"]["chain_id"]),
            int(bond["ligand_atom"]["resseq"]),
            str(bond["ligand_atom"]["atom_name"]),
        )
        if key in seen:
            continue
        seen.add(key)
        bonds.append(bond)

    atom_records = _parse_pdb_atom_records(lines)
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            src = int(fields[1])
        except ValueError:
            continue
        for token in fields[2:]:
            try:
                dst = int(token)
            except ValueError:
                continue
            atom_a = atom_records.get(src)
            atom_b = atom_records.get(dst)
            if atom_a is None or atom_b is None:
                continue
            bond = _pair_to_bond(
                atom_a,
                atom_b,
                protein_chain_id=protein_chain_id,
                ligand_ref=ligand_ref,
            )
            if bond is None:
                continue
            key = (
                str(bond["protein_atom"]["chain_id"]),
                int(bond["protein_atom"]["resseq"]),
                str(bond["protein_atom"]["atom_name"]),
                str(bond["ligand_atom"]["chain_id"]),
                int(bond["ligand_atom"]["resseq"]),
                str(bond["ligand_atom"]["atom_name"]),
            )
            if key in seen:
                continue
            seen.add(key)
            bonds.append(bond)
    return bonds


def residue_to_pdb_block(residue) -> str:
    from Bio.PDB import PDBIO  # type: ignore

    io_obj = PDBIO()
    io_obj.set_structure(residue)
    buffer = io.StringIO()
    io_obj.save(buffer, write_end=True)
    return buffer.getvalue()
