from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.SeqUtils import seq1


@dataclass(frozen=True, slots=True)
class ChainMapEntry:
    structure_chain: str
    protomer: str
    structure_residues: Any = None
    protomer_positions: Any = None


@dataclass(frozen=True, slots=True)
class StructureResidue:
    chain: str
    number: int
    insertion_code: str
    amino_acid: str

    @property
    def encoded_id(self) -> str:
        return f"{self.chain}{self.number}{self.insertion_code}"


@dataclass(frozen=True, slots=True)
class ResolvedChainMapping:
    structure_chain: str
    protomer: str
    structure_length: int
    structure_indices: tuple[int, ...]
    protomer_positions: tuple[int, ...]
    residue_ids: tuple[str, ...]
    context_only: bool = False


_RESIDUE_COMPONENT = re.compile(
    r"^(?P<chain>[A-Za-z]+)?(?P<start>\d+)(?P<start_i>[A-Za-z]?)"
    r"(?:-(?:(?P<end_chain>[A-Za-z]+)?(?P<end>\d+)(?P<end_i>[A-Za-z]?)))?$"
)


def normalize_chain_map(raw_value: Any) -> dict[str, ChainMapEntry]:
    if not isinstance(raw_value, dict) or not raw_value:
        raise ValueError("chain_map must map structure chains to protomers.")
    normalized: dict[str, ChainMapEntry] = {}
    for raw_chain, raw_entry in raw_value.items():
        chain = str(raw_chain).strip()
        if isinstance(raw_entry, dict):
            protomer = str(
                raw_entry.get("protomer") or raw_entry.get("target") or ""
            ).strip()
            structure_residues = raw_entry.get(
                "structure_residues",
                raw_entry.get("residues"),
            )
            protomer_positions = raw_entry.get(
                "protomer_positions",
                raw_entry.get("positions"),
            )
        else:
            protomer = str(raw_entry).strip()
            structure_residues = None
            protomer_positions = None
        if not chain or not protomer:
            raise ValueError("chain_map cannot contain empty chains or protomers.")
        normalized[chain] = ChainMapEntry(
            structure_chain=chain,
            protomer=protomer,
            structure_residues=structure_residues,
            protomer_positions=protomer_positions,
        )
    return normalized


def load_structure_residues(
    structure_path: Path,
) -> dict[str, tuple[StructureResidue, ...]]:
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(structure_path)
    lower_name = path.name.lower()
    parser = (
        MMCIFParser(QUIET=True)
        if any(ext in lower_name for ext in (".cif", ".mmcif"))
        else PDBParser(QUIET=True)
    )
    if lower_name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            structure = parser.get_structure("initializer", handle)
    else:
        structure = parser.get_structure("initializer", str(path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"Initializer structure contains no models: {path}")

    result: dict[str, tuple[StructureResidue, ...]] = {}
    for chain in model.get_chains():
        chain_id = str(chain.id).strip()
        residues = []
        for residue in chain.get_residues():
            hetero_flag, residue_number, insertion_code = residue.id
            if str(hetero_flag).strip():
                continue
            residues.append(
                StructureResidue(
                    chain=chain_id,
                    number=int(residue_number),
                    insertion_code=str(insertion_code).strip(),
                    amino_acid=seq1(
                        str(residue.get_resname()).strip().upper(),
                        undef_code="X",
                    ).upper(),
                )
            )
        if residues:
            result[chain_id] = tuple(residues)
    return result


def sequence_matches(pattern: str, observed: str) -> bool:
    return len(pattern) == len(observed) and all(
        expected == actual or expected == "X" or actual == "X"
        for expected, actual in zip(pattern.upper(), observed.upper(), strict=True)
    )


def find_sequence_matches(
    residues_by_chain: dict[str, tuple[StructureResidue, ...]],
    pattern: str,
    *,
    chain: str | None = None,
) -> list[tuple[str, tuple[int, ...]]]:
    matches = []
    for chain_id, residues in residues_by_chain.items():
        if chain is not None and chain_id != chain:
            continue
        observed = "".join(residue.amino_acid for residue in residues)
        for start in range(0, len(observed) - len(pattern) + 1):
            if sequence_matches(pattern, observed[start : start + len(pattern)]):
                matches.append(
                    (chain_id, tuple(range(start, start + len(pattern))))
                )
    return matches


def select_structure_indices(
    residues: tuple[StructureResidue, ...],
    selector: Any,
    *,
    expected_chain: str,
) -> tuple[int, ...]:
    if isinstance(selector, (list, tuple)):
        components = [str(item).strip() for item in selector]
    else:
        components = [item.strip() for item in str(selector).split(",")]
    if not components or any(not component for component in components):
        raise ValueError("Structure residue selection cannot be empty.")

    by_key = {
        (residue.number, residue.insertion_code): index
        for index, residue in enumerate(residues)
    }
    selected: list[int] = []
    for component in components:
        match = _RESIDUE_COMPONENT.fullmatch(component)
        if match is None:
            raise ValueError(f"Invalid structure residue selection: {component!r}")
        chain = match.group("chain") or expected_chain
        end_chain = match.group("end_chain") or chain
        if chain != expected_chain or end_chain != expected_chain:
            raise ValueError(
                f"Structure residue selection {component!r} does not belong to "
                f"chain {expected_chain!r}."
            )
        start_key = (int(match.group("start")), match.group("start_i") or "")
        end_key = (
            int(match.group("end")),
            match.group("end_i") or "",
        ) if match.group("end") else start_key
        if start_key not in by_key or end_key not in by_key:
            raise ValueError(
                f"Structure residue selection {component!r} is missing from chain "
                f"{expected_chain!r}."
            )
        start_index = by_key[start_key]
        end_index = by_key[end_key]
        if end_index < start_index:
            raise ValueError(f"Reversed structure residue selection: {component!r}")
        selected.extend(range(start_index, end_index + 1))
    if len(set(selected)) != len(selected):
        raise ValueError("Structure residue selection contains duplicate residues.")
    return tuple(selected)


def parse_protomer_positions(raw_value: Any, *, length: int) -> tuple[int, ...]:
    if isinstance(raw_value, (list, tuple)):
        components = [str(item).strip() for item in raw_value]
    else:
        components = [item.strip() for item in str(raw_value).split(",")]
    selected: list[int] = []
    for component in components:
        if not component:
            raise ValueError("Protomer position selection cannot be empty.")
        if "-" in component:
            start_text, end_text = component.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(component)
        if start < 1 or end < start or end > length:
            raise ValueError(
                f"Protomer position selection {component!r} is invalid for length {length}."
            )
        selected.extend(range(start - 1, end))
    if len(set(selected)) != len(selected):
        raise ValueError("Protomer position selection contains duplicate positions.")
    return tuple(selected)


def _anchored_window(
    *,
    protomer: Any,
    residues: tuple[StructureResidue, ...],
    segments: dict[str, Any],
) -> tuple[int, ...] | None:
    chain_sequence = "".join(residue.amino_acid for residue in residues)
    candidate_offsets: set[int] | None = None
    for index, segment_name in enumerate(protomer.components):
        segment = segments[segment_name]
        if not segment.is_fixed:
            continue
        segment_slice = protomer.components_slices[index]
        pattern = str(segment.sequence).upper()
        if not pattern or set(pattern) == {"X"}:
            continue
        offsets = {
            start - int(segment_slice.start)
            for start in range(0, len(chain_sequence) - len(pattern) + 1)
            if sequence_matches(pattern, chain_sequence[start : start + len(pattern)])
            and start - int(segment_slice.start) >= 0
            and start - int(segment_slice.start) + protomer.length <= len(residues)
        }
        candidate_offsets = offsets if candidate_offsets is None else candidate_offsets & offsets
    if candidate_offsets is None or len(candidate_offsets) != 1:
        return None
    offset = next(iter(candidate_offsets))
    return tuple(range(offset, offset + protomer.length))


def resolve_chain_mapping(
    entry: ChainMapEntry,
    *,
    residues_by_chain: dict[str, tuple[StructureResidue, ...]],
    protomer: Any,
    segments: dict[str, Any],
    context_only: bool,
) -> ResolvedChainMapping:
    residues = residues_by_chain.get(entry.structure_chain)
    if residues is None:
        available = ", ".join(sorted(residues_by_chain)) or "<none>"
        raise ValueError(
            f"Initializer structure is missing mapped chain {entry.structure_chain!r}. "
            f"Available chains: {available}."
        )
    if entry.structure_residues is not None:
        structure_indices = select_structure_indices(
            residues,
            entry.structure_residues,
            expected_chain=entry.structure_chain,
        )
    elif len(residues) == protomer.length:
        structure_indices = tuple(range(len(residues)))
    else:
        structure_indices = _anchored_window(
            protomer=protomer,
            residues=residues,
            segments=segments,
        ) or ()
        if not structure_indices:
            raise ValueError(
                f"Initializer structure chain {entry.structure_chain!r} has "
                f"{len(residues)} protein residues while protomer {entry.protomer!r} "
                f"has length {protomer.length}; provide chain_map.{entry.structure_chain}."
                "structure_residues and protomer_positions, or a uniquely matching "
                "fixed motif anchor."
            )

    if entry.protomer_positions is not None:
        protomer_positions = parse_protomer_positions(
            entry.protomer_positions,
            length=protomer.length,
        )
    elif len(structure_indices) == protomer.length:
        protomer_positions = tuple(range(protomer.length))
    else:
        raise ValueError(
            f"Partial chain_map for structure chain {entry.structure_chain!r} requires "
            "protomer_positions."
        )
    if len(structure_indices) != len(protomer_positions):
        raise ValueError(
            f"chain_map.{entry.structure_chain} selects {len(structure_indices)} structure "
            f"residues but {len(protomer_positions)} protomer positions."
        )
    if context_only:
        mismatches = [
            protomer_position + 1
            for structure_index, protomer_position in zip(
                structure_indices,
                protomer_positions,
                strict=True,
            )
            if not sequence_matches(
                str(protomer.sequence)[protomer_position],
                residues[structure_index].amino_acid,
            )
        ]
        if mismatches:
            raise ValueError(
                f"Initializer context chain {entry.structure_chain!r} does not match "
                f"fixed protomer {entry.protomer!r} at position(s): "
                + ", ".join(str(position) for position in mismatches)
            )
    return ResolvedChainMapping(
        structure_chain=entry.structure_chain,
        protomer=entry.protomer,
        structure_length=len(residues),
        structure_indices=structure_indices,
        protomer_positions=protomer_positions,
        residue_ids=tuple(residues[index].encoded_id for index in structure_indices),
        context_only=context_only,
    )
