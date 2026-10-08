from __future__ import annotations

import itertools
import re

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.base import loss_registry


def _parse_included_protomers(
    included_protomers: str,
    *,
    loss_name: str = "sequence_residue_fraction",
    reject_duplicates: bool = False,
) -> list[str]:
    protomer_names = [
        name.strip()
        for name in str(included_protomers).split(",")
        if name.strip()
    ]
    if not protomer_names:
        raise ValueError(
            f"{loss_name} requires at least one included protomer."
        )
    if reject_duplicates and len(set(protomer_names)) != len(protomer_names):
        raise ValueError(f"{loss_name} included_protomers contains duplicates.")
    return protomer_names


def _parse_residue_codes(residues: str) -> set[str]:
    if not isinstance(residues, str):
        raise ValueError("sequence_residue_fraction residues must be a string.")

    normalized = residues.strip().upper()
    if not normalized:
        raise ValueError(
            "sequence_residue_fraction requires at least one residue code."
        )

    if "," not in normalized and not re.search(r"\s", normalized):
        if len(normalized) != 1:
            raise ValueError(
                "sequence_residue_fraction residues must be a single-character code "
                "or a comma/space-separated list of single-character codes."
            )
        tokens = [normalized]
    else:
        tokens = [token for token in re.split(r"[\s,]+", normalized) if token]

    if any(len(token) != 1 or not token.isalpha() for token in tokens):
        raise ValueError(
            f"sequence_residue_fraction received invalid residue codes: {residues!r}"
        )
    return set(tokens)


def _protein_sequence(protomer_name: str, *, loss_name: str) -> str:
    protomer = global_config.protomers.get(protomer_name)
    if protomer is None:
        raise ValueError(
            f"{loss_name} unknown protomer: {protomer_name!r}"
        )

    if getattr(protomer, "type", None) is not ProtomerTypes.PROTEIN:
        raise ValueError(
            f"{loss_name} requires a protein protomer, got {protomer_name!r}."
        )

    sequence = getattr(protomer, "mutated_sequence", None)
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError(
            f"{loss_name} requires non-empty string mutated_sequence for protomer {protomer_name!r}."
        )

    return sequence.upper()


def _protein_sequence_for_fraction(protomer_name: str) -> str:
    return _protein_sequence(
        protomer_name,
        loss_name="sequence_residue_fraction",
    )


def _parse_region_positions(positions: str, *, loss_name: str) -> tuple[int, ...]:
    if not isinstance(positions, str):
        raise ValueError(f"{loss_name} positions must be a string.")

    raw = positions.strip()
    if not raw:
        raise ValueError(f"{loss_name} requires at least one position.")

    parsed: set[int] = set()
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(
                    f"{loss_name} received malformed position range: {token!r}"
                )
            try:
                lo = int(bounds[0])
                hi = int(bounds[1])
            except ValueError as exc:
                raise ValueError(
                    f"{loss_name} received non-integer position range: {token!r}"
                ) from exc
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError(
                    f"{loss_name} position ranges must be positive and increasing: {token!r}"
                )
            parsed.update(range(lo, hi + 1))
            continue

        try:
            pos = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{loss_name} received non-integer position: {token!r}"
            ) from exc
        if pos <= 0:
            raise ValueError(f"{loss_name} positions must be positive: {token!r}")
        parsed.add(pos)

    if not parsed:
        raise ValueError(f"{loss_name} requires at least one position.")
    return tuple(sorted(parsed))


def _parse_directed_edges(edges: str, *, loss_name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(edges, str):
        raise ValueError(f"{loss_name} edges must be a string.")
    raw = edges.strip()
    if not raw:
        raise ValueError(f"{loss_name} requires at least one directed edge.")

    parsed: list[tuple[str, str]] = []
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        if ">" in token:
            left, right = token.split(">", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        elif "-" in token:
            left, right = token.split("-", 1)
        else:
            raise ValueError(
                f"{loss_name} received malformed directed edge: {token!r}"
            )
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(
                f"{loss_name} received malformed directed edge: {token!r}"
            )
        parsed.append((left, right))

    if not parsed:
        raise ValueError(f"{loss_name} requires at least one directed edge.")
    return tuple(parsed)


@loss_registry.register(required_params=["included_protomers", "residues"])
def sequence_residue_fraction(eval_result, included_protomers: str, residues: str):
    protomer_names = _parse_included_protomers(included_protomers)
    residue_codes = _parse_residue_codes(residues)

    sequences = [
        _protein_sequence_for_fraction(protomer_name)
        for protomer_name in protomer_names
    ]

    total_residues = sum(len(sequence) for sequence in sequences)
    matched_residues = sum(
        residue in residue_codes
        for sequence in sequences
        for residue in sequence
    )
    return matched_residues / total_residues


@loss_registry.register(required_params=["included_protomers", "positions"])
def sequence_region_identity(
    eval_result,
    included_protomers: str,
    positions: str,
):
    loss_name = "sequence_region_identity"
    protomer_names = _parse_included_protomers(
        included_protomers,
        loss_name=loss_name,
        reject_duplicates=True,
    )
    if len(protomer_names) < 2:
        raise ValueError(f"{loss_name} requires at least two included protomers.")

    region_positions = _parse_region_positions(positions, loss_name=loss_name)
    sequences = {
        protomer_name: _protein_sequence(protomer_name, loss_name=loss_name)
        for protomer_name in protomer_names
    }

    lengths = {
        protomer_name: len(sequence)
        for protomer_name, sequence in sequences.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{loss_name} requires equal-length protomers: {lengths}")

    sequence_length = next(iter(lengths.values()))
    out_of_range = [pos for pos in region_positions if pos > sequence_length]
    if out_of_range:
        raise ValueError(
            f"{loss_name} positions outside protomer length {sequence_length}: {out_of_range!r}"
        )

    matches = 0
    total = 0
    for left, right in itertools.combinations(protomer_names, 2):
        left_sequence = sequences[left]
        right_sequence = sequences[right]
        for pos in region_positions:
            matches += left_sequence[pos - 1] == right_sequence[pos - 1]
            total += 1

    return matches / total


@loss_registry.register(required_params=["edges", "m_positions", "n_positions"])
def sequence_edge_region_identity(
    eval_result,
    edges: str,
    m_positions: str,
    n_positions: str,
):
    loss_name = "sequence_edge_region_identity"
    directed_edges = _parse_directed_edges(edges, loss_name=loss_name)
    if len(directed_edges) < 2:
        raise ValueError(f"{loss_name} requires at least two directed edges.")
    if len(set(directed_edges)) != len(directed_edges):
        raise ValueError(f"{loss_name} edges contains duplicates.")

    m_region_positions = _parse_region_positions(m_positions, loss_name=loss_name)
    n_region_positions = _parse_region_positions(n_positions, loss_name=loss_name)
    protomer_names = sorted({name for edge in directed_edges for name in edge})
    sequences = {
        protomer_name: _protein_sequence(protomer_name, loss_name=loss_name)
        for protomer_name in protomer_names
    }
    lengths = {
        protomer_name: len(sequence)
        for protomer_name, sequence in sequences.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{loss_name} requires equal-length protomers: {lengths}")

    sequence_length = next(iter(lengths.values()))
    out_of_range = [
        pos
        for pos in (*m_region_positions, *n_region_positions)
        if pos > sequence_length
    ]
    if out_of_range:
        raise ValueError(
            f"{loss_name} positions outside protomer length {sequence_length}: {out_of_range!r}"
        )

    edge_barcodes = []
    for src, dst in directed_edges:
        src_sequence = sequences[src]
        dst_sequence = sequences[dst]
        edge_barcodes.append(
            "".join(src_sequence[pos - 1] for pos in m_region_positions)
            + "".join(dst_sequence[pos - 1] for pos in n_region_positions)
        )

    matches = 0
    total = 0
    for left, right in itertools.combinations(edge_barcodes, 2):
        for left_residue, right_residue in zip(left, right, strict=True):
            matches += left_residue == right_residue
            total += 1

    return matches / total
