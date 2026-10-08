from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

import numpy as np
import termcolor

from astrolabe.utils.chemicals import one_letter_20

# Create constant sets for sequence validation
VALID_RNA_BASES: Set[str] = {"A", "U", "G", "C"}
VALID_DNA_BASES: Set[str] = {"A", "T", "G", "C"}
# NOTE: allow 'X' as an unknown/masked residue placeholder (common in design workflows).
VALID_PROTEIN_RESIDUES: Set[str] = set(one_letter_20).union({"X"})


def is_numeric_range(stmt: str) -> bool:
    """Check if string represents a numeric range (e.g., '1-10')"""
    parts = stmt.split("-")
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def validate_sequence_type(sequence: str, valid_chars: Set[str]) -> bool:
    """Generic sequence validator using a set of valid characters"""
    return set(sequence).issubset(valid_chars)


def is_valid_protein_sequence(sequence: str) -> bool:
    """Validate if string contains only valid protein residues"""
    return validate_sequence_type(sequence, VALID_PROTEIN_RESIDUES)


def is_valid_ptm_sequence(sequence: str) -> bool:
    """Validate if string is a valid post-translational modification sequence"""
    cleaned_seq = re.sub(r"\[.*?\]", "[]", sequence)
    valid_chars = VALID_PROTEIN_RESIDUES.union({"[", "]"})
    return set(cleaned_seq).issubset(valid_chars)


def is_valid_rna_sequence(sequence: str) -> bool:
    """Validate if string contains only valid RNA bases"""
    return validate_sequence_type(sequence, VALID_RNA_BASES)


def is_valid_dna_sequence(sequence: str) -> bool:
    """Validate if string contains only valid DNA bases"""
    return validate_sequence_type(sequence, VALID_DNA_BASES)


def is_ligand_sequence(sequence: str) -> bool:
    """Check if sequence is a ligand (not protein/RNA/DNA)"""
    return not any(
        [
            is_valid_protein_sequence(sequence),
            is_valid_rna_sequence(sequence),
            is_valid_dna_sequence(sequence),
        ]
    )


def is_range_selection(selector: str) -> bool:
    """Check if string represents a range selection (contains two colons)"""
    return selector.count(":") == 2


def is_chain_selection(selector: str) -> bool:
    """Check if string represents a chain selection (contains one colon)"""
    return selector.count(":") == 1


def apply_sequence_color(
    stmt: str, seqs: dict, colored_seqs: dict, key: str, color: str
) -> None:
    """Apply color coding to a sequence and store both colored and uncolored versions"""
    seqs[key] = stmt
    colored_seqs[key].append(termcolor.colored(stmt, color))


def generate_random_sequence(
    residue_frequencies: Dict[str, float], length_range: str
) -> str:
    """Generate random sequence based on residue frequencies and length range"""
    start, end = map(int, length_range.split("-"))
    sequence_length = np.random.randint(start, end + 1)
    residues = list(residue_frequencies.keys())
    probabilities = list(residue_frequencies.values())

    sequence = "".join(
        np.random.choice(residues, p=probabilities) for _ in range(sequence_length)
    )
    logging.info(f"Generated random sequence of length {sequence_length}: {sequence}")
    return sequence


def parse_hierarchy_string(hierarchy_str: str) -> Dict[str, List[int]]:
    """
    Parse hierarchy string into chain-to-sites mapping
    Format: 'A:1,2,3,4;B:1-5,6,7' -> {'A': [1,2,3,4], 'B': [1,2,3,4,5,6,7]}
    Format: 'A:1,2@3,3,4@3;B:1-5@5,6,7' -> {'A': [1,2,3,4], 'B': [1,2,3,4,5,6,7]}, {"A": [1,3,1,3], "B": [5,5,5,5,5,1,1]}
    """
    from collections import defaultdict

    chain_sites = defaultdict(list)
    chain_weights = defaultdict(list)
    for chain_spec in hierarchy_str.split(";"):
        chain, sites = chain_spec.strip().split(":")
        for site_range in sites.split(","):

            site_range, *weight = site_range.strip().split("@")
            if "-" in site_range:
                start, end = map(int, site_range.split("-"))
                chain_sites[chain].extend(range(start, end + 1))
            elif site_range.isdigit():
                start, end = int(site_range), int(site_range)
                chain_sites[chain].extend(range(start, end + 1))
            else:
                raise ValueError(f"Invalid site specification: {site_range}")

            if weight:
                chain_weights[chain].extend([int(weight[0])] * (end - start + 1))
            else:
                chain_weights[chain].extend([1] * (end - start + 1))

    return dict(chain_sites), chain_weights


def find_ptm_positions(sequence: str, offset: int = 0) -> List[int]:
    """Find positions of post-translational modifications marked with '?'"""
    return [match.start() + offset for match in re.finditer(r"\?", sequence)]
