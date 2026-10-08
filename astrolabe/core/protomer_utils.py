from __future__ import annotations

import functools
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import termcolor
from omegaconf import DictConfig

from astrolabe.core.folding_input import Template
from astrolabe.core.msa_utils import parse_a3m_sequences, apply_position_mask
from astrolabe.core.sequence_parsing import (
    parse_hierarchy_string,
    apply_sequence_color,
    is_chain_selection,
    is_numeric_range,
    generate_random_sequence,
    is_valid_protein_sequence,
    is_range_selection,
    is_valid_ptm_sequence,
    find_ptm_positions,
)
from astrolabe.core.sequence_visualization import (
    log_colored_sequences,
    log_fixed_regions,
)

# Types
SequenceDict = Dict[str, str]
IndexDict = Dict[str, List[int]]
ColoredSeqDict = Dict[str, List[str]]
PTMDict = Dict[str, Dict[int, str]]
ParsedResult = Dict[str, Dict[str, Union[str, List]]]

CHAIN_TYPE_MAP = {"p": "protein", "c": "ligand", "s": "ligand", "r": "RNA", "d": "DNA"}


@functools.cache
def cached_ccd(user_ccd: Optional[str] = None):
    """Get cached chemical component dictionary.

    Args:
        user_ccd: Optional path to custom CCD file

    Returns:
        Cached Ccd instance (requires AF3 python package)
    """
    try:
        from alphafold3.constants import chemical_components  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PTM parsing requires the AlphaFold3 python package (`alphafold3`). "
            "Install the vendored AF3 deps, or remove PTM annotations like `[PTM]` / `(CCD)`."
        ) from exc
    return chemical_components.Ccd(user_ccd=user_ccd)


def parse_protomer_statements(
    seq_stmt: Dict[str, str],
    parsed_seq: ParsedResult,
    aa_freq: Dict[str, float],
    types: Dict[str, str],
) -> Tuple[
    SequenceDict, IndexDict, IndexDict, PTMDict, Dict[str, List[Tuple[slice, bool]]]
]:
    """Parse protomer configuration and sequence statements.

    Args:
        seq_stmt: Sequence statements for each chain
        parsed_seq: Parsed structure data
        aa_freq: Amino acid frequencies
        types: Chain types (protein, RNA, etc)

    Returns:
        Tuple containing:
        - Parsed sequences
        - Fixed position indices
        - Reference position indices
        - Post-translational modification indices
        - Raw input indices
    """
    parsed_sequences = {key: "" for key in seq_stmt.keys()}
    colored_seqs = {key: [] for key in seq_stmt.keys()}
    fixed_residue_indices = {key: [] for key in seq_stmt.keys() if types[key] == "p"}
    reference_indices = {key: [] for key in seq_stmt.keys() if types[key] == "p"}
    ptm_positions = {key: {} for key in seq_stmt.keys() if types[key] == "p"}
    raw_input_indices = defaultdict(list)

    # Dispatch table for sequence processing
    type_processors = {
        "p": lambda key, statement: process_protein_statement(
            parsed_seq,
            aa_freq,
            parsed_sequences,
            colored_seqs,
            fixed_residue_indices,
            key,
            statement,
            reference_indices,
            ptm_positions,
            raw_input_indices,
        ),
        "s": lambda key, stmt: process_ligand_smiles(
            parsed_seq, stmt, parsed_sequences, colored_seqs, key
        ),
        "c": lambda key, stmt: process_ligand_ccds(
            parsed_seq, stmt, parsed_sequences, colored_seqs, key
        ),
        "r": lambda key, stmt: apply_sequence_color(
            stmt, parsed_sequences, colored_seqs, key, "magenta"
        ),
        "d": lambda key, stmt: apply_sequence_color(
            stmt, parsed_sequences, colored_seqs, key, "cyan"
        ),
    }

    # Process each sequence statement
    for key, stmt in seq_stmt.items():
        if key_type := types.get(key):
            if processor := type_processors.get(key_type):
                processor(key, stmt)

    # Display sequences and colored output
    log_fixed_regions(fixed_residue_indices, parsed_sequences)
    log_colored_sequences(colored_seqs)

    # Sort and deduplicate fix indices
    fixed_residue_indices = {
        k: sorted(set(v)) for k, v in fixed_residue_indices.items()
    }

    return (
        parsed_sequences,
        fixed_residue_indices,
        reference_indices,
        ptm_positions,
        raw_input_indices,
    )


def get_sequence_by_position(rslt, stmt):
    sign, chain, rge = stmt.split(":")
    validate_pdb_key(rslt, sign)
    start, end = map(int, rge.split("-"))
    if "pdb_idx" in rslt[sign]:
        seq = "".join(
            [
                rslt[sign]["seq_str"][i]
                for i, val in enumerate(rslt[sign]["pdb_idx"])
                if (val[0] == chain or not chain) and start <= val[1] <= end
            ]
        )
    else:
        seq = "".join([rslt[sign][chain]["seq_str"][i] for i in range(start - 1, end)])

    if not seq:
        raise ValueError(f"Residue not found in {sign}:{chain}:{rge}")
    if rslt[sign][chain]["type"] != "p":
        raise ValueError(
            f"Invalid sequence type for {sign}:{chain}:{rge}, expected protein"
        )
    logging.info(f"{sign}:{chain}:{rge} => {seq}")
    return seq


def get_sequence_by_chain(rslt, stmt):
    sign, chain = stmt.split(":")
    validate_pdb_key(rslt, sign)
    if "pdb_idx" in rslt[sign]:
        seq = "".join(
            [
                rslt[sign]["seq_str"][i]
                for i, val in enumerate(rslt[sign]["pdb_idx"])
                if val[0] == chain
            ]
        )
    else:
        seq = rslt[sign][chain]["seq_str"]
    logging.info(f"{sign}:{chain} => {seq}")
    return seq


def validate_pdb_key(rslt, sign):
    if sign not in rslt:
        raise ValueError(f"{sign} not found, available keys are {rslt.keys()}")


def append_sequence(seqs: dict, colored_seqs: dict, key, seq, color, hal_idx):
    seqs[key] += seq
    colored_seqs[key].append(termcolor.colored(seq, color))
    return hal_idx + len(seq)


def load_msa_for_protomer(
    extra_msa_cfg: Optional[DictConfig],
    sequences: SequenceDict,
    fix_res_list: IndexDict,
) -> Dict[str, str]:
    """Load and process MSA files for each protomer.

    Args:
        extra_msa_cfg: MSA configuration
        sequences: Protomer sequences
        fix_res_list: Fixed residue indices

    Returns:
        Dict mapping protomer keys to processed MSA strings

    Raises:
        ValueError: If MSA file missing or invalid
    """
    result = {}

    if extra_msa_cfg is None:
        return result

    for key, msa_paths in extra_msa_cfg.items():
        if key not in sequences:
            continue

        current_seq = sequences[key]
        fix_res = np.array(fix_res_list[key])

        raw_templates = msa_paths.get("templates")
        templates = extract_template_fixpos(fix_res, raw_templates)

        unpairedMsa = mask_msa_fixpos(
            current_seq, fix_res, msa_paths.get("unpairedMsa"), remove_mainly_gaps=True
        )
        pairedMsa = mask_msa_fixpos(current_seq, fix_res, msa_paths.get("pairedMsa"))
        result[key] = {
            "unpairedMsa": unpairedMsa,
            "pairedMsa": pairedMsa,
            "templates": templates,
        }

    return result


def mask_msa_fixpos(current_seq, fix_res, msa_string, remove_mainly_gaps=False):
    if msa_string is None:
        return None
    msa = parse_a3m_sequences(msa_string)
    headers, msa_seqs = zip(*msa)
    if len(current_seq) != len(msa_seqs[0]):
        raise ValueError(
            f"Sequence length mismatch: expected {len(current_seq)}, got {len(msa_seqs[0])}"
        )
    seqs = apply_position_mask(msa_seqs, fix_res)
    seqs[0] = current_seq
    if remove_mainly_gaps:
        seqs = [seq for seq in seqs if seq.count("-") / len(current_seq) < 0.7]
    masked_msa = "\n".join([f"{header}\n{seq}" for header, seq in zip(headers, seqs)])
    return masked_msa


def extract_template_fixpos(fix_res, raw_templates):
    if raw_templates is None:
        templates = None
    else:
        templates = []
        for raw_template in raw_templates:
            mmcif = raw_template.get("mmcif", None)
            query_to_template_map = dict(
                zip(
                    raw_template["queryIndices"],
                    raw_template["templateIndices"],
                )
            )
            query_to_template_map = {
                k: v for k, v in query_to_template_map.items() if k in fix_res
            }
            templates.append(
                Template(mmcif=mmcif, query_to_template_map=query_to_template_map)
            )
    return templates


def convert_dicrete_index_to_range(index_list):
    index_list = sorted(index_list)
    ranges = []
    start = index_list[0]
    end = index_list[0]
    for i in range(1, len(index_list)):
        if index_list[i] == end + 1:
            end = index_list[i]
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = index_list[i]
            end = index_list[i]
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def _validate_keys(actual, expected):
    """Validates that the JSON doesn't contain any extra unwanted keys."""
    if bad_keys := set(actual) - set(expected):
        raise ValueError(f'Unexpected JSON keys in: {", ".join(sorted(bad_keys))}')


def _parse_hotspot_positions(hotspot_positions_config, random_fragment_index):
    """Parse hotspot positions from configuration.

    Args:
        hotspot_positions_config: Configuration for hotspot positions, can be None, DictConfig, or string.
        random_fragment_index: Boolean indicating if random fragment index is used.

    Returns:
        Parsed hotspot positions as a dictionary or None if not applicable.

    Raises:
        ValueError: If the configuration type is invalid.
    """
    # todo implement hotspot weight during config parsing
    if not hotspot_positions_config:
        return None, None

    if isinstance(hotspot_positions_config, DictConfig):
        hotspots = defaultdict(list)
        weights = defaultdict(list)
        for chain, chain_spec in hotspot_positions_config.items():
            for site_range in str(chain_spec).split(","):
                site_range, *weight = site_range.strip().split("@")
                if "-" in site_range:
                    start, end = map(int, site_range.split("-"))
                    hotspots[chain].extend(range(start, end + 1))
                elif site_range.isdigit():
                    start, end = int(site_range), int(site_range)
                    hotspots[chain].extend(range(start, end + 1))
                else:
                    raise ValueError(f"Invalid site specification: {site_range}")

                if weight:
                    weights[chain].extend([int(weight[0])] * (end - start + 1))
                else:
                    weights[chain].extend([1] * (end - start + 1))

    elif isinstance(hotspot_positions_config, str):
        hotspots, weights = parse_hierarchy_string(hotspot_positions_config)
    else:
        raise ValueError(
            f"Invalid hotspot_positions configuration type: {type(hotspot_positions_config)}, should be either "
            "DictConfig or string."
        )

    return {k: np.array([i - 1 for i in v]) for k, v in hotspots.items()}, weights


def process_ligand_smiles(rslt, stmt, seqs, colored_seqs, key):
    """
    Process ligand-related sequences.
    """
    if ":" not in stmt:
        seqs[key] = stmt
    elif stmt.split(":")[0] in rslt and is_chain_selection(stmt):
        file, chain = stmt.split(":")
        seqs[key] = rslt[file][chain]["seq_str"]
        if rslt[file][chain]["type"] != "s":
            raise ValueError(
                f"Invalid sequence type for {stmt}, expected `smiles` in {file}"
            )
    else:
        raise ValueError(f"Invalid ligand sequence {stmt}")
    colored_seqs[key].append(termcolor.colored(seqs[key], "blue"))


def process_ligand_ccds(rslt, stmt, seqs, colored_seqs, key):
    """
    Process ligand-related sequences.
    """
    if ":" not in stmt:
        seqs[key] = stmt
    elif stmt.split(":")[0] in rslt and is_chain_selection(stmt):
        file, chain = stmt.split(":")
        seqs[key] = rslt[file][chain]["seq_str"]
        if rslt[file][chain]["type"] != "c":
            raise ValueError(
                f"Invalid sequence type for {stmt}, expected `ccdcodes` in {file}"
            )
    else:
        raise ValueError(f"Invalid ligand sequence {stmt}")
    colored_seqs[key].append(termcolor.colored(seqs[key], "blue"))


def process_protein_fragment(
    fragment: str,
    aa_frequencies: Dict[str, float],
    parsed_data: ParsedResult,
    current_index: int,
) -> tuple[str, str, Optional[List[int]]]:
    """Process a protein sequence fragment and determine its color"""
    is_fixed = fragment.startswith("+")
    fragment = fragment[1:] if is_fixed else fragment

    if is_numeric_range(fragment):
        return generate_random_sequence(aa_frequencies, fragment), "green", None
    elif is_valid_protein_sequence(fragment):
        return fragment, "red", None
    elif is_range_selection(fragment):
        return get_sequence_by_position(parsed_data, fragment), "yellow", None
    elif is_chain_selection(fragment):
        return get_sequence_by_chain(parsed_data, fragment), "yellow", None
    elif is_valid_ptm_sequence(fragment):
        # Extract base sequence by removing PTM markers
        base_sequence = re.sub(r"\[.*?]", "?", fragment)
        # Find PTM positions
        ptm_pos = find_ptm_positions(base_sequence, current_index)
        # Extract PTM types
        ptm_types = re.findall(r"\[(.*?)]", fragment)
        ptm = {pos - current_index: ptm for pos, ptm in zip(ptm_pos, ptm_types)}
        recover_sequence = "".join(
            [
                (
                    cached_ccd()[ptm[idx]]["_chem_comp.one_letter_code"][0]
                    if idx in ptm
                    else aa
                )
                for idx, aa in enumerate(base_sequence)
            ]
        )
        return recover_sequence, "blue", list(zip(ptm_pos, ptm_types))
    else:
        raise ValueError(f"Invalid protein fragment: {fragment}")


def process_protein_statement(
    parsed_data: ParsedResult,
    aa_frequencies: Dict[str, float],
    sequences: SequenceDict,
    colored_sequences: ColoredSeqDict,
    fixed_indices: IndexDict,
    key: str,
    statement: str,
    reference_indices: IndexDict,
    ptm_indices: PTMDict,
    raw_indices: Dict[str, List[Tuple[slice, bool]]],
) -> None:
    """Process protein sequence statement with improved organization"""
    current_index = 0
    fragments = [f for f in statement.strip().split() if f]

    for fragment in fragments:
        is_fixed = fragment.startswith("+")
        sequence, color, ptms = process_protein_fragment(
            fragment, aa_frequencies, parsed_data, current_index
        )

        fragment_length = len(sequence)
        fragment_range = range(current_index, current_index + fragment_length)

        # Update sequences and indices
        sequences[key] += sequence
        colored_sequences[key].append(termcolor.colored(sequence, color))

        if is_fixed:
            fixed_indices[key].extend(fragment_range)
        if not is_numeric_range(fragment):
            reference_indices[key].extend(fragment_range)
        if ptms:
            for pos, ptm_type in ptms:
                ptm_indices[key][pos] = ptm_type

        raw_indices[key].append(
            (slice(current_index, current_index + fragment_length), is_fixed)
        )
        current_index += fragment_length
