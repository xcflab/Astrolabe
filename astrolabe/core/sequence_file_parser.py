from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict

from astrolabe.core.protomer_utils import ParsedResult
from astrolabe.utils.file_parser import (
    read_fasta,
    parse_mmcif,
    parse_pdb_zip,
    parse_pdb,
)


# ParsedResult = Dict[str, Dict[str, Union[str, List]]]


class FileFormat(Enum):
    FASTA = "fasta"
    FA = "fa"
    CIF = "cif"
    PDB = "pdb"


def _parse_structure_file(file_path: str, is_compressed: bool) -> Dict:
    """Parse structure files (PDB/CIF) and return parsed data."""
    if "cif" in file_path:
        return parse_mmcif(file_path, parse_hetatom=True)
    elif "pdb" in file_path:
        return (
            parse_pdb_zip(file_path, parse_hetatom=True)
            if is_compressed
            else parse_pdb(file_path, parse_hetatom=True)
        )
    raise ValueError(f"Unsupported structure file format: {file_path}")


def _extract_chain_sequences(parsed_data: Dict) -> Dict[str, Dict[str, str]]:
    """Extract chain sequences from parsed structure data."""
    chain_sequences = defaultdict(list)
    for (chain, _), sequence in zip(parsed_data["pdb_idx"], parsed_data["seq_str"]):
        chain_sequences[chain].append(sequence)

    return {
        chain: {"seq_str": "".join(sequences), "type": "p"}
        for chain, sequences in chain_sequences.items()
    }


def parse_sequences_from_files(file_paths: Dict[str, str]) -> ParsedResult:
    """Parse sequences from structure files."""
    results = {key: None for key in file_paths}

    for key, file_path in file_paths.items():
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        file_extension = Path(file_path).suffix.lower().lstrip(".")
        is_compressed = file_extension == "gz"

        if is_compressed:
            file_extension = Path(file_path).stem.split(".")[-1]

        if file_extension in [FileFormat.FASTA.value, FileFormat.FA.value]:
            results[key] = read_fasta(file_path)
        elif file_extension in [FileFormat.CIF.value, FileFormat.PDB.value]:
            parsed_data = _parse_structure_file(file_path, is_compressed)
            results[key] = _extract_chain_sequences(parsed_data)

        logging.info(f"{key} => {file_path} => {time.time() - start_time:.2f} sec")

    return results


def write_sequences_to_fasta(protomer, step: int, path: Path, oligomer_names=None):
    fasta_str = ""
    proto2fasta = {}

    for proto, sequence in protomer.sequences.items():
        proto_type = protomer.full_types[proto]
        ptms = protomer.ptm_indices.get(proto, {})

        new_sequence = ""
        for site in range(len(sequence)):
            if site not in ptms:
                new_sequence += sequence[site]
            else:
                new_sequence += f"({ptms[site]})"
        proto_fasta = f">{proto_type}|name=proto_{proto}\n{new_sequence}\n"
        proto2fasta[proto] = proto_fasta
        fasta_str += proto_fasta

    with open(path / f"step_{step:03d}_protomers.fasta", "w") as f:
        f.write(fasta_str)

    if oligomer_names is not None:
        for oligomer_name in oligomer_names:
            with open(path / f"step_{step:03d}_{oligomer_name}_input.fasta", "w") as f:
                f.write("".join([proto2fasta[proto] for proto in oligomer_name]))
