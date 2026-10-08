from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class StructureParser:
    """Base class for structure file parsers."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.chains = defaultdict(lambda: ([], ""))

    def _open_file(self, file_path: Path):
        """Open file with automatic compression detection."""
        if file_path.suffix.lower() in [".gz", ".gzip"]:
            return gzip.open(file_path, "rt", encoding="utf-8")
        else:
            return open(file_path, "r", encoding="utf-8")

    def parse(self) -> Dict[str, Tuple[List[int], str]]:
        """Parse the structure file and return chain data."""
        raise NotImplementedError

    def get_chain_sequence(self, chain_id: str) -> Tuple[List[int], str]:
        """Get sequence and indices for a specific chain."""
        return self.chains.get(chain_id, ([], ""))


class PDBParser(StructureParser):
    """Parser for PDB format files."""

    # Standard amino acid mapping
    AA_3TO1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "SEC": "U",
        "PYL": "O",  # Non-standard amino acids
    }

    # Standard nucleotide mapping
    NUCLEOTIDE_MAP = {
        "A": "A",
        "T": "T",
        "G": "G",
        "C": "C",
        "U": "U",
        "DA": "A",
        "DT": "T",
        "DG": "G",
        "DC": "C",
    }

    def parse(self) -> Dict[str, Tuple[List[int], str]]:
        """Parse PDB file and extract sequences with residue indices."""
        chain_residues = defaultdict(
            dict
        )  # chain_id -> {res_num: (res_name, insertion_code)}

        with self._open_file(self.file_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = "HETATM" if line.startswith("HETATM") else "ATOM"
                    chain_id = line[21:22].strip()
                    if not chain_id:
                        chain_id = "A"  # Default chain

                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    insertion_code = line[26:27].strip()

                    # Create unique key for residue position
                    res_key = (res_num, insertion_code)

                    if (
                        record_type == "HETATM"
                        and res_name not in self.AA_3TO1
                        and res_name not in self.NUCLEOTIDE_MAP
                    ):
                        continue

                    # Only keep the first occurrence of each residue
                    if res_key not in chain_residues[chain_id]:
                        chain_residues[chain_id][res_key] = res_name

        # Convert to sequences
        for chain_id, residues in chain_residues.items():
            indices = []
            sequence = []

            # Sort by residue number, then by insertion code
            sorted_residues = sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))

            for (res_num, insertion_code), res_name in sorted_residues:
                indices.append(res_num)

                # Convert to single letter code
                if res_name in self.AA_3TO1:
                    sequence.append(self.AA_3TO1[res_name])
                elif res_name in self.NUCLEOTIDE_MAP:
                    sequence.append(self.NUCLEOTIDE_MAP[res_name])
                else:
                    sequence.append("X")  # Unknown residue

            self.chains[chain_id] = (indices, "".join(sequence))

        return dict(self.chains)


class CIFParser(StructureParser):
    """Parser for mmCIF format files."""

    # Reuse mappings from PDB parser
    AA_3TO1 = PDBParser.AA_3TO1
    NUCLEOTIDE_MAP = PDBParser.NUCLEOTIDE_MAP

    def parse(self) -> Dict[str, Tuple[List[int], str]]:
        """Parse mmCIF file and extract sequences with residue indices."""
        chain_residues = defaultdict(dict)
        in_atom_site = False
        headers = []

        with self._open_file(self.file_path) as f:
            for line in f:
                line = line.strip()

                if line.startswith("_atom_site."):
                    headers.append(line[11:])  # Remove '_atom_site.' prefix
                    in_atom_site = True
                elif line.startswith("#") or line.startswith("_") or not line:
                    if in_atom_site and headers:
                        break
                    continue
                elif in_atom_site and headers:
                    # Parse data line
                    values = self._parse_cif_line(line)
                    if len(values) >= len(headers):
                        data = dict(zip(headers, values))

                        # Extract relevant fields
                        group_pdb = data.get("group_PDB", "ATOM")
                        chain_id = data.get(
                            "auth_asym_id", data.get("label_asym_id", "A")
                        )
                        res_name = data.get(
                            "auth_comp_id", data.get("label_comp_id", "")
                        )
                        res_num_str = data.get(
                            "auth_seq_id", data.get("label_seq_id", "0")
                        )

                        try:
                            res_num = int(res_num_str)
                        except (ValueError, TypeError):
                            continue

                        insertion_code = data.get("pdbx_PDB_ins_code", "")
                        if insertion_code == "?":
                            insertion_code = ""

                        # Create unique key for residue position
                        res_key = (res_num, insertion_code)

                        if (
                            group_pdb == "HETATM"
                            and res_name not in self.AA_3TO1
                            and res_name not in self.NUCLEOTIDE_MAP
                        ):
                            continue

                        # Only keep the first occurrence of each residue
                        if res_key not in chain_residues[chain_id]:
                            chain_residues[chain_id][res_key] = res_name

        # Convert to sequences (same logic as PDB parser)
        for chain_id, residues in chain_residues.items():
            indices = []
            sequence = []

            sorted_residues = sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))

            for (res_num, insertion_code), res_name in sorted_residues:
                indices.append(res_num)

                if res_name in self.AA_3TO1:
                    sequence.append(self.AA_3TO1[res_name])
                elif res_name in self.NUCLEOTIDE_MAP:
                    sequence.append(self.NUCLEOTIDE_MAP[res_name])
                else:
                    sequence.append("X")

            self.chains[chain_id] = (indices, "".join(sequence))

        return dict(self.chains)

    def _parse_cif_line(self, line: str) -> List[str]:
        """Parse a CIF data line, handling quoted strings."""
        values = []
        in_quotes = False
        current_value = ""
        quote_char = None

        i = 0
        while i < len(line):
            char = line[i]

            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                elif char.isspace():
                    if current_value:
                        values.append(current_value)
                        current_value = ""
                else:
                    current_value += char
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_value += char

            i += 1

        if current_value:
            values.append(current_value)

        return values


def create_structure_parser(file_path: str) -> StructureParser:
    """Factory function to create appropriate parser based on file extension."""
    path = Path(file_path)

    # Handle compressed files
    if path.suffix.lower() in [".gz", ".gzip"]:
        # Get the extension before .gz
        actual_extension = path.stem.split(".")[-1].lower() if "." in path.stem else ""
    else:
        actual_extension = path.suffix.lower().lstrip(".")

    if actual_extension in ["pdb", "ent"]:
        return PDBParser(file_path)
    elif actual_extension in ["cif", "mmcif"]:
        return CIFParser(file_path)
    else:
        raise ValueError(f"Unsupported file format: {actual_extension}")
