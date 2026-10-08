from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple, Dict, List

import numpy as np
from omegaconf import DictConfig

from astrolabe.core import global_config
from astrolabe.core.msa_utils import parse_homology_arg, mask_msa_fixpos
from astrolabe.core.nonstandard_residues import resolve_nonstandard_alias
from astrolabe.core.structure_parser import create_structure_parser
from astrolabe.utils.constant import aa_freq, aa_order


def cached_ccd(*args, **kwargs):
    """
    Return AlphaFold3 chemical component dictionary used for PTM parsing.

    NOTE: Protenix-only runs should not require AF3 python deps. Keep this import lazy
    so users can run non-AF3 evaluators without installing/building `alphafold3`.
    """
    try:
        from alphafold3.constants import chemical_components  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PTM parsing requires the AlphaFold3 python package (`alphafold3`). "
            "Install the vendored AF3 deps (recommended: `gh doctor --strict`), "
            "or remove PTM annotations like `[PTM]` / `(CCD)` from sequences."
        ) from exc
    return chemical_components.Ccd(*args, **kwargs)


class ProtomerTypes(Enum):
    PROTEIN = "p"
    RNA = "r"
    DNA = "d"
    SMILES = "s"
    CCD = "c"

    @classmethod
    def from_alias(cls, alias: str):
        """Map aliases to the corresponding ProtomerType."""
        alias_map = {
            "p": cls.PROTEIN,
            "prot": cls.PROTEIN,
            "protein": cls.PROTEIN,
            "r": cls.RNA,
            "rna": cls.RNA,
            "Rna": cls.RNA,
            "RNA": cls.RNA,
            "d": cls.DNA,
            "dna": cls.DNA,
            "s": cls.SMILES,
            "smiles": cls.SMILES,
            "c": cls.CCD,
            "ccd": cls.CCD,
        }
        normalized_alias = alias.lower()
        if normalized_alias not in alias_map:
            raise ValueError(f"Invalid alias for ProtomerType: {alias}")
        return alias_map[normalized_alias]


StatementType = Literal["random", "literal", "predefined", "from_file"]


class StatementHandlerBase:
    registry = {}

    @classmethod
    def register(cls, name: ProtomerTypes, type: StatementType):
        def decorator(handler_cls):
            cls.registry[(name, type)] = handler_cls()
            return handler_cls

        return decorator

    def handle(self, config):
        raise NotImplementedError


@StatementHandlerBase.register(ProtomerTypes.SMILES, "literal")
class SmilesLiteralHandler(StatementHandlerBase):
    def handle(self, config: str):
        return config.strip(), {}


@StatementHandlerBase.register(ProtomerTypes.SMILES, "from_file")
class SmilesFileHandler(StatementHandlerBase):
    def handle(self, config: str):
        """Handle SMILES sequence from a file."""
        if not config:
            raise ValueError("SMILES sequence file path cannot be empty.")
        file_alias, statement = config.split(":") if ":" in config else (config, None)
        if file_alias not in global_config.named_files:
            raise ValueError(f"File alias '{file_alias}' not found in named files.")

        parsed_file: "ParsedFile" = global_config.named_files[file_alias]

        if statement not in parsed_file or not parsed_file[statement]:
            raise ValueError(
                f"Statement '{statement}' not found in file '{file_alias}'."
            )
        chain_id, *indices = list(statement)


@StatementHandlerBase.register(ProtomerTypes.CCD, "literal")
class CcdLiteralHandler(StatementHandlerBase):
    def handle(self, config: str):
        """Handle literal CCD sequence."""
        if not config:
            raise ValueError("CCD sequence cannot be empty.")

        # todo: Implement CCD validation
        # For now, we assume the input is valid
        return config.strip(), None


@StatementHandlerBase.register(ProtomerTypes.PROTEIN, "random")
class ProteinRandomHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle random protein sequence generation."""
        if not set(config).issubset(set("0123456789-")):
            raise ValueError("Invalid random protein sequence format.")

        start, end = map(int, config.split("-"))
        if start < 0 or end < start:
            raise ValueError("Invalid range for random protein sequence.")
        length = np.random.randint(start, end + 1)
        amino_acids_sampler = ResidueSampler(
            frequencies=aa_freq, exclude=global_config.exclude_AA
        )
        sequence = "".join(amino_acids_sampler.sample_n(length))
        return sequence, {}


@StatementHandlerBase.register(ProtomerTypes.PROTEIN, "literal")
class ProteinLiteralHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle literal protein sequence, including PTMs."""
        if not config:
            raise ValueError("Protein sequence cannot be empty.")

        sequence, ptm_dict = parse_ptm_sequence(config)
        return sequence, ptm_dict


class ParsedFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._parse_file(file_path)

    def _parse_file(self, file_path: str) -> Dict[str, Tuple[List[int], str]]:
        """Parse file based on its format."""
        path = Path(file_path)

        # Handle compressed files
        if path.suffix.lower() in ["gz", ".gzip"]:
            actual_extension = (
                path.stem.split(".")[-1].lower() if "." in path.stem else ""
            )
        else:
            actual_extension = path.suffix.lower().lstrip(".")

        # Check if it's a structure file
        if actual_extension in ["pdb", "ent", "cif", "mmcif"]:
            try:
                parser = create_structure_parser(file_path)
                return parser.parse()
            except Exception as e:
                raise ValueError(
                    f"Failed to parse structure file {file_path}: {str(e)}"
                )
        else:
            # Handle other file formats (existing logic)
            return self._parse_other_formats(file_path)

    def _parse_other_formats(self, file_path: str) -> Dict[str, Tuple[List[int], str]]:
        """Parse non-structure file formats."""
        # Placeholder for other file format parsing
        # This would include FASTA, text files, etc.
        return {}

    def __getitem__(self, item) -> Tuple[List[int], str]:
        return self.data.get(item, ([], ""))

    def __contains__(self, item) -> bool:
        return item in self.data


@StatementHandlerBase.register(ProtomerTypes.PROTEIN, "from_file")
class ProteinFileHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle protein sequence from a file."""
        if not config:
            raise ValueError("Protein sequence file path cannot be empty.")
        file_alias, statement = config.split(":", 1) if ":" in config else (config, None)
        if file_alias not in global_config.named_files:
            raise ValueError(f"File alias '{file_alias}' not found in named files.")

        parsed_file: "ParsedFile" = global_config.named_files[file_alias]

        chain_indices: Dict[str, List[int]] = parse_ranges(statement)
        if not chain_indices:
            raise ValueError(
                f"No valid chain indices found in statement '{statement}'."
            )
        sequences = []
        for chain_id, indices in chain_indices.items():
            pdb_indices, pdb_sequence = parsed_file[chain_id]
            if not indices:
                sequences.append(pdb_sequence)
            else:
                out_of_range = any(index not in pdb_indices for index in indices)
                if out_of_range:
                    raise ValueError(
                        f"Indices {indices} out of range for chain {chain_id} in file '{file_alias}'."
                    )
                sequences.append(
                    "".join(pdb_sequence[pdb_indices.index(i)] for i in indices)
                )
        sequence = "".join(sequences)
        return sequence, {}


@StatementHandlerBase.register(ProtomerTypes.PROTEIN, "predefined")
class ProteinPredefinedHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle predefined protein sequence."""
        if not config:
            raise ValueError("Predefined protein sequence cannot be empty.")

        if config not in global_config.predefined_statement:
            raise ValueError(f"Predefined statement '{config}' not found.")

        sequence, ptm_dict = global_config.predefined_statement[config]
        return sequence, ptm_dict


@StatementHandlerBase.register(ProtomerTypes.RNA, "literal")
class RnaLiteralHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle literal RNA sequence."""
        if not config:
            raise ValueError("RNA sequence cannot be empty.")

        sequence, ptm_dict = parse_ptm_sequence(config)

        if not all(base in "ACGU?" for base in sequence):
            raise ValueError("Invalid RNA sequence. Only A, C, G, U are allowed.")

        return sequence, ptm_dict


@StatementHandlerBase.register(ProtomerTypes.RNA, "random")
class RnaRandomHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle random RNA sequence generation."""
        if not set(config).issubset(set("0123456789-")):
            raise ValueError("Invalid random RNA sequence format.")

        start, end = map(int, config.split("-"))
        if start < 0 or end < start:
            raise ValueError("Invalid range for random RNA sequence.")
        length = np.random.randint(start, end + 1)
        bases = "ACGU"
        sequence = "".join(np.random.choice(list(bases), size=length))
        return sequence, {}


@StatementHandlerBase.register(ProtomerTypes.DNA, "literal")
class DnaLiteralHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle literal DNA sequence."""
        if not config:
            raise ValueError("DNA sequence cannot be empty.")

        sequence, ptm_dict = parse_ptm_sequence(config)

        if not all(base in "ACGT?" for base in sequence):
            raise ValueError("Invalid DNA sequence. Only A, C, G, T are allowed.")

        return sequence, ptm_dict


@StatementHandlerBase.register(ProtomerTypes.DNA, "random")
class DnaRandomHandler(StatementHandlerBase):
    def handle(self, config: str) -> Tuple[str, Dict[int, str]]:
        """Handle random DNA sequence generation."""
        if not set(config).issubset(set("0123456789-")):
            raise ValueError("Invalid random DNA sequence format.")

        start, end = map(int, config.split("-"))
        if start < 0 or end < start:
            raise ValueError("Invalid range for random DNA sequence.")
        length = np.random.randint(start, end + 1)
        bases = "ACGT"
        sequence = "".join(np.random.choice(list(bases), size=length))
        return sequence, {}


def get_aa_order(type: ProtomerTypes):
    if type == ProtomerTypes.PROTEIN:
        return np.array(aa_order[:20])
    elif type == ProtomerTypes.RNA:
        return np.array(["A", "C", "G", "U"])
    elif type == ProtomerTypes.DNA:
        return np.array(["A", "C", "G", "T"])
    return


@dataclass
class Segment:
    sequence: str
    ptm_dict: Dict[int, str] = None
    is_fixed: bool = False
    type: ProtomerTypes = ProtomerTypes.PROTEIN
    initialized_from_random: bool = False
    probs: List[np.ndarray] | None = None
    scores: List[float] | None = None
    mutations: List[Tuple[int, str, str, float, float]] = None

    @property
    def aa_order(self) -> np.ndarray:
        """Get amino acid order for the segment type."""
        return get_aa_order(self.type)

    def clean_up(self):
        self.probs = None
        self.scores = None

    def add_probs(self, probs: List[np.ndarray]):
        """Add probabilities to the segment."""
        if self.probs is None:
            self.probs = []
        self.probs.append(probs)

    def add_scores(self, scores: List[float]):
        """Add scores to the segment."""
        if self.scores is None:
            self.scores = []

        self.scores.append(scores)

    def sample_mutations(self) -> Tuple[int, str, str, float, float]:
        if (
            self.probs is None
            or len(self.probs) == 0
            or self.type not in MutableTypes
            or self.is_fixed
        ):
            return []

        average_probs = np.mean(self.probs, axis=0)

        # normalize probabilities (avoid divide-by-zero RuntimeWarning for fully masked rows)
        row_sums = np.sum(average_probs, axis=1, keepdims=True)
        average_probs = np.divide(
            average_probs,
            row_sums,
            out=np.zeros_like(average_probs),
            where=row_sums > 0,
        )
        average_probs[np.isnan(average_probs)] = 0.0

        if self.scores is None:
            average_scores = np.zeros(len(self.sequence))
        else:
            average_scores = np.mean(self.scores, axis=0)

        entropy = -np.sum(average_probs * np.log2(average_probs + 1e-10), axis=1)
        entropy = np.where(np.isclose(entropy, 0), 0.0, entropy)

        sorted_indices = np.argsort(entropy)

        result = []

        for idx in sorted_indices:
            if idx >= len(self.sequence) or (
                self.ptm_dict is not None and idx in self.ptm_dict
            ):
                continue
            from_base = self.sequence[idx]
            if from_base == "?":
                continue

            position_probs = average_probs[idx]
            if np.isclose(np.sum(position_probs), 0):
                continue
            if np.isclose(entropy[idx], 0):
                to_base = self.aa_order[int(np.argmax(position_probs))]
            else:
                to_base = np.random.choice(self.aa_order, p=position_probs)
            entropy_value = entropy[idx]
            score = average_scores[idx]
            if to_base == from_base:
                continue
            result.append((idx, from_base, to_base, entropy_value, score))
        return result

    def add_mutations(self, mutation) -> bool:
        if self.mutations is None:
            self.mutations = []
        sites = list(zip(*self.mutations))[0] if self.mutations else []
        if mutation[0] not in sites:
            self.mutations.append(mutation)
            is_existed = False
        else:
            is_existed = True
        return is_existed

    def apply_mutations(self) -> None:
        """Assign mutations to the segment."""
        if self.mutations is None or not self.mutations:
            return

        for mutation in self.mutations:
            site, from_base, to_base, _, _ = mutation
            if site in self.ptm_dict:
                continue
            assert (
                self.sequence[site] == from_base
            ), f"Inconsistent mutation at site {site}: expected {from_base}, got {self.sequence[site]}"
            self.sequence = self.sequence[:site] + to_base + self.sequence[site + 1 :]

    def clear_mutations(self):
        """Clear mutations for the segment."""
        self.mutations = []

    @property
    def mutated_sequence(self) -> str:
        """Get the sequence after applying mutations."""
        if self.mutations is None or not self.mutations:
            return self.sequence
        mutated_sequence = list(self.sequence)
        for mutation in self.mutations:
            site, from_base, to_base, _, _ = mutation
            if site in self.ptm_dict:
                continue
            mutated_sequence[site] = to_base
        return "".join(mutated_sequence)


def segment_handler_distributer(type: ProtomerTypes, segment_definition) -> Segment:
    is_fixed, segment_definition = (
        segment_definition.startswith("+")
        and (True, segment_definition[1:])
        or (False, segment_definition)
    )
    if type not in MutableTypes:
        is_fixed = True

    if (
        (alias := clean_statement(segment_definition))
        and alias in global_config.segments
        and type == global_config.segments[alias].type
    ):
        return global_config.segments[alias]

    elif set(segment_definition).issubset(set("0123456789-+")) and type in MutableTypes:
        sequence, ptm_dict = StatementHandlerBase.registry.get((type, "random")).handle(
            segment_definition
        )
        return Segment(
            sequence,
            ptm_dict,
            is_fixed,
            type,
            initialized_from_random=(type == ProtomerTypes.PROTEIN),
        )
    elif (
        file := (clean_statement(segment_definition))
    ) and file in global_config.named_files:
        if type != ProtomerTypes.PROTEIN:
            # TODO: Add explicit RNA/DNA/ligand extraction from structure files once
            # residue typing is preserved by ParsedFile/StructureParser.
            raise NotImplementedError(
                "Structure-backed seq segments currently support protein protomers only. "
                f"Got protomer type {type.value!r} for segment {segment_definition!r}."
            )
        sequence, ptm_dict = StatementHandlerBase.registry.get(
            (type, "from_file")
        ).handle(segment_definition)
        return Segment(sequence, ptm_dict, is_fixed, type)
    elif (literal := clean_statement(segment_definition)) and is_literal(type, literal):
        sequence, ptm_dict = StatementHandlerBase.registry.get(
            (type, "literal")
        ).handle(segment_definition)
        return Segment(sequence, ptm_dict, is_fixed, type)
    else:
        raise ValueError(
            f"Invalid segment definition '{segment_definition}' for type {type}. "
            "Must be a valid literal, random range, or predefined statement."
        )


def is_literal(type: ProtomerTypes, segment_definition: str) -> bool:
    if type == ProtomerTypes.SMILES:
        # todo : implement SMILES validation
        return True
    elif type == ProtomerTypes.PROTEIN:
        remove_ptm = re.sub(r"[\[\(].+?[\]\)]", "?", segment_definition)

        # Allow 'X' for unknown residues (e.g., extracted from structures).
        return all(base in "ACDEFGHIKLMNPQRSTVWY?X" for base in remove_ptm)
    elif type == ProtomerTypes.RNA:
        remove_ptm = re.sub(r"[\[\(].+?[\]\)]", "?", segment_definition)
        return all(base in "ACGU?" for base in remove_ptm)
    elif type == ProtomerTypes.DNA:
        remove_ptm = re.sub(r"[\[\(].+?[\]\)]", "?", segment_definition)
        return all(base in "ACGT?" for base in remove_ptm)
    elif type == ProtomerTypes.CCD:
        # todo : implement CCD validation
        return True
    else:
        raise ValueError(
            f"Unsupported Protomer type: {type}. Must be one of {ProtomerTypes.__members__.keys()}"
        )


def clean_statement(s: str) -> str:
    """Clean and normalize a protomer statement."""
    s = s.strip()
    if not s:
        raise ValueError("Protomer statement cannot be empty.")
    s = re.sub(r"@.+", "", s)
    s = s.replace("+", "")
    s = re.sub(r":.+", "", s)  # Remove all whitespace
    return s


class Protomer:
    def __init__(self, name, config: str, type, homologies=None) -> None:
        self.name = name
        self.type = ProtomerTypes.from_alias(type)
        self.config = config
        self.components = []
        self.component_ids = []
        self.components_slices = []
        self._ptm_dict = {}
        self._locked_positions: set[int] = set()
        self._mask_mutable_msa = True
        if isinstance(homologies, (dict, DictConfig)):
            self._mask_mutable_msa = bool(
                homologies.get("maskMutableResidues", True)
            )
        self.initialize(config)
        if homologies is not None:
            self._homology = parse_homology_arg(
                self.sequence, (~self.mutable_mask).nonzero()[0].tolist(), homologies
            )
        else:
            self._homology = {}

    def _register_name(self): ...

    def initialize(self, protomer_config) -> None:
        """Initialize Protomer with configuration.

        Args:
            args: Configuration containing protomer parameters

        Raises:
            ValueError: If configuration is missing or invalid
        """
        index = 1
        residue_index = 0
        for config in protomer_config.split():
            if self.type == ProtomerTypes.SMILES:
                config = config.strip('"').strip("'")
                aliases = []
            else:
                config, *aliases = config.split("@")

            segment = segment_handler_distributer(self.type, config)
            name = f"{self.name}_{index}"

            global_config.segments[name] = segment
            global_config.segment_manager.add_segment(name, segment)
            self.components.append(name)
            self.component_ids.append(id(segment))
            self.components_slices.append(
                slice(residue_index, residue_index + len(segment.sequence))
            )
            index += 1
            residue_index += len(segment.sequence)
            for alias in aliases:
                alias = alias.strip()
                if alias:
                    global_config.segments[alias] = segment

    @property
    def sequence(self):
        """Concatenate all segments to form the full sequence."""
        return "".join(
            segment.sequence
            for name, segment in global_config.segments.items()
            if name in self.components
        )

    @property
    def ptm_dict(self) -> Dict[int, str]:
        """Combine PTM dictionaries from all segments."""
        if self._ptm_dict:
            return self._ptm_dict

        ptm_dict = {}
        index = 0
        for name in self.components:
            segment = global_config.segments[name]
            if segment.ptm_dict:
                for pos, ptm in segment.ptm_dict.items():
                    ptm_dict[index + pos] = ptm
            index += len(segment.sequence)
        self._ptm_dict = ptm_dict
        return ptm_dict

    @property
    def homology(self) -> Dict[str, Dict[str, str]]:
        """Combine homology data from all segments."""
        if not self._homology:
            return self._homology

        seq = self.mutated_sequence
        if getattr(self, "_homology_seq", None) == seq:
            return self._homology

        retained_msa_positions = (
            (~self.mutable_mask).nonzero()[0].tolist()
            if self._mask_mutable_msa
            else list(range(len(seq)))
        )
        for key in ("unpairedMsa", "pairedMsa"):
            msa_string = self._homology.get(key, None)
            if not isinstance(msa_string, str) or msa_string.strip() == "":
                continue
            self._homology[key] = mask_msa_fixpos(
                seq,
                retained_msa_positions,
                msa_string,
                remove_mainly_gaps=True,
            )

        self._homology_seq = seq
        return self._homology

    @property
    def length(self) -> int:
        """Get the total length of the sequence."""
        return (
            sum(len(global_config.segments[name].sequence) for name in self.components)
            if self.type in MutableTypes
            else 1
        )

    @property
    def mutable_length(self) -> int:
        """Number of residues that are allowed to mutate."""
        if self.type not in MutableTypes:
            return self.length
        return int(self.mutable_mask.sum())

    @property
    def mutable_mask(self) -> np.ndarray:
        """True for mutable."""
        if self.type not in MutableTypes:
            raise ValueError(
                f"Fixed indices are not applicable for type {self.type}. "
                "Only mutable types support fixed indices."
            )
        if hasattr(self, "_mutable_mask"):
            return self._mutable_mask
        result = []
        index = 0
        for name in self.components:
            segment = global_config.segments[name]
            if segment.is_fixed:
                result.extend([i + index for i in range(len(segment.sequence))])
            index += len(segment.sequence)
        self._mutable_mask = np.ones(self.length, dtype=bool)
        self._mutable_mask[result] = False
        if self._locked_positions:
            locked = [p for p in self._locked_positions if 0 <= p < self.length]
            self._mutable_mask[locked] = False
        return self._mutable_mask

    @property
    def locked_positions(self) -> frozenset[int]:
        return frozenset(self._locked_positions)

    def add_locked_positions(self, positions: Iterable[int]) -> None:
        changed = False
        for p in positions:
            p0 = int(p)
            if p0 < 0 or p0 >= self.length:
                raise ValueError(
                    f"Locked position {p0} out of range for protomer {self.name} length={self.length}"
                )
            if p0 not in self._locked_positions:
                self._locked_positions.add(p0)
                changed = True
        if changed and hasattr(self, "_mutable_mask"):
            delattr(self, "_mutable_mask")
        if changed and hasattr(self, "_homology_seq"):
            delattr(self, "_homology_seq")

    def set_residue(self, pos1: int, aa: str) -> None:
        aa = str(aa or "").strip().upper()
        if len(aa) != 1 or not aa.isalpha():
            raise ValueError(f"Invalid residue letter: {aa!r}")
        pos0 = int(pos1) - 1
        if pos0 < 0 or pos0 >= self.length:
            raise ValueError(
                f"Residue position {pos1} out of range for protomer {self.name} length={self.length}"
            )
        if pos0 in self.ptm_dict:
            raise ValueError(
                f"Cannot overwrite residue at position {pos1} in protomer {self.name}: PTM present"
            )

        for idx, seg_slice in enumerate(self.components_slices):
            if pos0 < seg_slice.start or pos0 >= seg_slice.stop:
                continue
            segment_name = self.components[idx]
            segment = global_config.segments[segment_name]
            rel = pos0 - int(seg_slice.start)
            seq = list(segment.sequence)
            if rel < 0 or rel >= len(seq):
                raise AssertionError("Segment slice mapping out of bounds")
            seq[rel] = aa
            segment.sequence = "".join(seq)
            # Clear caches so any downstream consumers see updated state.
            self._ptm_dict = {}
            if hasattr(self, "_mutable_mask"):
                delattr(self, "_mutable_mask")
            if hasattr(self, "_homology_seq"):
                delattr(self, "_homology_seq")
            return

        raise AssertionError(
            f"Failed to locate residue position {pos1} within protomer {self.name} segments"
        )

    @property
    def tied_positions(self):
        """Get tied positions for symmetry."""
        if self.type not in MutableTypes:
            raise ValueError(
                f"Tied positions are not applicable for type {self.type}. "
                "Only mutable types support tied positions."
            )
        if hasattr(self, "_tied_positions"):
            return self._tied_positions
        # [[segment indices], ...]
        # [[[segment indices 1], [segment indices 2]], ...]
        # fixme this function currently not considering aliases in other protomers
        index = 0
        segment_indices = {}
        for name in self.components:
            segment_len = len(global_config.segments[name].sequence)
            segment_indices[name] = list(range(index, index + segment_len))
            index += segment_len

        tied_positions = []
        for _, pack in itertools.groupby(
            zip(self.components, self.component_ids), key=lambda x: x[1]
        ):
            comps, comp_ids = zip(*pack)
            segment_len = len(global_config.segments[comps[0]].sequence)
            for i in range(segment_len):
                tied_positions.append(tuple(segment_indices[comp][i] for comp in comps))

        self._tied_positions = tied_positions
        return tied_positions

    @property
    def tied_positions_dict(self):
        if hasattr(self, "_tied_positions_dict"):
            return self._tied_positions_dict

        self._tied_positions_dict = {
            pos: tuple(pos_tuple)
            for pos_tuple in self.tied_positions
            for pos in pos_tuple
        }
        return self._tied_positions_dict

    def aggregate_bool(
        self,
        bools: np.ndarray,
        merge: Literal["any", "all"] = "any",
    ) -> np.ndarray:
        """Aggregate boolean masks from segments."""
        if len(bools) != self.length:
            raise ValueError(
                f"Boolean mask length {len(bools)} does not match protomer length {self.length}."
            )
        for tied in self.tied_positions:
            tied_bools = bools[list(tied)]
            if merge == "any":
                bools[list(tied)] = np.any(tied_bools, axis=0)
            elif merge == "all":
                bools[list(tied)] = np.all(tied_bools, axis=0)
            else:
                raise ValueError(f"Invalid merge type: {merge}. Use 'any' or 'all'.")
        return bools

    def update_sequence(self, updated_sites: Dict[int, Tuple[str, str]]):
        to_mutate_sites = [
            self.tied_positions_dict.get(pos, pos) for pos in updated_sites
        ]

        if len(set(to_mutate_sites)) != len(to_mutate_sites):
            raise ValueError(
                "Tied positions should not be updated with different values. "
            )

    def propagating_probs(self, probs: np.ndarray) -> None:
        """Propagate probabilities to all segments."""
        if probs.shape[-2] != self.length:
            raise ValueError(
                f"Probabilities shape {probs.shape[-2]} does not match protomer length {self.length}."
            )
        averaged_probs = np.mean(probs, axis=0)
        processed = []
        for idx, name in enumerate(self.components):
            segment = global_config.segments[name]

            processed.append(id(segment))
            segment.add_probs(averaged_probs[self.components_slices[idx]])

    def propagating_scores(self, scores: np.ndarray) -> None:
        """Propagate scores to all segments."""
        scores = np.asarray(scores)
        if scores.shape[-1] != self.length:
            raise ValueError(
                f"Scores shape {scores.shape[-1]} does not match protomer length {self.length}."
            )
        if len(scores.shape) > 1:
            averaged_scores = np.mean(scores, axis=0)
        else:
            averaged_scores = scores
        processed = []
        for idx, name in enumerate(self.components):
            segment = global_config.segments[name]

            processed.append(id(segment))
            segment.add_scores(averaged_scores[self.components_slices[idx]])

    def collect_mutations(self) -> List[Tuple[int, str, str, float, float]]:
        """Collect mutations from all segments."""
        mutations = []
        for name in self.components:
            segment = global_config.segments[name]
            segment_mutations = segment.sample_mutations()
            if segment_mutations:
                mutations.extend(segment_mutations)

        return mutations

    def add_mutation(self, mutation): ...

    @property
    def mutated_sequence(self) -> str:
        """Get the sequence after applying mutations."""
        mutated_sequence = []
        for name in self.components:
            segment = global_config.segments[name]
            mutated_sequence.append(segment.mutated_sequence)
        return "".join(mutated_sequence)

    def select_residues_by_alias(self, expression: str) -> np.ndarray:
        """Select residues by alias expression and return a boolean mask."""
        if not expression.strip():
            raise ValueError("Expression cannot be empty")

        selected_mask = np.zeros(self.length)

        for alias_expr in expression.split(";"):
            alias_expr = alias_expr.strip()
            if not alias_expr:
                continue

            try:
                self._process_alias_expression(alias_expr, selected_mask)
            except Exception as e:
                raise ValueError(
                    f"Error processing alias expression '{alias_expr}': {e}"
                )

        return selected_mask

    def _process_alias_expression(
        self, alias_expr: str, selected_mask: np.ndarray
    ) -> None:
        """Process a single alias expression and update the mask."""
        segment_name, indices_str = self._parse_alias_expression(alias_expr)

        if segment_name not in global_config.segments:
            raise ValueError(
                f"Segment '{segment_name}' not found in global_config.segments."
            )

        segment_id = id(global_config.segments[segment_name])
        if segment_id not in self.component_ids:
            return

        segment_slice_indices = [
            i for i, comp_id in enumerate(self.component_ids) if comp_id == segment_id
        ]

        if not segment_slice_indices:
            return

        parsed_indices, weight = self._parse_indices(
            indices_str, segment_slice_indices[0]
        )
        self._update_mask_for_segments(
            selected_mask, segment_slice_indices, parsed_indices, weight
        )

    def _parse_alias_expression(self, alias_expr: str) -> tuple[str, str]:
        """Parse alias expression into segment name and indices string."""
        if ":" not in alias_expr:
            return alias_expr.strip(), "all"

        parts = alias_expr.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid alias expression format: {alias_expr}")

        return parts[0].strip(), parts[1].strip()

    def _parse_indices(
        self, indices_str: str, segment_slice_idx: int
    ) -> tuple[np.ndarray, float]:
        """Parse indices string into numpy array and weight."""
        # Parse weight
        if "@" in indices_str:
            indices_part, weight_str = indices_str.split("@", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(f"Invalid weight format: {weight_str}")
        else:
            indices_part, weight = indices_str, 1.0

        indices_part = indices_part.strip()
        segment_length = (
            self.components_slices[segment_slice_idx].stop
            - self.components_slices[segment_slice_idx].start
        )

        # Parse indices
        if indices_part == "all":
            return np.arange(1, segment_length + 1), weight
        else:
            mask = []
            for idx in indices_part.split(","):
                idx = idx.strip()
                if not idx:
                    raise ValueError(f"Empty index in indices part: {indices_part}")

                if "-" in idx:
                    mask.extend(self._parse_range_indices(idx, segment_length).tolist())
                else:
                    mask.extend(self._parse_single_index(idx, segment_length).tolist())
            return np.array(sorted(set(mask))), weight

    def _parse_range_indices(
        self, indices_part: str, segment_length: int
    ) -> np.ndarray:
        """Parse range indices (e.g., '1-5')."""
        try:
            start_str, end_str = indices_part.split("-", 1)
            start, end = int(start_str), int(end_str)
        except ValueError:
            raise ValueError(f"Invalid range format: {indices_part}")

        if start < 1 or end < start or end > segment_length:
            raise ValueError(
                f"Range {start}-{end} is invalid for segment length {segment_length}"
            )

        return np.arange(start, end + 1)

    def _parse_single_index(self, indices_part: str, segment_length: int) -> np.ndarray:
        """Parse single index."""
        try:
            index = int(indices_part)
        except ValueError:
            raise ValueError(f"Invalid index format: {indices_part}")

        if index < 1 or index > segment_length:
            raise ValueError(
                f"Index {index} is out of range for segment length {segment_length}"
            )

        return np.array([index])

    def _update_mask_for_segments(
        self,
        selected_mask: np.ndarray,
        segment_slice_indices: list[int],
        parsed_indices: np.ndarray,
        weight: float,
    ) -> None:
        """Update the selection mask for all segment slices."""
        for idx in segment_slice_indices:
            residue_range = self.components_slices[idx]
            residues = np.arange(residue_range.start, residue_range.stop)

            # Convert 1-based indices to 0-based and validate
            zero_based_indices = parsed_indices - 1
            if np.any(zero_based_indices >= len(residues)) or np.any(
                zero_based_indices < 0
            ):
                raise ValueError(
                    f"Indices {parsed_indices} out of range for segment with {len(residues)} residues"
                )

            selected_residues = residues[zero_based_indices]
            selected_mask[selected_residues] = weight


class SegmentManager:
    def __init__(self):
        self.segments: dict[str, Segment] = {}
        self.unique_segments: dict[int, Segment] = {}
        self.primary_names: dict[int, str] = {}
        self._pre_update_snapshot: dict[int, tuple[str, list | None]] | None = None

    def add_segment(self, name: str, segment: Segment):
        """Add a segment to the manager."""
        if name in self.segments:
            raise ValueError(f"Segment '{name}' already exists.")
        self.segments[name] = segment
        segment_id = id(segment)
        if segment_id not in self.unique_segments:
            self.unique_segments[segment_id] = segment
            self.primary_names[segment_id] = name

    def collect_mutations(
        self,
    ) -> List[(Tuple[str, Tuple[int, str, str, float, float]])]:
        """Collect mutations from all segments."""
        mutations = []
        for segment_id, segment in self.unique_segments.items():
            name = self.primary_names[segment_id]
            segment_mutations = segment.sample_mutations()
            segment_mutations = list(
                zip([name] * len(segment_mutations), segment_mutations)
            )

            if segment_mutations:
                mutations.extend(segment_mutations)
        return mutations

    def add_mutations(
        self,
        mutations: List[(Tuple[str, Tuple[int, str, str, float, float]])],
        num_mutations: int,
    ):
        if num_mutations <= 0:
            return
        for name, mutation in mutations:
            if name not in self.segments:
                raise ValueError(f"Segment '{name}' not found.")
            segment = self.segments[name]
            is_exist = segment.add_mutations(mutation)
            if not is_exist:
                num_mutations -= 1
            if num_mutations <= 0:
                return

    def apply_mutations(self):
        """Apply mutations to all segments."""
        for segment in self.unique_segments.values():
            segment.apply_mutations()
        self.clear_mutations()

    def clear_mutations(self):
        """Clear mutations from all segments."""
        for segment in self.unique_segments.values():
            segment.clear_mutations()

    def begin_pre_update_step(self) -> None:
        """Snapshot accepted sequences before staged mutator-stack updates."""
        if self._pre_update_snapshot is not None:
            return
        self._pre_update_snapshot = {
            segment_id: (
                segment.sequence,
                list(segment.mutations) if segment.mutations is not None else None,
            )
            for segment_id, segment in self.unique_segments.items()
        }

    def preapply_pending_mutations(self, *, max_entropy: float | None = None) -> None:
        """Expose queued mutations as the working sequence for later mutators.

        The snapshot created by ``begin_pre_update_step`` lets the caller roll the
        sequence back when the whole mutation step is rejected.
        """
        if self._pre_update_snapshot is None:
            self.begin_pre_update_step()
        for segment in self.unique_segments.values():
            if max_entropy is None:
                segment.apply_mutations()
                segment.clear_mutations()
                continue

            pending = list(segment.mutations or [])
            to_apply = [
                mutation
                for mutation in pending
                if len(mutation) < 4 or float(mutation[3]) <= float(max_entropy)
            ]
            to_keep = [
                mutation
                for mutation in pending
                if len(mutation) >= 4 and float(mutation[3]) > float(max_entropy)
            ]
            segment.mutations = to_apply
            segment.apply_mutations()
            segment.mutations = to_keep

    def finish_pre_update_step(self, *, accepted: bool) -> None:
        """Commit or roll back staged mutator-stack sequence updates."""
        snapshot = self._pre_update_snapshot
        if snapshot is None:
            return
        if not accepted:
            for segment_id, (sequence, mutations) in snapshot.items():
                segment = self.unique_segments.get(segment_id)
                if segment is None:
                    continue
                segment.sequence = sequence
                segment.mutations = list(mutations) if mutations is not None else []
        else:
            self.clear_mutations()
        self._pre_update_snapshot = None

    def clean_up(self):
        """Clear all segments."""
        for segment in self.unique_segments.values():
            segment.clean_up()


def parse_ptm_sequence(s: str):
    """
    Parse a protein sequence with PTMs (e.g., 'QAQ[PTM1]TDR(PTM2)VSLRNLRG').

    Args:
        s (str): Protein sequence with PTMs.

    Returns:
        Tuple[str, Dict[int, str]]:
            - Main sequence (PTMs removed).
            - PTM dictionary (0-based positions mapped to PTM names).
    """
    # Fast path: no PTMs; avoid importing AF3 deps for Protenix-only runs.
    if "[" not in s and "(" not in s:
        return s, {}

    pattern = re.compile(r"([A-Z])|\[([A-Za-z0-9_]+)\]?|\(([A-Za-z0-9_]+)\)")
    sequence = []
    ptm_dict = {}
    try:
        alias_map = getattr(global_config, "nonstandard_aliases")
    except Exception:
        alias_map = {}
    try:
        user_ccd = getattr(global_config, "user_ccd")
    except Exception:
        user_ccd = None
    ccd = cached_ccd(user_ccd=user_ccd)
    i = 0  # Current position index

    for match in pattern.finditer(s):
        std_aa, ptm_form1, ptm_form2 = match.groups()
        if std_aa:
            sequence.append(std_aa)
            i += 1
        elif ptm_form1 or ptm_form2:
            token = (ptm_form1 or ptm_form2).strip()
            ptm_name, parent_aa = resolve_nonstandard_alias(
                token,
                alias_map=alias_map,
                require_ptm=True,
            )
            validate_ptm(ptm_name, ccd)
            ptm_dict[i] = ptm_name
            std_aa = get_standard_aa(ptm_name, ccd, fallback_parent_aa=parent_aa)
            sequence.append(std_aa)
            i += 1
        else:
            raise ValueError(
                f"Invalid PTM format in sequence: {s}. Expected [PTM] or (CCD_CODE)."
            )

    return "".join(sequence), ptm_dict


def validate_ptm(ptm_name: str, ccd: dict) -> None:
    """Validate if the PTM exists in the cached CCD."""
    if ptm_name not in ccd:
        raise ValueError(f"Unknown PTM: {ptm_name}")


def get_standard_aa(
    ptm_name: str,
    ccd: dict,
    *,
    fallback_parent_aa: str | None = None,
) -> str:
    """Retrieve the standard amino acid for a given PTM."""
    ptm = ccd[ptm_name]
    try:
        std_aa = str(ptm["_chem_comp.one_letter_code"][0]).strip().upper()
    except Exception:
        std_aa = ""
    if std_aa and std_aa in "ACDEFGHIKLMNPQRSTVWYAGCTU":
        return std_aa
    if fallback_parent_aa:
        fallback = str(fallback_parent_aa).strip().upper()
        if fallback and fallback in "ACDEFGHIKLMNPQRSTVWYAGCTU":
            return fallback
    raise ValueError(
        f"Invalid PTM one-letter code for {ptm_name}: {std_aa!r}. "
        f"fallback_parent_aa={fallback_parent_aa!r}"
    )


def parse_ranges(s: str):
    if s is None:
        raise ValueError("Missing chain/range statement (expected like 'A' or 'A1-10').")
    result = defaultdict(list)
    s = str(s).strip()
    if not s:
        raise ValueError("Empty chain/range statement (expected like 'A' or 'A1-10').")
    items = s.split(",")

    for item in items:
        item = item.strip()
        if not item:
            continue
        # Allow specifying an entire chain with no residue indices, e.g. "A" or "A,B".
        match_chain_only = re.fullmatch(r"([A-Za-z])", item)
        if match_chain_only:
            result[match_chain_only.group(1)] = []
            continue

        match = re.fullmatch(r"([A-Za-z])(?::?)(\d+)(?:-(\d+))?", item)
        if not match:
            raise ValueError(f"Invalid format: {item}")
        prefix = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        result[prefix].extend(range(start, end + 1))

    return dict(result)


def parse_predefined_statement(
    config: DictConfig,
) -> None:
    """Parse predefined statements from the configuration.

    Args:
        config: Configuration containing predefined statements

    Raises:
        ValueError: If configuration is missing or invalid
    """
    if not config.predefined_statement:
        raise ValueError("Predefined statement configuration is empty.")

    try:
        predefined_statement = global_config.predefined_statement
    except RuntimeError:
        global_config.init_const("predefined_statement", {})
        predefined_statement = global_config.predefined_statement
    else:
        if not isinstance(predefined_statement, dict):
            raise TypeError(
                "global_config.predefined_statement must be a dict, got "
                f"{type(predefined_statement).__name__}"
            )
        predefined_statement.clear()

    for key, value in config.predefined_statement.items():
        if ":" in value:
            sequence, ptm = ProteinFileHandler().handle(value)
        else:
            sequence, ptm = ProteinLiteralHandler().handle(value)
        predefined_statement[key] = (sequence, ptm)


class ResidueSampler:
    def __init__(self, frequencies: dict[str, float], exclude: list[str] = None):
        self.frequencies = frequencies
        if exclude is not None:
            self.exclude = set(exclude)
            self.frequencies = {
                k: v for k, v in self.frequencies.items() if k not in self.exclude
            }
        self.normalized_frequencies = self._normalize_frequencies()

    def _normalize_frequencies(self):
        """Normalize the frequencies to sum to 1."""
        total = sum(self.frequencies.values())
        if total == 0:
            raise ValueError("Total frequency cannot be zero.")
        return {k: v / total for k, v in self.frequencies.items()}

    def sample(self):
        """Sample a residue based on the normalized frequencies."""
        import random

        return random.choices(
            list(self.normalized_frequencies.keys()),
            weights=list(self.normalized_frequencies.values()),
            k=1,
        )[0]

    def sample_n(self, n: int) -> str:
        """Sample n residues based on the normalized frequencies."""
        import random

        return "".join(
            random.choices(
                list(self.normalized_frequencies.keys()),
                weights=list(self.normalized_frequencies.values()),
                k=n,
            )
        )


def register_files(files: Dict[str, str]) -> None:
    """Register files in the global configuration."""
    global_config.init_const("named_files", {})
    for alias, file_path in files.items():
        if alias in global_config.named_files:
            raise ValueError(f"File alias '{alias}' already exists.")
        global_config.named_files[alias] = ParsedFile(file_path)


MutableTypes = [ProtomerTypes.PROTEIN, ProtomerTypes.RNA, ProtomerTypes.DNA]


if __name__ == "__main__":
    # Example usage
    config = {
        "A": "+CCRIEGDTIS[TPO]VMP[PTR]LLIVAFVLGAL@alias1 alias1",
        "B": "alias1 10-20 example_1:C1-10,C2-10@motif",
        "C": "5-15",
        "D": "[ATP]",
    }
    types = {
        "A": "p",
        "B": "p",
        "C": "r",
        "D": "c",
    }
    files = {
        "example_1": "test/inputs/1BC8_1.pdb",
    }
    exclude_AA = ["C"]
    global_config.init_const("exclude_AA", exclude_AA)

    register_files(files)

    for key, value in config.items():
        protomer = Protomer(key, value, types[key])
        print(protomer.sequence)  # Output: AB?CD?EF
        print(
            global_config.segments
        )  # Output: {'example_1': Segment(...), 'alias1': Segment(...), ...}
