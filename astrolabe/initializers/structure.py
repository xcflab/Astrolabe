from __future__ import annotations

import gzip
import json
import shutil
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.initializers.base import Initializer, InitializerResult
from astrolabe.initializers.mapping import (
    ChainMapEntry,
    ResolvedChainMapping,
    load_structure_residues,
    normalize_chain_map,
    resolve_chain_mapping,
)


def to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def resolve_structure_path(value: Any, *, field: str) -> Path:
    reference = str(value or "").strip()
    if not reference:
        raise ValueError(f"{field} is required.")
    try:
        parsed_file = (global_config.named_files or {}).get(reference)
    except RuntimeError:
        parsed_file = None
    if parsed_file is not None:
        path_value = getattr(parsed_file, "file_path", None)
        if not path_value:
            raise ValueError(f"{field} alias {reference!r} has no file_path.")
        path = Path(path_value).expanduser().resolve()
    else:
        path = Path(reference).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


@dataclass(frozen=True, slots=True)
class PreparedStructure:
    path: Path
    details: dict[str, Any]


class StructureSequenceInitializer(Initializer):
    """Shared structure -> mutator -> initial-sequence implementation."""

    def __init__(
        self,
        *,
        mode: str,
        args: Any,
        mutators: dict[str, Any],
        segment_plan: dict[str, list[tuple[str, slice, list[int]]]],
        mode_config: Any,
    ):
        self.mode = str(mode)
        self.args = args
        self.segment_plan = segment_plan
        self.output_dir = (
            Path(args.out_dir) / "structures" / f"{self.mode}_initializer"
        )

        initializer_cfg = getattr(args, "initializer", None)
        if not (isinstance(initializer_cfg, dict) or OmegaConf.is_dict(initializer_cfg)):
            raise ValueError(
                f"initializer.mode={self.mode!r} requires an initializer mapping."
            )
        self.sequence_designer_alias = str(
            initializer_cfg.get("sequence_designer") or ""
        ).strip()
        if not self.sequence_designer_alias:
            raise ValueError(
                f"initializer.mode={self.mode!r} requires initializer.sequence_designer "
                "to reference a mutator alias."
            )
        try:
            self.sequence_designer = mutators[self.sequence_designer_alias]
        except KeyError as exc:
            available = ", ".join(sorted(mutators)) or "<none>"
            raise ValueError(
                "initializer.sequence_designer references an unavailable mutator alias "
                f"{self.sequence_designer_alias!r}. Available aliases: {available}."
            ) from exc
        if not callable(
            getattr(self.sequence_designer, "design_initial_sequences", None)
        ):
            raise ValueError(
                f"Mutator alias {self.sequence_designer_alias!r} cannot design an "
                "initial sequence from a structure."
            )
        capabilities = getattr(self.sequence_designer, "capabilities", None)
        if not bool(
            getattr(capabilities, "supports_initialization_from_structure", False)
        ):
            raise ValueError(
                f"Mutator alias {self.sequence_designer_alias!r} does not declare "
                "supports_initialization_from_structure."
            )

        normalized_mode_config = to_plain(mode_config)
        if not isinstance(normalized_mode_config, dict):
            raise ValueError(
                f"initializer.mode={self.mode!r} requires initializer.{self.mode}."
            )
        self.mode_config = dict(normalized_mode_config)
        self.chain_entries = self._normalize_chain_map(self.mode_config.get("chain_map"))
        self.chain_map = {
            chain: entry.protomer for chain, entry in self.chain_entries.items()
        }
        self.allow_partial_mapping = bool(
            self.mode_config.get("allow_partial_mapping", False)
        )
        self._validate_segment_plan()

    def _normalize_chain_map(self, raw_value: Any) -> dict[str, ChainMapEntry]:
        raw = to_plain(raw_value)
        try:
            return normalize_chain_map(raw)
        except ValueError as exc:
            raise ValueError(
                f"initializer.{self.mode}.{exc}"
            ) from exc

    def _validate_segment_plan(self) -> None:
        mapped_protomers = set(self.chain_map.values())
        unknown = sorted(mapped_protomers - set(global_config.protomers))
        if unknown:
            raise ValueError(
                f"initializer.{self.mode}.chain_map references unknown protomer(s): "
                + ", ".join(unknown)
            )
        nonprotein = sorted(
            protomer_name
            for protomer_name in mapped_protomers
            if global_config.protomers[protomer_name].type != ProtomerTypes.PROTEIN
        )
        if nonprotein:
            raise ValueError(
                f"initializer.mode={self.mode!r} only supports protein protomers; got: "
                + ", ".join(nonprotein)
            )
        if not mapped_protomers.intersection(self.segment_plan):
            raise ValueError(
                f"initializer.{self.mode}.chain_map must include at least one protomer "
                "with mutable protein positions."
            )

    def _mutable_positions(self, protomer_name: str) -> set[int]:
        return {
            int(position)
            for _segment_name, _segment_slice, positions in self.segment_plan.get(
                protomer_name,
                [],
            )
            for position in positions
        }

    def _resolve_chain_mappings(
        self,
        structure_path: Path,
    ) -> list[ResolvedChainMapping]:
        residues_by_chain = load_structure_residues(structure_path)
        return [
            resolve_chain_mapping(
                entry,
                residues_by_chain=residues_by_chain,
                protomer=global_config.protomers[entry.protomer],
                segments=global_config.segments,
                context_only=entry.protomer not in self.segment_plan,
            )
            for entry in self.chain_entries.values()
        ]

    def _sequence_design_features(
        self,
        mappings: list[ResolvedChainMapping],
    ) -> dict[str, Any]:
        redesigned: list[str] = []
        chains_to_design: list[str] = []
        refs_by_protomer: dict[str, dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for mapping in mappings:
            mutable_positions = self._mutable_positions(mapping.protomer)
            mapped_any = False
            for protomer_position, residue_id in zip(
                mapping.protomer_positions,
                mapping.residue_ids,
                strict=True,
            ):
                if protomer_position not in mutable_positions:
                    continue
                redesigned.append(residue_id)
                refs_by_protomer[mapping.protomer][protomer_position].append(residue_id)
                mapped_any = True
            if mapped_any:
                chains_to_design.append(mapping.structure_chain)

        tied_groups: list[list[str]] = []
        for protomer_name, refs_by_position in refs_by_protomer.items():
            protomer = global_config.protomers[protomer_name]
            grouped_positions: set[int] = set()
            for group in protomer.tied_positions:
                mutable_group = sorted(
                    {
                        int(position)
                        for position in group
                        if refs_by_position.get(int(position))
                    }
                )
                refs = [
                    residue_id
                    for position in mutable_group
                    for residue_id in refs_by_position[position]
                ]
                if len(refs) > 1:
                    tied_groups.append(refs)
                    grouped_positions.update(mutable_group)
            for position, refs in refs_by_position.items():
                if position not in grouped_positions and len(refs) > 1:
                    tied_groups.append(refs)

        tied_groups = [group for group in tied_groups if len(group) > 1]
        return {
            "chains_to_design": chains_to_design,
            "redesigned_residues": " ".join(redesigned),
            "symmetry_residues": "|".join(
                ",".join(group) for group in tied_groups
            ),
            "symmetry_weights": "|".join(
                ",".join([str(1.0 / len(group))] * len(group))
                for group in tied_groups
            ),
        }

    def _apply_sequences(
        self,
        sequences_by_chain: dict[str, str],
        mappings: list[ResolvedChainMapping],
    ) -> dict[str, str]:
        designed_residues: dict[str, dict[int, list[tuple[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for mapping in mappings:
            if mapping.context_only:
                continue
            chain = mapping.structure_chain
            if chain not in sequences_by_chain:
                available = ", ".join(sorted(sequences_by_chain)) or "<none>"
                raise ValueError(
                    f"Structure sequence designer output is missing chain {chain!r}. "
                    f"Available chains: {available}."
                )
            sequence = str(sequences_by_chain[chain])
            if len(sequence) != mapping.structure_length:
                raise ValueError(
                    f"Initializer sequence length mismatch for structure chain {chain!r}: "
                    f"expected {mapping.structure_length}; got {len(sequence)}."
                )
            mutable_positions = self._mutable_positions(mapping.protomer)
            for structure_index, protomer_position in zip(
                mapping.structure_indices,
                mapping.protomer_positions,
                strict=True,
            ):
                if protomer_position not in mutable_positions:
                    continue
                designed_residues[mapping.protomer][protomer_position].append(
                    (chain, sequence[structure_index])
                )

        for protomer_name in sorted(set(self.chain_map.values()).intersection(self.segment_plan)):
            expected = self._mutable_positions(protomer_name)
            observed = set(designed_residues[protomer_name])
            missing = sorted(expected - observed)
            if missing and not self.allow_partial_mapping:
                raise ValueError(
                    f"Initializer mapping does not cover mutable position(s) for protomer "
                    f"{protomer_name!r}: "
                    + ", ".join(str(position + 1) for position in missing)
                    + ". Set allow_partial_mapping=true only when retaining the existing "
                    "sequence at unmapped positions is intentional."
                )
            inconsistent_positions = [
                position + 1
                for position, values in designed_residues[protomer_name].items()
                if len({residue for _chain, residue in values}) != 1
            ]
            if inconsistent_positions:
                raise ValueError(
                    f"Initializer tied chains for protomer {protomer_name!r} produced "
                    "different residues at mutable position(s): "
                    + ", ".join(str(position) for position in inconsistent_positions)
                    + "."
                )

        segment_updates: dict[int, tuple[Any, dict[int, str]]] = {}
        initialized_protomers = sorted(designed_residues)
        for protomer_name in initialized_protomers:
            segments = self.segment_plan[protomer_name]
            for segment_name, segment_slice, mutable_positions in segments:
                segment = global_config.segments[segment_name]
                for position in mutable_positions:
                    values = designed_residues[protomer_name].get(int(position))
                    if not values:
                        continue
                    relative_position = int(position - segment_slice.start)
                    designed_residue = values[0][1]
                    if designed_residue not in "ACDEFGHIKLMNPQRSTVWY":
                        raise ValueError(
                            f"Initializer sequence designer produced non-canonical residue "
                            f"{designed_residue!r} at mutable position "
                            f"{protomer_name}{int(position) + 1}."
                        )
                    _segment, updates = segment_updates.setdefault(
                        id(segment),
                        (segment, {}),
                    )
                    previous = updates.get(relative_position)
                    if previous is not None and previous != designed_residue:
                        raise ValueError(
                            f"Initializer sequence design produced inconsistent values for "
                            f"shared segment {segment_name!r} position "
                            f"{relative_position + 1}."
                        )
                    updates[relative_position] = designed_residue

        for segment, updates in segment_updates.values():
            tokens = list(segment.sequence)
            for relative_position, designed_residue in updates.items():
                if relative_position < 0 or relative_position >= len(tokens):
                    raise ValueError(
                        "Initializer segment mapping produced an out-of-range position."
                    )
                tokens[relative_position] = designed_residue
            segment.sequence = "".join(tokens)
            segment.initialized_from_random = False
        return {
            protomer_name: global_config.protomers[protomer_name].sequence
            for protomer_name in initialized_protomers
        }

    @abstractmethod
    def prepare_structure(self) -> PreparedStructure:
        pass

    def apply(self) -> InitializerResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self.prepare_structure()
        mappings = self._resolve_chain_mappings(prepared.path)
        design_features = self._sequence_design_features(mappings)
        sequences_by_chain = self.sequence_designer.design_initial_sequences(
            structure_path=prepared.path,
            **design_features,
        )
        initialized_sequences = self._apply_sequences(sequences_by_chain, mappings)

        manifest_path = self.output_dir / "initializer_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "mode": self.mode,
                    "sequence_designer": self.sequence_designer_alias,
                    "chain_map": self.chain_map,
                    "resolved_chain_map": [
                        {
                            "structure_chain": mapping.structure_chain,
                            "protomer": mapping.protomer,
                            "structure_residues": list(mapping.residue_ids),
                            "protomer_positions": [
                                position + 1 for position in mapping.protomer_positions
                            ],
                            "context_only": mapping.context_only,
                        }
                        for mapping in mappings
                    ],
                    "selected_structure": str(prepared.path),
                    "structure": prepared.details,
                    "sequence_design": design_features,
                    "designed_sequences_by_chain": sequences_by_chain,
                    "initialized_sequences": initialized_sequences,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return InitializerResult(
            mode=self.mode,
            sequences=initialized_sequences,
            structure_path=prepared.path,
            manifest_path=manifest_path,
        )


class ExistingStructureInitializer(StructureSequenceInitializer):
    def __init__(self, *, args, mutators, segment_plan):
        initializer_cfg = getattr(args, "initializer", None)
        mode_config = (
            initializer_cfg.get("structure")
            if isinstance(initializer_cfg, dict) or OmegaConf.is_dict(initializer_cfg)
            else None
        )
        super().__init__(
            mode="structure",
            args=args,
            mutators=mutators,
            segment_plan=segment_plan,
            mode_config=mode_config,
        )
        self.source_path = resolve_structure_path(
            self.mode_config.get("source"),
            field="initializer.structure.source",
        )

    def prepare_structure(self) -> PreparedStructure:
        destination = self.output_dir / "input_structure.pdb"
        suffixes = [suffix.lower() for suffix in self.source_path.suffixes]
        if suffixes[-1:] == [".pdb"]:
            shutil.copy2(self.source_path, destination)
        elif suffixes[-2:] == [".pdb", ".gz"]:
            with gzip.open(self.source_path, "rt", encoding="utf-8") as source:
                destination.write_text(source.read(), encoding="utf-8")
        elif suffixes[-1:] in ([".cif"], [".mmcif"]) or suffixes[-2:] in (
            [".cif", ".gz"],
            [".mmcif", ".gz"],
        ):
            from Bio.PDB import MMCIFParser, PDBIO

            parser = MMCIFParser(QUIET=True)
            if suffixes[-1:] == [".gz"]:
                with gzip.open(self.source_path, "rt", encoding="utf-8") as source:
                    structure = parser.get_structure("initializer", source)
            else:
                structure = parser.get_structure("initializer", str(self.source_path))
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(destination))
        else:
            raise ValueError(
                "initializer.structure.source must be PDB or mmCIF (optionally gzip-compressed)."
            )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise FileNotFoundError(
                f"Initializer structure snapshot is empty: {destination}"
            )
        return PreparedStructure(
            path=destination,
            details={
                "source": str(self.source_path),
                "snapshot": str(destination),
            },
        )
