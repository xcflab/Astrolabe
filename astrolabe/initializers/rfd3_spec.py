from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from astrolabe.core import global_config
from astrolabe.initializers.mapping import (
    ChainMapEntry,
    find_sequence_matches,
    load_structure_residues,
    select_structure_indices,
    sequence_matches,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id, str_id_to_int_id


RFD3_DIRECT_SPECIFICATION_FIELDS = frozenset(
    {
        "input",
        "contig",
        "unindex",
        "length",
        "ligand",
        "extra",
        "dialect",
        "select_fixed_atoms",
        "select_unfixed_sequence",
        "select_buried",
        "select_partially_buried",
        "select_exposed",
        "select_hbond_acceptor",
        "select_hbond_donor",
        "select_hotspots",
        "redesign_motif_sidechains",
        "symmetry",
        "ori_token",
        "infer_ori_strategy",
        "plddt_enhanced",
        "is_non_loopy",
        "partial_t",
    }
)


@dataclass(frozen=True, slots=True)
class RFD3SpecificationPlan:
    specification: dict[str, Any]
    engine_config: dict[str, Any]
    details: dict[str, Any]


def _format_residue_selection(residues: list[Any]) -> str:
    if not residues:
        raise ValueError("Cannot format an empty RFD3 motif selection.")
    if (
        len(residues) > 1
        and all(not residue.insertion_code for residue in residues)
        and all(
            right.number == left.number + 1
            for left, right in zip(residues, residues[1:])
        )
    ):
        return f"{residues[0].chain}{residues[0].number}-{residues[-1].number}"
    return ",".join(residue.encoded_id for residue in residues)


def _length_contains(raw_value: Any, expected: int) -> bool:
    text = str(raw_value).strip()
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start, end = int(start_text), int(end_text)
        return start <= expected <= end
    return int(text) == expected


class RFD3SpecificationBuilder:
    def __init__(
        self,
        *,
        mode_config: dict[str, Any],
        chain_entries: dict[str, ChainMapEntry],
        resolve_path: Callable[[Any], Path],
    ):
        self.mode_config = dict(mode_config)
        self.chain_entries = dict(chain_entries)
        self.resolve_path = resolve_path
        raw_motif_map = self.mode_config.get("motif_map") or {}
        if not isinstance(raw_motif_map, dict):
            raise ValueError("initializer.rfd3.motif_map must be a mapping.")
        self.motif_map = raw_motif_map

    def build(self) -> RFD3SpecificationPlan:
        engine_config = dict(self.mode_config)
        nested_specification = engine_config.pop("specification", None)
        engine_config.pop("chain_map", None)
        engine_config.pop("motif_map", None)
        engine_config.pop("allow_partial_mapping", None)
        engine_config.pop("n_batches", None)
        engine_config.pop("sample_index", None)

        direct_specification = {
            key: engine_config.pop(key)
            for key in tuple(engine_config)
            if key in RFD3_DIRECT_SPECIFICATION_FIELDS
        }
        if nested_specification is None:
            nested_specification = {}
        if not isinstance(nested_specification, dict):
            raise ValueError("initializer.rfd3.specification must be a mapping.")
        conflicts = sorted(
            key
            for key in direct_specification.keys() & nested_specification.keys()
            if direct_specification[key] != nested_specification[key]
        )
        if conflicts:
            raise ValueError(
                "RFD3 specification fields are set both directly and under "
                "initializer.rfd3.specification with different values: "
                + ", ".join(conflicts)
            )
        user_specification = {**direct_specification, **nested_specification}

        if user_specification.get("input") not in (None, "", "null", "None"):
            user_specification["input"] = str(
                self.resolve_path(user_specification["input"])
            )

        details: dict[str, Any] = {
            "user_specification": dict(user_specification),
            "derived_contig": None,
            "motif_assignments": [],
        }
        if user_specification.get("contig") not in (None, "", "null", "None"):
            specification = dict(user_specification)
            details["source"] = "user_contig"
            return RFD3SpecificationPlan(specification, engine_config, details)
        if user_specification.get("partial_t") not in (None, "", "null", "None"):
            specification = dict(user_specification)
            details["source"] = "user_partial_diffusion"
            return RFD3SpecificationPlan(specification, engine_config, details)

        derived, motif_assignments = self._derive_from_protomers(user_specification)
        specification = {**derived, **user_specification}
        expected_length = int(derived["length"])
        if "length" in user_specification:
            try:
                compatible = _length_contains(
                    user_specification["length"],
                    expected_length,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid initializer.rfd3 length: {user_specification['length']!r}"
                ) from exc
            if not compatible:
                raise ValueError(
                    f"initializer.rfd3 length {user_specification['length']!r} does not "
                    f"contain the realized seqs length {expected_length}."
                )
            specification["length"] = str(expected_length)
        if specification.get("input") and not any(
            specification.get(key) not in (None, "", "null", "None")
            for key in ("contig", "unindex", "ligand", "partial_t")
        ):
            raise ValueError(
                "initializer.rfd3.specification.input is not represented in the "
                "generated composition. Add a fixed + segment/context protomer, or "
                "provide an explicit contig, unindex, ligand, or partial_t."
            )
        details.update(
            {
                "source": "seqs",
                "derived_contig": derived.get("contig"),
                "motif_assignments": motif_assignments,
                "realized_length": expected_length,
            }
        )
        return RFD3SpecificationPlan(specification, engine_config, details)

    def _ordered_entries(self) -> list[ChainMapEntry]:
        try:
            return sorted(
                self.chain_entries.values(),
                key=lambda entry: str_id_to_int_id(entry.structure_chain),
            )
        except ValueError as exc:
            raise ValueError(
                "Automatic RFD3 contig generation requires uppercase output chain IDs."
            ) from exc

    def _derive_from_protomers(
        self,
        user_specification: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        entries = self._ordered_entries()
        expected_chain_ids = [int_id_to_str_id(index + 1) for index in range(len(entries))]
        actual_chain_ids = [entry.structure_chain for entry in entries]
        if actual_chain_ids != expected_chain_ids:
            raise ValueError(
                "Automatic RFD3 contig generation requires sequential output chain_map "
                f"keys {expected_chain_ids}; got {actual_chain_ids}. Provide an explicit "
                "specification.contig to control non-sequential output mapping."
            )

        input_path = user_specification.get("input")
        residues_by_chain = (
            load_structure_residues(Path(str(input_path))) if input_path else {}
        )
        chain_contigs = []
        motif_assignments: list[dict[str, Any]] = []
        has_motif = False
        total_length = 0
        for entry in entries:
            protomer = global_config.protomers[entry.protomer]
            total_length += protomer.length
            components = []
            for segment_index, segment_name in enumerate(protomer.components, start=1):
                segment = global_config.segments[segment_name]
                if not segment.is_fixed:
                    components.append(str(len(segment.sequence)))
                    continue
                has_motif = True
                if not input_path:
                    raise ValueError(
                        f"Fixed seqs segment {entry.protomer}:{segment_index} requires "
                        "initializer.rfd3.specification.input so RFD3 can place its motif."
                    )
                selection, assignment = self._resolve_motif(
                    protomer_name=entry.protomer,
                    segment_index=segment_index,
                    segment_name=segment_name,
                    sequence=str(segment.sequence),
                    residues_by_chain=residues_by_chain,
                )
                components.append(selection)
                motif_assignments.append(assignment)
            chain_contigs.append(",".join(components))

        derived: dict[str, Any] = {"length": str(total_length)}
        if len(chain_contigs) > 1 or has_motif:
            derived["contig"] = ",/0,".join(chain_contigs)
        return derived, motif_assignments

    def _motif_override(
        self,
        *,
        protomer_name: str,
        segment_index: int,
        segment_name: str,
    ) -> Any:
        nested = self.motif_map.get(protomer_name)
        if isinstance(nested, dict):
            for key in (str(segment_index), segment_index, segment_name):
                if key in nested:
                    return nested[key]
        for key in (
            f"{protomer_name}:{segment_index}",
            segment_name,
        ):
            if key in self.motif_map:
                return self.motif_map[key]
        return None

    def _resolve_motif(
        self,
        *,
        protomer_name: str,
        segment_index: int,
        segment_name: str,
        sequence: str,
        residues_by_chain: dict[str, tuple[Any, ...]],
    ) -> tuple[str, dict[str, Any]]:
        override = self._motif_override(
            protomer_name=protomer_name,
            segment_index=segment_index,
            segment_name=segment_name,
        )
        if override is not None:
            selection = str(override).strip()
            candidates = []
            for chain, residues in residues_by_chain.items():
                try:
                    indices = select_structure_indices(
                        residues,
                        selection,
                        expected_chain=chain,
                    )
                except ValueError:
                    continue
                candidates.append((chain, residues, indices))
            if len(candidates) != 1:
                available = ", ".join(sorted(residues_by_chain)) or "<none>"
                raise ValueError(
                    f"initializer.rfd3.motif_map selection {selection!r} matched "
                    f"{len(candidates)} input chains; available chains: {available}."
                )
            chain, residues, indices = candidates[0]
            selected = [residues[index] for index in indices]
            observed = "".join(residue.amino_acid for residue in selected)
            if not sequence_matches(sequence, observed):
                raise ValueError(
                    f"initializer.rfd3.motif_map selection {selection!r} has sequence "
                    f"{observed!r}, incompatible with fixed seqs motif {sequence!r}."
                )
            source = "motif_map"
        else:
            matches = find_sequence_matches(residues_by_chain, sequence)
            if len(matches) != 1:
                raise ValueError(
                    f"Fixed seqs motif {protomer_name}:{segment_index} ({sequence!r}) "
                    f"matched {len(matches)} input locations; set "
                    f"initializer.rfd3.motif_map.{protomer_name}.{segment_index} to an "
                    "explicit RFD3 residue selection."
                )
            chain, indices = matches[0]
            selected = [residues_by_chain[chain][index] for index in indices]
            selection = _format_residue_selection(selected)
            observed = "".join(residue.amino_acid for residue in selected)
            source = "sequence_match"
        return selection, {
            "protomer": protomer_name,
            "segment_index": segment_index,
            "segment_name": segment_name,
            "sequence": sequence,
            "selection": selection,
            "input_sequence": observed,
            "source": source,
        }
