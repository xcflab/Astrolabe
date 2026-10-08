from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_RNA_DNA, MutatorCapabilities
from astrolabe.mutators.utils import (
    aggregate_per_proto_scores,
    collect_blended_proto_probabilities,
    prepare_exclude_sites,
)
from astrolabe.structure_processors.base import run_nested_predocker_on_scores

logger = logging.getLogger(__name__)


class NAMPNNMutator(Mutator):
    """Mixed protein/RNA/DNA mutator backed by baker-laboratory/NA-MPNN."""

    name = "NAMPNN"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_RNA_DNA,
        needs_structure=True,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
    )
    supported_types = (
        ProtomerTypes.PROTEIN,
        ProtomerTypes.RNA,
        ProtomerTypes.DNA,
    )
    _protein_order = (
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    )
    _rna_order = ("A", "C", "G", "U")
    _dna_order = ("A", "C", "G", "T")
    _protein_tokens = {
        "A": "A",
        "R": "R",
        "N": "N",
        "D": "D",
        "C": "C",
        "Q": "Q",
        "E": "E",
        "G": "G",
        "H": "H",
        "I": "I",
        "L": "L",
        "K": "K",
        "M": "M",
        "F": "F",
        "P": "P",
        "S": "S",
        "T": "T",
        "W": "W",
        "Y": "Y",
        "V": "V",
        "X": "X",
    }
    _rna_tokens = {
        "b": "A",
        "d": "C",
        "h": "G",
        "u": "U",
        "a": "A",
        "c": "C",
        "g": "G",
        "t": "U",
        "x": "N",
        "y": "N",
    }
    _dna_tokens = {
        "a": "A",
        "c": "C",
        "g": "G",
        "t": "T",
        "b": "A",
        "d": "C",
        "h": "G",
        "u": "T",
        "x": "N",
        "y": "N",
    }

    def __init__(
        self,
        sequence_model_runner,
        position_selector=None,
        max_mutations_per_proto=None,
        predocker=None,
    ) -> None:
        self.runner = sequence_model_runner
        self.position_selector = position_selector
        self.max_mutations_per_proto = max_mutations_per_proto
        self.predocker = predocker

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config,
    ) -> Dict[str, Dict[str, object]]:
        if self.predocker is not None:
            evaluation_scores = run_nested_predocker_on_scores(
                self.predocker,
                evaluation_scores,
            )

        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )
        ignored = set(global_config.ignored_complexes or [])
        ignored.update(global_config.negative_design_complexes or [])
        mutation_input_features: Dict[str, Dict[str, object]] = {}

        for oligo_name in evaluation_scores.names:
            if oligo_name in ignored:
                continue
            complex_path = evaluation_scores.complex_paths.get(oligo_name)
            if complex_path is None:
                raise ValueError(f"NAMPNN requires structure_path for oligomer {oligo_name!r}.")

            proto_list: list[str] = []
            designable_proto_list: list[str] = []
            fixed_residues: list[str] = []
            excluded_by_proto: dict[str, set[int]] = {}

            for proto in oligo_name:
                proto_obj = protomers.get(proto)
                if proto_obj is None:
                    raise ValueError(f"Unknown protomer {proto!r} referenced by oligomer {oligo_name!r}.")
                if proto_obj.type not in self.supported_types:
                    raise ValueError(
                        "NAMPNN only supports protein/RNA/DNA protomers. "
                        f"Found unsupported protomer {proto!r} (type={proto_obj.type}) "
                        f"in oligomer {oligo_name!r}."
                    )

                proto_list.append(proto)
                excluded = set(int(pos) for pos in (excluded_positions.get(proto, []) or []))
                excluded_by_proto[proto] = excluded

                mutable_mask = self._mutable_mask(proto_obj)
                designable = []
                for idx in range(proto_obj.length):
                    can_design = bool(mutable_mask[idx]) and idx not in excluded
                    if can_design:
                        designable.append(idx)
                    else:
                        fixed_residues.append(f"{proto}{idx + 1}")
                if designable:
                    designable_proto_list.append(proto)

            if not proto_list or not designable_proto_list:
                continue

            designed_types = {protomers[proto].type for proto in designable_proto_list}
            if (
                getattr(self.runner, "na_shared_tokens", 0)
                and ProtomerTypes.RNA in designed_types
                and ProtomerTypes.DNA in designed_types
            ):
                raise ValueError(
                    "NAMPNN cannot design RNA and DNA in the same call when `na_shared_tokens=1`. "
                    "Set `mutator.NAMPNN.na_shared_tokens=0`."
                )

            mutation_input_features[oligo_name] = {
                "structure_path": complex_path,
                "protos": proto_list,
                "chains_to_design": ",".join(designable_proto_list),
                "fixed_residues": fixed_residues,
                "parse_these_chains_only": "".join(proto_list),
                "excluded": excluded_by_proto,
            }

        return mutation_input_features

    def compute_mutation(
        self, mutation_input_features: Dict[str, Dict[str, object]]
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        mutation_output: Dict[str, Dict[str, Dict[str, object]]] = {}

        for oligo_name, job in mutation_input_features.items():
            configured_job_name = getattr(self.runner, "job_name", None)
            if not configured_job_name:
                raise ValueError(
                    "NAMPNN requires externally configured `mutator.NAMPNN.job_name`."
                )
            design_result = self.runner.design_sequences(
                structure_path=job["structure_path"],
                job_name=configured_job_name,
                chains_to_design=job["chains_to_design"],
                fixed_residues=job["fixed_residues"],
                parse_these_chains_only=job["parse_these_chains_only"],
            )

            probability_map = self._extract_probability_map(
                design_result.get("probabilities"),
                job["protos"],
            )
            designed_protos = self._normalize_proto_list(job.get("chains_to_design"))
            designed_protos = [proto for proto in job["protos"] if proto in designed_protos]
            if not designed_protos:
                designed_protos = list(job["protos"])
            output_protos = [
                proto
                for proto in designed_protos
                if probability_map.get(proto) is not None and probability_map[proto].shape[1] > 0
            ]
            if not output_protos:
                output_protos = designed_protos

            sequence_map = self._resolve_sequence_map(
                str(design_result["sequence"]),
                all_protos=job["protos"],
                output_protos=output_protos,
            )

            proto_sequences: Dict[str, Dict[str, object]] = {}
            for proto in output_protos:
                proto_entry: Dict[str, object] = {
                    "sequence": sequence_map[proto],
                    "excluded": set(job["excluded"].get(proto, set())),
                }
                probs = probability_map.get(proto)
                if probs is not None:
                    proto_entry["sampling_probs"] = probs
                proto_sequences[proto] = proto_entry

            if proto_sequences:
                mutation_output[oligo_name] = {"proto_sequences": proto_sequences}

        return mutation_output

    def prepare_mutation_output(
        self,
        mutation_output,
        protomers,
        evaluation_scores,
        runtime_config,
    ):
        if not mutation_output:
            return None

        proto_names = self._collect_proto_names(mutation_output)
        budgets = self._compute_mutation_budgets(proto_names)
        queued_from_probs = self._schedule_probabilistic_mutations(
            mutation_output,
            evaluation_scores,
            budgets,
            runtime_config,
        )
        queued_total = sum(queued_from_probs.values())
        queued_total += self._backfill_sequence_mutations(
            mutation_output,
            budgets,
            queued_from_probs,
        )
        logger.info("Queued %d NAMPNN-driven mutations.", queued_total)
        return None

    def n_mutations(self):
        return super().n_mutations()

    def _collect_proto_names(self, mutation_output) -> set[str]:
        names: set[str] = set()
        for oligo_output in mutation_output.values():
            proto_map = oligo_output.get("proto_sequences", {})
            if isinstance(proto_map, dict):
                names.update(proto_map.keys())
        return names

    def _schedule_probabilistic_mutations(
        self,
        mutation_output,
        evaluation_scores: StandardScores,
        budgets: dict[str, int],
        runtime_config,
    ) -> dict[str, int]:
        for proto, scores in aggregate_per_proto_scores(
            evaluation_scores,
            runtime_config=runtime_config,
        ).items():
            protomer = global_config.protomers.get(proto)
            if protomer is None or protomer.type not in self.supported_types:
                continue
            scores_array = np.asarray(scores)
            if scores_array.ndim == 0:
                continue
            if scores_array.shape[-1] != protomer.length:
                logger.debug(
                    "Skipping score propagation for %s: got length %s, expected %s",
                    proto,
                    scores_array.shape[-1],
                    protomer.length,
                )
                continue
            protomer.propagating_scores(scores_array)
        excluded_by_proto: dict[str, set[int]] = defaultdict(set)
        for proto, data in self._iter_proto_entries(mutation_output):
            excluded_by_proto[proto].update(set(data.get("excluded", set())))

        proto_probabilities = collect_blended_proto_probabilities(
            evaluation_scores,
            mutation_output,
            protomer_type=self.supported_types,
            runtime_config=runtime_config,
        )
        for proto, probs in proto_probabilities.items():
            tensor = self._ensure_probability_tensor(
                proto,
                probs,
                expected_length=global_config.protomers[proto].length,
            )
            tensor = self._apply_exclusion_masks(
                proto,
                tensor,
                excluded_positions=excluded_by_proto.get(proto, set()),
            )
            if tensor.shape[1] == 0:
                continue
            global_config.protomers[proto].propagating_probs(tensor.cpu().numpy())

        candidate_mutations = list(global_config.segment_manager.collect_mutations())
        if not candidate_mutations:
            return {}
        candidate_mutations = self._sort_and_shuffle_mutations(candidate_mutations)
        candidate_mutations = self._apply_budget_limits(candidate_mutations, budgets)
        if not candidate_mutations:
            return {}

        n_mutations = max(min(self.n_mutations(), len(candidate_mutations)), 0)
        global_config.segment_manager.add_mutations(candidate_mutations, n_mutations)
        return self._count_queued_by_proto()

    def _backfill_sequence_mutations(
        self,
        mutation_output,
        budgets: dict[str, int],
        queued_from_probs: dict[str, int],
    ) -> int:
        total = 0
        for proto, data in self._iter_proto_entries(mutation_output):
            budget = budgets.get(proto, 0)
            if budget <= 0:
                continue
            if queued_from_probs.get(proto, 0) > 0:
                continue
            total += self._queue_proto_sequence_mutations(
                proto,
                designed_sequence=str(data.get("sequence", "")),
                excluded_positions=set(data.get("excluded", set())),
                budget=budget,
            )
        return total

    def _queue_proto_sequence_mutations(
        self,
        proto: str,
        designed_sequence: str,
        excluded_positions: Set[int],
        budget: Optional[int],
    ) -> int:
        protomer = global_config.protomers.get(proto)
        if protomer is None or protomer.type not in self.supported_types:
            return 0
        reference_sequence = protomer.sequence
        if len(reference_sequence) != len(designed_sequence):
            raise ValueError(
                f"NAMPNN output length mismatch for proto {proto!r}: "
                f"expected {len(reference_sequence)}, got {len(designed_sequence)}."
            )
        allowed = self._combine_budget_limits(proto, budget)
        if allowed == 0:
            return 0

        candidates: list[tuple[str, tuple[int, str, str, float, float]]] = []
        allowed_residues = set(self._alphabet_for_type(protomer.type))
        for idx, (ref_res, new_res) in enumerate(zip(reference_sequence, designed_sequence)):
            if idx in excluded_positions or new_res == ref_res:
                continue
            if new_res not in allowed_residues:
                continue
            segment_name, local_index = self._resolve_segment_position(protomer, idx)
            if segment_name is None:
                continue
            candidates.append(
                (segment_name, (local_index, ref_res, new_res, 0.0, 0.0))
            )
            if allowed is not None and len(candidates) >= allowed:
                break
        if not candidates:
            return 0
        global_config.segment_manager.add_mutations(candidates, len(candidates))
        return len(candidates)

    def _iter_proto_entries(self, mutation_output):
        for oligo_output in mutation_output.values():
            proto_map = oligo_output.get("proto_sequences", {})
            if not isinstance(proto_map, dict):
                continue
            for proto, data in proto_map.items():
                if isinstance(data, dict):
                    yield proto, data

    def _count_queued_by_proto(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for proto_name, protomer in global_config.protomers.items():
            total = 0
            for segment_name in protomer.components:
                segment = global_config.segments[segment_name]
                total += len(segment.mutations or [])
            if total:
                counts[proto_name] = total
        return counts

    def _sort_and_shuffle_mutations(self, mutations):
        sorted_mutations = sorted(mutations, key=lambda item: item[1][-2])
        to_shuffle = []
        not_shuffle = []
        for mutation in sorted_mutations:
            if mutation[1][-2] < 1.5:
                to_shuffle.append(mutation)
            else:
                not_shuffle.append(mutation)
        np.random.shuffle(to_shuffle)
        return to_shuffle + not_shuffle

    def _apply_budget_limits(self, mutations, budgets: dict[str, int]):
        segment_to_proto = {}
        for proto_name, proto in global_config.protomers.items():
            for segment_name in proto.components:
                segment_to_proto[segment_name] = proto_name

        proto_usage = defaultdict(int)
        budgeted = []
        for segment_name, mutation in mutations:
            proto_name = segment_to_proto.get(segment_name)
            if proto_name is None:
                continue
            limit = self._combine_budget_limits(proto_name, budgets.get(proto_name))
            if limit is not None and proto_usage[proto_name] >= limit:
                continue
            budgeted.append((segment_name, mutation))
            proto_usage[proto_name] += 1
        return budgeted

    def _ensure_probability_tensor(
        self,
        proto: str,
        probs,
        *,
        expected_length: int,
    ) -> torch.Tensor:
        if isinstance(probs, torch.Tensor):
            tensor = probs.detach().clone().to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(probs, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                f"NAMPNN probabilities for proto {proto!r} must have rank 2 or 3, got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[1] == 0:
            return tensor.cpu()
        if tensor.shape[1] != expected_length:
            raise ValueError(
                f"NAMPNN probabilities for proto {proto!r} have length {tensor.shape[1]}, expected {expected_length}."
            )
        tensor = torch.nan_to_num(tensor, nan=0.0)
        row_sums = tensor.sum(dim=-1, keepdim=True)
        tensor = torch.where(row_sums > 0, tensor / row_sums, torch.zeros_like(tensor))
        return tensor.cpu()

    def _apply_exclusion_masks(
        self,
        proto: str,
        probs: torch.Tensor,
        *,
        excluded_positions: Set[int],
    ) -> torch.Tensor:
        if probs.shape[1] == 0 or not excluded_positions:
            return probs
        protomer = global_config.protomers[proto]
        alphabet = self._alphabet_for_type(protomer.type)
        alpha_index = {aa: idx for idx, aa in enumerate(alphabet)}
        for pos in excluded_positions:
            if pos < 0 or pos >= probs.shape[1]:
                continue
            residue = self._normalize_residue_for_type(protomer.sequence[pos], protomer.type)
            idx = alpha_index.get(residue)
            if idx is None:
                continue
            probs[:, pos, :] = 0.0
            probs[:, pos, idx] = 1.0
        return probs

    def _compute_mutation_budgets(self, proto_names: Iterable[str]) -> dict[str, int]:
        total_budget = max(self.n_mutations(), 0)
        protos = {
            proto
            for proto in proto_names
            if (proto_obj := global_config.protomers.get(proto)) is not None
            and proto_obj.type in self.supported_types
            and proto_obj.mutable_length > 0
        }
        if not protos:
            return {}
        if total_budget == 0:
            return {proto: 0 for proto in protos}
        if len(protos) == 1:
            proto = next(iter(protos))
            return {proto: self._combine_budget_limits(proto, total_budget) or 0}

        lengths = {proto: max(global_config.protomers[proto].mutable_length, 0) for proto in protos}
        total_length = sum(lengths.values())
        if total_length <= 0:
            return {proto: self._combine_budget_limits(proto, 0) or 0 for proto in protos}

        ordered = sorted(protos)
        shares = {proto: total_budget * lengths[proto] / total_length for proto in ordered}
        budgets = {proto: int(shares[proto]) for proto in ordered}
        allocated = sum(budgets.values())
        remaining = total_budget - allocated
        fractional_order = sorted(
            ordered,
            key=lambda proto: (shares[proto] - int(shares[proto]), lengths[proto]),
            reverse=True,
        )
        eligible = [
            proto
            for proto in fractional_order
            if self._combine_budget_limits(proto, budgets[proto] + 1) != budgets[proto]
        ]
        idx = 0
        while remaining > 0 and eligible:
            proto = eligible[idx % len(eligible)]
            budgets[proto] += 1
            remaining -= 1
            if self._combine_budget_limits(proto, budgets[proto] + 1) == budgets[proto]:
                eligible = [name for name in eligible if name != proto]
            idx += 1
        return {
            proto: max(self._combine_budget_limits(proto, value) or 0, 0)
            for proto, value in budgets.items()
        }

    def _convert_sequence_map(self, raw_sequence: str, protos: Sequence[str]) -> dict[str, str]:
        raw_sequence = (raw_sequence or "").strip()
        if not raw_sequence:
            raise ValueError("NAMPNN returned an empty sequence string.")

        if "/" in raw_sequence:
            chains = raw_sequence.split("/")
            if len(chains) != len(protos):
                raise ValueError(
                    "NAMPNN output chain count does not match protomer count: "
                    f"got {len(chains)} chains, expected {len(protos)}."
                )
            split_sequences = chains
        else:
            split_sequences = self._split_by_length(raw_sequence, protos)

        converted = {}
        for proto, seq in zip(protos, split_sequences):
            proto_type = global_config.protomers[proto].type
            converted[proto] = self._translate_sequence(seq, proto_type)
        return converted

    def _resolve_sequence_map(
        self,
        raw_sequence: str,
        *,
        all_protos: Sequence[str],
        output_protos: Sequence[str],
    ) -> dict[str, str]:
        last_error: ValueError | None = None
        candidates: list[tuple[Sequence[str], bool]] = []
        if output_protos:
            candidates.append((output_protos, False))
        if list(all_protos) != list(output_protos):
            candidates.append((all_protos, True))

        for candidate_protos, subset_to_output in candidates:
            try:
                sequence_map = self._convert_sequence_map(raw_sequence, candidate_protos)
            except ValueError as exc:
                last_error = exc
                continue
            if subset_to_output:
                return {proto: sequence_map[proto] for proto in output_protos}
            return sequence_map

        assert last_error is not None
        raise last_error

    def _split_by_length(self, seq: str, protos: Sequence[str]) -> list[str]:
        total_needed = sum(global_config.protomers[proto].length for proto in protos)
        if len(seq) != total_needed:
            raise ValueError(
                f"NAMPNN output length {len(seq)} does not match expected {total_needed}."
            )
        parts: list[str] = []
        offset = 0
        for proto in protos:
            length = global_config.protomers[proto].length
            parts.append(seq[offset : offset + length])
            offset += length
        return parts

    def _translate_sequence(self, seq: str, proto_type: ProtomerTypes) -> str:
        if proto_type == ProtomerTypes.PROTEIN:
            mapping = self._protein_tokens
        elif proto_type == ProtomerTypes.RNA:
            mapping = self._rna_tokens
        elif proto_type == ProtomerTypes.DNA:
            mapping = self._dna_tokens
        else:
            raise ValueError(f"Unsupported NAMPNN protomer type: {proto_type}")
        translated = []
        for char in seq:
            translated.append(mapping.get(char, mapping.get(char.lower(), char.upper())))
        return "".join(translated)

    def _extract_probability_map(
        self,
        probability_bundle,
        protos: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if probability_bundle is None:
            return {}
        if isinstance(probability_bundle, torch.Tensor):
            if len(protos) != 1:
                raise ValueError("Legacy tensor NAMPNN probabilities only support single-proto outputs.")
            return {protos[0]: self._ensure_probability_tensor(protos[0], probability_bundle, expected_length=global_config.protomers[protos[0]].length)}
        if not isinstance(probability_bundle, dict):
            raise TypeError(
                f"NAMPNN probability bundle must be a dict or tensor, got {type(probability_bundle).__name__}."
            )

        predicted_ppm = np.asarray(probability_bundle.get("predicted_ppm"), dtype=np.float32)
        restype_to_int = probability_bundle.get("restype_to_int")
        if hasattr(restype_to_int, "item"):
            restype_to_int = restype_to_int.item()
        if not isinstance(restype_to_int, dict):
            raise ValueError("NAMPNN specificity bundle missing `restype_to_int` mapping.")

        encoded_residues = probability_bundle.get("encoded_residues")
        if encoded_residues is None:
            encoded_residues = []
        if hasattr(encoded_residues, "tolist"):
            encoded_residues = encoded_residues.tolist()
        encoded_residues = [str(item) for item in encoded_residues]

        row_indices_by_proto = defaultdict(list)
        for row_idx, encoded in enumerate(encoded_residues):
            if not encoded:
                continue
            row_indices_by_proto[encoded[0]].append(row_idx)

        if not row_indices_by_proto and predicted_ppm.ndim == 2:
            offset = 0
            for proto in protos:
                length = global_config.protomers[proto].length
                row_indices_by_proto[proto] = list(range(offset, offset + length))
                offset += length

        probability_map: dict[str, torch.Tensor] = {}
        for proto in protos:
            proto_type = global_config.protomers[proto].type
            column_names = self._probability_columns_for_type(proto_type)
            rows = row_indices_by_proto.get(proto, [])
            if not rows:
                probability_map[proto] = torch.zeros(
                    (1, 0, len(column_names)), dtype=torch.float32
                )
                continue
            missing = [name for name in column_names if name not in restype_to_int]
            if missing:
                raise ValueError(
                    f"NAMPNN specificity bundle missing columns for proto {proto!r}: {missing}"
                )
            cols = [restype_to_int[name] for name in column_names]
            proto_probs = predicted_ppm[np.asarray(rows, dtype=np.int64)][:, cols]
            probability_map[proto] = torch.from_numpy(proto_probs).unsqueeze(0).to(dtype=torch.float32)
        return probability_map

    def _normalize_proto_list(self, protos: Sequence[str] | str | None) -> list[str]:
        if protos is None:
            return []
        if isinstance(protos, str):
            items = protos.split(",")
        else:
            items = protos
        return [str(item).strip() for item in items if str(item).strip()]

    def _mutable_mask(self, protomer) -> np.ndarray:
        return np.asarray(protomer.mutable_mask, dtype=bool)

    def _max_mutations_for_proto(self, proto: str) -> Optional[int]:
        if self.max_mutations_per_proto is None:
            return None
        if isinstance(self.max_mutations_per_proto, dict):
            value = self.max_mutations_per_proto.get(proto)
        else:
            value = self.max_mutations_per_proto
        if value is None:
            return None
        return max(int(value), 0)

    def _combine_budget_limits(
        self,
        proto: str,
        runtime_budget: Optional[int],
    ) -> Optional[int]:
        configured = self._max_mutations_for_proto(proto)
        if runtime_budget is None:
            return configured
        if configured is None:
            return max(runtime_budget, 0)
        return max(min(runtime_budget, configured), 0)

    def _resolve_segment_position(self, protomer, absolute_index: int):
        offset = 0
        for segment_name in protomer.components:
            segment = global_config.segments[segment_name]
            seg_len = len(segment.sequence)
            if absolute_index < offset + seg_len:
                return segment_name, absolute_index - offset
            offset += seg_len
        return None, None

    def _alphabet_for_type(self, proto_type: ProtomerTypes) -> tuple[str, ...]:
        if proto_type == ProtomerTypes.PROTEIN:
            return self._protein_order
        if proto_type == ProtomerTypes.RNA:
            return self._rna_order
        if proto_type == ProtomerTypes.DNA:
            return self._dna_order
        raise ValueError(f"Unsupported NAMPNN protomer type: {proto_type}")

    def _probability_columns_for_type(self, proto_type: ProtomerTypes) -> tuple[str, ...]:
        if proto_type == ProtomerTypes.PROTEIN:
            return (
                "ALA",
                "CYS",
                "ASP",
                "GLU",
                "PHE",
                "GLY",
                "HIS",
                "ILE",
                "LYS",
                "LEU",
                "MET",
                "ASN",
                "PRO",
                "GLN",
                "ARG",
                "SER",
                "THR",
                "VAL",
                "TRP",
                "TYR",
            )
        if proto_type == ProtomerTypes.RNA:
            return ("A", "C", "G", "U")
        if proto_type == ProtomerTypes.DNA:
            return ("DA", "DC", "DG", "DT")
        raise ValueError(f"Unsupported NAMPNN protomer type: {proto_type}")

    def _normalize_residue_for_type(self, residue: str, proto_type: ProtomerTypes) -> str:
        residue = str(residue or "").strip().upper()
        if proto_type == ProtomerTypes.RNA and residue == "T":
            return "U"
        return residue
