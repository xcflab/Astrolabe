from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Set, Tuple

import torch

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.utils import prepare_exclude_sites, sample_mutations_by_aa_probs

logger = logging.getLogger(__name__)


class NucleicAcidDesignMutator(Mutator):
    """Common scheduling utilities for nucleic-acid design mutators.

    Subclasses are responsible for producing either:
    - per-position base probabilities (`sampling_probs`), OR
    - an explicit designed `sequence` per proto.
    """

    name = "NucleicAcidDesign"
    target_type = ProtomerTypes.RNA

    def __init__(
        self,
        position_selector=None,
        max_mutations_per_proto: Optional[int | Dict[str, int]] = None,
    ) -> None:
        self.position_selector = position_selector
        self.max_mutations_per_proto = max_mutations_per_proto
        self.alphabet_order = tuple(self._alphabet())
        self._alphabet_index = {base.upper(): idx for idx, base in enumerate(self.alphabet_order)}

    def prepare_mutation_output(
        self,
        mutation_output: Dict[str, Dict[str, Dict[str, object]]],
        protomers,
        evaluation_scores,
        runtime_config,
    ):
        if not mutation_output:
            return None
        if self._has_sampling_probs(mutation_output):
            self._schedule_probabilistic_mutations(
                mutation_output, protomers, evaluation_scores, runtime_config
            )
        else:
            self._schedule_sequence_mutations(mutation_output)
        return None

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def _normalize_sequence(self, sequence: str) -> str:
        """Subclasses may override to translate model-specific alphabets."""
        return sequence.upper()

    def _alphabet(self) -> Tuple[str, ...]:  # pragma: no cover - constant
        return ("A", "C", "G", "U")

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _schedule_probabilistic_mutations(
        self,
        mutation_output,
        protomers,
        evaluation_scores,
        runtime_config,
    ) -> None:
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )
        proto_names = self._collect_proto_names(mutation_output)
        budgets = self._compute_mutation_budgets(proto_names)

        for oligo_name, oligo_output in mutation_output.items():
            probs = self._extract_sampling_probs(oligo_output)
            if probs is None:
                continue
            probs = self._ensure_probability_tensor(probs)
            probs = self._apply_exclusion_masks(
                probs, oligo_name, protomers, excluded_positions
            )
            oligo_output["sampling_probs"] = probs

        sample_mutations_by_aa_probs(
            evaluation_scores,
            mutation_output,
            self.alphabet_order,
            getattr(self, "mutation_weight", 1.0),
            protomer_type=self.target_type,
            per_proto_limits=budgets,
            propagate_scores=False,
            runtime_config=runtime_config,
        )

    def _schedule_sequence_mutations(self, mutation_output) -> None:
        proto_names = self._collect_proto_names(mutation_output)
        budgets = self._compute_mutation_budgets(proto_names)
        total_mutations = 0
        for proto_map in self._iter_proto_maps(mutation_output):
            for proto, data in proto_map.items():
                sequence = self._normalize_sequence(data.get("sequence", ""))
                excluded = set(data.get("excluded", set()))
                allowed = budgets.get(proto)
                mutated = self._queue_proto_mutations(proto, sequence, excluded, allowed)
                total_mutations += mutated
        logger.info("Queued %d %s-driven mutations.", total_mutations, self.name)

    def _queue_proto_mutations(
        self,
        proto: str,
        designed_sequence: str,
        excluded_positions: Set[int],
        budget: Optional[int],
    ) -> int:
        protomer = global_config.protomers.get(proto)
        if protomer is None:
            logger.warning("Proto %s not found in global configuration.", proto)
            return 0
        if protomer.type != self.target_type:
            return 0
        reference_sequence = protomer.sequence
        if len(reference_sequence) != len(designed_sequence):
            logger.warning(
                "Length mismatch for proto %s (ref=%d, design=%d). Skipping.",
                proto,
                len(reference_sequence),
                len(designed_sequence),
            )
            return 0
        allowed = self._combine_budget_limits(proto, budget)
        if allowed == 0:
            return 0

        candidates: list[Tuple[str, Tuple[int, str, str, float, float]]] = []
        for idx, (ref_res, new_res) in enumerate(
            zip(reference_sequence, designed_sequence)
        ):
            if idx in excluded_positions or new_res == ref_res:
                continue
            if new_res not in {"A", "C", "G", "U"}:
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

    def _has_sampling_probs(self, mutation_output) -> bool:
        for oligo_output in mutation_output.values():
            if isinstance(oligo_output, dict) and "sampling_probs" in oligo_output:
                return True
        return False

    def _extract_proto_map(self, oligo_output) -> Optional[Dict[str, Dict[str, object]]]:
        if not isinstance(oligo_output, dict):
            return None
        if "proto_sequences" in oligo_output and isinstance(oligo_output["proto_sequences"], dict):
            return oligo_output["proto_sequences"]

        proto_like = {}
        for key, value in oligo_output.items():
            if isinstance(value, dict) and "sequence" in value:
                proto_like[key] = value
        if proto_like and len(proto_like) == len(oligo_output):
            return proto_like
        return None

    def _iter_proto_maps(self, mutation_output):
        for oligo_output in mutation_output.values():
            proto_map = self._extract_proto_map(oligo_output)
            if proto_map:
                yield proto_map

    def _collect_proto_names(self, mutation_output) -> Set[str]:
        names: Set[str] = set()
        for proto_map in self._iter_proto_maps(mutation_output):
            names.update(proto_map.keys())
        return names

    def _extract_sampling_probs(self, oligo_output):
        if not isinstance(oligo_output, dict):
            return None
        return oligo_output.get("sampling_probs")

    def _ensure_probability_tensor(self, probs):
        if isinstance(probs, torch.Tensor):
            tensor = probs.detach().clone()
        else:
            tensor = torch.as_tensor(probs, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != len(self.alphabet_order):
            raise ValueError(
                f"Expected probability dimension {len(self.alphabet_order)}, got {tensor.shape[-1]}"
            )
        tensor = torch.clamp(tensor, min=1e-8)
        tensor = tensor / tensor.sum(dim=-1, keepdim=True)
        return tensor.cpu()

    def _apply_exclusion_masks(
        self,
        probs: torch.Tensor,
        oligo_name: str,
        protomers,
        excluded_positions,
    ) -> torch.Tensor:
        offset = 0
        for proto in oligo_name:
            proto_obj = protomers.get(proto)
            if proto_obj is None or proto_obj.type != self.target_type:
                continue
            proto_len = proto_obj.length
            if offset + proto_len > probs.shape[1]:
                logger.warning(
                    "Probability tensor shorter than expected for proto %s (len=%d).",
                    proto,
                    proto_len,
                )
                break
            frozen = excluded_positions.get(proto, [])
            for pos in frozen:
                if pos >= proto_len:
                    continue
                column = self._base_to_index(proto_obj.sequence[pos])
                if column is None:
                    continue
                probs[:, offset + pos, :] = 0.0
                probs[:, offset + pos, column] = 1.0
            offset += proto_len
        return probs

    def _resolve_segment_position(self, protomer, absolute_index: int):
        offset = 0
        for segment_name in protomer.components:
            segment = global_config.segments[segment_name]
            seg_len = len(segment.sequence)
            if absolute_index < offset + seg_len:
                return segment_name, absolute_index - offset
            offset += seg_len
        return None, None

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
        self, proto: str, runtime_budget: Optional[int]
    ) -> Optional[int]:
        configured = self._max_mutations_for_proto(proto)
        if runtime_budget is None:
            return configured
        if configured is None:
            return max(runtime_budget, 0)
        return max(min(runtime_budget, configured), 0)

    def _compute_mutation_budgets(self, proto_names: Iterable[str]):
        total_budget = max(self.n_mutations(), 0)
        protos = {proto for proto in proto_names if self._is_target_proto(proto)}
        if not protos:
            return {}
        if total_budget == 0:
            return {proto: 0 for proto in protos}
        if len(protos) == 1:
            return {next(iter(protos)): total_budget}

        lengths = {proto: global_config.protomers[proto].length for proto in protos}
        total_length = sum(lengths.values())
        if total_length == 0:
            return {proto: 0 for proto in protos}

        ordered = sorted(protos)
        shares = {proto: total_budget * lengths[proto] / total_length for proto in ordered}
        budgets = {proto: int(shares[proto]) for proto in ordered}
        allocated = sum(budgets.values())
        remaining = total_budget - allocated
        fractional_order = sorted(ordered, key=lambda p: shares[p] - int(shares[p]), reverse=True)
        idx = 0
        while remaining > 0 and fractional_order:
            proto = fractional_order[idx % len(fractional_order)]
            budgets[proto] += 1
            remaining -= 1
            idx += 1

        for proto, value in budgets.items():
            budgets[proto] = max(value, 0)
        return budgets

    def _is_target_proto(self, proto: str) -> bool:
        proto_obj = global_config.protomers.get(proto)
        return proto_obj is not None and proto_obj.type == self.target_type

    def _base_to_index(self, residue: str) -> Optional[int]:
        base = residue.upper()
        if base == "T" and "U" in self._alphabet_index:
            base = "U"
        return self._alphabet_index.get(base)
