import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import MutableTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.selectors.utils import construct_position_selector
from astrolabe.selectors.sse import _proto_secstruct_map


@dataclass
class MotifTarget:
    proto: str
    start: int
    length: int
    motif: str
    context: str  # "helix" or "loop"


class MotifBreakerMutator(Mutator):
    """Force-inserts helix/loop breaking motifs to disrupt long secondary structure stretches."""

    name = "MotifBreaker"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=True,
        outputs_probabilities=False,
        outputs_scores=False,
        supports_transition_blending=False,
        supports_shared_selection_policy=False,
    )

    def __init__(
        self,
        helix_selector: str = "long_helix_core()",
        loop_selector: str = "long_loop_terminal()",
        helix_min_run: int = 17,
        loop_min_run: int = 10,
        proline_loop_fraction: float = 0.45,
        helix_break_motif: str = "GP",
        loop_break_motif: str = "GS",
        max_targets_per_proto: int = 2,
        max_mutations: int | None = None,
    ) -> None:
        self.helix_selector = helix_selector
        self.loop_selector = loop_selector
        self.helix_min_run = max(1, int(helix_min_run))
        self.loop_min_run = max(1, int(loop_min_run))
        self.proline_loop_fraction = np.clip(proline_loop_fraction, 0.0, 1.0)
        self.helix_break_motif = helix_break_motif.upper()
        self.loop_break_motif = loop_break_motif.upper()
        self.max_targets_per_proto = max(1, int(max_targets_per_proto))
        self.max_mutations = (
            int(max_mutations)
            if (max_mutations is not None and int(max_mutations) > 0)
            else None
        )

    def compute_mutation(self, mutation_input_features):
        # Deterministic mutator; computation already performed in prepare_mutation_input.
        return mutation_input_features

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        targets: Dict[str, List[MotifTarget]] = {p: [] for p in protomers.keys()}
        secstruct_map = _proto_secstruct_map(evaluation_scores.complex_poses)
        if self.helix_selector not in (None, "None"):
            helix_masks = construct_position_selector(
                self.helix_selector,
                evaluation_scores,
                protomers,
            )
            helix_context_masks = self._context_masks_from_ss(
                protomers,
                secstruct_map,
                target_ss="H",
            )
            self._collect_targets(
                targets,
                protomers,
                helix_masks,
                helix_context_masks,
                min_run=self.helix_min_run,
                motif=self.helix_break_motif,
                context="helix",
            )
        if self.loop_selector not in (None, "None"):
            loop_masks = construct_position_selector(
                self.loop_selector,
                evaluation_scores,
                protomers,
            )
            self._collect_targets(
                targets,
                protomers,
                loop_masks,
                self._context_masks_from_ss(
                    protomers,
                    secstruct_map,
                    target_ss="L",
                ),
                min_run=self.loop_min_run,
                motif=self.loop_break_motif,
                context="loop",
                require_proline=True,
            )

        return {"motif_targets": targets}

    def prepare_mutation_output(
        self,
        mutation_output,
        protomers,
        evaluation_scores,
        runtime_config,
    ):
        targets: Dict[str, List[MotifTarget]] = mutation_output.get("motif_targets", {})
        planned_mutations: List[tuple[str, tuple[int, str, str, float, float]]] = []
        total_mutations_allowed = (
            None if self.max_mutations is None else int(self.max_mutations)
        )
        for proto_name, proto_targets in targets.items():
            protomer = protomers.get(proto_name)
            if protomer is None or protomer.type not in MutableTypes:
                continue
            for target in proto_targets[: self.max_targets_per_proto]:
                if total_mutations_allowed is not None and len(planned_mutations) >= total_mutations_allowed:
                    break
                new_mutations = self._build_segment_mutations(protomer, target)
                if not new_mutations:
                    continue
                if total_mutations_allowed is not None:
                    remaining_budget = total_mutations_allowed - len(planned_mutations)
                    if remaining_budget <= 0:
                        break
                    if len(new_mutations) > remaining_budget:
                        new_mutations = new_mutations[:remaining_budget]
                planned_mutations.extend(new_mutations)
                if total_mutations_allowed is not None and len(planned_mutations) >= total_mutations_allowed:
                    break
        if planned_mutations:
            limit = (
                len(planned_mutations)
                if total_mutations_allowed is None
                else min(len(planned_mutations), total_mutations_allowed)
            )
            global_config.segment_manager.add_mutations(
                planned_mutations,
                limit,
            )
        return mutation_output

    def _collect_targets(
        self,
        targets: Dict[str, List[MotifTarget]],
        protomers,
        selector_masks,
        context_masks,
        min_run: int,
        motif: str,
        context: str,
        require_proline: bool = False,
    ) -> None:
        motif_len = len(motif)
        if motif_len == 0:
            return
        for proto_name, mask in selector_masks.items():
            protomer = protomers[proto_name]
            if protomer.type not in MutableTypes:
                continue
            mutable_mask = protomer.mutable_mask
            context_mask = context_masks.get(
                proto_name, np.zeros_like(mask, dtype=bool)
            )
            if context_mask.shape[0] != protomer.length:
                context_mask = np.resize(context_mask, protomer.length)
            runs = self._find_runs(
                np.logical_and(context_mask, mutable_mask),
                min_run,
            )
            for start, end in runs:
                if (end - start) < motif_len:
                    continue
                if not mask[start:end].any():
                    continue
                if require_proline and not self._passes_proline_threshold(
                    protomer.sequence[start:end],
                    self.proline_loop_fraction,
                ):
                    continue
                center_start = start + (end - start - motif_len) // 2
                target = MotifTarget(
                    proto=proto_name,
                    start=center_start,
                    length=motif_len,
                    motif=motif,
                    context=context,
                )
                targets[proto_name].append(target)

    @staticmethod
    def _find_runs(mask: np.ndarray, min_run: int) -> List[tuple[int, int]]:
        runs = []
        start = None
        for idx, flag in enumerate(mask):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                if idx - start >= min_run:
                    runs.append((start, idx))
                start = None
        if start is not None and len(mask) - start >= min_run:
            runs.append((start, len(mask)))
        return runs

    @staticmethod
    def _passes_proline_threshold(segment: str, threshold: float) -> bool:
        if not segment:
            return False
        prolines = segment.count("P")
        return prolines / len(segment) >= threshold

    def _build_segment_mutations(self, protomer, target: MotifTarget):
        mutations = []
        remaining = target.length
        motif_chars = list(target.motif)
        for offset in range(target.length):
            global_pos = target.start + offset
            seg_name, seg_idx = self._map_global_to_segment(protomer, global_pos)
            if seg_name is None:
                continue
            segment = global_config.segments[seg_name]
            current_res = segment.sequence[seg_idx]
            new_res = motif_chars[offset % len(motif_chars)]
            if current_res == new_res:
                continue
            priority = 2.5 if target.context == "helix" else 2.0
            mutations.append(
                (
                    seg_name,
                    (seg_idx, current_res, new_res, priority, 0.0),
                )
            )
        return mutations

    @staticmethod
    def _map_global_to_segment(protomer, global_pos: int):
        for name, seg_slice in zip(protomer.components, protomer.components_slices):
            if seg_slice.start <= global_pos < seg_slice.stop:
                return name, global_pos - seg_slice.start
        return None, None

    def _context_masks_from_ss(
        self, protomers, proto2ss, target_ss: str
    ) -> Dict[str, np.ndarray]:
        masks = {}
        for proto_name, protomer in protomers.items():
            length = protomer.length
            ss_strings = proto2ss.get(proto_name, [])
            if not ss_strings:
                masks[proto_name] = np.zeros(length, dtype=bool)
                continue
            per_chain = []
            for ss in ss_strings:
                arr = np.zeros(length, dtype=bool)
                for idx, char in enumerate(ss[:length]):
                    if char == target_ss:
                        arr[idx] = True
                per_chain.append(arr)
            masks[proto_name] = np.any(per_chain, axis=0)
        return masks
