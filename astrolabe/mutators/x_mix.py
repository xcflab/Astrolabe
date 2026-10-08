from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import prepare_exclude_sites
from astrolabe.utils.constant import aa_order


CANONICAL_AMINO_ACIDS = tuple(str(aa).upper() for aa in aa_order)


def _normalize_aa_set(raw_values) -> list[str]:
    if raw_values is None:
        raise ValueError(
            "XMix mutator `aa_set` must be a string, list, or {'residues': ...} when provided."
        )

    values = raw_values
    if isinstance(values, dict):
        if "residues" not in values:
            raise ValueError("XMix mutator aa_set mapping must contain `residues`.")
        values = values.get("residues")

    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            raise ValueError("XMix mutator aa_set cannot be empty.")
        tokens = raw.replace(",", " ").split()
        residues = list(tokens[0]) if len(tokens) == 1 else tokens
    elif isinstance(values, (list, tuple, set)):
        residues = list(values)
    else:
        raise ValueError("XMix mutator aa_set must be a string, list, or dict.")

    normalized: list[str] = []
    seen: set[str] = set()
    for residue in residues:
        aa = str(residue or "").strip().upper()
        if not aa:
            continue
        if len(aa) != 1 or aa not in CANONICAL_AMINO_ACIDS:
            raise ValueError(
                f"Invalid amino acid in XMix mutator aa_set: {residue!r}."
            )
        if aa in seen:
            continue
        seen.add(aa)
        normalized.append(aa)

    if not normalized:
        raise ValueError("XMix mutator aa_set has no valid amino acids after normalization.")
    return normalized


def _resolve_x_fraction(raw_value: Any) -> float:
    try:
        fraction = float(0.8 if raw_value is None else raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("XMix mutator x_fraction must be numeric.") from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"XMix mutator x_fraction must be within [0, 1], got {raw_value!r}.")
    return fraction


def _resolve_residue_set(raw_aa_set) -> list[str]:
    if raw_aa_set is not None:
        return _normalize_aa_set(raw_aa_set)

    excluded = {
        str(residue or "").strip().upper()
        for residue in (getattr(global_config, "exclude_AA", []) or [])
        if str(residue or "").strip()
    }
    residues = [aa for aa in CANONICAL_AMINO_ACIDS if aa not in excluded]
    if not residues:
        raise ValueError(
            "XMix mutator has no available amino acids after exclude_AA filtering."
        )
    return residues


class XMixMutator(Mutator):
    name = "XMix"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=False,
        outputs_probabilities=False,
        outputs_scores=False,
        supports_transition_blending=False,
        supports_shared_selection_policy=False,
    )

    def __init__(
        self,
        *,
        x_fraction: float = 0.8,
        aa_set=None,
        position_selector=None,
    ) -> None:
        self.position_selector = position_selector
        self.x_fraction = _resolve_x_fraction(x_fraction)
        self.residue_set = _resolve_residue_set(aa_set)
        self.last_mutation_telemetry: dict[str, Any] = {
            "metrics": {
                "entropy_norm": "N/A",
                "selection_score": "N/A",
                "secondary_structure": "N/A",
            },
            "protomers": {},
        }

    def prepare_mutation_input(
        self,
        protomers: Dict[str, Protomer],
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )
        mutation_inputs: dict[str, dict[str, Any]] = {}

        for proto_name, protomer in protomers.items():
            if protomer.type != ProtomerTypes.PROTEIN:
                continue
            raw_mutable_mask = getattr(protomer, "mutable_mask", None)
            if raw_mutable_mask is None:
                mutable_mask = []
            else:
                mutable_mask = list(np.asarray(raw_mutable_mask).astype(bool).tolist())
            excluded = set(excluded_positions.get(proto_name, []))
            available_positions = [
                idx
                for idx, is_mutable in enumerate(mutable_mask)
                if bool(is_mutable) and idx not in excluded
            ]
            if not available_positions:
                continue

            mutation_quota = min(self.n_mutations(), len(available_positions))
            if mutation_quota <= 0:
                continue

            sampled_positions = (
                np.random.choice(
                    available_positions,
                    size=mutation_quota,
                    replace=False,
                )
                .astype(int)
                .tolist()
            )
            mutation_inputs[proto_name] = {
                "sequence": protomer.sequence,
                "sampled_positions": sorted(sampled_positions),
            }

        return mutation_inputs

    def compute_mutation(self, mutation_input_features):
        mutation_output = {}
        per_proto_telemetry: dict[str, dict[str, Any]] = {}

        for proto_name, input_feat in mutation_input_features.items():
            sequence = str(input_feat.get("sequence") or "")
            sampled_positions = list(input_feat.get("sampled_positions") or [])
            tokens = list(sequence)

            x_count = 0
            for pos in sampled_positions:
                if random.random() < self.x_fraction:
                    tokens[pos] = "X"
                    x_count += 1
                else:
                    tokens[pos] = random.choice(self.residue_set)

            mutation_output[proto_name] = {
                "name": proto_name,
                "feature_dict": input_feat,
                "sampled_sequence": "".join(tokens),
            }
            per_proto_telemetry[proto_name] = {
                "sampled": len(sampled_positions),
                "x_count": x_count,
                "x_fraction_observed": (x_count / len(sampled_positions)) if sampled_positions else 0.0,
            }

        self.last_mutation_telemetry = {
            "metrics": {
                "entropy_norm": "N/A",
                "selection_score": "N/A",
                "secondary_structure": "N/A",
            },
            "protomers": per_proto_telemetry,
        }
        return mutation_output

    def prepare_mutation_output(
        self,
        mutation_output,
        protomers,
        evaluation_scores,
        runtime_config,
    ):
        planned_mutations: List[Tuple[str, Tuple[int, str, str, float, float]]] = []

        for proto_name, output in mutation_output.items():
            protomer = protomers.get(proto_name)
            if protomer is None:
                continue
            metadata = output.get("feature_dict", {})
            original_sequence = str(metadata.get("sequence") or "")
            sampled_sequence = str(output.get("sampled_sequence") or "")
            sampled_positions = list(metadata.get("sampled_positions") or [])
            if not original_sequence or not sampled_sequence or not sampled_positions:
                continue

            for pos in sampled_positions:
                if pos >= len(original_sequence) or pos >= len(sampled_sequence):
                    continue
                old_residue = original_sequence[pos]
                new_residue = sampled_sequence[pos]
                if old_residue == new_residue:
                    continue
                segment_info = self._map_global_to_segment(protomer, int(pos))
                if segment_info is None:
                    continue
                segment_name, local_idx = segment_info
                planned_mutations.append(
                    (
                        segment_name,
                        (local_idx, old_residue, new_residue, 0.0, 0.0),
                    )
                )

        if planned_mutations:
            global_config.segment_manager.add_mutations(
                planned_mutations,
                len(planned_mutations),
            )
        return mutation_output

    @staticmethod
    def _map_global_to_segment(
        protomer: Protomer, global_idx: int
    ) -> Optional[Tuple[str, int]]:
        for name, seg_slice in zip(protomer.components, protomer.components_slices):
            if seg_slice.start <= global_idx < seg_slice.stop:
                return name, global_idx - seg_slice.start
        return None
