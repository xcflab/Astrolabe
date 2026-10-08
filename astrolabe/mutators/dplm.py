from __future__ import annotations

import json
import os
import pathlib
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import prepare_exclude_sites
from astrolabe.per_residue_scores import mutator_site_score
from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner
from astrolabe.mutators.wrappers.dplm_quality_filter import (
    build_quality_filter_config,
    filter_dplm_sequences,
 )


@dataclass
class _DplmPrefetchCache:
    combined_sequence: str
    masked_sequence: str
    mask_positions: list[int]
    mutation_quota: int
    pending_samples: list[str]


class DPLMMutator(Mutator):
    """Mutator that fills masked residues using the external DPLM diffusion language model."""

    name = "DPLM"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=False,
        outputs_probabilities=False,
        outputs_scores=True,
        supports_transition_blending=False,
        supports_shared_selection_policy=False,
    )

    def __init__(self, position_selector=None, **runner_kwargs) -> None:
        self.position_selector = position_selector
        self.runner = DPLMRunner(runner_kwargs)
        self.mask_token = "<mask>"
        self._prefetch_save = bool(runner_kwargs.get("prefetch_save", True))
        # Default: safe behavior (invalidate cache when accepted sequence changes).
        # If disabled, we will reuse pending samples across accepted steps as long as
        # the chain length and mask positions are compatible.
        self._prefetch_invalidate_on_sequence_change = bool(
            runner_kwargs.get("prefetch_invalidate_on_sequence_change", True)
        )
        self._quality_filter_enable = bool(runner_kwargs.get("quality_filter_enable", True))
        self._quality_filter_fallback_to_raw = bool(
            runner_kwargs.get("quality_filter_fallback_to_raw", True)
        )
        # Defaults are conservative: do not silently increase sampling beyond runner.num_samples.
        # If you want stricter filtering without fallback, set:
        #   quality_filter_attempt_num_samples > num_samples
        #   quality_filter_max_rounds > 1
        #   quality_filter_fallback_to_raw=false
        self._quality_filter_max_rounds = int(runner_kwargs.get("quality_filter_max_rounds", 1) or 0)
        self._quality_filter_attempt_num_samples = int(
            runner_kwargs.get("quality_filter_attempt_num_samples", 0) or 0
        )
        self._quality_cfg = build_quality_filter_config(
            runner_kwargs.get("quality_filter", {}) if isinstance(runner_kwargs.get("quality_filter", {}), dict) else {}
        )
        self._prefetch: dict[str, _DplmPrefetchCache] = {}
        self.last_mutation_telemetry: dict[str, Any] = {
            "metrics": {
                "entropy_norm": "N/A",
                "selection_score": "N/A",
                "secondary_structure": "N/A",
            },
            "oligos": {},
        }

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _per_proto_site_score(
        evaluation_scores,
        proto_name: str,
        site_index: int,
        runtime_config=None,
    ) -> float:
        return mutator_site_score(
            evaluation_scores,
            proto_name=proto_name,
            site_index=site_index,
            runtime_config=runtime_config,
        )

    def _prefetch_dir(self) -> pathlib.Path:
        return pathlib.Path(self.runner.work_dir) / "prefetch"

    def shutdown(self) -> None:
        self.runner.close()

    def _write_json_atomic(self, path: pathlib.Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(tmp), str(path))

    def _masked_sequence(self, combined_sequence: str, mask_positions: Sequence[int]) -> str:
        tokens = list(combined_sequence)
        for idx in mask_positions:
            if 0 <= idx < len(tokens):
                tokens[idx] = self.mask_token
        return "".join(tokens)

    def _request_num_samples(self) -> int:
        base = int(getattr(self.runner, "num_samples", 1) or 1)
        if not self._quality_filter_enable:
            return max(1, base)
        return max(1, base, int(self._quality_filter_attempt_num_samples or 0))

    def _quality_filter_rounds(self) -> int:
        if not self._quality_filter_enable:
            return 1
        return max(1, int(self._quality_filter_max_rounds or 0))

    def compute_mutation(self, mutation_input_features: Dict[str, Dict[str, Any]]):
        tasks: list[dict[str, Any]] = []
        cached_samples: dict[str, list[str]] = {}
        for oligo_name, features in mutation_input_features.items():
            mask_positions: Sequence[int] = features.get("mask_positions") or ()
            if not mask_positions:
                continue
            combined_sequence = str(features.get("combined_sequence") or "")
            masked_sequence = str(features.get("masked_sequence") or "")
            mutation_quota = int(features.get("mutation_quota") or len(mask_positions))

            cache = self._prefetch.get(oligo_name)
            if (
                cache is not None
                and cache.combined_sequence == combined_sequence
                and cache.masked_sequence == masked_sequence
                and cache.mutation_quota == mutation_quota
                and cache.pending_samples
            ):
                cached_samples[oligo_name] = [cache.pending_samples.pop(0)]
                continue

            num_samples = self._request_num_samples()
            tasks.append(
                {
                    "name": oligo_name,
                    "masked_sequence": masked_sequence,
                    "seed": random.randint(0, 1_000_000),
                    "num_samples": max(1, num_samples),
                }
            )

        runner_outputs: dict[str, list[str]] = {k: [] for k in [t["name"] for t in tasks]}
        qstats_by_name: dict[str, dict[str, float | int | None] | None] = {
            k: None for k in runner_outputs.keys()
        }
        # Resample until at least one sequence passes quality filter (or max rounds).
        for round_idx in range(self._quality_filter_rounds()):
            if not tasks:
                break
            raw_round = self.runner.design_sequences(tasks)
            remaining: list[dict[str, Any]] = []
            for t in tasks:
                name = t["name"]
                combined_sequence = str(mutation_input_features.get(name, {}).get("combined_sequence") or "")
                raw_samples = (raw_round.get(name) or []) if isinstance(raw_round, dict) else []
                if not self._quality_filter_enable:
                    runner_outputs[name] = raw_samples
                    qstats_by_name[name] = None
                    continue

                filtered, stats = filter_dplm_sequences(
                    raw_samples,
                    cfg=self._quality_cfg,
                    expected_length=len(combined_sequence) if combined_sequence else None,
                )
                if filtered:
                    runner_outputs[name] = filtered
                    qstats_by_name[name] = stats
                else:
                    # Keep last stats for debugging.
                    qstats_by_name[name] = stats
                    remaining.append(t)
            tasks = remaining

        step = int(getattr(global_config, "step", 0) or 0)
        if self._prefetch_save:
            prefetch_root = self._prefetch_dir()
            prefetch_root.mkdir(parents=True, exist_ok=True)

        mutation_output = {}
        per_oligo_telemetry: dict[str, dict[str, Any]] = {}
        for oligo_name, features in mutation_input_features.items():
            combined_sequence = str(features.get("combined_sequence") or "")
            masked_sequence = str(features.get("masked_sequence") or "")
            mask_positions: list[int] = list(features.get("mask_positions") or [])
            mutation_quota = int(features.get("mutation_quota") or len(mask_positions))

            out_samples: list[str] = []
            if oligo_name in cached_samples:
                out_samples = cached_samples[oligo_name]
                per_oligo_telemetry[oligo_name] = {
                    "selection_score": "N/A",
                    "entropy_norm": "N/A",
                    "secondary_structure": "N/A",
                    "stats": {},
                    "sampled": len(out_samples),
                    "mutation_quota": mutation_quota,
                    "cache_hit": True,
                }
            else:
                samples = runner_outputs.get(oligo_name, []) or []
                qstats = qstats_by_name.get(oligo_name)
                if not samples and self._quality_filter_fallback_to_raw:
                    # Last-resort fallback: accept raw output from the last round (if any).
                    # This preserves the historical behavior (always returns a sample) but should
                    # be rare if num_samples/max_rounds are set reasonably.
                    raw_last = self.runner.design_sequences(
                        [
                            {
                                "name": oligo_name,
                                "masked_sequence": masked_sequence,
                                "seed": random.randint(0, 1_000_000),
                                "num_samples": max(1, int(getattr(self.runner, "num_samples", 1) or 1)),
                            }
                        ]
                    )
                    samples = raw_last.get(oligo_name, []) or []
                if samples:
                    out_samples = [samples[0]]
                leftover = list(samples[1:]) if len(samples) > 1 else []
                # Keep mask positions stable while the accepted sequence is unchanged.
                self._prefetch[oligo_name] = _DplmPrefetchCache(
                    combined_sequence=combined_sequence,
                    masked_sequence=masked_sequence,
                    mask_positions=mask_positions,
                    mutation_quota=mutation_quota,
                    pending_samples=leftover,
                )
                if self._prefetch_save and samples:
                    requested = self._request_num_samples()
                    dump = {
                        "step": step,
                        "oligo_name": oligo_name,
                        "combined_sequence": combined_sequence,
                        "masked_sequence": masked_sequence,
                        "mask_positions": mask_positions,
                        "mutation_quota": mutation_quota,
                        "num_samples": len(samples),
                        "runner_num_samples": int(getattr(self.runner, "num_samples", 1) or 1),
                        "requested_num_samples": int(requested),
                        "samples": samples,
                        "prefetch_invalidate_on_sequence_change": self._prefetch_invalidate_on_sequence_change,
                        "quality_filter_enable": self._quality_filter_enable,
                        "quality_filter_max_rounds": self._quality_filter_max_rounds,
                        "quality_filter_attempt_num_samples": self._quality_filter_attempt_num_samples,
                        "quality_filter_fallback_to_raw": self._quality_filter_fallback_to_raw,
                    }
                    if qstats is not None:
                        dump["quality_filter_stats"] = qstats
                    out_path = self._prefetch_dir() / f"step_{step:03d}_{oligo_name}_{uuid.uuid4().hex[:8]}.json"
                    self._write_json_atomic(out_path, dump)

                selection_score = None
                entropy_norm = None
                if isinstance(qstats, dict):
                    raw_count = self._to_float(qstats.get("raw"))
                    kept_count = self._to_float(qstats.get("kept"))
                    if raw_count and raw_count > 0 and kept_count is not None:
                        selection_score = kept_count / raw_count
                    entropy_norm = self._to_float(qstats.get("entropy_norm_mean"))
                per_oligo_telemetry[oligo_name] = {
                    "selection_score": selection_score,
                    "entropy_norm": entropy_norm,
                    "secondary_structure": "N/A",
                    "stats": qstats or {},
                    "sampled": len(out_samples),
                    "mutation_quota": mutation_quota,
                }

            mutation_output[oligo_name] = {
                "feature_dict": features,
                "name": oligo_name,
                "samples": out_samples,
                "quality_filter_stats": qstats_by_name.get(oligo_name),
            }
        entropy_values = [
            float(item["entropy_norm"])
            for item in per_oligo_telemetry.values()
            if isinstance(item.get("entropy_norm"), (int, float))
        ]
        selection_scores = [
            float(item["selection_score"])
            for item in per_oligo_telemetry.values()
            if isinstance(item.get("selection_score"), (int, float))
        ]
        self.last_mutation_telemetry = {
            "metrics": {
                "entropy_norm": (
                    float(sum(entropy_values) / len(entropy_values))
                    if entropy_values
                    else "N/A"
                ),
                "selection_score": (
                    float(sum(selection_scores) / len(selection_scores))
                    if selection_scores
                    else "N/A"
                ),
                "secondary_structure": "N/A",
            },
            "oligos": per_oligo_telemetry,
        }
        return mutation_output

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

        protein_sequences = {
            name: protomer.sequence
            for name, protomer in protomers.items()
            if protomer.type == ProtomerTypes.PROTEIN
        }

        mutation_inputs: Dict[str, Dict[str, Any]] = {}
        for oligo_name in evaluation_scores.names:
            combined_tokens: List[str] = []
            mapping: List[Tuple[str, int]] = []
            for proto in oligo_name:
                sequence = protein_sequences.get(proto)
                if not sequence:
                    continue
                for idx, residue in enumerate(sequence):
                    combined_tokens.append(residue)
                    mapping.append((proto, idx))

            if not combined_tokens:
                continue

            available_indices = []
            for idx, (proto, proto_idx) in enumerate(mapping):
                excluded = excluded_positions.get(proto, [])
                if proto_idx in excluded:
                    continue
                available_indices.append(idx)

            if not available_indices:
                continue

            mutation_quota = min(self.n_mutations(), len(available_indices))
            if mutation_quota <= 0:
                continue

            combined_sequence = "".join(combined_tokens)
            cache = self._prefetch.get(oligo_name)
            if cache is not None:
                # Always invalidate if length/quota changed (samples no longer align).
                if len(cache.combined_sequence) != len(combined_sequence) or cache.mutation_quota != int(
                    mutation_quota
                ):
                    cache = None
                    self._prefetch.pop(oligo_name, None)
                # Conditionally invalidate if accepted sequence changed.
                elif cache.combined_sequence != combined_sequence and self._prefetch_invalidate_on_sequence_change:
                    cache = None
                    self._prefetch.pop(oligo_name, None)

            mask_positions: list[int]
            if cache is not None:
                mask_positions = list(cache.mask_positions)
                avail = set(int(x) for x in available_indices)
                if (
                    len(mask_positions) != int(mutation_quota)
                    or any(int(x) not in avail for x in mask_positions)
                ):
                    cache = None
                    self._prefetch.pop(oligo_name, None)

            if cache is None:
                mask_positions = (
                    np.random.choice(
                        available_indices,
                        size=mutation_quota,
                        replace=False,
                    )
                    .astype(int)
                    .tolist()
                )
            else:
                # Keep cached mask positions; update cache metadata to the current sequence.
                # This allows optional reuse of pending samples across accepted steps.
                cache.combined_sequence = combined_sequence
                cache.masked_sequence = self._masked_sequence(combined_sequence, mask_positions)

            mutation_inputs[oligo_name] = {
                "masked_sequence": self._masked_sequence(combined_sequence, mask_positions),
                "combined_sequence": combined_sequence,
                "mask_positions": sorted(mask_positions),
                "mutation_quota": int(mutation_quota),
                "mapping": mapping,
            }

        return mutation_inputs

    def prepare_mutation_output(
        self,
        mutation_output: Dict[str, Dict[str, Any]],
        protomers: Dict[str, Protomer],
        evaluation_scores,
        runtime_config,
    ):
        planned_mutations: List[Tuple[str, Tuple[int, str, str, float, float]]] = []

        for oligo_name, output in mutation_output.items():
            metadata = output.get("feature_dict", {})
            mask_positions: Sequence[int] = metadata.get("mask_positions", [])
            mapping: Sequence[Tuple[str, int]] = metadata.get("mapping", [])
            original = metadata.get("combined_sequence")
            samples: Sequence[str] = output.get("samples") or ()
            qstats = output.get("quality_filter_stats")
            entropy_value = 0.0
            if isinstance(qstats, dict):
                raw_entropy = qstats.get("entropy_norm_mean")
                if isinstance(raw_entropy, (int, float)):
                    entropy_value = float(raw_entropy)
            if not mask_positions or not mapping or not original or not samples:
                continue

            sampled_sequence = samples[0]
            if len(sampled_sequence) != len(mapping):
                continue

            for idx in mask_positions:
                if idx >= len(mapping):
                    continue
                proto, proto_idx = mapping[idx]
                protomer = protomers.get(proto)
                if protomer is None:
                    continue
                new_residue = sampled_sequence[idx]
                old_residue = original[idx]
                if new_residue == old_residue:
                    continue
                segment_info = self._map_global_to_segment(protomer, proto_idx)
                if segment_info is None:
                    continue
                segment_name, local_idx = segment_info
                site_score = self._per_proto_site_score(
                    evaluation_scores,
                    proto_name=proto,
                    site_index=proto_idx,
                    runtime_config=runtime_config,
                )
                planned_mutations.append(
                    (
                        segment_name,
                        (local_idx, old_residue, new_residue, entropy_value, site_score),
                    )
                )

        if not planned_mutations:
            return mutation_output

        mutation_budget = getattr(global_config, "mutation_num", None)
        if mutation_budget is None:
            mutation_limit = len(planned_mutations)
        else:
            mutation_limit = max(
                0,
                min(
                    len(planned_mutations),
                    int(round(mutation_budget * getattr(self, "mutation_weight", 1.0))),
                ),
            )
        if mutation_limit > 0:
            global_config.segment_manager.add_mutations(
                planned_mutations,
                mutation_limit,
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
