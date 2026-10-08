import logging
from typing import Dict, Any, Optional, Sequence

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import MutableTypes, ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import MUTABLE_TYPE_SET, MutatorCapabilities
from astrolabe.mutators.utils import (
    prepare_exclude_sites,
    propagate_evaluation_scores_to_protomers,
)
from astrolabe.utils.constant import cached_aa_freq, restype_str_to_int


def generate_aa_freqs(
    sequence_length,
    type: ProtomerTypes = ProtomerTypes.PROTEIN,
    allowed_residues=None,
    disallowed_residues=None,
):
    if type == ProtomerTypes.PROTEIN:
        aa_freq = cached_aa_freq(exclude=global_config.exclude_AA)
        allowed = {aa.upper() for aa in allowed_residues} if allowed_residues else None
        disallowed = (
            {aa.upper() for aa in disallowed_residues}
            if disallowed_residues
            else set()
        )
        adjusted_freq = {}
        for aa, freq in aa_freq.items():
            if allowed is not None and aa not in allowed:
                continue
            if aa in disallowed:
                continue
            adjusted_freq[aa] = freq
        if not adjusted_freq:
            raise ValueError(
                "No amino acids remain after applying allowed/disallowed residue filters."
            )
        total = sum(adjusted_freq.values())
        aa_freq = {aa: freq / total for aa, freq in adjusted_freq.items()}
        aa_chance_array = np.zeros([20])
        for aa, freq in aa_freq.items():
            aa_chance_array[restype_str_to_int[aa]] = freq
        aa_freqs = np.repeat(
            aa_chance_array[np.newaxis, :],
            sequence_length,
            axis=0,
        )[np.newaxis, :]
    elif type == ProtomerTypes.RNA:
        aa_freqs = np.zeros([1, sequence_length, 4])
        aa_freqs[...] = 0.25  # Uniform distribution for RNA bases
    elif type == ProtomerTypes.DNA:
        aa_freqs = np.zeros([1, sequence_length, 4])
        aa_freqs[...] = 0.25
    else:
        return None
    return aa_freqs


class RandomSubstitutionMutator(Mutator):
    """Propose residue substitutions; acceptance is handled by the decision layer.

    Protein proposals use adjusted amino-acid frequencies; nucleic-acid proposals
    use uniform base probabilities. The serialized configuration key stays MCMC.
    """

    name: str = "MCMC"
    capabilities = MutatorCapabilities(
        supported_types=MUTABLE_TYPE_SET,
        needs_structure=False,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=False,
        supports_shared_selection_policy=True,
    )

    def __init__(
        self,
        mutation_method,
        select_positions,
        position_selector=None,
        predocker=None,
        allowed_residues=None,
        disallowed_residues=None,
        max_mutations=None,
        avoid_current_residue=False,
    ):
        self.mutation_method = mutation_method
        self.select_positions = select_positions
        self.enabled_types = MutableTypes
        self.position_selector = position_selector
        self.allowed_residues = (
            [aa.upper() for aa in allowed_residues]
            if allowed_residues
            else None
        )
        self.disallowed_residues = (
            [aa.upper() for aa in disallowed_residues]
            if disallowed_residues
            else None
        )
        self.max_mutations = (
            int(max_mutations)
            if (max_mutations is not None and int(max_mutations) > 0)
            else None
        )
        self.avoid_current_residue = bool(avoid_current_residue)

    def compute_mutation(self, mutation_input_features):
        # select positions to mutate
        protomers = global_config.protomers
        excluded_positions = mutation_input_features.get("excluded_pos_per_proto", {})

        for name, protomer in protomers.items():
            if protomer.type not in self.enabled_types:
                continue
            aa_probs = generate_aa_freqs(
                protomer.length,
                type=protomer.type,
                allowed_residues=self.allowed_residues
                if protomer.type == ProtomerTypes.PROTEIN
                else None,
                disallowed_residues=self.disallowed_residues
                if protomer.type == ProtomerTypes.PROTEIN
                else None,
            )
            aa_probs = self._apply_excluded_positions(
                protomer,
                self._apply_avoid_current_residue(
                    protomer,
                    aa_probs,
                    excluded_positions.get(name, ()),
                ),
                excluded_positions.get(name, ()),
            )
            protomer.propagating_probs(aa_probs)

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        for name, segment in global_config.segments.items():
            segment.clean_up()
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )

        return {
            "excluded_pos_per_proto": excluded_positions,
        }

    @staticmethod
    def _apply_excluded_positions(
        protomer,
        aa_probs: Optional[np.ndarray],
        excluded_positions: Sequence[int],
    ):
        if aa_probs is None or not excluded_positions:
            return aa_probs

        base_lookup = {
            ProtomerTypes.PROTEIN: restype_str_to_int,
            ProtomerTypes.RNA: {"A": 0, "C": 1, "G": 2, "U": 3},
            ProtomerTypes.DNA: {"A": 0, "C": 1, "G": 2, "T": 3},
        }.get(protomer.type)
        if base_lookup is None:
            return aa_probs

        seq = protomer.sequence
        alphabet_size = aa_probs.shape[-1]
        for pos in excluded_positions:
            if pos >= len(seq) or pos < 0:
                continue
            residue = seq[pos].upper()
            residue_idx = base_lookup.get(residue)
            if residue_idx is None or residue_idx >= alphabet_size:
                continue
            aa_probs[:, pos, :] = 0.0
            aa_probs[:, pos, residue_idx] = 1.0
        return aa_probs

    def _apply_avoid_current_residue(
        self,
        protomer,
        aa_probs: Optional[np.ndarray],
        excluded_positions: Sequence[int],
    ):
        if aa_probs is None or not self.avoid_current_residue:
            return aa_probs
        if protomer.type != ProtomerTypes.PROTEIN:
            return aa_probs

        excluded = {int(pos) for pos in excluded_positions}
        for pos, residue in enumerate(str(protomer.sequence)):
            if pos in excluded:
                continue
            residue_idx = restype_str_to_int.get(residue.upper())
            if residue_idx is None or residue_idx >= aa_probs.shape[-1]:
                continue
            if float(aa_probs[:, pos, residue_idx].sum()) <= 0.0:
                continue
            aa_probs[:, pos, residue_idx] = 0.0
            row_sums = aa_probs[:, pos, :].sum(axis=-1, keepdims=True)
            if np.any(row_sums <= 0.0):
                aa_probs[:, pos, residue_idx] = 1.0
            else:
                aa_probs[:, pos, :] = aa_probs[:, pos, :] / row_sums
        return aa_probs

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        propagate_evaluation_scores_to_protomers(evaluation_scores, runtime_config)

        mutations = global_config.segment_manager.collect_mutations()

        def sort_by_score(mutations):
            """
            Sort mutations by entropy, descending.
            """
            return sorted(
                mutations,
                key=lambda item: item[1][-1],
            )

        mutations = sort_by_score(mutations)

        n_mutations = round(global_config.mutation_num * self.mutation_weight)
        if self.max_mutations is not None:
            n_mutations = min(n_mutations, self.max_mutations)

        global_config.segment_manager.add_mutations(mutations, n_mutations)

        # apply mutation based on the mutation_output
        return mutation_output


class PoreBlockingSubstitutionMutator(RandomSubstitutionMutator):
    name: str = "PoreBlockerMCMC"

    def __init__(
        self,
        position_selector,
        target_residues="GAST",
        mutation_method="frequency_adjusted",
        select_positions="plddt",
        allowed_residues=None,
        avoid_current_residue=True,
        active_steps=None,
        active_step_interval=None,
        active_step_offset=0,
        **kwargs,
    ):
        if position_selector in (None, "None", ""):
            raise ValueError(
                "pore_blocker_mcmc requires a position_selector, usually pore_bottleneck(...)."
            )
        if allowed_residues is not None:
            target_residues = allowed_residues
        if isinstance(target_residues, str):
            allowed = [aa for aa in target_residues.upper() if aa.strip()]
        else:
            allowed = [
                str(aa).strip().upper()
                for aa in target_residues
                if str(aa).strip()
            ]
        if not allowed:
            raise ValueError("pore_blocker_mcmc target_residues cannot be empty.")
        self.active_steps = self._parse_active_steps(active_steps)
        self.active_step_interval = self._parse_optional_positive_int(
            active_step_interval,
            field_name="active_step_interval",
        )
        self.active_step_offset = int(active_step_offset or 0)
        self._active_invocations = 0
        self.last_mutation_telemetry: dict[str, Any] = {"metrics": {}}
        super().__init__(
            mutation_method=mutation_method,
            select_positions=select_positions,
            position_selector=position_selector,
            allowed_residues=allowed,
            avoid_current_residue=avoid_current_residue,
            **kwargs,
        )

    @staticmethod
    def _parse_active_steps(active_steps) -> set[int] | None:
        if active_steps in (None, "", "None"):
            return None
        if isinstance(active_steps, str):
            tokens = [
                token.strip()
                for token in active_steps.replace(";", ",").split(",")
                if token.strip()
            ]
        else:
            tokens = list(active_steps)
        parsed = {int(token) for token in tokens}
        return parsed or None

    @staticmethod
    def _parse_optional_positive_int(value, *, field_name: str) -> int | None:
        if value in (None, "", "None", 0, "0"):
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be a positive integer when set.")
        return parsed

    def _current_step(self) -> int:
        step = getattr(global_config, "step", None)
        if step is None:
            return 0
        return int(step)

    def _is_active_step(self, step: int) -> bool:
        if self.active_steps is not None:
            return int(step) in self.active_steps
        if self.active_step_interval is not None:
            return (
                int(step) >= self.active_step_offset
                and (int(step) - self.active_step_offset) % self.active_step_interval
                == 0
            )
        return True

    @staticmethod
    def _exclude_all_positions(protomers) -> dict[str, list[int]]:
        excluded = {}
        for proto, protomer in protomers.items():
            if protomer.type not in MutableTypes:
                continue
            excluded[proto] = list(range(int(protomer.length)))
        return excluded

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        for name, segment in global_config.segments.items():
            segment.clean_up()
        step = self._current_step()
        if not self._is_active_step(step):
            self.last_mutation_telemetry = {
                "metrics": {
                    "pore_blocker_active": False,
                    "pore_blocker_step": step,
                    "pore_blocker_reason": "inactive_step",
                }
            }
            return {
                "excluded_pos_per_proto": self._exclude_all_positions(protomers),
                "pore_blocker_active": False,
            }

        global_config.set_variable(
            "pore_bottleneck_activation_index",
            int(self._active_invocations),
        )
        self.last_mutation_telemetry = {
            "metrics": {
                "pore_blocker_active": True,
                "pore_blocker_step": step,
                "pore_blocker_activation_index": int(self._active_invocations),
            }
        }
        self._active_invocations += 1
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            fallback_to_all_mutable_if_selector_empty=False,
        )

        return {
            "excluded_pos_per_proto": excluded_positions,
            "pore_blocker_active": True,
        }

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        propagate_evaluation_scores_to_protomers(evaluation_scores, runtime_config)

        mutations = global_config.segment_manager.collect_mutations()
        mutations = sorted(
            mutations,
            key=lambda item: item[1][-1],
        )

        if self.last_mutation_telemetry.get("metrics", {}).get(
            "pore_blocker_active"
        ) is False:
            global_config.segment_manager.add_mutations(mutations, 0)
            return mutation_output

        n_mutations = len(mutations)
        if self.max_mutations is not None:
            n_mutations = min(n_mutations, self.max_mutations)

        self.last_mutation_telemetry.setdefault("metrics", {})[
            "pore_blocker_candidate_mutations"
        ] = len(mutations)
        self.last_mutation_telemetry["metrics"][
            "pore_blocker_queued_mutations"
        ] = n_mutations
        global_config.segment_manager.add_mutations(mutations, n_mutations)
        return mutation_output
