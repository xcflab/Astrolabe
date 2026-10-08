from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from astrolabe.mutators.base import Mutator
from astrolabe.mutators.ligandmpnn import GroupedLigandMPNNMutator, LigandMPNNMutator


class _NoSequenceRunner:
    def design_sequences(self, **_kwargs):
        raise RuntimeError("This helper only prepares GH structure features; it is not a sequence model.")


class DynamicMPNNMutator(Mutator):
    """Mutator for the real Alex-Abrudan/DynamicMPNN backend.

    This class does not subclass or run LigandMPNN.  It reuses GH's existing
    structure-selection/fixed-position plumbing through a helper object, then sends
    alias-grouped conformations to a real DynamicMPNN runner that must return one
    joint `sampling_probs` tensor for the conformational group.
    """

    name = "DynamicMPNN"
    capabilities = replace(
        GroupedLigandMPNNMutator.capabilities,
        supports_initialization_from_structure=False,
    )

    def __init__(self, sequence_model_runner, **kwargs):
        self.runner = sequence_model_runner
        self._feature_helper = LigandMPNNMutator(
            sequence_model_runner=_NoSequenceRunner(), **kwargs
        )
        self._group_helper = GroupedLigandMPNNMutator(
            sequence_model_runner=_NoSequenceRunner(), **kwargs
        )
        self.aa_order = self._feature_helper.aa_order
        self.entropy_cutoff = getattr(self._feature_helper, "entropy_cutoff", None)

    def prepare_mutation_input(
        self,
        protomer,
        evaluation_scores,
        runtime_config: Dict[str, Any],
    ):
        return self._feature_helper.prepare_mutation_input(
            protomer, evaluation_scores, runtime_config
        )

    def prepare_mutation_output(
        self, mutation_output, protomers, evaluation_scores, runtime_config
    ):
        self._feature_helper.mutation_weight = getattr(self, "mutation_weight", 1.0)
        self._feature_helper.prepare_mutation_output(
            mutation_output, protomers, evaluation_scores, runtime_config
        )

    def compute_mutation(self, mutation_input_features):
        primary_features = {
            oligo_name: self._group_helper._primary_features(input_feat)
            for oligo_name, input_feat in mutation_input_features.items()
        }
        outputs = {}
        for group in self._group_helper._group_shared_protomer_features(primary_features):
            if len(group) == 1:
                raise ValueError(
                    "DynamicMPNN requires at least two synchronized conformations. "
                    f"Ungrouped oligo {group[0][0]!r} has no shared @alias partner."
                )
            outputs.update(self._run_multi_state_group(group))
        for oligo_name, output in outputs.items():
            output.update(
                {
                    "name": oligo_name,
                    "feature_dict": primary_features[oligo_name],
                }
            )
        return outputs

    def _run_multi_state_group(
        self,
        group: list[tuple[str, dict[str, object]]],
    ) -> dict[str, dict[str, object]]:
        payload_dir = self._group_helper._multi_state_payload_dir()
        group_name = self._group_helper._safe_group_name(
            "_".join(oligo for oligo, _ in group)
        )
        pdb_paths = [str(features["pdb_path"]) for _, features in group]
        fixed_residues = {
            path: self._group_helper._split_residue_string(
                features.get("fixed_residues", "")
            )
            for path, (_, features) in zip(pdb_paths, group)
        }
        redesigned_residues = {
            path: self._group_helper._split_residue_string(
                features.get("redesigned_residues", "")
            )
            for path, (_, features) in zip(pdb_paths, group)
        }

        multi_features = self._group_helper._runner_features(group[0][1])
        for key in (
            "pdb_path",
            "pdb_path_multi",
            "fixed_residues",
            "fixed_residues_multi",
            "redesigned_residues",
            "redesigned_residues_multi",
            "symmetry_residues",
            "symmetry_weights",
            "transmembrane_interface",
            "transmembrane_interface_multi",
            "transmembrane_buried",
            "transmembrane_buried_multi",
        ):
            multi_features.pop(key, None)
        multi_features["pdb_path_multi"] = str(
            self._group_helper._write_group_json(
                payload_dir, group_name, "pdb_paths", pdb_paths
            )
        )
        multi_features["fixed_residues_multi"] = str(
            self._group_helper._write_group_json(
                payload_dir, group_name, "fixed_residues", fixed_residues
            )
        )
        if any(redesigned_residues.values()):
            multi_features["redesigned_residues_multi"] = str(
                self._group_helper._write_group_json(
                    payload_dir,
                    group_name,
                    "redesigned_residues",
                    redesigned_residues,
                )
            )

        output = self.runner.design_sequences(**multi_features)
        if not isinstance(output, dict) or "sampling_probs" not in output:
            raise RuntimeError(
                "DynamicMPNN backend must return a dict containing unified sampling_probs."
            )
        return {oligo_name: dict(output) for oligo_name, _features in group}
