import itertools
import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from astrolabe.utils.mmcif_ids import int_id_to_str_id

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.structure_inputs import prepare_structure_input_scores
from astrolabe.mutators.utils import (
    prepare_exclude_sites,
    probabilities_to_numpy,
    sample_mutations_by_aa_probs,
)
from astrolabe.structure_processors.base import run_nested_predocker_on_scores
from astrolabe.selectors.utils import construct_position_selector
from astrolabe.utils.constant import aa_order


class LigandMPNNMutator(Mutator):
    name = "LigandMPNN"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=True,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
        supports_initialization_from_structure=True,
    )

    def __init__(
        self,
        sequence_model_runner,
        position_selector=None,
        fallback_to_all_mutable_if_selector_empty=True,
        membrane_selector=None,
        membrane_label="interface",
        membrane_label_selectors=None,
        predocker=None,
        supplemental_structures=None,
        supplemental_structure_weight=1.0,
        entropy_cutoff=None,
        enforce_mutation_num=False,
        map_sequence_positions_to_pdb_residues=False,
    ):
        self.runner = sequence_model_runner
        self.position_selector = position_selector
        self.fallback_to_all_mutable_if_selector_empty = bool(
            fallback_to_all_mutable_if_selector_empty
        )
        self.membrane_selector = membrane_selector
        self.membrane_label = str(membrane_label or "interface").strip().lower()
        if self.membrane_label not in {"interface", "buried"}:
            raise ValueError("membrane_label must be 'interface' or 'buried'.")
        self.membrane_label_selectors = self._normalize_membrane_label_selectors(
            membrane_label_selectors
        )
        self.predocker = predocker
        self.supplemental_structures = supplemental_structures
        self.supplemental_structure_weight = float(supplemental_structure_weight)
        if self.supplemental_structure_weight < 0.0:
            raise ValueError("supplemental_structure_weight must be non-negative.")
        self.entropy_cutoff = entropy_cutoff
        self.enforce_mutation_num = bool(enforce_mutation_num)
        self.map_sequence_positions_to_pdb_residues = bool(
            map_sequence_positions_to_pdb_residues
        )
        self.aa_order = aa_order[:20]  #

    @staticmethod
    def _normalize_membrane_label_selectors(raw) -> dict[str, str]:
        if raw is None:
            return {}
        if isinstance(raw, str) and raw in ("", "None"):
            return {}
        if not hasattr(raw, "items"):
            raise ValueError(
                "membrane_label_selectors must be a mapping of label to selector."
            )
        result: dict[str, str] = {}
        for label, expression in raw.items():
            label_name = str(label).strip().lower()
            if label_name not in {"interface", "buried"}:
                raise ValueError(
                    "membrane_label_selectors labels must be 'interface' or 'buried'."
                )
            if expression in (None, "", "None"):
                continue
            result[label_name] = str(expression)
        return result

    def compute_mutation(self, mutation_input_features: Tuple[Any, Any]):
        mutation_output = {}
        for oligo_name, input_feat in mutation_input_features.items():
            input_feat = dict(input_feat)
            supplemental_inputs = input_feat.pop("supplemental_inputs", [])
            oligo_output = self.runner.design_sequences(**input_feat)
            if supplemental_inputs:
                supplemental_outputs = [
                    {
                        "source": supplemental_input["source"],
                        "features": supplemental_input["features"],
                        "output": self.runner.design_sequences(
                            **supplemental_input["features"]
                        ),
                    }
                    for supplemental_input in supplemental_inputs
                ]
                oligo_output = self._merge_supplemental_outputs(
                    oligo_name,
                    oligo_output,
                    supplemental_outputs,
                )
            oligo_output.update({"name": oligo_name, "feature_dict": input_feat})
            mutation_output[oligo_name] = oligo_output

        return mutation_output

    def design_initial_sequences(
        self,
        *,
        structure_path,
        chains_to_design,
        redesigned_residues,
        symmetry_residues="",
        symmetry_weights="",
    ) -> dict[str, str]:
        """Inverse-fold selected structure chains for an initializer."""
        chain_ids = [str(chain).strip() for chain in chains_to_design if str(chain).strip()]
        if not chain_ids:
            raise ValueError("Initial sequence design requires at least one structure chain.")
        output = self.runner.design_sequences(
            pdb_path=str(structure_path),
            chains_to_design=",".join(chain_ids),
            fixed_residues="",
            redesigned_residues=str(redesigned_residues),
            symmetry_residues=str(symmetry_residues or ""),
            symmetry_weights=str(symmetry_weights or ""),
        )
        generated = output.get("S")
        chain_letters = [str(chain) for chain in (output.get("chain_letters") or [])]
        if generated is None or not chain_letters:
            raise ValueError(
                "LigandMPNN initial sequence design output is missing S or chain_letters."
            )
        if hasattr(generated, "detach"):
            generated = generated.detach().cpu().numpy()
        generated = np.asarray(generated)
        if generated.ndim == 2:
            generated = generated[0]
        if generated.ndim != 1 or len(generated) != len(chain_letters):
            raise ValueError(
                "LigandMPNN initial sequence design output length does not match chain_letters."
            )

        sequences: dict[str, list[str]] = {chain: [] for chain in chain_ids}
        for residue_index, chain in zip(generated.tolist(), chain_letters, strict=True):
            if chain not in sequences:
                continue
            index = int(residue_index)
            if index == len(self.aa_order):
                sequences[chain].append("X")
                continue
            if index < 0 or index > len(self.aa_order):
                raise ValueError(
                    f"LigandMPNN produced unsupported residue index {index} for chain {chain!r}."
                )
            sequences[chain].append(self.aa_order[index])
        missing = [chain for chain, residues in sequences.items() if not residues]
        if missing:
            raise ValueError(
                "LigandMPNN output is missing requested chain(s): " + ", ".join(missing)
            )
        return {chain: "".join(residues) for chain, residues in sequences.items()}

    def _merge_supplemental_outputs(
        self,
        oligo_name: str,
        primary_output: dict,
        supplemental_outputs: list[dict],
    ) -> dict:
        primary_probs = probabilities_to_numpy(primary_output["sampling_probs"])
        weighted = np.asarray(primary_probs, dtype=float)
        total_weight = 1.0
        context_records = []
        for item in supplemental_outputs:
            source = item["source"]
            output = item["output"]
            supplemental_probs = probabilities_to_numpy(output["sampling_probs"])
            if supplemental_probs.shape != primary_probs.shape:
                raise ValueError(
                    "LigandMPNN supplemental structure output shape mismatch for "
                    f"{oligo_name!r} source {source!r}: "
                    f"{supplemental_probs.shape} vs {primary_probs.shape}."
                )
            weight = self.supplemental_structure_weight
            weighted = weighted + weight * np.asarray(supplemental_probs, dtype=float)
            total_weight += weight
            context_records.append(
                {
                    "source": source,
                    "feature_dict": item["features"],
                }
            )

        merged = dict(primary_output)
        merged["sampling_probs"] = weighted / total_weight
        merged["supplemental_structure_outputs"] = context_records
        return merged

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        excluded_complexes = set()
        if runtime_config is not None and hasattr(runtime_config, "get"):
            excluded_complexes.update(runtime_config.get("ignored_complexes") or [])
            excluded_complexes.update(
                runtime_config.get("negative_design_complexes") or []
            )
        active_oligos = [
            oligo_name
            for oligo_name in evaluation_scores.names
            if oligo_name not in excluded_complexes
        ]
        active_scores = (
            evaluation_scores.filter_by_oligomer(active_oligos)
            if active_oligos != list(evaluation_scores.names)
            else evaluation_scores
        )

        # Prepare mappings and exclusions
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            active_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
            fallback_to_all_mutable_if_selector_empty=(
                self.fallback_to_all_mutable_if_selector_empty
            ),
        )

        chain_protos_by_oligo = {}
        complex_fixed_residues = {}
        for oligo_name in active_scores.names:
            chain_protos = [
                (int_id_to_str_id(chain_index + 1), proto)
                for chain_index, proto in enumerate(oligo_name)
                if proto in protomers
            ]
            chain_protos_by_oligo[oligo_name] = chain_protos
            complex_fixed_residues[oligo_name] = " ".join(
                " ".join(f"{chain}{pos+1}" for pos in excluded_positions[proto])
                for chain, proto in chain_protos
            )

        # todo handle tied positions, especially for segments with aliases
        chain_tied_residues = {}
        chain_tied_residue_weights = {}
        for oligo_name in active_scores.names:
            chain_protos = chain_protos_by_oligo[oligo_name]
            result = []
            weights = []
            for proto, packs in itertools.groupby(chain_protos, key=lambda x: x[1]):
                chains = [chain for chain, _ in packs]
                tied_positions = protomers[proto].tied_positions

                for positions in tied_positions:
                    tied_names = [
                        f"{chain}{pos+1}" for chain in chains for pos in positions
                    ]
                    if len(tied_names) <= 1:
                        continue
                    result.append(",".join(tied_names))
                    normalized_weight = str(1.0 / len(tied_names))
                    weights.append(",".join([normalized_weight] * len(tied_names)))
            chain_tied_residues[oligo_name] = "|".join(result)
            chain_tied_residue_weights[oligo_name] = "|".join(weights)

        structure_inputs = prepare_structure_input_scores(
            active_scores,
            supplemental_structures=self.supplemental_structures,
            predocker=self.predocker,
            mutator_name=getattr(self, "alias", self.name),
            predocker_runner=run_nested_predocker_on_scores,
        )
        try:
            pdb_paths = {
                k: str(v) for k, v in structure_inputs.primary.complex_paths.items()
            }
        except IndexError:
            raise FileNotFoundError("No PDB/CIF files found in the specified directory")

        if self.map_sequence_positions_to_pdb_residues:
            complex_fixed_residues = {
                oligo_name: self._map_fixed_residues_to_pdb(
                    oligo_name,
                    chain_protos_by_oligo[oligo_name],
                    excluded_positions,
                    protomers,
                    pdb_paths[oligo_name],
                )
                for oligo_name in active_scores.names
            }

        membrane_residues = self._prepare_membrane_residue_strings(
            protomers,
            active_scores,
            excluded_positions,
        )
        prepared = {}
        for oligo_name in active_scores.names:
            base_features = {
                "pdb_path": pdb_paths[oligo_name],
                "fixed_residues": complex_fixed_residues.get(oligo_name, ""),
                "redesigned_residues": "",
                "symmetry_residues": chain_tied_residues.get(oligo_name, ""),
                "symmetry_weights": chain_tied_residue_weights.get(oligo_name, ""),
            }
            base_features.update(
                {
                    "transmembrane_interface": membrane_residues["interface"].get(
                        oligo_name, ""
                    ),
                    "transmembrane_buried": membrane_residues["buried"].get(
                        oligo_name, ""
                    ),
                }
            )
            supplemental_inputs = []
            for supplemental in structure_inputs.supplemental:
                supplemental_paths = supplemental.scores.complex_paths
                supplemental_pdb_path = str(supplemental_paths[oligo_name])
                supplemental_fixed_residues = base_features["fixed_residues"]
                if self.map_sequence_positions_to_pdb_residues:
                    supplemental_fixed_residues = self._map_fixed_residues_to_pdb(
                        oligo_name,
                        chain_protos_by_oligo[oligo_name],
                        excluded_positions,
                        protomers,
                        supplemental_pdb_path,
                    )
                supplemental_features = {
                    "pdb_path": supplemental_pdb_path,
                    "fixed_residues": supplemental_fixed_residues,
                    "symmetry_residues": base_features["symmetry_residues"],
                    "symmetry_weights": base_features["symmetry_weights"],
                }
                for optional_key in (
                    "redesigned_residues",
                    "transmembrane_interface",
                    "transmembrane_buried",
                ):
                    if base_features.get(optional_key):
                        supplemental_features[optional_key] = base_features[optional_key]
                supplemental_inputs.append(
                    {
                        "source": supplemental.source,
                        "features": supplemental_features,
                    }
                )
            if supplemental_inputs:
                base_features["supplemental_inputs"] = supplemental_inputs
            prepared[oligo_name] = base_features
        return prepared

    @staticmethod
    def _pdb_residue_ids_by_chain(pdb_path: str) -> dict[str, list[str]]:
        if pathlib.Path(pdb_path).suffix.lower() in {".cif", ".mmcif"}:
            from Bio.PDB import MMCIFParser

            structure = MMCIFParser(QUIET=True).get_structure("ligandmpnn", pdb_path)
            model = next(structure.get_models())
            residue_ids: dict[str, list[str]] = defaultdict(list)
            for chain in model:
                chain_id = str(chain.id).strip()
                if not chain_id:
                    raise ValueError(
                        f"Cannot map sequence positions in {pdb_path!r}: "
                        "an mmCIF atom chain has no identifier."
                    )
                for residue in chain:
                    hetero_flag, residue_number, insertion_code = residue.id
                    if hetero_flag != " ":
                        continue
                    residue_ids[chain_id].append(
                        f"{chain_id}{residue_number}{str(insertion_code).strip()}"
                    )
            return dict(residue_ids)

        residue_ids: dict[str, list[str]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        with open(pdb_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("ATOM  "):
                    continue
                chain = line[21].strip()
                resseq = line[22:26].strip()
                insertion_code = line[26].strip()
                if not chain or not resseq:
                    raise ValueError(
                        f"Cannot map sequence positions in {pdb_path!r}: "
                        "an ATOM record has no chain or residue number."
                    )
                residue_key = (resseq, insertion_code)
                if residue_key in seen[chain]:
                    continue
                seen[chain].add(residue_key)
                residue_ids[chain].append(f"{chain}{resseq}{insertion_code}")
        return dict(residue_ids)

    @classmethod
    def _map_fixed_residues_to_pdb(
        cls,
        oligo_name,
        chain_protos,
        excluded_positions,
        protomers,
        pdb_path,
    ) -> str:
        residue_ids_by_chain = cls._pdb_residue_ids_by_chain(pdb_path)
        mapped: list[str] = []
        for chain, proto in chain_protos:
            residue_ids = residue_ids_by_chain.get(chain)
            if residue_ids is None:
                raise ValueError(
                    f"Cannot map sequence positions for oligomer {oligo_name!r}: "
                    f"PDB {pdb_path!r} has no ATOM residues for chain {chain}."
                )
            expected_length = int(protomers[proto].length)
            if len(residue_ids) != expected_length:
                raise ValueError(
                    f"Cannot map sequence positions for oligomer {oligo_name!r}: "
                    f"PDB {pdb_path!r} chain {chain} residue count {len(residue_ids)} "
                    f"does not match protomer {proto!r} length {expected_length}."
                )
            for position in excluded_positions[proto]:
                position = int(position)
                if position < 0 or position >= len(residue_ids):
                    raise ValueError(
                        f"Cannot map selector position {position + 1} for protomer "
                        f"{proto!r} in oligomer {oligo_name!r}; chain {chain} has "
                        f"{len(residue_ids)} residues."
                    )
                mapped.append(residue_ids[position])
        return " ".join(mapped)

    def _prepare_membrane_residue_strings(
        self,
        protomers,
        evaluation_scores: StandardScores,
        excluded_positions,
    ) -> dict[str, dict[str, str]]:
        if self.membrane_label_selectors:
            return self._prepare_membrane_label_selector_strings(
                protomers,
                evaluation_scores,
                excluded_positions,
            )

        empty = self._empty_membrane_residue_strings(evaluation_scores)
        if self.membrane_selector in (None, "", "None"):
            return empty

        selected_masks = self._select_membrane_masks(
            self.membrane_selector,
            evaluation_scores,
            protomers,
        )
        residue_strings = self._encode_membrane_residue_strings(
            selected_masks,
            protomers,
            evaluation_scores,
            description="LigandMPNN membrane_selector",
            require_nonempty=True,
        )
        empty_label = {oligo_name: "" for oligo_name in evaluation_scores.names}
        return {
            "redesigned": empty_label,
            "interface": (
                residue_strings if self.membrane_label == "interface" else empty_label
            ),
            "buried": (
                residue_strings if self.membrane_label == "buried" else empty_label
            ),
        }

    def _prepare_membrane_label_selector_strings(
        self,
        protomers,
        evaluation_scores: StandardScores,
        excluded_positions,
    ) -> dict[str, dict[str, str]]:
        label_masks: dict[str, dict[str, np.ndarray]] = {}
        for label, expression in self.membrane_label_selectors.items():
            masks = self._select_membrane_masks(
                expression,
                evaluation_scores,
                protomers,
            )
            self._assert_membrane_masks_have_residues(
                masks,
                protomers,
                evaluation_scores,
                description=f"LigandMPNN membrane_label_selectors[{label!r}]",
            )
            label_masks[label] = masks

        if "interface" in label_masks and "buried" in label_masks:
            for proto, buried_mask in list(label_masks["buried"].items()):
                interface_mask = label_masks["interface"].get(proto)
                if interface_mask is not None:
                    label_masks["buried"][proto] = np.logical_and(
                        buried_mask,
                        np.logical_not(interface_mask),
                    )

        union_masks: dict[str, np.ndarray] = {}
        for masks in label_masks.values():
            for proto, mask in masks.items():
                if proto not in protomers:
                    continue
                current = union_masks.get(proto)
                if current is None:
                    union_masks[proto] = np.asarray(mask, dtype=bool).copy()
                else:
                    union_masks[proto] = np.logical_or(
                        current,
                        np.asarray(mask, dtype=bool),
                    )
        self._assert_membrane_masks_have_residues(
            union_masks,
            protomers,
            evaluation_scores,
            description="LigandMPNN membrane_label_selectors",
        )

        result = self._empty_membrane_residue_strings(evaluation_scores)
        for label in ("interface", "buried"):
            result[label] = self._encode_membrane_residue_strings(
                label_masks.get(label, {}),
                protomers,
                evaluation_scores,
                require_nonempty=False,
                description=f"LigandMPNN membrane_label_selectors[{label!r}]",
            )
        return result

    @staticmethod
    def _empty_membrane_residue_strings(
        evaluation_scores: StandardScores,
    ) -> dict[str, dict[str, str]]:
        empty = {oligo_name: "" for oligo_name in evaluation_scores.names}
        return {
            "redesigned": dict(empty),
            "interface": dict(empty),
            "buried": dict(empty),
        }

    def _select_membrane_masks(
        self,
        expression: str,
        evaluation_scores: StandardScores,
        protomers,
    ) -> dict[str, np.ndarray]:
        selected_masks = construct_position_selector(
            expression,
            evaluation_scores,
            protomers,
        )
        return {
            proto: np.asarray(mask, dtype=bool).copy()
            for proto, mask in selected_masks.items()
            if proto in protomers
        }

    def _encode_membrane_residue_strings(
        self,
        selected_masks: Mapping[str, np.ndarray],
        protomers,
        evaluation_scores: StandardScores,
        *,
        description: str,
        require_nonempty: bool,
    ) -> dict[str, str]:
        residue_strings: dict[str, str] = {}
        for oligo_name in evaluation_scores.names:
            encoded: list[str] = []
            for chain_index, proto in enumerate(oligo_name):
                if proto not in protomers:
                    continue
                mask = selected_masks.get(proto)
                if mask is None:
                    continue
                chain = int_id_to_str_id(chain_index + 1)
                for pos in np.where(mask)[0].tolist():
                    encoded.append(f"{chain}{int(pos) + 1}")
            residue_strings[oligo_name] = " ".join(encoded)
            if require_nonempty and not residue_strings[oligo_name]:
                raise ValueError(
                    f"{description} selected no residues for oligomer {oligo_name!r}."
                )
        return residue_strings

    def _assert_membrane_masks_have_residues(
        self,
        selected_masks: Mapping[str, np.ndarray],
        protomers,
        evaluation_scores: StandardScores,
        *,
        description: str,
    ) -> None:
        self._encode_membrane_residue_strings(
            selected_masks,
            protomers,
            evaluation_scores,
            description=description,
            require_nonempty=True,
        )

    def prepare_mutation_output(
        self, mutation_output, protomers, evaluation_scores, runtime_config
    ):
        """
        Prepares and mutates protein sequences based on probabilities in the mutation_output.

        Parameters:
            mutation_output (dict): Contains feature dictionary and probabilities for each sequence.
            evaluation_scores (list of StandardInput): Input protein structures with sequence data.
            runtime_config (dict): Contains runtime configurations.

        Returns:
            dict: Updated mutation sites for each prototype.
        """
        # Extract mappings and sequence lengths

        sample_mutations_by_aa_probs(
            evaluation_scores,
            mutation_output,
            self.aa_order,
            self.mutation_weight,
            runtime_config=runtime_config,
            entropy_cutoff=self.entropy_cutoff,
            enforce_mutation_num=self.enforce_mutation_num,
        )


class GroupedLigandMPNNMutator(LigandMPNNMutator):
    """LigandMPNN mutator that batches synchronized multi-state design jobs."""

    name = "GroupedLigandMPNN"

    def compute_mutation(self, mutation_input_features: Tuple[Any, Any]):
        primary_features = {
            oligo_name: self._primary_features(input_feat)
            for oligo_name, input_feat in mutation_input_features.items()
        }
        primary_outputs = self._run_grouped_feature_set(primary_features)

        supplemental_outputs_by_oligo: dict[str, list[dict[str, object]]] = {
            oligo_name: [] for oligo_name in primary_features
        }
        max_supplemental = max(
            (
                len(input_feat.get("supplemental_inputs", []) or [])
                for input_feat in mutation_input_features.values()
            ),
            default=0,
        )
        for supplemental_index in range(max_supplemental):
            supplemental_features: dict[str, dict[str, object]] = {}
            supplemental_sources: dict[str, str] = {}
            for oligo_name, input_feat in mutation_input_features.items():
                supplemental_inputs = input_feat.get("supplemental_inputs", []) or []
                if supplemental_index >= len(supplemental_inputs):
                    continue
                supplemental = supplemental_inputs[supplemental_index]
                supplemental_features[oligo_name] = dict(supplemental["features"])
                supplemental_sources[oligo_name] = str(supplemental["source"])
            if not supplemental_features:
                continue
            supplemental_outputs = self._run_grouped_feature_set(supplemental_features)
            for oligo_name, output in supplemental_outputs.items():
                supplemental_outputs_by_oligo.setdefault(oligo_name, []).append(
                    {
                        "source": supplemental_sources[oligo_name],
                        "features": supplemental_features[oligo_name],
                        "output": output,
                    }
                )

        mutation_output = {}
        for oligo_name, oligo_output in primary_outputs.items():
            supplemental_outputs = supplemental_outputs_by_oligo.get(oligo_name) or []
            if supplemental_outputs:
                oligo_output = self._merge_supplemental_outputs(
                    oligo_name,
                    oligo_output,
                    supplemental_outputs,
                )
            oligo_output = dict(oligo_output)
            oligo_output.update(
                {
                    "name": oligo_name,
                    "feature_dict": primary_features[oligo_name],
                }
            )
            mutation_output[oligo_name] = oligo_output
        return mutation_output

    def _run_grouped_feature_set(
        self,
        feature_by_oligo: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        outputs: dict[str, dict[str, object]] = {}
        for group in self._group_shared_protomer_features(feature_by_oligo):
            if len(group) == 1:
                oligo_name, features = group[0]
                outputs[oligo_name] = self.runner.design_sequences(
                    **self._runner_features(features)
                )
                continue
            outputs.update(self._run_multi_state_group(group))
        return outputs

    def _run_multi_state_group(
        self,
        group: list[tuple[str, dict[str, object]]],
    ) -> dict[str, dict[str, object]]:
        payload_dir = self._multi_state_payload_dir()
        group_name = self._safe_group_name("_".join(oligo for oligo, _ in group))
        pdb_paths = [str(features["pdb_path"]) for _, features in group]
        fixed_residues = {
            path: self._split_residue_string(features.get("fixed_residues", ""))
            for path, (_, features) in zip(pdb_paths, group)
        }
        redesigned_residues = {
            path: self._split_residue_string(features.get("redesigned_residues", ""))
            for path, (_, features) in zip(pdb_paths, group)
        }
        transmembrane_interface = {
            path: str(features.get("transmembrane_interface", "") or "")
            for path, (_, features) in zip(pdb_paths, group)
        }
        transmembrane_buried = {
            path: str(features.get("transmembrane_buried", "") or "")
            for path, (_, features) in zip(pdb_paths, group)
        }

        multi_features = self._runner_features(group[0][1])
        for key in (
            "pdb_path",
            "pdb_path_multi",
            "fixed_residues",
            "fixed_residues_multi",
            "redesigned_residues",
            "redesigned_residues_multi",
            "transmembrane_interface",
            "transmembrane_interface_multi",
            "transmembrane_buried",
            "transmembrane_buried_multi",
        ):
            multi_features.pop(key, None)
        multi_features["pdb_path_multi"] = str(
            self._write_group_json(payload_dir, group_name, "pdb_paths", pdb_paths)
        )
        multi_features["fixed_residues_multi"] = str(
            self._write_group_json(payload_dir, group_name, "fixed_residues", fixed_residues)
        )
        if any(redesigned_residues.values()):
            multi_features["redesigned_residues_multi"] = str(
                self._write_group_json(
                    payload_dir,
                    group_name,
                    "redesigned_residues",
                    redesigned_residues,
                )
            )
        if any(transmembrane_interface.values()):
            multi_features["transmembrane_interface_multi"] = str(
                self._write_group_json(
                    payload_dir,
                    group_name,
                    "transmembrane_interface",
                    transmembrane_interface,
                )
            )
        if any(transmembrane_buried.values()):
            multi_features["transmembrane_buried_multi"] = str(
                self._write_group_json(
                    payload_dir,
                    group_name,
                    "transmembrane_buried",
                    transmembrane_buried,
                )
            )

        result = self.runner.design_sequences(**multi_features)
        multi_outputs = result.get("multi_outputs") if isinstance(result, dict) else None
        if not isinstance(multi_outputs, dict):
            raise RuntimeError(
                "DynamicMPNN grouped inference requires runner output with `multi_outputs`."
            )

        outputs: dict[str, dict[str, object]] = {}
        for oligo_name, features in group:
            pdb_path = str(features["pdb_path"])
            oligo_output = multi_outputs.get(pdb_path)
            if oligo_output is None:
                resolved = str(pathlib.Path(pdb_path).expanduser().resolve())
                oligo_output = multi_outputs.get(resolved)
            if oligo_output is None:
                raise RuntimeError(
                    f"DynamicMPNN grouped inference missing output for {pdb_path!r}."
                )
            outputs[oligo_name] = oligo_output
        return outputs

    def _group_shared_protomer_features(
        self,
        feature_by_oligo: dict[str, dict[str, object]],
    ) -> list[list[tuple[str, dict[str, object]]]]:
        buckets: dict[tuple[object, ...], list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for oligo_name, features in feature_by_oligo.items():
            buckets[self._multi_state_compatibility_key(features)].append(
                (oligo_name, features)
            )

        groups: list[list[tuple[str, dict[str, object]]]] = []
        for bucket_items in buckets.values():
            remaining = list(bucket_items)
            while remaining:
                group = [remaining.pop(0)]
                group_protos = self._sync_protomers(group[0][0])
                changed = True
                while changed:
                    changed = False
                    for item in list(remaining):
                        item_protos = self._sync_protomers(item[0])
                        if group_protos and item_protos and group_protos.intersection(item_protos):
                            remaining.remove(item)
                            group.append(item)
                            group_protos.update(item_protos)
                            changed = True
                groups.append(group)
        return groups

    def _sync_protomers(self, oligo_name: str) -> set[str]:
        protomers = getattr(global_config, "protomers", {})
        try:
            segments = global_config.segments
        except RuntimeError:
            segments = {}
        sync: set[str] = set()
        for proto_name in str(oligo_name):
            protomer = protomers.get(proto_name)
            if protomer is None:
                continue
            if getattr(protomer, "type", None) != ProtomerTypes.PROTEIN:
                continue
            if int(getattr(protomer, "mutable_length", 0) or 0) <= 0:
                continue

            component_keys: list[str] = []
            for component_name in getattr(protomer, "components", []) or []:
                segment = segments.get(component_name)
                if segment is None or getattr(segment, "is_fixed", False):
                    continue
                component_keys.append(f"segment:{id(segment)}")
            if not component_keys:
                for component_id in getattr(protomer, "component_ids", []) or []:
                    component_keys.append(f"segment:{component_id}")
            sync.update(component_keys or [f"protomer:{proto_name}"])
        return sync

    @staticmethod
    def _multi_state_compatibility_key(features: dict[str, object]) -> tuple[object, ...]:
        return tuple(
            features.get(key, "")
            for key in (
                "parse_these_chains_only",
                "symmetry_residues",
                "symmetry_weights",
                "global_transmembrane_label",
            )
        )

    @staticmethod
    def _primary_features(input_feat: dict[str, object]) -> dict[str, object]:
        features = dict(input_feat)
        features.pop("supplemental_inputs", None)
        return features

    @staticmethod
    def _runner_features(features: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in features.items() if not str(key).startswith("_")}

    @staticmethod
    def _split_residue_string(value: object) -> list[str]:
        if value in (None, "", "None"):
            return []
        if isinstance(value, str):
            return [item for item in value.split() if item]
        return [str(item) for item in value]

    def _multi_state_payload_dir(self) -> pathlib.Path:
        try:
            root = pathlib.Path(global_config.raw_model_results_dir)
        except RuntimeError:
            root = pathlib.Path(global_config.out_dir) / "raw_model_results"
        path = root / self.name / "multi_state_inputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_group_json(
        payload_dir: pathlib.Path,
        group_name: str,
        kind: str,
        value,
    ) -> pathlib.Path:
        path = payload_dir / f"{group_name}.{kind}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _safe_group_name(value: str) -> str:
        step = getattr(global_config, "step", 0)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "group"))
        return f"step_{int(step):03d}_{safe}"
