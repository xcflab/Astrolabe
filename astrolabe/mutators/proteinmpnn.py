from __future__ import annotations

import io
import json
import math
import pathlib
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from prody import parsePDB, writePDBStream

from astrolabe.core import global_config
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator
from astrolabe.mutators.utils import probabilities_to_numpy
from astrolabe.utils.constant import restype_3to1


_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_REMAPPED_RESIDUE_KEYS = {
    "fixed_residues",
    "redesigned_residues",
    "transmembrane_interface",
    "transmembrane_buried",
}
_STATE_SPECIFIC_KEYS = _REMAPPED_RESIDUE_KEYS | {
    "pdb_path",
    "pdb_path_multi",
    "fixed_residues_multi",
    "redesigned_residues_multi",
    "transmembrane_interface_multi",
    "transmembrane_buried_multi",
    "symmetry_residues",
    "symmetry_weights",
    "parse_these_chains_only",
    "chains_to_design",
    "supplemental_inputs",
}


@dataclass(frozen=True)
class _Residue:
    token: str
    aa: str


@dataclass(frozen=True)
class _ParsedPDB:
    path: pathlib.Path
    atom_lines: list[str]
    chain_order: list[str]
    protein_chain_order: list[str]
    residues_by_chain: dict[str, list[_Residue]]
    atom_coords: np.ndarray


@dataclass(frozen=True)
class _StateLayout:
    oligo: str
    features: dict[str, object]
    parsed: _ParsedPDB
    chain_map: dict[str, str]
    protomer_chain_map: dict[str, str]
    protomer_new_chain_map: dict[str, str]
    output_chains: list[str]
    output_lengths: list[int]


ProteinMPNN = "ProteinMPNN"
MultistateProteinMPNN = "MultistateProteinMPNN"


class ProteinMPNNMutator(LigandMPNNMutator):
    """ProteinMPNN-backed specialization of the shared LigandMPNN adapter."""

    name = ProteinMPNN


class MultistateProteinMPNNMutator(ProteinMPNNMutator):
    """Run one far-apart MPNN job with signed shared-binder state logits."""

    name = MultistateProteinMPNN
    capabilities = replace(
        ProteinMPNNMutator.capabilities,
        supports_initialization_from_structure=False,
    )

    def __init__(
        self,
        sequence_model_runner,
        state_betas=None,
        translation_x=5000.0,
        **kwargs,
    ):
        super().__init__(sequence_model_runner=sequence_model_runner, **kwargs)
        self.state_oligos: list[str] | None = None
        self.shared_protomer: str | None = None
        self.state_betas = self._parse_configured_state_betas(state_betas)
        self.translation_x = float(translation_x)
        if self.translation_x < 1000.0:
            raise ValueError("translation_x must be at least 1000 Angstrom.")

    def _parse_configured_state_betas(
        self, state_betas
    ) -> dict[str, float] | None:
        if state_betas is None:
            return None
        if not isinstance(state_betas, Mapping):
            raise TypeError(
                "state_betas must map each unprefixed stage oligo to its signed beta."
            )
        parsed = {str(oligo): float(beta) for oligo, beta in state_betas.items()}
        if not all(math.isfinite(beta) for beta in parsed.values()):
            raise ValueError("state_betas values must all be finite.")
        if any(beta == 0.0 for beta in parsed.values()):
            raise ValueError(
                "state_betas values must be non-zero; use the '~' oligo prefix to ignore a state."
            )
        return parsed

    def _infer_state_contract(self, oligos) -> None:
        state_oligos = [str(oligo) for oligo in oligos]
        if len(state_oligos) != 2 or len(set(state_oligos)) != 2:
            raise ValueError(
                "MultistateProteinMPNN requires exactly two distinct non-ignored "
                f"oligos in the current stage, got {state_oligos!r}."
            )
        shared_candidates = [
            protomer
            for protomer in dict.fromkeys(state_oligos[0])
            if protomer != "+"
            and all(oligo.count(protomer) == 1 for oligo in state_oligos)
        ]
        if len(shared_candidates) != 1:
            raise ValueError(
                "MultistateProteinMPNN must infer exactly one shared protomer that "
                "appears once in each non-ignored stage oligo; "
                f"oligos={state_oligos!r}, candidates={shared_candidates!r}."
            )
        if self.state_betas is not None:
            expected = set(state_oligos)
            observed = set(self.state_betas)
            if observed != expected:
                missing = sorted(expected - observed)
                unexpected = sorted(observed - expected)
                raise ValueError(
                    "state_betas keys must exactly match the non-ignored stage oligos; "
                    f"missing={missing!r}, unexpected={unexpected!r}."
                )
            self.state_betas = {
                oligo: self.state_betas[oligo] for oligo in state_oligos
            }
        self.state_oligos = state_oligos
        self.shared_protomer = shared_candidates[0]

    def _resolve_state_betas(
        self, primary_features: dict[str, dict[str, object]]
    ) -> dict[str, float]:
        assert self.state_oligos is not None
        if self.state_betas is not None:
            return dict(self.state_betas)
        state_betas = {
            oligo: float(primary_features[oligo].get("_multistate_beta", 1.0))
            for oligo in self.state_oligos
        }
        if not all(
            math.isfinite(beta) and beta != 0.0 for beta in state_betas.values()
        ):
            raise ValueError(
                "Resolved multistate ProteinMPNN betas must be finite and non-zero."
            )
        return state_betas

    @staticmethod
    def _effective_state_logit_weights(
        state_betas: Mapping[str, float],
    ) -> dict[str, float]:
        state_count = len(state_betas)
        return {
            oligo: beta / state_count for oligo, beta in state_betas.items()
        }

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores,
        runtime_config: dict[str, Any] | None,
    ):
        input_runtime_config = dict(runtime_config or {})
        ignored_complexes = set(input_runtime_config.get("ignored_complexes") or [])
        stage_oligos = input_runtime_config.get("oligo")
        if stage_oligos is None:
            stage_oligos = evaluation_scores.names
        self._infer_state_contract(
            oligo for oligo in stage_oligos if str(oligo) not in ignored_complexes
        )
        assert self.state_oligos is not None
        configured_states = set(self.state_oligos)
        negative_design_complexes = list(
            input_runtime_config.get("negative_design_complexes") or []
        )
        negative_states = configured_states.intersection(negative_design_complexes)
        input_runtime_config["negative_design_complexes"] = [
            oligo
            for oligo in negative_design_complexes
            if oligo not in configured_states
        ]
        prepared = super().prepare_mutation_input(
            protomers,
            evaluation_scores,
            input_runtime_config,
        )
        inferred_betas = {
            oligo: -1.0 if oligo in negative_states else 1.0
            for oligo in self.state_oligos
        }
        state_betas = self.state_betas or inferred_betas
        sign_mismatches = [
            oligo
            for oligo, beta in state_betas.items()
            if (oligo in negative_states) != (beta < 0.0)
        ]
        if sign_mismatches:
            raise ValueError(
                "state_betas signs must match '-' negative-design markers in stages.*.oligo; "
                f"mismatched states: {sign_mismatches!r}."
            )
        for oligo, features in prepared.items():
            if oligo in state_betas:
                features["_multistate_beta"] = state_betas[oligo]
        return prepared

    def compute_mutation(self, mutation_input_features):
        self._infer_state_contract(mutation_input_features)
        primary_features = self._primary_features_by_state(mutation_input_features)
        state_betas = self._resolve_state_betas(primary_features)
        layouts = self._build_state_layouts(primary_features)
        combined_path, state_coords = self._write_combined_pdb(layouts)
        min_cross_state_distance = self._min_distance(state_coords[0], state_coords[1])
        if min_cross_state_distance <= 1000.0:
            raise ValueError(
                "Combined far-apart PDB cross-state minimum atom distance must be "
                f"> 1000 Angstrom, got {min_cross_state_distance:.3f}."
            )

        symmetry_residues, symmetry_weights = self._build_shared_binder_ties(
            layouts, state_betas
        )
        combined_features = self._combined_runner_features(
            layouts,
            combined_path=combined_path,
            symmetry_residues=symmetry_residues,
            symmetry_weights=symmetry_weights,
        )

        runner_output = self.runner.design_sequences(**combined_features)
        runner_call_count = 1
        probs = self._validated_sampling_probs(runner_output, layouts)
        outputs = self._split_runner_output(runner_output, probs, layouts)
        self._assert_shared_binder_probabilities_match(outputs, layouts)

        provenance = {
            "combined_pdb": str(combined_path),
            "translation_x": self.translation_x,
            "state_oligos": list(self.state_oligos),
            "shared_protomer": self.shared_protomer,
            "state_betas": dict(state_betas),
            "effective_state_logit_weights": self._effective_state_logit_weights(
                state_betas
            ),
            "negative_state_oligos": [
                oligo for oligo, beta in state_betas.items() if beta < 0.0
            ],
            "chain_maps": {
                layout.oligo: dict(layout.chain_map) for layout in layouts
            },
            "tie_count": len(symmetry_residues.split("|")) if symmetry_residues else 0,
            "runner_call_count": runner_call_count,
            "no_probability_average_fallback": True,
            "min_cross_state_atom_distance": min_cross_state_distance,
        }
        provenance_path = combined_path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        provenance["provenance_json"] = str(provenance_path)

        for layout in layouts:
            public_features = {
                key: value
                for key, value in layout.features.items()
                if not str(key).startswith("_")
            }
            outputs[layout.oligo].update(
                {
                    "name": layout.oligo,
                    "feature_dict": public_features,
                    "multistate_proteinmpnn_provenance": provenance,
                    "far_apart_tied_multitarget_provenance": provenance,
                }
            )
        return outputs

    def _primary_features_by_state(self, mutation_input_features) -> dict[str, dict[str, object]]:
        observed = set(mutation_input_features)
        expected = set(self.state_oligos)
        if observed != expected:
            raise ValueError(
                "MultistateProteinMPNN expects exactly the configured "
                f"state_oligos {self.state_oligos!r}, got {sorted(observed)!r}."
            )
        primary_features = {}
        for oligo in self.state_oligos:
            features = dict(mutation_input_features[oligo])
            if features.get("supplemental_inputs"):
                raise ValueError(
                    "MultistateProteinMPNN does not support "
                    "supplemental_inputs because it must make exactly one runner call."
                )
            features.pop("supplemental_inputs", None)
            if features.get("symmetry_residues"):
                raise ValueError(
                    "MultistateProteinMPNN owns symmetry_residues; "
                    f"state {oligo!r} provided pre-existing ties."
                )
            primary_features[oligo] = features
        return primary_features

    def _build_state_layouts(
        self, primary_features: dict[str, dict[str, object]]
    ) -> list[_StateLayout]:
        parsed_by_oligo = {
            oligo: self._parse_pdb(pathlib.Path(str(features["pdb_path"])))
            for oligo, features in primary_features.items()
        }
        chain_maps = self._assign_chain_maps(
            [parsed_by_oligo[oligo] for oligo in self.state_oligos]
        )

        layouts = []
        for oligo, chain_map in zip(self.state_oligos, chain_maps, strict=True):
            parsed = parsed_by_oligo[oligo]
            if len(parsed.protein_chain_order) < len(oligo):
                raise ValueError(
                    f"PDB {parsed.path} has fewer protein chains "
                    f"({len(parsed.protein_chain_order)}) than oligo {oligo!r}."
                )
            protomer_chain_map = {
                proto: parsed.protein_chain_order[index]
                for index, proto in enumerate(oligo)
            }
            protomer_new_chain_map = {
                proto: chain_map[chain]
                for proto, chain in protomer_chain_map.items()
            }
            output_chains = [protomer_new_chain_map[proto] for proto in oligo]
            output_lengths = [
                len(parsed.residues_by_chain[protomer_chain_map[proto]])
                for proto in oligo
            ]
            layouts.append(
                _StateLayout(
                    oligo=oligo,
                    features=primary_features[oligo],
                    parsed=parsed,
                    chain_map=chain_map,
                    protomer_chain_map=protomer_chain_map,
                    protomer_new_chain_map=protomer_new_chain_map,
                    output_chains=output_chains,
                    output_lengths=output_lengths,
                )
            )
        return layouts

    @staticmethod
    def _parse_pdb(path: pathlib.Path) -> _ParsedPDB:
        if not path.exists():
            raise FileNotFoundError(f"LigandMPNN input PDB not found: {path}")
        atoms = parsePDB(str(path))
        if atoms is None:
            raise ValueError(f"Could not parse LigandMPNN input structure: {path}")
        stream = io.StringIO()
        writePDBStream(stream, atoms)
        normalized_lines = stream.getvalue().splitlines()
        atom_lines: list[str] = []
        chain_order: list[str] = []
        protein_chain_order: list[str] = []
        residues_by_chain: dict[str, list[_Residue]] = OrderedDict()
        residue_keys_by_chain: dict[str, set[tuple[str, str]]] = {}
        atom_coords: list[tuple[float, float, float]] = []

        for raw_line in normalized_lines:
            record = raw_line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            line = raw_line.ljust(80)
            chain = line[21].strip()
            if len(chain) != 1:
                raise ValueError(f"PDB atom line has missing chain ID in {path}.")
            if chain not in chain_order:
                chain_order.append(chain)
            try:
                atom_coords.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Invalid atom coordinates in {path}: {raw_line!r}") from exc

            if record != "ATOM":
                continue
            if chain not in protein_chain_order:
                protein_chain_order.append(chain)
            residues = residues_by_chain.setdefault(chain, [])
            seen = residue_keys_by_chain.setdefault(chain, set())
            residue_key = (line[22:26].strip(), line[26].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            resname = line[17:20].strip().upper()
            token = f"{residue_key[0]}{residue_key[1]}"
            residues.append(_Residue(token=token, aa=restype_3to1.get(resname, "X")))

        if not atom_lines and not atom_coords:
            raise ValueError(f"PDB contains no ATOM/HETATM records: {path}")
        # Keep the same normalized atom stream that LigandMPNN's ProDy parser sees.
        atom_lines = [
            line.ljust(80)
            for line in normalized_lines
            if line[:6].strip() in {"ATOM", "HETATM"}
        ]
        return _ParsedPDB(
            path=path,
            atom_lines=atom_lines,
            chain_order=chain_order,
            protein_chain_order=protein_chain_order,
            residues_by_chain=dict(residues_by_chain),
            atom_coords=np.asarray(atom_coords, dtype=float),
        )

    @staticmethod
    def _assign_chain_maps(parsed_states: list[_ParsedPDB]) -> list[dict[str, str]]:
        used: set[str] = set()
        maps: list[dict[str, str]] = []
        for state_index, parsed in enumerate(parsed_states):
            chain_map: dict[str, str] = {}
            for chain in parsed.chain_order:
                if state_index == 0 and chain in _CHAIN_IDS and chain not in used:
                    new_chain = chain
                else:
                    new_chain = next(
                        (candidate for candidate in _CHAIN_IDS if candidate not in used),
                        None,
                    )
                    if new_chain is None:
                        raise ValueError("Ran out of one-character PDB chain IDs.")
                chain_map[chain] = new_chain
                used.add(new_chain)
            maps.append(chain_map)
        return maps

    def _write_combined_pdb(
        self,
        layouts: list[_StateLayout],
    ) -> tuple[pathlib.Path, list[np.ndarray]]:
        output_path = self._payload_dir(layouts[0].parsed.path) / (
            self._safe_group_name("_".join(self.state_oligos)) + ".combined.pdb"
        )
        output_lines: list[str] = []
        state_coords: list[list[tuple[float, float, float]]] = []
        atom_serial = 1
        for state_index, layout in enumerate(layouts):
            translated_coords: list[tuple[float, float, float]] = []
            previous_chain = None
            previous_line = None
            for raw_line in layout.parsed.atom_lines:
                line = raw_line.ljust(80)
                chain = line[21].strip()
                if previous_chain is not None and chain != previous_chain:
                    output_lines.append(
                        self._render_ter(
                            previous_line,
                            layout.chain_map[previous_chain],
                            atom_serial,
                        )
                    )
                    atom_serial += 1
                new_chain = layout.chain_map[chain]
                x = float(line[30:38]) + (self.translation_x if state_index else 0.0)
                y = float(line[38:46])
                z = float(line[46:54])
                translated_coords.append((x, y, z))
                output_lines.append(
                    f"{line[:6]}{atom_serial:5d}{line[11:21]}{new_chain}{line[22:30]}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                )
                previous_chain = chain
                previous_line = line
                atom_serial += 1
                if atom_serial > 99999:
                    raise ValueError("Combined PDB exceeds 99999 atoms.")
            if previous_line is not None:
                output_lines.append(
                    self._render_ter(
                        previous_line,
                        layout.chain_map[previous_chain],
                        atom_serial,
                    )
                )
                atom_serial += 1
            state_coords.append(translated_coords)
        output_lines.append("END")
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        return output_path, [np.asarray(coords, dtype=float) for coords in state_coords]

    @staticmethod
    def _render_ter(last_atom: str, chain: str, serial: int) -> str:
        return (
            f"TER   {serial:5d}      {last_atom[17:20]} {chain}"
            f"{last_atom[22:26]}{last_atom[26]}"
        )

    def _build_shared_binder_ties(
        self,
        layouts: list[_StateLayout],
        state_betas: Mapping[str, float],
    ) -> tuple[str, str]:
        first, second = layouts
        first_chain = first.protomer_chain_map[self.shared_protomer]
        second_chain = second.protomer_chain_map[self.shared_protomer]
        first_residues = first.parsed.residues_by_chain[first_chain]
        second_residues = second.parsed.residues_by_chain[second_chain]
        first_sequence = "".join(residue.aa for residue in first_residues)
        second_sequence = "".join(residue.aa for residue in second_residues)
        if first_sequence != second_sequence:
            raise ValueError(
                f"shared_protomer {self.shared_protomer!r} binder sequences differ "
                f"between {first.oligo!r} and {second.oligo!r}."
            )

        first_new_chain = first.chain_map[first_chain]
        second_new_chain = second.chain_map[second_chain]
        ties = [
            f"{first_new_chain}{residue_a.token},{second_new_chain}{residue_b.token}"
            for residue_a, residue_b in zip(first_residues, second_residues, strict=True)
        ]
        # Kuhlman-Lab ProteinMPNN applies beta * logits / tied-state count.
        state_weights = self._effective_state_logit_weights(state_betas)
        tied_weight = ",".join(
            str(state_weights[layout.oligo]) for layout in layouts
        )
        weights = [tied_weight] * len(ties)
        return "|".join(ties), "|".join(weights)

    def _combined_runner_features(
        self,
        layouts: list[_StateLayout],
        *,
        combined_path: pathlib.Path,
        symmetry_residues: str,
        symmetry_weights: str,
    ) -> dict[str, object]:
        combined_features = self._runner_features(layouts)
        combined_features.update(
            {
                "pdb_path": str(combined_path),
                "fixed_residues": self._merge_remapped_residue_key(
                    layouts, "fixed_residues"
                ),
                "redesigned_residues": self._merge_remapped_residue_key(
                    layouts, "redesigned_residues"
                ),
                "symmetry_residues": symmetry_residues,
                "symmetry_weights": symmetry_weights,
                "parse_these_chains_only": "".join(
                    chain for layout in layouts for chain in layout.output_chains
                ),
            }
        )
        for key in ("transmembrane_interface", "transmembrane_buried"):
            value = self._merge_remapped_residue_key(layouts, key)
            if value:
                combined_features[key] = value
            else:
                combined_features.pop(key, None)
        return combined_features

    @staticmethod
    def _runner_features(layouts: list[_StateLayout]) -> dict[str, object]:
        features = {
            key: value
            for key, value in layouts[0].features.items()
            if not str(key).startswith("_") and key not in _STATE_SPECIFIC_KEYS
        }
        for layout in layouts[1:]:
            for key, value in layout.features.items():
                if str(key).startswith("_") or key in _STATE_SPECIFIC_KEYS:
                    continue
                if key in features and features[key] != value:
                    raise ValueError(
                        f"Runner feature {key!r} differs across state_oligos."
                    )
                features.setdefault(key, value)
        return features

    def _merge_remapped_residue_key(
        self, layouts: list[_StateLayout], key: str
    ) -> str:
        values: list[str] = []
        for layout in layouts:
            values.extend(
                self._remap_residue_tokens(
                    layout.features.get(key, ""),
                    layout.chain_map,
                    key=key,
                    oligo=layout.oligo,
                )
            )
        return " ".join(values)

    @staticmethod
    def _remap_residue_tokens(
        value: object,
        chain_map: dict[str, str],
        *,
        key: str,
        oligo: str,
    ) -> list[str]:
        if value in (None, "", "None"):
            return []
        tokens = value.split() if isinstance(value, str) else [str(item) for item in value]
        remapped = []
        for token in tokens:
            match = re.fullmatch(r"([A-Za-z0-9])(.+)", str(token))
            if match is None:
                raise ValueError(
                    f"Cannot parse {key} residue token {token!r} for state {oligo!r}."
                )
            chain, suffix = match.groups()
            if chain not in chain_map:
                raise ValueError(
                    f"{key} residue token {token!r} references chain {chain!r}, "
                    f"which is absent from state {oligo!r}."
                )
            remapped.append(f"{chain_map[chain]}{suffix}")
        return remapped

    @staticmethod
    def _validated_sampling_probs(
        runner_output: dict[str, object],
        layouts: list[_StateLayout],
    ) -> np.ndarray:
        if not isinstance(runner_output, dict) or "sampling_probs" not in runner_output:
            raise RuntimeError(
                "MultistateProteinMPNN runner must return a dict containing "
                "sampling_probs."
            )
        probs = probabilities_to_numpy(runner_output["sampling_probs"])
        if probs.ndim != 3:
            raise ValueError(
                "MultistateProteinMPNN expected sampling_probs with shape "
                f"(N, L, A), got {probs.shape}."
            )
        expected_length = sum(sum(layout.output_lengths) for layout in layouts)
        if probs.shape[1] != expected_length:
            raise ValueError(
                "MultistateProteinMPNN sampling_probs length mismatch: expected "
                f"{expected_length}, got {probs.shape[1]}."
            )
        chain_letters = runner_output.get("chain_letters")
        if chain_letters is not None and len(chain_letters) != expected_length:
            raise ValueError(
                "MultistateProteinMPNN runner chain_letters length mismatch: "
                f"expected {expected_length}, got {len(chain_letters)}."
            )
        return np.asarray(probs)

    def _split_runner_output(
        self,
        runner_output: dict[str, object],
        probs: np.ndarray,
        layouts: list[_StateLayout],
    ) -> dict[str, dict[str, object]]:
        indices_by_oligo = self._output_indices_by_oligo(runner_output, layouts)
        outputs = {}
        for layout in layouts:
            indices = indices_by_oligo[layout.oligo]
            output = dict(runner_output)
            output["sampling_probs"] = probs[:, indices, :]
            output["chain_letters"] = tuple(
                chain
                for chain, length in zip(layout.output_chains, layout.output_lengths, strict=True)
                for _ in range(length)
            )
            outputs[layout.oligo] = output
        return outputs

    @staticmethod
    def _output_indices_by_oligo(
        runner_output: dict[str, object],
        layouts: list[_StateLayout],
    ) -> dict[str, list[int]]:
        chain_letters = runner_output.get("chain_letters")
        if chain_letters is not None:
            chain_letters = [str(chain) for chain in chain_letters]
            result: dict[str, list[int]] = {}
            for layout in layouts:
                indices: list[int] = []
                for chain, expected_length in zip(
                    layout.output_chains, layout.output_lengths, strict=True
                ):
                    chain_indices = [
                        index
                        for index, observed_chain in enumerate(chain_letters)
                        if observed_chain == chain
                    ]
                    if len(chain_indices) != expected_length:
                        raise ValueError(
                            "MultistateProteinMPNN runner chain_letters count "
                            f"mismatch for chain {chain!r}: expected {expected_length}, "
                            f"got {len(chain_indices)}."
                        )
                    indices.extend(chain_indices)
                result[layout.oligo] = indices
            return result

        result = {}
        offset = 0
        for layout in layouts:
            length = sum(layout.output_lengths)
            result[layout.oligo] = list(range(offset, offset + length))
            offset += length
        return result

    def _assert_shared_binder_probabilities_match(
        self,
        outputs: dict[str, dict[str, object]],
        layouts: list[_StateLayout],
    ) -> None:
        binder_blocks = []
        for layout in layouts:
            shared_index = layout.oligo.index(self.shared_protomer)
            start = sum(layout.output_lengths[:shared_index])
            stop = start + layout.output_lengths[shared_index]
            binder_blocks.append(outputs[layout.oligo]["sampling_probs"][:, start:stop, :])
        if not np.array_equal(binder_blocks[0], binder_blocks[1]):
            raise ValueError(
                "MultistateProteinMPNN runner returned non-identical shared "
                "binder probability blocks."
            )

    @staticmethod
    def _min_distance(first: np.ndarray, second: np.ndarray) -> float:
        if first.size == 0 or second.size == 0:
            raise ValueError("Cannot compute cross-state atom distance without atoms.")
        min_sq = float("inf")
        for start in range(0, len(first), 512):
            diff = first[start : start + 512, None, :] - second[None, :, :]
            chunk_min = float(np.min(np.sum(diff * diff, axis=-1)))
            min_sq = min(min_sq, chunk_min)
        return float(np.sqrt(min_sq))

    def _payload_dir(self, fallback_pdb_path: pathlib.Path) -> pathlib.Path:
        try:
            root = pathlib.Path(global_config.raw_model_results_dir)
        except RuntimeError:
            try:
                root = pathlib.Path(global_config.out_dir) / "raw_model_results"
            except RuntimeError:
                root = fallback_pdb_path.parent / "raw_model_results"
        alias = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(getattr(self, "alias", self.name)),
        )
        path = root / MultistateProteinMPNN / alias / "combined_inputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _safe_group_name(value: str) -> str:
        step = getattr(global_config, "step", 0)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "group"))
        return f"step_{int(step):03d}_{safe}"
