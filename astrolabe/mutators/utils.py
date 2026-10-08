import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import MutableTypes, ProtomerTypes
from astrolabe.selectors.utils import construct_position_selector
from astrolabe.per_residue_scores import (
    build_mutator_selector_score_context,
    combine_per_proto_mutator_scores,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id

# from alphafold.common.protein import Protein
from astrolabe.utils.constant import (
    cached_aa_freq,
    restype_int_to_str,
    restype_str_to_int,
)


def get_transition_consistency_gate_context(runtime_config):
    if runtime_config is None:
        return None
    if hasattr(runtime_config, "get"):
        return runtime_config.get("transition_consistency_gate_context")
    return getattr(runtime_config, "transition_consistency_gate_context", None)


def get_transition_consistency_gate_output(runtime_config):
    if runtime_config is None:
        return None
    if hasattr(runtime_config, "get"):
        return runtime_config.get("transition_consistency_gate_output")
    return getattr(runtime_config, "transition_consistency_gate_output", None)


def mutation_output_supports_sampling_probs(mutation_output) -> bool:
    for oligo_output in (mutation_output or {}).values():
        if not isinstance(oligo_output, dict):
            continue
        if "sampling_probs" in oligo_output:
            return True
        proto_map = oligo_output.get("proto_sequences")
        if not isinstance(proto_map, dict):
            continue
        for proto_output in proto_map.values():
            if isinstance(proto_output, dict) and "sampling_probs" in proto_output:
                return True
    return False


def probabilities_to_numpy(probs):
    if torch is not None and isinstance(probs, torch.Tensor):
        return probs.detach().cpu().numpy()
    return np.asarray(probs)


def _runtime_get(runtime_config, key, default=None):
    if runtime_config is None:
        return default
    if hasattr(runtime_config, "get"):
        return runtime_config.get(key, default)
    return getattr(runtime_config, key, default)


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def state_conflict_diagnostics_enabled(runtime_config) -> bool:
    cfg = _runtime_get(runtime_config, "state_conflict_diagnostics")
    return bool(_config_get(cfg, "enabled", False))


def _state_conflict_output_dir(runtime_config) -> Path:
    cfg = _runtime_get(runtime_config, "state_conflict_diagnostics")
    raw_output_dir = _config_get(cfg, "output_dir", None)
    if raw_output_dir not in (None, "", "null", "None"):
        return Path(str(raw_output_dir))
    return Path(global_config.out_dir) / "mutator_state_conflicts"


def _collect_state_proto_probabilities(
    evaluation_scores,
    mutation_output,
    *,
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None,
):
    supported_types = _normalize_supported_types(protomer_type)
    state_values: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    protomers = global_config.protomers

    for oligo_name in getattr(evaluation_scores, "names", []) or []:
        oligo_name = str(oligo_name)
        oligo_output = (mutation_output or {}).get(oligo_name)
        if not isinstance(oligo_output, dict):
            continue

        proto_map = oligo_output.get("proto_sequences")
        if isinstance(proto_map, dict):
            for proto_name, proto_output in proto_map.items():
                if not isinstance(proto_output, dict) or "sampling_probs" not in proto_output:
                    continue
                proto_obj = protomers.get(proto_name)
                if proto_obj is None:
                    raise ValueError(
                        f"State conflict diagnostics found unknown protomer {proto_name!r}."
                    )
                if supported_types is not None and proto_obj.type not in supported_types:
                    continue
                arr = probabilities_to_numpy(proto_output["sampling_probs"])
                if arr.ndim == 2:
                    arr = arr[None, ...]
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 3:
                    raise ValueError(
                        f"State conflict diagnostics expected proto sampling_probs for "
                        f"{proto_name!r} in {oligo_name!r} to have 2 or 3 dimensions, got {arr.shape}."
                    )
                if arr.shape[1] != int(proto_obj.length):
                    raise ValueError(
                        f"State conflict diagnostics probability length mismatch for protomer "
                        f"{proto_name!r} in {oligo_name!r}: expected {proto_obj.length}, got {arr.shape[1]}."
                    )
                state_values[str(proto_name)][oligo_name].append(arr)
            continue

        if "sampling_probs" in oligo_output:
            arr = probabilities_to_numpy(oligo_output["sampling_probs"])
            if arr.ndim == 2:
                arr = arr[None, ...]
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 3:
                raise ValueError(
                    f"State conflict diagnostics expected sampling_probs for {oligo_name!r} "
                    f"to have 2 or 3 dimensions, got {arr.shape}."
                )
            residue_index = 0
            for proto in oligo_name:
                proto_obj = protomers.get(proto)
                if proto_obj is None:
                    raise ValueError(
                        f"State conflict diagnostics found unknown protomer {proto!r} in {oligo_name!r}."
                    )
                chain_length = int(proto_obj.length)
                if supported_types is not None and proto_obj.type not in supported_types:
                    residue_index += chain_length
                    continue
                stop = residue_index + chain_length
                if stop > arr.shape[1]:
                    raise ValueError(
                        f"State conflict diagnostics probability length mismatch for {oligo_name!r}: "
                        f"need residues up to {stop}, got {arr.shape[1]}."
                    )
                state_values[str(proto)][oligo_name].append(arr[:, residue_index:stop, :])
                residue_index = stop

    return {
        proto_name: {
            oligo_name: sum(values) / len(values)
            for oligo_name, values in by_oligo.items()
            if values
        }
        for proto_name, by_oligo in state_values.items()
    }


def _mean_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected per-state probability matrix with shape (L,A), got {arr.shape}.")
    denom = arr.sum(axis=-1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("State conflict diagnostics encountered non-positive probability row sum.")
    return arr / denom


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=-1) + 0.5 * np.sum(q * np.log(q / m), axis=-1)


def write_state_conflict_diagnostics(
    evaluation_scores,
    mutation_output,
    aa_order: List[str],
    *,
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None,
    runtime_config: dict[str, Any] | None,
) -> Path | None:
    if not state_conflict_diagnostics_enabled(runtime_config):
        return None
    if not mutation_output_supports_sampling_probs(mutation_output):
        raise ValueError(
            "State conflict diagnostics require mutation output with sampling_probs "
            "or proto_sequences.*.sampling_probs."
        )

    state_probs = _collect_state_proto_probabilities(
        evaluation_scores,
        mutation_output,
        protomer_type=protomer_type,
    )
    rows: list[dict[str, Any]] = []
    step = _runtime_get(runtime_config, "step", getattr(global_config, "step", 0) or 0)
    mutator_alias = _runtime_get(runtime_config, "current_mutator_alias", "unknown")
    stage_name = None
    try:
        stage_name = getattr(global_config.stage_manager, "current_stage", None)
    except RuntimeError:
        stage_name = None

    for proto_name, by_oligo in state_probs.items():
        oligos = sorted(by_oligo)
        if len(oligos) < 2:
            continue
        protomer = global_config.protomers[proto_name]
        sequence = str(getattr(protomer, "sequence", ""))
        for left_index, oligo_a in enumerate(oligos[:-1]):
            probs_a = _mean_distribution(by_oligo[oligo_a])
            for oligo_b in oligos[left_index + 1:]:
                probs_b = _mean_distribution(by_oligo[oligo_b])
                if probs_a.shape != probs_b.shape:
                    raise ValueError(
                        f"State conflict diagnostics shape mismatch for protomer {proto_name!r}: "
                        f"{oligo_a} {probs_a.shape} vs {oligo_b} {probs_b.shape}."
                    )
                jsd = _js_divergence(probs_a, probs_b)
                for pos in range(probs_a.shape[0]):
                    top_a_idx = int(np.argmax(probs_a[pos]))
                    top_b_idx = int(np.argmax(probs_b[pos]))
                    current_aa = sequence[pos] if pos < len(sequence) else ""
                    current_idx = aa_order.index(current_aa) if current_aa in aa_order else None
                    rows.append(
                        {
                            "step": int(step),
                            "stage": "" if stage_name is None else str(stage_name),
                            "mutator": str(mutator_alias),
                            "protomer": proto_name,
                            "position": pos + 1,
                            "current_aa": current_aa,
                            "oligo_a": oligo_a,
                            "oligo_b": oligo_b,
                            "js_divergence": f"{float(jsd[pos]):.8g}",
                            "top_aa_a": aa_order[top_a_idx],
                            "top_prob_a": f"{float(probs_a[pos, top_a_idx]):.8g}",
                            "top_aa_b": aa_order[top_b_idx],
                            "top_prob_b": f"{float(probs_b[pos, top_b_idx]):.8g}",
                            "top_agree": "1" if top_a_idx == top_b_idx else "0",
                            "current_prob_a": (
                                "" if current_idx is None else f"{float(probs_a[pos, current_idx]):.8g}"
                            ),
                            "current_prob_b": (
                                "" if current_idx is None else f"{float(probs_b[pos, current_idx]):.8g}"
                            ),
                        }
                    )

    output_dir = _state_conflict_output_dir(runtime_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"step_{int(step):03d}_{str(mutator_alias)}.csv"
    fieldnames = [
        "step",
        "stage",
        "mutator",
        "protomer",
        "position",
        "current_aa",
        "oligo_a",
        "oligo_b",
        "js_divergence",
        "top_aa_a",
        "top_prob_a",
        "top_aa_b",
        "top_prob_b",
        "top_agree",
        "current_prob_a",
        "current_prob_b",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _normalize_supported_types(
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None,
) -> set[ProtomerTypes] | None:
    if protomer_type is None:
        return None
    if isinstance(protomer_type, ProtomerTypes):
        return {protomer_type}
    return set(protomer_type)


def _collect_proto_probabilities_from_proto_sequences(
    mutation_output,
    *,
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None,
    value_key: str = "sampling_probs",
):
    supported_types = _normalize_supported_types(protomer_type)
    proto_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    for oligo_output in (mutation_output or {}).values():
        proto_map = oligo_output.get("proto_sequences") if isinstance(oligo_output, dict) else None
        if not isinstance(proto_map, dict):
            continue
        for proto_name, proto_output in proto_map.items():
            if not isinstance(proto_output, dict) or value_key not in proto_output:
                continue
            protomer = global_config.protomers.get(proto_name)
            if protomer is None:
                continue
            if supported_types is not None and protomer.type not in supported_types:
                continue
            arr = probabilities_to_numpy(proto_output[value_key])
            if arr.ndim == 2:
                arr = arr[None, ...]
            proto_probabilities[str(proto_name)].append(np.asarray(arr, dtype=float))
    return {
        proto_name: sum(values) / len(values)
        for proto_name, values in proto_probabilities.items()
        if values
    }


def collect_proto_probabilities(
    evaluation_scores,
    mutation_output,
    *,
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None = ProtomerTypes.PROTEIN,
    value_key: str = "sampling_probs",
):
    supported_types = _normalize_supported_types(protomer_type)
    proto_probabilities: dict[str, np.ndarray] = {}
    if supported_types is not None and len(supported_types) == 1:
        single_type = next(iter(supported_types))
        current_names = [
            str(oligo_name)
            for oligo_name in getattr(evaluation_scores, "names", []) or []
            if isinstance((mutation_output or {}).get(str(oligo_name)), dict)
            and value_key in (mutation_output or {}).get(str(oligo_name), {})
        ]
        if current_names:
            complex_sample_probs = {
                str(oligo_name): mutation_output[str(oligo_name)][value_key]
                for oligo_name in current_names
            }
            excluded_complexes = set(global_config.ignored_complexes or [])
            excluded_complexes.update(global_config.negative_design_complexes or [])
            proto_probabilities.update(
                aggregate_proto_aa_probs(
                    current_names,
                    complex_sample_probs,
                    excluded_complexes,
                    type=single_type,
                )
            )

    for proto_name, probs in _collect_proto_probabilities_from_proto_sequences(
        mutation_output,
        protomer_type=protomer_type,
        value_key=value_key,
    ).items():
        if proto_name in proto_probabilities:
            if proto_probabilities[proto_name].shape != probs.shape:
                raise ValueError(
                    f"Probability shape mismatch for protomer {proto_name!r}: "
                    f"{proto_probabilities[proto_name].shape} vs {probs.shape}."
                )
            proto_probabilities[proto_name] = (
                np.asarray(proto_probabilities[proto_name], dtype=float)
                + np.asarray(probs, dtype=float)
            ) / 2.0
        else:
            proto_probabilities[proto_name] = np.asarray(probs, dtype=float)
    return proto_probabilities


def collect_blended_proto_probabilities(
    evaluation_scores,
    mutation_output,
    *,
    protomer_type: ProtomerTypes | Iterable[ProtomerTypes] | None = ProtomerTypes.PROTEIN,
    runtime_config: dict[str, Any] | None = None,
):
    current_proto_probabilities = collect_proto_probabilities(
        evaluation_scores,
        mutation_output,
        protomer_type=protomer_type,
    )
    extra_context = get_transition_consistency_gate_context(runtime_config)
    extra_output = get_transition_consistency_gate_output(runtime_config)
    extra_weight = float(getattr(extra_context, "weight", 0.0) or 0.0)
    if extra_context is None or extra_output is None or extra_weight <= 0.0:
        return current_proto_probabilities

    extra_proto_probabilities = collect_proto_probabilities(
        extra_context.reference_scores,
        extra_output,
        protomer_type=protomer_type,
    )

    supported_types = _normalize_supported_types(protomer_type)
    allowed_protos = {
        proto
        for oligo_name in getattr(evaluation_scores, "names", []) or []
        for proto in str(oligo_name)
        if global_config.protomers.get(proto) is not None
        and (supported_types is None or global_config.protomers[proto].type in supported_types)
    }

    blended = dict(current_proto_probabilities)
    for proto_name in allowed_protos:
        current = current_proto_probabilities.get(proto_name)
        extra = extra_proto_probabilities.get(proto_name)
        if current is None or extra is None:
            continue
        current_np = np.asarray(probabilities_to_numpy(current), dtype=float)
        extra_np = np.asarray(probabilities_to_numpy(extra), dtype=float)
        if current_np.shape != extra_np.shape:
            raise ValueError(
                f"transition-consistency-gate probability shape mismatch for protomer {proto_name!r}: "
                f"{extra_np.shape} vs {current_np.shape}"
            )
        blended[proto_name] = extra_weight * extra_np + (1.0 - extra_weight) * current_np
    return blended


def select_mutation_sites(
    excluded_positions,
    n_mutations,
    proto_probabilities,
    protomers,
    aa_order=restype_int_to_str,
    entropy_cutoff: float | None = None,
):
    """
    Generate mutation sites for each protomer based on probability distributions.

    Args:
        excluded_positions: Dictionary mapping protomers to positions that should not be mutated
        n_mutations: Dictionary mapping protomers to the number of mutations to make
        proto_probabilities: Dictionary mapping protomers to amino acid probabilities
        proto_sequences: Dictionary mapping protomers to their original sequences

    Returns:
        Dictionary mapping protomers to their mutated sites (position: [original_aa, new_aa])
    """
    updated_sites = {}
    for proto, mutation_count in n_mutations.items():
        # Skip if no mutations requested
        if mutation_count <= 0:
            updated_sites[proto] = {}
            continue

        original_sequence = protomers[proto].sequence

        # todo move this part into Protomer class
        # sampling considering the tied positions
        # Process probability matrix
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "select_mutation_sites requires `torch`, but torch is not installed."
            ) from exc

        probabilities = proto_probabilities[proto].squeeze().cpu()
        probabilities = torch.clamp(probabilities, min=1e-5)
        probabilities /= probabilities.sum(-1, keepdim=True)

        # Calculate entropy for each position (higher entropy = more uncertainty)
        entropy_values = -torch.sum(probabilities * torch.log(probabilities), axis=-1)
        sorted_indices = (
            entropy_values.argsort().tolist()
        )  # Sort from lowest to highest entropy

        # Track mutations
        curr_mutated_sites = {}
        remaining_mutations = mutation_count

        sorted_indices = [
            i for i in sorted_indices if i not in excluded_positions[proto]
        ]
        if entropy_cutoff is not None:
            cutoff = float(entropy_cutoff)
            sorted_indices = [
                i for i in sorted_indices if float(entropy_values[i]) <= cutoff
            ]

        to_shuffle_end = 0
        for i, idx in enumerate(sorted_indices):
            to_shuffle_end = i
            if entropy_values[idx] > 1:
                break
        to_shuffle_indices = sorted_indices[:to_shuffle_end]
        remaining = sorted_indices[to_shuffle_end:]
        to_mutate = np.random.permutation(to_shuffle_indices).tolist() + remaining

        # Iterate through positions in order of increasing entropy
        for idx in to_mutate:
            if remaining_mutations <= 0:
                break

            # Skip positions that should not be mutated
            if idx in excluded_positions[proto]:
                continue

            # Sample amino acid according to probability distribution
            position_probs = probabilities[idx]
            sampled_aa = aa_order[np.random.choice(range(20), p=position_probs.numpy())]

            # Only apply mutation if it's different from the original
            if sampled_aa != original_sequence[idx]:
                curr_mutated_sites[idx] = [original_sequence[idx], sampled_aa]
                logging.debug(
                    "Mutated site %d from %s to %s",
                    idx,
                    original_sequence[idx],
                    sampled_aa,
                )
                remaining_mutations -= 1

        updated_sites[proto] = curr_mutated_sites
    return updated_sites


def aggregate_proto_aa_probs(
    oligomer_names,
    complex_sample_probs,
    ignored_complexes,
    type: ProtomerTypes = ProtomerTypes.PROTEIN,
):
    """
    Calculate amino acid probabilities for each protomer by aggregating over chains.

    Args:
        chain2proto: Mapping of chain IDs to protomer IDs
        oligo2complex: Mapping of oligomer names to lists of chain IDs
        chain_lengths: Dictionary with chain lengths
        complex_sample_probs: Probabilities from oligomer sampling
        protein_sequences: Dictionary of protein sequences keyed by protomer
        ignored_complexes: Set of complex names to exclude (for negative design)

    Returns:
        Dictionary mapping prototypes to their aggregated probability distributions
    """
    # First pass: collect chain-level probabilities from complexes
    proto_probabilities = defaultdict(list)
    protomers = global_config.protomers
    for oligo_name in oligomer_names:
        residue_index = 0
        if oligo_name in ignored_complexes:
            continue
        for i, proto in enumerate(oligo_name):
            if protomers[proto].type != type:
                continue
            chain_length = global_config.protomers[proto].length
            probs = complex_sample_probs[oligo_name][
                :, residue_index : residue_index + chain_length
            ]
            proto_probabilities[proto].append(probs)
            residue_index += chain_length
    # Average the probabilities for each protomer
    proto_probabilities = {
        proto: sum(probs) / len(probs) for proto, probs in proto_probabilities.items()
    }
    return proto_probabilities


def _derive_chain2proto_from_oligomers(
    oligomer_names: Iterable[str] | None,
) -> dict[str, str]:
    chain2proto: dict[str, str] = {}
    for oligo_name in oligomer_names or ():
        for idx, proto_name in enumerate(str(oligo_name)):
            chain_id = int_id_to_str_id(idx + 1)
            existing = chain2proto.get(chain_id)
            if existing is not None and existing != proto_name:
                raise ValueError(
                    f"Ambiguous chain-to-protomer mapping for chain {chain_id!r}: "
                    f"{existing!r} vs {proto_name!r} across oligomers."
                )
            chain2proto[chain_id] = proto_name
    return chain2proto


def aggregate_per_proto_scores(
    evaluation_scores,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Fold evaluator per-chain/per-protomer scores back onto configured protomers."""
    combined_scores = combine_per_proto_mutator_scores(
        evaluation_scores,
        runtime_config=runtime_config,
    )
    if combined_scores is not None:
        return {
            proto_name: np.asarray(score_values, dtype=float)
            for proto_name, score_values in combined_scores.items()
        }

    protomer_names = set(global_config.protomers.keys())
    raw_scores = getattr(evaluation_scores, "per_proto_scores", None) or {}
    explicit_chain2proto = dict((runtime_config or {}).get("chain2proto") or {})
    try:
        derived_chain2proto = _derive_chain2proto_from_oligomers(
            getattr(evaluation_scores, "names", None)
        )
    except ValueError:
        derived_chain2proto = {}
    resolved_scores: dict[str, list[np.ndarray]] = defaultdict(list)

    for source_name, score_list in raw_scores.items():
        target_name = explicit_chain2proto.get(source_name)
        if target_name is None:
            target_name = derived_chain2proto.get(source_name)
        if target_name is None and source_name in protomer_names:
            target_name = source_name
        if target_name not in protomer_names:
            raise KeyError(
                f"Per-proto score key {source_name!r} could not be resolved to a configured protomer. "
                f"Known protomers: {sorted(protomer_names)}; "
                f"explicit chain2proto keys: {sorted(explicit_chain2proto)}; "
                f"derived chain2proto keys: {sorted(derived_chain2proto)}."
            )
        resolved_scores[target_name].extend(score_list)

    return {
        proto_name: np.asarray(score_list, dtype=float)
        for proto_name, score_list in resolved_scores.items()
    }


def propagate_evaluation_scores_to_protomers(
    evaluation_scores,
    runtime_config: Mapping[str, Any] | None = None,
) -> None:
    for proto_name, scores in aggregate_per_proto_scores(
        evaluation_scores, runtime_config
    ).items():
        global_config.protomers[proto_name].propagating_scores(scores)


def concatenate_proteins(protein_A, protein_B):
    """
    Concatenates two protein structures along the X-axis, aligning their C-alpha atoms.

    Parameters:
        protein_A (Protein): First protein structure
        protein_B (Protein): Second protein structure

    Returns:
        Protein: Concatenated protein structure
    """
    # Convert atom positions to numpy arrays for manipulation
    atom_positions_A = np.array(protein_A.atom_positions)
    atom_positions_B = np.array(protein_B.atom_positions)

    # Extract C-alpha X-coordinates to align protein structures
    CA_pos_A = atom_positions_A[:, 1, 0]
    CA_pos_B = atom_positions_B[:, 1, 0]

    # Calculate boundary displacements to avoid atom overlap
    max_coord_A = np.max(CA_pos_A)
    min_coord_B = np.min(CA_pos_B)
    atom_positions_A[:, :, 0] -= max_coord_A + 10  # Offset by 10 units
    atom_positions_B[:, :, 0] -= min_coord_B - 10  # Offset by 10 units

    # Concatenate all attributes of the proteins into a single structure
    combined_positions = np.concatenate((atom_positions_A, atom_positions_B), axis=0)
    combined_chain_index = np.concatenate(
        (protein_A.chain_index, protein_B.chain_index + max(protein_A.chain_index) + 1),
        axis=0,
    )
    combined_residue_index = np.concatenate(
        (
            protein_A.residue_index,
            protein_B.residue_index + max(protein_A.residue_index) + 1,
        ),
        axis=0,
    )
    combined_atom_mask = np.concatenate(
        (protein_A.atom_mask, protein_B.atom_mask), axis=0
    )
    combined_aatype = np.concatenate((protein_A.aatype, protein_B.aatype), axis=0)
    combined_b_factors = np.concatenate(
        (protein_A.b_factors, protein_B.b_factors), axis=0
    )

    return Protein(
        atom_positions=combined_positions,
        chain_index=combined_chain_index,
        residue_index=combined_residue_index,
        atom_mask=combined_atom_mask,
        aatype=combined_aatype,
        b_factors=combined_b_factors,
    )


def _select_random_positions(proto_seqs, n_mutations, excluded_pos_per_proto):
    """Select positions randomly for each protomer."""
    mutable_positions = {}
    for proto, seq in proto_seqs.items():
        exclude_pos = excluded_pos_per_proto[proto]
        design_pos = np.setdiff1d(np.arange(len(seq)), exclude_pos)
        n_mut = min(n_mutations[proto], len(design_pos))
        mutable_positions[proto] = np.random.choice(
            design_pos, size=n_mut, replace=False
        )
    return mutable_positions


def _select_plddt_positions(
    proto_seqs,
    n_mutations,
    excluded_pos_per_proto,
    per_proto_scores,
    mutate_plddt_quantile=1.0,
):
    """Select positions based on pLDDT scores."""
    mutable_positions = {}

    for proto, plddts in per_proto_scores.items():
        proto_len = len(proto_seqs[proto])
        exclude_pos = excluded_pos_per_proto[proto]

        # Generate position weights with reduced weights at termini
        weights = np.array(
            [0.25, 0.5, 0.75] + [1.0] * (proto_len - 6) + [0.75, 0.5, 0.25]
        )

        # Get lowest pLDDT positions within quantile
        n_potential = round(proto_len * mutate_plddt_quantile)
        consensus_min = np.min(plddts, axis=0)
        potential_sites = [
            pos for pos in np.argsort(consensus_min) if pos not in exclude_pos
        ][:n_potential]

        if not potential_sites:
            mutable_positions[proto] = np.array([])
            continue

        # Calculate weights for potential sites
        site_weights = weights[potential_sites]
        site_weights = site_weights / site_weights.sum()

        # Select final mutation sites
        n_mut = min(n_mutations[proto], len(potential_sites))
        selected_sites = np.random.choice(
            potential_sites, size=n_mut, replace=False, p=site_weights
        )

        mutable_positions[proto] = selected_sites

    return mutable_positions


def select_positions(
    n_mutations,
    proto_seqs,
    per_proto_scores,
    select_positions,
    select_position_params,
    excluded_pos_per_proto,
):
    """
    Select mutable positions in each protomer based on a specific method.

    Args:
        n_mutations: Dict mapping protomers to number of mutations
        proto_seqs: Dict mapping protomers to sequences
        per_proto_scores: Dict mapping protomers to pLDDT scores
        select_positions: Method for selecting positions ('random' or 'plddt')
        select_position_params: Parameters for position selection (e.g. pLDDT quantile)
        excluded_pos_per_proto: Dict mapping protomers to positions that cannot be mutated

    Returns:
        Dict mapping protomers to arrays of mutable positions
    """
    if select_positions == "random":
        return _select_random_positions(proto_seqs, n_mutations, excluded_pos_per_proto)

    elif select_positions == "plddt":
        mutate_plddt_quantile = (
            float(select_position_params) if select_position_params else 1.0
        )
        return _select_plddt_positions(
            proto_seqs,
            n_mutations,
            excluded_pos_per_proto,
            per_proto_scores,
            mutate_plddt_quantile,
        )
    else:
        raise ValueError(f"Unknown position selection method: {select_positions}")


def mutate(method, proto_sequences, mutable_positions=None):
    """Mutate mutable positions based on a specific method."""
    # aa_freq = cached_aa_freq(self.exclude_AA)
    aa_freq = cached_aa_freq(global_config.exclude_AA)
    mutated_proto_sequences = {}
    mutated_sites = {
        proto: {} for proto in proto_sequences.keys()
    }  # {proto: {sites : (original_aa, mutated_aa)}}

    if method not in ["uniform", "frequency_adjusted", "blosum62", "pssm"]:
        logging.info(
            f"ERROR: sequence mutation method [{method}] does not exist. Only 'uniform', 'frequency_adjusted', 'blosum62', 'pssm' are supported."
        )
        sys.exit()

    if method == "uniform":
        # Mutate positions with uniform probabilities.
        for proto, seq in proto_sequences.items():
            for p in mutable_positions[proto]:
                aa_sample = np.random.choice(list(set(aa_freq.keys()) - set(seq[p])))
                mutated_sites[proto][p] = (seq[p], aa_sample)
                seq = seq[:p] + aa_sample + seq[p + 1 :]

            mutated_proto_sequences[proto] = seq

    elif method == "frequency_adjusted":
        # Mutate positions based on frequency-ajusted AA substitutions.
        for proto, seq in proto_sequences.items():
            for p in mutable_positions[proto]:
                aa_sample = np.random.choice(
                    list(aa_freq.keys()), p=list(aa_freq.values())
                )
                mutated_sites[proto][p] = (seq[p], aa_sample)
                seq = seq[:p] + aa_sample + seq[p + 1 :]

            mutated_proto_sequences[proto] = seq

    elif method == "blosum62":
        # Mutate positions based on BLOSUM62 substituation matrix.
        # BLOSUM62 probabilies taken from:
        # https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/src/algo/blast/composition_adjustment/matrix_frequency_data.c

        # Joint probabilities for BLOSUM62.
        joint_p = JOINT_PROBABILITY

        # Background frequencies for BLOSUM62.
        bg_freqs = BACKGROUND_FREQUENCIES

        # AA order.
        aas = np.array(list("ARNDCQEGHILKMFPSTWYV"))

        # Generate substituation probability matrix -- code from Chris Norn.
        M = np.zeros((20, 20))
        S = np.zeros((20, 20))
        for i in range(0, 20):
            for j in range(0, 20):
                M[i, j] = joint_p[i][j]
                S[i, j] = (
                    2 * np.log(joint_p[i][j] / (bg_freqs[i] * bg_freqs[j])) / np.log(2)
                )

        P_conditional = M / bg_freqs
        np.fill_diagonal(
            P_conditional, 0.0
        )  # we don't want to make self substitutions.

        # Make BLOSUM62-informed substituations.
        for proto, seq in proto_sequences.items():
            for p in mutable_positions[proto]:
                current_aa = seq[p]
                idx = np.argwhere(aas == current_aa)[0][0]
                sub_prob_renorm = (
                    P_conditional[:, idx] / P_conditional[:, idx].sum()
                )  # get subsitutation vector for that aa, and renormalise the vector.
                sub_prob = {a: f for a, f in list(zip(aas, sub_prob_renorm))}

                aa_sample = np.random.choice(
                    list(sub_prob.keys()), p=list(sub_prob.values())
                )
                mutated_sites[proto][p] = (seq[p], aa_sample)
                seq = seq[:p] + aa_sample + seq[p + 1 :]

            mutated_proto_sequences[proto] = seq

    elif method == "pssm":
        # Mutate positions based on a user-defined PSSM.
        # PSSM needs to be a .csv of shape L,20
        # Each vector needs to sum to 1.
        # AA order.

        aas = np.array(list("ARNDCQEGHILKMFPSTWYV"))
        with open("pssm.csv", "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        pssm = np.array([l.split(",") for l in lines], dtype=float)

        for proto, seq in proto_sequences.items():
            for p in mutable_positions[proto]:
                frequencies = pssm[p]
                aa_sample = np.random.choice(aas, p=frequencies)
                mutated_sites[proto][p] = (seq[p], aa_sample)

                seq = seq[:p] + aa_sample + seq[p + 1 :]

            mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences, mutated_sites


def build_mutator_runtime_config(
    step, updated_sites=None, outprefix=None, former_config=None
):
    former_config = former_config or {}
    stage_oligos = former_config.get("oligo")
    try:
        stage_manager = getattr(global_config, "stage_manager", None)
    except Exception:
        stage_manager = None
    current_stage_oligos = (
        getattr(stage_manager, "current_oligo", None)
        if stage_manager is not None
        else None
    )
    if current_stage_oligos is not None:
        stage_oligos = list(current_stage_oligos)
    transition_consistency_gate_context = getattr(
        stage_manager, "transition_consistency_gate_context", None
    )
    ignored_complexes = None
    negative_design_complexes = None
    try:
        ignored_complexes = list(global_config.ignored_complexes or [])
    except Exception:
        ignored_complexes = former_config.get("ignored_complexes")
    try:
        negative_design_complexes = list(
            global_config.negative_design_complexes or []
        )
    except Exception:
        negative_design_complexes = former_config.get("negative_design_complexes")
    former_config.update(
        {
            "step": step,
            "updated_sites": updated_sites,
            "outprefix": outprefix,
            "transition_consistency_gate_context": transition_consistency_gate_context,
            "oligo": stage_oligos,
            "ignored_complexes": ignored_complexes,
            "negative_design_complexes": negative_design_complexes,
        }
    )
    return former_config


def prepare_exclude_sites(
    position_selector,
    evaluation_result,
    protomers,
    runtime_config: Mapping[str, Any] | None = None,
    mutator_alias: str | None = None,
    *,
    fallback_to_all_mutable_if_selector_empty: bool = True,
):
    """Prepare a list of sites to exclude from mutation.

    Args:
        position_selector: Position position_selector configuration
        evaluation_result: Results from previous evaluations
        protomers[: Protomer object containing sequence information
        updated_sites: Sites that have already been mutated

    Returns:
        Dictionary mapping protomers to lists of positions to exclude from mutation
    """
    # Get mask of positions that can be mutated based on position_selector criteria
    mutable_mask_selector = construct_position_selector(
        position_selector, evaluation_result, protomers
    )
    mutable_mask_protomer = {
        proto: protomers[proto].mutable_mask for proto in protomers.keys()
    }

    def aggregate_bools(all_masks):
        mutable_mask = {}
        for proto in protomers:
            if protomers[proto].type not in MutableTypes:
                continue

            all_masks_per_proto = [
                mask.get(proto, np.ones(protomers[proto].length, dtype=bool))
                for mask in all_masks
            ]
            if len(all_masks_per_proto) < 2:
                mutable_mask[proto] = all_masks_per_proto[0]
            else:
                mutable_mask[proto] = np.logical_and(
                    *all_masks_per_proto,
                )
        return mutable_mask

    mutable_mask_wt_selector = aggregate_bools(
        [mutable_mask_selector, mutable_mask_protomer]
    )
    mutable_mask_wo_selector = aggregate_bools([mutable_mask_protomer])
    if runtime_config is not None and bool(
        runtime_config.get("record_mutator_selector_scores", False)
    ):
        alias = mutator_alias or runtime_config.get("current_mutator_alias") or "mutator"
        context = build_mutator_selector_score_context(
            mutator_alias=str(alias),
            selector_expression=position_selector,
            evaluation_result=evaluation_result,
            selected_masks=mutable_mask_wt_selector,
            runtime_config=runtime_config,
        )
        runtime_config.setdefault("mutator_selector_score_contexts", []).append(context)

    # Identify positions that cannot be mutated based on the mask

    # Combine all types of positions that should be excluded:
    # 1. Fixed positions specified by the user
    # 2. Already mutated sites
    # 3. Positions excluded by the position_selector

    # todo aggregate across different segments
    for mask in [mutable_mask_wt_selector, mutable_mask_wo_selector]:
        for name, protomer in protomers.items():
            mask[name] = protomer.aggregate_bool(mask[name])

    final_excluded_positions = {}
    for name, protomer in protomers.items():
        if protomer.type not in MutableTypes:
            continue
        if (
            sum(mutable_mask_wt_selector[name]) == 0
            and fallback_to_all_mutable_if_selector_empty
        ):
            logging.warning(
                "Position selector for protomer %s selected no mutable positions; "
                "falling back to all mutable positions.",
                name,
            )
            final_excluded_positions[name] = list(
                np.where(~mutable_mask_wo_selector[name])[0]
            )
        else:
            final_excluded_positions[name] = list(
                np.where(~mutable_mask_wt_selector[name])[0]
            )
    return final_excluded_positions


BACKGROUND_FREQUENCIES = [
    7.4216205067993410e-02,
    5.1614486141284638e-02,
    4.4645808512757915e-02,
    5.3626000838554413e-02,
    2.4687457167944848e-02,
    3.4259650591416023e-02,
    5.4311925684587502e-02,
    7.4146941452644999e-02,
    2.6212984805266227e-02,
    6.7917367618953756e-02,
    9.8907868497150955e-02,
    5.8155682303079680e-02,
    2.4990197579643110e-02,
    4.7418459742284751e-02,
    3.8538003320306206e-02,
    5.7229029476494421e-02,
    5.0891364550287033e-02,
    1.3029956129972148e-02,
    3.2281512313758580e-02,
    7.2919098205619245e-02,
]
JOINT_PROBABILITY = [
    [
        2.1497573378347484e-02,
        2.3470224274721213e-03,
        1.9493235258876179e-03,
        2.1674844853066858e-03,
        1.5903351423026848e-03,
        1.9242657898716525e-03,
        2.9879059292799641e-03,
        5.8158526388051033e-03,
        1.1076584657559144e-03,
        3.1880644746334580e-03,
        4.4186245468471547e-03,
        3.3466571942021082e-03,
        1.3412107617355408e-03,
        1.6360627863999076e-03,
        2.1568959784943114e-03,
        6.2524987419815400e-03,
        3.7180506975672363e-03,
        4.0281679108936688e-04,
        1.2999956675626666e-03,
        5.0679056444508912e-03,
    ],
    [
        2.3470224274721213e-03,
        1.7757465118386322e-02,
        1.9786027128591904e-03,
        1.5865480081162602e-03,
        3.9365984789376245e-04,
        2.4858611089731411e-03,
        2.6933867548771758e-03,
        1.7221140903704937e-03,
        1.2407382229440791e-03,
        1.2435878276496955e-03,
        2.4193952633248727e-03,
        6.2339060289407083e-03,
        8.0309461712520876e-04,
        9.3181986323789834e-04,
        9.5783034332718700e-04,
        2.2660898636037261e-03,
        1.7802796534180537e-03,
        2.6571979312581875e-04,
        9.2634607111251918e-04,
        1.5810185245264004e-03,
    ],
    [
        1.9493235258876179e-03,
        1.9786027128591904e-03,
        1.4140291972553610e-02,
        3.7201973506001745e-03,
        4.3845466068066216e-04,
        1.5304436972610567e-03,
        2.2097156829738759e-03,
        2.8591871815612977e-03,
        1.4301072616183181e-03,
        9.9437221166923172e-04,
        1.3690958423974782e-03,
        2.4402105140841090e-03,
        5.2943633069226512e-04,
        7.5004227978192801e-04,
        8.6016459857770028e-04,
        3.1466019144814608e-03,
        2.2360795375444384e-03,
        1.6159545671597605e-04,
        7.0048422794024819e-04,
        1.2014015528772706e-03,
    ],
    [
        2.1674844853066858e-03,
        1.5865480081162602e-03,
        3.7201973506001745e-03,
        2.1274574617480089e-02,
        3.9909227141697264e-04,
        1.6481246723433428e-03,
        4.9158017471929655e-03,
        2.5221102126636373e-03,
        9.5384849402143984e-04,
        1.2347404942429857e-03,
        1.5202051791453383e-03,
        2.4453087721980561e-03,
        4.6429229320514104e-04,
        7.6023722413111566e-04,
        1.2373315413524663e-03,
        2.8035127901697272e-03,
        1.8961512776990257e-03,
        1.6218020183662784e-04,
        5.9842263937853702e-04,
        1.3158365660538270e-03,
    ],
    [
        1.5903351423026848e-03,
        3.9365984789376245e-04,
        4.3845466068066216e-04,
        3.9909227141697264e-04,
        1.1931352277704348e-02,
        3.0937204045913537e-04,
        3.8338775043186374e-04,
        7.6951976030099293e-04,
        2.2976387481074697e-04,
        1.0956590131781735e-03,
        1.5682982157153873e-03,
        5.0124929379033781e-04,
        3.7717165634097634e-04,
        5.1389991547056834e-04,
        3.6111795849154795e-04,
        1.0432626586831986e-03,
        9.3041313726939057e-04,
        1.4474923964368156e-04,
        3.4603772624580643e-04,
        1.3606607271146112e-03,
    ],
    [
        1.9242657898716525e-03,
        2.4858611089731411e-03,
        1.5304436972610567e-03,
        1.6481246723433428e-03,
        3.0937204045913537e-04,
        7.3292255467189687e-03,
        3.5385780499965817e-03,
        1.3683038039160171e-03,
        1.0489026828741754e-03,
        8.9102936026571569e-04,
        1.6174411456311808e-03,
        3.0968229715707327e-03,
        7.3993258722701268e-04,
        5.4255147972143906e-04,
        8.4668181752066874e-04,
        1.8931125300036275e-03,
        1.3796838284921874e-03,
        2.2737931366728891e-04,
        6.7584155312457842e-04,
        1.1660966117775285e-03,
    ],
    [
        2.9879059292799641e-03,
        2.6933867548771758e-03,
        2.2097156829738759e-03,
        4.9158017471929655e-03,
        3.8338775043186374e-04,
        3.5385780499965817e-03,
        1.6133927472163669e-02,
        1.9380952488713059e-03,
        1.3667885452189439e-03,
        1.2192061706431622e-03,
        2.0030316026648431e-03,
        4.1322603720305197e-03,
        6.7909745467514783e-04,
        8.5179405867513139e-04,
        1.4216207127018586e-03,
        2.9539180653600089e-03,
        2.0493063257644955e-03,
        2.6488552587183780e-04,
        8.7044186256788659e-04,
        1.6987763526262680e-03,
    ],
    [
        5.8158526388051033e-03,
        1.7221140903704937e-03,
        2.8591871815612977e-03,
        2.5221102126636373e-03,
        7.6951976030099293e-04,
        1.3683038039160171e-03,
        1.9380952488713059e-03,
        3.7804346453413303e-02,
        9.5813607255887238e-04,
        1.3849118546156933e-03,
        2.0864716056392773e-03,
        2.5392537741810947e-03,
        7.3281559749652399e-04,
        1.1976708695723554e-03,
        1.3641171883713547e-03,
        3.8342830901664762e-03,
        2.1858459940987062e-03,
        4.0740829083805248e-04,
        8.3467413018106177e-04,
        1.8218235950233687e-03,
    ],
    [
        1.1076584657559144e-03,
        1.2407382229440791e-03,
        1.4301072616183181e-03,
        9.5384849402143984e-04,
        2.2976387481074697e-04,
        1.0489026828741754e-03,
        1.3667885452189439e-03,
        9.5813607255887238e-04,
        9.2802502369336622e-03,
        5.8089627083019206e-04,
        9.8696608463236094e-04,
        1.1873625842258938e-03,
        3.8264639620910225e-04,
        8.1041076335565583e-04,
        4.7770135861914477e-04,
        1.1052034635193162e-03,
        7.4371746073077327e-04,
        1.5168037757411286e-04,
        1.5213771111755425e-03,
        6.4882907765797669e-04,
    ],
    [
        3.1880644746334580e-03,
        1.2435878276496955e-03,
        9.9437221166923172e-04,
        1.2347404942429857e-03,
        1.0956590131781735e-03,
        8.9102936026571569e-04,
        1.2192061706431622e-03,
        1.3849118546156933e-03,
        5.8089627083019206e-04,
        1.8441526588740136e-02,
        1.1382470627796603e-02,
        1.5655862274689192e-03,
        2.5081290988482057e-03,
        3.0458868657559346e-03,
        1.0068164685944146e-03,
        1.7225081689171561e-03,
        2.6953622613315018e-03,
        3.6183761166072852e-04,
        1.3821121844492116e-03,
        1.1972663837662637e-02,
    ],
    [
        4.4186245468471547e-03,
        2.4193952633248727e-03,
        1.3690958423974782e-03,
        1.5202051791453383e-03,
        1.5682982157153873e-03,
        1.6174411456311808e-03,
        2.0030316026648431e-03,
        2.0864716056392773e-03,
        9.8696608463236094e-04,
        1.1382470627796603e-02,
        3.7141460156350926e-02,
        2.4634345023228079e-03,
        4.9293545515183088e-03,
        5.4151301166464015e-03,
        1.4146090399381900e-03,
        2.4277107072013821e-03,
        3.3238031308707055e-03,
        7.3206640617832933e-04,
        2.2096734692836624e-03,
        9.4786263030457313e-03,
    ],
    [
        3.3466571942021082e-03,
        6.2339060289407083e-03,
        2.4402105140841090e-03,
        2.4453087721980561e-03,
        5.0124929379033781e-04,
        3.0968229715707327e-03,
        4.1322603720305197e-03,
        2.5392537741810947e-03,
        1.1873625842258938e-03,
        1.5655862274689192e-03,
        2.4634345023228079e-03,
        1.6113385590544604e-02,
        9.0876633395557617e-04,
        9.4875149773685364e-04,
        1.5773020912564391e-03,
        3.1016069999481111e-03,
        2.3467014804084987e-03,
        2.7198500003555514e-04,
        9.9908866586876396e-04,
        1.9360424083099779e-03,
    ],
    [
        1.3412107617355408e-03,
        8.0309461712520876e-04,
        5.2943633069226512e-04,
        4.6429229320514104e-04,
        3.7717165634097634e-04,
        7.3993258722701268e-04,
        6.7909745467514783e-04,
        7.3281559749652399e-04,
        3.8264639620910225e-04,
        2.5081290988482057e-03,
        4.9293545515183088e-03,
        9.0876633395557617e-04,
        4.0477309321969848e-03,
        1.1901770463553603e-03,
        4.0824445213456919e-04,
        8.5603787638552766e-04,
        1.0095451907679563e-03,
        1.9872537223131380e-04,
        5.7145288352831449e-04,
        2.3123361470140736e-03,
    ],
    [
        1.6360627863999076e-03,
        9.3181986323789834e-04,
        7.5004227978192801e-04,
        7.6023722413111566e-04,
        5.1389991547056834e-04,
        5.4255147972143906e-04,
        8.5179405867513139e-04,
        1.1976708695723554e-03,
        8.1041076335565583e-04,
        3.0458868657559346e-03,
        5.4151301166464015e-03,
        9.4875149773685364e-04,
        1.1901770463553603e-03,
        1.8277684015431908e-02,
        5.2528021756783813e-04,
        1.1939618185901600e-03,
        1.1624184369750680e-03,
        8.4917468952377874e-04,
        4.2392005745634370e-03,
        2.5763052227920180e-03,
    ],
    [
        2.1568959784943114e-03,
        9.5783034332718700e-04,
        8.6016459857770028e-04,
        1.2373315413524663e-03,
        3.6111795849154795e-04,
        8.4668181752066874e-04,
        1.4216207127018586e-03,
        1.3641171883713547e-03,
        4.7770135861914477e-04,
        1.0068164685944146e-03,
        1.4146090399381900e-03,
        1.5773020912564391e-03,
        4.0824445213456919e-04,
        5.2528021756783813e-04,
        1.9066033679132538e-02,
        1.6662567934883051e-03,
        1.3511005665728870e-03,
        1.4152209821874487e-04,
        4.5224391125285910e-04,
        1.2451325046931832e-03,
    ],
    [
        6.2524987419815400e-03,
        2.2660898636037261e-03,
        3.1466019144814608e-03,
        2.8035127901697272e-03,
        1.0432626586831986e-03,
        1.8931125300036275e-03,
        2.9539180653600089e-03,
        3.8342830901664762e-03,
        1.1052034635193162e-03,
        1.7225081689171561e-03,
        2.4277107072013821e-03,
        3.1016069999481111e-03,
        8.5603787638552766e-04,
        1.1939618185901600e-03,
        1.6662567934883051e-03,
        1.2585947097159817e-02,
        4.7004857686835334e-03,
        2.8731729176487776e-04,
        1.0299846310599138e-03,
        2.3587292053265561e-03,
    ],
    [
        3.7180506975672363e-03,
        1.7802796534180537e-03,
        2.2360795375444384e-03,
        1.8961512776990257e-03,
        9.3041313726939057e-04,
        1.3796838284921874e-03,
        2.0493063257644955e-03,
        2.1858459940987062e-03,
        7.4371746073077327e-04,
        2.6953622613315018e-03,
        3.3238031308707055e-03,
        2.3467014804084987e-03,
        1.0095451907679563e-03,
        1.1624184369750680e-03,
        1.3511005665728870e-03,
        4.7004857686835334e-03,
        1.2514818886617953e-02,
        2.8575770858467209e-04,
        9.4161039895612720e-04,
        3.6402328079338207e-03,
    ],
    [
        4.0281679108936688e-04,
        2.6571979312581875e-04,
        1.6159545671597605e-04,
        1.6218020183662784e-04,
        1.4474923964368156e-04,
        2.2737931366728891e-04,
        2.6488552587183780e-04,
        4.0740829083805248e-04,
        1.5168037757411286e-04,
        3.6183761166072852e-04,
        7.3206640617832933e-04,
        2.7198500003555514e-04,
        1.9872537223131380e-04,
        8.4917468952377874e-04,
        1.4152209821874487e-04,
        2.8731729176487776e-04,
        2.8575770858467209e-04,
        6.4699301717154852e-03,
        8.8744160259272527e-04,
        3.5578318710317554e-04,
    ],
    [
        1.2999956675626666e-03,
        9.2634607111251918e-04,
        7.0048422794024819e-04,
        5.9842263937853702e-04,
        3.4603772624580643e-04,
        6.7584155312457842e-04,
        8.7044186256788659e-04,
        8.3467413018106177e-04,
        1.5213771111755425e-03,
        1.3821121844492116e-03,
        2.2096734692836624e-03,
        9.9908866586876396e-04,
        5.7145288352831449e-04,
        4.2392005745634370e-03,
        4.5224391125285910e-04,
        1.0299846310599138e-03,
        9.4161039895612720e-04,
        8.8744160259272527e-04,
        1.0246100213822419e-02,
        1.5489827890922993e-03,
    ],
    [
        5.0679056444508912e-03,
        1.5810185245264004e-03,
        1.2014015528772706e-03,
        1.3158365660538270e-03,
        1.3606607271146112e-03,
        1.1660966117775285e-03,
        1.6987763526262680e-03,
        1.8218235950233687e-03,
        6.4882907765797669e-04,
        1.1972663837662637e-02,
        9.4786263030457313e-03,
        1.9360424083099779e-03,
        2.3123361470140736e-03,
        2.5763052227920180e-03,
        1.2451325046931832e-03,
        2.3587292053265561e-03,
        3.6402328079338207e-03,
        3.5578318710317554e-04,
        1.5489827890922993e-03,
        1.9631915140537640e-02,
    ],
]


def sort_probabilities_by_aaorder(probabilities: np.ndarray, aa_order: List[str]):
    try:
        reorder = [restype_str_to_int[aa] for aa in aa_order]
    except KeyError:
        # Non-protein alphabets (e.g., RNA A/C/G/U) are not part of restype_str_to_int.
        # In that case, assume the last dim already matches the requested order.
        if probabilities.shape[-1] != len(aa_order):
            raise
        reorder = list(range(len(aa_order)))

    reordered_probabilities = probabilities[..., reorder]
    return (
        reordered_probabilities.cpu().numpy()
        if (torch is not None and isinstance(reordered_probabilities, torch.Tensor))
        else reordered_probabilities
    )


def _condition_non_wt_probabilities(
    probabilities: np.ndarray,
    log_probabilities: np.ndarray,
    sequence: str,
    aa_order: List[str],
    *,
    protomer_name: str,
) -> np.ndarray:
    original = np.asarray(probabilities, dtype=float)
    adjusted_logits = np.asarray(log_probabilities, dtype=float).copy()
    if original.shape != adjusted_logits.shape:
        raise ValueError(
            f"Cannot enforce mutation_num for protomer {protomer_name!r}: "
            f"sampling_probs shape {original.shape} does not match "
            f"log_probs shape {adjusted_logits.shape}."
        )
    if adjusted_logits.ndim < 2:
        raise ValueError(
            f"Cannot enforce mutation_num for protomer {protomer_name!r}: "
            "expected probabilities with at least 2 dimensions, "
            f"got {adjusted_logits.shape}."
        )
    if adjusted_logits.shape[-2] != len(sequence):
        raise ValueError(
            f"Cannot enforce mutation_num for protomer {protomer_name!r}: "
            f"sequence length {len(sequence)} does not match probability length "
            f"{adjusted_logits.shape[-2]}."
        )

    active_rows = original.sum(axis=-1) > 0
    aa_to_index = {str(aa): idx for idx, aa in enumerate(aa_order)}
    for position, current_aa in enumerate(sequence):
        aa_index = aa_to_index.get(current_aa)
        if aa_index is None:
            raise ValueError(
                f"Cannot enforce mutation_num for protomer {protomer_name!r}: "
                f"current amino acid {current_aa!r} at position {position + 1} "
                "is not present in aa_order."
            )
        adjusted_logits[..., position, aa_index] = -np.inf

    safe_logits = np.where(active_rows[..., None], adjusted_logits, 0.0)
    row_max = np.max(safe_logits, axis=-1, keepdims=True)
    adjusted = np.exp(safe_logits - row_max)
    adjusted = np.where(active_rows[..., None], adjusted, 0.0)
    row_sums = adjusted.sum(axis=-1, keepdims=True)
    return np.divide(
        adjusted,
        row_sums,
        out=np.zeros_like(adjusted),
        where=row_sums > 0,
    )


def sample_mutations_by_aa_probs(
    evaluation_scores,
    mutation_output,
    aa_order,
    mutation_weight,
    protomer_type: ProtomerTypes = ProtomerTypes.PROTEIN,
    per_proto_limits: dict[str, int] | None = None,
    propagate_scores: bool = True,
    runtime_config: dict[str, Any] | None = None,
    entropy_cutoff: float | None = None,
    enforce_mutation_num: bool = False,
):
    if propagate_scores:
        per_proto_scores = aggregate_per_proto_scores(
            evaluation_scores,
            runtime_config=runtime_config,
        )
        for name, scores in per_proto_scores.items():
            protomer = global_config.protomers.get(name)
            if protomer is None or protomer.type != protomer_type:
                continue
            scores_array = np.asarray(scores)
            if scores_array.ndim == 0:
                continue
            if scores_array.shape[-1] != protomer.length:
                logging.debug(
                    "Skipping score propagation for %s: got length %s, expected %s",
                    name,
                    scores_array.shape[-1],
                    protomer.length,
                )
                continue
            protomer.propagating_scores(scores_array)
    write_state_conflict_diagnostics(
        evaluation_scores,
        mutation_output,
        aa_order,
        protomer_type=protomer_type,
        runtime_config=runtime_config,
    )
    proto_probabilities = collect_blended_proto_probabilities(
        evaluation_scores,
        mutation_output,
        protomer_type=protomer_type,
        runtime_config=runtime_config,
    )
    proto_log_probabilities = None
    if enforce_mutation_num:
        proto_log_probabilities = collect_proto_probabilities(
            evaluation_scores,
            mutation_output,
            protomer_type=protomer_type,
            value_key="log_probs",
        )

    for name, probs in proto_probabilities.items():
        protomer = global_config.protomers[name]
        sorted_probs = sort_probabilities_by_aaorder(probs, aa_order)
        if enforce_mutation_num:
            if name not in proto_log_probabilities:
                raise ValueError(
                    f"enforce_mutation_num requires log_probs for protomer {name!r}."
                )
            sorted_log_probs = sort_probabilities_by_aaorder(
                proto_log_probabilities[name], aa_order
            )
            sorted_probs = _condition_non_wt_probabilities(
                sorted_probs,
                sorted_log_probs,
                protomer.sequence,
                aa_order,
                protomer_name=name,
            )
        protomer.propagating_probs(sorted_probs)
    mutations = global_config.segment_manager.collect_mutations()

    # todo filter out by position_selector

    def sort_by_entropy(mutations):
        """
        Sort mutations by entropy in ascending order.
        """
        return sorted(
            mutations,
            key=lambda item: item[1][-2],
        )

    def shuffle_mutations_by_entropy(mutations, threshold=1.5):
        """
        Shuffle the low-entropy prefix and keep the full mutation list.
        """
        # split by threshold
        to_shuffle, not_shuffle = [], []
        for i, mutation in enumerate(mutations):
            if mutation[1][-2] < threshold:
                to_shuffle.append(mutation)
            else:
                not_shuffle.append(mutation)

        np.random.shuffle(to_shuffle)
        result = to_shuffle + not_shuffle
        return result

    sorted_mutations = sort_by_entropy(mutations)
    if entropy_cutoff is not None:
        cutoff = float(entropy_cutoff)
        before_count = len(sorted_mutations)
        sorted_mutations = [
            mutation for mutation in sorted_mutations if float(mutation[1][-2]) <= cutoff
        ]
        logging.debug(
            "Entropy cutoff %.3f kept %d/%d mutation candidates",
            cutoff,
            len(sorted_mutations),
            before_count,
        )

    shuffled_mutations = shuffle_mutations_by_entropy(sorted_mutations)

    if per_proto_limits:
        segment_to_proto = {}
        for proto_name, proto in global_config.protomers.items():
            for segment_name in proto.components:
                segment_to_proto[segment_name] = proto_name

        proto_usage = defaultdict(int)
        budgeted = []
        for segment_name, mutation in shuffled_mutations:
            proto_name = segment_to_proto.get(segment_name)
            if proto_name is None:
                continue
            limit = per_proto_limits.get(proto_name)
            if limit is not None and proto_usage[proto_name] >= limit:
                continue
            budgeted.append((segment_name, mutation))
            proto_usage[proto_name] += 1
        shuffled_mutations = budgeted

    requested_mutations = round(global_config.mutation_num * mutation_weight)
    if (
        enforce_mutation_num
        and requested_mutations > 0
        and len(shuffled_mutations) < requested_mutations
    ):
        raise ValueError(
            "enforce_mutation_num requested "
            f"{requested_mutations} mutations but only {len(shuffled_mutations)} "
            "non-WT mutation candidates are available."
        )
    n_mutations = max(min(requested_mutations, len(shuffled_mutations)), 0)

    global_config.segment_manager.add_mutations(shuffled_mutations, n_mutations)
