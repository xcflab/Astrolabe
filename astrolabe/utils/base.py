import logging
import colorsys
from collections import defaultdict
from typing import Dict

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes
from astrolabe.utils.log_events import (
    build_mutation_summary_message,
    build_no_change_message,
)
from astrolabe.utils.logging_utils import (
    colorize,
    format_metric_value,
    mean_or_none,
)
from astrolabe.core.folding_input import (
    DnaChain,
    Input,
    Ligand,
    ProteinChain,
    RnaChain,
)
from astrolabe.core.cyclization import (
    derive_cyclic_chain_ids,
    derive_highfold_cycpoem_relative_patches,
    derive_cyclic_relative_patches,
    derive_global_cyclic_relative_patches,
    expand_cyclization_bonds_for_oligo,
    filter_expanded_bonds_for_signal_modes,
    parse_cyclization,
    resolve_cyclization_bonds,
    resolve_cyclization_bond_specs,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id

TELEMETRY_FIELDS = ("entropy_norm", "selection_score", "secondary_structure")
MAX_SITE_TELEMETRY_PER_PROTO = 12


def _normalize_finite_values(values: list[float | None]) -> list[float]:
    finite_values: list[float] = []
    for value in values:
        numeric_value = _optional_float(value)
        if numeric_value is not None:
            finite_values.append(numeric_value)
    return finite_values


def _metric_color(
    value: float | None,
    *,
    all_values: list[float | None],
    lower_is_better: bool,
    fallback: str,
) -> str:
    numeric_value = _optional_float(value)
    if numeric_value is None:
        return fallback
    finite_values = _normalize_finite_values(all_values)
    if len(finite_values) < 2:
        return "ansi256:226"
    minimum = min(finite_values)
    maximum = max(finite_values)
    span = maximum - minimum
    if span <= 1e-12:
        return "ansi256:226"
    normalized = (numeric_value - minimum) / span
    quality = 1.0 - normalized if lower_is_better else normalized
    quality = max(0.0, min(1.0, quality))
    hue = quality / 3.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    red_6 = int(round(red * 5))
    green_6 = int(round(green * 5))
    blue_6 = int(round(blue * 5))
    ansi_index = 16 + 36 * red_6 + 6 * green_6 + blue_6
    return f"ansi256:{ansi_index}"


def _ss_color(ss_char: str) -> str:
    if not ss_char or ss_char == "N/A":
        return "yellow"
    letter = str(ss_char).upper()
    if letter in {"H", "G", "I"}:
        return "magenta"
    if letter in {"E", "B"}:
        return "cyan"
    return "yellow"


def _build_telemetry_metrics(telemetry: dict | None) -> dict[str, object]:
    metrics = {field: "N/A" for field in TELEMETRY_FIELDS}
    if not telemetry:
        return metrics
    telemetry_metrics = telemetry.get("metrics", {})
    if not isinstance(telemetry_metrics, dict):
        return metrics
    for field in TELEMETRY_FIELDS:
        value = telemetry_metrics.get(field)
        if value is not None:
            metrics[field] = value
    return metrics


def _optional_float(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric_value):
        return numeric_value
    return None


def _find_segment_site_scores(proto_name: str, global_idx: int, to_base: str) -> tuple[float | None, float | None]:
    try:
        protomers = global_config.protomers
        segments = global_config.segments
    except Exception:
        return None, None

    protomer = protomers.get(proto_name)
    if protomer is None:
        return None, None

    for segment_name, seg_slice in zip(protomer.components, protomer.components_slices):
        if not (seg_slice.start <= global_idx < seg_slice.stop):
            continue
        local_idx = int(global_idx - seg_slice.start)
        segment = segments.get(segment_name)
        if segment is None:
            return None, None
        mutations = getattr(segment, "mutations", None) or []
        for site, _from_base, mutation_to_base, entropy, score in mutations:
            if int(site) == local_idx and str(mutation_to_base) == str(to_base):
                entropy_value = _optional_float(entropy)
                score_value = _optional_float(score)
                return entropy_value, score_value
        return None, None
    return None, None


def snapshot_segment_mutation_site_metrics() -> dict[str, dict[int, dict[str, object]]]:
    try:
        protomers = global_config.protomers
        segments = global_config.segments
    except Exception:
        return {}

    snapshot: dict[str, dict[int, dict[str, object]]] = {}
    for proto_name, protomer in protomers.items():
        per_site: dict[int, dict[str, object]] = {}
        for segment_name, seg_slice in zip(protomer.components, protomer.components_slices):
            segment = segments.get(segment_name)
            if segment is None:
                continue
            mutations = getattr(segment, "mutations", None) or []
            for mutation in mutations:
                if not isinstance(mutation, (tuple, list)) or len(mutation) < 5:
                    continue
                local_site, _from_base, to_base, entropy, score = mutation[:5]
                global_site = int(seg_slice.start) + int(local_site)
                per_site[global_site] = {
                    "to": str(to_base),
                    "entropy": _optional_float(entropy),
                    "score": _optional_float(score),
                }
        if per_site:
            snapshot[str(proto_name)] = per_site
    return snapshot


def _find_site_scores_for_log(
    proto_name: str,
    global_idx: int,
    to_base: str,
    site_metrics_snapshot: dict[str, dict[int, dict[str, object]]] | None = None,
) -> tuple[float | None, float | None]:
    if isinstance(site_metrics_snapshot, dict):
        proto_metrics = site_metrics_snapshot.get(str(proto_name), {})
        if isinstance(proto_metrics, dict):
            entry = proto_metrics.get(global_idx)
            if entry is None:
                entry = proto_metrics.get(str(global_idx))
            if isinstance(entry, dict):
                entry_to = entry.get("to")
                if entry_to is None or str(entry_to) == str(to_base):
                    return _optional_float(entry.get("entropy")), _optional_float(entry.get("score"))
    return _find_segment_site_scores(proto_name, global_idx, to_base)


def extract_secondary_structure_by_proto(evaluation_result) -> dict[str, str]:
    complex_poses = getattr(evaluation_result, "complex_poses", None)
    poses = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    if not isinstance(poses, dict) or not poses:
        return {}

    try:
        import pyrosetta  # type: ignore
        from pyrosetta.rosetta.core.pose import get_chain_from_chain_id  # type: ignore
    except Exception:
        return {}

    proto2ss = defaultdict(list)
    for oligo_name, pose in poses.items():
        if pose is None:
            continue
        chain2proto = {
            int_id_to_str_id(i + 1): proto for i, proto in enumerate(str(oligo_name))
        }
        try:
            dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
            dssp.insert_ss_into_pose(pose)
        except Exception:
            continue

        for chain in pose.split_by_chain():
            try:
                chain_name = get_chain_from_chain_id(chain.chain(1), chain)
            except Exception:
                continue
            proto_name = chain2proto.get(chain_name)
            if not proto_name:
                continue
            proto2ss[proto_name].append(str(chain.secstruct()))

    merged: dict[str, str] = {}
    for proto_name, seqs in proto2ss.items():
        if not seqs:
            continue
        min_len = min(len(seq) for seq in seqs)
        if min_len <= 0:
            continue
        consensus_chars = []
        for idx in range(min_len):
            votes: dict[str, int] = {}
            for seq in seqs:
                ss = seq[idx]
                votes[ss] = votes.get(ss, 0) + 1
            consensus = max(votes.items(), key=lambda kv: kv[1])[0]
            consensus_chars.append(consensus)
        merged[proto_name] = "".join(consensus_chars)
    return merged


def summarize_sequence_changes(before_sequences, after_sequences) -> dict[str, int]:
    changed_positions = 0
    changed_protomers = 0
    for proto_name, before in before_sequences.items():
        after = after_sequences[proto_name]
        if before == after:
            continue
        changed_protomers += 1
        changed_positions += sum(
            before_residue != after_residue
            for before_residue, after_residue in zip(before, after)
        )
    return {
        "changed_positions": changed_positions,
        "changed_protomers": changed_protomers,
    }


def log_mutator_sequence_changes(
    mutator_name,
    before_sequences,
    after_sequences,
    telemetry: dict | None = None,
):
    """Log mutator-level details and return the sequence-change summary."""
    summary = summarize_sequence_changes(before_sequences, after_sequences)
    changed_positions = summary["changed_positions"]
    changed_protomers = summary["changed_protomers"]
    changed_any = changed_protomers > 0
    site_details_by_proto: dict[str, list[str]] = {}

    metrics = _build_telemetry_metrics(telemetry)
    secondary_structure_by_proto = {}
    site_metrics_snapshot: dict[str, dict[int, dict[str, object]]] = {}
    if isinstance(telemetry, dict):
        ss_map = telemetry.get("secondary_structure_by_proto", {})
        if isinstance(ss_map, dict):
            secondary_structure_by_proto = {
                str(k): str(v) for k, v in ss_map.items() if isinstance(v, str)
            }
        metrics_snapshot = telemetry.get("site_metrics_by_proto", {})
        if isinstance(metrics_snapshot, dict):
            site_metrics_snapshot = metrics_snapshot

    for proto_name in before_sequences.keys():
        before = before_sequences[proto_name]
        after = after_sequences[proto_name]
        if before == after:
            continue
        colored_before = []
        colored_after = []
        mutation_markers = []
        site_details = []
        site_metrics: dict[int, dict[str, float | str | None]] = {}
        for site_index, (b_char, a_char) in enumerate(zip(before, after)):
            if b_char == a_char:
                colored_before.append(b_char)
                colored_after.append(a_char)
                mutation_markers.append(" ")
            else:
                entropy, score = _find_site_scores_for_log(
                    proto_name=proto_name,
                    global_idx=site_index,
                    to_base=a_char,
                    site_metrics_snapshot=site_metrics_snapshot,
                )
                secondary_structure = "N/A"
                ss_seq = secondary_structure_by_proto.get(proto_name, "")
                if site_index < len(ss_seq):
                    secondary_structure = ss_seq[site_index]
                site_metrics[site_index] = {
                    "score": score,
                    "entropy": entropy,
                    "ss": secondary_structure,
                }
                colored_before.append(b_char)
                colored_after.append(a_char)
                mutation_markers.append("|")
                site_details.append(
                    (
                        f"{proto_name}{site_index + 1}:{b_char}->{a_char}"
                        f"(score={format_metric_value(score)}"
                        f" entropy={format_metric_value(entropy)}"
                        f" ss={secondary_structure})"
                    )
                )

        if site_metrics:
            score_values = [entry.get("score") for entry in site_metrics.values()]
            entropy_values = [entry.get("entropy") for entry in site_metrics.values()]
            for idx, entry in site_metrics.items():
                score_color = _metric_color(
                    entry.get("score"),
                    all_values=score_values,
                    lower_is_better=False,
                    fallback="yellow",
                )
                colored_before[idx] = colorize(before[idx], score_color)
                ss_color = _ss_color(str(entry.get("ss", "N/A")))
                mutation_markers[idx] = colorize("|", ss_color)
                entropy_color = _metric_color(
                    entry.get("entropy"),
                    all_values=entropy_values,
                    lower_is_better=True,
                    fallback="yellow",
                )
                colored_after[idx] = colorize(after[idx], entropy_color)

        logging.info(
            "[%s] %-4s %-18s %s",
            mutator_name,
            proto_name,
            "Before(score↑):",
            "".join(colored_before),
        )
        logging.info(
            "[%s] %-4s %-18s %s",
            mutator_name,
            proto_name,
            "Mutate(ss):",
            "".join(mutation_markers),
        )
        logging.info(
            "[%s] %-4s %-18s %s",
            mutator_name,
            proto_name,
            "After(entropy↓):",
            "".join(colored_after),
        )
        if site_details:
            site_details_by_proto[proto_name] = site_details

    if changed_any:
        logging.info(
            build_mutation_summary_message(
                mutator_name=mutator_name,
                changed_positions=changed_positions,
                changed_protomers=changed_protomers,
                metrics=metrics,
            )
        )
    else:
        logging.info(build_no_change_message(mutator_name, metrics))

    for proto_name, details in site_details_by_proto.items():
        visible = details[:MAX_SITE_TELEMETRY_PER_PROTO]
        remainder = len(details) - len(visible)
        suffix = f" ... (+{remainder} more)" if remainder > 0 else ""
        logging.info("[%s] sites %s: %s%s", mutator_name, proto_name, ", ".join(visible), suffix)

    if telemetry and isinstance(telemetry, dict):
        per_oligo = telemetry.get("oligos", {})
        if isinstance(per_oligo, dict) and per_oligo:
            entropy_values = [
                float(stats.get("entropy_norm"))
                for stats in per_oligo.values()
                if isinstance(stats, dict) and isinstance(stats.get("entropy_norm"), (int, float))
            ]
            mean_entropy = mean_or_none(entropy_values)
            logging.debug(
                "[%s] telemetry_oligos=%s entropy_norm_mean=%s",
                mutator_name,
                list(per_oligo.keys()),
                format_metric_value(mean_entropy),
            )
    return summary


def calculate_temperature(step):
    """Calculate decision-layer Metropolis temperature for a step.

    Args:
        step: Current step.

    Returns:
        A decision acceptance temperature value.
    """
    initial_temperature = float(global_config.initial_temperature)
    if initial_temperature <= 0:
        return 0.0
    return initial_temperature * (
        np.exp(np.log(initial_temperature) / 100) ** step
    )


def mutation_number_scheduler(n_mutation, step): ...


def construct_standard_input(oligo_names, seed):
    inputs = {}
    protomers: Dict[Protomer] = global_config.protomers
    try:
        user_ccd = getattr(global_config, "user_ccd")
    except Exception:
        user_ccd = None
    for name, protomer in protomers.items():
        before_sequence = protomer.sequence
        after_sequence = protomer.mutated_sequence
        mutation_markers = "".join(
            [
                " " if before_sequence[i] == after_sequence[i] else "|"
                for i in range(len(before_sequence))
            ]
        )

        logging.debug("Before: %s", before_sequence)
        logging.debug("Mutate: %s", mutation_markers)
        logging.debug("After:  %s", after_sequence)

    for oligo_name in oligo_names:
        chains = []
        for chain_idx, chain in enumerate(oligo_name):
            if chain not in protomers:
                raise ValueError(f"{chain} not found in protomer sequences")
            protomer = protomers[chain]
            mods = [(v, k + 1) for k, v in protomer.ptm_dict.items()]
            chains.append(
                construct_chain_input(
                    int_id_to_str_id(chain_idx + 1),
                    protomer.mutated_sequence,
                    mods,
                    unpaired_msa=protomer.homology.get("unpairedMsa", None),
                    paired_msa=protomer.homology.get("pairedMsa", None),
                    templates=protomer.homology.get("templates", None),
                    type=protomer.type,
                )
            )

        bonded_atom_pairs = []
        bonded_atom_pair_meta = None
        cyclic_chain_ids = None
        cyclic_relative_patches = None
        try:
            cyc_cfg = getattr(global_config, "cyclization")
        except Exception:
            cyc_cfg = None
        cyc_spec = parse_cyclization(cyc_cfg)
        if cyc_spec is not None:
            protomer_lengths = {name: len(p.sequence) for name, p in protomers.items()}
            resolved = resolve_cyclization_bonds(
                cyc_spec, protomer_lengths=protomer_lengths
            )
            if resolved:
                derived_cyclic_chain_ids = derive_cyclic_chain_ids(
                    resolved,
                    oligo_name=str(oligo_name),
                    protomer_lengths=protomer_lengths,
                )
                if derived_cyclic_chain_ids:
                    cyclic_chain_ids = derived_cyclic_chain_ids
                if cyc_spec.encoding.enabled and cyc_spec.encoding.mode == "highfold_cycpoem":
                    derived_cyclic_relative_patches = derive_highfold_cycpoem_relative_patches(
                        resolved,
                        oligo_name=str(oligo_name),
                        protomer_lengths=protomer_lengths,
                    )
                elif cyc_spec.encoding.enabled and cyc_spec.encoding.mode == "global_equivalent":
                    derived_cyclic_relative_patches = derive_global_cyclic_relative_patches(
                        resolved,
                        oligo_name=str(oligo_name),
                        protomer_lengths=protomer_lengths,
                    )
                else:
                    derived_cyclic_relative_patches = derive_cyclic_relative_patches(
                        resolved,
                        oligo_name=str(oligo_name),
                        protomer_lengths=protomer_lengths,
                        default_offset_window=int(cyc_spec.offset_window or 3),
                    )
            if derived_cyclic_relative_patches:
                cyclic_relative_patches = derived_cyclic_relative_patches
            if any(
                str(getattr(bond, "signal_mode", "both")).strip().lower()
                in {"bond_only", "both"}
                for bond in resolved
            ):
                resolved_specs = resolve_cyclization_bond_specs(
                    cyc_spec,
                    protomer_lengths=protomer_lengths,
                    require_order_for_generic=True,
                )
                expanded = expand_cyclization_bonds_for_oligo(
                    resolved_specs,
                    oligo_name=str(oligo_name),
                )
                selected_expanded = filter_expanded_bonds_for_signal_modes(expanded)
                if selected_expanded:
                    bonded_atom_pairs.extend((b.atom1, b.atom2) for b in selected_expanded)
                    bonded_atom_pair_meta = [
                        {
                            "order": b.order,
                            "target_distance": float(b.target_distance),
                            "is_head_to_tail": bool(b.is_head_to_tail),
                        }
                        for b in selected_expanded
                    ]

        try:
            raw_covalent_bonds = getattr(global_config, "covalent_bonds")
        except Exception:
            raw_covalent_bonds = None
        if raw_covalent_bonds:
            for bond in list(raw_covalent_bonds):
                atom1 = dict(bond.get("atom1") or {})
                atom2 = dict(bond.get("atom2") or {})
                chain_id1 = str(atom1.get("chain_id") or "").strip()
                chain_id2 = str(atom2.get("chain_id") or "").strip()
                if not chain_id1 or not chain_id2:
                    raise ValueError(
                        "Each covalent_bonds entry must define atom1.chain_id and atom2.chain_id."
                    )
                if chain_id1 not in oligo_name or chain_id2 not in oligo_name:
                    continue
                try:
                    bonded_atom_pairs.append(
                        (
                            (
                                chain_id1,
                                int(atom1["resseq"]),
                                str(atom1["atom_name"]),
                            ),
                            (
                                chain_id2,
                                int(atom2["resseq"]),
                                str(atom2["atom_name"]),
                            ),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(
                        "Each covalent_bonds entry must define atom1/atom2 resseq and atom_name."
                    ) from exc

        inputs[oligo_name] = Input(
            chains=chains,
            name=oligo_name,
            rng_seeds=[seed],
            bonded_atom_pairs=bonded_atom_pairs or None,
            cyclic_chain_ids=cyclic_chain_ids,
            cyclic_relative_patches=cyclic_relative_patches,
            bonded_atom_pair_meta=bonded_atom_pair_meta,
            user_ccd=user_ccd,
        )

    return inputs


def construct_chain_input(
    chain_id: str,
    sequence,
    modifications,
    unpaired_msa=None,
    paired_msa=None,
    templates=None,
    type: ProtomerTypes = ProtomerTypes.PROTEIN,
):
    match type:
        case ProtomerTypes.PROTEIN:
            return ProteinChain(
                id=chain_id,
                sequence=sequence,
                ptms=modifications,
                unpaired_msa=unpaired_msa or "",
                paired_msa=paired_msa or "",
                templates=templates or [],
            )
        case ProtomerTypes.SMILES:
            return Ligand(
                id=chain_id,
                smiles=sequence,
            )
        case ProtomerTypes.CCD:
            return Ligand(
                id=chain_id, ccd_ids=[i.strip() for i in sequence.split(",")]
            )
        case ProtomerTypes.RNA:
            return RnaChain(
                id=chain_id,
                sequence=sequence,
                unpaired_msa=unpaired_msa or "",
                modifications=[],
            )
        case ProtomerTypes.DNA:
            return DnaChain(
                id=chain_id, sequence=sequence, modifications=[]
            )

def construct_mutated_protomer(protomers, mutated_sites):
    for name, protomer in protomers.items():
        protomer.update_sequence(
            updated_sites=mutated_sites.get(name, {}),
        )


def make_evaluation_runtime_config(
    step,
    inputs,
    former_config=None,
    chain2proto=None,
    oligo2complex=None,
):
    former_config = former_config or {}

    former_config.update(
        {
            "step": step,
            "chain2proto": chain2proto or former_config.get("chain2proto", None),
            "oligo2complex": oligo2complex or former_config.get("oligo2complex", None),
        }
    )
    return former_config
