"""
Astrolabe closed-loop sequence–structure design for molecular systems.

This module orchestrates the mutation and evaluation process for protein design,
using configured structure predictors, sequence mutators and stage objectives.
"""

import functools
import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

# Third-party imports
import hydra
import numpy as np
try:  # pragma: no cover - optional (should be installed via deps)
    from absl import logging as absl_logging
except Exception:  # pragma: no cover
    absl_logging = None
from omegaconf import OmegaConf

if TYPE_CHECKING:  # pragma: no cover
    from astrolabe.core.folding_input import Input

# Local imports
from astrolabe import run_config_lifecycle
from astrolabe.argparser import parse_arguments
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import (
    MutableTypes,
    Protomer,
    ProtomerTypes,
    SegmentManager,
)
from astrolabe.decision_context import passes_stage_entry_acceptance_gate
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.utils import ComplexPoses, StandardScores, construct_evaluators
from astrolabe.graceful_shutdown import install_handlers, stop_requested
from astrolabe.initializers import (
    AminoAcidSetInitializer,
    AllAInitializer,
    DPLMInitializer,
    ExistingStructureInitializer,
    MaskMixInitializer,
    PSSMInitializer,
    RFD3Initializer,
    RandomInitializer,
    XMixInitializer,
)
from astrolabe.losses.base import LossEngine
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.utils import build_mutator_runtime_config
from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner
from astrolabe.mutators.wrappers.dplm_quality_filter import (
    build_quality_filter_config,
    filter_dplm_sequences,
)
from astrolabe.structure_processors.base import (
    initialize_predockers,
    assign_predocker_to_mutators,
)
from astrolabe.per_residue_scores import (
    attach_per_residue_scores,
    per_residue_scores_enabled,
    per_residue_scores_requires_complex_poses,
)
from astrolabe.score_contributions import normalize_score_contributions
from astrolabe.stage_manager import StageManager
from astrolabe.stage_transaction import (
    begin_active_transition_gate_attempt,
    begin_direct_stage_entry,
    begin_seed_transition_gate_attempt,
    begin_stage_entry_transaction,
    clear_transition_gate_runtime,
    finish_active_transition_gate_decision,
    finish_direct_stage_entry_decision,
    finish_seed_transition_gate_decision,
    preserve_seed_transition_source,
    resolve_active_transition_gate_decision,
    resolve_already_accepted_direct_stage_entry_decision,
    resolve_seed_transition_gate_decision,
)
from astrolabe.structure_start import (
    build_structure_start_state,
    select_structure_start_candidate,
)
from astrolabe.utils.base import (
    construct_standard_input,
    extract_secondary_structure_by_proto,
    log_mutator_sequence_changes,
    make_evaluation_runtime_config,
    snapshot_segment_mutation_site_metrics,
    summarize_sequence_changes,
)
from astrolabe.utils.log_events import (
    build_step_outcome_message,
    build_step_start_message,
    event_span,
    log_event,
    summarize_event_items,
)
from astrolabe.utils.logging_utils import (
    AnsiPolicyFormatter,
    normalize_logging_config,
    parse_log_level,
    should_strip_console_ansi,
    should_strip_file_ansi,
)
from astrolabe.utils.extra_residues import setup_extra_residues
from astrolabe.core.cyclization import (
    derive_aa_constraints,
    derive_locked_positions,
    parse_cyclization,
    resolve_cyclization_bonds,
 )
from astrolabe.core.nonstandard_residues import (
    build_nonstandard_aliases_and_user_ccd,
)


PYROSETTA_LOSS_TYPES = {
    "autodock",
    "binding_site_buried_ratio",
    "bindcraft_binder_score",
    "bindcraft_hotspot_rmsd",
    "bindcraft_binder_rmsd",
    "bindcraft_binder_loop_pct",
    "bindcraft_interface_dG",
    "bindcraft_interface_dSASA",
    "bindcraft_interface_delta_unsat_hbonds",
    "bindcraft_interface_interface_hbonds",
    "bindcraft_interface_nres",
    "bindcraft_interface_plddt_norm",
    "bindcraft_interface_sc",
    "bindcraft_ss_plddt_norm",
    "bindcraft_surface_hydrophobicity",
    "chain_pair_holes_diff",
    "chain_holes_diff",
    "contact_molecular_surface",
    "hotspot_buried_ratio",
    "hotspot_delta_sasa",
    "hotspot_dsasa",
    "hotspot_sasa",
    "interface_buried_unsatisfied_hbonds",
    "ligand_contact",
    "ligand_delta_sasa_fraction",
    "ligand_delta_sasa_ratio",
    "ligand_exposed_sasa",
    "ligand_pocket_occupancy",
    "ligand_reference_pose_rmsd",
    "ligand_unsat_polar",
    "ligand_unsat_polars",
    "one_sidechain_in",
    "one_sidechain_in_distance",
    "pairwise_interface_contacts",
    "physical_interface_score",
    "protomer_shape_complementarity",
    "sap_score",
    "shape_complementarity",
    "target_ligand_contact",
}


logger = logging.getLogger(__name__)

POST_EVAL_HOOK_LOSS_TYPES = {
    "bindcraft_binder_score",
    "bindcraft_interface_dG",
    "bindcraft_interface_dSASA",
    "bindcraft_interface_delta_unsat_hbonds",
    "bindcraft_interface_interface_hbonds",
    "bindcraft_interface_sc",
    "bindcraft_surface_hydrophobicity",
}


class RawModelResultsCleanupError(RuntimeError):
    pass


CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def timer(name: str = ""):
    """Decorator that reports the execution time of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"{name or func.__name__} executed in {elapsed_time:.6f} seconds"
            )
            return result

        return wrapper

    return decorator


def setup_logging(outdir: pathlib.Path, logging_config: dict | None = None) -> None:
    """Configure logging handlers for the application."""
    logging_cfg = normalize_logging_config(logging_config or {})
    color_cfg = logging_cfg["color"]

    logger = logging.getLogger()
    # Keep root at DEBUG so debug.log can capture tracebacks + debug diagnostics,
    # while console output remains INFO via its handler level.
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers (for repeated runs)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_format = "%(message)s"
    formatter_kwargs = {
        "fmt": base_format,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    stream = logging.StreamHandler().stream
    console_strip_ansi = should_strip_console_ansi(color_cfg["console"], stream)
    file_strip_ansi = should_strip_file_ansi(color_cfg["file"])

    info_formatter = AnsiPolicyFormatter(
        **formatter_kwargs,
        strip_ansi_sequences=file_strip_ansi,
    )
    debug_formatter = AnsiPolicyFormatter(
        **formatter_kwargs,
        strip_ansi_sequences=file_strip_ansi,
    )
    console_formatter = AnsiPolicyFormatter(
        fmt=console_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        strip_ansi_sequences=console_strip_ansi,
    )

    # Create log directory if it doesn't exist
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handlers
    info_handler = logging.FileHandler(log_dir / "info.log", encoding="utf-8")
    info_handler.setLevel(parse_log_level(logging_cfg.get("info_file_level"), logging.INFO))
    info_handler.setFormatter(info_formatter)
    logger.addHandler(info_handler)

    debug_handler = logging.FileHandler(log_dir / "debug.log", encoding="utf-8")
    debug_handler.setLevel(parse_log_level(logging_cfg.get("debug_file_level"), logging.DEBUG))
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(parse_log_level(logging_cfg.get("console_level"), logging.INFO))
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    # Suppress verbose external logging
    _suppress_external_logging()


def _suppress_external_logging() -> None:
    """Suppress verbose logging from external libraries."""
    if absl_logging is not None:
        absl_logging.set_verbosity(absl_logging.ERROR)

    jax_logger = logging.getLogger("jax._src.xla_bridge")
    jax_logger.setLevel(logging.WARNING)
    logging.getLogger("protenix").setLevel(logging.WARNING)
    logging.getLogger("protenix.data.constraint_featurizer").setLevel(logging.WARNING)
    logging.getLogger("pyrosetta").setLevel(logging.WARNING)
    logging.getLogger("pyrosetta.rosetta").setLevel(logging.WARNING)
    logging.getLogger("prody").setLevel(logging.ERROR)
    logging.getLogger(".prody").setLevel(logging.ERROR)

    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="hydra._internal.defaults_list"
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="huggingface_hub.file_download",
    )

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["GLOG_minloglevel"] = "3"

    # Optional: make ProDy quiet if available (used by some docking utilities).
    try:
        import prody  # type: ignore

        prody.utilities.logger.PackageLogger.verbosity = None
    except Exception:
        pass


def initialize_global_config(args) -> None:
    """Initialize global configuration constants."""
    nonstandard_aliases, user_ccd = build_nonstandard_aliases_and_user_ccd(
        getattr(args, "nonstandard_residues", None)
    )
    try:
        named_files = global_config.named_files
    except RuntimeError:
        named_files = {}
    direct_user_ccd = getattr(args, "user_ccd", None)
    if direct_user_ccd:
        direct_user_ccd = str(direct_user_ccd)
        direct_user_ccd_path = None
        if "\n" not in direct_user_ccd and not direct_user_ccd.lstrip().startswith("data_"):
            direct_user_ccd_path = pathlib.Path(direct_user_ccd).expanduser()
        if direct_user_ccd_path is not None and direct_user_ccd_path.exists():
            direct_user_ccd = direct_user_ccd_path.read_text(encoding="utf-8")
        user_ccd = (
            f"{str(user_ccd).rstrip()}\n{direct_user_ccd.strip()}\n"
            if user_ccd
            else direct_user_ccd
        )
    config_items = {
        "exclude_AA": args.exclude_AA,
        "covalent_bonds": getattr(args, "covalent_bonds", None),
        "protomers": {},
        "segments": {},
        "named_files": named_files,
        "mutable_protomers": {},
        "segment_manager": SegmentManager(),
        "initial_temperature": args.initial_temperature,
        "out_dir": args.out_dir,
        "raw_model_results_dir": args.raw_model_results_dir,
        "is_debug": getattr(args, "debug", False),
        "use_metropolis_acceptance": args.use_metropolis_acceptance,
        "enable_pyrosetta": getattr(args, "enable_pyrosetta", False),
        "skip_complex_poses": getattr(args, "skip_complex_poses", False),
        "allow_mmcif_pdb_fallback": getattr(args, "allow_mmcif_pdb_fallback", False),
        "per_residue_scores": getattr(args, "per_residue_scores", None),
        "pre_eval_hooks": getattr(args, "pre_eval_hooks", None),
        # Optional: Protenix per-run pocket/contact constraints (soft guidance).
        # Stored on global_config so evaluators can access without threading args through.
        "protenix_constraint": getattr(args, "protenix_constraint", None),
        # Optional: binder/target binding-site spec (used to generate backend constraints).
        "binding_site": getattr(args, "binding_site", None),
        # Optional: runtime-derived binder/target binding-site spec. Predockers can
        # update this after initialization, so evaluator setup uses the explicit
        # enable flag to select constraint-capable checkpoints up front.
        "dynamic_binding_site": None,
        "dynamic_binding_site_enabled": bool(
            getattr(args, "dynamic_binding_site_enabled", False)
        ),
        # Optional: explicit Chai pairwise restraint rows or per-oligomer row mapping.
        "chai_restraints": getattr(args, "chai_restraints", None),
        # Optional: channel-specific runtime metadata for RF3 / HOLE workflows.
        "channel": getattr(args, "channel", None),
        # Optional: cyclic peptide / bond constraints.
        "cyclization": getattr(args, "cyclization", None),
        # Optional: map ligand entity/protomer to Rosetta params name.
        "ligand_resnames": getattr(args, "ligand_resnames", None),
        # Optional: alias->[ccd_id,parent_aa] map for nonstandard PTM tokens.
        "nonstandard_aliases": nonstandard_aliases,
        # Optional: user-provided CCD mmCIF text injected into AF3 Ccd(user_ccd=...).
        "user_ccd": user_ccd,
    }

    for key, value in config_items.items():
        global_config.init_const(key, value)
    try:
        global_config.get_const("named_files")
    except RuntimeError:
        global_config.init_const("named_files", {})


def setup_protomers(args) -> None:
    """Initialize protomers from sequence configurations."""
    for proto, statement in args.seqs.items():
        protomer = Protomer(
            proto,
            statement,
            args.types.get(proto, "protein"),
            args.homologies.get(proto),
        )
        global_config.protomers[proto] = protomer

        if protomer.type in MutableTypes:
            global_config.mutable_protomers[proto] = protomer


def _validate_nonstandard_backend_compat(args, stage_manager) -> None:
    try:
        aliases = getattr(global_config, "nonstandard_aliases")
    except Exception:
        aliases = {}
    if not aliases:
        return

    evaluator_terms = getattr(args.evaluator, "evaluator_terms", None) or {}
    default_alias = getattr(stage_manager.default_setting, "evaluator", None)
    used_aliases: set[str] = set()
    if default_alias:
        used_aliases.add(str(default_alias))
    for stage_name, stage_unit in stage_manager.stage_args.items():
        if stage_name == "default":
            continue
        alias = getattr(stage_unit, "evaluator", None) or default_alias
        if alias:
            used_aliases.add(str(alias))

    non_af3: list[str] = []
    for alias in sorted(used_aliases):
        conf = evaluator_terms.get(alias)
        if conf is None:
            non_af3.append(f"{alias}=<undefined>")
            continue
        try:
            typ = str(conf.get("_type") or "").strip().lower()
        except Exception:
            typ = str(getattr(conf, "_type", "") or "").strip().lower()
        if typ != "alphafold3":
            non_af3.append(f"{alias}={typ or '<unknown>'}")
    if non_af3:
        raise ValueError(
            "nonstandard_residues is configured, but stage evaluators are not AF3. "
            "Current evaluators: "
            + ", ".join(non_af3)
            + ". Nonstandard residues currently require evaluator _type=Alphafold3 only."
        )


def apply_cyclization(args) -> None:
    spec = parse_cyclization(getattr(args, "cyclization", None))
    if spec is None:
        return

    protomers = global_config.protomers
    protomer_lengths = {name: len(p.sequence) for name, p in protomers.items()}
    resolved = resolve_cyclization_bonds(spec, protomer_lengths=protomer_lengths)

    # Enforce optional residue identity constraints (e.g. disulfide requires Cys).
    aa_constraints = derive_aa_constraints(resolved)
    for proto, mapping in aa_constraints.items():
        if proto not in protomers:
            raise ValueError(f"cyclization references unknown protomer {proto!r}")
        p = protomers[proto]
        for pos1, aa in mapping.items():
            cur = p.sequence[int(pos1) - 1]
            if cur != aa:
                pos0 = int(pos1) - 1
                if p.type in MutableTypes and not bool(p.mutable_mask[pos0]):
                    raise ValueError(
                        f"cyclization requires {proto}:{pos1} to be {aa}, but that position is fixed ({cur})"
                    )
                p.set_residue(int(pos1), aa)

    # Lock endpoints so mutators (and DPLM init masking) do not change them.
    locked = derive_locked_positions(resolved)
    for proto, positions in locked.items():
        if proto not in protomers:
            raise ValueError(f"cyclization references unknown protomer {proto!r}")
        protomers[proto].add_locked_positions(sorted(positions))


def _to_plain_object(value):
    try:
        value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _initializer_config(args):
    value = getattr(args, "initializer", None)
    if isinstance(value, dict) or OmegaConf.is_dict(value):
        return value
    return {}


def _collect_random_protein_segments():
    """Collect protein segments that were created from numeric random ranges."""
    plan = {}
    for proto_name, protomer in global_config.protomers.items():
        if protomer.type != ProtomerTypes.PROTEIN:
            continue

        mutable_mask = protomer.mutable_mask
        segment_plan = []
        for idx, segment_name in enumerate(protomer.components):
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            if not getattr(segment, "initialized_from_random", False):
                continue

            seg_slice = protomer.components_slices[idx]
            mutable_positions = [
                pos for pos in range(seg_slice.start, seg_slice.stop) if bool(mutable_mask[pos])
            ]
            if not mutable_positions:
                continue
            segment_plan.append((segment_name, seg_slice, mutable_positions))

        if segment_plan:
            plan[proto_name] = segment_plan
    return plan


def _collect_mutable_protein_segments():
    """Collect every mutable protein segment for structure-based initialization."""
    plan = {}
    for proto_name, protomer in global_config.protomers.items():
        if protomer.type != ProtomerTypes.PROTEIN:
            continue

        mutable_mask = protomer.mutable_mask
        segment_plan = []
        for idx, segment_name in enumerate(protomer.components):
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            segment_slice = protomer.components_slices[idx]
            mutable_positions = [
                position
                for position in range(segment_slice.start, segment_slice.stop)
                if bool(mutable_mask[position])
            ]
            if mutable_positions:
                segment_plan.append(
                    (segment_name, segment_slice, mutable_positions)
                )
        if segment_plan:
            plan[proto_name] = segment_plan
    return plan


def _resolve_initializer_mode(args) -> str:
    init_cfg = _initializer_config(args)
    mode_raw = init_cfg.get("mode")
    mode = str(mode_raw).strip().lower() if mode_raw is not None else ""

    mode_aliases = {
        "none": "random",
        "default": "random",
        "aa-set": "aa_set",
        "aaset": "aa_set",
        "mask-mix": "mask_mix",
        "maskmix": "mask_mix",
        "all-a": "all_a",
        "alla": "all_a",
        "x-mix": "x_mix",
        "xmix": "x_mix",
        "rfd-3": "rfd3",
        "start-from-structure": "structure",
        "start_from_structure": "structure",
        "existing-structure": "structure",
        "existing_structure": "structure",
    }
    mode = mode_aliases.get(mode, mode)

    legacy_dplm_flag = bool(getattr(args, "reinitialize_random_segments_with_dplm", False))
    if legacy_dplm_flag:
        if mode in {"", "random", "dplm"}:
            return "dplm"
        raise ValueError(
            "Conflicting initializer settings: legacy `reinitialize_random_segments_with_dplm=true` "
            f"cannot be combined with `initializer.mode={mode!r}`."
        )

    if not mode:
        return "random"
    return mode


def _normalize_aa_set(raw_values) -> list[str]:
    if raw_values is None:
        raise ValueError(
            "initializer.mode='aa_set' requires `initializer.aa_set` (e.g. 'AGST' or ['A','G','S','T'])."
        )

    values = _to_plain_object(raw_values)
    if isinstance(values, dict):
        if "residues" in values:
            values = values.get("residues")
        else:
            raise ValueError("initializer.aa_set mapping must contain `residues`.")

    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            raise ValueError("initializer.aa_set cannot be empty.")
        tokens = raw.replace(",", " ").split()
        residues = list(tokens[0]) if len(tokens) == 1 else tokens
    elif isinstance(values, (list, tuple, set)):
        residues = list(values)
    else:
        raise ValueError("initializer.aa_set must be a string, list, or dict with `residues`.")

    normalized = []
    seen = set()
    for residue in residues:
        aa = str(residue or "").strip().upper()
        if not aa:
            continue
        if len(aa) != 1 or aa not in CANONICAL_AMINO_ACIDS:
            raise ValueError(
                f"Invalid amino acid in initializer.aa_set: {residue!r}. "
                "Use one-letter canonical amino acids."
            )
        if aa in seen:
            continue
        seen.add(aa)
        normalized.append(aa)

    if not normalized:
        raise ValueError("initializer.aa_set has no valid amino acids after normalization.")
    return normalized


def _resolve_x_fraction(init_cfg: dict[str, Any]) -> float:
    raw_fraction = init_cfg.get("x_fraction", 0.8)
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("initializer.x_fraction must be numeric.") from exc
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(
            f"initializer.x_fraction must be within [0, 1], got {fraction!r}."
        )
    return fraction


def _normalize_mask_fraction(raw_value) -> float:
    if raw_value is None:
        raise ValueError("initializer.mode='mask_mix' requires `initializer.mask_fraction`.")
    try:
        mask_fraction = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"initializer.mask_fraction must be a number between 0 and 1, got {raw_value!r}."
        ) from exc
    if not math.isfinite(mask_fraction) or not 0.0 <= mask_fraction <= 1.0:
        raise ValueError(
            f"initializer.mask_fraction must be between 0 and 1 inclusive, got {raw_value!r}."
        )
    return mask_fraction


def _resolve_mask_mix_residue_set(args, init_cfg: dict[str, Any]) -> list[str]:
    raw_aa_set = init_cfg.get("aa_set")
    if raw_aa_set is not None:
        return _normalize_aa_set(raw_aa_set)

    excluded = {
        str(residue or "").strip().upper()
        for residue in (_to_plain_object(getattr(args, "exclude_AA", [])) or [])
        if str(residue or "").strip()
    }
    residues = [aa for aa in CANONICAL_AMINO_ACIDS if aa not in excluded]
    if not residues:
        raise ValueError(
            "initializer.mode='mask_mix' has no available amino acids after applying exclude_AA."
        )
    return residues


def _resolve_aa_set_initializer_config(
    raw_values,
) -> tuple[list[str] | None, dict[int, list[str]]]:
    values = _to_plain_object(raw_values)
    if values is None:
        raise ValueError(
            "initializer.mode='aa_set' requires `initializer.aa_set` (e.g. 'AGST' or {'segments': {'loop_1': 'GSTNQPD'}})."
        )

    if not isinstance(values, dict):
        return _normalize_aa_set(values), {}

    has_residues = "residues" in values and values.get("residues") is not None
    has_segments = "segments" in values and values.get("segments") is not None
    if not has_residues and not has_segments:
        raise ValueError(
            "initializer.aa_set mapping must contain `residues` and/or `segments`."
        )

    default_residues = (
        _normalize_aa_set(values.get("residues")) if has_residues else None
    )
    segment_overrides_raw = _to_plain_object(values.get("segments")) if has_segments else {}
    if segment_overrides_raw is None:
        segment_overrides_raw = {}
    if not isinstance(segment_overrides_raw, dict):
        raise ValueError("initializer.aa_set.segments must be a mapping of segment names to residue sets.")

    segment_residues_by_id: dict[int, list[str]] = {}
    for segment_ref, residues_raw in segment_overrides_raw.items():
        segment_name = str(segment_ref or "").strip()
        if not segment_name:
            raise ValueError("initializer.aa_set.segments contains an empty segment name.")
        segment = global_config.segments.get(segment_name)
        if segment is None:
            raise ValueError(
                f"initializer.aa_set.segments references unknown segment {segment_name!r}."
            )
        segment_residues_by_id[id(segment)] = _normalize_aa_set(residues_raw)

    return default_residues, segment_residues_by_id


def _collect_unique_random_protein_segments():
    plan = {}
    for proto_name, protomer in global_config.protomers.items():
        if protomer.type != ProtomerTypes.PROTEIN:
            continue

        locked = set(getattr(protomer, "locked_positions", frozenset()))
        segment_plan: dict[int, tuple[str, object, set[int]]] = {}
        for idx, segment_name in enumerate(protomer.components):
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            if not getattr(segment, "initialized_from_random", False):
                continue

            seg_slice = protomer.components_slices[idx]
            rel_mutable_positions = {
                int(pos - seg_slice.start)
                for pos in range(seg_slice.start, seg_slice.stop)
                if pos not in locked
            }
            if not rel_mutable_positions:
                continue

            segment_id = id(segment)
            if segment_id in segment_plan:
                prev_name, prev_segment, prev_positions = segment_plan[segment_id]
                segment_plan[segment_id] = (
                    prev_name,
                    prev_segment,
                    prev_positions & rel_mutable_positions,
                )
            else:
                segment_plan[segment_id] = (segment_name, segment, rel_mutable_positions)

        filtered = [
            (name, segment, sorted(rel_positions))
            for name, segment, rel_positions in segment_plan.values()
            if rel_positions
        ]
        if filtered:
            plan[proto_name] = filtered
    return plan


def _choose_weighted_amino_acid(weights_by_aa: dict[str, float]) -> str:
    residues = []
    weights = []
    for aa, weight in weights_by_aa.items():
        w = float(weight)
        if w <= 0:
            continue
        residues.append(aa)
        weights.append(w)
    if not residues:
        raise ValueError("PSSM row has no positive amino-acid weights.")
    return random.choices(residues, weights=weights, k=1)[0]


def _load_pssm_mapping(raw_pssm) -> dict[str, Any]:
    pssm_cfg = _to_plain_object(raw_pssm)
    if pssm_cfg is None:
        raise ValueError("initializer.mode='pssm' requires `initializer.pssm`.")

    def _load_pssm_file(path_value) -> dict[str, Any]:
        p = pathlib.Path(str(path_value)).expanduser()
        if not p.exists():
            raise ValueError(f"initializer.pssm path does not exist: {p}")
        if p.suffix.lower() == ".json":
            payload = json.loads(p.read_text(encoding="utf-8"))
        else:
            payload = _to_plain_object(OmegaConf.load(str(p)))
        if not isinstance(payload, dict):
            raise ValueError(f"initializer.pssm file must contain a mapping: {p}")
        if isinstance(payload.get("protomers"), dict):
            return payload["protomers"]
        return payload

    if isinstance(pssm_cfg, str):
        return _load_pssm_file(pssm_cfg)

    if not isinstance(pssm_cfg, dict):
        raise ValueError("initializer.pssm must be a mapping, file path, or {'path': ...}.")

    loaded = {}
    if pssm_cfg.get("path"):
        loaded.update(_load_pssm_file(pssm_cfg["path"]))

    inline = pssm_cfg.get("protomers")
    if inline is None:
        inline = {
            k: v
            for k, v in pssm_cfg.items()
            if k not in {"path", "protomers", "format", "temperature"}
        }
    inline = _to_plain_object(inline)
    if isinstance(inline, dict):
        loaded.update(inline)

    if not loaded:
        raise ValueError("No protomer PSSM entries found in initializer.pssm.")
    return loaded


def _normalize_protomer_pssm_rows(rows_obj, *, protomer_name: str, expected_length: int):
    rows = _to_plain_object(rows_obj)
    if not isinstance(rows, list):
        raise ValueError(
            f"initializer.pssm[{protomer_name!r}] must be a list with one row per residue."
        )
    if len(rows) != expected_length:
        raise ValueError(
            f"initializer.pssm[{protomer_name!r}] length mismatch: expected {expected_length}, got {len(rows)}."
        )

    amino_acids = list(CANONICAL_AMINO_ACIDS)
    normalized_rows = []
    for idx, row in enumerate(rows):
        row = _to_plain_object(row)
        weights = {}
        if isinstance(row, dict):
            for aa, value in row.items():
                aa_norm = str(aa or "").strip().upper()
                if aa_norm not in amino_acids:
                    continue
                try:
                    weight = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid PSSM value at {protomer_name}[{idx}] for amino acid {aa_norm!r}: {value!r}"
                    ) from exc
                if weight > 0:
                    weights[aa_norm] = weight
        elif isinstance(row, (list, tuple)):
            if len(row) != len(amino_acids):
                raise ValueError(
                    f"initializer.pssm[{protomer_name!r}][{idx}] must have {len(amino_acids)} values "
                    f"(order: {''.join(amino_acids)})."
                )
            for aa, value in zip(amino_acids, row):
                try:
                    weight = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid PSSM value at {protomer_name}[{idx}] for amino acid {aa!r}: {value!r}"
                    ) from exc
                if weight > 0:
                    weights[aa] = weight
        else:
            raise ValueError(
                f"initializer.pssm[{protomer_name!r}][{idx}] must be dict or list, got {type(row).__name__}."
            )

        if not weights:
            raise ValueError(
                f"initializer.pssm[{protomer_name!r}][{idx}] has no positive amino-acid weights."
            )
        normalized_rows.append(weights)
    return normalized_rows


def _apply_aa_set_initializer(args, *, fixed_residue_set: list[str] | None = None) -> None:
    random_plan = _collect_unique_random_protein_segments()
    if not random_plan:
        logging.info("No randomly initialized protein segments detected for aa_set initialization.")
        return

    init_cfg = _initializer_config(args)
    default_residues, segment_residues_by_id = (
        (fixed_residue_set, {})
        if fixed_residue_set is not None
        else _resolve_aa_set_initializer_config(init_cfg.get("aa_set"))
    )
    if segment_residues_by_id:
        residues_label = "segment overrides"
        if default_residues is not None:
            residues_label = f"{''.join(default_residues)} + segment overrides"
    else:
        residues_label = "".join(default_residues or [])

    updated_segments = 0
    for _proto_name, segments in random_plan.items():
        for segment_name, segment, mutable_positions in segments:
            residues = segment_residues_by_id.get(id(segment), default_residues)
            if residues is None:
                continue
            tokens = list(segment.sequence)
            for rel in mutable_positions:
                tokens[int(rel)] = random.choice(residues)
            segment.sequence = "".join(tokens)
            segment.initialized_from_random = False
            updated_segments += 1

    logging.info(
        "Reinitialized %d protein segment(s) with aa_set initializer (%s).",
        updated_segments,
        residues_label,
    )


def _apply_mask_mix_initializer(args) -> None:
    random_plan = _collect_random_protein_segments()
    if not random_plan:
        logging.info("No randomly initialized protein segments detected for mask_mix initialization.")
        return

    init_cfg = _initializer_config(args)
    mask_fraction = _normalize_mask_fraction(init_cfg.get("mask_fraction"))
    residues = _resolve_mask_mix_residue_set(args, init_cfg)

    updated_segments = 0
    total_masked = 0
    for _proto_name, segments in random_plan.items():
        protomer_positions = [pos for _name, _slice, mutable_positions in segments for pos in mutable_positions]
        mask_count = round(len(protomer_positions) * mask_fraction)
        masked_positions = (
            set(random.sample(protomer_positions, k=mask_count)) if mask_count else set()
        )
        for segment_name, seg_slice, mutable_positions in segments:
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            tokens = list(segment.sequence)
            for pos in mutable_positions:
                rel = int(pos - seg_slice.start)
                tokens[rel] = "X" if pos in masked_positions else random.choice(residues)
            segment.sequence = "".join(tokens)
            segment.initialized_from_random = False
            updated_segments += 1
        total_masked += mask_count

    logging.info(
        "Reinitialized %d protein segment(s) with mask_mix initializer (%d masked positions, aa_set=%s).",
        updated_segments,
        total_masked,
        "".join(residues),
    )


def _apply_pssm_initializer(args) -> None:
    random_plan = _collect_random_protein_segments()
    if not random_plan:
        logging.info("No randomly initialized protein segments detected for PSSM initialization.")
        return

    init_cfg = _initializer_config(args)
    pssm_mapping = _load_pssm_mapping(init_cfg.get("pssm"))

    missing = [name for name in random_plan if name not in pssm_mapping]
    if missing:
        raise ValueError(
            "initializer.pssm is missing protomer entries for: " + ", ".join(sorted(missing))
        )

    updated_segments = 0
    for proto_name, segments in random_plan.items():
        protomer = global_config.protomers.get(proto_name)
        if protomer is None:
            continue
        rows = _normalize_protomer_pssm_rows(
            pssm_mapping[proto_name],
            protomer_name=proto_name,
            expected_length=protomer.length,
        )
        for segment_name, seg_slice, mutable_positions in segments:
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            tokens = list(segment.sequence)
            for pos in mutable_positions:
                rel = int(pos - seg_slice.start)
                tokens[rel] = _choose_weighted_amino_acid(rows[int(pos)])
            segment.sequence = "".join(tokens)
            segment.initialized_from_random = False
            updated_segments += 1

    logging.info("Reinitialized %d protein segment(s) with PSSM initializer.", updated_segments)


def _apply_x_mix_initializer(args) -> None:
    random_plan = _collect_random_protein_segments()
    if not random_plan:
        logging.info("No randomly initialized protein segments detected for x_mix initialization.")
        return

    init_cfg = _initializer_config(args)
    x_fraction = _resolve_x_fraction(init_cfg)

    if init_cfg.get("aa_set") is not None:
        residues = _normalize_aa_set(init_cfg.get("aa_set"))
    else:
        excluded = {str(res).strip().upper() for res in getattr(global_config, "exclude_AA", [])}
        residues = [aa for aa in "ACDEFGHIKLMNPQRSTVWY" if aa not in excluded]
        if not residues:
            raise ValueError("initializer.mode='x_mix' has no available amino acids after exclude_AA filtering.")

    updated_segments = 0
    for _proto_name, segments in random_plan.items():
        for segment_name, seg_slice, mutable_positions in segments:
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            tokens = list(segment.sequence)
            for pos in mutable_positions:
                rel = int(pos - seg_slice.start)
                tokens[rel] = "X" if random.random() < x_fraction else random.choice(residues)
            segment.sequence = "".join(tokens)
            segment.initialized_from_random = False
            updated_segments += 1

    logging.info(
        "Reinitialized %d protein segment(s) with x_mix initializer (x_fraction=%.2f).",
        updated_segments,
        x_fraction,
    )


def _resolve_dplm_runner_config(args) -> dict[str, Any]:
    mutator_cfg = _to_plain_object(getattr(args, "mutator", None))
    mutator_cfg = mutator_cfg if isinstance(mutator_cfg, dict) else {}

    runner_conf = _to_plain_object(mutator_cfg.get("DPLM"))
    runner_conf = dict(runner_conf) if isinstance(runner_conf, dict) else {}

    init_cfg = _initializer_config(args)
    alias = init_cfg.get("dplm_alias")

    mutator_terms = _to_plain_object(mutator_cfg.get("mutator_terms"))
    mutator_terms = mutator_terms if isinstance(mutator_terms, dict) else {}
    enabled_aliases = _to_plain_object(mutator_cfg.get("enabled_aliases"))
    enabled_aliases = enabled_aliases if isinstance(enabled_aliases, list) else []

    if alias:
        term = _to_plain_object(mutator_terms.get(alias))
        if not isinstance(term, dict):
            raise ValueError(f"initializer.dplm_alias references unknown mutator alias: {alias!r}")
        if str(term.get("_type") or "").lower() != "dplm":
            raise ValueError(
                f"initializer.dplm_alias={alias!r} does not point to a DPLM mutator."
            )
        term_override = {k: v for k, v in term.items() if k != "_type"}
        runner_conf.update(term_override)
        return runner_conf

    alias_order = [str(a) for a in enabled_aliases if str(a)]
    for candidate in mutator_terms.keys():
        candidate_str = str(candidate)
        if candidate_str not in alias_order:
            alias_order.append(candidate_str)

    for candidate in alias_order:
        term = _to_plain_object(mutator_terms.get(candidate))
        if not isinstance(term, dict):
            continue
        if str(term.get("_type") or "").lower() != "dplm":
            continue
        term_override = {k: v for k, v in term.items() if k != "_type"}
        runner_conf.update(term_override)
        break

    init_dplm = _to_plain_object(init_cfg.get("dplm"))
    if isinstance(init_dplm, dict):
        runner_conf.update(init_dplm)

    return runner_conf


def _apply_dplm_initializer(args) -> None:
    """Use DPLM to fill protein segments that originated from numeric ranges."""
    runner_conf = _resolve_dplm_runner_config(args)
    if not runner_conf:
        logging.warning(
            "Requested DPLM-based initialization but no DPLM configuration was provided."
        )
        return

    random_plan = _collect_random_protein_segments()
    tasks = []
    reinit_plan = {}
    for proto_name, segments in random_plan.items():
        protomer = global_config.protomers.get(proto_name)
        if protomer is None:
            continue
        masked_sequence = list(protomer.sequence)
        for _segment_name, _seg_slice, mutable_positions in segments:
            for pos in mutable_positions:
                masked_sequence[int(pos)] = "<mask>"
        tasks.append({"name": proto_name, "masked_sequence": "".join(masked_sequence)})
        reinit_plan[proto_name] = [(name, seg_slice) for name, seg_slice, _ in segments]

    if not tasks:
        logging.info(
            "No randomly initialized protein segments detected for DPLM initialization."
        )
        return

    if not runner_conf.get("work_dir"):
        runner_conf["work_dir"] = str(
            pathlib.Path(args.raw_model_results_dir) / "DPLM_init"
        )
    runner = DPLMRunner(runner_conf)

    q_enable = bool(runner_conf.get("quality_filter_enable", True))
    q_fallback = bool(runner_conf.get("quality_filter_fallback_to_raw", False))
    # Keep defaults conservative; do not silently increase sampling beyond num_samples.
    q_max_rounds = int(runner_conf.get("quality_filter_max_rounds", 1) or 0)
    q_attempt_num_samples = int(runner_conf.get("quality_filter_attempt_num_samples", 0) or 0)
    q_cfg = build_quality_filter_config(
        runner_conf.get("quality_filter", {}) if isinstance(runner_conf.get("quality_filter", {}), dict) else {}
    )
    task_by_name = {t["name"]: t for t in tasks}
    base_num_samples = int(runner_conf.get("num_samples", 1) or 1)
    request_num_samples = max(1, base_num_samples, q_attempt_num_samples) if q_enable else max(1, base_num_samples)

    # Resample until at least one passes filter (or max rounds).
    runner_outputs = {t["name"]: [] for t in tasks}
    remaining = list(tasks)
    rounds = max(1, q_max_rounds) if q_enable else 1
    for _ in range(rounds):
        if not remaining:
            break
        req_tasks = [{**t, "num_samples": request_num_samples} for t in remaining]
        raw_round = runner.design_sequences(req_tasks)
        next_remaining = []
        for t in remaining:
            name = t["name"]
            raw_samples = raw_round.get(name) or []
            if not q_enable:
                runner_outputs[name] = raw_samples
                continue
            expected_len = getattr(global_config.protomers.get(name), "length", None)
            filtered, _ = filter_dplm_sequences(raw_samples, cfg=q_cfg, expected_length=expected_len)
            if filtered:
                runner_outputs[name] = filtered
            else:
                next_remaining.append(t)
        remaining = next_remaining

    updated = 0
    for proto_name, segments in reinit_plan.items():
        samples = runner_outputs.get(proto_name) or []
        if not samples and q_fallback:
            # Last resort: accept raw output.
            t = task_by_name.get(proto_name)
            if t is None:
                raw_samples = []
            else:
                raw_samples = runner.design_sequences(
                    [{"name": proto_name, "masked_sequence": t["masked_sequence"], "num_samples": base_num_samples}]
                ).get(proto_name) or []
            samples = raw_samples
        if not samples:
            logging.warning(
                "DPLM returned no sequences for protomer '%s'; skipping.", proto_name
            )
            continue
        new_sequence = samples[0]
        protomer = global_config.protomers.get(proto_name)
        if protomer is None:
            continue
        if len(new_sequence) != protomer.length:
            logging.warning(
                "Length mismatch while applying DPLM sequence to protomer '%s' "
                "(expected %d, got %d).",
                proto_name,
                protomer.length,
                len(new_sequence),
            )
            continue
        for segment_name, seg_slice in segments:
            segment = global_config.segments.get(segment_name)
            if segment is None:
                continue
            segment.sequence = new_sequence[seg_slice.start : seg_slice.stop]
            segment.initialized_from_random = False
        updated += 1

    logging.info("Reinitialized %d protomer(s) with DPLM.", updated)


def reinitialize_random_segments_with_dplm(args) -> None:
    if not getattr(args, "reinitialize_random_segments_with_dplm", False):
        return
    _apply_dplm_initializer(args)

def build_initializer(args, *, mutators=None):
    """Build the configured initializer implementation."""
    mode = _resolve_initializer_mode(args)
    if mode == "random":
        return RandomInitializer()
    if mode == "dplm":
        return DPLMInitializer(functools.partial(_apply_dplm_initializer, args))
    if mode == "aa_set":
        return AminoAcidSetInitializer(functools.partial(_apply_aa_set_initializer, args))
    if mode == "mask_mix":
        return MaskMixInitializer(functools.partial(_apply_mask_mix_initializer, args))
    if mode == "all_a":
        return AllAInitializer(
            functools.partial(
                _apply_aa_set_initializer,
                args,
                fixed_residue_set=["A"],
            )
        )
    if mode == "pssm":
        return PSSMInitializer(functools.partial(_apply_pssm_initializer, args))
    if mode == "x_mix":
        return XMixInitializer(functools.partial(_apply_x_mix_initializer, args))
    if mode in {"structure", "rfd3"}:
        if mutators is None:
            raise ValueError(
                f"initializer.mode={mode!r} requires initialized mutators."
            )
        initializer_cls = (
            ExistingStructureInitializer if mode == "structure" else RFD3Initializer
        )
        return initializer_cls(
            args=args,
            mutators=mutators,
            segment_plan=_collect_mutable_protein_segments(),
        )
    raise ValueError(
        f"Unknown initializer.mode={mode!r}. Expected one of: random, dplm, aa_set, "
        "mask_mix, pssm, all_a, x_mix, structure, rfd3."
    )


def apply_initializer(args, *, mutators=None):
    """Apply one initializer implementation to the parsed random protein segments."""
    mode = _resolve_initializer_mode(args)
    initializer = build_initializer(args, mutators=mutators)
    with event_span(
        logger=logger,
        component="initializer",
        alias=mode,
        stage="initialization",
        step=0,
        detail=initializer.__class__.__name__,
    ) as span:
        result = initializer.apply()
        sequence_names = sorted(result.sequences)
        event_fields = {
            "sequence_count": len(sequence_names),
        }
        if sequence_names:
            event_fields["sequence_names"] = summarize_event_items(sequence_names)
        if result.manifest_path is not None:
            event_fields["manifest"] = result.manifest_path
        span.finish(
            artifact=result.structure_path or result.manifest_path,
            result=result.mode,
            **event_fields,
        )
        return result


def _uses_alphafold3_evaluator(args, stage_manager) -> bool:
    evaluator = getattr(args, "evaluator", None)
    if evaluator is None:
        return False

    if isinstance(evaluator, dict) or OmegaConf.is_dict(evaluator):
        terms = evaluator.get("evaluator_terms")
    else:
        terms = getattr(evaluator, "evaluator_terms", None)

    if terms is None:
        raise ValueError(
            "Invalid `evaluator` config: expected mapping with `evaluator_terms`."
        )
    if not (isinstance(terms, dict) or OmegaConf.is_dict(terms)):
        raise ValueError(
            "Invalid `evaluator.evaluator_terms`: expected mapping of alias -> config."
        )

    used_aliases: set[str] = set()
    default_alias = getattr(stage_manager.default_setting, "evaluator", None)

    for unit in getattr(stage_manager, "stage_args", {}).values():
        alias = getattr(unit, "evaluator", None) or default_alias
        if alias:
            used_aliases.add(str(alias))
        gate = getattr(unit, "transition_consistency_gate", None)
        gate_alias = getattr(gate, "evaluator", None) if gate is not None else None
        if gate_alias:
            used_aliases.add(str(gate_alias))

    if not used_aliases:
        raise ValueError(
            "No evaluator aliases were selected by stages; expected at least one alias."
        )

    for alias in used_aliases:
        conf = terms.get(alias)
        if conf is None:
            available = sorted(str(k) for k, v in terms.items() if v is not None)
            raise ValueError(
                f"Stage-selected evaluator alias {alias!r} is not defined. "
                f"Available aliases: {available}."
            )
        if not (isinstance(conf, dict) or OmegaConf.is_dict(conf)):
            raise ValueError(
                f"Invalid evaluator term for alias {alias!r}: expected mapping, got {type(conf).__name__}."
            )
        if str(conf.get("_type") or "").lower() == "alphafold3":
            return True
    return False


def calculate_token_numbers(inputs: Dict[str, "Input"]) -> Dict[str, int]:
    """
    Calculate token numbers for each input oligomer.

    This is used as a best-effort fallback for evaluator bucket selection.

    We intentionally derive token counts from the same tokenization logic used by
    AF3 featurisation, because it accounts for non-polymer chains (e.g. small-
    molecule ligands) as well as polymers.

    NOTE: AF3's vendor cleaning drops chains whose sequence is entirely unknown
    (`X...X`). We treat that as an intentional masked initialization, so token
    counting uses the same compatibility patch as evaluator featurisation.
    """
    try:
        from alphafold3.constants import chemical_components
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "AlphaFold3 evaluator selected but AF3 featurisation deps are missing. "
            "Fix the env (recommended: `gh doctor --strict --task <task>`), or switch evaluator."
        ) from exc

    def count_tokens(*, fold_input, ccd) -> int:
        # Avoid calling the full AF3 pipeline (MSA/templates/etc). We only need
        # the token count used for bucketing, which is computed right after
        # structure cleaning and tokenization.
        from alphafold3.model import features
        from alphafold3.model.pipeline import inter_chain_bonds
        from alphafold3.model.pipeline import pipeline as pipeline_lib
        from alphafold3.model.pipeline import structure_cleaning

        config = pipeline_lib.WholePdbPipeline.Config()
        logging_name = getattr(fold_input, "name", "<input>")

        struct = fold_input.to_structure(ccd=ccd)
        cleaned_struc, _cleaning_metadata = structure_cleaning.clean_structure(
            struct,
            ccd=ccd,
            drop_non_standard_atoms=True,
            drop_missing_sequence=True,
            filter_clashes=config.filter_clashes,
            filter_crystal_aids=config.filter_crystal_aids,
            filter_waters=True,
            filter_hydrogens=True,
            filter_leaving_atoms=config.drop_ligand_leaving_atoms,
            only_glycan_ligands_for_leaving_atoms=True,
            covalent_bonds_only=True,
            remove_polymer_polymer_bonds=True,
            remove_bad_bonds=True,
            remove_nonsymmetric_bonds=config.remove_nonsymmetric_bonds,
        )
        if cleaned_struc.num_chains == 0:
            raise pipeline_lib.MmcifNumChainsError(f"{logging_name}: No chains in structure!")

        polymer_ligand_bonds, ligand_ligand_bonds = (
            inter_chain_bonds.get_polymer_ligand_and_ligand_ligand_bonds(
                cleaned_struc,
                only_glycan_ligands=False,
                allow_multiple_bonds_per_atom=True,
            )
        )
        if ligand_ligand_bonds is not None and not ligand_ligand_bonds.atom_name.size:
            ligand_ligand_bonds = None
        if polymer_ligand_bonds is not None and not polymer_ligand_bonds.atom_name.size:
            polymer_ligand_bonds = None

        _empty_output_struc, flat_output_layout = (
            structure_cleaning.create_empty_output_struc_and_layout(
                struc=cleaned_struc,
                ccd=ccd,
                polymer_ligand_bonds=polymer_ligand_bonds,
                ligand_ligand_bonds=ligand_ligand_bonds,
                drop_ligand_leaving_atoms=config.drop_ligand_leaving_atoms,
            )
        )
        all_tokens, _all_token_atoms_layout, _standard_token_idxs = features.tokenizer(
            flat_output_layout,
            ccd=ccd,
            max_atoms_per_token=config.max_atoms_per_token,
            flatten_non_standard_residues=config.flatten_non_standard_residues,
            logging_name=logging_name,
        )
        return int(len(all_tokens.atom_name))

    from astrolabe.utils.af3_compat import (
        allow_all_x_protein_chains_during_featurisation,
        to_af3_input,
    )

    num_tokens: dict[str, int] = {}
    for oligo_name, input_data in inputs.items():
        af3_input = to_af3_input(input_data)
        ccd = chemical_components.Ccd(user_ccd=af3_input.user_ccd)
        with allow_all_x_protein_chains_during_featurisation(af3_input):
            token_count = count_tokens(fold_input=af3_input, ccd=ccd)
        num_tokens[oligo_name] = int(token_count)

    return num_tokens


def _needs_molfile_to_params(args) -> bool:
    if not bool(getattr(args, "enable_pyrosetta", False)):
        return False
    has_smiles_or_ccd = any(
        p.type in (ProtomerTypes.SMILES, ProtomerTypes.CCD)
        for p in global_config.protomers.values()
    )
    has_nonstandard_aliases = bool(getattr(global_config, "nonstandard_aliases", {}) or {})
    return bool(has_smiles_or_ccd or has_nonstandard_aliases)


def _init_pyrosetta_runtime(structure_files=None) -> None:
    setup_extra_residues(structure_files=structure_files)
    try:
        from contextlib import redirect_stderr, redirect_stdout

        import pyrosetta  # type: ignore
        from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized

        if pyrosetta_was_initialized(pyrosetta):
            return

        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                param_files = list(
                    pathlib.Path(global_config.out_dir).glob("raw_model_results/*.params")
                )
                extra_res = (
                    f" -extra_res_fa {' '.join(str(i) for i in param_files)}"
                    if param_files
                    else ""
                )
                init_opts = f"-ignore_zero_occupancy false -mute all -beta{extra_res}"

                # Some PyRosetta-based metrics require a DAlphaBall executable.
                # Prefer an explicit override via GH_DALPHABALL; otherwise fall back
                # to the bundled binary when available.
                dalpha = os.environ.get("GH_DALPHABALL", "").strip()
                dalpha_path: pathlib.Path | None = None
                if dalpha:
                    expanded = os.path.expanduser(os.path.expandvars(dalpha))
                    p = pathlib.Path(expanded)
                    if p.exists():
                        dalpha_path = p
                else:
                    bundled = (
                        pathlib.Path(__file__).resolve().parent
                        / "astrolabe"
                        / "assets"
                        / "DAlphaBall.gcc"
                    )
                    if bundled.exists():
                        dalpha_path = bundled

                if dalpha_path is not None:
                    try:
                        mode = dalpha_path.stat().st_mode
                        if (mode & 0o100) == 0:
                            dalpha_path.chmod(mode | 0o100)
                    except Exception:
                        pass
                    init_opts = f"{init_opts} -holes:dalphaball {dalpha_path}"

                pyrosetta.init(init_opts)
    except Exception as exc:
        raise RuntimeError("PyRosetta init failed") from exc


def _loss_computer_needs_complex_poses(loss_computer: LossEngine | None) -> bool:
    """Return whether active loss specs require PyRosetta-backed poses."""
    if loss_computer is None:
        return False
    for spec_tuple in getattr(loss_computer, "loss_specs", {}).values():
        if not spec_tuple:
            continue
        spec_name = spec_tuple[0]
        if spec_name in PYROSETTA_LOSS_TYPES:
            return True
        loss_func = spec_tuple[1] if len(spec_tuple) > 1 else None
        code = getattr(loss_func, "__code__", None)
        if code is not None and "_require_pyrosetta" in code.co_names:
            return True
    return False


def _loss_computer_needs_pyrosetta_runtime(loss_computer: LossEngine | None) -> bool:
    return bool(
        _loss_computer_needs_complex_poses(loss_computer)
        or _loss_computer_needs_post_eval_hook(loss_computer)
    )


def _loss_computer_specs(loss_computer: LossEngine | None) -> set[str]:
    if loss_computer is None:
        return set()
    specs = set()
    for spec_tuple in getattr(loss_computer, "loss_specs", {}).values():
        if spec_tuple:
            specs.add(str(spec_tuple[0]))
    return specs


def _loss_computer_needs_post_eval_hook(loss_computer: LossEngine | None) -> bool:
    return bool(_loss_computer_specs(loss_computer).intersection(POST_EVAL_HOOK_LOSS_TYPES))


def _global_config_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(global_config, name))
    except RuntimeError:
        return default


def _sync_enable_pyrosetta_from_loss_computer(
    args,
    loss_computer: LossEngine | None,
) -> bool:
    if not _loss_computer_needs_pyrosetta_runtime(loss_computer):
        return False
    if not bool(getattr(args, "enable_pyrosetta", False)):
        setattr(args, "enable_pyrosetta", True)
    if not _global_config_bool("enable_pyrosetta", False):
        try:
            global_config.update_const("enable_pyrosetta", True)
        except RuntimeError:
            pass
    return True


def _evaluation_result_needs_complex_poses(
    args,
    evaluation_result,
    loss_computer: LossEngine | None = None,
) -> bool:
    if bool(getattr(args, "skip_complex_poses", False)):
        return False
    if getattr(evaluation_result, "complex_poses", None) is not None:
        return False
    return bool(
        getattr(args, "enable_pyrosetta", False)
        or _global_config_bool("enable_pyrosetta", False)
        or per_residue_scores_requires_complex_poses(getattr(args, "per_residue_scores", None))
        or _loss_computer_needs_complex_poses(loss_computer)
    )


def _current_loss_computer(loss_computer: LossEngine | None) -> LossEngine | None:
    if loss_computer is not None:
        return loss_computer
    try:
        return global_config.get_const("loss_computer")
    except RuntimeError:
        return None


def _ensure_evaluation_complex_poses(
    args,
    evaluation_result,
    loss_computer: LossEngine | None = None,
):
    loss_computer = _current_loss_computer(loss_computer)
    _sync_enable_pyrosetta_from_loss_computer(args, loss_computer)
    if not _evaluation_result_needs_complex_poses(args, evaluation_result, loss_computer):
        return evaluation_result
    _init_pyrosetta_runtime(structure_files=evaluation_result.complex_paths)
    evaluation_result.complex_poses = ComplexPoses.construct_complex_poses(
        evaluation_result
    )
    return evaluation_result


def _attach_evaluation_per_residue_scores(args, evaluation_result):
    cfg = getattr(args, "per_residue_scores", None)
    if not per_residue_scores_enabled(cfg):
        return evaluation_result
    return attach_per_residue_scores(evaluation_result, cfg)


def _finalize_evaluation_result(args, evaluation_result, loss_computer: LossEngine | None = None):
    evaluation_result = _ensure_evaluation_complex_poses(
        args,
        evaluation_result,
        loss_computer,
    )
    return _attach_evaluation_per_residue_scores(args, evaluation_result)


def _post_eval_hook_config(args, loss_computer: LossEngine | None = None):
    raw = getattr(args, "post_eval_hooks", None)
    try:
        raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    if raw in (None, {}, "null", "None"):
        raw = None
    if raw is None and _loss_computer_needs_post_eval_hook(loss_computer):
        return {"rosetta_minimize": {"enabled": True, "mode": "repack"}}
    if not isinstance(raw, dict):
        return None
    hook = raw.get("rosetta_minimize") or raw.get("rosetta_repack")
    if hook in (None, {}, "null", "None"):
        return None
    if not isinstance(hook, dict):
        raise ValueError("post_eval_hooks.rosetta_minimize must be a mapping.")
    if not bool(hook.get("enabled", False)):
        return None
    return {"rosetta_minimize": hook}


def _apply_post_eval_hooks(
    args,
    evaluation_result,
    *,
    runtime_config: Dict,
    loss_computer: LossEngine | None = None,
):
    hook_config = _post_eval_hook_config(args, loss_computer)
    if hook_config is None:
        return evaluation_result
    hook = hook_config["rosetta_minimize"]
    from astrolabe.post_eval_hooks import rosetta_minimize

    scheduling = _evaluation_scheduling_config(args)
    if not bool(_config_get(scheduling, "overlap_cpu_postprocess", False)):
        _init_pyrosetta_runtime(structure_files=evaluation_result.complex_paths)
    step = int(runtime_config.get("step", getattr(global_config, "step", 0)) or 0)
    out_dir = global_config.out_dir / "structures" / "post_eval_hooks"
    evaluation_result = rosetta_minimize.apply_rosetta_minimize_hook(
        evaluation_result,
        hook,
        out_dir=out_dir,
        step=step,
    )
    # Paths changed; any old pose cache on this StandardScores object is stale.
    evaluation_result.complex_poses = None
    candidates = getattr(evaluation_result, "candidate_scores", None)
    if candidates:
        updated_candidates = []
        for idx, candidate in enumerate(candidates):
            candidate = rosetta_minimize.apply_rosetta_minimize_hook(
                candidate,
                hook,
                out_dir=out_dir / "candidates" / f"candidate_{idx:03d}",
                step=step,
            )
            candidate.complex_poses = None
            updated_candidates.append(candidate)
        evaluation_result.candidate_scores = updated_candidates
    return evaluation_result


def _standard_scores_field_names() -> set[str]:
    return set(getattr(StandardScores, "__dataclass_fields__", {}).keys())


def _merge_standard_scores(results: list[StandardScores]) -> StandardScores:
    if not results:
        raise ValueError("Cannot merge empty evaluation results.")
    if len(results) == 1:
        return results[0]

    merged: dict[str, Any] = {"names": []}
    field_names = _standard_scores_field_names()
    for field in field_names - {"names"}:
        merged[field] = None

    per_proto_scores: dict[str, Any] = {}
    per_proto_score_terms: dict[str, Any] = {}
    score_contributions: list[Any] = []
    complex_pose_dict: dict[str, Any] = {}
    hook_reports: dict[str, Any] = {}
    for result in results:
        merged["names"].extend(list(getattr(result, "names", []) or []))
        for field in field_names - {
            "names",
            "per_proto_scores",
            "complex_poses",
            "score_contributions",
        }:
            value = getattr(result, field, None)
            if isinstance(value, dict):
                if merged[field] is None:
                    merged[field] = {}
                merged[field].update(value)
            elif value is not None and merged[field] is None:
                merged[field] = value

        for proto, scores in (getattr(result, "per_proto_scores", None) or {}).items():
            per_proto_scores.setdefault(proto, scores)

        for proto, term_map in (getattr(result, "per_proto_score_terms", None) or {}).items():
            target = per_proto_score_terms.setdefault(proto, {})
            if isinstance(term_map, dict):
                target.update(term_map)

        score_contributions.extend(
            normalize_score_contributions(getattr(result, "score_contributions", None))
        )

        complex_poses = getattr(result, "complex_poses", None)
        poses = getattr(complex_poses, "poses", None)
        if isinstance(poses, dict):
            complex_pose_dict.update(poses)

        reports = getattr(result, "post_eval_hook_reports", None)
        if isinstance(reports, dict):
            hook_reports.update(reports)

    if per_proto_scores:
        merged["per_proto_scores"] = per_proto_scores
    if per_proto_score_terms:
        merged["per_proto_score_terms"] = per_proto_score_terms
    if score_contributions:
        merged["score_contributions"] = score_contributions
    if complex_pose_dict:
        merged["complex_poses"] = ComplexPoses(complex_pose_dict)

    out = StandardScores(**merged)
    if hook_reports:
        out.post_eval_hook_reports = hook_reports
    return out


def _should_evaluate_oligomers_incrementally(
    args,
    oligo_names: list[str] | None,
    loss_computer: LossEngine | None,
) -> bool:
    if not oligo_names or len(oligo_names) <= 1:
        return False
    return _post_eval_hook_config(args, loss_computer) is not None


def initialize_pipeline_components(args) -> Tuple[Any, Any, Any, Dict[str, "Input"]]:
    """Initialize all core pipeline components."""
    initialize_global_config(args)
    runtime_context = global_config.build_runtime_context()
    setup_protomers(args)
    apply_cyclization(args)

    mutators = None
    if _resolve_initializer_mode(args) in {"structure", "rfd3"}:
        mutators = build_mutators(args.mutator, args.raw_model_results_dir)
        global_config.init_const("mutators", mutators)
    initializer_result = (
        apply_initializer(args)
        if mutators is None
        else apply_initializer(args, mutators=mutators)
    )
    global_config.init_const("initializer_result", initializer_result)

    # Initialize components
    if mutators is None:
        mutators = build_mutators(args.mutator, args.raw_model_results_dir)
        global_config.init_const("mutators", mutators)
    loss_computer = LossEngine(args.loss, runtime_context=runtime_context)
    global_config.init_const("loss_computer", loss_computer)
    stage_manager = StageManager(
        args.stages,
        loss_computer=loss_computer,
        runtime_context=runtime_context,
    )
    _sync_enable_pyrosetta_from_loss_computer(args, loss_computer)
    global_config.init_const("stage_manager", stage_manager)
    _validate_nonstandard_backend_compat(args, stage_manager)
    evaluators = construct_evaluators(
        args.evaluator,
        args.raw_model_results_dir,
        stage_manager.all_oligos,
        enabled_aliases=getattr(args.evaluator, "enabled_aliases", None),
    )
    global_config.init_const("evaluators", evaluators)

    inputs = construct_standard_input(stage_manager.current_oligo, args.seed)

    # Setup token numbers
    num_tokens = {}
    if _uses_alphafold3_evaluator(args, stage_manager):
        num_tokens = calculate_token_numbers(
            construct_standard_input(stage_manager.all_oligos, args.seed)
        )
    global_config.init_const("num_tokens", num_tokens)
    global_config.set_variable("step", 0)

    initialize_predockers(args.predocker)
    # -------------------------
    global_config.init_const("obabel_path", args.obabel_path)
    needs_molfile_to_params = _needs_molfile_to_params(args)
    if needs_molfile_to_params:
        if OmegaConf.is_missing(args, "molfile_to_params") or not args.molfile_to_params:
            raise ValueError(
                "Missing `molfile_to_params`. This is required when `enable_pyrosetta=true` "
                "and the design includes SMILES/CCD protomers. Set GH_MOLFILE_TO_PARAMS to "
                "point to Rosetta's molfile_to_params.py, or disable PyRosetta / remove ligands."
            )
        global_config.init_const("molfile_to_params", args.molfile_to_params)
    else:
        # Avoid resolving missing mandatory values for tasks that don't need Rosetta.
        global_config.init_const("molfile_to_params", None)
    global_config.init_const("dssp_path", getattr(args, "dssp_path", None))
    global_config.init_const("usalign_path", getattr(args, "usalign_path", None))
    global_config.init_const("ananas_path", getattr(args, "ananas_path", None))
    if getattr(args, "structure_start", None) is not None:
        _init_pyrosetta_runtime()
    structure_start_state = build_structure_start_state(
        getattr(args, "structure_start", None),
        loss_computer=loss_computer,
    )
    global_config.init_const("structure_start", structure_start_state)

    # Materialize standalone ligand files before the first RF3 evaluation so
    # fixed_ligand inputs can use a complete conformer source instead of a
    # truncated residue copied from the complex structure.
    if needs_molfile_to_params:
        setup_extra_residues(ligand_resnames=getattr(args, "ligand_resnames", None))

    # assign_predocker_to_mutators(mutators, args.predocker)
    return loss_computer, stage_manager, inputs


def _prepare_structure_start_runtime(
    evaluation_result,
    *,
    timing: str,
    loss_computer: LossEngine,
    runtime_config: Dict,
):
    runtime_config.pop("reference_gate_result", None)
    evaluation_result, result = select_structure_start_candidate(
        evaluation_result,
        timing=timing,
        loss_computer=loss_computer,
    )
    if result is not None:
        runtime_config["reference_gate_result"] = result
    return evaluation_result


def _current_stage_name() -> str | None:
    try:
        stage_manager = global_config.stage_manager
    except RuntimeError:
        return None
    if stage_manager is None:
        return None
    return getattr(stage_manager, "current_stage", None)


def _evaluator_event_alias(evaluator: Any) -> str:
    for attr in ("alias", "name", "_alias"):
        value = getattr(evaluator, attr, None)
        if value:
            return str(value)
    return evaluator.__class__.__name__


def _format_oligo_names(oligo_names: list[str] | None) -> str | None:
    if oligo_names is None:
        return None
    return ",".join(str(name) for name in oligo_names)


@dataclass(frozen=True)
class EvaluationWorkItem:
    """Main-loop evaluator work item with stable work/result mapping."""

    item_id: str
    step: int
    stage: str | None
    evaluator_alias: str
    evaluator: Any
    oligo_names: list[str] | None
    inputs: Dict
    runtime_config: Dict
    detail: str


@dataclass(frozen=True)
class EvaluationResultItem:
    work_item: EvaluationWorkItem
    evaluation_result: Any


def _evaluation_item_id(*, step: int, oligo_names: list[str] | None) -> str:
    label = _format_oligo_names(oligo_names) or "all"
    return f"step:{int(step)}:{label}"


def _build_evaluation_work_items(
    args,
    evaluator: Any,
    runtime_config: Dict,
    step: int,
    oligo_names: list[str] | None,
    loss_computer: LossEngine | None,
) -> list[EvaluationWorkItem]:
    evaluator_alias = _evaluator_event_alias(evaluator)
    stage = _current_stage_name()
    if _should_evaluate_oligomers_incrementally(args, oligo_names, loss_computer):
        work_items = []
        for oligo_name in list(oligo_names or []):
            item_oligos = [str(oligo_name)]
            inputs = construct_standard_input(item_oligos, args.seed)
            one_runtime = make_evaluation_runtime_config(
                step,
                inputs,
                dict(runtime_config),
            )
            work_items.append(
                EvaluationWorkItem(
                    item_id=_evaluation_item_id(step=step, oligo_names=item_oligos),
                    step=int(step),
                    stage=stage,
                    evaluator_alias=evaluator_alias,
                    evaluator=evaluator,
                    oligo_names=item_oligos,
                    inputs=inputs,
                    runtime_config=one_runtime,
                    detail="incremental",
                )
            )
        return work_items

    inputs = construct_standard_input(oligo_names, args.seed)
    work_runtime = make_evaluation_runtime_config(step, inputs, runtime_config)
    return [
        EvaluationWorkItem(
            item_id=_evaluation_item_id(step=step, oligo_names=oligo_names),
            step=int(step),
            stage=stage,
            evaluator_alias=evaluator_alias,
            evaluator=evaluator,
            oligo_names=None if oligo_names is None else list(oligo_names),
            inputs=inputs,
            runtime_config=work_runtime,
            detail="step",
        )
    ]


def _execute_evaluation_work_item(work_item: EvaluationWorkItem) -> EvaluationResultItem:
    result = _evaluate_with_observability(
        work_item.evaluator,
        work_item.inputs,
        work_item.runtime_config,
        step=work_item.step,
        oligo_names=work_item.oligo_names,
        detail=work_item.detail,
    )
    return EvaluationResultItem(work_item=work_item, evaluation_result=result)


def _finalize_evaluation_result_item(
    args,
    result_item: EvaluationResultItem,
    loss_computer: LossEngine | None,
) -> EvaluationResultItem:
    evaluation_result = _apply_post_eval_hooks(
        args,
        result_item.evaluation_result,
        runtime_config=result_item.work_item.runtime_config,
        loss_computer=loss_computer,
    )
    evaluation_result = _finalize_evaluation_result(
        args,
        evaluation_result,
        loss_computer,
    )
    return EvaluationResultItem(
        work_item=result_item.work_item,
        evaluation_result=evaluation_result,
    )


def _merge_evaluation_result_items(result_items: list[EvaluationResultItem]) -> Any:
    results = [item.evaluation_result for item in result_items]
    if len(results) == 1:
        return results[0]
    return _merge_standard_scores(results)


def _config_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        if OmegaConf.is_config(node):
            return node.get(key, default)
    except Exception:
        pass
    return getattr(node, key, default)


def _evaluation_scheduling_config(args: Any) -> Any:
    return _config_get(_config_get(args, "evaluator", None), "scheduling", None)


def _overlap_cpu_postprocess_enabled(args: Any, work_items: list[EvaluationWorkItem]) -> bool:
    if len(work_items) <= 1:
        return False
    scheduling = _evaluation_scheduling_config(args)
    return bool(_config_get(scheduling, "overlap_cpu_postprocess", False))


def _cpu_postprocess_workers(args: Any, work_items: list[EvaluationWorkItem]) -> int:
    scheduling = _evaluation_scheduling_config(args)
    raw_workers = _config_get(scheduling, "cpu_postprocess_workers", 1)
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluator.scheduling.cpu_postprocess_workers must be an integer >= 1") from exc
    if workers < 1:
        raise ValueError("evaluator.scheduling.cpu_postprocess_workers must be an integer >= 1")
    return min(workers, len(work_items))


def _execute_evaluation_work_items(
    args,
    work_items: list[EvaluationWorkItem],
    loss_computer: LossEngine | None,
) -> list[EvaluationResultItem]:
    if not _overlap_cpu_postprocess_enabled(args, work_items):
        return [
            _finalize_evaluation_result_item(
                args,
                _execute_evaluation_work_item(work_item),
                loss_computer,
            )
            for work_item in work_items
        ]

    max_workers = _cpu_postprocess_workers(args, work_items)
    finalized_items: list[EvaluationResultItem] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for work_item in work_items:
            result_item = _execute_evaluation_work_item(work_item)
            futures.append(
                executor.submit(
                    _finalize_evaluation_result_item,
                    args,
                    result_item,
                    loss_computer,
                )
            )
        for future in futures:
            finalized_items.append(future.result())
    return finalized_items


def _evaluation_artifact_summary(evaluation_result: Any) -> str | None:
    complex_paths = getattr(evaluation_result, "complex_paths", None)
    if isinstance(complex_paths, dict) and complex_paths:
        parts = []
        for name in sorted(complex_paths):
            parts.append(f"{name}:{complex_paths[name]}")
            if len(parts) == 3:
                break
        if len(complex_paths) > len(parts):
            parts.append(f"+{len(complex_paths) - len(parts)}")
        return ",".join(parts)

    names = getattr(evaluation_result, "names", None)
    if names:
        return ",".join(str(name) for name in names)
    return None


def _standard_score_metric_mean(
    evaluation_result: Any,
    metric_name: str,
) -> float | None:
    values_by_oligomer = getattr(evaluation_result, metric_name, None)
    if not isinstance(values_by_oligomer, dict):
        return None

    run_means = []
    for values in values_by_oligomer.values():
        run_values = values if isinstance(values, (list, tuple)) else [values]
        for value in run_values:
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                if value.size == 0 or not np.issubdtype(value.dtype, np.number):
                    continue
                finite_values = value[np.isfinite(value)]
                if finite_values.size == 0:
                    continue
                scalar = float(np.mean(finite_values))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                scalar = float(value)
            else:
                continue
            if math.isfinite(scalar):
                run_means.append(scalar)
    if not run_means:
        return None
    return sum(run_means) / len(run_means)


def _evaluation_result_event_fields(evaluation_result: Any) -> dict[str, Any]:
    names = list(getattr(evaluation_result, "names", None) or [])
    complex_paths = getattr(evaluation_result, "complex_paths", None)
    fields: dict[str, Any] = {
        "result": "standard_scores",
        "oligomer_count": len(names),
        "artifact_count": len(complex_paths) if isinstance(complex_paths, dict) else 0,
    }
    if names:
        fields["oligomers"] = summarize_event_items(names)
    for metric_name in ("plddt", "ptm", "iptm"):
        metric_mean = _standard_score_metric_mean(evaluation_result, metric_name)
        if metric_mean is not None:
            fields[f"{metric_name}_mean"] = metric_mean
    return fields


def _evaluate_with_observability(
    evaluator: Any,
    inputs: Dict,
    runtime_config: Dict,
    *,
    step: int,
    oligo_names: list[str] | None,
    detail: str,
) -> Any:
    with event_span(
        logger=logger,
        component="evaluator",
        alias=_evaluator_event_alias(evaluator),
        stage=_current_stage_name(),
        step=step,
        oligo=_format_oligo_names(oligo_names),
        detail=detail,
    ) as span:
        evaluation_result = evaluator.evaluate(inputs, runtime_config)
        span.finish(
            artifact=_evaluation_artifact_summary(evaluation_result),
            **_evaluation_result_event_fields(evaluation_result),
        )
        return evaluation_result


@timer("evaluation")
def perform_evaluation(
    args,
    evaluator,
    runtime_config: Dict,
    step: int,
    oligo_names: list[str] = None,
    loss_computer: LossEngine | None = None,
) -> Any:
    """Perform evaluation for a given step."""
    work_items = _build_evaluation_work_items(
        args,
        evaluator,
        runtime_config,
        step,
        oligo_names,
        loss_computer,
    )
    result_items = _execute_evaluation_work_items(args, work_items, loss_computer)
    return _merge_evaluation_result_items(result_items)


def load_or_evaluate_initial(
    args, evaluator, inputs: Dict, runtime_config: Dict, task_name: str, loss_computer: LossEngine | None = None
):
    """Load cached evaluation results or perform initial evaluation."""
    eval_file = f"{task_name}_eval_rslt.pkl"
    is_debug = os.getenv("DEBUG") == "1"

    # Load from cache if in debug mode and file exists
    if is_debug and os.path.exists(eval_file):
        logging.info(f"Loading cached evaluation results from {eval_file}")
        log_event(
            logger=logger,
            level=logging.INFO,
            event="span_end",
            component="evaluator",
            alias=_evaluator_event_alias(evaluator),
            stage=_current_stage_name(),
            step=0,
            status="cache_hit",
            artifact=eval_file,
            detail_level="summary",
            detail="initial_cache",
        )
        with open(eval_file, "rb") as f:
            evaluation_result = pickle.load(f)
            # evaluation_result.complex_poses = ComplexPoses.construct_complex_poses(
            #     evaluation_result
            # )
    else:
        # Perform evaluation
        logging.info("Performing initial evaluation")
        evaluation_result = _evaluate_with_observability(
            evaluator,
            inputs,
            runtime_config,
            step=0,
            oligo_names=list(inputs.keys()) if isinstance(inputs, dict) else None,
            detail="initial",
        )
        evaluation_result = _apply_post_eval_hooks(
            args,
            evaluation_result,
            runtime_config=runtime_config,
            loss_computer=loss_computer,
        )

        # Cache results if in debug mode
        if "DEBUG" in os.environ:
            with open(eval_file, "wb") as f:
                pickle.dump(evaluation_result, f)
            logging.info(f"Cached evaluation results to {eval_file}")

    return _finalize_evaluation_result(args, evaluation_result, loss_computer)


def calculate_mutation_counts(
    protomers: Dict[str, Protomer],
    mutation_fraction: float | Dict[str, float],
    mutation_num: int | Dict[str, int] | float,
) -> Dict[str, int]:
    """Calculate number of mutations for each protomer."""

    def _mutable_length(protomer: Protomer) -> int:
        return getattr(protomer, "mutable_length", len(protomer.sequence))

    def _parse_percentage(value: str) -> float | None:
        stripped = value.strip()
        if stripped.endswith("%"):
            number_str = stripped[:-1].strip()
            if not number_str:
                raise ValueError("Percentage value for mutation_num is empty.")
            try:
                return float(number_str) / 100.0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid percentage value for mutation_num: '{value}'"
                ) from exc
        return None

    def _convert_string_value(value: str) -> float:
        stripped = value.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for mutation_num: '{value}'"
            ) from exc

    if isinstance(mutation_num, str):
        percent = _parse_percentage(mutation_num)
        if percent is not None:
            return {
                name: round(_mutable_length(protomer) * percent)
                for name, protomer in protomers.items()
            }
        mutation_num = int(round(_convert_string_value(mutation_num)))

    if isinstance(mutation_num, int):
        return {proto: mutation_num for proto in protomers}

    if isinstance(mutation_num, dict):
        result = {}
        for proto, protomer in protomers.items():
            value = mutation_num.get(proto, 0)
            if isinstance(value, str):
                percent = _parse_percentage(value)
                if percent is not None:
                    result[proto] = round(_mutable_length(protomer) * percent)
                    continue
                value = _convert_string_value(value)
            result[proto] = int(round(value))
        return result

    if isinstance(mutation_fraction, float):
        return {
            name: round(_mutable_length(protomer) * mutation_fraction)
            for name, protomer in protomers.items()
        }

    if isinstance(mutation_fraction, dict):
        return {
            name: round(_mutable_length(protomer) * mutation_fraction.get(name, 0))
            for name, protomer in protomers.items()
        }

    raise ValueError("Invalid mutation configuration")


def validate_mutation_counts(
    protomers: Dict[str, Protomer], mutation_counts: Dict[str, int]
) -> None:
    """Validate that mutation counts don't exceed sequence lengths."""
    for name, protomer in protomers.items():
        limit = getattr(protomer, "mutable_length", len(protomer.sequence))
        if mutation_counts[name] > limit:
            raise ValueError(
                f"Mutation count ({mutation_counts[name]}) exceeds sequence length "
                f"({limit}) for protomer '{name}'"
            )


@timer("mutation step")
def execute_mutation_step(
    mutators, evaluation_result, runtime_config: Dict, step: int
) -> None:
    """Execute mutation for all mutators."""
    secondary_structure_by_proto = extract_secondary_structure_by_proto(evaluation_result)
    runtime_config["mutator_selector_score_contexts"] = []
    pre_update = bool(runtime_config.get("mutator_stack_pre_update", False))
    if pre_update:
        global_config.segment_manager.begin_pre_update_step()
    max_entropy = runtime_config.get("mutator_stack_pre_update_max_entropy")
    max_entropy = None if max_entropy in (None, "null", "None") else float(max_entropy)
    for index, mutator in enumerate(mutators):
        runtime_config = build_mutator_runtime_config(
            step=step,
            outprefix="test",
            former_config=runtime_config,
        )
        runtime_config["current_mutator_alias"] = (
            getattr(mutator, "alias", None)
            or getattr(mutator, "name", None)
            or type(mutator).__name__
        )
        before_sequences = {
            name: protomer.mutated_sequence
            for name, protomer in global_config.protomers.items()
        }
        with event_span(
            logger=logger,
            component="mutator",
            alias=getattr(mutator, "alias", getattr(mutator, "name", mutator.__class__.__name__)),
            stage=_current_stage_name(),
            step=step,
            detail=getattr(mutator, "name", mutator.__class__.__name__),
        ) as span:
            mutator.mutate(
                global_config.mutable_protomers, evaluation_result, runtime_config
            )
            after_sequences = {
                name: protomer.mutated_sequence
                for name, protomer in global_config.protomers.items()
            }
            mutation_summary = summarize_sequence_changes(
                before_sequences,
                after_sequences,
            )
            span.finish(
                result=(
                    "changed"
                    if mutation_summary["changed_positions"] > 0
                    else "unchanged"
                ),
                **mutation_summary,
            )
        site_metrics_snapshot = snapshot_segment_mutation_site_metrics()
        telemetry = getattr(mutator, "last_mutation_telemetry", None)
        if telemetry is None or not isinstance(telemetry, dict):
            telemetry = {"metrics": {}}
        else:
            telemetry = dict(telemetry)
            telemetry["metrics"] = dict(telemetry.get("metrics", {}) or {})
        if site_metrics_snapshot:
            telemetry["site_metrics_by_proto"] = site_metrics_snapshot
        if secondary_structure_by_proto:
            telemetry["metrics"]["secondary_structure"] = "computed"
            telemetry["secondary_structure_by_proto"] = secondary_structure_by_proto
        else:
            telemetry["metrics"].setdefault("secondary_structure", "N/A")
        log_mutator_sequence_changes(
            mutator.name,
            before_sequences,
            after_sequences,
            telemetry=telemetry,
        )
        if pre_update and index < len(mutators) - 1:
            global_config.segment_manager.preapply_pending_mutations(
                max_entropy=max_entropy
            )


def _finalize_mutator_stack_pre_update(runtime_config: Dict, accepted: bool) -> None:
    if not bool(runtime_config.get("mutator_stack_pre_update", False)):
        return
    global_config.segment_manager.finish_pre_update_step(accepted=bool(accepted))


def _clone_eval_result(eval_result):
    if hasattr(eval_result, "clone"):
        return eval_result.clone()
    return eval_result


def _perform_stage_reevaluation(
    *,
    args,
    stage_manager,
    stage_unit,
    runtime_config: Dict,
    step: int,
    oligo: list[str],
    loss_computer: LossEngine | None = None,
):
    return perform_evaluation(
        args,
        stage_manager.resolve_stage_evaluator_instance(stage_unit),
        runtime_config,
        step,
        oligo,
        loss_computer=loss_computer,
    )


def _handle_transition_gate_stage_entry(
    *,
    args,
    step: int,
    decisions: Decisions,
    loss_computer: LossEngine,
    runtime_config: Dict,
    stage_manager,
    stage_entry,
):
    transition_plan = stage_entry.plan
    previous_stage_result = stage_entry.previous_stage_result
    transition_runtime = stage_entry.decision_runtime(runtime_config)
    target_stage_name = transition_plan.target_stage_name

    preserved_source = preserve_seed_transition_source(
        transition_plan,
        previous_stage_result=previous_stage_result,
        out_dir=global_config.out_dir,
        step=step,
    )
    if transition_plan.needs_stage_reevaluation:
        evaluation_result = perform_evaluation(
            args,
            stage_manager.resolve_transition_consistency_gate_evaluator_instance(
                target_stage_name
            ),
            transition_runtime,
            step,
            transition_plan.target_oligo,
        )
    else:
        evaluation_result = _clone_eval_result(previous_stage_result)

    gate_attempt = begin_seed_transition_gate_attempt(
        transition_runtime,
        stage_manager,
        transition_plan,
        previous_stage_result=preserved_source.previous_stage_result,
        gate_seed_scores=evaluation_result,
        artifact_root=preserved_source.artifact_root,
        step=step,
    )
    gate_result = gate_attempt.gate_result
    metadata = gate_attempt.metadata
    if gate_result.passed:
        if passes_stage_entry_acceptance_gate(
            eval_result=evaluation_result,
            runtime_config=transition_runtime,
            loss_computer=loss_computer,
        ):
            logging.info(
                "step=%s transition_consistency_gate seed_pass score=%.5f threshold=%.5f source_stage=%s target_stage=%s",
                step,
                float(getattr(gate_result, "score", 0.0)),
                float(getattr(gate_result, "threshold", 0.0)),
                metadata["source_stage"],
                metadata["target_stage"],
            )
            seed_decision = finish_seed_transition_gate_decision(
                transition_runtime,
                stage_manager,
                gate_attempt,
                evaluation_result=evaluation_result,
                is_accepted=True,
            )
            evaluation_result = resolve_seed_transition_gate_decision(
                transition_runtime,
                seed_decision,
                evaluate_target=lambda: _perform_stage_reevaluation(
                    args=args,
                    stage_manager=stage_manager,
                    stage_unit=transition_plan.target_stage_unit,
                    runtime_config=transition_runtime,
                    step=step,
                    oligo=transition_plan.target_oligo,
                    loss_computer=loss_computer,
                ),
                recompute_last_score=lambda target_result: decisions.recompute_last_score(
                    loss_computer,
                    transition_runtime,
                    target_result,
                ),
            )
        else:
            logging.info(
                "step=%s transition_consistency_gate seed_reject reason=acceptance_gate score=%.5f threshold=%.5f source_stage=%s target_stage=%s",
                step,
                float(getattr(gate_result, "score", 0.0)),
                float(getattr(gate_result, "threshold", 0.0)),
                metadata["source_stage"],
                metadata["target_stage"],
            )
            seed_decision = finish_seed_transition_gate_decision(
                transition_runtime,
                stage_manager,
                gate_attempt,
                evaluation_result=evaluation_result,
                is_accepted=False,
            )
            evaluation_result = seed_decision.evaluation_result
    else:
        seed_decision = finish_seed_transition_gate_decision(
            transition_runtime,
            stage_manager,
            gate_attempt,
            evaluation_result=evaluation_result,
            is_accepted=False,
        )
        evaluation_result = seed_decision.evaluation_result

    return evaluation_result


def _handle_direct_stage_entry(
    *,
    args,
    step: int,
    decisions: Decisions,
    loss_computer: LossEngine,
    runtime_config: Dict,
    stage_manager,
    stage_entry,
):
    transition_plan = stage_entry.plan
    previous_stage_result = stage_entry.previous_stage_result
    transition_runtime = stage_entry.decision_runtime(runtime_config)
    direct_attempt = begin_direct_stage_entry(
        stage_manager,
        transition_plan,
        previous_stage_result=previous_stage_result,
        evaluate_target=lambda: _perform_stage_reevaluation(
            args=args,
            stage_manager=stage_manager,
            stage_unit=transition_plan.target_stage_unit,
            runtime_config=runtime_config,
            step=step,
            oligo=transition_plan.target_oligo,
            loss_computer=loss_computer,
        ),
        clone_result=_clone_eval_result,
    )
    evaluation_result = direct_attempt.evaluation_result
    direct_decision = finish_direct_stage_entry_decision(
        stage_manager,
        direct_attempt,
        previous_stage_result=previous_stage_result,
        evaluation_result=evaluation_result,
        is_accepted=passes_stage_entry_acceptance_gate(
            eval_result=evaluation_result,
            runtime_config=transition_runtime,
            loss_computer=loss_computer,
        ),
    )
    return resolve_already_accepted_direct_stage_entry_decision(
        direct_decision,
        recompute_last_score=lambda result: decisions.recompute_last_score(
            loss_computer,
            transition_runtime,
            result,
        ),
    )


def _handle_stage_entry(
    *,
    args,
    step: int,
    evaluation_result,
    decisions: Decisions,
    loss_computer: LossEngine,
    runtime_config: Dict,
    stage_manager,
):
    transaction = begin_stage_entry_transaction(
        stage_manager,
        evaluation_result,
        loss_computer,
        clone_result=_clone_eval_result,
    )
    if transaction is None:
        return None

    if transaction.target_stage_has_gate:
        return _handle_transition_gate_stage_entry(
            args=args,
            step=step,
            decisions=decisions,
            loss_computer=loss_computer,
            runtime_config=runtime_config,
            stage_manager=stage_manager,
            stage_entry=transaction,
        )

    return _handle_direct_stage_entry(
        args=args,
        step=step,
        decisions=decisions,
        loss_computer=loss_computer,
        runtime_config=runtime_config,
        stage_manager=stage_manager,
        stage_entry=transaction,
    )


def run_optimization_loop(
    args,
    loss_computer: LossEngine,
    initial_evaluation_result,
    stage_manager,
    runtime_config: Dict,
) -> Any:
    """Execute the main optimization loop."""
    initial_evaluation_result = _prepare_structure_start_runtime(
        initial_evaluation_result,
        timing="step0",
        loss_computer=loss_computer,
        runtime_config=runtime_config,
    )
    decisions = Decisions.create(
        initial_evaluation_result, runtime_config, loss_computer.compute_loss
    )
    global_config.init_const("decisions", decisions)
    step0_reference_gate = runtime_config.get("reference_gate_result")
    if bool(getattr(step0_reference_gate, "gate_active", False)):
        if not bool(getattr(step0_reference_gate, "passed", False)):
            decisions.row_history[-1]["accepted"] = False
            decisions.row_history[-1]["note"] = "reference gate failed"
            decisions.dump_results(global_config.out_dir, incremental=True)
            raise ValueError(
                "structure_start step0 reference gate failed: "
                f"reference_tmscore={getattr(step0_reference_gate, 'score', None)} "
                f"threshold={getattr(step0_reference_gate, 'threshold', None)} "
                f"hotspot_score={getattr(step0_reference_gate, 'hotspot_score', None)} "
                f"hotspot_threshold={getattr(step0_reference_gate, 'hotspot_threshold', None)} "
                f"hotspot_passed={getattr(step0_reference_gate, 'hotspot_passed', None)}"
            )
        decisions.dump_results(global_config.out_dir, incremental=True)

    evaluation_result = initial_evaluation_result
    global_config.set_variable("step", 0)
    initial_stage_result = _handle_stage_entry(
        args=args,
        step=0,
        evaluation_result=evaluation_result,
        decisions=decisions,
        loss_computer=loss_computer,
        runtime_config=runtime_config,
        stage_manager=stage_manager,
    )
    if initial_stage_result is not None:
        evaluation_result = initial_stage_result
    runtime_config.pop("reference_gate_result", None)

    if int(args.steps) <= 1:
        decisions.dump_results(global_config.out_dir, incremental=False)

    step = 1
    while step < args.steps:
        clear_transition_gate_runtime(runtime_config)
        runtime_config.pop("reference_gate_result", None)
        logging.info(
            build_step_start_message(
                step=step,
                total_steps=args.steps,
                stage_name=getattr(stage_manager, "current_stage", None),
            )
        )
        global_config.set_variable("step", step)
        mutation_input_scores = (
            stage_manager.current_transition_input_scores
            if getattr(stage_manager, "transition_consistency_gate_context", None)
            is not None
            and getattr(stage_manager, "current_transition_input_scores", None) is not None
            else evaluation_result
        )
        # Mutation phase
        is_accepted = False
        try:
            execute_mutation_step(
                stage_manager.current_mutators, mutation_input_scores, runtime_config, step
            )

            # Evaluation phase
            evaluation_result = perform_evaluation(
                args,
                stage_manager.current_evaluator,
                runtime_config,
                step,
                stage_manager.current_oligo,
                loss_computer=loss_computer,
            )
            evaluation_result = _prepare_structure_start_runtime(
                evaluation_result,
                timing="pre_accept",
                loss_computer=loss_computer,
                runtime_config=runtime_config,
            )

            # Decision making
            gate_active = (
                getattr(stage_manager, "transition_consistency_gate_context", None)
                is not None
            )
            if gate_active:
                gate_attempt = begin_active_transition_gate_attempt(
                    runtime_config,
                    stage_manager,
                    evaluation_result,
                )
                is_early_stop, evaluation_result, is_accepted = (
                    decisions.make_mutation_decision(
                        step, evaluation_result, loss_computer, runtime_config
                    )
                )
                active_decision = finish_active_transition_gate_decision(
                    runtime_config,
                    stage_manager,
                    gate_attempt,
                    step=step,
                    evaluation_result=evaluation_result,
                    is_accepted=is_accepted,
                    decisions=decisions,
                )
                evaluation_result = resolve_active_transition_gate_decision(
                    active_decision,
                    evaluate_target=lambda: _perform_stage_reevaluation(
                        args=args,
                        stage_manager=stage_manager,
                        stage_unit=gate_attempt.plan.target_stage_unit,
                        runtime_config=runtime_config,
                        step=step,
                        oligo=gate_attempt.plan.target_oligo,
                        loss_computer=loss_computer,
                    ),
                    recompute_last_score=lambda target_result: decisions.recompute_last_score(
                        loss_computer,
                        runtime_config,
                        target_result,
                    ),
                    dump_results=lambda: decisions.dump_results(
                        global_config.out_dir,
                        incremental=False,
                    ),
                )
            else:
                is_early_stop, evaluation_result, is_accepted = (
                    decisions.make_mutation_decision(
                        step, evaluation_result, loss_computer, runtime_config
                    )
                )
                prefix_early_stop = bool(
                    (getattr(decisions, "scores_detail", {}) or {})
                    .get(step, {})
                    .get("prefix_early_stop_triggered", 0.0)
                )
                stage_entry_result = _handle_stage_entry(
                    args=args,
                    step=step,
                    evaluation_result=evaluation_result,
                    decisions=decisions,
                    loss_computer=loss_computer,
                    runtime_config=runtime_config,
                    stage_manager=stage_manager,
                )
                if stage_entry_result is not None:
                    evaluation_result = stage_entry_result
                    if (
                        getattr(stage_manager, "transition_consistency_gate_context", None)
                        is not None
                    ):
                        is_early_stop = prefix_early_stop
                    else:
                        scores_detail = getattr(decisions, "scores_detail", {})
                        latest_detail = scores_detail.get(
                            getattr(decisions, "last_accepted", None),
                            {},
                        )
                        is_early_stop = bool(
                            prefix_early_stop
                            or getattr(
                                loss_computer,
                                "is_early_stopping",
                                lambda _detail: False,
                            )(
                                latest_detail
                            )
                        )
        finally:
            _finalize_mutator_stack_pre_update(runtime_config, accepted=is_accepted)

        if is_early_stop:
            logging.info("Metrics have converged. Early stopping triggered.")
            logging.info(
                build_step_outcome_message(
                    step=step,
                    accepted=is_accepted,
                    early_stop=is_early_stop,
                    stage_name=getattr(stage_manager, "current_stage", None),
                )
            )
            break
        logging.info(
            build_step_outcome_message(
                step=step,
                accepted=is_accepted,
                early_stop=is_early_stop,
                stage_name=getattr(stage_manager, "current_stage", None),
            )
        )
        step += 1
    return evaluation_result


def _shutdown_pipeline_resources() -> None:
    """Release per-run resources before the next repeat starts."""
    try:
        mutators = global_config.get_const("mutators")
    except RuntimeError:
        mutators = {}
    for alias, mutator in (mutators or {}).items():
        shutdown = getattr(mutator, "shutdown", None)
        if not callable(shutdown):
            continue
        try:
            shutdown()
        except Exception:
            logging.exception("Failed to shutdown mutator '%s'.", alias)


def _finish_run_provenance(
    args, state: str, *, error: BaseException | None = None
) -> None:
    """Record the terminal run state; never raises, so it cannot mask pipeline errors."""
    if args is None:
        return
    run_config_lifecycle.finish_run_provenance(args, state, error=error)


def _run_pipeline(cfg) -> None:
    """Execute a single astrolabe run."""
    args = None
    succeeded = False
    try:
        args = parse_arguments(cfg)
        setup_logging(args.out_dir, getattr(args, "logging", None))
        logging.info("Starting protein astrolabe pipeline")

        # Initialize components
        loss_computer, stage_manager, inputs = initialize_pipeline_components(args)
        runtime_config = {}
        runtime_config["mutator_stack_pre_update"] = bool(
            getattr(getattr(args, "mutator", None), "stack_pre_update", False)
        )
        runtime_config["state_conflict_diagnostics"] = getattr(
            getattr(args, "mutator", None),
            "state_conflict_diagnostics",
            None,
        )
        max_entropy = getattr(
            getattr(args, "mutator", None),
            "stack_pre_update_max_entropy",
            None,
        )
        if max_entropy not in (None, "null", "None"):
            runtime_config["mutator_stack_pre_update_max_entropy"] = float(max_entropy)

        # Initial evaluation
        initial_result = load_or_evaluate_initial(
            args,
            stage_manager.current_evaluator,
            inputs,
            runtime_config,
            args.task_name,
            loss_computer=loss_computer,
        )

        # Run optimization
        run_optimization_loop(
            args,
            loss_computer,
            initial_result,
            stage_manager,
            runtime_config,
        )

        logging.info("Pipeline completed successfully")
        succeeded = True

    except Exception as exc:
        # Include full traceback in debug/info logs.
        logging.exception("Pipeline failed")
        _finish_run_provenance(args, "failed", error=exc)
        raise
    except BaseException as exc:
        # Interruptions (KeyboardInterrupt/SystemExit) must never be recorded as
        # completion; provenance keeps the failure visible for the run directory.
        _finish_run_provenance(args, "interrupted", error=exc)
        raise
    else:
        _finish_run_provenance(args, "completed")
    finally:
        _shutdown_pipeline_resources()
        _cleanup_auto_managed_raw_dir(args=args, succeeded=succeeded)


def _cleanup_auto_managed_raw_dir(*, args, succeeded: bool) -> None:
    if args is None:
        return
    if not getattr(args, "_raw_model_results_dir_auto_managed", False):
        return
    if not succeeded:
        # Keep auto-managed raw results on failure to aid debugging.
        logging.info(
            "Keeping auto-managed raw_model_results_dir due to failure: %s",
            getattr(args, "raw_model_results_dir", None),
        )
        return
    raw_dir = getattr(args, "raw_model_results_dir", None)
    if raw_dir is None:
        return
    raw_dir = pathlib.Path(str(raw_dir))

    marker = getattr(args, "_raw_model_results_dir_marker", None)
    if not marker:
        raise RawModelResultsCleanupError(
            f"Refusing to delete auto-managed raw_model_results_dir without marker: {raw_dir}"
        )

    marker_path = pathlib.Path(str(marker))
    try:
        marker_exists = marker_path.exists()
    except OSError as exc:
        raise RawModelResultsCleanupError(
            f"Failed to verify cleanup marker for auto-managed raw_model_results_dir: {raw_dir}"
        ) from exc
    if not marker_exists:
        raise RawModelResultsCleanupError(
            f"Refusing to delete auto-managed raw_model_results_dir without marker: {raw_dir}"
        )

    try:
        shutil.rmtree(raw_dir)
        logging.info("Deleted auto-managed raw_model_results_dir: %s", raw_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RawModelResultsCleanupError(
            f"Failed to delete auto-managed raw_model_results_dir: {raw_dir}"
        ) from exc



def run_repeats(
    base_cfg: Any,
    *,
    on_stop_requested: Callable[[int, int], None] | None = None,
) -> None:
    """Run `_run_pipeline` for configured repeats with unified seed/reset semantics."""
    base_obj = OmegaConf.to_container(base_cfg, resolve=False)
    lifecycle_cfg = OmegaConf.create(base_obj)

    # Resolve only fields required to schedule the lifecycle. Per-repeat configs
    # continue to be cloned from the unresolved base object below.
    repeats_raw = lifecycle_cfg.get("repeats", 1)
    repeats_is_decimal_string = False
    if type(repeats_raw) is str:
        digits = repeats_raw[1:] if repeats_raw[:1] in "+-" else repeats_raw
        repeats_is_decimal_string = (
            bool(digits) and digits.isascii() and digits.isdecimal()
        )
    if type(repeats_raw) is not int and not repeats_is_decimal_string:
        raise ValueError(
            f"Invalid repeats={repeats_raw!r}; expected a positive decimal integer."
        )
    repeats = int(repeats_raw)
    if repeats <= 0:
        raise ValueError(
            f"Invalid repeats={repeats_raw!r}; expected a positive decimal integer."
        )

    raw_seed = lifecycle_cfg.get("seed")
    base_seed = None
    if raw_seed is not None:
        seed_is_decimal_string = False
        if type(raw_seed) is str:
            digits = raw_seed[1:] if raw_seed[:1] in "+-" else raw_seed
            seed_is_decimal_string = (
                bool(digits) and digits.isascii() and digits.isdecimal()
            )
        if type(raw_seed) is not int and not seed_is_decimal_string:
            raise ValueError(
                f"Invalid seed value {raw_seed!r}; expected null or a decimal integer."
            )
        base_seed = int(raw_seed)
        last_seed = base_seed + repeats - 1
        if base_seed < 1 or last_seed > 2**32 - 1:
            raise ValueError(
                "Invalid seed sequence: explicit seeds must stay within "
                f"1..{2**32 - 1}, got {base_seed}..{last_seed}."
            )

    install_handlers()

    for repeat_idx in range(repeats):
        run_cfg = OmegaConf.create(base_obj)
        run_cfg.repeats = repeats
        run_cfg.repeat_index = repeat_idx
        if base_seed is None:
            task_name = str(run_cfg.get("task_name") or "")
            digest = hashlib.sha256(
                f"{task_name}:{repeat_idx}".encode("utf-8")
            ).hexdigest()
            run_cfg.seed = (int(digest[:8], 16) % 1_000_000) + 1
        else:
            run_cfg.seed = base_seed + repeat_idx

        raw_dir = run_cfg.get("raw_model_results_dir")
        if raw_dir not in (None, "", "null", "None"):
            run_cfg.raw_model_results_dir = (
                pathlib.Path(str(raw_dir)).expanduser()
                / f"repeat_{repeat_idx + 1:03d}"
            )

        global_config.reset()
        print(f"=== Repeat {repeat_idx + 1}/{repeats} ===")
        _run_pipeline(run_cfg)
        if stop_requested():
            if on_stop_requested is None:
                logging.warning(
                    "Graceful stop requested; exiting after repeat %d/%d.",
                    repeat_idx + 1,
                    repeats,
                )
            else:
                on_stop_requested(repeat_idx + 1, repeats)
            break


@hydra.main(version_base=None)
def main(args) -> None:
    """Main entry point for the protein astrolabe pipeline."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    run_repeats(args)


if __name__ == "__main__":
    print(
        "[DEPRECATED] `python run.py` is a compatibility path. Use `gh run ...` instead.",
        file=os.sys.stderr,
    )
    main()
