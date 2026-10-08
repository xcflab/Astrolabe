from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.utils.mmcif_ids import int_id_to_str_id

logger = logging.getLogger(__name__)


PerResidueMetricFn = Callable[[Any, "PerResidueMetricContext"], np.ndarray]


@dataclass(frozen=True)
class PerResidueMetricContext:
    name: str
    oligo_name: str
    config: dict[str, Any]


class PerResidueScoreRegistry:
    """Registry for pose-wide per-residue metrics.

    Registered metric functions return one value per pose residue. The caller is
    responsible for mapping pose chains back to configured protomers.
    """

    _registry: dict[str, PerResidueMetricFn] = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(func: PerResidueMetricFn):
            for name in names:
                cls._registry[_normalize_metric_type(name)] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> PerResidueMetricFn:
        key = _normalize_metric_type(name)
        try:
            return cls._registry[key]
        except KeyError as exc:
            available = ", ".join(cls.list_available()) or "<none>"
            raise KeyError(
                f"Unknown per-residue score metric {name!r}. Available metrics: {available}"
            ) from exc

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._registry)


def _normalize_metric_type(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _to_plain_config(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _config_mapping(value: Any) -> dict[str, Any]:
    value = _to_plain_config(value)
    return dict(value) if isinstance(value, Mapping) else {}


def per_residue_scores_enabled(config: Any) -> bool:
    cfg = _config_mapping(config)
    return bool(cfg.get("enabled", False))


def _existing_metric_type(metric_type: str) -> bool:
    return _normalize_metric_type(metric_type) in {"plddt", "existing", "legacy"}


def per_residue_scores_requires_complex_poses(config: Any) -> bool:
    cfg = _config_mapping(config)
    if not per_residue_scores_enabled(cfg):
        return False
    terms_cfg = _config_mapping(cfg.get("terms"))
    for alias, raw_term_conf in terms_cfg.items():
        term_conf = _config_mapping(raw_term_conf)
        metric_type = str(term_conf.get("_type") or alias)
        if not _existing_metric_type(metric_type):
            return True
    return False


def _score_config(runtime_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if runtime_config is not None and "per_residue_scores" in runtime_config:
        return _config_mapping(runtime_config.get("per_residue_scores"))
    try:
        return _config_mapping(getattr(global_config, "per_residue_scores", None))
    except RuntimeError:
        return {}


def _pose_total_residue(pose) -> int:
    total = getattr(pose, "total_residue", None)
    if callable(total):
        return int(total())
    if total is not None:
        return int(total)
    return int(pose.size())


def _vector1_to_array(values, total_residue: int) -> np.ndarray:
    return np.asarray([float(values[idx]) for idx in range(1, total_residue + 1)], dtype=float)


def _mapping_to_array(values, total_residue: int) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=float)
        if arr.shape == (total_residue,):
            return arr
    out = np.zeros(total_residue, dtype=float)
    for idx in range(1, total_residue + 1):
        if hasattr(values, "get"):
            val = values.get(idx, 0.0)
        else:
            val = values[idx]
        out[idx - 1] = float(val)
    return out


def _chain2proto_for_oligo(oligo_name: str) -> dict[str, str]:
    return {
        int_id_to_str_id(i + 1): proto_name
        for i, proto_name in enumerate(str(oligo_name))
        if proto_name in global_config.protomers
    }


def _split_pose_metric_by_proto(
    pose,
    values: np.ndarray,
    *,
    oligo_name: str,
) -> dict[str, list[np.ndarray]]:
    chain2proto = _chain2proto_for_oligo(oligo_name)
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    values = np.asarray(values, dtype=float)
    for chain_index in range(1, pose.num_chains() + 1):
        chain_id = int_id_to_str_id(chain_index)
        proto_name = chain2proto.get(chain_id)
        protomer = global_config.protomers.get(proto_name)
        if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
            continue
        begin = int(pose.chain_begin(chain_index)) - 1
        end = int(pose.chain_end(chain_index))
        arr = np.asarray(values[begin:end], dtype=float)
        if arr.shape[-1] != int(protomer.length):
            raise ValueError(
                f"Per-residue score length mismatch for protomer {proto_name!r} in oligo {oligo_name!r}: "
                f"got {arr.shape[-1]}, expected {protomer.length}."
            )
        out[str(proto_name)].append(arr)
    return out


def _per_residue_sasa(pose, *, sidechain_only: bool = False) -> np.ndarray:
    total = _pose_total_residue(pose)
    if sidechain_only:
        from pyrosetta.rosetta.core.scoring.sasa import per_res_sc_sasa  # type: ignore

        return _vector1_to_array(per_res_sc_sasa(pose), total)

    from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import (  # type: ignore
        PerResidueSasaMetric,
    )

    return _mapping_to_array(PerResidueSasaMetric().calculate(pose), total)


@PerResidueScoreRegistry.register("sasa", "rosetta_sasa")
def _metric_sasa(pose, context: PerResidueMetricContext) -> np.ndarray:
    return _per_residue_sasa(
        pose,
        sidechain_only=bool(context.config.get("sidechain_only", False)),
    )


@PerResidueScoreRegistry.register("sidechain_sasa", "sc_sasa", "rosetta_sidechain_sasa")
def _metric_sidechain_sasa(pose, context: PerResidueMetricContext) -> np.ndarray:
    return _per_residue_sasa(pose, sidechain_only=True)


@PerResidueScoreRegistry.register("buried_sasa", "delta_sasa", "rosetta_buried_sasa")
def _metric_buried_sasa(pose, context: PerResidueMetricContext) -> np.ndarray:
    bound = _per_residue_sasa(
        pose,
        sidechain_only=bool(context.config.get("sidechain_only", False)),
    )
    unbound_parts = [
        _per_residue_sasa(
            chain_pose,
            sidechain_only=bool(context.config.get("sidechain_only", False)),
        )
        for chain_pose in pose.split_by_chain()
    ]
    unbound = np.concatenate(unbound_parts) if unbound_parts else np.zeros_like(bound)
    if bound.shape != unbound.shape:
        raise ValueError(
            f"Bound/unbound SASA vectors do not align for {context.name!r}: "
            f"{bound.shape} vs {unbound.shape}."
        )
    return np.maximum(0.0, unbound - bound)


@PerResidueScoreRegistry.register("sap", "rosetta_sap")
def _metric_sap(pose, context: PerResidueMetricContext) -> np.ndarray:
    from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import (  # type: ignore
        PerResidueSapScoreMetric,
    )

    return _mapping_to_array(PerResidueSapScoreMetric().calculate(pose), _pose_total_residue(pose))


@PerResidueScoreRegistry.register("buried_unsat", "buns", "buried_unsatisfied", "rosetta_buns")
def _metric_buried_unsat(pose, context: PerResidueMetricContext) -> np.ndarray:
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector  # type: ignore
    from pyrosetta.rosetta.protocols.simple_filters import BuriedUnsatHbondFilter  # type: ignore

    total = _pose_total_residue(pose)
    out = np.zeros(total, dtype=float)
    ignore_surface_res = bool(context.config.get("ignore_surface_res", False))
    for idx in range(1, total + 1):
        residue = pose.residue(idx)
        if not residue.is_protein():
            continue
        filt = BuriedUnsatHbondFilter()
        filt.set_report_all_heavy_atom_unsats(True)
        filt.set_ignore_surface_res(ignore_surface_res)
        filt.set_residue_selector(ResidueIndexSelector(str(idx)))
        out[idx - 1] = float(filt.report_sm(pose))
    return out


@PerResidueScoreRegistry.register("holes", "rosetta_holes")
def _metric_holes(pose, context: PerResidueMetricContext) -> np.ndarray:
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector  # type: ignore
    from pyrosetta.rosetta.protocols.simple_filters import HolesFilter  # type: ignore

    total = _pose_total_residue(pose)
    out = np.zeros(total, dtype=float)
    for idx in range(1, total + 1):
        residue = pose.residue(idx)
        if not residue.is_protein():
            continue
        holes = HolesFilter()
        holes.set_residue_selector(ResidueIndexSelector(str(idx)))
        out[idx - 1] = float(holes.report_sm(pose))
    return out


@PerResidueScoreRegistry.register("packstat", "residue_packstat", "rosetta_packstat")
def _metric_packstat(pose, context: PerResidueMetricContext) -> np.ndarray:
    from pyrosetta.rosetta.core.scoring import packstat  # type: ignore

    oversample = int(context.config.get("oversample", 1) or 1)
    return _vector1_to_array(
        packstat.compute_residue_packing_scores(pose, oversample),
        _pose_total_residue(pose),
    )


def _normalized_terms_from_scores(eval_result) -> dict[str, dict[str, list[np.ndarray]]]:
    terms: dict[str, dict[str, list[np.ndarray]]] = {}
    existing_terms = getattr(eval_result, "per_proto_score_terms", None) or {}
    for proto_name, metric_map in existing_terms.items():
        if not isinstance(metric_map, Mapping):
            continue
        terms[str(proto_name)] = {
            str(metric_name): [np.asarray(v, dtype=float) for v in (values if isinstance(values, list) else [values])]
            for metric_name, values in metric_map.items()
        }

    for proto_name, values in (getattr(eval_result, "per_proto_scores", None) or {}).items():
        wrapped = values if isinstance(values, list) else [values]
        terms.setdefault(str(proto_name), {}).setdefault(
            "plddt",
            [np.asarray(v, dtype=float) for v in wrapped],
        )
    return terms


def attach_per_residue_scores(eval_result, config: Any):
    cfg = _config_mapping(config)
    if not per_residue_scores_enabled(cfg):
        return eval_result

    terms_cfg = _config_mapping(cfg.get("terms"))
    if "plddt" in _score_term_specs(cfg):
        terms_cfg.setdefault("plddt", {"_type": "plddt"})
    if not terms_cfg:
        return eval_result

    score_terms = _normalized_terms_from_scores(eval_result)
    pose_terms = {}
    for alias, raw_term_conf in terms_cfg.items():
        term_conf = _config_mapping(raw_term_conf)
        metric_type = str(term_conf.get("_type") or alias)
        if _existing_metric_type(metric_type):
            continue
        pose_terms[str(alias)] = (metric_type, term_conf)

    if not pose_terms:
        eval_result.per_proto_score_terms = score_terms
        return eval_result

    complex_poses = getattr(eval_result, "complex_poses", None)
    poses = getattr(complex_poses, "poses", None)
    if not isinstance(poses, Mapping) or not poses:
        raise ValueError(
            "Rosetta per_residue_scores terms require evaluation_result.complex_poses."
        )

    for alias, (metric_type, term_conf) in pose_terms.items():
        metric_fn = PerResidueScoreRegistry.get(metric_type)
        for oligo_name in getattr(eval_result, "names", []) or []:
            pose = poses.get(str(oligo_name))
            if pose is None:
                continue
            context = PerResidueMetricContext(
                name=str(alias),
                oligo_name=str(oligo_name),
                config=term_conf,
            )
            values = np.asarray(metric_fn(pose, context), dtype=float)
            if values.shape != (_pose_total_residue(pose),):
                raise ValueError(
                    f"Metric {alias!r} returned shape {values.shape}; expected one value per pose residue."
                )
            for proto_name, arrays in _split_pose_metric_by_proto(
                pose,
                values,
                oligo_name=str(oligo_name),
            ).items():
                score_terms.setdefault(proto_name, {}).setdefault(str(alias), []).extend(
                    list(arrays)
                )

    eval_result.per_proto_score_terms = score_terms
    return eval_result


def _score_term_specs(config: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("mutator_score_terms", "score_terms"):
        value = config.get(key)
        if isinstance(_to_plain_config(value), Mapping):
            return _config_mapping(value)
    mutator_score = _config_mapping(config.get("mutator_score"))
    terms = _config_mapping(mutator_score.get("terms"))
    return terms


def _term_transform(term_name: str, raw_spec: Any, config: Mapping[str, Any]) -> tuple[float, float, float, float | None, float | None]:
    terms_cfg = _config_mapping(config.get("terms"))
    term_cfg = _config_mapping(terms_cfg.get(term_name))
    spec_cfg = _config_mapping(raw_spec)
    if isinstance(raw_spec, (int, float)):
        spec_cfg["weight"] = float(raw_spec)
    weight = float(spec_cfg.get("weight", term_cfg.get("weight", 1.0)))
    scale = float(spec_cfg.get("scale", term_cfg.get("scale", 1.0)) or 1.0)
    offset = float(spec_cfg.get("offset", term_cfg.get("offset", 0.0)) or 0.0)
    clip_min = spec_cfg.get("clip_min", term_cfg.get("clip_min"))
    clip_max = spec_cfg.get("clip_max", term_cfg.get("clip_max"))
    return (
        weight,
        scale,
        offset,
        None if clip_min is None else float(clip_min),
        None if clip_max is None else float(clip_max),
    )


def combine_per_proto_mutator_scores(
    eval_result,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray] | None:
    cfg = _score_config(runtime_config)
    score_specs = _score_term_specs(cfg)
    if not score_specs:
        return None

    terms = _normalized_terms_from_scores(eval_result)
    combined: dict[str, np.ndarray] = {}
    allow_missing = bool(cfg.get("allow_missing_mutator_score_terms", False))

    for proto_name, metric_map in terms.items():
        parts: list[np.ndarray] = []
        for term_name, raw_spec in score_specs.items():
            arrays = metric_map.get(str(term_name))
            if not arrays:
                if allow_missing:
                    logger.warning(
                        "Missing per-residue mutator score term %s for protomer %s.",
                        term_name,
                        proto_name,
                    )
                    continue
                raise KeyError(
                    f"Missing per-residue mutator score term {term_name!r} for protomer {proto_name!r}."
                )
            stacked = np.stack([np.asarray(arr, dtype=float) for arr in arrays], axis=0)
            arr = np.mean(stacked, axis=0)
            weight, scale, offset, clip_min, clip_max = _term_transform(
                str(term_name),
                raw_spec,
                cfg,
            )
            arr = (arr - offset) / scale
            if clip_min is not None or clip_max is not None:
                arr = np.clip(
                    arr,
                    -np.inf if clip_min is None else clip_min,
                    np.inf if clip_max is None else clip_max,
                )
            parts.append(float(weight) * arr)
        if parts:
            combined[str(proto_name)] = np.sum(np.stack(parts, axis=0), axis=0)
    return combined or None


def _selected_positions_from_masks(
    selected_masks: Mapping[str, Any],
) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for proto_name, raw_mask in (selected_masks or {}).items():
        try:
            mask = np.asarray(raw_mask, dtype=bool)
        except Exception:
            continue
        if mask.ndim != 1:
            continue
        positions[str(proto_name)] = [int(idx) for idx in np.where(mask)[0].tolist()]
    return positions


def _valid_positions(positions: list[int], length: int) -> list[int]:
    return [int(pos) for pos in positions if 0 <= int(pos) < int(length)]


def summarize_selected_per_residue_scores(
    eval_result,
    selected_positions: Mapping[str, list[int]],
    *,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize per-residue score terms over a fixed set of protomer positions."""
    terms = _normalized_terms_from_scores(eval_result)
    combined = combine_per_proto_mutator_scores(eval_result, runtime_config)
    if combined is not None:
        for proto_name, arr in combined.items():
            terms.setdefault(str(proto_name), {})["combined"] = [np.asarray(arr, dtype=float)]

    per_proto: dict[str, dict[str, Any]] = {}
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    total_count = 0

    for proto_name, raw_positions in (selected_positions or {}).items():
        metric_map = terms.get(str(proto_name))
        if not metric_map:
            continue
        positions = [int(pos) for pos in raw_positions]
        proto_terms: dict[str, float] = {}
        proto_count = len(positions)
        total_count += proto_count
        for term_name, arrays in metric_map.items():
            wrapped = arrays if isinstance(arrays, list) else [arrays]
            valid_arrays = []
            for arr_like in wrapped:
                try:
                    arr = np.asarray(arr_like, dtype=float)
                except Exception:
                    continue
                if arr.ndim == 0:
                    continue
                valid_arrays.append(arr)
            if not valid_arrays:
                continue
            stacked = np.stack(valid_arrays, axis=0)
            arr = np.mean(stacked, axis=0)
            valid = _valid_positions(positions, int(arr.shape[-1]))
            if not valid:
                continue
            values = np.asarray(arr[valid], dtype=float)
            if values.size == 0:
                continue
            proto_terms[str(term_name)] = float(np.mean(values))
            aggregate_values[str(term_name)].extend(float(v) for v in values.tolist())
        per_proto[str(proto_name)] = {
            "selected_count": int(proto_count),
            "terms": proto_terms,
        }

    return {
        "selected_count": int(total_count),
        "terms": {
            term_name: float(np.mean(values))
            for term_name, values in aggregate_values.items()
            if values
        },
        "protomers": per_proto,
    }


def build_mutator_selector_score_context(
    *,
    mutator_alias: str,
    selector_expression: Any,
    evaluation_result,
    selected_masks: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    positions = _selected_positions_from_masks(selected_masks)
    return {
        "mutator": str(mutator_alias),
        "selector": "" if selector_expression is None else str(selector_expression),
        "positions": positions,
        "before": summarize_selected_per_residue_scores(
            evaluation_result,
            positions,
            runtime_config=runtime_config,
        ),
    }


def _safe_detail_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return label or "unknown"


def _add_before_after_delta(
    details: dict[str, float],
    *,
    prefix: str,
    term_name: str,
    before_terms: Mapping[str, Any],
    after_terms: Mapping[str, Any],
) -> None:
    before_value = before_terms.get(term_name)
    after_value = after_terms.get(term_name)
    if before_value is not None:
        details[f"{prefix}_{term_name}_before_mean"] = float(before_value)
    if after_value is not None:
        details[f"{prefix}_{term_name}_after_mean"] = float(after_value)
    if before_value is not None and after_value is not None:
        details[f"{prefix}_{term_name}_delta"] = float(after_value) - float(before_value)


def mutator_selector_score_delta_details(
    eval_result,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return score details comparing fixed selector positions before/after mutation."""
    contexts = []
    if runtime_config is not None:
        raw_contexts = runtime_config.get("mutator_selector_score_contexts", [])
        if isinstance(raw_contexts, list):
            contexts = raw_contexts
    details: dict[str, float] = {}
    for context in contexts:
        if not isinstance(context, Mapping):
            continue
        positions = context.get("positions")
        if not isinstance(positions, Mapping):
            continue
        mutator_label = _safe_detail_label(str(context.get("mutator") or "mutator"))
        before = context.get("before") if isinstance(context.get("before"), Mapping) else {}
        after = summarize_selected_per_residue_scores(
            eval_result,
            positions,
            runtime_config=runtime_config,
        )
        base_prefix = f"mutator_selector_{mutator_label}"
        details[f"{base_prefix}_selected_count"] = float(
            before.get("selected_count", after.get("selected_count", 0)) or 0
        )
        before_terms = before.get("terms") if isinstance(before.get("terms"), Mapping) else {}
        after_terms = after.get("terms") if isinstance(after.get("terms"), Mapping) else {}
        for term_name in sorted(set(before_terms) | set(after_terms)):
            _add_before_after_delta(
                details,
                prefix=base_prefix,
                term_name=str(term_name),
                before_terms=before_terms,
                after_terms=after_terms,
            )

        before_protos = before.get("protomers") if isinstance(before.get("protomers"), Mapping) else {}
        after_protos = after.get("protomers") if isinstance(after.get("protomers"), Mapping) else {}
        for proto_name in sorted(set(before_protos) | set(after_protos)):
            before_proto = before_protos.get(proto_name) if isinstance(before_protos.get(proto_name), Mapping) else {}
            after_proto = after_protos.get(proto_name) if isinstance(after_protos.get(proto_name), Mapping) else {}
            proto_prefix = f"{base_prefix}_p{_safe_detail_label(str(proto_name))}"
            count = before_proto.get("selected_count", after_proto.get("selected_count", 0))
            details[f"{proto_prefix}_selected_count"] = float(count or 0)
            before_proto_terms = before_proto.get("terms") if isinstance(before_proto.get("terms"), Mapping) else {}
            after_proto_terms = after_proto.get("terms") if isinstance(after_proto.get("terms"), Mapping) else {}
            for term_name in sorted(set(before_proto_terms) | set(after_proto_terms)):
                _add_before_after_delta(
                    details,
                    prefix=proto_prefix,
                    term_name=str(term_name),
                    before_terms=before_proto_terms,
                    after_terms=after_proto_terms,
                )
    return details


def mutator_site_score(
    eval_result,
    *,
    proto_name: str,
    site_index: int,
    runtime_config: Mapping[str, Any] | None = None,
) -> float:
    combined = combine_per_proto_mutator_scores(eval_result, runtime_config)
    if combined is not None and proto_name in combined:
        arr = np.asarray(combined[proto_name], dtype=float)
        if 0 <= int(site_index) < arr.shape[-1]:
            return float(arr[int(site_index)])

    per_proto_scores = getattr(eval_result, "per_proto_scores", None) or {}
    proto_scores = per_proto_scores.get(proto_name)
    if proto_scores is None:
        return 0.0
    score_array = np.asarray(proto_scores)
    if score_array.size == 0:
        return 0.0
    if score_array.ndim > 1:
        score_array = np.mean(score_array, axis=0)
    if score_array.ndim == 0:
        return float(score_array)
    if site_index < 0 or site_index >= int(score_array.shape[0]):
        return 0.0
    try:
        return float(score_array[site_index])
    except Exception:
        return 0.0
