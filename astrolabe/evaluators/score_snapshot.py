from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from astrolabe.score_contributions import merge_score_contribution_snapshot


DIRECT_SCORE_FIELDS = ("plddt", "ptm", "iptm", "pde")
DIRECT_SCORE_NORMALIZERS = {"pde": 20.0}


def safe_metric_mean(values: Any) -> float | None:
    if values is None:
        return None
    wrapped = values if isinstance(values, list) else [values]
    per_item_means: list[float] = []
    for item in wrapped:
        try:
            arr = np.asarray(item, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        try:
            per_item_means.append(float(np.mean(arr)))
        except Exception:
            continue
    if not per_item_means:
        return None
    return float(np.mean(per_item_means))


def _copy_mapping_like(value: dict | None) -> dict:
    if value is None:
        return {}
    try:
        return type(value)(value)
    except Exception:
        return dict(value)


def deep_fill_missing(base: dict, defaults: dict) -> dict:
    merged = _copy_mapping_like(base)
    for key, value in (defaults or {}).items():
        if isinstance(value, dict):
            existing = merged.get(key)
            if isinstance(existing, dict):
                merged[key] = deep_fill_missing(existing, value)
            elif key not in merged:
                merged[key] = _copy_mapping_like(value)
        elif key not in merged:
            merged[key] = value
    return merged


def flatten_score_details(values: dict | None, *, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in dict(values or {}).items():
        key_str = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_score_details(value, prefix=key_str))
            continue
        flat[key_str] = value
    return flat


def _add_metric_value(
    *,
    metric_name: str,
    oligo_name: str,
    value: float | None,
    per_oligo_snapshot: dict[str, dict[str, float]],
    aggregate_values: dict[str, list[float]],
) -> None:
    if value is None:
        return
    val = float(value)
    per_oligo_snapshot.setdefault(str(oligo_name), {})[metric_name] = val
    aggregate_values[metric_name].append(val)


def _mean_square_matrix_metric(values: Any, *, mask_kind: str | None) -> float | None:
    mats = values if isinstance(values, list) else [values]
    per_matrix_values: list[float] = []
    for matrix_like in mats:
        try:
            matrix = np.asarray(matrix_like, dtype=float)
        except Exception:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            continue
        if mask_kind == "cross":
            if matrix.shape[0] < 2:
                continue
            mask = ~np.eye(int(matrix.shape[0]), dtype=bool)
            values_arr = matrix[mask]
        elif mask_kind == "intra":
            mask = np.eye(int(matrix.shape[0]), dtype=bool)
            values_arr = matrix[mask]
        else:
            values_arr = matrix.reshape(-1)
        if values_arr.size == 0:
            continue
        per_matrix_values.append(float(np.mean(values_arr)))
    if not per_matrix_values:
        return None
    return float(np.mean(per_matrix_values))


def _chain_pair_metric_names(raw_name: str) -> tuple[str, str] | None:
    if not raw_name.startswith("chain_pair_"):
        return None
    suffix = raw_name.removeprefix("chain_pair_")
    if not suffix:
        return None
    return f"cross_{suffix}", f"intra_{suffix}"


def _add_chain_pair_snapshot(
    eval_result,
    *,
    per_oligo_snapshot: dict[str, dict[str, float]],
    aggregate_values: dict[str, list[float]],
) -> None:
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None) or {}
    if not isinstance(chain_pair_scores, dict):
        return

    for oligo_name in getattr(eval_result, "names", []) or []:
        metric_map = chain_pair_scores.get(str(oligo_name)) or {}
        if not isinstance(metric_map, dict):
            continue
        for raw_name, raw_values in metric_map.items():
            direct_mean = safe_metric_mean(raw_values)
            _add_metric_value(
                metric_name=str(raw_name),
                oligo_name=str(oligo_name),
                value=direct_mean,
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )

            pair_names = _chain_pair_metric_names(str(raw_name))
            if pair_names is None:
                continue
            cross_name, intra_name = pair_names
            _add_metric_value(
                metric_name=cross_name,
                oligo_name=str(oligo_name),
                value=_mean_square_matrix_metric(raw_values, mask_kind="cross"),
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )
            _add_metric_value(
                metric_name=intra_name,
                oligo_name=str(oligo_name),
                value=_mean_square_matrix_metric(raw_values, mask_kind="intra"),
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )

        lis_value = _mean_square_matrix_metric(
            metric_map.get("chain_pair_lis"),
            mask_kind="cross",
        )
        _add_metric_value(
            metric_name="lis",
            oligo_name=str(oligo_name),
            value=lis_value,
            per_oligo_snapshot=per_oligo_snapshot,
            aggregate_values=aggregate_values,
        )


def _add_direct_score_snapshot(
    eval_result,
    *,
    per_oligo_snapshot: dict[str, dict[str, float]],
    aggregate_values: dict[str, list[float]],
) -> None:
    for metric_name in DIRECT_SCORE_FIELDS:
        metric_map = getattr(eval_result, metric_name, None) or {}
        if not isinstance(metric_map, dict):
            continue
        for oligo_name in getattr(eval_result, "names", []) or []:
            value = safe_metric_mean(metric_map.get(oligo_name))
            normalizer = DIRECT_SCORE_NORMALIZERS.get(metric_name)
            if value is not None and normalizer:
                value = float(value) / float(normalizer)
            _add_metric_value(
                metric_name=metric_name,
                oligo_name=str(oligo_name),
                value=value,
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )


def _add_per_residue_score_term_snapshot(
    eval_result,
    *,
    aggregate_values: dict[str, list[float]],
) -> None:
    terms = getattr(eval_result, "per_proto_score_terms", None) or {}
    if not isinstance(terms, dict):
        return
    for proto_name, metric_map in terms.items():
        if not isinstance(metric_map, dict):
            continue
        for metric_name, values in metric_map.items():
            value = safe_metric_mean(values)
            if value is None:
                continue
            aggregate_values[f"per_residue_{metric_name}_mean"].append(float(value))
            aggregate_values[f"p{proto_name}_per_residue_{metric_name}_mean"].append(float(value))


def _add_derived_pae_snapshot(
    eval_result,
    *,
    per_oligo_snapshot: dict[str, dict[str, float]],
    aggregate_values: dict[str, list[float]],
) -> None:
    try:
        from astrolabe.core.chain_resolution import (
            chain_ids_for_oligo,
            resolve_selector_chain_ids,
        )
        from astrolabe.losses.confidence import (
            _infer_binder_selector_for_oligo,
            ipsae as compute_ipsae,
            pae_min as compute_pae_min,
        )
        from astrolabe.losses.interaction import compute_interface_pae
    except Exception:
        return

    for oligo_name in getattr(eval_result, "names", []) or []:
        chain_ids = chain_ids_for_oligo(str(oligo_name))
        if len(chain_ids) < 2:
            continue

        # Two-chain metrics are unambiguous.  For larger complexes, derive a
        # binder-target snapshot only when the binding-site config identifies
        # the binder; never guess from chain order.
        binder_selector = _infer_binder_selector_for_oligo(str(oligo_name))
        if binder_selector:
            binder_ids = resolve_selector_chain_ids(
                str(oligo_name), str(binder_selector)
            )
        elif len(chain_ids) == 2:
            binder_ids = [str(chain_ids[0])]
        else:
            continue
        binder_ids = [chain_id for chain_id in binder_ids if chain_id in chain_ids]
        if not binder_ids:
            continue
        target_ids = [chain_id for chain_id in chain_ids if chain_id not in binder_ids]
        if not target_ids:
            continue
        target_selector = ",".join(target_ids)
        binder_selector = ",".join(binder_ids)

        try:
            filtered = eval_result.filter_by_oligomer([str(oligo_name)])
        except Exception:
            continue
        try:
            for metric_name, symmetric, reduction in (
                ("ipae", "mean", "mean"),
                ("ipae_min", "min", "mean"),
                ("ipae_minimal", "min", "minimal"),
                ("ipae_max", "max", "mean"),
            ):
                metric_scores = compute_interface_pae(
                    filtered,
                    symmetric=symmetric,
                    reduction=reduction,
                    chain_a=binder_selector,
                    chain_b=target_selector,
                )
                _add_metric_value(
                    metric_name=metric_name,
                    oligo_name=str(oligo_name),
                    value=metric_scores.get(str(oligo_name)),
                    per_oligo_snapshot=per_oligo_snapshot,
                    aggregate_values=aggregate_values,
                )
        except Exception:
            pass
        try:
            ipsae_scores = compute_ipsae(
                filtered,
                chain_a=binder_selector,
                chain_b=target_selector,
            )
            _add_metric_value(
                metric_name="ipsae",
                oligo_name=str(oligo_name),
                value=ipsae_scores.get(str(oligo_name)),
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )
        except Exception:
            pass
        try:
            pae_min_scores = compute_pae_min(
                filtered,
                chain_a=binder_selector,
                chain_b=target_selector,
            )
            _add_metric_value(
                metric_name="pae_min",
                oligo_name=str(oligo_name),
                value=pae_min_scores.get(str(oligo_name)),
                per_oligo_snapshot=per_oligo_snapshot,
                aggregate_values=aggregate_values,
            )
        except Exception:
            pass


def build_standard_score_snapshot(eval_result) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    per_oligo_snapshot: dict[str, dict[str, float]] = {}
    aggregate_values: dict[str, list[float]] = defaultdict(list)

    _add_direct_score_snapshot(
        eval_result,
        per_oligo_snapshot=per_oligo_snapshot,
        aggregate_values=aggregate_values,
    )
    _add_per_residue_score_term_snapshot(
        eval_result,
        aggregate_values=aggregate_values,
    )
    _add_chain_pair_snapshot(
        eval_result,
        per_oligo_snapshot=per_oligo_snapshot,
        aggregate_values=aggregate_values,
    )
    _add_derived_pae_snapshot(
        eval_result,
        per_oligo_snapshot=per_oligo_snapshot,
        aggregate_values=aggregate_values,
    )

    for metric_name, values in aggregate_values.items():
        if values:
            snapshot[metric_name] = float(np.mean(values))
    snapshot.update(per_oligo_snapshot)
    contributions = getattr(eval_result, "score_contributions", None)
    if contributions:
        snapshot = merge_score_contribution_snapshot(snapshot, contributions)
    return snapshot


def merge_score_details(evaluation_result, score_details: dict | None) -> dict:
    base_details = score_details or {}
    defaults = build_standard_score_snapshot(evaluation_result)
    return deep_fill_missing(base_details, defaults)
