from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrolabe.evaluators.score_snapshot import merge_score_details
from astrolabe.per_residue_scores import mutator_selector_score_delta_details
from astrolabe.stage_transition import transition_consistency_gate_score_context
from astrolabe.utils.logging_utils import to_numeric


SCORE_CONTEXT_PREFIXES = ("previous_", "delta_")
NOISE_COMPONENT_KEYS = {"iteration", "step", "stage_entry", "acceptance_gate_passed"}
NOISE_COMPONENT_SUFFIXES = ("_length",)


@dataclass(frozen=True)
class DecisionScoreContext:
    weighted_score: Any
    scores_detail: dict[str, Any]
    early_stop: bool = False
    soft_accepted: bool = False
    hard_gate_passed: bool = True


def get_transition_consistency_gate_result(runtime_config: Any | None):
    if runtime_config is None:
        return None
    if hasattr(runtime_config, "get"):
        return runtime_config.get("transition_consistency_gate_result")
    return getattr(runtime_config, "transition_consistency_gate_result", None)


def get_reference_gate_result(runtime_config: Any | None):
    if runtime_config is None:
        return None
    if hasattr(runtime_config, "get"):
        return runtime_config.get("reference_gate_result")
    return getattr(runtime_config, "reference_gate_result", None)


def _get_transition_consistency_gate_metadata(runtime_config: Any | None):
    if runtime_config is None:
        return None
    if hasattr(runtime_config, "get"):
        return runtime_config.get("transition_consistency_gate_metadata")
    return getattr(runtime_config, "transition_consistency_gate_metadata", None)


def _copy_mapping_preserve_type(values: dict | None) -> dict:
    if values is None:
        return {}
    try:
        return type(values)(values)
    except Exception:
        return dict(values)


def _deep_override(base: dict, updates: dict) -> dict:
    merged = _copy_mapping_preserve_type(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_override(merged[key], value)
        elif isinstance(value, dict):
            merged[key] = _copy_mapping_preserve_type(value)
        else:
            merged[key] = value
    return merged


def attach_reference_gate_details(
    scores_detail: dict | None,
    runtime_config: Any | None,
) -> dict:
    result = get_reference_gate_result(runtime_config)
    if result is None:
        return _copy_mapping_preserve_type(scores_detail)

    from astrolabe.structure_start import reference_gate_score_context

    scores_detail = _deep_override(
        _copy_mapping_preserve_type(scores_detail),
        reference_gate_score_context(result),
    )
    scores_detail["reference_gate_active"] = (
        1.0 if bool(getattr(result, "gate_active", False)) else 0.0
    )
    return scores_detail


def attach_transition_consistency_gate_details(
    scores_detail: dict | None,
    runtime_config: Any | None,
) -> dict:
    consistency = get_transition_consistency_gate_result(runtime_config)
    if consistency is None or not bool(getattr(consistency, "gate_active", False)):
        return _copy_mapping_preserve_type(scores_detail)

    scores_detail = _copy_mapping_preserve_type(scores_detail)

    metadata = _get_transition_consistency_gate_metadata(runtime_config)
    if not metadata:
        raise ValueError(
            "transition_consistency_gate_metadata is required when "
            "transition_consistency_gate_result is active"
        )
    source_stage = metadata.get("source_stage", "") or ""
    target_stage = metadata.get("target_stage", "") or ""
    attempt = metadata.get("attempt")

    scores_detail.update(
        transition_consistency_gate_score_context(
            consistency,
            source_stage=source_stage,
            target_stage=target_stage,
            attempt=attempt,
        )
    )
    return scores_detail


def component_deltas(
    current_scores: dict,
    previous_scores: dict,
) -> list[tuple[str, float | None, float | None, float | None]]:
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    keys = sorted(set(current_scores.keys()) | set(previous_scores.keys()))
    for key in keys:
        current = to_numeric(current_scores.get(key))
        previous = to_numeric(previous_scores.get(key))
        delta = None
        if current is not None and previous is not None:
            delta = current - previous
        rows.append((str(key), previous, current, delta))
    return rows


def previous_delta_score_details(
    *,
    previous_weighted: Any | None,
    current_weighted: Any | None,
    previous_scores: dict | None,
    current_scores: dict | None,
) -> dict[str, float]:
    details: dict[str, float] = {}
    previous_weighted_num = to_numeric(previous_weighted)
    current_weighted_num = to_numeric(current_weighted)
    if previous_weighted_num is not None:
        details["previous_weighted_score"] = float(previous_weighted_num)
    if previous_weighted_num is not None and current_weighted_num is not None:
        details["delta_weighted_score"] = float(current_weighted_num) - float(
            previous_weighted_num
        )
    for name, previous, _current, delta in component_deltas(
        current_scores=current_scores or {},
        previous_scores=previous_scores or {},
    ):
        if name.startswith(SCORE_CONTEXT_PREFIXES):
            continue
        if name in NOISE_COMPONENT_KEYS:
            continue
        if any(name.endswith(suffix) for suffix in NOISE_COMPONENT_SUFFIXES):
            continue
        if previous is not None:
            details[f"previous_{name}"] = float(previous)
        if delta is not None:
            details[f"delta_{name}"] = float(delta)
    return details


def self_delta_score_details(
    scores_detail: dict | None,
    weighted_score: Any,
) -> dict[str, float]:
    details: dict[str, float] = {}
    weighted_score_num = to_numeric(weighted_score)
    if weighted_score_num is not None:
        details["previous_weighted_score"] = float(weighted_score_num)
        details["delta_weighted_score"] = 0.0
    for name, value in (scores_detail or {}).items():
        if str(name).startswith(SCORE_CONTEXT_PREFIXES):
            continue
        value_num = to_numeric(value)
        if value_num is None:
            continue
        details[f"previous_{name}"] = float(value_num)
        details[f"delta_{name}"] = 0.0
    return details


def _call_loss_bool(loss_computer: Any, method_name: str, scores_detail: dict) -> bool:
    method = getattr(loss_computer, method_name, None)
    if not callable(method):
        return False
    return bool(method(scores_detail))


def _call_acceptance_gate(loss_computer: Any, scores_detail: dict) -> bool:
    method = getattr(loss_computer, "passes_acceptance_gate", None)
    if not callable(method):
        return True
    return bool(method(scores_detail))


def build_decision_score_context(
    *,
    eval_result: Any,
    loss_computer: Any,
    runtime_config: Any,
    previous_weighted: Any | None = None,
    previous_scores: dict | None = None,
    stage_entry: float | None = None,
    include_default_scores: bool = True,
    include_mutator_selector_deltas: bool = False,
    include_previous_delta: bool = False,
    include_self_delta: bool = False,
    evaluate_decision_flags: bool = True,
    evaluate_acceptance_gate: bool = True,
) -> DecisionScoreContext:
    weighted_score, scores_detail = loss_computer.compute_loss(
        eval_result,
        runtime_config,
    )
    scores_detail = attach_reference_gate_details(scores_detail, runtime_config)
    scores_detail = attach_transition_consistency_gate_details(
        scores_detail,
        runtime_config,
    )
    if include_default_scores:
        scores_detail = merge_score_details(eval_result, scores_detail)
    if include_mutator_selector_deltas:
        scores_detail.update(
            mutator_selector_score_delta_details(eval_result, runtime_config)
        )
    if include_self_delta:
        scores_detail.update(self_delta_score_details(scores_detail, weighted_score))
    elif include_previous_delta:
        scores_detail.update(
            previous_delta_score_details(
                previous_weighted=previous_weighted,
                current_weighted=weighted_score,
                previous_scores=previous_scores,
                current_scores=dict(scores_detail or {}),
            )
        )
    if stage_entry is not None:
        scores_detail["stage_entry"] = float(stage_entry)

    early_stop = (
        _call_loss_bool(loss_computer, "is_early_stopping", scores_detail)
        if evaluate_decision_flags
        else False
    )
    soft_accepted = (
        _call_loss_bool(loss_computer, "is_soft_accepted", scores_detail)
        if evaluate_decision_flags
        else False
    )
    hard_gate_passed = (
        _call_acceptance_gate(loss_computer, scores_detail)
        if evaluate_acceptance_gate
        else True
    )
    if evaluate_acceptance_gate:
        scores_detail["acceptance_gate_passed"] = (
            1.0 if hard_gate_passed else 0.0
        )

    return DecisionScoreContext(
        weighted_score=weighted_score,
        scores_detail=scores_detail,
        early_stop=early_stop,
        soft_accepted=soft_accepted,
        hard_gate_passed=hard_gate_passed,
    )


def passes_stage_entry_acceptance_gate(
    *,
    eval_result: Any,
    loss_computer: Any,
    runtime_config: Any,
) -> bool:
    context = build_decision_score_context(
        eval_result=eval_result,
        loss_computer=loss_computer,
        runtime_config=runtime_config,
        stage_entry=1.0,
        include_self_delta=True,
        evaluate_decision_flags=False,
    )
    return context.hard_gate_passed
