from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreContribution:
    """Opt-in score metric contributed by a runtime component."""

    source: str
    metric_name: str
    value: Any
    scope: str = "global"
    entity: str | None = None
    shape: str = "scalar"
    aggregation: str = "identity"
    flatten_as: str | None = None
    provenance: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not str(self.source or "").strip():
            raise ValueError("ScoreContribution.source is required")
        if not str(self.metric_name or "").strip():
            raise ValueError("ScoreContribution.metric_name is required")
        scope = str(self.scope or "").strip()
        if scope not in {"global", "oligo", "protomer"}:
            raise ValueError(
                "ScoreContribution.scope must be one of: global, oligo, protomer"
            )
        if scope != "global" and not str(self.entity or "").strip():
            raise ValueError(
                "ScoreContribution.entity is required for oligo/protomer scope"
            )
        if self.flatten_as is not None and not str(self.flatten_as).strip():
            raise ValueError("ScoreContribution.flatten_as cannot be empty")

    @property
    def output_key(self) -> str:
        return str(self.flatten_as or self.metric_name)


def normalize_score_contributions(raw: Any) -> list[ScoreContribution]:
    if raw is None:
        return []
    if isinstance(raw, ScoreContribution):
        return [raw]
    if isinstance(raw, Mapping):
        if "metric_name" in raw or "value" in raw:
            return [_contribution_from_mapping(raw)]
        source = str(raw.get("source", "runtime")) if "source" in raw else "runtime"
        return [
            ScoreContribution(source=source, metric_name=str(key), value=value)
            for key, value in raw.items()
            if key != "source"
        ]
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        contributions: list[ScoreContribution] = []
        for item in raw:
            contributions.extend(normalize_score_contributions(item))
        return contributions
    raise TypeError(f"Unsupported score contribution payload: {type(raw).__name__}")


def score_contributions_from_runtime_config(runtime_config: Any) -> list[ScoreContribution]:
    if runtime_config is None:
        return []
    if hasattr(runtime_config, "get"):
        raw = runtime_config.get("score_contributions")
    else:
        raw = getattr(runtime_config, "score_contributions", None)
    return normalize_score_contributions(raw)


def score_contributions_to_snapshot(raw: Any) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for contribution in normalize_score_contributions(raw):
        if not contribution.enabled:
            continue
        _merge_contribution(snapshot, contribution)
    return snapshot


def merge_score_contribution_snapshot(
    base: dict[str, object],
    raw: Any,
) -> dict[str, object]:
    merged = dict(base or {})
    for contribution in normalize_score_contributions(raw):
        if not contribution.enabled:
            continue
        _merge_contribution(merged, contribution)
    return merged


def filter_score_contributions(
    raw: Any,
    *,
    allowed_oligos: set[str],
    allowed_protomers: set[str],
) -> list[ScoreContribution]:
    filtered: list[ScoreContribution] = []
    for contribution in normalize_score_contributions(raw):
        if contribution.scope == "oligo" and str(contribution.entity) not in allowed_oligos:
            continue
        if (
            contribution.scope == "protomer"
            and str(contribution.entity) not in allowed_protomers
        ):
            continue
        filtered.append(contribution)
    return filtered


def _contribution_from_mapping(raw: Mapping[str, Any]) -> ScoreContribution:
    if "metric_name" not in raw:
        raise ValueError("Score contribution mapping requires metric_name")
    if "value" not in raw:
        raise ValueError("Score contribution mapping requires value")
    return ScoreContribution(
        source=str(raw.get("source", "runtime")),
        metric_name=str(raw["metric_name"]),
        value=raw["value"],
        scope=str(raw.get("scope", "global")),
        entity=None if raw.get("entity") is None else str(raw.get("entity")),
        shape=str(raw.get("shape", "scalar")),
        aggregation=str(raw.get("aggregation", "identity")),
        flatten_as=None
        if raw.get("flatten_as") is None
        else str(raw.get("flatten_as")),
        provenance=None
        if raw.get("provenance") is None
        else str(raw.get("provenance")),
        enabled=bool(raw.get("enabled", True)),
    )


def _aggregate_value(value: Any, aggregation: str) -> object:
    aggregation = str(aggregation or "identity")
    if aggregation in {"identity", "none"}:
        return value
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"Cannot aggregate score contribution value with {aggregation!r}"
        ) from exc
    if arr.size == 0:
        raise ValueError("Cannot aggregate empty score contribution value")
    if aggregation == "mean":
        return float(np.mean(arr))
    if aggregation == "sum":
        return float(np.sum(arr))
    if aggregation == "min":
        return float(np.min(arr))
    if aggregation == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown score contribution aggregation: {aggregation!r}")


def _merge_contribution(target: dict[str, object], contribution: ScoreContribution) -> None:
    value = _aggregate_value(contribution.value, contribution.aggregation)
    if contribution.scope == "global":
        _set_new_key(target, contribution.output_key, value)
        _set_provenance(target, contribution)
        return

    entity = str(contribution.entity)
    nested = target.get(entity)
    if nested is None:
        nested = {}
        target[entity] = nested
    if not isinstance(nested, dict):
        raise ValueError(
            f"Score contribution entity {entity!r} conflicts with existing score key"
        )
    _set_new_key(nested, contribution.output_key, value)
    _set_provenance(nested, contribution)


def _set_provenance(target: dict[str, object], contribution: ScoreContribution) -> None:
    if contribution.provenance is None:
        return
    _set_new_key(target, f"{contribution.output_key}_provenance", contribution.provenance)


def _set_new_key(target: dict[str, object], key: str, value: object) -> None:
    if key in target:
        raise ValueError(f"Score contribution would overwrite existing score key: {key}")
    target[key] = value
