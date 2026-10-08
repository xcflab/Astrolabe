from __future__ import annotations

from typing import Any

from astrolabe.losses.base import loss_registry


@loss_registry.register(
    required_params=["metric"],
    description="Read an oligomer-level metric emitted by RosettaCrystalEvaluator",
    supports_oligomer_specific=True,
)
def rosetta_crystal_metric(eval_result, metric: str) -> dict[str, float]:
    raw = getattr(eval_result, "rosetta_metrics", None)
    if not isinstance(raw, dict):
        raise ValueError("rosetta_crystal_metric requires RosettaCrystalEvaluator output")
    scores: dict[str, float] = {}
    for oligo_name in list(getattr(eval_result, "names", []) or []):
        values: Any = raw.get(oligo_name)
        if not isinstance(values, dict) or metric not in values:
            raise ValueError(
                f"Rosetta metric {metric!r} missing for oligomer {oligo_name!r}"
            )
        scores[str(oligo_name)] = float(values[metric])
    return scores
