from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.losses.base import LossEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_channel_benchmark_confidence_terms_saturate_above_threshold() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "benchmark_channel_v1_refine.yaml")
    engine = LossEngine(
        OmegaConf.to_container(cfg.loss, resolve=False),
        validate_config=False,
    )
    expr = str(cfg.stages.default.weighted_score_expression)
    baseline = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 1.2,
            "iptm": 0.8,
            "ptm": 0.8,
            "plddt": 0.85,
        },
        expr,
    )
    boosted = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 1.2,
            "iptm": 0.95,
            "ptm": 0.93,
            "plddt": 0.97,
        },
        expr,
    )
    low_conf = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 1.2,
            "iptm": 0.4,
            "ptm": 0.5,
            "plddt": 0.6,
        },
        expr,
    )

    assert boosted == pytest.approx(baseline)
    assert low_conf < baseline


def test_channel_benchmark_radius_bonus_rewards_larger_min_radius() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "benchmark_channel_v1_refine.yaml")
    engine = LossEngine(
        OmegaConf.to_container(cfg.loss, resolve=False),
        validate_config=False,
    )
    expr = str(cfg.stages.default.weighted_score_expression)

    narrow = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 0.8,
            "iptm": 0.8,
            "ptm": 0.8,
            "plddt": 0.85,
        },
        expr,
    )
    gate = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 1.2,
            "iptm": 0.8,
            "ptm": 0.8,
            "plddt": 0.85,
        },
        expr,
    )
    broad = engine.weighted_score(
        {
            "channel_hole": 0.4,
            "best_channel_min_radius": 1.6,
            "iptm": 0.8,
            "ptm": 0.8,
            "plddt": 0.85,
        },
        expr,
    )

    assert narrow < gate < broad
    assert gate - narrow == pytest.approx(broad - gate, rel=0.2)
