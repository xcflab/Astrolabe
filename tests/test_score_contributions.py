from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from astrolabe.evaluators.score_snapshot import (
    build_standard_score_snapshot,
    flatten_score_details,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.posthoc.filtering import (
    apply_filter_expression,
    compile_filter_expression,
)
from astrolabe.posthoc.io_utils import parse_scores_csv
from astrolabe.score_contributions import (
    ScoreContribution,
    score_contributions_to_snapshot,
)


def _scores(*, score_contributions=None) -> StandardScores:
    return StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.8], "AC": [0.6]},
        iptm={"AB": [0.7], "AC": [0.5]},
        complex_paths={"AB": Path("ab.cif"), "AC": Path("ac.cif")},
        score_contributions=score_contributions,
    )


def test_score_contributions_snapshot_global_and_oligo_scope() -> None:
    snapshot = score_contributions_to_snapshot(
        [
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_entropy_norm",
                value=[0.4, 0.6],
                aggregation="mean",
                provenance="last_mutation_telemetry",
            ),
            ScoreContribution(
                source="predocker.vina",
                metric_name="vina_contact_score",
                value=0.72,
                scope="oligo",
                entity="AB",
                provenance="predocker_context",
            ),
        ]
    )

    assert snapshot["dplm_entropy_norm"] == pytest.approx(0.5)
    assert snapshot["dplm_entropy_norm_provenance"] == "last_mutation_telemetry"
    assert snapshot["AB"]["vina_contact_score"] == pytest.approx(0.72)
    assert snapshot["AB"]["vina_contact_score_provenance"] == "predocker_context"
    assert flatten_score_details(snapshot)["AB.vina_contact_score"] == pytest.approx(0.72)


def test_standard_score_snapshot_preserves_old_keys_and_filters_contributions() -> None:
    scores = _scores(
        score_contributions=[
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_entropy_norm",
                value=0.5,
            ),
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_contact_score",
                value=0.7,
                scope="oligo",
                entity="AB",
            ),
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_contact_score",
                value=0.2,
                scope="oligo",
                entity="AC",
            ),
        ]
    )

    filtered = scores.filter_by_oligomer(["AB"])
    snapshot = build_standard_score_snapshot(filtered)

    assert snapshot["plddt"] == pytest.approx(0.8)
    assert snapshot["iptm"] == pytest.approx(0.7)
    assert snapshot["dplm_entropy_norm"] == pytest.approx(0.5)
    assert snapshot["AB"]["dplm_contact_score"] == pytest.approx(0.7)
    assert "AC" not in snapshot


def test_runtime_score_contribution_enters_weighted_gate_and_posthoc_context(
    tmp_path: Path,
) -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "telemetry_bonus": "dplm_entropy_norm + AB.dplm_contact_score",
            },
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="plddt + telemetry_bonus",
        early_stopping=None,
        acceptance_threshold=None,
        acceptance_gate="dplm_entropy_norm >= 0.5 and AB.dplm_contact_score >= 0.7",
        validate_config=False,
    )
    runtime_config = {
        "score_contributions": [
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_entropy_norm",
                value=np.array([0.4, 0.6]),
                aggregation="mean",
                provenance="last_mutation_telemetry",
            ),
            ScoreContribution(
                source="mutator.dplm",
                metric_name="dplm_contact_score",
                value=0.7,
                scope="oligo",
                entity="AB",
            ),
        ]
    }

    score, detail = engine.compute_loss(_scores(), runtime_config=runtime_config)
    context = engine.score_context(_scores(), runtime_config=runtime_config)

    assert score == pytest.approx(1.9)
    assert detail["telemetry_bonus"] == pytest.approx(1.2)
    assert detail["dplm_entropy_norm"] == pytest.approx(0.5)
    assert detail["dplm_entropy_norm_provenance"] == "last_mutation_telemetry"
    assert detail["AB"].dplm_contact_score == pytest.approx(0.7)
    assert context["AB"].dplm_contact_score == pytest.approx(0.7)
    assert engine.passes_acceptance_gate(detail) is True

    flat = {
        key: value
        for key, value in flatten_score_details(detail).items()
        if key not in {"step", "weighted_score", "accepted"}
    }
    scores_csv = tmp_path / "scores.csv"
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["step", "weighted_score", "accepted", *sorted(flat)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "step": 1,
                "weighted_score": score,
                "accepted": True,
                **flat,
            }
        )
    metrics = parse_scores_csv(scores_csv)[1]
    compiled = compile_filter_expression(
        "dplm_entropy_norm >= 0.5 and AB.dplm_contact_score >= 0.7"
    )

    assert apply_filter_expression(metrics, compiled).passed is True


def test_score_contribution_cannot_overwrite_existing_score_key() -> None:
    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    engine.setup(
        weighted_score_expression="plddt",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )

    with pytest.raises(ValueError, match="overwrite existing score key: plddt"):
        engine.compute_loss(
            _scores(),
            runtime_config={
                "score_contributions": [
                    ScoreContribution(
                        source="mutator.dplm",
                        metric_name="plddt",
                        value=0.1,
                    )
                ]
            },
        )


def test_standard_score_contribution_cannot_overwrite_snapshot_key() -> None:
    with pytest.raises(ValueError, match="overwrite existing score key: plddt"):
        build_standard_score_snapshot(
            _scores(
                score_contributions=[
                    ScoreContribution(
                        source="postprocessor",
                        metric_name="plddt",
                        value=0.1,
                    )
                ]
            )
        )


def test_parse_scores_csv_normalizes_padded_column_names(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    scores_csv.write_text(
        " step , weighted_score     , ABCD.binder_ptm    \n"
        " 0    , 0.75               , 0.63               \n",
        encoding="utf-8",
    )

    metrics = parse_scores_csv(scores_csv)[0]

    assert metrics["weighted_score"] == pytest.approx(0.75)
    assert metrics["ABCD.binder_ptm"] == pytest.approx(0.63)
    assert all(key == key.strip() for key in metrics)


def test_parse_scores_csv_rejects_columns_that_collide_after_normalization(
    tmp_path: Path,
) -> None:
    scores_csv = tmp_path / "scores.csv"
    scores_csv.write_text(
        "step,weighted_score, weighted_score \n"
        "0,0.75,0.63\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate columns"):
        parse_scores_csv(scores_csv)
