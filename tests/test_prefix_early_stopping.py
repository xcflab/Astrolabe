from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions
from astrolabe.losses.base import LossEngine


PREFIX_EXPR = (
    "(prefix_steps == 8 and best.AC.binder_ptm <= 0.71) or "
    "(prefix_steps == 12 and best.AC.ipsae <= 0.03)"
)


@pytest.fixture(autouse=True)
def _reset_global_config():
    global_config.reset()
    yield
    global_config.reset()


def _init_runtime(tmp_path: Path) -> None:
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(apply_mutations=lambda: None, clear_mutations=lambda: None),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="default"))
    global_config.set_variable("step", 0)


def _decisions(initial_detail: dict) -> Decisions:
    return Decisions(
        sequences={0: {}},
        scores={0: 1.0},
        evaluation_results={0: object()},
        scores_detail={0: initial_detail},
        accepted_steps=[0],
        last_accepted=0,
        stage_history={0: "default"},
    )


class _PrefixLoss(LossEngine):
    def __init__(
        self,
        *,
        score_detail: dict,
        prefix_early_stopping: str | None = PREFIX_EXPR,
        early_stopping: bool = False,
        acceptance_gate: bool = True,
    ) -> None:
        super().__init__(
            {
                "loss_terms": {
                    "binder_ptm": {"_type": "binder_ptm"},
                    "ipsae": {
                        "_type": "ipsae",
                        "chain_a": "A",
                        "chain_b": "C",
                    },
                }
            },
            validate_config=False,
        )
        self.score_detail = score_detail
        self._early_stopping = early_stopping
        self._acceptance_gate = acceptance_gate
        self.setup(
            weighted_score_expression="weighted",
            early_stopping=None,
            acceptance_threshold=None,
            acceptance_gate=None,
            validate_config=False,
            prefix_early_stopping=prefix_early_stopping,
        )

    def compute_loss(self, _eval_result, _runtime_config):
        detail = self._default_expression_symbols()
        detail.update(self.score_detail)
        return float(self.score_detail["weighted"]), detail

    def is_early_stopping(self, _scores_detail):
        return self._early_stopping

    def is_soft_accepted(self, _scores_detail):
        return False

    def passes_acceptance_gate(self, _scores_detail):
        return self._acceptance_gate


def _run_decision_step(
    *,
    decisions: Decisions,
    step: int,
    score_detail: dict,
    prefix_early_stopping: str | None = PREFIX_EXPR,
    early_stopping: bool = False,
    acceptance_gate: bool = True,
):
    global_config.set_variable("step", step)
    return decisions.make_mutation_decision(
        step=step,
        eval_result=object(),
        loss_computer=_PrefixLoss(
            score_detail=score_detail,
            prefix_early_stopping=prefix_early_stopping,
            early_stopping=early_stopping,
            acceptance_gate=acceptance_gate,
        ),
        runtime_config={},
    )


def _detail(*, weighted: float, binder_ptm: float, ipsae: float) -> dict:
    return {
        "weighted": weighted,
        "AC": {"binder_ptm": binder_ptm, "ipsae": ipsae},
    }


def test_prefix_best_context_uses_all_scores_through_current_step() -> None:
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.60, ipsae=0.01))
    decisions.scores_detail[5] = _detail(weighted=1.2, binder_ptm=0.80, ipsae=0.02)
    decisions.scores_detail[8] = _detail(weighted=0.8, binder_ptm=0.70, ipsae=0.05)
    decisions.scores_detail[13] = _detail(weighted=2.0, binder_ptm=0.99, ipsae=0.99)
    decisions.accepted_steps = [0, 5]

    best = decisions._build_prefix_best_score_context(through_step=8)

    assert best.AC.binder_ptm == pytest.approx(0.80)
    assert best.AC.ipsae == pytest.approx(0.05)


def test_prefix_early_stopping_triggers_on_step8_failure(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.70, ipsae=0.04))

    early_stop, accepted_eval, accepted = _run_decision_step(
        decisions=decisions,
        step=7,
        score_detail=_detail(weighted=0.0, binder_ptm=0.69, ipsae=0.04),
    )

    assert early_stop is True
    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert decisions.scores_detail[7]["prefix_early_stop_triggered"] == 1.0
    assert decisions.scores_detail[7]["prefix_early_stopping_expression"] == PREFIX_EXPR
    assert decisions.scores_detail[7]["prefix_steps"] == 8


def test_prefix_early_stopping_triggers_on_step12_failure(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.72, ipsae=0.02))

    early_stop, _accepted_eval, accepted = _run_decision_step(
        decisions=decisions,
        step=11,
        score_detail=_detail(weighted=0.0, binder_ptm=0.73, ipsae=0.03),
    )

    assert early_stop is True
    assert accepted is False
    assert decisions.scores_detail[11]["prefix_early_stop_triggered"] == 1.0
    assert decisions.scores_detail[11]["prefix_early_stopping_expression"] == PREFIX_EXPR
    assert decisions.scores_detail[11]["prefix_steps"] == 12


def test_prefix_early_stopping_does_not_trigger_when_both_prefixes_pass(
    tmp_path: Path,
) -> None:
    _init_runtime(tmp_path)
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.72, ipsae=0.04))

    step8_stop, _accepted_eval, _accepted = _run_decision_step(
        decisions=decisions,
        step=7,
        score_detail=_detail(weighted=0.0, binder_ptm=0.70, ipsae=0.02),
    )
    step12_stop, _accepted_eval, _accepted = _run_decision_step(
        decisions=decisions,
        step=11,
        score_detail=_detail(weighted=0.0, binder_ptm=0.70, ipsae=0.02),
    )

    assert step8_stop is False
    assert step12_stop is False
    assert decisions.scores_detail[7]["prefix_early_stop_triggered"] == 0.0
    assert decisions.scores_detail[11]["prefix_early_stop_triggered"] == 0.0


def test_prefix_early_stopping_unconfigured_does_not_trigger(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.70, ipsae=0.02))

    early_stop, _accepted_eval, _accepted = _run_decision_step(
        decisions=decisions,
        step=7,
        score_detail=_detail(weighted=0.0, binder_ptm=0.69, ipsae=0.01),
        prefix_early_stopping=None,
    )

    assert early_stop is False
    assert "prefix_early_stop_triggered" not in decisions.scores_detail[7]
    assert "prefix_early_stopping_expression" not in decisions.scores_detail[7]


def test_prefix_early_stopping_is_or_merged_with_existing_early_stopping(
    tmp_path: Path,
) -> None:
    _init_runtime(tmp_path)
    decisions = _decisions(_detail(weighted=1.0, binder_ptm=0.72, ipsae=0.04))

    early_stop, _accepted_eval, accepted = _run_decision_step(
        decisions=decisions,
        step=7,
        score_detail=_detail(weighted=0.0, binder_ptm=0.72, ipsae=0.04),
        early_stopping=True,
        acceptance_gate=False,
    )

    assert early_stop is True
    assert accepted is False
    assert decisions.scores_detail[7]["prefix_early_stop_triggered"] == 0.0
