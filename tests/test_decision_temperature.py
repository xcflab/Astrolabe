from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import astrolabe.decision_maker as decision_mod
from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions


def _init_runtime(
    tmp_path: Path,
    *,
    use_metropolis_acceptance: bool,
    initial_temperature: float = 1000.0,
) -> SimpleNamespace:
    calls = SimpleNamespace(applied=False, cleared=False)
    global_config.reset()
    global_config.init_const("initial_temperature", initial_temperature)
    global_config.init_const("use_metropolis_acceptance", use_metropolis_acceptance)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(calls, "applied", True),
            clear_mutations=lambda: setattr(calls, "cleared", True),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="default"))
    global_config.init_const(
        "protomers",
        {"A": SimpleNamespace(type=SimpleNamespace(value="p"), mutated_sequence="AAAT")},
    )
    return calls


def _decisions() -> Decisions:
    return Decisions(
        sequences={0: {"pA": "AAAA"}},
        scores={0: 1.0},
        evaluation_results={0: object()},
        scores_detail={0: {"weighted": 1.0}},
        accepted_steps=[0],
        last_accepted=0,
        stage_history={0: "default"},
    )


class _WorseScoreLoss:
    def compute_loss(self, _eval_result, _runtime_config):
        return 0.0, {"weighted": 0.0}

    def is_early_stopping(self, _scores_detail):
        return False

    def is_soft_accepted(self, _scores_detail):
        return False


def test_metropolis_disabled_rejects_worse_score_without_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _init_runtime(tmp_path, use_metropolis_acceptance=False)
    decisions = _decisions()
    monkeypatch.setattr(
        decision_mod,
        "calculate_temperature",
        lambda _step: pytest.fail("temperature should not be evaluated"),
    )

    early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        step=1,
        eval_result=object(),
        loss_computer=_WorseScoreLoss(),
        runtime_config={},
    )

    assert early_stop is False
    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert decisions.last_accepted == 0
    assert calls.applied is False
    assert calls.cleared is True
    assert decisions.row_history[-1]["accepted"] is False


def test_metropolis_explicit_true_keeps_compatibility_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = _init_runtime(tmp_path, use_metropolis_acceptance=True)
    decisions = _decisions()
    monkeypatch.setattr(decision_mod.np.random, "uniform", lambda _low, _high: 0.0)

    with caplog.at_level("INFO"):
        _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
            step=1,
            eval_result=object(),
            loss_computer=_WorseScoreLoss(),
            runtime_config={},
        )

    assert accepted is True
    assert accepted_eval is decisions.evaluation_results[1]
    assert decisions.last_accepted == 1
    assert calls.applied is True
    assert calls.cleared is False
    assert any("reason=metropolis" in record.getMessage() for record in caplog.records)
