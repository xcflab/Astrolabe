from __future__ import annotations

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import run as run_mod
from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager
from astrolabe.stage_transition import compute_reference_consistency
from astrolabe.structure_start import ReferenceGateResult


def _scores(*names: str) -> StandardScores:
    return StandardScores(
        names=list(names),
        plddt={name: [1.0] for name in names},
        complex_paths={name: Path(f"{name.lower()}.pdb") for name in names},
    )


def _scores_at(tmp_path: Path, *names: str) -> StandardScores:
    complex_paths = {}
    for name in names:
        structure_path = tmp_path / f"{name.lower()}.pdb"
        structure_path.write_text(f"{name}\n", encoding="utf-8")
        complex_paths[name] = structure_path
    return StandardScores(
        names=list(names),
        plddt={name: [1.0] for name in names},
        complex_paths=complex_paths,
    )


def _init_loss() -> LossEngine:
    global_config.reset()
    loss_computer = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss_computer)
    global_config.init_const("protomers", {})
    return loss_computer


def test_run_optimization_loop_keeps_source_stage_while_transition_consistency_gate_is_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("use_metropolis_acceptance", False)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "transition_consistency_gate": {
                "weight": 0.5,
                "consistency_threshold": 0.9,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_eval = _scores_at(tmp_path, "A")
    reevaluated = _scores_at(tmp_path, "AB")

    class _Decisions:
        def __init__(self):
            self.recompute_called = False

        def make_mutation_decision(
            self, step, evaluation_result, loss_computer, runtime_config, force_accept=False,
            stage_entry=0.0,
        ):
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_called = True

    decisions = _Decisions()

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    eval_results = iter([accepted_eval, reevaluated])
    monkeypatch.setattr(
        run_mod, "perform_evaluation", lambda *args, **kwargs: next(eval_results)
    )
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1" if int(global_config.step) == 1 else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: SimpleNamespace(
            gate_active=True,
            score=0.4,
            threshold=0.9,
            passed=False,
            comparisons={},
        ),
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=SimpleNamespace(compute_loss=lambda *args, **kwargs: (0.0, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert decisions.recompute_called is False
    assert stage_manager.transition_consistency_gate_context is not None
    assert stage_manager.current_stage == "default"
    assert stage_manager.current_transition_input_scores is reevaluated
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert result is reevaluated


def test_run_optimization_loop_stays_in_source_stage_after_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "transition_consistency_gate": {
                "weight": 0.5,
                "consistency_threshold": 0.9,
                "max_attempts": 1,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_eval = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")

    class _Decisions:
        def __init__(self):
            self.calls = 0

        def make_mutation_decision(
            self, step, evaluation_result, loss_computer, runtime_config, force_accept=False,
            stage_entry=0.0,
        ):
            self.calls += 1
            return False, evaluation_result, True if self.calls == 1 else False

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            return None

    decisions = _Decisions()

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    eval_results = iter([accepted_eval, gate_seed, gate_seed])
    monkeypatch.setattr(
        run_mod, "perform_evaluation", lambda *args, **kwargs: next(eval_results)
    )
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1" if int(global_config.step) == 1 else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: SimpleNamespace(
            gate_active=True,
            score=0.4,
            threshold=0.9,
            passed=False,
            comparisons={},
        ),
    )

    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=SimpleNamespace(compute_loss=lambda *args, **kwargs: (0.0, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config={},
    )

    assert stage_manager.transition_consistency_gate_context is None
    assert stage_manager.current_stage == "default"
    assert result.names == accepted_eval.names


def test_run_optimization_loop_keeps_gate_result_visible_on_seed_pass(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_eval = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")
    passed_gate = SimpleNamespace(
        gate_active=True,
        score=0.91,
        threshold=0.8,
        passed=True,
        comparisons={"AB~A": 0.91},
    )

    class _Decisions:
        def __init__(self):
            self.recompute_called = False
            self.recompute_runtime = None
            self.recompute_eval = None

        def make_mutation_decision(
            self, step, evaluation_result, loss_computer, runtime_config, force_accept=False,
            stage_entry=0.0,
        ):
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_called = True
            self.recompute_runtime = dict(runtime_config)
            self.recompute_eval = evaluation_result

    decisions = _Decisions()

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    eval_results = iter([accepted_eval, gate_seed])
    monkeypatch.setattr(
        run_mod, "perform_evaluation", lambda *args, **kwargs: next(eval_results)
    )
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1" if int(global_config.step) == 1 else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: passed_gate,
    )

    accepted_scores_detail = {}

    def _passes_seed_acceptance_gate(scores_detail):
        accepted_scores_detail.update(scores_detail)
        return (
            scores_detail["transition_consistency_gate_score"] >= 0.8
            and scores_detail["stage_entry"] == 1.0
            and scores_detail["plddt"] > 0.5
            and scores_detail["previous_plddt"] == scores_detail["plddt"]
            and scores_detail["delta_plddt"] == 0.0
        )

    gate_loss = SimpleNamespace(
        compute_loss=lambda *_args, **_kwargs: (
            0.0,
            {"plddt": 0.9},
        ),
        passes_acceptance_gate=_passes_seed_acceptance_gate,
    )

    with caplog.at_level(logging.INFO):
        result = run_mod.run_optimization_loop(
            args=SimpleNamespace(steps=2),
            loss_computer=gate_loss,
            initial_evaluation_result=initial_eval,
            stage_manager=stage_manager,
            runtime_config={},
        )

    assert decisions.recompute_called is True
    assert decisions.recompute_eval is gate_seed
    assert decisions.recompute_runtime is not None
    assert decisions.recompute_runtime["transition_consistency_gate_result"] is passed_gate
    assert accepted_scores_detail["transition_consistency_gate_score"] == 0.91
    assert accepted_scores_detail["previous_plddt"] == 0.9
    assert accepted_scores_detail["delta_plddt"] == 0.0
    assert stage_manager.current_stage == "stage_1"
    assert result is gate_seed
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "transition_consistency_gate" in message and "seed_pass" in message
        for message in messages
    )


def test_initial_gated_stage_entry_strips_reference_gate_from_seed_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object()})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["AB"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval0",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "AB")
    passed_gate = SimpleNamespace(
        gate_active=True,
        score=0.91,
        threshold=0.8,
        passed=True,
        comparisons={"AB~AB": 0.91},
    )

    class _Decisions:
        def __init__(self):
            self.recompute_runtime = None

        def dump_results(self, *_args, **_kwargs):
            return None

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_runtime = dict(runtime_config)

    class _GateLoss:
        def __init__(self):
            self.acceptance_scores_detail = None

        def compute_loss(self, eval_result, runtime_config):
            return 0.0, {"plddt": 0.9}

        def passes_acceptance_gate(self, scores_detail):
            self.acceptance_scores_detail = dict(scores_detail)
            assert "transition_consistency_gate_score" in scores_detail
            assert "reference_gate_active" not in scores_detail
            assert "reference_tmscore" not in scores_detail
            return True

    decisions = _Decisions()
    gate_loss = _GateLoss()

    def _fake_prepare(evaluation_result, *, timing, loss_computer, runtime_config):
        runtime_config["reference_gate_result"] = ReferenceGateResult(
            gate_active=True,
            score=0.91,
            threshold=0.87,
            passed=True,
            comparisons={},
            per_oligo_tmscore={"AB": 0.91},
        )
        return evaluation_result

    monkeypatch.setattr(run_mod, "_prepare_structure_start_runtime", _fake_prepare)
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1"
        if int(global_config.step) == 0
        else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: passed_gate,
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=1),
        loss_computer=gate_loss,
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert result.names == ["AB"]
    assert stage_manager.current_stage == "stage_1"
    assert gate_loss.acceptance_scores_detail is not None
    assert decisions.recompute_runtime is not None
    assert "transition_consistency_gate_result" in decisions.recompute_runtime
    assert "reference_gate_result" not in decisions.recompute_runtime
    assert "reference_gate_result" not in runtime_config


def test_run_optimization_loop_blocks_seed_pass_on_acceptance_gate(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "acceptance_gate": "pair_hotspot_support >= 0.14",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_eval = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")
    passed_gate = SimpleNamespace(
        gate_active=True,
        score=0.91,
        threshold=0.8,
        passed=True,
        comparisons={"AB~A": 0.91},
    )

    class _Decisions:
        def __init__(self):
            self.recompute_called = False

        def make_mutation_decision(
            self, step, evaluation_result, loss_computer, runtime_config, force_accept=False,
            stage_entry=0.0,
        ):
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_called = True

    decisions = _Decisions()

    hard_gate_loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            0.0,
            {"pair_hotspot_support": 0.0},
        ),
        passes_acceptance_gate=lambda scores_detail: scores_detail[
            "pair_hotspot_support"
        ]
        >= 0.14,
    )

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    eval_results = iter([accepted_eval, gate_seed])
    monkeypatch.setattr(
        run_mod, "perform_evaluation", lambda *args, **kwargs: next(eval_results)
    )
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1" if int(global_config.step) == 1 else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: passed_gate,
    )

    runtime_config = {}
    with caplog.at_level(logging.INFO):
        result = run_mod.run_optimization_loop(
            args=SimpleNamespace(steps=2),
            loss_computer=hard_gate_loss,
            initial_evaluation_result=initial_eval,
            stage_manager=stage_manager,
            runtime_config=runtime_config,
        )

    assert decisions.recompute_called is False
    assert stage_manager.current_stage == "default"
    assert stage_manager.transition_consistency_gate_context is not None
    assert stage_manager.current_transition_input_scores is gate_seed
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert result is gate_seed
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "transition_consistency_gate" in message
        and "seed_reject reason=acceptance_gate" in message
        for message in messages
    )



def test_run_optimization_loop_preserves_source_reference_before_target_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["AB"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    canonical = tmp_path / "structures" / "step_001_AB_model.cif"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("SOURCE\n", encoding="utf-8")

    initial_eval = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": canonical},
    )
    accepted_eval = initial_eval.clone()

    class _Decisions:
        def make_mutation_decision(
            self, step, evaluation_result, loss_computer, runtime_config, force_accept=False,
            stage_entry=0.0,
        ):
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            return None

    decisions = _Decisions()

    def _perform_target_evaluation(*args, **kwargs):
        canonical.write_text("TARGET\n", encoding="utf-8")
        return StandardScores(
            names=["AB"],
            plddt={"AB": [1.0]},
            complex_paths={"AB": canonical},
        )

    eval_call_count = {"count": 0}

    def _perform_evaluation(*args, **kwargs):
        eval_call_count["count"] += 1
        if eval_call_count["count"] == 1:
            return accepted_eval
        return _perform_target_evaluation(*args, **kwargs)

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "perform_evaluation", _perform_evaluation)
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1" if int(global_config.step) == 1 else None,
    )

    def _evaluate_transition_consistency(eval_result):
        assert stage_manager.transition_consistency_gate_context is not None
        return compute_reference_consistency(
            current_scores=eval_result,
            reference_scores=stage_manager.transition_consistency_gate_context.reference_scores,
            threshold=0.8,
            compare_fn=lambda current_structure, reference_structure, **kwargs: (
                1.0
                if Path(current_structure).read_text(encoding="utf-8")
                == Path(reference_structure).read_text(encoding="utf-8")
                else 0.0
            ),
        )

    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        _evaluate_transition_consistency,
    )

    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=SimpleNamespace(compute_loss=lambda *args, **kwargs: (0.0, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config={},
    )

    assert stage_manager.current_stage == "default"
    assert stage_manager.transition_consistency_gate_context is not None
    assert result.names == ["AB"]


def test_run_optimization_loop_reevaluates_target_after_active_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    eval_default = object()
    eval_gate = object()
    eval_target = object()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const(
        "evaluators",
        {"Eval0": eval_default, "EvalGate": eval_gate, "EvalTarget": eval_target},
    )
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "EvalTarget",
            "transition_consistency_gate": {
                "evaluator": "EvalGate",
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_default = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")
    gate_attempt = _scores_at(tmp_path, "AB")
    target_eval = _scores_at(tmp_path, "AB")

    class _Decisions:
        def __init__(self):
            self.recompute_eval = None
            self.stage_during_decision = None

        def make_mutation_decision(
            self,
            step,
            evaluation_result,
            loss_computer,
            runtime_config,
            force_accept=False,
            stage_entry=0.0,
        ):
            self.stage_during_decision = stage_manager.current_stage
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_eval = evaluation_result

    decisions = _Decisions()
    eval_calls = []
    eval_runtime_keys = []
    eval_results = iter([accepted_default, gate_seed, gate_attempt, target_eval])

    def _perform_evaluation(_args, evaluator, runtime_config, *_rest, **_kwargs):
        eval_calls.append(evaluator)
        eval_runtime_keys.append((evaluator, set(runtime_config)))
        return next(eval_results)

    gate_calls = {"count": 0}

    def _evaluate_transition_consistency(eval_result):
        gate_calls["count"] += 1
        return SimpleNamespace(
            gate_active=True,
            score=0.91 if gate_calls["count"] == 2 else 0.4,
            threshold=0.8,
            passed=gate_calls["count"] == 2,
            comparisons={},
        )

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "perform_evaluation", _perform_evaluation)
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1"
        if int(global_config.step) == 1
        else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        _evaluate_transition_consistency,
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=SimpleNamespace(compute_loss=lambda *args, **kwargs: (0.0, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert eval_calls == [eval_default, eval_gate, eval_gate, eval_target]
    assert eval_runtime_keys[-1] == (eval_target, set())
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert decisions.stage_during_decision == "default"
    assert stage_manager.current_stage == "stage_1"
    assert decisions.recompute_eval is target_eval
    assert result is target_eval


def test_run_optimization_loop_records_target_reevaluation_after_active_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    loss_computer.compute_loss = lambda eval_result, runtime_config: (
        float(getattr(eval_result, "score")),
        {"mock_score": float(getattr(eval_result, "score"))},
    )
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: None,
            clear_mutations=lambda: None,
        ),
    )
    eval_default = object()
    eval_gate = object()
    eval_target = object()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const(
        "evaluators",
        {"Eval0": eval_default, "EvalGate": eval_gate, "EvalTarget": eval_target},
    )
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("use_metropolis_acceptance", False)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "mock_score",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "mock_score > 0.0",
            "evaluator": "EvalTarget",
            "transition_consistency_gate": {
                "evaluator": "EvalGate",
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    def _with_score(scores: StandardScores, score: float) -> StandardScores:
        scores.score = score
        return scores

    initial_eval = _with_score(_scores_at(tmp_path, "A"), 0.0)
    accepted_default = _with_score(_scores_at(tmp_path, "A"), 1.0)
    gate_seed = _with_score(_scores_at(tmp_path, "AB"), 2.0)
    gate_attempt = _with_score(_scores_at(tmp_path, "AB"), 3.0)
    target_eval = _with_score(_scores_at(tmp_path, "AB"), 4.0)
    eval_results = iter([accepted_default, gate_seed, gate_attempt, target_eval])

    eval_calls = []

    def _perform_evaluation(_args, evaluator, runtime_config, *_rest, **_kwargs):
        eval_calls.append(evaluator)
        return next(eval_results)

    gate_calls = {"count": 0}

    def _evaluate_transition_consistency(eval_result):
        gate_calls["count"] += 1
        return SimpleNamespace(
            gate_active=True,
            score=0.91 if gate_calls["count"] == 2 else 0.4,
            threshold=0.8,
            passed=gate_calls["count"] == 2,
            comparisons={},
        )

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "perform_evaluation", _perform_evaluation)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1"
        if int(global_config.step) == 1
        else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        _evaluate_transition_consistency,
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=loss_computer,
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    rows = list(csv.DictReader((tmp_path / "scores.csv").open("r", encoding="utf-8")))
    step2_rows = [row for row in rows if row["step"] == "2"]

    assert eval_calls == [eval_default, eval_gate, eval_gate, eval_target]
    assert len(step2_rows) == 2
    assert step2_rows[0]["stage"] == "default"
    assert step2_rows[0]["note"] == "enter stage"
    assert float(step2_rows[0]["weighted_score"]) == 3.0
    assert step2_rows[1]["stage"] == "stage_1"
    assert step2_rows[1]["note"] == "re-evaluate due to stage change"
    assert float(step2_rows[1]["weighted_score"]) == 4.0
    assert stage_manager.current_stage == "stage_1"
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert result is target_eval


def test_run_optimization_loop_commits_active_gate_without_decision_stage_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    eval_default = object()
    eval_stage = object()
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": eval_default, "Eval1": eval_stage})
    global_config.init_const("out_dir", tmp_path)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_default = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")
    gate_attempt = _scores_at(tmp_path, "AB")

    class _Decisions:
        def __init__(self):
            self.stage_during_gate_decision = None
            self.stage_history = {0: "default"}
            self.row_history = []
            self.dump_calls = 0
            self.recompute_called = False

        def make_mutation_decision(
            self,
            step,
            evaluation_result,
            loss_computer,
            runtime_config,
            force_accept=False,
            stage_entry=0.0,
        ):
            if "transition_consistency_gate_result" in runtime_config:
                self.stage_during_gate_decision = stage_manager.current_stage
                self.stage_history[step] = stage_manager.current_stage
                self.row_history.append({"step": step, "stage": stage_manager.current_stage})
            return False, evaluation_result, True

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_called = True

        def dump_results(self, out_dir, incremental=True):
            self.dump_calls += 1

    decisions = _Decisions()
    eval_calls = []
    eval_results = iter([accepted_default, gate_seed, gate_attempt])

    def _perform_evaluation(_args, evaluator, runtime_config, *_rest, **_kwargs):
        eval_calls.append(evaluator)
        return next(eval_results)

    gate_calls = {"count": 0}

    def _evaluate_transition_consistency(eval_result):
        gate_calls["count"] += 1
        return SimpleNamespace(
            gate_active=True,
            score=0.91 if gate_calls["count"] == 2 else 0.4,
            threshold=0.8,
            passed=gate_calls["count"] == 2,
            comparisons={},
        )

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "perform_evaluation", _perform_evaluation)
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1"
        if int(global_config.step) == 1
        else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        _evaluate_transition_consistency,
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=SimpleNamespace(compute_loss=lambda *args, **kwargs: (0.0, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert eval_calls == [eval_default, eval_stage, eval_stage]
    assert decisions.stage_during_gate_decision == "default"
    assert decisions.stage_history[2] == "stage_1"
    assert decisions.row_history[-1]["stage"] == "stage_1"
    assert decisions.dump_calls == 1
    assert decisions.recompute_called is False
    assert stage_manager.current_stage == "stage_1"
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert result is gate_attempt


def test_run_optimization_loop_keeps_active_gate_after_acceptance_gate_reject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loss_computer = _init_loss()
    loss_computer.compute_loss = lambda *_args, **_kwargs: (
        0.0,
        {"pair_hotspot_support": 0.0},
    )
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: None,
            clear_mutations=lambda: None,
        ),
    )
    global_config.init_const("mutators", {"MCMC": SimpleNamespace()})
    global_config.init_const("evaluators", {"Eval0": object(), "Eval1": object()})
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("use_metropolis_acceptance", False)

    stage_args = {
        "default": {
            "oligo": ["A"],
            "mutation_num": 1,
            "weighted_score_expression": "plddt",
            "evaluator": "Eval0",
            "mutators": ["MCMC"],
        },
        "stage_1": {
            "oligo": ["AB"],
            "_entrance": "plddt > 0.0",
            "evaluator": "Eval1",
            "acceptance_gate": "pair_hotspot_support >= 0.14",
            "transition_consistency_gate": {
                "consistency_threshold": 0.8,
                "max_attempts": 3,
            },
        },
    }
    stage_manager = StageManager(stage_args, loss_computer=loss_computer)
    global_config.init_const("stage_manager", stage_manager)

    initial_eval = _scores_at(tmp_path, "A")
    accepted_default = _scores_at(tmp_path, "A")
    gate_seed = _scores_at(tmp_path, "AB")
    gate_attempt = _scores_at(tmp_path, "AB")
    eval_results = iter([accepted_default, gate_seed, gate_attempt])

    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "perform_evaluation",
        lambda *args, **kwargs: next(eval_results),
    )
    monkeypatch.setattr(
        stage_manager,
        "find_entrance_stage",
        lambda evaluation_result, loss_computer: "stage_1"
        if int(global_config.step) == 1
        else None,
    )
    monkeypatch.setattr(
        stage_manager,
        "evaluate_transition_consistency",
        lambda eval_result: SimpleNamespace(
            gate_active=True,
            score=0.91,
            threshold=0.8,
            passed=True,
            comparisons={},
        ),
    )

    runtime_config = {}
    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=3),
        loss_computer=loss_computer,
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert stage_manager.current_stage == "default"
    assert stage_manager.transition_consistency_gate_context is not None
    assert stage_manager.transition_consistency_gate_attempts == 1
    assert "transition_consistency_gate_result" not in runtime_config
    assert "transition_consistency_gate_metadata" not in runtime_config
    assert result is gate_seed
