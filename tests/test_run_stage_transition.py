from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import run as run_mod
from astrolabe.core import global_config
from astrolabe.decision_context import passes_stage_entry_acceptance_gate
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.structure_start import ReferenceGateResult


class _EvalResult:
    def __init__(self, names, *, score: float = 0.0, iptm: float = 0.0):
        self.names = list(names)
        self.score = score
        self.iptm = iptm

    def clone(self):
        return _EvalResult(self.names, score=self.score, iptm=self.iptm)


def test_run_optimization_loop_evaluates_initial_result_for_stage_entry(
    monkeypatch,
) -> None:
    global_config.reset()

    events: list[tuple] = []
    initial_eval = _EvalResult(["A"], score=1.0, iptm=0.9)
    loop_eval = _EvalResult(["A"], score=1.1, iptm=0.95)

    class _FakeDecisions:
        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            events.append(("recompute", int(global_config.step), tuple(evaluation_result.names)))

        def make_mutation_decision(
            self,
            step,
            eval_result,
            loss_computer,
            runtime_config,
            force_accept: bool = False,
            stage_entry: float = 0.0,
        ):
            events.append(("decision", int(step), tuple(eval_result.names), bool(force_accept)))
            return False, eval_result, False

    class _FakeStageManager:
        def __init__(self) -> None:
            self.default_setting = SimpleNamespace(
                evaluator="Eval0",
                oligo=["A"],
                mutators=["default-mutator"],
            )
            self.stage_args = {
                "default": self.default_setting,
                "stage_1": SimpleNamespace(
                    evaluator="Eval0",
                    oligo=["A"],
                    mutators=["target-mutator"],
                ),
            }
            self.set_stage("default")

        def set_stage(self, stage_name):
            events.append(("set_stage", stage_name))
            self.current_stage = stage_name
            self.current_setting = self.stage_args[stage_name]
            self.current_evaluator = self.current_setting.evaluator
            self.current_oligo = self.current_setting.oligo
            self.current_mutators = self.current_setting.mutators

        def find_entrance_stage(self, eval_result, loss_computer):
            events.append(("find", int(global_config.step), tuple(eval_result.names)))
            return "stage_1" if int(global_config.step) == 0 else None

        def get_transition_consistency_gate(self, stage_name):
            return None

        def resolve_stage_evaluator_instance(self, stage_unit):
            return stage_unit.evaluator

    stage_manager = _FakeStageManager()

    monkeypatch.setattr(
        run_mod,
        "Decisions",
        SimpleNamespace(create=lambda *_args, **_kwargs: _FakeDecisions()),
    )
    monkeypatch.setattr(
        run_mod,
        "execute_mutation_step",
        lambda mutators, *_args, **_kwargs: events.append(("mutate", tuple(mutators))),
    )
    monkeypatch.setattr(
        run_mod,
        "perform_evaluation",
        lambda *_args, **_kwargs: loop_eval,
    )

    run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=SimpleNamespace(compute_loss=lambda result, _runtime: (result.score, {})),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config={},
    )

    assert events.index(("set_stage", "stage_1")) < events.index(
        ("mutate", ("target-mutator",))
    )
    assert ("recompute", 0, ("A",)) in events


def test_initial_stage_entry_can_use_step0_reference_gate_context(
    monkeypatch,
    tmp_path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)

    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": tmp_path / "ab.cif"},
    )

    class _FakeDecisions:
        def __init__(self) -> None:
            self.recompute_runtime = None

        def dump_results(self, *_args, **_kwargs):
            return None

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            self.recompute_runtime = dict(runtime_config)

    decisions = _FakeDecisions()

    class _FakeStageManager:
        def __init__(self) -> None:
            self.default_setting = SimpleNamespace(evaluator="Eval0", oligo=["AB"])
            self.stage_args = {
                "default": self.default_setting,
                "stage_1": SimpleNamespace(evaluator="Eval0", oligo=["AB"]),
            }
            self.set_stage("default")

        def set_stage(self, stage_name):
            self.current_stage = stage_name
            self.current_setting = self.stage_args[stage_name]
            self.current_evaluator = self.current_setting.evaluator
            self.current_oligo = self.current_setting.oligo
            self.current_mutators = []

        def find_entrance_stage(self, eval_result, loss_computer):
            return "stage_1" if int(global_config.step) == 0 else None

        def get_transition_consistency_gate(self, stage_name):
            return None

        def resolve_stage_evaluator_instance(self, stage_unit):
            return stage_unit.evaluator

    stage_manager = _FakeStageManager()

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

    loss_computer = LossEngine({"loss_terms": {}}, validate_config=False)
    loss_computer.setup(
        weighted_score_expression="reference_tmscore",
        early_stopping=None,
        acceptance_threshold=None,
        acceptance_gate="reference_tmscore >= 0.87",
        validate_config=False,
    )

    monkeypatch.setattr(run_mod, "_prepare_structure_start_runtime", _fake_prepare)
    monkeypatch.setattr(run_mod.Decisions, "create", lambda *_args, **_kwargs: decisions)

    runtime_config = {}
    run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=1),
        loss_computer=loss_computer,
        initial_evaluation_result=eval_result,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert decisions.recompute_runtime is not None
    assert "reference_gate_result" in decisions.recompute_runtime
    assert "reference_gate_result" not in runtime_config
    assert stage_manager.current_stage == "stage_1"


def test_current_acceptance_gate_supplies_self_delta_context_for_stage_entry() -> None:
    loss_computer = LossEngine({"loss_terms": {}}, validate_config=False)
    loss_computer.setup(
        weighted_score_expression="iptm",
        early_stopping=None,
        acceptance_threshold=None,
        acceptance_gate=(
            "(stage_entry > 0) and (cross_contact_lis > 0) and "
            "((previous_iptm < 0.30) or (delta_iptm >= -0.02))"
        ),
        validate_config=False,
    )

    loss_computer.compute_loss = lambda _eval_result, _runtime_config: (
        0.62,
        {"iptm": 0.62, "cross_contact_lis": 1.0},
    )

    assert passes_stage_entry_acceptance_gate(
        eval_result=SimpleNamespace(),
        runtime_config={},
        loss_computer=loss_computer,
    )

    loss_computer.compute_loss = lambda _eval_result, _runtime_config: (
        0.62,
        {"iptm": 0.62, "cross_contact_lis": 0.0},
    )

    assert not passes_stage_entry_acceptance_gate(
        eval_result=SimpleNamespace(),
        runtime_config={},
        loss_computer=loss_computer,
    )


def test_run_optimization_loop_preserves_source_row_for_direct_stage_reevaluation(
    monkeypatch,
    tmp_path,
) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(apply_mutations=lambda: None, clear_mutations=lambda: None),
    )
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                type=SimpleNamespace(value="p"),
                sequence="AAAA",
                mutated_sequence="AAAT",
            )
        },
    )
    global_config.init_const("out_dir", tmp_path)

    initial_eval = _EvalResult(["A"], score=0.1, iptm=0.1)
    source_eval = _EvalResult(["A"], score=2.5, iptm=0.71)
    target_eval = _EvalResult(["AB"], score=0.8, iptm=0.075)

    class _FakeStageManager:
        def __init__(self) -> None:
            self.default_setting = SimpleNamespace(evaluator="Eval0", oligo=["A"])
            self.stage_args = {
                "default": self.default_setting,
                "target_protenix": SimpleNamespace(evaluator="Eval1", oligo=["AB"]),
            }
            self.set_stage("default")

        def set_stage(self, stage_name):
            self.current_stage = stage_name
            self.current_setting = self.stage_args[stage_name]
            self.current_evaluator = self.current_setting.evaluator
            self.current_oligo = self.current_setting.oligo
            self.current_mutators = []

        def find_entrance_stage(self, eval_result, loss_computer):
            return "target_protenix" if int(global_config.step) == 1 else None

        def get_transition_consistency_gate(self, stage_name):
            return None

        def resolve_stage_evaluator_instance(self, stage_unit):
            return stage_unit.evaluator

    stage_manager = _FakeStageManager()
    global_config.init_const("stage_manager", stage_manager)

    class _Loss:
        def compute_loss(self, eval_result, _runtime_config):
            return eval_result.score, {"iptm": eval_result.iptm}

        def is_early_stopping(self, _scores_detail):
            return False

        def is_soft_accepted(self, _scores_detail):
            return False

        def passes_acceptance_gate(self, _scores_detail):
            return True

    eval_results = iter([source_eval, target_eval])
    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "perform_evaluation",
        lambda *_args, **_kwargs: next(eval_results),
    )

    run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=_Loss(),
        initial_evaluation_result=initial_eval,
        stage_manager=stage_manager,
        runtime_config={},
    )

    rows = list(csv.DictReader((tmp_path / "scores.csv").open("r", encoding="utf-8")))
    step_rows = [row for row in rows if row["step"] == "1"]

    assert len(step_rows) == 2
    source_row, target_row = step_rows
    assert source_row["stage"] == "default"
    assert source_row["note"] == "enter stage"
    assert float(source_row["weighted_score"]) == 2.5
    assert float(source_row["iptm"]) == 0.71

    assert target_row["stage"] == "target_protenix"
    assert target_row["note"] == "re-evaluate due to stage change"
    assert float(target_row["weighted_score"]) == 0.8
    assert float(target_row["iptm"]) == 0.075


def test_run_optimization_loop_does_not_enter_stage_from_rejected_candidate(
    monkeypatch,
) -> None:
    global_config.reset()

    decision_calls: list[tuple[int, tuple[str, ...], bool]] = []
    stage_calls: list[tuple[str, ...]] = []

    eval_default = _EvalResult(["A"])
    eval_candidate = _EvalResult(["AB"])

    class _FakeDecisions:
        def make_mutation_decision(
            self,
            step,
            eval_result,
            loss_computer,
            runtime_config,
            force_accept: bool = False,
            stage_entry: float = 0.0,
        ):
            decision_calls.append((int(step), tuple(eval_result.names), bool(force_accept)))
            return False, eval_default, False

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            return None

    class _FakeStageManager:
        def __init__(self) -> None:
            self.current_stage = "default"
            self.default_setting = SimpleNamespace(evaluator="DefaultEval", oligo=["A"])
            self.last_setting = self.default_setting
            self.current_setting = self.default_setting
            self.stage_args = {
                "default": self.default_setting,
                "target": SimpleNamespace(evaluator="TargetEval", oligo=["AB"]),
            }
            self.current_mutators = []
            self.current_evaluator = "DefaultEval"
            self.current_oligo = ["A"]

        def find_entrance_stage(self, eval_result, loss_computer):
            stage_calls.append(tuple(eval_result.names))
            if tuple(eval_result.names) == ("AB",):
                self.last_setting = self.current_setting
                self.current_setting = self.stage_args["target"]
                self.current_stage = "target"
                self.current_evaluator = "TargetEval"
                self.current_oligo = ["AB"]
                return "target"
            return None

        def set_stage(self, stage_name):
            self.last_setting = self.current_setting
            self.current_setting = self.stage_args[stage_name]
            self.current_stage = stage_name
            self.current_evaluator = self.current_setting.evaluator
            self.current_oligo = self.current_setting.oligo

        def decide_entrance(self, eval_result, loss_computer):
            stage_name = self.find_entrance_stage(eval_result, loss_computer)
            if stage_name:
                self.set_stage(stage_name)
            return stage_name

        def get_transition_consistency_gate(self, stage_name):
            return None

        def resolve_stage_evaluator_instance(self, stage_unit):
            return stage_unit.evaluator

    stage_manager = _FakeStageManager()

    monkeypatch.setattr(
        run_mod,
        "Decisions",
        SimpleNamespace(create=lambda *_args, **_kwargs: _FakeDecisions()),
    )
    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "perform_evaluation",
        lambda *args, **kwargs: eval_candidate,
    )

    runtime_config = {}
    run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=SimpleNamespace(compute_loss=lambda *_args, **_kwargs: (0.0, {})),
        initial_evaluation_result=eval_default,
        stage_manager=stage_manager,
        runtime_config=runtime_config,
    )

    assert stage_manager.current_stage == "default"
    assert all(names == ("A",) for names in stage_calls)
    assert all(force_accept is False for _, _, force_accept in decision_calls)


def test_run_optimization_loop_keeps_source_stage_when_decision_rejects_candidate(
    monkeypatch,
) -> None:
    global_config.reset()

    eval_default = _EvalResult(["A"])
    eval_target = _EvalResult(["AB"])

    class _FakeDecisions:
        def make_mutation_decision(
            self,
            step,
            eval_result,
            loss_computer,
            runtime_config,
            force_accept: bool = False,
            stage_entry: float = 0.0,
        ):
            return False, eval_default, False

        def recompute_last_score(self, loss_computer, runtime_config, evaluation_result):
            return None

    class _FakeStageManager:
        def __init__(self) -> None:
            self.current_stage = "default"
            self.current_evaluator = "Protenix"
            self.current_oligo = ["A"]
            self.current_mutators = []
            self.default_setting = SimpleNamespace(evaluator="Protenix", oligo=["A"])
            self.current_setting = self.default_setting
            self.stage_args = {
                "default": self.default_setting,
                "target_protenix": SimpleNamespace(
                    evaluator="Protenix",
                    oligo=["AB"],
                ),
            }

        def find_entrance_stage(self, eval_result, loss_computer):
            if int(global_config.step) != 1:
                return None
            return "target_protenix" if tuple(eval_result.names) == ("AB",) else None

        def decide_entrance(self, eval_result, loss_computer):
            stage_name = self.find_entrance_stage(eval_result, loss_computer)
            if stage_name:
                self.set_stage(stage_name)
            return stage_name

        def set_stage(self, stage_name):
            self.current_stage = stage_name
            self.current_setting = self.stage_args[stage_name]
            self.current_oligo = (
                ["A"] if stage_name == "default" else self.stage_args[stage_name].oligo
            )

        def get_transition_consistency_gate(self, stage_name):
            return None

        def resolve_stage_evaluator_instance(self, stage_unit):
            return stage_unit.evaluator

    stage_manager = _FakeStageManager()

    monkeypatch.setattr(
        run_mod,
        "Decisions",
        SimpleNamespace(create=lambda *_args, **_kwargs: _FakeDecisions()),
    )
    monkeypatch.setattr(run_mod, "execute_mutation_step", lambda *_args, **_kwargs: None)

    def _fake_perform_evaluation(
        args,
        evaluator,
        runtime_config,
        step,
        oligo_names,
        **_kwargs,
    ):
        if tuple(oligo_names) == ("A",):
            return eval_default
        if tuple(oligo_names) == ("AB",):
            return eval_target
        raise AssertionError(f"Unexpected oligo_names: {oligo_names}")

    monkeypatch.setattr(run_mod, "perform_evaluation", _fake_perform_evaluation)

    result = run_mod.run_optimization_loop(
        args=SimpleNamespace(steps=2),
        loss_computer=SimpleNamespace(compute_loss=lambda *_args, **_kwargs: (0.0, {})),
        initial_evaluation_result=eval_default,
        stage_manager=stage_manager,
        runtime_config={},
    )

    assert stage_manager.current_stage == "default"
    assert result is eval_default
