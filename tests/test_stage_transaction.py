from __future__ import annotations

from types import SimpleNamespace

from astrolabe.stage_transaction import (
    begin_active_transition_gate_attempt,
    begin_direct_stage_entry,
    begin_seed_transition_gate_attempt,
    begin_stage_entry_transaction,
    clear_transition_gate_runtime,
    commit_accepted_transition_gate_attempt,
    finish_active_transition_gate_decision,
    finish_direct_stage_entry_decision,
    finish_rejected_active_transition_gate_attempt,
    finish_rejected_seed_transition_gate_attempt,
    finish_seed_transition_gate_decision,
    mark_transition_gate_committed_stage,
    plan_stage_transaction,
    preserve_seed_transition_source,
    resolve_active_transition_gate_decision,
    resolve_already_accepted_direct_stage_entry_decision,
    resolve_seed_transition_gate_decision,
    rollback_direct_stage_entry,
    runtime_without_reference_gate,
    set_transition_gate_runtime,
    transition_gate_runtime_metadata,
)
from astrolabe.evaluators.utils import StandardScores


class _StageManager:
    def __init__(self) -> None:
        self.default_setting = SimpleNamespace(evaluator="Eval0", oligo=["A"])
        self.stage_args = {
            "default": self.default_setting,
            "stage_1": SimpleNamespace(evaluator="Eval1", oligo=["AB"]),
            "stage_2": SimpleNamespace(evaluator=None, oligo=None),
            "stage_3": SimpleNamespace(evaluator="Eval0", oligo=["A"]),
        }
        self.current_stage = "default"
        self.current_setting = self.default_setting
        self.transition_consistency_gate_attempts = 2
        self.transition_consistency_gate_context = SimpleNamespace(
            source_stage="default",
            target_stage="stage_1",
        )
        self.gate_result = SimpleNamespace(gate_active=True, score=0.9)
        self.evaluated_result = None
        self.current_transition_input_scores = SimpleNamespace(name="retry")
        self.rollback_scores = SimpleNamespace(name="rollback")
        self.fail_limit_reached = False
        self.record_failure_calls = 0
        self.abort_calls = 0
        self.activated = None
        self.registered_reference = None
        self.stage_set_calls = []
        self.complete_calls = 0
        self.entrance_stage = None

    def get_transition_consistency_gate(self, stage_name: str):
        if stage_name == "stage_1":
            return SimpleNamespace(evaluator="GateEval")
        if stage_name == "stage_3":
            return SimpleNamespace(evaluator=None)
        return None

    def find_entrance_stage(self, evaluation_result, loss_computer):
        return self.entrance_stage

    def evaluate_transition_consistency(self, evaluation_result):
        self.evaluated_result = evaluation_result
        return self.gate_result

    def activate_transition_consistency_gate(
        self,
        *,
        target_stage,
        previous_stage_result,
        previous_stage_oligo,
        gate_seed_scores,
        artifact_root,
        activation_step,
    ):
        self.activated = {
            "target_stage": target_stage,
            "previous_stage_result": previous_stage_result,
            "previous_stage_oligo": previous_stage_oligo,
            "gate_seed_scores": gate_seed_scores,
            "artifact_root": artifact_root,
            "activation_step": activation_step,
        }
        self.transition_consistency_gate_context = SimpleNamespace(
            source_stage=self.current_stage,
            target_stage=target_stage,
        )
        self.current_transition_input_scores = gate_seed_scores

    def register_stage_entry_reference(
        self,
        *,
        target_stage,
        previous_stage_result,
        previous_stage_oligo,
    ):
        self.registered_reference = {
            "target_stage": target_stage,
            "previous_stage_result": previous_stage_result,
            "previous_stage_oligo": previous_stage_oligo,
        }

    def set_stage(self, stage_name: str):
        self.stage_set_calls.append(stage_name)
        self.current_stage = stage_name
        self.current_setting = self.stage_args[stage_name]

    def complete_transition_consistency_gate(self) -> str:
        self.complete_calls += 1
        target_stage = self.transition_consistency_gate_context.target_stage
        self.set_stage(target_stage)
        self.transition_consistency_gate_context = None
        return target_stage

    def record_transition_consistency_gate_failure(self):
        self.record_failure_calls += 1
        return self.fail_limit_reached

    def abort_transition_consistency_gate(self):
        self.abort_calls += 1
        return self.rollback_scores


class _LegacyDecideEntranceStageManager:
    def __init__(self) -> None:
        self.default_setting = SimpleNamespace(evaluator="Eval0", oligo=["A"])
        self.stage_args = {
            "default": self.default_setting,
            "target": SimpleNamespace(evaluator="Eval1", oligo=["AB"]),
        }
        self.current_stage = "default"
        self.current_setting = self.default_setting
        self.stage_set_calls = []

    def decide_entrance(self, evaluation_result, loss_computer):
        self.set_stage("target")
        return "target"

    def set_stage(self, stage_name: str):
        self.stage_set_calls.append(stage_name)
        self.current_stage = stage_name
        self.current_setting = self.stage_args[stage_name]

    def get_transition_consistency_gate(self, stage_name: str):
        return None


def test_plan_stage_transaction_uses_gate_evaluator_and_target_oligo() -> None:
    plan = plan_stage_transaction(_StageManager(), "stage_1")

    assert plan.source_stage_name == "default"
    assert plan.target_stage_name == "stage_1"
    assert plan.source_evaluator_alias == "Eval0"
    assert plan.target_evaluator_alias == "Eval1"
    assert plan.gate_evaluator_alias == "GateEval"
    assert plan.source_oligo == ["A"]
    assert plan.target_oligo == ["AB"]
    assert plan.needs_stage_reevaluation is True
    assert plan.target_stage_has_gate is True


def test_plan_stage_transaction_falls_back_to_default_stage_values() -> None:
    plan = plan_stage_transaction(_StageManager(), "stage_2")

    assert plan.target_evaluator_alias == "Eval0"
    assert plan.gate_evaluator_alias == "Eval0"
    assert plan.target_oligo == ["A"]
    assert plan.needs_stage_reevaluation is False
    assert plan.target_stage_has_gate is False


def test_plan_stage_transaction_detects_gate_without_forcing_reevaluation() -> None:
    plan = plan_stage_transaction(_StageManager(), "stage_3")

    assert plan.target_evaluator_alias == "Eval0"
    assert plan.gate_evaluator_alias == "Eval0"
    assert plan.source_oligo == plan.target_oligo
    assert plan.needs_stage_reevaluation is False
    assert plan.target_stage_has_gate is True


def test_begin_stage_entry_transaction_returns_none_without_entrance() -> None:
    transaction = begin_stage_entry_transaction(
        _StageManager(),
        SimpleNamespace(name="candidate"),
        SimpleNamespace(),
        clone_result=lambda result: SimpleNamespace(name=f"{result.name}-clone"),
    )

    assert transaction is None


def test_begin_stage_entry_transaction_clones_and_plans_target() -> None:
    stage_manager = _StageManager()
    stage_manager.entrance_stage = "stage_1"
    candidate = SimpleNamespace(name="candidate")
    cloned = SimpleNamespace(name="candidate-clone")

    transaction = begin_stage_entry_transaction(
        stage_manager,
        candidate,
        SimpleNamespace(),
        clone_result=lambda result: cloned if result is candidate else None,
    )

    assert transaction is not None
    assert transaction.plan.target_stage_name == "stage_1"
    assert transaction.previous_stage_result is cloned
    assert transaction.target_stage_has_gate is True


def test_stage_entry_transaction_runtime_strips_reference_for_gate_entries() -> None:
    stage_manager = _StageManager()
    stage_manager.entrance_stage = "stage_1"
    transaction = begin_stage_entry_transaction(
        stage_manager,
        SimpleNamespace(name="candidate"),
        SimpleNamespace(),
        clone_result=lambda result: result,
    )
    runtime_config = {
        "reference_gate_result": object(),
        "transition_consistency_gate_result": object(),
        "step": 0,
    }

    assert transaction is not None
    decision_runtime = transaction.decision_runtime(runtime_config)

    assert sorted(decision_runtime) == ["step", "transition_consistency_gate_result"]
    assert "reference_gate_result" in runtime_config


def test_stage_entry_transaction_runtime_keeps_reference_for_accepted_direct_entry() -> None:
    stage_manager = _StageManager()
    stage_manager.entrance_stage = "stage_2"
    transaction = begin_stage_entry_transaction(
        stage_manager,
        SimpleNamespace(name="candidate"),
        SimpleNamespace(),
        clone_result=lambda result: result,
    )
    runtime_config = {"reference_gate_result": object(), "step": 0}

    assert transaction is not None
    assert transaction.decision_runtime(runtime_config) is runtime_config


def test_begin_stage_entry_transaction_restores_legacy_decide_entrance_mutation() -> None:
    stage_manager = _LegacyDecideEntranceStageManager()

    transaction = begin_stage_entry_transaction(
        stage_manager,
        SimpleNamespace(name="candidate"),
        SimpleNamespace(),
        clone_result=lambda result: result,
    )

    assert transaction is not None
    assert transaction.plan.source_stage_name == "default"
    assert transaction.plan.target_stage_name == "target"
    assert transaction.plan.source_oligo == ["A"]
    assert transaction.plan.target_oligo == ["AB"]
    assert transaction.plan.needs_stage_reevaluation is True
    assert stage_manager.current_stage == "default"
    assert stage_manager.stage_set_calls == ["target", "default"]


def test_transition_gate_runtime_metadata_prefers_active_context() -> None:
    metadata = transition_gate_runtime_metadata(
        _StageManager(),
        target_stage_name="fallback_stage",
    )

    assert metadata == {
        "source_stage": "default",
        "target_stage": "stage_1",
        "attempt": 3,
    }


def test_transition_gate_runtime_metadata_uses_target_fallback() -> None:
    stage_manager = _StageManager()
    stage_manager.transition_consistency_gate_context = None

    metadata = transition_gate_runtime_metadata(
        stage_manager,
        target_stage_name="fallback_stage",
    )

    assert metadata == {
        "source_stage": "",
        "target_stage": "fallback_stage",
        "attempt": 3,
    }


def test_clear_transition_gate_runtime_removes_only_gate_keys() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
        "reference_gate_result": object(),
    }

    clear_transition_gate_runtime(runtime_config)

    assert sorted(runtime_config) == ["reference_gate_result"]


def test_runtime_without_reference_gate_keeps_transition_gate_fields() -> None:
    result = object()
    runtime_config = {
        "reference_gate_result": object(),
        "transition_consistency_gate_result": result,
        "step": 4,
    }

    filtered = runtime_without_reference_gate(runtime_config)

    assert filtered == {
        "transition_consistency_gate_result": result,
        "step": 4,
    }
    assert "reference_gate_result" in runtime_config


def test_set_transition_gate_runtime_sets_result_and_metadata() -> None:
    runtime_config = {}
    result = SimpleNamespace(gate_active=True, score=0.9)

    metadata = set_transition_gate_runtime(
        runtime_config,
        _StageManager(),
        result,
        target_stage_name="fallback_stage",
    )

    assert runtime_config["transition_consistency_gate_result"] is result
    assert runtime_config["transition_consistency_gate_metadata"] == metadata
    assert metadata == {
        "source_stage": "default",
        "target_stage": "stage_1",
        "attempt": 3,
    }


def test_begin_active_transition_gate_attempt_sets_runtime_and_plan() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    evaluation_result = SimpleNamespace(name="attempt")

    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        evaluation_result,
    )

    assert stage_manager.evaluated_result is evaluation_result
    assert attempt.plan.target_stage_name == "stage_1"
    assert attempt.plan.gate_evaluator_alias == "GateEval"
    assert attempt.needs_target_reevaluation is True
    assert attempt.gate_result is stage_manager.gate_result
    assert (
        runtime_config["transition_consistency_gate_result"]
        is stage_manager.gate_result
    )
    assert runtime_config["transition_consistency_gate_metadata"] == {
        "source_stage": "default",
        "target_stage": "stage_1",
        "attempt": 3,
    }


def test_begin_seed_transition_gate_attempt_activates_and_sets_runtime(tmp_path) -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    source_path = tmp_path / "source_a.pdb"
    source_path.write_text("source\n", encoding="utf-8")
    previous_stage_result = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": source_path},
    )
    gate_seed_scores = SimpleNamespace(name="seed")

    preserved = preserve_seed_transition_source(
        plan,
        previous_stage_result=previous_stage_result,
        out_dir=tmp_path,
        step=7,
    )
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=preserved.previous_stage_result,
        gate_seed_scores=gate_seed_scores,
        artifact_root=preserved.artifact_root,
        step=7,
    )

    assert attempt.plan is plan
    assert attempt.gate_seed_scores is gate_seed_scores
    assert attempt.gate_result is stage_manager.gate_result
    assert stage_manager.evaluated_result is gate_seed_scores
    assert stage_manager.activated["target_stage"] == "stage_1"
    assert stage_manager.activated["previous_stage_oligo"] == ["A"]
    snapshot_path = stage_manager.activated["previous_stage_result"].complex_paths["A"]
    assert snapshot_path.exists()
    assert snapshot_path.parent.name == "source_stage"
    assert runtime_config["transition_consistency_gate_result"] is stage_manager.gate_result
    assert runtime_config["transition_consistency_gate_metadata"] == {
        "source_stage": "default",
        "target_stage": "stage_1",
        "attempt": 3,
    }


def test_commit_accepted_transition_gate_attempt_clears_runtime_before_target_reevaluation() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }
    stage_manager = _StageManager()
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )

    commit = commit_accepted_transition_gate_attempt(
        runtime_config,
        stage_manager,
        attempt,
    )

    assert commit.target_stage_name == "stage_1"
    assert commit.needs_target_reevaluation is True
    assert stage_manager.complete_calls == 1
    assert stage_manager.current_stage == "stage_1"
    assert runtime_config == {}


def test_commit_accepted_transition_gate_attempt_keeps_runtime_when_recompute_needs_gate_score() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    stage_manager.transition_consistency_gate_context = SimpleNamespace(
        source_stage="default",
        target_stage="stage_3",
    )
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )

    commit = commit_accepted_transition_gate_attempt(
        runtime_config,
        stage_manager,
        attempt,
    )

    assert commit.target_stage_name == "stage_3"
    assert commit.needs_target_reevaluation is False
    assert runtime_config["transition_consistency_gate_result"] is stage_manager.gate_result
    assert stage_manager.current_stage == "stage_3"


def test_mark_transition_gate_committed_stage_updates_stage_history_and_latest_row() -> None:
    decisions = SimpleNamespace(
        stage_history={1: "default"},
        row_history=[
            {"step": 1, "stage": "default"},
            {"step": 2, "stage": "default"},
        ],
    )

    mark_transition_gate_committed_stage(decisions, 2, "stage_3")

    assert decisions.stage_history[2] == "stage_3"
    assert decisions.row_history == [
        {"step": 1, "stage": "default"},
        {"step": 2, "stage": "stage_3"},
    ]


def test_finish_active_transition_gate_decision_requests_target_reevaluation() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }
    stage_manager = _StageManager()
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )
    gate_result = SimpleNamespace(name="attempt")

    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=4,
        evaluation_result=gate_result,
        is_accepted=True,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )

    assert decision.evaluation_result is gate_result
    assert decision.is_accepted is True
    assert decision.committed_stage_name == "stage_1"
    assert decision.needs_target_reevaluation is True
    assert decision.should_dump_results is False
    assert stage_manager.current_stage == "stage_1"
    assert runtime_config == {}


def test_finish_active_transition_gate_decision_marks_stage_without_target_reevaluation() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    stage_manager.transition_consistency_gate_context = SimpleNamespace(
        source_stage="default",
        target_stage="stage_3",
    )
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )
    decisions = SimpleNamespace(
        stage_history={2: "default"},
        row_history=[{"step": 2, "stage": "default"}],
    )

    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=2,
        evaluation_result=SimpleNamespace(name="attempt"),
        is_accepted=True,
        decisions=decisions,
    )

    assert decision.is_accepted is True
    assert decision.committed_stage_name == "stage_3"
    assert decision.needs_target_reevaluation is False
    assert decision.should_dump_results is True
    assert decisions.stage_history[2] == "stage_3"
    assert decisions.row_history == [{"step": 2, "stage": "stage_3"}]
    assert runtime_config == {}


def test_finish_active_transition_gate_decision_rejects_with_retry_input() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }
    stage_manager = _StageManager()
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )

    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=2,
        evaluation_result=SimpleNamespace(name="attempt"),
        is_accepted=False,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )

    assert decision.evaluation_result is stage_manager.current_transition_input_scores
    assert decision.is_accepted is False
    assert decision.committed_stage_name is None
    assert decision.needs_target_reevaluation is False
    assert decision.should_dump_results is False
    assert stage_manager.record_failure_calls == 1
    assert runtime_config == {}


def test_finish_active_transition_gate_decision_reject_abort_returns_rollback() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }
    stage_manager = _StageManager()
    stage_manager.fail_limit_reached = True
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )

    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=2,
        evaluation_result=SimpleNamespace(name="attempt"),
        is_accepted=False,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )

    assert decision.evaluation_result is stage_manager.rollback_scores
    assert decision.is_accepted is False
    assert decision.committed_stage_name is None
    assert decision.needs_target_reevaluation is False
    assert decision.should_dump_results is False
    assert stage_manager.record_failure_calls == 1
    assert stage_manager.abort_calls == 1
    assert runtime_config == {}


def test_resolve_active_transition_gate_decision_reevaluates_and_recomputes() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )
    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=4,
        evaluation_result=SimpleNamespace(name="gate"),
        is_accepted=True,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )
    target_result = SimpleNamespace(name="target")
    recomputed = []
    dumped = []

    result = resolve_active_transition_gate_decision(
        decision,
        evaluate_target=lambda: target_result,
        recompute_last_score=recomputed.append,
        dump_results=lambda: dumped.append(True),
    )

    assert result is target_result
    assert recomputed == [target_result]
    assert dumped == []


def test_resolve_active_transition_gate_decision_dumps_without_reevaluation() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    stage_manager.transition_consistency_gate_context = SimpleNamespace(
        source_stage="default",
        target_stage="stage_3",
    )
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )
    accepted_result = SimpleNamespace(name="accepted")
    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=4,
        evaluation_result=accepted_result,
        is_accepted=True,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )
    recomputed = []
    dumped = []

    result = resolve_active_transition_gate_decision(
        decision,
        evaluate_target=lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
        recompute_last_score=recomputed.append,
        dump_results=lambda: dumped.append(True),
    )

    assert result is accepted_result
    assert recomputed == []
    assert dumped == [True]


def test_resolve_active_transition_gate_decision_rejects_without_callbacks() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    attempt = begin_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="attempt"),
    )
    decision = finish_active_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        step=4,
        evaluation_result=SimpleNamespace(name="candidate"),
        is_accepted=False,
        decisions=SimpleNamespace(stage_history={}, row_history=[]),
    )
    recomputed = []
    dumped = []

    result = resolve_active_transition_gate_decision(
        decision,
        evaluate_target=lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
        recompute_last_score=recomputed.append,
        dump_results=lambda: dumped.append(True),
    )

    assert result is decision.evaluation_result
    assert recomputed == []
    assert dumped == []


def test_finish_rejected_seed_transition_gate_attempt_keeps_seed_input() -> None:
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }
    stage_manager = _StageManager()
    seed_scores = SimpleNamespace(name="seed")
    stage_manager.current_transition_input_scores = seed_scores

    result = finish_rejected_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="candidate"),
    )

    assert result is seed_scores
    assert runtime_config == {}


def test_finish_seed_transition_gate_decision_requests_target_reevaluation() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )

    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=seed_scores,
        is_accepted=True,
    )

    assert decision.evaluation_result is seed_scores
    assert decision.is_accepted is True
    assert decision.committed_stage_name == "stage_1"
    assert decision.needs_target_reevaluation is True
    assert stage_manager.current_stage == "stage_1"
    assert runtime_config == {}


def test_finish_seed_transition_gate_decision_keeps_runtime_for_recompute() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_3")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )

    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=seed_scores,
        is_accepted=True,
    )

    assert decision.evaluation_result is seed_scores
    assert decision.is_accepted is True
    assert decision.committed_stage_name == "stage_3"
    assert decision.needs_target_reevaluation is False
    assert stage_manager.current_stage == "stage_3"
    assert runtime_config["transition_consistency_gate_result"] is stage_manager.gate_result


def test_finish_seed_transition_gate_decision_rejects_with_seed_input() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )

    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=SimpleNamespace(name="candidate"),
        is_accepted=False,
    )

    assert decision.evaluation_result is seed_scores
    assert decision.is_accepted is False
    assert decision.committed_stage_name is None
    assert decision.needs_target_reevaluation is False
    assert stage_manager.current_stage == "default"
    assert runtime_config == {}


def test_resolve_seed_transition_gate_decision_reevaluates_recomputes_and_clears() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )
    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=seed_scores,
        is_accepted=True,
    )
    target_result = SimpleNamespace(name="target")
    recomputed = []

    result = resolve_seed_transition_gate_decision(
        runtime_config,
        decision,
        evaluate_target=lambda: target_result,
        recompute_last_score=recomputed.append,
    )

    assert result is target_result
    assert recomputed == [target_result]
    assert runtime_config == {}


def test_resolve_seed_transition_gate_decision_recomputes_seed_result_before_clear() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_3")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )
    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=seed_scores,
        is_accepted=True,
    )
    recompute_runtime_keys = []

    result = resolve_seed_transition_gate_decision(
        runtime_config,
        decision,
        evaluate_target=lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
        recompute_last_score=lambda result: recompute_runtime_keys.append(
            (result, sorted(runtime_config))
        ),
    )

    assert result is seed_scores
    assert recompute_runtime_keys == [
        (
            seed_scores,
            [
                "transition_consistency_gate_metadata",
                "transition_consistency_gate_result",
            ],
        )
    ]
    assert runtime_config == {}


def test_resolve_seed_transition_gate_decision_rejects_without_callbacks() -> None:
    runtime_config = {}
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    seed_scores = SimpleNamespace(name="seed")
    attempt = begin_seed_transition_gate_attempt(
        runtime_config,
        stage_manager,
        plan,
        previous_stage_result=SimpleNamespace(name="source"),
        gate_seed_scores=seed_scores,
        artifact_root=object(),
        step=3,
    )
    decision = finish_seed_transition_gate_decision(
        runtime_config,
        stage_manager,
        attempt,
        evaluation_result=SimpleNamespace(name="candidate"),
        is_accepted=False,
    )
    recomputed = []

    result = resolve_seed_transition_gate_decision(
        runtime_config,
        decision,
        evaluate_target=lambda: (_ for _ in ()).throw(AssertionError("unexpected")),
        recompute_last_score=recomputed.append,
    )

    assert result is seed_scores
    assert recomputed == []
    assert runtime_config == {}


def test_begin_direct_stage_entry_registers_sets_and_reevaluates() -> None:
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    previous_stage_result = SimpleNamespace(name="source")
    reevaluated = SimpleNamespace(name="target")

    attempt = begin_direct_stage_entry(
        stage_manager,
        plan,
        previous_stage_result=previous_stage_result,
        evaluate_target=lambda: reevaluated,
        clone_result=lambda _result: SimpleNamespace(name="clone"),
    )

    assert attempt.plan is plan
    assert attempt.evaluation_result is reevaluated
    assert stage_manager.registered_reference == {
        "target_stage": "stage_1",
        "previous_stage_result": previous_stage_result,
        "previous_stage_oligo": ["A"],
    }
    assert stage_manager.current_stage == "stage_1"
    assert stage_manager.stage_set_calls == ["stage_1"]


def test_begin_direct_stage_entry_clones_when_no_reevaluation_needed() -> None:
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_2")
    previous_stage_result = SimpleNamespace(name="source")
    cloned = SimpleNamespace(name="clone")
    evaluate_calls = {"count": 0}

    attempt = begin_direct_stage_entry(
        stage_manager,
        plan,
        previous_stage_result=previous_stage_result,
        evaluate_target=lambda: evaluate_calls.__setitem__("count", 1),
        clone_result=lambda _result: cloned,
    )

    assert attempt.evaluation_result is cloned
    assert evaluate_calls["count"] == 0
    assert stage_manager.current_stage == "stage_2"


def test_rollback_direct_stage_entry_returns_to_source_stage() -> None:
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    stage_manager.set_stage("stage_1")

    rollback_direct_stage_entry(stage_manager, plan)

    assert stage_manager.current_stage == "default"
    assert stage_manager.stage_set_calls == ["stage_1", "default"]


def test_finish_direct_stage_entry_decision_accepts_target_result() -> None:
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    previous_stage_result = SimpleNamespace(name="source")
    evaluation_result = SimpleNamespace(name="target")
    attempt = begin_direct_stage_entry(
        stage_manager,
        plan,
        previous_stage_result=previous_stage_result,
        evaluate_target=lambda: evaluation_result,
        clone_result=lambda result: result,
    )

    decision = finish_direct_stage_entry_decision(
        stage_manager,
        attempt,
        previous_stage_result=previous_stage_result,
        evaluation_result=evaluation_result,
        is_accepted=True,
    )

    assert decision.evaluation_result is evaluation_result
    assert decision.is_accepted is True
    assert decision.entered_stage_name == "stage_1"
    assert stage_manager.current_stage == "stage_1"


def test_finish_direct_stage_entry_decision_rejects_and_rolls_back() -> None:
    stage_manager = _StageManager()
    plan = plan_stage_transaction(stage_manager, "stage_1")
    previous_stage_result = SimpleNamespace(name="source")
    evaluation_result = SimpleNamespace(name="target")
    attempt = begin_direct_stage_entry(
        stage_manager,
        plan,
        previous_stage_result=previous_stage_result,
        evaluate_target=lambda: evaluation_result,
        clone_result=lambda result: result,
    )

    decision = finish_direct_stage_entry_decision(
        stage_manager,
        attempt,
        previous_stage_result=previous_stage_result,
        evaluation_result=evaluation_result,
        is_accepted=False,
    )

    assert decision.evaluation_result is previous_stage_result
    assert decision.is_accepted is False
    assert decision.entered_stage_name == "stage_1"
    assert stage_manager.current_stage == "default"
    assert stage_manager.stage_set_calls == ["stage_1", "default"]


def test_resolve_already_accepted_direct_stage_entry_decision_recomputes_on_accept() -> None:
    decision = SimpleNamespace(
        evaluation_result=SimpleNamespace(name="target"),
        is_accepted=True,
    )
    recomputed = []

    result = resolve_already_accepted_direct_stage_entry_decision(
        decision,
        recompute_last_score=recomputed.append,
    )

    assert result is decision.evaluation_result
    assert recomputed == [decision.evaluation_result]


def test_resolve_already_accepted_direct_stage_entry_decision_skips_recompute_on_reject() -> None:
    decision = SimpleNamespace(
        evaluation_result=SimpleNamespace(name="source"),
        is_accepted=False,
    )
    recomputed = []

    result = resolve_already_accepted_direct_stage_entry_decision(
        decision,
        recompute_last_score=recomputed.append,
    )

    assert result is decision.evaluation_result
    assert recomputed == []


def test_finish_rejected_active_transition_gate_attempt_keeps_retry_input() -> None:
    stage_manager = _StageManager()
    evaluation_result = SimpleNamespace(name="candidate")
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }

    result = finish_rejected_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        evaluation_result,
    )

    assert result is stage_manager.current_transition_input_scores
    assert stage_manager.record_failure_calls == 1
    assert stage_manager.abort_calls == 0
    assert runtime_config == {}


def test_finish_rejected_active_transition_gate_attempt_aborts_on_limit() -> None:
    stage_manager = _StageManager()
    stage_manager.fail_limit_reached = True
    runtime_config = {
        "transition_consistency_gate_result": object(),
        "transition_consistency_gate_metadata": {"target_stage": "stage_1"},
    }

    result = finish_rejected_active_transition_gate_attempt(
        runtime_config,
        stage_manager,
        SimpleNamespace(name="candidate"),
    )

    assert result is stage_manager.rollback_scores
    assert stage_manager.record_failure_calls == 1
    assert stage_manager.abort_calls == 1
    assert runtime_config == {}
