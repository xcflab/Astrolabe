from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


REFERENCE_GATE_RESULT_KEY = "reference_gate_result"
TRANSITION_GATE_RESULT_KEY = "transition_consistency_gate_result"
TRANSITION_GATE_METADATA_KEY = "transition_consistency_gate_metadata"


@dataclass(frozen=True)
class StageTransactionPlan:
    source_stage_name: str | None
    target_stage_name: str
    source_stage_unit: Any
    target_stage_unit: Any
    source_evaluator_alias: str | None
    target_evaluator_alias: str | None
    gate_evaluator_alias: str | None
    source_oligo: list[str]
    target_oligo: list[str]
    needs_stage_reevaluation: bool
    target_stage_has_gate: bool


@dataclass(frozen=True)
class StageEntryTransaction:
    plan: StageTransactionPlan
    previous_stage_result: Any

    @property
    def target_stage_name(self) -> str:
        return self.plan.target_stage_name

    @property
    def target_stage_has_gate(self) -> bool:
        return self.plan.target_stage_has_gate

    def decision_runtime(self, runtime_config: dict) -> dict:
        if not self.target_stage_has_gate:
            return runtime_config
        return runtime_without_reference_gate(runtime_config)


@dataclass(frozen=True)
class ActiveTransitionGateAttempt:
    plan: StageTransactionPlan
    gate_result: Any
    metadata: dict[str, object]

    @property
    def needs_target_reevaluation(self) -> bool:
        return self.plan.gate_evaluator_alias != self.plan.target_evaluator_alias


@dataclass(frozen=True)
class SeedTransitionGateAttempt:
    plan: StageTransactionPlan
    gate_seed_scores: Any
    gate_result: Any
    metadata: dict[str, object]

    @property
    def needs_target_reevaluation(self) -> bool:
        return self.plan.gate_evaluator_alias != self.plan.target_evaluator_alias


@dataclass(frozen=True)
class PreservedSeedTransitionSource:
    previous_stage_result: Any
    artifact_root: Any


@dataclass(frozen=True)
class DirectStageEntryAttempt:
    plan: StageTransactionPlan
    evaluation_result: Any


@dataclass(frozen=True)
class DirectStageEntryDecision:
    evaluation_result: Any
    is_accepted: bool
    entered_stage_name: str | None = None


@dataclass(frozen=True)
class AcceptedTransitionGateCommit:
    target_stage_name: str
    needs_target_reevaluation: bool


@dataclass(frozen=True)
class ActiveTransitionGateDecision:
    evaluation_result: Any
    is_accepted: bool
    committed_stage_name: str | None = None
    needs_target_reevaluation: bool = False
    should_dump_results: bool = False


@dataclass(frozen=True)
class SeedTransitionGateDecision:
    evaluation_result: Any
    is_accepted: bool
    committed_stage_name: str | None = None
    needs_target_reevaluation: bool = False


def resolve_stage_evaluator_alias(stage_manager: Any, stage_unit: Any) -> str | None:
    resolver = getattr(stage_manager, "resolve_stage_evaluator", None)
    if callable(resolver):
        return resolver(stage_unit)
    default_stage = stage_manager.default_setting
    return getattr(stage_unit, "evaluator", None) or getattr(default_stage, "evaluator", None)


def resolve_stage_oligo(stage_manager: Any, stage_unit: Any) -> list[str]:
    resolver = getattr(stage_manager, "resolve_stage_oligo", None)
    if callable(resolver):
        return list(resolver(stage_unit))
    default_stage = stage_manager.default_setting
    oligo = getattr(stage_unit, "oligo", None)
    if oligo is None:
        oligo = getattr(default_stage, "oligo", None)
    return list(oligo or [])


def resolve_transition_gate_evaluator_alias(
    stage_manager: Any,
    stage_name: str,
    fallback_alias: str | None,
) -> str | None:
    resolver = getattr(stage_manager, "resolve_transition_consistency_gate_evaluator", None)
    if callable(resolver):
        return resolver(stage_name)
    transition_gate = stage_manager.get_transition_consistency_gate(stage_name)
    if transition_gate is not None and getattr(transition_gate, "evaluator", None):
        return transition_gate.evaluator
    return fallback_alias


def plan_stage_transaction(
    stage_manager: Any,
    target_stage_name: str,
) -> StageTransactionPlan:
    source_stage_unit = stage_manager.current_setting
    source_stage_name = getattr(stage_manager, "current_stage", None)
    target_stage_unit = stage_manager.stage_args[target_stage_name]
    source_evaluator_alias = resolve_stage_evaluator_alias(
        stage_manager,
        source_stage_unit,
    )
    target_evaluator_alias = resolve_stage_evaluator_alias(
        stage_manager,
        target_stage_unit,
    )
    gate_evaluator_alias = resolve_transition_gate_evaluator_alias(
        stage_manager,
        target_stage_name,
        target_evaluator_alias,
    )
    source_oligo = resolve_stage_oligo(stage_manager, source_stage_unit)
    target_oligo = resolve_stage_oligo(stage_manager, target_stage_unit)
    return StageTransactionPlan(
        source_stage_name=source_stage_name,
        target_stage_name=target_stage_name,
        source_stage_unit=source_stage_unit,
        target_stage_unit=target_stage_unit,
        source_evaluator_alias=source_evaluator_alias,
        target_evaluator_alias=target_evaluator_alias,
        gate_evaluator_alias=gate_evaluator_alias,
        source_oligo=source_oligo,
        target_oligo=target_oligo,
        needs_stage_reevaluation=(
            source_evaluator_alias != gate_evaluator_alias
            or source_oligo != target_oligo
        ),
        target_stage_has_gate=(
            stage_manager.get_transition_consistency_gate(target_stage_name) is not None
        ),
    )


def find_stage_entry_target(
    stage_manager: Any,
    evaluation_result: Any,
    loss_computer: Any,
) -> str | None:
    finder = getattr(stage_manager, "find_entrance_stage", None)
    if callable(finder):
        return finder(evaluation_result, loss_computer)
    return stage_manager.decide_entrance(evaluation_result, loss_computer)


def begin_stage_entry_transaction(
    stage_manager: Any,
    evaluation_result: Any,
    loss_computer: Any,
    *,
    clone_result: Callable[[Any], Any],
) -> StageEntryTransaction | None:
    source_stage_name = getattr(stage_manager, "current_stage", None)
    target_stage_name = find_stage_entry_target(
        stage_manager,
        evaluation_result,
        loss_computer,
    )
    if not target_stage_name:
        return None
    if (
        source_stage_name is not None
        and getattr(stage_manager, "current_stage", None) != source_stage_name
    ):
        setter = getattr(stage_manager, "set_stage", None)
        if not callable(setter):
            raise RuntimeError(
                "Stage entry target discovery changed the current stage but the "
                "stage manager cannot restore the source stage."
            )
        setter(source_stage_name)
    return StageEntryTransaction(
        plan=plan_stage_transaction(stage_manager, target_stage_name),
        previous_stage_result=clone_result(evaluation_result),
    )


def transition_gate_runtime_metadata(
    stage_manager: Any,
    *,
    target_stage_name: str | None = None,
) -> dict[str, object]:
    gate_context = getattr(stage_manager, "transition_consistency_gate_context", None)
    return {
        "source_stage": getattr(gate_context, "source_stage", None) or "",
        "target_stage": (
            getattr(gate_context, "target_stage", None)
            or target_stage_name
            or ""
        ),
        "attempt": int(
            getattr(stage_manager, "transition_consistency_gate_attempts", 0)
        )
        + 1,
    }


def runtime_without_reference_gate(runtime_config: dict) -> dict:
    return {
        key: value
        for key, value in dict(runtime_config).items()
        if key != REFERENCE_GATE_RESULT_KEY
    }


def set_transition_gate_runtime(
    runtime_config: dict,
    stage_manager: Any,
    result: Any,
    *,
    target_stage_name: str | None = None,
) -> dict[str, object]:
    runtime_config[TRANSITION_GATE_RESULT_KEY] = result
    metadata = transition_gate_runtime_metadata(
        stage_manager,
        target_stage_name=target_stage_name,
    )
    runtime_config[TRANSITION_GATE_METADATA_KEY] = metadata
    return metadata


def begin_active_transition_gate_attempt(
    runtime_config: dict,
    stage_manager: Any,
    evaluation_result: Any,
) -> ActiveTransitionGateAttempt:
    gate_context = getattr(stage_manager, "transition_consistency_gate_context", None)
    target_stage_name = getattr(gate_context, "target_stage", None)
    if target_stage_name is None:
        raise RuntimeError(
            "Active transition-consistency gate has no target stage context."
        )
    plan = plan_stage_transaction(stage_manager, target_stage_name)
    gate_result = stage_manager.evaluate_transition_consistency(evaluation_result)
    metadata = set_transition_gate_runtime(
        runtime_config,
        stage_manager,
        gate_result,
        target_stage_name=target_stage_name,
    )
    return ActiveTransitionGateAttempt(
        plan=plan,
        gate_result=gate_result,
        metadata=metadata,
    )


def begin_seed_transition_gate_attempt(
    runtime_config: dict,
    stage_manager: Any,
    plan: StageTransactionPlan,
    *,
    previous_stage_result: Any,
    gate_seed_scores: Any,
    artifact_root: Any,
    step: int,
) -> SeedTransitionGateAttempt:
    stage_manager.activate_transition_consistency_gate(
        target_stage=plan.target_stage_name,
        previous_stage_result=previous_stage_result,
        previous_stage_oligo=plan.source_oligo,
        gate_seed_scores=gate_seed_scores,
        artifact_root=artifact_root,
        activation_step=step,
    )
    gate_result = stage_manager.evaluate_transition_consistency(gate_seed_scores)
    metadata = set_transition_gate_runtime(
        runtime_config,
        stage_manager,
        gate_result,
        target_stage_name=plan.target_stage_name,
    )
    return SeedTransitionGateAttempt(
        plan=plan,
        gate_seed_scores=gate_seed_scores,
        gate_result=gate_result,
        metadata=metadata,
    )


def preserve_seed_transition_source(
    plan: StageTransactionPlan,
    *,
    previous_stage_result: Any,
    out_dir: Any,
    step: int,
) -> PreservedSeedTransitionSource:
    from astrolabe.stage_transition import (
        build_transition_gate_artifact_root,
        snapshot_standard_scores,
    )

    artifact_root = build_transition_gate_artifact_root(
        out_dir=out_dir,
        step=step,
        source_stage=plan.source_stage_name,
        target_stage=plan.target_stage_name,
    )
    preserved_previous_stage_result = snapshot_standard_scores(
        previous_stage_result,
        output_dir=artifact_root / "source_stage",
    )
    return PreservedSeedTransitionSource(
        previous_stage_result=preserved_previous_stage_result,
        artifact_root=artifact_root,
    )


def begin_direct_stage_entry(
    stage_manager: Any,
    plan: StageTransactionPlan,
    *,
    previous_stage_result: Any,
    evaluate_target: Callable[[], Any],
    clone_result: Callable[[Any], Any],
) -> DirectStageEntryAttempt:
    if hasattr(stage_manager, "register_stage_entry_reference"):
        stage_manager.register_stage_entry_reference(
            target_stage=plan.target_stage_name,
            previous_stage_result=previous_stage_result,
            previous_stage_oligo=plan.source_oligo,
        )
    stage_manager.set_stage(plan.target_stage_name)
    if plan.needs_stage_reevaluation:
        evaluation_result = evaluate_target()
    else:
        evaluation_result = clone_result(previous_stage_result)
    return DirectStageEntryAttempt(
        plan=plan,
        evaluation_result=evaluation_result,
    )


def rollback_direct_stage_entry(stage_manager: Any, plan: StageTransactionPlan) -> None:
    if (
        plan.source_stage_name
        and getattr(stage_manager, "current_stage", None) != plan.source_stage_name
    ):
        stage_manager.set_stage(plan.source_stage_name)


def finish_direct_stage_entry_decision(
    stage_manager: Any,
    attempt: DirectStageEntryAttempt,
    *,
    previous_stage_result: Any,
    evaluation_result: Any,
    is_accepted: bool,
) -> DirectStageEntryDecision:
    if not is_accepted:
        rollback_direct_stage_entry(stage_manager, attempt.plan)
        return DirectStageEntryDecision(
            evaluation_result=previous_stage_result,
            is_accepted=False,
            entered_stage_name=attempt.plan.target_stage_name,
        )
    return DirectStageEntryDecision(
        evaluation_result=evaluation_result,
        is_accepted=True,
        entered_stage_name=attempt.plan.target_stage_name,
    )


def resolve_already_accepted_direct_stage_entry_decision(
    decision: DirectStageEntryDecision,
    *,
    recompute_last_score: Callable[[Any], None],
) -> Any:
    if decision.is_accepted:
        recompute_last_score(decision.evaluation_result)
    return decision.evaluation_result


def commit_accepted_transition_gate_attempt(
    runtime_config: dict,
    stage_manager: Any,
    attempt: ActiveTransitionGateAttempt | SeedTransitionGateAttempt,
) -> AcceptedTransitionGateCommit:
    target_stage_name = stage_manager.complete_transition_consistency_gate()
    if attempt.needs_target_reevaluation:
        clear_transition_gate_runtime(runtime_config)
    return AcceptedTransitionGateCommit(
        target_stage_name=target_stage_name,
        needs_target_reevaluation=attempt.needs_target_reevaluation,
    )


def mark_transition_gate_committed_stage(
    decisions: Any,
    step: int,
    stage_name: str,
) -> None:
    if hasattr(decisions, "stage_history"):
        decisions.stage_history[step] = stage_name
    for row in reversed(getattr(decisions, "row_history", []) or []):
        if int(row.get("step", -1)) == int(step):
            row["stage"] = stage_name
            return


def finish_active_transition_gate_decision(
    runtime_config: dict,
    stage_manager: Any,
    attempt: ActiveTransitionGateAttempt,
    *,
    step: int,
    evaluation_result: Any,
    is_accepted: bool,
    decisions: Any,
) -> ActiveTransitionGateDecision:
    if not is_accepted:
        return ActiveTransitionGateDecision(
            evaluation_result=finish_rejected_active_transition_gate_attempt(
                runtime_config,
                stage_manager,
                evaluation_result,
            ),
            is_accepted=False,
        )

    commit = commit_accepted_transition_gate_attempt(
        runtime_config,
        stage_manager,
        attempt,
    )
    if not commit.needs_target_reevaluation:
        mark_transition_gate_committed_stage(
            decisions,
            step,
            commit.target_stage_name,
        )
    clear_transition_gate_runtime(runtime_config)
    return ActiveTransitionGateDecision(
        evaluation_result=evaluation_result,
        is_accepted=True,
        committed_stage_name=commit.target_stage_name,
        needs_target_reevaluation=commit.needs_target_reevaluation,
        should_dump_results=not commit.needs_target_reevaluation,
    )


def resolve_active_transition_gate_decision(
    decision: ActiveTransitionGateDecision,
    *,
    evaluate_target: Callable[[], Any],
    recompute_last_score: Callable[[Any], None],
    dump_results: Callable[[], None],
) -> Any:
    if decision.needs_target_reevaluation:
        evaluation_result = evaluate_target()
        recompute_last_score(evaluation_result)
        return evaluation_result
    if decision.should_dump_results:
        dump_results()
    return decision.evaluation_result


def finish_seed_transition_gate_decision(
    runtime_config: dict,
    stage_manager: Any,
    attempt: SeedTransitionGateAttempt,
    *,
    evaluation_result: Any,
    is_accepted: bool,
) -> SeedTransitionGateDecision:
    if not is_accepted:
        return SeedTransitionGateDecision(
            evaluation_result=finish_rejected_seed_transition_gate_attempt(
                runtime_config,
                stage_manager,
                evaluation_result,
            ),
            is_accepted=False,
        )

    commit = commit_accepted_transition_gate_attempt(
        runtime_config,
        stage_manager,
        attempt,
    )
    return SeedTransitionGateDecision(
        evaluation_result=evaluation_result,
        is_accepted=True,
        committed_stage_name=commit.target_stage_name,
        needs_target_reevaluation=commit.needs_target_reevaluation,
    )


def resolve_seed_transition_gate_decision(
    runtime_config: dict,
    decision: SeedTransitionGateDecision,
    *,
    evaluate_target: Callable[[], Any],
    recompute_last_score: Callable[[Any], None],
) -> Any:
    if not decision.is_accepted:
        return decision.evaluation_result
    evaluation_result = decision.evaluation_result
    if decision.needs_target_reevaluation:
        evaluation_result = evaluate_target()
    recompute_last_score(evaluation_result)
    clear_transition_gate_runtime(runtime_config)
    return evaluation_result


def finish_rejected_seed_transition_gate_attempt(
    runtime_config: dict,
    stage_manager: Any,
    evaluation_result: Any,
) -> Any:
    clear_transition_gate_runtime(runtime_config)
    return stage_manager.current_transition_input_scores or evaluation_result


def finish_rejected_active_transition_gate_attempt(
    runtime_config: dict,
    stage_manager: Any,
    evaluation_result: Any,
) -> Any:
    if stage_manager.record_transition_consistency_gate_failure():
        evaluation_result = stage_manager.abort_transition_consistency_gate()
        clear_transition_gate_runtime(runtime_config)
        return evaluation_result
    evaluation_result = (
        stage_manager.current_transition_input_scores or evaluation_result
    )
    clear_transition_gate_runtime(runtime_config)
    return evaluation_result


def clear_transition_gate_runtime(runtime_config: dict) -> None:
    runtime_config.pop(TRANSITION_GATE_RESULT_KEY, None)
    runtime_config.pop(TRANSITION_GATE_METADATA_KEY, None)
