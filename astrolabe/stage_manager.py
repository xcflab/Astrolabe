import copy
import logging
import re
import ast
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.losses.base import LossEngine
from astrolabe.stage_transition import (
    TransitionConsistencyGateConfig,
    TransitionConsistencyGateContext,
    TransitionConsistencyGateResult,
    compute_reference_consistency,
    snapshot_standard_scores,
    write_transition_gate_manifest,
)
from astrolabe.utils.log_events import build_stage_enter_message




def _extract_missing_var_names(exc: Exception) -> set[str]:
    missing: set[str] = set()
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()

    while pending:
        current = pending.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        text = str(current)
        match = re.search(
            r"Variables in weighted_score_expression not found: (\{.*\})", text
        )
        if match:
            missing.update(re.findall(r"'([^']+)'", match.group(1)))
        missing.update(re.findall(r"name '([^']+)' is not defined", text))

        pending.append(getattr(current, "__cause__", None))
        pending.append(getattr(current, "__context__", None))

    return missing


def _extract_attribute_roots(expr: str | None) -> set[str]:
    if not expr:
        return set()
    tree = ast.parse(expr, mode="eval")
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            roots.add(node.value.id)
    return roots


def resolve_mutation_num(raw_value, *, protomer_pool=None) -> Optional[int]:
    """Convert a mutation_num spec into an integer count."""

    if raw_value is None:
        return None

    def _total_mutable_length():
        pool = protomer_pool
        if pool is None:
            pool = (
                global_config.mutable_protomers
                if getattr(global_config, "mutable_protomers", None)
                else global_config.protomers
            )
        total = 0
        for protomer in pool.values():
            try:
                total += protomer.mutable_length
            except AttributeError:
                total += len(protomer.sequence)
        return total or 1

    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.endswith("%"):
            number_part = stripped[:-1].strip()
            if not number_part:
                raise ValueError("Percentage value for mutation_num is empty.")
            try:
                percent = float(number_part) / 100.0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid percentage value for mutation_num: '{raw_value}'"
                ) from exc
            total_len = _total_mutable_length()
            return max(1, int(round(total_len * percent)))
        try:
            return int(round(float(stripped)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for mutation_num: '{raw_value}'"
            ) from exc

    if isinstance(raw_value, (int, float)):
        return int(round(raw_value))

    raise TypeError(
        "mutation_num must be an int, float, or percentage string (e.g., '5%')."
    )


@dataclass
class StageUnit:
    oligo: List[str] | None = None
    weighted_score_expression: str | None = None
    acceptance_gate: str | None = None
    acceptance_threshold: str | None = None
    early_stopping: str | None = None
    prefix_early_stopping: str | None = None
    ignored_complexes: List[str] | None = field(default_factory=list)
    negative_design_complexes: List[str] | None = field(default_factory=list)
    evaluator: str | None = None
    mutators: List[str] | None = None
    mutator_weights: List[str] | None = None
    transition_consistency_gate: TransitionConsistencyGateConfig | dict[str, Any] | None = None
    mutation_num: int | float | str | None = None
    _entrance: str | None = None  # Condition to enter this stage
    _has_weighted_score_expression: bool = field(init=False, default=False)
    _has_acceptance_gate: bool = field(init=False, default=False)
    _has_acceptance_threshold: bool = field(init=False, default=False)
    _has_early_stopping: bool = field(init=False, default=False)
    _has_prefix_early_stopping: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.oligo is not None:
            self.negative_design_complexes = [
                oligo[1:] for oligo in self.oligo if oligo.startswith("-")
            ]
            self.ignored_complexes = [
                oligo[1:] for oligo in self.oligo if oligo.startswith("~")
            ]
            self.oligo = [
                oligo[1:] if oligo.startswith(("-", "~")) else oligo
                for oligo in self.oligo
            ]

    def set_field_flags(self, source_dict):
        self._has_weighted_score_expression = "weighted_score_expression" in source_dict
        self._has_acceptance_gate = "acceptance_gate" in source_dict
        self._has_acceptance_threshold = "acceptance_threshold" in source_dict
        self._has_early_stopping = "early_stopping" in source_dict
        self._has_prefix_early_stopping = "prefix_early_stopping" in source_dict


class StageManager:
    def __init__(self, stage_args, *, loss_computer: "LossEngine", runtime_context=None):
        # Allow disabling stages via overlays by setting them to `null`.
        self.loss_computer = loss_computer
        self.runtime_context = runtime_context or global_config.build_runtime_context()
        stage_args = {k: v for k, v in stage_args.items() if v is not None}
        if "default" not in stage_args:
            raise ValueError("Stage arguments must contain a 'default' stage.")
        self.stage_names = list(stage_args.keys())
        self.nondefault_names = [name for name in self.stage_names if name != "default"]
        self.stage_args = stage_args
        for stage_name, args in stage_args.items():
            if "_entrance" not in args and stage_name != "default":
                raise ValueError(
                    f"Stage '{stage_name}' must contain '_entrance' key in its arguments to decide when to enter the stage."
                )
            try:
                if OmegaConf.is_config(args):
                    stage_unit_args = OmegaConf.to_container(args, resolve=True)
                else:
                    stage_unit_args = dict(args)
                # stage_unit_args.pop("_entrance", None)  # Remove entrance condition
                unit = StageUnit(**stage_unit_args)
                if unit.transition_consistency_gate is not None:
                    unit.transition_consistency_gate = TransitionConsistencyGateConfig.from_raw(
                        unit.transition_consistency_gate,
                        stage_name=stage_name,
                    )
                unit.set_field_flags(stage_unit_args)
                self.stage_args[stage_name] = unit
            except TypeError as e:
                raise ValueError(
                    f"Invalid arguments for stage '{stage_name}': {e}"
                ) from e
            raw_oligos = list(stage_unit_args.get("oligo") or [])
            if any(
                re.fullmatch(r"[-~]?[A-Za-z+]+", str(oligo)) is None
                for oligo in raw_oligos
            ):
                raise ValueError(
                    f"Invalid oligomer {raw_oligos!r}: expected an optional '-' "
                    "(negative design) or '~' (ignored) prefix followed by A-Z/a-z/+"
                )
            sorted_oligo = [
                "".join(
                    sorted(
                        oligo[1:] if oligo.startswith(("-", "~")) else oligo
                    )
                )
                for oligo in raw_oligos
            ]
            if len(sorted_oligo) != len(set(sorted_oligo)):
                raise ValueError(
                    "Oligomers must be unique, but found duplicates in the provided oligomers."
                )
        self.default_setting = self.stage_args["default"]
        if self.default_setting.transition_consistency_gate is not None:
            raise ValueError("Default stage cannot define 'transition_consistency_gate'.")
        self.transition_consistency_gate_context: TransitionConsistencyGateContext | None = None
        self.current_transition_input_scores = None
        self.transition_consistency_gate_attempts = 0
        self.stage_entry_reference_scores = {}
        self.last_stage = None
        if self.default_setting.evaluator is None:
            raise ValueError(
                "Default stage must define 'evaluator' since there is no global fallback."
            )
        if self.default_setting.weighted_score_expression is None:
            raise ValueError(
                "Default stage must define 'weighted_score_expression' since there is no global fallback."
            )
        if not self.default_setting.mutators:
            raise ValueError(
                "Default stage must define 'mutators' since there is no global fallback."
            )
        if self.default_setting.mutation_num is None:
            raise ValueError(
                "Default stage must define 'mutation_num' since no global fallback is provided."
            )
        self.set_stage("default")

    def set_stage(self, stage_name: str) -> None:
        """Set the current stage to a new stage."""
        if stage_name not in self.stage_names:
            raise ValueError(f"Stage '{stage_name}' does not exist.")
        self.last_stage = getattr(self, "current_stage", None)
        self.current_stage = stage_name
        self.last_setting = (
            self.current_setting
            if hasattr(self, "current_setting")
            else self.default_setting
        )
        self.current_setting = self.stage_args[stage_name]
        self.clear_transition_consistency_gate()
        self._sync_design_complexes()
        self.loss_computer.setup(
            self.current_weighted_score_expression,
            self.current_early_stopping,
            self.current_acceptance_threshold,
            self.current_acceptance_gate,
            prefix_early_stopping=self.current_prefix_early_stopping,
        )
        self._apply_mutation_num()
        mutator_aliases = list(
            self.current_setting.mutators or self.default_setting.mutators or []
        )
        evaluator_alias = self.current_setting.evaluator or self.default_setting.evaluator
        logging.info(
            build_stage_enter_message(
                stage_name=self.current_stage,
                evaluator=evaluator_alias,
                oligo=self.current_oligo,
                mutators=mutator_aliases,
                mutation_num=self.runtime_context.get_variable("mutation_num"),
            )
        )

    def get_current_stage(self) -> str:
        """Get the name of the current stage."""
        return self.current_stage

    @property
    def current_oligo(self) -> List[str]:
        """Get the current oligo for the current stage."""
        stage_setting = self.execution_setting
        if stage_setting.oligo is None:
            return self.default_setting.oligo

        return stage_setting.oligo

    @property
    def all_oligos(self) -> List[str]:
        """Get all oligos from all stages."""
        ordered: list[str] = []
        seen: set[str] = set()
        for stage_name in self.stage_names:
            stage = self.stage_args[stage_name]
            for oligo in stage.oligo or []:
                if oligo not in seen:
                    ordered.append(oligo)
                    seen.add(oligo)
        return ordered

    @property
    def current_ignored_complexes(self) -> List[str]:
        """Get the current ignored complexes for the current stage."""
        stage_setting = self.execution_setting
        if stage_setting.ignored_complexes is None:
            return self.default_setting.ignored_complexes

        return stage_setting.ignored_complexes

    @property
    def current_negative_design_complexes(self) -> List[str]:
        """Get the negative-design complexes for the current stage."""
        stage_setting = self.execution_setting
        if stage_setting.negative_design_complexes is None:
            return self.default_setting.negative_design_complexes

        return stage_setting.negative_design_complexes

    def _sync_design_complexes(self) -> None:
        self.runtime_context.set_variable(
            "ignored_complexes", self.current_ignored_complexes
        )
        self.runtime_context.set_variable(
            "negative_design_complexes", self.current_negative_design_complexes
        )

    @property
    def current_weighted_score_expression(self) -> str:
        """Get the current weighted score expression for the current stage."""
        stage_setting = self.execution_setting
        if not stage_setting._has_weighted_score_expression:
            return self.default_setting.weighted_score_expression

        return stage_setting.weighted_score_expression

    @property
    def execution_setting(self) -> StageUnit:
        if self.transition_consistency_gate_context is not None:
            return self.stage_args[self.transition_consistency_gate_context.target_stage]
        return self.current_setting

    def get_transition_consistency_gate(
        self, stage_name: str
    ) -> TransitionConsistencyGateConfig | None:
        if stage_name == "default":
            return None
        return self.stage_args[stage_name].transition_consistency_gate

    @property
    def current_transition_consistency_gate(self) -> TransitionConsistencyGateConfig | None:
        if self.transition_consistency_gate_context is not None:
            return self.get_transition_consistency_gate(
                self.transition_consistency_gate_context.target_stage
            )
        return self.get_transition_consistency_gate(self.current_stage)

    def resolve_stage_evaluator(self, stage_unit: StageUnit | None) -> str | None:
        if stage_unit is None:
            return self.default_setting.evaluator
        return stage_unit.evaluator or self.default_setting.evaluator

    def resolve_stage_evaluator_instance(self, stage_unit: StageUnit | None):
        alias = self.resolve_stage_evaluator(stage_unit)
        try:
            return self.runtime_context.evaluators[alias]
        except KeyError as exc:
            missing = str(exc.args[0]) if exc.args else "<unknown>"
            available = sorted((self.runtime_context.evaluators or {}).keys())
            raise ValueError(
                "Stage references an unknown/disabled evaluator alias "
                f"{missing!r}. Available evaluators: {available}. "
                "Fix by defining it under `evaluator.evaluator_terms` (stages reference those aliases)."
            ) from exc

    def resolve_transition_consistency_gate_evaluator(self, stage_name: str) -> str | None:
        transition_gate = self.get_transition_consistency_gate(stage_name)
        if transition_gate is not None and transition_gate.evaluator:
            return transition_gate.evaluator
        return self.resolve_stage_evaluator(self.stage_args[stage_name])

    def resolve_transition_consistency_gate_evaluator_instance(self, stage_name: str):
        alias = self.resolve_transition_consistency_gate_evaluator(stage_name)
        try:
            return self.runtime_context.evaluators[alias]
        except KeyError as exc:
            missing = str(exc.args[0]) if exc.args else "<unknown>"
            available = sorted((self.runtime_context.evaluators or {}).keys())
            raise ValueError(
                "Transition consistency gate references an unknown/disabled evaluator alias "
                f"{missing!r}. Available evaluators: {available}. "
                "Fix by defining it under `evaluator.evaluator_terms`."
            ) from exc

    def resolve_stage_oligo(self, stage_unit: StageUnit | None) -> List[str]:
        if stage_unit is None or stage_unit.oligo is None:
            return list(self.default_setting.oligo or [])
        return list(stage_unit.oligo)

    def clear_transition_consistency_gate(self) -> None:
        self.transition_consistency_gate_context = None
        self.current_transition_input_scores = None
        self.transition_consistency_gate_attempts = 0

    def register_stage_entry_reference(
        self,
        *,
        target_stage: str,
        previous_stage_result,
        previous_stage_oligo: list[str] | None,
    ) -> None:
        if previous_stage_result is None:
            raise ValueError(
                f"Stage '{target_stage}' requested a stage-entry reference, but no previous stage result is available."
            )
        if previous_stage_oligo is None:
            raise ValueError(
                f"Stage '{target_stage}' requested a stage-entry reference, but previous stage oligomers are unknown."
            )
        reference_scores = previous_stage_result.filter_by_oligomer(previous_stage_oligo)
        if not reference_scores.names:
            raise ValueError(
                f"Stage '{target_stage}' requested a stage-entry reference, but the previous stage result contains no structures for oligomers {previous_stage_oligo!r}."
            )
        self.stage_entry_reference_scores[str(target_stage)] = reference_scores

    def get_stage_entry_reference_scores(self, stage_name: str | None = None):
        stage_key = str(stage_name or self.current_stage)
        try:
            return self.stage_entry_reference_scores[stage_key]
        except KeyError as exc:
            raise ValueError(
                f"Stage '{stage_key}' has no stage-entry reference. This source is only available after entering a non-default stage from a previous stage."
            ) from exc

    def activate_transition_consistency_gate(
        self,
        *,
        target_stage: str,
        previous_stage_result,
        previous_stage_oligo: list[str] | None,
        gate_seed_scores,
        artifact_root=None,
        activation_step: int | None = None,
    ) -> None:
        transition_gate = self.get_transition_consistency_gate(target_stage)
        if transition_gate is None:
            self.clear_transition_consistency_gate()
            return
        if previous_stage_result is None:
            raise ValueError(
                f"Stage '{target_stage}' requested transition_consistency_gate, but no previous stage result is available."
            )
        if previous_stage_oligo is None:
            raise ValueError(
                f"Stage '{target_stage}' requested transition_consistency_gate, but previous stage oligomers are unknown."
            )
        reference_scores = previous_stage_result.filter_by_oligomer(previous_stage_oligo)
        if not reference_scores.names:
            raise ValueError(
                f"Stage '{target_stage}' requested transition_consistency_gate, but the previous stage result contains no structures for oligomers {previous_stage_oligo!r}."
            )
        seed_scores = None
        if artifact_root is not None:
            seed_scores = snapshot_standard_scores(
                gate_seed_scores,
                output_dir=artifact_root / "seed",
            )
        self.transition_consistency_gate_context = TransitionConsistencyGateContext(
            weight=float(transition_gate.weight),
            consistency_threshold=float(transition_gate.consistency_threshold),
            max_attempts=int(transition_gate.max_attempts),
            evaluator=transition_gate.evaluator,
            source_stage=self.current_stage,
            target_stage=target_stage,
            reference_scores=reference_scores,
            rollback_scores=previous_stage_result,
            artifact_root=artifact_root,
            activation_step=activation_step,
            seed_scores=seed_scores,
        )
        self.current_transition_input_scores = gate_seed_scores
        self.transition_consistency_gate_attempts = 0
        self._sync_design_complexes()
        self.loss_computer.setup(
            self.current_weighted_score_expression,
            self.current_early_stopping,
            self.current_acceptance_threshold,
            self.current_acceptance_gate,
            prefix_early_stopping=self.current_prefix_early_stopping,
        )
        self._apply_mutation_num()

    def record_transition_consistency_gate_failure(self) -> bool:
        if self.transition_consistency_gate_context is None:
            return False
        self.transition_consistency_gate_attempts += 1
        return self.transition_consistency_gate_attempts >= int(
            self.transition_consistency_gate_context.max_attempts
        )

    def complete_transition_consistency_gate(self) -> str:
        if self.transition_consistency_gate_context is None:
            raise RuntimeError(
                "Cannot complete transition-consistency gate without context."
            )
        target_stage = self.transition_consistency_gate_context.target_stage
        self.set_stage(target_stage)
        return target_stage

    def abort_transition_consistency_gate(self):
        if self.transition_consistency_gate_context is None:
            raise RuntimeError(
                "Cannot abort transition-consistency gate without context."
            )
        rollback_scores = self.transition_consistency_gate_context.rollback_scores
        source_stage = self.transition_consistency_gate_context.source_stage
        if source_stage is None:
            raise RuntimeError(
                "Cannot abort transition-consistency gate without source stage."
            )
        self.clear_transition_consistency_gate()
        self._sync_design_complexes()
        self.loss_computer.setup(
            self.current_weighted_score_expression,
            self.current_early_stopping,
            self.current_acceptance_threshold,
            self.current_acceptance_gate,
            prefix_early_stopping=self.current_prefix_early_stopping,
        )
        self._apply_mutation_num()
        return rollback_scores

    def evaluate_transition_consistency(
        self,
        eval_result,
    ) -> TransitionConsistencyGateResult:
        if self.transition_consistency_gate_context is None:
            raise RuntimeError(
                "Transition consistency requested without an active transition-consistency gate."
            )
        context = self.transition_consistency_gate_context
        step = int(getattr(global_config, "step", 0) or 0)
        label = "seed"
        current_scores = eval_result
        use_seed_snapshot = (
            context.seed_scores is not None
            and context.activation_step is not None
            and step == int(context.activation_step)
        )
        if use_seed_snapshot:
            current_scores = context.seed_scores
        elif context.artifact_root is not None:
            label = f"attempt_{self.transition_consistency_gate_attempts + 1:02d}"
            current_scores = snapshot_standard_scores(
                eval_result,
                output_dir=context.artifact_root / label,
            )
        result = compute_reference_consistency(
            current_scores=current_scores,
            reference_scores=context.reference_scores,
            threshold=context.consistency_threshold,
        )
        if context.artifact_root is not None:
            write_transition_gate_manifest(
                artifact_root=context.artifact_root,
                source_stage=context.source_stage,
                target_stage=context.target_stage,
                threshold=context.consistency_threshold,
                max_attempts=context.max_attempts,
                activation_step=context.activation_step,
                label=label,
                step=step,
                reference_scores=context.reference_scores,
                current_scores=current_scores,
                result=result,
            )
        return result

    @property
    def current_evaluator(self):
        if self.transition_consistency_gate_context is not None:
            return self.resolve_transition_consistency_gate_evaluator_instance(
                self.transition_consistency_gate_context.target_stage
            )
        return self.resolve_stage_evaluator_instance(self.execution_setting)

    @property
    def current_mutators(self):
        stage_setting = self.execution_setting
        if stage_setting.mutators is None:
            mutator_names = self.default_setting.mutators or []
        else:
            mutator_names = stage_setting.mutators

        try:
            mutators = [self.runtime_context.mutators[mutator] for mutator in mutator_names]
        except KeyError as exc:
            missing = str(exc.args[0]) if exc.args else "<unknown>"
            available = sorted((self.runtime_context.mutators or {}).keys())
            raise ValueError(
                "Stage references an unknown/disabled mutator alias "
                f"{missing!r}. Available mutators: {available}. "
                "Fix by defining it under `mutator.mutator_terms` (stages reference those aliases)."
            ) from exc
        if not mutators:
            raise ValueError("No mutators configured for the current stage or default stage.")

        weights = getattr(stage_setting, "mutator_weights", None)
        if weights is not None and len(weights) == len(mutators):
            weights = [float(w) for w in weights]
            weights = [weight / sum(weights) for weight in weights]
        else:
            default_weights = (
                getattr(self.default_setting, "mutator_weights", None)
                if stage_setting.mutators is None
                else None
            )
            if (
                default_weights is not None
                and len(default_weights) == len(mutators)
            ):
                weights = [float(w) for w in default_weights]
                weights = [weight / sum(weights) for weight in weights]
            else:
                weights = [1.0 / len(mutators)] * len(mutators)

        for mutator, weight in zip(mutators, weights):
            mutator.mutation_weight = weight

        return mutators

    @property
    def current_acceptance_threshold(self) -> str | None:
        """Get the current acceptance threshold for the current stage."""
        stage_setting = self.execution_setting
        if stage_setting is self.default_setting:
            return self.default_setting.acceptance_threshold
        if not stage_setting._has_acceptance_threshold:
            return None

        return stage_setting.acceptance_threshold

    @property
    def current_acceptance_gate(self) -> str | None:
        """Get the hard acceptance gate for the current stage."""
        stage_setting = self.execution_setting
        if stage_setting is self.default_setting:
            return self.default_setting.acceptance_gate
        if not stage_setting._has_acceptance_gate:
            return None

        return stage_setting.acceptance_gate

    @property
    def current_early_stopping(self) -> str:
        """Get the current early stopping condition for the current stage."""
        stage_setting = self.execution_setting
        if not stage_setting._has_early_stopping:
            return self.default_setting.early_stopping

        return stage_setting.early_stopping

    @property
    def current_prefix_early_stopping(self) -> str | None:
        """Get the prefix-history early stopping condition for the current stage."""
        stage_setting = self.execution_setting
        if not stage_setting._has_prefix_early_stopping:
            return self.default_setting.prefix_early_stopping

        return stage_setting.prefix_early_stopping

    def _resolve_stage_mutation_value(self):
        raw_value = self.execution_setting.mutation_num
        if raw_value is None:
            raw_value = self.default_setting.mutation_num
        return raw_value

    def _apply_mutation_num(self):
        raw_value = self._resolve_stage_mutation_value()
        if raw_value is None:
            return
        protomer_pool = (
            self.runtime_context.mutable_protomers
            if self.runtime_context.mutable_protomers
            else self.runtime_context.protomers
        )
        resolved = resolve_mutation_num(raw_value, protomer_pool=protomer_pool)
        self.runtime_context.set_variable("mutation_num", resolved)

    def find_entrance_stage(self, eval_result, loss_computer: "LossEngine"):
        current_index = self.stage_names.index(self.current_stage)
        for stage_name in self.stage_names[current_index + 1 :]:
            if stage_name == "default":
                continue
            args = self.stage_args[stage_name]
            entrance_condition = args._entrance
            try:
                score_context_for_expressions = getattr(
                    loss_computer, "score_context_for_expressions", None
                )
                score_context = getattr(loss_computer, "score_context", None)
                current_eval_names = set(getattr(eval_result, "names", []) or [])
                required_attr_roots: set[str] = set()
                ordered_expression_terms = getattr(
                    loss_computer, "_ordered_expression_terms_for_expressions", None
                )
                expression_terms = getattr(loss_computer, "expression_terms", {}) or {}
                if callable(ordered_expression_terms):
                    required_attr_roots.update(
                        _extract_attribute_roots(entrance_condition)
                    )
                    for term_name in ordered_expression_terms([entrance_condition]):
                        required_attr_roots.update(
                            _extract_attribute_roots(expression_terms.get(term_name))
                        )

                if callable(score_context) and "score_context" in vars(loss_computer):
                    total_env = score_context(eval_result)
                elif (
                    callable(score_context_for_expressions)
                    and required_attr_roots.issubset(current_eval_names)
                ):
                    total_env = score_context_for_expressions(
                        eval_result, [entrance_condition]
                    )
                elif callable(score_context):
                    total_env = score_context(eval_result)
                else:
                    per_oligo_scores, per_terms_score = (
                        loss_computer._extract_scores_with_registry(eval_result)
                    )
                    total_env = {**per_oligo_scores, **per_terms_score}
                is_entrance = loss_computer.eval_condition(
                    total_env, entrance_condition
                )
            except Exception as e:
                missing_vars = _extract_missing_var_names(e)
                stage_oligos = set(getattr(args, "oligo", None) or [])
                if missing_vars and stage_oligos and missing_vars.issubset(stage_oligos):
                    logging.debug(
                        "Skip stage entrance for %s; missing vars %s are expected before stage oligos %s exist.",
                        stage_name,
                        sorted(missing_vars),
                        sorted(stage_oligos),
                    )
                    continue
                raise RuntimeError(
                    f"Failed to evaluate stage entrance condition for stage {stage_name!r}: {entrance_condition!r}"
                ) from e
            if is_entrance:
                return stage_name
        return None

    def decide_entrance(self, eval_result, loss_computer: "LossEngine"):
        stage_name = self.find_entrance_stage(eval_result, loss_computer)
        if stage_name is None:
            return None
        if self.get_transition_consistency_gate(stage_name) is None:
            self.set_stage(stage_name)
        return stage_name
