import logging
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None

from astrolabe.core import global_config
from astrolabe.decision_context import (
    build_decision_score_context,
    component_deltas,
    get_reference_gate_result,
    get_transition_consistency_gate_result,
)
from astrolabe.evaluators.score_snapshot import (
    flatten_score_details,
    merge_score_details,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossTerms
from astrolabe.utils.base import calculate_temperature
from astrolabe.utils.log_events import (
    build_component_score_summary,
    build_weighted_score_summary,
    event_span,
)
from astrolabe.utils.logging_utils import (
    colorize_delta,
    format_metric_value,
    to_numeric,
)


@dataclass
class Decisions:
    """Class for managing protein mutation decisions and tracking mutation history.

    Attributes:
        mutated_protomers: Maps step number to protomer state
        scores: Maps step number to score value
        evaluation_results: Maps step number to evaluation results
        scores_detail: Maps step number to detailed score metrics
        accepted_steps: List of steps where mutations were accepted
        last_accepted: Most recent accepted step number
    """

    sequences: Dict[int, Dict[str, str]]
    scores: Dict[int, float]
    evaluation_results: Dict[int, "StandardScores"]
    scores_detail: Dict[int, Dict[str, float]]
    accepted_steps: List[int]
    last_accepted: int
    stage_history: Dict[int, str]
    row_history: List[Dict[str, object]] = field(default_factory=list)
    NOISE_COMPONENT_KEYS = {
        "iteration",
        "step",
        "prefix_steps",
        "stage_entry",
        "acceptance_gate_passed",
    }
    SCORE_CONTEXT_PREFIXES = ("previous_", "delta_")
    NOISE_COMPONENT_SUFFIXES = ("_length",)
    SCORE_COMPONENT_MAX_INFO = 10
    SCORE_COMPONENT_MIN_INFO = 4
    SCORE_COMPONENT_MIN_ABS_DELTA = 1e-6

    @staticmethod
    def _merge_default_score_details(
        evaluation_result: "StandardScores",
        score_details: dict | None,
    ) -> dict:
        return merge_score_details(evaluation_result, score_details)

    def _build_score_row(
        self,
        *,
        step: int,
        accepted: bool,
        note: str = "",
    ) -> dict[str, object]:
        row = {
            "step": step,
            "weighted_score": self.scores.get(step),
            "accepted": accepted,
            "stage": self.stage_history.get(step, ""),
            "note": note,
        }
        row.update(flatten_score_details(self.scores_detail.get(step, {}) or {}))
        for key, value in dict(self.sequences.get(step, {}) or {}).items():
            row[f"{key}_sequence"] = value
        return row

    def _seed_row_history_from_state(self) -> None:
        if self.row_history:
            return
        for step in sorted(self.scores.keys()):
            self.row_history.append(
                self._build_score_row(
                    step=step,
                    accepted=step in self.accepted_steps,
                )
            )

    def _record_score_row(
        self,
        *,
        step: int,
        accepted: bool,
        note: str = "",
    ) -> None:
        self.row_history.append(
            self._build_score_row(step=step, accepted=accepted, note=note)
        )

    def _update_latest_step_row_note(self, *, step: int, note: str) -> None:
        self._seed_row_history_from_state()
        for row in reversed(self.row_history):
            if int(row.get("step", -1)) != step:
                continue
            row["note"] = note
            return
        self._record_score_row(
            step=step,
            accepted=step in self.accepted_steps,
            note=note,
        )

    @classmethod
    def _merge_prefix_best_metrics(cls, best: dict[str, Any], values: dict) -> None:
        for raw_key, value in dict(values or {}).items():
            key = str(raw_key)
            if isinstance(value, dict):
                existing = best.setdefault(key, {})
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"Cannot build prefix early-stopping best context: "
                        f"score key {key!r} is both scalar and nested."
                    )
                cls._merge_prefix_best_metrics(existing, value)
                continue

            numeric = to_numeric(value)
            if numeric is None:
                continue
            existing = best.get(key)
            if isinstance(existing, dict):
                raise ValueError(
                    f"Cannot build prefix early-stopping best context: "
                    f"score key {key!r} is both nested and scalar."
                )
            if existing is None or float(numeric) > float(existing):
                best[key] = float(numeric)

    @classmethod
    def _as_loss_terms(cls, values: dict[str, Any]) -> LossTerms:
        return LossTerms(
            {
                key: cls._as_loss_terms(value) if isinstance(value, dict) else value
                for key, value in values.items()
            }
        )

    def _build_prefix_best_score_context(self, *, through_step: int) -> LossTerms:
        best: dict[str, Any] = {}
        for historical_step in sorted(self.scores_detail.keys()):
            if int(historical_step) > int(through_step):
                continue
            self._merge_prefix_best_metrics(
                best,
                self.scores_detail.get(historical_step, {}) or {},
            )
        return self._as_loss_terms(best)

    def _evaluate_prefix_early_stop(
        self,
        *,
        step: int,
        loss_computer: "LossEngine",
        scores_detail: dict,
    ) -> bool:
        expression = getattr(loss_computer, "prefix_early_stopping", None)
        if not expression:
            return False
        method = getattr(loss_computer, "is_prefix_early_stopping", None)
        if not callable(method):
            raise TypeError(
                "Configured prefix_early_stopping requires "
                "loss_computer.is_prefix_early_stopping()."
            )
        best_context = self._build_prefix_best_score_context(through_step=step)
        triggered = bool(method(scores_detail, best_context))
        scores_detail["prefix_early_stop_triggered"] = (
            1.0 if triggered else 0.0
        )
        scores_detail["prefix_early_stopping_expression"] = str(expression)
        return triggered

    @staticmethod
    def _compute_component_deltas(
        current_scores: dict, previous_scores: dict
    ) -> list[tuple[str, float | None, float | None, float | None]]:
        return component_deltas(current_scores, previous_scores)

    @staticmethod
    def _format_delta(delta: float | None) -> str:
        if delta is None:
            return "N/A"
        signed = f"{delta:+.3f}"
        return colorize_delta(delta, signed)

    def _log_score_deltas(
        self,
        step: int,
        previous_weighted: float | None,
        current_weighted: float | None,
        previous_scores: dict,
        current_scores: dict,
    ) -> None:
        weighted_delta = None
        if (
            isinstance(previous_weighted, (int, float))
            and isinstance(current_weighted, (int, float))
        ):
            weighted_delta = float(current_weighted) - float(previous_weighted)
        logging.info(
            build_weighted_score_summary(
                step=step,
                previous_weighted=previous_weighted,
                current_weighted=current_weighted,
                weighted_delta=self._format_delta(weighted_delta),
            )
        )

        component_rows = self._compute_component_deltas(
            current_scores=current_scores,
            previous_scores=previous_scores,
        )
        changed_rows = []
        availability_rows = []
        stable_rows = []
        for name, previous, current, delta in component_rows:
            if name.startswith(self.SCORE_CONTEXT_PREFIXES):
                continue
            if name in self.NOISE_COMPONENT_KEYS:
                continue
            if any(name.endswith(suffix) for suffix in self.NOISE_COMPONENT_SUFFIXES):
                continue
            if delta is None:
                if current is None and previous is None:
                    continue
                availability_rows.append((name, previous, current, delta))
                continue
            if abs(delta) < self.SCORE_COMPONENT_MIN_ABS_DELTA:
                stable_rows.append((name, previous, current, delta))
                continue
            changed_rows.append((name, previous, current, delta))
        changed_rows.sort(
            key=lambda row: abs(row[3]) if row[3] is not None else -1.0,
            reverse=True,
        )
        availability_rows.sort(
            key=lambda row: (
                abs(row[2]) if row[2] is not None else 0.0,
                abs(row[1]) if row[1] is not None else 0.0,
            ),
            reverse=True,
        )
        stable_rows.sort(
            key=lambda row: (
                abs(row[2]) if row[2] is not None else 0.0,
                abs(row[1]) if row[1] is not None else 0.0,
            ),
            reverse=True,
        )

        selected_rows: list[tuple[str, float | None, float | None, float | None]] = []
        selected_names: set[str] = set()

        def _take_rows(
            rows: list[tuple[str, float | None, float | None, float | None]],
            limit: int,
        ) -> None:
            for row in rows:
                name = row[0]
                if name in selected_names:
                    continue
                selected_rows.append(row)
                selected_names.add(name)
                if len(selected_rows) >= limit:
                    return

        _take_rows(changed_rows, self.SCORE_COMPONENT_MAX_INFO)
        _take_rows(availability_rows, self.SCORE_COMPONENT_MAX_INFO)
        min_target = min(self.SCORE_COMPONENT_MIN_INFO, self.SCORE_COMPONENT_MAX_INFO)
        if len(selected_rows) < min_target:
            _take_rows(stable_rows, min_target)

        for name, previous, current, delta in selected_rows:
            logging.info(
                build_component_score_summary(
                    step=step,
                    component=name,
                    previous=previous,
                    current=current,
                    delta=self._format_delta(delta),
                )
            )
        logging.debug(
            "Step %d score_context current=%s previous=%s shown=%d changed=%d availability=%d stable=%d",
            step,
            current_scores,
            previous_scores,
            len(selected_rows),
            len(changed_rows),
            len(availability_rows),
            len(stable_rows),
        )

    def __getitem__(self, item: str) -> any:
        """Allows attribute-style access via indexing.

        Args:
            item: Name of attribute to access

        Returns:
            Value of requested attribute
        """
        return getattr(self, item)

    @classmethod
    def create(
        cls,
        evaluation_result: "StandardScores",
        runtime_config: dict,
        score_func: callable,
    ) -> "Decisions":
        """Factory method to create a new Decisions instance.

        Args:
            protomer: Initial protomer state
            evaluation_result: Initial evaluation results
            score_func: Function to compute scores

        Returns:
            Initialized Decisions instance
        """
        class _InitialLossComputer:
            def compute_loss(self, eval_result, runtime):
                return score_func(eval_result, runtime)

        context = build_decision_score_context(
            eval_result=evaluation_result,
            loss_computer=_InitialLossComputer(),
            runtime_config=runtime_config,
            evaluate_decision_flags=False,
            evaluate_acceptance_gate=False,
        )
        score = context.weighted_score
        scores_detail = context.scores_detail
        sequences = {
            0: {
                f"{p.type.value}{name}": p.sequence
                for name, p in global_config.protomers.items()
            }
        }
        current_stage = getattr(global_config.stage_manager, "current_stage", "default")
        return cls(
            sequences=sequences,
            scores={0: score},
            evaluation_results={0: evaluation_result},
            scores_detail={0: scores_detail},
            accepted_steps=[0],
            last_accepted=0,
            stage_history={0: current_stage},
            row_history=[
                {
                    "step": 0,
                    "weighted_score": score,
                    "accepted": True,
                    "stage": current_stage,
                    "note": "",
                    **flatten_score_details(scores_detail),
                    **{f"{key}_sequence": value for key, value in sequences[0].items()},
                }
            ],
        )

    def _evaluate_mutation(
        self,
        step: int,
        eval_result: "StandardScores",
        loss_computer: "LossEngine",
        runtime_config: dict,
    ) -> Tuple[float, dict, bool, bool]:
        """Evaluate mutation and compute scores."""
        self.evaluation_results[step] = eval_result
        context = build_decision_score_context(
            eval_result=eval_result,
            loss_computer=loss_computer,
            runtime_config=runtime_config,
            include_mutator_selector_deltas=True,
            evaluate_acceptance_gate=False,
        )
        score = context.weighted_score
        scores_detail = context.scores_detail
        self.scores[step] = context.weighted_score
        self.scores_detail[step] = context.scores_detail
        current_stage = getattr(global_config.stage_manager, "current_stage", "unknown")
        self.stage_history[step] = current_stage
        self.sequences[step] = {
            f"{p.type.value}{name}": p.mutated_sequence
            for name, p in global_config.protomers.items()
        }
        return score, scores_detail, context.early_stop, context.soft_accepted

    def _accept_mutation(
        self, step: int, score: float, mutated_protomers: "Protomer"
    ) -> "Protomer":
        """Handle accepted mutation state updates."""
        self.accepted_steps.append(step)
        self.last_accepted = step
        return mutated_protomers

    def make_mutation_decision(
        self,
        step: int,
        eval_result: "StandardScores",
        loss_computer: "LossEngine",
        runtime_config: dict,
        force_accept: bool = False,
        stage_entry: float = 0.0,
    ) -> tuple[bool, StandardScores, bool]:
        """Makes a decision to accept or reject a mutation.

        Args:
            step: Current step number
            eval_result: Evaluation results for mutation
            loss_computer: Loss computation object
            runtime_config: Runtime configuration dict
            force_accept: If True, accept regardless of score (e.g. stage change)

        Returns:
            Tuple of (early_stop, last_accepted_eval_result, accepted)
        """
        current_stage = getattr(global_config.stage_manager, "current_stage", "unknown")
        with event_span(
            component="decision_maker",
            alias=self.__class__.__name__,
            stage=current_stage,
            step=step,
            detail="make_mutation_decision",
            force_accept=force_accept,
        ) as span:
            outcome = self._make_mutation_decision(
                step=step,
                eval_result=eval_result,
                loss_computer=loss_computer,
                runtime_config=runtime_config,
                force_accept=force_accept,
                stage_entry=stage_entry,
            )
            early_stop, _accepted_eval_result, accepted = outcome
            finish_fields = {
                "result": "accept" if accepted else "reject",
                "accepted": accepted,
                "early_stop": early_stop,
                "weighted_score": float(self.scores[step]),
            }
            score_detail = self.scores_detail.get(step, {}) or {}
            if "prefix_early_stop_triggered" in score_detail:
                finish_fields["prefix_early_stop"] = bool(
                    to_numeric(score_detail.get("prefix_early_stop_triggered")) or 0.0
                )
                finish_fields["prefix_early_stopping_expression"] = score_detail.get(
                    "prefix_early_stopping_expression"
                )
            span.finish(**finish_fields)
            return outcome

    def _make_mutation_decision(
        self,
        step: int,
        eval_result: "StandardScores",
        loss_computer: "LossEngine",
        runtime_config: dict,
        force_accept: bool = False,
        stage_entry: float = 0.0,
    ) -> tuple[bool, StandardScores, bool]:
        previous_step = self.last_accepted
        previous_weighted = self.scores.get(previous_step)
        previous_scores_detail = dict(self.scores_detail.get(previous_step, {}) or {})
        context = build_decision_score_context(
            eval_result=eval_result,
            loss_computer=loss_computer,
            runtime_config=runtime_config,
            previous_weighted=previous_weighted,
            previous_scores=previous_scores_detail,
            stage_entry=stage_entry,
            include_mutator_selector_deltas=True,
            include_previous_delta=True,
        )
        score = context.weighted_score
        scores_detail = context.scores_detail
        standard_early_stop = context.early_stop
        soft_accepted = context.soft_accepted
        hard_gate_passed = context.hard_gate_passed
        self.evaluation_results[step] = eval_result
        self.scores[step] = score
        self.scores_detail[step] = scores_detail
        current_stage = getattr(global_config.stage_manager, "current_stage", "unknown")
        self.stage_history[step] = current_stage
        self.sequences[step] = {
            f"{p.type.value}{name}": p.mutated_sequence
            for name, p in global_config.protomers.items()
        }
        self._log_score_deltas(
            step=step,
            previous_weighted=previous_weighted,
            current_weighted=score,
            previous_scores=previous_scores_detail,
            current_scores=dict(scores_detail or {}),
        )
        last_score_text = format_metric_value(self.scores[self.last_accepted])
        current_score_text = format_metric_value(score)
        consistency = get_transition_consistency_gate_result(runtime_config)
        gate_active = bool(getattr(consistency, "gate_active", False))
        reference_gate = get_reference_gate_result(runtime_config)
        reference_gate_active = bool(getattr(reference_gate, "gate_active", False))

        if gate_active and force_accept:
            raise ValueError(
                "force_accept cannot bypass an active transition-consistency gate."
            )

        if reference_gate_active and force_accept:
            raise ValueError("force_accept cannot bypass an active reference gate.")

        prefix_early_stop = self._evaluate_prefix_early_stop(
            step=step,
            loss_computer=loss_computer,
            scores_detail=scores_detail,
        )
        early_stop = bool(standard_early_stop or prefix_early_stop)

        if reference_gate_active and not bool(getattr(reference_gate, "passed", False)):
            score_text = format_metric_value(getattr(reference_gate, "score", None))
            threshold_text = format_metric_value(
                getattr(reference_gate, "threshold", None)
            )
            hotspot_score_text = format_metric_value(
                getattr(reference_gate, "hotspot_score", None)
            )
            hotspot_threshold_text = format_metric_value(
                getattr(reference_gate, "hotspot_threshold", None)
            )
            logging.info(
                f"step={step} decision=reject reason=reference_gate "
                f"previous={last_score_text} current={current_score_text} "
                f"reference_tmscore={score_text} threshold={threshold_text} "
                f"hotspot_score={hotspot_score_text} "
                f"hotspot_threshold={hotspot_threshold_text} "
                f"hotspot_passed={getattr(reference_gate, 'hotspot_passed', None)}"
            )
            global_config.segment_manager.clear_mutations()
            self._record_score_row(step=step, accepted=False)
            self.dump_results(global_config.out_dir, incremental=True)
            return early_stop, self.evaluation_results[self.last_accepted], False

        if not hard_gate_passed:
            logging.info(
                f"step={step} decision=reject reason=acceptance_gate "
                f"previous={last_score_text} current={current_score_text}"
            )
            global_config.segment_manager.clear_mutations()
            self._record_score_row(step=step, accepted=False)
            self.dump_results(global_config.out_dir, incremental=True)
            return early_stop, self.evaluation_results[self.last_accepted], False

        if gate_active:
            passed = bool(getattr(consistency, "passed", False))
            score_text = format_metric_value(getattr(consistency, "score", None))
            threshold_text = format_metric_value(getattr(consistency, "threshold", None))
            if passed:
                self.accepted_steps.append(step)
                self.last_accepted = step
                logging.info(
                    f"step={step} decision=accept reason=transition_consistency_gate "
                    f"previous={last_score_text} current={current_score_text} "
                    f"consistency={score_text} threshold={threshold_text}"
                )
                global_config.segment_manager.apply_mutations()
                self._record_score_row(step=step, accepted=True)
                self.dump_results(global_config.out_dir, incremental=True)
                return early_stop, self.evaluation_results[self.last_accepted], True

            logging.info(
                f"step={step} decision=reject reason=transition_consistency_gate "
                f"previous={last_score_text} current={current_score_text} "
                f"consistency={score_text} threshold={threshold_text}"
            )
            global_config.segment_manager.clear_mutations()
            self._record_score_row(step=step, accepted=False)
            self.dump_results(global_config.out_dir, incremental=True)
            return early_stop, self.evaluation_results[self.last_accepted], False

        if not hard_gate_passed:
            logging.info(
                f"step={step} decision=reject reason=acceptance_gate "
                f"previous={last_score_text} current={current_score_text}"
            )
            global_config.segment_manager.clear_mutations()
            self._record_score_row(step=step, accepted=False)
            self.dump_results(global_config.out_dir, incremental=True)
            return early_stop, self.evaluation_results[self.last_accepted], False
        if force_accept:
            self.accepted_steps.append(step)
            logging.info(
                f"step={step} decision=accept reason=stage_change "
                f"previous={last_score_text} current={current_score_text}"
            )
            self.last_accepted = step
            global_config.segment_manager.apply_mutations()
            self._record_score_row(step=step, accepted=True)
            self.dump_results(global_config.out_dir, incremental=True)
            return early_stop, self.evaluation_results[self.last_accepted], True
        delta = self.scores[self.last_accepted] - score

        ACCEPTED = True
        if delta < 0:
            logging.info(
                f"step={step} decision=accept reason=better "
                f"previous={last_score_text} current={current_score_text}"
            )
        else:
            mh_enabled = bool(
                getattr(global_config, "use_metropolis_acceptance", False)
            )
            accepted_by_mh = False
            if mh_enabled:
                temperature = calculate_temperature(step)
            else:
                temperature = 0.0
            if temperature > 0:
                acceptance_probability = np.exp(-delta / temperature)
                if np.random.uniform(0, 1) < acceptance_probability:
                    logging.info(
                        f"step={step} decision=accept reason=metropolis "
                        f"previous={last_score_text} current={current_score_text}"
                    )
                    accepted_by_mh = True

            if accepted_by_mh:
                pass
            elif soft_accepted:
                logging.info(
                    f"step={step} decision=accept reason=threshold "
                    f"previous={last_score_text} current={current_score_text}"
                )
            else:
                logging.info(
                    f"step={step} decision=reject reason=worse "
                    f"previous={last_score_text} current={current_score_text}"
                )
                ACCEPTED = False
        if ACCEPTED:
            self.accepted_steps.append(step)
            self.last_accepted = step
            global_config.segment_manager.apply_mutations()
        else:
            global_config.segment_manager.clear_mutations()

        self._record_score_row(step=step, accepted=ACCEPTED)
        self.dump_results(global_config.out_dir, incremental=True)
        return (
            early_stop,
            self.evaluation_results[self.last_accepted],
            ACCEPTED,
        )

    def recompute_last_score(self, loss_computer, runtime_config, eval_result=None):
        """Recalculate the last accepted score under the current stage setup."""
        step = self.last_accepted
        current_stage = getattr(global_config.stage_manager, "current_stage", "unknown")
        previous_stage = self.stage_history.get(step)
        stage_changed = bool(previous_stage and previous_stage != current_stage)
        if stage_changed:
            self._update_latest_step_row_note(step=step, note="enter stage")
        if eval_result is not None:
            self.evaluation_results[step] = eval_result
        else:
            eval_result = self.evaluation_results[step]
        context = build_decision_score_context(
            eval_result=eval_result,
            loss_computer=loss_computer,
            runtime_config=runtime_config,
            evaluate_decision_flags=False,
            evaluate_acceptance_gate=False,
        )
        score = context.weighted_score
        scores_detail = context.scores_detail
        self.scores[step] = score
        self.scores_detail[step] = scores_detail
        self.stage_history[step] = current_stage
        if stage_changed:
            self._record_score_row(
                step=step,
                accepted=step in self.accepted_steps,
                note="re-evaluate due to stage change",
            )
        self.dump_results(global_config.out_dir, incremental=False)

    def dump_results(self, out_dir: Path, incremental: bool = True) -> None:
        """Saves results to a CSV file.

        Args:
            out_dir: Directory to save results
            incremental: If True, append new data to existing file; if False, overwrite
        """
        csv_path = out_dir / "scores.csv"
        self._seed_row_history_from_state()
        rows = list(self.row_history)
        preferred = ["step", "weighted_score", "accepted", "stage", "note"]

        if pd is None:
            all_keys = set(preferred)
            for row in rows:
                all_keys.update(row.keys())
            fieldnames = preferred + sorted(
                key for key in all_keys if key not in set(preferred)
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return

        score = pd.DataFrame(rows)
        other_cols = [col for col in score.columns if col not in preferred]
        score = score.reindex(columns=preferred + sorted(other_cols))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        score.to_csv(csv_path, mode="w", header=True, index=False)
