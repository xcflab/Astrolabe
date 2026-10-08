from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any

from astrolabe.utils.logging_utils import format_metric_value


EVENT_SUMMARY_ITEM_LIMIT = 5

_EVENT_FIELD_ORDER = (
    "event",
    "span_id",
    "component",
    "alias",
    "stage",
    "step",
    "status",
    "duration_ms",
    "artifact",
    "result",
    "error",
    "detail_level",
    "detail",
)


def summarize_event_items(
    values: Any,
    *,
    limit: int = EVENT_SUMMARY_ITEM_LIMIT,
) -> str | None:
    """Return a bounded comma-separated item summary."""
    items = [str(value) for value in values]
    if not items:
        return None
    visible = items[:limit]
    if len(items) > limit:
        visible.append(f"+{len(items) - limit}")
    return ",".join(visible)


def _format_event_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(_format_event_value(item) for item in value)
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or "=" in text:
        return json.dumps(text, ensure_ascii=True)
    return text


def build_event_message(
    *,
    event: str,
    span_id: str | None = None,
    component: str,
    alias: Any | None = None,
    stage: Any | None = None,
    step: Any | None = None,
    status: str | None = None,
    duration_ms: float | None = None,
    artifact: Any | None = None,
    result: Any | None = None,
    error: Any | None = None,
    detail_level: str | None = None,
    detail: Any | None = None,
    **fields: Any,
) -> str:
    """Render a stable single-line observability event."""
    data: dict[str, Any] = {
        "event": event,
        "span_id": span_id,
        "component": component,
        "alias": alias,
        "stage": stage,
        "step": step,
        "status": status,
        "duration_ms": duration_ms,
        "artifact": artifact,
        "result": result,
        "error": error,
        "detail_level": detail_level,
        "detail": detail,
    }
    data.update(fields)

    ordered_keys = [key for key in _EVENT_FIELD_ORDER if key in data]
    ordered_keys.extend(sorted(key for key in data if key not in _EVENT_FIELD_ORDER))
    parts = [
        f"{key}={_format_event_value(data[key])}"
        for key in ordered_keys
        if data.get(key) is not None
    ]
    return " ".join(parts)


def log_event(
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    event: str,
    component: str,
    **fields: Any,
) -> None:
    target = logger if logger is not None else logging.getLogger()
    target.log(
        level,
        build_event_message(event=event, component=component, **fields),
    )


class EventSpan:
    """Small logging-only lifecycle span; it never suppresses exceptions."""

    def __init__(
        self,
        *,
        component: str,
        span_id: str | None = None,
        alias: Any | None = None,
        stage: Any | None = None,
        step: Any | None = None,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        detail_level: str = "summary",
        detail: Any | None = None,
        **fields: Any,
    ) -> None:
        self.component = component
        self.span_id = span_id or uuid.uuid4().hex
        self.alias = alias
        self.stage = stage
        self.step = step
        self.logger = logger if logger is not None else logging.getLogger()
        self.level = level
        self.detail_level = detail_level
        self.detail = detail
        self.fields = dict(fields)
        self._start_time: float | None = None
        self._closed = False

    def __enter__(self) -> "EventSpan":
        self.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is not None:
            self.error(exc if exc is not None else exc_type)
            return False
        if not self._closed:
            self.finish()
        return False

    def start(self, **fields: Any) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        log_event(
            logger=self.logger,
            level=self.level,
            event="span_start",
            span_id=self.span_id,
            component=self.component,
            alias=self.alias,
            stage=self.stage,
            step=self.step,
            status="start",
            detail_level=self.detail_level,
            detail=self.detail,
            **self.fields,
            **fields,
        )

    def finish(
        self,
        *,
        status: str = "success",
        artifact: Any | None = None,
        result: Any | None = None,
        detail: Any | None = None,
        level: int | None = None,
        event: str = "span_end",
        **fields: Any,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        log_event(
            logger=self.logger,
            level=self.level if level is None else level,
            event=event,
            span_id=self.span_id,
            component=self.component,
            alias=self.alias,
            stage=self.stage,
            step=self.step,
            status=status,
            duration_ms=self._duration_ms(),
            artifact=artifact,
            result=result,
            detail_level=self.detail_level,
            detail=self.detail if detail is None else detail,
            **self.fields,
            **fields,
        )

    def skip(self, *, artifact: Any | None = None, detail: Any | None = None, **fields: Any) -> None:
        self.finish(
            status="skip",
            artifact=artifact,
            detail=detail,
            event="span_skip",
            **fields,
        )

    def error(self, exc: BaseException | type[BaseException] | None, **fields: Any) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(exc, type):
            error_name = exc.__name__
            detail = None
        elif exc is None:
            error_name = "Exception"
            detail = None
        else:
            error_name = exc.__class__.__name__
            detail = str(exc) or None
        message = build_event_message(
            event="span_error",
            span_id=self.span_id,
            component=self.component,
            alias=self.alias,
            stage=self.stage,
            step=self.step,
            status="error",
            duration_ms=self._duration_ms(),
            error=error_name,
            detail_level=self.detail_level,
            detail=detail,
            **self.fields,
            **fields,
        )
        self.logger.exception(message)

    def _duration_ms(self) -> float | None:
        if self._start_time is None:
            return None
        return (time.perf_counter() - self._start_time) * 1000.0


def event_span(**kwargs: Any) -> EventSpan:
    return EventSpan(**kwargs)


def build_step_start_message(
    step: int, total_steps: int, stage_name: str | None = None
) -> str:
    if stage_name:
        return f"step_start step={step}/{total_steps} stage={stage_name}"
    return f"step_start step={step}/{total_steps}"


def build_step_outcome_message(
    step: int,
    accepted: bool,
    early_stop: bool,
    stage_name: str | None = None,
) -> str:
    stage_part = f" stage={stage_name}" if stage_name else ""
    return (
        f"step_end step={step} accepted={str(bool(accepted)).lower()} "
        f"early_stop={str(bool(early_stop)).lower()}{stage_part}"
    )


def build_mutation_summary_message(
    mutator_name: str,
    changed_positions: int,
    changed_protomers: int,
    metrics: dict[str, Any],
) -> str:
    metrics_str = " ".join(
        f"{name}={format_metric_value(value)}" for name, value in metrics.items()
    )
    return (
        f"[{mutator_name}] changed_positions={changed_positions} "
        f"changed_protomers={changed_protomers} {metrics_str}"
    )


def build_no_change_message(mutator_name: str, metrics: dict[str, Any]) -> str:
    metrics_str = " ".join(
        f"{name}={format_metric_value(value)}" for name, value in metrics.items()
    )
    return f"[{mutator_name}] No sequence changes applied. {metrics_str}"


def build_weighted_score_summary(
    step: int,
    previous_weighted: Any,
    current_weighted: Any,
    weighted_delta: str,
) -> str:
    return (
        f"step={step} score "
        f"previous={format_metric_value(previous_weighted)} "
        f"current={format_metric_value(current_weighted)} "
        f"delta={weighted_delta}"
    )


def build_component_score_summary(
    step: int,
    component: str,
    previous: Any,
    current: Any,
    delta: str,
) -> str:
    return (
        f"component={component} "
        f"previous={format_metric_value(previous)} "
        f"current={format_metric_value(current)} delta={delta}"
    )


def build_stage_enter_message(
    stage_name: str,
    evaluator: Any,
    oligo: Any,
    mutators: Any,
    mutation_num: Any,
) -> str:
    return (
        f"stage_enter stage={stage_name} "
        f"evaluator={evaluator} oligo={oligo} "
        f"mutators={mutators} mutation_num={format_metric_value(mutation_num)}"
    )
