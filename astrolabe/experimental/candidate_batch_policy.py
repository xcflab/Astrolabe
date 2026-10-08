from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal

from astrolabe.utils.log_events import event_span

FailPolicy = Literal["record-and-continue", "fail-fast"]
CandidateStatus = Literal["completed", "failed"]
CollectionTrigger = Literal["min_completed", "max_wait_s", "drain"]


@dataclass(frozen=True)
class CandidatePoolConfig:
    num_candidates: int = 4
    min_completed: int = 2
    max_wait_s: float = 30.0
    max_in_flight: int | None = None
    fail_policy: FailPolicy = "record-and-continue"

    def __post_init__(self) -> None:
        if int(self.num_candidates) < 1:
            raise ValueError("num_candidates must be >= 1")
        if int(self.min_completed) < 1:
            raise ValueError("min_completed must be >= 1")
        if float(self.max_wait_s) <= 0:
            raise ValueError("max_wait_s must be > 0")
        if self.max_in_flight is not None and int(self.max_in_flight) < 1:
            raise ValueError("max_in_flight must be >= 1 when set")
        if self.fail_policy not in {"record-and-continue", "fail-fast"}:
            raise ValueError(
                "fail_policy must be 'record-and-continue' or 'fail-fast'"
            )

    @property
    def worker_count(self) -> int:
        return int(self.max_in_flight or self.num_candidates)


@dataclass(frozen=True)
class CandidateTask:
    candidate_id: str
    generation: int
    payload: Any = None


@dataclass(frozen=True)
class CandidateResult:
    task: CandidateTask
    status: CandidateStatus
    score: float | None = None
    payload: Any = None
    error: str | None = None
    duration_s: float = 0.0

    @property
    def candidate_id(self) -> str:
        return self.task.candidate_id


CandidateSelector = Callable[[Iterable[CandidateResult]], CandidateResult | None]


def select_highest_score_candidate(
    results: Iterable[CandidateResult],
) -> CandidateResult | None:
    completed = [result for result in results if result.status == "completed"]
    if not completed:
        return None
    return sorted(
        completed,
        key=lambda result: (-float(result.score), result.candidate_id),
    )[0]


def payload_to_summary(payload: Any) -> object:
    if payload is None:
        return None
    if hasattr(payload, "as_summary"):
        return payload.as_summary()
    if isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, dict):
        return dict(payload)
    return repr(payload)


@dataclass(frozen=True)
class CandidateCollection:
    collection_index: int
    trigger: CollectionTrigger
    results: tuple[CandidateResult, ...]
    selected: CandidateResult | None
    carried_forward_candidate_ids: tuple[str, ...]
    started_at_s: float
    finished_at_s: float

    @property
    def failed(self) -> tuple[CandidateResult, ...]:
        return tuple(result for result in self.results if result.status == "failed")

    @property
    def completed(self) -> tuple[CandidateResult, ...]:
        return tuple(result for result in self.results if result.status == "completed")

    def as_row(self) -> dict[str, object]:
        failed_candidates = [
            {
                "candidate_id": result.candidate_id,
                "error": result.error or "",
                "duration_s": result.duration_s,
                "payload": payload_to_summary(result.payload),
            }
            for result in self.failed
        ]
        return {
            "collection_index": self.collection_index,
            "trigger": self.trigger,
            "selected_candidate_id": (
                self.selected.candidate_id if self.selected is not None else ""
            ),
            "selected_score": (
                self.selected.score if self.selected is not None else None
            ),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "failed_candidates": failed_candidates,
            "carried_forward": ",".join(self.carried_forward_candidate_ids),
            "duration_s": self.finished_at_s - self.started_at_s,
        }


class CandidateExecutionError(RuntimeError):
    pass


class CandidatePoolScheduler:
    """Collect asynchronous candidate evaluations in decision-sized windows."""

    def __init__(
        self,
        config: CandidatePoolConfig,
        worker: Callable[[CandidateTask], CandidateResult | float],
        *,
        selector: CandidateSelector | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.selector = selector or select_highest_score_candidate
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=config.worker_count)
        self._pending: dict[Future[CandidateResult], CandidateTask] = {}
        self._ready: list[CandidateResult] = []
        self._collection_index = 0
        self._window_started = time.monotonic()
        self._generation_started: dict[int, float] = {}
        self._closed = False

    def submit_generation(self, generation: int, tasks: Iterable[CandidateTask]) -> None:
        if self._closed:
            raise RuntimeError("CandidatePoolScheduler is closed")
        submitted = 0
        generation_int = int(generation)
        for task in tasks:
            future = self._executor.submit(self._run_task, task)
            self._pending[future] = task
            submitted += 1
        if submitted:
            started = time.monotonic()
            self._generation_started[generation_int] = started
            if not self._ready:
                self._window_started = started

    def collect_until_ready(
        self,
        *,
        target_generation: int | None = None,
    ) -> CandidateCollection | None:
        generation = None if target_generation is None else int(target_generation)
        while True:
            ready = self._ready_for_generation(generation)
            if len(ready) >= self.config.min_completed:
                return self._finalize_collection(
                    "min_completed",
                    target_generation=generation,
                )
            pending = self._pending_for_generation(generation)
            if not pending:
                if ready:
                    return self._finalize_collection(
                        "drain",
                        target_generation=generation,
                    )
                return None

            window_started = self._collection_started_at(generation)
            elapsed = time.monotonic() - window_started
            remaining = max(0.0, float(self.config.max_wait_s) - elapsed)
            if remaining <= 0:
                if ready:
                    return self._finalize_collection(
                        "max_wait_s",
                        target_generation=generation,
                    )
                raise CandidateExecutionError(
                    f"No candidate finished within max_wait_s={self.config.max_wait_s}"
                )

            done, _not_done = wait(
                list(self._pending.keys()),
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                ready = self._ready_for_generation(generation)
                if ready:
                    return self._finalize_collection(
                        "max_wait_s",
                        target_generation=generation,
                    )
                raise CandidateExecutionError(
                    f"No candidate finished within max_wait_s={self.config.max_wait_s}"
                )
            for future in done:
                task = self._pending.pop(future)
                result = future.result()
                if (
                    result.status == "failed"
                    and self.config.fail_policy == "fail-fast"
                ):
                    raise CandidateExecutionError(
                        f"Candidate {task.candidate_id} failed: {result.error}"
                    )
                self._ready.append(result)

    def drain(self) -> list[CandidateCollection]:
        collections: list[CandidateCollection] = []
        if self._pending:
            done, _not_done = wait(list(self._pending.keys()))
            for future in done:
                task = self._pending.pop(future)
                result = future.result()
                if result.status == "failed" and self.config.fail_policy == "fail-fast":
                    raise CandidateExecutionError(
                        f"Candidate {task.candidate_id} failed: {result.error}"
                    )
                self._ready.append(result)
        if self._ready:
            collection = self._finalize_collection("drain")
            collections.append(collection)
        return collections

    def close(self, *, wait_for_pending: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait_for_pending, cancel_futures=not wait_for_pending)

    def __enter__(self) -> CandidatePoolScheduler:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close(wait_for_pending=_exc_type is None)
        return False

    def _run_task(self, task: CandidateTask) -> CandidateResult:
        started = time.monotonic()
        with event_span(
            logger=self.logger,
            component="candidate",
            alias=task.candidate_id,
            detail=f"generation={task.generation}",
        ) as span:
            try:
                raw = self.worker(task)
                duration = time.monotonic() - started
                if isinstance(raw, CandidateResult):
                    result = raw
                else:
                    result = CandidateResult(
                        task=task,
                        status="completed",
                        score=float(raw),
                        duration_s=duration,
                    )
                if result.duration_s <= 0:
                    result = replace(result, duration_s=duration)
                if result.status == "failed" and not result.error:
                    result = replace(result, error="candidate failed")
                if result.status == "completed" and result.score is None:
                    raise ValueError("completed candidate result must include a score")
                if result.status == "failed":
                    span.finish(
                        event="span_error",
                        status="failed",
                        artifact=task.candidate_id,
                        error=result.error,
                        duration_s=result.duration_s,
                    )
                else:
                    span.finish(
                        artifact=task.candidate_id,
                        score=result.score,
                        duration_s=result.duration_s,
                    )
                return result
            except Exception as exc:
                duration = time.monotonic() - started
                span.error(exc, duration_s=duration)
                return CandidateResult(
                    task=task,
                    status="failed",
                    error=str(exc) or exc.__class__.__name__,
                    duration_s=duration,
                )

    def _finalize_collection(
        self,
        trigger: CollectionTrigger,
        *,
        target_generation: int | None = None,
    ) -> CandidateCollection:
        self._collection_index += 1
        finished = time.monotonic()
        results = tuple(self._ready_for_generation(target_generation))
        selected = self.selector(results)
        if selected is not None:
            if selected not in results:
                raise CandidateExecutionError(
                    "Candidate selector returned a result outside the collection"
                )
            if selected.status != "completed":
                raise CandidateExecutionError(
                    "Candidate selector returned a non-completed candidate"
                )
        if target_generation is None:
            self._ready = []
        else:
            self._ready = [
                result
                for result in self._ready
                if result.task.generation != target_generation
            ]
        carried = tuple(task.candidate_id for task in self._pending.values())
        collection = CandidateCollection(
            collection_index=self._collection_index,
            trigger=trigger,
            results=results,
            selected=selected,
            carried_forward_candidate_ids=carried,
            started_at_s=self._collection_started_at(target_generation),
            finished_at_s=finished,
        )
        self._window_started = finished
        if target_generation is not None:
            self._generation_started.pop(target_generation, None)
        self.logger.info(
            "candidate_collection index=%s trigger=%s completed=%s failed=%s selected=%s carried_forward=%s",
            collection.collection_index,
            collection.trigger,
            len(collection.completed),
            len(collection.failed),
            collection.selected.candidate_id if collection.selected else "",
            ",".join(collection.carried_forward_candidate_ids),
        )
        return collection

    def _ready_for_generation(
        self,
        target_generation: int | None,
    ) -> list[CandidateResult]:
        if target_generation is None:
            return list(self._ready)
        return [
            result
            for result in self._ready
            if int(result.task.generation) == int(target_generation)
        ]

    def _pending_for_generation(
        self,
        target_generation: int | None,
    ) -> dict[Future[CandidateResult], CandidateTask]:
        if target_generation is None:
            return dict(self._pending)
        return {
            future: task
            for future, task in self._pending.items()
            if int(task.generation) == int(target_generation)
        }

    def _collection_started_at(self, target_generation: int | None) -> float:
        if target_generation is None:
            return self._window_started
        return self._generation_started.get(int(target_generation), self._window_started)
