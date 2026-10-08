from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.structure_processors.base import run_nested_predocker_on_scores

SUPPLEMENTAL_STRUCTURE_START_REFERENCE = "structure_start_reference"
SUPPLEMENTAL_STRUCTURE_SOURCES = {
    SUPPLEMENTAL_STRUCTURE_START_REFERENCE,
}


@dataclass(frozen=True)
class SupplementalStructureScores:
    source: str
    scores: StandardScores


@dataclass(frozen=True)
class StructureInputScores:
    primary: StandardScores
    supplemental: tuple[SupplementalStructureScores, ...] = ()


def normalize_supplemental_structures(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        items = value
    else:
        raise ValueError(
            "supplemental_structures must be a string or a list of strings."
        )

    normalized: list[str] = []
    for item in items:
        source = str(item).strip().lower()
        if not source:
            continue
        if source not in SUPPLEMENTAL_STRUCTURE_SOURCES:
            allowed = ", ".join(sorted(SUPPLEMENTAL_STRUCTURE_SOURCES))
            raise ValueError(
                f"supplemental_structures entries must be one of: {allowed}; "
                f"got {item!r}."
            )
        if source not in normalized:
            normalized.append(source)
    return tuple(normalized)


def prepare_structure_input_scores(
    evaluation_scores: StandardScores,
    *,
    supplemental_structures: Any = None,
    predocker: Any = None,
    mutator_name: str = "mutator",
    predocker_runner: Any | None = None,
) -> StructureInputScores:
    primary = _maybe_predock(
        predocker,
        evaluation_scores,
        predocker_runner=predocker_runner,
    )
    supplemental = tuple(
        SupplementalStructureScores(
            source=source,
            scores=_maybe_predock(
                predocker,
                _resolve_supplemental_structure_scores(
                    evaluation_scores,
                    source,
                    mutator_name,
                ),
                predocker_runner=predocker_runner,
            ),
        )
        for source in normalize_supplemental_structures(supplemental_structures)
    )
    return StructureInputScores(primary=primary, supplemental=supplemental)


def _maybe_predock(
    predocker: Any,
    scores: StandardScores,
    *,
    predocker_runner: Any | None = None,
) -> StandardScores:
    if predocker is None:
        return scores
    runner = run_nested_predocker_on_scores if predocker_runner is None else predocker_runner
    return runner(predocker, scores)


def _resolve_supplemental_structure_scores(
    evaluation_scores: StandardScores,
    source: str,
    mutator_name: str,
) -> StandardScores:
    if source == SUPPLEMENTAL_STRUCTURE_START_REFERENCE:
        return _structure_start_reference_scores(evaluation_scores, mutator_name)
    raise AssertionError(f"Unhandled supplemental structure source: {source}")


def _structure_start_reference_scores(
    evaluation_scores: StandardScores,
    mutator_name: str,
) -> StandardScores:
    try:
        state = global_config.structure_start
    except RuntimeError as exc:
        raise ValueError(
            f"{mutator_name} supplemental_structures='structure_start_reference' "
            "requires global_config.structure_start to be initialized."
        ) from exc
    if state is None or getattr(state, "reference_scores", None) is None:
        raise ValueError(
            f"{mutator_name} supplemental_structures='structure_start_reference' "
            "requires structure_start references."
        )

    reference_scores = state.reference_scores
    reference_paths = getattr(reference_scores, "complex_paths", {}) or {}
    reference_names = set(getattr(reference_scores, "names", []) or [])
    missing = [
        oligo_name
        for oligo_name in evaluation_scores.names
        if oligo_name not in reference_names or oligo_name not in reference_paths
    ]
    if missing:
        missing_list = ", ".join(missing)
        available = ", ".join(sorted(str(k) for k in reference_paths)) or "<none>"
        raise ValueError(
            f"{mutator_name} supplemental_structures='structure_start_reference' is "
            f"missing structure_start reference(s) for oligo(s): {missing_list}. "
            f"Available references: {available}."
        )

    return reference_scores.filter_by_oligomer(list(evaluation_scores.names))
