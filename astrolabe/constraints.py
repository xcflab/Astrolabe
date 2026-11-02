"""Explicit constraint categories; unsupported combinations fail closed."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConstraintKind(str, Enum):
    TOPOLOGY = "topology"
    SOFT = "soft"
    HARD = "hard"
    TEMPLATE = "template"
    INITIAL_COORDINATES = "initial_coordinates"
    POSTHOC = "posthoc"


@dataclass(frozen=True)
class Constraint:
    kind: ConstraintKind
    payload: Any
    mechanism: str


def validate_constraints(identity: str, constraints: list[Constraint], supported: dict[str, set[tuple[ConstraintKind, str]]]) -> None:
    allowed = supported.get(identity, set())
    for constraint in constraints:
        if (constraint.kind, constraint.mechanism) not in allowed:
            raise ValueError(f"Unsupported constraint {constraint.kind.value}/{constraint.mechanism} for {identity}")
