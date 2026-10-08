from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_REFOLD_BACKENDS = ("af3", "protenix")
REFOLD_BACKEND_ALIASES = {
    "af3": "af3",
    "alphafold3": "af3",
    "px": "protenix",
    "protenix": "protenix",
}
REFOLD_BACKEND_CHOICES = DEFAULT_REFOLD_BACKENDS
REFOLD_BACKEND_COMPLETION_CHOICES = (*DEFAULT_REFOLD_BACKENDS, ",".join(DEFAULT_REFOLD_BACKENDS))


@dataclass(frozen=True)
class RefoldBackendSelection:
    backends: tuple[str, ...]

    @property
    def backend_set(self) -> set[str]:
        return set(self.backends)

    def as_list(self) -> list[str]:
        return list(self.backends)


def canonical_refold_backend_name(raw: object) -> str | None:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    return REFOLD_BACKEND_ALIASES.get(value)


def parse_refold_backend_selection(
    raw_values: Sequence[object] | str | None,
    *,
    default: Iterable[str] = DEFAULT_REFOLD_BACKENDS,
) -> RefoldBackendSelection:
    tokens = _split_backend_tokens(raw_values)
    if not tokens:
        tokens = [str(value) for value in default]

    backends: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        backend = canonical_refold_backend_name(token)
        if backend is None:
            invalid.append(str(token))
            continue
        if backend not in seen:
            seen.add(backend)
            backends.append(backend)

    if invalid:
        allowed = ",".join(REFOLD_BACKEND_CHOICES)
        bad = ",".join(invalid)
        raise ValueError(f"Unsupported refold backend(s): {bad}; expected one of: {allowed}")
    return RefoldBackendSelection(tuple(backends))


def requested_refold_backends(evaluators: Sequence[object] | str | None) -> set[str]:
    return parse_refold_backend_selection(evaluators).backend_set


def refold_backend_names(evaluators: Sequence[object] | str | None) -> list[str]:
    return parse_refold_backend_selection(evaluators).as_list()


def item_needed_backends(
    raw_needed: object,
    requested: set[str],
) -> set[str]:
    if not isinstance(raw_needed, list) or not raw_needed:
        return set(requested)
    needed = parse_refold_backend_selection(raw_needed, default=()).backend_set
    return needed or set(requested)


def _split_backend_tokens(raw_values: Sequence[object] | str | None) -> list[str]:
    if raw_values is None:
        return []
    values: list[object]
    if isinstance(raw_values, str):
        values = [raw_values]
    else:
        values = list(raw_values)
    tokens: list[str] = []
    for value in values:
        for part in str(value or "").split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tokens
