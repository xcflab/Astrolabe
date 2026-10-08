from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from astrolabe.core.protomer_config_parser import MutableTypes, ProtomerTypes


@dataclass(frozen=True)
class MutatorCapabilities:
    supported_types: frozenset[ProtomerTypes] | None = None
    needs_structure: bool = False
    outputs_probabilities: bool = False
    outputs_scores: bool = False
    supports_transition_blending: bool = False
    supports_shared_selection_policy: bool = False
    supports_initialization_from_structure: bool = False

    def supports_type(self, protomer_type: ProtomerTypes) -> bool:
        if self.supported_types is None:
            return True
        return protomer_type in self.supported_types


@dataclass(frozen=True)
class MutatorSelectionPolicy:
    """Opt-in helper for future shared mutation candidate selection."""

    sort_by: str = "none"
    descending: bool = False
    shuffle: bool = False
    limit: int | None = None
    per_entity_limit: dict[str, int] = field(default_factory=dict)
    seed: int | None = None

    def apply(self, candidates: Iterable[Any]) -> list[Any]:
        items = list(candidates)
        if self.sort_by not in {"none", "entropy", "score"}:
            raise ValueError(
                "MutatorSelectionPolicy.sort_by must be one of: none, entropy, score"
            )
        if self.sort_by != "none":
            items = sorted(
                items,
                key=lambda item: _candidate_metric(item, self.sort_by),
                reverse=self.descending,
            )
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(items)
        if self.per_entity_limit:
            items = _apply_per_entity_limit(items, self.per_entity_limit)
        if self.limit is not None:
            items = items[: max(0, int(self.limit))]
        return items


MUTABLE_TYPE_SET = frozenset(MutableTypes)
PROTEIN_ONLY = frozenset({ProtomerTypes.PROTEIN})
PROTEIN_RNA_DNA = frozenset(
    {ProtomerTypes.PROTEIN, ProtomerTypes.RNA, ProtomerTypes.DNA}
)


def validate_mutator_capabilities(
    mutator: Any,
    *,
    alias: str,
    protomers: Mapping[str, Any],
) -> None:
    capabilities = getattr(mutator, "capabilities", None)
    if capabilities is None or capabilities.supported_types is None:
        return
    observed_types = {
        getattr(protomer, "type", None)
        for protomer in (protomers or {}).values()
        if getattr(protomer, "type", None) is not None
    }
    if not observed_types:
        return
    if observed_types.intersection(capabilities.supported_types):
        return
    supported = ", ".join(t.value for t in sorted(capabilities.supported_types, key=lambda t: t.value))
    observed = ", ".join(t.value for t in sorted(observed_types, key=lambda t: t.value))
    raise ValueError(
        f"Mutator alias {alias!r} does not support any configured protomer types "
        f"(observed: {observed}; supported: {supported})."
    )


def _candidate_metric(candidate: Any, name: str) -> float:
    if isinstance(candidate, Mapping):
        value = candidate.get(name)
        if value is None and name == "entropy":
            value = candidate.get("entropy_norm")
        if value is None:
            return 0.0
        return float(value)
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        payload = candidate[1] if len(candidate) > 1 else candidate
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            idx = 3 if name == "entropy" else -1
            try:
                return float(payload[idx])
            except Exception:
                return 0.0
    return float(getattr(candidate, name, 0.0) or 0.0)


def _candidate_entity(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        for key in ("entity", "protomer", "proto", "name"):
            value = candidate.get(key)
            if value is not None:
                return str(value)
        return ""
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        return str(candidate[0]) if candidate else ""
    return str(getattr(candidate, "entity", ""))


def _apply_per_entity_limit(
    candidates: list[Any],
    limits: Mapping[str, int],
) -> list[Any]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[Any] = []
    for candidate in candidates:
        entity = _candidate_entity(candidate)
        limit = limits.get(entity)
        if limit is not None and counts[entity] >= int(limit):
            continue
        selected.append(candidate)
        counts[entity] += 1
    return selected
