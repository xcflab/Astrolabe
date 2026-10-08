from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeUsage:
    evaluator_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    mutator_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    predocker_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    loss_terms: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def evaluator_types(self) -> set[str]:
        return _term_types(self.evaluator_terms)

    @property
    def mutator_types(self) -> set[str]:
        return _term_types(self.mutator_terms)

    @property
    def predocker_types(self) -> set[str]:
        return _term_types(self.predocker_terms)

    @property
    def loss_types(self) -> set[str]:
        return _term_types(self.loss_terms)

    @property
    def used_predocker_aliases(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for conf in self.mutator_terms.values():
            for alias in _as_list(conf.get("predocker")):
                alias_str = str(alias)
                if alias_str in seen or alias_str not in self.predocker_terms:
                    continue
                seen.add(alias_str)
                ordered.append(alias_str)
        return ordered


def analyze_runtime_usage(task_cfg: Any) -> RuntimeUsage:
    if not isinstance(task_cfg, Mapping):
        return RuntimeUsage()

    evaluator_terms = _terms(task_cfg, "evaluator", "evaluator_terms")
    mutator_terms = _terms(task_cfg, "mutator", "mutator_terms")
    predocker_terms = _terms(task_cfg, "predocker", "predocker_terms")
    loss_terms = _terms(task_cfg, "loss", "loss_terms")

    used_evaluator_aliases = _used_evaluator_aliases(task_cfg)
    used_mutator_aliases = _used_mutator_aliases(task_cfg)

    if used_evaluator_aliases:
        evaluator_terms = _select_terms(evaluator_terms, used_evaluator_aliases)
    if used_mutator_aliases:
        mutator_terms = _select_terms(mutator_terms, used_mutator_aliases)

    used_predocker_aliases: list[str] = []
    seen_predockers: set[str] = set()
    for conf in mutator_terms.values():
        for alias in _as_list(conf.get("predocker")):
            alias_str = str(alias)
            if alias_str in seen_predockers or alias_str not in predocker_terms:
                continue
            seen_predockers.add(alias_str)
            used_predocker_aliases.append(alias_str)
    if used_predocker_aliases:
        predocker_terms = _select_terms(predocker_terms, used_predocker_aliases)

    return RuntimeUsage(
        evaluator_terms=evaluator_terms,
        mutator_terms=mutator_terms,
        predocker_terms=predocker_terms,
        loss_terms=loss_terms,
    )


def _terms(task_cfg: Mapping[str, Any], section: str, terms_key: str) -> dict[str, dict[str, Any]]:
    section_cfg = task_cfg.get(section) or {}
    if not isinstance(section_cfg, Mapping):
        return {}
    raw_terms = section_cfg.get(terms_key) or {}
    if not isinstance(raw_terms, Mapping):
        return {}
    return {
        str(alias): dict(conf)
        for alias, conf in raw_terms.items()
        if str(alias) and isinstance(conf, Mapping)
    }


def _select_terms(
    terms: Mapping[str, dict[str, Any]],
    aliases: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        str(alias): dict(terms[str(alias)])
        for alias in aliases
        if str(alias) in terms
    }


def _term_types(terms: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        term_type
        for conf in terms.values()
        for term_type in [str(conf.get("_type") or "").strip()]
        if term_type
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _used_evaluator_aliases(task_cfg: Mapping[str, Any]) -> list[str]:
    stages = task_cfg.get("stages") or {}
    if not isinstance(stages, Mapping):
        return []
    default_alias = None
    default_stage = stages.get("default")
    if isinstance(default_stage, Mapping):
        default_alias = default_stage.get("evaluator")

    ordered: list[str] = []
    seen: set[str] = set()

    def note(value: Any) -> None:
        alias = str(value or "")
        if alias and alias not in seen:
            seen.add(alias)
            ordered.append(alias)

    for stage_conf in stages.values():
        if not isinstance(stage_conf, Mapping):
            continue
        note(stage_conf.get("evaluator") or default_alias)
        gate = stage_conf.get("transition_consistency_gate")
        if isinstance(gate, Mapping):
            note(gate.get("evaluator"))
    return ordered


def _used_mutator_aliases(task_cfg: Mapping[str, Any]) -> list[str]:
    stages = task_cfg.get("stages") or {}
    if not isinstance(stages, Mapping):
        return []
    default_stage = stages.get("default")
    default_mutators = (
        [str(alias) for alias in _as_list(default_stage.get("mutators")) if str(alias)]
        if isinstance(default_stage, Mapping)
        else []
    )

    ordered: list[str] = []
    seen: set[str] = set()

    def note_many(values: list[str]) -> None:
        for alias in values:
            if alias and alias not in seen:
                seen.add(alias)
                ordered.append(alias)

    note_many(default_mutators)
    for stage_conf in stages.values():
        if not isinstance(stage_conf, Mapping):
            continue
        mutators = stage_conf.get("mutators")
        if mutators:
            note_many([str(alias) for alias in _as_list(mutators) if str(alias)])
        elif default_mutators:
            note_many(default_mutators)
    initializer = task_cfg.get("initializer") or {}
    if isinstance(initializer, Mapping):
        mode = str(initializer.get("mode") or "").strip().lower()
        if mode in {
            "structure",
            "start_from_structure",
            "start-from-structure",
            "existing_structure",
            "existing-structure",
            "rfd3",
            "rfd-3",
        }:
            alias = str(initializer.get("sequence_designer") or "").strip()
            if alias:
                note_many([alias])
    return ordered
