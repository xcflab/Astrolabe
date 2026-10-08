from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import ComplexPoses, StandardScores
from astrolabe.stage_transition import (
    TransitionConsistencyGateResult,
    compute_reference_consistency,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id


@dataclass(frozen=True)
class ReferenceTmScoreConfig:
    enabled: bool = True
    threshold: float = 0.85


@dataclass(frozen=True)
class HotspotGateConfig:
    enabled: bool = False
    expression: str = "pair_hotspot_support"
    floor: float = 0.25
    reference_fraction: float = 0.8
    tolerance: float = 0.05


@dataclass(frozen=True)
class StructureStartReference:
    oligo: str
    structure_alias: str
    chain_map: dict[str, str]


@dataclass(frozen=True)
class StructureStartConfig:
    mode: str
    gate_timing: tuple[str, ...]
    aggregate: str
    references: dict[str, StructureStartReference]
    reference_tmscore: ReferenceTmScoreConfig
    hotspot_gate: HotspotGateConfig
    sequence_policy: str = "strict"

    @classmethod
    def from_raw(cls, raw_value: Any) -> "StructureStartConfig | None":
        if raw_value is None:
            return None
        raw = _to_plain(raw_value)
        if raw in ({}, "null", "None"):
            return None
        if not isinstance(raw, dict):
            raise ValueError("structure_start must be a mapping.")

        allowed = {
            "mode",
            "gate_timing",
            "aggregate",
            "sequence_policy",
            "references",
            "reference_tmscore",
            "hotspot_gate",
        }
        unknown = sorted(set(raw.keys()) - allowed)
        if unknown:
            raise ValueError(f"structure_start has unsupported fields: {unknown}.")

        mode = str(raw.get("mode", "repredict_gate")).strip()
        if mode not in {"repredict_gate", "soft_select"}:
            raise ValueError(
                "structure_start.mode must be one of: repredict_gate, soft_select."
            )

        timings_raw = raw.get("gate_timing", ["step0", "pre_accept"])
        timings = _as_string_tuple(timings_raw, field="structure_start.gate_timing")
        allowed_timings = {"step0", "pre_accept", "stage_transition"}
        bad_timings = sorted(set(timings) - allowed_timings)
        if bad_timings:
            raise ValueError(
                f"structure_start.gate_timing has unsupported values: {bad_timings}."
            )

        aggregate = str(raw.get("aggregate", "min")).strip()
        if aggregate != "min":
            raise ValueError("structure_start.aggregate currently supports only 'min'.")

        sequence_policy = str(raw.get("sequence_policy", "strict")).strip()
        if sequence_policy not in {"strict", "length_only"}:
            raise ValueError(
                "structure_start.sequence_policy must be one of: strict, length_only."
            )

        references_raw = raw.get("references")
        if not isinstance(references_raw, dict) or not references_raw:
            raise ValueError("structure_start.references must be a non-empty mapping.")
        references: dict[str, StructureStartReference] = {}
        for oligo_raw, ref_raw in references_raw.items():
            oligo = str(oligo_raw).strip()
            if not oligo:
                raise ValueError("structure_start.references contains an empty oligo.")
            if not isinstance(ref_raw, dict):
                raise ValueError(f"structure_start.references.{oligo} must be a mapping.")
            ref_unknown = sorted(set(ref_raw.keys()) - {"structure_alias", "chain_map"})
            if ref_unknown:
                raise ValueError(
                    f"structure_start.references.{oligo} has unsupported fields: {ref_unknown}."
                )
            structure_alias = str(ref_raw.get("structure_alias", "")).strip()
            if not structure_alias:
                raise ValueError(
                    f"structure_start.references.{oligo}.structure_alias is required."
                )
            chain_map_raw = ref_raw.get("chain_map")
            if not isinstance(chain_map_raw, dict) or not chain_map_raw:
                raise ValueError(
                    f"structure_start.references.{oligo}.chain_map must be a non-empty mapping."
                )
            chain_map = {
                str(input_chain).strip(): str(proto).strip()
                for input_chain, proto in chain_map_raw.items()
            }
            if any(not k or not v for k, v in chain_map.items()):
                raise ValueError(
                    f"structure_start.references.{oligo}.chain_map cannot contain empty keys/values."
                )
            references[oligo] = StructureStartReference(
                oligo=oligo,
                structure_alias=structure_alias,
                chain_map=chain_map,
            )

        tm_raw = raw.get("reference_tmscore") or {}
        if not isinstance(tm_raw, dict):
            raise ValueError("structure_start.reference_tmscore must be a mapping.")
        tm_unknown = sorted(set(tm_raw.keys()) - {"enabled", "threshold"})
        if tm_unknown:
            raise ValueError(
                f"structure_start.reference_tmscore has unsupported fields: {tm_unknown}."
            )
        tm_threshold = _float_in_unit_interval(
            tm_raw.get("threshold", 0.85),
            field="structure_start.reference_tmscore.threshold",
        )
        reference_tmscore = ReferenceTmScoreConfig(
            enabled=bool(tm_raw.get("enabled", True)),
            threshold=tm_threshold,
        )

        hotspot_raw = raw.get("hotspot_gate") or {}
        if not isinstance(hotspot_raw, dict):
            raise ValueError("structure_start.hotspot_gate must be a mapping.")
        hotspot_unknown = sorted(
            set(hotspot_raw.keys())
            - {"enabled", "expression", "floor", "reference_fraction", "tolerance"}
        )
        if hotspot_unknown:
            raise ValueError(
                f"structure_start.hotspot_gate has unsupported fields: {hotspot_unknown}."
            )
        hotspot_gate = HotspotGateConfig(
            enabled=bool(hotspot_raw.get("enabled", False)),
            expression=str(hotspot_raw.get("expression", "pair_hotspot_support")).strip()
            or "pair_hotspot_support",
            floor=_float_in_unit_interval(
                hotspot_raw.get("floor", 0.25),
                field="structure_start.hotspot_gate.floor",
            ),
            reference_fraction=_float_in_unit_interval(
                hotspot_raw.get("reference_fraction", 0.8),
                field="structure_start.hotspot_gate.reference_fraction",
            ),
            tolerance=_float_in_unit_interval(
                hotspot_raw.get("tolerance", 0.05),
                field="structure_start.hotspot_gate.tolerance",
            ),
        )

        return cls(
            mode=mode,
            gate_timing=timings,
            aggregate=aggregate,
            references=references,
            reference_tmscore=reference_tmscore,
            hotspot_gate=hotspot_gate,
            sequence_policy=sequence_policy,
        )


@dataclass(frozen=True)
class StructureStartState:
    config: StructureStartConfig
    reference_scores: StandardScores
    manifest_path: Path
    reference_hotspot_score: float | None = None
    reference_hotspot_threshold: float | None = None


@dataclass(frozen=True)
class ReferenceGateResult:
    gate_active: bool
    score: float
    threshold: float
    passed: bool
    comparisons: dict[str, float]
    per_oligo_tmscore: dict[str, float] = dataclasses.field(default_factory=dict)
    hotspot_score: float | None = None
    hotspot_threshold: float | None = None
    hotspot_passed: bool | None = None
    candidate_count: int = 1
    selected_candidate_index: int = 0


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    value = _to_plain(value)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"{field} must be a string or list of strings.")
    out = tuple(str(item).strip() for item in items if str(item).strip())
    if not out:
        raise ValueError(f"{field} cannot be empty.")
    return out


def _float_in_unit_interval(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field} must be within [0, 1].")
    return parsed


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip() or "reference")


def _source_path_for_alias(alias: str) -> Path:
    try:
        named_files = global_config.named_files
    except RuntimeError as exc:
        raise ValueError("structure_start requires named_files to be registered.") from exc
    parsed_file = (named_files or {}).get(alias)
    if parsed_file is None:
        raise ValueError(f"structure_start reference alias not found: {alias!r}.")
    path = getattr(parsed_file, "file_path", None)
    if not path:
        raise ValueError(f"structure_start named_files[{alias!r}] has no file_path.")
    return Path(path)


def _chain_sequence_for_alias(alias: str, chain_id: str) -> str:
    parsed_file = global_config.named_files.get(alias)
    indices, sequence = parsed_file[str(chain_id)]
    if not indices and not sequence:
        raise ValueError(
            f"structure_start reference alias {alias!r} does not contain chain {chain_id!r}."
        )
    return sequence


def _ordered_input_chains(ref: StructureStartReference) -> list[str]:
    expected_counts = Counter(str(ref.oligo))
    mapped_by_proto: dict[str, list[str]] = defaultdict(list)
    for input_chain, mapped_proto in ref.chain_map.items():
        if mapped_proto not in expected_counts:
            raise ValueError(
                "structure_start reference "
                f"{ref.oligo!r} maps input chain {input_chain!r} to protomer "
                f"{mapped_proto!r}, which is not present in the oligo."
            )
        mapped_by_proto[mapped_proto].append(input_chain)

    for proto, expected_count in expected_counts.items():
        got = mapped_by_proto.get(proto, [])
        if len(got) != expected_count:
            raise ValueError(
                "structure_start reference "
                f"{ref.oligo!r} must map exactly {expected_count} input chain(s) "
                f"to protomer {proto!r}; got {got}."
            )

    ordered: list[str] = []
    cursors: dict[str, int] = {}
    for proto in ref.oligo:
        idx = cursors.get(proto, 0)
        ordered.append(mapped_by_proto[proto][idx])
        cursors[proto] = idx + 1
    return ordered


def _write_normalized_reference_structure(
    *,
    source_path: Path,
    input_chains: list[str],
    output_path: Path,
) -> None:
    from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Structure

    parser = (
        MMCIFParser(QUIET=True)
        if source_path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("structure_start", str(source_path))
    normalized = Structure.Structure("structure_start")

    for model in structure:
        new_model = Model.Model(model.id)
        available = {str(chain.id): chain for chain in model}
        for idx, input_chain in enumerate(input_chains):
            chain = available.get(str(input_chain))
            if chain is None:
                raise ValueError(
                    f"structure_start requested chain {input_chain!r}, "
                    f"but it is missing from {source_path}."
                )
            copied = chain.copy()
            copied.id = int_id_to_str_id(idx + 1)
            new_model.add(copied)
        normalized.add(new_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(normalized)
    io.save(str(output_path))


def build_structure_start_state(raw_value: Any, *, loss_computer=None) -> StructureStartState | None:
    config = StructureStartConfig.from_raw(raw_value)
    if config is None:
        return None

    artifact_root = Path(global_config.out_dir) / "structure_start" / "references"
    artifact_root.mkdir(parents=True, exist_ok=True)
    complex_paths: dict[str, Path] = {}
    manifest_refs: dict[str, dict[str, Any]] = {}

    protomers = global_config.protomers
    for oligo, ref in config.references.items():
        missing_protos = [proto for proto in oligo if proto not in protomers]
        if missing_protos:
            raise ValueError(
                f"structure_start reference {oligo!r} uses unknown protomer(s): {missing_protos}."
            )
        input_chains = _ordered_input_chains(ref)
        for input_chain, proto in zip(input_chains, oligo, strict=True):
            ref_seq = _chain_sequence_for_alias(ref.structure_alias, input_chain)
            proto_seq = protomers[proto].sequence
            if len(ref_seq) != len(proto_seq):
                raise ValueError(
                    "structure_start reference sequence length mismatch for "
                    f"{oligo}:{input_chain}->{proto}: reference length={len(ref_seq)} "
                    f"protomer length={len(proto_seq)}."
                )
            if config.sequence_policy == "strict" and ref_seq != proto_seq:
                raise ValueError(
                    "structure_start reference sequence mismatch for "
                    f"{oligo}:{input_chain}->{proto}: reference length={len(ref_seq)} "
                    f"protomer length={len(proto_seq)}."
                )

        source_path = _source_path_for_alias(ref.structure_alias)
        snapshot_path = artifact_root / f"{_safe_label(oligo)}_reference.pdb"
        _write_normalized_reference_structure(
            source_path=source_path,
            input_chains=input_chains,
            output_path=snapshot_path,
        )
        complex_paths[oligo] = snapshot_path
        manifest_refs[oligo] = {
            "structure_alias": ref.structure_alias,
            "source_path": str(source_path),
            "chain_map": dict(ref.chain_map),
            "ordered_input_chains": list(input_chains),
            "normalized_path": str(snapshot_path),
        }

    reference_scores = StandardScores(
        names=list(config.references.keys()),
        plddt={oligo: [1.0] for oligo in config.references.keys()},
        complex_paths=complex_paths,
    )
    if config.hotspot_gate.enabled:
        reference_scores = dataclasses.replace(
            reference_scores,
            complex_poses=ComplexPoses.construct_complex_poses(reference_scores),
        )

    reference_hotspot_score = None
    reference_hotspot_threshold = None
    if config.hotspot_gate.enabled:
        if loss_computer is None:
            raise ValueError("structure_start.hotspot_gate requires a loss computer.")
        reference_hotspot_score = _evaluate_hotspot_expression(
            loss_computer,
            reference_scores,
            config.hotspot_gate.expression,
            context_label="reference",
        )
        reference_hotspot_threshold = max(
            float(config.hotspot_gate.floor),
            float(config.hotspot_gate.reference_fraction)
            * float(reference_hotspot_score)
            - float(config.hotspot_gate.tolerance),
        )

    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": config.mode,
                "gate_timing": list(config.gate_timing),
                "aggregate": config.aggregate,
                "sequence_policy": config.sequence_policy,
                "reference_tmscore": dataclasses.asdict(config.reference_tmscore),
                "hotspot_gate": dataclasses.asdict(config.hotspot_gate),
                "reference_hotspot_score": reference_hotspot_score,
                "reference_hotspot_threshold": reference_hotspot_threshold,
                "references": manifest_refs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return StructureStartState(
        config=config,
        reference_scores=reference_scores,
        manifest_path=manifest_path,
        reference_hotspot_score=reference_hotspot_score,
        reference_hotspot_threshold=reference_hotspot_threshold,
    )


def _evaluate_hotspot_expression(
    loss_computer,
    scores: StandardScores,
    expression: str,
    *,
    context_label: str,
) -> float:
    try:
        env = loss_computer.score_context_for_expressions(scores, [expression])
        value = loss_computer.weighted_score(env, expression)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate structure_start.hotspot_gate.expression "
            f"{expression!r} on {context_label} structures."
        ) from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"structure_start.hotspot_gate.expression {expression!r} on "
            f"{context_label} structures did not produce a numeric value: {value!r}."
        ) from exc


def _per_oligo_tmscore(comparisons: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in comparisons.items():
        if not str(key).startswith("min~"):
            continue
        oligo = str(key).split("~", 1)[1]
        out[oligo] = float(value)
    return out


def evaluate_structure_start_gate(
    eval_result: StandardScores,
    *,
    timing: str,
    loss_computer=None,
    compare_fn=None,
    candidate_count: int = 1,
    selected_candidate_index: int = 0,
) -> ReferenceGateResult | None:
    try:
        state = global_config.structure_start
    except RuntimeError:
        return None
    if state is None:
        return None
    if timing not in state.config.gate_timing:
        return None

    tm_result = TransitionConsistencyGateResult(
        gate_active=True,
        score=1.0,
        threshold=0.0,
        passed=True,
        comparisons={},
    )
    if state.config.reference_tmscore.enabled:
        tm_result = compute_reference_consistency(
            current_scores=eval_result,
            reference_scores=state.reference_scores,
            threshold=state.config.reference_tmscore.threshold,
            compare_fn=compare_fn,
        )

    hotspot_score = None
    hotspot_threshold = state.reference_hotspot_threshold
    hotspot_passed = None
    if state.config.hotspot_gate.enabled:
        if loss_computer is None:
            raise ValueError("structure_start hotspot gate requires a loss computer.")
        hotspot_score = _evaluate_hotspot_expression(
            loss_computer,
            eval_result,
            state.config.hotspot_gate.expression,
            context_label="candidate",
        )
        hotspot_passed = bool(hotspot_score >= float(hotspot_threshold))

    hard_gate = state.config.mode == "repredict_gate"
    passed = bool(tm_result.passed)
    if hotspot_passed is not None:
        passed = passed and bool(hotspot_passed)

    return ReferenceGateResult(
        gate_active=hard_gate,
        score=float(tm_result.score),
        threshold=float(tm_result.threshold),
        passed=passed,
        comparisons={k: float(v) for k, v in tm_result.comparisons.items()},
        per_oligo_tmscore=_per_oligo_tmscore(tm_result.comparisons),
        hotspot_score=hotspot_score,
        hotspot_threshold=hotspot_threshold,
        hotspot_passed=hotspot_passed,
        candidate_count=int(candidate_count),
        selected_candidate_index=int(selected_candidate_index),
    )


def select_structure_start_candidate(
    eval_result: StandardScores,
    *,
    timing: str,
    loss_computer=None,
    compare_fn=None,
) -> tuple[StandardScores, ReferenceGateResult | None]:
    try:
        state = global_config.structure_start
    except RuntimeError:
        state = None
    if state is None or state.config.mode != "soft_select":
        return eval_result, evaluate_structure_start_gate(
            eval_result,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
        )

    candidates = (
        getattr(eval_result, "candidate_scores", None)
        or getattr(eval_result, "candidate_results", None)
        or None
    )
    if not candidates:
        result = evaluate_structure_start_gate(
            eval_result,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
            candidate_count=1,
            selected_candidate_index=0,
        )
        return eval_result, result

    scored: list[tuple[float, int, StandardScores, ReferenceGateResult | None]] = []
    for idx, candidate in enumerate(candidates):
        result = evaluate_structure_start_gate(
            candidate,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
            candidate_count=len(candidates),
            selected_candidate_index=idx,
        )
        score = float(getattr(result, "score", 0.0) if result is not None else 0.0)
        scored.append((score, idx, candidate, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    _score, idx, selected, result = scored[0]
    if result is not None:
        result = dataclasses.replace(
            result,
            candidate_count=len(candidates),
            selected_candidate_index=idx,
        )
    return selected, result


def reference_gate_score_context(result: ReferenceGateResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    context: dict[str, Any] = {
        "reference_tmscore": float(result.score),
        "reference_gate_passed": 1.0 if bool(result.passed) else 0.0,
        "reference_tmscore_threshold": float(result.threshold),
        "reference_soft_select_candidate_count": int(result.candidate_count),
        "reference_soft_select_selected_index": int(result.selected_candidate_index),
    }
    if result.hotspot_score is not None:
        context["reference_hotspot_score"] = float(result.hotspot_score)
    if result.hotspot_threshold is not None:
        context["reference_hotspot_threshold"] = float(result.hotspot_threshold)
    if result.hotspot_passed is not None:
        context["reference_hotspot_passed"] = 1.0 if result.hotspot_passed else 0.0
    for oligo, score in result.per_oligo_tmscore.items():
        context.setdefault(oligo, {})["reference_tmscore"] = float(score)
    return context
