from __future__ import annotations

import dataclasses
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from astrolabe.evaluators.utils import StandardScores
from astrolabe.utils.mmcif_ids import int_id_to_str_id


@dataclass(frozen=True)
class TransitionConsistencyGateConfig:
    weight: float = 1.0
    consistency_threshold: float = 0.8
    max_attempts: int = 10
    evaluator: str | None = None

    @classmethod
    def from_raw(
        cls, raw_value: Any, *, stage_name: str
    ) -> "TransitionConsistencyGateConfig":
        if not isinstance(raw_value, dict):
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate must be a mapping."
            )

        unknown_fields = sorted(
            set(raw_value.keys())
            - {"weight", "consistency_threshold", "max_attempts", "evaluator"}
        )
        if unknown_fields:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate has unsupported fields: {unknown_fields}."
            )

        try:
            weight = float(raw_value.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.weight must be numeric."
            ) from exc
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.weight must be within [0, 1]."
            )

        raw_threshold = raw_value.get("consistency_threshold", 0.8)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.consistency_threshold must be numeric."
            ) from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.consistency_threshold must be within [0, 1]."
            )

        raw_max_attempts = raw_value.get("max_attempts", 10)
        try:
            max_attempts = int(raw_max_attempts)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.max_attempts must be an integer >= 1."
            ) from exc
        if max_attempts < 1:
            raise ValueError(
                f"Stage '{stage_name}' transition_consistency_gate.max_attempts must be an integer >= 1."
            )

        raw_evaluator = raw_value.get("evaluator")
        evaluator = None
        if raw_evaluator is not None:
            evaluator = str(raw_evaluator).strip()
            if not evaluator:
                raise ValueError(
                    f"Stage '{stage_name}' transition_consistency_gate.evaluator must be a non-empty string."
                )

        return cls(
            weight=weight,
            consistency_threshold=threshold,
            max_attempts=max_attempts,
            evaluator=evaluator,
        )


@dataclass(frozen=True)
class TransitionConsistencyGateContext:
    weight: float
    consistency_threshold: float
    max_attempts: int
    evaluator: str | None
    source_stage: str | None
    target_stage: str
    reference_scores: StandardScores
    rollback_scores: StandardScores
    artifact_root: Path | None = None
    activation_step: int | None = None
    seed_scores: StandardScores | None = None


@dataclass(frozen=True)
class TransitionConsistencyGateResult:
    gate_active: bool
    score: float
    threshold: float
    passed: bool
    comparisons: dict[str, float]


def transition_consistency_gate_score_context(
    result: TransitionConsistencyGateResult | Any | None,
    *,
    source_stage: str | None = "",
    target_stage: str | None = "",
    attempt: int | None = None,
) -> dict[str, object]:
    if result is None or not bool(getattr(result, "gate_active", False)):
        return {}

    context: dict[str, object] = {
        "transition_consistency_gate_score": float(getattr(result, "score", 0.0)),
        "transition_consistency_gate_threshold": float(
            getattr(result, "threshold", 0.0)
        ),
        "transition_consistency_gate_passed": (
            1.0 if bool(getattr(result, "passed", False)) else 0.0
        ),
        "transition_consistency_gate_source_stage": source_stage or "",
        "transition_consistency_gate_target_stage": target_stage or "",
    }
    if attempt is not None:
        context["transition_consistency_gate_attempt"] = int(attempt)
    return context


def transition_consistency_gate_score_context_from_runtime(
    runtime_config: Any | None,
) -> dict[str, object]:
    if runtime_config is None:
        return {}
    if hasattr(runtime_config, "get"):
        result = runtime_config.get("transition_consistency_gate_result")
        metadata = runtime_config.get("transition_consistency_gate_metadata") or {}
    else:
        result = getattr(runtime_config, "transition_consistency_gate_result", None)
        metadata = (
            getattr(runtime_config, "transition_consistency_gate_metadata", {}) or {}
        )
    if not hasattr(metadata, "get"):
        metadata = {}
    return transition_consistency_gate_score_context(
        result,
        source_stage=metadata.get("source_stage", ""),
        target_stage=metadata.get("target_stage", ""),
        attempt=metadata.get("attempt"),
    )


def _sanitize_transition_label(value: str | None) -> str:
    label = (value or "unknown").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def build_transition_gate_artifact_root(
    *,
    out_dir: Path,
    step: int,
    source_stage: str | None,
    target_stage: str,
) -> Path:
    return Path(out_dir) / "transition_gate" / (
        f"step_{step:03d}_{_sanitize_transition_label(source_stage)}_to_{_sanitize_transition_label(target_stage)}"
    )


def snapshot_standard_scores(scores: StandardScores, *, output_dir: Path) -> StandardScores:
    from astrolabe.evaluators.utils import StandardScores

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: dict[str, Path] = {}
    for oligo_name in scores.names:
        source_path = Path(scores.complex_paths[str(oligo_name)])
        suffix = source_path.suffix or ".cif"
        snapshot_path = output_dir / f"{oligo_name}_model{suffix}"
        shutil.copy2(source_path, snapshot_path)
        snapshot_paths[str(oligo_name)] = snapshot_path
    return dataclasses.replace(scores, complex_paths=snapshot_paths)


def _relative_score_paths(scores: StandardScores, artifact_root: Path) -> dict[str, str]:
    return {
        str(oligo_name): str(Path(path).relative_to(artifact_root))
        for oligo_name, path in scores.complex_paths.items()
    }


def write_transition_gate_manifest(
    *,
    artifact_root: Path,
    source_stage: str | None,
    target_stage: str,
    threshold: float,
    max_attempts: int,
    activation_step: int | None,
    label: str,
    step: int | None,
    reference_scores: StandardScores,
    current_scores: StandardScores,
    result: TransitionConsistencyGateResult,
) -> Path:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "source_stage": source_stage or "",
            "target_stage": target_stage,
            "threshold": float(threshold),
            "max_attempts": int(max_attempts),
            "activation_step": activation_step,
            "reference_structures": _relative_score_paths(reference_scores, artifact_root),
            "evaluations": [],
        }

    evaluations = [
        item
        for item in payload.get("evaluations", [])
        if not (item.get("label") == label and item.get("step") == step)
    ]
    evaluations.append(
        {
            "label": label,
            "step": step,
            "score": float(result.score),
            "threshold": float(result.threshold),
            "passed": bool(result.passed),
            "comparisons": {k: float(v) for k, v in result.comparisons.items()},
            "structures": _relative_score_paths(current_scores, artifact_root),
        }
    )
    payload["evaluations"] = evaluations
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _map_reference_occurrences(reference_oligo: str, current_oligo: str) -> list[int] | None:
    current_occurrences: dict[tuple[str, int], int] = {}
    current_counts: dict[str, int] = {}
    for idx, proto in enumerate(str(current_oligo)):
        current_counts[proto] = current_counts.get(proto, 0) + 1
        current_occurrences[(proto, current_counts[proto])] = idx

    mapping: list[int] = []
    reference_counts: dict[str, int] = {}
    for proto in str(reference_oligo):
        reference_counts[proto] = reference_counts.get(proto, 0) + 1
        current_idx = current_occurrences.get((proto, reference_counts[proto]))
        if current_idx is None:
            return None
        mapping.append(current_idx)
    return mapping


def _extract_chain_subset_to_pdb(structure_path: Path, chain_ids: list[str]) -> Path:
    from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Structure

    path = Path(structure_path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("subset", str(path))
    subset = Structure.Structure("subset")

    for model in structure:
        subset_model = Model.Model(model.id)
        available = {chain.id: chain for chain in model}
        for chain_id in chain_ids:
            chain = available.get(chain_id)
            if chain is None:
                raise ValueError(
                    f"Requested chain {chain_id!r} is missing from structure {structure_path}."
                )
            subset_model.add(chain.copy())
        subset.add(subset_model)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".pdb",
    )
    handle.close()
    io = PDBIO()
    io.set_structure(subset)
    io.save(handle.name)
    return Path(handle.name)


def _compare_reference_to_current(
    *,
    current_structure: Path,
    current_oligo: str,
    reference_structure: Path,
    reference_oligo: str,
) -> float:
    from astrolabe.losses.structure import usalign_wrapper

    mapping = _map_reference_occurrences(reference_oligo, current_oligo)
    if mapping is None:
        return 0.0

    current_chain_ids = [int_id_to_str_id(idx + 1) for idx in mapping]
    reference_chain_ids = [
        int_id_to_str_id(idx + 1) for idx in range(len(str(reference_oligo)))
    ]

    current_subset = _extract_chain_subset_to_pdb(current_structure, current_chain_ids)
    reference_subset = _extract_chain_subset_to_pdb(
        reference_structure, reference_chain_ids
    )
    try:
        multimer = len(reference_chain_ids) > 1
        _, tm_query = usalign_wrapper(
            current_subset,
            reference_subset,
            None,
            multimer=multimer,
            tm_normalize_by="query",
        )
        _, tm_reference = usalign_wrapper(
            current_subset,
            reference_subset,
            None,
            multimer=multimer,
            tm_normalize_by="reference",
        )
        return float((tm_query + tm_reference) / 2.0)
    finally:
        for path in (current_subset, reference_subset):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def compute_reference_consistency(
    *,
    current_scores: StandardScores,
    reference_scores: StandardScores,
    threshold: float,
    compare_fn: Callable[..., float] | None = None,
) -> TransitionConsistencyGateResult:
    compare = compare_fn or _compare_reference_to_current
    comparisons: dict[str, float] = {}
    per_reference_scores: list[float] = []

    for reference_oligo in reference_scores.names:
        matched_scores: list[float] = []
        for current_oligo in current_scores.names:
            if _map_reference_occurrences(str(reference_oligo), str(current_oligo)) is None:
                continue
            score = float(
                compare(
                    current_structure=Path(current_scores.complex_paths[str(current_oligo)]),
                    current_oligo=str(current_oligo),
                    reference_structure=Path(reference_scores.complex_paths[str(reference_oligo)]),
                    reference_oligo=str(reference_oligo),
                )
            )
            comparisons[f"{current_oligo}~{reference_oligo}"] = score
            matched_scores.append(score)

        if not matched_scores:
            comparisons[f"missing~{reference_oligo}"] = 0.0
            return TransitionConsistencyGateResult(
                gate_active=True,
                score=0.0,
                threshold=float(threshold),
                passed=False,
                comparisons=comparisons,
            )

        reference_score = min(matched_scores)
        comparisons[f"min~{reference_oligo}"] = reference_score
        per_reference_scores.append(reference_score)

    overall = min(per_reference_scores) if per_reference_scores else 0.0
    threshold_value = float(threshold)
    return TransitionConsistencyGateResult(
        gate_active=True,
        score=float(overall),
        threshold=threshold_value,
        passed=bool(overall >= threshold_value),
        comparisons=comparisons,
    )
