from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf

from astrolabe.posthoc.db import (
    get_run_metadata,
    get_run_signature,
    merge_out_roots,
    open_db,
    upsert_records,
    upsert_run_signature,
)
from astrolabe.posthoc.derive import (
    derive_af3,
    derive_boltz,
    derive_chai1,
    derive_channel_hole,
    derive_protenix,
)
from astrolabe.posthoc.io_utils import (
    chain_order_from_structure,
    file_signature,
    index_structures,
    parse_scores_csv,
    scan_run_dirs,
)
from astrolabe.posthoc.refold_backends import canonical_refold_backend_name


class MissingRequiredMetricsError(RuntimeError):
    pass


_COLLECTION_DERIVATION_VERSION = 3
_ALIAS_METRIC_EXACT_EXCLUDES = {
    "task",
    "run",
    "step",
    "oligo",
    "stage",
    "iteration",
    "accepted",
    "weighted_score",
    "binder_chain",
    "target_chain",
    "parse_error_count",
    "required_metrics_ok",
    "src_run_dir",
    "refold_backends",
}
_ALIAS_METRIC_EXACT_INCLUDES = {
    "plddt",
    "ptm",
    "iptm",
    "pde",
    "binder_ptm",
    "binder_iptm",
    "lis",
    "ranking_score",
    "fraction_disordered",
    "has_clash",
}
_ALIAS_METRIC_PREFIXES = (
    "chain_",
    "cross_",
    "intra_",
    "af3_",
    "halludesign_af3_",
    "protenix_",
    "boltz_",
    "ipae",
    "ipsae",
    "pae_",
    "target_chain_",
)


def _canonical_refold_backend_name(value: Any) -> str | None:
    return canonical_refold_backend_name(value)


def _is_valid_required_metric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed and parsed not in {float("inf"), float("-inf")}


def _resolve_required_metrics(
    *,
    required_metrics: Iterable[str] | None,
    required_metrics_preset: Path | None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for key in required_metrics or []:
        key_norm = str(key).strip()
        if not key_norm or key_norm in seen:
            continue
        seen.add(key_norm)
        ordered.append(key_norm)

    if required_metrics_preset is not None:
        from astrolabe.posthoc.filtering import load_preset

        preset = load_preset(Path(required_metrics_preset))
        for rule in preset.rules:
            if rule.allow_missing:
                continue
            key_norm = str(rule.key).strip()
            if not key_norm or key_norm in seen:
                continue
            seen.add(key_norm)
            ordered.append(key_norm)
    return ordered


def _match_optional_regex(s: str, pat: str | None) -> bool:
    if not pat:
        return True
    return re.search(pat, s) is not None


def _normalize_out_roots(
    *,
    out_root: Path | None = None,
    out_roots: Iterable[Path] | None = None,
) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    candidates: list[Path] = []
    if out_root is not None:
        candidates.append(Path(out_root))
    for item in out_roots or []:
        candidates.append(Path(item))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Output root is not a directory: {resolved}. "
                "Use --out-root for an existing GH output directory containing scores.csv."
            )
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)
    if not ordered:
        raise ValueError("At least one out_root is required")
    return ordered


def _collection_context(*, binder_chain: str, target_chain: str) -> Dict[str, Any]:
    return {
        "derivation_version": _COLLECTION_DERIVATION_VERSION,
        "binder_chain": str(binder_chain),
        "target_chain": str(target_chain),
    }


def _archive_relpath_for_run(*, run_dir: Path) -> Path:
    digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:8]
    task = run_dir.parent.name
    run = run_dir.name
    return Path(task) / f"{run}__{digest}__raw_model_results.tar.gz"


def _archive_run_raw_model_results(
    *,
    run_dir: Path,
    archive_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    raw_dir = run_dir / "raw_model_results"
    archive_path = archive_dir / _archive_relpath_for_run(run_dir=run_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        return {
            "run_dir": str(run_dir),
            "archive_path": str(archive_path),
            "status": "missing_raw_model_results",
        }
    if archive_path.exists() and not overwrite:
        return {
            "run_dir": str(run_dir),
            "archive_path": str(archive_path),
            "status": "exists",
        }
    tmp = archive_path.with_suffix(archive_path.suffix + f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(raw_dir, arcname="raw_model_results")
    os.replace(str(tmp), str(archive_path))
    return {
        "run_dir": str(run_dir),
        "archive_path": str(archive_path),
        "status": "created",
    }


def _load_run_evaluator_context(run_dir: Path) -> Dict[str, Any]:
    cfg_path = Path(run_dir) / "config.yaml"
    metadata: Dict[str, Any] = {
        "stage_evaluator_aliases": {},
        "evaluator_alias_map": {},
    }
    if not cfg_path.exists():
        return metadata
    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception:
        return metadata

    try:
        raw_stages = OmegaConf.to_container(getattr(cfg, "stages", {}), resolve=False)
    except Exception:
        raw_stages = {}
    if isinstance(raw_stages, dict):
        for stage_name, stage_conf in raw_stages.items():
            if not isinstance(stage_conf, dict):
                continue
            alias = str(stage_conf.get("evaluator") or "").strip()
            if not alias:
                continue
            metadata["stage_evaluator_aliases"][str(stage_name)] = alias

    evaluator_node = getattr(cfg, "evaluator", None)
    try:
        raw_terms = OmegaConf.to_container(
            getattr(evaluator_node, "evaluator_terms", {}), resolve=False
        )
    except Exception:
        raw_terms = {}
    if isinstance(raw_terms, dict):
        for alias, conf in raw_terms.items():
            if conf is None:
                continue
            entry: Dict[str, Any] = {}
            if isinstance(conf, dict):
                actual = str(conf.get("_type") or "").strip()
                if actual:
                    entry["type"] = actual
                    entry["normalized_type"] = actual.lower()
            metadata["evaluator_alias_map"][str(alias)] = entry
    return metadata


def _is_alias_metric_candidate(key: str) -> bool:
    if key in _ALIAS_METRIC_EXACT_EXCLUDES:
        return False
    if key.startswith("__"):
        return False
    if "." in key:
        return False
    if re.match(r"^p[A-Z]_sequence$", key):
        return False
    if key.endswith("_length"):
        return False
    if key.endswith("_provenance") or key.endswith("_invalid_reason"):
        return False
    return key in _ALIAS_METRIC_EXACT_INCLUDES or key.startswith(
        _ALIAS_METRIC_PREFIXES
    )


def _merge_evaluator_contexts(*contexts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "stage_evaluator_aliases": {},
        "evaluator_alias_map": {},
    }
    for context in contexts:
        if not isinstance(context, dict):
            continue
        stage_aliases = context.get("stage_evaluator_aliases") or {}
        if isinstance(stage_aliases, dict):
            merged_stage_aliases = merged["stage_evaluator_aliases"]
            for stage_name, alias in stage_aliases.items():
                alias_str = str(alias or "").strip()
                if alias_str:
                    merged_stage_aliases[str(stage_name)] = alias_str
        alias_map = context.get("evaluator_alias_map") or {}
        if isinstance(alias_map, dict):
            merged_alias_map = merged["evaluator_alias_map"]
            for alias, entry in alias_map.items():
                if not isinstance(entry, dict):
                    continue
                merged_alias_map[str(alias)] = dict(entry)
    return merged


def _aliases_for_refold_backends(
    *,
    metrics: Dict[str, Any],
    evaluator_context: Dict[str, Any],
) -> list[str]:
    requested_backends: set[str] = set()
    for part in str(metrics.get("refold_backends") or "").split(","):
        backend = _canonical_refold_backend_name(part)
        if backend is not None:
            requested_backends.add(backend)
    if not requested_backends:
        return []

    aliases: list[str] = []
    alias_map = evaluator_context.get("evaluator_alias_map") or {}
    if not isinstance(alias_map, dict):
        return aliases
    for alias, entry in alias_map.items():
        if not isinstance(entry, dict):
            continue
        backend = _canonical_refold_backend_name(
            entry.get("normalized_type") or entry.get("type")
        )
        alias_str = str(alias or "").strip()
        if backend in requested_backends and alias_str:
            aliases.append(alias_str)
    return aliases


def _apply_evaluator_alias_metrics(
    *,
    metrics: Dict[str, Any],
    evaluator_context: Dict[str, Any],
) -> None:
    aliases: list[str] = []
    stage_name = str(metrics.get("stage") or "").strip()
    if stage_name:
        stage_aliases = evaluator_context.get("stage_evaluator_aliases") or {}
        alias = str(stage_aliases.get(stage_name) or "").strip()
        if alias:
            aliases.append(alias)
    aliases.extend(_aliases_for_refold_backends(metrics=metrics, evaluator_context=evaluator_context))
    if not aliases:
        return

    source_items = [
        (key, value)
        for key, value in list(metrics.items())
        if isinstance(key, str) and _is_alias_metric_candidate(key)
    ]
    seen_aliases: set[str] = set()
    for alias in aliases:
        alias_str = str(alias or "").strip()
        if not alias_str or alias_str in seen_aliases:
            continue
        seen_aliases.add(alias_str)
        for key, value in source_items:
            metrics.setdefault(f"{alias_str}.{key}", value)


def _augment_with_scalar_defaults(*, metrics: Dict[str, Any], oligo: str) -> None:
    """
    Promote explicit oligo-scoped binder metrics to their unscoped aliases.

    Whole-complex ``<oligo>.iptm`` and ``<oligo>.ptm`` values are not binder
    metrics and must never be used as fallbacks.
    """

    parse_errors = metrics.setdefault("__parse_errors__", [])
    if not isinstance(parse_errors, list):
        parse_errors = []
        metrics["__parse_errors__"] = parse_errors

    for metric_name in ("binder_iptm", "binder_ptm"):
        if metric_name in metrics:
            continue
        source_key = f"{oligo}.{metric_name}"
        if source_key not in metrics:
            continue
        try:
            metrics[metric_name] = float(metrics[source_key])
            metrics[f"{metric_name}_provenance"] = f"scores:{source_key}"
        except (TypeError, ValueError) as exc:
            metrics[f"{metric_name}_invalid_reason"] = f"non_numeric:{source_key}"
            parse_errors.append(f"{metric_name}<-{source_key}: {exc}")


def collect_scores(
    *,
    out_root: Path | None = None,
    out_roots: Iterable[Path] | None = None,
    db_path: Path,
    workers: int,
    binder_chain: str = "B",
    target_chain: str = "A",
    task_regex: str | None = None,
    run_regex: str | None = None,
    export_csv: Path | None = None,
    archive_dir: Path | None = None,
    archive_workers: int = 0,
    archive_overwrite: bool = False,
    required_metrics: Iterable[str] | None = None,
    required_metrics_preset: Path | None = None,
    fail_on_missing_required: bool = True,
) -> dict[str, Any]:
    roots = _normalize_out_roots(out_root=out_root, out_roots=out_roots)
    db_path = Path(db_path).expanduser().resolve()
    required_keys = _resolve_required_metrics(
        required_metrics=required_metrics,
        required_metrics_preset=Path(required_metrics_preset).expanduser().resolve()
        if required_metrics_preset is not None
        else None,
    )
    run_dirs: list[Path] = []
    seen_run_dirs: set[str] = set()
    for root in roots:
        for run_dir in scan_run_dirs(root):
            key = str(run_dir)
            if key in seen_run_dirs:
                continue
            seen_run_dirs.add(key)
            run_dirs.append(run_dir)

    # Filter scan list.
    filtered = []
    for run_dir in run_dirs:
        task = run_dir.parent.name
        run = run_dir.name
        if not _match_optional_regex(task, task_regex):
            continue
        if not _match_optional_regex(run, run_regex):
            continue
        filtered.append(run_dir)
    run_dirs = filtered

    conn = open_db(db_path)
    merge_out_roots(conn, roots=[str(root) for root in roots])
    collection_context = _collection_context(
        binder_chain=binder_chain,
        target_chain=target_chain,
    )
    to_process: List[Path] = []
    skipped = 0
    for run_dir in run_dirs:
        sig = file_signature(run_dir / "scores.csv")
        existing = get_run_signature(conn, str(run_dir))
        existing_metadata = get_run_metadata(conn, run_dir=run_dir)
        if (
            existing is not None
            and existing == sig
            and existing_metadata.get("posthoc_collection_context")
            == collection_context
        ):
            skipped += 1
            continue
        to_process.append(run_dir)

    results: List[Dict[str, Any]] = []

    def _process_one(
        run_dir: Path,
    ) -> tuple[
        Path,
        tuple[int, int],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        int,
        Dict[str, int],
        Dict[str, int],
        int,
    ]:
        scores_path = run_dir / "scores.csv"
        scores_sig = file_signature(scores_path)
        task = run_dir.parent.name
        run = run_dir.name
        run_parse_errors = 0
        run_missing_required_counts: Dict[str, int] = {}
        run_invalid_required_counts: Dict[str, int] = {}
        run_required_failed_records = 0

        step_rows = parse_scores_csv(scores_path)
        structures = index_structures(run_dir / "structures")
        evaluator_context = _load_run_evaluator_context(run_dir)
        run_evaluator_context = _merge_evaluator_contexts(evaluator_context)
        run_evaluator_context["posthoc_collection_context"] = collection_context
        source_evaluator_context_cache: Dict[str, Dict[str, Any]] = {}

        # Derived metrics (best-effort).
        derived: Dict[tuple[int, str], Dict[str, Any]] = {}
        for derive_fn in (
            lambda: derive_af3(run_dir, binder_chain=binder_chain, target_chain=target_chain),
            lambda: derive_protenix(
                run_dir,
                binder_chain=binder_chain,
                target_chain=target_chain,
            ),
            lambda: derive_boltz(run_dir, binder_chain=binder_chain, target_chain=target_chain),
            lambda: derive_chai1(run_dir),
            lambda: derive_channel_hole(run_dir),
        ):
            try:
                d = derive_fn()
            except Exception as exc:
                run_parse_errors += 1
                d = {}
                if not step_rows:
                    key = (0, "__step__")
                    derived.setdefault(key, {}).setdefault("__parse_errors__", []).append(
                        f"derive_failed:{type(exc).__name__}:{exc}"
                    )
            for k, v in d.items():
                derived.setdefault(k, {}).update(v)

        # Prefer structure path in structures/, but fall back to raw_model_results if present.
        records: List[Dict[str, Any]] = []
        for step, base_metrics in step_rows.items():
            # Candidate oligomers: those with structures, or those with derived metrics.
            oligos: set[str] = set()
            for (s, o) in structures.keys():
                if s == step:
                    oligos.add(o)
            for (s, o) in derived.keys():
                if s == step:
                    oligos.add(o)
            if not oligos:
                # Still keep a row for step-level inspection.
                oligos = {"__step__"}

            for oligo in sorted(oligos):
                metrics = dict(base_metrics)
                metrics.update(derived.get((step, oligo), {}))
                metrics["task"] = task
                metrics["run"] = run
                metrics["step"] = step
                metrics["oligo"] = oligo
                metrics["binder_chain"] = binder_chain
                metrics["target_chain"] = target_chain
                _augment_with_scalar_defaults(metrics=metrics, oligo=str(oligo))

                structure_path = None
                if (step, oligo) in structures:
                    structure_path = structures[(step, oligo)]
                else:
                    # AF3 fallback
                    p = run_dir / "raw_model_results" / "alphafold3" / f"step_{step:03d}_{oligo}" / f"step_{step:03d}_{oligo}_model.cif"
                    if p.exists():
                        structure_path = p
                    # Boltz fallback
                    if structure_path is None:
                        cands = sorted(
                            (run_dir / "raw_model_results" / "boltz" / f"step_{step:03d}_{oligo}").glob("predictions/*/*_model_0.cif")
                        )
                        if cands:
                            structure_path = cands[0]
                    # Protenix fallback (match best sample)
                    if structure_path is None:
                        cands = sorted(
                            (run_dir / "raw_model_results" / "protenix" / f"step_{step:03d}" / str(oligo)).glob("seed_*/predictions/*_sample_*.cif")
                        )
                        if cands:
                            structure_path = cands[0]

                # Refine binder metrics when raw chain vectors are present.
                # NOTE: Parsing structures is expensive; only do it when we actually have
                # per-chain vectors that need mapping.
                need_chain_map = False
                if "binder_ptm" not in metrics and isinstance(metrics.get("protenix_chain_ptm"), list):
                    need_chain_map = True
                if "binder_iptm" not in metrics and isinstance(metrics.get("protenix_chain_iptm"), list):
                    need_chain_map = True

                if need_chain_map and structure_path is not None:
                    chain_order = chain_order_from_structure(Path(structure_path))
                    if chain_order and str(binder_chain) in chain_order:
                        b_idx = int(chain_order.index(str(binder_chain)))
                        parse_errors = metrics.setdefault("__parse_errors__", [])
                        if not isinstance(parse_errors, list):
                            parse_errors = []
                            metrics["__parse_errors__"] = parse_errors
                        if "binder_ptm" not in metrics:
                            vec = metrics.get("protenix_chain_ptm")
                            if isinstance(vec, list) and b_idx < len(vec):
                                try:
                                    metrics["binder_ptm"] = float(vec[b_idx])
                                    metrics["binder_ptm_provenance"] = f"protenix_chain_ptm[{b_idx}]"
                                except (TypeError, ValueError) as exc:
                                    metrics["binder_ptm_invalid_reason"] = f"non_numeric:protenix_chain_ptm[{b_idx}]"
                                    parse_errors.append(f"binder_ptm<-protenix_chain_ptm[{b_idx}]: {exc}")
                        if "binder_iptm" not in metrics:
                            vec = metrics.get("protenix_chain_iptm")
                            if isinstance(vec, list) and b_idx < len(vec):
                                try:
                                    metrics["binder_iptm"] = float(vec[b_idx])
                                    metrics["binder_iptm_provenance"] = f"protenix_chain_iptm[{b_idx}]"
                                except (TypeError, ValueError) as exc:
                                    metrics["binder_iptm_invalid_reason"] = f"non_numeric:protenix_chain_iptm[{b_idx}]"
                                    parse_errors.append(f"binder_iptm<-protenix_chain_iptm[{b_idx}]: {exc}")

                parse_errors = metrics.get("__parse_errors__", [])
                if not isinstance(parse_errors, list):
                    parse_errors = []
                    metrics["__parse_errors__"] = parse_errors

                missing_required: list[str] = []
                invalid_required: list[str] = []
                for key in required_keys:
                    if key not in metrics:
                        missing_required.append(key)
                        run_missing_required_counts[key] = (
                            run_missing_required_counts.get(key, 0) + 1
                        )
                        parse_errors.append(f"required_missing:{key}")
                        continue
                    if not _is_valid_required_metric(metrics.get(key)):
                        invalid_required.append(key)
                        run_invalid_required_counts[key] = (
                            run_invalid_required_counts.get(key, 0) + 1
                        )
                        parse_errors.append(f"required_invalid:{key}")

                if missing_required or invalid_required:
                    run_required_failed_records += 1
                    if missing_required:
                        metrics["missing_required_metrics"] = sorted(
                            set(missing_required)
                        )
                    if invalid_required:
                        metrics["invalid_required_metrics"] = sorted(
                            set(invalid_required)
                        )
                    metrics["required_metrics_ok"] = False
                else:
                    metrics["required_metrics_ok"] = True

                row_evaluator_context = run_evaluator_context
                source_run_dir_raw = str(metrics.get("src_run_dir") or "").strip()
                if source_run_dir_raw:
                    source_run_dir = Path(source_run_dir_raw).expanduser()
                    try:
                        source_cache_key = str(source_run_dir.resolve())
                    except Exception:
                        source_cache_key = str(source_run_dir)
                    source_evaluator_context = source_evaluator_context_cache.get(source_cache_key)
                    if source_evaluator_context is None:
                        source_evaluator_context = _load_run_evaluator_context(source_run_dir)
                        source_evaluator_context_cache[source_cache_key] = source_evaluator_context
                    row_evaluator_context = _merge_evaluator_contexts(
                        source_evaluator_context,
                        run_evaluator_context,
                    )
                    run_evaluator_context = _merge_evaluator_contexts(
                        run_evaluator_context,
                        source_evaluator_context,
                    )

                run_parse_errors += len(parse_errors)
                metrics["parse_error_count"] = len(parse_errors)
                _apply_evaluator_alias_metrics(
                    metrics=metrics,
                    evaluator_context=row_evaluator_context,
                )

                records.append(
                    {
                        "run_dir": str(run_dir),
                        "task": task,
                        "run": run,
                        "step": step,
                        "oligo": oligo,
                        "structure_path": str(structure_path) if structure_path else None,
                        "metrics": metrics,
                        "scores_mtime_ns": scores_sig[0],
                        "scores_size": scores_sig[1],
                    }
                )
        run_evaluator_context["posthoc_collection_context"] = collection_context
        return (
            run_dir,
            scores_sig,
            task,
            run,
            run_evaluator_context,
            records,
            run_parse_errors,
            run_missing_required_counts,
            run_invalid_required_counts,
            run_required_failed_records,
        )

    max_workers = int(workers) if int(workers) > 0 else max(1, (os.cpu_count() or 1) // 2)
    missing_required_counts: Dict[str, int] = {}
    invalid_required_counts: Dict[str, int] = {}
    required_failed_records_total = 0
    archived_created = 0
    archived_skipped = 0
    archive_failed = 0
    archive_paths: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_one, rd): rd for rd in to_process}
        for fut in as_completed(futs):
            (
                run_dir,
                scores_sig,
                task,
                run,
                evaluator_context,
                recs,
                run_parse_errors,
                run_missing_required_counts,
                run_invalid_required_counts,
                run_required_failed_records,
            ) = fut.result()
            upsert_records(conn, recs)
            upsert_run_signature(
                conn,
                run_dir=str(run_dir),
                task=task,
                run=run,
                metadata=evaluator_context,
                scores_mtime_ns=scores_sig[0],
                scores_size=scores_sig[1],
            )
            results.append(
                {
                    "run_dir": str(run_dir),
                    "task": task,
                    "run": run,
                    "records": len(recs),
                    "parse_error_count": int(run_parse_errors),
                    "required_failed_records": int(run_required_failed_records),
                }
            )
            required_failed_records_total += int(run_required_failed_records)
            for key, count in run_missing_required_counts.items():
                missing_required_counts[key] = missing_required_counts.get(key, 0) + int(
                    count
                )
            for key, count in run_invalid_required_counts.items():
                invalid_required_counts[key] = invalid_required_counts.get(key, 0) + int(
                    count
                )

    if archive_dir is not None:
        archive_dir = Path(archive_dir).expanduser().resolve()
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_pool_workers = (
            int(archive_workers)
            if int(archive_workers) > 0
            else max(1, min(4, len(results) or 1))
        )
        with ThreadPoolExecutor(max_workers=archive_pool_workers) as ex:
            futs = {
                ex.submit(
                    _archive_run_raw_model_results,
                    run_dir=Path(item["run_dir"]),
                    archive_dir=archive_dir,
                    overwrite=bool(archive_overwrite),
                ): item
                for item in results
            }
            for fut in as_completed(futs):
                try:
                    archive_result = fut.result()
                except Exception:
                    archive_failed += 1
                    continue
                if archive_result.get("archive_path"):
                    archive_paths.append(str(archive_result["archive_path"]))
                status = str(archive_result.get("status") or "")
                if status == "created":
                    archived_created += 1
                elif status in {"exists", "missing_raw_model_results"}:
                    archived_skipped += 1
                else:
                    archive_failed += 1

    summary = {
        "out_root": str(roots[0]) if len(roots) == 1 else "",
        "out_roots": [str(root) for root in roots],
        "db_path": str(db_path),
        "runs_found": len(run_dirs),
        "runs_skipped": skipped,
        "runs_processed": len(to_process),
        "workers": max_workers,
        "processed": results,
        "parse_error_total": int(sum(int(r.get("parse_error_count", 0)) for r in results)),
        "required_metrics": list(required_keys),
        "required_failed_records_total": int(required_failed_records_total),
        "missing_required_counts": {
            k: int(v) for k, v in sorted(missing_required_counts.items())
        },
        "invalid_required_counts": {
            k: int(v) for k, v in sorted(invalid_required_counts.items())
        },
    }
    if archive_dir is not None:
        summary.update(
            {
                "archive_dir": str(archive_dir),
                "archived_created": int(archived_created),
                "archived_skipped": int(archived_skipped),
                "archive_failed": int(archive_failed),
                "archive_paths": sorted(archive_paths),
            }
        )

    if export_csv is not None:
        export_csv = Path(export_csv).expanduser().resolve()
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        export_records_to_csv(db_path=db_path, csv_path=export_csv)
        summary["export_csv"] = str(export_csv)

    if (
        fail_on_missing_required
        and required_keys
        and (missing_required_counts or invalid_required_counts)
    ):
        def _fmt_counts(d: Dict[str, int]) -> str:
            return ", ".join(
                f"{k}({v})"
                for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
            )

        msg = (
            "Required metrics are missing/invalid for collected records.\n"
            f"- db: {db_path}\n"
            f"- required_metrics: {', '.join(required_keys)}\n"
            f"- required_failed_records: {required_failed_records_total}\n"
        )
        if missing_required_counts:
            msg += f"- missing_required: {_fmt_counts(missing_required_counts)}\n"
        if invalid_required_counts:
            msg += f"- invalid_required: {_fmt_counts(invalid_required_counts)}\n"
        msg += (
            "Fix upstream outputs or rerun with `--allow-missing-required` for exploratory collection."
        )
        raise MissingRequiredMetricsError(msg)

    return summary


def export_records_to_csv(*, db_path: Path, csv_path: Path, oligo: str | None = None) -> None:
    """
    Export all records to a wide CSV for human inspection.
    """
    import csv
    import json

    from astrolabe.posthoc.db import iter_records, open_db

    conn = open_db(db_path)
    # First pass: union keys
    keys: set[str] = set()
    for row in iter_records(conn, oligo=oligo):
        try:
            d = json.loads(row["metrics_json"])
        except (TypeError, json.JSONDecodeError):
            d = {}
        if isinstance(d, dict):
            keys.update(d.keys())
    fixed = ["task", "run", "step", "oligo", "run_dir", "structure_path"]
    fieldnames = fixed + sorted(k for k in keys if k not in set(fixed))

    tmp = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in iter_records(conn, oligo=oligo):
            try:
                d = json.loads(row["metrics_json"])
            except (TypeError, json.JSONDecodeError):
                d = {}
            if not isinstance(d, dict):
                d = {}
            out = {k: "" for k in fieldnames}
            out["task"] = row["task"]
            out["run"] = row["run"]
            out["step"] = row["step"]
            out["oligo"] = row["oligo"]
            out["run_dir"] = row["run_dir"]
            out["structure_path"] = row["structure_path"] or ""
            for k, v in d.items():
                if k in out:
                    out[k] = v
            w.writerow(out)
    os.replace(str(tmp), str(csv_path))
