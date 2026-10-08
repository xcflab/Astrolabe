from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.nonstandard_residues import build_nonstandard_aliases_and_user_ccd
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes, SegmentManager
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.posthoc import refold_backends
from astrolabe.posthoc import refold_plan
from astrolabe.posthoc.db import iter_records, open_db
from astrolabe.posthoc.filtering import (
    apply_filter_expression,
    apply_preset,
    build_filter_context,
    compile_filter_expression,
    iter_selected_rows,
    load_metrics_json,
    load_preset,
)
from astrolabe.posthoc.io_utils import sanitize_token
from astrolabe.utils.base import construct_standard_input


_SEQ_KEY_RE = re.compile(r"^p([A-Z])_sequence$")


@dataclass(frozen=True)
class PathMap:
    src: str
    dst: str


def _parse_path_maps(raw: Sequence[str]) -> list[PathMap]:
    out: list[PathMap] = []
    for s in raw:
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"--path-map expects FROM=TO, got: {s}")
        a, b = s.split("=", 1)
        a = a.strip()
        b = b.strip()
        if not a or not b:
            raise ValueError(f"--path-map expects FROM=TO, got: {s}")
        out.append(PathMap(src=a, dst=b))
    return out


def _apply_path_maps(s: str, maps: Sequence[PathMap]) -> str:
    if not maps:
        return s
    for m in maps:
        if s.startswith(m.src):
            return m.dst + s[len(m.src) :]
    return s


def _split_path_and_chain_suffix(s: str) -> tuple[str, str]:
    """
    Split "path:CHAIN" style homology strings.

    Returns (path_part, suffix_part_including_colon_or_empty).
    """
    if ":" not in s:
        return s, ""
    a, b = s.rsplit(":", 1)
    # In this codebase, homology strings typically suffix a 1-char chain ID.
    if len(b) == 1 and b.isalnum():
        return a, ":" + b
    return s, ""


def _resolve_abs_or_relbase(
    *,
    raw_path: str,
    relpath_base: Path | None,
    path_maps: Sequence[PathMap],
    require_abs: bool,
) -> Path:
    raw_path = os.path.expanduser(os.path.expandvars(raw_path))
    if "${" in raw_path:
        raise ValueError(f"Unresolved interpolation in path: {raw_path}")
    raw_path = _apply_path_maps(raw_path, path_maps)
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if relpath_base is None:
        msg = f"Relative path not allowed: {raw_path}"
        if require_abs:
            msg += " (provide --relpath-base or rewrite config to absolute paths)"
        raise ValueError(msg)
    return (Path(relpath_base) / p).resolve()


def _resolve_homology_value(
    *,
    v: Any,
    relpath_base: Path | None,
    path_maps: Sequence[PathMap],
    require_abs: bool,
) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        path_part, suffix = _split_path_and_chain_suffix(v)
        p = _resolve_abs_or_relbase(
            raw_path=path_part,
            relpath_base=relpath_base,
            path_maps=path_maps,
            require_abs=require_abs,
        )
        if not p.exists():
            raise FileNotFoundError(f"Homology/MSA/template path missing: {p}")
        return str(p) + suffix
    if isinstance(v, list):
        # AF3-style templates list (inline mmcif), or list of dicts.
        return [_resolve_homology_value(v=x, relpath_base=relpath_base, path_maps=path_maps, require_abs=require_abs) for x in v]
    if isinstance(v, dict):
        # AF3 template dicts contain inline mmcif strings; do not treat them as paths.
        if "mmcif" in v and ("queryIndices" in v or "templateIndices" in v):
            return v
        out: dict[str, Any] = {}
        for k, vv in v.items():
            out[str(k)] = _resolve_homology_value(
                v=vv,
                relpath_base=relpath_base,
                path_maps=path_maps,
                require_abs=require_abs,
            )
        return out
    # Keep unknown structures as-is (rare).
    return v


def _extract_sequences(metrics: Mapping[str, Any]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for k, v in metrics.items():
        if not isinstance(k, str):
            continue
        m = _SEQ_KEY_RE.match(k)
        if not m:
            continue
        chain = m.group(1)
        if not isinstance(v, str) or not v.strip():
            continue
        seqs[chain] = v.strip()
    return seqs


def _is_valid_float(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    try:
        x = float(v)
    except Exception:
        return False
    return x == x and x not in {float("inf"), float("-inf")}


def _hash_manifest_inputs(obj: Any) -> str:
    return refold_plan.hash_manifest_inputs(obj)


def _canonical_refold_backend_name(raw: str | None) -> str | None:
    return refold_plan.canonical_refold_backend_name(raw)


def _extract_evaluator_backend_context(cfg: Any) -> tuple[dict[str, str], dict[str, str]]:
    alias_backend_map: dict[str, str] = {}
    stage_backend_map: dict[str, str] = {}

    evaluator_node = getattr(cfg, "evaluator", None)
    try:
        raw_terms = OmegaConf.to_container(
            getattr(evaluator_node, "evaluator_terms", {}), resolve=False
        )
    except Exception:
        raw_terms = {}
    if isinstance(raw_terms, dict):
        for alias, conf in raw_terms.items():
            if not isinstance(conf, dict):
                continue
            backend = _canonical_refold_backend_name(conf.get("_type"))
            if backend is not None:
                alias_backend_map[str(alias)] = backend

    try:
        raw_stages = OmegaConf.to_container(getattr(cfg, "stages", {}), resolve=False)
    except Exception:
        raw_stages = {}
    if isinstance(raw_stages, dict):
        default_alias = None
        default_stage = raw_stages.get("default")
        if isinstance(default_stage, dict):
            default_alias = str(default_stage.get("evaluator") or "").strip()
            backend = alias_backend_map.get(default_alias)
            if backend is not None:
                stage_backend_map["default"] = backend
        for stage_name, stage_conf in raw_stages.items():
            if not isinstance(stage_conf, dict):
                continue
            alias = str(stage_conf.get("evaluator") or default_alias or "").strip()
            backend = alias_backend_map.get(alias)
            if backend is not None:
                stage_backend_map[str(stage_name)] = backend

    return alias_backend_map, stage_backend_map


def _infer_needed_evaluators(
    *,
    required_keys: Sequence[str],
    metrics: Mapping[str, Any],
    alias_backend_map: Mapping[str, str],
    stage_backend_map: Mapping[str, str],
) -> list[str]:
    stage_name = str(metrics.get("stage") or "").strip()
    stage_backend = stage_backend_map.get(stage_name) or stage_backend_map.get("default")
    ordered: list[str] = []
    seen: set[str] = set()

    def note(name: str | None) -> None:
        backend = _canonical_refold_backend_name(name)
        if backend is None or backend in seen:
            return
        seen.add(backend)
        ordered.append(backend)

    for raw_key in required_keys:
        key = str(raw_key or "").strip()
        if not key:
            continue
        prefix, dot, _rest = key.partition(".")
        if dot and prefix in alias_backend_map:
            note(alias_backend_map[prefix])
            continue
        lowered = key.lower()
        if lowered.startswith("protenix_"):
            note("protenix")
            continue
        if lowered.startswith(("af3_", "alphafold3_", "halludesign_af3_")):
            note("af3")
            continue
        if lowered in {
            "iptm",
            "ptm",
            "plddt",
            "binder_iptm",
            "binder_ptm",
            "target_chain_ptm",
            "target_chain_iptm",
            "target_chain_plddt_mean",
            "ipae_mean",
            "ipae_min",
            "ipsae_ab",
            "chain_pair_lis_ab",
            "chain_pair_contact_lis_ab",
        }:
            note(stage_backend)

    if not ordered and stage_backend is not None:
        note(stage_backend)
    return ordered


def export_refold_manifest(
    *,
    db_path: Path,
    preset_path: Path | None = None,
    filter_expr: str | None = None,
    out_manifest: Path,
    prefer_oligo: str = "AB",
    run_selection: str = "all-passing",
    passed_only: bool = True,
    report_csv: Path | None = None,
    relpath_base: Path | None = None,
    path_map: Sequence[str] = (),
    require_abs_paths: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Export a JSONL manifest for refold second-pass prediction.

    Strict mode:
      - sequences come from scores.csv metrics (stored in DB)
      - homologies/templates/MSA come from run_dir/config.yaml
      - all referenced paths must exist; relative paths are rejected unless --relpath-base is provided
    """

    db_path = Path(db_path).expanduser().resolve()
    if bool(preset_path) == bool(filter_expr):
        raise ValueError("Exactly one of preset_path or filter_expr must be provided")

    preset_name = ""
    compiled_expr = None
    selector_kind = "preset"
    selector = ""
    if preset_path is not None:
        preset_path = Path(preset_path).expanduser().resolve()
        preset = load_preset(preset_path)
        preset_name = preset.name
        selector = preset.name
        required_keys = tuple(
            dict.fromkeys(rule.key for rule in preset.rules if not rule.allow_missing)
        )
    else:
        preset = None
        selector_kind = "dsl"
        compiled_expr = compile_filter_expression(str(filter_expr))
        selector = compiled_expr.source
        required_keys = tuple(compiled_expr.required_keys)
    out_manifest = Path(out_manifest).expanduser().resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    if report_csv is not None:
        report_csv = Path(report_csv).expanduser().resolve()
        report_csv.parent.mkdir(parents=True, exist_ok=True)

    maps = _parse_path_maps(list(path_map))
    if relpath_base is not None:
        relpath_base = Path(relpath_base).expanduser().resolve()

    conn = open_db(db_path)
    written = 0
    total = 0
    passed = 0
    failed = 0
    candidates = 0

    cfg_cache: dict[Path, tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]] = {}

    tmp = out_manifest.with_suffix(out_manifest.suffix + f".{os.getpid()}.tmp")
    rep_tmp = None
    rep_writer = None
    rep_fh = None
    if report_csv is not None:
        rep_tmp = report_csv.with_suffix(report_csv.suffix + f".{os.getpid()}.tmp")
        rep_fh = rep_tmp.open("w", encoding="utf-8", newline="")
        rep_writer = csv.DictWriter(
            rep_fh,
            fieldnames=[
                "task",
                "run",
                "step",
                "oligo",
                "status",
                "reason",
                "src_run_dir",
                "src_config",
            ],
        )
        rep_writer.writeheader()

    def _report(row: Mapping[str, Any], status: str, reason: str) -> None:
        nonlocal failed
        failed += 1
        if rep_writer is None:
            return
        def _g(k: str) -> Any:
            try:
                if hasattr(row, "get"):
                    return row.get(k, "")
            except Exception:
                pass
            try:
                return row[k]
            except Exception:
                return ""
        rep_writer.writerow(
            {
                "task": _g("task"),
                "run": _g("run"),
                "step": _g("step"),
                "oligo": _g("oligo"),
                "status": status,
                "reason": reason,
                "src_run_dir": _g("run_dir"),
                "src_config": str(Path(str(_g("run_dir"))) / "config.yaml"),
            }
        )

    with tmp.open("w", encoding="utf-8") as out_f:
        for row in iter_selected_rows(
            conn,
            prefer_oligo=str(prefer_oligo),
            run_selection=run_selection,
        ):
            total += 1

            try:
                metrics = load_metrics_json(row)
            except Exception:
                metrics = {}
            context = build_filter_context(row, metrics)
            if preset is not None:
                res = apply_preset(context, preset)
            else:
                assert compiled_expr is not None
                res = apply_filter_expression(context, compiled_expr)
            if res.passed:
                passed += 1
            missing_required = [key for key in required_keys if key not in context]
            invalid_required = [
                key
                for key in required_keys
                if key in context and preset is not None and not _is_valid_float(context.get(key))
            ]
            blocked_only_by_required = False
            if missing_required or invalid_required:
                if preset is not None:
                    blocking = set(missing_required) | set(invalid_required)
                    blocked_only_by_required = set(res.failed).issubset(blocking)
                else:
                    blocked_only_by_required = True
            include_passing_row = bool(res.passed and not passed_only)
            if not include_passing_row and not blocked_only_by_required:
                continue
            candidates += 1
            if limit is not None and candidates > int(limit):
                break

            seqs = _extract_sequences(metrics)
            missing_chains = [c for c in str(row["oligo"]) if c.isalpha() and c not in seqs]
            if missing_chains:
                _report(row, "error", f"missing sequences in scores.csv: {missing_chains}")
                continue

            source_run_dir = str(metrics.get("src_run_dir") or row["run_dir"])
            run_dir = Path(source_run_dir).expanduser().resolve()
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                _report(row, "error", f"missing config.yaml: {cfg_path}")
                continue

            cached = cfg_cache.get(cfg_path)
            if cached is None:
                try:
                    cfg = OmegaConf.load(str(cfg_path))
                except Exception as e:
                    _report(row, "error", f"failed to load config.yaml: {e}")
                    continue
                input_signature = _load_refold_input_signature(cfg)

                # Resolve homologies subtree only (safe even when unrelated env vars are missing).
                try:
                    raw_homologies = OmegaConf.to_container(
                        getattr(cfg, "homologies", {}), resolve=True
                    )
                except Exception as e:
                    _report(row, "error", f"failed to resolve homologies in config.yaml: {e}")
                    continue
                if raw_homologies is None:
                    raw_homologies = {}
                if not isinstance(raw_homologies, dict):
                    _report(row, "error", "config.yaml homologies is not a mapping")
                    continue

                # Types are helpful for non-protein protomers; keep best-effort.
                try:
                    raw_types = OmegaConf.to_container(getattr(cfg, "types", {}), resolve=True)
                except Exception:
                    raw_types = {}
                if raw_types is None:
                    raw_types = {}
                if not isinstance(raw_types, dict):
                    raw_types = {}

                alias_backend_map, stage_backend_map = _extract_evaluator_backend_context(cfg)
                cfg_cache[cfg_path] = (
                    dict(raw_homologies),
                    dict(raw_types),
                    dict(alias_backend_map),
                    dict(stage_backend_map),
                )
            else:
                raw_homologies, raw_types, alias_backend_map, stage_backend_map = cached

            if cached is None:
                raw_homologies, raw_types, alias_backend_map, stage_backend_map = cfg_cache[cfg_path]

            needed = sorted(seqs.keys())
            homologies: dict[str, Any] = {}
            try:
                for chain in needed:
                    if chain in raw_homologies:
                        homologies[chain] = _resolve_homology_value(
                            v=raw_homologies.get(chain),
                            relpath_base=relpath_base,
                            path_maps=maps,
                            require_abs=require_abs_paths,
                        )
                    else:
                        homologies[chain] = None
            except Exception as e:
                _report(row, "error", str(e))
                continue

            if include_passing_row:
                needed_evaluators = []
            else:
                needed_evaluators = _infer_needed_evaluators(
                    required_keys=[*missing_required, *invalid_required],
                    metrics=metrics,
                    alias_backend_map=alias_backend_map,
                    stage_backend_map=stage_backend_map,
                )

            item = {
                "id": f"{sanitize_token(row['task'])}__{sanitize_token(row['run'])}__step_{int(row['step']):03d}__{sanitize_token(row['oligo'])}",
                "task": row["task"],
                "run": row["run"],
                "step": int(row["step"]),
                "oligo": row["oligo"],
                "src_run_dir": str(run_dir),
                "src_config": str(cfg_path),
                "seqs": {k: seqs[k] for k in needed},
                "types": {k: raw_types.get(k, "protein") for k in needed},
                "homologies": homologies,
                "needed_evaluators": needed_evaluators,
                "cyclization": input_signature.get("cyclization"),
                "nonstandard_residues": input_signature.get("nonstandard_residues"),
                "stage1": {
                    "selector_kind": selector_kind,
                    "selector": selector,
                    "passed": bool(res.passed),
                    "failed": list(res.failed),
                    "missing": list(res.missing),
                },
                "created_at": time.time(),
            }
            item["inputs_hash"] = _hash_manifest_inputs(
                {
                    "task": item["task"],
                    "run": item["run"],
                    "step": item["step"],
                    "oligo": item["oligo"],
                    "seqs": item["seqs"],
                    "types": item["types"],
                    "homologies": item["homologies"],
                    "cyclization": item["cyclization"],
                    "nonstandard_residues": item["nonstandard_residues"],
                }
            )

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    os.replace(str(tmp), str(out_manifest))
    if rep_fh is not None and rep_tmp is not None and report_csv is not None:
        rep_fh.close()
        os.replace(str(rep_tmp), str(report_csv))

    return {
        "db_path": str(db_path),
        "selector_kind": selector_kind,
        "selector": selector,
        "preset": preset_name,
        "filter_expr": compiled_expr.source if compiled_expr is not None else "",
        "prefer_oligo": str(prefer_oligo),
        "run_selection": str(run_selection),
        "passed_only": bool(passed_only),
        "out_manifest": str(out_manifest),
        "report_csv": str(report_csv) if report_csv else None,
        "records_considered": total,
        "records_passed": passed,
        "candidates_considered": candidates,
        "written": written,
        "failed": failed,
        "require_abs_paths": bool(require_abs_paths),
        "relpath_base": str(relpath_base) if relpath_base else None,
        "path_map": [f"{m.src}={m.dst}" for m in maps],
        "limit": int(limit) if limit is not None else None,
    }


def refold_from_db(
    *,
    db_path: Path,
    dst_dir: Path,
    preset_path: Path | None = None,
    filter_expr: str | None = None,
    prefer_oligo: str = "AB",
    run_selection: str = "all-passing",
    evaluators: Sequence[str] = ("af3", "protenix"),
    collect_workers: int = 0,
    relpath_base: Path | None = None,
    path_map: Sequence[str] = (),
    keep_protenix_structures: bool = False,
    config_yaml: Path | None = None,
    workers: int = 1,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    evaluator_backends = refold_backends.refold_backend_names(evaluators)
    db_path = Path(db_path).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    manifest_jsonl = dst_dir / "refold_manifest.jsonl"
    export_report = dst_dir / "refold_export_report.csv"
    refold_out_root = dst_dir / "refold_runs"

    export_summary = export_refold_manifest(
        db_path=db_path,
        preset_path=Path(preset_path) if preset_path is not None else None,
        filter_expr=str(filter_expr) if filter_expr is not None else None,
        out_manifest=manifest_jsonl,
        prefer_oligo=str(prefer_oligo),
        run_selection=str(run_selection),
        passed_only=True,
        report_csv=export_report,
        relpath_base=Path(relpath_base) if relpath_base is not None else None,
        path_map=path_map,
        require_abs_paths=True,
        limit=limit,
    )

    run_summary = run_refold(
        manifest_jsonl=manifest_jsonl,
        out_root=refold_out_root,
        evaluators=evaluator_backends,
        relpath_base=Path(relpath_base) if relpath_base is not None else None,
        path_map=path_map,
        keep_protenix_structures=keep_protenix_structures,
        config_yaml=Path(config_yaml) if config_yaml is not None else None,
        workers=int(workers),
        dry_run=dry_run,
    )

    collect_summary: dict[str, Any] | None = None
    if not dry_run:
        from astrolabe.posthoc.collect import collect_scores

        collect_summary = collect_scores(
            out_root=refold_out_root,
            db_path=db_path,
            workers=int(collect_workers),
            fail_on_missing_required=False,
        )

    return {
        "db_path": str(db_path),
        "selector_kind": str(export_summary.get("selector_kind", "preset")),
        "selector": str(export_summary.get("selector", "")),
        "preset": str(export_summary.get("preset", "")),
        "filter_expr": str(export_summary.get("filter_expr", "")),
        "prefer_oligo": str(prefer_oligo),
        "run_selection": str(run_selection),
        "dst_dir": str(dst_dir),
        "manifest_jsonl": str(manifest_jsonl),
        "export_report_csv": str(export_report),
        "refold_out_root": str(refold_out_root),
        "evaluators": evaluator_backends,
        "collect_db_path": str(db_path),
        "collect_workers": int(collect_workers),
        "exported": int(export_summary.get("written", 0)),
        "processed": int(run_summary.get("processed", 0)),
        "skipped": int(run_summary.get("skipped", 0)),
        "failed": int(run_summary.get("failed", 0)),
        "ran_af3": int(run_summary.get("ran_af3", 0)),
        "ran_protenix": int(run_summary.get("ran_protenix", 0)),
        "reused_af3": int(run_summary.get("reused_af3", 0)),
        "reused_protenix": int(run_summary.get("reused_protenix", 0)),
        "groups": int(run_summary.get("groups", 0)),
        "resource_groups": int(run_summary.get("resource_groups", 0)),
        "evaluator_cache_hits": int(run_summary.get("evaluator_cache_hits", 0)),
        "evaluator_cache_misses": int(run_summary.get("evaluator_cache_misses", 0)),
        "collected_runs": int(collect_summary.get("runs_processed", 0)) if collect_summary else 0,
        "collect_summary": collect_summary,
        "config_yaml": str(config_yaml) if config_yaml is not None else None,
        "workers_requested": int(run_summary.get("workers_requested", workers)),
        "workers": int(run_summary.get("workers", 1)),
        "dry_run": bool(dry_run),
        "failure_report": run_summary.get("failure_report"),
    }


def _load_optional_cfg_node(cfg: Any, key: str) -> Any:
    node = getattr(cfg, key, None)
    if node is None:
        return None
    raw = OmegaConf.to_container(node, resolve=True)
    return raw


def _manifest_or_cfg_node(
    *,
    item: Mapping[str, Any] | None,
    cfg: Any,
    key: str,
) -> Any:
    if item is not None and key in item:
        value = item.get(key)
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return _load_optional_cfg_node(cfg, key)


def _load_refold_input_context(
    cfg: Any,
    *,
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    nonstandard_cfg = _manifest_or_cfg_node(
        item=item,
        cfg=cfg,
        key="nonstandard_residues",
    )
    nonstandard_aliases, user_ccd = build_nonstandard_aliases_and_user_ccd(
        nonstandard_cfg
    )
    return {
        "cyclization": _manifest_or_cfg_node(item=item, cfg=cfg, key="cyclization"),
        "nonstandard_aliases": nonstandard_aliases,
        "user_ccd": user_ccd,
    }


def _load_refold_input_signature(
    cfg: Any,
    *,
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cyclization": _manifest_or_cfg_node(item=item, cfg=cfg, key="cyclization"),
        "nonstandard_residues": _manifest_or_cfg_node(
            item=item,
            cfg=cfg,
            key="nonstandard_residues",
        ),
    }


def _load_refold_runtime_overrides(cfg: Any) -> dict[str, Any]:
    return {
        "binding_site": _load_optional_cfg_node(cfg, "binding_site"),
        "protenix_constraint": _load_optional_cfg_node(cfg, "protenix_constraint"),
    }


def _init_refold_global_config(
    *,
    out_dir: Path,
    binding_site: Any = None,
    protenix_constraint: Any = None,
    cyclization: Any = None,
    nonstandard_aliases: Mapping[str, Mapping[str, Any]] | None = None,
    user_ccd: str | None = None,
) -> None:
    global_config.reset()
    global_config.init_const("exclude_AA", [])
    global_config.init_const("covalent_bonds", None)
    global_config.init_const("protomers", {})
    global_config.init_const("segments", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("initial_temperature", 0.0)
    global_config.init_const("out_dir", Path(out_dir))
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const("enable_pyrosetta", False)
    global_config.init_const("skip_complex_poses", True)
    global_config.init_const("protenix_constraint", protenix_constraint)
    global_config.init_const("binding_site", binding_site)
    global_config.init_const("cyclization", cyclization)
    global_config.init_const("nonstandard_aliases", dict(nonstandard_aliases or {}))
    global_config.init_const("user_ccd", user_ccd)
    global_config.init_const("num_tokens", {})


def _build_protomers(
    *,
    seqs: Mapping[str, str],
    types: Mapping[str, str],
    homologies: Mapping[str, Any],
) -> None:
    for proto, seq in seqs.items():
        protomer = Protomer(
            proto,
            seq,
            types.get(proto, "protein"),
            homologies.get(proto),
        )
        global_config.protomers[proto] = protomer
        if protomer.type in (ProtomerTypes.PROTEIN, ProtomerTypes.RNA, ProtomerTypes.DNA):
            global_config.mutable_protomers[proto] = protomer


def _load_eval_subcfg(cfg: Any, key: str) -> Dict[str, Any]:
    node = getattr(getattr(cfg, "evaluator", None), key, None)
    if node is None:
        raise ValueError(f"Missing evaluator.{key} in src config")
    raw = OmegaConf.to_container(node, resolve=True)
    if raw is None or not isinstance(raw, dict):
        raise ValueError(f"Invalid evaluator.{key} config")
    return dict(raw)


def _load_cfg_for_refold(src_config: Path) -> Any:
    try:
        return OmegaConf.load(str(src_config))
    except Exception as e:
        raise RuntimeError(f"Failed to load src config: {src_config} ({e})") from e


def _ensure_scores_csv(
    *,
    run_dir: Path,
    rows_by_step: Mapping[int, Dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "scores.csv"

    # Discover chains from per-step rows (pA_sequence, pB_sequence, ...).
    chains: list[str] = []
    for row in rows_by_step.values():
        for k in row.keys():
            if isinstance(k, str) and k.startswith("p") and k.endswith("_sequence") and len(k) == len("pA_sequence"):
                c = k[1]
                if c.isalpha() and c not in chains:
                    chains.append(c)
    chains.sort()

    stable_first = [
        "step",
        "stage",
        "iteration",
        "accepted",
        "weighted_score",
        *[f"{c}_length" for c in chains],
        "iptm",
        "ptm",
        "plddt",
        "AB.iptm",
        "AB.ptm",
        "AB.plddt",
        "src_run_dir",
        *[f"p{c}_sequence" for c in chains],
    ]
    # Append any additional columns present in rows_by_step (stable order).
    extra: list[str] = []
    for row in rows_by_step.values():
        for k in row.keys():
            if not isinstance(k, str):
                continue
            if k in stable_first:
                continue
            if k not in extra:
                extra.append(k)
    header = stable_first + extra

    tmp = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for step in sorted(rows_by_step.keys()):
            row = dict(rows_by_step[step])
            row.setdefault("step", int(step))
            row.setdefault("stage", "refold")
            row.setdefault("iteration", 0)
            row.setdefault("accepted", True)
            row.setdefault("weighted_score", "")
            w.writerow(row)
    os.replace(str(tmp), str(out))


def _af3_done_marker(run_dir: Path, *, step: int, oligo: str) -> Path:
    return (
        Path(run_dir)
        / "raw_model_results"
        / "alphafold3"
        / f"step_{step:03d}_{oligo}"
        / f"step_{step:03d}_{oligo}_summary_confidences.json"
    )


def _protenix_done_marker(run_dir: Path, *, step: int, oligo: str) -> Path:
    # Best-effort: any summary_confidence indicates dumper completed.
    base = Path(run_dir) / "raw_model_results" / "protenix" / f"step_{step:03d}" / str(oligo)
    return base


def _requested_refold_backends(evaluators: Sequence[str]) -> set[str]:
    return refold_plan.requested_refold_backends(evaluators)


def _item_needed_backends(item: Mapping[str, Any], requested: set[str]) -> set[str]:
    return refold_plan.item_needed_backends(item, requested)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def _mirror_tree(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            _link_or_copy_file(root_path / name, dst_root / name)
        for name in dirs:
            (dst_root / name).mkdir(parents=True, exist_ok=True)


def _backend_config_key(backend: str) -> str:
    backend_name = _canonical_refold_backend_name(backend)
    if backend_name == "af3":
        return "alphafold3"
    if backend_name == "protenix":
        return "protenix"
    raise ValueError(f"Unsupported refold backend for config lookup: {backend}")


def _load_backend_cfg_if_available(config_path: Path, backend: str) -> dict[str, Any] | None:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        return None
    try:
        cfg = _load_cfg_for_refold(cfg_path)
        return _load_eval_subcfg(cfg, _backend_config_key(backend))
    except Exception:
        return None


def _load_refold_runtime_signature_if_available(
    config_path: Path,
    backend: str,
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        return None
    try:
        cfg = _load_cfg_for_refold(cfg_path)
        signature: dict[str, Any] = {
            "backend_cfg": _load_eval_subcfg(cfg, _backend_config_key(backend)),
        }
        if _canonical_refold_backend_name(backend) == "protenix":
            signature.update(_load_refold_runtime_overrides(cfg))
            weight_path_cfg = _load_optional_cfg_node(cfg, "weight_path")
            if weight_path_cfg is not None:
                signature["weight_path"] = weight_path_cfg
        input_signature = _load_refold_input_signature(cfg, item=item)
        signature["cyclization"] = input_signature.get("cyclization")
        signature["nonstandard_residues"] = input_signature.get("nonstandard_residues")
        return signature
    except Exception:
        return None


def _can_reuse_source_backend_outputs(
    *,
    backend: str,
    source_run_dir: Path,
    target_config_path: Path | None,
    item: Mapping[str, Any] | None = None,
) -> bool:
    source_cfg_path = Path(source_run_dir) / "config.yaml"
    if target_config_path is None:
        target_config_path = source_cfg_path
    target_cfg_path = Path(target_config_path).expanduser().resolve()
    source_cfg_path = source_cfg_path.expanduser().resolve()
    if (
        source_cfg_path == target_cfg_path
        and not (item and ("cyclization" in item or "nonstandard_residues" in item))
    ):
        return True
    source_signature = _load_refold_runtime_signature_if_available(source_cfg_path, backend)
    target_signature = _load_refold_runtime_signature_if_available(
        target_cfg_path,
        backend,
        item=item,
    )
    if source_signature is None or target_signature is None:
        return False
    return _hash_manifest_inputs(source_signature) == _hash_manifest_inputs(target_signature)


def _reuse_source_backend_outputs(
    *,
    backend: str,
    source_run_dir: Path,
    dst_run_dir: Path,
    step: int,
    oligo: str,
) -> bool:
    backend_name = _canonical_refold_backend_name(backend)
    if backend_name == "af3":
        src_dir = _af3_done_marker(source_run_dir, step=step, oligo=oligo).parent
        src_marker = src_dir / f"step_{step:03d}_{oligo}_summary_confidences.json"
        if not src_marker.exists():
            return False
        dst_dir = _af3_done_marker(dst_run_dir, step=step, oligo=oligo).parent
        _mirror_tree(src_dir, dst_dir)
        src_structure = source_run_dir / "structures" / f"step_{step:03d}_{oligo}_model.cif"
        if src_structure.exists():
            _link_or_copy_file(
                src_structure,
                dst_run_dir / "structures" / f"step_{step:03d}_{oligo}_model.cif",
            )
        return True
    if backend_name == "protenix":
        src_dir = _protenix_done_marker(source_run_dir, step=step, oligo=oligo)
        if not src_dir.exists() or not any(src_dir.glob("seed_*/predictions/*_summary_confidence_sample_*.json")):
            return False
        dst_dir = _protenix_done_marker(dst_run_dir, step=step, oligo=oligo)
        _mirror_tree(src_dir, dst_dir)
        return True
    return False


def _append_refold_note(row: Dict[str, Any], message: str) -> None:
    msg = str(message or "").strip()
    if not msg:
        return
    existing = [part.strip() for part in str(row.get("note") or "").split("; ") if part.strip()]
    if msg not in existing:
        existing.append(msg)
    row["note"] = "; ".join(existing)


def _append_refold_backend(row: Dict[str, Any], backend: str, *, action: str) -> None:
    backend_name = _canonical_refold_backend_name(backend)
    if backend_name is None:
        return
    current = [part.strip() for part in str(row.get("refold_backends") or "").split(",") if part.strip()]
    if backend_name not in current:
        current.append(backend_name)
    row["refold_backends"] = ",".join(current)
    _append_refold_note(row, f"{action}:{backend_name}")


def _merge_backend_metrics_into_rows(
    *,
    rows_by_step: Dict[int, Dict[str, Any]],
    run_dir: Path,
    binder_chain: str,
    target_chain: str,
) -> None:
    from astrolabe.posthoc.derive import derive_af3, derive_protenix

    for (_step, _oligo), metrics in derive_af3(
        run_dir,
        binder_chain=binder_chain,
        target_chain=target_chain,
    ).items():
        row = rows_by_step.setdefault(int(_step), {})
        for key, value in metrics.items():
            row.setdefault(key, value)
    for (_step, _oligo), metrics in derive_protenix(
        run_dir,
        binder_chain=binder_chain,
        target_chain=target_chain,
    ).items():
        row = rows_by_step.setdefault(int(_step), {})
        for key, value in metrics.items():
            row.setdefault(key, value)


def _repo_root_for_refold() -> Path:
    return Path(__file__).resolve().parents[2]


def _merge_worker_outputs(*, worker_out_root: Path, out_root: Path) -> None:
    for child in Path(worker_out_root).iterdir():
        if child.name == "refold_failures.jsonl":
            continue
        if not child.is_dir():
            continue
        _mirror_tree(child, out_root / child.name)


def _append_worker_failures(*, worker_out_root: Path, failure_lines: list[str]) -> None:
    path = Path(worker_out_root) / "refold_failures.jsonl"
    if not path.exists():
        return
    failure_lines.extend([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _refold_backend_resource_key(
    *,
    item: Mapping[str, Any],
    backend: str,
    config_yaml: Path | None,
) -> tuple[str, str]:
    return refold_plan.refold_backend_resource_key(
        item=item,
        backend=backend,
        config_yaml=config_yaml,
        load_runtime_signature=_load_refold_runtime_signature_if_available,
    )


def _retarget_refold_evaluators(
    *,
    evals: Mapping[str, Any],
    backend: str,
    dst_run_dir: Path,
) -> None:
    backend_name = _canonical_refold_backend_name(backend)
    raw_root = Path(dst_run_dir) / "raw_model_results"
    if backend_name == "af3":
        evaluator = evals.get("Alphafold3")
        if evaluator is not None and hasattr(evaluator, "output_dir"):
            evaluator.output_dir = raw_root / "alphafold3"
        return

    if backend_name == "protenix":
        evaluator = evals.get("Protenix")
        runner = getattr(evaluator, "model_runner", None)
        if runner is None:
            return
        dump_dir = raw_root / "protenix"
        try:
            runner.configs.dump_dir = dump_dir
        except Exception:
            pass
        runner.dump_dir = dump_dir
        runner.error_dir = os.path.join(str(dump_dir), "ERR")
        os.makedirs(str(dump_dir), exist_ok=True)
        os.makedirs(str(runner.error_dir), exist_ok=True)
        dumper = getattr(runner, "dumper", None)
        if dumper is not None and hasattr(dumper, "base_dir"):
            dumper.base_dir = dump_dir


def _run_refold_parallel(
    *,
    plan: refold_plan.RefoldPlan,
    manifest_jsonl: Path,
    out_root: Path,
    evaluators: Sequence[str],
    binder_chain: str,
    target_chain: str,
    relpath_base: Path | None,
    path_map: Sequence[str],
    keep_protenix_structures: bool,
    config_yaml: Path | None,
    workers: int,
) -> dict[str, Any]:
    worker_count = max(1, min(int(workers), len(plan.groups)))
    parallel_root = out_root / "_parallel_workers"
    if parallel_root.exists():
        shutil.rmtree(parallel_root)
    parallel_root.mkdir(parents=True, exist_ok=True)

    shards, resource_group_count = refold_plan.assign_refold_plan_to_worker_shards(
        plan=plan,
        worker_count=worker_count,
    )

    repo_root = _repo_root_for_refold()
    worker_summaries: list[dict[str, Any]] = []
    failure_lines: list[str] = []

    worker_code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "repo_root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(repo_root / 'vendor'))\n"
        "sys.path.insert(0, str(repo_root))\n"
        "from astrolabe.posthoc.refold import run_refold\n"
        "summary = run_refold(\n"
        "    manifest_jsonl=Path(sys.argv[2]),\n"
        "    out_root=Path(sys.argv[3]),\n"
        "    evaluators=json.loads(sys.argv[4]),\n"
        "    binder_chain=sys.argv[5],\n"
        "    target_chain=sys.argv[6],\n"
        "    relpath_base=(Path(sys.argv[7]) if sys.argv[7] else None),\n"
        "    path_map=json.loads(sys.argv[8]),\n"
        "    keep_protenix_structures=(sys.argv[9] == '1'),\n"
        "    config_yaml=(Path(sys.argv[10]) if sys.argv[10] else None),\n"
        "    dry_run=False,\n"
        "    workers=1,\n"
        ")\n"
        "Path(sys.argv[11]).write_text(json.dumps(summary, sort_keys=True), encoding='utf-8')\n"
    )

    procs: list[tuple[subprocess.Popen[str], Path, Path]] = []
    for idx, shard_items in enumerate(shards, start=1):
        shard_manifest = parallel_root / f"worker_{idx}.manifest.jsonl"
        shard_out_root = parallel_root / f"worker_{idx}.out"
        shard_summary = parallel_root / f"worker_{idx}.summary.json"
        with shard_manifest.open("w", encoding="utf-8") as f:
            for item in shard_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        cmd = [
            sys.executable,
            "-c",
            worker_code,
            str(repo_root),
            str(shard_manifest),
            str(shard_out_root),
            json.dumps(list(evaluators)),
            str(binder_chain),
            str(target_chain),
            str(relpath_base) if relpath_base else "",
            json.dumps(list(path_map)),
            "1" if keep_protenix_structures else "0",
            str(config_yaml) if config_yaml else "",
            str(shard_summary),
        ]
        procs.append((subprocess.Popen(cmd, text=True), shard_out_root, shard_summary))

    for proc, shard_out_root, shard_summary in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Refold worker exited with status {rc}: {shard_out_root}")
        worker_summaries.append(json.loads(shard_summary.read_text(encoding="utf-8")))
        _merge_worker_outputs(worker_out_root=shard_out_root, out_root=out_root)
        _append_worker_failures(worker_out_root=shard_out_root, failure_lines=failure_lines)

    if failure_lines:
        failure_jsonl = out_root / "refold_failures.jsonl"
        failure_jsonl.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
        failure_report = str(failure_jsonl)
    else:
        failure_report = None

    return {
        "manifest_jsonl": str(manifest_jsonl),
        "out_root": str(out_root),
        "evaluators": list(evaluators),
        "processed": sum(int(item.get("processed", 0)) for item in worker_summaries),
        "skipped": sum(int(item.get("skipped", 0)) for item in worker_summaries),
        "failed": sum(int(item.get("failed", 0)) for item in worker_summaries),
        "ran_af3": sum(int(item.get("ran_af3", 0)) for item in worker_summaries),
        "ran_protenix": sum(int(item.get("ran_protenix", 0)) for item in worker_summaries),
        "reused_af3": sum(int(item.get("reused_af3", 0)) for item in worker_summaries),
        "reused_protenix": sum(int(item.get("reused_protenix", 0)) for item in worker_summaries),
        "groups": len(plan.groups),
        "resource_groups": resource_group_count,
        "evaluator_cache_hits": sum(int(item.get("evaluator_cache_hits", 0)) for item in worker_summaries),
        "evaluator_cache_misses": sum(int(item.get("evaluator_cache_misses", 0)) for item in worker_summaries),
        "dry_run": False,
        "workers_requested": int(workers),
        "workers": worker_count,
        "binder_chain": str(binder_chain),
        "target_chain": str(target_chain),
        "config_yaml": str(config_yaml) if config_yaml is not None else None,
        "relpath_base": str(relpath_base) if relpath_base else None,
        "path_map": list(path_map),
        "failure_report": failure_report,
    }


def run_refold(
    *,
    manifest_jsonl: Path,
    out_root: Path,
    evaluators: Sequence[str],
    binder_chain: str = "B",
    target_chain: str = "A",
    relpath_base: Path | None = None,
    path_map: Sequence[str] = (),
    keep_protenix_structures: bool = False,
    config_yaml: Path | None = None,
    workers: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute second-pass prediction for each manifest record, writing GH-like run folders:

      <out_root>/<task>/<run>/
        scores.csv
        structures/step_###_<oligo>_model.cif         (canonical: AF3)
        raw_model_results/alphafold3/step_###_<oligo>/...
        raw_model_results/protenix/step_###/<oligo>/seed_*/predictions/...
    """

    evaluator_backends = refold_backends.refold_backend_names(evaluators)
    manifest_jsonl = Path(manifest_jsonl).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    maps = _parse_path_maps(list(path_map))
    if relpath_base is not None:
        relpath_base = Path(relpath_base).expanduser().resolve()

    todo: list[dict[str, Any]] = []
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            todo.append(json.loads(line))

    requested_backends = set(evaluator_backends)
    plan = refold_plan.build_refold_plan(
        items=todo,
        requested_backends=requested_backends,
        config_yaml=config_yaml,
        load_runtime_signature=_load_refold_runtime_signature_if_available,
    )
    resource_group_count = plan.resource_group_count

    if not dry_run and int(workers) > 1 and len(plan.groups) > 1:
        return _run_refold_parallel(
            plan=plan,
            manifest_jsonl=manifest_jsonl,
            out_root=out_root,
            evaluators=evaluator_backends,
            binder_chain=binder_chain,
            target_chain=target_chain,
            relpath_base=relpath_base,
            path_map=path_map,
            keep_protenix_structures=keep_protenix_structures,
            config_yaml=config_yaml,
            workers=int(workers),
        )

    processed = 0
    skipped = 0
    failed = 0
    ran_af3 = 0
    ran_protenix = 0
    reused_af3 = 0
    reused_protenix = 0
    evaluator_cache_hits = 0
    evaluator_cache_misses = 0
    evaluator_cache: dict[tuple[str, str], Any] = {}
    failure_records: list[dict[str, Any]] = []
    failure_jsonl = out_root / "refold_failures.jsonl"

    def _record_failure(it: dict[str, Any], *, reason: str, exc: Exception | None = None) -> None:
        nonlocal failed
        failed += 1
        rec = {
            "task": str(it.get("task", "")),
            "run": str(it.get("run", "")),
            "step": int(it.get("step", -1) or -1),
            "oligo": str(it.get("oligo", "")),
            "reason": reason,
            "src_run_dir": str(it.get("src_run_dir", "")),
            "src_config": str(it.get("src_config", "")),
        }
        if exc is not None:
            rec["error_type"] = type(exc).__name__
            rec["error"] = str(exc)
        failure_records.append(rec)

    for group in plan.groups:
        task, _run, _src_run_dir = group.key
        items = group.manifest_items()
        dst_run_name = group.output_run_name
        dst_run_dir = out_root / sanitize_token(task) / dst_run_name
        dst_run_dir.mkdir(parents=True, exist_ok=True)
        (dst_run_dir / "raw_model_results").mkdir(parents=True, exist_ok=True)
        (dst_run_dir / "structures").mkdir(parents=True, exist_ok=True)

        # Track per-step scores rows (write once per run dir).
        rows_by_step: dict[int, Dict[str, Any]] = {}
        for it in sorted(items, key=lambda x: (int(x["step"]), str(x["oligo"]))):
            step = int(it["step"])
            oligo = str(it["oligo"])
            seqs = it.get("seqs") or {}
            if not isinstance(seqs, dict) or not seqs:
                _record_failure(it, reason="missing_or_invalid_seqs")
                continue

            # Ensure a base scores row exists for this step (even if evaluators are skipped).
            base_row: Dict[str, Any] = {
                "src_run_dir": str(it.get("src_run_dir", "")),
            }
            for c, s in seqs.items():
                if not isinstance(c, str) or not isinstance(s, str):
                    continue
                if len(c) == 1 and c.isalpha() and s:
                    base_row[f"p{c}_sequence"] = s
                    base_row[f"{c}_length"] = len(s)
            rows_by_step.setdefault(step, {}).update(base_row)

            # Resume checks (do not require loading evaluator stacks).
            needed_backends = _item_needed_backends(it, requested_backends)
            want_af3 = "af3" in requested_backends and "af3" in needed_backends
            want_px = "protenix" in requested_backends and "protenix" in needed_backends
            af3_done = _af3_done_marker(dst_run_dir, step=step, oligo=oligo).exists()
            px_done = _protenix_done_marker(dst_run_dir, step=step, oligo=oligo).exists() and any(
                _protenix_done_marker(dst_run_dir, step=step, oligo=oligo).glob("seed_*/predictions/*_summary_confidence_sample_*.json")
            )
            src_run_dir = refold_plan.refold_source_run_dir_path(it)
            target_config_path = refold_plan.refold_target_config_path(
                item=it,
                config_yaml=config_yaml,
            )
            if (
                want_af3
                and not af3_done
                and src_run_dir is not None
                and src_run_dir.exists()
                and _can_reuse_source_backend_outputs(
                    backend="af3",
                    source_run_dir=src_run_dir,
                    target_config_path=target_config_path,
                    item=it,
                )
                and _reuse_source_backend_outputs(
                    backend="af3",
                    source_run_dir=src_run_dir,
                    dst_run_dir=dst_run_dir,
                    step=step,
                    oligo=oligo,
                )
            ):
                af3_done = True
                reused_af3 += 1
                _append_refold_backend(rows_by_step.setdefault(step, {}), "af3", action="reuse-source")
            if (
                want_px
                and not px_done
                and src_run_dir is not None
                and src_run_dir.exists()
                and _can_reuse_source_backend_outputs(
                    backend="protenix",
                    source_run_dir=src_run_dir,
                    target_config_path=target_config_path,
                    item=it,
                )
                and _reuse_source_backend_outputs(
                    backend="protenix",
                    source_run_dir=src_run_dir,
                    dst_run_dir=dst_run_dir,
                    step=step,
                    oligo=oligo,
                )
            ):
                px_done = True
                reused_protenix += 1
                _append_refold_backend(rows_by_step.setdefault(step, {}), "protenix", action="reuse-source")

            if (not want_af3 or af3_done) and (not want_px or px_done):
                skipped += 1
                continue

            if dry_run:
                processed += 1
                continue

            src_config = refold_plan.refold_target_config_path(
                item=it,
                config_yaml=config_yaml,
            )
            if src_config is None:
                _record_failure(it, reason="missing_src_config")
                continue
            cfg = _load_cfg_for_refold(src_config)

            af3_cfg = None
            px_cfg = None
            try:
                if want_af3 and not af3_done:
                    af3_cfg = _load_eval_subcfg(cfg, "alphafold3")
                if want_px and not px_done:
                    px_cfg = _load_eval_subcfg(cfg, "protenix")
                runtime_overrides = _load_refold_runtime_overrides(cfg)
                input_context = _load_refold_input_context(cfg, item=it)
            except Exception as e:
                _record_failure(it, reason="evaluator_config_resolution_failed", exc=e)
                continue

            # Resolve homology paths again at runtime to allow path-map/relpath-base overrides.
            homologies = it.get("homologies") or {}
            try:
                homologies = _resolve_homology_value(
                    v=homologies,
                    relpath_base=relpath_base,
                    path_maps=maps,
                    require_abs=True,
                ) or {}
            except Exception as e:
                _record_failure(it, reason="homology_path_resolution_failed", exc=e)
                continue

            types = it.get("types") or {}

            # Run Protenix first to avoid overwriting canonical AF3 structure file.
            if want_px and not px_done and px_cfg is not None:
                tmp_out = dst_run_dir / "_refold_tmp_protenix"
                _init_refold_global_config(
                    out_dir=tmp_out,
                    **runtime_overrides,
                    **input_context,
                )
                _build_protomers(seqs=seqs, types=types, homologies=homologies)
                global_config.set_variable("step", step)
                inputs = construct_standard_input([oligo], seed=int(getattr(cfg, "seed", 0) or 0))

                cache_key = _refold_backend_resource_key(
                    item=it,
                    backend="protenix",
                    config_yaml=config_yaml,
                )
                evals = evaluator_cache.get(cache_key)
                if evals is None:
                    evaluator_payload = {
                        "evaluator_terms": {"Protenix": {"_type": "Protenix"}},
                        "protenix": px_cfg,
                    }
                    weight_path_cfg = _load_optional_cfg_node(cfg, "weight_path")
                    if weight_path_cfg is not None:
                        evaluator_payload["weight_path"] = weight_path_cfg
                    evaluator_conf = OmegaConf.create(evaluator_payload)
                    evals = construct_evaluators(evaluator_conf, dst_run_dir / "raw_model_results", oligo_names=[oligo])
                    evaluator_cache[cache_key] = evals
                    evaluator_cache_misses += 1
                else:
                    evaluator_cache_hits += 1
                _retarget_refold_evaluators(
                    evals=evals,
                    backend="protenix",
                    dst_run_dir=dst_run_dir,
                )
                evals["Protenix"].evaluate(inputs, runtime_config={})
                ran_protenix += 1
                _append_refold_backend(rows_by_step.setdefault(step, {}), "protenix", action="run")
                if not keep_protenix_structures:
                    shutil.rmtree(str(tmp_out), ignore_errors=True)

            if want_af3 and not af3_done and af3_cfg is not None:
                _init_refold_global_config(
                    out_dir=dst_run_dir,
                    **runtime_overrides,
                    **input_context,
                )
                _build_protomers(seqs=seqs, types=types, homologies=homologies)
                global_config.set_variable("step", step)
                inputs = construct_standard_input([oligo], seed=int(getattr(cfg, "seed", 0) or 0))

                cache_key = _refold_backend_resource_key(
                    item=it,
                    backend="af3",
                    config_yaml=config_yaml,
                )
                evals = evaluator_cache.get(cache_key)
                if evals is None:
                    evaluator_conf = OmegaConf.create(
                        {
                            "evaluator_terms": {"Alphafold3": {"_type": "Alphafold3"}},
                            "alphafold3": af3_cfg,
                        }
                    )
                    evals = construct_evaluators(evaluator_conf, dst_run_dir / "raw_model_results", oligo_names=[oligo])
                    evaluator_cache[cache_key] = evals
                    evaluator_cache_misses += 1
                else:
                    evaluator_cache_hits += 1
                _retarget_refold_evaluators(
                    evals=evals,
                    backend="af3",
                    dst_run_dir=dst_run_dir,
                )
                std_scores = evals["Alphafold3"].evaluate(inputs, runtime_config={})
                ran_af3 += 1
                _append_refold_backend(rows_by_step.setdefault(step, {}), "af3", action="run")

                # Populate a minimal scores row for downstream posthoc collection.
                row: Dict[str, Any] = {
                    "src_run_dir": str(it.get("src_run_dir", "")),
                }
                try:
                    iptm = None
                    ptm = None
                    plddt = None
                    if getattr(std_scores, "iptm", None) and oligo in std_scores.iptm:
                        iptm = std_scores.iptm[oligo][0]
                    if getattr(std_scores, "ptm", None) and oligo in std_scores.ptm:
                        ptm = std_scores.ptm[oligo][0]
                    if getattr(std_scores, "plddt", None) and oligo in std_scores.plddt:
                        # AF3 evaluator returns scalar plddt list; keep first.
                        plddt = std_scores.plddt[oligo][0]
                    if iptm is not None:
                        row["AB.iptm"] = float(iptm)
                        row["iptm"] = float(iptm)
                        row["af3_iptm"] = float(iptm)
                    if ptm is not None:
                        row["AB.ptm"] = float(ptm)
                        row["ptm"] = float(ptm)
                        row["af3_ptm"] = float(ptm)
                    if plddt is not None:
                        try:
                            row["AB.plddt"] = float(plddt)
                            row["plddt"] = float(plddt)
                            row["af3_plddt"] = float(plddt)
                        except Exception:
                            pass
                except Exception:
                    pass
                rows_by_step.setdefault(step, {}).update(row)

            processed += 1

        # Always ensure a scores.csv for this run dir (even if empty), for consistent scanning.
        if rows_by_step:
            _merge_backend_metrics_into_rows(
                rows_by_step=rows_by_step,
                run_dir=dst_run_dir,
                binder_chain=binder_chain,
                target_chain=target_chain,
            )
            _ensure_scores_csv(run_dir=dst_run_dir, rows_by_step=rows_by_step)

    if failure_records:
        tmp = failure_jsonl.with_suffix(failure_jsonl.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in failure_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(str(tmp), str(failure_jsonl))

    return {
        "manifest_jsonl": str(manifest_jsonl),
        "out_root": str(out_root),
        "evaluators": evaluator_backends,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "ran_af3": ran_af3,
        "ran_protenix": ran_protenix,
        "reused_af3": reused_af3,
        "reused_protenix": reused_protenix,
        "groups": len(plan.groups),
        "resource_groups": resource_group_count,
        "evaluator_cache_hits": evaluator_cache_hits,
        "evaluator_cache_misses": evaluator_cache_misses,
        "dry_run": bool(dry_run),
        "workers_requested": int(workers),
        "workers": 1,
        "config_yaml": str(config_yaml) if config_yaml is not None else None,
        "binder_chain": str(binder_chain),
        "target_chain": str(target_chain),
        "relpath_base": str(relpath_base) if relpath_base else None,
        "path_map": [f"{m.src}={m.dst}" for m in maps],
        "failure_report": str(failure_jsonl) if failure_records else None,
    }
