from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

from astrolabe.posthoc.db import (
    infer_db_out_root,
    iter_records,
    open_db,
    set_meta,
    set_out_roots,
    upsert_records,
    upsert_run_signature,
)
from astrolabe.posthoc.io_utils import sanitize_token


_MISSING = object()
_BACKTICK_METRIC_RE = re.compile(r"`([^`]+)`")
_ALLOWED_EXPR_FUNCS = {
    "abs": abs,
    "max": max,
    "min": min,
    "round": round,
    "exists": lambda value: value is not _MISSING,
    "missing": lambda value: value is _MISSING,
}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
)
_OP_SYMBOLS = {
    "gt": ">",
    "ge": ">=",
    "lt": "<",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
}
_RUN_SELECTION_ALL = "all-passing"
_RUN_SELECTION_REPRESENTATIVE = "representative-per-run"
_VALID_RUN_SELECTIONS = {
    _RUN_SELECTION_ALL,
    _RUN_SELECTION_REPRESENTATIVE,
}
_ROW_METADATA_KEYS = (
    "task",
    "run",
    "step",
    "oligo",
    "run_dir",
    "structure_path",
)
_REPORT_METRIC_ORDER = (
    "weighted_score",
    "accepted",
    "stage",
    "iteration",
    "binder_ptm",
    "binder_iptm",
    "iptm",
    "ptm",
    "plddt",
    "chain_iptm",
    "chain_pair_iptm",
    "cross_iptm",
    "confidence",
    "contact_molecular_surface",
)
_REPORT_DOTTED_SUFFIX_ORDER = _REPORT_METRIC_ORDER[4:]
_REPORT_DOTTED_SUFFIXES = frozenset(_REPORT_DOTTED_SUFFIX_ORDER)
_REPORT_PROVIDER_PREFIXES = (
    "af3_",
    "halludesign_af3_",
    "protenix_",
    "boltz_",
)


@dataclass(frozen=True)
class Rule:
    key: str
    op: str
    value: float
    allow_missing: bool = False


@dataclass(frozen=True)
class Preset:
    name: str
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class CompiledFilterExpression:
    source: str
    code: Any
    symbol_to_metric: tuple[tuple[str, str], ...]
    referenced_keys: tuple[str, ...]
    optional_keys: tuple[str, ...]
    required_keys: tuple[str, ...]


@dataclass(frozen=True)
class FilterResult:
    selector: str
    passed: bool
    failed: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class PreparedFilterExpression:
    expr_env: dict[str, Any]
    missing: tuple[str, ...]
    invalid: tuple[str, ...]
    required_missing: tuple[str, ...]
    required_invalid: tuple[str, ...]


class MissingRequiredMetricsError(RuntimeError):
    pass


class FilterExpressionRuntimeError(RuntimeError):
    pass


class _MetricAttrTransformer(ast.NodeTransformer):
    def __init__(self, existing: Dict[str, str] | None = None):
        self.symbol_to_metric: Dict[str, str] = dict(existing or {})
        self.metric_to_symbol: Dict[str, str] = {
            metric: symbol for symbol, metric in self.symbol_to_metric.items()
        }
        self._counter = len(self.symbol_to_metric)

    def _allocate_symbol(self, metric_name: str) -> str:
        symbol = self.metric_to_symbol.get(metric_name)
        if symbol is not None:
            return symbol
        symbol = f"__metric_{self._counter}"
        self._counter += 1
        self.symbol_to_metric[symbol] = metric_name
        self.metric_to_symbol[metric_name] = symbol
        return symbol

    def _attribute_metric_name(self, node: ast.Attribute) -> str | None:
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        parts.append(cur.id)
        return ".".join(reversed(parts))

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        metric_name = self._attribute_metric_name(node)
        if metric_name is None:
            return self.generic_visit(node)
        return ast.copy_location(
            ast.Name(id=self._allocate_symbol(metric_name), ctx=ast.Load()),
            node,
        )


class _FilterExprValidator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Unsupported syntax in filter expression: {ast.dump(node, include_attributes=False)}"
            )
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_EXPR_FUNCS:
            raise ValueError(
                f"Unsupported function in filter expression: {ast.dump(node.func, include_attributes=False)}"
            )
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in filter expressions")
        for arg in node.args:
            self.visit(arg)


class _OptionalMetricCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.optional_symbols: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"missing", "exists"}:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.optional_symbols.append(arg.id)
        self.generic_visit(node)


def _format_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    try:
        import numpy as np  # type: ignore

        if isinstance(v, (np.integer, np.floating)):
            return str(v.item())
    except Exception:
        pass
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(v)
    return str(v)


def _default_filtered_db_path(dst_dir: Path) -> Path:
    return Path(dst_dir) / "filtered.sqlite"


def _sqlite_output_paths(db_path: Path) -> tuple[Path, Path, Path]:
    db_path = Path(db_path)
    return (
        db_path,
        Path(str(db_path) + "-wal"),
        Path(str(db_path) + "-shm"),
    )


def _prepare_fresh_sqlite_output(out_db: Path, *, source_db: Path) -> Path:
    out_db = Path(out_db).expanduser().resolve()
    source_db = Path(source_db).expanduser().resolve()
    if out_db == source_db:
        raise ValueError("--out-db must not overwrite the source --db")
    out_db.parent.mkdir(parents=True, exist_ok=True)
    for path in _sqlite_output_paths(out_db):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return out_db


def load_metrics_json(row: Any) -> Dict[str, Any]:
    try:
        metrics = json.loads(row["metrics_json"])
    except Exception:
        metrics = {}
    if not isinstance(metrics, dict):
        metrics = {}
    return metrics


def _write_filtered_db_snapshot(
    *,
    source_db_path: Path,
    out_db_path: Path,
    dst_dir: Path,
    report_csv: Path | None,
    selector_kind: str,
    selector: str,
    preset_name: str,
    filter_expr: str,
    prefer_oligo: str,
    run_selection: str,
    records: list[dict[str, Any]],
) -> str:
    out_db_path = _prepare_fresh_sqlite_output(out_db_path, source_db=source_db_path)
    conn = open_db(out_db_path)
    try:
        set_meta(conn, key="dataset_kind", value="posthoc_filtered")
        set_meta(conn, key="filtered_source_db", value=str(source_db_path))
        set_meta(conn, key="filtered_dst_dir", value=str(dst_dir))
        set_meta(conn, key="filtered_report_csv", value=str(report_csv) if report_csv else "")
        set_meta(conn, key="filtered_selector_kind", value=str(selector_kind))
        set_meta(conn, key="filtered_selector", value=str(selector))
        set_meta(conn, key="filtered_preset", value=str(preset_name))
        set_meta(conn, key="filtered_filter_expr", value=str(filter_expr))
        set_meta(conn, key="filtered_prefer_oligo", value=str(prefer_oligo))
        set_meta(conn, key="filtered_run_selection", value=str(run_selection))
        set_meta(conn, key="filtered_created_at", value=str(time.time()))
        set_out_roots(conn, roots=[str(dst_dir)])

        seen_runs: set[str] = set()
        for record in records:
            run_dir = str(record["run_dir"])
            if run_dir in seen_runs:
                continue
            seen_runs.add(run_dir)
            upsert_run_signature(
                conn,
                run_dir=run_dir,
                task=str(record["task"]),
                run=str(record["run"]),
                metadata={
                    "dataset_kind": "posthoc_filtered",
                    "filtered_source_db": str(source_db_path),
                    "filtered_dst_dir": str(dst_dir),
                    "filtered_report_csv": str(report_csv) if report_csv else "",
                    "filtered_selector_kind": str(selector_kind),
                    "filtered_selector": str(selector),
                },
                scores_mtime_ns=int(record["scores_mtime_ns"]),
                scores_size=int(record["scores_size"]),
            )
        upsert_records(conn, records)
    finally:
        conn.close()
    return str(out_db_path)


def build_filter_context(row: Any, metrics: Dict[str, Any]) -> Dict[str, Any]:
    context = dict(metrics)
    context["task"] = str(row["task"])
    context["run"] = str(row["run"])
    context["step"] = int(row["step"])
    context["oligo"] = str(row["oligo"])
    context["run_dir"] = str(row["run_dir"])
    context["structure_path"] = str(row["structure_path"] or "")
    return context


def _normalize_run_selection(value: str | None) -> str:
    normalized = str(value or _RUN_SELECTION_ALL).strip()
    if normalized not in _VALID_RUN_SELECTIONS:
        raise ValueError(
            f"Invalid run selection {value!r}; expected one of {sorted(_VALID_RUN_SELECTIONS)}"
        )
    return normalized


def _logical_run_dir(row: Any, metrics: Dict[str, Any]) -> str:
    raw = metrics.get("src_run_dir")
    if raw is None:
        return str(row["run_dir"])
    logical = str(raw).strip()
    return logical or str(row["run_dir"])


def _logical_run_sort_key(row: Any, metrics: Dict[str, Any]) -> tuple[int, int, float, str]:
    logical = _logical_run_dir(row, metrics)
    origin_rank = 0 if str(row["run_dir"]) == logical else 1
    try:
        step = int(row["step"])
    except Exception:
        step = 0
    try:
        updated_at = float(row["updated_at"])
    except Exception:
        updated_at = 0.0
    return (origin_rank, step, updated_at, str(row["run_dir"]))


def _coerce_weighted_score(metrics: Dict[str, Any]) -> float | None:
    value = metrics.get("weighted_score")
    if not _is_valid_float(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _prefer_representative_row(
    *,
    current_row: Any,
    current_weighted_score: float | None,
    best_row: Any | None,
    best_weighted_score: float | None,
) -> bool:
    if best_row is None:
        return True
    if current_weighted_score is not None and best_weighted_score is None:
        return True
    if current_weighted_score is None and best_weighted_score is not None:
        return False
    if current_weighted_score is not None and best_weighted_score is not None:
        if current_weighted_score > best_weighted_score:
            return True
        if current_weighted_score < best_weighted_score:
            return False
    return int(current_row["step"]) >= int(best_row["step"])


def iter_selected_rows(
    conn,
    *,
    prefer_oligo: str,
    run_selection: str,
):
    normalized_selection = _normalize_run_selection(run_selection)
    if normalized_selection == _RUN_SELECTION_ALL:
        yield from iter_records(conn, oligo=str(prefer_oligo))
        return

    grouped: Dict[tuple[str, str, str], list[tuple[Any, Dict[str, Any]]]] = {}
    for row in iter_records(conn, oligo=str(prefer_oligo)):
        metrics = load_metrics_json(row)
        run_key = (_logical_run_dir(row, metrics), str(row["task"]), str(row["run"]))
        grouped.setdefault(run_key, []).append((row, metrics))

    for run_key in sorted(grouped):
        current_stage: str | None = None
        best_row: Any | None = None
        best_weighted_score: float | None = None
        for row, metrics in sorted(
            grouped[run_key],
            key=lambda item: _logical_run_sort_key(item[0], item[1]),
        ):
            row_stage = str(metrics.get("stage") or "")
            row_weighted_score = _coerce_weighted_score(metrics)
            if current_stage != row_stage:
                current_stage = row_stage
                best_row = row
                best_weighted_score = row_weighted_score
                continue
            if _prefer_representative_row(
                current_row=row,
                current_weighted_score=row_weighted_score,
                best_row=best_row,
                best_weighted_score=best_weighted_score,
            ):
                best_row = row
                best_weighted_score = row_weighted_score
        if best_row is not None:
            yield best_row


_load_metrics_json = load_metrics_json
_build_filter_context = build_filter_context
_iter_selected_rows = iter_selected_rows


def _cmp(op: str, x: float, y: float) -> bool:
    if op == "gt":
        return x > y
    if op == "ge":
        return x >= y
    if op == "lt":
        return x < y
    if op == "le":
        return x <= y
    if op == "eq":
        return x == y
    if op == "ne":
        return x != y
    raise ValueError(f"Unknown op: {op}")


def _is_valid_float(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    try:
        x = float(v)
    except Exception:
        return False
    if math.isnan(x) or math.isinf(x):
        return False
    return True


def _has_report_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return not isinstance(value, (list, dict, set, tuple))


def _observed_report_keys(
    rows,
    *,
    prefer_oligo: str,
) -> list[str]:
    observed: set[str] = set()
    for row in rows:
        metrics = load_metrics_json(row)
        context = build_filter_context(row, metrics)
        observed.update(
            str(key)
            for key, value in context.items()
            if key not in _ROW_METADATA_KEYS and _has_report_value(value)
        )

    ordered = [key for key in _REPORT_METRIC_ORDER if key in observed]
    ordered.extend(
        sorted(
            key
            for key in observed
            if key.endswith("_length") and key not in ordered
        )
    )

    for suffix in _REPORT_DOTTED_SUFFIX_ORDER:
        key = f"{prefer_oligo}.{suffix}"
        if key in observed and key not in ordered:
            ordered.append(key)

    dotted_aliases: list[str] = []
    for key in observed:
        if key.count(".") != 1:
            continue
        prefix, suffix = key.split(".", 1)
        if suffix not in _REPORT_DOTTED_SUFFIXES:
            continue
        if prefix == str(prefer_oligo):
            continue
        if re.fullmatch(r"[A-Z]+", prefix):
            continue
        dotted_aliases.append(key)
    ordered.extend(sorted(key for key in dotted_aliases if key not in ordered))

    provider_keys = sorted(
        key
        for key in observed
        if key.startswith(_REPORT_PROVIDER_PREFIXES) and key not in ordered
    )
    ordered.extend(provider_keys)
    return ordered


def _coerce_expr_value(v: Any) -> Any:
    if v is _MISSING:
        return v
    if isinstance(v, str):
        vv = v.strip()
        if vv == "True":
            return True
        if vv == "False":
            return False
        try:
            if "." in vv or "e" in vv.lower():
                return float(vv)
            return int(vv)
        except Exception:
            return v
    return v


def _is_invalid_expr_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, dict, set, tuple)):
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, str):
        vv = v.strip()
        if not vv:
            return True
        if vv in {"True", "False"}:
            return False
        try:
            return not math.isfinite(float(vv))
        except Exception:
            return False
    try:
        return not math.isfinite(float(v))
    except Exception:
        return True


def _prepare_filter_expression(
    metrics: Dict[str, Any],
    compiled: CompiledFilterExpression,
) -> PreparedFilterExpression:
    missing: List[str] = []
    invalid: List[str] = []
    expr_env: Dict[str, Any] = {}
    required_keys = set(compiled.required_keys)

    for symbol, metric_name in compiled.symbol_to_metric:
        if metric_name not in metrics:
            missing.append(metric_name)
            expr_env[symbol] = _MISSING
            continue
        raw_value = metrics[metric_name]
        if _is_invalid_expr_value(raw_value):
            invalid.append(metric_name)
            expr_env[symbol] = raw_value
            continue
        expr_env[symbol] = _coerce_expr_value(raw_value)

    required_missing = tuple(key for key in missing if key in required_keys)
    required_invalid = tuple(key for key in invalid if key in required_keys)
    return PreparedFilterExpression(
        expr_env=expr_env,
        missing=tuple(dict.fromkeys(missing)),
        invalid=tuple(dict.fromkeys(invalid)),
        required_missing=tuple(dict.fromkeys(required_missing)),
        required_invalid=tuple(dict.fromkeys(required_invalid)),
    )


def _render_metric_ref(key: str) -> str:
    parts = [part for part in str(key).split(".") if part]
    if parts and all(part.isidentifier() for part in parts):
        return str(key)
    return f"`{key}`"


def _render_rule_preview(rule: Rule) -> str:
    metric_ref = _render_metric_ref(rule.key)
    clause = f"{metric_ref} {_OP_SYMBOLS[rule.op]} {rule.value:g}"
    if rule.allow_missing:
        return f"(missing({metric_ref}) or {clause})"
    return clause


def _replace_backtick_metric_refs(expr: str) -> tuple[str, Dict[str, str]]:
    symbol_to_metric: Dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        metric_name = match.group(1).strip()
        if not metric_name:
            raise ValueError("Empty backtick metric name in filter expression")
        symbol = f"__metric_{len(symbol_to_metric)}"
        symbol_to_metric[symbol] = metric_name
        return symbol

    return _BACKTICK_METRIC_RE.sub(_repl, expr), symbol_to_metric


def load_preset(path: Path) -> Preset:
    cfg = OmegaConf.load(str(Path(path).expanduser().resolve()))
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid preset YAML: {path}")
    name = str(raw.get("name", Path(path).stem))
    rules = []
    for r in raw.get("rules", []) or []:
        if not isinstance(r, dict):
            continue
        rules.append(
            Rule(
                key=str(r["key"]),
                op=str(r["op"]),
                value=float(r["value"]),
                allow_missing=bool(r.get("allow_missing", False)),
            )
        )
    return Preset(name=name, rules=tuple(rules))


def compile_filter_expression(expr: str) -> CompiledFilterExpression:
    source = str(expr or "").strip()
    if not source:
        raise ValueError("Empty filter expression")

    prepared_expr, backtick_refs = _replace_backtick_metric_refs(source)
    try:
        tree = ast.parse(prepared_expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid filter expression: {source!r}") from exc

    transformer = _MetricAttrTransformer(existing=backtick_refs)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    _FilterExprValidator().visit(tree)

    symbol_to_metric = dict(transformer.symbol_to_metric)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if node.id in _ALLOWED_EXPR_FUNCS or node.id in {"True", "False", "None"}:
            continue
        symbol_to_metric.setdefault(node.id, node.id)

    referenced_keys = tuple(dict.fromkeys(symbol_to_metric.values()))
    optional_collector = _OptionalMetricCollector()
    optional_collector.visit(tree)
    optional_keys = tuple(
        dict.fromkeys(
            symbol_to_metric[symbol]
            for symbol in optional_collector.optional_symbols
            if symbol in symbol_to_metric
        )
    )
    required_keys = tuple(key for key in referenced_keys if key not in set(optional_keys))
    return CompiledFilterExpression(
        source=source,
        code=compile(tree, "<filter_expr>", "eval"),
        symbol_to_metric=tuple(symbol_to_metric.items()),
        referenced_keys=referenced_keys,
        optional_keys=optional_keys,
        required_keys=required_keys,
    )


def apply_preset(metrics: Dict[str, Any], preset: Preset) -> FilterResult:
    failed: List[str] = []
    missing: List[str] = []
    for rule in preset.rules:
        if rule.key not in metrics:
            missing.append(rule.key)
            if not rule.allow_missing:
                failed.append(rule.key)
            continue
        try:
            x = float(metrics[rule.key])
        except Exception:
            failed.append(rule.key)
            continue
        if not _cmp(rule.op, x, rule.value):
            failed.append(rule.key)
    return FilterResult(
        selector=preset.name,
        passed=(len(failed) == 0),
        failed=tuple(sorted(set(failed))),
        missing=tuple(sorted(set(missing))),
    )


def apply_filter_expression(metrics: Dict[str, Any], compiled: CompiledFilterExpression) -> FilterResult:
    prepared = _prepare_filter_expression(metrics, compiled)
    if prepared.required_missing or prepared.required_invalid:
        return FilterResult(
            selector=compiled.source,
            passed=False,
            failed=tuple(
                sorted(
                    set(
                        [
                            *prepared.required_missing,
                            *prepared.required_invalid,
                            "filter_expr",
                        ]
                    )
                )
            ),
            missing=prepared.missing,
        )
    if prepared.invalid:
        invalid_keys = ", ".join(prepared.invalid)
        raise FilterExpressionRuntimeError(
            f"Filter expression encountered invalid metric value(s): {invalid_keys}"
        )

    try:
        result = eval(
            compiled.code,
            {"__builtins__": {}},
            {**_ALLOWED_EXPR_FUNCS, **prepared.expr_env},
        )
    except Exception as exc:
        raise FilterExpressionRuntimeError(
            f"Filter expression runtime error for {compiled.source!r}: {exc}"
        ) from exc

    if not isinstance(result, bool):
        raise FilterExpressionRuntimeError(
            f"Filter expression must evaluate to a boolean value, got {result!r}"
        )

    failed: tuple[str, ...] = ()
    if not result:
        failed = tuple(sorted(set([*prepared.missing, "filter_expr"])))
    return FilterResult(
        selector=compiled.source,
        passed=bool(result),
        failed=failed,
        missing=prepared.missing,
    )


def preview_preset(path: Path) -> dict[str, Any]:
    preset_path = Path(path).expanduser().resolve()
    preset = load_preset(preset_path)
    rule_metrics = list(dict.fromkeys([rule.key for rule in preset.rules]))
    required_metrics = list(
        dict.fromkeys([rule.key for rule in preset.rules if not rule.allow_missing])
    )
    preview_filter_expr = " and ".join(_render_rule_preview(rule) for rule in preset.rules) or "True"
    return {
        "selector_kind": "preset",
        "selector": preset.name,
        "preset": preset.name,
        "preset_path": str(preset_path),
        "rule_count": len(preset.rules),
        "rule_metrics": rule_metrics,
        "required_metrics": required_metrics,
        "preview_filter_expr": preview_filter_expr,
        "rules": [
            {
                "key": rule.key,
                "op": rule.op,
                "value": rule.value,
                "allow_missing": rule.allow_missing,
            }
            for rule in preset.rules
        ],
    }


def filter_structures(
    *,
    db_path: Path,
    dst_dir: Path,
    preset_path: Path | None = None,
    filter_expr: str | None = None,
    prefer_oligo: str = "AB",
    hardlink: bool = True,
    overwrite: bool = False,
    flat: bool = False,
    report_csv: Path | None = None,
    out_db: Path | None = None,
    fail_on_missing_required: bool = True,
    run_selection: str = _RUN_SELECTION_ALL,
    dry_run: bool = False,
) -> dict[str, Any]:
    if bool(preset_path) == bool(filter_expr):
        raise ValueError("Exactly one of preset_path or filter_expr must be provided")

    db_path = Path(db_path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Scores database not found: {db_path}. "
            "Run gh posthoc collect --out-root <outputs> --db <scores.sqlite> first."
        )
    dst_dir = Path(dst_dir).expanduser().resolve()
    conn = open_db(db_path)

    preset: Preset | None = None
    compiled_expr: CompiledFilterExpression | None = None
    selector_kind = "preset"
    selector = ""

    if preset_path is not None:
        preset_path = Path(preset_path).expanduser().resolve()
        preset = load_preset(preset_path)
        selector = preset.name
        rule_keys = list(dict.fromkeys([rule.key for rule in preset.rules]))
        required_keys = list(
            dict.fromkeys([rule.key for rule in preset.rules if not rule.allow_missing])
        )
    else:
        selector_kind = "dsl"
        compiled_expr = compile_filter_expression(str(filter_expr))
        selector = compiled_expr.source
        rule_keys = list(compiled_expr.referenced_keys)
        required_keys = list(compiled_expr.required_keys)

    if report_csv is None and not dry_run:
        report_csv = dst_dir / "filter_report.csv"
    if out_db is None and not dry_run:
        out_db = _default_filtered_db_path(dst_dir)
    if out_db is not None:
        out_db = Path(out_db).expanduser().resolve()
    run_selection = _normalize_run_selection(run_selection)
    db_out_root = infer_db_out_root(conn)

    selected = 0
    linked = 0
    linkable = 0
    structure_missing = 0
    total = 0
    filtered_records: list[dict[str, Any]] = []

    source_rows = lambda: iter_selected_rows(  # noqa: E731
        conn,
        prefer_oligo=str(prefer_oligo),
        run_selection=run_selection,
    )
    default_keys = _observed_report_keys(
        source_rows(),
        prefer_oligo=str(prefer_oligo),
    )
    extra_keys = [k for k in default_keys if k not in rule_keys] + rule_keys

    missing_required_counts: Dict[str, int] = {}
    invalid_required_counts: Dict[str, int] = {}
    if fail_on_missing_required and required_keys:
        for row in source_rows():
            metrics = load_metrics_json(row)
            context = build_filter_context(row, metrics)
            if preset is not None:
                for key in required_keys:
                    if key not in context:
                        missing_required_counts[key] = missing_required_counts.get(key, 0) + 1
                        continue
                    if not _is_valid_float(context.get(key)):
                        invalid_required_counts[key] = invalid_required_counts.get(key, 0) + 1
                continue
            assert compiled_expr is not None
            prepared = _prepare_filter_expression(context, compiled_expr)
            for key in prepared.required_missing:
                missing_required_counts[key] = missing_required_counts.get(key, 0) + 1
            for key in prepared.required_invalid:
                invalid_required_counts[key] = invalid_required_counts.get(key, 0) + 1

    strict_missing = bool(missing_required_counts or invalid_required_counts)
    do_link = not strict_missing

    tmp = report_csv.with_suffix(report_csv.suffix + f".{os.getpid()}.tmp") if report_csv else None
    writer = None
    handle = None
    if report_csv is not None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        handle = tmp.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "run",
                "step",
                "oligo",
                "run_dir",
                "passed",
                "failed",
                "missing",
                "link_issue",
                "src_structure",
                "dst_structure",
                *extra_keys,
            ],
        )
        writer.writeheader()

    try:
        for row in source_rows():
            total += 1
            metrics = load_metrics_json(row)
            context = build_filter_context(row, metrics)

            if preset is not None:
                res = apply_preset(context, preset)
            else:
                assert compiled_expr is not None
                res = apply_filter_expression(context, compiled_expr)

            src = row["structure_path"]
            dst = ""
            link_issue = ""
            if not res.passed:
                continue

            selected += 1

            if do_link and src:
                src_path = Path(src)
                if not src_path.exists():
                    link_issue = "structure_missing"
                    structure_missing += 1
                else:
                    linkable += 1
                    if not dry_run:
                        task = sanitize_token(row["task"])
                        run = sanitize_token(row["run"])
                        step = int(row["step"])
                        oligo = sanitize_token(row["oligo"])
                        ext = src_path.suffix
                        fname = f"{task}__{run}__step_{step:03d}__{oligo}_model{ext}"
                        out_dir = dst_dir if flat else (dst_dir / task)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        dst_path = out_dir / fname
                        if dst_path.exists() and not overwrite:
                            dst = str(dst_path)
                        else:
                            if dst_path.exists():
                                dst_path.unlink()
                            if hardlink:
                                try:
                                    os.link(src_path, dst_path)
                                except OSError as exc:
                                    raise OSError(
                                        f"Hardlink failed ({exc}). Use a destination on the same filesystem, "
                                        "or rerun with hardlink disabled (explicit copy mode is not enabled by default)."
                                    ) from exc
                            else:
                                import shutil

                                shutil.copy2(src_path, dst_path)
                            dst = str(dst_path)
                        linked += 1
            elif do_link and not src:
                link_issue = "structure_missing"
                structure_missing += 1

            if not dry_run:
                filtered_metrics = dict(metrics)
                filtered_metrics.update(
                    {
                        "filtered_source_db": str(db_path),
                        "filtered_source_run_dir": str(row["run_dir"]),
                        "filtered_source_structure": str(src or ""),
                        "filtered_dst_structure": str(dst),
                        "filtered_link_issue": str(link_issue),
                        "filtered_selector_kind": selector_kind,
                        "filtered_selector": selector,
                        "filtered_preset": preset.name if preset is not None else "",
                        "filtered_filter_expr": compiled_expr.source if compiled_expr is not None else "",
                        "filtered_prefer_oligo": str(prefer_oligo),
                        "filtered_run_selection": run_selection,
                        "filtered_report_csv": str(report_csv) if report_csv is not None else "",
                        "filtered_dst_dir": str(dst_dir),
                    }
                )
                filtered_records.append(
                    {
                        "run_dir": str(row["run_dir"]),
                        "task": str(row["task"]),
                        "run": str(row["run"]),
                        "step": int(row["step"]),
                        "oligo": str(row["oligo"]),
                        "structure_path": str(dst),
                        "metrics": filtered_metrics,
                        "scores_mtime_ns": int(row["scores_mtime_ns"]),
                        "scores_size": int(row["scores_size"]),
                    }
                )

            if writer is not None:
                out_row: Dict[str, Any] = {
                    "task": row["task"],
                    "run": row["run"],
                    "step": row["step"],
                    "oligo": row["oligo"],
                    "run_dir": row["run_dir"],
                    "passed": bool(res.passed),
                    "failed": ";".join(res.failed),
                    "missing": ";".join(res.missing),
                    "link_issue": link_issue,
                    "src_structure": src or "",
                    "dst_structure": dst,
                }
                for key in extra_keys:
                    out_row[key] = _format_cell(context.get(key))
                writer.writerow(out_row)
    finally:
        if handle is not None:
            handle.close()

    if report_csv is not None and tmp is not None:
        os.replace(str(tmp), str(report_csv))

    if strict_missing:
        def _fmt_counts(counts: Dict[str, int]) -> str:
            parts = [f"{key}({value})" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
            return ", ".join(parts) if parts else ""

        if preset is not None:
            msg = (
                "Preset requires metrics that are missing/invalid for some records; refusing to select/link structures.\n"
                f"- preset: {preset.name}\n"
            )
        else:
            msg = (
                "Filter expression references metrics that are missing/invalid for some records; refusing to select/link structures.\n"
                f"- filter_expr: {compiled_expr.source if compiled_expr is not None else ''}\n"
            )
        msg += (
            f"- db: {db_path}\n"
            f"- prefer_oligo: {prefer_oligo}\n"
            f"- report_csv: {report_csv or ''}\n"
        )
        if missing_required_counts:
            msg += f"- missing_required: {_fmt_counts(missing_required_counts)}\n"
        if invalid_required_counts:
            msg += f"- invalid_required: {_fmt_counts(invalid_required_counts)}\n"
        if preset is not None:
            msg += (
                "To generate missing metrics, run selector-driven refold against the same DB, then rerun filter.\n"
                "Suggested:\n"
                f"  gh posthoc refold --db '{db_path}' --preset '{preset_path}' --prefer-oligo '{prefer_oligo}' "
                f"--run-selection '{run_selection}' --dst '{dst_dir}' --dry-run\n"
            )
        else:
            msg += (
                "Rerun with --allow-missing-required for exploratory filtering, or run selector-driven refold on the same DB:\n"
                f"  gh posthoc refold --db '{db_path}' --filter-expr {compiled_expr.source!r} --prefer-oligo '{prefer_oligo}' "
                f"--run-selection '{run_selection}' --dst '{dst_dir}' --dry-run\n"
            )
        raise MissingRequiredMetricsError(msg.rstrip())

    out_db_written = ""
    if out_db is not None and not dry_run:
        out_db_written = _write_filtered_db_snapshot(
            source_db_path=db_path,
            out_db_path=out_db,
            dst_dir=dst_dir,
            report_csv=report_csv,
            selector_kind=selector_kind,
            selector=selector,
            preset_name=preset.name if preset is not None else "",
            filter_expr=compiled_expr.source if compiled_expr is not None else "",
            prefer_oligo=str(prefer_oligo),
            run_selection=run_selection,
            records=filtered_records,
        )

    return {
        "db_path": str(db_path),
        "db_out_root": str(db_out_root or ""),
        "selector_kind": selector_kind,
        "selector": selector,
        "preset": preset.name if preset is not None else "",
        "filter_expr": compiled_expr.source if compiled_expr is not None else "",
        "prefer_oligo": str(prefer_oligo),
        "run_selection": run_selection,
        "dst_dir": str(dst_dir),
        "dry_run": bool(dry_run),
        "records_considered": total,
        "selected": selected,
        "linked": linked,
        "linkable": linkable,
        "structure_missing": structure_missing,
        "report_csv": str(report_csv) if report_csv is not None else "",
        "out_db": out_db_written,
    }


def prune_outputs(
    *,
    db_path: Path,
    delete_scope: str,
    preset_path: Path | None = None,
    filter_expr: str | None = None,
    prefer_oligo: str = "AB",
    run_selection: str = _RUN_SELECTION_ALL,
    report_csv: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    if bool(preset_path) == bool(filter_expr):
        raise ValueError("Exactly one of preset_path or filter_expr must be provided")
    normalized_scope = str(delete_scope or "").strip()
    if normalized_scope not in {"raw_model_results", "run_dir"}:
        raise ValueError(
            "delete_scope must be one of ['raw_model_results', 'run_dir']"
        )

    db_path = Path(db_path).expanduser().resolve()
    conn = open_db(db_path)
    preset: Preset | None = None
    compiled_expr: CompiledFilterExpression | None = None
    selector_kind = "preset"
    selector = ""

    if preset_path is not None:
        preset_path = Path(preset_path).expanduser().resolve()
        preset = load_preset(preset_path)
        selector = preset.name
    else:
        selector_kind = "dsl"
        compiled_expr = compile_filter_expression(str(filter_expr))
        selector = compiled_expr.source

    missing_required_counts: dict[str, int] = {}
    invalid_required_counts: dict[str, int] = {}
    for row in iter_selected_rows(
        conn,
        prefer_oligo=str(prefer_oligo),
        run_selection=run_selection,
    ):
        metrics = load_metrics_json(row)
        context = build_filter_context(row, metrics)
        if preset is not None:
            for rule in preset.rules:
                if rule.allow_missing:
                    continue
                if rule.key not in context:
                    missing_required_counts[rule.key] = missing_required_counts.get(rule.key, 0) + 1
                    continue
                if not _is_valid_float(context.get(rule.key)):
                    invalid_required_counts[rule.key] = invalid_required_counts.get(rule.key, 0) + 1
            continue
        assert compiled_expr is not None
        prepared = _prepare_filter_expression(context, compiled_expr)
        for key in prepared.required_missing:
            missing_required_counts[key] = missing_required_counts.get(key, 0) + 1
        for key in prepared.required_invalid:
            invalid_required_counts[key] = invalid_required_counts.get(key, 0) + 1

    if missing_required_counts or invalid_required_counts:
        def _fmt_counts(counts: Dict[str, int]) -> str:
            parts = [
                f"{key}({value})"
                for key, value in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            return ", ".join(parts) if parts else ""

        if preset is not None:
            msg = (
                "Preset requires metrics that are missing/invalid for some records; refusing to prune outputs.\n"
                f"- preset: {preset.name}\n"
            )
        else:
            msg = (
                "Filter expression references metrics that are missing/invalid for some records; refusing to prune outputs.\n"
                f"- filter_expr: {compiled_expr.source if compiled_expr is not None else ''}\n"
            )
        msg += (
            f"- db: {db_path}\n"
            f"- prefer_oligo: {prefer_oligo}\n"
            f"- delete_scope: {normalized_scope}\n"
        )
        if missing_required_counts:
            msg += f"- missing_required: {_fmt_counts(missing_required_counts)}\n"
        if invalid_required_counts:
            msg += f"- invalid_required: {_fmt_counts(invalid_required_counts)}\n"
        raise MissingRequiredMetricsError(msg.rstrip())

    selected_rows: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    deleted_paths = 0
    missing_paths = 0
    delete_errors = 0

    for row in iter_selected_rows(
        conn,
        prefer_oligo=str(prefer_oligo),
        run_selection=run_selection,
    ):
        metrics = load_metrics_json(row)
        context = build_filter_context(row, metrics)
        if preset is not None:
            res = apply_preset(context, preset)
        else:
            assert compiled_expr is not None
            res = apply_filter_expression(context, compiled_expr)
        if not res.passed:
            continue

        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        target = run_dir if normalized_scope == "run_dir" else run_dir / "raw_model_results"
        target_key = str(target)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        issue = ""
        deleted = False
        if not target.exists():
            issue = "missing_path"
            missing_paths += 1
        elif not dry_run:
            try:
                shutil.rmtree(target)
                deleted = True
                deleted_paths += 1
            except Exception as exc:
                issue = f"delete_failed:{type(exc).__name__}"
                delete_errors += 1

        selected_rows.append(
            {
                "task": row["task"],
                "run": row["run"],
                "step": row["step"],
                "oligo": row["oligo"],
                "run_dir": row["run_dir"],
                "target_path": target_key,
                "delete_scope": normalized_scope,
                "deleted": bool(deleted),
                "delete_issue": issue,
            }
        )

    if report_csv is not None:
        report_csv = Path(report_csv).expanduser().resolve()
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_csv.with_suffix(report_csv.suffix + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task",
                    "run",
                    "step",
                    "oligo",
                    "run_dir",
                    "target_path",
                    "delete_scope",
                    "deleted",
                    "delete_issue",
                ],
            )
            writer.writeheader()
            for item in selected_rows:
                writer.writerow(item)
        os.replace(str(tmp), str(report_csv))

    return {
        "db_path": str(db_path),
        "selector_kind": selector_kind,
        "selector": selector,
        "preset": preset.name if preset is not None else "",
        "filter_expr": compiled_expr.source if compiled_expr is not None else "",
        "prefer_oligo": str(prefer_oligo),
        "run_selection": run_selection,
        "delete_scope": normalized_scope,
        "selected": len(selected_rows),
        "deleted_paths": int(deleted_paths),
        "missing_paths": int(missing_paths),
        "delete_errors": int(delete_errors),
        "dry_run": bool(dry_run),
        "report_csv": str(report_csv) if report_csv is not None else "",
    }
