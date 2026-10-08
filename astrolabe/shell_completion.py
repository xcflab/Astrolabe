from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CompletionItem:
    value: str
    description: str | None = None


@dataclass
class _ParseState:
    parser: argparse.ArgumentParser
    parsers: tuple[argparse.ArgumentParser, ...]
    path: tuple[str, ...]
    seen: dict[str, list[str]]
    present: set[str]
    all_seen: dict[str, list[str]] | None = None
    all_present: set[str] | None = None
    pending_action: argparse.Action | None = None
    pending_remaining: int = 0


def _next_positional_action(state: _ParseState) -> argparse.Action | None:
    for action in state.parser._actions:
        if action.option_strings or isinstance(action, argparse._SubParsersAction):
            continue
        count = len(state.seen.get(action.dest, []))
        if action.nargs in ("*", "+", argparse.REMAINDER) or count < _option_value_arity(action):
            return action
    return None


_METRIC_NAME_CACHE: dict[str, tuple[float, list[str]]] = {}
_METRIC_NAME_CACHE_VERSION = 2
_METRIC_NAME_CACHE_TTL_SECONDS = 300.0
_FIXED_FILTER_COLUMNS = (
    "task",
    "run",
    "step",
    "oligo",
    "run_dir",
    "structure_path",
)
_FILTER_PRESET_RULES_RE = re.compile(r"(?m)^\s*rules\s*:")
_PRESET_SEARCH_DIR_NAMES = ("presets", "preset", "examples")


def _single_line(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    return " ".join(first.split())


def _option_value_arity(action: argparse.Action) -> int:
    if isinstance(
        action,
        (
            argparse._HelpAction,
            argparse._SubParsersAction,
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse._CountAction,
            argparse.BooleanOptionalAction,
        ),
    ):
        return 0
    nargs = action.nargs
    if nargs in (None, 1, "?"):
        return 1
    if isinstance(nargs, int):
        return max(0, nargs)
    return 1


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _subparser_help_map(
    sub: argparse._SubParsersAction | None,
) -> tuple[dict[str, str], dict[int, str]]:
    if sub is None:
        return {}, {}
    help_by_name: dict[str, str] = {}
    parser_to_name: dict[int, str] = {}
    for choice_action in getattr(sub, "_choices_actions", []) or []:
        name = str(getattr(choice_action, "dest", "") or "")
        if not name:
            continue
        help_by_name[name] = _single_line(getattr(choice_action, "help", None))
        subparser = (sub.choices or {}).get(name)
        if subparser is not None:
            parser_to_name[id(subparser)] = name
    return help_by_name, parser_to_name


def _find_option_action(
    parser: argparse.ArgumentParser,
    token: str,
) -> argparse.Action | None:
    for action in parser._actions:
        if token in getattr(action, "option_strings", []):
            return action
    return None


def _find_option_action_in_parsers(
    parsers: Sequence[argparse.ArgumentParser],
    token: str,
) -> argparse.Action | None:
    for parser in reversed(tuple(parsers)):
        action = _find_option_action(parser, token)
        if action is not None:
            return action
    return None


def _option_alias_label(action: argparse.Action) -> str:
    opts = list(getattr(action, "option_strings", []) or [])
    short_opts = [opt for opt in opts if opt.startswith("-") and not opt.startswith("--")]
    long_opts = [opt for opt in opts if opt.startswith("--")]
    remainder = [opt for opt in opts if opt not in short_opts and opt not in long_opts]
    ordered = short_opts + long_opts + remainder
    return " ".join(ordered)


def _select_option_value(action: argparse.Action, prefix: str) -> str | None:
    opts = list(getattr(action, "option_strings", []) or [])
    if not opts:
        return None

    long_opts = [opt for opt in opts if opt.startswith("--")]
    short_opts = [opt for opt in opts if opt.startswith("-") and not opt.startswith("--")]

    if prefix.startswith("--"):
        matches = [opt for opt in long_opts if opt.startswith(prefix)]
        if matches:
            return matches[0]
        matches = [opt for opt in opts if opt.startswith(prefix)]
        if matches:
            return matches[0]
        return long_opts[0] if long_opts else opts[0]

    if prefix.startswith("-"):
        matches = [opt for opt in short_opts if opt.startswith(prefix)]
        if matches:
            return matches[0]
        matches = [opt for opt in long_opts if opt.startswith(prefix)]
        if matches:
            return matches[0]
        if prefix == "-" and short_opts:
            return short_opts[0]
        matches = [opt for opt in opts if opt.startswith(prefix)]
        if matches:
            return matches[0]

    return long_opts[0] if long_opts else opts[0]


def _iter_visible_option_items(
    parser: argparse.ArgumentParser,
    prefix: str = "",
) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        help_text = getattr(action, "help", None)
        if help_text == argparse.SUPPRESS:
            continue
        opts = list(getattr(action, "option_strings", []) or [])
        if not opts:
            continue
        if prefix.startswith("-") and not any(opt.startswith(prefix) for opt in opts):
            continue
        selected = _select_option_value(action, prefix)
        if selected is None:
            continue
        alias_label = _option_alias_label(action)
        desc = alias_label if not help_text else f"{alias_label}: {help_text}"
        items.append(CompletionItem(selected, desc))
    return items


def _iter_subcommand_items(parser: argparse.ArgumentParser) -> list[CompletionItem]:
    sub = _subparser_action(parser)
    if sub is None:
        return []
    help_by_name, parser_to_name = _subparser_help_map(sub)
    items: list[CompletionItem] = []
    seen: set[str] = set()
    for name, subparser in (sub.choices or {}).items():
        if name in seen:
            continue
        seen.add(name)
        canonical = parser_to_name.get(id(subparser), str(name))
        detail = help_by_name.get(canonical) or _single_line(
            getattr(subparser, "description", None)
        )
        if str(name) == canonical:
            display = str(name) if not detail else f"{name}: {detail}"
        else:
            alias_text = f"{name} (alias: {canonical})"
            display = alias_text if not detail else f"{alias_text}: {detail}"
        items.append(CompletionItem(str(name), display))
    return items


def _filter_items(items: Iterable[CompletionItem], prefix: str) -> list[CompletionItem]:
    p = str(prefix or "")
    return [item for item in items if item.value.startswith(p)]


def _complete_paths(
    prefix: str,
    *,
    cwd: Path,
    suffixes: Sequence[str] | None = None,
    directories: bool = True,
    files: bool = True,
) -> list[CompletionItem]:
    prefix = str(prefix or "")
    raw_prefix = os.path.expanduser(prefix)
    prefix_path = Path(raw_prefix)

    if prefix_path.is_absolute():
        base_dir = prefix_path.parent if prefix_path.name else prefix_path
        name_prefix = prefix_path.name
        render_parent = prefix_path.parent
        absolute = True
    else:
        relative_parent = prefix_path.parent if prefix_path.parent != Path(".") else Path("")
        base_dir = (cwd / relative_parent).resolve()
        name_prefix = prefix_path.name
        render_parent = relative_parent
        absolute = False

    if not base_dir.exists():
        return []

    suffixes_norm = tuple(s.lower() for s in (suffixes or ()))
    items: list[CompletionItem] = []
    for entry in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.startswith(name_prefix):
            continue
        if entry.is_dir():
            if not directories:
                continue
            if absolute:
                value = str(entry) + os.sep
            else:
                value = str(render_parent / entry.name) if str(render_parent) else entry.name
                value += os.sep
            items.append(CompletionItem(value, None))
            continue
        if not files:
            continue
        if suffixes_norm and entry.suffix.lower() not in suffixes_norm:
            continue
        if absolute:
            value = str(entry)
        else:
            value = str(render_parent / entry.name) if str(render_parent) else entry.name
        items.append(CompletionItem(value, None))
    return items


def _complete_paths_from_roots(
    prefix: str,
    *,
    roots: Sequence[Path],
    suffixes: Sequence[str] | None = None,
    directories: bool = True,
    files: bool = True,
) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    for root in roots:
        items.extend(
            _complete_paths(
                prefix,
                cwd=root,
                suffixes=suffixes,
                directories=directories,
                files=files,
            )
        )
    return _dedupe_items(items)


def _complete_db_paths(prefix: str, *, cwd: Path) -> list[CompletionItem]:
    text = str(prefix or "")
    ends_with_sep = text.endswith("/") or text.endswith(os.sep)
    if ends_with_sep:
        raw_prefix = os.path.expanduser(text.rstrip("/"))
        prefix_path = Path(raw_prefix)
        if prefix_path.is_absolute():
            base_dir = prefix_path
            render_parent = prefix_path
            absolute = True
        else:
            render_parent = prefix_path
            base_dir = (cwd / prefix_path).resolve()
            absolute = False
        if not base_dir.exists():
            return []
        items: list[CompletionItem] = []
        for entry in sorted(base_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
                continue
            value = str(entry) if absolute else str(render_parent / entry.name)
            items.append(CompletionItem(value, None))
        return items
    return _complete_paths(
        prefix,
        cwd=cwd,
        suffixes=(".db", ".sqlite", ".sqlite3"),
        directories=not ends_with_sep,
        files=True,
    )


def _metric_name_cache_dir() -> Path:
    root = os.environ.get("ASTROLABE_COMPLETION_CACHE_DIR") or os.environ.get(
        "GH_COMPLETION_CACHE_DIR"
    )
    if root:
        return Path(root).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "astrolabe" / "completion"
    return Path.home().resolve() / ".cache" / "astrolabe" / "completion"


def _metric_name_cache_identity(db_path: Path) -> tuple[int, int]:
    st = db_path.stat()
    return int(st.st_dev), int(st.st_ino)


def _metric_name_cache_file(
    db_path: Path,
    *,
    identity: tuple[int, int] | None = None,
) -> Path:
    if identity is not None:
        raw = f"{identity[0]}:{identity[1]}"
    else:
        raw = str(db_path)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return _metric_name_cache_dir() / f"metric-names-{digest}.json"


def _metric_name_cache_candidates(
    db_path: Path,
    *,
    identity: tuple[int, int] | None,
) -> list[Path]:
    items: list[Path] = []
    if identity is not None:
        items.append(_metric_name_cache_file(db_path, identity=identity))
    legacy = _metric_name_cache_file(db_path)
    if legacy not in items:
        items.append(legacy)
    return items


def _metric_name_cache_signature(db_path: Path) -> tuple[int, int]:
    st = db_path.stat()
    return int(st.st_mtime_ns), int(st.st_size)


def _read_metric_name_cache(
    db_path: Path,
    *,
    identity: tuple[int, int] | None,
    signature: tuple[int, int],
) -> list[str] | None:
    now = time.time()
    for cache_path in _metric_name_cache_candidates(db_path, identity=identity):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if int(payload.get("version", -1)) != _METRIC_NAME_CACHE_VERSION:
            continue
        names = payload.get("names")
        if not isinstance(names, list):
            continue
        payload_identity = None
        if "db_dev" in payload and "db_ino" in payload:
            try:
                payload_identity = (int(payload["db_dev"]), int(payload["db_ino"]))
            except (TypeError, ValueError):
                payload_identity = None
        if identity is not None and payload_identity == identity:
            cached_at = float(payload.get("cached_at", 0.0) or 0.0)
            if now - cached_at <= _METRIC_NAME_CACHE_TTL_SECONDS:
                return sorted(str(name) for name in names)
            if (
                int(payload.get("mtime_ns", -1)) == signature[0]
                and int(payload.get("size", -1)) == signature[1]
            ):
                return sorted(str(name) for name in names)
        if (
            str(payload.get("db_path")) == str(db_path)
            and int(payload.get("mtime_ns", -1)) == signature[0]
            and int(payload.get("size", -1)) == signature[1]
        ):
            return sorted(str(name) for name in names)
    return None


def _write_metric_name_cache(
    db_path: Path,
    *,
    identity: tuple[int, int] | None,
    signature: tuple[int, int],
    names: Sequence[str],
) -> None:
    payload = {
        "version": _METRIC_NAME_CACHE_VERSION,
        "db_path": str(db_path),
        "mtime_ns": int(signature[0]),
        "size": int(signature[1]),
        "cached_at": float(time.time()),
        "names": list(names),
    }
    if identity is not None:
        payload["db_dev"] = int(identity[0])
        payload["db_ino"] = int(identity[1])
    cache_path = _metric_name_cache_file(db_path, identity=identity)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        return


def _scan_metric_names_from_db(db_path: str | Path) -> list[str]:
    from astrolabe.posthoc.db import iter_records, open_db

    conn = open_db(db_path)
    try:
        names: set[str] = set(_FIXED_FILTER_COLUMNS)
        for row in iter_records(conn):
            try:
                metrics = json.loads(row["metrics_json"])
            except Exception:
                metrics = {}
            if isinstance(metrics, dict):
                names.update(str(k) for k in metrics.keys())
        return sorted(names)
    finally:
        conn.close()


def _load_metric_names_from_db(db_path: str) -> list[str]:
    resolved = Path(db_path).expanduser().resolve()
    try:
        identity = _metric_name_cache_identity(resolved)
    except OSError:
        identity = None
    try:
        signature = _metric_name_cache_signature(resolved)
    except OSError:
        return _scan_metric_names_from_db(resolved)

    now = time.time()
    if identity is not None:
        cached = _METRIC_NAME_CACHE.get(f"ino:{identity[0]}:{identity[1]}")
        if cached is not None and cached[0] >= now:
            return list(cached[1])
    signature_key = f"sig:{resolved}:{signature[0]}:{signature[1]}"
    cached = _METRIC_NAME_CACHE.get(signature_key)
    if cached is not None:
        return list(cached[1])

    cached_names = _read_metric_name_cache(resolved, identity=identity, signature=signature)
    if cached_names is not None:
        _METRIC_NAME_CACHE[signature_key] = (float("inf"), list(cached_names))
        if identity is not None:
            _METRIC_NAME_CACHE[f"ino:{identity[0]}:{identity[1]}"] = (
                now + _METRIC_NAME_CACHE_TTL_SECONDS,
                list(cached_names),
            )
        return list(cached_names)

    names = _scan_metric_names_from_db(resolved)
    _METRIC_NAME_CACHE[signature_key] = (float("inf"), list(names))
    if identity is not None:
        _METRIC_NAME_CACHE[f"ino:{identity[0]}:{identity[1]}"] = (
            now + _METRIC_NAME_CACHE_TTL_SECONDS,
            list(names),
        )
    _write_metric_name_cache(resolved, identity=identity, signature=signature, names=names)
    return list(names)


def _extract_expression_tail(prefix: str) -> tuple[str, str]:
    raw = str(prefix or "")
    idx = len(raw)
    while idx > 0 and (raw[idx - 1].isalnum() or raw[idx - 1] in {"_", "."}):
        idx -= 1
    return raw[:idx], raw[idx:]


_FILTER_EXPR_OPERATOR_TOKENS: tuple[CompletionItem, ...] = (
    CompletionItem("+", "+"),
    CompletionItem("-", "-"),
    CompletionItem("*", "*"),
    CompletionItem("/", "/"),
    CompletionItem("%", "%"),
    CompletionItem(">", ">"),
    CompletionItem(">=", ">="),
    CompletionItem("<", "<"),
    CompletionItem("<=", "<="),
    CompletionItem("==", "=="),
    CompletionItem("!=", "!="),
    CompletionItem("and", "and"),
    CompletionItem("or", "or"),
    CompletionItem("not", "not"),
)

_FILTER_EXPR_FUNCTION_TOKENS: tuple[CompletionItem, ...] = (
    CompletionItem("missing(", "missing("),
    CompletionItem("exists(", "exists("),
    CompletionItem("abs(", "abs("),
    CompletionItem("min(", "min("),
    CompletionItem("max(", "max("),
    CompletionItem("round(", "round("),
)


_OPERAND_SLOT_SUFFIX_RE = re.compile(
    r"(?:\(|,|>=|<=|==|!=|>|<|\+|-|\*|/|%|\band\b|\bor\b|\bnot\b)\s*$"
)
_LEADING_MISSING_OPERAND_RE = re.compile(
    r"^(?P<leading>\s*)(?P<rest>(?:>=|<=|==|!=|>|<|\+|-|\*|/|%|\band\b|\bor\b).*)$"
)


def _prefers_operand_candidates(prefix: str) -> bool:
    raw = str(prefix or "")
    if not raw.strip():
        return True
    return _OPERAND_SLOT_SUFFIX_RE.search(raw) is not None


def _join_filter_expr_candidate(prefix: str, candidate: str, suffix: str) -> str:
    left = str(prefix or "")
    right = str(suffix or "").lstrip()
    if not right:
        return left + candidate
    if right[0] in {")", "]", "}", ","}:
        return left + candidate + right
    return left + candidate + " " + right


def _rewrite_filter_expr_leading_gap(
    prefix: str,
    *,
    metric_names: Sequence[str],
) -> list[CompletionItem]:
    match = _LEADING_MISSING_OPERAND_RE.match(str(prefix or ""))
    if match is None:
        return []
    leading = str(match.group("leading") or "")
    rest = str(match.group("rest") or "")
    if not rest:
        return []
    items: list[CompletionItem] = []
    for item in _FILTER_EXPR_FUNCTION_TOKENS:
        items.append(
            CompletionItem(
                _join_filter_expr_candidate(leading, item.value, rest),
                item.description,
            )
        )
    for metric_name in metric_names:
        items.append(
            CompletionItem(
                _join_filter_expr_candidate(leading, metric_name, rest),
                metric_name,
            )
        )
    return items


def _complete_filter_expr(prefix: str, *, db_path: str | None) -> list[CompletionItem]:
    metrics: list[str] = []
    if db_path:
        try:
            metrics = _load_metric_names_from_db(db_path)
        except (sqlite3.Error, OSError):
            metrics = []

    expr_head, tail = _extract_expression_tail(prefix)
    metric_items = [
        CompletionItem(expr_head + metric, metric)
        for metric in metrics
        if metric.startswith(tail)
    ]
    function_items = [
        CompletionItem(expr_head + item.value, item.description)
        for item in _FILTER_EXPR_FUNCTION_TOKENS
        if item.value.startswith(tail)
    ]
    operator_items = [
        CompletionItem(expr_head + item.value, item.description)
        for item in _FILTER_EXPR_OPERATOR_TOKENS
        if item.value.startswith(tail)
    ]
    items: list[CompletionItem] = []
    if _prefers_operand_candidates(prefix):
        items.extend(function_items)
        items.extend(metric_items)
        items.extend(operator_items)
    else:
        items.extend(operator_items)
        items.extend(function_items)
        items.extend(metric_items)
    if items:
        return items
    return _rewrite_filter_expr_leading_gap(prefix, metric_names=metrics)


def _task_completion_items(config_dir: str | None, prefix: str) -> list[CompletionItem]:
    import astrolabe.cli as cli

    cfg_dir = Path(config_dir).expanduser().resolve() if config_dir else cli._default_config_dir()
    return _filter_items(
        [CompletionItem(task, f"{task}: task template") for task in cli._list_available_tasks(cfg_dir)],
        prefix,
    )


def _seen_values(state: _ParseState, dest: str) -> list[str]:
    if state.all_seen is not None and dest in state.all_seen:
        return state.all_seen[dest]
    return state.seen.get(dest, [])


def _option_present(state: _ParseState, dest: str) -> bool:
    if dest in state.present:
        return True
    return state.all_present is not None and dest in state.all_present


def _looks_like_filter_preset_file(path: Path) -> bool:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _FILTER_PRESET_RULES_RE.search(text) is not None


def _path_display_value(path: Path, *, cwd: Path) -> str:
    resolved_cwd = cwd.resolve()
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(resolved_cwd):
        return str(resolved_path.relative_to(resolved_cwd))
    return str(resolved_path)


def _preview_preset_completion_items(
    prefix: str,
    *,
    cwd: Path,
    db_path: str | None,
) -> list[CompletionItem]:
    roots: list[Path] = [cwd.resolve()]
    if db_path:
        db_parent = Path(db_path).expanduser().resolve().parent
        if db_parent not in roots:
            roots.append(db_parent)
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root not in roots:
        roots.append(repo_root)

    text = str(prefix or "")
    if (
        text.startswith("/")
        or text.startswith(".")
        or text.startswith("~")
        or "/" in text
        or os.sep in text
    ):
        return _complete_paths_from_roots(
            text,
            roots=roots,
            suffixes=(".yaml", ".yml"),
        )

    candidate_dirs: list[Path] = []
    for root in roots:
        if root.exists():
            candidate_dirs.append(root)
        for name in _PRESET_SEARCH_DIR_NAMES:
            child = root / name
            if child.exists() and child.is_dir():
                candidate_dirs.append(child)

    items: list[CompletionItem] = []
    seen_values: set[str] = set()
    for directory in candidate_dirs:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if not _looks_like_filter_preset_file(entry):
                continue
            value = _path_display_value(entry, cwd=cwd)
            if value in seen_values:
                continue
            seen_values.add(value)
            items.append(CompletionItem(value, value))
    return _filter_items(items, text)


def _static_action_value_items(action: argparse.Action) -> list[CompletionItem]:
    completion_choices = getattr(action, "completion_choices", None)
    if completion_choices:
        return [CompletionItem(str(choice), str(choice)) for choice in completion_choices]
    if getattr(action, "choices", None):
        return [CompletionItem(str(choice), str(choice)) for choice in action.choices]
    return []


def _context_key(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "__root__"


def _shell_here_doc(text: str) -> str:
    if not text:
        return "cat <<'EOF'\nEOF"
    return f"cat <<'EOF'\n{text}\nEOF"


def _render_items_text(items: Sequence[CompletionItem]) -> str:
    return render_items_for_shell(items)


def _collect_static_completion_spec() -> dict[str, object]:
    import astrolabe.cli as cli

    parser = cli._build_parser()
    spec: dict[str, object] = {
        "subcommands": {},
        "options_default": {},
        "options_short": {},
        "options_long": {},
        "next_context": {},
        "option_arity": {},
        "static_values": {},
    }
    visited: set[str] = set()

    def visit(
        parser_obj: argparse.ArgumentParser,
        path: tuple[str, ...],
        parsers: tuple[argparse.ArgumentParser, ...],
    ) -> None:
        context = _context_key(path)
        if context in visited:
            return
        visited.add(context)

        spec["subcommands"][context] = _render_items_text(_iter_subcommand_items(parser_obj))
        if any(not action.option_strings and not isinstance(action, argparse._SubParsersAction)
               and getattr(action, "choices", None) for action in parser_obj._actions):
            spec.setdefault("positional_contexts", {})[context] = "1"
        spec["options_default"][context] = _render_items_text(
            _iter_visible_option_items(parser_obj, "")
        )
        option_items_short: list[CompletionItem] = []
        option_items_long: list[CompletionItem] = []
        arity_keys: set[str] = set()
        static_value_keys: set[str] = set()
        for option_parser in reversed(parsers):
            option_items_short.extend(_iter_visible_option_items(option_parser, "-"))
            option_items_long.extend(_iter_visible_option_items(option_parser, "--"))
            for action in option_parser._actions:
                opts = list(getattr(action, "option_strings", []) or [])
                if not opts:
                    continue
                arity = _option_value_arity(action)
                static_values = _render_items_text(_static_action_value_items(action))
                for opt in opts:
                    key = f"{context}:{opt}"
                    if key not in arity_keys:
                        spec["option_arity"][key] = str(arity)
                        arity_keys.add(key)
                    if static_values and key not in static_value_keys:
                        spec["static_values"][key] = static_values
                        static_value_keys.add(key)
        spec["options_short"][context] = _render_items_text(_dedupe_items(option_items_short))
        spec["options_long"][context] = _render_items_text(_dedupe_items(option_items_long))

        sub = _subparser_action(parser_obj)
        _, parser_to_name = _subparser_help_map(sub)

        if sub is None:
            return
        for name, subparser in (sub.choices or {}).items():
            canonical = parser_to_name.get(id(subparser), str(name))
            next_context = _context_key(tuple([*path, canonical]))
            spec["next_context"][f"{context}:{name}"] = next_context
        for canonical in parser_to_name.values():
            child = (sub.choices or {}).get(canonical)
            if child is None:
                continue
            visit(child, tuple([*path, canonical]), tuple([*parsers, child]))

    visit(parser, (), (parser,))
    return spec


def _render_zsh_static_dispatch() -> str:
    spec = _collect_static_completion_spec()

    def emit_case_fn(name: str, entries: dict[str, str], *, key2: bool = False) -> list[str]:
        lines = [f"{name}() {{", '  case "$1' + (':$2' if key2 else '') + '" in']
        for key, value in entries.items():
            pattern = shlex.quote(str(key))
            lines.append(f"    ({pattern})")
            lines.append(f"      {_shell_here_doc(value)}")
            lines.append("      ;;")
        lines.append("    (*) return 1 ;;")
        lines.append("  esac")
        lines.append("}")
        return lines

    lines: list[str] = []
    lines.extend(emit_case_fn("_astrolabe_static_subcommands", spec["subcommands"]))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_options_default", spec["options_default"]))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_options_short", spec["options_short"]))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_options_long", spec["options_long"]))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_next_context", spec["next_context"], key2=True))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_option_arity", spec["option_arity"], key2=True))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_values", spec["static_values"], key2=True))  # type: ignore[arg-type]
    lines.append("")
    lines.extend(emit_case_fn("_astrolabe_static_has_positionals", spec.get("positional_contexts", {})))
    lines.append("")
    lines.extend(
        [
            "_astrolabe_filter_completion_lines() {",
            '  local prefix="$1"',
            "  local value desc",
            "  while IFS=$'\\t' read -r value desc; do",
            '    [[ -z "$value" ]] && continue',
            '    [[ -n "$prefix" && "$value" != ${prefix}* ]] && continue',
            '    print -r -- "$value"$\'\\t\'"${desc:-$value}"',
            "  done",
            "}",
            "",
            "_astrolabe_static_complete() {",
            '  local cword="$1"',
            '  local current_prefix="$2"',
            "  shift 2",
            "  local -a query_words",
            '  query_words=("$@")',
            '  local context="__root__"',
            '  local pending=""',
            "  local remaining=0",
            "  local idx token arity next_ctx static_values sub_lines opt_lines",
            "  idx=1",
            "  while (( idx <= cword )); do",
            '    token="${query_words[idx]}"',
            '    if [[ -n "$pending" && $remaining -gt 0 ]]; then',
            "      (( remaining -= 1 ))",
            '      if (( remaining == 0 )); then pending=""; fi',
            "      (( idx += 1 ))",
            "      continue",
            "    fi",
            '    if [[ "$token" == "--" ]]; then',
            "      break",
            "    fi",
            '    if [[ "$token" == -* ]]; then',
            '      arity="$(_astrolabe_static_option_arity "$context" "$token" 2>/dev/null || true)"',
            '      if [[ -n "$arity" && "$arity" != "0" ]]; then',
            '        pending="$token"',
            '        remaining="$arity"',
            "      fi",
            "      (( idx += 1 ))",
            "      continue",
            "    fi",
            '    next_ctx="$(_astrolabe_static_next_context "$context" "$token" 2>/dev/null || true)"',
            '    if [[ -n "$next_ctx" ]]; then',
            '      context="$next_ctx"',
            "    fi",
            "    (( idx += 1 ))",
            "  done",
            "",
            '  if [[ -n "$pending" && $remaining -gt 0 ]]; then',
            '    static_values="$(_astrolabe_static_values "$context" "$pending" 2>/dev/null || true)"',
            '    if [[ -n "$static_values" ]]; then',
            '      print -r -- "$static_values" | _astrolabe_filter_completion_lines "$current_prefix"',
            "      return 0",
            "    fi",
            "    return 1",
            "  fi",
            "",
            '  if [[ "$current_prefix" == -* ]]; then',
            '    if [[ "$current_prefix" == --* ]]; then',
            '      opt_lines="$(_astrolabe_static_options_long "$context" 2>/dev/null || true)"',
            "    else",
            '      opt_lines="$(_astrolabe_static_options_short "$context" 2>/dev/null || true)"',
            "    fi",
            '    [[ -z "$opt_lines" ]] && return 1',
            '    print -r -- "$opt_lines" | _astrolabe_filter_completion_lines "$current_prefix"',
            "    return 0",
            "  fi",
            "",
            '  if _astrolabe_static_has_positionals "$context" >/dev/null 2>&1; then return 1; fi',
            '  sub_lines="$(_astrolabe_static_subcommands "$context" 2>/dev/null || true)"',
            '  opt_lines="$(_astrolabe_static_options_default "$context" 2>/dev/null || true)"',
            '  if [[ -n "$sub_lines" ]]; then',
            '    print -r -- "$sub_lines" | _astrolabe_filter_completion_lines "$current_prefix"',
            "  fi",
            '  if [[ -n "$opt_lines" && ( -z "$sub_lines" || -z "$current_prefix" ) ]]; then',
            '    print -r -- "$opt_lines" | _astrolabe_filter_completion_lines "$current_prefix"',
            "  fi",
            '  [[ -n "$sub_lines" || -n "$opt_lines" ]]',
            "}",
        ]
    )
    return "\n".join(lines)


def _complete_action_value(
    *,
    action: argparse.Action,
    state: _ParseState,
    prefix: str,
    cwd: Path,
) -> list[CompletionItem]:
    dest = getattr(action, "dest", "")
    option_names = set(getattr(action, "option_strings", []) or [])
    completion_choices = getattr(action, "completion_choices", None)
    if completion_choices:
        return _filter_items(
            [CompletionItem(str(choice), None) for choice in completion_choices],
            prefix,
        )
    if getattr(action, "choices", None):
        return _filter_items(
            [CompletionItem(str(choice), None) for choice in action.choices],
            prefix,
        )

    if dest == "task" and state.path in {
        ("run",),
        ("config", "wizard"),
        ("config", "scaffold"),
        ("config", "compose"),
        ("env", "wizard"),
        ("env", "doctor"),
    }:
        cfg_dir = (_seen_values(state, "config_dir") or [None])[-1]
        return _task_completion_items(cfg_dir, prefix)

    if dest in {"config_dir", "out_dir", "data_dir", "tool_dir"} or "--config-dir" in option_names or "--out-dir" in option_names:
        return _complete_paths(prefix, cwd=cwd)

    if dest == "config_env" and state.path[:1] == ("paper",):
        import astrolabe.cli as cli
        config_root = Path((_seen_values(state, "config_dir") or [cli._default_config_dir()])[-1])
        return _filter_items([CompletionItem(path.stem) for path in sorted((config_root / "env").glob("*.yaml"))], prefix)

    if dest in {"db"}:
        return _complete_db_paths(prefix, cwd=cwd)

    if dest in {
        "preset",
        "required_metrics_preset",
        "overlay",
        "standalone",
        "env",
        "out",
        "overlay_out",
        "standalone_out",
        "export_csv",
        "report_csv",
        "inputs",
        "planned",
        "records",
        "overrides_file",
    }:
        roots = [cwd]
        db_path = (_seen_values(state, "db") or [None])[-1]
        if dest == "preset" and _option_present(state, "preview_preset"):
            return _preview_preset_completion_items(prefix, cwd=cwd, db_path=db_path)
        if dest in {"preset", "required_metrics_preset"} and db_path:
            db_parent = Path(db_path).expanduser().resolve().parent
            if db_parent not in roots:
                roots.append(db_parent)
        return _complete_paths_from_roots(
            prefix,
            roots=roots,
            suffixes=(".yaml", ".yml", ".csv"),
        )

    if dest in {"out_root", "dst", "out_manifest", "manifest"}:
        return _complete_paths(prefix, cwd=cwd)

    if dest in {"structure", "framework_pdb", "target_structure", "parent_structure", "reference", "spheres"}:
        return _complete_paths(prefix, cwd=cwd, suffixes=(".pdb", ".cif", ".gz"))

    if dest in {"target_homology", "framework_homology", "protein_homology", "homology", "msa"}:
        return _complete_paths(prefix, cwd=cwd, suffixes=(".json", ".a3m", ".yaml", ".yml"))

    if dest in {"mutable_positions"}:
        return _complete_paths(prefix, cwd=cwd, suffixes=(".txt", ".tsv", ".csv"))

    if dest in {"ligand_conformer"}:
        return _complete_paths(
            prefix,
            cwd=cwd,
            suffixes=(".sdf", ".mol", ".mol2", ".pdb", ".cif", ".gz"),
        )

    if dest == "filter_expr" and state.path == ("posthoc", "filter"):
        db_path = (_seen_values(state, "db") or [None])[-1]
        return _complete_filter_expr(prefix, db_path=db_path)

    return []


def _record_consumed_value(state: _ParseState, action: argparse.Action, value: str) -> None:
    state.seen.setdefault(action.dest, []).append(value)


def _consume_words(words: Sequence[str], cword: int) -> _ParseState:
    import astrolabe.cli as cli

    parser = cli._build_parser()
    state = _ParseState(parser=parser, parsers=(parser,), path=(), seen={}, present=set())
    idx = 0
    while idx < min(cword, len(words)):
        token = words[idx]

        if state.pending_action is not None and state.pending_remaining > 0:
            _record_consumed_value(state, state.pending_action, token)
            state.pending_remaining -= 1
            if state.pending_remaining == 0:
                state.pending_action = None
            idx += 1
            continue

        if token == "--":
            break

        if token.startswith("-"):
            action = _find_option_action_in_parsers(state.parsers, token)
            if action is None:
                idx += 1
                continue
            state.present.add(str(action.dest))
            arity = _option_value_arity(action)
            if arity > 0:
                state.pending_action = action
                state.pending_remaining = arity
            idx += 1
            continue

        sub = _subparser_action(state.parser)
        if sub is not None and token in (sub.choices or {}):
            state.parser = sub.choices[token]
            _, parser_to_name = _subparser_help_map(sub)
            canonical = parser_to_name.get(id(state.parser), token)
            state.parsers = tuple([*state.parsers, state.parser])
            state.path = tuple([*state.path, canonical])
            idx += 1
            continue

        positional = _next_positional_action(state)
        if positional is not None:
            _record_consumed_value(state, positional, token)

        idx += 1
    return state


def complete_words(
    words: Sequence[str],
    cword: int,
    *,
    cwd: Path | None = None,
) -> list[CompletionItem]:
    words = list(words)
    cwd = (cwd or Path.cwd()).resolve()
    if cword < 0:
        cword = 0
    prefix = words[cword] if cword < len(words) else ""
    state = _consume_words(words, cword)
    full_state = _consume_words(words, len(words))
    state.all_seen = full_state.seen
    state.all_present = full_state.present

    if state.pending_action is not None and state.pending_remaining > 0:
        items = _complete_action_value(
            action=state.pending_action,
            state=state,
            prefix=prefix,
            cwd=cwd,
        )
        return _dedupe_items(items)

    sub_items = _iter_subcommand_items(state.parser)
    option_items = _iter_visible_option_items(state.parser, prefix)

    if prefix.startswith("-"):
        return _dedupe_items(option_items)

    items: list[CompletionItem] = []
    if sub_items:
        items.extend(_filter_items(sub_items, prefix))
    positional = _next_positional_action(state)
    if positional is not None:
        items.extend(_complete_action_value(action=positional, state=state, prefix=prefix, cwd=cwd))
    items.extend(_filter_items(option_items, prefix))
    return _dedupe_items(items)


def _dedupe_items(items: Iterable[CompletionItem]) -> list[CompletionItem]:
    out: list[CompletionItem] = []
    seen: set[str] = set()
    for item in items:
        if item.value in seen:
            continue
        seen.add(item.value)
        out.append(item)
    return out


def render_bash_completion(*, init: bool = False) -> str:
    init_block = ""
    if init:
        init_block = """# enable programmable completion when sourcing in a fresh bash session
shopt -s progcomp 2>/dev/null || true

"""
    return f"""# bash completion for astrolabe
{init_block}_astrolabe_completion() {{
  local cword=$((COMP_CWORD - 1))
  local out
  out="$("${{COMP_WORDS[0]}}" completion complete --shell bash --cword "$cword" -- "${{COMP_WORDS[@]:1}}" 2>/dev/null)" || return 0
  COMPREPLY=()
  while IFS=$'\\t' read -r value _desc; do
    [[ -z "$value" ]] && continue
    COMPREPLY+=("$value")
  done <<< "$out"
}}

complete -o nosort -F _astrolabe_completion astrolabe
"""


def render_zsh_completion(*, init: bool = False) -> str:
    init_block = ""
    if init:
        init_block = """if ! (( $+functions[compdef] )); then
  autoload -Uz compinit
  compinit -i
fi

"""
    static_block = _render_zsh_static_dispatch()
    return f"""#compdef astrolabe
{init_block}_astrolabe_completion() {{
  local cword
  local word_index
  if (( CURRENT <= 1 )); then
    cword=0
    word_index=0
  else
    cword=$((CURRENT - 2))
    word_index=$((CURRENT - 1))
  fi
  local current_prefix
  current_prefix="${{IPREFIX-}}${{PREFIX-}}"
  local -a query_words
  query_words=("${{words[@]:1}}")
  if (( word_index > 0 )); then
    query_words[$word_index]="$current_prefix"
  fi
  local out
  out="$(_astrolabe_static_complete "$cword" "$current_prefix" "${{query_words[@]}}" 2>/dev/null)" || out=""
  if [[ -z "$out" ]]; then
    out="$("${{words[1]}}" completion complete --shell zsh --cword "$cword" -- "${{query_words[@]}}" 2>/dev/null)" || return 0
  fi
  local -a values
  local -a displays
  local line value desc insert_value iprefix_len iprefix_raw
  iprefix_raw="${{IPREFIX-}}"
  iprefix_len=${{#iprefix_raw}}
  while IFS=$'\\t' read -r value desc; do
    [[ -z "$value" ]] && continue
    insert_value="$value"
    if (( iprefix_len > 0 && ${{#insert_value}} >= iprefix_len )) && [[ "${{insert_value[1,iprefix_len]}}" == "$iprefix_raw" ]]; then
      insert_value="${{insert_value[iprefix_len+1,-1]}}"
    fi
    values+=("$insert_value")
    displays+=("${{desc:-$value}}")
  done <<< "$out"
  if (( ${{#values[@]}} == 0 )); then
    return 0
  fi
  compadd -Q -U -d displays -- "${{values[@]}}"
}}

{static_block}

compdef _astrolabe_completion astrolabe
compdef -p _astrolabe_completion '*/astrolabe'
"""


def render_items_for_shell(items: Sequence[CompletionItem]) -> str:
    lines = []
    for item in items:
        desc = item.description or ""
        lines.append(f"{item.value}\t{desc}")
    return "\n".join(lines)


def shell_words_from_remainder(remainder: Sequence[str]) -> list[str]:
    if remainder and remainder[0] == "--":
        return list(remainder[1:])
    return list(remainder)
