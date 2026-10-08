from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from omegaconf import OmegaConf

from astrolabe.runtime_requirements import task_requirements
from astrolabe.runtime_usage import analyze_runtime_usage


_ENV_PRUNE_SECTION_KEYS = {
    "autodock",
    "carbondesign",
    "hmmer",
    "protenix",
    "rfd3",
    "rhofold",
    "subprocess_envs",
    "weight_path",
}
_ENV_PRUNE_SCALAR_KEYS = {
    "dssp_path",
    "molfile_to_params",
    "obabel_path",
    "usalign_path",
}
_REFOLD_REPLAY_EVALUATOR_TYPES = frozenset({"alphafold3", "protenix"})
_REFOLD_REPLAY_ENV_KEYS = {
    "alphafold3": {"weight_path.alphafold3"},
    "protenix": {
        "weight_path.protenix",
        "protenix.ccd_components_file",
        "protenix.ccd_components_rdkit_mol_file",
    },
}
_FALSE_VALUES = {False, 0, "false", "False", "FALSE", "0", "no", "No", "NO"}
_TRUE_VALUES = {True, 1, "true", "True", "TRUE", "1", "yes", "Yes", "YES"}


def _collect_refold_replay_evaluator_types(raw: dict[str, Any]) -> set[str]:
    evaluator_cfg = raw.get("evaluator")
    if not isinstance(evaluator_cfg, dict):
        return set()
    return {
        backend
        for backend in _REFOLD_REPLAY_EVALUATOR_TYPES
        if isinstance(evaluator_cfg.get(backend), dict)
    }


def _collect_enabled_predocker_aliases(raw: dict[str, Any]) -> list[str]:
    return analyze_runtime_usage(raw).used_predocker_aliases


def _prune_mapping_to_required_paths(
    node: dict[str, Any],
    paths: list[tuple[str, ...]],
) -> dict[str, Any]:
    if not isinstance(node, dict):
        return node
    if not paths:
        return {}

    grouped: dict[str, list[tuple[str, ...]]] = {}
    for path in paths:
        if not path:
            return dict(node)
        grouped.setdefault(path[0], []).append(path[1:])

    kept: dict[str, Any] = {}
    for key, child_paths in grouped.items():
        if key not in node:
            continue
        value = node[key]
        if any(len(child) == 0 for child in child_paths):
            kept[key] = value
            continue
        if isinstance(value, dict):
            pruned_child = _prune_mapping_to_required_paths(value, child_paths)
            if pruned_child:
                kept[key] = pruned_child
    return kept


def _prune_env_sections(
    raw: dict[str, Any],
    *,
    replay_evaluator_types: set[str] | None = None,
) -> None:
    required_env_keys = {
        str(key)
        for key in (task_requirements(raw).get("required_env_keys") or [])
        if str(key)
    }
    for backend in replay_evaluator_types or set():
        required_env_keys.update(_REFOLD_REPLAY_ENV_KEYS.get(str(backend), set()))

    for key in list(_ENV_PRUNE_SCALAR_KEYS):
        if key in raw and key not in required_env_keys:
            raw.pop(key, None)

    for section in list(_ENV_PRUNE_SECTION_KEYS):
        value = raw.get(section)
        if not isinstance(value, dict):
            continue
        section_paths = [
            tuple(part for part in req_key.split(".")[1:] if part)
            for req_key in required_env_keys
            if req_key.startswith(f"{section}.")
        ]
        if not section_paths:
            raw.pop(section, None)
            continue
        pruned = _prune_mapping_to_required_paths(value, section_paths)
        if pruned:
            raw[section] = pruned
        else:
            raw.pop(section, None)


def _values_equivalent(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    return str(left) == str(right)


def _normalize_initializer(raw: dict[str, Any]) -> None:
    legacy_value = raw.get("reinitialize_random_segments_with_dplm")
    initializer_cfg = raw.get("initializer")
    if not isinstance(initializer_cfg, dict):
        initializer_cfg = {}
        raw["initializer"] = initializer_cfg

    if legacy_value in _TRUE_VALUES:
        raw_mode = initializer_cfg.get("mode")
        mode = str(raw_mode).strip().lower() if raw_mode is not None else ""
        if mode in {"", "random", "dplm"}:
            initializer_cfg["mode"] = "dplm"
            raw.pop("reinitialize_random_segments_with_dplm", None)
            return
        return

    if legacy_value in _FALSE_VALUES or legacy_value is None:
        raw.pop("reinitialize_random_segments_with_dplm", None)


def _cleanup_legacy_root_fields(raw: dict[str, Any]) -> None:
    _normalize_initializer(raw)

    if raw.get("use_metropolis_acceptance") in _FALSE_VALUES:
        raw.pop("initial_temperature", None)

    stages_cfg = raw.get("stages")
    default_stage = stages_cfg.get("default") if isinstance(stages_cfg, dict) else None
    if isinstance(default_stage, dict):
        root_mutation_num = raw.get("mutation_num")
        stage_mutation_num = default_stage.get("mutation_num")
        if _values_equivalent(root_mutation_num, stage_mutation_num):
            raw.pop("mutation_num", None)


def _collect_stage_expressions(stages_cfg: dict[str, Any] | None) -> list[str]:
    if not isinstance(stages_cfg, dict):
        return []
    expressions: list[str] = []
    default_cfg = stages_cfg.get("default")
    if isinstance(default_cfg, dict):
        for key in (
            "weighted_score_expression",
            "acceptance_gate",
            "acceptance_threshold",
            "early_stopping",
            "prefix_early_stopping",
        ):
            value = default_cfg.get(key)
            if isinstance(value, str) and value.strip():
                expressions.append(value)
    for stage_name, stage_cfg in stages_cfg.items():
        if str(stage_name) == "default" or not isinstance(stage_cfg, dict):
            continue
        for key in (
            "weighted_score_expression",
            "acceptance_gate",
            "acceptance_threshold",
            "early_stopping",
            "prefix_early_stopping",
            "_entrance",
        ):
            value = stage_cfg.get(key)
            if isinstance(value, str) and value.strip():
                expressions.append(value)
    return expressions


def _ordered_expression_terms(
    *,
    expression_terms: dict[str, Any],
    expressions: list[str],
) -> list[str]:
    from astrolabe.losses.base import extract_best_namespace_terms, extract_variables

    if not expression_terms:
        return []
    requested = set()
    for expr in expressions:
        requested.update(
            extract_variables(expr).intersection(expression_terms.keys())
        )
        requested.update(
            extract_best_namespace_terms(expr).intersection(expression_terms.keys())
        )
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(
                f"Cyclic expression term dependency detected at '{name}'"
            )
        visiting.add(name)
        deps = extract_variables(str(expression_terms[name])).intersection(
            expression_terms.keys()
        )
        for dep in expression_terms.keys():
            if dep in deps:
                visit(str(dep))
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in expression_terms.keys():
        if name in requested:
            visit(str(name))
    return ordered


def _prune_component_section(
    section: dict[str, Any] | None,
    *,
    terms_key: str,
    preserve_keys: set[str] | None = None,
) -> None:
    if not isinstance(section, dict):
        return
    preserve_keys = preserve_keys or set()
    enabled_aliases = [
        str(alias)
        for alias in (section.get("enabled_aliases") or [])
        if str(alias)
    ]
    raw_terms = section.get(terms_key) or {}
    kept_terms = {}
    actual_types: set[str] = set()
    if isinstance(raw_terms, dict):
        for alias in enabled_aliases:
            conf = raw_terms.get(alias)
            if conf is None:
                continue
            kept_terms[alias] = conf
            if isinstance(conf, dict):
                actual = str(conf.get("_type") or "").strip()
                if actual:
                    actual_types.add(actual)

    kept: dict[str, Any] = {}
    for key, value in section.items():
        if key == terms_key:
            kept[key] = kept_terms
            continue
        if key == "enabled_aliases":
            kept[key] = enabled_aliases
            continue
        if key == "selectors":
            if isinstance(value, dict):
                kept[key] = {
                    str(alias): selector
                    for alias, selector in value.items()
                    if str(alias) in enabled_aliases
                }
            else:
                kept[key] = value
            continue
        if key in preserve_keys:
            kept[key] = value
            continue
        if key in actual_types:
            kept[key] = value
            continue
        if not isinstance(value, dict):
            kept[key] = value
    section.clear()
    section.update(kept)


def _prune_loss_section(
    loss_cfg: dict[str, Any] | None,
    *,
    stages_cfg: dict[str, Any] | None,
) -> None:
    if not isinstance(loss_cfg, dict):
        return
    from astrolabe.losses.base import extract_loss_terms

    expressions = _collect_stage_expressions(stages_cfg)
    expression_terms = loss_cfg.get("expression_terms") or {}
    required_expression_terms: list[str] = []
    if isinstance(expression_terms, dict):
        required_expression_terms = _ordered_expression_terms(
            expression_terms=expression_terms,
            expressions=expressions,
        )
    included_loss_terms = set(
        extract_loss_terms(
            expressions=[
                *expressions,
                *[
                    str(expression_terms[name])
                    for name in required_expression_terms
                    if name in expression_terms
                ],
            ],
            excluded_terms=list(expression_terms.keys())
            if isinstance(expression_terms, dict)
            else None,
        )
    )

    kept: dict[str, Any] = {}
    for key, value in loss_cfg.items():
        if key == "loss_terms":
            if isinstance(value, dict):
                kept[key] = {
                    str(name): conf
                    for name, conf in value.items()
                    if str(name) in included_loss_terms
                }
            else:
                kept[key] = value
            continue
        if key == "expression_terms":
            if isinstance(value, dict):
                kept[key] = {
                    str(name): conf
                    for name, conf in value.items()
                    if str(name) in required_expression_terms
                }
            else:
                kept[key] = value
            continue
        if not isinstance(value, dict):
            kept[key] = value
    loss_cfg.clear()
    loss_cfg.update(kept)


def build_persisted_run_config(args: Any) -> Any:
    raw = OmegaConf.to_container(args, resolve=False)
    if not isinstance(raw, dict):
        return args
    _cleanup_legacy_root_fields(raw)
    replay_evaluator_types = _collect_refold_replay_evaluator_types(raw)
    evaluator_cfg = raw.get("evaluator")
    preserved_replay_cfg = {
        backend: evaluator_cfg.get(backend)
        for backend in replay_evaluator_types
        if isinstance(evaluator_cfg, dict) and evaluator_cfg.get(backend) is not None
    }
    _prune_component_section(
        raw.get("evaluator"),
        terms_key="evaluator_terms",
        preserve_keys={"scheduling"},
    )
    if isinstance(raw.get("evaluator"), dict):
        raw["evaluator"].update(preserved_replay_cfg)
    _prune_component_section(raw.get("mutator"), terms_key="mutator_terms")
    predocker_cfg = raw.get("predocker")
    if isinstance(predocker_cfg, dict):
        predocker_cfg["enabled_aliases"] = _collect_enabled_predocker_aliases(raw)
    _prune_component_section(predocker_cfg, terms_key="predocker_terms")
    _prune_loss_section(raw.get("loss"), stages_cfg=raw.get("stages"))
    _prune_env_sections(
        raw,
        replay_evaluator_types=replay_evaluator_types,
    )
    return OmegaConf.create(raw)


# ---------------------------------------------------------------------------
# Run provenance (`provenance.json`)
#
# One small machine-readable record per run directory, written next to the
# persisted `config.yaml`: what code ran, on which host/devices, with which
# persisted config (digest) and which explicit input files, plus the run state.
# Everything here is best-effort: provenance never changes pipeline behavior and
# never masks a scientific exception.
# ---------------------------------------------------------------------------

_PROVENANCE_FILENAME = "provenance.json"
_PROVENANCE_SCHEMA_VERSION = 1
_TERMINAL_PROVENANCE_STATES = frozenset({"completed", "failed", "interrupted"})
_DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "torch",
    "hydra-core",
    "omegaconf",
    "biopython",
)
_VISIBILITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)
_NVIDIA_SMI_QUERY = "--query-gpu=index,name,driver_version,memory.total,compute_cap"
_GIT_TIMEOUT_SECONDS = 30.0
_NVIDIA_SMI_TIMEOUT_SECONDS = 10.0
_HASH_CHUNK_BYTES = 1024 * 1024


def _utc_now_iso() -> str:
    """Current UTC timestamp (ISO 8601, second resolution) for provenance records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_scalar(value: Any) -> Any:
    """Keep JSON scalars as-is; stringify anything else so the record stays readable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _stderr_tail(text: Any) -> str:
    return str(text).strip()[:200]


def _source_root() -> pathlib.Path:
    """Source/install root that contains the `astrolabe` package (never the cwd)."""
    import astrolabe

    resolver = getattr(astrolabe, "install_root", None)
    if callable(resolver):
        return pathlib.Path(resolver())
    return pathlib.Path(__file__).resolve().parents[1]


def _git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )


def collect_git_provenance(source_root: Any = None) -> dict[str, Any]:
    """Collect commit/branch/dirty state from the source root's own `.git`.

    Limitations: only the source root is inspected (`git -C <source_root>`), never the
    current working directory, so an installed wheel or copied source tree reports
    `available: false` rather than inferring an unrelated checkout. `dirty` covers
    tracked modifications only (`--untracked-files=no --ignore-submodules=all`) and is
    `None` when that check could not run. Every failure is reported in `reason`; this
    function never raises.
    """
    root = pathlib.Path(source_root) if source_root is not None else _source_root()
    record: dict[str, Any] = {
        "available": False,
        "root": str(root),
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_entries": None,
        "dirty_scope": (
            "tracked modifications; untracked files and submodule worktrees excluded"
        ),
        "reason": None,
    }
    if not (root / ".git").exists():
        record["reason"] = "no .git in source root (installed or copied source tree)"
        return record
    if shutil.which("git") is None:
        record["reason"] = "git executable not found on PATH"
        return record
    try:
        commit = _git(root, "rev-parse", "HEAD")
        if commit.returncode != 0:
            record["reason"] = f"git rev-parse HEAD failed: {_stderr_tail(commit.stderr)}"
            return record
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = _git(
            root,
            "status",
            "--porcelain",
            "--ignore-submodules=all",
            "--untracked-files=no",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record["reason"] = f"git invocation failed: {exc}"
        return record

    record["available"] = True
    record["commit"] = commit.stdout.strip() or None
    if branch.returncode == 0 and branch.stdout.strip():
        record["branch"] = branch.stdout.strip()
    if dirty.returncode != 0:
        record["reason"] = f"git status failed: {_stderr_tail(dirty.stderr)}"
    else:
        entries = [line for line in dirty.stdout.splitlines() if line.strip()]
        record["dirty"] = bool(entries)
        record["dirty_entries"] = len(entries)
    return record


def collect_version_provenance() -> dict[str, Any]:
    """Running code version: package attribute first, installed metadata as fallback.

    Limitations: `astrolabe.__version__` is authoritative when present; installed
    distribution metadata is only a fallback and a missing/lookup-failing version is
    reported as `source: "unavailable"` with a null value instead of raising.
    """
    import astrolabe

    version = getattr(astrolabe, "__version__", None)
    source = "package attribute"
    if version in (None, ""):
        try:
            version = importlib.metadata.version("astrolabe")
            source = "installed distribution metadata"
        except Exception:
            version, source = None, "unavailable"
    return {
        "astrolabe": None if version in (None, "") else str(version),
        "source": source,
    }


def collect_dependency_versions() -> dict[str, str]:
    """Installed versions of a small fixed set of runtime dependencies.

    Limitations: values come from distribution metadata only; no dependency module is
    imported (in particular no GPU library), so a dependency without installed metadata
    is omitted rather than detected through an import.
    """
    versions: dict[str, str] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = str(importlib.metadata.version(distribution))
        except Exception:
            continue
    return versions


def _collect_gpu_provenance() -> dict[str, Any]:
    """Best-effort `nvidia-smi` inventory, collected without importing GPU libraries."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    command = (executable, _NVIDIA_SMI_QUERY, "--format=csv,noheader")
    command_text = " ".join(command)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason": f"nvidia-smi invocation failed: {exc}",
            "command": command_text,
        }
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": (
                f"nvidia-smi exited {proc.returncode}: {_stderr_tail(proc.stderr)}"
            ),
            "command": command_text,
        }
    gpus: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or not fields[0]:
            continue
        gpus.append(
            {
                "index": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total": fields[3],
                "compute_capability": fields[4],
            }
        )
    return {
        "available": True,
        "count": len(gpus),
        "gpus": gpus,
        "command": command_text,
    }


def collect_hardware_provenance() -> dict[str, Any]:
    """Host facts plus device visibility, collected without importing GPU libraries.

    Limitations: device visibility is the raw environment (`None` means "unset", i.e.
    provider default visibility), and the GPU inventory is a point-in-time `nvidia-smi`
    query that may disagree with what a container actually maps in.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device_visibility_env": {
            name: os.environ.get(name) for name in _VISIBILITY_ENV_VARS
        },
        "nvidia_smi": _collect_gpu_provenance(),
    }


def _stream_sha256(path: pathlib.Path) -> tuple[str | None, int | None, str | None]:
    """Stream-hash a file; return `(sha256, size_bytes, reason)`."""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        return None, None, f"unreadable: {exc}"
    return digest.hexdigest(), size, None


def _candidate_paths(raw: str) -> list[pathlib.Path]:
    """Candidate paths for one explicit value, tolerating a trailing `:CHAIN` suffix."""
    text = str(raw).strip()
    candidates = [text]
    head, sep, tail = text.rpartition(":")
    if sep and head and tail and "." in head:
        candidates.append(head)
    paths: list[pathlib.Path] = []
    for candidate in candidates:
        expanded = pathlib.Path(os.path.expanduser(candidate))
        paths.append(expanded if expanded.is_absolute() else pathlib.Path.cwd() / expanded)
    return paths


def identify_input_file(raw: Any) -> dict[str, Any] | None:
    """Identify one explicit input file by streamed sha256 (never by reading it whole).

    Limitations: only the values passed in are identified; model weights referenced
    elsewhere in the config are never hashed. Relative paths resolve against the launch
    directory, and an unresolvable value keeps a null hash plus a `reason`.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return {
            "value_type": type(raw).__name__,
            "path": None,
            "sha256": None,
            "reason": "non-path value (not hashed)",
        }
    candidates = _candidate_paths(raw)
    for candidate in candidates:
        if candidate.is_file():
            sha256, size, reason = _stream_sha256(candidate)
            return {
                "value": raw,
                "path": str(candidate),
                "sha256": sha256,
                "size_bytes": size,
                "reason": reason,
            }
    return {
        "value": raw,
        "path": str(candidates[0]),
        "sha256": None,
        "reason": "not found",
    }


def _identify_input_entries(values: Any) -> dict[str, Any] | None:
    """Identify a mapping of explicit input values (`named_files`, `homologies`)."""
    if not isinstance(values, dict) and not OmegaConf.is_dict(values):
        return None
    entries: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            entries[str(key)] = {
                "value_type": "list",
                "entries": [identify_input_file(item) for item in value],
            }
        elif isinstance(value, dict):
            entries[str(key)] = {
                "value_type": "dict",
                "path": None,
                "sha256": None,
                "reason": "inline/non-path value (not hashed)",
            }
        else:
            entries[str(key)] = identify_input_file(value)
    return entries


def build_run_provenance(args: Any) -> dict[str, Any]:
    """Assemble the provenance record for the run directory described by `args`.

    Limitations: the config digest covers the persisted `config.yaml` bytes (pruned,
    interpolations unresolved) rather than the resolved in-memory config; timestamps are
    UTC ISO-8601 with second resolution; `input_files` covers only explicit
    `named_files`/`homologies` entries, not every file a stage may read later; and the
    Git section follows :func:`collect_git_provenance` (source root only).
    """
    now = _utc_now_iso()
    out_dir = getattr(args, "out_dir", None)
    config_record: dict[str, Any] = {
        "path": "config.yaml",
        "algorithm": "sha256",
        "sha256": None,
        "reason": None,
    }
    if out_dir is not None:
        config_path = pathlib.Path(str(out_dir)) / "config.yaml"
        if config_path.is_file():
            config_record["sha256"], _, config_record["reason"] = _stream_sha256(
                config_path
            )
        else:
            config_record["reason"] = "config.yaml not found"
    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "state": "started",
        "terminal": False,
        "started_at": now,
        "updated_at": now,
        "ended_at": None,
        "error": None,
        "status_history": [{"state": "started", "at": now}],
        "run": {
            "task_name": _json_scalar(getattr(args, "task_name", None)),
            "run_dir": None if out_dir is None else str(out_dir),
            "seed": _json_scalar(getattr(args, "seed", None)),
            "repeat_index": _json_scalar(getattr(args, "repeat_index", None)),
            "repeats": _json_scalar(getattr(args, "repeats", None)),
        },
        "version": collect_version_provenance(),
        "git": collect_git_provenance(),
        "config": config_record,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependencies": collect_dependency_versions(),
        "hardware": collect_hardware_provenance(),
        "input_files": {
            "named_files": _identify_input_entries(getattr(args, "named_files", None)),
            "homologies": _identify_input_entries(getattr(args, "homologies", None)),
        },
    }


def _provenance_path(args: Any) -> pathlib.Path | None:
    out_dir = getattr(args, "out_dir", None)
    if not out_dir:
        return None
    return pathlib.Path(str(out_dir)) / _PROVENANCE_FILENAME


def _write_provenance(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Atomically replace `provenance.json` (temp file beside it, then `os.replace`)."""
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_provenance(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def start_run_provenance(args: Any) -> pathlib.Path | None:
    """Write the initial `provenance.json` for a run that is about to start.

    Called once per run directory right after the persisted `config.yaml` exists, so the
    record can carry its digest. Limitations/contract: entirely best-effort — failures
    are logged as warnings and `None` is returned instead of raising, so provenance can
    never abort or alter a scientific run.
    """
    try:
        path = _provenance_path(args)
        if path is None:
            return None
        _write_provenance(path, build_run_provenance(args))
        return path
    except Exception:
        logging.getLogger(__name__).warning("Failed to write run provenance", exc_info=True)
        return None


def finish_run_provenance(
    args: Any, state: str, *, error: BaseException | None = None
) -> pathlib.Path | None:
    """Record a terminal state (`completed`/`failed`/`interrupted`) on the run record.

    Limitations/contract: best-effort like :func:`start_run_provenance` — failures are
    logged and never raised, so a provenance problem cannot mask the caller's scientific
    exception. Only `state == "completed"` means success: an interrupted or failed run
    records that terminal state, and a process killed before finalizing keeps `started`
    (it must never be read as completion).
    """
    try:
        path = _provenance_path(args)
        if path is None:
            return None
        payload = _load_provenance(path)
        if payload is None:
            payload = build_run_provenance(args)
        now = _utc_now_iso()
        history = payload.get("status_history")
        if not isinstance(history, list):
            history = []
        history.append({"state": state, "at": now})
        payload.update(
            {
                "state": state,
                "terminal": state in _TERMINAL_PROVENANCE_STATES,
                "updated_at": now,
                "ended_at": now,
                "status_history": history,
            }
        )
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_provenance(path, payload)
        return path
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to update run provenance", exc_info=True
        )
        return None
