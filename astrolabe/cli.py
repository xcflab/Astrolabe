from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import sys
import ast
import glob
import subprocess
import importlib.util
from collections import namedtuple
from functools import lru_cache
from shutil import which
from typing import Any, Iterable, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException
from hydra.errors import HydraException, MissingConfigException
from yaml import YAMLError

from astrolabe import cli_command_definitions as cli_defs
from astrolabe import design_preset_builder
from astrolabe import runtime_requirements
from astrolabe.dev_wrapper import (
    default_local_env_path,
    default_local_env_path_for_config_dir,
    write_astrolabe_python_hint,
    write_astrolabe_python_hint_for_config_dir,
)
from astrolabe.posthoc import refold_backends
from astrolabe.structure_processors.wrappers import default_rapidock_daemon_script_path

_VALID_CYCLIZE_BOND_ORDERS = frozenset({"single", "double", "triple", "aromatic"})
_CLOSURE_PROFILE_PRESETS = {
    "strict": {"sigma": 0.20, "aggregate": "min"},
    "min": {"sigma": 0.20, "aggregate": "min"},
    "mean": {"sigma": 0.20, "aggregate": "mean"},
}
_LIGAND_TYPE_COMPLETION_CHOICES = ("ccd", "smiles")
_FRAMEWORK_KIND_COMPLETION_CHOICES = ("auto", "vhh", "scfv")
_SCFV_ORDER_COMPLETION_CHOICES = ("vh_linker_vl", "vl_linker_vh")
_MULTITARGET_BINDER_VARIANT_TO_CONFIG = {
    "dplm-protenix": "benchmark_multitarget_binder_dplm_v1",
    "x-init-direct-dual-protenix": "benchmark_multitarget_binder_xinit_v1",
}
_MULTIDOMAIN_BINDER_VARIANTS = ("af3", "protenix")


def _sanitize_name(raw: str) -> str:
    s = (raw or "").strip().lower().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    out = "".join([c for c in s if c in allowed])
    return out or "compose"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _ensure_sys_path_has_repo_root(_config_dir: pathlib.Path | None = None) -> pathlib.Path:
    repo_root = _repo_root()
    if _config_dir is not None:
        repo_root = pathlib.Path(_config_dir).resolve().parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # Vendored deps: allow importing pure-Python parts without requiring
    # `pip install -e vendor/...`.
    af2_root = repo_root / "vendor" / "alphafold2"
    af2_root_str = str(af2_root)
    if af2_root.exists() and af2_root_str not in sys.path:
        sys.path.insert(0, af2_root_str)

    # This is used by the AlphaFold3 evaluator path; non-AF3 evaluators should
    # avoid importing AF3 modules unless strictly necessary.
    af3_src = repo_root / "vendor" / "alphafold3" / "src"
    af3_src_str = str(af3_src)
    # A source checkout does not contain AF3's generated CCD/extension files.
    # Preserve an installed (possibly editable) package; source-only use remains
    # available when no package is installed or through explicit PYTHONPATH.
    if (af3_src.exists() and af3_src_str not in sys.path
            and importlib.util.find_spec("alphafold3") is None):
        sys.path.insert(0, af3_src_str)
    return repo_root


def _default_config_dir() -> pathlib.Path:
    return _repo_root() / "config"


def _list_available_tasks(config_dir: pathlib.Path) -> List[str]:
    if not config_dir.exists():
        return []
    tasks = []  # type: List[str]
    for path in sorted(config_dir.glob("*.yaml")):
        if path.name in {"env.yaml"}:
            continue
        tasks.append(path.stem)
    return tasks


def _resolve_template_config(config_dir: pathlib.Path, task: Optional[str]) -> str:
    if task:
        config_path = config_dir / f"{task}.yaml"
        if not config_path.exists():
            tasks = _list_available_tasks(config_dir)
            hint = f" Available tasks: {', '.join(tasks)}" if tasks else ""
            raise FileNotFoundError(
                f"Task template not found: {config_path}.{hint}"
            )
        return task

    base_path = config_dir / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base template not found: {base_path}")
    return "base"


def _read_overrides_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    override_path = pathlib.Path(path)
    lines = []  # type: List[str]
    for raw_line in override_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _split_overrides_string(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [token for token in shlex.split(value) if token]


def _normalize_remainder_overrides(values):  # type: (List[str]) -> List[str]
    if not values:
        return []
    if values and values[0] == "--":
        values = values[1:]
    return [v for v in values if v]


def _compose_hydra_config(
    *,
    config_dir: pathlib.Path,
    config_name: str,
    overrides: List[str],
) -> Any:
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    # Local env overlays are intentionally partial; if a newer repo schema adds
    # keys under `env.yaml`, promote matching overrides to Hydra `+key=value`
    # so sparse env_local files do not block forward-compatible config updates.
    overrides = _normalize_env_local_overrides(config_dir, overrides)

    # Hydra keeps global state; clear it to allow repeated calls in the same process.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name="astrolabe",
    )
    try:
        return hydra.compose(config_name=config_name, overrides=overrides)
    except MissingConfigException as exc:
        # A fresh distribution has no site-local file. Preserve explicit env
        # choices and all configured checkouts; only the absent default changes.
        if exc.missing_cfg_file != "env/env_local" or any(
            token.lstrip("+").startswith("env=") for token in overrides
        ):
            raise
        print(
            "No site-local environment: using env=env. Configure model paths "
            "before inference (see README installation).",
            file=sys.stderr,
        )
        return hydra.compose(config_name=config_name, overrides=["env=env", *overrides])


def _apply_override_config_file(cfg, override_config_path):  # type: (Any, Optional[str]) -> Any
    if not override_config_path:
        return cfg
    override_cfg = OmegaConf.load(override_config_path)
    return OmegaConf.merge(cfg, override_cfg)


def _strip_defaults_key(cfg: Any) -> Any:
    # Standalone configs often include Hydra-style `defaults:`. In `astrolabe run --standalone`
    # mode we treat that file as an overlay on top of the repo template, so `defaults`
    # is ignored (the template already defines the base config composition).
    if not isinstance(cfg, (dict, DictConfig)):
        return cfg
    if "defaults" not in cfg:
        return cfg

    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg_copy):
        if "defaults" in cfg_copy:
            del cfg_copy["defaults"]
    return cfg_copy


def _get_by_dotted_key(obj: Any, dotted_key: str) -> Any:
    cur = obj
    for part in str(dotted_key).split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _has_dotted_key(obj: Any, dotted_key: str) -> bool:
    cur = obj
    for part in str(dotted_key).split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _normalize_env_local_overrides(config_dir: pathlib.Path, overrides: List[str]) -> List[str]:
    env_schema_path = config_dir / "env.yaml"
    env_local_path = config_dir / "env" / "env_local.yaml"
    if not env_schema_path.exists() or not env_local_path.exists() or not overrides:
        return overrides

    env_schema = OmegaConf.to_container(OmegaConf.load(env_schema_path), resolve=False)
    env_local = OmegaConf.to_container(OmegaConf.load(env_local_path), resolve=False)
    normalized: List[str] = []
    for override in overrides:
        if override.startswith(("+", "~")) or "=" not in override:
            normalized.append(override)
            continue
        key, _value = override.split("=", 1)
        if _has_dotted_key(env_schema, key) and not _has_dotted_key(env_local, key):
            normalized.append(f"+{override}")
        else:
            normalized.append(override)
    return normalized


def _set_by_dotted_key(obj: dict, dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        return
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _ast_literal(node: Any) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return None


@lru_cache(maxsize=1)
def _loss_index_from_source() -> dict[str, dict]:
    """
    Fast loss registry snapshot without importing `astrolabe.losses` (which is slow).

    Returns mapping: loss_type -> {description, required_params, optional_params, supports_oligomer_specific}
    """
    losses_dir = _repo_root() / "astrolabe" / "losses"
    out: dict[str, dict] = {}
    if not losses_dir.exists():
        return out

    skip = {"__init__.py", "base.py", "utils.py"}
    for path in sorted(losses_dir.glob("*.py")):
        if path.name in skip:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            dec_call = None
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fn = dec.func
                if not (isinstance(fn, ast.Attribute) and fn.attr == "register"):
                    continue
                if not (isinstance(fn.value, ast.Name) and fn.value.id == "loss_registry"):
                    continue
                dec_call = dec
                break
            if dec_call is None:
                continue

            meta: dict[str, Any] = {
                "description": "",
                "required_params": [],
                "optional_params": [],
                "supports_oligomer_specific": False,
            }
            explicit_params = False
            name = None
            for kw in dec_call.keywords:
                if not kw.arg:
                    continue
                if kw.arg == "name":
                    v = _ast_literal(kw.value)
                    if isinstance(v, str) and v.strip():
                        name = v.strip()
                elif kw.arg == "description":
                    v = _ast_literal(kw.value)
                    if isinstance(v, str):
                        meta["description"] = v
                elif kw.arg == "supports_oligomer_specific":
                    v = _ast_literal(kw.value)
                    if isinstance(v, bool):
                        meta["supports_oligomer_specific"] = v
                elif kw.arg in {"params", "required_params", "optional_params"}:
                    v = _ast_literal(kw.value)
                    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                        if kw.arg == "required_params":
                            meta["required_params"] = list(v)
                            explicit_params = True
                        elif kw.arg == "optional_params":
                            meta["optional_params"] = list(v)
                            explicit_params = True
                        else:
                            # "params" isn't enough to split req/opt; treat as optional list.
                            meta["optional_params"] = list(v)
                            explicit_params = True

            loss_name = name or node.name
            if not explicit_params:
                # Match LossRegistry.register() auto-inference rules.
                exclude = {"eval_result", "runtime_config", "oligo2complex"}
                args = [a.arg for a in (node.args.args or []) if a.arg not in exclude and a.arg != "self"]
                # positional defaults apply to last N args
                n_pos = len(args)
                n_def = len(node.args.defaults or [])
                req = args[: max(0, n_pos - n_def)]
                opt = args[max(0, n_pos - n_def) :]
                # kw-only
                kwonly = [a.arg for a in (node.args.kwonlyargs or []) if a.arg not in exclude and a.arg != "self"]
                kw_defs = list(node.args.kw_defaults or [])
                for i, k in enumerate(kwonly):
                    d = kw_defs[i] if i < len(kw_defs) else None
                    if d is None:
                        req.append(k)
                    else:
                        opt.append(k)
                meta["required_params"] = req
                meta["optional_params"] = opt

            out[str(loss_name)] = meta

    return out


def _task_requirements(task_cfg: Any) -> dict:
    return runtime_requirements.task_requirements(task_cfg)


def _normalize_dotlist_overrides(overrides: List[str]) -> List[str]:
    out: List[str] = []
    for item in overrides:
        if not item:
            continue
        if item.startswith("+++"):
            item = item[3:]
        elif item.startswith("++"):
            item = item[2:]
        elif item.startswith("+"):
            item = item[1:]
        out.append(item)
    return out


def _run_from_config(cfg: Any, *, autoselect_gpu: bool) -> None:
    _maybe_autoselect_cuda_visible_devices(enabled=autoselect_gpu)

    # Import late so `astrolabe env doctor` doesn't drag heavy deps.
    from astrolabe.app import run_from_cfg

    run_from_cfg(cfg)


def _maybe_autoselect_cuda_visible_devices(*, enabled: bool) -> None:
    """Pick a GPU automatically when users didn't specify CUDA_VISIBLE_DEVICES.

    Root cause: many components default to torch's "cuda" (cuda:0). On shared
    machines this frequently OOMs if GPU0 is busy even though other GPUs are free.
    We avoid importing torch here (must run before any CUDA init).
    """

    if not enabled:
        return
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        return

    if which("nvidia-smi") is None:
        return

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return

    candidates: list[tuple[int, int]] = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpu_idx = int(parts[0])
            mem_free_mib = int(parts[1])
        except ValueError:
            continue
        candidates.append((gpu_idx, mem_free_mib))

    if not candidates:
        return

    best_gpu_idx, best_free_mib = max(candidates, key=lambda x: x[1])
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_idx)
    print(
        f"[astrolabe] Auto-selected CUDA_VISIBLE_DEVICES={best_gpu_idx} "
        f"(free={best_free_mib} MiB via nvidia-smi)."
    )


DoctorItem = namedtuple("DoctorItem", ["key", "value", "exists", "kind"])


def _iter_string_leaves(
    obj, prefix=""
):  # type: (Any, str) -> Iterable[Tuple[str, str]]
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_string_leaves(v, key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield from _iter_string_leaves(v, key)
    elif isinstance(obj, str):
        yield (prefix, obj)


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith("${"):
        return True
    if value.startswith(("/", "./", "../")):
        return True
    if value.startswith("~"):
        return True
    return False


def _doctor_scan_env(env_path, resolve):  # type: (pathlib.Path, bool) -> List[DoctorItem]
    cfg = OmegaConf.load(str(env_path))
    as_obj = OmegaConf.to_container(cfg, resolve=resolve)
    items = []  # type: List[DoctorItem]

    for key, value in _iter_string_leaves(as_obj):
        if not _looks_like_path(value):
            continue
        if "${" in value:
            items.append(DoctorItem(key=key, value=value, exists=None, kind="unknown"))
            continue
        path = _resolve_existing_env_path(value, env_path=env_path)
        if path is None:
            items.append(DoctorItem(key=key, value=value, exists=False, kind="unknown"))
            continue
        if path.exists():
            kind = "dir" if path.is_dir() else "file" if path.is_file() else "path"
            items.append(DoctorItem(key=key, value=value, exists=True, kind=kind))
        else:
            items.append(DoctorItem(key=key, value=value, exists=False, kind="unknown"))

    return items


def _probe_rapidock_imports(*, python_bin: str, repo_path: str) -> None:
    cmd = [
        str(python_bin),
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "import RAPiDock, MDAnalysis, torch_geometric, e3nn, esm, PeptideBuilder"
        ),
        str(repo_path),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"RAPiDock subprocess import probe failed with exit code {proc.returncode}: {detail}"
        )


def _probe_meeko_imports(*, python_bin: str, pythonpath: str | None = None) -> None:
    cmd = [
        str(python_bin),
        "-c",
        "import meeko, rdkit, gemmi",
    ]
    env = None
    if pythonpath:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(pythonpath) if not existing else str(pythonpath) + os.pathsep + existing
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Meeko subprocess import probe failed with exit code {proc.returncode}: {detail}"
        )


def _probe_rapidock_daemon(
    *, python_bin: str, repo_path: str, model_dir: str, ckpt: str
) -> None:
    daemon_path = default_rapidock_daemon_script_path()
    if not daemon_path.exists():
        raise FileNotFoundError(f"RAPiDock daemon script not found: {daemon_path}")

    proc = subprocess.Popen(
        [
            str(python_bin),
            "-u",
            str(daemon_path),
            "--repo-path",
            str(pathlib.Path(repo_path).expanduser()),
            "--model-dir",
            str(model_dir),
            "--ckpt",
            str(ckpt),
        ],
        cwd=str(pathlib.Path(repo_path).expanduser()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps({"type": "ping", "id": "doctor"}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read().strip()
            raise RuntimeError(
                f"RAPiDock daemon ping failed to return output. stderr={stderr}"
            )
        response = json.loads(line)
        if response.get("type") != "pong" or response.get("id") != "doctor":
            raise RuntimeError(f"Unexpected RAPiDock daemon ping response: {response}")
        proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def _cmd_run(ns: argparse.Namespace) -> int:
    if not any(
        [
            getattr(ns, "task", None),
            getattr(ns, "standalone", None),
            getattr(ns, "overlay", None),
            getattr(ns, "overrides", None),
            getattr(ns, "overrides_file", None),
            bool(getattr(ns, "print_config", False)),
            bool(getattr(ns, "dry_run", False)),
            list(getattr(ns, "hydra_overrides", []) or []),
        ]
    ):
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print("astrolabe run requires a template/standalone config or overrides.", file=sys.stderr)
        return 2

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()

    config_file_path = pathlib.Path(ns.standalone).expanduser().resolve() if ns.standalone else None
    if config_file_path is not None:
        if not config_file_path.exists():
            raise FileNotFoundError(f"--standalone not found: {config_file_path}")
        if ns.chdir:
            os.chdir(str(config_file_path.parent))

    config_name = _resolve_template_config(config_dir, ns.task)

    overrides: List[str] = []
    overrides.extend(_read_overrides_file(ns.overrides_file))
    overrides.extend(_split_overrides_string(ns.overrides))
    overrides.extend(_normalize_remainder_overrides(ns.hydra_overrides))

    compose_overrides = overrides
    post_overrides: List[str] = []
    if config_file_path is not None:
        # Standalone configs may introduce keys not present in the repo template;
        # apply CLI overrides after layering the file to avoid Hydra strict errors.
        #
        # BUT: config-group switches (e.g. `env=env_local`) must be applied at compose time,
        # otherwise they become plain dotlist keys and won't affect Hydra defaults.
        def _is_group_override(token: str) -> bool:
            t = (token or "").strip()
            if "=" not in t:
                return False
            # Dotted keys are almost certainly config values, not groups.
            lhs = t.split("=", 1)[0].strip()
            if not lhs:
                return False
            if lhs.startswith("+"):
                lhs = lhs[1:]
            if "." in lhs:
                return False
            # Only treat as a group override if a matching group directory exists.
            return (config_dir / lhs).is_dir()

        compose_overrides = []
        post_overrides = []
        for ov in overrides:
            if _is_group_override(ov):
                compose_overrides.append(ov)
            else:
                post_overrides.append(ov)

    cfg = _compose_hydra_config(
        config_dir=config_dir,
        config_name=config_name,
        overrides=compose_overrides,
    )
    if config_file_path is not None:
        external_cfg = OmegaConf.load(str(config_file_path))
        external_cfg = _strip_defaults_key(external_cfg)
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, external_cfg)
        if post_overrides:
            dot_cfg = OmegaConf.from_dotlist(_normalize_dotlist_overrides(post_overrides))
            with open_dict(cfg):
                cfg = OmegaConf.merge(cfg, dot_cfg)
    cfg = _apply_override_config_file(cfg, ns.overlay)

    if ns.print_config or ns.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=False))
        if ns.dry_run:
            return 0

    _run_from_config(cfg, autoselect_gpu=bool(getattr(ns, "autoselect_gpu", False)))
    return 0


def _cmd_completion_bash(_ns: argparse.Namespace) -> int:
    from astrolabe.shell_completion import render_bash_completion

    print(render_bash_completion(init=bool(getattr(_ns, "init", False))), end="")
    return 0


def _cmd_completion_zsh(_ns: argparse.Namespace) -> int:
    from astrolabe.shell_completion import render_zsh_completion

    print(render_zsh_completion(init=bool(getattr(_ns, "init", False))), end="")
    return 0


def _cmd_completion_complete(ns: argparse.Namespace) -> int:
    from astrolabe.shell_completion import (
        complete_words,
        render_items_for_shell,
        shell_words_from_remainder,
    )

    words = shell_words_from_remainder(getattr(ns, "words", []) or [])
    cword = int(getattr(ns, "cword", 0) or 0)
    items = complete_words(words, cword, cwd=pathlib.Path.cwd())
    print(render_items_for_shell(items))
    return 0


def _cmd_experimental_candidate_batch(ns: argparse.Namespace) -> int:
    from astrolabe.experimental.candidate_batch_runner import run_from_namespace

    return run_from_namespace(ns, parser=getattr(ns, "_parser", None))


def _parse_len_range(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty length range")
    if "-" in s:
        a, b = s.split("-", 1)
        lo = int(a.strip())
        hi = int(b.strip())
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Invalid length range: {raw}")
        return f"{lo}-{hi}"
    n = int(s)
    if n <= 0:
        raise ValueError(f"Invalid length: {raw}")
    return f"{n}-{n}"


_PROTEIN_INITIAL_SEQ_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX]+$")
_RNA_INITIAL_SEQ_RE = re.compile(r"^[AUGC]+$")
_DNA_INITIAL_SEQ_RE = re.compile(r"^[ATGC]+$")


def _normalize_initial_sequence(
    raw: str,
    *,
    flag_name: str,
    expr_flag_name: str,
    pattern: re.Pattern[str],
    alphabet_description: str,
) -> str:
    seq = str(raw or "").strip().upper()
    if not seq:
        raise ValueError(f"{flag_name} is required")
    if not pattern.fullmatch(seq):
        raise ValueError(
            f"{flag_name} expects one contiguous sequence using "
            f"{alphabet_description}; use {expr_flag_name} for protomer "
            "statements such as '+ACD 20-30' or 'input:A'."
        )
    return seq


def _normalize_initial_binder_seq(
    raw: str,
    *,
    flag_name: str,
    pattern: re.Pattern[str],
    alphabet_description: str,
) -> str:
    return _normalize_initial_sequence(
        raw,
        flag_name=flag_name,
        expr_flag_name="--initial-binder-expr",
        pattern=pattern,
        alphabet_description=alphabet_description,
    )


def _resolve_initial_a_statement(
    ns: argparse.Namespace,
    *,
    length_attr: str,
    seq_pattern: re.Pattern[str] = _PROTEIN_INITIAL_SEQ_RE,
    alphabet_description: str = "ACDEFGHIKLMNPQRSTVWYX",
) -> str:
    initial_seq = getattr(ns, "initial_binder_seq", None)
    initial_expr = getattr(ns, "initial_binder_expr", None)
    if initial_seq is not None:
        return _normalize_initial_binder_seq(
            initial_seq,
            flag_name="--initial-binder-seq",
            pattern=seq_pattern,
            alphabet_description=alphabet_description,
        )
    if initial_expr is not None:
        expr = str(initial_expr or "").strip()
        if not expr:
            raise ValueError("--initial-binder-expr is required")
        return expr
    return _parse_len_range(getattr(ns, length_attr, None))


def _resolve_initial_binder_statement(ns: argparse.Namespace) -> str:
    return _resolve_initial_a_statement(ns, length_attr="binder_len")


def _aptamer_type_settings(raw: str | None) -> tuple[str, re.Pattern[str], str]:
    value = str(raw or "rna").strip().lower()
    if value in {"r", "rna"}:
        return "r", _RNA_INITIAL_SEQ_RE, "AUGC"
    if value in {"d", "dna"}:
        return "d", _DNA_INITIAL_SEQ_RE, "ATGC"
    raise ValueError(f"Invalid aptamer type: {raw!r} (expected: rna|dna)")


def _resolve_initial_statement_from_attrs(
    ns: argparse.Namespace,
    *,
    length_attr: str,
    seq_attr: str,
    expr_attr: str,
    length_flag: str,
    seq_flag: str,
    expr_flag: str,
    seq_pattern: re.Pattern[str] = _PROTEIN_INITIAL_SEQ_RE,
    alphabet_description: str = "ACDEFGHIKLMNPQRSTVWYX",
) -> str:
    initial_seq = getattr(ns, seq_attr, None)
    initial_expr = getattr(ns, expr_attr, None)
    if initial_seq is not None:
        return _normalize_initial_sequence(
            initial_seq,
            flag_name=seq_flag,
            expr_flag_name=expr_flag,
            pattern=seq_pattern,
            alphabet_description=alphabet_description,
        )
    if initial_expr is not None:
        expr = str(initial_expr or "").strip()
        if not expr:
            raise ValueError(f"{expr_flag} is required")
        return expr
    return _parse_len_range(getattr(ns, length_attr, None))


def _add_initial_binder_args(
    parser: argparse.ArgumentParser,
    *,
    length_flags: tuple[str, ...],
    len_help: str,
    seq_help: str = (
        "Mutable initial binder sequence for chain A. Writes seqs.A as this "
        "bare sequence; use --initial-binder-expr for fixed segments or "
        "multi-segment statements."
    ),
    expr_help: str = (
        "Advanced binder statement written directly to seqs.A, e.g. "
        "'+ACD 20-30' or 'input:A'."
    ),
) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(*length_flags, default=None, help=len_help)
    group.add_argument(
        "--initial-binder-seq",
        default=None,
        metavar="SEQ",
        help=seq_help,
    )
    group.add_argument(
        "--initial-binder-expr",
        default=None,
        metavar="EXPR",
        help=expr_help,
    )


def _add_initial_protein_chain_args(
    parser: argparse.ArgumentParser,
    *,
    chain_label: str,
    length_flag: str,
    length_dest: str,
    seq_flag: str,
    seq_dest: str,
    expr_flag: str,
    expr_dest: str,
) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        length_flag,
        dest=length_dest,
        default=None,
        help=f"Mutable protein {chain_label} length range, e.g. '80-120' or '100'.",
    )
    group.add_argument(
        seq_flag,
        dest=seq_dest,
        default=None,
        metavar="SEQ",
        help=f"Mutable protein {chain_label} initial sequence.",
    )
    group.add_argument(
        expr_flag,
        dest=expr_dest,
        default=None,
        metavar="EXPR",
        help=(
            f"Advanced protomer statement for protein {chain_label}, e.g. "
            "'+ACD 20-30' or 'input:A'."
        ),
    )


def _parse_cyclize_pos(raw: str) -> int | str:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty cyclization position")
    if s in {"-1"}:
        return -1
    if s.isdigit():
        v = int(s)
        if v <= 0:
            raise ValueError(f"Cyclization position must be 1-indexed (>0), got {raw!r}")
        return v
    key = s.lower()
    if key in {"nterm", "cterm"}:
        return key
    raise ValueError(f"Invalid cyclization position {raw!r} (expected int, -1, nterm, cterm)")


def _parse_cyclize_atom_ref(raw: str) -> dict[str, object]:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty cyclization atom ref")
    if "@" not in s:
        raise ValueError(f"Invalid cyclization atom ref {raw!r} (missing '@')")
    left, atom = s.split("@", 1)
    atom = atom.strip()
    if not atom:
        raise ValueError(f"Invalid cyclization atom ref {raw!r} (empty atom)")
    if ":" in left:
        proto, pos_s = left.split(":", 1)
        proto = proto.strip()
        pos_s = pos_s.strip()
    else:
        raise ValueError(f"Invalid cyclization atom ref {raw!r} (missing '<proto>:<pos>@<atom>')")
    if not proto:
        raise ValueError(f"Invalid cyclization atom ref {raw!r} (empty proto)")
    pos = _parse_cyclize_pos(pos_s)
    return {"proto": proto, "pos": pos, "atom": atom}


def _parse_disulfide_pair(raw: str) -> tuple[int | str, int | str]:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty disulfide spec")
    if "-" not in s:
        raise ValueError(f"Invalid disulfide spec {raw!r} (expected '<pos1>-<pos2>')")
    a, b = s.split("-", 1)
    return _parse_cyclize_pos(a.strip()), _parse_cyclize_pos(b.strip())


def _parse_cyclize_bond_order(raw: str) -> str:
    value = str(raw or "").strip().lower()
    alias = {
        "s": "single",
        "single": "single",
        "d": "double",
        "double": "double",
        "t": "triple",
        "triple": "triple",
        "a": "aromatic",
        "aromatic": "aromatic",
    }
    order = alias.get(value, value)
    if order not in _VALID_CYCLIZE_BOND_ORDERS:
        raise ValueError(
            f"Invalid cyclize bond order {raw!r} "
            "(expected single|double|triple|aromatic)."
        )
    return order


def _parse_cyclize_bond_typed_spec(raw: str) -> dict[str, object]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Empty --cyclize-bond-typed spec")
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in {3, 4}:
        raise ValueError(
            "Invalid --cyclize-bond-typed spec. Expected "
            "'<ATOM1>,<ATOM2>,<ORDER>[,<TARGET_DISTANCE>]'"
        )
    a_ref = _parse_cyclize_atom_ref(parts[0])
    b_ref = _parse_cyclize_atom_ref(parts[1])
    order = _parse_cyclize_bond_order(parts[2])
    bond: dict[str, object] = {"a": a_ref, "b": b_ref, "order": order}
    if len(parts) == 4:
        try:
            target_distance = float(parts[3])
        except ValueError as exc:
            raise ValueError(
                f"Invalid target distance in --cyclize-bond-typed: {parts[3]!r}"
            ) from exc
        if target_distance <= 0:
            raise ValueError(
                f"--cyclize-bond-typed target distance must be > 0, got {target_distance}"
            )
        bond["target_distance"] = target_distance
    return bond


def _parse_cyclic_encoding(raw: str) -> dict[str, object]:
    value = str(raw or "").strip().lower()
    if not value:
        raise ValueError("--cyclic-encoding cannot be empty")
    if value in {"off", "none", "disabled", "false", "0"}:
        return {
            "enabled": False,
            "mode": "equivalent",
            "scope": "head_to_tail_only",
        }
    if value in {"head_to_tail", "head-to-tail", "head_to_tail_only", "default"}:
        return {
            "enabled": True,
            "mode": "equivalent",
            "scope": "head_to_tail_only",
        }
    if value in {"all", "all_bonds", "all-bonds"}:
        return {"enabled": True, "mode": "equivalent", "scope": "all_bonds"}
    if value in {
        "global",
        "global-equivalent",
        "global_equivalent",
        "global-generic",
        "global_generic",
        "rfpeptide",
        "rfpeptide-like",
        "rfpeptide_like",
    }:
        return {"enabled": True, "mode": "global_equivalent", "scope": "all_bonds"}
    if value in {
        "highfold",
        "highfold-cycpoem",
        "highfold_cycpoem",
        "cycpoem",
    }:
        return {"enabled": True, "mode": "highfold_cycpoem", "scope": "all_bonds"}
    raise ValueError(
        "Invalid --cyclic-encoding value "
        f"{raw!r} (expected head-to-tail|all-bonds|rfpeptide-like|highfold-cycpoem|off)"
    )


def _parse_closure_profile(raw: str) -> dict[str, object]:
    key = str(raw or "").strip().lower()
    if not key:
        raise ValueError("--closure-profile cannot be empty")
    if key not in _CLOSURE_PROFILE_PRESETS:
        allowed = ", ".join(sorted(_CLOSURE_PROFILE_PRESETS.keys()))
        raise ValueError(
            f"Invalid --closure-profile {raw!r}. Expected one of: {allowed}."
        )
    return dict(_CLOSURE_PROFILE_PRESETS[key])


def _load_structured_config_file(path_like: str) -> dict[str, Any]:
    path = pathlib.Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    else:
        payload = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return dict(payload)


def _load_nonstandard_map(path_like: str) -> dict[str, object]:
    payload = _load_structured_config_file(path_like)
    if "nonstandard_residues" in payload:
        cfg = payload["nonstandard_residues"]
    elif "aliases" in payload:
        cfg = {"aliases": payload["aliases"]}
    else:
        raise ValueError(
            "Nonstandard map must contain either top-level 'nonstandard_residues' "
            "or top-level 'aliases'."
        )
    if not isinstance(cfg, dict):
        raise ValueError("nonstandard_residues config must be a mapping")
    aliases = cfg.get("aliases")
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("nonstandard_residues.aliases must be a non-empty mapping")
    return {"aliases": aliases}


def _normalize_fixed_sequence(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty sequence")
    return s if s.startswith("+") else f"+{s}"


def _strip_fixed_prefix(seq: str) -> str:
    s = (seq or "").strip()
    return s[1:] if s.startswith("+") else s


def _sequence_from_a3m(path: pathlib.Path) -> str:
    # Extract the first (query) sequence from an A3M file.
    seq_lines: list[str] = []
    in_seq = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_seq and seq_lines:
                    break
                in_seq = True
                continue
            if in_seq:
                seq_lines.append(line)

    if not seq_lines:
        raise ValueError(f"No sequence records found in A3M: {path}")

    raw = "".join(seq_lines)
    # A3M: lowercase letters = insertions; drop them. Remove gaps.
    cleaned = "".join([c for c in raw if c.isupper() and c.isalpha()])
    if not cleaned:
        raise ValueError(f"Empty query sequence after cleaning A3M: {path}")
    return cleaned


def _sequence_from_af3_json(homology_arg: str) -> str:
    import json
    from astrolabe.core.msa_utils import load_af3_chain_data

    file_str, *chain_parts = (homology_arg or "").split(":")
    if not file_str.endswith(".json"):
        raise ValueError(f"Not an AF3 JSON homology: {homology_arg!r}")
    if not chain_parts:
        raise ValueError(f"AF3 JSON homology must include chain id: {homology_arg!r}")
    chain_id = chain_parts[0]

    path = pathlib.Path(file_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"AF3 JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if "dialect" not in data:
        raise ValueError(f"{path} is not a valid AlphaFold3 JSON file (missing dialect)")
    _polymer_key, chain_data = load_af3_chain_data(data, chain_id)
    seq = chain_data.get("sequence")
    if not isinstance(seq, str) or not seq.strip():
        raise ValueError(f"AF3 JSON chain {chain_id!r} has no sequence field in {path}")
    return seq.strip()


def _infer_target_seq_from_homology(homology_arg: str) -> str:
    if not homology_arg:
        raise ValueError("Empty homology")
    file_str = homology_arg.split(":", 1)[0]
    if file_str.endswith(".json"):
        return _sequence_from_af3_json(homology_arg)
    if file_str.endswith(".a3m"):
        return _sequence_from_a3m(pathlib.Path(file_str).expanduser().resolve())
    raise ValueError(f"Cannot infer sequence from homology: {homology_arg!r} (expected .json:<chain> or .a3m)")


def _normalize_chain_list(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",")]
    out = [p for p in parts if p]
    if not out:
        raise ValueError("Empty chain list")
    # De-dupe but keep order.
    seen = set()
    deduped: list[str] = []
    for c in out:
        if c in seen:
            continue
        deduped.append(c)
        seen.add(c)
    return deduped


def _load_structure_chain_sequences(structure_path: pathlib.Path) -> dict[str, str]:
    # Local import: avoids loading optional deps during `astrolabe env doctor`.
    from astrolabe.core.structure_parser import create_structure_parser

    parser = create_structure_parser(str(structure_path))
    parsed = parser.parse()  # chain_id -> (indices, sequence)
    return {cid: seq for cid, (_idx, seq) in parsed.items() if seq}


def _template_release_date(structure_path: pathlib.Path) -> str:
    import gzip
    import re
    from datetime import datetime, timezone

    name = structure_path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    with opener(structure_path, "rt", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    for pattern in (
        r"_pdbx_audit_revision_history\.revision_date\s+([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"_pdbx_database_status\.recvd_initial_deposition_date\s+([0-9]{4}-[0-9]{2}-[0-9]{2})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # PDB inputs and ad hoc local structures may not carry deposition metadata.
    return datetime.fromtimestamp(
        structure_path.stat().st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%d")


def _single_chain_mmcif_string(structure_path: pathlib.Path, chain_id: str) -> str:
    import copy
    import gzip
    import io

    from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser  # type: ignore
    from Bio.PDB.Structure import Structure as BioStructure  # type: ignore
    from Bio.PDB.Model import Model  # type: ignore

    if chain_id is None or str(chain_id).strip() == "":
        raise ValueError("Missing chain id for template extraction")

    # Parse input (support .gz for convenience).
    name = structure_path.name.lower()
    is_cif = any(ext in name for ext in [".cif", ".mmcif"])
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    handle = None
    try:
        if name.endswith(".gz"):
            handle = gzip.open(str(structure_path), "rt")
            structure = parser.get_structure("complex", handle)
        else:
            structure = parser.get_structure("complex", str(structure_path))
    finally:
        if handle is not None:
            handle.close()

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"No models found in structure: {structure_path}")

    chain = None
    for ch in model.get_chains():
        if str(ch.id) == str(chain_id):
            chain = ch
            break
    if chain is None:
        available = sorted({str(ch.id) for ch in model.get_chains()})
        raise ValueError(
            f"Chain '{chain_id}' not found in {structure_path}. Available: {available}"
        )

    out_struct = BioStructure("template")
    out_model = Model(0)
    out_struct.add(out_model)
    out_model.add(copy.deepcopy(chain))

    io_out = MMCIFIO()
    io_out.set_structure(out_struct)
    buf = io.StringIO()
    io_out.save(buf)
    mmcif_str = buf.getvalue()
    if "_pdbx_audit_revision_history.revision_date" in mmcif_str:
        return mmcif_str

    mmcif_str = mmcif_str.rstrip()
    if mmcif_str.endswith("#"):
        mmcif_str = mmcif_str[:-1].rstrip()
    release_date = _template_release_date(structure_path)
    return (
        mmcif_str
        + f"\n_pdbx_audit_revision_history.revision_date {release_date}\n#\n"
    )


def _sequence_from_structure_chain(structure_arg: str, chain_id: str) -> tuple[pathlib.Path, str]:
    structure_path = pathlib.Path(str(structure_arg)).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"--target-structure not found: {structure_path}")
    chain = str(chain_id or "").strip()
    if not chain:
        raise ValueError("--target-chain is required when --target-structure is set")

    chain2seq = _load_structure_chain_sequences(structure_path)
    if chain not in chain2seq:
        available = sorted(chain2seq.keys())
        raise ValueError(
            f"Target chain '{chain}' not found in parsed sequences from {structure_path}. "
            f"Available: {available}"
        )
    return structure_path, chain2seq[chain]


def _template_homology_from_structure(
    *,
    structure_path: pathlib.Path,
    chain_id: str,
    sequence: str,
) -> dict[str, object]:
    mmcif_str = _single_chain_mmcif_string(structure_path, chain_id)
    L = len(str(sequence or "").strip())
    return {
        "templates": [
            {
                "mmcif": mmcif_str,
                "queryIndices": list(range(L)),
                "templateIndices": list(range(L)),
            }
        ]
    }


def _framework_template_homology_from_layout(
    framework_layout: dict[str, object],
) -> dict[str, object]:
    structure_path = pathlib.Path(str(framework_layout["structure_path"])).expanduser().resolve()
    kind = str(framework_layout["kind"]).strip().lower()

    if kind == "vhh":
        heavy_chain = str(framework_layout["heavy_chain"]).strip()
        heavy_seq = str(framework_layout["source_chain_sequences"]["heavy"])
        return _template_homology_from_structure(
            structure_path=structure_path,
            chain_id=heavy_chain,
            sequence=heavy_seq,
        )

    if kind != "scfv":
        raise ValueError(f"Unsupported framework structure kind for template homology: {kind!r}")

    heavy_chain = str(framework_layout["heavy_chain"]).strip()
    light_chain = str(framework_layout["light_chain"]).strip()
    source_chain_sequences = framework_layout["source_chain_sequences"]
    heavy_seq = str(source_chain_sequences["heavy"])
    light_seq = str(source_chain_sequences["light"])
    order = str(framework_layout["scfv_order"]).strip().lower()
    linker = str(framework_layout["scfv_linker"])

    if order == "vh_linker_vl":
        heavy_start = 0
        light_start = len(heavy_seq) + len(linker)
    elif order == "vl_linker_vh":
        light_start = 0
        heavy_start = len(light_seq) + len(linker)
    else:
        raise ValueError(f"Unsupported scFv order for template homology: {order!r}")

    templates = []
    for chain_id, sequence, query_start in (
        (heavy_chain, heavy_seq, heavy_start),
        (light_chain, light_seq, light_start),
    ):
        mmcif_str = _single_chain_mmcif_string(structure_path, chain_id)
        length = len(sequence)
        templates.append(
            {
                "mmcif": mmcif_str,
                "queryIndices": list(range(query_start, query_start + length)),
                "templateIndices": list(range(length)),
            }
        )
    return {"templates": templates}


def _binder_seq_has_ranged_loop_lengths(binder_seq: str) -> bool:
    tokens = [str(tok).strip() for tok in str(binder_seq).split() if str(tok).strip()]
    for token in tokens:
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            continue
        if "-" not in token:
            continue
        start_s, end_s = token.split("-", 1)
        if int(start_s) != int(end_s):
            return True
    return False


def _derive_target_loop_lengths_from_binder_seq(
    *,
    binder_seq: str,
    fixed_segments: list[str],
    loop_order: list[str],
) -> dict[str, int]:
    if len(fixed_segments) != len(loop_order) + 1:
        raise ValueError(
            "framework layout requires len(fixed_segments) == len(loop_order) + 1"
        )

    tokens = [str(tok).strip() for tok in str(binder_seq).split() if str(tok).strip()]
    range_tokens = [tok for tok in tokens if re.fullmatch(r"\d+(?:-\d+)?", tok)]
    if len(range_tokens) != len(loop_order):
        raise ValueError(
            "Custom framework protomer statement must contain exactly one fixed length token per loop"
        )

    result: dict[str, int] = {}
    for loop, token in zip(loop_order, range_tokens, strict=True):
        if "-" in token:
            start_s, end_s = token.split("-", 1)
        else:
            start_s = end_s = token
        start = int(start_s)
        end = int(end_s)
        if start != end:
            raise ValueError(
                "Custom framework homology preprocessing requires exact loop lengths; "
                f"got {loop}:{token}"
            )
        result[str(loop)] = start
    return result


def _prepare_framework_homology_payload(
    *,
    binder_seq: str,
    framework_layout: dict[str, object],
    framework_homology_source: Any | None,
) -> dict[str, object]:
    from astrolabe.core.msa_utils import (
        _load_framework_homology_source,
        _transform_antibody_msa,
        _transform_antibody_templates,
    )

    fixed_segments = [str(part) for part in framework_layout["fixed_segments"]]
    loop_order = [str(loop) for loop in framework_layout["loop_order"]]
    source_loop_lengths = {
        str(loop): int(framework_layout["source_loop_lengths"][loop])
        for loop in loop_order
    }
    target_loop_lengths = _derive_target_loop_lengths_from_binder_seq(
        binder_seq=binder_seq,
        fixed_segments=fixed_segments,
        loop_order=loop_order,
    )
    current_query_seq_parts: list[str] = []
    for idx, fixed_segment in enumerate(fixed_segments[:-1]):
        current_query_seq_parts.append(fixed_segment)
        current_query_seq_parts.append("A" * int(target_loop_lengths[loop_order[idx]]))
    current_query_seq_parts.append(fixed_segments[-1])
    current_query_seq = "".join(current_query_seq_parts)

    source_value = (
        framework_homology_source
        if framework_homology_source is not None
        else _framework_template_homology_from_layout(framework_layout)
    )
    source_data = _load_framework_homology_source(source_value)

    payload: dict[str, object] = {}
    if "unpairedMsa" in source_data:
        transformed = _transform_antibody_msa(
            current_seq=current_query_seq,
            source_msa=source_data.get("unpairedMsa"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if transformed is not None:
            payload["unpairedMsa"] = transformed
    if "pairedMsa" in source_data:
        transformed = _transform_antibody_msa(
            current_seq=current_query_seq,
            source_msa=source_data.get("pairedMsa"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if transformed is not None:
            payload["pairedMsa"] = transformed
    if "templates" in source_data:
        transformed = _transform_antibody_templates(
            raw_templates=source_data.get("templates"),
            fixed_segments=fixed_segments,
            loop_order=loop_order,
            source_loop_lengths=source_loop_lengths,
            target_loop_lengths=target_loop_lengths,
        )
        if transformed is not None:
            payload["templates"] = transformed
    if not payload:
        raise ValueError("Framework homology source did not produce any usable payload")
    return payload


def _load_af3_chain_dict_from_homology(homology_arg: str) -> dict[str, object]:
    from astrolabe.core.msa_utils import load_af3_chain_data

    file_str, *chain_parts = (homology_arg or "").split(":")
    if not file_str.endswith(".json"):
        raise ValueError(f"Not an AF3 JSON homology: {homology_arg!r}")
    if not chain_parts:
        raise ValueError(f"AF3 JSON homology must include chain id: {homology_arg!r}")
    chain_id = str(chain_parts[0]).strip()
    if not chain_id:
        raise ValueError(f"AF3 JSON homology has empty chain id: {homology_arg!r}")

    path = pathlib.Path(file_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"AF3 JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if "dialect" not in data:
        raise ValueError(f"{path} is not a valid AlphaFold3 JSON file (missing dialect)")
    _polymer_key, chain_data = load_af3_chain_data(data, chain_id)
    return chain_data


def _parse_domain_index_groups(raw_groups: Optional[list[str]]) -> list[list[int]]:
    groups = list(raw_groups or [])
    if len(groups) < 2:
        raise ValueError("Provide at least two --domain-indices groups")

    return [
        _parse_residue_list(group, flag_name="--domain-indices")
        for group in groups
    ]


def _parse_multidomain_domain_specs(
    *,
    raw_domain_indices: Optional[list[str]],
    raw_domain_hotspots: Optional[list[str]],
) -> tuple[list[dict[str, object]], list[int]]:
    domain_indices = _parse_domain_index_groups(raw_domain_indices)
    hotspot_groups_raw = list(raw_domain_hotspots or [])
    if hotspot_groups_raw and len(hotspot_groups_raw) != len(domain_indices):
        raise ValueError("Domain hotspot groups must match domain index groups")

    domains: list[dict[str, object]] = []
    combined_hotspots: list[int] = []
    seen_hotspots: set[int] = set()
    for idx, indices in enumerate(domain_indices):
        hotspots = (
            _parse_residue_list(hotspot_groups_raw[idx], flag_name="--domain-hotspots")
            if hotspot_groups_raw
            else list(indices)
        )
        index_set = set(indices)
        missing = [resi for resi in hotspots if resi not in index_set]
        if missing:
            raise ValueError(
                f"--domain-hotspots group {idx + 1} must be a subset of "
                f"--domain-indices group {idx + 1}, got {missing!r}"
            )
        domains.append(
            {
                "label": f"domain_{idx + 1}",
                "indices": list(indices),
                "hotspots": list(hotspots),
            }
        )
        for residue in hotspots:
            if residue in seen_hotspots:
                continue
            combined_hotspots.append(int(residue))
            seen_hotspots.add(int(residue))
    return domains, combined_hotspots


def _parse_fasta_records(msa_string: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_lines: list[str] = []
    for raw_line in str(msa_string or "").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip() or "query"
            seq_lines = []
            continue
        seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _format_fasta_records(records: list[tuple[str, str]]) -> str | None:
    if not records:
        return None
    return "\n".join(f">{header}\n{sequence}" for header, sequence in records)


def _template_to_raw_dict(template: Any) -> dict[str, object] | None:
    mapping = dict(getattr(template, "query_to_template_map", {}) or {})
    if not mapping:
        return None
    query_indices = sorted(int(key) for key in mapping.keys())
    template_indices = [int(mapping[key]) for key in query_indices]
    return {
        "mmcif": getattr(template, "mmcif", None),
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


def _merge_masked_msa_strings(sequence: str, msa_strings: list[str | None]) -> str | None:
    merged: list[tuple[str, str]] = []
    seen_rows: set[tuple[str, str]] = set()
    query_added = False
    for msa_string in msa_strings:
        if not msa_string:
            continue
        records = _parse_fasta_records(msa_string)
        if not records:
            continue
        query_header, _query_sequence = records[0]
        if not query_added:
            merged.append((query_header, sequence))
            query_added = True
        for header, row_sequence in records[1:]:
            key = (header, row_sequence)
            if key in seen_rows:
                continue
            merged.append((header, row_sequence))
            seen_rows.add(key)
    return _format_fasta_records(merged)


def _split_full_homology_by_domains(
    *,
    sequence: str,
    homology_source: object | None,
    domains: list[dict[str, object]],
) -> object | None:
    if homology_source is None:
        return None

    from astrolabe.core.msa_utils import parse_homology_arg

    base_sequence = _strip_fixed_prefix(sequence)
    domain_results: list[dict[str, object]] = []
    for domain in domains:
        indices = [int(residue) - 1 for residue in list(domain["indices"])]
        domain_results.append(
            parse_homology_arg(
                base_sequence,
                indices,
                homology_source,
            )
        )

    unpaired_msa = _merge_masked_msa_strings(
        base_sequence,
        [result.get("unpairedMsa") for result in domain_results],
    )
    paired_msa = _merge_masked_msa_strings(
        base_sequence,
        [result.get("pairedMsa") for result in domain_results],
    )
    templates: list[dict[str, object]] = []
    for result in domain_results:
        for template in list(result.get("templates") or []):
            raw_template = _template_to_raw_dict(template)
            if raw_template is not None:
                templates.append(raw_template)

    out: dict[str, object] = {}
    if unpaired_msa:
        out["unpairedMsa"] = unpaired_msa
    if paired_msa:
        out["pairedMsa"] = paired_msa
    if templates:
        out["templates"] = templates
    return out or None


def _parse_full_homology_payload(sequence: str, homology_source: str) -> dict[str, object]:
    from astrolabe.core.msa_utils import parse_homology_arg

    base_sequence = _strip_fixed_prefix(sequence)
    parsed = parse_homology_arg(
        base_sequence,
        list(range(len(base_sequence))),
        homology_source,
    )
    out: dict[str, object] = {}
    for key in ("unpairedMsa", "pairedMsa"):
        value = parsed.get(key)
        if value:
            out[key] = value
    templates = []
    for template in list(parsed.get("templates") or []):
        raw_template = _template_to_raw_dict(template)
        if raw_template is not None:
            templates.append(raw_template)
    if templates:
        out["templates"] = templates
    return out


def _resolve_multidomain_target_inputs(
    *,
    target_seq: Optional[str],
    target_homology: Optional[str],
    target_structure: Optional[str],
    target_chain: Optional[str],
    domains: list[dict[str, object]],
    domain_split_mode: str = "full",
) -> tuple[str, object | None]:
    resolved_target_seq, resolved_full_homology = _resolve_target_inputs(
        target_seq=target_seq,
        target_homology=target_homology,
        target_structure=target_structure,
        target_chain=target_chain,
    )
    mode = str(domain_split_mode or "full").strip().lower()
    if mode == "template":
        if isinstance(resolved_full_homology, str):
            return resolved_target_seq, _parse_full_homology_payload(
                resolved_target_seq,
                resolved_full_homology,
            )
        return resolved_target_seq, resolved_full_homology
    if mode != "full":
        raise ValueError(
            f"Unsupported --domain-split-mode {domain_split_mode!r}; expected full|template"
        )
    resolved_split_homology = _split_full_homology_by_domains(
        sequence=resolved_target_seq,
        homology_source=resolved_full_homology,
        domains=domains,
    )
    return resolved_target_seq, resolved_split_homology


def _homology_has_nonempty_templates(homology_arg: str) -> bool:
    file_str = str(homology_arg or "").split(":", 1)[0]
    if file_str.endswith(".a3m"):
        return False
    if file_str.endswith(".json"):
        chain_data = _load_af3_chain_dict_from_homology(homology_arg)
        templates = chain_data.get("templates", [])
        return bool(list(templates or []))
    raise ValueError(
        f"Cannot inspect homology templates for {homology_arg!r} "
        "(expected .json:<chain> or .a3m)"
    )


def _merge_homology_with_structure_template(
    *,
    homology_arg: str,
    structure_template_homology: dict[str, object],
) -> dict[str, object]:
    file_str = str(homology_arg or "").split(":", 1)[0]
    templates = structure_template_homology.get("templates", None)
    if templates is None:
        raise ValueError("Missing templates when merging homology with structure template")
    if file_str.endswith(".a3m"):
        return {
            "unpairedMsa": str(homology_arg),
            "templates": templates,
        }
    if file_str.endswith(".json"):
        return {
            "unpairedMsa": str(homology_arg),
            "pairedMsa": str(homology_arg),
            "templates": templates,
        }
    raise ValueError(
        f"Cannot merge homology with structure template for {homology_arg!r} "
        "(expected .json:<chain> or .a3m)"
    )


def _resolve_target_inputs(
    *,
    target_seq: Optional[str],
    target_homology: Optional[str],
    target_structure: Optional[str],
    target_chain: Optional[str],
) -> tuple[str, object | None]:
    seq_sources: list[tuple[str, str]] = []

    target_seq_arg = str(target_seq or "").strip()
    target_homology_arg = str(target_homology or "").strip()
    target_structure_arg = str(target_structure or "").strip()
    target_chain_arg = str(target_chain or "").strip()

    if target_chain_arg and not target_structure_arg:
        raise ValueError("--target-chain requires --target-structure")

    if target_seq_arg:
        seq_sources.append(("target-seq", _strip_fixed_prefix(target_seq_arg)))

    homology_seq: Optional[str] = None
    if target_homology_arg:
        homology_seq = _infer_target_seq_from_homology(target_homology_arg)
        seq_sources.append(("target-homology", homology_seq))

    structure_path: Optional[pathlib.Path] = None
    structure_seq: Optional[str] = None
    if target_structure_arg:
        structure_path, structure_seq = _sequence_from_structure_chain(
            target_structure_arg, target_chain_arg
        )
        seq_sources.append(("target-structure", structure_seq))

    if not seq_sources:
        raise ValueError(
            "Provide at least one of: --target-seq, --target-homology, or "
            "--target-structure + --target-chain"
        )

    base_name, base_seq = seq_sources[0]
    for name, seq in seq_sources[1:]:
        if str(seq).strip() != str(base_seq).strip():
            raise ValueError(
                "Mismatch between target inputs. "
                f"{base_name}_len={len(str(base_seq).strip())} "
                f"{name}_len={len(str(seq).strip())}"
            )

    resolved_target_seq = _normalize_fixed_sequence(base_seq)

    resolved_homology: object | None = None
    if target_homology_arg and structure_path is not None and structure_seq is not None:
        structure_template = _template_homology_from_structure(
            structure_path=structure_path,
            chain_id=target_chain_arg,
            sequence=structure_seq,
        )
        if _homology_has_nonempty_templates(target_homology_arg):
            resolved_homology = target_homology_arg
        else:
            resolved_homology = _merge_homology_with_structure_template(
                homology_arg=target_homology_arg,
                structure_template_homology=structure_template,
            )
    elif target_homology_arg:
        resolved_homology = target_homology_arg
    elif structure_path is not None and structure_seq is not None:
        resolved_homology = _template_homology_from_structure(
            structure_path=structure_path,
            chain_id=target_chain_arg,
            sequence=structure_seq,
        )
    else:
        resolved_homology = None

    return resolved_target_seq, resolved_homology


def _set_stage_evaluator(cfg: Any, evaluator_alias: Optional[str]) -> None:
    if not evaluator_alias:
        return
    if not hasattr(cfg, "stages") or cfg.stages is None:
        return
    with open_dict(cfg):
        for stage_name in cfg.stages.keys():
            stage = cfg.stages[stage_name]
            if isinstance(stage, DictConfig) and "evaluator" in stage:
                stage.evaluator = str(evaluator_alias)


def _set_stage_oligos(cfg: Any, oligos: list[str]) -> None:
    if not oligos:
        return
    if not hasattr(cfg, "stages") or cfg.stages is None:
        return
    with open_dict(cfg):
        for stage_name in cfg.stages.keys():
            stage = cfg.stages[stage_name]
            if isinstance(stage, DictConfig) and "oligo" in stage:
                stage.oligo = list(oligos)


def _parse_residue_list(raw: str, *, flag_name: str) -> list[int]:
    spec = str(raw or "").strip()
    if not spec:
        raise ValueError(f"{flag_name} is required")

    ordered: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            parts = [p.strip() for p in token.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"{flag_name} has invalid range {token!r}; expected 'start-end'"
                )
            start = int(parts[0])
            end = int(parts[1])
            if start <= 0 or end <= 0:
                raise ValueError(f"{flag_name} residues must be positive, got {token!r}")
            if end < start:
                raise ValueError(
                    f"{flag_name} range must be ascending, got {token!r}"
                )
            values = range(start, end + 1)
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"{flag_name} residues must be positive, got {token!r}")
            values = [value]

        for value in values:
            if value not in seen:
                ordered.append(int(value))
                seen.add(int(value))

    if not ordered:
        raise ValueError(f"{flag_name} produced no residues from {raw!r}")
    return ordered


def _format_hotspot_target(chain_id: str, residues: list[int]) -> str:
    return f"{chain_id}:{','.join(str(v) for v in residues)}"


def _format_hotspot_select(chain_id: str, residues: list[int]) -> str:
    return f"{chain_id}_1:{','.join(str(v) for v in residues)}"


def _cmd_design_multitarget_binder(ns: argparse.Namespace) -> int:
    config_name = _MULTITARGET_BINDER_VARIANT_TO_CONFIG[str(ns.variant)]
    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name=config_name,
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    binder_statement = _resolve_initial_binder_statement(ns)
    target_b_seq, target_b_homology_cfg = _resolve_target_inputs(
        target_seq=getattr(ns, "target_b_seq", None),
        target_homology=getattr(ns, "target_b_homology", None),
        target_structure=getattr(ns, "target_b_structure", None),
        target_chain=getattr(ns, "target_b_chain", None),
    )
    target_c_seq, target_c_homology_cfg = _resolve_target_inputs(
        target_seq=getattr(ns, "target_c_seq", None),
        target_homology=getattr(ns, "target_c_homology", None),
        target_structure=getattr(ns, "target_c_structure", None),
        target_chain=getattr(ns, "target_c_chain", None),
    )
    target_b_hotspots = _parse_residue_list(
        ns.target_b_hotspots,
        flag_name="--target-b-hotspots",
    )
    target_c_hotspots = _parse_residue_list(
        ns.target_c_hotspots,
        flag_name="--target-c-hotspots",
    )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = binder_statement
        cfg.seqs.B = target_b_seq
        cfg.seqs.C = target_c_seq
        cfg.homologies.B = target_b_homology_cfg
        cfg.homologies.C = target_c_homology_cfg
        if ns.repeats is not None:
            cfg.repeats = int(ns.repeats)
        if ns.steps is not None:
            cfg.steps = int(ns.steps)

        cfg.binding_site.AB.target = _format_hotspot_target("B", target_b_hotspots)
        cfg.binding_site.AC.target = _format_hotspot_target("C", target_c_hotspots)

        cfg.loss.loss_terms.hotspot_target_b_global.select_expression = _format_hotspot_select(
            "B", target_b_hotspots
        )
        cfg.loss.loss_terms.hotspot_target_c_global.select_expression = _format_hotspot_select(
            "C", target_c_hotspots
        )

        cfg.multitarget.pair_name = str(ns.pair_name)
        cfg.multitarget.shared_hotspot_manifest = str(ns.shared_hotspot_manifest)
        cfg.multitarget.shared_hotspots.B = list(target_b_hotspots)
        cfg.multitarget.shared_hotspots.C = list(target_c_hotspots)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_design_multidomain_binder(ns: argparse.Namespace) -> int:
    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_multidomain_binder_v1",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    binder_statement = _resolve_initial_binder_statement(ns)
    domains, combined_hotspots = _parse_multidomain_domain_specs(
        raw_domain_indices=getattr(ns, "domain_indices", None),
        raw_domain_hotspots=getattr(ns, "domain_hotspots", None),
    )
    target_seq, target_homology_cfg = _resolve_multidomain_target_inputs(
        target_seq=getattr(ns, "target_seq", None),
        target_homology=getattr(ns, "target_homology", None),
        target_structure=getattr(ns, "target_structure", None),
        target_chain=getattr(ns, "target_chain", None),
        domains=domains,
        domain_split_mode=getattr(ns, "domain_split_mode", "full"),
    )

    variant = str(ns.variant)
    target_complex_evaluator = "Alphafold3" if variant == "af3" else "Protenix"

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = binder_statement
        cfg.seqs.B = target_seq
        cfg.homologies.B = target_homology_cfg
        if ns.repeats is not None:
            cfg.repeats = int(ns.repeats)
        if ns.steps is not None:
            cfg.steps = int(ns.steps)
        cfg.binding_site.target = _format_hotspot_target("B", combined_hotspots)
        cfg.loss.loss_terms.hotspot_target_global.select_expression = _format_hotspot_select(
            "B", combined_hotspots
        )
        cfg.multidomain_target.target_name = str(ns.target_name)
        cfg.multidomain_target.full_homology = (
            str(getattr(ns, "target_homology", "") or "").strip() or None
        )
        cfg.multidomain_target.domain_split_mode = str(
            getattr(ns, "domain_split_mode", "full")
        )
        cfg.multidomain_target.domains = list(domains)
        cfg.stages.target_complex.evaluator = target_complex_evaluator
    _apply_common_one_shot_config_overrides(cfg, ns)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


_OLIGOMER_DESIGN_SHAPE_COMPLETIONS = tuple(
    [f"C{i}" for i in range(2, 13)] + ["D2", "D3", "D4"]
)
_DIHEDRAL_OLIGOMER_CONFIGS = {
    "D2": "benchmark_dihedral_d2_dplm_protenix_af2_rpxdock_v1",
    "D3": "benchmark_dihedral_d3_dplm_protenix_af2_rpxdock_v1",
    "D4": "benchmark_dihedral_d4_dplm_protenix_af2_rpxdock_v1",
}


def _oligomer_design_shape(raw_shape: str) -> tuple[str, int, str, str]:
    shape = str(raw_shape or "").strip().upper()
    match = re.fullmatch(r"([CD])(\d+)", shape)
    if match is None:
        raise ValueError(
            f"Unsupported oligomer shape {raw_shape!r}; expected one of: "
            "C<N> for N>=2 or D2-D4"
        )
    family = match.group(1)
    order = int(match.group(2))
    if family == "C" and order < 2:
        raise ValueError(f"Unsupported oligomer shape {raw_shape!r}; C order must be >= 2")
    if family == "D" and shape not in _DIHEDRAL_OLIGOMER_CONFIGS:
        raise ValueError(
            f"Unsupported oligomer shape {raw_shape!r}; dihedral presets support D2-D4"
        )
    if family == "C":
        return family, order, "A" * order, "benchmark_oligomer_morphology_cyclic_v1"
    return family, order, "A" * (2 * order), _DIHEDRAL_OLIGOMER_CONFIGS[shape]


def _cmd_design_oligomer(ns: argparse.Namespace) -> int:
    family, order, oligo, config_name = _oligomer_design_shape(ns.shape)
    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name=config_name,
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    protein_statement = _resolve_initial_a_statement(ns, length_attr="protein_len")
    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = protein_statement
        cfg.homologies.A = None
        if "oligomer" in cfg.stages:
            cfg.stages.oligomer.oligo = [oligo]
        if "default" in cfg.stages and family == "C":
            cfg.stages.default.oligo = ["A"]
        symmetry = cfg.predocker.predocker_terms.symmetry
        symmetry.order = int(order)
        symmetry.oligo = oligo
        if family == "C":
            symmetry.mode = "cyclic_oligomer"
            symmetry.backend = "rpxdock"
        else:
            symmetry.mode = "dihedral_oligomer"
            symmetry.backend = "rpxdock"
    _apply_common_one_shot_config_overrides(cfg, ns)
    _set_stage_evaluator(cfg, ns.evaluator)

    print("Protomer mapping: A (homooligomer)")
    print(f"Shape: {family}{order}")
    print(f"Oligomer: {oligo}")
    print(f"Template: {config_name}")

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _protomer_letter_names(copies: int) -> list[str]:
    if copies <= 0:
        raise ValueError("--copies must be > 0")
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if copies > len(letters):
        raise ValueError(f"--copies must be <= {len(letters)} for one-letter protomer ids")
    return letters[:copies]


def _build_internal_repeat_statement(unit_spec: str, copies: int) -> str:
    if copies <= 1:
        raise ValueError("--copies must be > 1 for repeat design commands")
    alias = "repeat_1"
    return " ".join([f"{unit_spec}@{alias}"] + [alias] * (copies - 1))


def _parse_residue_ranges(raw: str, *, length: int) -> set[int]:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty residue range specification")
    positions: set[int] = set()
    for token in s.split(","):
        part = token.strip()
        if not part:
            raise ValueError(f"Invalid empty range in {raw!r}")
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo <= 0 or hi < lo or hi > length:
                raise ValueError(f"Invalid range {part!r} for sequence length {length}")
            positions.update(range(lo - 1, hi))
        else:
            pos = int(part)
            if pos <= 0 or pos > length:
                raise ValueError(f"Invalid residue index {part!r} for sequence length {length}")
            positions.add(pos - 1)
    return positions


def _build_heterooligomer_seqs(
    sequence: str,
    *,
    interface_ranges: str,
    protomer_names: list[str],
) -> dict[str, str]:
    seq = _strip_fixed_prefix(sequence).strip().upper()
    if not seq:
        raise ValueError("Empty monomer sequence for heterooligomer refinement")
    interface_positions = _parse_residue_ranges(interface_ranges, length=len(seq))
    if not interface_positions:
        raise ValueError("--interface must select at least one residue")
    if len(interface_positions) == len(seq):
        raise ValueError("--interface cannot cover the entire protomer; shared core would be empty")

    descriptors: list[tuple[bool, int, str]] = []
    start = 0
    shared_idx = 1
    interface_idx = 1
    current_is_interface = 0 in interface_positions
    for idx in range(1, len(seq) + 1):
        next_is_interface = idx < len(seq) and idx in interface_positions
        if idx < len(seq) and next_is_interface == current_is_interface:
            continue
        chunk = seq[start:idx]
        if current_is_interface:
            descriptors.append((True, interface_idx, chunk))
            interface_idx += 1
        else:
            descriptors.append((False, shared_idx, chunk))
            shared_idx += 1
        start = idx
        current_is_interface = next_is_interface

    seqs: dict[str, str] = {}
    ref_proto = protomer_names[0]
    for proto in protomer_names:
        tokens: list[str] = []
        for is_interface, seg_idx, chunk in descriptors:
            if is_interface:
                tokens.append(f"{chunk}@itf_{proto}_{seg_idx}")
            else:
                alias = f"shared_{seg_idx}"
                tokens.append(f"{chunk}@{alias}" if proto == ref_proto else alias)
        seqs[proto] = " ".join(tokens)
    return seqs


def _mutator_supports_predocker(mutator_type: str) -> bool:
    return str(mutator_type or "").strip().lower() in {
        "ligandmpnn",
        "esmif",
        "carbondesign",
        "nampnn",
    }


def _validate_symmetry_predocker_backend(mode: str, backend: str | None) -> None:
    if backend is None:
        return
    mode_norm = str(mode or "").strip()
    backend_norm = str(backend).strip().lower()
    allowed_by_mode = {
        "pseudocycle": {"legacy", "rfdiffusion", "rpxdock"},
        "tandem_repeat": {"legacy"},
        "heterooligomer_shared_core": {"legacy"},
        "cyclic_oligomer": {"rfdiffusion", "rpxdock"},
        "dihedral_oligomer": {"rfdiffusion", "rpxdock"},
    }
    allowed = allowed_by_mode.get(mode_norm)
    if allowed is None:
        raise ValueError(f"Unsupported symmetry canonicalizer mode {mode_norm!r}.")
    if backend_norm not in allowed:
        raise ValueError(f"backend={backend_norm!r} is not supported for mode={mode_norm!r}.")


def _enable_symmetry_predocker(
    cfg: Any,
    *,
    mode: str,
    oligo: str,
    included_protomer: str | None = None,
    repeat_canonicalizer_mode: str | None = None,
    backend: str | None = None,
    order: int | None = None,
    interface_selection: str | None = None,
    interface: str | None = None,
) -> None:
    if order is not None and int(order) <= 0:
        raise ValueError("Symmetry order must be > 0.")
    _validate_symmetry_predocker_backend(mode, backend)
    if order is not None and str(backend or "").strip().lower() == "legacy":
        raise ValueError("Symmetry order is not supported for backend='legacy'.")
    interface_selection_norm = (
        str(interface_selection).strip().lower() if interface_selection is not None else None
    )
    if interface_selection_norm is not None:
        if interface_selection_norm not in {"auto", "manual"}:
            raise ValueError("--symmetry-interface-selection must be auto|manual")
        if interface_selection_norm == "manual" and not str(interface or "").strip():
            raise ValueError(
                "--symmetry-interface-selection manual requires --symmetry-interface"
            )
    predocker_cfg: dict[str, object] = {
        "_type": "symmetry_canonicalizer",
        "mode": str(mode),
        "oligo": str(oligo),
    }
    if backend is not None:
        predocker_cfg["backend"] = str(backend)
    if order is not None:
        predocker_cfg["order"] = int(order)
    if included_protomer is not None:
        predocker_cfg["included_protomer"] = str(included_protomer)
    if repeat_canonicalizer_mode is not None:
        predocker_cfg["repeat_canonicalizer_mode"] = str(repeat_canonicalizer_mode)
    if interface_selection_norm is not None:
        predocker_cfg["interface_selection"] = interface_selection_norm
    if interface is not None:
        predocker_cfg["interface"] = str(interface)

    with open_dict(cfg):
        cfg.predocker.predocker_terms.symmetry = predocker_cfg
        attached = 0
        for alias in cfg.mutator.mutator_terms.keys():
            term = cfg.mutator.mutator_terms[alias]
            if term is None:
                continue
            if _mutator_supports_predocker(term.get("_type", "")):
                term.predocker = ["symmetry"]
                attached += 1
        if attached == 0:
            raise ValueError("No predocker-capable mutator terms found to attach symmetry_canonicalizer.")


def _maybe_dump_config(cfg: Any, out_path: Optional[str]) -> None:
    if not out_path:
        return
    path = pathlib.Path(out_path).expanduser().resolve()
    header = "# @package _global_\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(header + OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")
    os.replace(str(tmp), str(path))
    print(f"Wrote: {path}")


def _run_design_preset(
    ns: argparse.Namespace,
    *,
    config_name: str,
    mutate_config: design_preset_builder.MutateConfig,
) -> int:
    return design_preset_builder.run_design_preset(
        ns,
        config_name=config_name,
        mutate_config=mutate_config,
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _apply_common_one_shot_config_overrides(cfg: Any, ns: argparse.Namespace) -> None:
    with open_dict(cfg):
        if getattr(ns, "repeats", None) is not None:
            cfg.repeats = int(ns.repeats)
        if getattr(ns, "seed", None) is not None:
            cfg.seed = int(ns.seed)
        if getattr(ns, "steps", None) is not None:
            cfg.steps = int(ns.steps)


def _add_common_one_shot_run_args(
    parser: argparse.ArgumentParser,
    *,
    task_name_required: bool = True,
    out_dir_required: bool = True,
) -> None:
    parser.add_argument(
        "-t",
        "--task-name",
        required=task_name_required,
        help="Run name (creates out_dir/<task-name>/###_run).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=out_dir_required,
        help="Output root directory (task subdir is created under it).",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        "--repeats",
        dest="repeats",
        type=int,
        default=None,
        help="Override number of runs/repeats (default: template value).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Repeat 1 uses this seed; later repeats use seed+index.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override steps (default: template value).",
    )
    parser.add_argument(
        "--dump-config",
        default=None,
        help="Optional: write the filled overlay YAML to this path.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print composed config before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running.",
    )


def _add_protein_binder_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    _add_initial_binder_args(
        parser,
        length_flags=("--binder-len",),
        len_help='Binder length range, e.g. "80-120" or "100".',
    )
    parser.add_argument("--target-seq", default=None, help="Target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    parser.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    parser.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    parser.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_idr_binder_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    _add_initial_binder_args(
        parser,
        length_flags=("--binder-len",),
        len_help='Binder length range, e.g. "80-120" or "100".',
    )
    parser.add_argument("--target-seq", default=None, help="IDR target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    parser.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    parser.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    parser.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_peptide_binder_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    _add_initial_binder_args(
        parser,
        length_flags=("--peptide-len",),
        len_help='Peptide length range, e.g. "8-16" or "12".',
        seq_help=(
            "Mutable initial peptide binder sequence for chain A. Writes seqs.A "
            "as this bare sequence; use --initial-binder-expr for fixed segments "
            "or multi-segment statements."
        ),
    )
    parser.add_argument("--target-seq", default=None, help="Target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    parser.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    parser.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    parser.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    parser.add_argument("--binding-site-target", default=None, help="Optional epitope spec on target chain, e.g. 'B:64-68'. When set, inject binding_site constraints into the run config.")
    parser.add_argument("--nonstd-map", default=None, help="Path to YAML/JSON nonstandard residue alias map. Accepts either top-level `aliases` or top-level `nonstandard_residues`.")
    parser.add_argument("--cyclize-head-to-tail", action="store_true", help="Make the peptide binder (protomer A) head-to-tail cyclic by adding a covalent bond (A:1@N <-> A:-1@C).")
    parser.add_argument("--cyclize-disulfide", action="append", default=None, help="Add a disulfide-like covalent bond within protomer A. Repeatable. Format: '<pos1>-<pos2>' (1-indexed, or -1/nterm/cterm).")
    parser.add_argument("--cyclize-bond", action="append", nargs=2, default=None, metavar=("ATOM1", "ATOM2"), help="Add a generic covalent bond. Repeatable. Format: '<proto>:<pos>@<atom>' for each endpoint (pos is 1-indexed, or -1/nterm/cterm).")
    parser.add_argument("--cyclize-bond-typed", action="append", default=None, metavar="SPEC", help="Add a typed covalent bond. Repeatable. Format: '<ATOM1>,<ATOM2>,<ORDER>[,<TARGET_DISTANCE>]'. Example: 'A:2@CA,A:7@CB,double,1.34'.")
    cli_defs.add_cyclic_encoding_argument(parser)
    closure_profile_action = parser.add_argument("--closure-profile", default=None, help="Cyclic closure loss preset: strict|min|mean (default when cyclized: strict).")
    cli_defs.add_completion_choices(closure_profile_action, tuple(_CLOSURE_PROFILE_PRESETS))
    parser.add_argument("--closure-weight", type=float, default=None, help="Weight for cyclic_closure_norm in weighted_score_expression (default when cyclized: 0.3).")
    parser.add_argument("--closure-gate", type=float, default=None, help="Acceptance gate threshold for cyclic_closure_norm in [0,1] (default when cyclized: 0.90).")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_aptamer_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    _add_initial_binder_args(
        parser,
        length_flags=("--aptamer-len",),
        len_help='RNA aptamer length range, e.g. "28-36" or "31".',
        seq_help=(
            "Mutable initial RNA aptamer sequence for chain A. Writes seqs.A as "
            "this bare sequence; use --initial-binder-expr for fixed segments or "
            "multi-segment statements."
        ),
        expr_help=(
            "Advanced RNA aptamer statement written directly to seqs.A, e.g. "
            "'AUGC 28-36' or 'input:A'."
        ),
    )
    cli_defs.add_aptamer_type_argument(parser)
    parser.add_argument("--target-seq", default=None, help="Target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    parser.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    parser.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    parser.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_antibody_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser, task_name_required=False, out_dir_required=False)
    parser.add_argument("--binder-seq", required=False, help="Binder protomer statement for chain A (framework fixed with '+'; CDR/paratope mutable).")
    parser.add_argument("--framework", default=None, help="Use a built-in antibody/scFv framework preset (sequence-only; canonical name or alias). Use --list-frameworks to view names.")
    parser.add_argument("--framework-pdb", default=None, help="Build binder framework from a Chothia-numbered structure (.pdb/.cif[.gz]).")
    parser.add_argument("--framework-heavy-chain", default=None, help="Heavy-chain ID in --framework-pdb (required for explicit vhh/scfv modes).")
    parser.add_argument("--framework-light-chain", default=None, help="Light-chain ID in --framework-pdb (required for scFv mode).")
    framework_kind_action = parser.add_argument("--framework-kind", default="auto", help="Framework kind for --framework-pdb: auto|vhh|scfv (default: auto).")
    cli_defs.add_completion_choices(framework_kind_action, _FRAMEWORK_KIND_COMPLETION_CHOICES)
    scfv_order_action = parser.add_argument("--scfv-order", default="vh_linker_vl", help="scFv chain order for --framework-pdb: vh_linker_vl|vl_linker_vh.")
    cli_defs.add_completion_choices(scfv_order_action, _SCFV_ORDER_COMPLETION_CHOICES)
    parser.add_argument("--scfv-linker", default="GGGGSGGGGSGGGGS", help="scFv linker sequence for --framework-pdb (default: (GGGGS)3).")
    parser.add_argument("--framework-homology", default=None, help="Binder framework homology resource for chain A. Built-in --framework presets use repo-managed defaults unless this overrides them; --framework-pdb uses structure-derived templates when omitted and uses this source when provided. Supports .json:<chain>, .a3m, or YAML/JSON homology spec files.")
    parser.add_argument("--cdr-lengths", default=None, help="Optional loop length overrides, e.g. 'H1:7-11,H2:6-10,L1:8-13'. Supports partial overrides; requires --framework or --framework-pdb.")
    parser.add_argument("--list-frameworks", action="store_true", help="Print built-in antibody/scFv framework preset names (+aliases) and exit.")
    parser.add_argument("--target-seq", default=None, help="Target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    parser.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    parser.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    parser.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    parser.add_argument("--binding-site-target", default=None, help="Optional epitope spec on target chain, e.g. 'B:64-68'. When set, inject binding_site constraints.")
    parser.add_argument("--binding-site-binder", action="append", default=None, help="Optional repeatable paratope spec on binder chain A, e.g. 'A:26-35'. If omitted, pocket-only binding_site is used with binder='A'.")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_refine_ligand_common_args(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    parser.add_argument("--structure", required=True, help="Input structure (.pdb/.cif[.gz]).")
    parser.add_argument("--protein-chain", required=True, help="Chain ID of the protein in the input structure.")
    parser.add_argument("--ligand", required=True, help="Ligand identifier: CCD code (e.g. FAD) or SMILES.")
    ligand_type_action = parser.add_argument("--ligand-type", default="ccd", help="Ligand type: ccd|smiles (default: ccd; CCD keeps standard residue naming from the CCD definition).")
    cli_defs.add_completion_choices(ligand_type_action, _LIGAND_TYPE_COMPLETION_CHOICES)
    parser.add_argument("--ligand-resname", default=None, help="Optional Rosetta params name override for ligand entity B (e.g. LIG or 9TP). This only renames the ligand entry; it does not switch CCD vs SMILES chemistry parsing.")
    parser.add_argument("--protein-homology", default=None, help="Optional protein homology (AF3 JSON path + chain, or .a3m).")
    parser.add_argument(
        "--protein-template-from-structure",
        action="store_true",
        help="If no --protein-homology, embed protein chain structure as an AF3 template.",
    )
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_refine_channel_common_args(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    parser.add_argument("--structure", required=True, help="Input structure (.pdb/.cif[.gz]) that contains the protein and ligand.")
    parser.add_argument("--protein-chain", required=True, help="Protein chain ID in --structure.")
    parser.add_argument(
        "--ligand-residue",
        required=True,
        help="Ligand residue selector from --structure: '<chain>:<resname>:<resseq>[:<icode>]'.",
    )
    parser.add_argument(
        "--mutable-positions",
        required=True,
        help="Path to mutable positions file for the protein chain.",
    )
    parser.add_argument(
        "--spheres",
        required=True,
        help="Reference spheres PDB used for HOLE rescanning.",
    )
    parser.add_argument(
        "--msa",
        default=None,
        help="Optional protein MSA (.a3m) for the mutable chain.",
    )
    parser.add_argument(
        "--ligand-conformer",
        default=None,
        help="Optional ligand conformer override path. Default uses the ligand residue from --structure.",
    )


def _chemical_component_id_from_cif(path: pathlib.Path) -> str | None:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".cif" not in suffixes and ".mmcif" not in suffixes:
        return None

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif = MMCIF2Dict(str(path))
    value = cif.get("_chem_comp.id")
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    component_id = str(value).strip()
    return component_id or None


def _add_refine_oligomer_common_args(parser: argparse.ArgumentParser) -> None:
    _add_common_one_shot_run_args(parser)
    parser.add_argument("--seq", default=None, help="Starting protein sequence for protomer A (mutable).")
    parser.add_argument("--protein-len", default=None, help='Alternative: random length range for protomer A, e.g. "150-150" or "120-160".')
    parser.add_argument("--structure", default=None, help="Alternative: input structure (.pdb/.cif[.gz]) to extract chain sequence.")
    parser.add_argument("--chain", default=None, help="Chain ID to extract as protomer A (required with --structure).")
    parser.add_argument("--oligo", default=None, help='Comma-separated oligomer strings, e.g. "AA,AAA" or "AAAAA".')
    parser.add_argument("--copies", type=int, default=None, help="Shorthand for a single homo-oligomer: repeats of 'A' (e.g. 5 -> AAAAA).")
    parser.add_argument("--homology", default=None, help="Optional homology for protomer A (AF3 JSON path + chain, or .a3m).")
    parser.add_argument(
        "--template-from-structure",
        action="store_true",
        help="If no --homology, embed the extracted chain structure as an AF3 template.",
    )
    parser.add_argument("--symmetry-predocker", action="store_true", help="Attach symmetry_canonicalizer to predocker-capable mutators.")
    parser.add_argument(
        "--symmetry-predocker-backend",
        default="rfdiffusion",
        choices=cli_defs.SYMMETRY_PREDOCKER_BACKEND_CHOICES,
        help="Backend used by symmetry_canonicalizer (default: rfdiffusion).",
    )
    parser.add_argument("--symmetry-order", type=int, default=None, help="Optional explicit symmetry order for symmetry_canonicalizer.")
    parser.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")


def _add_symmetry_interface_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--symmetry-interface-selection",
        default=None,
        choices=cli_defs.SYMMETRY_INTERFACE_SELECTION_CHOICES,
        help=(
            "Optional interface-selection mode for symmetry_canonicalizer. "
            "auto uses all repeat residues; manual uses --symmetry-interface ranges."
        ),
    )
    parser.add_argument(
        "--symmetry-interface",
        default=None,
        help="Manual 1-index residue ranges for symmetry fit residues, e.g. 4-8,15.",
    )


def _cmd_design_binder_template(ns: argparse.Namespace, *, config_name: str) -> int:
    def mutate_config(cfg: Any) -> None:
        binder_statement = _resolve_initial_binder_statement(ns)
        target_seq, target_homology_cfg = _resolve_target_inputs(
            target_seq=ns.target_seq,
            target_homology=ns.target_homology,
            target_structure=getattr(ns, "target_structure", None),
            target_chain=getattr(ns, "target_chain", None),
        )

        with open_dict(cfg):
            cfg.task_name = ns.task_name
            cfg.out_dir = ns.out_dir
            cfg.seqs.A = binder_statement
            cfg.seqs.B = target_seq
            cfg.homologies.B = target_homology_cfg
        _apply_common_one_shot_config_overrides(cfg, ns)
        _set_stage_evaluator(cfg, ns.evaluator)

    return _run_design_preset(
        ns,
        config_name=config_name,
        mutate_config=mutate_config,
    )


def _cmd_bench_binder_v1_4(ns: argparse.Namespace) -> int:
    return _cmd_design_binder_template(ns, config_name="benchmark_binder_v1_4")


def _cmd_design_idr_binder(ns: argparse.Namespace) -> int:
    return _cmd_design_binder_template(ns, config_name="benchmark_binder_idr_v1")


def _cmd_bench_peptide_binder_v1(ns: argparse.Namespace) -> int:
    def mutate_config(cfg: Any) -> None:
        peptide_statement = _resolve_initial_a_statement(ns, length_attr="peptide_len")
        target_seq, target_homology_cfg = _resolve_target_inputs(
            target_seq=ns.target_seq,
            target_homology=ns.target_homology,
            target_structure=getattr(ns, "target_structure", None),
            target_chain=getattr(ns, "target_chain", None),
        )

        with open_dict(cfg):
            cfg.task_name = ns.task_name
            cfg.out_dir = ns.out_dir
            cfg.seqs.A = peptide_statement
            cfg.seqs.B = target_seq
            cfg.homologies.B = target_homology_cfg
        _apply_common_one_shot_config_overrides(cfg, ns)

        binding_site_target = (ns.binding_site_target or "").strip()
        if binding_site_target:
            with open_dict(cfg):
                cfg.binding_site = {
                    "binder": "A",
                    "target": binding_site_target,
                    "mode": "pocket",
                    "max_distance": 12.0,
                }
                cfg.loss.loss_terms.binding_site_allowed_min_dist = {
                    "_type": "binding_site_allowed_min_dist"
                }
                base_expr = str(cfg.stages.default.weighted_score_expression)
                cfg.stages.default.weighted_score_expression = (
                    base_expr + " - 0.05 * binding_site_allowed_min_dist"
                )

        nonstd_map = str(getattr(ns, "nonstd_map", "") or "").strip()
        if nonstd_map:
            nonstd_cfg = _load_nonstandard_map(nonstd_map)
            with open_dict(cfg):
                cfg.nonstandard_residues = nonstd_cfg

        cyclization: dict[str, object] = {}
        bonds: list[dict[str, object]] = []
        if getattr(ns, "cyclize_head_to_tail", False):
            cyclization["head_to_tail"] = True

        for spec in list(getattr(ns, "cyclize_disulfide", []) or []):
            pos1, pos2 = _parse_disulfide_pair(spec)
            bonds.append(
                {
                    "a": {"proto": "A", "pos": pos1, "atom": "SG", "aa": "C"},
                    "b": {"proto": "A", "pos": pos2, "atom": "SG", "aa": "C"},
                }
            )

        for pair in list(getattr(ns, "cyclize_bond", []) or []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Invalid --cyclize-bond value: {pair!r}")
            a_ref = _parse_cyclize_atom_ref(pair[0])
            b_ref = _parse_cyclize_atom_ref(pair[1])
            bonds.append({"a": a_ref, "b": b_ref})

        for raw_spec in list(getattr(ns, "cyclize_bond_typed", []) or []):
            bonds.append(_parse_cyclize_bond_typed_spec(raw_spec))

        if bonds:
            cyclization["bonds"] = bonds

        cyclic_encoding_arg = getattr(ns, "cyclic_encoding", None)
        if cyclic_encoding_arg is not None:
            if not cyclization:
                raise ValueError("--cyclic-encoding requires cyclization to be enabled")
            cyclization["encoding"] = _parse_cyclic_encoding(cyclic_encoding_arg)

        if cyclization:
            with open_dict(cfg):
                cfg.cyclization = cyclization

        closure_overrides_used = any(
            getattr(ns, key, None) is not None
            for key in ("closure_profile", "closure_weight", "closure_gate")
        )
        if closure_overrides_used and not cyclization:
            raise ValueError(
                "--closure-profile/--closure-weight/--closure-gate require cyclization to be enabled"
            )

        if cyclization:
            closure_profile = _parse_closure_profile(
                getattr(ns, "closure_profile", None) or "strict"
            )
            closure_weight = (
                0.3 if getattr(ns, "closure_weight", None) is None else float(ns.closure_weight)
            )
            closure_gate = (
                0.90 if getattr(ns, "closure_gate", None) is None else float(ns.closure_gate)
            )
            if closure_gate < 0.0 or closure_gate > 1.0:
                raise ValueError(
                    f"--closure-gate must be within [0,1], got {closure_gate}"
                )

            with open_dict(cfg):
                cfg.loss.loss_terms.cyclic_closure_norm = {
                    "_type": "cyclic_closure_norm",
                    "sigma": float(closure_profile["sigma"]),
                    "aggregate": str(closure_profile["aggregate"]),
                }

                if closure_weight != 0.0:
                    base_expr = str(cfg.stages.default.weighted_score_expression)
                    if closure_weight >= 0:
                        extra = f"{closure_weight:g} * cyclic_closure_norm"
                        cfg.stages.default.weighted_score_expression = (
                            f"{base_expr} + {extra}"
                        )
                    else:
                        extra = f"{abs(closure_weight):g} * cyclic_closure_norm"
                        cfg.stages.default.weighted_score_expression = (
                            f"{base_expr} - {extra}"
                        )

                gate_expr = f"cyclic_closure_norm >= {closure_gate:g}"
                prev_gate = cfg.stages.default.acceptance_threshold
                if prev_gate:
                    cfg.stages.default.acceptance_threshold = (
                        f"({prev_gate}) and ({gate_expr})"
                    )
                else:
                    cfg.stages.default.acceptance_threshold = gate_expr

        _set_stage_evaluator(cfg, ns.evaluator)

    return _run_design_preset(
        ns,
        config_name="benchmark_peptide_binder_v1",
        mutate_config=mutate_config,
    )


def _cmd_bench_aptamer_v1(ns: argparse.Namespace) -> int:
    def mutate_config(cfg: Any) -> None:
        aptamer_type, seq_pattern, alphabet_description = _aptamer_type_settings(
            getattr(ns, "aptamer_type", "rna")
        )
        aptamer_statement = _resolve_initial_a_statement(
            ns,
            length_attr="aptamer_len",
            seq_pattern=seq_pattern,
            alphabet_description=alphabet_description,
        )
        target_seq, target_homology_cfg = _resolve_target_inputs(
            target_seq=ns.target_seq,
            target_homology=ns.target_homology,
            target_structure=getattr(ns, "target_structure", None),
            target_chain=getattr(ns, "target_chain", None),
        )

        with open_dict(cfg):
            cfg.task_name = ns.task_name
            cfg.out_dir = ns.out_dir
            cfg.types.A = aptamer_type
            cfg.types.B = "p"
            cfg.seqs.A = aptamer_statement
            cfg.seqs.B = target_seq
            cfg.homologies.B = target_homology_cfg
        _apply_common_one_shot_config_overrides(cfg, ns)
        _set_stage_evaluator(cfg, ns.evaluator)

    return _run_design_preset(
        ns,
        config_name="benchmark_aptamer_v1",
        mutate_config=mutate_config,
    )


def _cmd_bench_antibody_v1(ns: argparse.Namespace) -> int:
    if getattr(ns, "list_frameworks", False):
        from astrolabe.antibody_framework_presets import list_antibody_framework_presets

        for preset in list_antibody_framework_presets():
            aliases = ",".join(preset.aliases) if preset.aliases else "-"
            print(f"{preset.name}\t{aliases}\t{preset.description}")
        return 0

    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_antibody_v1",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    task_name = str(getattr(ns, "task_name", "") or "").strip()
    out_dir = str(getattr(ns, "out_dir", "") or "").strip()
    if not task_name:
        raise ValueError("--task-name is required")
    if not out_dir:
        raise ValueError("--out-dir is required")

    binder_seq_raw = str(getattr(ns, "binder_seq", "") or "").strip()
    framework = str(getattr(ns, "framework", "") or "").strip()
    framework_pdb = str(getattr(ns, "framework_pdb", "") or "").strip()
    framework_homology = str(getattr(ns, "framework_homology", "") or "").strip()
    cdr_lengths_raw = str(getattr(ns, "cdr_lengths", "") or "").strip()

    binder_modes = [bool(binder_seq_raw), bool(framework), bool(framework_pdb)]
    if sum(1 for x in binder_modes if x) > 1:
        raise ValueError(
            "Use only one of: --binder-seq, --framework, --framework-pdb"
        )

    if cdr_lengths_raw and binder_seq_raw:
        raise ValueError("--cdr-lengths cannot be used with --binder-seq")
    if cdr_lengths_raw and not (framework or framework_pdb):
        raise ValueError("--cdr-lengths requires --framework or --framework-pdb")
    if framework_homology and binder_seq_raw:
        raise ValueError("--framework-homology cannot be used with --binder-seq")

    binder_homology_cfg = None
    if framework or framework_pdb:
        from astrolabe.antibody_framework_presets import (
            build_framework_layout_from_structure,
            build_framework_homology_config,
            get_antibody_framework_preset,
            parse_cdr_length_overrides,
        )

        cdr_overrides = parse_cdr_length_overrides(cdr_lengths_raw)
        if framework:
            preset = get_antibody_framework_preset(framework)
            binder_seq = preset.binder_seq(cdr_overrides)
            binder_homology_cfg = build_framework_homology_config(
                framework=preset.name,
                framework_homology=framework_homology or None,
            )
        else:
            binder_seq, framework_layout = build_framework_layout_from_structure(
                structure=framework_pdb,
                heavy_chain=getattr(ns, "framework_heavy_chain", None),
                light_chain=getattr(ns, "framework_light_chain", None),
                kind=getattr(ns, "framework_kind", "auto"),
                scfv_order=getattr(ns, "scfv_order", "vh_linker_vl"),
                scfv_linker=getattr(ns, "scfv_linker", "GGGGSGGGGSGGGGS"),
                cdr_lengths=cdr_overrides,
            )
            if _binder_seq_has_ranged_loop_lengths(binder_seq):
                binder_homology_cfg = build_framework_homology_config(
                    framework_name=str(framework_layout["framework_name"]),
                    framework_homology=(
                        framework_homology or _framework_template_homology_from_layout(framework_layout)
                    ),
                    fixed_segments=framework_layout["fixed_segments"],
                    loop_order=framework_layout["loop_order"],
                    source_loop_lengths=framework_layout["source_loop_lengths"],
                )
            else:
                binder_homology_cfg = _prepare_framework_homology_payload(
                    binder_seq=binder_seq,
                    framework_layout=framework_layout,
                    framework_homology_source=framework_homology or None,
                )
    else:
        binder_seq = binder_seq_raw

    if not binder_seq:
        raise ValueError(
            "Provide one of: --binder-seq, --framework, --framework-pdb (or use --list-frameworks)"
        )

    target_seq, target_homology_cfg = _resolve_target_inputs(
        target_seq=ns.target_seq,
        target_homology=ns.target_homology,
        target_structure=getattr(ns, "target_structure", None),
        target_chain=getattr(ns, "target_chain", None),
    )

    with open_dict(cfg):
        cfg.task_name = task_name
        cfg.out_dir = out_dir
        cfg.seqs.A = binder_seq
        cfg.seqs.B = target_seq
        cfg.homologies.A = binder_homology_cfg
        cfg.homologies.B = target_homology_cfg
    _apply_common_one_shot_config_overrides(cfg, ns)

    binding_site_target = (ns.binding_site_target or "").strip()
    if binding_site_target:
        raw_binder_specs = list(getattr(ns, "binding_site_binder", []) or [])
        binder_specs = [str(x).strip() for x in raw_binder_specs if str(x).strip()]
        for spec in binder_specs:
            if not spec.startswith("A:"):
                raise ValueError(
                    "--binding-site-binder must use chain 'A' for astrolabe design antibody "
                    f"(got: {spec!r})"
                )

        with open_dict(cfg):
            cfg.binding_site = {
                "binder": binder_specs if binder_specs else "A",
                "target": binding_site_target,
                "mode": "pocket",
                "max_distance": 12.0,
            }
            cfg.loss.loss_terms.binding_site_allowed_min_dist = {
                "_type": "binding_site_allowed_min_dist"
            }
            base_expr = str(cfg.stages.default.weighted_score_expression)
            expr = base_expr + " - 0.05 * binding_site_allowed_min_dist"
            if binder_specs:
                cfg.loss.loss_terms.binding_site_framework_min_dist = {
                    "_type": "binding_site_framework_min_dist"
                }
                expr = expr + " + 0.05 * binding_site_framework_min_dist"
            cfg.stages.default.weighted_score_expression = expr

    _set_stage_evaluator(cfg, ns.evaluator)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_refine_binder_v1_3(ns: argparse.Namespace) -> int:
    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()

    structure_path = pathlib.Path(ns.structure).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"--structure not found: {structure_path}")
    reference_path = None
    if getattr(ns, "reference", None):
        reference_path = pathlib.Path(ns.reference).expanduser().resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"--reference not found: {reference_path}")

    target_chain = str(ns.target_chain).strip()
    if not target_chain:
        raise ValueError("--target-chain is required")

    binder_chains = _normalize_chain_list(ns.binder_chains)
    if target_chain in binder_chains:
        raise ValueError("--binder-chains must not include --target-chain")

    chain2seq = _load_structure_chain_sequences(structure_path)
    if target_chain not in chain2seq:
        available = sorted(chain2seq.keys())
        raise ValueError(
            f"Target chain '{target_chain}' not found in parsed sequences from {structure_path}. Available: {available}"
        )

    missing_binders = [c for c in binder_chains if c not in chain2seq]
    if missing_binders:
        available = sorted(chain2seq.keys())
        raise ValueError(
            f"Binder chain(s) not found in parsed sequences from {structure_path}: {missing_binders}. Available: {available}"
        )

    if len(binder_chains) > 25:
        raise ValueError("Too many binder chains; max supported is 25 (A + C..Z with B reserved for target).")

    # Protomer naming convention:
    #   - target is always protomer 'B' and is the 2nd chain in the oligomer.
    #   - binder chain #1 -> 'A' (1st chain), binder chain #2 -> 'C' (3rd chain), then 'D', ...
    binder_protomers: list[str] = ["A"]
    for i in range(1, len(binder_chains)):
        binder_protomers.append(chr(ord("C") + (i - 1)))
    oligo = "".join(["A", "B", *binder_protomers[1:]])

    # Build config from task template.
    cfg = _compose_hydra_config(
        config_dir=config_dir,
        config_name="benchmark_binder_v1_3_refine",
        overrides=[],
    )

    target_seq = _normalize_fixed_sequence(chain2seq[target_chain])

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        # Ensure oligomer covers all protomers.
        _set_stage_oligos(cfg, [oligo])

        # Target is fixed.
        cfg.seqs.B = target_seq

        # Binder chain(s) are mutable by default (no '+').
        for chain_id, proto in zip(binder_chains, binder_protomers, strict=True):
            cfg.seqs[proto] = chain2seq[chain_id]

        # Homology: prefer explicit AF3 JSON; otherwise optional template-from-structure.
        if ns.target_homology:
            cfg.homologies.B = str(ns.target_homology)
        elif ns.target_template_from_structure:
            mmcif_str = _single_chain_mmcif_string(structure_path, target_chain)
            L = len(chain2seq[target_chain])
            cfg.homologies.B = {
                "templates": [
                    {
                        "mmcif": mmcif_str,
                        "queryIndices": list(range(L)),
                        "templateIndices": list(range(L)),
                    }
                ]
            }
        else:
            cfg.homologies.B = None

        if reference_path is not None:
            if cfg.named_files is None:
                cfg.named_files = {}
            cfg.named_files.structure_start_ref = str(reference_path)
            hotspot_expression = getattr(ns, "reference_hotspot_expression", None)
            cfg.structure_start = {
                "mode": str(getattr(ns, "reference_mode", "repredict_gate")),
                "gate_timing": ["step0", "pre_accept"],
                "aggregate": "min",
                "references": {
                    oligo: {
                        "structure_alias": "structure_start_ref",
                        "chain_map": {
                            **{target_chain: "B"},
                            **{
                                chain_id: proto
                                for chain_id, proto in zip(
                                    binder_chains, binder_protomers, strict=True
                                )
                            },
                        },
                    }
                },
                "reference_tmscore": {
                    "enabled": True,
                    "threshold": float(ns.reference_tmscore_threshold),
                },
                "hotspot_gate": {
                    "enabled": bool(hotspot_expression),
                    "expression": hotspot_expression or "pair_hotspot_support",
                    "floor": float(ns.reference_hotspot_floor),
                    "reference_fraction": float(
                        ns.reference_hotspot_reference_fraction
                    ),
                },
            }
    _apply_common_one_shot_config_overrides(cfg, ns)

    _set_stage_evaluator(cfg, ns.evaluator)

    # UX: print mapping so users can interpret logs/losses.
    print("Protomer mapping (input_chain -> protomer):")
    print(f"- target  {target_chain} -> B")
    for chain_id, proto in zip(binder_chains, binder_protomers, strict=True):
        print(f"- binder  {chain_id} -> {proto}")
    print(f"Oligomer: {oligo}")

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _ligand_type_to_alias(raw: str) -> str:
    t = (raw or "").strip().lower()
    if t in {"c", "ccd"}:
        return "c"
    if t in {"s", "smiles"}:
        return "s"
    raise ValueError(f"Invalid ligand type: {raw!r} (expected: ccd|smiles)")


def _load_refine_ligand_inputs(
    ns: argparse.Namespace,
) -> tuple[pathlib.Path, str, dict[str, str], str, str]:
    structure_path = pathlib.Path(ns.structure).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"--structure not found: {structure_path}")

    protein_chain = str(ns.protein_chain).strip()
    if not protein_chain:
        raise ValueError("--protein-chain is required")

    chain2seq = _load_structure_chain_sequences(structure_path)
    if protein_chain not in chain2seq:
        available = sorted(chain2seq.keys())
        raise ValueError(
            f"Protein chain '{protein_chain}' not found in parsed sequences from {structure_path}. Available: {available}"
        )

    ligand = (ns.ligand or "").strip()
    if not ligand:
        raise ValueError("--ligand is required")
    ligand_type = _ligand_type_to_alias(ns.ligand_type)
    return structure_path, protein_chain, chain2seq, ligand, ligand_type


def _build_refine_ligand_config(
    ns: argparse.Namespace,
    *,
    config_name: str,
    prefer_structure_template_with_homology: bool = False,
) -> tuple[Any, pathlib.Path, str]:
    structure_path, protein_chain, chain2seq, ligand, ligand_type = _load_refine_ligand_inputs(ns)
    protein_seq = chain2seq[protein_chain]

    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name=config_name,
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    resolved_homology = None
    if ns.protein_homology and ns.protein_template_from_structure:
        structure_template = _template_homology_from_structure(
            structure_path=structure_path,
            chain_id=protein_chain,
            sequence=chain2seq[protein_chain],
        )
        if prefer_structure_template_with_homology:
            resolved_homology = _merge_homology_with_structure_template(
                homology_arg=ns.protein_homology,
                structure_template_homology=structure_template,
            )
        elif _homology_has_nonempty_templates(ns.protein_homology):
            resolved_homology = ns.protein_homology
        else:
            resolved_homology = _merge_homology_with_structure_template(
                homology_arg=ns.protein_homology,
                structure_template_homology=structure_template,
            )
    elif ns.protein_homology:
        resolved_homology = str(ns.protein_homology)
    elif ns.protein_template_from_structure:
        resolved_homology = _template_homology_from_structure(
            structure_path=structure_path,
            chain_id=protein_chain,
            sequence=chain2seq[protein_chain],
        )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = protein_seq
        cfg.seqs.B = ligand
        cfg.types.B = ligand_type
        if ns.ligand_resname is not None and str(ns.ligand_resname).strip():
            if cfg.get("ligand_resnames") is None:
                cfg.ligand_resnames = {}
            cfg.ligand_resnames.B = str(ns.ligand_resname).strip()
        cfg.homologies.A = resolved_homology
    _apply_common_one_shot_config_overrides(cfg, ns)
    return cfg, structure_path, protein_chain


def _parse_reference_pocket_positions(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("--reference-pocket must not be empty")

    positions: list[int] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid --reference-pocket range: {token!r}")
            values = range(start, end + 1)
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"Invalid --reference-pocket position: {token!r}")
            values = [value]
        for pos in values:
            if pos not in seen:
                seen.add(pos)
                positions.append(pos)
    if not positions:
        raise ValueError("--reference-pocket did not resolve to any positions")
    return positions


def _apply_refine_ligand_pocket_overrides(
    cfg: Any,
    ns: argparse.Namespace,
    *,
    current_structure_chain: str,
) -> None:
    mutation_num = getattr(ns, "mutation_num", None)
    if mutation_num is not None:
        if int(mutation_num) <= 0:
            raise ValueError("--mutation-num must be > 0")
        with open_dict(cfg):
            cfg.mutation_num = int(mutation_num)
            cfg.stages.default.mutation_num = int(mutation_num)

    initial_temperature = getattr(ns, "initial_temperature", None)
    if initial_temperature is not None:
        with open_dict(cfg):
            cfg.initial_temperature = float(initial_temperature)

    contact_distance = getattr(ns, "contact_distance", None)
    if contact_distance is not None:
        distance_value = float(contact_distance)
        if distance_value <= 0:
            raise ValueError("--contact-distance must be > 0")
        with open_dict(cfg.mutator.mutator_terms.LMPNN_allala_vina):
            cfg.mutator.mutator_terms.LMPNN_allala_vina.position_selector = (
                f"ligand_contact(distance_max={distance_value:g})"
            )

    scoring_mode = str(getattr(ns, "scoring_mode", "vina") or "vina").strip().lower()
    if scoring_mode not in {"vina", "retarget"}:
        raise ValueError("--scoring-mode must be vina|retarget")

    reference_structure = str(getattr(ns, "reference_structure", "") or "").strip()
    reference_chain = str(getattr(ns, "reference_chain", "") or "").strip()
    reference_pocket = str(getattr(ns, "reference_pocket", "") or "").strip()
    has_reference_args = any([reference_structure, reference_chain, reference_pocket])
    if scoring_mode == "retarget" and not has_reference_args:
        raise ValueError(
            "--scoring-mode retarget requires --reference-structure, --reference-chain, and --reference-pocket"
        )
    if not has_reference_args:
        return
    if not all([reference_structure, reference_chain, reference_pocket]):
        raise ValueError(
            "--reference-structure, --reference-chain, and --reference-pocket must be provided together"
        )

    reference_path = pathlib.Path(reference_structure).expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"--reference-structure not found: {reference_path}")
    reference_positions = _parse_reference_pocket_positions(reference_pocket)

    predocker_term = cfg.predocker.predocker_terms.autodock_vina_box
    with open_dict(predocker_term):
        predocker_term.binding_site_strategy = "aligned_reference_pocket"
        predocker_term.reference_structure_path = str(reference_path)
        predocker_term.reference_structure_chain = reference_chain
        predocker_term.current_structure_chain = current_structure_chain
        predocker_term.reference_pocket_positions = reference_positions
    if scoring_mode == "retarget":
        with open_dict(cfg):
            cfg.ligand_retarget_scoring = {
                "mode": "retarget",
                "reference_structure_path": str(reference_path),
                "reference_structure_chain": reference_chain,
                "current_structure_chain": current_structure_chain,
                "reference_pocket_positions": reference_positions,
                "ligand_targets": "B",
            }
            cfg.loss.loss_terms.ligand_pocket_occupancy = {
                "_type": "ligand_aligned_reference_pocket_box_occupancy",
                "reference_structure_path": str(reference_path),
                "reference_structure_chain": reference_chain,
                "current_structure_chain": current_structure_chain,
                "reference_pocket_positions": reference_positions,
                "ligand_targets": "B",
            }
            expr = str(cfg.stages.default.weighted_score_expression)
            if "ligand_pocket_occupancy" not in expr:
                cfg.stages.default.weighted_score_expression = (
                    expr + " + ligand_pocket_occupancy"
                )


def _cmd_bench_ligand_v1(ns: argparse.Namespace) -> int:
    def mutate_config(cfg: Any) -> None:
        protein_statement = _resolve_initial_a_statement(ns, length_attr="protein_len")
        ligand = (ns.ligand or "").strip()
        if not ligand:
            raise ValueError("--ligand is required")
        ligand_type = _ligand_type_to_alias(ns.ligand_type)

        with open_dict(cfg):
            cfg.task_name = ns.task_name
            cfg.out_dir = ns.out_dir
            cfg.seqs.A = protein_statement
            cfg.seqs.B = ligand
            cfg.types.B = ligand_type
            if ns.ligand_resname is not None and str(ns.ligand_resname).strip():
                if cfg.get("ligand_resnames") is None:
                    cfg.ligand_resnames = {}
                cfg.ligand_resnames.B = str(ns.ligand_resname).strip()
        _apply_common_one_shot_config_overrides(cfg, ns)
        _set_stage_evaluator(cfg, ns.evaluator)

    return _run_design_preset(
        ns,
        config_name="benchmark_ligand_v1",
        mutate_config=mutate_config,
    )


def _cmd_design_cid_generate(ns: argparse.Namespace) -> int:
    def mutate_config(cfg: Any) -> None:
        protein_a = _resolve_initial_statement_from_attrs(
            ns,
            length_attr="protein_a_len",
            seq_attr="protein_a_seq",
            expr_attr="protein_a_expr",
            length_flag="--protein-a-len",
            seq_flag="--protein-a-seq",
            expr_flag="--protein-a-expr",
        )
        protein_b = _resolve_initial_statement_from_attrs(
            ns,
            length_attr="protein_b_len",
            seq_attr="protein_b_seq",
            expr_attr="protein_b_expr",
            length_flag="--protein-b-len",
            seq_flag="--protein-b-seq",
            expr_flag="--protein-b-expr",
        )
        ligand = (ns.ligand or "").strip()
        if not ligand:
            raise ValueError("--ligand is required")
        ligand_type = _ligand_type_to_alias(ns.ligand_type)

        with open_dict(cfg):
            cfg.task_name = ns.task_name
            cfg.out_dir = ns.out_dir
            cfg.seqs.A = protein_a
            cfg.seqs.B = protein_b
            cfg.seqs.C = ligand
            cfg.types.C = ligand_type
            if ns.ligand_resname is not None and str(ns.ligand_resname).strip():
                if cfg.get("ligand_resnames") is None:
                    cfg.ligand_resnames = {}
                cfg.ligand_resnames.C = str(ns.ligand_resname).strip()
        _apply_common_one_shot_config_overrides(cfg, ns)
        _set_stage_evaluator(cfg, ns.evaluator)

        print("Protomer mapping: A/B proteins, C ligand")
        print("Oligomer: ABC")
        print("Template: benchmark_cid_generate_v1")

    return _run_design_preset(
        ns,
        config_name="benchmark_cid_generate_v1",
        mutate_config=mutate_config,
    )


def _cmd_refine_ligand_v1(ns: argparse.Namespace) -> int:
    cfg, _structure_path, protein_chain = _build_refine_ligand_config(
        ns,
        config_name="benchmark_ligand_v1_refine",
    )

    _set_stage_evaluator(cfg, ns.evaluator)

    print("Protomer mapping (input_chain -> protomer):")
    print(f"- protein {protein_chain} -> A")
    print("Oligomer: AB")

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_refine_ligand_pocket_v1(ns: argparse.Namespace) -> int:
    cfg, _structure_path, protein_chain = _build_refine_ligand_config(
        ns,
        config_name="benchmark_ligand_v1_pocket_redesign",
        prefer_structure_template_with_homology=True,
    )
    _apply_refine_ligand_pocket_overrides(
        cfg,
        ns,
        current_structure_chain=protein_chain,
    )
    _set_stage_evaluator(cfg, ns.evaluator)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _build_refine_channel_config(
    ns: argparse.Namespace,
) -> tuple[Any, pathlib.Path]:
    from astrolabe.cli_channel import (
        build_mutable_sequence_statement,
        extract_structure_covalent_bonds,
        find_residue,
        load_mutable_positions,
        parse_ligand_residue_selector,
    )

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()

    structure_path = pathlib.Path(ns.structure).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"--structure not found: {structure_path}")

    mutable_positions_path = pathlib.Path(ns.mutable_positions).expanduser().resolve()
    if not mutable_positions_path.exists():
        raise FileNotFoundError(f"--mutable-positions not found: {mutable_positions_path}")

    spheres_path = pathlib.Path(ns.spheres).expanduser().resolve()
    if not spheres_path.exists():
        raise FileNotFoundError(f"--spheres not found: {spheres_path}")

    msa_path = None
    if getattr(ns, "msa", None):
        msa_path = pathlib.Path(ns.msa).expanduser().resolve()
        if not msa_path.exists():
            raise FileNotFoundError(f"--msa not found: {msa_path}")

    ligand_conformer_path = None
    ligand_conformer_component_id = None
    if getattr(ns, "ligand_conformer", None):
        ligand_conformer_path = pathlib.Path(ns.ligand_conformer).expanduser().resolve()
        if not ligand_conformer_path.exists():
            raise FileNotFoundError(
                f"--ligand-conformer not found: {ligand_conformer_path}"
            )
        ligand_conformer_component_id = _chemical_component_id_from_cif(
            ligand_conformer_path
        )

    protein_chain = str(ns.protein_chain).strip()
    if not protein_chain:
        raise ValueError("--protein-chain is required")

    chain2seq = _load_structure_chain_sequences(structure_path)
    if protein_chain not in chain2seq:
        available = sorted(chain2seq.keys())
        raise ValueError(
            f"Protein chain '{protein_chain}' not found in parsed sequences from {structure_path}. Available: {available}"
        )
    protein_sequence = chain2seq[protein_chain]

    mutable_positions = load_mutable_positions(
        mutable_positions_path,
        sequence_length=len(protein_sequence),
    )
    mutable_statement = build_mutable_sequence_statement(
        protein_sequence,
        mutable_positions,
    )

    ligand_ref = parse_ligand_residue_selector(ns.ligand_residue)
    find_residue(structure_path, ligand_ref)
    covalent_bonds = extract_structure_covalent_bonds(
        structure_path,
        protein_chain_id=protein_chain,
        ligand_ref=ligand_ref,
    )
    if not covalent_bonds:
        raise ValueError(
            "No protein-ligand covalent bond was found between the selected ligand residue "
            f"and protein chain {protein_chain!r} in {structure_path}"
        )

    cfg = _compose_hydra_config(
        config_dir=config_dir,
        config_name="benchmark_channel_v1_refine",
        overrides=[],
    )

    translated_covalent_bonds = [
        {
            "atom1": {
                "chain_id": str(item["protein_atom"]["chain_id"]),
                "resseq": int(item["protein_atom"]["resseq"]),
                "atom_name": str(item["protein_atom"]["atom_name"]),
            },
            "atom2": {
                "chain_id": "B",
                "resseq": 1,
                "atom_name": str(item["ligand_atom"]["atom_name"]),
            },
        }
        for item in covalent_bonds
    ]
    rf3_ligand_resname = ligand_conformer_component_id or ligand_ref.resname

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = mutable_statement
        cfg.seqs.B = rf3_ligand_resname
        cfg.types.B = "c"
        if cfg.get("ligand_resnames") is None:
            cfg.ligand_resnames = {}
        cfg.ligand_resnames.B = ligand_ref.resname
        if ligand_conformer_component_id is not None:
            cfg.user_ccd = ligand_conformer_path.read_text(encoding="utf-8")
        cfg.homologies.A = str(msa_path) if msa_path is not None else None
        cfg.covalent_bonds = translated_covalent_bonds
        cfg.evaluator.rf3.fixed_ligands = {
            "B": {
                "source": "override" if ligand_conformer_path is not None else "structure_residue",
                "path": str(ligand_conformer_path) if ligand_conformer_path is not None else None,
                "ccd_id": rf3_ligand_resname,
                "output_resname": ligand_ref.resname,
                "structure_path": str(structure_path),
                "residue": {
                    "chain_id": ligand_ref.chain_id,
                    "resname": ligand_ref.resname,
                    "resseq": int(ligand_ref.resseq),
                    "icode": ligand_ref.icode,
                },
            }
        }
        cfg.evaluator.rf3.template_selection = [f"B/{rf3_ligand_resname}/1/*"]
        cfg.evaluator.rf3.ground_truth_conformer_selection = [f"B/{rf3_ligand_resname}/1/*"]
        cfg.loss.loss_terms.channel_hole.reference_structure_path = str(structure_path)
        cfg.loss.loss_terms.channel_hole.alignment_chains = protein_chain
        cfg.loss.loss_terms.channel_hole.reference_spheres_path = str(spheres_path)
        cfg.mutator.mutator_terms.LMPNN_channel.position_selector = (
            "hole_surface("
            f"reference_structure_path='{str(structure_path)}', "
            f"reference_spheres_path='{str(spheres_path)}', "
            f"alignment_chains='{protein_chain}', "
            "surface_gap_max=1.5"
            ") and not ligand_contact(distance_max=5.0)"
        )
    _apply_common_one_shot_config_overrides(cfg, ns)
    return cfg, structure_path


def _cmd_refine_channel_v1(ns: argparse.Namespace) -> int:
    cfg, _structure_path = _build_refine_channel_config(ns)

    if ns.print_config or ns.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=False))
        if ns.dry_run:
            _maybe_dump_config(cfg, ns.dump_config)
            return 0

    _maybe_dump_config(cfg, ns.dump_config)
    _run_from_config(cfg, autoselect_gpu=bool(getattr(ns, "autoselect_gpu", False)))
    return 0


def _parse_oligo_list(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",")]
    out = [p for p in parts if p]
    if not out:
        raise ValueError("Empty oligo list")
    return out


def _cmd_refine_oligomer_v3(ns: argparse.Namespace) -> int:
    seq = (ns.seq or "").strip()
    protein_len = (ns.protein_len or "").strip() if hasattr(ns, "protein_len") else ""
    structure_raw = (ns.structure or "").strip()
    chain = (ns.chain or "").strip()

    structure_path = None
    chain2seq: dict[str, str] | None = None
    if structure_raw:
        structure_path = pathlib.Path(structure_raw).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"--structure not found: {structure_path}")
        if not chain:
            raise ValueError("--chain is required when using --structure")
        chain2seq = _load_structure_chain_sequences(structure_path)
        if chain not in chain2seq:
            available = sorted(chain2seq.keys())
            raise ValueError(
                f"Chain '{chain}' not found in parsed sequences from {structure_path}. Available: {available}"
            )

    if seq:
        seq_A = seq
    elif chain2seq is not None and structure_path is not None:
        seq_A = chain2seq[chain]
    elif protein_len:
        seq_A = _parse_len_range(protein_len)
    else:
        raise ValueError("Provide one of: --seq, --protein-len, or --structure+--chain")

    oligos: list[str] | None = None
    if ns.oligo and ns.copies is not None:
        raise ValueError("Use only one of: --oligo or --copies")
    if ns.oligo:
        oligos = _parse_oligo_list(ns.oligo)
    elif ns.copies is not None:
        n = int(ns.copies)
        if n <= 0:
            raise ValueError("--copies must be > 0")
        oligos = ["A" * n]

    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_oligomer_v3_refine",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = seq_A
        if oligos is not None:
            _set_stage_oligos(cfg, oligos)

        if ns.homology:
            cfg.homologies.A = str(ns.homology)
        elif ns.template_from_structure:
            if structure_path is None or chain2seq is None:
                raise ValueError("--template-from-structure requires --structure and --chain")
            mmcif_str = _single_chain_mmcif_string(structure_path, chain)
            L = len(chain2seq[chain])
            cfg.homologies.A = {
                "templates": [
                    {
                        "mmcif": mmcif_str,
                        "queryIndices": list(range(L)),
                        "templateIndices": list(range(L)),
                    }
                ]
            }
        else:
            cfg.homologies.A = None
    _apply_common_one_shot_config_overrides(cfg, ns)

    if getattr(ns, "symmetry_predocker", False):
        stage_oligos = oligos
        if stage_oligos is None:
            stage_oligos = [str(item) for item in list(cfg.stages.default.oligo)]
        if len(stage_oligos) != 1:
            raise ValueError(
                "Symmetry predocker for refine-oligomer-v3 requires exactly one target oligomer."
            )
        _enable_symmetry_predocker(
            cfg,
            mode="cyclic_oligomer",
            oligo=str(stage_oligos[0]),
            backend=getattr(ns, "symmetry_predocker_backend", "rfdiffusion"),
            order=getattr(ns, "symmetry_order", None),
        )

    _set_stage_evaluator(cfg, ns.evaluator)

    print("Protomer mapping: A (homooligomer)")
    if oligos:
        print(f"Oligomer(s): {oligos}")

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_bench_pseudocycle_v1(ns: argparse.Namespace) -> int:
    config_dir = design_preset_builder.config_dir_from_namespace(
        ns,
        default_config_dir=_default_config_dir,
    )
    _ensure_sys_path_has_repo_root(config_dir)

    copies = int(ns.copies)
    unit_len = _parse_len_range(ns.unit_len)
    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_pseudocycle_v1",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = _build_internal_repeat_statement(unit_len, copies)
        if ns.repeats is not None:
            cfg.repeats = int(ns.repeats)
        if ns.steps is not None:
            cfg.steps = int(ns.steps)

    if getattr(ns, "symmetry_predocker", False):
        _enable_symmetry_predocker(
            cfg,
            mode="pseudocycle",
            oligo="A",
            included_protomer="A",
            repeat_canonicalizer_mode="helical",
            backend=getattr(ns, "symmetry_predocker_backend", "legacy"),
            order=getattr(ns, "symmetry_order", None),
            interface_selection=getattr(ns, "symmetry_interface_selection", None),
            interface=getattr(ns, "symmetry_interface", None),
        )

    _set_stage_evaluator(cfg, ns.evaluator)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_bench_tandem_repeat_v1(ns: argparse.Namespace) -> int:
    config_dir = design_preset_builder.config_dir_from_namespace(
        ns,
        default_config_dir=_default_config_dir,
    )
    _ensure_sys_path_has_repo_root(config_dir)

    copies = int(ns.copies)
    unit_len = _parse_len_range(ns.unit_len)
    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_tandem_repeat_v1",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs.A = _build_internal_repeat_statement(unit_len, copies)
        if ns.repeats is not None:
            cfg.repeats = int(ns.repeats)
        if ns.steps is not None:
            cfg.steps = int(ns.steps)

    if getattr(ns, "symmetry_predocker", False):
        _enable_symmetry_predocker(
            cfg,
            mode="tandem_repeat",
            oligo="A",
            included_protomer="A",
            repeat_canonicalizer_mode=ns.repeat_canonicalizer_mode,
            backend=getattr(ns, "symmetry_predocker_backend", "legacy"),
            order=getattr(ns, "symmetry_order", None),
            interface_selection=getattr(ns, "symmetry_interface_selection", None),
            interface=getattr(ns, "symmetry_interface", None),
        )

    _set_stage_evaluator(cfg, ns.evaluator)

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_refine_heterooligomer_v1(ns: argparse.Namespace) -> int:
    config_dir = design_preset_builder.config_dir_from_namespace(
        ns,
        default_config_dir=_default_config_dir,
    )
    _ensure_sys_path_has_repo_root(config_dir)

    seq = (ns.seq or "").strip()
    structure_raw = (ns.structure or "").strip()
    chain = (ns.chain or "").strip()
    if bool(seq) == bool(structure_raw):
        raise ValueError("Provide exactly one of: --seq or --structure")

    structure_path = None
    monomer_seq = ""
    if structure_raw:
        structure_path = pathlib.Path(structure_raw).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"--structure not found: {structure_path}")
        if not chain:
            raise ValueError("--chain is required when using --structure")
        chain2seq = _load_structure_chain_sequences(structure_path)
        if chain not in chain2seq:
            available = sorted(chain2seq.keys())
            raise ValueError(
                f"Chain '{chain}' not found in parsed sequences from {structure_path}. Available: {available}"
            )
        monomer_seq = _strip_fixed_prefix(chain2seq[chain]).strip().upper()
    else:
        monomer_seq = _strip_fixed_prefix(seq).strip().upper()

    protomer_names = _protomer_letter_names(int(ns.copies))
    oligo = "".join(protomer_names)
    seqs = _build_heterooligomer_seqs(
        monomer_seq,
        interface_ranges=ns.interface,
        protomer_names=protomer_names,
    )

    cfg = design_preset_builder.compose_config_from_namespace(
        ns,
        config_name="benchmark_heterooligomer_v1_refine",
        default_config_dir=_default_config_dir,
        compose_config=_compose_hydra_config,
    )

    with open_dict(cfg):
        cfg.task_name = ns.task_name
        cfg.out_dir = ns.out_dir
        cfg.seqs = dict(seqs)
        cfg.homologies = {proto: None for proto in protomer_names}
        if ns.repeats is not None:
            cfg.repeats = int(ns.repeats)
        if ns.steps is not None:
            cfg.steps = int(ns.steps)
        _set_stage_oligos(cfg, [oligo])
        if ns.template_from_structure:
            if structure_path is None:
                raise ValueError("--template-from-structure requires --structure and --chain")
            mmcif_str = _single_chain_mmcif_string(structure_path, chain)
            L = len(monomer_seq)
            template = {
                "templates": [
                    {
                        "mmcif": mmcif_str,
                        "queryIndices": list(range(L)),
                        "templateIndices": list(range(L)),
                    }
                ]
            }
            cfg.homologies = {proto: template for proto in protomer_names}

    if getattr(ns, "symmetry_predocker", False):
        _enable_symmetry_predocker(
            cfg,
            mode="heterooligomer_shared_core",
            oligo=oligo,
            backend=getattr(ns, "symmetry_predocker_backend", "legacy"),
            order=getattr(ns, "symmetry_order", None),
        )

    _set_stage_evaluator(cfg, ns.evaluator)

    print("Protomer mapping (copy -> protomer):")
    if structure_path is not None:
        print(f"- template {chain} -> {','.join(protomer_names)}")
    for proto in protomer_names:
        print(f"- copy {proto} -> {proto}")
    print(f"Oligomer: {oligo}")

    return design_preset_builder.finish_config(
        ns,
        cfg,
        dump_config=_maybe_dump_config,
        run_config=_run_from_config,
    )


def _cmd_wizard(ns: argparse.Namespace) -> int:
    from astrolabe.cli_interactive import _cmd_wizard as _impl

    return _impl(ns)


def _scan_missing_keys_for_scaffold(cfg_obj: Any) -> List[str]:
    missing_keys: List[str] = []

    def _scan(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k)
                next_prefix = f"{prefix}.{key}" if prefix else key
                _scan(v, next_prefix)
            return
        if isinstance(node, list):
            for i, v in enumerate(node):
                next_prefix = f"{prefix}[{i}]"
                _scan(v, next_prefix)
            return
        if isinstance(node, str) and node.strip() == "???":
            if prefix:
                missing_keys.append(prefix)
            return

    _scan(cfg_obj)
    return [k for k in missing_keys if k]


def _parse_scalar_value(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    low = s.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    # int/float best-effort
    try:
        if low.startswith("0") and len(low) > 1 and low[1].isdigit():
            return s  # keep leading-zero strings as strings
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _classify_scaffold_key_kind(key: str) -> str:
    if key == "out_dir":
        return "dir_may_create"
    if key.startswith("seqs."):
        return "sequence"
    if key.startswith("homologies."):
        return "file_or_null"
    if key.endswith(".json") or key.endswith(".jsonl"):
        return "file"
    return "string"


def _validate_scaffold_value(key: str, value: Any) -> tuple[bool, str]:
    kind = _classify_scaffold_key_kind(key)
    if kind == "string":
        if value is None:
            return False, "null_not_allowed"
        if str(value).strip() == "":
            return False, "empty"
        return True, "ok"

    if kind == "sequence":
        if value is None:
            return False, "null_not_allowed"
        s = str(value).strip()
        if not s:
            return False, "empty"
        if s.startswith("+"):
            s = s[1:]
        # Allow common AA alphabet + X
        allowed = set("ACDEFGHIKLMNPQRSTVWYXBZUOJ")  # permissive
        bad = [c for c in s.upper() if c not in allowed]
        if bad:
            return False, "bad_chars:%s" % "".join(sorted(set(bad))[:12])
        return True, "ok"

    if kind == "dir_may_create":
        if value is None:
            return False, "null_not_allowed"
        s = str(value).strip()
        if not s:
            return False, "empty"
        p = pathlib.Path(_expand_user_vars(s))
        if p.exists() and not p.is_dir():
            return False, "exists_not_dir"
        # allow non-existing (will be created by pipeline)
        return True, "ok"

    if kind == "file_or_null":
        if value is None:
            return True, "ok"
        s = str(value).strip()
        if not s:
            return False, "empty"
        p = pathlib.Path(_expand_user_vars(s))
        if not p.exists():
            return False, "missing"
        if not p.is_file():
            return False, "not_a_file"
        return True, "ok"

    return True, "ok"


def _prompt_line(
    prompt_text: str,
    *,
    default: str = "",
    path_kind: Optional[str] = None,
) -> str:
    """
    Read one line, best-effort with path completion + fuzzy matching.

    `path_kind`: "file" | "dir" | "path" | None
    """
    from prompt_toolkit import prompt as pt_prompt  # type: ignore
    from prompt_toolkit.completion import FuzzyCompleter, PathCompleter  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings  # type: ignore

    completer = None
    if path_kind in {"file", "dir", "path"}:
        completer = FuzzyCompleter(
            PathCompleter(
                expanduser=True,
                only_directories=(path_kind == "dir"),
            )
        )

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event):  # pragma: no cover - interactive
        buf = event.current_buffer
        # If a completion menu is active, accept the current completion but do not submit.
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            return
        buf.validate_and_handle()

    return pt_prompt(
        prompt_text,
        default=default or "",
        completer=completer,
        complete_while_typing=False,
        key_bindings=kb,
    ).strip()


def _prompt_choice_from_list(
    prompt_text: str,
    options: List[str],
    *,
    default: Optional[str] = None,
    allow_empty: bool = False,
) -> str:
    opts = [o for o in options if o]
    opt_map = {str(i + 1): o for i, o in enumerate(opts)}
    # When we show a narrowed "Matches:" list, allow choosing by 1-based index within that list.
    last_match_map: Optional[dict[str, str]] = None
    from prompt_toolkit import prompt as pt_prompt  # type: ignore
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings  # type: ignore
    while True:
        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event):  # pragma: no cover - interactive
            buf = event.current_buffer
            if buf.complete_state and buf.complete_state.current_completion:
                buf.apply_completion(buf.complete_state.current_completion)
                return
            buf.validate_and_handle()

        completer = FuzzyCompleter(
            WordCompleter(opts + list(opt_map.keys()), ignore_case=True)
        )
        raw = pt_prompt(
            prompt_text,
            default=(default or ""),
            completer=completer,
            complete_while_typing=False,
            key_bindings=kb,
        ).strip()
        if not raw:
            if allow_empty:
                return ""
            if default is not None:
                return default
            continue

        # If user is picking from the most recently displayed narrowed match list, interpret
        # numbers against that list (not the global numbering of all options).
        if raw.isdigit() and last_match_map and raw in last_match_map:
            return last_match_map[raw]
        # Any non-numeric input invalidates the previous match list context.
        if not raw.isdigit():
            last_match_map = None

        if raw in {":back", "b", "back"}:
            if allow_empty:
                return ""
            continue
        if raw in opt_map:
            return opt_map[raw]
        low = raw.lower()
        matches = [o for o in opts if low in o.lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            print("Matches:")
            matches = matches[:30]
            last_match_map = {str(i + 1): o for i, o in enumerate(matches)}
            for i, o in enumerate(matches, start=1):
                print(f"{i:02d} {o}")
            continue
        print("ERR: no match")


def _prompt_words(prompt_text: str, words: List[str], *, default: str = "") -> str:
    """
    Prompt with fuzzy word completion (Tab to complete).
    """
    from prompt_toolkit import prompt as pt_prompt  # type: ignore
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings  # type: ignore

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event):  # pragma: no cover - interactive
        buf = event.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            return
        buf.validate_and_handle()

    completer = FuzzyCompleter(WordCompleter([w for w in words if w], ignore_case=True))
    return pt_prompt(
        prompt_text,
        default=default or "",
        completer=completer,
        complete_while_typing=False,
        key_bindings=kb,
    ).strip()


def _prompt_expr_line(prompt_text: str, *, default: str, words: List[str]) -> str:
    """
    Prompt for a free-form expression string with word completion (Tab).

    Uses prompt_toolkit when available; falls back to input().
    """
    from prompt_toolkit import prompt as pt_prompt  # type: ignore
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings  # type: ignore

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event):  # pragma: no cover - interactive
        buf = event.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            return
        buf.validate_and_handle()

    uniq: List[str] = []
    seen = set()
    for w in words:
        ww = str(w)
        if not ww or ww in seen:
            continue
        uniq.append(ww)
        seen.add(ww)

    completer = FuzzyCompleter(WordCompleter(uniq, ignore_case=True))
    return pt_prompt(
        prompt_text,
        default=default or "",
        completer=completer,
        complete_while_typing=False,
        complete_in_thread=True,
        key_bindings=kb,
    ).strip()


def _prompt_multi_select_from_list(prompt_text: str, options: List[str]) -> List[str]:
    """
    Multi-select via text input with completion.

    Input forms:
    - numbers: "1 3 4"
    - names: "MCMC, LigandMPNN"
    - mixed
    Empty / :back / b / back => []
    """
    opts = [o for o in options if o]
    if not opts:
        return []
    opt_map = {str(i + 1): o for i, o in enumerate(opts)}
    raw = _prompt_words(prompt_text, opts + list(opt_map.keys()), default="").strip()
    if raw in {"", ":back", "b", "back"}:
        return []
    toks = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    picked: List[str] = []
    for t in toks:
        if t in opt_map:
            picked.append(opt_map[t])
            continue
        matches = [o for o in opts if t.lower() == o.lower()]
        if matches:
            picked.append(matches[0])
            continue
        matches = [o for o in opts if t.lower() in o.lower()]
        if len(matches) == 1:
            picked.append(matches[0])
            continue
    # de-dupe keep order
    seen = set()
    out: List[str] = []
    for x in picked:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _compose_build_oligo_from_parts(parts: List[str]) -> str:
    # Allow input like ["A","+","B"] or ["A+B"] etc -> "AB"
    joined = "".join([p for p in parts if p and p != "+"]).strip()
    if "+" in joined:
        joined = "".join([x.strip() for x in joined.split("+") if x.strip()])
    return joined


def _compose_prompt_oligo_list(overlay_obj: dict, *, default: Optional[List[str]] = None) -> List[str]:
    seqs = overlay_obj.get("seqs")
    protos: List[str] = []
    if isinstance(seqs, dict):
        protos = sorted([str(k) for k in seqs.keys() if str(k)])
    if not protos:
        protos = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    cur = default or []
    cur_s = ", ".join(cur) if cur else ""
    raw = _prompt_words(
        "oligo (comma-sep; ':build' interactive) [%s]: " % cur_s if cur_s else "oligo (comma-sep; ':build' interactive): ",
        [":build", ":back"] + protos,
        default=cur_s,
    ).strip()
    if raw in {"", ":back", "b", "back"}:
        return cur
    if raw == ":build":
        built: List[str] = []
        while True:
            parts: List[str] = []
            while True:
                p = _prompt_words("add protomer (+ allowed; enter to finish oligomer): ", protos + ["+"], default="").strip()
                if p in {"", ":back", "b", "back"}:
                    break
                parts.append(p)
            ol = _compose_build_oligo_from_parts(parts)
            if ol:
                built.append(ol)
            more = input("add another oligomer? [y/N]: ").strip().lower()
            if more not in {"y", "yes"}:
                break
        return built or cur

    # Try literal list
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, str):
            return [_compose_build_oligo_from_parts([v])]
        if isinstance(v, list):
            out = []
            for it in v:
                if it is None:
                    continue
                out.append(_compose_build_oligo_from_parts([str(it)]))
            return [x for x in out if x]
    except (ValueError, SyntaxError):
        pass

    # Comma/space list
    items = [x.strip() for x in raw.replace(" ", ",").split(",") if x.strip()]
    return [_compose_build_oligo_from_parts([x]) for x in items if _compose_build_oligo_from_parts([x])]


def _compose_expression_words(overlay_obj: dict, stage_conf: dict) -> List[str]:
    # Loss aliases (user-defined)
    loss_terms = _get_by_dotted_key(overlay_obj, "loss.loss_terms") or {}
    loss_aliases: List[str] = sorted([str(k) for k in loss_terms.keys()]) if isinstance(loss_terms, dict) else []

    # Standard evaluator score keys (common)
    score_keys = [
        "plddt",
        "ptm",
        "iptm",
        "pae",
        "pde",
    ]

    # Handy functions/constants for expressions
    funcs = ["min", "max", "abs", "sum", "len", "float", "int"]
    literals = ["true", "false", "null", "None"]
    ops = ["and", "or", "not"]

    # Oligomer prefixes (AB., ACD., ...) + expanded tokens like AB.plddt
    oligos = stage_conf.get("oligo")
    prefixes: List[str] = []
    expanded: List[str] = []
    if isinstance(oligos, list):
        for o in oligos:
            oo = str(o).strip()
            if not oo:
                continue
            prefixes.append(oo)
            prefixes.append(oo + ".")
            for k in score_keys:
                expanded.append(f"{oo}.{k}")
            for k in loss_aliases:
                expanded.append(f"{oo}.{k}")

    return expanded + prefixes + score_keys + loss_aliases + funcs + ops + literals


def _compose_default_config_tree(config_dir: pathlib.Path) -> dict:
    base_path = config_dir / "base.yaml"
    if not base_path.exists():
        return {}
    cfg = OmegaConf.load(str(base_path))
    obj = OmegaConf.to_container(cfg, resolve=False)
    return obj if isinstance(obj, dict) else {}


def _flatten_dict_keys(obj: Any, *, prefix: str = "", max_depth: int = 2) -> List[str]:
    out: List[str] = []

    def _walk(node: Any, pfx: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                kk = str(k)
                np = f"{pfx}.{kk}" if pfx else kk
                out.append(np)
                _walk(v, np, depth + 1)

    _walk(obj, prefix, 1)
    # de-dupe keep order
    seen = set()
    dedup: List[str] = []
    for k in out:
        if k not in seen:
            dedup.append(k)
            seen.add(k)
    return dedup


def _ensure_dict_at(obj: dict, dotted_key: str) -> dict:
    """
    Ensure a dict exists at dotted_key and return the live reference.

    Important: do NOT use `... or {}` when retrieving nested dicts; that creates a
    detached dict when the real value is an empty dict.
    """
    cur = _get_by_dotted_key(obj, dotted_key)
    if isinstance(cur, dict):
        return cur
    _set_by_dotted_key(obj, dotted_key, {})
    cur2 = _get_by_dotted_key(obj, dotted_key)
    return cur2 if isinstance(cur2, dict) else {}


def _ensure_list_at(obj: dict, dotted_key: str) -> list:
    """
    Ensure a list exists at dotted_key and return the live reference.
    """
    cur = _get_by_dotted_key(obj, dotted_key)
    if isinstance(cur, list):
        return cur
    _set_by_dotted_key(obj, dotted_key, [])
    cur2 = _get_by_dotted_key(obj, dotted_key)
    return cur2 if isinstance(cur2, list) else []


def _compose_write_yaml(path: pathlib.Path, obj: dict, *, with_header: bool = True) -> None:
    header = "# @package _global_\n\n" if with_header else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(header + OmegaConf.to_yaml(OmegaConf.create(obj), resolve=False), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _compose_get_section_status(overlay_obj: dict) -> dict:
    loss_terms = _get_by_dotted_key(overlay_obj, "loss.loss_terms") or {}
    eval_terms = _get_by_dotted_key(overlay_obj, "evaluator.evaluator_terms") or {}
    mut_terms = _get_by_dotted_key(overlay_obj, "mutator.mutator_terms") or {}
    stages = _get_by_dotted_key(overlay_obj, "stages") or {}
    pred_terms = _get_by_dotted_key(overlay_obj, "predocker.predocker_terms") or {}
    return {
        "basic": any(k in overlay_obj for k in ["task_name", "out_dir", "steps", "seed", "mutation_num"]),
        "evaluator": bool(isinstance(eval_terms, dict) and eval_terms),
        "mutator": bool(isinstance(mut_terms, dict) and mut_terms),
        "loss": bool(isinstance(loss_terms, dict) and loss_terms),
        "predocker": bool(isinstance(pred_terms, dict) and pred_terms),
        "stages": bool(isinstance(stages, dict) and stages),
    }


def _compose_validate_basic(overlay_obj: dict) -> List[str]:
    errs: List[str] = []
    for dotted in ["loss.loss_terms", "evaluator.evaluator_terms", "mutator.mutator_terms", "predocker.predocker_terms", "stages"]:
        v = _get_by_dotted_key(overlay_obj, dotted)
        if v is None:
            continue
        if not isinstance(v, dict):
            errs.append(f"{dotted}: expected dict, got {type(v).__name__}")

    eval_terms = _get_by_dotted_key(overlay_obj, "evaluator.evaluator_terms") or {}
    if isinstance(eval_terms, dict):
        for alias, conf in eval_terms.items():
            if not isinstance(conf, dict):
                errs.append(f"evaluator.evaluator_terms.{alias}: expected dict")
                continue
            if "_type" not in conf:
                errs.append(f"evaluator.evaluator_terms.{alias} missing _type")

    mut_terms = _get_by_dotted_key(overlay_obj, "mutator.mutator_terms") or {}
    if isinstance(mut_terms, dict):
        for alias, conf in mut_terms.items():
            if not isinstance(conf, dict):
                errs.append(f"mutator.mutator_terms.{alias}: expected dict")
                continue
            if "_type" not in conf:
                errs.append(f"mutator.mutator_terms.{alias} missing _type")

    initializer = _get_by_dotted_key(overlay_obj, "initializer") or {}
    if isinstance(initializer, dict):
        initializer_mode = str(initializer.get("mode") or "").strip().lower()
        if initializer_mode in {
            "structure",
            "start_from_structure",
            "start-from-structure",
            "existing_structure",
            "existing-structure",
            "rfd3",
            "rfd-3",
        }:
            sequence_designer = str(
                initializer.get("sequence_designer") or ""
            ).strip()
            if not sequence_designer:
                errs.append(
                    "initializer.sequence_designer is required for structure/rfd3 mode"
                )
            elif isinstance(mut_terms, dict) and mut_terms and sequence_designer not in mut_terms:
                errs.append(
                    "initializer.sequence_designer references unknown mutator alias: "
                    + sequence_designer
                )
            mode_key = "rfd3" if initializer_mode in {"rfd3", "rfd-3"} else "structure"
            mode_config = initializer.get(mode_key)
            if not isinstance(mode_config, dict):
                errs.append(f"initializer.{mode_key}: expected dict")
            elif not isinstance(mode_config.get("chain_map"), dict) or not mode_config.get(
                "chain_map"
            ):
                errs.append(
                    f"initializer.{mode_key}.chain_map must be a non-empty dict"
                )

    loss_terms = _get_by_dotted_key(overlay_obj, "loss.loss_terms") or {}
    if isinstance(loss_terms, dict):
        for alias, conf in loss_terms.items():
            if not isinstance(conf, dict):
                errs.append(f"loss.loss_terms.{alias}: expected dict")
                continue
            if "_type" not in conf:
                errs.append(f"loss.loss_terms.{alias} missing _type")

    pred_terms = _get_by_dotted_key(overlay_obj, "predocker.predocker_terms") or {}
    if isinstance(pred_terms, dict):
        for alias, conf in pred_terms.items():
            if not isinstance(conf, dict):
                errs.append(f"predocker.predocker_terms.{alias}: expected dict")
                continue
            if "_type" not in conf:
                errs.append(f"predocker.predocker_terms.{alias} missing _type")

    stages = _get_by_dotted_key(overlay_obj, "stages") or {}
    if isinstance(stages, dict):
        eval_aliases = set(eval_terms.keys()) if isinstance(eval_terms, dict) else set()
        mut_aliases = set(mut_terms.keys()) if isinstance(mut_terms, dict) else set()
        for stage_name, conf in stages.items():
            if not isinstance(conf, dict):
                errs.append(f"stages.{stage_name}: expected dict")
                continue
            ev = conf.get("evaluator")
            if ev and eval_aliases and str(ev) not in eval_aliases:
                errs.append(f"stages.{stage_name}.evaluator references unknown alias: {ev}")
            gate = conf.get("transition_consistency_gate")
            if isinstance(gate, dict):
                gate_ev = gate.get("evaluator")
                if gate_ev and eval_aliases and str(gate_ev) not in eval_aliases:
                    errs.append(
                        f"stages.{stage_name}.transition_consistency_gate.evaluator references unknown alias: {gate_ev}"
                    )
            muts = conf.get("mutators")
            if muts and isinstance(muts, list) and mut_aliases:
                for m in muts:
                    if str(m) not in mut_aliases:
                        errs.append(f"stages.{stage_name}.mutators references unknown alias: {m}")
            w = conf.get("mutator_weights")
            if isinstance(muts, list) and isinstance(w, list) and len(muts) != len(w):
                errs.append(f"stages.{stage_name}.mutator_weights length != mutators length")
            if "extra_template" in conf:
                errs.append(
                    f"stages.{stage_name}.extra_template is not supported; use transition_consistency_gate instead"
                )
            if gate is None:
                continue
            if stage_name == "default":
                errs.append("stages.default.transition_consistency_gate is not allowed")
                continue
            try:
                from astrolabe.stage_transition import (
                    TransitionConsistencyGateConfig,
                )

                TransitionConsistencyGateConfig.from_raw(gate, stage_name=stage_name)
            except ValueError as exc:
                message = str(exc).rstrip(".")
                prefix = f"Stage '{stage_name}' "
                if message.startswith(prefix):
                    message = f"stages.{stage_name}." + message[len(prefix):]
                errs.append(message)
    structure_start = _get_by_dotted_key(overlay_obj, "structure_start")
    if structure_start is not None:
        try:
            from astrolabe.structure_start import StructureStartConfig

            StructureStartConfig.from_raw(structure_start)
        except ValueError as exc:
            errs.append(str(exc).rstrip("."))
    return errs


def _compose_section_basic(overlay_obj: dict) -> None:
    key_hints = {
        "task_name": "string",
        "out_dir": "dir_may_create",
        "steps": "int",
        "mutation_num": "int",
        "seed": "int|null",
        "initializer.mode": "string(random|dplm|aa_set|pssm|all_a|mask_mix|x_mix|structure|rfd3)",
        "initializer.aa_set": "string|list",
        "initializer.pssm.path": "file",
        "initializer.dplm_alias": "string|null",
        "initializer.sequence_designer": "string|null",
        "initializer.structure.source": "file",
        "initializer.structure.chain_map": "mapping",
        "initializer.rfd3.specification": "mapping|null",
        "initializer.rfd3.motif_map": "mapping",
        "enable_pyrosetta": "bool",
        "skip_complex_poses": "bool",
        "allow_mmcif_pdb_fallback": "bool",
    }
    keys = list(key_hints.keys()) + ["<custom dotted key>"]
    while True:
        print("basic: pick key to set (:back to menu)")
        for i, k in enumerate(keys, start=1):
            print(f"{i:02d} {k}")
        raw = input("basic> ").strip()
        if raw in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if raw in {":?", ":help"}:
            print("Enter number/key; or 'foo.bar=123'. Commands: :back")
            continue
        if "=" in raw and not raw.startswith("="):
            k, vraw = raw.split("=", 1)
            k = k.strip()
            val = _parse_scalar_value(vraw)
            _set_by_dotted_key(overlay_obj, k, val)
            continue

        choice = raw
        if not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            choice = keys[int(choice) - 1]
        if choice == "<custom dotted key>":
            dk = input("dotted key: ").strip()
            if not dk:
                continue
            cur = _get_by_dotted_key(overlay_obj, dk)
            rawv = input(f"{dk} = ").strip()
            if not rawv and cur is not None:
                rawv = str(cur)
            _set_by_dotted_key(overlay_obj, dk, _parse_scalar_value(rawv))
            continue
        if choice not in key_hints:
            print("ERR: unknown key")
            continue
        hint = key_hints[choice]
        cur = _get_by_dotted_key(overlay_obj, choice)
        cur_s = "" if cur is None else str(cur)
        pk = "dir" if hint == "dir_may_create" else None
        rawv = _prompt_line(f"{choice} ({hint}) [{cur_s}]: " if cur_s else f"{choice} ({hint}): ", default=cur_s, path_kind=pk)
        if rawv == ":back":
            continue
        if not rawv and cur_s:
            rawv = cur_s
        _set_by_dotted_key(overlay_obj, choice, _parse_scalar_value(rawv))


def _compose_section_evaluator(overlay_obj: dict, config_dir: pathlib.Path) -> None:
    evaluator_types = [
        "Alphafold3",
        "Alphafold2",
        "ESMFold",
        "Chai1",
        "Boltz",
        "OpenDDE",
        "Protenix",
        "Dummy",
    ]
    eval_terms = _ensure_dict_at(overlay_obj, "evaluator.evaluator_terms")

    base_tree = _compose_default_config_tree(config_dir)

    def _default_keys_for_type(t: str) -> List[str]:
        key = (t or "").strip().lower()
        # match evaluator/utils.py normalization
        mapping = {
            "alphafold3": "alphafold3",
            "alphafold2": "alphafold2",
            "chai1": "chai1",
            "boltz": "boltz",
            "protenix": "protenix",
        }
        sub = mapping.get(key)
        if not sub:
            return []
        node = ((base_tree.get("evaluator") or {}) if isinstance(base_tree, dict) else {}).get(sub)
        return _flatten_dict_keys(node, prefix="", max_depth=2) if node is not None else []

    def _edit_alias(alias: str) -> None:
        conf = eval_terms.get(alias) or {}
        if not isinstance(conf, dict):
            conf = {}
        t = str(conf.get("_type") or "").strip()
        default_keys = _default_keys_for_type(t)
        while True:
            print(json.dumps(conf, indent=2, sort_keys=True))
            raw = _prompt_words(
                "edit> field (Tab) or key=value (:back to stop): ",
                [":back", ":show", ":del"]
                + [k.split(".", 1)[0] for k in default_keys]
                + list(conf.keys()),
                default="",
            ).strip()
            if raw in {"", "b", "back", ":back"}:
                break
            if raw in {":show"}:
                continue
            if raw.startswith(":del"):
                dk = _prompt_choice_from_list(
                    "delete key: ",
                    sorted([k for k in conf.keys() if k != "_type"]),
                    allow_empty=True,
                )
                if dk:
                    conf.pop(dk, None)
                continue
            if "=" in raw:
                k, v = raw.split("=", 1)
                kk = k.strip()
                if kk == "_type":
                    print("ERR: _type locked; delete+re-add instead")
                    continue
                conf[kk] = _parse_scalar_value(v)
                continue

            field = raw.strip()
            if not field or field == "_type":
                continue
            cur = conf.get(field)
            cur_s = "" if cur is None else str(cur)
            hint = ""
            # pull hint from base defaults if present
            base_node = ((base_tree.get("evaluator") or {}) if isinstance(base_tree, dict) else {}).get((t or "").lower())
            if isinstance(base_node, dict) and field in base_node:
                hint = f" (default type={type(base_node[field]).__name__})"
            v = _prompt_line(f"{field}{hint} [{cur_s}]: " if cur_s else f"{field}{hint}: ", default=cur_s)
            if v.strip() == "":
                continue
            try:
                conf[field] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                conf[field] = _parse_scalar_value(v)
        eval_terms[alias] = conf

    while True:
        print("evaluator: [a]dd [e]dit [d]elete :back")
        if eval_terms:
            for k in sorted(eval_terms.keys()):
                t = (eval_terms.get(k) or {}).get("_type") if isinstance(eval_terms.get(k), dict) else None
                print(f"- {k} (_type={t})")
        cmd = input("evaluator> ").strip()
        if cmd in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if cmd in {"a", "add"}:
            t = _prompt_choice_from_list("type (Tab to complete): ", evaluator_types, allow_empty=True)
            if not t:
                continue
            alias_default = t
            alias = input(f"alias [{alias_default}]: ").strip() or alias_default
            if not alias:
                continue
            eval_terms[alias] = {"_type": t}
            # Immediately allow setting type-specific fields (press Enter to skip).
            _edit_alias(alias)
            continue
        if cmd in {"d", "del", "delete"}:
            if not eval_terms:
                continue
            alias = _prompt_choice_from_list("delete alias: ", sorted(list(eval_terms.keys())), allow_empty=True)
            if not alias:
                continue
            eval_terms.pop(alias, None)
            continue
        if cmd in {"e", "edit"}:
            if not eval_terms:
                continue
            alias = _prompt_choice_from_list("edit alias: ", sorted(list(eval_terms.keys())), allow_empty=True)
            if not alias:
                continue
            _edit_alias(alias)
            continue
        if cmd in {":?", ":help"}:
            print("Add/edit evaluator_terms entries used by stages.evaluator.")
            continue


def _compose_section_mutator(overlay_obj: dict, config_dir: pathlib.Path) -> None:
    mutator_types = [
        "MCMC",
        "LigandMPNN",
        "CarbonDesign",
        "ESMIF",
        "ESM2",
        "ESM3",
        "MotifBreaker",
        "DPLM",
        "SurfDesign",
        "SurfPro",
    ]
    mut_terms = _ensure_dict_at(overlay_obj, "mutator.mutator_terms")

    base_tree = _compose_default_config_tree(config_dir)

    def _default_keys_for_type(t: str) -> List[str]:
        key = (t or "").strip()
        # mutator section keys in base.yaml use exact casing for common ones
        node = ((base_tree.get("mutator") or {}) if isinstance(base_tree, dict) else {}).get(key)
        return _flatten_dict_keys(node, prefix="", max_depth=2) if node is not None else []

    def _edit_alias(alias: str) -> None:
        conf = mut_terms.get(alias) or {}
        if not isinstance(conf, dict):
            conf = {}
        t = str(conf.get("_type") or "").strip()
        default_keys = _default_keys_for_type(t)
        while True:
            print(json.dumps(conf, indent=2, sort_keys=True))
            raw = _prompt_words(
                "edit> field (Tab) or key=value (:back to stop): ",
                [":back", ":show", ":del"]
                + [k.split(".", 1)[0] for k in default_keys]
                + list(conf.keys()),
                default="",
            ).strip()
            if raw in {"", "b", "back", ":back"}:
                break
            if raw in {":show"}:
                continue
            if raw.startswith(":del"):
                dk = _prompt_choice_from_list(
                    "delete key: ",
                    sorted([k for k in conf.keys() if k != "_type"]),
                    allow_empty=True,
                )
                if dk:
                    conf.pop(dk, None)
                continue
            if "=" in raw:
                k, v = raw.split("=", 1)
                kk = k.strip()
                if kk == "_type":
                    print("ERR: _type locked; delete+re-add instead")
                    continue
                conf[kk] = _parse_scalar_value(v)
                continue

            field = raw.strip()
            if not field or field == "_type":
                continue
            cur = conf.get(field)
            cur_s = "" if cur is None else str(cur)
            hint = ""
            base_node = ((base_tree.get("mutator") or {}) if isinstance(base_tree, dict) else {}).get(t)
            if isinstance(base_node, dict) and field in base_node:
                hint = f" (default type={type(base_node[field]).__name__})"
            v = _prompt_line(f"{field}{hint} [{cur_s}]: " if cur_s else f"{field}{hint}: ", default=cur_s)
            if v.strip() == "":
                continue
            try:
                conf[field] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                conf[field] = _parse_scalar_value(v)
        mut_terms[alias] = conf

    while True:
        print("mutator: [a]dd [e]dit [d]elete :back")
        if mut_terms:
            for k in sorted(mut_terms.keys()):
                t = (mut_terms.get(k) or {}).get("_type") if isinstance(mut_terms.get(k), dict) else None
                print(f"- {k} (_type={t})")
        cmd = input("mutator> ").strip()
        if cmd in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if cmd in {"a", "add"}:
            t = _prompt_choice_from_list("type (Tab to complete): ", mutator_types, allow_empty=True)
            if not t:
                continue
            alias_default = t
            alias = input(f"alias [{alias_default}]: ").strip() or alias_default
            if not alias:
                continue
            mut_terms[alias] = {"_type": t}
            # Immediately allow setting type-specific fields (press Enter to skip).
            _edit_alias(alias)
            continue
        if cmd in {"d", "del", "delete"}:
            if not mut_terms:
                continue
            alias = _prompt_choice_from_list("delete alias: ", sorted(list(mut_terms.keys())), allow_empty=True)
            if not alias:
                continue
            mut_terms.pop(alias, None)
            continue
        if cmd in {"e", "edit"}:
            if not mut_terms:
                continue
            alias = _prompt_choice_from_list("edit alias: ", sorted(list(mut_terms.keys())), allow_empty=True)
            if not alias:
                continue
            _edit_alias(alias)
            continue
        if cmd in {":?", ":help"}:
            print("Add/edit mutator_terms entries used by stages.mutators (mutator.names is ignored).")
            continue


def _compose_section_loss(overlay_obj: dict) -> None:
    # NOTE: do NOT import `astrolabe.losses` here (slow). Use source index instead.
    loss_index = _loss_index_from_source()
    loss_types = sorted(list(loss_index.keys()))
    loss_terms = _ensure_dict_at(overlay_obj, "loss.loss_terms")

    while True:
        print("loss: [a]dd [e]dit [d]elete :back")
        if loss_terms:
            for k in sorted(loss_terms.keys()):
                t = (loss_terms.get(k) or {}).get("_type") if isinstance(loss_terms.get(k), dict) else None
                print(f"- {k} (_type={t})")
        cmd = input("loss> ").strip()
        if cmd in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if cmd in {"a", "add"}:
            lt = _prompt_choice_from_list("loss type (Tab to complete): ", loss_types, allow_empty=True)
            if not lt:
                continue
            meta = loss_index.get(lt) or {}
            desc = str(meta.get("description") or "").strip()
            if desc:
                print(f"desc: {desc}")
            req_params = list(meta.get("required_params") or [])
            opt_params = list(meta.get("optional_params") or [])
            alias_default = lt
            alias = input(f"alias [{alias_default}]: ").strip() or alias_default
            if not alias:
                continue
            if alias in loss_terms:
                print("ERR: alias exists")
                continue
            conf: dict = {"_type": lt}
            raw_for = input("_for (oligo, comma-sep; blank skip): ").strip()
            if raw_for:
                conf["_for"] = [x.strip() for x in raw_for.split(",") if x.strip()]
            for p in req_params:
                cur = conf.get(p)
                cur_s = "" if cur is None else str(cur)
                v = _prompt_line(f"{p} (required) [{cur_s}]: " if cur_s else f"{p} (required): ", default=cur_s)
                conf[p] = _parse_scalar_value(v)
            for p in opt_params:
                cur = conf.get(p)
                cur_s = "" if cur is None else str(cur)
                v = _prompt_line(f"{p} (optional) [{cur_s}] (blank skip): ", default=cur_s)
                if v.strip() == "" and cur is None:
                    continue
                if v.strip() == "":
                    continue
                conf[p] = _parse_scalar_value(v)
            loss_terms[alias] = conf
            continue
        if cmd in {"d", "del", "delete"}:
            if not loss_terms:
                continue
            alias = _prompt_choice_from_list("delete alias: ", sorted(list(loss_terms.keys())), allow_empty=True)
            if not alias:
                continue
            loss_terms.pop(alias, None)
            continue
        if cmd in {"e", "edit"}:
            if not loss_terms:
                continue
            alias = _prompt_choice_from_list("edit alias: ", sorted(list(loss_terms.keys())), allow_empty=True)
            if not alias:
                continue
            conf = loss_terms.get(alias) or {}
            if not isinstance(conf, dict):
                conf = {}
            print(json.dumps(conf, indent=2, sort_keys=True))
            raw = input("set 'key=value' (under loss.loss_terms.<alias>), blank stops: ").strip()
            while raw:
                if "=" in raw:
                    k, v = raw.split("=", 1)
                    conf[k.strip()] = _parse_scalar_value(v)
                raw = input("... ").strip()
            loss_terms[alias] = conf
            continue
        if cmd in {":?", ":help"}:
            print("Losses live under loss.loss_terms; stage expressions reference aliases.")
            continue


def _compose_section_predocker(overlay_obj: dict) -> None:
    from astrolabe.structure_processors.base import _PREDOCKER_TYPE_TO_MODULE  # type: ignore

    pred_terms = _ensure_dict_at(overlay_obj, "predocker.predocker_terms")

    types = sorted([str(x) for x in _PREDOCKER_TYPE_TO_MODULE.keys()])

    def _edit_alias(alias: str) -> None:
        conf = pred_terms.get(alias) or {}
        if not isinstance(conf, dict):
            conf = {}
        print(json.dumps(conf, indent=2, sort_keys=True))
        raw = input("set 'key=value' (under predocker.predocker_terms.<alias>), blank stops: ").strip()
        while raw:
            if "=" in raw:
                k, v = raw.split("=", 1)
                kk = k.strip()
                if kk == "_type":
                    print("ERR: _type locked; delete+re-add instead")
                else:
                    conf[kk] = _parse_scalar_value(v)
            raw = input("... ").strip()
        pred_terms[alias] = conf

    while True:
        print("predocker: [a]dd [e]dit [d]elete :back")
        if pred_terms:
            for k in sorted(pred_terms.keys()):
                t = (pred_terms.get(k) or {}).get("_type") if isinstance(pred_terms.get(k), dict) else None
                print(f"- {k} (_type={t})")
        cmd = input("predocker> ").strip()
        if cmd in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if cmd in {"a", "add"}:
            t = _prompt_choice_from_list("type (Tab to complete): ", types, allow_empty=True)
            if not t:
                continue
            alias_default = t
            alias = input(f"alias [{alias_default}]: ").strip() or alias_default
            if not alias:
                continue
            pred_terms[alias] = {"_type": t}
            # Immediately allow setting fields like exhaustiveness (press Enter to skip).
            _edit_alias(alias)
            continue
        if cmd in {"d", "del", "delete"}:
            if not pred_terms:
                continue
            alias = _prompt_choice_from_list("delete alias: ", sorted(list(pred_terms.keys())), allow_empty=True)
            if not alias:
                continue
            pred_terms.pop(alias, None)
            continue
        if cmd in {"e", "edit"}:
            if not pred_terms:
                continue
            alias = _prompt_choice_from_list("edit alias: ", sorted(list(pred_terms.keys())), allow_empty=True)
            if not alias:
                continue
            _edit_alias(alias)
            continue
        if cmd in {":?", ":help"}:
            print("Predockers are optional; mutator terms can reference predocker aliases.")
            continue


def _compose_section_stages(overlay_obj: dict) -> None:
    stages = _ensure_dict_at(overlay_obj, "stages")

    eval_terms = _get_by_dotted_key(overlay_obj, "evaluator.evaluator_terms") or {}
    mut_terms = _get_by_dotted_key(overlay_obj, "mutator.mutator_terms") or {}

    eval_aliases = sorted(list(eval_terms.keys())) if isinstance(eval_terms, dict) else []
    mut_aliases = sorted(list(mut_terms.keys())) if isinstance(mut_terms, dict) else []

    def _edit_stage(name: str) -> None:
        conf = stages.get(name) or {}
        if not isinstance(conf, dict):
            conf = {}
        print(f"edit stage: {name} (enter/b/back to stop)")
        fields = [
            "evaluator",
            "oligo",
            "mutators",
            "mutator_weights",
            "weighted_score_expression",
            "acceptance_threshold",
            "early_stopping",
            "_entrance",
            "mutation_num",
        ]
        while True:
            print(json.dumps(conf, indent=2, sort_keys=True))
            raw = _prompt_words(
                "stage> field (Tab) or key=value: ",
                fields
                + [":show", ":del", ":back"]
                + eval_aliases
                + mut_aliases,
                default="",
            ).strip()
            if raw in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
                break
            if raw in {":show"}:
                continue
            if raw.startswith(":del"):
                _, *rest = raw.split(" ", 1)
                tgt = rest[0].strip() if rest else ""
                if not tgt:
                    tgt = _prompt_choice_from_list("delete field: ", fields, allow_empty=True)
                if tgt and tgt in conf:
                    conf.pop(tgt, None)
                continue
            if "=" in raw:
                k, v = raw.split("=", 1)
                kk = k.strip()
                vv = v.strip()
                if kk in {"oligo", "mutators", "mutator_weights"}:
                    try:
                        conf[kk] = ast.literal_eval(vv)
                    except (ValueError, SyntaxError):
                        conf[kk] = [x.strip() for x in vv.split(",") if x.strip()]
                else:
                    conf[kk] = vv if kk.endswith("expression") or kk.endswith("threshold") or kk.endswith("stopping") or kk == "_entrance" else _parse_scalar_value(vv)
                continue

            field = raw
            if field == "evaluator":
                if eval_aliases:
                    ev = _prompt_choice_from_list("evaluator alias (Tab): ", eval_aliases, allow_empty=True)
                    if ev:
                        conf["evaluator"] = ev
                else:
                    ev = input("evaluator alias: ").strip()
                    if ev:
                        conf["evaluator"] = ev
                continue
            if field == "mutators":
                if mut_aliases:
                    muts = _prompt_multi_select_from_list("mutators (Tab; comma/space): ", mut_aliases)
                    if muts:
                        conf["mutators"] = muts
                else:
                    vv = input("mutators (comma): ").strip()
                    if vv:
                        conf["mutators"] = [x.strip() for x in vv.split(",") if x.strip()]
                continue
            if field == "oligo":
                cur_list = conf.get("oligo") if isinstance(conf.get("oligo"), list) else None
                ols = _compose_prompt_oligo_list(overlay_obj, default=cur_list or [])
                if ols:
                    conf["oligo"] = ols
                continue
            if field == "mutator_weights":
                muts = conf.get("mutators") if isinstance(conf.get("mutators"), list) else []
                if muts:
                    weights: List[float] = []
                    for m in muts:
                        w = _prompt_line(f"weight for {m} [1.0]: ", default="1.0")
                        weights.append(float(w))
                    conf["mutator_weights"] = weights
                else:
                    vv = input("mutator_weights (python list or comma): ").strip()
                    if vv:
                        try:
                            conf["mutator_weights"] = ast.literal_eval(vv)
                        except (ValueError, SyntaxError):
                            conf["mutator_weights"] = [float(x) for x in vv.split(",") if x.strip()]
                continue
            if field in {"weighted_score_expression", "acceptance_threshold", "early_stopping", "_entrance"}:
                cur = str(conf.get(field) or "")
                words = _compose_expression_words(overlay_obj, conf)
                v = _prompt_expr_line(f"{field}: ", default=cur, words=words)
                if v.strip() == "":
                    conf[field] = None
                else:
                    conf[field] = v
                continue
            if field in {"mutation_num"}:
                cur = conf.get("mutation_num")
                cur_s = "" if cur is None else str(cur)
                v = _prompt_line(f"mutation_num [{cur_s}]: " if cur_s else "mutation_num: ", default=cur_s)
                conf["mutation_num"] = _parse_scalar_value(v)
                continue

            print("ERR: unknown field")
        stages[name] = conf

    while True:
        print("stages: [a]dd [e]dit [d]elete :back")
        if stages:
            for k in sorted(stages.keys()):
                ev = (stages.get(k) or {}).get("evaluator") if isinstance(stages.get(k), dict) else None
                print(f"- {k} (evaluator={ev})")
        cmd = _prompt_words(
            "stages> ",
            ["add", "a", "edit", "e", "delete", "del", "d", ":back", "back", "b"]
            + sorted(list(stages.keys())),
            default="",
        ).strip()
        if cmd in {"", "b", "back", ":back", ":q", ":quit", ":exit"}:
            return
        if cmd in {"a", "add"}:
            name = input("stage name [default]: ").strip() or "default"
            if name in stages:
                print("ERR: exists")
                continue
            stages[name] = {}
            _edit_stage(name)
            continue
        if cmd in {"e", "edit"}:
            if not stages:
                stages["default"] = {}
            name = _prompt_choice_from_list(
                "stage name: ", sorted(list(stages.keys())), default="default", allow_empty=True
            ) or "default"
            _edit_stage(name)
            continue
        if cmd in {"d", "del", "delete"}:
            if not stages:
                continue
            name = _prompt_choice_from_list(
                "delete stage: ", sorted(list(stages.keys())), allow_empty=True
            )
            if not name:
                continue
            stages.pop(name, None)
            continue
        if cmd in {":?", ":help"}:
            if eval_aliases:
                print("Known evaluator aliases: " + ", ".join(eval_aliases))
            if mut_aliases:
                print("Known mutator aliases: " + ", ".join(mut_aliases))
            continue


def _cmd_compose(ns: argparse.Namespace) -> int:
    from astrolabe.cli_interactive import _cmd_compose as _impl

    return _impl(ns)


def _interactive_scaffold_prompt(key: str, current: Optional[Any]) -> Any:
    kind = _classify_scaffold_key_kind(key)
    cur_s = "" if current is None else str(current)

    while True:
        if cur_s:
            prompt = f"{key} ({kind}) [{cur_s}]: "
        else:
            prompt = f"{key} ({kind}): "
        raw = _prompt_line(prompt, default=cur_s, path_kind="dir" if kind == "dir_may_create" else "file" if kind == "file_or_null" else None)
        if not raw:
            print("ERR: empty")
            continue

        if raw in {":q", ":quit", ":exit"}:
            raise SystemExit(1)
        if raw in {":?", ":help"}:
            print("Commands: :back previous | :q quit | !<cmd> use stdout | :which <exe> | :glob <pattern>")
            continue
        if raw == ":back":
            return ":back"
        if raw.startswith(":which "):
            exe = raw.split(" ", 1)[1].strip()
            p = which(exe) or ""
            print(p if p else f"ERR: not found in PATH: {exe}")
            if p:
                cur_s = p
            continue
        if raw.startswith(":glob "):
            pat = raw.split(" ", 1)[1].strip()
            matches = sorted(glob.glob(_expand_user_vars(pat)))
            if not matches:
                print(f"ERR: no matches: {pat}")
                continue
            if len(matches) == 1:
                cur_s = matches[0]
                print(cur_s)
                continue
            for i, m in enumerate(matches[:50], start=1):
                print(f"{i:02d} {m}")
            choice = input("pick number: ").strip()
            try:
                idx = int(choice)
                cur_s = matches[idx - 1]
                print(cur_s)
            except (ValueError, IndexError):
                print("ERR: bad choice")
            continue
        if raw.startswith("!"):
            cmd = raw[1:].strip()
            if not cmd:
                print("ERR: empty shell cmd")
                continue
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"ERR: cmd failed rc={res.returncode}")
                if res.stderr:
                    print(res.stderr.strip().splitlines()[-1])
                continue
            out = (res.stdout or "").strip().splitlines()
            if not out:
                print("ERR: cmd stdout empty")
                continue
            raw = out[-1].strip()

        val = _parse_scalar_value(raw)
        ok, reason = _validate_scaffold_value(key, val)
        if ok:
            return val
        print(f"ERR: {reason}")


def _cmd_scaffold(ns: argparse.Namespace) -> int:
    from astrolabe.cli_interactive import _cmd_scaffold as _impl

    return _impl(ns)


def _classify_env_key_kind(key: str) -> str:
    if key.startswith("weight_path."):
        # Most weights are directories, but some are single checkpoint files.
        if key in {
            "weight_path.protenix",
            "weight_path.protenix_constraint",
        }:
            return "file"
        return "dir"
    if key.endswith(".python_bin"):
        return "file"
    if key.endswith(".repo_path") or key.endswith("repo_path"):
        return "dir"
    if key.isupper() and key.endswith("_PATH"):
        return "path"
    if key.endswith("_file") or key.endswith("_ckpt") or key.endswith("_config") or key.endswith("_features"):
        return "file"
    if key.endswith("_path") or key in {"dssp_path", "usalign_path", "obabel_path", "molfile_to_params"}:
        return "file"
    return "path"


def _get_doctor_env_value(env_obj: Any, key: str) -> Any:
    value = _get_by_dotted_key(env_obj, key)
    if value is None and "." not in str(key) and str(key).isupper():
        return os.environ.get(str(key))
    return value


def _expand_user_vars(raw: str) -> str:
    return os.path.expanduser(os.path.expandvars(raw))


def _candidate_env_paths(value: str, *, env_path: pathlib.Path | None = None) -> list[pathlib.Path]:
    expanded = _expand_user_vars(value)
    raw_path = pathlib.Path(expanded)
    if raw_path.is_absolute():
        return [raw_path]

    candidates: list[pathlib.Path] = []
    if env_path is not None:
        candidates.append((env_path.expanduser().resolve().parent / raw_path).resolve())
    repo_candidate = (_repo_root() / raw_path).resolve()
    if repo_candidate not in candidates:
        candidates.append(repo_candidate)
    cwd_candidate = (pathlib.Path.cwd() / raw_path).resolve()
    if cwd_candidate not in candidates:
        candidates.append(cwd_candidate)
    return candidates


def _resolve_existing_env_path(value: str, *, env_path: pathlib.Path | None = None) -> pathlib.Path | None:
    for candidate in _candidate_env_paths(value, env_path=env_path):
        if candidate.exists():
            return candidate
    return None


def _extract_oc_env_name(value: str) -> str | None:
    match = re.search(r"\$\{oc\.env:([^,}]+)", str(value or ""))
    if not match:
        return None
    return match.group(1).strip() or None


def _validate_path_value(
    key: str,
    value: str,
    *,
    env_path: pathlib.Path | None = None,
) -> tuple[bool, str]:
    kind = _classify_env_key_kind(key)
    if not value:
        return False, "empty"
    if "${" in value:
        return True, "interpolated"

    first_existing = _resolve_existing_env_path(value, env_path=env_path)
    if first_existing is None:
        return False, "missing"
    path = first_existing
    if kind == "file" and not path.is_file():
        return False, "not_a_file"
    if kind == "dir" and not path.is_dir():
        return False, "not_a_dir"
    if kind == "file" and (key.endswith("_path") or key.endswith(".python_bin")):
        # Best-effort executable check.
        if not os.access(str(path), os.X_OK):
            return False, "not_executable"
    return True, "ok"


def _doctor_env_repo_file_items(
    req: dict,
    env_obj: Any,
    *,
    env_path: pathlib.Path,
) -> List[DoctorItem]:
    items: List[DoctorItem] = []
    for spec in req.get("required_env_repo_files", []) or []:
        if not isinstance(spec, dict):
            continue
        base_key = str(spec.get("base_env_key") or "").strip()
        repo_rel = str(spec.get("path") or "").strip()
        label = str(spec.get("label") or pathlib.Path(repo_rel).name or "repo_file")
        if not base_key or not repo_rel:
            continue

        item_key = f"repo_file::{label}"
        base_value = _get_doctor_env_value(env_obj, base_key)
        if base_value is None:
            items.append(
                DoctorItem(
                    key=item_key,
                    value=f"{base_key}/{repo_rel}",
                    exists=False,
                    kind="missing_base",
                )
            )
            continue
        if not isinstance(base_value, str):
            items.append(
                DoctorItem(
                    key=item_key,
                    value=f"{base_value}/{repo_rel}",
                    exists=False,
                    kind="not_a_path",
                )
            )
            continue
        if "${" in base_value or "${" in repo_rel:
            items.append(
                DoctorItem(
                    key=item_key,
                    value=f"{base_value}/{repo_rel}",
                    exists=None,
                    kind="unknown",
                )
            )
            continue

        rel_path = pathlib.Path(_expand_user_vars(repo_rel))
        if rel_path.is_absolute():
            path = rel_path
        else:
            repo_path = _resolve_existing_env_path(base_value, env_path=env_path)
            if repo_path is None:
                repo_path = _candidate_env_paths(base_value, env_path=env_path)[0]
            path = (repo_path / rel_path).resolve()
        items.append(
            DoctorItem(
                key=item_key,
                value=str(path),
                exists=path.exists(),
                kind="file" if path.exists() and path.is_file() else "unknown",
            )
        )
    return items


def _interactive_env_prompt(
    key: str,
    current: Optional[str],
    *,
    env_path: pathlib.Path | None = None,
) -> str:
    kind = _classify_env_key_kind(key)
    cur = "" if current is None else str(current)
    while True:
        hint = f" ({kind})" if kind else ""
        if cur:
            prompt = f"{key}{hint} [{cur}]: "
        else:
            prompt = f"{key}{hint}: "
        raw = _prompt_line(prompt, default=cur, path_kind=kind if kind in {"file", "dir", "path"} else "path")
        if not raw:
            print("ERR: empty")
            continue

        if raw in {":q", ":quit", ":exit"}:
            raise SystemExit(1)
        if raw == ":back":
            return ":back"
        if raw in {":?", ":help"}:
            print("Commands: :back previous | :q quit | !<cmd> use stdout as value | :which <exe> | :glob <pattern>")
            continue
        if raw.startswith(":which "):
            exe = raw.split(" ", 1)[1].strip()
            p = which(exe) or ""
            print(p if p else f"ERR: not found in PATH: {exe}")
            if p:
                cur = p
            continue
        if raw.startswith(":glob "):
            pat = raw.split(" ", 1)[1].strip()
            matches = sorted(glob.glob(_expand_user_vars(pat)))
            if not matches:
                print(f"ERR: no matches: {pat}")
                continue
            if len(matches) == 1:
                cur = matches[0]
                print(cur)
                continue
            for i, m in enumerate(matches[:50], start=1):
                print(f"{i:02d} {m}")
            choice = input("pick number: ").strip()
            try:
                idx = int(choice)
                cur = matches[idx - 1]
                print(cur)
            except (ValueError, IndexError):
                print("ERR: bad choice")
            continue
        if raw.startswith("!"):
            cmd = raw[1:].strip()
            if not cmd:
                print("ERR: empty shell cmd")
                continue
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"ERR: cmd failed rc={res.returncode}")
                if res.stderr:
                    print(res.stderr.strip().splitlines()[-1])
                continue
            out = (res.stdout or "").strip().splitlines()
            if not out:
                print("ERR: cmd stdout empty")
                continue
            raw = out[-1].strip()

        # UX: if a file is required but user hits enter on an existing directory,
        # treat it as "drill into this dir" instead of a hard error / advancing.
        if kind == "file" and "${" not in raw:
            expanded = _expand_user_vars(raw)
            p = pathlib.Path(expanded)
            if p.exists() and p.is_dir():
                cur = str(p.resolve())
                if not cur.endswith(os.sep):
                    cur += os.sep
                print(cur)
                continue

        ok, reason = _validate_path_value(key, raw, env_path=env_path)
        if ok:
            return raw
        print(f"ERR: {reason} -> {raw}")


def _cmd_env_wizard(ns: argparse.Namespace) -> int:
    from astrolabe.cli_interactive import _cmd_env_wizard as _impl

    return _impl(ns)


def _cmd_doctor(ns: argparse.Namespace) -> int:
    import importlib

    config_dir = pathlib.Path(ns.config_dir).resolve() if ns.config_dir else _default_config_dir()
    if ns.env:
        env_path = pathlib.Path(ns.env).resolve()
    else:
        env_path = default_local_env_path_for_config_dir(config_dir)
        if not env_path.exists():
            fallback_env_path = config_dir / "env.yaml"
            if fallback_env_path.exists():
                env_path = fallback_env_path.resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"env.yaml not found: {env_path}")

    resolve = bool(getattr(ns, "resolve", False) or getattr(ns, "strict", False))
    task_name = getattr(ns, "task", None)
    python_deps = bool(getattr(ns, "python_deps", False) or getattr(ns, "strict", False))

    req = None
    task_cfg = None
    rapidock_ckpt = "rapidock_global.pt"
    if task_name:
        task_path = config_dir / f"{task_name}.yaml"
        if not task_path.exists():
            tasks = _list_available_tasks(config_dir)
            hint = f" Available tasks: {', '.join(tasks)}" if tasks else ""
            raise FileNotFoundError(f"Task template not found: {task_path}.{hint}")
        task_cfg = OmegaConf.load(str(task_path))
        task_cfg = _strip_defaults_key(task_cfg)
        req = _task_requirements(OmegaConf.to_container(task_cfg, resolve=False))
        rapidock_cfg = ((task_cfg.get("predocker") or {}).get("RAPiDock") or {})
        if isinstance(rapidock_cfg, dict):
            rapidock_ckpt = str(rapidock_cfg.get("ckpt") or rapidock_ckpt)

    env_cfg = OmegaConf.load(str(env_path))
    env_obj = OmegaConf.to_container(env_cfg, resolve=resolve)

    items: List[DoctorItem] = []
    if req is None:
        items = _doctor_scan_env(env_path, resolve=resolve)
    else:
        for key in req.get("required_env_keys", []) or []:
            value = _get_doctor_env_value(env_obj, key)
            if value is None:
                items.append(DoctorItem(key=key, value=None, exists=False, kind="missing"))
                continue
            if not isinstance(value, str):
                items.append(DoctorItem(key=key, value=str(value), exists=True, kind="value"))
                continue
            if "${" in value:
                items.append(DoctorItem(key=key, value=value, exists=None, kind="unknown"))
                continue
            ok, reason = _validate_path_value(key, value, env_path=env_path)
            resolved = _resolve_existing_env_path(value, env_path=env_path)
            if ok and resolved is not None:
                kind = "dir" if resolved.is_dir() else "file" if resolved.is_file() else "path"
                items.append(DoctorItem(key=key, value=value, exists=True, kind=kind))
            else:
                items.append(DoctorItem(key=key, value=value, exists=False, kind=reason))

        # Derived checks: LigandMPNN checkpoint files
        if "weight_path.ligandmpnn" in set(req.get("required_env_keys") or []):
            base = _get_by_dotted_key(env_obj, "weight_path.ligandmpnn")
            if isinstance(base, str) and "${" not in base:
                base_p = pathlib.Path(_expand_user_vars(base))
                ckpts = [
                    "global_label_membrane_mpnn_v_48_020.pt",
                    "ligandmpnn_v_32_020_25.pt",
                    "per_residue_label_membrane_mpnn_v_48_020.pt",
                    "proteinmpnn_v_48_020.pt",
                    "solublempnn_v_48_020.pt",
                ]
                for fn in ckpts:
                    p = base_p / fn
                    items.append(
                        DoctorItem(
                            key=f"weight_path.ligandmpnn::{fn}",
                            value=str(p),
                            exists=p.exists(),
                            kind="file" if p.exists() else "unknown",
                        )
                    )

        if "weight_path.rapidock" in set(req.get("required_env_keys") or []):
            base = _get_by_dotted_key(env_obj, "weight_path.rapidock")
            if isinstance(base, str) and "${" not in base:
                base_p = pathlib.Path(_expand_user_vars(base))
                for fn in ("model_parameters.yml", rapidock_ckpt):
                    p = base_p / fn
                    items.append(
                        DoctorItem(
                            key=f"weight_path.rapidock::{fn}",
                            value=str(p),
                            exists=p.exists(),
                            kind="file" if p.exists() else "unknown",
                        )
                    )

        # Repo-local artifacts (e.g. AF3 pickles)
        for p_str in req.get("required_repo_files", []) or []:
            p = pathlib.Path(p_str)
            items.append(
                DoctorItem(
                    key=f"repo_file::{p.name}",
                    value=str(p),
                    exists=p.exists(),
                    kind="file" if p.exists() else "unknown",
                )
            )
        items.extend(
            _doctor_env_repo_file_items(req, env_obj, env_path=env_path)
        )

    items_sorted = sorted(items, key=lambda it: str(it.key))

    missing = [it for it in items_sorted if it.exists is False]
    interpolated = [it for it in items_sorted if it.exists is None]
    ok = [it for it in items_sorted if it.exists is True]

    py_missing = []
    py_ok = []
    if python_deps:
        dep_mods = ["absl", "pyrosetta", "prody", "meeko"]
        if req is not None:
            dep_mods = [m for m in req.get("required_imports", []) or []]
        for mod in dep_mods:
            if mod.startswith("__"):
                py_missing.append({"type": "config", "name": mod})
                continue
            try:
                importlib.import_module(mod)
                py_ok.append({"type": "import", "name": mod})
            except Exception as exc:
                py_missing.append({"type": "import", "name": mod, "error": str(exc)})

        if req is not None and "rapidock" in {
            str(t).lower() for t in (req.get("predocker_types") or [])
        }:
            rapidock_python = _get_by_dotted_key(env_obj, "subprocess_envs.rapidock.python_bin")
            rapidock_repo = _get_by_dotted_key(env_obj, "subprocess_envs.rapidock.repo_path")
            rapidock_model_dir = _get_by_dotted_key(env_obj, "weight_path.rapidock")
            if (
                isinstance(rapidock_python, str)
                and isinstance(rapidock_repo, str)
                and isinstance(rapidock_model_dir, str)
                and "${" not in rapidock_python
                and "${" not in rapidock_repo
                and "${" not in rapidock_model_dir
            ):
                try:
                    _probe_rapidock_imports(
                        python_bin=str(_expand_user_vars(rapidock_python)),
                        repo_path=str(_expand_user_vars(rapidock_repo)),
                    )
                    py_ok.append({"type": "rapidock_subprocess", "name": "imports"})
                except Exception as exc:
                    py_missing.append(
                        {
                            "type": "rapidock_subprocess",
                            "name": "imports",
                            "error": str(exc),
                        }
                    )
                try:
                    _probe_rapidock_daemon(
                        python_bin=str(_expand_user_vars(rapidock_python)),
                        repo_path=str(_expand_user_vars(rapidock_repo)),
                        model_dir=str(_expand_user_vars(rapidock_model_dir)),
                        ckpt=rapidock_ckpt,
                    )
                    py_ok.append({"type": "rapidock_daemon", "name": "ping"})
                except Exception as exc:
                    py_missing.append(
                        {
                            "type": "rapidock_daemon",
                            "name": "ping",
                            "error": str(exc),
                        }
                    )

        if req is not None and any(
            str(t).lower() in {"autodock_gpu", "autodock_vina"}
            for t in (req.get("predocker_types") or [])
        ):
            meeko_python = _get_by_dotted_key(env_obj, "subprocess_envs.meeko.python_bin")
            meeko_pythonpath = _get_by_dotted_key(env_obj, "subprocess_envs.meeko.pythonpath")
            if isinstance(meeko_python, str) and "${" not in meeko_python:
                try:
                    _probe_meeko_imports(
                        python_bin=str(_expand_user_vars(meeko_python)),
                        pythonpath=str(_expand_user_vars(meeko_pythonpath))
                        if isinstance(meeko_pythonpath, str) and "${" not in meeko_pythonpath
                        else None,
                    )
                    py_ok.append({"type": "meeko_subprocess", "name": "imports"})
                except Exception as exc:
                    py_missing.append(
                        {
                            "type": "meeko_subprocess",
                            "name": "imports",
                            "error": str(exc),
                        }
                    )

    if ns.format == "json":
        payload = {
            "env": str(env_path),
            "task": task_name,
            "requirements": req,
            "ok": [it._asdict() for it in ok],
            "missing": [it._asdict() for it in missing],
            "interpolated": [it._asdict() for it in interpolated],
            "python_deps": {"ok": py_ok, "missing": py_missing},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        reason_map = {}
        if isinstance(req, dict):
            reason_map = dict(req.get("env_key_reasons") or {})

        def _doctor_hint(item: DoctorItem) -> str:
            env_var = _extract_oc_env_name(str(item.value or ""))
            if item.exists is None:
                base_hint = "set env vars or use --resolve"
                if env_var:
                    base_hint = f"set env var {env_var} (shell or .env) or use --resolve"
            else:
                base_hint = "mount or edit env.yaml"
            reason = reason_map.get(str(item.key))
            if reason:
                return f"{base_hint}; {reason}"
            return base_hint

        print(f"env: {env_path}")
        if task_name:
            print(f"task: {task_name}")
        for it in ok:
            if ns.show_ok:
                print(f"OK        {it.key} = {it.value}")
        for it in interpolated:
            print(f"UNRESOLVED {it.key} = {it.value}  ({_doctor_hint(it)})")
        for it in missing:
            print(f"MISSING   {it.key} = {it.value}  ({_doctor_hint(it)})")
        print(
            f"summary: ok={len(ok)} missing={len(missing)} unresolved={len(interpolated)}"
        )
        if python_deps:
            for it in py_ok:
                if ns.show_ok:
                    print(f"PY OK    import {it['name']}")
            for it in py_missing:
                if it["type"] == "config":
                    print("PY MISS  config: enable_pyrosetta=false but task requires PyRosetta-based terms")
                else:
                    print(f"PY MISS  import {it['name']}  ({it.get('error')})")

    strict = bool(getattr(ns, "strict", False))
    if (missing or (strict and interpolated) or py_missing) and not ns.soft_fail:
        return 2
    return 0


def _cmd_collect_scores(ns: argparse.Namespace) -> int:
    from astrolabe.posthoc.collect import (
        MissingRequiredMetricsError,
        collect_scores,
    )

    raw_out_roots = list(getattr(ns, "out_root", []) or [])
    try:
        kwargs = dict(
            db_path=pathlib.Path(ns.db),
            workers=int(ns.workers),
            binder_chain=str(ns.binder_chain),
            target_chain=str(ns.target_chain),
            task_regex=getattr(ns, "task_regex", None),
            run_regex=getattr(ns, "run_regex", None),
            export_csv=pathlib.Path(ns.export_csv) if ns.export_csv else None,
            archive_dir=pathlib.Path(ns.archive_dir) if getattr(ns, "archive_dir", None) else None,
            archive_workers=int(getattr(ns, "archive_workers", 0)),
            archive_overwrite=bool(getattr(ns, "archive_overwrite", False)),
            required_metrics=list(getattr(ns, "required_metric", []) or []),
            required_metrics_preset=pathlib.Path(ns.preset)
            if getattr(ns, "preset", None)
            else None,
            fail_on_missing_required=not bool(
                getattr(ns, "allow_missing_required", False)
            ),
        )
        if len(raw_out_roots) == 1:
            kwargs["out_root"] = pathlib.Path(raw_out_roots[0])
        else:
            kwargs["out_roots"] = [pathlib.Path(p) for p in raw_out_roots]
        summary = collect_scores(**kwargs)
    except MissingRequiredMetricsError as e:
        print(str(e), file=sys.stderr)
        return 2

    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        out_roots = list(summary.get("out_roots") or [])
        if len(out_roots) <= 1:
            print(f"out_root: {summary['out_root']}")
        else:
            print(f"out_roots: {','.join(out_roots)}")
        print(f"db: {summary['db_path']}")
        print(
            f"runs: found={summary['runs_found']} skipped={summary['runs_skipped']} processed={summary['runs_processed']}"
        )
        print(f"parse_error_total: {summary.get('parse_error_total', 0)}")
        required_metrics = summary.get("required_metrics") or []
        if required_metrics:
            print(f"required_metrics: {','.join(required_metrics)}")
            print(
                f"required_failed_records_total: {summary.get('required_failed_records_total', 0)}"
            )
        if summary.get("archive_dir"):
            print(f"archive_dir: {summary['archive_dir']}")
            print(
                f"archive: created={summary.get('archived_created', 0)} skipped={summary.get('archived_skipped', 0)} failed={summary.get('archive_failed', 0)}"
            )
        if summary.get("export_csv"):
            print(f"export_csv: {summary['export_csv']}")
    return 0


def _slugify_posthoc_filter_selector(raw: str) -> str:
    text = str(raw or "").strip()
    replacements = [
        (">=", "_ge_"),
        ("<=", "_le_"),
        ("!=", "_ne_"),
        ("==", "_eq_"),
        (">", "_gt_"),
        ("<", "_lt_"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = re.sub(r"\band\b", "_and_", text, flags=re.IGNORECASE)
    text = re.sub(r"\bor\b", "_or_", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not text:
        return "selector"
    if len(text) <= 48:
        return text
    return text[:48].rstrip("_")


def _default_posthoc_filter_dst(
    *,
    db_path: pathlib.Path,
    preset_path: str | None,
    filter_expr: str | None,
) -> pathlib.Path:
    root = pathlib.Path(db_path).expanduser().resolve().parent
    if preset_path:
        label = f"preset__{pathlib.Path(preset_path).stem}"
    else:
        label = f"filter__{_slugify_posthoc_filter_selector(str(filter_expr or 'selector'))}"
    return root / label


def _default_posthoc_refold_dst(
    *,
    db_path: pathlib.Path,
    preset_path: str | None,
    filter_expr: str | None,
) -> pathlib.Path:
    root = pathlib.Path(db_path).expanduser().resolve().parent
    if preset_path:
        label = f"refold__preset__{pathlib.Path(preset_path).stem}"
    else:
        label = f"refold__filter__{_slugify_posthoc_filter_selector(str(filter_expr or 'selector'))}"
    return root / label


def _default_posthoc_report_dst(
    *,
    db_path: pathlib.Path,
    preset_path: str | None,
    filter_expr: str | None,
) -> pathlib.Path:
    root = pathlib.Path(db_path).expanduser().resolve().parent
    if preset_path:
        label = f"report__preset__{pathlib.Path(preset_path).stem}"
    else:
        label = f"report__filter__{_slugify_posthoc_filter_selector(str(filter_expr or 'selector'))}"
    return root / label


def _default_posthoc_channel_screen_dst(
    *,
    structure_path: pathlib.Path,
) -> pathlib.Path:
    root = pathlib.Path(structure_path).expanduser().resolve().parent
    name = pathlib.Path(structure_path).name
    for suffix in (".gz", ".mmcif", ".cif", ".pdb", ".ent"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    label = f"{_sanitize_name(name)}__channel_screen"
    return root / label


def _cmd_filter_structures(ns: argparse.Namespace) -> int:
    import importlib

    posthoc_filtering = importlib.import_module('astrolabe.posthoc.filtering')

    if bool(getattr(ns, "preview_preset", False)):
        if not getattr(ns, "preset", None):
            print("--preview-preset requires --preset", file=sys.stderr)
            return 2
        preview_preset = getattr(posthoc_filtering, "preview_preset", None)
        if preview_preset is None:
            raise RuntimeError("posthoc filtering module does not expose preview_preset")
        summary = preview_preset(pathlib.Path(ns.preset))
        if getattr(ns, "format", "text") == "json":
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"preset: {summary['preset']}")
            print(f"rules: {summary['rule_count']}")
            if summary.get("required_metrics"):
                print(f"required_metrics: {','.join(summary['required_metrics'])}")
            print(f"preview_filter_expr: {summary['preview_filter_expr']}")
        return 0

    if bool(getattr(ns, "preset", None)) == bool(getattr(ns, "filter_expr", None)):
        print(
            "Exactly one of --preset or --filter-expr/--expr is required.",
            file=sys.stderr,
        )
        return 2

    if not getattr(ns, "db", None):
        print("--db is required unless --preview-preset is used", file=sys.stderr)
        return 2
    dst_dir = (
        pathlib.Path(ns.dst)
        if getattr(ns, "dst", None)
        else _default_posthoc_filter_dst(
            db_path=pathlib.Path(ns.db),
            preset_path=getattr(ns, "preset", None),
            filter_expr=getattr(ns, "filter_expr", None),
        )
    )

    runtime_error_cls = getattr(posthoc_filtering, "FilterExpressionRuntimeError", RuntimeError)
    try:
        summary = posthoc_filtering.filter_structures(
            db_path=pathlib.Path(ns.db),
            preset_path=pathlib.Path(ns.preset) if ns.preset else None,
            filter_expr=str(ns.filter_expr) if getattr(ns, "filter_expr", None) else None,
            dst_dir=dst_dir,
            prefer_oligo=str(ns.prefer_oligo),
            run_selection=str(getattr(ns, "run_selection", "all-passing")),
            hardlink=not bool(getattr(ns, "copy", False)),
            overwrite=bool(getattr(ns, "overwrite", False)),
            flat=bool(getattr(ns, "flat", False)),
            report_csv=pathlib.Path(ns.report_csv) if ns.report_csv else None,
            out_db=pathlib.Path(ns.out_db) if getattr(ns, "out_db", None) else None,
            fail_on_missing_required=not bool(getattr(ns, "allow_missing_required", False)),
            dry_run=bool(getattr(ns, "dry_run", False)),
        )
    except posthoc_filtering.MissingRequiredMetricsError as e:
        print(str(e), file=sys.stderr)
        return 2
    except runtime_error_cls as e:
        print(str(e), file=sys.stderr)
        return 2
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"db: {summary['db_path']}")
        if summary.get("db_out_root"):
            print(f"db_out_root: {summary['db_out_root']}")
        print(f"selector_kind: {summary.get('selector_kind', 'preset')}")
        if summary.get("preset"):
            print(f"preset: {summary['preset']}")
        if summary.get("filter_expr"):
            print(f"filter_expr: {summary['filter_expr']}")
        print(f"prefer_oligo: {summary['prefer_oligo']}")
        print(f"run_selection: {summary.get('run_selection', 'all-passing')}")
        if summary.get("dry_run"):
            print("dry_run: true")
        print(f"dst: {summary['dst_dir']}")
        print(f"selected: {summary['selected']} / {summary['records_considered']}")
        if summary.get("dry_run"):
            print(f"linkable: {summary.get('linkable', 0)}")
        elif "linked" in summary:
            print(f"linked: {summary['linked']}")
        if summary.get("structure_missing"):
            print(f"structure_missing: {summary['structure_missing']}")
        if summary.get("report_csv"):
            print(f"report_csv: {summary['report_csv']}")
        if summary.get("out_db"):
            print(f"out_db: {summary['out_db']}")
    return 0


def _cmd_prune_outputs(ns: argparse.Namespace) -> int:
    import importlib

    posthoc_filtering = importlib.import_module("astrolabe.posthoc.filtering")
    if bool(getattr(ns, "preset", None)) == bool(getattr(ns, "filter_expr", None)):
        print(
            "Exactly one of --preset or --filter-expr/--expr is required.",
            file=sys.stderr,
        )
        return 2

    summary = posthoc_filtering.prune_outputs(
        db_path=pathlib.Path(ns.db),
        preset_path=pathlib.Path(ns.preset) if ns.preset else None,
        filter_expr=str(ns.filter_expr) if getattr(ns, "filter_expr", None) else None,
        prefer_oligo=str(ns.prefer_oligo),
        run_selection=str(getattr(ns, "run_selection", "all-passing")),
        delete_scope=str(ns.delete_scope),
        report_csv=pathlib.Path(ns.report_csv) if ns.report_csv else None,
        dry_run=not bool(getattr(ns, "yes", False)),
    )

    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"db: {summary['db_path']}")
        print(f"selector_kind: {summary.get('selector_kind', 'preset')}")
        if summary.get("preset"):
            print(f"preset: {summary['preset']}")
        if summary.get("filter_expr"):
            print(f"filter_expr: {summary['filter_expr']}")
        print(f"prefer_oligo: {summary['prefer_oligo']}")
        print(f"run_selection: {summary['run_selection']}")
        print(f"delete_scope: {summary['delete_scope']}")
        if summary.get("dry_run"):
            print("dry_run: true")
        print(f"selected: {summary['selected']}")
        print(f"deleted_paths: {summary['deleted_paths']}")
        if summary.get("missing_paths"):
            print(f"missing_paths: {summary['missing_paths']}")
        if summary.get("delete_errors"):
            print(f"delete_errors: {summary['delete_errors']}")
        if summary.get("report_csv"):
            print(f"report_csv: {summary['report_csv']}")
    return 0


def _cmd_posthoc_report(ns: argparse.Namespace) -> int:
    from astrolabe.posthoc.report import generate_posthoc_report

    if bool(getattr(ns, "preset", None)) == bool(getattr(ns, "filter_expr", None)):
        print(
            "Exactly one of --preset or --filter-expr/--expr is required.",
            file=sys.stderr,
        )
        return 2
    dst_dir = (
        pathlib.Path(ns.dst)
        if getattr(ns, "dst", None)
        else _default_posthoc_report_dst(
            db_path=pathlib.Path(ns.db),
            preset_path=getattr(ns, "preset", None),
            filter_expr=getattr(ns, "filter_expr", None),
        )
    )
    summary = generate_posthoc_report(
        db_path=pathlib.Path(ns.db),
        preset_path=pathlib.Path(ns.preset) if ns.preset else None,
        filter_expr=str(ns.filter_expr) if getattr(ns, "filter_expr", None) else None,
        prefer_oligo=str(ns.prefer_oligo),
        dst_dir=dst_dir,
        trend_metrics=list(getattr(ns, "trend_metrics", []) or []),
    )
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"db: {summary['db_path']}")
        print(f"selector_kind: {summary.get('selector_kind', 'preset')}")
        if summary.get("preset"):
            print(f"preset: {summary['preset']}")
        if summary.get("filter_expr"):
            print(f"filter_expr: {summary['filter_expr']}")
        print(f"prefer_oligo: {summary['prefer_oligo']}")
        print(f"dst: {summary['dst_dir']}")
        print(f"trend_metrics: {','.join(summary.get('trend_metrics', []))}")
        print(f"summary_csv: {summary['summary_csv']}")
        print(f"per_step_csv: {summary['per_step_csv']}")
        print(f"report_md: {summary['report_md']}")
        print(f"step_trends_png: {summary['step_trends_png']}")
        print(f"success_rate_png: {summary['success_rate_png']}")
    return 0


def _cmd_posthoc_channel_screen(ns: argparse.Namespace) -> int:
    from astrolabe.cli_channel import parse_ligand_residue_selector
    from astrolabe.posthoc.rosetta_direct_channel import run_pocket_fixed_screen

    structure_path = pathlib.Path(ns.structure).expanduser().resolve()
    mutable_positions_path = pathlib.Path(ns.mutable_positions).expanduser().resolve()
    spheres_path = pathlib.Path(ns.spheres).expanduser().resolve()
    reference_structure_path = (
        pathlib.Path(ns.reference_structure).expanduser().resolve()
        if getattr(ns, "reference_structure", None)
        else structure_path
    )
    out_dir = (
        pathlib.Path(ns.out_dir).expanduser().resolve()
        if getattr(ns, "out_dir", None)
        else _default_posthoc_channel_screen_dst(structure_path=structure_path)
    )
    ligand_ref = parse_ligand_residue_selector(ns.ligand_residue)
    summary = run_pocket_fixed_screen(
        structure_path=structure_path,
        protein_chain_id=str(ns.protein_chain),
        mutable_positions_path=mutable_positions_path,
        ligand_chain_id=ligand_ref.chain_id,
        ligand_resname=ligand_ref.resname,
        ligand_resseq=int(ligand_ref.resseq),
        ligand_icode=ligand_ref.icode,
        reference_spheres_path=spheres_path,
        output_root=out_dir,
        reference_structure_path=reference_structure_path,
        alignment_chains=str(getattr(ns, "alignment_chains", None) or ns.protein_chain),
        target_aas=str(getattr(ns, "target_aas", "A,S,T,G")),
        contact_distance=float(getattr(ns, "contact_distance", 5.0)),
        radius_gate_path_len_threshold=float(
            getattr(ns, "radius_gate_path_len_threshold", 40.0)
        ),
        hole_executable=getattr(ns, "hole_executable", None) or None,
        hole_radius_file=(
            getattr(ns, "hole_radius_file", None)
            or os.environ.get("GH_HOLE_RADIUS_FILE")
            or None
        ),
    )
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"structure: {summary['structure_path']}")
        print(f"out_dir: {summary['output_root']}")
        print(f"contact_distance: {summary['contact_distance']}")
        print(
            f"mutable_positions: {summary['mutable_positions_count']} protected={summary['protected_positions_count']} design={summary['design_positions_count']}"
        )
        if summary.get("protected_residue_ids"):
            print(f"protected_residues: {','.join(summary['protected_residue_ids'])}")
        if summary.get("design_residue_ids"):
            print(f"design_residues: {','.join(summary['design_residue_ids'])}")
        print(f"target_aas: {','.join(summary.get('target_aas', []))}")
        if summary.get("best_variant"):
            print(
                f"best_variant: {summary['best_variant']} radius={summary['best_variant_min_radius']}"
            )
        print(f"plan_json: {summary['plan_json']}")
        print(f"summary_json: {summary['summary_json']}")
        print(f"summary_csv: {summary['summary_csv']}")
    return 0


def _cmd_refold_export(ns: argparse.Namespace) -> int:
    from astrolabe.posthoc.refold import export_refold_manifest

    summary = export_refold_manifest(
        db_path=pathlib.Path(ns.db),
        preset_path=pathlib.Path(ns.preset) if ns.preset else None,
        filter_expr=str(ns.filter_expr) if getattr(ns, "filter_expr", None) else None,
        out_manifest=pathlib.Path(ns.out_manifest),
        prefer_oligo=str(ns.prefer_oligo),
        run_selection=str(getattr(ns, "run_selection", "all-passing")),
        passed_only=bool(ns.passed_only),
        report_csv=pathlib.Path(ns.report_csv) if ns.report_csv else None,
        relpath_base=pathlib.Path(ns.relpath_base) if ns.relpath_base else None,
        path_map=list(getattr(ns, "path_map", []) or []),
        limit=int(ns.limit) if ns.limit is not None else None,
    )
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"db: {summary['db_path']}")
        print(f"selector_kind: {summary.get('selector_kind', 'preset')}")
        if summary.get("preset"):
            print(f"preset: {summary['preset']}")
        if summary.get("filter_expr"):
            print(f"filter_expr: {summary['filter_expr']}")
        print(f"prefer_oligo: {summary['prefer_oligo']}")
        print(f"run_selection: {summary.get('run_selection', 'all-passing')}")
        print(f"passed_only: {summary['passed_only']}")
        print(f"out_manifest: {summary['out_manifest']}")
        if summary.get("report_csv"):
            print(f"report_csv: {summary['report_csv']}")
        print(
            f"records: considered={summary['records_considered']} passed={summary['records_passed']} written={summary['written']} failed={summary['failed']}"
        )
    return 0


def _cmd_refold_from_db(ns: argparse.Namespace) -> int:
    from astrolabe.posthoc.refold import refold_from_db

    if bool(getattr(ns, "preset", None)) == bool(getattr(ns, "filter_expr", None)):
        print(
            "Exactly one of --preset or --filter-expr/--expr is required.",
            file=sys.stderr,
        )
        return 2
    try:
        evaluator_backends = refold_backends.refold_backend_names(ns.evaluators)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dst_dir = (
        pathlib.Path(ns.dst)
        if getattr(ns, "dst", None)
        else _default_posthoc_refold_dst(
            db_path=pathlib.Path(ns.db),
            preset_path=getattr(ns, "preset", None),
            filter_expr=getattr(ns, "filter_expr", None),
        )
    )

    summary = refold_from_db(
        db_path=pathlib.Path(ns.db),
        preset_path=pathlib.Path(ns.preset) if ns.preset else None,
        filter_expr=str(ns.filter_expr) if getattr(ns, "filter_expr", None) else None,
        dst_dir=dst_dir,
        prefer_oligo=str(ns.prefer_oligo),
        run_selection=str(getattr(ns, "run_selection", "all-passing")),
        evaluators=evaluator_backends,
        collect_workers=int(getattr(ns, "collect_workers", 0)),
        relpath_base=pathlib.Path(ns.relpath_base) if ns.relpath_base else None,
        path_map=list(getattr(ns, "path_map", []) or []),
        keep_protenix_structures=bool(getattr(ns, "keep_protenix_structures", False)),
        config_yaml=pathlib.Path(ns.config_yaml) if getattr(ns, "config_yaml", None) else None,
        workers=int(getattr(ns, "workers", 1) or 1),
        dry_run=bool(getattr(ns, "dry_run", False)),
        limit=int(ns.limit) if ns.limit is not None else None,
    )
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"db: {summary['db_path']}")
        print(f"selector_kind: {summary.get('selector_kind', 'preset')}")
        if summary.get("preset"):
            print(f"preset: {summary['preset']}")
        if summary.get("filter_expr"):
            print(f"filter_expr: {summary['filter_expr']}")
        print(f"prefer_oligo: {summary['prefer_oligo']}")
        print(f"run_selection: {summary['run_selection']}")
        print(f"dst: {summary['dst_dir']}")
        print(f"manifest: {summary['manifest_jsonl']}")
        print(f"refold_out_root: {summary['refold_out_root']}")
        print(f"evaluators: {','.join(summary['evaluators'])}")
        print(f"exported: {summary['exported']}")
        print(
            f"records: processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']} dry_run={summary['dry_run']}"
        )
        print(
            "groups: "
            f"output={summary.get('groups', 0)} "
            f"resource={summary.get('resource_groups', 0)} "
            f"evaluator_cache={summary.get('evaluator_cache_hits', 0)}/{summary.get('evaluator_cache_misses', 0)}"
        )
        print(
            "workers: "
            f"effective={summary.get('workers', 1)} "
            f"requested={summary.get('workers_requested', summary.get('workers', 1))}"
        )
        print(
            "ran: "
            f"af3={summary.get('ran_af3', 0)} "
            f"protenix={summary.get('ran_protenix', 0)} "
            f"reused_af3={summary.get('reused_af3', 0)} "
            f"reused_protenix={summary.get('reused_protenix', 0)}"
        )
        if not summary.get("dry_run"):
            print(f"collect_db: {summary['collect_db_path']}")
            print(f"collected_runs: {summary['collected_runs']}")
        if summary.get("failure_report"):
            print(f"failure_report: {summary['failure_report']}")
    return 0


def _cmd_refold_run(ns: argparse.Namespace) -> int:
    from astrolabe.posthoc.refold import run_refold

    try:
        evaluator_backends = refold_backends.refold_backend_names(ns.evaluators)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = run_refold(
        manifest_jsonl=pathlib.Path(ns.manifest),
        out_root=pathlib.Path(ns.out_root),
        evaluators=evaluator_backends,
        relpath_base=pathlib.Path(ns.relpath_base) if ns.relpath_base else None,
        path_map=list(getattr(ns, "path_map", []) or []),
        keep_protenix_structures=bool(getattr(ns, "keep_protenix_structures", False)),
        config_yaml=pathlib.Path(ns.config_yaml) if getattr(ns, "config_yaml", None) else None,
        workers=int(getattr(ns, "workers", 1) or 1),
        dry_run=bool(getattr(ns, "dry_run", False)),
    )
    if getattr(ns, "format", "text") == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"manifest: {summary['manifest_jsonl']}")
        print(f"out_root: {summary['out_root']}")
        print(f"evaluators: {','.join(summary['evaluators'])}")
        print(
            f"records: processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']} dry_run={summary['dry_run']}"
        )
        print(
            "groups: "
            f"output={summary.get('groups', 0)} "
            f"resource={summary.get('resource_groups', 0)} "
            f"evaluator_cache={summary.get('evaluator_cache_hits', 0)}/{summary.get('evaluator_cache_misses', 0)}"
        )
        print(
            "workers: "
            f"effective={summary.get('workers', 1)} "
            f"requested={summary.get('workers_requested', summary.get('workers', 1))}"
        )
        print(f"ran: af3={summary['ran_af3']} protenix={summary['ran_protenix']}")
        if summary.get("failure_report"):
            print(f"failure_report: {summary['failure_report']}")
    return 0


def _canonical_leaf_parsers(
    parser: argparse.ArgumentParser,
) -> Tuple[argparse.ArgumentParser, ...]:
    leaves = []
    stack = [parser]
    while stack:
        current = stack.pop()
        children = {
            id(child): child
            for action in current._actions
            if isinstance(action, argparse._SubParsersAction)
            for child in action.choices.values()
        }
        if children:
            stack.extend(children.values())
        else:
            leaves.append(current)
    return tuple(leaves)


def _validate_handler_registrations(parser: argparse.ArgumentParser) -> None:
    leaves = _canonical_leaf_parsers(parser)
    unregistered = [
        leaf.prog
        for leaf in leaves
        if not callable(handler := leaf._defaults.get("_handler"))
        or leaf._defaults.get("_handler_registered") is not handler
    ]
    if unregistered:
        raise RuntimeError(
            f"Expected canonical CLI leaves registered via cli_defs.set_handler; "
            f"found {len(leaves)}, unregistered: {', '.join(sorted(unregistered)) or 'none'}."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe", description="Astrolabe all-atom biomolecular design CLI")
    from astrolabe import __version__

    parser.add_argument("--version", action="version", version=f"Astrolabe {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show the full traceback for input/configuration errors.")
    parser.set_defaults(_cuda_preinitialize=True)
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Hydra config directory (default: <repo>/config).",
    )
    parser.add_argument(
        "--autoselect-gpu",
        action="store_true",
        help=(
            "When CUDA_VISIBLE_DEVICES is unset, pick the GPU with the most free memory "
            "(via nvidia-smi). Explicit opt-in; default: disabled."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    from astrolabe.paper import register_commands as register_paper_commands
    register_paper_commands(sub)

    run_p = sub.add_parser(
        "run",
        help="Run generation/optimization.",
        description=(
            "Compose a run config then execute it.\n"
            "Choose one source: -t/--task for a repo template, or --standalone for a standalone YAML.\n"
            "Use -o/--overlay to layer an overlay on top of the chosen base."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run_p.add_argument(
        "-t",
        "--task",
        default=None,
        help="Template name under config/<task>.yaml. Optional when using --standalone.",
    )
    run_p.add_argument(
        "--standalone",
        default=None,
        help="Path to a standalone config YAML layered on top of the repo template; its defaults list is ignored.",
    )
    run_p.add_argument(
        "-o",
        "--overlay",
        default=None,
        help="Overlay config YAML layered after --task/--standalone composition.",
    )
    run_p.add_argument(
        "--chdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --standalone, chdir into its parent directory before resolving relative paths (default: true).",
    )
    run_p.add_argument(
        "--overrides",
        default=None,
        help='Hydra overrides as a single string, e.g. \'a=b c=[1,2]\'.',
    )
    run_p.add_argument(
        "--overrides-file",
        default=None,
        help="Text file with one Hydra override per line (recommended).",
    )
    run_p.add_argument(
        "--print-config",
        action="store_true",
        help="Print the composed config (after layering) before running.",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose/print config and exit without running the pipeline.",
    )
    run_p.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides (pass-through).",
    )
    cli_defs.set_handler(run_p, _cmd_run, parser_ref=True)

    completion_p = sub.add_parser(
        "completion",
        help="Generate Bash/Zsh completion scripts.",
    )
    completion_p.set_defaults(_cuda_preinitialize=False)
    completion_sub = completion_p.add_subparsers(dest="completion_cmd", required=True)

    completion_bash_p = completion_sub.add_parser(
        "bash",
        help="Print the Bash completion script.",
    )
    completion_bash_p.add_argument(
        "--init",
        action="store_true",
        help="Include shell initialization commands so one `source`/`eval` is enough.",
    )
    cli_defs.set_handler(completion_bash_p, _cmd_completion_bash)

    completion_zsh_p = completion_sub.add_parser(
        "zsh",
        help="Print the Zsh completion script.",
    )
    completion_zsh_p.add_argument(
        "--init",
        action="store_true",
        help="Include shell initialization commands so one `source`/`eval` is enough.",
    )
    cli_defs.set_handler(completion_zsh_p, _cmd_completion_zsh)

    completion_exec_p = completion_sub.add_parser(
        "complete",
        help=argparse.SUPPRESS,
    )
    completion_exec_p.add_argument(
        "--shell",
        choices=cli_defs.COMPLETION_SHELL_CHOICES,
        required=True,
        help=argparse.SUPPRESS,
    )
    completion_exec_p.add_argument(
        "--cword",
        type=int,
        required=True,
        help=argparse.SUPPRESS,
    )
    completion_exec_p.add_argument(
        "words",
        nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )
    cli_defs.set_handler(completion_exec_p, _cmd_completion_complete)

    experimental_p = sub.add_parser(
        "experimental",
        help="Experimental tools with unstable interfaces.",
    )
    experimental_p.set_defaults(_cuda_preinitialize=False)
    experimental_sub = experimental_p.add_subparsers(dest="experimental_cmd", required=True)

    candidate_batch_p = experimental_sub.add_parser(
        "candidate-batch",
        help="Run process-isolated candidate-pool rounds.",
        description="Experimental candidate-pool runner for process-isolated GH candidates.",
    )
    from astrolabe.experimental.candidate_batch_runner import add_candidate_batch_arguments

    add_candidate_batch_arguments(candidate_batch_p)
    cli_defs.set_handler(candidate_batch_p, _cmd_experimental_candidate_batch, parser_ref=True)

    bench_binder_p = sub.add_parser(
        "bench-binder-v1-4",
        help="One-shot: fill binder benchmark v1.4 config and run.",
    )
    _add_protein_binder_arguments(bench_binder_p)
    cli_defs.set_handler(bench_binder_p, _cmd_bench_binder_v1_4)

    bench_idr_p = sub.add_parser(
        "bench-idr-binder-v1",
        help="One-shot: fill IDR-target binder benchmark v1 config and run.",
    )
    _add_idr_binder_arguments(bench_idr_p)
    cli_defs.set_handler(bench_idr_p, _cmd_design_idr_binder)

    bench_pep_p = sub.add_parser(
        "bench-peptide-binder-v1",
        help="One-shot: fill peptide binder benchmark v1 config and run.",
    )
    _add_peptide_binder_arguments(bench_pep_p)
    cli_defs.set_handler(bench_pep_p, _cmd_bench_peptide_binder_v1)

    bench_apt_p = sub.add_parser(
        "bench-aptamer-v1",
        help="One-shot: fill RNA aptamer benchmark v1 config and run.",
    )
    _add_aptamer_arguments(bench_apt_p)
    cli_defs.set_handler(bench_apt_p, _cmd_bench_aptamer_v1)

    bench_ab_p = sub.add_parser(
        "bench-antibody-v1",
        help="One-shot: fill antibody/scFv benchmark v1 config and run.",
    )
    _add_antibody_arguments(bench_ab_p)
    cli_defs.set_handler(bench_ab_p, _cmd_bench_antibody_v1)

    design_p = sub.add_parser("design", help="One-shot design presets (fill template + run).")
    design_sub = design_p.add_subparsers(dest="design_cmd", required=True)

    design_binder_p = design_sub.add_parser(
        "protein-binder",
        aliases=["binder"],
        help="Protein binder benchmark preset (v1.4).",
    )
    _add_protein_binder_arguments(design_binder_p)
    cli_defs.set_handler(design_binder_p, _cmd_bench_binder_v1_4)

    design_idr_p = design_sub.add_parser(
        "idr-binder",
        help="IDR-target protein binder benchmark preset (v1).",
    )
    _add_idr_binder_arguments(design_idr_p)
    cli_defs.set_handler(design_idr_p, _cmd_design_idr_binder)

    design_multidomain_p = design_sub.add_parser(
        "multidomain-binder",
        help="Flexible multi-domain target binder preset (single target chain B).",
    )
    _add_common_one_shot_run_args(design_multidomain_p)
    design_multidomain_p.add_argument(
        "--variant",
        required=True,
        choices=sorted(_MULTIDOMAIN_BINDER_VARIANTS),
        help="Preset route for the target-complex stage evaluator: af3|protenix.",
    )
    design_multidomain_p.add_argument(
        "--target-name",
        required=True,
        help="Logical target name stored under multidomain_target.target_name.",
    )
    _add_initial_binder_args(
        design_multidomain_p,
        length_flags=("--binder-len",),
        len_help='Binder length range for chain A, e.g. "80-120" or "100".',
    )
    design_multidomain_p.add_argument("--target-seq", default=None, help="Full target protein sequence on chain B (fixed; '+' auto-added).")
    design_multidomain_p.add_argument(
        "--target-homology",
        default=None,
        help="Full target homology source (AF3 JSON path + chain, or .a3m). The command derives domain-split MSA/templates from this source.",
    )
    design_multidomain_p.add_argument(
        "--domain-split-mode",
        default="full",
        choices=cli_defs.DOMAIN_SPLIT_MODE_CHOICES,
        help=(
            "How to use target homology for domain runs: full splits a full-chain "
            "homology source by --domain-indices; template keeps the resolved "
            "target homology payload unchanged."
        ),
    )
    design_multidomain_p.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    design_multidomain_p.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    design_multidomain_p.add_argument(
        "--domain-indices",
        action="append",
        default=None,
        help="Repeatable residue group for one target domain. Accepts comma-separated residues and ranges, e.g. '1-128' or '1-45,80-128'. Provide at least two groups.",
    )
    design_multidomain_p.add_argument(
        "--domain-hotspots",
        action="append",
        default=None,
        help="Optional repeatable hotspot group for one target domain. Must align 1:1 with --domain-indices. When omitted, each domain index group is also used as its hotspot group.",
    )
    cli_defs.set_handler(design_multidomain_p, _cmd_design_multidomain_binder)

    design_oligomer_p = design_sub.add_parser(
        "oligomer",
        help="De novo homo-oligomer morphology preset (C2-N or D2-D4).",
    )
    _add_common_one_shot_run_args(design_oligomer_p)
    design_oligomer_shape_action = design_oligomer_p.add_argument(
        "--shape",
        required=True,
        metavar="SHAPE",
        help="Target point-group morphology: C<N> for N>=2 or D2-D4.",
    )
    cli_defs.add_completion_choices(
        design_oligomer_shape_action,
        _OLIGOMER_DESIGN_SHAPE_COMPLETIONS,
    )
    _add_initial_binder_args(
        design_oligomer_p,
        length_flags=("--protein-len",),
        len_help='Protomer A length range, e.g. "72-72" or "60-90".',
        seq_help=(
            "Mutable initial protomer A sequence. Writes seqs.A as this bare "
            "sequence; use --initial-binder-expr for fixed segments or "
            "multi-segment statements."
        ),
        expr_help=(
            "Advanced protomer A statement written directly to seqs.A, e.g. "
            "'+ACD 20-30' or 'input:A'."
        ),
    )
    design_oligomer_p.add_argument("--evaluator", default=None, help="Override all stage evaluator aliases (e.g. alphafold2, Protenix).")
    cli_defs.set_handler(design_oligomer_p, _cmd_design_oligomer)

    design_multitarget_p = design_sub.add_parser(
        "multitarget-binder",
        help="Shared-hotspot no-AF3 multi-target binder preset (binder A against targets B/C).",
    )
    design_multitarget_p.add_argument("--task-name", required=True, help="Run name (creates out_dir/<task-name>/###_run).")
    design_multitarget_p.add_argument("--out-dir", required=True, help="Output root directory (task subdir is created under it).")
    design_multitarget_p.add_argument(
        "--variant",
        required=True,
        choices=sorted(_MULTITARGET_BINDER_VARIANT_TO_CONFIG.keys()),
        help="Preset route. dplm-protenix = dplm monomer start; x-init-direct-dual-protenix = x-init Chai1 start then dual Protenix.",
    )
    design_multitarget_p.add_argument("--pair-name", required=True, help="Logical pair name stored under multitarget.pair_name.")
    design_multitarget_p.add_argument("--shared-hotspot-manifest", required=True, help="Path to the shared hotspot manifest YAML recorded under multitarget.shared_hotspot_manifest.")
    _add_initial_binder_args(
        design_multitarget_p,
        length_flags=("--binder-len",),
        len_help='Binder length range for chain A, e.g. "80-120" or "100".',
    )
    design_multitarget_p.add_argument("--target-b-seq", default=None, help="Fixed sequence for target chain B (prefix '+' auto-added).")
    design_multitarget_p.add_argument("--target-b-homology", default=None, help="Optional homology for target chain B (AF3 JSON path + chain, or .a3m).")
    design_multitarget_p.add_argument("--target-b-structure", default=None, help="Optional target-B structure (.pdb/.cif[.gz]) used to infer sequence and template homology.")
    design_multitarget_p.add_argument("--target-b-chain", default=None, help="Target chain ID in --target-b-structure.")
    design_multitarget_p.add_argument("--target-b-hotspots", required=True, help="Hotspot residues on target chain B. Accepts comma-separated residues and ranges, e.g. '5,6,9,88-92'.")
    design_multitarget_p.add_argument("--target-c-seq", default=None, help="Fixed sequence for target chain C (prefix '+' auto-added).")
    design_multitarget_p.add_argument("--target-c-homology", default=None, help="Optional homology for target chain C (AF3 JSON path + chain, or .a3m).")
    design_multitarget_p.add_argument("--target-c-structure", default=None, help="Optional target-C structure (.pdb/.cif[.gz]) used to infer sequence and template homology.")
    design_multitarget_p.add_argument("--target-c-chain", default=None, help="Target chain ID in --target-c-structure.")
    design_multitarget_p.add_argument("--target-c-hotspots", required=True, help="Hotspot residues on target chain C. Accepts comma-separated residues and ranges, e.g. '8,9,12,62,65-69'.")
    design_multitarget_p.add_argument("--repeats", type=int, default=None, help="Override repeats (default: template value).")
    design_multitarget_p.add_argument("--steps", type=int, default=None, help="Override steps (default: template value).")
    design_multitarget_p.add_argument("--dump-config", default=None, help="Optional: write the filled overlay YAML to this path.")
    design_multitarget_p.add_argument("--print-config", action="store_true", help="Print composed config before running.")
    design_multitarget_p.add_argument("--dry-run", action="store_true", help="Print config and exit without running.")
    cli_defs.set_handler(design_multitarget_p, _cmd_design_multitarget_binder)

    design_pep_p = design_sub.add_parser(
        "peptide",
        aliases=["peptide-binder"],
        help="Peptide binder benchmark preset (v1).",
    )
    _add_peptide_binder_arguments(design_pep_p)
    cli_defs.set_handler(design_pep_p, _cmd_bench_peptide_binder_v1)

    design_apt_p = design_sub.add_parser(
        "aptamer",
        help="RNA aptamer benchmark preset (v1).",
    )
    _add_aptamer_arguments(design_apt_p)
    cli_defs.set_handler(design_apt_p, _cmd_bench_aptamer_v1)

    design_dna_apt_p = design_sub.add_parser(
        "dna-aptamer",
        help="DNA aptamer benchmark preset (v1).",
    )
    _add_common_one_shot_run_args(design_dna_apt_p)
    _add_initial_binder_args(
        design_dna_apt_p,
        length_flags=("--aptamer-len",),
        len_help='DNA aptamer length range, e.g. "28-36" or "31".',
        seq_help=(
            "Mutable initial DNA aptamer sequence for chain A. Writes seqs.A as "
            "this bare sequence; use --initial-binder-expr for fixed segments or "
            "multi-segment statements."
        ),
        expr_help=(
            "Advanced DNA aptamer statement written directly to seqs.A, e.g. "
            "'ATGC 28-36' or 'input:A'."
        ),
    )
    design_dna_apt_p.add_argument("--target-seq", default=None, help="Target protein sequence (fixed; '+' auto-added). If omitted, inferred from --target-homology when possible.")
    design_dna_apt_p.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m). If set and --target-seq omitted, infer sequence from it.")
    design_dna_apt_p.add_argument("--target-structure", default=None, help="Optional target structure (.pdb/.cif[.gz]). When set with --target-chain, infer sequence and embed as template homology.")
    design_dna_apt_p.add_argument("--target-chain", default=None, help="Target chain ID in --target-structure.")
    design_dna_apt_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")
    cli_defs.set_handler(design_dna_apt_p, _cmd_bench_aptamer_v1, aptamer_type="dna")

    design_ab_p = design_sub.add_parser(
        "antibody",
        help="Antibody/scFv benchmark preset (v1).",
    )
    _add_antibody_arguments(design_ab_p)
    cli_defs.set_handler(design_ab_p, _cmd_bench_antibody_v1)

    design_ligand_p = design_sub.add_parser(
        "ligand-binder",
        aliases=["ligand"],
        help="Protein + ligand benchmark preset (v1).",
    )
    _add_common_one_shot_run_args(design_ligand_p)
    _add_initial_binder_args(
        design_ligand_p,
        length_flags=("--protein-len",),
        len_help='Protein length range, e.g. "140-180" or "160".',
        seq_help=(
            "Mutable initial protein binder sequence for chain A. Writes seqs.A "
            "as this bare sequence; use --initial-binder-expr for fixed segments "
            "or multi-segment statements."
        ),
    )
    design_ligand_p.add_argument("--ligand", required=True, help="Ligand identifier: CCD code (e.g. FAD) or SMILES.")
    design_ligand_type_action = design_ligand_p.add_argument("--ligand-type", default="ccd", help="Ligand type: ccd|smiles (default: ccd; CCD keeps standard residue naming from the CCD definition).")
    cli_defs.add_completion_choices(design_ligand_type_action, _LIGAND_TYPE_COMPLETION_CHOICES)
    design_ligand_p.add_argument("--ligand-resname", default=None, help="Optional Rosetta params name override for ligand entity B (e.g. LIG or 9TP). This only renames the ligand entry; it does not switch CCD vs SMILES chemistry parsing.")
    design_ligand_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")
    cli_defs.set_handler(design_ligand_p, _cmd_bench_ligand_v1)

    design_cid_p = design_sub.add_parser(
        "cid-generate",
        aliases=["cid"],
        help="Chemical-induced dimer sensor generation preset (proteins A/B plus ligand C).",
    )
    _add_common_one_shot_run_args(design_cid_p)
    _add_initial_protein_chain_args(
        design_cid_p,
        chain_label="A",
        length_flag="--protein-a-len",
        length_dest="protein_a_len",
        seq_flag="--protein-a-seq",
        seq_dest="protein_a_seq",
        expr_flag="--protein-a-expr",
        expr_dest="protein_a_expr",
    )
    _add_initial_protein_chain_args(
        design_cid_p,
        chain_label="B",
        length_flag="--protein-b-len",
        length_dest="protein_b_len",
        seq_flag="--protein-b-seq",
        seq_dest="protein_b_seq",
        expr_flag="--protein-b-expr",
        expr_dest="protein_b_expr",
    )
    design_cid_p.add_argument("--ligand", required=True, help="Ligand identifier for protomer C: CCD code (e.g. ATP) or SMILES.")
    design_cid_ligand_type_action = design_cid_p.add_argument("--ligand-type", default="ccd", help="Ligand type for protomer C: ccd|smiles (default: ccd).")
    cli_defs.add_completion_choices(design_cid_ligand_type_action, _LIGAND_TYPE_COMPLETION_CHOICES)
    design_cid_p.add_argument("--ligand-resname", default=None, help="Optional Rosetta params name override for ligand entity C (e.g. LIG or 9TP).")
    design_cid_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")
    cli_defs.set_handler(design_cid_p, _cmd_design_cid_generate)

    refine_oligo_alias_p = sub.add_parser(
        "refine-oligomer-v3",
        help="Compatibility alias for `astrolabe refine oligomer`.",
    )
    _add_refine_oligomer_common_args(refine_oligo_alias_p)
    cli_defs.set_handler(refine_oligo_alias_p, _cmd_refine_oligomer_v3)

    refine_p = sub.add_parser("refine", help="One-shot refine presets (start from structure/seq + run).")
    refine_sub = refine_p.add_subparsers(dest="refine_cmd", required=True)

    refine_binder_p = refine_sub.add_parser(
        "binder",
        help="Refine an existing complex (supports multi-chain binders).",
    )
    _add_common_one_shot_run_args(refine_binder_p)
    refine_binder_p.add_argument("--structure", required=True, help="Input complex structure (.pdb/.cif[.gz]).")
    refine_binder_p.add_argument("--target-chain", required=True, help="Chain ID of target in the input structure.")
    refine_binder_p.add_argument(
        "--binder-chains",
        required=True,
        help="Comma-separated binder chain IDs in the input structure (order preserved).",
    )
    refine_binder_p.add_argument("--target-homology", default=None, help="Optional target homology (AF3 JSON path + chain, or .a3m).")
    refine_binder_p.add_argument(
        "--target-template-from-structure",
        action="store_true",
        help="If no --target-homology, embed target chain structure as an AF3 template.",
    )
    refine_binder_p.add_argument(
        "--reference",
        default=None,
        help="Optional reference complex for structure_start reference gating.",
    )
    refine_binder_p.add_argument(
        "--reference-mode",
        choices=["repredict_gate", "soft_select"],
        default="repredict_gate",
        help="structure_start mode for --reference (default: repredict_gate).",
    )
    refine_binder_p.add_argument(
        "--reference-tmscore-threshold",
        type=float,
        default=0.85,
        help="Minimum reference TMscore for structure_start gate (default: 0.85).",
    )
    refine_binder_p.add_argument(
        "--reference-hotspot-expression",
        default=None,
        help="Optional hotspot expression for structure_start hotspot_gate, e.g. pair_hotspot_support.",
    )
    refine_binder_p.add_argument(
        "--reference-hotspot-floor",
        type=float,
        default=0.25,
        help="Minimum hotspot gate threshold floor when --reference-hotspot-expression is set.",
    )
    refine_binder_p.add_argument(
        "--reference-hotspot-reference-fraction",
        type=float,
        default=0.8,
        help="Fraction of reference hotspot score required when --reference-hotspot-expression is set.",
    )
    refine_binder_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Protenix).")
    cli_defs.set_handler(refine_binder_p, _cmd_refine_binder_v1_3)

    refine_ligand_p = refine_sub.add_parser(
        "ligand",
        help="Refine/design protein+ligand from an existing structure.",
    )
    _add_refine_ligand_common_args(refine_ligand_p)
    cli_defs.set_handler(refine_ligand_p, _cmd_refine_ligand_v1)

    refine_ligand_pocket_p = refine_sub.add_parser(
        "ligand-pocket",
        help="Pocket-restricted protein redesign around a ligand from an existing structure.",
    )
    _add_refine_ligand_common_args(refine_ligand_pocket_p)
    refine_ligand_pocket_p.add_argument(
        "--mutation-num",
        type=int,
        default=None,
        help="Residues to mutate per step (default: template value 6).",
    )
    refine_ligand_pocket_p.add_argument(
        "--initial-temperature",
        type=float,
        default=None,
        help="Decision-layer Metropolis acceptance temperature override (default: template value 0.02).",
    )
    refine_ligand_pocket_p.add_argument(
        "--contact-distance",
        type=float,
        default=None,
        help="Ligand-contact selector distance in Angstroms for mutable pocket residues (default: template value 5.0).",
    )
    refine_ligand_pocket_p.add_argument(
        "--scoring-mode",
        default="vina",
        choices=cli_defs.REFINE_LIGAND_POCKET_SCORING_MODE_CHOICES,
        help=(
            "Pocket scoring preset. vina keeps the template Autodock penalty; "
            "retarget adds ligand_pocket_occupancy from --reference-pocket."
        ),
    )
    refine_ligand_pocket_p.add_argument(
        "--reference-structure",
        default=None,
        help="Optional reference structure (.pdb/.cif[.gz]) used to align a pocket-defined docking box. Requires --reference-chain and --reference-pocket.",
    )
    refine_ligand_pocket_p.add_argument(
        "--reference-chain",
        default=None,
        help="Protein chain ID in --reference-structure used for aligned reference pocket boxing.",
    )
    refine_ligand_pocket_p.add_argument(
        "--reference-pocket",
        default=None,
        help="Reference pocket residue positions on --reference-chain, e.g. '72,103-105,107'. Requires --reference-structure and --reference-chain.",
    )
    cli_defs.set_handler(refine_ligand_pocket_p, _cmd_refine_ligand_pocket_v1)

    refine_channel_p = refine_sub.add_parser(
        "channel",
        help="Channel-broaden refine preset from a protein-ligand structure.",
    )
    _add_refine_channel_common_args(refine_channel_p)
    cli_defs.set_handler(refine_channel_p, _cmd_refine_channel_v1)

    refine_oligo_p = refine_sub.add_parser(
        "oligomer",
        help="Refine a homo-oligomeric assembly (protomer A repeated).",
    )
    _add_refine_oligomer_common_args(refine_oligo_p)
    cli_defs.set_handler(refine_oligo_p, _cmd_refine_oligomer_v3)

    bench_pseudocycle_p = sub.add_parser(
        "bench-pseudocycle-v1",
        help="One-shot: de novo design of a pseudo-cycle from tied internal repeat units.",
    )
    bench_pseudocycle_p.add_argument("--task-name", required=True, help="Run name (creates out_dir/<task-name>/###_run).")
    bench_pseudocycle_p.add_argument("--out-dir", required=True, help="Output root directory (task subdir is created under it).")
    bench_pseudocycle_p.add_argument("--unit-len", required=True, help='Repeat-unit length range, e.g. "40-40".')
    bench_pseudocycle_p.add_argument("--copies", type=int, required=True, help="Number of internal repeat copies.")
    bench_pseudocycle_p.add_argument("--symmetry-predocker", action="store_true", help="Attach symmetry_canonicalizer to predocker-capable mutators.")
    bench_pseudocycle_p.add_argument(
        "--symmetry-predocker-backend",
        default="legacy",
        choices=cli_defs.SYMMETRY_PREDOCKER_BACKEND_CHOICES,
        help="Backend used by symmetry_canonicalizer (default: legacy).",
    )
    bench_pseudocycle_p.add_argument("--symmetry-order", type=int, default=None, help="Optional explicit symmetry order for symmetry_canonicalizer.")
    _add_symmetry_interface_args(bench_pseudocycle_p)
    bench_pseudocycle_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Alphafold2).")
    bench_pseudocycle_p.add_argument("--repeats", type=int, default=None, help="Override repeats (default: template value).")
    bench_pseudocycle_p.add_argument("--steps", type=int, default=None, help="Override steps (default: template value).")
    bench_pseudocycle_p.add_argument("--dump-config", default=None, help="Optional: write the filled overlay YAML to this path.")
    bench_pseudocycle_p.add_argument("--print-config", action="store_true", help="Print composed config before running.")
    bench_pseudocycle_p.add_argument("--dry-run", action="store_true", help="Print config and exit without running.")
    cli_defs.set_handler(bench_pseudocycle_p, _cmd_bench_pseudocycle_v1)

    bench_tandem_p = sub.add_parser(
        "bench-tandem-repeat-v1",
        help="One-shot: de novo design of a linear/helical tandem repeat from tied internal units.",
    )
    bench_tandem_p.add_argument("--task-name", required=True, help="Run name (creates out_dir/<task-name>/###_run).")
    bench_tandem_p.add_argument("--out-dir", required=True, help="Output root directory (task subdir is created under it).")
    bench_tandem_p.add_argument("--unit-len", required=True, help='Repeat-unit length range, e.g. "40-40".')
    bench_tandem_p.add_argument("--copies", type=int, required=True, help="Number of internal repeat copies.")
    bench_tandem_p.add_argument("--symmetry-predocker", action="store_true", help="Attach symmetry_canonicalizer to predocker-capable mutators.")
    bench_tandem_p.add_argument(
        "--symmetry-predocker-backend",
        default="legacy",
        choices=cli_defs.SYMMETRY_PREDOCKER_BACKEND_CHOICES,
        help="Backend used by symmetry_canonicalizer (default: legacy).",
    )
    bench_tandem_p.add_argument("--symmetry-order", type=int, default=None, help="Optional explicit symmetry order for symmetry_canonicalizer.")
    _add_symmetry_interface_args(bench_tandem_p)
    bench_tandem_p.add_argument(
        "--repeat-canonicalizer-mode",
        default="helical",
        choices=["helical", "translation"],
        help="Canonicalization mode for the repeat predocker.",
    )
    bench_tandem_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Alphafold2).")
    bench_tandem_p.add_argument("--repeats", type=int, default=None, help="Override repeats (default: template value).")
    bench_tandem_p.add_argument("--steps", type=int, default=None, help="Override steps (default: template value).")
    bench_tandem_p.add_argument("--dump-config", default=None, help="Optional: write the filled overlay YAML to this path.")
    bench_tandem_p.add_argument("--print-config", action="store_true", help="Print composed config before running.")
    bench_tandem_p.add_argument("--dry-run", action="store_true", help="Print config and exit without running.")
    cli_defs.set_handler(bench_tandem_p, _cmd_bench_tandem_repeat_v1)

    refine_hetero_p = sub.add_parser(
        "refine-heterooligomer-v1",
        help="One-shot: split monomer into shared core vs interface segments and refine as a heterooligomer.",
    )
    refine_hetero_p.add_argument("--task-name", required=True, help="Run name (creates out_dir/<task-name>/###_run).")
    refine_hetero_p.add_argument("--out-dir", required=True, help="Output root directory (task subdir is created under it).")
    refine_hetero_p.add_argument("--seq", default=None, help="Starting monomer sequence for all protomer copies.")
    refine_hetero_p.add_argument("--structure", default=None, help="Alternative: input structure (.pdb/.cif[.gz]) to extract monomer chain sequence.")
    refine_hetero_p.add_argument("--chain", default=None, help="Chain ID to extract when using --structure.")
    refine_hetero_p.add_argument("--copies", type=int, required=True, help="Number of heterooligomer protomers to generate.")
    refine_hetero_p.add_argument("--interface", required=True, help='Residue ranges (1-indexed) that should stay protomer-specific, e.g. "10-20,35-40".')
    refine_hetero_p.add_argument(
        "--template-from-structure",
        action="store_true",
        help="Embed the extracted monomer chain structure as an AF3 template for every protomer.",
    )
    refine_hetero_p.add_argument("--symmetry-predocker", action="store_true", help="Attach symmetry_canonicalizer to predocker-capable mutators.")
    refine_hetero_p.add_argument(
        "--symmetry-predocker-backend",
        default="legacy",
        choices=cli_defs.SYMMETRY_PREDOCKER_BACKEND_CHOICES,
        help="Backend used by symmetry_canonicalizer (default: legacy).",
    )
    refine_hetero_p.add_argument("--symmetry-order", type=int, default=None, help="Optional explicit symmetry order for symmetry_canonicalizer.")
    refine_hetero_p.add_argument("--evaluator", default=None, help="Override stage evaluator alias (e.g. Alphafold3, Alphafold2).")
    refine_hetero_p.add_argument("--repeats", type=int, default=None, help="Override repeats (default: template value).")
    refine_hetero_p.add_argument("--steps", type=int, default=None, help="Override steps (default: template value).")
    refine_hetero_p.add_argument("--dump-config", default=None, help="Optional: write the filled overlay YAML to this path.")
    refine_hetero_p.add_argument("--print-config", action="store_true", help="Print composed config before running.")
    refine_hetero_p.add_argument("--dry-run", action="store_true", help="Print config and exit without running.")
    cli_defs.set_handler(refine_hetero_p, _cmd_refine_heterooligomer_v1)

    posthoc_p = sub.add_parser("posthoc", help="Post-hoc tools (collect/filter/refold).")
    posthoc_p.set_defaults(_cuda_preinitialize=False)
    posthoc_sub = posthoc_p.add_subparsers(dest="posthoc_cmd", required=True)

    collect_p = posthoc_sub.add_parser(
        "collect",
        help="Incrementally collect run scores into SQLite (optional CSV export).",
        description=(
            "Scan an outputs tree, derive post-hoc metrics, and store them in SQLite.\n"
            "`--binder-chain` and `--target-chain` control derived binder/interface metrics.\n"
            "Use required-metric flags to fail fast when a downstream preset depends on missing metrics."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    collect_p.add_argument("--out-root", "--outputs", dest="out_root", action="append", required=True, help="Output root directory to scan recursively. Repeat to collect multiple outputs roots into one DB.")
    collect_p.add_argument("--db", "--sqlite", dest="db", required=True, help="SQLite DB path to write (created if missing).")
    collect_p.add_argument("--export-csv", "--csv", dest="export_csv", default=None, help="Optional: export a wide CSV from the DB.")
    collect_p.add_argument("--archive-dir", default=None, help="Optional archive destination. When set, package each processed run's raw_model_results into tar.gz files under this directory after DB upsert.")
    collect_p.add_argument("--archive-workers", type=int, default=0, help="Archive worker threads (default: min(4, processed runs)).")
    collect_p.add_argument("--archive-overwrite", action="store_true", help="Overwrite existing archive tarballs.")
    collect_p.add_argument("--workers", type=int, default=0, help="Worker threads (default: CPU/2).")
    collect_p.add_argument("--task-regex", default=None, help="Only scan tasks matching this regex.")
    collect_p.add_argument("--run-regex", default=None, help="Only scan runs matching this regex.")
    collect_p.add_argument("--binder-chain", default="B", help="Binder chain ID used when deriving binder_* metrics from raw outputs (default: B).")
    collect_p.add_argument("--target-chain", default="A", help="Target chain ID used when deriving interface metrics such as iPAE from raw outputs (default: A).")
    collect_p.add_argument(
        "--required-metric",
        action="append",
        default=[],
        help="Metric name that must be present/valid for every collected record. Repeatable.",
    )
    collect_p.add_argument(
        "--required-metrics-preset",
        "--preset",
        dest="preset",
        default=None,
        help="Preset YAML path. All non-optional preset metrics become required during collection.",
    )
    collect_p.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Collect anyway when required metrics are missing/invalid; useful for exploratory scans.",
    )
    cli_defs.add_output_format_argument(collect_p)
    cli_defs.set_handler(collect_p, _cmd_collect_scores)

    channel_screen_p = posthoc_sub.add_parser(
        "channel-screen",
        help="Directly mutate mutable residues to fixed amino acids and rescore HOLE.",
    )
    channel_screen_p.add_argument(
        "--structure",
        required=True,
        help="Input structure (.pdb/.cif[.gz]) containing the protein and ligand.",
    )
    channel_screen_p.add_argument(
        "--protein-chain",
        required=True,
        help="Protein chain ID in --structure.",
    )
    channel_screen_p.add_argument(
        "--ligand-residue",
        required=True,
        help="Ligand residue selector from --structure: '<chain>:<resname>:<resseq>[:<icode>]'.",
    )
    channel_screen_p.add_argument(
        "--mutable-positions",
        required=True,
        help="Path to mutable positions file for the protein chain.",
    )
    channel_screen_p.add_argument(
        "--spheres",
        required=True,
        help="Reference spheres PDB used for HOLE rescanning.",
    )
    channel_screen_p.add_argument(
        "--out-dir",
        default=None,
        help="Output root. Default writes '<structure_stem>__channel_screen' next to --structure.",
    )
    channel_screen_p.add_argument(
        "--reference-structure",
        default=None,
        help="Optional reference structure used for HOLE alignment/scoring. Default uses --structure.",
    )
    channel_screen_p.add_argument(
        "--alignment-chains",
        default=None,
        help="Chains used to align structures before HOLE rescoring. Default uses --protein-chain.",
    )
    channel_screen_p.add_argument(
        "--target-aas",
        default="A,S,T,G",
        help="Target amino acids for non-pocket mutable residues. Accepts 1-letter or 3-letter tokens separated by comma, space, or '|'.",
    )
    channel_screen_p.add_argument(
        "--contact-distance",
        type=float,
        default=5.0,
        help="Mutable residues with ligand heavy-atom distance <= this cutoff stay fixed as pocket residues.",
    )
    channel_screen_p.add_argument(
        "--radius-gate-path-len-threshold",
        type=float,
        default=40.0,
        help="Path-length threshold used by the channel_hole radius gate.",
    )
    channel_screen_p.add_argument(
        "--hole-executable",
        default=None,
        help="Optional HOLE executable path override.",
    )
    channel_screen_p.add_argument(
        "--hole-radius-file",
        default=None,
        help="Optional HOLE radius parameter file. Default also honors GH_HOLE_RADIUS_FILE.",
    )
    channel_screen_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    cli_defs.set_handler(channel_screen_p, _cmd_posthoc_channel_screen)

    filt_p = posthoc_sub.add_parser(
        "filter",
        help="Filter structures using a DSL expression or a preset and hardlink renamed outputs.",
        description=(
            "`--preset` expects a YAML file path, not a built-in preset name.\n"
            "Use `--preview-preset --preset <file>` to inspect the resolved rule set.\n"
            "If --dst is omitted, a destination directory is auto-generated next to the SQLite DB."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    filt_p.add_argument("--db", "--sqlite", dest="db", required=False, help="SQLite DB path produced by posthoc collect. Required unless --preview-preset is used.")
    selector_group = filt_p.add_mutually_exclusive_group(required=False)
    selector_group.add_argument("--preset", required=False, help="Filter preset YAML path (design-metrics compatible).")
    selector_group.add_argument(
        "--filter-expr",
        "--expr",
        dest="filter_expr",
        required=False,
        help="DSL filter expression over collected metric names, e.g. 'weighted_score > 0 and AB.iptm >= 0.7'.",
    )
    filt_p.add_argument("--preview-preset", action="store_true", help="Preview the resolved preset rules/expression and exit. Requires --preset; does not need --db or --dst.")
    filt_p.add_argument("--dst", "--dest", dest="dst", required=False, help="Destination directory for filtered structures (default: auto-generated next to the DB).")
    filt_p.add_argument("--prefer-oligo", "--oligo", dest="prefer_oligo", default="AB", help="Which oligo to filter/link (default: AB).")
    cli_defs.add_run_selection_argument(
        filt_p,
        help="How to choose records per run before filtering: all passing rows, or only the representative row from the last stage (highest weighted_score wins, ties prefer later step).",
    )
    filt_p.add_argument("--flat", action="store_true", help="Write all outputs directly under --dst (no task subdirs).")
    filt_p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    filt_p.add_argument("--copy", action="store_true", help="Copy instead of hardlink (explicit; uses extra disk).")
    filt_p.add_argument("-n", "--dry-run", action="store_true", help="Only compute selection counts; do not write filtered structures or report CSV.")
    filt_p.add_argument("--report-csv", default=None, help="Optional report CSV path (default: <dst>/filter_report.csv).")
    filt_p.add_argument("--out-db", default=None, help="Optional filtered SQLite snapshot path (default: <dst>/filtered.sqlite; not written during --dry-run).")
    filt_p.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Allow selectors with required metrics missing/invalid in the DB (default: fail and suggest refold when possible).",
    )
    cli_defs.add_output_format_argument(filt_p)
    cli_defs.set_handler(filt_p, _cmd_filter_structures)

    prune_p = posthoc_sub.add_parser(
        "prune",
        help="Delete collected output directories using the same selector semantics as posthoc filter.",
        description=(
            "Select rows from the SQLite DB using a preset or DSL expression, collapse to unique target paths, "
            "and delete either raw_model_results or whole run directories.\n"
            "Default is dry-run; pass --yes to actually delete."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    prune_p.add_argument("--db", "--sqlite", dest="db", required=True, help="SQLite DB path produced by posthoc collect.")
    prune_selector = prune_p.add_mutually_exclusive_group(required=False)
    prune_selector.add_argument("--preset", required=False, help="Filter preset YAML path.")
    prune_selector.add_argument(
        "--filter-expr",
        "--expr",
        dest="filter_expr",
        required=False,
        help="DSL filter expression over collected metric names.",
    )
    prune_p.add_argument("--prefer-oligo", "--oligo", dest="prefer_oligo", default="AB", help="Which oligo to select from the DB (default: AB).")
    cli_defs.add_run_selection_argument(
        prune_p,
        help="How to choose records per run before pruning.",
    )
    prune_p.add_argument(
        "--delete-scope",
        choices=cli_defs.DELETE_SCOPE_CHOICES,
        default="raw_model_results",
        help="Delete only raw_model_results, or the whole run_dir.",
    )
    prune_p.add_argument("--report-csv", default=None, help="Optional prune report CSV path.")
    prune_p.add_argument("--yes", action="store_true", help="Actually delete selected paths. Default is dry-run.")
    cli_defs.add_output_format_argument(prune_p)
    cli_defs.set_handler(prune_p, _cmd_prune_outputs)

    report_p = posthoc_sub.add_parser(
        "report",
        help="Generate static post-hoc trends and success-rate reports from a SQLite DB.",
        description=(
            "Read a posthoc SQLite DB, evaluate a selector, and emit CSV + PNG + Markdown summaries.\n"
            "Exactly one of --preset or --filter-expr/--expr is required."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    report_p.add_argument("--db", "--sqlite", dest="db", required=True, help="SQLite DB path produced by posthoc collect.")
    report_selector = report_p.add_mutually_exclusive_group(required=False)
    report_selector.add_argument("--preset", required=False, help="Filter preset YAML path.")
    report_selector.add_argument(
        "--filter-expr",
        "--expr",
        dest="filter_expr",
        required=False,
        help="DSL filter expression used to define success.",
    )
    report_p.add_argument("--dst", "--dest", dest="dst", required=False, help="Destination directory for report files (default: auto-generated next to the DB).")
    report_p.add_argument("--prefer-oligo", "--oligo", dest="prefer_oligo", default="AB", help="Which oligo to analyze (default: AB).")
    report_p.add_argument(
        "--trend-metric",
        dest="trend_metrics",
        action="append",
        default=[],
        help="Metric to include in the step-trend plot/CSV. Repeatable.",
    )
    cli_defs.add_output_format_argument(report_p)
    cli_defs.set_handler(report_p, _cmd_posthoc_report)

    refold_p = posthoc_sub.add_parser(
        "refold",
        help="Selector-driven second-pass prediction that writes results back into the same DB.",
        description=(
            "Use the same selector semantics as `posthoc filter`, then rerun predictors, collect the "
            "refold outputs back into the same SQLite DB, and continue filtering on that DB.\n"
            "Exactly one of `--preset` or `--filter-expr/--expr` is required.\n"
            "If --dst is omitted, a refold work directory is auto-generated next to the SQLite DB."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    refold_p.set_defaults(_cuda_preinitialize=True)
    refold_p.add_argument("--db", required=True, help="SQLite DB path produced by posthoc collect.")
    refold_selector = refold_p.add_mutually_exclusive_group(required=False)
    refold_selector.add_argument("--preset", required=False, help="Filter preset YAML path used to choose rows for refold.")
    refold_selector.add_argument(
        "--filter-expr",
        "--expr",
        dest="filter_expr",
        required=False,
        help="DSL filter expression used to choose rows for refold.",
    )
    refold_p.add_argument("--dst", "--dest", dest="dst", required=False, help="Working directory for manifest/report/refold outputs (default: auto-generated next to the DB).")
    refold_p.add_argument("--prefer-oligo", default="AB", help="Which oligo to select from the DB (default: AB).")
    cli_defs.add_run_selection_argument(
        refold_p,
        help="How to choose records per logical run before refold: all passing rows, or only the representative row from the last stage.",
    )
    refold_p.add_argument(
        "--evaluators",
        default="af3,protenix",
        help="Comma-separated prediction backends to rerun: af3,protenix (default: af3,protenix).",
    )
    refold_p.add_argument("--collect-workers", type=int, default=0, help="Worker threads for the automatic collect-back step (default: CPU/2).")
    refold_p.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional path mapping applied at read time: FROM=TO (repeatable).",
    )
    refold_p.add_argument(
        "--relpath-base",
        default=None,
        help="If config homology paths are relative, join them under this base before export/run.",
    )
    refold_p.add_argument(
        "--keep-protenix-structures",
        action="store_true",
        help="Keep Protenix temporary structures (default: delete to save space).",
    )
    refold_p.add_argument("--limit", type=int, default=None, help="Optional: only refold the first N selected records.")
    refold_p.add_argument("--config-yaml", default=None, help="Optional external YAML/config.yaml to reuse for evaluator settings instead of each source run config.")
    refold_p.add_argument("--workers", type=int, default=1, help="Number of refold worker processes over run groups (default: 1).")
    refold_p.add_argument("--dry-run", action="store_true", help="Prepare manifest and validate/count work without running models or collect-back.")
    cli_defs.add_output_format_argument(refold_p)
    cli_defs.set_handler(refold_p, _cmd_refold_from_db)

    refexp_p = posthoc_sub.add_parser(
        "refold-export",
        aliases=["refold-prepare"],
        help="Low-level refold manifest export. Prefer `astrolabe posthoc refold` for the normal workflow.",
    )
    refexp_p.add_argument("--db", required=True, help="SQLite DB path produced by posthoc collect.")
    selector_group = refexp_p.add_mutually_exclusive_group(required=True)
    selector_group.add_argument("--preset", required=False, help="Filter preset YAML path used to choose rows for refold.")
    selector_group.add_argument(
        "--filter-expr",
        "--expr",
        dest="filter_expr",
        required=False,
        help="DSL filter expression used to choose rows for refold.",
    )
    refexp_p.add_argument("--out-manifest", required=True, help="Output manifest JSONL path consumed by `astrolabe posthoc refold-run`.")
    refexp_p.add_argument("--prefer-oligo", default="AB", help="Which oligo to select from the DB (default: AB).")
    cli_defs.add_run_selection_argument(
        refexp_p,
        help="How to choose records per logical run before export.",
    )
    refexp_p.add_argument(
        "--passed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only export records that pass the preset (default: true).",
    )
    refexp_p.add_argument("--report-csv", default=None, help="Optional export report CSV (lists errors).")
    refexp_p.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional path mapping applied at read time: FROM=TO (repeatable).",
    )
    refexp_p.add_argument(
        "--relpath-base",
        default=None,
        help="If a config.yaml homology path is relative, join it under this base (explicit).",
    )
    refexp_p.add_argument("--limit", type=int, default=None, help="Optional: only export first N records.")
    cli_defs.add_output_format_argument(refexp_p)
    cli_defs.set_handler(refexp_p, _cmd_refold_export)

    refrun_p = posthoc_sub.add_parser(
        "refold-run",
        aliases=["refold-predict"],
        help="Low-level refold run from a prepared manifest. Prefer `astrolabe posthoc refold` for the normal workflow.",
    )
    refrun_p.set_defaults(_cuda_preinitialize=True)
    refrun_p.add_argument("--manifest", required=True, help="Manifest JSONL produced by `astrolabe posthoc refold-export`.")
    refrun_p.add_argument("--out-root", required=True, help="Destination output root for refold run folders.")
    refrun_p.add_argument(
        "--evaluators",
        default="af3,protenix",
        help="Comma-separated prediction backends to rerun: af3,protenix (default: af3,protenix).",
    )
    refrun_p.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional path mapping applied at read time: FROM=TO (repeatable).",
    )
    refrun_p.add_argument(
        "--relpath-base",
        default=None,
        help="If manifest homology paths are relative, join them under this base before loading.",
    )
    refrun_p.add_argument(
        "--keep-protenix-structures",
        action="store_true",
        help="Keep Protenix temporary structures (default: delete to save space).",
    )
    refrun_p.add_argument("--config-yaml", default=None, help="Optional external YAML/config.yaml to reuse for evaluator settings instead of each manifest item src_config.")
    refrun_p.add_argument("--workers", type=int, default=1, help="Number of refold worker processes over run groups (default: 1).")
    refrun_p.add_argument("--dry-run", action="store_true", help="Validate and count work without running models.")
    cli_defs.add_output_format_argument(refrun_p)
    cli_defs.set_handler(refrun_p, _cmd_refold_run)

    cfg_p = sub.add_parser("config", help="Config helpers (interactive).")
    cfg_p.set_defaults(_cuda_preinitialize=False)
    cfg_sub = cfg_p.add_subparsers(dest="config_cmd", required=True)

    wiz_p = cfg_sub.add_parser(
        "wizard",
        help="Interactive helper to fill missing values in a standalone config.",
        description=(
            "Fill `???` values in an existing standalone YAML.\n"
            "If --standalone is omitted, the wizard asks for the file path.\n"
            "--task is optional and only controls the final `astrolabe run` command it prints."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    wiz_p.add_argument("--task", default=None, help="Optional template name under config/<task>.yaml for the final printed `astrolabe run` command.")
    wiz_p.add_argument("--standalone", default=None, help="Path to a standalone config YAML to inspect and fill.")
    wiz_p.add_argument("--overrides", default=None, help="Extra Hydra overrides as a single string.")
    wiz_p.add_argument(
        "--print-config",
        action="store_true",
        help="Print the composed config before running (if --execute).",
    )
    wiz_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose/print config and exit without running (if --execute).",
    )
    wiz_p.add_argument(
        "--execute",
        action="store_true",
        help="Run immediately after prompting (default: print command only).",
    )
    cli_defs.set_handler(wiz_p, _cmd_wizard)

    scaffold_p = cfg_sub.add_parser(
        "scaffold",
        help="Interactive helper to create an overlay YAML for a task template.",
        description=(
            "Start from an existing repo template, scan for `???`, and prompt for values.\n"
            "This writes an overlay YAML by default; use it together with `astrolabe run -t <task> -o <overlay>`."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    scaffold_p.add_argument(
        "--task",
        required=True,
        help="Template name under config/<task>.yaml (scans for '???' and prompts).",
    )
    scaffold_p.add_argument(
        "--out",
        default=None,
        help="Output overlay YAML path (default: ./d.<task>.overlay.yaml).",
    )
    cli_defs.set_handler(scaffold_p, _cmd_scaffold)

    compose_p = cfg_sub.add_parser(
        "compose",
        help="Interactive helper to build a draft overlay/config from scratch.",
        description=(
            "Build a draft config interactively from `base` or a repo template.\n"
            "`:write` writes only <name>.yaml.\n"
            "`:write-all` writes both <name>.yaml and d.<name>.overlay.yaml.\n"
            "`:write-overlay` writes only d.<name>.overlay.yaml."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    compose_p.add_argument(
        "--base",
        choices=cli_defs.CONFIG_BASE_CHOICES,
        default=None,
        help="Base kind: `base` for config/base.yaml, or `template` for config/<task>.yaml (default: prompt; enter => base).",
    )
    compose_p.add_argument(
        "--task",
        default=None,
        help="When --base template: template name under config/<task>.yaml. Omit to choose interactively.",
    )
    compose_p.add_argument(
        "--name",
        default=None,
        help="Output name (default: prompt; used to form d.<name>.overlay.yaml and <name>.yaml).",
    )
    compose_p.add_argument("--overlay-out", default=None, help="Overlay output path.")
    compose_p.add_argument("--standalone-out", default=None, help="Standalone output path.")
    cli_defs.set_handler(compose_p, _cmd_compose)

    env_p = sub.add_parser("env", help="Environment helpers (doctor/wizard).")
    env_p.set_defaults(_cuda_preinitialize=False)
    env_sub = env_p.add_subparsers(dest="env_cmd", required=True)

    env_wiz_p = env_sub.add_parser(
        "wizard",
        help="Interactive helper to fill env.yaml paths for a task.",
        description=(
            "Fill env.yaml entries required by config/<task>.yaml.\n"
            "--task should be a template name under config/.\n"
            "If env.yaml uses `${oc.env:VAR,???}` placeholders, you can either replace them here or keep them and set the corresponding VAR in your shell/.env."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    env_wiz_p.add_argument(
        "--task",
        required=True,
        help="Template name under config/<task>.yaml (used to infer required env keys).",
    )
    env_wiz_p.add_argument(
        "--env",
        default=None,
        help="Input env.yaml path (default: config/env/env_local.yaml).",
    )
    env_wiz_p.add_argument(
        "--out",
        default=None,
        help="Output env.yaml path (default: overwrite --env).",
    )
    env_wiz_p.add_argument(
        "--force",
        action="store_true",
        help="Prompt for all required keys even if current value is valid.",
    )
    env_wiz_p.add_argument(
        "--show-ok",
        action="store_true",
        help="Show OK entries in the final doctor report.",
    )
    env_wiz_p.add_argument(
        "--soft-fail",
        action="store_true",
        help="Exit 0 even if doctor finds missing deps (still writes the env file).",
    )
    cli_defs.set_handler(env_wiz_p, _cmd_env_wizard)

    doc_p = env_sub.add_parser(
        "doctor",
        help="Check external tool/model paths from env.yaml.",
        description=(
            "Validate env.yaml entries either globally or for a specific task template.\n"
            "When --task is set, doctor explains which env keys are required by that task's evaluators/mutators/predockers/losses."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    doc_p.add_argument("--env", default=None, help="Path to env.yaml (default: config/env/env_local.yaml).")
    doc_p.add_argument(
        "--task",
        default=None,
        help="Optional template name under config/<task>.yaml; when set, limit checks to deps implied by that task.",
    )
    doc_p.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve OmegaConf interpolations using current environment variables.",
    )
    cli_defs.add_output_format_argument(doc_p)
    doc_p.add_argument(
        "--show-ok",
        action="store_true",
        help="Show OK entries in text output (default: only unresolved/missing).",
    )
    doc_p.add_argument(
        "--soft-fail",
        action="store_true",
        help="Exit 0 even if required paths are missing.",
    )
    doc_p.add_argument(
        "--python-deps",
        action="store_true",
        help="Also check importability of key Python deps (task-aware when --task is set).",
    )
    doc_p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required path is missing OR any interpolation remains unresolved; implies --resolve and --python-deps.",
    )
    cli_defs.set_handler(doc_p, _cmd_doctor)
    _validate_handler_registrations(parser)
    return parser


LEGACY_COMMAND_REWRITES = (
    (("posthoc", "refold", "export"), ("posthoc", "refold-export")),
    (("posthoc", "refold", "prepare"), ("posthoc", "refold-export")),
    (("posthoc", "refold", "run"), ("posthoc", "refold-run")),
    (("posthoc", "refold", "predict"), ("posthoc", "refold-run")),
)


def _preinitialize_cuda_driver(ns: argparse.Namespace) -> None:
    if (
        os.environ.get("GH_CUDA_PREINITIALIZE") != "1"
        or not getattr(ns, "_cuda_preinitialize", False)
        or getattr(ns, "dry_run", False)
    ):
        return
    import ctypes

    driver = ctypes.CDLL("libcuda.so.1")
    result = int(driver.cuInit(0))
    if result != 0:
        raise RuntimeError(f"CUDA driver pre-initialization failed with cuInit status {result}.")


def _rewrite_legacy_posthoc_refold_argv(
    argv: List[str],
    parser: argparse.ArgumentParser,
) -> List[str]:
    raw = list(argv)
    route_index = 0
    while route_index < len(raw) and raw[route_index].startswith("-"):
        option, separator, _value = raw[route_index].partition("=")
        action = parser._option_string_actions.get(option)
        if (
            action is None
            or action.nargs not in (None, 0)
            or (action.nargs == 0 and separator)
        ):
            return raw
        route_index += 1
        if action.nargs is None and not separator:
            if route_index >= len(raw):
                return raw
            route_index += 1
    for legacy, canonical in LEGACY_COMMAND_REWRITES:
        route_end = route_index + len(legacy)
        if tuple(raw[route_index:route_end]) == legacy:
            return [*raw[:route_index], *canonical, *raw[route_end:]]
    return raw


def main(argv=None):  # type: (Optional[List[str]]) -> None
    parser = _build_parser()
    parse_argv = list(sys.argv[1:]) if argv is None else argv
    ns = parser.parse_args(_rewrite_legacy_posthoc_refold_argv(parse_argv, parser))
    handler = getattr(ns, "_handler", None)
    if handler is None:
        parser.error("No subcommand handler selected.")
    try:
        _preinitialize_cuda_driver(ns)
        code = int(handler(ns))
    except BrokenPipeError:
        # Allow piping to `head` etc.
        code = 0
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError,
            HydraException, OmegaConfBaseException, YAMLError) as exc:
        if ns.debug:
            raise
        print(f"astrolabe: {exc}", file=sys.stderr)
        print("Check the supplied paths/configuration and the command's --help. "
              "Use astrolabe --debug <command> for a traceback.", file=sys.stderr)
        code = 2
    except ModuleNotFoundError as exc:
        if ns.debug:
            raise
        print(f"astrolabe: missing dependency {exc.name!r}. Install the selected backend "
              "using scripts/install.py (README: model installation), then "
              "run astrolabe env doctor --strict --task <task>. "
              "Use astrolabe --debug <command> for a traceback.", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
