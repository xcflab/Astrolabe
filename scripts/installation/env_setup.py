#!/usr/bin/env python3
"""
Host-side third-party environment setup (micromamba/conda) with resume + interactive confirmations.

Design goals:
- Prepare multiple isolated environments for tools invoked via subprocess.
- Resume safely using per-env state files (idempotent re-runs).
- Allow interactive decisions when an env already exists.

This script intentionally does NOT attempt to download model weights; it focuses on
creating Python/tool environments and optionally cloning repos / running upstream
install commands (as configured).
"""

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


STATE_DIRNAME = ".gh_env_setup"
PRINT_LOCK = threading.Lock()
BOOTSTRAP_ENV_NAME = "_gh_bootstrap"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PROFILES = {
    "runtime": (
        "gh",
        "dock",
        "boltz_cache",
        "alphafold3_assets",
        "alphafold2_assets",
        "protenix_assets",
        "usalign",
    ),
    "needed_softwares": (
        "gh",
        "dock",
        "dplm",
        "rpxdock",
        "boltz_cache",
        "alphafold3_assets",
        "alphafold2_assets",
        "protenix_assets",
        "usalign",
    ),
    "all": None,
}

_HOST_COMPILE_ENV_KEYS = [
    # Commonly exported in HPC shells; can break conda-forge toolchains / CMake tests.
    # Example symptom: linker picks up wrong-arch crt objects (ELFCLASS32) via LIBRARY_PATH.
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
]

_HOST_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
]

_KNOWN_BAD_PROXY_SUBSTRINGS = [
    "127.0.0.1:18901",
    "localhost:18901",
]


def eprint(msg):
    with PRINT_LOCK:
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()


def sprint(msg):
    with PRINT_LOCK:
        sys.stdout.write(str(msg) + "\n")
        sys.stdout.flush()


def _sanitize_host_compile_env(env: dict, *, preserve: bool) -> dict:
    if preserve:
        return {}
    removed = {}
    for k in _HOST_COMPILE_ENV_KEYS:
        v = env.pop(k, None)
        if v:
            removed[k] = v
    if removed:
        sprint(
            "sanitize: removed %s from host env (avoids broken builds inside micromamba envs)"
            % ",".join(sorted(removed.keys()))
        )
    return removed


def _sanitize_broken_proxy_env(env: dict) -> dict:
    removed = {}
    for k in _HOST_PROXY_ENV_KEYS:
        v = env.get(k)
        if not v:
            continue
        v_str = str(v)
        if any(bad in v_str for bad in _KNOWN_BAD_PROXY_SUBSTRINGS):
            removed[k] = v_str
            env.pop(k, None)
    if removed:
        sprint(
            "sanitize: removed broken proxy env (%s) to avoid git/pip failures"
            % ",".join(sorted(removed.keys()))
        )
    return removed


def _require_omegaconf():
    try:
        from omegaconf import OmegaConf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "This script requires `omegaconf`. Run it from the project environment "
            "(e.g. `pip install -e .` or `pip install omegaconf`)."
        ) from exc


def _expand_path(path_str):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(path_str))).resolve()


def _json_load(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(str(tmp), str(path))


def _write_gh_python_hint(repo_root, python_bin):
    if not python_bin:
        return None
    value = str(python_bin).strip()
    if not value:
        return None
    hint_path = pathlib.Path(repo_root) / "config" / "env" / ".astrolabe_python_path"
    hint_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(hint_path, value + "\n")
    return hint_path


def _write_text_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    os.replace(str(tmp), str(path))


def _run(cmd, env=None, cwd=None, check=True):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    sprint("+ " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, env=env, cwd=cwd, check=check)


def _prompt_choice(question, choices, default_key=None, non_interactive=False):
    """
    choices: list of (key, label)
    returns selected key
    """
    keys = [k for k, _ in choices]
    if default_key is not None and default_key not in keys:
        raise ValueError("default_key must be one of choices")
    if non_interactive:
        if default_key is None:
            raise RuntimeError("Non-interactive mode requires a default choice.")
        return default_key

    labels = " ".join(["[%s] %s" % (k, lbl) for k, lbl in choices])
    prompt = "%s %s" % (question, labels)
    if default_key:
        prompt += " (default: %s)" % default_key
    prompt += ": "
    while True:
        try:
            resp = input(prompt).strip().lower()
        except EOFError:
            resp = ""
        if not resp and default_key is not None:
            return default_key
        if resp in keys:
            return resp
        sprint("Invalid choice. Choose one of: %s" % ", ".join(keys))


def _detect_executable(name):
    from shutil import which

    return which(name)


def _bootstrap_micromamba(bin_dir):
    """
    Download micromamba into bin_dir if not present.
    """
    import tarfile
    import urllib.request

    bin_dir.mkdir(parents=True, exist_ok=True)
    mm = bin_dir / "micromamba"
    if mm.exists():
        return str(mm)

    url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
    sprint("Downloading micromamba: %s" % url)
    tar_path = bin_dir / "micromamba.tar.bz2"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with tar_path.open("wb") as f:
        f.write(data)

    with tarfile.open(str(tar_path), "r:bz2") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith("/micromamba"):
                member = m
                break
        if member is None:
            raise RuntimeError("micromamba tarball did not contain micromamba binary")
        member.name = pathlib.Path(member.name).name  # flatten
        tf.extract(member, str(bin_dir))

    os.chmod(str(mm), 0o755)
    tar_path.unlink()
    return str(mm)


def _env_prefix_micromamba(root_prefix, env_name):
    return root_prefix / "envs" / env_name


def _env_exists(prefix):
    return prefix.exists() and any(prefix.iterdir())


def _external_python_is_usable(python_bin):
    return python_bin.is_file() and os.access(str(python_bin), os.X_OK)


def _resolve_external_env_prefix(env_name, env_spec, substitutions):
    ext_prefix = env_spec.get("prefix") or env_spec.get("env_prefix")
    py_bin = env_spec.get("python_bin")

    if ext_prefix:
        prefix = _expand_path(_apply_env_substitutions(str(ext_prefix), substitutions))
        python_bin = prefix / "bin" / "python"
        if not _env_exists(prefix) or not python_bin.exists():
            raise RuntimeError(
                "external env '%s' prefix is missing or not a Python env: %s "
                "(expected %s). Point `prefix` at an existing env, use `python_bin`, "
                "or switch to manager=micromamba to create it."
                % (env_name, prefix, python_bin)
            )
        if not _external_python_is_usable(python_bin):
            raise RuntimeError(
                "external env '%s' prefix has a non-executable Python: %s"
                % (env_name, python_bin)
            )
        return prefix

    if py_bin:
        python_bin = _expand_path(_apply_env_substitutions(str(py_bin), substitutions))
        if not python_bin.exists():
            raise RuntimeError(
                "external env '%s' python_bin does not exist: %s. Point `python_bin` "
                "at an existing interpreter or switch to manager=micromamba to create it."
                % (env_name, python_bin)
            )
        if not _external_python_is_usable(python_bin):
            raise RuntimeError(
                "external env '%s' python_bin is not executable: %s"
                % (env_name, python_bin)
            )
        prefix = python_bin.parent.parent
        if not _env_exists(prefix):
            raise RuntimeError(
                "external env '%s' python_bin exists but its env prefix is invalid: %s"
                % (env_name, prefix)
            )
        return prefix

    raise ValueError(
        "external env '%s' requires 'prefix' (env prefix) or 'python_bin'."
        % env_name
    )


def _state_paths(state_root, env_name):
    env_state_dir = state_root / env_name
    return env_state_dir / "state.json", env_state_dir


def _load_state(state_root, env_name):
    path, _ = _state_paths(state_root, env_name)
    state = _json_load(path) or {"completed_steps": [], "updated_at": None}
    if "completed_steps" not in state or not isinstance(state.get("completed_steps"), list):
        state["completed_steps"] = []
    if "fingerprints" not in state or not isinstance(state.get("fingerprints"), dict):
        state["fingerprints"] = {}
    return state


def _save_state(state_root, env_name, state):
    path, _ = _state_paths(state_root, env_name)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _json_dump_atomic(path, state)


def _mark_step_done(state_root, env_name, state, step_name):
    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)
        _save_state(state_root, env_name, state)


def _render_env_summary(env_name, prefix, state):
    steps = state.get("completed_steps", [])
    return "%s  prefix=%s  steps_done=%d" % (env_name, prefix, len(steps))


def _parse_env_name_csv(value):
    if not value:
        return set()
    return set([x.strip() for x in str(value).split(",") if x.strip()])


def _select_env_specs(envs, *, only=None, exclude=None):
    only_names = _parse_env_name_csv(only)
    exclude_names = _parse_env_name_csv(exclude)

    selected = []
    for env_spec in envs:
        if not isinstance(env_spec, dict) or "name" not in env_spec:
            raise ValueError("Each env spec must be a mapping with a 'name'")
        if only_names and env_spec["name"] not in only_names:
            continue
        if env_spec["name"] in exclude_names:
            continue
        selected.append(env_spec)
    return selected


def _hash12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _fingerprint_lines(lines) -> str:
    if not lines:
        return ""
    joined = "\n".join([str(x) for x in lines])
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()


def _is_editable_pip_spec(spec: str) -> bool:
    s = (spec or "").strip()
    return s.startswith("-e ") or s.startswith("--editable ")


def _pip_install_target(spec: str):
    parts = shlex.split(str(spec or ""))
    if not parts:
        return None
    for idx, part in enumerate(parts):
        if part in {"-e", "--editable"}:
            if idx + 1 >= len(parts):
                raise ValueError("Editable pip spec is missing a path: %s" % spec)
            return parts[idx + 1]
    for part in parts:
        if part.startswith("-"):
            continue
        return part
    return None


def _looks_like_local_pip_target(target: str) -> bool:
    if not target:
        return False
    if "://" in target or target.startswith("git+"):
        return False
    return (
        target.startswith("/")
        or target.startswith("./")
        or target.startswith("../")
        or target.startswith("~")
    )


def _local_pip_target_path(target: str) -> pathlib.Path:
    path = _expand_path(target)
    if path.exists():
        return path
    if target.endswith("]") and "[" in target:
        base_target = target[: target.rfind("[")]
        if base_target:
            return _expand_path(base_target)
    return path


def _validate_local_pip_specs(env_name, pip_specs):
    missing = []
    not_projects = []
    for spec in pip_specs:
        spec_stripped = str(spec).strip()
        target = _pip_install_target(spec_stripped)
        if not target or not _looks_like_local_pip_target(target):
            continue
        path = _local_pip_target_path(target)
        if not path.exists():
            missing.append((spec_stripped, path))
            continue
        if path.is_dir() and not any(
            (path / marker).exists()
            for marker in ("pyproject.toml", "setup.py", "setup.cfg")
        ):
            not_projects.append((spec_stripped, path))

    if not missing and not not_projects:
        return

    lines = [
        "Local pip source validation failed for env '%s'." % env_name,
    ]
    if missing:
        lines.append("Missing local pip paths:")
        lines.extend(["  - %s -> %s" % item for item in missing])
    if not_projects:
        lines.append("Local pip paths are not Python projects:")
        lines.extend(["  - %s -> %s" % item for item in not_projects])
    lines.extend(
        [
            "Initialize vendored submodules with `git submodule update --init --recursive --depth 1 --jobs 4`.",
            "Use the reviewed repo checkout and the submodule URLs recorded in `.gitmodules`; do not upload a local source tree as the normal install source.",
        ]
    )
    raise RuntimeError("\n".join(lines))


_VENDOR_PATCHABLE_PIP_DIRS = {
    "alphafold2": "alphafold2",
    "chai_lab": "chai_lab",
    "boltz": "boltz",
}


def _vendor_patch_groups_for_pip_specs(pip_specs):
    repo_root = REPO_ROOT
    groups = []
    for spec in pip_specs:
        spec_stripped = str(spec).strip()
        target = _pip_install_target(spec_stripped)
        if not target or not _looks_like_local_pip_target(target):
            continue
        target_path = _local_pip_target_path(target).resolve()
        for group, vendor_dir in _VENDOR_PATCHABLE_PIP_DIRS.items():
            expected = (repo_root / "vendor" / vendor_dir).resolve()
            if target_path == expected and group not in groups:
                groups.append(group)
    return groups


def _apply_repo_vendor_patches_for_pip_specs(pip_specs):
    groups = _vendor_patch_groups_for_pip_specs(pip_specs)
    if not groups:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "maintenance" / "apply_vendor_patches.py"),
    ]
    for group in groups:
        cmd.extend(["--only", group])
    _run(cmd)


def _gitmodules_urls(repo_root):
    out = {}
    gitmodules = pathlib.Path(repo_root) / ".gitmodules"
    if not gitmodules.exists():
        return out
    try:
        raw = subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                str(gitmodules),
                "--get-regexp",
                r"^submodule\..*\.url$",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return out
    for line in raw.splitlines():
        key, url = line.split(None, 1)
        prefix = "submodule."
        suffix = ".url"
        if key.startswith(prefix) and key.endswith(suffix):
            out[key[len(prefix) : -len(suffix)]] = url.strip()
    return out


def _gitmodules_branches(repo_root):
    repo_root = pathlib.Path(repo_root)
    gitmodules = repo_root / ".gitmodules"
    out = {}
    if not gitmodules.exists():
        return out
    try:
        raw = subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                str(gitmodules),
                "--get-regexp",
                r"^submodule\..*\.branch$",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return out
    for line in raw.splitlines():
        key, branch = line.split(None, 1)
        prefix = "submodule."
        suffix = ".branch"
        if key.startswith(prefix) and key.endswith(suffix):
            out[key[len(prefix) : -len(suffix)]] = branch.strip()
    return out


def _gitlink_commits(repo_root):
    try:
        raw = subprocess.check_output(
            ["git", "ls-tree", "-rz", "HEAD"],
            cwd=str(repo_root),
        )
    except subprocess.CalledProcessError:
        return {}
    out = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, object_type, commit = metadata.decode("ascii").split()
        if mode == "160000" and object_type == "commit":
            out[os.fsdecode(raw_path)] = commit
    return out


def _vendor_patch_script_groups(repo_root):
    import importlib.util

    script_path = pathlib.Path(repo_root) / "scripts" / "maintenance" / "apply_vendor_patches.py"
    spec = importlib.util.spec_from_file_location("_gh_apply_vendor_patches_inventory", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vendor patch helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.VENDOR_PATCHES)


def _validate_third_party_source_inventory(repo_root):
    from omegaconf import OmegaConf

    repo_root = pathlib.Path(repo_root).resolve()
    inventory_path = repo_root / "config" / "third_party_sources.yaml"
    inventory_path = pathlib.Path(os.environ.get("GH_THIRD_PARTY_SOURCES", inventory_path))
    data = OmegaConf.to_container(OmegaConf.load(str(inventory_path)), resolve=True)
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, dict):
        raise ValueError(f"Invalid source inventory: {inventory_path}")

    gitmodule_urls = _gitmodules_urls(repo_root)
    gitmodule_branches = _gitmodules_branches(repo_root)
    gitlink_commits = _gitlink_commits(repo_root)
    patch_groups = _vendor_patch_script_groups(repo_root)
    inventory_patch_groups = set()
    inventory_paths = set()
    errors = []

    for name, spec in components.items():
        if not isinstance(spec, dict):
            errors.append(f"{name}: source spec must be a mapping")
            continue
        path = spec.get("path")
        expected_url = spec.get("expected_url")
        source_mode = spec.get("source_mode")
        migration_status = spec.get("migration_status")
        patch_group = spec.get("patch_group")
        if not path or not expected_url or not source_mode or not migration_status:
            errors.append(f"{name}: path, expected_url, source_mode, and migration_status are required")
            continue
        inventory_paths.add(str(path))
        actual_url = gitmodule_urls.get(str(path))
        if actual_url != expected_url:
            errors.append(
                f"{name}: .gitmodules url for {path} is {actual_url!r}, expected {expected_url!r}"
            )
        expected_branch = spec.get("expected_branch")
        actual_branch = gitmodule_branches.get(str(path))
        if expected_branch is not None and actual_branch != expected_branch:
            errors.append(
                f"{name}: .gitmodules branch for {path} is {actual_branch!r}, expected {expected_branch!r}"
            )
        if expected_branch is None and actual_branch is not None:
            errors.append(
                f"{name}: .gitmodules branch for {path} is {actual_branch!r}, but source inventory has no expected_branch"
            )
        expected_commit = spec.get("expected_commit")
        actual_commit = gitlink_commits.get(str(path))
        if expected_branch is not None and expected_commit is None:
            errors.append(f"{name}: expected_branch requires an exact expected_commit")
        if expected_commit is not None and actual_commit != expected_commit:
            errors.append(
                f"{name}: gitlink for {path} is {actual_commit!r}, expected {expected_commit!r}"
            )
        if patch_group is not None:
            inventory_patch_groups.add(str(patch_group))
            if str(patch_group) != str(name):
                errors.append(f"{name}: patch_group should match component name")
            if "patch" not in str(source_mode):
                errors.append(f"{name}: patch_group set but source_mode is {source_mode!r}")
        if bool(spec.get("fork_candidate")) and migration_status != "fork_pending":
            errors.append(f"{name}: fork_candidate entries must use migration_status=fork_pending")
        if migration_status == "fork_pending":
            if not bool(spec.get("fork_candidate")):
                errors.append(f"{name}: fork_pending entries must set fork_candidate=true")
            if source_mode != "official_submodule_plus_patch":
                errors.append(
                    f"{name}: fork_pending entries must use source_mode=official_submodule_plus_patch"
                )

    missing_patch_groups = sorted(patch_groups - inventory_patch_groups)
    extra_patch_groups = sorted(inventory_patch_groups - patch_groups)
    if missing_patch_groups:
        errors.append(f"Patch groups missing from source inventory: {missing_patch_groups}")
    if extra_patch_groups:
        errors.append(f"Unknown patch groups in source inventory: {extra_patch_groups}")
    missing_inventory_paths = sorted(set(gitmodule_urls) - inventory_paths)
    if missing_inventory_paths:
        errors.append(f"Git submodules missing from source inventory: {missing_inventory_paths}")
    if errors:
        raise RuntimeError("Third-party source inventory validation failed:\n" + "\n".join(errors))


def _apply_env_substitutions(value, substitutions):
    """
    Very small templating: replace ${VAR} with env var, and {{KEY}} with substitutions[KEY].
    """
    if not isinstance(value, str):
        return value
    out = os.path.expandvars(value)
    for k, v in substitutions.items():
        out = out.replace("{{%s}}" % k, str(v))
    return out


def _repo_submodule_paths(repo_root=None):
    repo_root = pathlib.Path(repo_root or REPO_ROOT).resolve()
    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        return set()
    paths = set()
    for line in gitmodules.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path"):
            continue
        key, sep, value = stripped.partition("=")
        if sep and key.strip() == "path":
            paths.add((repo_root / value.strip()).resolve())
    return paths


def _validate_repo_dest_not_submodule(repo_spec, substitutions):
    dest_raw = _apply_env_substitutions(repo_spec.get("dest"), substitutions)
    if not dest_raw:
        return
    dest = _expand_path(dest_raw).resolve()
    repo_root = REPO_ROOT
    submodule_paths = _repo_submodule_paths()
    owner = None
    for submodule_path in sorted(
        submodule_paths,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if dest == submodule_path or submodule_path in dest.parents:
            owner = submodule_path
            break
    if owner is None:
        return
    rel_dest = dest.relative_to(repo_root)
    rel_owner = owner.relative_to(repo_root)
    raise RuntimeError(
        "Env repo spec points inside vendored submodule path %s: %s. "
        "Initialize it with `git submodule update --init --recursive %s` "
        "instead of cloning/checking out that path through env_setup.py."
        % (rel_owner, rel_dest, rel_owner)
    )


def _materialize_list(items, substitutions):
    out = []
    for item in items or []:
        out.append(_apply_env_substitutions(item, substitutions))
    return out


def _run_env_command(cmd, *, manager, env_name, root_prefix, micromamba, conda_exe, env, cwd=None):
    if manager == "micromamba":
        _run(
            [micromamba, "run", "-n", env_name, "--root-prefix", str(root_prefix)]
            + shlex.split(cmd),
            env=env,
            cwd=cwd,
        )
    elif manager == "conda":
        _run(
            [conda_exe, "run", "-n", env_name] + shlex.split(cmd),
            env=env,
            cwd=cwd,
        )
    elif manager == "external":
        _run(shlex.split(cmd), env=env, cwd=cwd)
    elif manager in {"path", "asset"}:
        _run(["bash", "-lc", cmd], env=env, cwd=cwd)


def _ensure_git_repo(repo_spec, substitutions, non_interactive):
    """
    Ensure a git repository exists at dest. Supports idempotent re-runs.

    repo_spec:
      - name (optional)
      - url (required)
      - dest (required)
      - ref (optional): branch/tag/commit
      - recursive (optional bool): use --recursive on clone
    """
    url = _apply_env_substitutions(repo_spec.get("url"), substitutions)
    dest_raw = _apply_env_substitutions(repo_spec.get("dest"), substitutions)
    if not url or not dest_raw:
        raise ValueError("repo spec must include url and dest")
    _validate_repo_dest_not_submodule(repo_spec, substitutions)
    dest = _expand_path(dest_raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = repo_spec.get("ref")
    recursive = bool(repo_spec.get("recursive", False))

    if dest.exists() and (dest / ".git").exists():
        _run(["git", "fetch", "--all", "--tags"], cwd=str(dest))
    else:
        if dest.exists() and any(dest.iterdir()):
            if non_interactive:
                raise RuntimeError("Repo dest exists and is non-empty: %s" % dest)
            act = _prompt_choice(
                "Repo dest exists and is non-empty: %s. Continue anyway?" % dest,
                choices=[("y", "yes"), ("n", "no")],
                default_key="n",
                non_interactive=non_interactive,
            )
            if act == "n":
                raise RuntimeError("Aborted due to non-empty repo dest: %s" % dest)
        clone_cmd = ["git", "clone"]
        if recursive:
            clone_cmd.append("--recursive")
        clone_cmd.extend([url, str(dest)])
        _run(clone_cmd)

    if ref:
        ref = _apply_env_substitutions(ref, substitutions)
        _run(["git", "checkout", ref], cwd=str(dest))

    return dest


def _set_nested_key(mapping, dotted_key, value):
    """
    Set a dotted key like "a.b.c" into a nested dict.
    """
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        return
    cur = mapping
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _resolve_export_entry(key, entry, substitutions, env):
    optional = False
    env_var = None
    value = entry
    if isinstance(entry, dict):
        if "value" not in entry:
            raise ValueError(
                "export entry for key '%s' must be a string or a dict with 'value'"
                % key
            )
        value = entry.get("value")
        optional = bool(entry.get("optional"))
        env_var = entry.get("env_var")
        if env_var is not None and not isinstance(env_var, str):
            raise ValueError("export entry for key '%s' has non-string env_var" % key)

    if env_var:
        env_value = env.get(env_var)
        if env_value:
            return str(env_value), {
                "optional": optional,
                "env_var": env_var,
                "used_env_var": True,
            }
    return _apply_env_substitutions(value, substitutions), {
        "optional": optional,
        "env_var": env_var,
        "used_env_var": False,
    }


def _collect_export_path_issues(exports, export_meta):
    missing = []
    unresolved = []
    for k, v in sorted(exports.items()):
        if not isinstance(v, str) or not v:
            continue
        if "{{" in v or "}}" in v or "${" in v:
            unresolved.append((k, v))
            continue
        if v.startswith("~"):
            vv = os.path.expanduser(v)
        else:
            vv = v
        pth = pathlib.Path(vv)
        if not pth.exists():
            meta = export_meta.get(k, {})
            if meta.get("optional") and not meta.get("used_env_var"):
                continue
            missing.append((k, v))
    return missing, unresolved


def _maybe_bootstrap_omegaconf(args, root_prefix, bin_dir, decisions_file_hint=None):
    """
    If omegaconf isn't importable in the current interpreter, bootstrap a tiny
    micromamba env with omegaconf installed and re-exec this script within it.
    """
    try:
        import omegaconf  # noqa: F401

        return
    except Exception:
        pass

    if os.environ.get("GH_ENV_SETUP_BOOTSTRAPPED") == "1":
        raise RuntimeError(
            "Failed to import omegaconf even after bootstrapping. "
            "Check micromamba installation and network access."
        )

    micromamba = _detect_executable("micromamba")
    if not micromamba:
        micromamba = _bootstrap_micromamba(bin_dir)

    env = dict(os.environ)
    env["MAMBA_ROOT_PREFIX"] = str(root_prefix)

    bootstrap_prefix = _env_prefix_micromamba(root_prefix, BOOTSTRAP_ENV_NAME)
    if not _env_exists(bootstrap_prefix):
        _run(
            [
                micromamba,
                "create",
                "-y",
                "-n",
                BOOTSTRAP_ENV_NAME,
                "--root-prefix",
                str(root_prefix),
                "-c",
                "conda-forge",
                "python=3.11",
                "pip",
            ],
            env=env,
        )
        _run(
            [
                micromamba,
                "run",
                "-n",
                BOOTSTRAP_ENV_NAME,
                "--root-prefix",
                str(root_prefix),
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "omegaconf",
            ],
            env=env,
        )

    # Re-exec inside the bootstrap env.
    cmd = [
        micromamba,
        "run",
        "-n",
        BOOTSTRAP_ENV_NAME,
        "--root-prefix",
        str(root_prefix),
        "python",
        str(pathlib.Path(__file__).resolve()),
    ] + sys.argv[1:]
    env["GH_ENV_SETUP_BOOTSTRAPPED"] = "1"
    if decisions_file_hint:
        env.setdefault("GH_ENV_SETUP_DECISIONS", decisions_file_hint)
    sprint("Re-executing via bootstrap env to get omegaconf...")
    subprocess.run(cmd, env=env, check=True)
    raise SystemExit(0)


def _setup_one_env(env_spec, ctx):
    """
    Executes steps for a single environment.
    """
    env_name = env_spec["name"]
    non_interactive = ctx["non_interactive"]
    root_prefix = ctx["root_prefix"]
    micromamba = ctx["micromamba"]
    state_root = ctx["state_root"]
    substitutions = ctx["substitutions"].copy()
    substitutions["ENV_NAME"] = env_name
    substitutions["CLONE_ROOT"] = str(ctx.get("clone_root") or (root_prefix / "repos"))

    manager = (env_spec.get("manager") or "micromamba").strip().lower()

    if manager == "micromamba":
        prefix = _env_prefix_micromamba(root_prefix, env_name)
    elif manager in {"conda", "external"}:
        # For non-micromamba managers we create a stable symlink under <root>/envs/<name>
        # that points to the real environment prefix.
        prefix = _env_prefix_micromamba(root_prefix, env_name)
    elif manager in {"path", "asset"}:
        p = env_spec.get("path") or env_spec.get("prefix")
        if not p:
            p = str(pathlib.Path(substitutions["CLONE_ROOT"]) / env_name)
        prefix = _expand_path(_apply_env_substitutions(p, substitutions))
    else:
        raise ValueError("Unknown manager for env '%s': %s" % (env_name, manager))

    # Allow specs to reference the computed prefix in pip/commands via {{PREFIX}}.
    substitutions["PREFIX"] = str(prefix)
    state = _load_state(state_root, env_name)
    pip_specs = _materialize_list(env_spec.get("pip", []), substitutions)
    _validate_local_pip_specs(env_name, pip_specs)

    rerun_steps = set([s.strip() for s in (ctx.get("rerun_steps") or []) if s.strip()])
    # Special pip rerun modes: apply later once we have pip specs for this env.
    pip_rerun_mode = None
    if "pip_wheels" in rerun_steps:
        pip_rerun_mode = "wheels"
        rerun_steps.discard("pip_wheels")
    if "pip_editable" in rerun_steps:
        pip_rerun_mode = "editable"
        rerun_steps.discard("pip_editable")
    if "pip_all" in rerun_steps:
        pip_rerun_mode = "all"
        rerun_steps.discard("pip_all")
    if pip_rerun_mode:
        existing = ctx.get("pip_rerun_mode")
        if existing and existing != pip_rerun_mode:
            raise ValueError("Conflicting pip rerun modes requested.")
        ctx["pip_rerun_mode"] = pip_rerun_mode
    if rerun_steps:
        completed = list(state.get("completed_steps", []) or [])
        if "all" in rerun_steps:
            state["completed_steps"] = []
            state["fingerprints"] = {}
        else:
            # Support both coarse steps (e.g. "pip_install") and itemized steps
            # (e.g. "pip_install::<hash>") for incremental installs.
            new_completed = []
            for s in completed:
                drop = False
                for rr in rerun_steps:
                    if s == rr or s.startswith(rr + "::"):
                        drop = True
                        break
                if not drop:
                    new_completed.append(s)
            state["completed_steps"] = new_completed
            fps = state.get("fingerprints") or {}
            if not isinstance(fps, dict):
                fps = {}
            for rr in rerun_steps:
                fps.pop(rr, None)
            state["fingerprints"] = fps
        _save_state(state_root, env_name, state)

    existing_prefix = None
    if manager == "micromamba":
        existing_prefix = prefix if _env_exists(prefix) else None
    elif manager in {"path", "asset"}:
        existing_prefix = prefix if prefix.exists() else None
    elif manager == "external":
        existing_prefix = _resolve_external_env_prefix(
            env_name,
            env_spec,
            substitutions,
        )
    elif manager == "conda":
        conda_exe = ctx.get("conda_exe")
        if not conda_exe:
            raise RuntimeError(
                "env '%s' uses manager=conda but `conda` is not available in PATH. "
                "Either install conda or use manager=micromamba/external."
                % env_name
            )
        existing_prefix = _conda_find_env_prefix(conda_exe, env_name, env_spec.get("prefix"))

    if existing_prefix and not ctx["force"]:
        action = ctx["decisions"].get(env_name)
        if action is None:
            action = _prompt_choice(
                "Env '%s' already exists. What to do?" % env_name,
                choices=[
                    ("u", "use existing (continue/resume)"),
                    ("m", "mark done (skip all steps)"),
                    ("r", "remove and recreate"),
                    ("s", "skip"),
                ],
                default_key="u",
                non_interactive=non_interactive,
            )
            ctx["decisions"][env_name] = action
        if action == "s":
            return {"env": env_name, "status": "skipped", "prefix": str(prefix)}
        if action == "m":
            state["completed_steps"] = ["ALL"]
            state["prefix"] = str(existing_prefix)
            _save_state(state_root, env_name, state)
            return {"env": env_name, "status": "marked_done", "prefix": str(prefix)}
        if action == "r":
            step_rm = "remove_env"
            if step_rm not in state.get("completed_steps", []):
                if manager == "micromamba":
                    _run(
                        [
                            micromamba,
                            "remove",
                            "-y",
                            "-n",
                            env_name,
                            "--root-prefix",
                            str(root_prefix),
                        ],
                        env=ctx["env"],
                    )
                elif manager == "conda":
                    conda_exe = ctx["conda_exe"]
                    _run([conda_exe, "remove", "-y", "-n", env_name, "--all"], env=ctx["env"])
                elif manager == "external":
                    raise RuntimeError(
                        "Cannot remove external env '%s' automatically. Remove it manually or choose 'use/skip/mark-done'."
                        % env_name
                    )
                elif manager in {"path", "asset"}:
                    # Only allow deletion if the target is under our managed roots.
                    clone_root = _expand_path(substitutions["CLONE_ROOT"])
                    managed_roots = [root_prefix, clone_root]
                    allowed = False
                    for rr in managed_roots:
                        try:
                            allowed = prefix.resolve().is_relative_to(rr.resolve())
                        except Exception:
                            allowed = False
                        if allowed:
                            break
                    if not allowed:
                        raise RuntimeError(
                            "Refusing to delete path env '%s' at %s (outside %s). "
                            "Delete it manually or choose 'use/skip/mark-done'."
                            % (env_name, prefix, ", ".join(str(r) for r in managed_roots))
                        )
                    import shutil

                    shutil.rmtree(str(prefix), ignore_errors=True)
                # Clear state
                state = {"completed_steps": [], "updated_at": None}
                _save_state(state_root, env_name, state)
            else:
                sprint("skip: %s already completed" % step_rm)

    if "ALL" in state.get("completed_steps", []):
        return {"env": env_name, "status": "already_done", "prefix": str(prefix)}

    # Step 0: ensure repos
    step_repos = "repos"
    if step_repos not in state["completed_steps"]:
        repos = env_spec.get("repos") or []
        if repos:
            repo_map = {}
            for repo_spec in repos:
                if not isinstance(repo_spec, dict):
                    raise ValueError("repos entries must be mappings")
                name = repo_spec.get("name") or "repo"
                # Default dest: <clone_root>/<name>
                if not repo_spec.get("dest"):
                    repo_spec = dict(repo_spec)
                    repo_spec["dest"] = str(
                        pathlib.Path(substitutions["CLONE_ROOT"]) / str(name)
                    )
                dest = _ensure_git_repo(repo_spec, substitutions, non_interactive)
                key = "REPO_%s" % str(name).upper().replace("-", "_")
                repo_map[key] = str(dest)
            substitutions.update(repo_map)
        _mark_step_done(state_root, env_name, state, step_repos)
    else:
        sprint("skip: %s already completed" % step_repos)

    # For conda/external, create/refresh a stable symlink under root_prefix/envs/<name>.
    if manager in {"conda", "external"}:
        link_target = None
        if manager == "external":
            link_target = existing_prefix
        else:
            conda_exe = ctx["conda_exe"]
            link_target = _conda_find_env_prefix(conda_exe, env_name, env_spec.get("prefix"))
        link_target = pathlib.Path(str(link_target)).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if prefix.exists() or prefix.is_symlink():
            # If it's a symlink to the right target, keep it.
            try:
                if prefix.is_symlink() and prefix.resolve() == link_target:
                    pass
                else:
                    if not non_interactive:
                        act = _prompt_choice(
                            "Link path %s exists but doesn't match %s. Replace?" % (prefix, link_target),
                            choices=[("y", "yes"), ("n", "no")],
                            default_key="y",
                            non_interactive=non_interactive,
                        )
                        if act == "n":
                            raise RuntimeError("Refusing to overwrite existing link/path: %s" % prefix)
                    # Remove existing
                    if prefix.is_symlink() or prefix.is_file():
                        prefix.unlink()
                    else:
                        raise RuntimeError(
                            "Expected %s to be a symlink/file. Remove it manually if you want to relink."
                            % prefix
                        )
                    prefix.symlink_to(link_target)
            except FileNotFoundError:
                pass
        else:
            prefix.symlink_to(link_target)
        state["prefix"] = str(link_target)
        _mark_step_done(state_root, env_name, state, "link_prefix")

    # Step 1: create env
    step_create = "create_env"
    if step_create not in state["completed_steps"]:
        if manager == "micromamba":
            python_version = str(env_spec.get("python", "3.11"))
            channels = env_spec.get("channels") or ["conda-forge"]
            conda_pkgs = list(env_spec.get("conda", []) or [])
            args = [
                micromamba,
                "create",
                "-y",
                "-n",
                env_name,
                "--root-prefix",
                str(root_prefix),
            ]
            for ch in channels:
                args.extend(["-c", ch])
            args.append("python=%s" % python_version)
            args.append("pip")
            args.extend(conda_pkgs)
            _run(args, env=ctx["env"])
            _mark_step_done(state_root, env_name, state, step_create)
            _mark_step_done(state_root, env_name, state, "conda_install")
        elif manager == "conda":
            conda_exe = ctx["conda_exe"]
            python_version = str(env_spec.get("python", "3.11"))
            conda_pkgs = list(env_spec.get("conda", []) or [])
            env_file = env_spec.get("env_file") or env_spec.get("environment_yml")
            if env_file:
                env_file = _apply_env_substitutions(env_file, substitutions)
                env_file = str(_expand_path(env_file))
                _run([conda_exe, "env", "update", "-n", env_name, "-f", env_file], env=ctx["env"])
            else:
                args = [conda_exe, "create", "-y", "-n", env_name, "python=%s" % python_version, "pip"]
                args.extend(conda_pkgs)
                _run(args, env=ctx["env"])
            _mark_step_done(state_root, env_name, state, step_create)
            _mark_step_done(state_root, env_name, state, "conda_install")
        elif manager == "external":
            # Nothing to create.
            _mark_step_done(state_root, env_name, state, step_create)
            _mark_step_done(state_root, env_name, state, "conda_install")
        elif manager in {"path", "asset"}:
            prefix.mkdir(parents=True, exist_ok=True)
            _mark_step_done(state_root, env_name, state, step_create)
            _mark_step_done(state_root, env_name, state, "conda_install")
    else:
        sprint("skip: %s already completed" % step_create)

    # Step 1.5: ensure conda packages (useful when spec changes after env creation).
    step_conda = "conda_install"
    if step_conda not in state["completed_steps"]:
        conda_pkgs = list(env_spec.get("conda", []) or [])
        channels = env_spec.get("channels") or ["conda-forge"]
        if conda_pkgs and manager == "micromamba":
            args = [
                micromamba,
                "install",
                "-y",
                "-n",
                env_name,
                "--root-prefix",
                str(root_prefix),
            ]
            for ch in channels:
                args.extend(["-c", ch])
            args.extend(conda_pkgs)
            _run(args, env=ctx["env"])
        elif conda_pkgs and manager == "conda":
            conda_exe = ctx["conda_exe"]
            if not conda_exe:
                raise RuntimeError("manager=conda but conda executable not found")
            _run([conda_exe, "install", "-y", "-n", env_name] + conda_pkgs, env=ctx["env"])
        _mark_step_done(state_root, env_name, state, step_conda)
    else:
        sprint("skip: %s already completed" % step_conda)

    # Step 1.7: commands that must run before local editable pip installs.
    step_pre_pip = "pre_pip_commands"
    pre_pip_commands = _materialize_list(env_spec.get("pre_pip_commands", []), substitutions)
    pre_pip_fp = _fingerprint_lines(pre_pip_commands)
    pre_pip_prev_fp = (state.get("fingerprints") or {}).get(step_pre_pip)
    if step_pre_pip in state["completed_steps"] and pre_pip_prev_fp != pre_pip_fp:
        sprint("pre_pip_commands changed; will run missing commands only.")
        try:
            state["completed_steps"].remove(step_pre_pip)
        except Exception:
            pass
        _save_state(state_root, env_name, state)

    if step_pre_pip not in state["completed_steps"]:
        cwd = env_spec.get("cwd")
        if cwd:
            cwd = _apply_env_substitutions(cwd, substitutions)
            cwd = str(_expand_path(cwd))
        for cmd in pre_pip_commands:
            item_step = "pre_pip_commands::%s" % _hash12(cmd)
            if item_step in state.get("completed_steps", []):
                continue
            _run_env_command(
                cmd,
                manager=manager,
                env_name=env_name,
                root_prefix=root_prefix,
                micromamba=micromamba,
                conda_exe=ctx.get("conda_exe"),
                env=ctx["env"],
                cwd=cwd or (str(prefix) if manager in {"path", "asset"} else None),
            )
            _mark_step_done(state_root, env_name, state, item_step)
        state.setdefault("fingerprints", {})[step_pre_pip] = pre_pip_fp
        _save_state(state_root, env_name, state)
        _mark_step_done(state_root, env_name, state, step_pre_pip)
    else:
        sprint("skip: %s already completed" % step_pre_pip)

    # Step 2: pip installs
    step_pip = "pip_install"
    pip_rerun_mode = ctx.get("pip_rerun_mode")
    if pip_rerun_mode:
        # Ensure pip step executes even if already completed, but avoid reinstalling everything:
        #   - pip_wheels: rerun only non-editable specs
        #   - pip_editable: rerun only -e/--editable specs
        #   - pip_all: rerun all pip specs
        for spec in pip_specs:
            spec_stripped = str(spec).strip()
            is_editable = _is_editable_pip_spec(spec_stripped)
            if pip_rerun_mode == "wheels" and is_editable:
                continue
            if pip_rerun_mode == "editable" and (not is_editable):
                continue
            item_step = "pip_install::%s" % _hash12(spec_stripped)
            try:
                state["completed_steps"].remove(item_step)
            except Exception:
                pass
        if pip_rerun_mode == "all":
            for spec in pip_specs:
                item_step = "pip_install::%s" % _hash12(str(spec).strip())
                try:
                    state["completed_steps"].remove(item_step)
                except Exception:
                    pass
        try:
            state["completed_steps"].remove(step_pip)
        except Exception:
            pass
        _save_state(state_root, env_name, state)
    pip_fp = _fingerprint_lines(pip_specs)
    pip_prev_fp = (state.get("fingerprints") or {}).get(step_pip)
    pip_delta_update = bool(pip_prev_fp) and (pip_prev_fp != pip_fp)
    if step_pip in state["completed_steps"] and pip_prev_fp != pip_fp:
        # If the user explicitly asked to rerun some other step(s), don't surprise them by
        # re-running pip installs due to spec drift. They can opt in via `--rerun pip_install`.
        allow_auto_reconcile = (not rerun_steps) or ("all" in rerun_steps) or (step_pip in rerun_steps)
        if allow_auto_reconcile:
            sprint("pip_install changed; will install missing pip specs only.")
            try:
                state["completed_steps"].remove(step_pip)
            except Exception:
                pass
            _save_state(state_root, env_name, state)
        else:
            sprint(
                "pip_install changed but --rerun does not include pip_install; skipping pip reconciliation "
                "(run with `--rerun pip_install` to apply)."
            )

    if step_pip not in state["completed_steps"]:
        pip_env = dict(ctx["env"])
        pip_env_overrides = env_spec.get("pip_env") or {}
        if not isinstance(pip_env_overrides, dict):
            raise ValueError("env '%s' field 'pip_env' must be a mapping" % env_name)
        for k, v in pip_env_overrides.items():
            pip_env[str(k)] = str(_apply_env_substitutions(str(v), substitutions))

        if pip_specs:
            if manager in {"path", "asset"}:
                raise RuntimeError(
                    "env '%s' uses manager=%s but declares `pip:`. "
                    "Use manager=micromamba/conda/external for Python envs."
                    % (env_name, manager)
                )
            _apply_repo_vendor_patches_for_pip_specs(pip_specs)

            # One-time upgrade pip marker.
            step_pip_upgrade = "pip_install::upgrade_pip"
            if step_pip_upgrade not in state.get("completed_steps", []):
                pip_upgrade = env_spec.get("pip_upgrade", "pip")
                pip_upgrade_text = "" if pip_upgrade is False else str(pip_upgrade).strip()
                pip_upgrade_disabled = pip_upgrade_text.lower() in {"", "false", "none", "skip"}
                if pip_upgrade_disabled:
                    sprint("skip: pip upgrade disabled")
                else:
                    pip_upgrade_args = shlex.split(pip_upgrade_text)
                    if manager == "micromamba":
                        _run(
                            [
                                micromamba,
                                "run",
                                "-n",
                                env_name,
                                "--root-prefix",
                                str(root_prefix),
                                "python",
                                "-m",
                                "pip",
                                "--isolated",
                                "install",
                                "--upgrade",
                            ]
                            + pip_upgrade_args,
                            env=pip_env,
                        )
                    elif manager == "conda":
                        conda_exe = ctx["conda_exe"]
                        _run(
                            [
                                conda_exe,
                                "run",
                                "-n",
                                env_name,
                                "python",
                                "-m",
                                "pip",
                                "--isolated",
                                "install",
                                "--upgrade",
                            ]
                            + pip_upgrade_args,
                            env=pip_env,
                        )
                _mark_step_done(state_root, env_name, state, step_pip_upgrade)
            for spec in pip_specs:
                spec_stripped = str(spec).strip()
                item_step = "pip_install::%s" % _hash12(spec_stripped)
                if item_step in state.get("completed_steps", []):
                    continue

                # Delta update mode: avoid re-running expensive editable installs when the env was
                # previously provisioned and only the spec list changed (common case: adding a new
                # requirement). Users can force full reinstall via `--rerun pip_install`.
                if pip_delta_update and _is_editable_pip_spec(spec_stripped):
                    sprint("skip (delta): %s" % spec_stripped)
                    _mark_step_done(state_root, env_name, state, item_step)
                    continue
                if pip_rerun_mode == "wheels" and _is_editable_pip_spec(spec_stripped):
                    sprint("skip (rerun wheels): %s" % spec_stripped)
                    _mark_step_done(state_root, env_name, state, item_step)
                    continue
                if pip_rerun_mode == "editable" and (not _is_editable_pip_spec(spec_stripped)):
                    sprint("skip (rerun editable): %s" % spec_stripped)
                    _mark_step_done(state_root, env_name, state, item_step)
                    continue
                # `pip install <spec...>` is kept as a string so users can include extras/versions.
                if manager == "micromamba":
                    _run(
                        [
                            micromamba,
                            "run",
                            "-n",
                            env_name,
                            "--root-prefix",
                            str(root_prefix),
                            "python",
                            "-m",
                            "pip",
                            "--isolated",
                            "install",
                        ]
                        + shlex.split(spec_stripped),
                        env=pip_env,
                    )
                elif manager == "conda":
                    conda_exe = ctx["conda_exe"]
                    _run(
                        [conda_exe, "run", "-n", env_name, "python", "-m", "pip", "--isolated", "install"]
                        + shlex.split(spec_stripped),
                        env=pip_env,
                    )
                elif manager == "external":
                    # Best-effort pip install into the external env via its python.
                    py_bin = env_spec.get("python_bin")
                    if not py_bin:
                        py_bin = str(pathlib.Path(str(existing_prefix)) / "bin" / "python")
                    py_bin = str(_expand_path(py_bin))
                    _run([py_bin, "-m", "pip", "--isolated", "install"] + shlex.split(spec_stripped), env=pip_env)
                _mark_step_done(state_root, env_name, state, item_step)

        # Update fingerprint + mark step complete.
        state.setdefault("fingerprints", {})[step_pip] = pip_fp
        _save_state(state_root, env_name, state)
        _mark_step_done(state_root, env_name, state, step_pip)
    else:
        sprint("skip: %s already completed" % step_pip)

    # Step 3: optional commands (clone/build/etc.)
    step_cmds = "commands"
    commands = _materialize_list(env_spec.get("commands", []), substitutions)
    cmds_fp = _fingerprint_lines(commands)
    cmds_prev_fp = (state.get("fingerprints") or {}).get(step_cmds)
    if step_cmds in state["completed_steps"] and cmds_prev_fp != cmds_fp:
        sprint("commands changed; will run missing commands only.")
        try:
            state["completed_steps"].remove(step_cmds)
        except Exception:
            pass
        _save_state(state_root, env_name, state)

    if step_cmds not in state["completed_steps"]:
        if ctx.get("skip_commands"):
            state.setdefault("fingerprints", {})[step_cmds] = cmds_fp
            _save_state(state_root, env_name, state)
            _mark_step_done(state_root, env_name, state, step_cmds)
            return {"env": env_name, "status": "ok", "prefix": str(prefix)}

        cwd = env_spec.get("cwd")
        if cwd:
            cwd = _apply_env_substitutions(cwd, substitutions)
            cwd = str(_expand_path(cwd))
        for cmd in commands:
            item_step = "commands::%s" % _hash12(cmd)
            if item_step in state.get("completed_steps", []):
                continue
            _run_env_command(
                cmd,
                manager=manager,
                env_name=env_name,
                root_prefix=root_prefix,
                micromamba=micromamba,
                conda_exe=ctx.get("conda_exe"),
                env=ctx["env"],
                cwd=cwd or (str(prefix) if manager in {"path", "asset"} else None),
            )
            _mark_step_done(state_root, env_name, state, item_step)

        state.setdefault("fingerprints", {})[step_cmds] = cmds_fp
        _save_state(state_root, env_name, state)
        _mark_step_done(state_root, env_name, state, step_cmds)
    else:
        sprint("skip: %s already completed" % step_cmds)

    return {"env": env_name, "status": "ok", "prefix": str(prefix)}


def _conda_find_env_prefix(conda_exe, env_name, fallback_prefix=None):
    """
    Returns a pathlib.Path for the conda env prefix if it exists, else None.
    """
    if fallback_prefix:
        p = _expand_path(fallback_prefix)
        if _env_exists(p):
            return p
    try:
        out = subprocess.check_output([conda_exe, "env", "list", "--json"], text=True)
        payload = json.loads(out)
        for p in payload.get("envs", []):
            pp = pathlib.Path(p)
            if pp.name == env_name and _env_exists(pp):
                return pp
    except Exception:
        return None
    return None

def main():
    p = argparse.ArgumentParser(
        description="Prepare third-party environments (host-side)."
    )
    p.add_argument(
        "--spec",
        default=str(REPO_ROOT / "config" / "third_party_envs.yaml"),
        help="YAML spec file describing environments to create.",
    )
    p.add_argument(
        "--root-prefix",
        default=str(pathlib.Path.home() / ".gh_envs"),
        help="micromamba root prefix (envs will be under <root>/envs/<name>).",
    )
    p.add_argument(
        "--bin-dir",
        default=None,
        help="Directory to place/download micromamba (default: <root>/.bin).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallelism across envs (non-interactive only).",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive; accept defaults for existing env decisions.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing envs and proceed (still resumable by step state).",
    )
    p.add_argument(
        "--profile",
        choices=sorted(ENV_PROFILES),
        default="all",
        help=(
            "Environment profile to process. 'runtime' installs the default GH "
            "runtime path and skips optional/heavy backends such as DPLM, "
            "RAPiDock, RPXDock/WillUtil, and NA-MPNN. 'needed_softwares' adds "
            "the subprocess envs and path assets required by needed_softwares.md "
            "without pulling unrelated optional backends. 'all' preserves the "
            "legacy full spec selection."
        ),
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated env names to process (overrides --profile selection).",
    )
    p.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated env names to skip after --only/default selection.",
    )
    p.add_argument(
        "--clone-root",
        default=None,
        help="Where to place cloned repos referenced by the spec (default: <root-prefix>/repos).",
    )
    p.add_argument(
        "--skip-commands",
        action="store_true",
        help="Skip running env 'commands' steps (useful for dry bootstraps).",
    )
    p.add_argument(
        "--rerun",
        default=None,
        help=(
            "Comma-separated step names to re-run (e.g. 'conda_install,pip_install,commands' or 'all'). "
            "Special: 'pip_wheels' reruns only non-editable pip specs, 'pip_editable' reruns only '-e/--editable' specs, "
            "and 'pip_all' reruns all pip specs. "
            "When provided, other steps will NOT be auto-rerun just because the spec changed; "
            "include that step explicitly if you want it updated."
        ),
    )
    p.add_argument(
        "--preserve-host-compile-env",
        action="store_true",
        help="Preserve host compile-related env vars (e.g. LIBRARY_PATH) when invoking micromamba/pip.",
    )
    p.add_argument(
        "--decisions",
        default=None,
        help="JSON file with per-env decisions for existing envs (env->u/m/r/s).",
    )
    p.add_argument(
        "--write-decisions",
        default=None,
        help="Write decisions taken during this run to a JSON file.",
    )
    p.add_argument(
        "--write-env-yaml",
        default=None,
        help=(
            "Update an env.yaml (or write a new one) with exported values from the env spec. "
            "Example: config/env/env_local.yaml."
        ),
    )
    p.add_argument(
        "--verify-env-yaml",
        action="store_true",
        help="After writing --write-env-yaml, verify that exported paths exist.",
    )
    p.add_argument(
        "--soft-fail",
        action="store_true",
        help="Exit 0 even if verification finds missing paths.",
    )
    p.add_argument(
        "--env-yaml-mode",
        choices=["merge", "overwrite"],
        default="merge",
        help="How to write --write-env-yaml: merge into existing file or overwrite.",
    )
    args = p.parse_args()

    root_prefix = _expand_path(args.root_prefix)
    bin_dir = _expand_path(args.bin_dir) if args.bin_dir else (root_prefix / ".bin")
    state_root = root_prefix / STATE_DIRNAME
    non_interactive = bool(args.yes)

    if non_interactive and args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if (not non_interactive) and args.jobs != 1:
        sprint("Interactive mode forces --jobs=1")
        args.jobs = 1

    # Bootstrap omegaconf if missing (allows running from system python).
    _maybe_bootstrap_omegaconf(
        args=args,
        root_prefix=root_prefix,
        bin_dir=bin_dir,
        decisions_file_hint=args.decisions,
    )

    # Now omegaconf must be available.
    from omegaconf import OmegaConf

    spec_path = _expand_path(args.spec)
    spec = OmegaConf.to_container(OmegaConf.load(str(spec_path)), resolve=True)
    if not isinstance(spec, dict) or "envs" not in spec:
        raise ValueError("Invalid spec file: expected top-level 'envs'")
    _validate_third_party_source_inventory(REPO_ROOT)

    envs = spec["envs"]
    if not isinstance(envs, list):
        raise ValueError("Spec 'envs' must be a list")

    profile_only = None
    if args.only is None and ENV_PROFILES[args.profile] is not None:
        profile_only = ",".join(ENV_PROFILES[args.profile])
    selected = _select_env_specs(envs, only=args.only or profile_only, exclude=args.exclude)

    if not selected:
        raise RuntimeError("No environments selected.")

    # micromamba selection/bootstrap
    micromamba = _detect_executable("micromamba")
    if not micromamba:
        micromamba = _bootstrap_micromamba(bin_dir)
    sprint("micromamba: %s" % micromamba)

    conda_exe = _detect_executable("conda")

    decisions = {}
    if args.decisions:
        decisions_path = _expand_path(args.decisions)
        decisions = _json_load(decisions_path) or {}
        if not isinstance(decisions, dict):
            raise ValueError("--decisions must be a JSON object (env_name -> action)")

    ctx = {
        "root_prefix": root_prefix,
        "micromamba": micromamba,
        "state_root": state_root,
        "non_interactive": non_interactive,
        "force": bool(args.force),
        "decisions": decisions,
        "substitutions": {
            "ROOT_PREFIX": str(root_prefix),
            "REPO_ROOT": str(REPO_ROOT),
        },
        "env": dict(os.environ),
        "conda_exe": conda_exe,
    }
    ctx["skip_commands"] = bool(args.skip_commands)
    ctx["rerun_steps"] = [s.strip() for s in (args.rerun or "").split(",") if s.strip()]
    ctx["clone_root"] = (
        _expand_path(args.clone_root) if args.clone_root else (root_prefix / "repos")
    )

    _sanitize_host_compile_env(ctx["env"], preserve=bool(args.preserve_host_compile_env))
    _sanitize_broken_proxy_env(ctx["env"])

    # Ensure micromamba sees its root prefix.
    ctx["env"]["MAMBA_ROOT_PREFIX"] = str(root_prefix)

    sprint("Planned environments:")
    for env_spec in selected:
        env_name = env_spec["name"]
        manager = (env_spec.get("manager") or "micromamba").strip().lower()
        if manager == "micromamba" or manager in {"conda", "external"}:
            prefix = _env_prefix_micromamba(root_prefix, env_name)
        elif manager in {"path", "asset"}:
            subs0 = ctx["substitutions"].copy()
            subs0["ENV_NAME"] = env_name
            subs0["CLONE_ROOT"] = str(ctx.get("clone_root") or (root_prefix / "repos"))
            p = env_spec.get("path") or env_spec.get("prefix")
            if not p:
                p = str(pathlib.Path(subs0["CLONE_ROOT"]) / env_name)
            prefix = _expand_path(_apply_env_substitutions(p, subs0))
        else:
            prefix = _env_prefix_micromamba(root_prefix, env_name)
        state = _load_state(state_root, env_name)
        sprint("- " + _render_env_summary(env_name, prefix, state))

    results = []

    if args.jobs == 1:
        for env_spec in selected:
            res = _setup_one_env(env_spec, ctx)
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_setup_one_env, env_spec, ctx) for env_spec in selected]
            for fut in as_completed(futs):
                results.append(fut.result())

    if args.write_decisions:
        out_path = _expand_path(args.write_decisions)
        _json_dump_atomic(out_path, ctx["decisions"])
        sprint("Wrote decisions: %s" % out_path)

    gh_python_hint = None
    for res in results:
        if str(res.get("env") or "") == "gh" and res.get("prefix"):
            gh_python_hint = str(pathlib.Path(str(res["prefix"])) / "bin" / "python")
            break

    _write_gh_python_hint(REPO_ROOT, gh_python_hint)

    if args.write_env_yaml:
        env_yaml_path = _expand_path(args.write_env_yaml)
        env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        exports = {}
        export_meta = {}
        for env_spec in selected:
            env_name = env_spec.get("name")
            if not env_name:
                continue
            manager = (env_spec.get("manager") or "micromamba").strip().lower()
            # Determine actual prefix
            if manager == "micromamba":
                actual_prefix = _env_prefix_micromamba(root_prefix, env_name)
            elif manager in {"path", "asset"}:
                subs0 = ctx["substitutions"].copy()
                subs0["ENV_NAME"] = env_name
                subs0["CLONE_ROOT"] = str(ctx.get("clone_root") or (root_prefix / "repos"))
                p = env_spec.get("path") or env_spec.get("prefix")
                if not p:
                    p = str(pathlib.Path(subs0["CLONE_ROOT"]) / env_name)
                actual_prefix = _expand_path(_apply_env_substitutions(p, subs0))
            elif manager == "external":
                subs0 = ctx["substitutions"].copy()
                subs0["ENV_NAME"] = env_name
                subs0["CLONE_ROOT"] = str(ctx.get("clone_root") or (root_prefix / "repos"))
                actual_prefix = _resolve_external_env_prefix(
                    env_name,
                    env_spec,
                    subs0,
                )
            elif manager == "conda":
                actual_prefix = _conda_find_env_prefix(
                    ctx.get("conda_exe"), env_name, env_spec.get("prefix")
                )
                if not actual_prefix:
                    continue
            else:
                continue

            substitutions = ctx["substitutions"].copy()
            substitutions["ENV_NAME"] = env_name
            substitutions["PREFIX"] = str(actual_prefix)
            if manager in {"path", "asset"}:
                substitutions["PYTHON_BIN"] = ""
            else:
                substitutions["PYTHON_BIN"] = str(actual_prefix / "bin" / "python")
            substitutions["CLONE_ROOT"] = str(ctx.get("clone_root") or (root_prefix / "repos"))

            # Allow exports to reference repo paths (REPO_<NAME>) if repos are declared.
            repos = env_spec.get("repos") or []
            if repos:
                for repo_spec in repos:
                    if not isinstance(repo_spec, dict):
                        continue
                    repo_name = repo_spec.get("name") or "repo"
                    dest = repo_spec.get("dest")
                    if not dest:
                        dest = str(pathlib.Path(substitutions["CLONE_ROOT"]) / str(repo_name))
                    dest = _apply_env_substitutions(dest, substitutions)
                    key = "REPO_%s" % str(repo_name).upper().replace("-", "_")
                    substitutions[key] = str(_expand_path(dest))

            export_map = env_spec.get("export") or env_spec.get("exports") or {}
            if isinstance(export_map, dict):
                for k, v in export_map.items():
                    value, meta = _resolve_export_entry(
                        k,
                        v,
                        substitutions,
                        ctx["env"],
                    )
                    exports[k] = value
                    export_meta[k] = meta

        header = "# @package _global_\n\n"
        if args.env_yaml_mode == "overwrite":
            out_obj = {}
            for k, v in exports.items():
                _set_nested_key(out_obj, k, v)
            payload = header + OmegaConf.to_yaml(OmegaConf.create(out_obj), resolve=False)
            _write_text_atomic(env_yaml_path, payload)
            sprint("Wrote env yaml (overwrite): %s" % env_yaml_path)
        else:
            had_file = env_yaml_path.exists()
            if had_file:
                cfg = OmegaConf.load(str(env_yaml_path))
                out_obj = OmegaConf.to_container(cfg, resolve=False)
                if not isinstance(out_obj, dict):
                    out_obj = {}
            else:
                seed_path = (
                    REPO_ROOT / "config" / "env.yaml"
                )
                if seed_path.exists():
                    cfg = OmegaConf.load(str(seed_path))
                    out_obj = OmegaConf.to_container(cfg, resolve=False)
                    if not isinstance(out_obj, dict):
                        out_obj = {}
                else:
                    out_obj = {}

            for k, v in exports.items():
                _set_nested_key(out_obj, k, v)
            payload = header + OmegaConf.to_yaml(OmegaConf.create(out_obj), resolve=False)
            _write_text_atomic(env_yaml_path, payload)
            if had_file:
                sprint("Updated env yaml (merge): %s" % env_yaml_path)
            else:
                sprint("Wrote env yaml (merge, seeded): %s" % env_yaml_path)

        if args.verify_env_yaml:
            missing, unresolved = _collect_export_path_issues(exports, export_meta)

            if unresolved:
                sprint("Verification: unresolved values (placeholders/interpolations):")
                for k, v in unresolved:
                    sprint("  - %s = %s" % (k, v))
            if missing:
                sprint("Verification: missing paths:")
                for k, v in missing:
                    sprint("  - %s = %s" % (k, v))
            sprint(
                "Verification summary: exported=%d missing=%d unresolved=%d"
                % (len(exports), len(missing), len(unresolved))
            )
            if missing and not args.soft_fail:
                raise SystemExit(2)

    sprint("Summary:")
    for r in sorted(results, key=lambda x: x["env"]):
        sprint("- %(env)s: %(status)s  (%(prefix)s)" % r)


if __name__ == "__main__":
    main()
