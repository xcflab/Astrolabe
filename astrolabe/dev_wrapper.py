from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _config_env_dir(config_dir: str | Path) -> Path:
    return Path(config_dir) / "env"


def default_local_env_path_for_config_dir(config_dir: str | Path) -> Path:
    return _config_env_dir(config_dir) / "env_local.yaml"


def default_local_env_path(repo_root: str | Path) -> Path:
    return default_local_env_path_for_config_dir(Path(repo_root) / "config")


def default_astrolabe_python_hint_path_for_config_dir(config_dir: str | Path) -> Path:
    return _config_env_dir(config_dir) / ".astrolabe_python_path"


def default_astrolabe_python_hint_path(repo_root: str | Path) -> Path:
    return default_astrolabe_python_hint_path_for_config_dir(Path(repo_root) / "config")


def _legacy_gh_python_hint_path_for_config_dir(config_dir: str | Path) -> Path:
    return _config_env_dir(config_dir) / ".gh_python_path"


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value.rstrip("\n")
    return value.split("#", 1)[0].rstrip()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _read_nested_yaml_scalar(path: Path, keys: Iterable[str]) -> str | None:
    if not path.exists():
        return None
    key_list = list(keys)
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        parents = [name for _, name in stack]
        if parents == key_list[:-1] and key == key_list[-1]:
            if value in {"", "null", "None"}:
                return None
            return _strip_quotes(value)
        if value == "":
            stack.append((indent, key))
    return None


def _executable_path(value: str | None) -> str | None:
    if not value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if not expanded.is_absolute():
        return None
    if expanded.is_file() and os.access(str(expanded), os.X_OK):
        return str(expanded)
    return None


def _iter_container_python_wrappers(home: Path | None = None) -> Iterable[Path]:
    base = home or Path.home()
    patterns = (
        base / ".astrolabe_envs" / "tools" / "astrolabe_*_container_python.py",
        base / ".gh_envs" / "tools" / "gh_*_container_python.py",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(pattern.parent.glob(pattern.name)))
    return paths


def _iter_standard_astrolabe_python_candidates(home: Path | None = None) -> Iterable[Path]:
    base = home or Path.home()
    return [
        base / ".astrolabe_envs" / "envs" / "astrolabe" / "bin" / "python",
        base / ".astrolabe_envs" / "envs" / "astrolabe" / "bin" / "python3",
        base / ".gh_envs" / "envs" / "gh" / "bin" / "python",
        base / ".gh_envs" / "envs" / "gh" / "bin" / "python3",
    ]


def resolve_default_astrolabe_python(repo_root: str | Path) -> str | None:
    root = Path(repo_root)
    config_dir = root / "config"
    for hint_path in (
        default_astrolabe_python_hint_path(root),
        _legacy_gh_python_hint_path_for_config_dir(config_dir),
    ):
        if hint_path.exists():
            valid = _executable_path(hint_path.read_text(encoding="utf-8").strip())
            if valid:
                return valid

    env_local_path = default_local_env_path(root)
    for keys in (
        ["subprocess_envs", "astrolabe", "python_bin"],
        ["subprocess_envs", "gh", "python_bin"],
    ):
        valid = _executable_path(_read_nested_yaml_scalar(env_local_path, keys))
        if valid:
            return valid

    for candidate in _iter_standard_astrolabe_python_candidates():
        valid = _executable_path(str(candidate))
        if valid:
            return valid

    for candidate in _iter_container_python_wrappers():
        valid = _executable_path(str(candidate))
        if valid:
            return valid
    return None


def write_astrolabe_python_hint(repo_root: str | Path, python_bin: str | None) -> Path | None:
    if not python_bin:
        return None
    value = str(python_bin).strip()
    if not value:
        return None
    hint_path = default_astrolabe_python_hint_path(repo_root)
    hint_path.parent.mkdir(parents=True, exist_ok=True)
    hint_path.write_text(value + "\n", encoding="utf-8")
    return hint_path


def write_astrolabe_python_hint_for_config_dir(
    config_dir: str | Path, python_bin: str | None
) -> Path | None:
    if not python_bin:
        return None
    value = str(python_bin).strip()
    if not value:
        return None
    hint_path = default_astrolabe_python_hint_path_for_config_dir(config_dir)
    hint_path.parent.mkdir(parents=True, exist_ok=True)
    hint_path.write_text(value + "\n", encoding="utf-8")
    return hint_path


# Compatibility aliases for existing bootstrap/configuration callers.
def default_gh_python_hint_path_for_config_dir(config_dir: str | Path) -> Path:
    return _legacy_gh_python_hint_path_for_config_dir(config_dir)


def default_gh_python_hint_path(repo_root: str | Path) -> Path:
    return default_gh_python_hint_path_for_config_dir(Path(repo_root) / "config")


def resolve_default_gh_python(repo_root: str | Path) -> str | None:
    return resolve_default_astrolabe_python(repo_root)


def write_gh_python_hint(repo_root: str | Path, python_bin: str | None) -> Path | None:
    return write_gh_python_hint_for_config_dir(Path(repo_root) / "config", python_bin)


def write_gh_python_hint_for_config_dir(
    config_dir: str | Path, python_bin: str | None
) -> Path | None:
    if not python_bin or not str(python_bin).strip():
        return None
    hint_path = default_gh_python_hint_path_for_config_dir(config_dir)
    hint_path.parent.mkdir(parents=True, exist_ok=True)
    hint_path.write_text(str(python_bin).strip() + "\n", encoding="utf-8")
    return hint_path
