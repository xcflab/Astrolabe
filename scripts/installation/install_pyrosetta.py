#!/usr/bin/env python3
"""
Install PyRosetta into the *current* Python environment.

Motivation:
- `pyrosetta` wheels are hosted on the Graylab PyRosetta4 archives (not on PyPI).
- The core env bootstrap (`scripts/installation/env_setup.py`) is YAML-driven; this helper gives a
  reproducible, debuggable way to install PyRosetta as an explicit step.

Behavior:
- If PyRosetta is already importable, exit 0 by default.
- If GH_PYROSETTA_WHEEL is set (local path or https URL), install that wheel.
- Else, delegate to `pyrosetta_installer.install_pyrosetta()` (installs latest compatible wheel).
"""

import argparse
import os
import pathlib
import platform
import subprocess
import sys
from typing import Optional


def _pip_install(spec: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", spec])


def _try_import_pyrosetta() -> bool:
    try:
        import pyrosetta  # noqa: F401

        return True
    except Exception:
        return False


def _derive_root_prefix_from_python_bin(python_bin: str | os.PathLike[str]) -> Optional[pathlib.Path]:
    python_path = pathlib.Path(python_bin).expanduser().resolve(strict=False)
    if python_path.parent.name != "bin":
        return None
    env_prefix = python_path.parent.parent
    envs_dir = env_prefix.parent
    if envs_dir.name != "envs":
        return None
    return envs_dir.parent


def _current_wheel_abi_tag(py_version: tuple[int, int] | None = None) -> str:
    major, minor = py_version or sys.version_info[:2]
    return f"cp{major}{minor}"


def _current_wheel_platform_fragment() -> str:
    return f"{platform.system().lower()}_{platform.machine().lower()}"


def _find_compatible_cached_wheel(
    download_dir: str | os.PathLike[str],
    *,
    py_version: tuple[int, int] | None = None,
) -> Optional[pathlib.Path]:
    wheel_dir = pathlib.Path(download_dir)
    if not wheel_dir.is_dir():
        return None

    abi_tag = _current_wheel_abi_tag(py_version)
    platform_fragment = _current_wheel_platform_fragment()
    candidates = sorted(wheel_dir.glob("pyrosetta-*.whl"), reverse=True)
    for wheel_path in candidates:
        name = wheel_path.name
        if f"-{abi_tag}-{abi_tag}-" not in name:
            continue
        if platform_fragment not in name:
            continue
        return wheel_path
    return None


def _find_cached_wheel_for_current_env() -> Optional[pathlib.Path]:
    root_prefix = _derive_root_prefix_from_python_bin(sys.executable)
    if root_prefix is None:
        return None
    return _find_compatible_cached_wheel(root_prefix / "downloads" / "pyrosetta")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-skip", action="store_true", help="Always attempt install even if pyrosetta imports.")
    args = p.parse_args()

    if _try_import_pyrosetta() and not args.no_skip:
        print("PyRosetta already importable; skipping.")
        return 0

    wheel_override = os.environ.get("GH_PYROSETTA_WHEEL", "").strip()
    if wheel_override:
        print(f"Installing PyRosetta from GH_PYROSETTA_WHEEL={wheel_override}")
        _pip_install(wheel_override)
        ok = _try_import_pyrosetta()
        if not ok:
            raise SystemExit("PyRosetta install finished but import still fails.")
        return 0

    cached_wheel = _find_cached_wheel_for_current_env()
    if cached_wheel is not None:
        print(f"Installing PyRosetta from cached wheel: {cached_wheel}")
        _pip_install(str(cached_wheel))
        ok = _try_import_pyrosetta()
        if not ok:
            raise SystemExit("PyRosetta cached-wheel install finished but import still fails.")
        return 0

    # Default: install latest compatible wheel using the official installer.
    try:
        import pyrosetta_installer  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "pyrosetta-installer not installed in this env. "
            "Install it first: `python -m pip install pyrosetta-installer`."
        ) from exc

    # NOTE: pyrosetta_installer always installs the latest compatible wheel for the current
    # Python version/OS from the official archive.
    pyrosetta_installer.install_pyrosetta(silent=False, skip_if_installed=(not args.no_skip))

    ok = _try_import_pyrosetta()
    if not ok:
        raise SystemExit("PyRosetta install finished but import still fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
