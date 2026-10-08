from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "installation" / "install_pyrosetta.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gh_install_pyrosetta", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_prefix_is_derived_from_micromamba_python_path(tmp_path: Path) -> None:
    mod = _load_module()

    python_bin = tmp_path / ".gh_envs" / "envs" / "gh" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")

    assert mod._derive_root_prefix_from_python_bin(python_bin) == tmp_path / ".gh_envs"


def test_find_cached_wheel_prefers_current_python_abi(tmp_path: Path) -> None:
    mod = _load_module()

    download_dir = tmp_path / "downloads" / "pyrosetta"
    download_dir.mkdir(parents=True)
    good = download_dir / "pyrosetta-2026.6+release.e5a76a2dbd-cp311-cp311-linux_x86_64.whl"
    bad = download_dir / "pyrosetta-2026.6+release.e5a76a2dbd-cp310-cp310-linux_x86_64.whl"
    bad.write_text("", encoding="utf-8")
    good.write_text("", encoding="utf-8")

    assert mod._find_compatible_cached_wheel(download_dir, py_version=(3, 11)) == good


def test_find_cached_wheel_ignores_non_matching_files(tmp_path: Path) -> None:
    mod = _load_module()

    download_dir = tmp_path / "downloads" / "pyrosetta"
    download_dir.mkdir(parents=True)
    (download_dir / "README.txt").write_text("no wheel", encoding="utf-8")

    assert mod._find_compatible_cached_wheel(download_dir, py_version=(3, 11)) is None
