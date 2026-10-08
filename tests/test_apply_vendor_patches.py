from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "apply_vendor_patches.py"
VENDOR_CASES = {
    "alphafold2": {
        "path": REPO_ROOT / "vendor" / "alphafold2",
        "official_sha": "c77e5d2a8961d1a353632c462914ff0a32a950f6",
        "files": {
            "pyproject.toml": ("tool.setuptools.packages.find", "alphafold*"),
            "alphafold/model/modules.py": ("initial_prev_pos", "cyclic_offset"),
            "alphafold/model/modules_multimer.py": ("initial_prev_pos", "cyclic_offset"),
        },
    },
    "alphafold3": {
        "path": REPO_ROOT / "vendor" / "alphafold3",
        "official_sha": "4b154d922759c918dd4b5a4d5a779853100fd20e",
        "files": {
            "CMakeLists.txt": ("pybind11/include/pybind11/detail/common.h",),
        },
    },
    "chai_lab": {
        "path": REPO_ROOT / "vendor" / "chai_lab",
        "official_sha": "c544fb183e865c4950909444db860a9d50604f66",
        "files": {
            "chai_lab/chai1.py": (
                "cyclic_relative_patches",
                "_apply_cyclic_relative_separation",
            ),
        },
    },
    "boltz": {
        "path": REPO_ROOT / "vendor" / "boltz",
        "official_sha": "cb04aeccdd480fd4db707f0bbafde538397fa2ac",
        "files": {
            "src/boltz/data/types.py": ("cyclic_relative_patches",),
            "src/boltz/model/modules/encoders.py": ("cyclic_relative_patches",),
        },
    },
}


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_test_gh_apply_vendor_patches", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendor_patches_apply_to_official_checkouts(tmp_path: Path) -> None:
    module = _load_script_module()

    for name, case in VENDOR_CASES.items():
        vendor_path = case["path"]
        worktree = tmp_path / name
        subprocess.check_call(
            [
                "git",
                "-C",
                str(vendor_path),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                case["official_sha"],
            ]
        )
        try:
            assert module.main(["--only", name, "--vendor-root", str(worktree), "--check"]) == 0
            assert module.main(["--only", name, "--vendor-root", str(worktree)]) == 0
            assert module.main(["--only", name, "--vendor-root", str(worktree), "--check"]) == 0
            for rel_path, needles in case["files"].items():
                text = (worktree / rel_path).read_text(encoding="utf-8")
                for needle in needles:
                    assert needle in text
            if name == "alphafold2":
                site_dir = tmp_path / "af2_site"
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--no-deps",
                        "--no-build-isolation",
                        "--target",
                        str(site_dir),
                        str(worktree),
                    ]
                )
                env = dict(os.environ)
                env["PYTHONPATH"] = str(site_dir)
                subprocess.check_call(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib.util; "
                            "assert importlib.util.find_spec('alphafold') is not None; "
                            "assert importlib.util.find_spec('alphafold.model') is not None"
                        ),
                    ],
                    env=env,
                )
        finally:
            subprocess.call(
                ["git", "-C", str(vendor_path), "worktree", "remove", "--force", str(worktree)]
            )
            subprocess.call(["git", "-C", str(vendor_path), "worktree", "prune"])


def test_patch_managed_submodules_are_ignored_as_dirty_in_parent_status() -> None:
    for name in VENDOR_CASES:
        rel_path = VENDOR_CASES[name]["path"].relative_to(REPO_ROOT).as_posix()
        value = subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                str(REPO_ROOT / ".gitmodules"),
                "--get",
                f"submodule.{rel_path}.ignore",
            ],
            text=True,
        ).strip()
        assert value == "dirty"


def test_materialize_fork_worktree_creates_patch_commit(tmp_path: Path) -> None:
    module = _load_script_module()
    vendor_repo = tmp_path / "vendor_repo"
    vendor_repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=vendor_repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=vendor_repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.invalid"], cwd=vendor_repo)
    package_dir = vendor_repo / "pkg"
    package_dir.mkdir()
    source_file = package_dir / "module.py"
    source_file.write_text("VALUE = 'official'\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "."], cwd=vendor_repo)
    subprocess.check_call(["git", "commit", "-m", "official"], cwd=vendor_repo)

    source_file.write_text("VALUE = 'patched'\n", encoding="utf-8")
    patch_path = tmp_path / "demo.patch"
    patch_text = subprocess.check_output(["git", "diff"], cwd=vendor_repo, text=True)
    patch_path.write_text(patch_text, encoding="utf-8")
    subprocess.check_call(["git", "checkout", "--", "pkg/module.py"], cwd=vendor_repo)

    module.VENDOR_PATCHES = {
        "demo": {
            "default_root": vendor_repo,
            "patch": patch_path,
            "sentinels": {"pkg/module.py": ("patched",)},
        }
    }
    out_dir = tmp_path / "fork_worktree"

    assert module.main(["--only", "demo", "--materialize-fork-worktree", str(out_dir)]) == 0

    assert (out_dir / "pkg" / "module.py").read_text(encoding="utf-8") == "VALUE = 'patched'\n"
    assert (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=out_dir, text=True)
        == ""
    )
    assert (
        subprocess.check_output(["git", "log", "-1", "--format=%s"], cwd=out_dir, text=True).strip()
        == "Apply GH demo vendor patch"
    )


def test_materialize_fork_worktree_rejects_uninitialized_vendor_dir(tmp_path: Path) -> None:
    module = _load_script_module()
    parent_repo = tmp_path / "parent"
    parent_repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=parent_repo)
    vendor_dir = parent_repo / "vendor" / "demo"
    vendor_dir.mkdir(parents=True)
    patch_path = tmp_path / "demo.patch"
    patch_path.write_text("", encoding="utf-8")
    module.VENDOR_PATCHES = {
        "demo": {
            "default_root": vendor_dir,
            "patch": patch_path,
            "sentinels": {"pkg/module.py": ("patched",)},
        }
    }
    out_dir = tmp_path / "fork_worktree"

    try:
        module.main(["--only", "demo", "--materialize-fork-worktree", str(out_dir)])
    except SystemExit as exc:
        text = str(exc)
        assert "Vendor checkout path is not a git repository root" in text
        assert str(parent_repo.resolve()) in text
    else:
        raise AssertionError("uninitialized vendor directory should fail before worktree add")
    assert not out_dir.exists()
