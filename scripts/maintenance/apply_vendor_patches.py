#!/usr/bin/env python3
"""Apply repo-owned patches to patch-managed vendored checkouts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATCHES = {
    "alphafold2": {
        "default_root": REPO_ROOT / "vendor" / "alphafold2",
        "patch": REPO_ROOT / "patches" / "vendor" / "alphafold2" / "gh-af2-cyclic-initial-recycle.patch",
        "sentinels": {
            "pyproject.toml": ("tool.setuptools.packages.find", "alphafold*"),
            "alphafold/model/model.py": ("_compute_chain_pair_iptm", "_compute_chain_ptm"),
            "alphafold/model/modules.py": ("initial_prev_pos", "cyclic_offset"),
            "alphafold/model/modules_multimer.py": ("initial_prev_pos", "cyclic_offset"),
        },
    },
    "alphafold3": {
        "default_root": REPO_ROOT / "vendor" / "alphafold3",
        "patch": REPO_ROOT / "patches" / "vendor" / "alphafold3" / "gh-af3-complete-local-third-party.patch",
        "sentinels": {
            "CMakeLists.txt": (
                "abseil-cpp-source/CMakeLists.txt",
                "pybind11-source/include/pybind11/detail/common.h",
                "pybind11-abseil-source/CMakeLists.txt",
                "libcifpp-source/CMakeLists.txt",
                "dssp-source/CMakeLists.txt",
                "NOT DEFINED CIFPP_DOWNLOAD_CCD",
            ),
        },
    },
    "chai_lab": {
        "default_root": REPO_ROOT / "vendor" / "chai_lab",
        "patch": REPO_ROOT / "patches" / "vendor" / "chai_lab" / "gh-chai-cyclic-relative-separation.patch",
        "sentinels": {
            "chai_lab/chai1.py": ("cyclic_relative_patches", "_apply_cyclic_relative_separation"),
        },
    },
    "boltz": {
        "default_root": REPO_ROOT / "vendor" / "boltz",
        "patch": REPO_ROOT / "patches" / "vendor" / "boltz" / "gh-boltz-cyclic-relative-patches.patch",
        "sentinels": {
            "src/boltz/data/types.py": ("cyclic_relative_patches",),
            "src/boltz/model/modules/encoders.py": ("cyclic_relative_patches",),
        },
    },
    "protenix": {
        "default_root": REPO_ROOT / "vendor" / "protenix",
        "patch": REPO_ROOT / "patches" / "vendor" / "protenix" / "gh-protenix-use-vendored-esm2.patch",
        "sentinels": {
            "protenix/data/esm/compute_esm.py": (
                "from esm2 import FastaBatchedDataset, pretrained",
            ),
        },
    },
    "esm2": {
        "default_root": REPO_ROOT / "vendor" / "esm2",
        "patch": REPO_ROOT / "patches" / "vendor" / "esm2" / "gh-esm2-biotite-1.5.patch",
        "sentinels": {
            "esm2/inverse_folding/util.py": (
                "from biotite.structure import filter_peptide_backbone as filter_backbone",
            ),
        },
    },
}


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def _git_output(cmd: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def _vendor_root_hint(vendor_root: Path) -> str:
    try:
        rel_path = vendor_root.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "Pass an initialized git checkout as --vendor-root."
    return f"Initialize it with `git submodule update --init --recursive {rel_path}`."


def _ensure_git_checkout(vendor_root: Path) -> None:
    try:
        top_level = Path(
            _git_output(["git", "rev-parse", "--show-toplevel"], cwd=vendor_root)
        ).resolve()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Vendor checkout is not an initialized git repository: {vendor_root}. "
            f"{_vendor_root_hint(vendor_root)}"
        ) from exc
    if top_level != vendor_root:
        raise SystemExit(
            f"Vendor checkout path is not a git repository root: {vendor_root}; "
            f"git resolves it to {top_level}. {_vendor_root_hint(vendor_root)}"
        )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing expected vendor file: {path}") from exc


def _all_needles_present(path: Path, needles: Iterable[str]) -> bool:
    text = _read_text(path)
    return all(needle in text for needle in needles)


def _apply_patch_once(
    vendor_root: Path,
    patch_path: Path,
    sentinels: dict[str, Iterable[str]],
    *,
    check_only: bool,
) -> bool:
    already_patched = all(
        _all_needles_present(vendor_root / rel_path, needles)
        for rel_path, needles in sentinels.items()
    )
    if not patch_path.exists():
        raise SystemExit(f"Missing patch file: {patch_path}")
    if already_patched:
        print(f"already patched: {vendor_root}")
        if check_only:
            _run(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=vendor_root)
        return False
    _run(["git", "apply", "--check", str(patch_path)], cwd=vendor_root)
    if not check_only:
        _run(["git", "apply", str(patch_path)], cwd=vendor_root)
    return True


def _materialize_fork_worktree(
    *,
    name: str,
    vendor_root: Path,
    patch_path: Path,
    sentinels: dict[str, Iterable[str]],
    out_dir: Path,
) -> None:
    if out_dir.exists():
        raise SystemExit(f"Output worktree already exists: {out_dir}")
    _run(["git", "worktree", "add", "--detach", str(out_dir), "HEAD"], cwd=vendor_root)
    _apply_patch_once(
        out_dir,
        patch_path,
        sentinels,
        check_only=False,
    )
    _run(["git", "add", "-A"], cwd=out_dir)
    if not _git_output(["git", "status", "--porcelain"], cwd=out_dir):
        print(f"fork worktree already contains {name} patch: {out_dir}")
        return
    _run(
        [
            "git",
            "-c",
            "user.name=Guided Hallucination",
            "-c",
            "user.email=guided-astrolabe@example.invalid",
            "commit",
            "-m",
            f"Apply GH {name} vendor patch",
        ],
        cwd=out_dir,
    )
    commit = _git_output(["git", "rev-parse", "--short", "HEAD"], cwd=out_dir)
    print(f"fork worktree ready: {out_dir} @ {commit}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=tuple(VENDOR_PATCHES),
        action="append",
        default=None,
        help="Vendor patch group to apply. Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--vendor-root",
        default=None,
        help="Path to the vendored checkout. Only valid with exactly one --only.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the patch applies, but do not modify files.",
    )
    parser.add_argument(
        "--materialize-fork-worktree",
        default=None,
        help=(
            "Create a detached, patch-applied commit worktree for one --only "
            "component. Use the result as the source to push into a reviewed fork."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = tuple(args.only or VENDOR_PATCHES)
    if args.vendor_root and len(names) != 1:
        raise SystemExit("--vendor-root requires exactly one --only value.")
    if args.materialize_fork_worktree and len(names) != 1:
        raise SystemExit("--materialize-fork-worktree requires exactly one --only value.")
    if args.materialize_fork_worktree and args.check:
        raise SystemExit("--materialize-fork-worktree cannot be combined with --check.")
    for name in names:
        spec = VENDOR_PATCHES[name]
        vendor_root = (
            Path(args.vendor_root).expanduser().resolve()
            if args.vendor_root
            else Path(spec["default_root"]).resolve()
        )
        if not vendor_root.exists():
            raise SystemExit(f"Vendor checkout does not exist for {name}: {vendor_root}")
        _ensure_git_checkout(vendor_root)
        if args.materialize_fork_worktree:
            _materialize_fork_worktree(
                name=name,
                vendor_root=vendor_root,
                patch_path=Path(spec["patch"]),
                sentinels=spec["sentinels"],
                out_dir=Path(args.materialize_fork_worktree).expanduser().resolve(),
            )
            continue
        _apply_patch_once(
            vendor_root,
            Path(spec["patch"]),
            spec["sentinels"],
            check_only=bool(args.check),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
