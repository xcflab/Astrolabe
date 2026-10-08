#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _find_transients(repo_root: Path) -> tuple[list[Path], list[Path]]:
    pycache_dirs = sorted([p for p in repo_root.rglob("__pycache__") if p.is_dir()])
    pyc_files = sorted([p for p in repo_root.rglob("*.pyc") if p.is_file()])
    return pycache_dirs, pyc_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean transient Python cache artifacts from the repository tree.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be removed.")
    ns = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    pycache_dirs, pyc_files = _find_transients(repo_root)

    print(f"repo_root: {repo_root}")
    print(f"pycache_dirs: {len(pycache_dirs)}")
    print(f"pyc_files: {len(pyc_files)}")

    if ns.dry_run:
        return 0

    for path in pyc_files:
        path.unlink(missing_ok=True)
    for path in pycache_dirs:
        shutil.rmtree(path, ignore_errors=True)

    print("cleaned: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
