#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return proc.returncode, (proc.stdout or "").strip()


def _git_dirty(repo_root: Path) -> tuple[bool, str]:
    rc, out = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if rc != 0:
        return True, "failed to run `git status --porcelain`"
    return bool(out), out


def _count_transients(repo_root: Path) -> tuple[int, int]:
    pycache = sum(1 for _ in repo_root.rglob("__pycache__"))
    pyc = sum(1 for _ in repo_root.rglob("*.pyc"))
    return pycache, pyc


def _normalize_third_party_manifest(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("- generated_at_utc:"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _third_party_status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 6:
            continue
        status = cols[5].strip(" `")
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def _check_third_party_manifest_fresh(repo_root: Path, target: Path) -> tuple[bool, str]:
    rel_tmp = Path(".release_audit.third_party.tmp.md")
    tmp_path = repo_root / rel_tmp
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        rc, _ = _run(
            [
                sys.executable,
                "scripts/release/generate_third_party_licenses.py",
                "--out",
                str(rel_tmp),
            ],
            cwd=repo_root,
        )
        if rc != 0:
            return False, "failed to regenerate THIRD_PARTY_LICENSES.md for freshness check"
        if not target.exists() or not tmp_path.exists():
            return False, "THIRD_PARTY_LICENSES.md missing after regeneration check"

        current = _normalize_third_party_manifest(target.read_text(encoding="utf-8"))
        regenerated = _normalize_third_party_manifest(tmp_path.read_text(encoding="utf-8"))
        return current == regenerated, "THIRD_PARTY_LICENSES.md is stale; regenerate with just third-party-licenses"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release readiness audit for repo cleanliness/compliance.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Do not fail on dirty git tree (still reported).",
    )
    parser.add_argument(
        "--allow-stale-third-party-licenses",
        action="store_true",
        help="Do not fail when THIRD_PARTY_LICENSES.md does not match regenerated inventory.",
    )
    parser.add_argument(
        "--allow-license-review-needed",
        action="store_true",
        help="Do not fail when THIRD_PARTY_LICENSES.md contains review-needed vendor statuses.",
    )
    ns = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    failed = False

    dirty, porcelain = _git_dirty(repo_root)
    if dirty:
        print("FAIL: git tree is dirty.")
        if porcelain:
            preview = porcelain.splitlines()[:20]
            for line in preview:
                print(f"  {line}")
            if len(porcelain.splitlines()) > 20:
                print("  ...")
        if not ns.allow_dirty:
            failed = True
    else:
        print("OK: git tree is clean.")

    pycache_count, pyc_count = _count_transients(repo_root)
    if pycache_count or pyc_count:
        print(f"FAIL: transient artifacts found (__pycache__={pycache_count}, *.pyc={pyc_count}).")
        failed = True
    else:
        print("OK: no transient Python cache artifacts.")

    third_party = repo_root / "THIRD_PARTY_LICENSES.md"
    if third_party.is_file():
        print("OK: THIRD_PARTY_LICENSES.md exists.")
        fresh, freshness_msg = _check_third_party_manifest_fresh(repo_root, third_party)
        if fresh:
            print("OK: THIRD_PARTY_LICENSES.md matches regenerated inventory.")
        else:
            print(f"FAIL: {freshness_msg}")
            if not ns.allow_stale_third_party_licenses:
                failed = True

        status_counts = _third_party_status_counts(third_party)
        review_needed = int(status_counts.get("license_file_not_found_top_level", 0))
        if review_needed > 0:
            print(
                "FAIL: THIRD_PARTY_LICENSES.md contains review-needed components "
                f"(license_file_not_found_top_level={review_needed})."
            )
            if not ns.allow_license_review_needed:
                failed = True
        else:
            print("OK: no review-needed vendor license statuses.")
    else:
        print("FAIL: THIRD_PARTY_LICENSES.md missing.")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
