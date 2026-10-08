#!/usr/bin/env python3
"""Smoke test: convert A3M -> Chai `.aligned.pqt` (no inference)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from astrolabe.evaluators.chai1 import _maybe_write_aligned_pqt

    out_dir = Path("/tmp/gh_chai_msa_smoke/msas")
    out_dir.mkdir(parents=True, exist_ok=True)

    query = "ACDEFGHIKL"
    a3m = f">query\n{query}\n>hit\nACD--GHIKL\n"

    _maybe_write_aligned_pqt(
        msas_dir=out_dir,
        query_sequence=query,
        unpaired_a3m=a3m,
        paired_a3m=None,
        max_unpaired_rows=None,
        max_paired_rows=None,
    )

    # Ensure a single parquet file is written.
    files = list(out_dir.glob("*.aligned.pqt"))
    assert len(files) == 1, files
    assert files[0].stat().st_size > 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
