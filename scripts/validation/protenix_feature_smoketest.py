#!/usr/bin/env python3
"""Smoke test: Protenix feature builder w/ precomputed MSA dir.

Goal: catch import-time compat issues (Biotite/Protenix) and verify we can
construct MSA features without running model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from astrolabe.evaluators.wrappers.protenix import protenix_from_dict_to_feature

    msa_dir = Path("/tmp/gh_protenix_msa_test/1")
    msa_dir.mkdir(parents=True, exist_ok=True)
    seq = "ACDEFGHIKL"
    (msa_dir / "non_pairing.a3m").write_text(f">q\n{seq}\n", encoding="utf-8")
    (msa_dir / "pairing.a3m").write_text(f">q\n{seq}\n", encoding="utf-8")

    sample = {
        "name": "toy",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                    "unpairedMsaPath": str(msa_dir / "non_pairing.a3m"),
                    "pairedMsaPath": str(msa_dir / "pairing.a3m"),
                }
            }
        ],
        "modelSeeds": [],
    }

    data, _atom_array, _time_tracker = protenix_from_dict_to_feature(sample)
    feat = data["input_feature_dict"]

    assert int(data["N_token"].item()) == len(seq)
    assert int(data["N_msa"].item()) >= 1
    assert tuple(feat["msa"].shape) == (int(data["N_msa"].item()), len(seq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
