#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

# Allow running without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _make_sequence(length: int, *, seed: int, mode: str) -> str:
    if length <= 0:
        raise ValueError("length must be > 0")
    if mode == "polyA":
        return "A" * length
    if mode == "random":
        rng = random.Random(seed)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        return "".join(rng.choice(alphabet) for _ in range(length))
    raise ValueError(f"Unknown --sequence-mode {mode!r}")


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * (part / total)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile AF3 featurisation (no inference) and print per-step timings."
    )
    p.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[500, 1000],
        help="Protein lengths to profile (default: 500 1000).",
    )
    p.add_argument("--runs", type=int, default=3, help="Timed runs to average.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs to discard.")
    p.add_argument(
        "--sequence-mode",
        choices=("random", "polyA"),
        default="random",
        help="Sequence generator (default: random).",
    )
    p.add_argument(
        "--sequence-seed",
        type=int,
        default=0,
        help="RNG seed for random sequence generation (default: 0).",
    )
    p.add_argument(
        "--deterministic-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include deterministic frames path (default: true).",
    )
    p.add_argument(
        "--msa-crop-size",
        type=int,
        default=16384,
        help="MSA crop size used in padding shapes (default: 16384).",
    )
    p.add_argument(
        "--max-templates",
        type=int,
        default=4,
        help="Max templates (default: 4; set 0 to disable templates).",
    )
    p.add_argument(
        "--pad-num-chains",
        type=int,
        default=1000,
        help="Pad num_chains feature dimension (default: 1000).",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full results as JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from astrolabe.utils.af3_featurization_profile import profile_featurization_steps

    all_results = {}
    for length in args.lengths:
        seq = _make_sequence(length, seed=args.sequence_seed, mode=args.sequence_mode)
        result = profile_featurization_steps(
            sequence=seq,
            name=f"bench_{length}",
            deterministic_frames=args.deterministic_frames,
            msa_crop_size=args.msa_crop_size,
            max_templates=args.max_templates,
            pad_num_chains=args.pad_num_chains,
            runs=args.runs,
            warmup=args.warmup,
        )
        all_results[str(length)] = result

        total = result["total_s"]
        print(f"\n=== AF3 featurisation profile: {length} aa ===")
        print(
            f"tokens={result['num_tokens']} bucket={result['bucket_tokens']} "
            f"runs={result['runs']} warmup={result['warmup']} total={total:.3f}s"
        )
        items = sorted(result["times_s"].items(), key=lambda kv: kv[1], reverse=True)
        for name, sec in items:
            print(f"{name:32s} {sec:8.3f}s  ({_pct(sec, total):5.1f}%)")
        if items:
            top_name, top_sec = items[0]
            print(f"TOP: {top_name} {top_sec:.3f}s ({_pct(top_sec, total):.1f}%)")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()

