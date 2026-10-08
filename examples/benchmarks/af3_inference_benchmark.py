#!/usr/bin/env python3
"""Benchmark AlphaFold3 end-to-end (featurisation + inference).

Goal: quantify how much the RefStructure CPU parallelism + cache helps overall runtime,
and verify that enabling them does not change prediction outputs (same seeds).

This intentionally uses synthetic protein-only inputs (2 chains) to:
- match the "mutate sequence only" use case
- keep the setup lightweight (empty MSAs/templates)
- ensure ipTM is defined (single-chain ipTM can be NaN)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _ensure_af3_on_sys_path() -> None:
    """Ensure vendored AlphaFold3 sources are importable without external installs."""
    try:
        import alphafold3  # noqa: F401

        return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    af3_src = repo_root / "vendor" / "alphafold3" / "src"
    if af3_src.exists():
        af3_src_str = str(af3_src)
        if af3_src_str not in sys.path:
            sys.path.insert(0, af3_src_str)

    import alphafold3  # noqa: F401  # Retry for clearer error.


def _load_default_model_dir() -> str | None:
    try:
        from omegaconf import OmegaConf

        repo_root = Path(__file__).resolve().parents[2]
        env_cfg = OmegaConf.load(repo_root / "config" / "env.yaml")
        model_dir = env_cfg.get("weight_path", {}).get("alphafold3", None)
        return str(model_dir) if model_dir else None
    except Exception:
        return None


def _set_refstructure_env(
    *,
    workers: int,
    cache: bool,
    cache_max_entries: int,
) -> None:
    os.environ["ALPHAFOLD3_REFSTRUCTURE_WORKERS"] = str(int(workers))
    os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE"] = "1" if cache else "0"
    os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES"] = str(int(cache_max_entries))


def _clear_refstructure_cache() -> None:
    # Cache is an implementation detail; clear it if present to avoid interference.
    from alphafold3.model import features as af3_features

    cache = getattr(af3_features, "_REFSTRUCTURE_REFERENCE_CACHE", None)
    if cache is not None:
        cache.clear()


def _make_sequence(length: int, *, seed: int, mode: str) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    if mode == "repeat":
        return "A" * length
    if mode == "random":
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(length))
    raise ValueError(f"Unknown sequence mode: {mode!r}")


def _mutate_single_residue(seq: str, *, pos: int, to_aa: str) -> str:
    if not (0 <= pos < len(seq)):
        raise ValueError(f"Mutation position out of range: pos={pos} len={len(seq)}")
    if len(to_aa) != 1:
        raise ValueError(f"Invalid target AA: {to_aa!r}")
    if seq[pos] == to_aa:
        raise ValueError("Mutation would not change the residue.")
    return seq[:pos] + to_aa + seq[pos + 1 :]


def _make_fold_input(*, name: str, chain_seqs: Sequence[str], rng_seed: int):
    from alphafold3.common.folding_input import Input, ProteinChain
    from alphafold3.constants import mmcif_names
    from alphafold3.data import msa as msa_module

    chains = []
    for idx, seq in enumerate(chain_seqs):
        chain_id = chr(ord("A") + idx)
        unpaired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        paired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        chains.append(
            ProteinChain(
                id=chain_id,
                sequence=seq,
                ptms=[],
                unpaired_msa=unpaired,
                paired_msa=paired,
                templates=[],
            )
        )

    return Input(
        name=name,
        chains=chains,
        rng_seeds=[int(rng_seed)],
        bonded_atom_pairs=None,
        user_ccd=None,
    )


@dataclass(frozen=True)
class RunResult:
    timings_s: dict[str, float]
    scalars: dict[str, float]


def _extract_scalar_metrics(all_inference_results) -> dict[str, float]:
    import numpy as np

    from astrolabe.evaluators.wrappers.alphafold3 import get_af3_scores

    scores = get_af3_scores(all_inference_results)
    ranking_scores = np.asarray(scores["per_str_scores"]["ranking_score"], dtype=np.float64)
    best_idx = int(np.nanargmax(ranking_scores))

    ptm = float(scores["per_str_scores"]["ptm"][best_idx])
    iptm = float(scores["per_str_scores"]["iptm"][best_idx])
    ranking_score = float(scores["per_str_scores"]["ranking_score"][best_idx])
    per_res_plddt = np.asarray(
        scores["per_res_scores"]["per_res_plddt"][best_idx], dtype=np.float64
    )
    mean_plddt = float(np.mean(per_res_plddt))
    return {
        "best_sample_idx": float(best_idx),
        "ranking_score": ranking_score,
        "ptm": ptm,
        "iptm": iptm,
        "mean_plddt": mean_plddt,
    }


def _run_once(
    *,
    fold_input,
    model_runner,
    buckets: Sequence[int] | None,
    ref_max_modified_date=None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
) -> RunResult:
    import jax

    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from astrolabe.evaluators.wrappers.alphafold3 import ResultsForSeed

    timings_s: dict[str, float] = {}

    t0 = time.perf_counter()
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    timings_s["ccd_init_s"] = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=False,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
    )
    timings_s["featurise_s"] = float(time.perf_counter() - t1)

    t2 = time.perf_counter()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        rng_key = jax.random.PRNGKey(int(seed))
        result = model_runner.run_inference(example, rng_key)
        inference_results = model_runner.extract_structures(
            batch=example, result=result, target_name=fold_input.name
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=int(seed),
                inference_results=inference_results,
                full_fold_input=fold_input,
            )
        )
    timings_s["inference_s"] = float(time.perf_counter() - t2)
    timings_s["total_s"] = float(sum(timings_s.values()))

    scalars = _extract_scalar_metrics(all_inference_results)
    return RunResult(timings_s=timings_s, scalars=scalars)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _diff(a: float, b: float) -> float:
    import math

    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    return abs(a - b)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AF3 end-to-end inference (featurisation + inference) and "
            "compare baseline vs ref_structure parallel+cache."
        )
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--chain-count", type=int, default=2)
    parser.add_argument("--sequence-mode", choices=["random", "repeat"], default="random")
    parser.add_argument("--sequence-seed", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=0, help="AF3 rng seed (fold_input.rng_seeds).")

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--jax-cache-dir", type=str, default="~/.cache/jax_cache")
    parser.add_argument("--flash-attention", type=str, default="cudnn")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-recycles", type=int, default=1)

    parser.add_argument("--baseline-workers", type=int, default=1)
    parser.add_argument("--opt-workers", type=int, default=8)
    parser.add_argument(
        "--opt-cache",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable RefStructure in-process cache for the optimised run (0/1).",
    )
    parser.add_argument("--opt-cache-max-entries", type=int, default=20000)

    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.chain_count < 1:
        raise SystemExit("--chain-count must be >= 1")

    _ensure_af3_on_sys_path()

    from omegaconf import OmegaConf

    from alphafold3.model.pipeline import pipeline as pipeline_lib
    from astrolabe.evaluators.wrappers.alphafold3 import BUCKETS, alphafold3_setup

    model_dir = args.model_dir or _load_default_model_dir()
    if not model_dir:
        raise SystemExit(
            "Missing --model-dir and could not infer from config/env.yaml "
            "(weight_path.alphafold3)."
        )

    setup_cfg = OmegaConf.create(
        {
            "jax_compilation_cache_dir": args.jax_cache_dir,
            "flash_attention_implementation": args.flash_attention,
            "num_samples": int(args.num_samples),
            "num_recycles": int(args.num_recycles),
            "save_embeddings": False,
            "model_dir": str(model_dir),
        }
    )
    model_runner = alphafold3_setup(setup_cfg)

    print("=== AF3 inference benchmark ===")
    print(f"model_dir={model_dir}")
    print(
        "model_config: "
        f"flash_attention={args.flash_attention} num_recycles={args.num_recycles} num_samples={args.num_samples}"
    )
    print(
        "ref_structure: "
        f"baseline(workers={args.baseline_workers}, cache=0) "
        f"opt(workers={args.opt_workers}, cache={int(bool(args.opt_cache))})"
    )

    for length in args.lengths:
        total_len = int(length)
        chain_lens = [total_len // args.chain_count] * args.chain_count
        for i in range(total_len % args.chain_count):
            chain_lens[i] += 1

        chain_seqs = [
            _make_sequence(n, seed=args.sequence_seed + i, mode=args.sequence_mode)
            for i, n in enumerate(chain_lens)
        ]
        fold_input = _make_fold_input(
            name=f"bench_{total_len}",
            chain_seqs=chain_seqs,
            rng_seed=args.rng_seed,
        )
        num_tokens = sum(len(s) for s in chain_seqs)
        bucket_tokens = pipeline_lib.calculate_bucket_size(num_tokens, BUCKETS)

        print(f"\n--- length={total_len}aa chains={args.chain_count} tokens={num_tokens} bucket={bucket_tokens} ---")

        # Warmup: compile for this bucket (not part of timings below).
        _set_refstructure_env(
            workers=args.baseline_workers,
            cache=False,
            cache_max_entries=args.opt_cache_max_entries,
        )
        _clear_refstructure_cache()
        print("warmup: compiling/jitting...")
        _run_once(fold_input=fold_input, model_runner=model_runner, buckets=BUCKETS)

        # Baseline: workers=1, cache=0
        _set_refstructure_env(
            workers=args.baseline_workers,
            cache=False,
            cache_max_entries=args.opt_cache_max_entries,
        )
        _clear_refstructure_cache()
        baseline = _run_once(fold_input=fold_input, model_runner=model_runner, buckets=BUCKETS)

        # Optimised: workers=N, cache on/off (first run: cold cache)
        _set_refstructure_env(
            workers=args.opt_workers,
            cache=bool(args.opt_cache),
            cache_max_entries=args.opt_cache_max_entries,
        )
        _clear_refstructure_cache()
        opt_base = _run_once(fold_input=fold_input, model_runner=model_runner, buckets=BUCKETS)

        # Optimised (mutant): same length/seed; cache should hit except mutated residues.
        mutant_chain_seqs = list(chain_seqs)
        if mutant_chain_seqs:
            mid = len(mutant_chain_seqs[0]) // 2
            current = mutant_chain_seqs[0][mid]
            for to_aa in ("C", "A", "W"):
                if to_aa != current:
                    mutant_chain_seqs[0] = _mutate_single_residue(
                        mutant_chain_seqs[0], pos=mid, to_aa=to_aa
                    )
                    break
        mutant_input = _make_fold_input(
            name=f"bench_{total_len}_mut",
            chain_seqs=mutant_chain_seqs,
            rng_seed=args.rng_seed,
        )
        opt_mutant = _run_once(fold_input=mutant_input, model_runner=model_runner, buckets=BUCKETS)

        def show(label: str, res: RunResult) -> None:
            t = res.timings_s
            s = res.scalars
            print(
                f"{label}: total={_format_seconds(t['total_s'])} "
                f"(feat={_format_seconds(t['featurise_s'])} inf={_format_seconds(t['inference_s'])}) "
                f"ranking={s['ranking_score']:.4f} ptm={s['ptm']:.4f} iptm={s['iptm']:.4f} "
                f"mean_plddt={s['mean_plddt']:.2f}"
            )

        show("baseline", baseline)
        show("opt(base)", opt_base)
        show("opt(mut)", opt_mutant)

        diffs = {
            key: _diff(float(baseline.scalars[key]), float(opt_base.scalars[key]))
            for key in ("ranking_score", "ptm", "iptm", "mean_plddt")
        }
        print(
            "A/B diffs (baseline vs opt(base), best-sample scalars): "
            + ", ".join(f"{k}={v:.3e}" for k, v in diffs.items())
        )

        if any(v > float(args.tolerance) for v in diffs.values()):
            print(f"FAIL: scalar diffs exceed tolerance={args.tolerance:g}")
            return 1

    print("\nOK: baseline vs optimised scalars match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
