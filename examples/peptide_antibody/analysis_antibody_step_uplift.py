#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MODALITIES = {
    "antibody_init_alla": {
        "method": "Antibody (All-A CDR)",
        "suffix": "_ab_alla",
    },
    "antibody_init_randomcdr": {
        "method": "Antibody (Random CDR)",
        "suffix": "_ab_randcdr",
    },
}


@dataclass(frozen=True)
class BootstrapResult:
    diff_mean: float
    ci_low: float
    ci_high: float


def _target_from_task(task_name: str, suffix: str) -> str:
    if task_name.endswith(suffix):
        return task_name[: -len(suffix)]
    return task_name


def _load_run_metrics(root: Path, *, steps_label: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for subdir, cfg in MODALITIES.items():
        run_root = root / subdir
        for score_file in sorted(run_root.glob("*/*/scores.csv")):
            task_name = score_file.parent.parent.name
            run_name = score_file.parent.name
            df = pd.read_csv(score_file)
            if df.empty or "iptm" not in df.columns:
                continue
            if "stage" in df.columns:
                stage = df["stage"].astype(str).str.lower()
                df = df.loc[stage.str.startswith("defaul")].copy()
            if df.empty:
                continue
            df["step"] = df["step"].astype(int)
            best_idx = df["iptm"].idxmax()
            final_idx = df["step"].idxmax()
            best_iptm = float(df.loc[best_idx, "iptm"])
            final_iptm = float(df.loc[final_idx, "iptm"])
            max_step = int(df.loc[final_idx, "step"])
            records.append(
                {
                    "steps_label": steps_label,
                    "modality": subdir,
                    "method": str(cfg["method"]),
                    "task": task_name,
                    "target": _target_from_task(task_name, str(cfg["suffix"])),
                    "target_label": _target_from_task(task_name, str(cfg["suffix"])).upper(),
                    "run": run_name,
                    "run_id": f"{task_name}/{run_name}",
                    "best_iptm": best_iptm,
                    "final_iptm": final_iptm,
                    "max_step": max_step,
                    "score_file": str(score_file),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "steps_label",
                "modality",
                "method",
                "task",
                "target",
                "target_label",
                "run",
                "run_id",
                "best_iptm",
                "final_iptm",
                "max_step",
                "score_file",
            ]
        )
    return pd.DataFrame.from_records(records)


def _bootstrap_diff(
    baseline: np.ndarray,
    new: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    if baseline.size == 0 or new.size == 0:
        return BootstrapResult(diff_mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    diffs = np.empty(n_boot, dtype=float)
    n0 = baseline.size
    n1 = new.size
    for i in range(n_boot):
        b0 = baseline[rng.integers(0, n0, size=n0)]
        b1 = new[rng.integers(0, n1, size=n1)]
        diffs[i] = float(np.mean(b1) - np.mean(b0))
    return BootstrapResult(
        diff_mean=float(np.mean(new) - np.mean(baseline)),
        ci_low=float(np.quantile(diffs, 0.025)),
        ci_high=float(np.quantile(diffs, 0.975)),
    )


def _expected_count_check(df: pd.DataFrame, expected_runs: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method", "target_label", "n_runs"])
    out = (
        df.groupby(["method", "target_label"], as_index=False)
        .agg(n_runs=("run_id", "nunique"))
        .sort_values(["method", "target_label"])
    )
    out["expected_runs"] = expected_runs
    out["ok"] = out["n_runs"] >= expected_runs
    return out


def _iter_groups(df: pd.DataFrame, keys: Iterable[str]):
    grouped = df.groupby(list(keys), as_index=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        yield key, group


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare antibody iPTM uplift between baseline and step-100 runs.")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("benchmark_outputs/peptide_targets_v1"),
        help="Baseline root containing antibody_init_alla and antibody_init_randomcdr (e.g., 30-step runs).",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("benchmark_outputs/peptide_targets_v1_100step"),
        help="New run root containing antibody_init_alla and antibody_init_randomcdr (e.g., 100-step runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_outputs/peptide_targets_v1_100step/analysis"),
        help="Output directory for comparison CSVs.",
    )
    parser.add_argument(
        "--expected-runs-per-task",
        type=int,
        default=60,
        help="Expected minimum run count for each method-target group.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CI.")
    parser.add_argument("--seed", type=int, default=7, help="Bootstrap random seed.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow comparison with incomplete new runs (otherwise requires expected runs in all groups).",
    )
    args = parser.parse_args(argv)

    baseline_root = args.baseline_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = _load_run_metrics(baseline_root, steps_label="baseline")
    new_df = _load_run_metrics(new_root, steps_label="new")

    if base_df.empty:
        raise ValueError(f"No baseline antibody score files found under {baseline_root}")
    if new_df.empty:
        raise ValueError(f"No new antibody score files found under {new_root}")

    count_base = _expected_count_check(base_df, args.expected_runs_per_task)
    count_new = _expected_count_check(new_df, args.expected_runs_per_task)
    count_base.to_csv(out_dir / "run_count_baseline.csv", index=False)
    count_new.to_csv(out_dir / "run_count_new.csv", index=False)

    if not args.allow_partial:
        if not bool(count_new["ok"].all()):
            bad = count_new.loc[~count_new["ok"], ["method", "target_label", "n_runs", "expected_runs"]]
            bad.to_csv(out_dir / "run_count_new_incomplete.csv", index=False)
            raise ValueError(
                "New runs are incomplete for one or more method-target groups. "
                f"See {out_dir / 'run_count_new_incomplete.csv'}"
            )

    merged_target_records: list[dict[str, object]] = []
    for (method, target_label), base_group in _iter_groups(base_df, ["method", "target_label"]):
        new_group = new_df[(new_df["method"] == method) & (new_df["target_label"] == target_label)]
        if new_group.empty:
            continue

        best_ci = _bootstrap_diff(
            baseline=base_group["best_iptm"].to_numpy(dtype=float),
            new=new_group["best_iptm"].to_numpy(dtype=float),
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        final_ci = _bootstrap_diff(
            baseline=base_group["final_iptm"].to_numpy(dtype=float),
            new=new_group["final_iptm"].to_numpy(dtype=float),
            n_boot=args.bootstrap,
            seed=args.seed + 1000,
        )
        merged_target_records.append(
            {
                "method": method,
                "target_label": target_label,
                "n_baseline": int(base_group["run_id"].nunique()),
                "n_new": int(new_group["run_id"].nunique()),
                "baseline_best_mean": float(base_group["best_iptm"].mean()),
                "new_best_mean": float(new_group["best_iptm"].mean()),
                "uplift_best_mean": best_ci.diff_mean,
                "uplift_best_ci_low": best_ci.ci_low,
                "uplift_best_ci_high": best_ci.ci_high,
                "baseline_final_mean": float(base_group["final_iptm"].mean()),
                "new_final_mean": float(new_group["final_iptm"].mean()),
                "uplift_final_mean": final_ci.diff_mean,
                "uplift_final_ci_low": final_ci.ci_low,
                "uplift_final_ci_high": final_ci.ci_high,
            }
        )

    by_target = pd.DataFrame.from_records(merged_target_records).sort_values(["method", "target_label"])
    by_target.to_csv(out_dir / "uplift_by_target.csv", index=False)

    method_records: list[dict[str, object]] = []
    for (method,), base_group in _iter_groups(base_df, ["method"]):
        new_group = new_df[new_df["method"] == method]
        if new_group.empty:
            continue
        best_ci = _bootstrap_diff(
            baseline=base_group["best_iptm"].to_numpy(dtype=float),
            new=new_group["best_iptm"].to_numpy(dtype=float),
            n_boot=args.bootstrap,
            seed=args.seed + 2000,
        )
        final_ci = _bootstrap_diff(
            baseline=base_group["final_iptm"].to_numpy(dtype=float),
            new=new_group["final_iptm"].to_numpy(dtype=float),
            n_boot=args.bootstrap,
            seed=args.seed + 3000,
        )
        method_records.append(
            {
                "method": method,
                "n_baseline": int(base_group["run_id"].nunique()),
                "n_new": int(new_group["run_id"].nunique()),
                "baseline_best_mean": float(base_group["best_iptm"].mean()),
                "new_best_mean": float(new_group["best_iptm"].mean()),
                "uplift_best_mean": best_ci.diff_mean,
                "uplift_best_ci_low": best_ci.ci_low,
                "uplift_best_ci_high": best_ci.ci_high,
                "baseline_final_mean": float(base_group["final_iptm"].mean()),
                "new_final_mean": float(new_group["final_iptm"].mean()),
                "uplift_final_mean": final_ci.diff_mean,
                "uplift_final_ci_low": final_ci.ci_low,
                "uplift_final_ci_high": final_ci.ci_high,
            }
        )

    by_method = pd.DataFrame.from_records(method_records).sort_values("method")
    by_method.to_csv(out_dir / "uplift_by_method.csv", index=False)

    all_best = _bootstrap_diff(
        baseline=base_df["best_iptm"].to_numpy(dtype=float),
        new=new_df["best_iptm"].to_numpy(dtype=float),
        n_boot=args.bootstrap,
        seed=args.seed + 4000,
    )
    all_final = _bootstrap_diff(
        baseline=base_df["final_iptm"].to_numpy(dtype=float),
        new=new_df["final_iptm"].to_numpy(dtype=float),
        n_boot=args.bootstrap,
        seed=args.seed + 5000,
    )
    summary = pd.DataFrame(
        [
            {
                "baseline_root": str(baseline_root),
                "new_root": str(new_root),
                "n_baseline": int(base_df["run_id"].nunique()),
                "n_new": int(new_df["run_id"].nunique()),
                "baseline_best_mean": float(base_df["best_iptm"].mean()),
                "new_best_mean": float(new_df["best_iptm"].mean()),
                "uplift_best_mean": all_best.diff_mean,
                "uplift_best_ci_low": all_best.ci_low,
                "uplift_best_ci_high": all_best.ci_high,
                "baseline_final_mean": float(base_df["final_iptm"].mean()),
                "new_final_mean": float(new_df["final_iptm"].mean()),
                "uplift_final_mean": all_final.diff_mean,
                "uplift_final_ci_low": all_final.ci_low,
                "uplift_final_ci_high": all_final.ci_high,
                "bootstrap": int(args.bootstrap),
                "seed": int(args.seed),
            }
        ]
    )
    summary.to_csv(out_dir / "uplift_summary.csv", index=False)

    print(f"baseline_runs={base_df['run_id'].nunique()} new_runs={new_df['run_id'].nunique()}")
    print(f"out_dir={out_dir}")
    print("Top-line uplift(best-up-to iPTM):")
    print(summary[["baseline_best_mean", "new_best_mean", "uplift_best_mean", "uplift_best_ci_low", "uplift_best_ci_high"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
