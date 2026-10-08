#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

MODALITIES = {
    "peptide_linear": {
        "subdir": "peptide_linear",
        "method": "Linear peptide",
        "kind": "peptide",
        "suffix": "_linpep",
    },
    "peptide_cyc_ht": {
        "subdir": "peptide_cyc_ht",
        "method": "Cyclized (head-tail)",
        "kind": "peptide",
        "suffix": "_cyc_ht",
    },
    "peptide_cyc_ss": {
        "subdir": "peptide_cyc_ss",
        "method": "Cyclized (Cys-Cys)",
        "kind": "peptide",
        "suffix": "_cyc_ss",
    },
    "antibody_init_alla": {
        "subdir": "antibody_init_alla",
        "method": "Antibody (All-A CDR)",
        "kind": "antibody",
        "suffix": "_ab_alla",
    },
    "antibody_init_randomcdr": {
        "subdir": "antibody_init_randomcdr",
        "method": "Antibody (Random CDR)",
        "kind": "antibody",
        "suffix": "_ab_randcdr",
    },
}

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#3C5488"]


def _target_from_task(task_name: str, suffix: str) -> str:
    if task_name.endswith(suffix):
        return task_name[: -len(suffix)]
    return task_name


def _stage_mask(df: pd.DataFrame) -> pd.Series:
    if "stage" not in df.columns:
        return pd.Series(True, index=df.index)
    stage = df["stage"].astype(str).str.lower()
    return stage.str.startswith("defaul")


def _ensure_required_columns(df: pd.DataFrame, required: list[str], file_path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{file_path}: missing required columns {missing}")


def _compute_custom_score(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "peptide":
        _ensure_required_columns(
            df,
            ["iptm", "binder_ptm", "plddt", "interface_sc", "ipae_min"],
            Path("<in-memory>"),
        )
        penalty = 0.5 * (1.0 - (df["ipae_min"] / 10.0)).clip(lower=0.0)
        return df["iptm"] + df["binder_ptm"] + df["plddt"] + 0.2 * df["interface_sc"] + penalty
    if kind == "antibody":
        _ensure_required_columns(df, ["iptm", "ptm", "plddt", "interface_sc"], Path("<in-memory>"))
        return df["iptm"] + df["ptm"] + df["plddt"] + 0.2 * df["interface_sc"]
    raise ValueError(f"Unsupported modality kind: {kind}")


def _load_run_best_rows(input_root: Path, select_by: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for modality_key, cfg in MODALITIES.items():
        run_root = input_root / str(cfg["subdir"])
        score_files = sorted(run_root.glob("*/*/scores.csv"))
        for score_file in score_files:
            run_dir = score_file.parent
            task_name = run_dir.parent.name
            run_name = run_dir.name
            df = pd.read_csv(score_file)
            if df.empty:
                continue
            df = df.loc[_stage_mask(df)].copy()
            if df.empty:
                continue
            _ensure_required_columns(df, ["iptm", "plddt", "interface_sc"], score_file)
            df["custom_score"] = _compute_custom_score(df, str(cfg["kind"]))
            if "ptm" in df.columns:
                df["ptm_like"] = df["ptm"]
            else:
                df["ptm_like"] = df["binder_ptm"]
            df["plddt_pct"] = df["plddt"] * 100.0
            if select_by not in df.columns:
                raise ValueError(f"{score_file}: selection column '{select_by}' not found")
            best_row = df.loc[df[select_by].idxmax()]
            records.append(
                {
                    "modality": modality_key,
                    "method": cfg["method"],
                    "kind": cfg["kind"],
                    "task": task_name,
                    "target": _target_from_task(task_name, str(cfg["suffix"])),
                    "task_label": _target_from_task(task_name, str(cfg["suffix"])).upper(),
                    "run": run_name,
                    "step": int(best_row["step"]),
                    "weighted_score": float(best_row.get("weighted_score", math.nan)),
                    "iptm": float(best_row["iptm"]),
                    "ptm_like": float(best_row["ptm_like"]),
                    "plddt_pct": float(best_row["plddt_pct"]),
                    "interface_sc": float(best_row["interface_sc"]),
                    "custom_score": float(best_row["custom_score"]),
                    "score_file": str(score_file),
                }
            )
    if not records:
        raise ValueError(f"No scores.csv found under {input_root}")
    return pd.DataFrame.from_records(records)


def _task_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["task_label"].dropna().unique().tolist())


def _method_order(df: pd.DataFrame) -> list[str]:
    declared = [str(cfg["method"]) for cfg in MODALITIES.values()]
    present = set(df["method"].dropna().unique().tolist())
    return [method for method in declared if method in present]


def _draw_grouped_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    task_col: str,
    score_col: str,
    method_col: str,
    task_order: list[str],
    method_order: list[str],
) -> None:
    total_width = 0.8
    per_width = total_width / max(1, len(method_order))
    for idx, method in enumerate(method_order):
        method_df = data.loc[data[method_col] == method]
        values: list[list[float]] = []
        positions: list[float] = []
        for task_idx, task in enumerate(task_order):
            vals = method_df.loc[method_df[task_col] == task, score_col].dropna().tolist()
            values.append(vals)
            positions.append(task_idx - total_width / 2 + per_width * (idx + 0.5))
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=per_width * 0.85,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        color = PALETTE[idx % len(PALETTE)]
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.95)
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set(color="black", linewidth=0.8)


def _plot_metric_panels(
    data: pd.DataFrame,
    out_dir: Path,
    *,
    prefix: str,
    methods: list[str],
    tasks: list[str],
) -> None:
    metric_map = {
        "iptm": "iPTM",
        "ptm_like": "pTM",
        "plddt_pct": "pLDDT",
        "interface_sc": "Interface SC",
        "custom_score": "Design Score",
    }

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["lines.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    for metric_key, metric_name in metric_map.items():
        current = data[["task_label", "method", metric_key]].rename(columns={metric_key: "score"}).dropna(subset=["score"])
        if current.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        _draw_grouped_boxplot(
            ax,
            current,
            task_col="task_label",
            score_col="score",
            method_col="method",
            task_order=tasks,
            method_order=methods,
        )
        ax.set_xlabel("")
        ax.set_xticks(list(range(len(tasks))))
        ax.set_xticklabels(tasks, rotation=45, ha="right", color="black")
        ax.set_ylabel(metric_name, fontweight="bold")
        if metric_name == "pLDDT":
            ax.axhline(70, ls="--", c="gray", lw=0.8, alpha=0.6, zorder=0)
            ax.axhline(90, ls="--", c="black", lw=0.8, alpha=0.6, zorder=0)
            ax.set_ylim(40, 100)
        elif metric_name in {"iPTM", "pTM"}:
            ax.axhline(0.8, ls="--", c="#E64B35", lw=0.8, alpha=0.6, zorder=0)
            ax.set_ylim(0, 1.05)
        handles = [Patch(facecolor=PALETTE[i], edgecolor="black", label=method) for i, method in enumerate(methods)]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(3, len(methods)),
            title=None,
            frameon=False,
            fontsize=6,
        )
        fig.tight_layout()
        out_path = out_dir / f"{prefix}_{metric_name.replace(' ', '_')}.svg"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _top_fraction(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"top_fraction must be in (0, 1], got {fraction}")

    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(["task_label", "method"], sort=False):
        keep = max(1, int(math.ceil(len(group) * fraction)))
        parts.append(group.nlargest(keep, "iptm"))
    return pd.concat(parts, ignore_index=True)


def _write_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "selected_best_rows.csv", index=False)

    summary = (
        df.groupby(["task_label", "method"], as_index=False)
        .agg(
            n=("iptm", "size"),
            iptm_mean=("iptm", "mean"),
            iptm_std=("iptm", "std"),
            ptm_mean=("ptm_like", "mean"),
            plddt_mean=("plddt_pct", "mean"),
            interface_sc_mean=("interface_sc", "mean"),
            design_score_mean=("custom_score", "mean"),
        )
        .sort_values(["task_label", "method"])
    )
    summary.to_csv(out_dir / "summary_by_task_method.csv", index=False)

    method_summary = (
        df.groupby("method", as_index=False)
        .agg(
            n=("iptm", "size"),
            iptm_mean=("iptm", "mean"),
            iptm_std=("iptm", "std"),
            ptm_mean=("ptm_like", "mean"),
            plddt_mean=("plddt_pct", "mean"),
            interface_sc_mean=("interface_sc", "mean"),
            design_score_mean=("custom_score", "mean"),
        )
        .sort_values("iptm_mean", ascending=False)
    )
    method_summary.to_csv(out_dir / "summary_by_method.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze peptide/antibody benchmark outputs and draw binder-style boxplots."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("benchmark_outputs/peptide_targets_v1"),
        help="Benchmark root containing modality subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_outputs/peptide_targets_v1/analysis"),
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--select-by",
        default="iptm",
        choices=["iptm", "custom_score", "weighted_score"],
        help="Which per-step metric to use when selecting one row per run.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.5,
        help="Fraction of highest-iPTM rows (per task/method) for top-fraction plots.",
    )
    args = parser.parse_args(argv)

    input_root = args.input_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_rows = _load_run_best_rows(input_root, select_by=args.select_by)
    _write_summary_tables(best_rows, out_dir)

    tasks = _task_order(best_rows)
    methods = _method_order(best_rows)
    _plot_metric_panels(best_rows, fig_dir, prefix="binder_style_all", methods=methods, tasks=tasks)

    top_rows = _top_fraction(best_rows, args.top_fraction)
    top_rows.to_csv(out_dir / "selected_top_fraction_rows.csv", index=False)
    _plot_metric_panels(
        top_rows,
        fig_dir,
        prefix=f"binder_style_top{int(args.top_fraction * 100)}p",
        methods=methods,
        tasks=tasks,
    )

    print(f"input_root: {input_root}")
    print(f"runs: {len(best_rows)}")
    print(f"tasks: {len(tasks)}")
    print(f"methods: {len(methods)}")
    print(f"out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
