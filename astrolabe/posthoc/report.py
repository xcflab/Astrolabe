from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from astrolabe.posthoc.db import iter_records, open_db
from astrolabe.posthoc.filtering import (
    _coerce_weighted_score,
    _logical_run_dir,
    _logical_run_sort_key,
    _prefer_representative_row,
    apply_filter_expression,
    apply_preset,
    build_filter_context,
    compile_filter_expression,
    load_metrics_json,
    load_preset,
)


def _is_valid_float(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        x = float(value)
    except Exception:
        return False
    return x == x and x not in {float("inf"), float("-inf")}


def _default_trend_metrics() -> list[str]:
    return ["weighted_score", "binder_ptm", "binder_iptm", "iptm", "ipsae"]


def _aggregate_scope_rows(
    rows: Iterable[dict[str, Any]],
    *,
    trend_metrics: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(rows)
    total = len(rows)
    passed = sum(1 for item in rows if bool(item["passed"]))
    by_step: dict[int, list[dict[str, Any]]] = {}
    for item in rows:
        by_step.setdefault(int(item["step"]), []).append(item)

    per_step_rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        items = by_step[step]
        row: dict[str, Any] = {
            "step": int(step),
            "records": len(items),
            "passed": sum(1 for item in items if bool(item["passed"])),
        }
        row["success_rate"] = (
            float(row["passed"]) / float(row["records"]) if row["records"] else 0.0
        )
        for metric in trend_metrics:
            vals = [
                float(item["context"][metric])
                for item in items
                if metric in item["context"] and _is_valid_float(item["context"][metric])
            ]
            row[f"{metric}_mean"] = sum(vals) / len(vals) if vals else None
        per_step_rows.append(row)

    summary = {
        "total": int(total),
        "passed": int(passed),
        "success_rate": float(passed) / float(total) if total else 0.0,
    }
    return per_step_rows, summary


def _representative_prefix_rows(
    all_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in all_rows:
        key = (
            str(item["logical_run_dir"]),
            str(item["row"]["task"]),
            str(item["row"]["run"]),
        )
        grouped.setdefault(key, []).append(item)

    all_steps = sorted({int(item["step"]) for item in all_rows})
    snapshots_by_run: list[dict[int, dict[str, Any]]] = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: _logical_run_sort_key(item["row"], item["metrics"]),
        )
        snapshots: dict[int, dict[str, Any]] = {}
        current_stage: str | None = None
        best_item: dict[str, Any] | None = None
        best_weighted_score: float | None = None
        for item in items:
            row_stage = str(item["metrics"].get("stage") or "")
            row_weighted_score = _coerce_weighted_score(item["metrics"])
            if current_stage != row_stage:
                current_stage = row_stage
                best_item = item
                best_weighted_score = row_weighted_score
            elif _prefer_representative_row(
                current_row=item["row"],
                current_weighted_score=row_weighted_score,
                best_row=best_item["row"] if best_item is not None else None,
                best_weighted_score=best_weighted_score,
            ):
                best_item = item
                best_weighted_score = row_weighted_score
            snapshots[int(item["step"])] = best_item
        snapshots_by_run.append(snapshots)

    representative_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for step in all_steps:
        for snapshots in snapshots_by_run:
            eligible_steps = [candidate for candidate in snapshots.keys() if candidate <= step]
            if not eligible_steps:
                continue
            representative_rows.append(snapshots[max(eligible_steps)])
    for snapshots in snapshots_by_run:
        if not snapshots:
            continue
        final_rows.append(snapshots[max(snapshots.keys())])
    return representative_rows, final_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_step_trends(
    *,
    path: Path,
    trend_metrics: list[str],
    all_designs_rows: list[dict[str, Any]],
    representative_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(trend_metrics), 1, figsize=(8, 3 * max(1, len(trend_metrics))))
    if len(trend_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, trend_metrics):
        plotted = False
        all_x = [row["step"] for row in all_designs_rows if row.get(f"{metric}_mean") is not None]
        all_y = [row[f"{metric}_mean"] for row in all_designs_rows if row.get(f"{metric}_mean") is not None]
        rep_x = [row["step"] for row in representative_rows if row.get(f"{metric}_mean") is not None]
        rep_y = [row[f"{metric}_mean"] for row in representative_rows if row.get(f"{metric}_mean") is not None]
        if all_x:
            ax.plot(all_x, all_y, marker="o", label="all_designs")
            plotted = True
        if rep_x:
            ax.plot(rep_x, rep_y, marker="s", label="representative_per_run")
            plotted = True
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.set_ylabel("mean")
        if plotted:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_success_rates(
    *,
    path: Path,
    all_designs_rows: list[dict[str, Any]],
    representative_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [row["step"] for row in all_designs_rows],
        [row["success_rate"] for row in all_designs_rows],
        marker="o",
        label="all_designs",
    )
    ax.plot(
        [row["step"] for row in representative_rows],
        [row["success_rate"] for row in representative_rows],
        marker="s",
        label="representative_per_run",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("success_rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_posthoc_report(
    *,
    db_path: Path,
    dst_dir: Path,
    preset_path: Path | None = None,
    filter_expr: str | None = None,
    prefer_oligo: str = "AB",
    trend_metrics: Iterable[str] | None = None,
) -> dict[str, Any]:
    if bool(preset_path) == bool(filter_expr):
        raise ValueError("Exactly one of preset_path or filter_expr must be provided")

    db_path = Path(db_path).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    trend_metrics = [str(metric) for metric in (trend_metrics or _default_trend_metrics()) if str(metric)]

    selector_kind = "preset"
    selector = ""
    preset = None
    compiled_expr = None
    if preset_path is not None:
        preset_path = Path(preset_path).expanduser().resolve()
        preset = load_preset(preset_path)
        selector = preset.name
    else:
        selector_kind = "dsl"
        compiled_expr = compile_filter_expression(str(filter_expr))
        selector = compiled_expr.source

    conn = open_db(db_path)
    all_rows: list[dict[str, Any]] = []
    for row in iter_records(conn, oligo=str(prefer_oligo)):
        metrics = load_metrics_json(row)
        context = build_filter_context(row, metrics)
        if preset is not None:
            res = apply_preset(context, preset)
        else:
            assert compiled_expr is not None
            res = apply_filter_expression(context, compiled_expr)
        all_rows.append(
            {
                "row": row,
                "metrics": metrics,
                "context": context,
                "passed": bool(res.passed),
                "step": int(row["step"]),
                "logical_run_dir": _logical_run_dir(row, metrics),
            }
        )

    rep_prefix_rows, rep_final_rows = _representative_prefix_rows(all_rows)
    all_designs_per_step, all_designs_summary = _aggregate_scope_rows(
        all_rows,
        trend_metrics=trend_metrics,
    )
    rep_per_step, _ = _aggregate_scope_rows(
        rep_prefix_rows,
        trend_metrics=trend_metrics,
    )
    _, rep_summary = _aggregate_scope_rows(
        rep_final_rows,
        trend_metrics=trend_metrics,
    )

    summary_csv = dst_dir / "posthoc_summary.csv"
    per_step_csv = dst_dir / "posthoc_per_step.csv"
    report_md = dst_dir / "posthoc_report.md"
    step_trends_png = dst_dir / "step_trends.png"
    success_rate_png = dst_dir / "success_rate.png"

    _write_csv(
        summary_csv,
        [
            {
                "scope": "all_designs",
                **all_designs_summary,
            },
            {
                "scope": "representative_per_run",
                **rep_summary,
            },
        ],
        fieldnames=["scope", "total", "passed", "success_rate"],
    )

    per_step_rows = []
    metric_headers = [f"{metric}_mean" for metric in trend_metrics]
    for scope, rows in (
        ("all_designs", all_designs_per_step),
        ("representative_per_run", rep_per_step),
    ):
        for row in rows:
            per_step_rows.append({"scope": scope, **row})
    _write_csv(
        per_step_csv,
        per_step_rows,
        fieldnames=["scope", "step", "records", "passed", "success_rate", *metric_headers],
    )

    _plot_step_trends(
        path=step_trends_png,
        trend_metrics=trend_metrics,
        all_designs_rows=all_designs_per_step,
        representative_rows=rep_per_step,
    )
    _plot_success_rates(
        path=success_rate_png,
        all_designs_rows=all_designs_per_step,
        representative_rows=rep_per_step,
    )

    report_md.write_text(
        "\n".join(
            [
                "# Posthoc Report",
                "",
                f"- db: {db_path}",
                f"- selector_kind: {selector_kind}",
                f"- selector: {selector}",
                f"- prefer_oligo: {prefer_oligo}",
                f"- trend_metrics: {', '.join(trend_metrics)}",
                "",
                "## Summary",
                "",
                f"- all_designs: total={all_designs_summary['total']} passed={all_designs_summary['passed']} success_rate={all_designs_summary['success_rate']}",
                f"- representative_per_run: total={rep_summary['total']} passed={rep_summary['passed']} success_rate={rep_summary['success_rate']}",
                "",
                "## Files",
                "",
                f"- summary_csv: {summary_csv}",
                f"- per_step_csv: {per_step_csv}",
                f"- step_trends_png: {step_trends_png}",
                f"- success_rate_png: {success_rate_png}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "db_path": str(db_path),
        "selector_kind": selector_kind,
        "selector": selector,
        "preset": preset.name if preset is not None else "",
        "filter_expr": compiled_expr.source if compiled_expr is not None else "",
        "prefer_oligo": str(prefer_oligo),
        "dst_dir": str(dst_dir),
        "trend_metrics": trend_metrics,
        "summary_csv": str(summary_csv),
        "per_step_csv": str(per_step_csv),
        "report_md": str(report_md),
        "step_trends_png": str(step_trends_png),
        "success_rate_png": str(success_rate_png),
    }
