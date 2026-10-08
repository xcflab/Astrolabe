"""Frozen-cohort statistics for the Astrolabe manuscript, using CSV and NumPy.

rank_ligands consumes one fixed representative per unit_id/method/group. The
four groups are FAD, IAI, OQO, SAM; columns maps identity and metric names to
exact CSV headers. Missing metric values are omitted only for that metric.
Incomplete four-ligand or seven-metric averages stay undefined, with coverage
reported, rather than silently changing equal-weight sets.

threshold_sensitivity uses the planned/records schema and selection rules of
paper_scores. It fixes representatives/variants before any threshold changes.
The denominator is explicitly 'planned' (all declared units; missing units do
not pass) or 'available' (units with measured selected records). Availability
is fixed across all panels: observed selected records must contain all required
metrics. Missing observations are never inferred from their numerical values.
Ligand intervals resample binary design-unit outcomes within the four fixed
ligands, using the exact binomial representation of this bootstrap. They do
not resample variants or ligand labels. comparison_methods supplies explicit
Astrolabe-AF3 and ProteinHunter method IDs; no label aliases are guessed.

bootstrap_ddg additionally requires condition and molecule_type columns, with
molecule_type explicitly 'minibinder' or 'VHH'. Each row is one fixed paired
design. The signed delta is AC minus AB; missing pairs are reported separately.
The 5,000-resample percentile bootstrap follows the original median bootstrap.

Recovered methods:
* 013_paper_writing/figures/fig4/ligand_benchmark_rank_success_snapshot_20260602/
  code/plot_confidence_boxplots.py::build_rank_tables (SI 210 clarifies medians).
* 013_paper_writing/figures/supplementary_data/08_threshold_sensitivity/code/
  analyze_threshold_sensitivity.py::current_benchmarks and ligand_sensitivity.
* 013_paper_writing/figures/fig3/vhh_rosetta_balance/code/
  01_plot_vhh_rosetta_balance.py::bootstrap_median_ci (MainPaper 163; SI 244).
No original run directory is read or executed by these functions.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from astrolabe.analysis.paper_scores import (
    _FAILED, _KEY, _Metrics, _DEFAULT_SELECTION, _SELECTIONS,
    _planned, _read_csv, _select, _write_csv,
)


_LIGANDS = ("FAD", "IAI", "OQO", "SAM")
_RANK_METRICS = ("iptm", "ptm", "plddt", "ddg", "dsasa", "ligand_hbonds", "sc")
_PROVENANCE = {
    "ligand_rank": "013_paper_writing/figures/fig4/ligand_benchmark_rank_success_snapshot_20260602/code/plot_confidence_boxplots.py::build_rank_tables; SI 169,210",
    "sensitivity": "013_paper_writing/figures/supplementary_data/08_threshold_sensitivity/code/analyze_threshold_sensitivity.py::current_benchmarks,ligand_sensitivity; MainPaper 216-217; SI 205",
    "ddg": "013_paper_writing/figures/fig3/vhh_rosetta_balance/code/01_plot_vhh_rosetta_balance.py::bootstrap_median_ci; MainPaper 163; SI 244",
}


def _mapped_rows(path, names, columns, *, required):
    path = Path(path).expanduser().resolve()
    rows, header = _read_csv(path)
    mapping = {name: name for name in names}
    if columns:
        if set(columns) - set(names) or any(not isinstance(value, str) or not value for value in columns.values()):
            raise ValueError(f"Invalid canonical-to-CSV columns mapping: {columns}")
        mapping.update(columns)
    missing = [name for name in required if mapping[name] not in header]
    if missing:
        raise ValueError(f"{path}: required CSV columns missing: {missing}")
    return [{name: row.get(column, "") for name, column in mapping.items()} for row in rows], mapping, path


def _optional_number(raw, *, name, unit):
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unit {unit}: {name} must be numeric or blank, got {raw!r}") from exc
    if math.isnan(value):
        return None
    if not math.isfinite(value):
        raise ValueError(f"Unit {unit}: {name} must be finite")
    return value


def _unique(rows, keys):
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in keys)
        if not all(key) or key in seen:
            raise ValueError(f"Empty or repeated design-unit identity {key}; supply one fixed representative per unit")
        seen.add(key)


def _ranks(values, *, lower_is_better):
    valid = [value for value in values if value is not None]
    return [None if value is None else float(1 + sum(other < value if lower_is_better else other > value for other in valid)
            + (sum(other == value for other in valid) - 1) / 2) for value in values]


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def rank_ligands(representatives_csv, out_dir, columns=None):
    """Median per method/ligand, average-tie ranks, four-ligand/seven-metric means."""
    rows, mapping, source = _mapped_rows(representatives_csv, (*_KEY, *_RANK_METRICS), columns, required=_KEY)
    if not rows:
        raise ValueError("Representative CSV contains no design units")
    _unique(rows, _KEY)
    if any(row["group"] not in _LIGANDS for row in rows):
        raise ValueError(f"Ligand ranking groups must be exactly the named panel {_LIGANDS}")
    methods = sorted({row["method"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["group"])].append(row)
    cells = []
    for ligand in _LIGANDS:
        for metric in _RANK_METRICS:
            block = []
            for method in methods:
                group = grouped[(method, ligand)]
                values = [_optional_number(row[metric], name=metric, unit=row["unit_id"]) for row in group]
                observed = [value for value in values if value is not None]
                block.append(dict(method=method, group=ligand, metric=metric, units=len(group), n=len(observed),
                    missing_values=len(group)-len(observed), median=float(np.median(observed)) if observed else None,
                    status="missing_group" if not group else "missing_metric" if not observed else "observed"))
            ranks = _ranks([cell["median"] for cell in block], lower_is_better=metric == "ddg")
            for cell, rank in zip(block, ranks):
                cell.update(rank=rank, ranked_methods=sum(value is not None for value in ranks))
            cells.extend(block)
    metric_means, overall = [], []
    for method in methods:
        summaries = []
        for metric in _RANK_METRICS:
            selected = [cell for cell in cells if cell["method"] == method and cell["metric"] == metric]
            missing = [cell["group"] for cell in selected if cell["rank"] is None]
            summary = dict(method=method, metric=metric, observed_ligands=4-len(missing), expected_ligands=4,
                           missing_ligands=missing, mean_rank=None if missing else float(np.mean([cell["rank"] for cell in selected])),
                           status="incomplete_ligands" if missing else "complete")
            metric_means.append(summary)
            summaries.append(summary)
        missing = [item["metric"] for item in summaries if item["mean_rank"] is None]
        overall.append(dict(method=method, observed_metrics=7-len(missing), expected_metrics=7,
                            missing_metrics=missing, average_rank=None if missing else float(np.mean([item["mean_rank"] for item in summaries])),
                            status="incomplete_metrics" if missing else "complete"))
    for row, rank in zip(overall, _ranks([row["average_rank"] for row in overall], lower_is_better=True)):
        row["overall_rank"] = rank
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"ligand_metric_ranks_csv": out_dir / "ligand_metric_ranks.csv", "metric_mean_ranks_csv": out_dir / "metric_mean_ranks.csv",
               "overall_ranks_csv": out_dir / "overall_ranks.csv", "summary_json": out_dir / "ligand_ranking.json"}
    _write_csv(outputs["ligand_metric_ranks_csv"], cells)
    _write_csv(outputs["metric_mean_ranks_csv"], metric_means)
    _write_csv(outputs["overall_ranks_csv"], overall)
    _json(outputs["summary_json"], dict(analysis_version=1, source=str(source), provenance=_PROVENANCE["ligand_rank"], columns=mapping,
        parameters=dict(groups=list(_LIGANDS), metrics=list(_RANK_METRICS), within_group_statistic="median", ties="average",
                        lower_is_better=["ddg"], aggregation="four fixed ligand ranks equally, then seven metrics equally",
                        missing_policy="metric-wise omission; incomplete fixed-set aggregates remain undefined"),
        fixed_design_units=len(rows), overall=overall))
    return outputs


def _sensitivity_units(planned_csv, records_csv, endpoint, columns):
    planned_csv, records_csv = Path(planned_csv).expanduser().resolve(), Path(records_csv).expanduser().resolve()
    plans = _planned(planned_csv)
    records, header = _read_csv(records_csv)
    if not set(_KEY) <= set(header):
        raise ValueError("Records CSV requires unit_id, method and group")
    keys = {tuple(plan[name] for name in _KEY) for plan in plans}
    by_unit, seen = defaultdict(list), set()
    for index, record in enumerate(records):
        key = tuple(record[name] for name in _KEY)
        if key not in keys and record.get("parent_id"):
            key = (record["parent_id"], record["method"], record["group"])
        if key not in keys:
            raise ValueError(f"Record is not associated with a planned unit: {key}")
        record["record_id"] = record.get("record_id") or f"record_{index:07d}"
        identity = (*key, record["record_id"])
        if identity in seen:
            raise ValueError(f"Duplicate record identity: {identity}")
        seen.add(identity)
        status = record.get("record_status") or "observed"
        if status != "observed" and status not in _FAILED:
            raise ValueError(f"Unknown record_status {status!r}")
        if status == "observed":
            by_unit[key].append(record)
    metrics = _Metrics(columns)
    names = ("iptm", "rg_ratio", "sap_norm") if endpoint == "protein-binder" else ("iptm", "rg_ratio")
    units, selected_records = [], []
    for plan in plans:
        key = tuple(plan[name] for name in _KEY)
        observed = by_unit[key]
        candidates = [record for record in observed if not plan.get("stage") or record.get("stage") == plan["stage"]]
        selection = plan.get("selection") or _DEFAULT_SELECTION[endpoint]
        if selection not in _SELECTIONS:
            raise ValueError(f"Unknown selection {selection!r}")
        status = plan.get("generation_status") or plan.get("status") or ""
        selected = [] if status in _FAILED else _select(candidates, selection, endpoint, metrics)
        values = []
        for record in selected:
            measured = {name: metrics.number(record, name) for name in names}
            values.append(list(measured.values()))
            selected_records.append(dict(**{name: plan[name] for name in _KEY}, record_id=record["record_id"], **measured))
        units.append(dict(**{name: plan[name] for name in _KEY}, parent_id=plan.get("parent_id", ""), selection=selection,
            status=status if status in _FAILED else "available" if selected else "unavailable",
            observed_record_count=len(observed), selected_record_count=len(selected), available=bool(selected),
            selected_record_ids=[record["record_id"] for record in selected],
            observed_variant_count=len(candidates) if selection == "all_variants" else None,
            evaluated_variant_count=len(selected) if selection == "all_variants" else None,
            variant_set_complete=len(selected) == 20 if endpoint == "protein-binder" and selection == "all_variants" else None,
            values=np.asarray(values, dtype=float).reshape(-1, len(names))))
    return units, selected_records, metrics.columns


def _group_arrays(units, methods, groups):
    result = {}
    for method in methods:
        for group in groups:
            members = [unit for unit in units if unit["method"] == method and unit["group"] == group]
            measured = [unit["values"] for unit in members if len(unit["values"])]
            indexes = np.concatenate([np.full(len(unit["values"]), i, dtype=int) for i, unit in enumerate(members)]) if members else np.zeros(0, dtype=int)
            result[(method, group)] = dict(planned=len(members), available=sum(unit["available"] for unit in members),
                row_unit=indexes, values=np.concatenate(measured) if measured else np.zeros((0, 3)),
                unit_available=np.asarray([unit["available"] for unit in members], dtype=bool))
    return result


def _group_rate(group, denominator, iptm, rg, sap=None):
    values = group["values"]
    passing = (values[:, 0] > iptm) & (values[:, 1] < rg)
    if sap is not None:
        passing &= values[:, 2] < sap
    unit_pass = np.zeros(group["planned"], dtype=bool)
    np.logical_or.at(unit_pass, group["row_unit"], passing)
    n = group[denominator]
    passed = int(unit_pass.sum())
    return dict(planned_units=group["planned"], available_units=group["available"], denominator_units=n,
                passing_units=passed, fraction=passed/n if n else None,
                status="missing_group" if not group["planned"] else "no_available_units" if not n else "complete")


def _macro(rows):
    missing = [row["group"] for row in rows if row["fraction"] is None]
    return dict(fraction=None if missing else float(np.mean([row["fraction"] for row in rows])),
                status="incomplete_groups" if missing else "complete", missing_groups=missing,
                groups_expected=len(rows), groups_available=len(rows)-len(missing),
                planned_units=sum(row["planned_units"] for row in rows),
                available_units=sum(row["available_units"] for row in rows),
                denominator_units=sum(row["denominator_units"] for row in rows),
                passing_units=sum(row["passing_units"] for row in rows))


def threshold_sensitivity(planned_csv, records_csv, out_dir, endpoint, *, denominator="planned", columns=None,
                          seed=0, bootstrap_resamples=20000, comparison_methods=("Astrolabe-AF3", "ProteinHunter")):
    """Paper threshold grids on fixed units, plus fixed-ligand bootstrap and Rg envelope."""
    if endpoint not in {"protein-binder", "ligand-binder"}:
        raise ValueError("threshold_sensitivity endpoint must be protein-binder or ligand-binder")
    if denominator not in {"planned", "available"}:
        raise ValueError("denominator must be explicitly planned or available")
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if len(comparison_methods) != 2 or comparison_methods[0] == comparison_methods[1]:
        raise ValueError("comparison_methods must identify Astrolabe-AF3 and ProteinHunter separately")
    units, fixed_records, mapping = _sensitivity_units(planned_csv, records_csv, endpoint, columns)
    methods = sorted({unit["method"] for unit in units})
    groups = list(_LIGANDS) if endpoint == "ligand-binder" else sorted({unit["group"] for unit in units})
    if endpoint == "ligand-binder" and any(unit["group"] not in groups for unit in units):
        raise ValueError(f"Ligand groups must belong to the fixed panel {_LIGANDS}")
    arrays = _group_arrays(units, methods, groups)
    specs = [("iptm", np.round(np.arange(.65, .901, .01), 2))] if endpoint == "ligand-binder" else [
        ("iptm", np.round(np.arange(.60, .901, .01), 2)),
        ("rg_ratio", np.round(np.arange(1.10, 1.501, .02), 2)),
        ("sap_norm", np.round(np.arange(.20, .701, .02), 2))]
    group_rows, curves, intervals = [], [], []
    rng = np.random.default_rng(seed)
    for varied, thresholds in specs:
        for threshold in thresholds:
            for method in methods:
                rows = []
                for group in groups:
                    rate = _group_rate(arrays[(method, group)], denominator,
                        float(threshold) if varied == "iptm" else .8,
                        float(threshold) if varied == "rg_ratio" else 1.3,
                        float(threshold) if varied == "sap_norm" else None)
                    rows.append(dict(method=method, group=group, varied_metric=varied, threshold=float(threshold), **rate))
                group_rows.extend(rows)
                macro = _macro(rows)
                curves.append(dict(method=method, varied_metric=varied, threshold=float(threshold), **macro))
                if endpoint == "ligand-binder" and method in comparison_methods:
                    low = high = None
                    if macro["status"] == "complete":
                        # Exact bootstrap distribution for binary design-unit outcomes.
                        simulations = [rng.binomial(row["denominator_units"], row["fraction"], size=bootstrap_resamples)
                                       / row["denominator_units"] for row in rows]
                        low, high = map(float, np.quantile(np.mean(simulations, axis=0), [.025, .975]))
                    intervals.append(dict(method=method, threshold=float(threshold), fraction=macro["fraction"],
                        ci_low=low, ci_high=high, status=macro["status"], missing_groups=macro["missing_groups"],
                        denominator_units=macro["denominator_units"], bootstrap_resamples=bootstrap_resamples))
    grid, envelope = [], []
    if endpoint == "ligand-binder":
        for iptm in np.round(np.arange(.70, .901, .01), 2):
            for rg in np.round(np.arange(1.15, 1.451, .01), 2):
                rates = []
                missing = []
                for method in comparison_methods:
                    if method not in methods:
                        rates.append(None); missing.append(method)
                        continue
                    macro = _macro([dict(group=group, **_group_rate(arrays[(method, group)], denominator, iptm, rg)) for group in groups])
                    rates.append(macro["fraction"])
                    if macro["fraction"] is None:
                        missing.append(method)
                grid.append(dict(iptm_threshold=float(iptm), rg_threshold=float(rg),
                    astrolabe_method=comparison_methods[0], proteinhunter_method=comparison_methods[1],
                    astrolabe_fraction=rates[0], proteinhunter_fraction=rates[1],
                    difference=None if missing else rates[0]-rates[1], status="incomplete_comparison" if missing else "complete",
                    missing_methods=missing))
            points = [row for row in grid if row["iptm_threshold"] == float(iptm)]
            differences = [row["difference"] for row in points if row["difference"] is not None]
            reference = next(row["difference"] for row in points if row["rg_threshold"] == 1.30)
            envelope.append(dict(iptm_threshold=float(iptm), rg_cutoffs_expected=31, rg_cutoffs_available=len(differences),
                difference_at_rg_1_30=reference, minimum_difference=min(differences) if len(differences) == 31 else None,
                maximum_difference=max(differences) if len(differences) == 31 else None,
                status="complete" if len(differences) == 31 else "incomplete_comparison"))
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"units_csv": out_dir / "sensitivity_units.csv", "fixed_records_csv": out_dir / "fixed_records.csv",
               "groups_csv": out_dir / "threshold_groups.csv", "curves_csv": out_dir / "threshold_curves.csv",
               "summary_json": out_dir / "threshold_sensitivity.json"}
    _write_csv(outputs["units_csv"], [{key: value for key, value in unit.items() if key != "values"} for unit in units])
    _write_csv(outputs["fixed_records_csv"], fixed_records, (*_KEY, "record_id"))
    _write_csv(outputs["groups_csv"], group_rows)
    _write_csv(outputs["curves_csv"], curves)
    if endpoint == "ligand-binder":
        outputs.update(intervals_csv=out_dir / "pointwise_intervals.csv", joint_grid_csv=out_dir / "joint_threshold_grid.csv",
                       envelope_csv=out_dir / "compactness_difference_envelope.csv")
        _write_csv(outputs["intervals_csv"], intervals, ("method", "threshold", "ci_low", "ci_high"))
        _write_csv(outputs["joint_grid_csv"], grid)
        _write_csv(outputs["envelope_csv"], envelope)
    _json(outputs["summary_json"], dict(analysis_version=1, endpoint=endpoint, provenance=_PROVENANCE["sensitivity"],
        planned_csv=str(Path(planned_csv).expanduser().resolve()), records_csv=str(Path(records_csv).expanduser().resolve()),
        columns=mapping, parameters=dict(denominator=denominator, available_definition="unit with measured selected records; fixed across thresholds",
            groups=groups, methods=methods, thresholds={name: values.tolist() for name, values in specs},
            fixed_iptm=.8, fixed_rg_ratio=1.3, sap_applied_only_in_sap_panel=True,
            iptm_operator=">", rg_operator="<", sap_operator="<", variant_index="zero-based 0–19; no replacement of missing indices",
            seed=seed, bootstrap_resamples=bootstrap_resamples if endpoint == "ligand-binder" else 0,
            bootstrap_unit="design unit within each fixed ligand; binomial representation of binary bootstrap",
            bootstrap_percentiles=[2.5,97.5], comparison_methods=list(comparison_methods),
            envelope_rg_cutoffs=np.round(np.arange(1.15,1.451,.01),2).tolist() if endpoint == "ligand-binder" else [],
            envelope_is_confidence_interval=False), planned_units=len(units), available_units=sum(unit["available"] for unit in units),
        compactness_envelope=envelope))
    return outputs


def bootstrap_ddg(representatives_csv, out_dir, *, columns=None, seed=0, bootstrap_resamples=5000):
    """Signed AC−AB medians and 95% percentile CIs, stratified by design class."""
    keys = ("unit_id", "method", "group", "condition", "molecule_type")
    rows, mapping, source = _mapped_rows(representatives_csv, (*keys, "ddg_ab", "ddg_ac"), columns, required=keys)
    if not rows:
        raise ValueError("Representative CSV contains no design units")
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    for row in rows:
        if row["molecule_type"].lower() not in {"minibinder", "vhh"}:
            raise ValueError("molecule_type must explicitly distinguish minibinder and VHH")
        row["molecule_type"] = "VHH" if row["molecule_type"].lower() == "vhh" else "minibinder"
    _unique(rows, keys)
    groups, details = defaultdict(list), []
    for row in rows:
        a = _optional_number(row["ddg_ab"], name="ddg_ab", unit=row["unit_id"])
        b = _optional_number(row["ddg_ac"], name="ddg_ac", unit=row["unit_id"])
        paired = a is not None and b is not None
        detail = {**row, "ddg_ab": a, "ddg_ac": b, "signed_delta": b-a if paired else None,
                  "absolute_delta": abs(b-a) if paired else None,
                  "status": "paired" if paired else "missing_pair"}
        details.append(detail)
        groups[tuple(row[name] for name in keys[1:])].append(detail)
    rng, summaries = np.random.default_rng(seed), []
    for key, members in sorted(groups.items()):
        observed = [row for row in members if row["status"] == "paired"]
        values = np.asarray([row["signed_delta"] for row in observed], dtype=float)
        stats = dict(median_signed_delta=None, q25=None, q75=None, ci_low=None, ci_high=None, median_absolute_delta=None)
        if len(values):
            indices = rng.integers(0, len(values), size=(bootstrap_resamples, len(values)))
            medians = np.median(values[indices], axis=1)
            q25, median, q75 = map(float, np.percentile(values, [25,50,75]))
            low, high = map(float, np.percentile(medians, [2.5,97.5]))
            stats.update(median_signed_delta=median, q25=q25, q75=q75, ci_low=low, ci_high=high,
                         median_absolute_delta=float(np.median(np.abs(values))))
        summaries.append(dict(zip(keys[1:], key), input_units=len(members), paired_units=len(values),
                              missing_pairs=len(members)-len(values), status="complete" if len(values)==len(members) else "incomplete_pairs",
                              **stats))
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"designs_csv": out_dir / "ddg_designs.csv", "groups_csv": out_dir / "ddg_groups.csv", "summary_json": out_dir / "ddg_bootstrap.json"}
    _write_csv(outputs["designs_csv"], details)
    _write_csv(outputs["groups_csv"], summaries)
    _json(outputs["summary_json"], dict(analysis_version=1, source=str(source), provenance=_PROVENANCE["ddg"], columns=mapping,
        parameters=dict(signed_delta="AC minus AB", units="REU", strata=list(keys[1:]), seed=seed,
                        bootstrap_resamples=bootstrap_resamples, bootstrap_percentiles=[2.5,97.5],
                        bootstrap_unit="paired design", missing_policy="omit incomplete pairs from statistics, retain input-unit count"),
        groups=summaries))
    return outputs
