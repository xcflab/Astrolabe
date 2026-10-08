"""Planned-unit accounting and paper computational endpoints.

A planned CSV has one row per (unit_id, method, group). Optional fields are
run_dir (relative to the planned CSV), selection, parent_id, stage, and status.
A SolubleMPNN parent is one planned unit; its variant records share unit_id, or
explicitly refer to that unit through parent_id. Parent IDs do not implicitly
collapse rows of the planned manifest. Collect reads only each run_dir/scores.csv.

The records CSV retains wide, same-state measurements: paired-target metrics
must be in the same row. ``columns`` maps canonical names to exact CSV columns.
Default names are identical except native_iptm -> iptm, complex_iptm -> AB.iptm,
iptm_ab -> AB.iptm and iptm_ac -> AC.iptm. plddt and all confidence/occupancy
fractions are 0–1, without automatic scale conversion; rg_ratio and sap_norm
must already be the paper's length-normalized quantities. Signed ddg_ab/ddg_ac
are optional paired values in REU. They enable AC-minus-AB and weaker-side
summaries. stage is an explicit planned-row filter, useful for monomer-to-complex
trajectories; no stage names are guessed.

Cyclic endpoints accept closure_all_bonds (0/1) only for the verified native
cyclic_all_bonds_within_error loss. Otherwise they require closure_missing_atoms
plus either closure_errors (a
JSON array of signed bond-length errors, Angstrom) or closure_max_abs_error.
The latter must summarize every specified bond. Missing endpoint atoms fail
closure without inventing a distance. Continuous closure scores are not used
as geometric closure flags.

Selections: native_max_iptm, complex_max_iptm, max_objective,
accepted_max_objective, last_accepted, max_objective_first_20, step_29,
all_variants, any_step, already_selected. Pocket defaults to its paper-defined
occupancy-first ipTM selection. First-20 means state indices 0–19 for aptamers;
protein-binder all_variants takes zero-based variant_index 0–19 when
supplied, otherwise the first 20 records in input generation order. Missing
indices are never replaced with later variants. Variants and repeated sequences are never
deduplicated. Incomplete variant sets are counted and explicitly labelled.

Missing planned units fail endpoints and remain in denominators. An observed
selected/candidate row with a missing required metric raises ValueError. All
conjunctions are evaluated within one record before any-step/variant reduction.
Group fractions use planned counts; method fractions give groups equal weight.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from astrolabe.posthoc.io_utils import parse_scores_csv


_KEY = ("unit_id", "method", "group")
_ENDPOINTS = {
    "protein-binder": ("pass_iptm", "pass_iptm_rg", "pass_iptm_rg_sap"),
    "ligand-binder": ("pass_iptm", "pass_compact"),
    "dual-target": ("pass_paired",),
    "vhh": ("pass_paired", "pass_paired_strict"),
    "idr": ("pass_iptm", "pass_joint"),
    "pocket": ("pass_occupancy", "pass_joint"),
    "cid": ("pass_joint",),
    "symmetry": ("pass_joint",),
    "cyclic-bound": ("pass_closure", "pass_joint"),
    "cyclic-multiring": ("pass_closure", "pass_joint"),
    "aptamer": (),
}
_DEFAULT_SELECTION = {
    "protein-binder": "already_selected", "ligand-binder": "already_selected",
    "dual-target": "last_accepted", "vhh": "last_accepted", "idr": "max_objective",
    "pocket": "pocket_max_iptm", "cid": "any_step", "symmetry": "any_step",
    "cyclic-bound": "step_29", "cyclic-multiring": "max_objective",
    "aptamer": "max_objective_first_20",
}
_SELECTIONS = {"native_max_iptm", "complex_max_iptm", "max_objective", "accepted_max_objective",
               "last_accepted", "max_objective_first_20", "step_29", "all_variants", "any_step",
               "already_selected", "pocket_max_iptm"}
_METRICS = {"iptm", "native_iptm", "complex_iptm", "iptm_ab", "iptm_ac", "plddt", "rg_ratio",
            "sap_norm", "ipsae_min", "target_non_loop", "apo_median_non_loop", "occupancy", "ca90",
            "ananas_score", "weighted_score", "step", "accepted", "variant_index", "ddg_ab", "ddg_ac",
            "sequence_ab", "sequence_ac", "closure_all_bonds", "closure_errors", "closure_missing_atoms", "closure_max_abs_error"}
_FAILED = {"generation_failed", "prediction_failed", "failed", "missing_run", "missing_scores",
           "empty_scores", "not_generated", "not_started", "missing_records"}
_UNIT_INTERVAL = {"iptm", "native_iptm", "complex_iptm", "iptm_ab", "iptm_ac", "plddt", "ipsae_min",
                  "target_non_loop", "apo_median_non_loop", "occupancy", "ca90", "ananas_score"}


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = [str(name).strip() for name in (reader.fieldnames or [])]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError(f"Empty or duplicate CSV column in {path}")
        reader.fieldnames = header
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"CSV row has more fields than its header: {path}")
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows, header


def _planned(path: Path) -> list[dict]:
    rows, header = _read_csv(path)
    if not set(_KEY) <= set(header):
        raise ValueError("Planned CSV requires unit_id, method and group columns")
    if not rows:
        raise ValueError("Planned CSV contains no design units")
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in _KEY)
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate planned unit: {key}")
        seen.add(key)
    return rows


def _write_csv(path: Path, rows: list[dict], columns=()) -> None:
    fields = list(columns)
    for row in rows:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def collect_records(planned_csv: str | Path, output_csv: str | Path) -> dict:
    """Collect declared run CSVs and explicit missing-unit placeholders."""
    planned_csv, output_csv = Path(planned_csv).expanduser().resolve(), Path(output_csv).expanduser()
    plans = _planned(planned_csv)
    records, statuses = [], Counter()
    for plan in plans:
        meta = {name: plan[name] for name in _KEY}
        meta.update(parent_id=plan.get("parent_id", ""), selection=plan.get("selection", ""))
        run = Path(plan["run_dir"]).expanduser() if plan.get("run_dir") else None
        if run is not None and not run.is_absolute():
            run = planned_csv.parent / run
        path = run / "scores.csv" if run is not None else None
        status = "observed"
        native = {}
        if run is None or not run.is_dir():
            status = "missing_run"
        elif not path.is_file():
            status = "missing_scores"
        else:
            native = parse_scores_csv(path)
            source_rows, _ = _read_csv(path)
            steps = []
            for row in source_rows:
                try:
                    step = int(row.get("step", ""))
                except ValueError as exc:
                    raise ValueError(f"Invalid native step in {path}: {row.get('step')!r}") from exc
                steps.append(step)
            if len(steps) != len(set(steps)):
                raise ValueError(f"Duplicate step in {path}; parse_scores_csv would overwrite source states")
            if not native:
                status = "empty_scores"
        statuses[status] += 1
        meta.update(run_dir=str(run) if run is not None else "", source_scores_path=str(path) if path else "")
        if status != "observed":
            records.append({**meta, "record_id": "", "record_status": status})
            continue
        for index, row in enumerate(native.values()):
            records.append({**row, **meta, "record_id": f"{plan['method']}/{plan['group']}/{plan['unit_id']}:{index}",
                            "source_row": index + 2, "record_status": "observed"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, records, (*_KEY, "record_id", "record_status"))
    return {"output_csv": str(output_csv), "planned_units": len(plans), "observed_units": statuses["observed"],
            "observed_records": sum(r["record_status"] == "observed" for r in records), "status_counts": dict(statuses)}


class _Metrics:
    def __init__(self, columns):
        self.columns = {name: name for name in _METRICS}
        self.columns.update(native_iptm="iptm", complex_iptm="AB.iptm", iptm_ab="AB.iptm", iptm_ac="AC.iptm")
        if columns:
            if set(columns) - _METRICS or any(not isinstance(v, str) or not v for v in columns.values()):
                raise ValueError(f"columns requires canonical metric names and nonempty CSV column names: {columns}")
            self.columns.update(columns)

    def present(self, row, name):
        return row.get(self.columns[name], "") not in ("", None)

    def raw(self, row, name):
        column = self.columns[name]
        if not self.present(row, name):
            raise ValueError(f"Record {row.get('record_id', '?')}: required metric {name!r} missing from column {column!r}")
        return row[column]

    def number(self, row, name):
        value = self.raw(row, name)
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Record {row.get('record_id', '?')}: {name} must be numeric, got {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Record {row.get('record_id', '?')}: {name} must be finite")
        if name in _UNIT_INTERVAL and not 0 <= number <= 1:
            raise ValueError(f"Record {row.get('record_id', '?')}: {name} must be on the 0–1 scale; no automatic conversion")
        return number

    def integer(self, row, name):
        value = self.number(row, name)
        if value < 0 or value != int(value):
            raise ValueError(f"Record {row.get('record_id', '?')}: {name} must be a nonnegative integer")
        return int(value)

    def accepted(self, row):
        value = str(self.raw(row, "accepted")).lower()
        if value not in {"true", "false", "1", "0"}:
            raise ValueError(f"Record {row.get('record_id', '?')}: accepted must be true/false or 1/0")
        return value in {"true", "1"}


def _select(rows, selection, endpoint, metrics):
    if selection == "already_selected":
        if len(rows) > 1:
            raise ValueError("already_selected requires at most one observed row per planned unit; declare its method-specific selection")
        return rows
    if selection == "any_step":
        return rows
    if selection == "all_variants":
        has_index = [metrics.present(row, "variant_index") for row in rows]
        if any(has_index):
            if not all(has_index):
                raise ValueError("variant_index must be supplied for every variant when used")
            indices = [metrics.integer(row, "variant_index") for row in rows]
            if len(indices) != len(set(indices)):
                raise ValueError("all_variants requires one evaluated record per variant_index")
            rows = [row for index, row in sorted(zip(indices, rows), key=lambda pair: pair[0])
                    if endpoint != "protein-binder" or index < 20]
        return rows[:20] if endpoint == "protein-binder" else rows
    if selection in {"accepted_max_objective", "last_accepted"}:
        rows = [row for row in rows if metrics.accepted(row)]
    if selection == "max_objective_first_20":
        rows = [row for row in rows if metrics.integer(row, "step") < 20]
    if selection == "step_29":
        rows = [row for row in rows if metrics.integer(row, "step") == 29]
        if len(rows) > 1:
            raise ValueError("step_29 requires one same-state record at index 29")
        return rows
    if selection == "complex_max_iptm":
        rows = [row for row in rows if metrics.present(row, "complex_iptm")]
    if not rows:
        return []
    if selection == "pocket_max_iptm":
        occupied = [row for row in rows if metrics.number(row, "occupancy") > 0.8]
        # Validate every observed candidate's ipTM before deciding the representative.
        for row in rows:
            metrics.number(row, "iptm")
        return [max(occupied or rows, key=lambda row: metrics.number(row, "iptm"))]
    metric = {"native_max_iptm": "native_iptm", "complex_max_iptm": "complex_iptm",
              "last_accepted": "step"}.get(selection, "weighted_score")
    return [max(rows, key=lambda row: metrics.number(row, metric))]


def _closure(row, metrics):
    if metrics.present(row, "closure_all_bonds"):
        closed = metrics.integer(row, "closure_all_bonds")
        if closed not in (0, 1):
            raise ValueError("closure_all_bonds must be 0 or 1 from cyclic_all_bonds_within_error")
        diagnostics = {"closure_all_bonds": closed}
        for name in ("closure_missing_atoms", "closure_max_abs_error"):
            if metrics.present(row, name):
                diagnostics[name] = metrics.number(row, name)
        return bool(closed), diagnostics
    missing = metrics.integer(row, "closure_missing_atoms")
    if missing:
        return False, {"closure_missing_atoms": missing, "closure_max_abs_error": None}
    if metrics.present(row, "closure_errors"):
        errors = json.loads(metrics.raw(row, "closure_errors"))
        if not isinstance(errors, list) or not errors:
            raise ValueError(f"Record {row['record_id']}: closure_errors must be a nonempty JSON array")
        try:
            errors = [float(value) for value in errors]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Record {row['record_id']}: closure_errors requires every specified bond error") from exc
        if not all(math.isfinite(value) for value in errors):
            raise ValueError(f"Record {row['record_id']}: closure_errors must be finite")
        maximum = max(map(abs, errors))
    else:
        maximum = metrics.number(row, "closure_max_abs_error")
        if maximum < 0:
            raise ValueError("closure_max_abs_error must be nonnegative")
    return maximum <= 0.30, {"closure_missing_atoms": 0, "closure_max_abs_error": maximum}


def _endpoint(row, endpoint, metrics, use_ipsae=False, use_ddg=False):
    values, flags = {}, {}
    def get(name):
        value = metrics.number(row, name)
        values[name] = value
        return value
    if endpoint == "protein-binder":
        confidence, compact, sap = get("iptm") > .8, get("rg_ratio") < 1.3, get("sap_norm") < .4
        flags = dict(pass_iptm=confidence, pass_iptm_rg=confidence and compact, pass_iptm_rg_sap=confidence and compact and sap)
        if use_ipsae:
            confidence = get("ipsae_min") > .61
            flags.update(pass_ipsae=confidence, pass_ipsae_rg=confidence and compact, pass_ipsae_rg_sap=confidence and compact and sap)
    elif endpoint == "ligand-binder":
        confidence, compact = get("iptm") > .8, get("rg_ratio") < 1.3
        flags = dict(pass_iptm=confidence, pass_compact=confidence and compact)
    elif endpoint in {"dual-target", "vhh"}:
        ab, ac = get("iptm_ab"), get("iptm_ac")
        values["pair_min_iptm"] = min(ab, ac)
        sequence_fields = [metrics.present(row, name) for name in ("sequence_ab", "sequence_ac")]
        if any(sequence_fields):
            if not all(sequence_fields) or metrics.raw(row, "sequence_ab") != metrics.raw(row, "sequence_ac"):
                raise ValueError(f"Record {row['record_id']}: paired-target values must belong to the same binder sequence")
        flags = dict(pass_paired=min(ab, ac) > .8) if endpoint == "dual-target" else dict(pass_paired=min(ab, ac) >= .7, pass_paired_strict=min(ab, ac) >= .8)
        if use_ddg:
            a, b = get("ddg_ab"), get("ddg_ac")
            values.update(ddg_ac_minus_ab=b-a, ddg_absolute_difference=abs(b-a), weaker_ddg=max(a, b),
                          weaker_ddg_target="AB" if a > b else "AC" if b > a else "tie")
    elif endpoint == "idr":
        confidence, ordered = get("iptm") > .7, get("target_non_loop") > get("apo_median_non_loop")
        flags = dict(pass_iptm=confidence, pass_joint=confidence and ordered)
    elif endpoint == "pocket":
        occupied, confident = get("occupancy") > .8, get("iptm") > .8
        flags = dict(pass_occupancy=occupied, pass_joint=occupied and confident)
    elif endpoint == "cid":
        localized, confident = get("ca90") > .8, get("iptm") > .8
        flags = dict(pass_joint=localized and confident)
    elif endpoint == "symmetry":
        symmetric, confident, interface = get("ananas_score") > .8, get("plddt") > .8, get("iptm") > .6
        flags = dict(pass_joint=symmetric and confident and interface)
    elif endpoint.startswith("cyclic-"):
        closure, geometry = _closure(row, metrics)
        values.update(geometry)
        confident = get("iptm") > .6 if endpoint == "cyclic-bound" else get("plddt") > .8
        flags = dict(pass_closure=closure, pass_joint=closure and confident)
    else:
        get("iptm"); get("plddt")
    return values, flags


def analyze_scores(planned_csv: str | Path, records_csv: str | Path, out_dir: str | Path,
                   endpoint: str, columns: dict | None = None) -> dict[str, Path]:
    """Write units, selected records, group counts and equally weighted methods."""
    if endpoint not in _ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {', '.join(sorted(_ENDPOINTS))}")
    planned_csv, records_csv, out_dir = Path(planned_csv).expanduser().resolve(), Path(records_csv).expanduser().resolve(), Path(out_dir).expanduser()
    plans = _planned(planned_csv)
    records, header = _read_csv(records_csv)
    if not set(_KEY) <= set(header):
        raise ValueError("Records CSV requires unit_id, method and group columns")
    metrics = _Metrics(columns)
    plan_keys = {tuple(plan[name] for name in _KEY) for plan in plans}
    grouped = defaultdict(list)
    seen = set()
    for index, row in enumerate(records):
        key = tuple(row[name] for name in _KEY)
        if key not in plan_keys and row.get("parent_id"):
            key = (row["parent_id"], row["method"], row["group"])
        if key not in plan_keys:
            raise ValueError(f"Record is not associated with a planned unit: {key}")
        row["record_id"] = row.get("record_id") or f"record_{index:07d}"
        identity = (*key, row["record_id"])
        if identity in seen:
            raise ValueError(f"Duplicate record identity: {identity}")
        seen.add(identity)
        status = row.get("record_status") or "observed"
        if status != "observed" and status not in _FAILED:
            raise ValueError(f"Unknown record_status {status!r} for {row['record_id']}")
        row["record_status"] = status
        grouped[key].append(row)
    use_ipsae = endpoint == "protein-binder" and ("ipsae_min" in (columns or {}) or metrics.columns["ipsae_min"] in header)
    use_ddg = endpoint in {"dual-target", "vhh"} and any(name in (columns or {}) or metrics.columns[name] in header for name in ("ddg_ab", "ddg_ac"))
    flag_names = list(_ENDPOINTS[endpoint])
    if use_ipsae:
        flag_names.extend(("pass_ipsae", "pass_ipsae_rg", "pass_ipsae_rg_sap"))
    units, selected_rows = [], []
    for plan in plans:
        key = tuple(plan[name] for name in _KEY)
        observed = [row for row in grouped[key] if row["record_status"] == "observed"]
        selection = plan.get("selection") or _DEFAULT_SELECTION[endpoint]
        if selection not in _SELECTIONS:
            raise ValueError(f"Unknown selection {selection!r} for {key}")
        candidates = [row for row in observed if not plan.get("stage") or row.get("stage") == plan["stage"]]
        status = plan.get("generation_status") or plan.get("status") or ""
        if status in _FAILED:
            selected = []
        else:
            selected = _select(candidates, selection, endpoint, metrics)
            status = ("observed" if selected else "no_eligible_records" if observed else
                      grouped[key][0]["record_status"] if grouped[key] else "missing_records")
        unit = {**{name: plan[name] for name in _KEY}, "parent_id": plan.get("parent_id", ""),
                "selection": selection, "status": status, "observed_record_count": len(observed),
                "eligible_record_count": len(candidates), "selected_record_count": len(selected),
                "selected_record_ids": [row["record_id"] for row in selected],
                "observed_variant_count": len(candidates) if selection == "all_variants" else None,
                "evaluated_variant_count": len(selected) if selection == "all_variants" else None,
                "variant_set_complete": len(selected) == 20 if endpoint == "protein-binder" and selection == "all_variants" else None}
        if selected and endpoint == "protein-binder" and selection == "all_variants" and len(selected) < 20:
            unit["status"] = "incomplete_variant_set"
        unit.update({name: False for name in flag_names})
        witnesses, paired = {}, []
        for row in selected:
            values, flags = _endpoint(row, endpoint, metrics, use_ipsae, use_ddg)
            selected_rows.append({**row, "analysis_unit_id": plan["unit_id"], "analysis_selection": selection, **values, **flags})
            for flag, passed in flags.items():
                if passed:
                    unit[flag] = True
                    witnesses.setdefault(flag, row["record_id"])
            if len(selected) == 1:
                unit.update(values)
            if "pair_min_iptm" in values:
                paired.append(values["pair_min_iptm"])
        unit["criterion_witnesses"] = witnesses
        if paired:
            unit["pair_min_iptm_for_ccdf"] = max(paired)
        units.append(unit)
    unit_groups = defaultdict(list)
    for unit in units:
        unit_groups[(unit["method"], unit["group"])].append(unit)
    group_rows = []
    for (method, group), members in sorted(unit_groups.items()):
        result = dict(method=method, group=group, planned_units=len(members),
                      observed_units=sum(unit["observed_record_count"] > 0 for unit in members),
                      selected_units=sum(unit["selected_record_count"] > 0 for unit in members),
                      status_counts=dict(Counter(unit["status"] for unit in members)))
        for flag in flag_names:
            passed = sum(unit[flag] for unit in members)
            result[f"{flag}_count"], result[f"{flag}_fraction"] = passed, passed / len(members)
        # Descriptive statistics use measured representatives, never missing-unit zeros.
        descriptive = ("iptm", "plddt") if endpoint == "aptamer" else ("ddg_ac_minus_ab", "ddg_absolute_difference", "weaker_ddg") if use_ddg else ()
        for name in descriptive:
            values = [unit[name] for unit in members if name in unit]
            result[f"{name}_n"] = len(values)
            result[f"{name}_mean"] = float(np.mean(values)) if values else None
            result[f"{name}_median"] = float(np.median(values)) if values else None
        group_rows.append(result)
    method_rows = []
    for method in sorted({plan["method"] for plan in plans}):
        groups = [row for row in group_rows if row["method"] == method]
        result = dict(method=method, groups=len(groups), planned_units=sum(row["planned_units"] for row in groups),
                      observed_units=sum(row["observed_units"] for row in groups), selected_units=sum(row["selected_units"] for row in groups))
        for flag in flag_names:
            result[f"{flag}_fraction"] = float(np.mean([row[f"{flag}_fraction"] for row in groups]))
        method_rows.append(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"units_csv": out_dir / "units.csv", "selected_records_csv": out_dir / "selected_records.csv",
               "groups_csv": out_dir / "groups.csv", "overall_csv": out_dir / "overall.csv", "overall_json": out_dir / "overall.json"}
    _write_csv(outputs["units_csv"], units)
    _write_csv(outputs["selected_records_csv"], selected_rows, (*_KEY, "record_id"))
    _write_csv(outputs["groups_csv"], group_rows)
    _write_csv(outputs["overall_csv"], method_rows)
    outputs["overall_json"].write_text(json.dumps({"analysis_version": 1, "endpoint": endpoint,
        "planned_csv": str(planned_csv), "records_csv": str(records_csv), "columns": metrics.columns,
        "denominator": "planned units within group; equal mean of group fractions within method",
        "planned_units": len(plans), "record_rows": len(records), "groups": group_rows, "methods": method_rows}, indent=2, allow_nan=False) + "\n")
    if endpoint == "dual-target":
        ccdf = []
        for threshold in np.linspace(0, 1, 401):
            for method in sorted({unit["method"] for unit in units}):
                fractions = []
                for (group_method, group), members in sorted(unit_groups.items()):
                    if group_method != method:
                        continue
                    passed = sum(unit.get("pair_min_iptm_for_ccdf", -1) > threshold for unit in members)
                    fraction = passed / len(members)
                    fractions.append(fraction)
                    ccdf.append(dict(method=method, group=group, threshold=float(threshold), passed_units=passed,
                                     planned_units=len(members), fraction=fraction))
                ccdf.append(dict(method=method, group="__equal_group_mean__", threshold=float(threshold), passed_units=None,
                                 planned_units=sum(row["planned_units"] for row in group_rows if row["method"] == method), fraction=float(np.mean(fractions))))
        outputs["ccdf_csv"] = out_dir / "dual_target_ccdf.csv"
        _write_csv(outputs["ccdf_csv"], ccdf)
    return outputs
