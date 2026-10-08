import csv
import json

import pytest

from astrolabe.analysis.paper_statistics import rank_ligands, threshold_sensitivity, bootstrap_ddg


LIGANDS = ("FAD", "IAI", "OQO", "SAM")
METRICS = ("iptm", "ptm", "plddt", "ddg", "dsasa", "ligand_hbonds", "sc")


def _csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _unit(unit_id, method, group, **kwargs):
    return dict(unit_id=unit_id, method=method, group=group, **kwargs)


def test_ligand_ranking_uses_fixed_medians_average_ties_and_metric_specific_missing(tmp_path):
    rows = []
    for ligand in LIGANDS:
        for method, values in [("A", [0, 0, 1]), ("B", [.2, .2, .2])]:
            for index, value in enumerate(values):
                metrics = {name: value for name in METRICS}
                metrics["ddg"] = [-1, -1, 100][index] if method == "A" else 0
                metrics["sc"] = .5
                if method == "A" and index == 2:
                    metrics["plddt"] = ""
                rows.append(_unit(str(index), method, ligand, **metrics))
    outputs = rank_ligands(_csv(tmp_path / "fixed.csv", rows), tmp_path / "rank")
    cells = _read(outputs["ligand_metric_ranks_csv"])
    lookup = {(row["method"], row["group"], row["metric"]): row for row in cells}
    # A's mean ipTM exceeds B's; its median is lower. Representatives cannot be reselected.
    assert float(lookup["A", "FAD", "iptm"]["median"]) == 0
    assert float(lookup["A", "FAD", "iptm"]["rank"]) == 2
    assert float(lookup["A", "FAD", "ddg"]["rank"]) == 1
    assert float(lookup["A", "FAD", "sc"]["rank"]) == 1.5
    assert lookup["A", "FAD", "plddt"]["n"] == "2"
    assert lookup["A", "FAD", "iptm"]["n"] == "3"
    overall = {row["method"]: row for row in _read(outputs["overall_ranks_csv"])}
    assert float(overall["B"]["average_rank"]) < float(overall["A"]["average_rank"])
    assert all(row["observed_metrics"] == "7" for row in overall.values())


def test_missing_ligand_does_not_silently_reduce_equal_weight_set(tmp_path):
    rows = [_unit("one", method, ligand, **{name: .5 for name in METRICS})
            for method in ("A", "B") for ligand in LIGANDS if not (method == "A" and ligand == "SAM")]
    outputs = rank_ligands(_csv(tmp_path / "fixed.csv", rows), tmp_path / "rank")
    means = _read(outputs["metric_mean_ranks_csv"])
    assert all(row["mean_rank"] == "" and row["observed_ligands"] == "3" for row in means if row["method"] == "A")
    cells = _read(outputs["ligand_metric_ranks_csv"])
    assert all(row["status"] == "missing_group" for row in cells if row["method"] == "A" and row["group"] == "SAM")
    with pytest.raises(ValueError, match="one fixed representative"):
        rank_ligands(_csv(tmp_path / "duplicate.csv", rows + [rows[0]]), tmp_path / "invalid")


def test_protein_sensitivity_keeps_same_variant_and_denominator_window(tmp_path):
    planned = _csv(tmp_path / "planned.csv", [_unit("parent", "SolubleMPNN", "target", selection="all_variants"),
                                            _unit("missing", "SolubleMPNN", "target", selection="all_variants")])
    records = _csv(tmp_path / "records.csv", [
        _unit("parent", "SolubleMPNN", "target", variant_index=0, iptm=.9, rg_ratio=1.4, sap_norm=.1),
        _unit("parent", "SolubleMPNN", "target", variant_index=1, iptm=.7, rg_ratio=1.0, sap_norm=.1),
        _unit("parent", "SolubleMPNN", "target", variant_index=2, iptm=.9, rg_ratio=1.0, sap_norm=.5),
        _unit("parent", "SolubleMPNN", "target", variant_index=20, iptm=.9, rg_ratio=1.0, sap_norm=.1)])
    for denominator, fraction in [("planned", .5), ("available", 1.0)]:
        outputs = threshold_sensitivity(planned, records, tmp_path / denominator, "protein-binder", denominator=denominator)
        curves = _read(outputs["curves_csv"])
        confidence = next(row for row in curves if row["varied_metric"] == "iptm" and float(row["threshold"]) == .8)
        sap = next(row for row in curves if row["varied_metric"] == "sap_norm" and float(row["threshold"]) == .4)
        assert float(confidence["fraction"]) == fraction  # ipTM panel fixes Rg only; no SAP gate.
        assert float(sap["fraction"]) == 0
        units = _read(outputs["units_csv"])
        assert units[0]["evaluated_variant_count"] == "3"
        assert units[0]["variant_set_complete"] == "False"
        assert units[1]["available"] == "False"
        assert len(_read(outputs["fixed_records_csv"])) == 3


def test_ligand_bootstrap_resamples_design_units_not_variants_and_keeps_rg_envelope(tmp_path):
    plans, records = [], []
    for method in ("Astrolabe-AF3", "ProteinHunter"):
        for ligand in LIGANDS:
            for unit_id in ("many", "one"):
                plans.append(_unit(unit_id, method, ligand, selection="all_variants"))
                for index in range(20 if unit_id == "many" else 1):
                    passes = method == "Astrolabe-AF3" and unit_id == "many"
                    records.append(_unit(unit_id, method, ligand, variant_index=index, iptm=.95 if passes else .1, rg_ratio=1.2))
    planned, records_csv = _csv(tmp_path / "planned.csv", plans), _csv(tmp_path / "records.csv", records)
    outputs = threshold_sensitivity(planned, records_csv, tmp_path / "sensitivity", "ligand-binder", seed=0)
    intervals = _read(outputs["intervals_csv"])
    point = next(row for row in intervals if row["method"] == "Astrolabe-AF3" and float(row["threshold"]) == .8)
    assert float(point["fraction"]) == .5  # Flattening variants would give 20/21.
    assert float(point["ci_low"]) == .125
    assert float(point["ci_high"]) == .875
    assert point["denominator_units"] == "8"
    envelope = _read(outputs["envelope_csv"])
    nominal = next(row for row in envelope if float(row["iptm_threshold"]) == .8)
    assert nominal["rg_cutoffs_available"] == "31"
    assert float(nominal["minimum_difference"]) == 0
    assert float(nominal["maximum_difference"]) == .5
    assert float(nominal["difference_at_rg_1_30"]) == .5
    assert len(_read(outputs["joint_grid_csv"])) == 651
    params = json.loads(outputs["summary_json"].read_text())["parameters"]
    assert params["groups"] == list(LIGANDS)
    assert params["bootstrap_resamples"] == 20000
    assert params["seed"] == 0
    assert params["envelope_is_confidence_interval"] is False


def test_ligand_missing_group_leaves_macro_and_interval_undefined(tmp_path):
    plans = [_unit("one", method, ligand) for method in ("Astrolabe-AF3", "ProteinHunter") for ligand in LIGANDS
             if not (method == "ProteinHunter" and ligand == "SAM")]
    records = [{**plan, "iptm": .9, "rg_ratio": 1} for plan in plans]
    outputs = threshold_sensitivity(_csv(tmp_path / "planned.csv", plans), _csv(tmp_path / "records.csv", records),
                                    tmp_path / "missing", "ligand-binder", bootstrap_resamples=16)
    curves = _read(outputs["curves_csv"])
    assert all(row["fraction"] == "" and row["status"] == "incomplete_groups" for row in curves if row["method"] == "ProteinHunter")
    ci = _read(outputs["intervals_csv"])
    assert all(row["ci_low"] == "" for row in ci if row["method"] == "ProteinHunter")
    assert all(row["difference"] == "" for row in _read(outputs["joint_grid_csv"]))


def test_ddg_bootstrap_signed_vs_absolute_and_separate_design_classes(tmp_path):
    rows = [_unit(str(i), "Astrolabe", "IL4_IL13", condition="dual", molecule_type="minibinder", left=-20, right=-20+delta)
            for i, delta in enumerate([-10, 1, 2])]
    rows.append(_unit("missing", "Astrolabe", "IL4_IL13", condition="dual", molecule_type="minibinder", left=-20, right=""))
    rows += [_unit(str(i), "Astrolabe", "IL4_IL13", condition="dual", molecule_type="VHH", left=-20, right=-16) for i in range(2)]
    source = _csv(tmp_path / "ddg.csv", rows)
    outputs = bootstrap_ddg(source, tmp_path / "ddg", columns=dict(ddg_ab="left", ddg_ac="right"), seed=0)
    summaries = {row["molecule_type"]: row for row in _read(outputs["groups_csv"])}
    mini = summaries["minibinder"]
    assert mini["input_units"] == "4" and mini["paired_units"] == "3"
    assert float(mini["median_signed_delta"]) == 1
    assert float(mini["median_absolute_delta"]) == 2
    assert float(mini["ci_low"]) == -10 and float(mini["ci_high"]) == 2
    assert summaries["VHH"]["paired_units"] == "2"
    assert float(summaries["VHH"]["ci_low"]) == float(summaries["VHH"]["ci_high"]) == 4
    again = bootstrap_ddg(source, tmp_path / "again", columns=dict(ddg_ab="left", ddg_ac="right"), seed=0)
    assert outputs["groups_csv"].read_text() == again["groups_csv"].read_text()
    assert json.loads(outputs["summary_json"].read_text())["parameters"]["bootstrap_resamples"] == 5000
