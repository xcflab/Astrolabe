import csv
import json

import pytest

from astrolabe.analysis.paper_scores import analyze_scores, collect_records


def _csv(path, rows, fields=("unit_id", "method", "group")):
    columns = list(fields)
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plan(unit="one", group="target", **extra):
    return dict(unit_id=unit, method="Astrolabe", group=group, **extra)


def _record(unit="one", group="target", **metrics):
    return dict(unit_id=unit, method="Astrolabe", group=group, **metrics)


def _run(tmp_path, endpoint, plans, records, columns=None):
    outputs = analyze_scores(_csv(tmp_path / "planned.csv", plans), _csv(tmp_path / "records.csv", records),
                             tmp_path / "analysis", endpoint, columns)
    with outputs["units_csv"].open() as handle:
        units = list(csv.DictReader(handle))
    return outputs, units, json.loads(outputs["overall_json"].read_text())


def test_collect_preserves_rejections_and_repeated_sequences_and_missing_units(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _csv(run / "scores.csv", [dict(step=0, accepted=True, A_sequence="AAA", weighted_score=.1),
                              dict(step=1, accepted=False, A_sequence="AAA", weighted_score=.0),
                              dict(step=2, accepted=True, A_sequence="ACA", weighted_score=.2)], fields=())
    planned = _csv(tmp_path / "planned.csv", [_plan(run_dir="run"), _plan("missing", run_dir="absent")])
    summary = collect_records(planned, tmp_path / "collected.csv")
    with (tmp_path / "collected.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert summary["planned_units"] == 2
    assert summary["observed_records"] == 3
    assert summary["status_counts"] == {"observed": 1, "missing_run": 1}
    assert [row["A_sequence"] for row in rows[:3]] == ["AAA", "AAA", "ACA"]
    assert rows[1]["accepted"] == "False"
    assert rows[-1]["record_status"] == "missing_run"
    assert len({row["record_id"] for row in rows[:3]}) == 3


def test_planned_missing_units_remain_denominator_and_targets_have_equal_weight(tmp_path):
    plans = [_plan("a", "large"), _plan("b", "large"), _plan("c", "large"), _plan("d", "small")]
    records = [_record("a", "large", iptm=.8, rg_ratio=1, sap_norm=.1),
               _record("d", "small", iptm=.9, rg_ratio=1, sap_norm=.1)]
    _, units, result = _run(tmp_path, "protein-binder", plans, records)
    assert len(units) == 4
    assert [row["status"] for row in units] == ["observed", "missing_records", "missing_records", "observed"]
    assert result["methods"][0]["pass_iptm_rg_sap_fraction"] == .5  # pooled fraction would be .25
    assert result["groups"][0]["planned_units"] == 3
    assert result["groups"][0]["pass_iptm_count"] == 0


def test_cumulative_protein_filters_cannot_mix_variants_and_first_twenty_is_fixed(tmp_path):
    rows = []
    for index in range(21):
        metrics = dict(iptm=.7, rg_ratio=1, sap_norm=.1)
        if index == 0:
            metrics.update(iptm=.9, rg_ratio=2)
        elif index == 2:
            metrics.update(iptm=.9, sap_norm=.5)
        elif index == 20:
            metrics.update(iptm=.9)
        rows.append(_record(variant_index=index, **metrics))
    _, units, _ = _run(tmp_path, "protein-binder", [_plan(selection="all_variants")], rows)
    row = units[0]
    assert row["pass_iptm"] == row["pass_iptm_rg"] == "True"
    assert row["pass_iptm_rg_sap"] == "False"
    assert row["observed_variant_count"] == "21"
    assert row["evaluated_variant_count"] == "20"
    assert row["variant_set_complete"] == "True"
    assert len(json.loads(row["selected_record_ids"])) == 20


def test_short_variant_set_is_reported_without_losing_parent_denominator(tmp_path):
    _, units, result = _run(tmp_path, "protein-binder", [_plan(selection="all_variants")],
                           [_record(iptm=.9, rg_ratio=1, sap_norm=.1)])
    assert units[0]["status"] == "incomplete_variant_set"
    assert units[0]["variant_set_complete"] == "False"
    assert result["methods"][0]["planned_units"] == 1
    assert units[0]["pass_iptm_rg_sap"] == "True"


@pytest.mark.parametrize("endpoint,metric_rows,expected", [
    ("protein-binder", [dict(iptm=.9, rg_ratio=1.3, sap_norm=.1)], False),
    ("protein-binder", [dict(iptm=.9, rg_ratio=1, sap_norm=.4)], False),
    ("ligand-binder", [dict(iptm=.8, rg_ratio=1)], False),
    ("ligand-binder", [dict(iptm=.9, rg_ratio=1.3)], False),
    ("dual-target", [{"AB.iptm": .8, "AC.iptm": .9}], False),
    ("vhh", [{"AB.iptm": .7, "AC.iptm": .8}], True),
    ("idr", [dict(iptm=.9, target_non_loop=.5, apo_median_non_loop=.5)], False),
    ("cid", [dict(iptm=.9, ca90=.8)], False),
    ("symmetry", [dict(iptm=.6, plddt=.9, ananas_score=.9)], False),
])
def test_endpoint_threshold_operators(tmp_path, endpoint, metric_rows, expected):
    _, units, result = _run(tmp_path, endpoint, [_plan(selection="already_selected")], [_record(**row) for row in metric_rows])
    flag = {"protein-binder": "pass_iptm_rg_sap", "ligand-binder": "pass_compact", "dual-target": "pass_paired", "vhh": "pass_paired"}.get(endpoint, "pass_joint")
    assert units[0][flag] == str(expected)


def test_any_step_and_paired_scores_do_not_combine_different_states(tmp_path):
    records = [_record(step=0, **{"AB.iptm": .99, "AC.iptm": .1}),
               _record(step=1, **{"AB.iptm": .1, "AC.iptm": .99})]
    outputs, units, _ = _run(tmp_path, "dual-target", [_plan(selection="any_step")], records)
    assert units[0]["pass_paired"] == "False"
    with outputs["ccdf_csv"].open() as handle:
        ccdf = list(csv.DictReader(handle))
    assert len(ccdf) == 802  # 401 thresholds × (one group + its equal-group mean)
    assert all(row["fraction"] == "0.0" for row in ccdf if float(row["threshold"]) >= .1)
    records = [_record(step=0, iptm=.99, ca90=.1), _record(step=1, iptm=.1, ca90=.99)]
    _, units, _ = _run(tmp_path, "cid", [_plan()], records)
    assert units[0]["pass_joint"] == "False"


def test_pocket_prefers_occupied_state_even_when_global_iptm_is_higher(tmp_path):
    _, units, _ = _run(tmp_path, "pocket", [_plan()],
        [_record(step=0, iptm=.99, occupancy=.2), _record(step=1, iptm=.81, occupancy=.9)])
    assert float(units[0]["iptm"]) == .81
    assert units[0]["pass_joint"] == "True"


def test_aptamer_selects_one_same_state_within_indices_zero_to_nineteen(tmp_path):
    _, units, result = _run(tmp_path, "aptamer", [_plan()],
        [_record(step=19, weighted_score=2, iptm=.8, plddt=.7),
         _record(step=20, weighted_score=100, iptm=1, plddt=1),
         _record(step=2, weighted_score=1, iptm=.99, plddt=.99)])
    assert float(units[0]["iptm"]) == .8
    assert float(units[0]["plddt"]) == .7
    assert not any(key.startswith("pass_") for key in units[0])
    assert result["groups"][0]["iptm_n"] == 1


def test_cyclic_every_bond_missing_atoms_and_verified_native_flag(tmp_path):
    rows = [_record("closed", step=29, closure_errors="[0.3, -0.3]", closure_missing_atoms=0, iptm=.61),
            _record("one_open", step=29, closure_errors="[0.0, 0.3001]", closure_missing_atoms=0, iptm=.9),
            _record("missing", step=29, closure_missing_atoms=1, iptm=.9)]
    _, units, _ = _run(tmp_path, "cyclic-bound", [_plan(name) for name in ("closed", "one_open", "missing")], rows)
    assert [row["pass_joint"] for row in units] == ["True", "False", "False"]
    _, units, _ = _run(tmp_path, "cyclic-multiring", [_plan(selection="already_selected")],
                       [_record(closure_all_bonds=1, plddt=.8)])
    assert units[0]["pass_closure"] == "True"
    assert units[0]["pass_joint"] == "False"


def test_required_metrics_and_scale_errors_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="required metric 'sap_norm'"):
        _run(tmp_path, "protein-binder", [_plan()], [_record(iptm=.9, rg_ratio=1)])
    with pytest.raises(ValueError, match="0–1 scale"):
        _run(tmp_path, "aptamer", [_plan()], [_record(step=0, weighted_score=1, iptm=.9, plddt=90)])
    with pytest.raises(ValueError, match="closure_missing_atoms"):
        _run(tmp_path, "cyclic-bound", [_plan()], [_record(step=29, closure_errors="[0.1]", iptm=.8)])
    with pytest.raises(ValueError, match="same binder sequence"):
        _run(tmp_path, "dual-target", [_plan(selection="already_selected")],
             [_record(**{"AB.iptm": .9, "AC.iptm": .9, "sequence_ab": "AAA", "sequence_ac": "ACA"})])


def test_method_selection_explicit_stage_and_generation_failure(tmp_path):
    plans = [_plan("complex", selection="complex_max_iptm"),
             _plan("last", selection="last_accepted", stage="oligomer"),
             _plan("failed", selection="any_step", status="generation_failed")]
    records = [_record("complex", step=0, iptm=.99, rg_ratio=1, sap_norm=.1),
               _record("complex", step=1, iptm=.81, rg_ratio=1, sap_norm=.1, **{"AB.iptm": .81}),
               _record("last", step=1, accepted=True, stage="monomer"),
               _record("last", step=2, accepted=True, stage="oligomer", iptm=.9, rg_ratio=1, sap_norm=.1),
               _record("last", step=3, accepted=False, stage="oligomer", iptm=.99, rg_ratio=1, sap_norm=.1),
               _record("failed", step=1, iptm=.99, rg_ratio=1, sap_norm=.1)]
    _, units, _ = _run(tmp_path, "protein-binder", plans, records)
    assert float(units[0]["iptm"]) == .81
    assert float(units[1]["iptm"]) == .9
    assert units[2]["status"] == "generation_failed"
    assert units[2]["pass_iptm"] == "False"


def test_column_mapping_and_vhh_signed_ddg(tmp_path):
    _, units, _ = _run(tmp_path, "vhh", [_plan(selection="already_selected")],
                       [_record(target1=.8, target2=.8, score1=-20, score2=-5)],
                       dict(iptm_ab="target1", iptm_ac="target2", ddg_ab="score1", ddg_ac="score2"))
    assert units[0]["pass_paired_strict"] == "True"
    assert float(units[0]["ddg_ac_minus_ab"]) == 15
    assert float(units[0]["ddg_absolute_difference"]) == 15
    assert units[0]["weaker_ddg_target"] == "AC"


def test_missing_early_variant_is_not_replaced_by_index_twenty(tmp_path):
    records = [_record(variant_index=index, iptm=.7 if index < 20 else .9, rg_ratio=1, sap_norm=.1)
               for index in [*range(19), 20]]
    _, units, result = _run(tmp_path, "protein-binder", [_plan(selection="all_variants")], records)
    assert units[0]["evaluated_variant_count"] == "19"
    assert units[0]["variant_set_complete"] == "False"
    assert units[0]["pass_iptm_rg_sap"] == "False"
    assert result["methods"][0]["planned_units"] == 1
