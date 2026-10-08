import csv
import json
from pathlib import Path

import numpy as np
import pytest

from astrolabe.analysis.paper_agreement import analyze_agreement, compare_binding_modes


TARGET = "ACDEF"
BINDER = "GHIK"


def _structure(path, offset=0, *, distal_shift=0, mismatch=False):
    target = np.array([[0, 0, 0], [0, 2, 0], [2, 0, 0], [100, 0, 0], [100, 2, 0]], dtype=float)
    target[3:] += [distal_shift, 0, 0]
    binder = np.array([[0, 0, 4], [0, 2, 4], [2, 0, 4], [2, 2, 4]], dtype=float) + [0, 0, offset]
    # Arbitrary chain labels and residue numbering must not define correspondence.
    target_residues = ["ALA", "CYS", "ASP", "GLU", "PHE"]
    binder_residues = ["GLY", "HIS", "ILE", "ALA" if mismatch else "LYS"]
    records = []
    for chain, residues, coordinates in [("T", target_residues, target), ("Q", binder_residues, binder)]:
        for index, (residue, point) in enumerate(zip(residues, coordinates), 11):
            x, y, z = point
            records.append(f"ATOM  {len(records)+1:5d}  CA  {residue:3s} {chain}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C")
    # This hydrogen would incorrectly make a distant target residue a local anchor.
    records.append(f"ATOM  {len(records)+1:5d}  H   PHE T{15:4d}    {0:8.3f}{0:8.3f}{4:8.3f}  1.00 80.00           H")
    path.write_text("\n".join(records) + "\nEND\n")
    return path


def _read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def _manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(design, side, prediction, *, method="Astrolabe", group="g1", score=.9, comparable=True):
    return {"design_id": design, "method": method, "group": group, "side": side,
            "target_sequence": TARGET, "binder_sequence": BINDER,
            "af3_structure": "ref.pdb", "protenix_structure": prediction,
            "af3_iptm": score, "protenix_iptm": score,
            "comparable": str(comparable),
            "exclusion_reason": "different target constructs" if not comparable else ""}


def test_local_target_alignment_separates_pose_from_fold_and_ignores_distal_motion(tmp_path):
    reference = _structure(tmp_path / "ref.pdb")
    prediction = _structure(tmp_path / "pred.pdb", offset=6, distal_shift=40)
    result = compare_binding_modes(reference, prediction, target_sequence=TARGET, binder_sequence=BINDER)
    assert result["local_target_ca_count"] == 3
    assert result["pose_rmsd"] == pytest.approx(6)
    assert result["fold_rmsd"] == pytest.approx(0, abs=1e-12)
    assert result["contact_recall"] == 0
    # At exactly 5 A, the three coincident target/binder CA pairs still count.
    boundary = _structure(tmp_path / "boundary.pdb", offset=1)
    result = compare_binding_modes(reference, boundary, target_sequence=TARGET, binder_sequence=BINDER)
    assert result["protenix_contacts"] == 3
    assert result["shared_contacts"] == 3


def test_exact_sequence_mismatch_is_rejected(tmp_path):
    reference = _structure(tmp_path / "ref.pdb")
    wrong = _structure(tmp_path / "wrong.pdb", mismatch=True)
    with pytest.raises(ValueError, match="exact sequence matches=0"):
        compare_binding_modes(reference, wrong, target_sequence=TARGET, binder_sequence=BINDER)


def test_complete_two_side_reduction_and_missing_geometry_accounting(tmp_path):
    _structure(tmp_path / "ref.pdb")
    _structure(tmp_path / "near.pdb", offset=1)
    _structure(tmp_path / "far.pdb", offset=6)
    rows = [_row("complete", "AB", "near.pdb", score=.95), _row("complete", "AC", "far.pdb", score=.7),
            _row("missing", "AB", "near.pdb"), _row("missing", "AC", "absent.pdb"),
            _row("excluded", "AB", "absent.pdb", comparable=False), _row("excluded", "AC", "absent.pdb", comparable=False)]
    result = analyze_agreement(_manifest(tmp_path / "manifest.csv", rows), tmp_path / "analysis")
    designs = {r["design_id"]: r for r in _read(result["designs"])}
    assert float(designs["complete"]["pose_rmsd"]) == pytest.approx(6)
    assert float(designs["complete"]["contact_recall"]) == 0
    assert float(designs["complete"]["confidence_AF3"]) == .7
    assert designs["missing"]["geometry_status"] == "incomplete"
    assert designs["missing"]["contact_recall"] == ""
    assert designs["excluded"]["geometry_status"] == "not_comparable"
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["design_status_counts"] == {"complete": 1, "incomplete": 1, "not_comparable": 1}
    # Excluded/missing geometry does not erase observed matched confidence.
    overall = next(r for r in _read(result["confidence_overlap"]) if r["group"] == "ALL")
    assert int(overall["n_matched"]) == 3
    strata = {r["stratum"]: r for r in _read(result["confidence_strata"]) if r["group"] == "ALL"}
    assert int(strata["intermediate"]["n"]) == 1
    assert strata["high"]["median_contact_recall"] == ""


def test_target_equal_statistics_ties_strict_overlap_and_constant_not_estimable(tmp_path):
    _structure(tmp_path / "ref.pdb")
    for offset in [0, 1, 4, 6]:
        _structure(tmp_path / f"p{offset}.pdb", offset=offset)
    rows = []
    for group in ["g1", "g2"]:
        for index, (score, offset) in enumerate(zip([.6, .8, .9, 1], [0, 1, 4, 6])):
            rows.extend(_row(f"a{index}", side, f"p{offset}.pdb", group=group, score=score) for side in ["AB", "AC"])
        # Unequal group sizes expose an erroneous pooling-weighted win statistic.
        for index in range(2 if group == "g1" else 6):
            rows.extend(_row(f"b{index}", side, "p0.pdb", method="Baseline", group=group,
                             score=1 if group == "g1" else 0) for side in ["AB", "AC"])
    for row in rows:
        row["target_pair"] = row.pop("group")
        row["native_af3"] = row.pop("af3_iptm")
    result = analyze_agreement(_manifest(tmp_path / "manifest.csv", rows), tmp_path / "analysis",
                               columns={"group": "target_pair"}, confidence_columns={"AF3": "native_af3"})
    wins = _read(result["win_probabilities"])
    af3 = {r["group"]: r for r in wins if r["predictor"] == "AF3"}
    assert float(af3["g1"]["win_probability"]) == .125  # Exact tie at 1 contributes 0.5.
    assert float(af3["g2"]["win_probability"]) == 1
    assert float(af3["ALL"]["win_probability"]) == .5625
    correlations = _read(result["method_correlations"])
    pose = next(r for r in correlations if r["method"] == "Astrolabe" and r["metric"] == "pose_rmsd")
    assert float(pose["spearman"]) == pytest.approx(1)
    assert int(pose["groups_estimable"]) == 2
    assert 0 < int(pose["valid_bootstrap_replicates"]) < 2000
    assert float(pose["ci_low"]) == pytest.approx(1)
    baseline = [r for r in correlations if r["method"] == "Baseline"]
    assert all(r["status"] == "not_estimable" and r["spearman"] == "" for r in baseline)
    contact = next(r for r in correlations if r["method"] == "Astrolabe" and r["metric"] == "contact_recall")
    assert float(contact["spearman"]) == pytest.approx(-3 / np.sqrt(10))
    overlap = next(r for r in _read(result["confidence_overlap"]) if r["method"] == "Astrolabe" and r["group"] == "ALL")
    assert int(overlap["n_matched"]) == 8
    assert int(overlap["n_both_high"]) == 4  # Exactly 0.8 does not pass.
    strata = {r["stratum"]: int(r["n"]) for r in _read(result["confidence_strata"]) if r["method"] == "Astrolabe" and r["group"] == "ALL"}
    assert strata == {"low": 2, "intermediate": 2, "high": 4}
