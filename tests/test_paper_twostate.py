"""Analytic coordinate ensembles test geometry and hierarchical statistics."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from astrolabe.analysis.paper_twostate import analyze_twostate


_COLUMNS = ("design_id", "method", "ligand", "predictor", "state", "seed", "sample",
            "structure_path", "protein_chain", "ligand_chain", "iptm", "protein_plddt", "protein_length")


def _pdb(path, ca, ligand=None, confidence=90):
    lines = []
    for index, (x, y, z) in enumerate(ca, 1):
        lines.append(f"ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{confidence:6.2f}           C  \n")
    if ligand is not None:
        x, y, z = ligand
        lines.append(f"HETATM{len(ca)+1:5d}  C1  LIG B   1    {x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{0:6.2f}           C  \n")
    path.write_text("".join(lines) + "END\n")


def _manifest(tmp_path, rows):
    path = tmp_path / "ensemble.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rows(tmp_path, design, predictor, make_coords, *, rotate=False, confidence=90, make_ligand=None):
    rows = []
    for state in ("apo", "holo"):
        for seed in range(5):
            for sample in range(5):
                coords = make_coords(state, seed, sample).copy()
                ligand = (np.array([0, 0, 0], dtype=float) if make_ligand is None
                          else make_ligand(state, seed, sample)) if state == "holo" else None
                if rotate:
                    angle = (seed * 5 + sample) / 8
                    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                    translation = np.array([seed, sample, seed - sample], dtype=float)
                    coords = coords @ rotation + translation
                    if ligand is not None:
                        ligand = ligand @ rotation + translation
                path = tmp_path / f"{design}_{predictor}_{state}_{seed}_{sample}.pdb"
                _pdb(path, coords, ligand, confidence)
                rows.append(dict(design_id=design, method="Astrolabe", ligand="SAM", predictor=predictor,
                                 state=state, seed=str(seed), sample=str(sample), structure_path=path.name,
                                 protein_chain="A", ligand_chain="B" if state == "holo" else "", iptm="0.9"))
    return rows


def test_displacement_confidence_and_predictors_stay_independent(tmp_path):
    cube = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    coordinates = lambda state, seed, sample: cube * (0.2 if state == "apo" else 1.5)
    rows = _rows(tmp_path, "design1", "AF3", coordinates, rotate=True, confidence=65)
    means = [90, 85, 80, 10, 10]
    iptm = [0.9, 0.85, 0.8, 0.1, 0.1]
    for row in rows:
        seed = int(row["seed"])
        row["protein_plddt"] = str(90 if row["state"] == "apo" else means[seed])
        if row["state"] == "holo":
            row["iptm"] = str(iptm[seed])
    rows += _rows(tmp_path, "design1", "Protenix2", lambda state, seed, sample: cube, rotate=True)
    # An incomplete observed design is retained, with no invented predictions.
    rows.append(dict(rows[-1], design_id="missing", structure_path="absent.pdb"))
    outputs = analyze_twostate(_manifest(tmp_path, rows), tmp_path / "out")
    content = json.loads(outputs["design_json"].read_text())
    results = {(r["design_id"], r["predictor"]): r for r in content["designs"]}
    af3 = results["design1", "AF3"]
    assert af3["status"] == "ok"
    assert af3["holo_plddt"] == 80
    assert af3["holo_iptm"] == pytest.approx(0.8)
    assert af3["apo_plddt"] == 90
    assert af3["displacement_count"] == 8
    assert af3["holo_rg_ratio"] == pytest.approx(np.sqrt(3) * 1.5 / (2.2 * 8**0.38), abs=2e-4)
    assert all(value for name, value in af3.items() if name.startswith("cumulative_"))
    assert af3["holo_scalar_plddt_rows"] == 25
    assert results["design1", "Protenix2"]["displacement_count"] == 0
    missing = results["missing", "Protenix2"]
    assert missing["status"] == "incomplete_ensemble"
    assert missing["displacement_count"] is None
    assert not missing["cumulative_displacement"]
    assert content["prediction_rows"] == 101
    assert content["implementation_conventions"]["kabsch_max_iterations"] == 100
    with outputs["residue_csv"].open() as handle:
        residues = list(csv.DictReader(handle))
    assert len(residues) == 16
    assert all(float(r["holo_plddt"]) == 65 for r in residues if r["predictor"] == "AF3")
    assert all(float(r["apo_rmsf"]) < 0.002 for r in residues)


def test_undefined_geometry_stays_in_output_and_keeps_confidence(tmp_path):
    line = np.column_stack([np.arange(8), np.zeros(8), np.zeros(8)])
    rows = _rows(tmp_path, "collinear", "AF3", lambda state, seed, sample: line)
    result = json.loads(analyze_twostate(_manifest(tmp_path, rows), tmp_path / "out")["design_json"].read_text())["designs"][0]
    assert result["ensemble_complete"]
    assert result["status"] == "geometry_undefined"
    assert "non-collinear" in result["reason"]
    assert result["cumulative_holo_iptm"]
    assert result["displacement_count"] is None
    assert result["ordering_residue_count"] is None
    assert not result["cumulative_displacement"]


def test_ordering_regions_require_seed_consensus_not_total_contact_votes(tmp_path):
    anchors = np.array(list(itertools.product([-10, 10], repeat=3)), dtype=float)
    repeat = np.array([[0, 0, -2], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)

    def coords(state, seed, sample):
        mobile = repeat.copy()
        mobile[:, 1] += (5 if state == "holo" else [-6, -3, 0, 3, 6][sample])
        return np.concatenate([anchors, mobile])

    rows = []
    for design in ("consensus", "concentrated"):
        def ligand(state, seed, sample):
            near = (seed < 4 and sample < 3) if design == "consensus" else seed < 3
            return np.array([0, 5, 0] if near else [100, 100, 100], dtype=float)
        rows += _rows(tmp_path, design, "AF3", coords, rotate=True, make_ligand=ligand)
    outputs = analyze_twostate(_manifest(tmp_path, rows), tmp_path / "out")
    designs = {r["design_id"]: r for r in json.loads(outputs["design_json"].read_text())["designs"]}
    assert designs["consensus"]["geometry_defined"]
    assert designs["consensus"]["ordering_region_count"] == 1
    assert designs["consensus"]["ordering_residue_count"] == 5
    assert designs["concentrated"]["ordering_region_count"] == 0
    assert designs["concentrated"]["ordering_residue_count"] == 0
    with outputs["residue_csv"].open() as handle:
        residues = list(csv.DictReader(handle))
    candidates = [r for r in residues if r["ordering_candidate"] == "True"]
    assert len(candidates) == 10
    for row in candidates:
        assert float(row["apo_rmsf"]) > 3
        assert float(row["holo_rmsf"]) <= 2
        assert float(row["displacement"]) > 3
        assert row["consensus_ligand_contact"] == str(row["design_id"] == "consensus")


def test_duplicate_prediction_identity_is_rejected(tmp_path):
    row = dict(design_id="duplicate", method="Astrolabe", ligand="SAM", predictor="AF3",
               state="apo", seed="0", sample="0", structure_path="unused.pdb", protein_chain="A", ligand_chain="", iptm="")
    with pytest.raises(ValueError, match="Duplicate prediction identity"):
        analyze_twostate(_manifest(tmp_path, [row, row]), tmp_path / "out")


def test_shared_residues_use_all_fifty_structures_with_explicit_length(tmp_path):
    from Bio.PDB import MMCIFIO, PDBParser

    cube = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    rows = _rows(tmp_path, "shared", "AF3", lambda state, seed, sample: cube)
    # One prediction lacks residue 8; a complete prediction is encoded as mmCIF.
    incomplete = tmp_path / rows[-1]["structure_path"]
    incomplete.write_text("".join(line for line in incomplete.read_text().splitlines(keepends=True)
                                 if not (line.startswith("ATOM") and int(line[22:26]) == 8)))
    source = tmp_path / rows[0]["structure_path"]
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("prediction", str(source)))
    writer.save(str(source.with_suffix(".cif")))
    rows[0]["structure_path"] = source.with_suffix(".cif").name
    for row in rows:
        row["protein_length"] = "8"
    outputs = analyze_twostate(_manifest(tmp_path, rows), tmp_path / "out")
    result = json.loads(outputs["design_json"].read_text())["designs"][0]
    assert result["status"] == "ok"
    assert result["shared_residues"] == 7
    assert result["protein_length"] == 8
    assert result["displacement_count"] == 0
    with outputs["residue_csv"].open() as handle:
        assert {int(row["residue_number"]) for row in csv.DictReader(handle)} == set(range(1, 8))
