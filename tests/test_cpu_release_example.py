"""Real-coordinate E2E regression with an independent C-alpha distance answer."""
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFParser


def test_cpu_example_science_and_cli_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "result"
    command = [sys.executable, str(root / "examples/01_cpu_reproduce.py"), "--out-dir", str(out)]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with (out / "raw/9tp_geometry/001_run/scores.csv").open() as handle:
        row = next(csv.DictReader(handle))

    # Reference computation uses a single broadcast distance matrix, independent
    # of GH's residue-pair loop and selector handling.
    model = MMCIFParser(QUIET=True).get_structure("input", root / "examples/input/9TP_cid.cif")[0]
    a = np.asarray([res["CA"].coord for res in model["A"] if "CA" in res], dtype=float)
    b = np.asarray([res["CA"].coord for res in model["B"] if "CA" in res], dtype=float)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    assert distances.shape == (85, 87)
    assert np.count_nonzero(distances <= 8) == 19
    assert float(row["ca_allowed_min_dist"]) == pytest.approx(distances.min(), abs=1e-6)
    assert float(row["ca_contact_satisfied_fraction"]) == pytest.approx(19 / 7395, abs=1e-12)
    assert float(row["aa_allowed_min_dist"]) == pytest.approx(2.3769934158267843, abs=1e-6)
    assert float(row["all_contact_satisfied_fraction"]) == pytest.approx(205 / 7395, abs=1e-12)
    with sqlite3.connect(out / "scores.sqlite") as connection:
        values = connection.execute("SELECT metrics_json FROM records").fetchall()
    assert len(values) == 1
    assert json.loads(values[0][0])["ca_allowed_min_dist"] == pytest.approx(distances.min(), abs=1e-6)
    assert len(list((out / "selected").rglob("*.cif"))) == 1
    with (out / "selected/filter_report.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert (out / "report/posthoc_report.md").is_file()

    # Rerunning into an existing directory must leave the completed result intact.
    refused = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert refused.returncode == 2
    assert "Output already exists" in refused.stderr
