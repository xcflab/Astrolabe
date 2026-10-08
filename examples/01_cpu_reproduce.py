#!/usr/bin/env python3
"""Reproduce contact geometry on the bundled AF3 complex; no model inference.

From an installed checkout (Python >=3.11):
    python examples/01_cpu_reproduce.py --out-dir outputs/cpu-example

Input: examples/input/9TP_cid.cif, protein chains A:1-85 and B:1-87.
Output: scores.csv, provenance.json, SQLite collection, CSV/Markdown reports,
and the selected structure. Contact cutoff: 8 Angstrom; no randomness.
The input is a prediction, not an experimental structure or a new design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf

from astrolabe.analysis.constraint_compliance import compute_geometry_metrics
from astrolabe.core.binding_site import parse_binding_site

# %% Explicit, fixed scientific input and a new output directory.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out-dir", type=Path, required=True, help="New output directory; existing results are never overwritten.")
args = parser.parse_args()
source = Path(__file__).resolve().parent / "input" / "9TP_cid.cif"
out = args.out_dir.expanduser().resolve()
if out.exists():
    parser.error(f"Output already exists: {out}. Choose a new --out-dir to preserve previous results.")
specification = {"binder": "A:1-85", "target": "B:1-87", "mode": "contact", "max_distance": 8.0}
spec = parse_binding_site(specification)

# %% Compute existing Astrolabe metrics from real coordinates. No fabricated scores.
metrics = compute_geometry_metrics(source, spec)
# There is no framework subset when all of chain A is allowed; its minimum is
# infinity. Exclude that undefined quantity from the numerical result table.
metrics = {key: value for key, value in metrics.items() if "framework" not in key}
run_dir = out / "raw" / "9tp_geometry" / "001_run"
(run_dir / "structures").mkdir(parents=True)
shutil.copyfile(source, run_dir / "structures" / "step_000_ABC_model.cif")
OmegaConf.save(OmegaConf.create({
    "task_name": "9tp_geometry", "oligo": ["ABC"],
    "analysis": {"kind": "coordinate_geometry", **specification},
}), run_dir / "config.yaml")
row = {"step": 0, "accepted": True, **metrics}
with (run_dir / "scores.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
provenance = {
    "analysis": "Astrolabe contact geometry on existing AF3 prediction; no inference",
    "version": importlib.metadata.version("astrolabe"),
    "python": sys.version,
    "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "biopython", "omegaconf")},
    "input": "examples/input/9TP_cid.cif",
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "parameters": specification, "seed": None, "device": "CPU",
    "state": "geometry_completed",
}
(run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

# %% Exercise the canonical CLI against these computed results.
cli = [sys.executable, "-m", "astrolabe.cli"]
subprocess.run([*cli, "posthoc", "collect", "--out-root", str(out / "raw"),
                "--db", str(out / "scores.sqlite"), "--export-csv", str(out / "scores.csv"),
                "--binder-chain", "A", "--target-chain", "B", "--workers", "1"], check=True)
subprocess.run([*cli, "posthoc", "filter", "--db", str(out / "scores.sqlite"),
                "--expr", "ca_allowed_min_dist <= 8", "--prefer-oligo", "ABC",
                "--dst", str(out / "selected")], check=True)
subprocess.run([*cli, "posthoc", "report", "--db", str(out / "scores.sqlite"),
                "--prefer-oligo", "ABC", "--expr", "ca_allowed_min_dist <= 8",
                "--trend-metric", "ca_allowed_min_dist",
                "--dst", str(out / "report")], check=True)
print(json.dumps({"output": str(out), **metrics}, indent=2, allow_nan=False))
