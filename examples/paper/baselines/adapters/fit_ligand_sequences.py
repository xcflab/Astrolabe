#!/usr/bin/env python3
"""Fit five native LigandMPNN sequences to each RF-family backbone.

The sampling flags follow LigandMPNN run.py. The 100-backbone / five-sequence
counts follow the Astrolabe MainPaper ligand benchmark, not an inferred archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def build_commands(outputs, *, python, tool_dir, checkpoint, chains, out_dir, seed):
    if len(outputs) != 100:
        raise ValueError(f"The paper recipe requires 100 backbones, received {len(outputs)}")
    names = [Path(item["structure_path"]).stem for item in outputs]
    if len(set(names)) != 100:
        raise ValueError("Backbone names must be unique within the ligand cohort")
    return [
        [str(python), str(Path(tool_dir) / "run.py"), "--model_type", "ligand_mpnn",
         "--checkpoint_ligand_mpnn", str(checkpoint), "--pdb_path", str(item["structure_path"]),
         "--out_folder", str(Path(out_dir) / name), "--chains_to_design", str(chains),
         "--batch_size", "5", "--number_of_batches", "1", "--seed", str(int(seed) + index)]
        for index, (name, item) in enumerate(zip(names, outputs))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--tool-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    outputs = json.loads(args.backbone_manifest.read_text())["outputs"]
    for output in outputs:
        if not Path(output["structure_path"]).is_file():
            raise FileNotFoundError(output["structure_path"])
    commands = build_commands(outputs, python=args.python, tool_dir=args.tool_dir,
                              checkpoint=args.checkpoint, chains=args.chains,
                              out_dir=args.out_dir, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "native_sequence_commands.json").write_text(
        json.dumps({"backbones": 100, "sequences_per_backbone": 5, "commands": commands}, indent=2) + "\n"
    )
    for command in commands:
        subprocess.run(command, cwd=args.tool_dir, check=True)


if __name__ == "__main__":
    main()
