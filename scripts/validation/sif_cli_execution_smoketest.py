#!/usr/bin/env python3
"""Run one non-help acceptance case for every canonical ``gh`` CLI leaf."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


CONTAINER_REPO_ROOT = Path("/opt/Astrolabe")
CONTAINER_WORK_ROOT = Path("/gh-cli-smoke")
CONTAINER_SCRIPT = Path("/tmp/sif_cli_execution_smoketest.py")


@dataclass(frozen=True)
class Case:
    path: tuple[str, ...]
    argv: tuple[str, ...]
    expected_returncode: int = 0
    require_stdout: bool = False
    stderr_contains: str | None = None
    artifacts: tuple[str, ...] = ()
    allow_empty_artifacts: bool = False
    json_stdout: bool = False
    timeout_seconds: float = 180.0


def _detect_engine() -> str | None:
    for name in ("apptainer", "singularity"):
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _canonical_children(
    action: argparse._SubParsersAction,
) -> list[tuple[str, argparse.ArgumentParser]]:
    seen: set[int] = set()
    children: list[tuple[str, argparse.ArgumentParser]] = []
    for choice_action in getattr(action, "_choices_actions", ()):
        name = str(getattr(choice_action, "dest", "") or "")
        child = action.choices.get(name)
        if not name or child is None or id(child) in seen:
            continue
        seen.add(id(child))
        children.append((name, child))
    for name, child in action.choices.items():
        if id(child) in seen:
            continue
        seen.add(id(child))
        children.append((str(name), child))
    return children


def _canonical_leaf_paths() -> set[tuple[str, ...]]:
    from astrolabe import cli

    leaves: set[tuple[str, ...]] = set()

    def visit(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        subparsers = _subparser_action(parser)
        if subparsers is None:
            leaves.add(path)
            return
        for name, child in _canonical_children(subparsers):
            visit(child, (*path, name))

    visit(cli._build_parser(), ())
    return leaves


def _pdb_atom(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
    *,
    record: str = "ATOM",
    occupancy: float = 1.0,
    bfactor: float = 20.0,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:<4} {resname:>3} {chain_id:1}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2}"
    )


def _write_protein_pdb(path: Path, chains: dict[str, tuple[str, ...]]) -> None:
    lines: list[str] = []
    serial = 1
    chain_offset = 0.0
    for chain_id, residues in chains.items():
        for index, resname in enumerate(residues, start=1):
            base = 3.8 * (index - 1)
            atoms = (
                ("N", base, chain_offset, 0.0, "N"),
                ("CA", base + 1.45, chain_offset + 0.2, 0.0, "C"),
                ("C", base + 2.45, chain_offset, 0.0, "C"),
                ("O", base + 3.15, chain_offset - 0.4, 0.0, "O"),
            )
            for atom_name, x, y, z, element in atoms:
                lines.append(
                    _pdb_atom(
                        serial,
                        atom_name,
                        resname,
                        chain_id,
                        index,
                        x,
                        y,
                        z,
                        element,
                    )
                )
                serial += 1
        lines.append("TER")
        chain_offset += 12.0
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_channel_pdb(path: Path, *, screen: bool) -> None:
    residues = ("ALA", "CYS", "GLY", "SER")
    lines: list[str] = []
    serial = 1
    for index, resname in enumerate(residues, start=1):
        base = 3.8 * (index - 1)
        atoms = [
            ("N", base, 0.0, 0.0, "N"),
            ("CA", base + 1.45, 0.2, 0.0, "C"),
            ("C", base + 2.45, 0.0, 0.0, "C"),
            ("O", base + 3.15, -0.4, 0.0, "O"),
        ]
        if resname == "CYS":
            atoms.append(("SG", base + 1.8, 1.8, 0.0, "S"))
        for atom_name, x, y, z, element in atoms:
            lines.append(
                _pdb_atom(
                    serial,
                    atom_name,
                    resname,
                    "A",
                    index,
                    x,
                    y,
                    z,
                    element,
                )
            )
            serial += 1
    if screen:
        lines.append(
            _pdb_atom(
                serial,
                "O",
                "HOH",
                "L",
                101,
                5.9,
                2.0,
                0.0,
                "O",
                record="HETATM",
            )
        )
    else:
        ligand_serial = serial
        lines.append(
            _pdb_atom(
                serial,
                "C1",
                "LIG",
                "L",
                101,
                5.9,
                2.0,
                0.0,
                "C",
                record="HETATM",
            )
        )
        serial += 1
        lines.append(
            _pdb_atom(
                serial,
                "O1",
                "LIG",
                "L",
                101,
                6.8,
                2.0,
                0.0,
                "O",
                record="HETATM",
            )
        )
        lines.append(f"CONECT    9{ligand_serial:5d}")
        lines.append(f"CONECT{ligand_serial:5d}    9{serial:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _prepare_fixtures(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    _write_protein_pdb(
        work_root / "protein.pdb",
        {"A": ("ALA", "CYS", "GLY", "SER")},
    )
    _write_protein_pdb(
        work_root / "complex.pdb",
        {
            "A": ("ALA", "CYS", "GLY", "SER"),
            "B": ("GLY", "ALA", "SER", "GLY"),
        },
    )
    _write_channel_pdb(work_root / "channel.pdb", screen=False)
    _write_channel_pdb(work_root / "channel_screen.pdb", screen=True)
    (work_root / "mutable.txt").write_text("2-4\n", encoding="ascii")
    (work_root / "spheres.pdb").write_text(
        "\n".join(
            [
                _pdb_atom(1, "X", "SPH", "S", 1, 6.0, 0.0, -10.0, "C", record="HETATM", occupancy=2.0),
                _pdb_atom(2, "X", "SPH", "S", 2, 6.0, 0.0, 0.0, "C", record="HETATM", occupancy=1.0),
                _pdb_atom(3, "X", "SPH", "S", 3, 6.0, 0.0, 10.0, "C", record="HETATM", occupancy=2.0),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    (work_root / "radius.rad").write_text("remark smoke radius file\n", encoding="ascii")
    hole = work_root / "hole-smoke"
    hole.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "cat >/dev/null\n"
        "printf ' cenxyz.cvec\n-30.0 2.0 0 0\n0.0 1.0 0 0\n30.0 2.0 0 0\n Minimum radius found: 1.0\n'\n",
        encoding="ascii",
    )
    hole.chmod(0o755)
    (work_root / "preset.yaml").write_text(
        "name: smoke\nrules:\n  - key: weighted_score\n    op: ge\n    value: 0\n",
        encoding="ascii",
    )
    (work_root / "shared_hotspots.yaml").write_text(
        "shared_hotspots: {}\n", encoding="ascii"
    )
    (work_root / "empty_manifest.jsonl").write_text("", encoding="ascii")
    (work_root / "env.yaml").write_text("paths: {}\n", encoding="ascii")

    run_dir = work_root / "posthoc_outputs" / "smoke_task" / "001_run"
    structures = run_dir / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work_root / "complex.pdb", structures / "step_000_AB_model.pdb")
    (run_dir / "scores.csv").write_text(
        "step,stage,weighted_score,binder_ptm\n0,default,1.0,0.9\n",
        encoding="ascii",
    )
    (run_dir / "config.yaml").write_text(
        "seqs: {A: ACDE, B: FGHI}\n"
        "types: {A: protein, B: protein}\n"
        "homologies: {A: null, B: null}\n",
        encoding="ascii",
    )
    _prepare_paper_fixtures(work_root)


def _prepare_paper_fixtures(work_root: Path) -> None:
    """Small analytic inputs exercise the CLI; these are not model predictions."""
    root = work_root / "paper"
    root.mkdir(parents=True, exist_ok=True)

    def table(name: str, rows: list[dict]) -> None:
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (root / "evaluation.yaml").write_text(
        "seqs: {A: ACGS, B: GASG}\ntypes: {A: protein, B: protein}\n"
        "homologies: {A: null, B: null}\n", encoding="ascii"
    )
    (root / "baseline.yaml").write_text(
        "settings: settings.json\nfilters: filters.json\nadvanced: advanced.json\n",
        encoding="ascii",
    )
    (root / "settings.json").write_text(json.dumps({
        "design_path": "unused", "starting_pdb": str(work_root / "protein.pdb"),
        "lengths": [80, 120],
    }), encoding="ascii")
    for name in ("filters.json", "advanced.json"):
        (root / name).write_text("{}\n", encoding="ascii")

    (root / "run").mkdir(exist_ok=True)
    table("run/scores.csv", [dict(step=0, accepted=True, weighted_score=1.0,
                                 iptm=0.9, rg_ratio=1.1, sap_norm=0.2)])
    table("planned.csv", [dict(unit_id=unit, method="Astrolabe", group="target",
                               run_dir=run, selection="already_selected")
                          for unit, run in (("observed", "run"), ("missing", "absent_run"))])
    table("ligands.csv", [dict(unit_id="design1", method=method, group=ligand,
                               iptm=0.9 - offset, ptm=0.9 - offset, plddt=0.9 - offset,
                               ddg=-20 + offset, dsasa=1000 - offset,
                               ligand_hbonds=4 - offset, sc=0.8 - offset)
                          for ligand in ("FAD", "IAI", "OQO", "SAM")
                          for method, offset in (("Astrolabe", 0), ("Comparator", 0.1))])
    table("ddg.csv", [dict(unit_id=f"design{index}", method="Astrolabe", group="pair",
                           condition="dual", molecule_type="minibinder", ddg_ab=-20, ddg_ac=value)
                      for index, value in enumerate((-18, -16), 1)])

    for state in ("apo", "holo"):
        atoms = [_pdb_atom(index, "CA", "ALA", "A", index, *point, "C", bfactor=90)
                 for index, point in enumerate(itertools.product((-1.0, 1.0), repeat=3), 1)]
        if state == "holo":
            atoms.append(_pdb_atom(9, "C1", "LIG", "B", 1, 0, 0, 0, "C", record="HETATM"))
        (root / f"{state}.pdb").write_text("\n".join(atoms) + "\nEND\n", encoding="ascii")
    table("ensemble.csv", [dict(design_id="design1", method="Astrolabe", ligand="SAM", predictor="AF3",
                                state=state, seed=seed, sample=sample, structure_path=f"{state}.pdb",
                                protein_chain="A", ligand_chain="B" if state == "holo" else "",
                                iptm=0.9 if state == "holo" else "", protein_plddt=90, protein_length=8)
                           for state in ("apo", "holo") for seed in range(5) for sample in range(5)])

    target = [(0, 0, 0), (0, 2, 0), (2, 0, 0), (100, 0, 0), (100, 2, 0)]
    binder = [(0, 0, 4), (0, 2, 4), (2, 0, 4), (2, 2, 4)]
    for name, offset in (("reference", 0), ("prediction", 1)):
        atoms = []
        for chain, names, points in (("T", ("ALA", "CYS", "ASP", "GLU", "PHE"), target),
                                      ("Q", ("GLY", "HIS", "ILE", "LYS"), binder)):
            for index, (residue, (x, y, z)) in enumerate(zip(names, points), 1):
                atoms.append(_pdb_atom(len(atoms) + 1, "CA", residue, chain, index,
                                       x, y, z + (offset if chain == "Q" else 0), "C"))
        (root / f"agreement_{name}.pdb").write_text("\n".join(atoms) + "\nEND\n", encoding="ascii")
    table("agreement.csv", [dict(design_id="design1", method="Astrolabe", group="pair", side=side,
                                 target_sequence="ACDEF", binder_sequence="GHIK",
                                 af3_structure="agreement_reference.pdb", protenix_structure="agreement_prediction.pdb",
                                 af3_iptm=0.9, protenix_iptm=0.85) for side in ("AB", "AC")])

    atoms = [_pdb_atom(index, "CA", "ALA", "B", residue, *point, "C")
             for index, (residue, point) in enumerate(zip(
                 (1, 2, 128, 139, 140, 343),
                 ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 30), (2, 0, 30), (0, 2, 30)),
             ), 1)]
    (root / "cre.pdb").write_text("\n".join(atoms) + "\nEND\n", encoding="ascii")
    atoms = []
    for index, chain in enumerate("ABCD"):
        angle = index * math.pi / 2
        for residue, (x, y, z) in enumerate(((8, 0, 0), (9, 1, 2), (8, 2, 4)), 1):
            atoms.append(_pdb_atom(len(atoms) + 1, "C4'", "A", chain, residue,
                                   x * math.cos(angle) - y * math.sin(angle),
                                   x * math.sin(angle) + y * math.cos(angle), z, "C"))
    (root / "rna.pdb").write_text("\n".join(atoms) + "\nEND\n", encoding="ascii")


def _case_specs(work_root: Path, repo_root: Path) -> list[Case]:
    p = lambda relative: str(work_root / relative)
    config = str(repo_root / "config")

    def configured(*argv: str) -> tuple[str, ...]:
        return ("--config-dir", config, *argv)

    cases = [
        Case(("run",), configured("run", "--task", "benchmark_binder_v1_4", "--dry-run"), require_stdout=True),
        Case(("completion", "bash"), ("completion", "bash", "--init"), require_stdout=True),
        Case(("completion", "zsh"), ("completion", "zsh", "--init"), require_stdout=True),
        Case(("completion", "complete"), ("completion", "complete", "--shell", "bash", "--cword", "1", "astrolabe", ""), require_stdout=True),
        Case(("experimental", "candidate-batch"), ("experimental", "candidate-batch", "--config-yaml", str(repo_root / "config" / "container_smoke.yaml"), "--out-root", p("candidate_batch"), "--runner", "/bin/false", "--num-candidates", "1", "--min-completed", "1", "--max-wait-s", "1", "--decision-rounds", "1", "--candidate-steps", "1", "--dry-run", "--format", "json"), require_stdout=True, json_stdout=True),
        Case(("bench-binder-v1-4",), configured("bench-binder-v1-4", "-t", "cli_binder", "-o", p("out_binder"), "--binder-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("binder.yaml")), artifacts=(p("binder.yaml"),)),
        Case(("bench-idr-binder-v1",), configured("bench-idr-binder-v1", "-t", "cli_idr", "-o", p("out_idr"), "--binder-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("idr.yaml")), artifacts=(p("idr.yaml"),)),
        Case(("bench-peptide-binder-v1",), configured("bench-peptide-binder-v1", "-t", "cli_peptide", "-o", p("out_peptide"), "--peptide-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("peptide.yaml")), artifacts=(p("peptide.yaml"),)),
        Case(("bench-aptamer-v1",), configured("bench-aptamer-v1", "-t", "cli_aptamer", "-o", p("out_aptamer"), "--aptamer-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("aptamer.yaml")), artifacts=(p("aptamer.yaml"),)),
        Case(("bench-antibody-v1",), configured("bench-antibody-v1", "-t", "cli_antibody", "-o", p("out_antibody"), "--binder-seq", "+AAAA 3-3 +BBBB", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("antibody.yaml")), artifacts=(p("antibody.yaml"),)),
        Case(("design", "protein-binder"), configured("design", "protein-binder", "-t", "cli_design_binder", "-o", p("out_design_binder"), "--binder-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_binder.yaml")), artifacts=(p("design_binder.yaml"),)),
        Case(("design", "idr-binder"), configured("design", "idr-binder", "-t", "cli_design_idr", "-o", p("out_design_idr"), "--binder-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_idr.yaml")), artifacts=(p("design_idr.yaml"),)),
        Case(("design", "multidomain-binder"), configured("design", "multidomain-binder", "-t", "cli_multidomain", "-o", p("out_multidomain"), "--variant", "af3", "--target-name", "smoke", "--binder-len", "8", "--target-seq", "ACDEFGHIKL", "--domain-indices", "1-2", "--domain-indices", "8-9", "--dry-run", "--dump-config", p("multidomain.yaml")), artifacts=(p("multidomain.yaml"),)),
        Case(("design", "oligomer"), configured("design", "oligomer", "-t", "cli_oligomer", "-o", p("out_oligomer"), "--shape", "C2", "--protein-len", "8", "--dry-run", "--dump-config", p("oligomer.yaml")), artifacts=(p("oligomer.yaml"),)),
        Case(("design", "multitarget-binder"), configured("design", "multitarget-binder", "--task-name", "cli_multitarget", "--out-dir", p("out_multitarget"), "--variant", "dplm-protenix", "--pair-name", "smoke", "--shared-hotspot-manifest", p("shared_hotspots.yaml"), "--binder-len", "8", "--target-b-seq", "ACDE", "--target-b-hotspots", "1", "--target-c-seq", "FGHI", "--target-c-hotspots", "2", "--dry-run", "--dump-config", p("multitarget.yaml")), artifacts=(p("multitarget.yaml"),)),
        Case(("design", "peptide"), configured("design", "peptide", "-t", "cli_design_peptide", "-o", p("out_design_peptide"), "--peptide-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_peptide.yaml")), artifacts=(p("design_peptide.yaml"),)),
        Case(("design", "aptamer"), configured("design", "aptamer", "-t", "cli_design_aptamer", "-o", p("out_design_aptamer"), "--aptamer-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_aptamer.yaml")), artifacts=(p("design_aptamer.yaml"),)),
        Case(("design", "dna-aptamer"), configured("design", "dna-aptamer", "-t", "cli_design_dna", "-o", p("out_design_dna"), "--aptamer-len", "8", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_dna.yaml")), artifacts=(p("design_dna.yaml"),)),
        Case(("design", "antibody"), configured("design", "antibody", "-t", "cli_design_antibody", "-o", p("out_design_antibody"), "--binder-seq", "+AAAA 3-3 +BBBB", "--target-seq", "ACDEFGHIK", "--dry-run", "--dump-config", p("design_antibody.yaml")), artifacts=(p("design_antibody.yaml"),)),
        Case(("design", "ligand-binder"), configured("design", "ligand-binder", "-t", "cli_ligand", "-o", p("out_ligand"), "--protein-len", "8", "--ligand", "CCO", "--ligand-type", "smiles", "--dry-run", "--dump-config", p("ligand.yaml")), artifacts=(p("ligand.yaml"),)),
        Case(("design", "cid-generate"), configured("design", "cid-generate", "-t", "cli_cid", "-o", p("out_cid"), "--protein-a-len", "8", "--protein-b-len", "8", "--ligand", "CCO", "--ligand-type", "smiles", "--dry-run", "--dump-config", p("cid.yaml")), artifacts=(p("cid.yaml"),)),
        Case(("refine-oligomer-v3",), configured("refine-oligomer-v3", "-t", "cli_refine_v3", "-o", p("out_refine_v3"), "--seq", "ACDE", "--copies", "2", "--dry-run", "--dump-config", p("refine_v3.yaml")), artifacts=(p("refine_v3.yaml"),)),
        Case(("refine", "binder"), configured("refine", "binder", "-t", "cli_refine_binder", "-o", p("out_refine_binder"), "--structure", p("complex.pdb"), "--target-chain", "B", "--binder-chains", "A", "--dry-run", "--dump-config", p("refine_binder.yaml")), artifacts=(p("refine_binder.yaml"),)),
        Case(("refine", "ligand"), configured("refine", "ligand", "-t", "cli_refine_ligand", "-o", p("out_refine_ligand"), "--structure", p("protein.pdb"), "--protein-chain", "A", "--ligand", "CCO", "--ligand-type", "smiles", "--dry-run", "--dump-config", p("refine_ligand.yaml")), artifacts=(p("refine_ligand.yaml"),)),
        Case(("refine", "ligand-pocket"), configured("refine", "ligand-pocket", "-t", "cli_refine_pocket", "-o", p("out_refine_pocket"), "--structure", p("protein.pdb"), "--protein-chain", "A", "--ligand", "CCO", "--ligand-type", "smiles", "--dry-run", "--dump-config", p("refine_pocket.yaml")), artifacts=(p("refine_pocket.yaml"),)),
        Case(("refine", "channel"), configured("refine", "channel", "-t", "cli_refine_channel", "-o", p("out_refine_channel"), "--structure", p("channel.pdb"), "--protein-chain", "A", "--ligand-residue", "L:LIG:101", "--mutable-positions", p("mutable.txt"), "--spheres", p("spheres.pdb"), "--dry-run", "--dump-config", p("refine_channel.yaml")), artifacts=(p("refine_channel.yaml"),)),
        Case(("refine", "oligomer"), configured("refine", "oligomer", "-t", "cli_refine_oligomer", "-o", p("out_refine_oligomer"), "--seq", "ACDE", "--copies", "2", "--dry-run", "--dump-config", p("refine_oligomer.yaml")), artifacts=(p("refine_oligomer.yaml"),)),
        Case(("bench-pseudocycle-v1",), configured("bench-pseudocycle-v1", "--task-name", "cli_pseudocycle", "--out-dir", p("out_pseudocycle"), "--unit-len", "8", "--copies", "2", "--dry-run", "--dump-config", p("pseudocycle.yaml")), artifacts=(p("pseudocycle.yaml"),)),
        Case(("bench-tandem-repeat-v1",), configured("bench-tandem-repeat-v1", "--task-name", "cli_tandem", "--out-dir", p("out_tandem"), "--unit-len", "8", "--copies", "2", "--dry-run", "--dump-config", p("tandem.yaml")), artifacts=(p("tandem.yaml"),)),
        Case(("refine-heterooligomer-v1",), configured("refine-heterooligomer-v1", "--task-name", "cli_hetero", "--out-dir", p("out_hetero"), "--seq", "ACDEFG", "--copies", "2", "--interface", "2", "--dry-run", "--dump-config", p("hetero.yaml")), artifacts=(p("hetero.yaml"),)),
        Case(("posthoc", "collect"), ("posthoc", "collect", "--out-root", p("posthoc_outputs"), "--db", p("scores.sqlite"), "--workers", "1", "--format", "json"), artifacts=(p("scores.sqlite"),), json_stdout=True),
        Case(("posthoc", "channel-screen"), ("posthoc", "channel-screen", "--structure", p("channel_screen.pdb"), "--protein-chain", "A", "--ligand-residue", "L:HOH:101", "--mutable-positions", p("mutable.txt"), "--spheres", p("spheres.pdb"), "--out-dir", p("channel_screen_out"), "--target-aas", "G", "--contact-distance", "2.5", "--hole-executable", p("hole-smoke"), "--hole-radius-file", p("radius.rad"), "--format", "json"), require_stdout=True, artifacts=(p("channel_screen_out/03_collect/pocket_fixed_screen_summary.json"),), timeout_seconds=300.0),
        Case(("posthoc", "filter"), ("posthoc", "filter", "--preset", p("preset.yaml"), "--preview-preset", "--format", "json"), json_stdout=True),
        Case(("posthoc", "prune"), ("posthoc", "prune", "--db", p("scores.sqlite"), "--expr", "weighted_score < 0", "--format", "json"), json_stdout=True),
        Case(("posthoc", "report"), ("posthoc", "report", "--db", p("scores.sqlite"), "--expr", "weighted_score >= 0", "--dst", p("posthoc_report"), "--format", "json"), artifacts=(p("posthoc_report/posthoc_summary.csv"), p("posthoc_report/posthoc_report.md"), p("posthoc_report/step_trends.png"), p("posthoc_report/success_rate.png")), json_stdout=True, timeout_seconds=300.0),
        Case(("posthoc", "refold"), ("posthoc", "refold", "--db", p("scores.sqlite"), "--expr", "weighted_score < 0", "--dst", p("refold"), "--dry-run", "--format", "json"), json_stdout=True),
        Case(("posthoc", "refold-export"), ("posthoc", "refold-export", "--db", p("scores.sqlite"), "--expr", "weighted_score < 0", "--out-manifest", p("export.jsonl"), "--format", "json"), artifacts=(p("export.jsonl"),), allow_empty_artifacts=True, json_stdout=True),
        Case(("posthoc", "refold-run"), ("posthoc", "refold-run", "--manifest", p("empty_manifest.jsonl"), "--out-root", p("refold_out"), "--dry-run", "--format", "json"), json_stdout=True),
        Case(("config", "wizard"), ("config", "wizard"), expected_returncode=2, stderr_contains="interactive TTY"),
        Case(("config", "scaffold"), configured("config", "scaffold", "--task", "benchmark_binder_v1_4"), expected_returncode=2, stderr_contains="interactive TTY"),
        Case(("config", "compose"), configured("config", "compose", "--base", "base", "--name", "smoke"), expected_returncode=2, stderr_contains="interactive TTY"),
        Case(("env", "wizard"), configured("env", "wizard", "--task", "benchmark_binder_v1_4"), expected_returncode=2, stderr_contains="interactive TTY"),
        Case(("env", "doctor"), ("env", "doctor", "--env", p("env.yaml"), "--format", "json", "--soft-fail"), json_stdout=True),
        Case(("paper", "list"), ("paper", "list", "--format", "json"), json_stdout=True),
        Case(("paper", "show"), configured("paper", "show", "evaluate-af3", "--config-env", "env"), require_stdout=True),
        Case(("paper", "prepare-inputs"),
             ("paper", "prepare-inputs", "--bundle", "interventions", "-o", p("paper/bundled")),
             json_stdout=True, artifacts=(p("paper/bundled/interventions/pocket_catm/reference.pdb"),)),
        Case(("paper", "run"),
             configured("paper", "run", "evaluate-af3", "--config-env", "env", "--inputs", p("paper/evaluation.yaml"),
                        "-o", p("paper/evaluation"), "--dry-run", "--dump-config", p("paper/evaluation_resolved.yaml")),
             require_stdout=True, artifacts=(p("paper/evaluation_resolved.yaml"),)),
        Case(("paper", "baseline"),
             ("paper", "baseline", "bindcraft", "--inputs", p("paper/baseline.yaml"),
              "--tool-dir", str(repo_root / "vendor" / "bindcraft"), "-o", p("paper/baseline"), "--dry-run"),
             json_stdout=True),
        Case(("paper", "collect"),
             ("paper", "collect", "--planned", p("paper/planned.csv"), "--output", p("paper/records.csv")),
             json_stdout=True, artifacts=(p("paper/records.csv"),)),
        Case(("paper", "analyze", "scores"),
             ("paper", "analyze", "scores", "--endpoint", "protein-binder", "--planned", p("paper/planned.csv"),
              "--records", p("paper/records.csv"), "-o", p("paper/scores")),
             json_stdout=True, artifacts=(p("paper/scores/units.csv"), p("paper/scores/overall.json"))),
        Case(("paper", "analyze", "ligand-rank"),
             ("paper", "analyze", "ligand-rank", "--records", p("paper/ligands.csv"), "-o", p("paper/ligand_rank")),
             json_stdout=True, artifacts=(p("paper/ligand_rank/overall_ranks.csv"), p("paper/ligand_rank/ligand_ranking.json"))),
        Case(("paper", "analyze", "sensitivity"),
             ("paper", "analyze", "sensitivity", "--endpoint", "protein-binder", "--planned", p("paper/planned.csv"),
              "--records", p("paper/records.csv"), "--denominator", "planned", "-o", p("paper/sensitivity")),
             json_stdout=True, artifacts=(p("paper/sensitivity/threshold_curves.csv"), p("paper/sensitivity/threshold_sensitivity.json"))),
        Case(("paper", "analyze", "ddg"),
             ("paper", "analyze", "ddg", "--records", p("paper/ddg.csv"), "--bootstrap-samples", "32", "--seed", "0", "-o", p("paper/ddg")),
             json_stdout=True, artifacts=(p("paper/ddg/ddg_designs.csv"), p("paper/ddg/ddg_bootstrap.json"))),
        Case(("paper", "analyze", "agreement"),
             ("paper", "analyze", "agreement", "--manifest", p("paper/agreement.csv"), "--bootstrap-samples", "32", "--seed", "0", "-o", p("paper/agreement")),
             json_stdout=True, artifacts=(p("paper/agreement/02_designs.csv"), p("paper/agreement/08_metadata.json"))),
        Case(("paper", "analyze", "two-state"),
             ("paper", "analyze", "two-state", "--manifest", p("paper/ensemble.csv"), "-o", p("paper/two_state")),
             json_stdout=True, artifacts=(p("paper/two_state/per_design.json"), p("paper/two_state/per_residue.csv"))),
        Case(("paper", "analyze", "cre"),
             ("paper", "analyze", "cre", "--reference", p("paper/cre.pdb"), "--structure", p("paper/cre.pdb"), "--reference-chain", "B", "--target-chain", "B"),
             json_stdout=True),
        Case(("paper", "analyze", "rna"),
             ("paper", "analyze", "rna", "--kind", "oligomer", "--structure", p("paper/rna.pdb")),
             json_stdout=True),
    ]
    return cases


def _invoke_case(case: Case, *, repo_root: Path, env: dict[str, str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "astrolabe.cli", *case.argv]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=case.timeout_seconds,
            check=False,
        )
        return {
            "path": list(case.path),
            "command": command,
            "returncode": proc.returncode,
            "seconds": round(time.monotonic() - started, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "path": list(case.path),
            "command": command,
            "returncode": None,
            "seconds": round(time.monotonic() - started, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout": case.timeout_seconds,
        }


def _validate_case(case: Case, result: dict[str, Any]) -> list[str]:
    label = " ".join(case.path)
    failures: list[str] = []
    if result["returncode"] != case.expected_returncode:
        failures.append(
            f"{label}: returncode={result['returncode']} expected={case.expected_returncode}"
        )
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    if case.require_stdout and not stdout.strip():
        failures.append(f"{label}: stdout is empty")
    if case.stderr_contains and case.stderr_contains not in stderr:
        failures.append(f"{label}: stderr lacks {case.stderr_contains!r}")
    if case.json_stdout and result["returncode"] == case.expected_returncode:
        try:
            json.loads(stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"{label}: stdout is not JSON: {exc}")
    for raw_path in case.artifacts:
        path = Path(raw_path)
        if not path.is_file():
            failures.append(f"{label}: missing artifact {path}")
        elif not case.allow_empty_artifacts and path.stat().st_size <= 0:
            failures.append(f"{label}: empty artifact {path}")
    result["stdout_tail"] = stdout[-3000:]
    result["stderr_tail"] = stderr[-3000:]
    result.pop("stdout", None)
    result.pop("stderr", None)
    return failures


def _inside(args: argparse.Namespace) -> int:
    work_root = Path(args.work_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    _prepare_fixtures(work_root)

    from astrolabe import cli

    source_file = str(Path(cli.__file__).resolve())
    failures: list[str] = []
    if args.expected_source_prefix and not source_file.startswith(
        str(Path(args.expected_source_prefix)) + os.sep
    ):
        failures.append(
            f"CLI source is not baked source: {source_file}; "
            f"expected prefix {args.expected_source_prefix}"
        )

    cases = _case_specs(work_root, repo_root)
    canonical_leaves = _canonical_leaf_paths()
    case_paths = {case.path for case in cases}
    if len(cases) != len(case_paths):
        failures.append("execution matrix contains duplicate canonical leaf paths")
    if case_paths != canonical_leaves:
        failures.append(
            "execution matrix differs from parser leaves: "
            f"missing={sorted(canonical_leaves - case_paths)} "
            f"extra={sorted(case_paths - canonical_leaves)}"
        )

    env = os.environ.copy()
    env.pop("GH_HOST_REPO_ROOT", None)
    env.pop("GH_HOST_REPO_MAP_ROOT", None)
    env["GH_CUDA_PREINITIALIZE"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["HOME"] = str(work_root / "home")
    env["MPLCONFIGDIR"] = str(work_root / "matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        result = _invoke_case(case, repo_root=repo_root, env=env)
        case_failures = _validate_case(case, result)
        result["failures"] = case_failures
        failures.extend(case_failures)
        results.append(result)

    bash_result = next(
        result for result in results if result["path"] == ["completion", "bash"]
    )
    bash_script = subprocess.run(
        [sys.executable, "-m", "astrolabe.cli", "completion", "bash", "--init"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    bash_path = work_root / "completion.bash"
    bash_path.write_text(bash_script, encoding="utf-8")
    bash_syntax = subprocess.run(
        ["bash", "-n", str(bash_path)], text=True, capture_output=True, check=False
    )
    if bash_result["returncode"] != 0 or bash_syntax.returncode != 0:
        failures.append("completion bash: generated script failed bash -n")

    raw_model_dirs = sorted(str(path) for path in work_root.rglob("raw_model_results"))
    if raw_model_dirs:
        failures.append(f"dry-run matrix created raw model directories: {raw_model_dirs}")

    report = {
        "schema_version": 1,
        "source_file": source_file,
        "canonical_leaves": len(canonical_leaves),
        "non_help_cases": len(cases),
        "passed_cases": sum(not result["failures"] for result in results),
        "bash_syntax_returncode": bash_syntax.returncode,
        "raw_model_directories": raw_model_dirs,
        "results": results,
        "failures": failures,
    }
    report_path = Path(args.inside_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "canonical_leaves": report["canonical_leaves"],
                "non_help_cases": report["non_help_cases"],
                "failures": len(failures),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return int(bool(failures))


def _outer(args: argparse.Namespace) -> int:
    engine = args.engine or _detect_engine()
    if engine is None:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")
    engine = str(Path(engine).resolve())
    sif = Path(os.path.expandvars(os.path.expanduser(args.sif))).resolve()
    if not sif.is_file():
        raise SystemExit(f"SIF not found: {sif}")

    work_parent = Path(args.work_parent).resolve() if args.work_parent else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    host_work = Path(
        tempfile.mkdtemp(
            prefix="gh-cli-execution-",
            dir=str(work_parent) if work_parent is not None else None,
        )
    ).resolve()
    inside_report = CONTAINER_WORK_ROOT / "inside-report.json"
    command = [
        engine,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "cwd",
        "--pwd",
        str(CONTAINER_REPO_ROOT),
        "-B",
        f"{host_work}:{CONTAINER_WORK_ROOT}",
        "-B",
        f"{Path(__file__).resolve()}:{CONTAINER_SCRIPT}:ro",
        str(sif),
        "/usr/bin/env",
        f"HOME={CONTAINER_WORK_ROOT / 'home'}",
        "GH_CUDA_PREINITIALIZE=0",
        "CUDA_VISIBLE_DEVICES=-1",
        "PYTHONDONTWRITEBYTECODE=1",
        "/alphafold3_venv/bin/python",
        str(CONTAINER_SCRIPT),
        "--inside",
        "--work-root",
        str(CONTAINER_WORK_ROOT),
        "--repo-root",
        str(CONTAINER_REPO_ROOT),
        "--expected-source-prefix",
        str(CONTAINER_REPO_ROOT),
        "--inside-report",
        str(inside_report),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    host_inside_report = host_work / inside_report.name
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not host_inside_report.is_file():
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"Container CLI execution smoke produced no report (exit {proc.returncode})."
        )
    report = json.loads(host_inside_report.read_text(encoding="utf-8"))
    report["container_command"] = command
    report["container_stdout_tail"] = proc.stdout[-4000:]
    report["container_stderr_tail"] = proc.stderr[-4000:]
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"commands: non_help={report['non_help_cases']}/{report['canonical_leaves']} "
        f"failures={len(report['failures'])}"
    )
    print(f"report: {report_path}")
    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    if args.keep_workdir:
        print(f"workdir: {host_work}")
    else:
        shutil.rmtree(host_work)
    return int(proc.returncode != 0 or bool(report["failures"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sif",
        default=str(
            Path.home()
            / ".gh_envs"
            / "images"
            / "Astrolabe_af3_licensed_20260722.sif"
        ),
    )
    parser.add_argument("--engine", default=None)
    parser.add_argument("--report", default="logs/gh_sif_cli_execution_smoketest.json")
    parser.add_argument("--work-parent", default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--work-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-prefix", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--inside-report", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        if not args.work_root or not args.repo_root or not args.inside_report:
            parser.error("--inside requires --work-root, --repo-root and --inside-report")
        return _inside(args)
    return _outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
