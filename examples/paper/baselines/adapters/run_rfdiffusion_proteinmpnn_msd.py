#!/usr/bin/env python3
"""RFdiffusion single-target backbone + ProteinMPNN-MSD baseline.

This is the corrected cross-target protocol.  A backbone is generated once
against target 1.  The configured hotspot correspondence defines a rigid
transform; the generated binder is copied into target-2's native frame with
the inverse transform.  Each state PDB therefore contains its target and
binder in one shared frame.  The maintained ProteinMPNN-MSD two-stage runner
then combines the states and ties B/D positions, producing one shared
sequence per sample.

The script intentionally handles one independent backbone unit.  A formal
200-backbone run should schedule this command 200 times with unique unit IDs
and seeds; each unit emits exactly 50 shared sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = BENCHMARK_ROOT / "config" / "rfdiffusion_proteinmpnn_msd_pairs.yaml"
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MSD_SEQUENCE_COUNT = 50
UNIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MsdError(RuntimeError):
    """Visible contract, geometry, or runtime failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise MsdError(f"config root must be a mapping: {path}")
    return value


def materialize_pair_config(raw: Mapping[str, Any], pair: str | None) -> dict[str, Any]:
    """Expand the central three-pair contract into one runner config.

    Keeping the pair table in one file prevents a silently stale copy of a
    target PDB, hotspot list, or runtime path from being used by a formal unit.
    The runner still accepts the older already-expanded example config.
    """
    config = dict(raw)
    pairs = config.get("pairs")
    if pairs is None:
        return config
    if not isinstance(pairs, Mapping):
        raise MsdError("config.pairs must be a mapping")
    selected = pair or "il4_il13"
    entry = pairs.get(selected)
    if not isinstance(entry, Mapping):
        raise MsdError(f"unknown RF-MSD target pair: {selected}")
    target1 = dict(entry.get("target_1", {}))
    target2 = dict(entry.get("target_2", {}))
    if not target1 or not target2:
        raise MsdError(f"pair {selected} must define target_1 and target_2")
    paired_raw = entry.get("paired_hotspots")
    if isinstance(paired_raw, Mapping):
        paired = [paired_raw.get("target_1"), paired_raw.get("target_2")]
    else:
        paired = paired_raw
    if not isinstance(paired, list) or len(paired) != 2:
        raise MsdError(f"pair {selected} must define paired_hotspots for both targets")
    design = dict(config.get("design", {}))
    lengths = design.pop("binder_length_by_pair", {})
    if not isinstance(lengths, Mapping) or selected not in lengths:
        raise MsdError(f"design.binder_length_by_pair lacks {selected}")
    design["binder_length"] = int(lengths[selected])
    config["design"] = design
    config["targets"] = [target1, target2]
    config["paired_hotspots"] = paired
    config["pair"] = selected
    # Pair-local entries are not part of the effective runtime config after
    # expansion; retaining them would make provenance needlessly ambiguous.
    config.pop("pairs", None)
    return config


def resolve_path(value: Any, field: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise MsdError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Configs are also materialized below ``config/inputs/<pair>`` for the
    # formal batch.  Prefer the benchmark-root convention used by the public
    # examples, then retain the older config-relative fallback for external
    # callers.
    benchmark_candidate = (BENCHMARK_ROOT / path).resolve()
    if benchmark_candidate.exists():
        return benchmark_candidate
    return (config_path.parent.parent / path).resolve()


def require_file(path: Path, field: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MsdError(f"missing or empty {field}: {path}")


def git_commit(path: Path) -> str | None:
    """Return a source commit without turning an untracked source into a hard failure."""
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_status_identity(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"available": False, "clean": None, "line_count": None, "sha256": None}
    text = result.stdout if result.returncode == 0 else ""
    if text and not text.endswith("\n"):
        text += "\n"
    return {
        "available": result.returncode == 0,
        "clean": result.returncode == 0 and not result.stdout,
        "line_count": len(result.stdout.splitlines()) if result.returncode == 0 else None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if result.returncode == 0 else None,
    }


def verify_declared_sha(path: Path, declared: Any, field: str) -> str:
    """Verify a declared SHA256 when a config supplies one, and return the actual hash."""
    require_file(path, field)
    observed = sha256_file(path)
    if declared not in (None, ""):
        if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise MsdError(f"{field} sha256 must be 64 lowercase hexadecimal characters")
        if observed != declared:
            raise MsdError(f"{field} sha256 mismatch: expected {declared}, observed {observed}")
    return observed


def runtime_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the files actually used by RFdiffusion and ProteinMPNN."""
    rf = config["rfdiffusion"]
    mpnn = config["proteinmpnn"]
    rf_source = Path(str(rf["source"])).resolve()
    mpnn_source = Path(str(mpnn["source"])).resolve()
    rf_checkpoint = Path(str(rf["checkpoint"])).resolve()
    model_name = str(mpnn.get("model_name", "v_48_020"))
    declared_weights = mpnn.get("model_weights") or mpnn.get("path_to_model_weights")
    if declared_weights:
        weights_path = Path(str(declared_weights)).expanduser()
        if not weights_path.is_absolute():
            weights_path = (mpnn_source / weights_path).resolve()
        elif weights_path.suffix != ".pt":
            weights_path = (weights_path / f"{model_name}.pt").resolve()
    else:
        weights_path = (mpnn_source / "run" / "model_weights" / f"{model_name}.pt").resolve()
    return {
        "rfdiffusion_source": {
            "path": str(rf_source),
            "commit": git_commit(rf_source),
            "worktree": git_status_identity(rf_source),
        },
        "rfdiffusion_checkpoint": {
            "path": str(rf_checkpoint),
            "sha256": verify_declared_sha(
                rf_checkpoint, rf.get("checkpoint_sha256"), "RFdiffusion checkpoint"
            ),
        },
        "proteinmpnn_source": {
            "path": str(mpnn_source),
            "commit": git_commit(mpnn_source),
            "worktree": git_status_identity(mpnn_source),
        },
        "proteinmpnn_model": {
            "name": model_name,
            "path": str(weights_path),
            "sha256": verify_declared_sha(
                weights_path, mpnn.get("model_weight_sha256"), "ProteinMPNN model weight"
            ),
            "loading_contract": "explicit_model_weights_path" if declared_weights else "source_local_run/model_weights",
        },
    }


def require_unit_id(value: str) -> str:
    if value in {".", ".."} or not UNIT_ID_RE.fullmatch(value):
        raise MsdError(f"unit id is not path-safe: {value!r}")
    return value


def parse_pdb_atoms(path: Path, chain: str) -> list[dict[str, Any]]:
    """Return ATOM/HETATM records for one chain, preserving input lines."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or line[21:22] != chain:
                continue
            resname = line[17:20].strip().upper()
            if resname == "MSE":
                resname = "MET"
            if resname not in AA3_TO_1:
                raise MsdError(f"unsupported residue {resname!r} in {path}")
            try:
                xyz = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
                resseq = int(line[22:26].strip())
            except (ValueError, IndexError) as exc:
                raise MsdError(f"malformed PDB atom line in {path}: {line.rstrip()}") from exc
            records.append(
                {
                    "line": line.rstrip("\n"),
                    "atom": line[12:16].strip(),
                    "resname": resname,
                    "resseq": resseq,
                    "icode": line[26:27].strip(),
                    "xyz": xyz,
                }
            )
    if not records:
        raise MsdError(f"chain {chain!r} is absent from {path}")
    return records


def pdb_chain_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21:22]
            if chain not in seen:
                seen.add(chain)
                ids.append(chain)
    return ids


def residue_ca_map(path: Path, chain: str) -> tuple[list[tuple[int, str]], dict[tuple[int, str], np.ndarray]]:
    records = parse_pdb_atoms(path, chain)
    order: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    ca: dict[tuple[int, str], np.ndarray] = {}
    for record in records:
        key = (record["resseq"], record["icode"])
        if key not in seen:
            seen.add(key)
            order.append(key)
        if record["atom"] == "CA":
            ca[key] = record["xyz"]
    return order, ca


def hotspot_ca_coordinates(path: Path, chain: str, positions_1_based: Sequence[int]) -> np.ndarray:
    order, ca = residue_ca_map(path, chain)
    if len(set(positions_1_based)) != len(positions_1_based) or not positions_1_based:
        raise MsdError("hotspot correspondence positions must be non-empty and unique")
    selected: list[np.ndarray] = []
    for position in positions_1_based:
        if not isinstance(position, int) or isinstance(position, bool) or position < 1 or position > len(order):
            raise MsdError(f"hotspot position {position!r} is outside chain {chain} length {len(order)}")
        key = order[position - 1]
        if key not in ca:
            raise MsdError(f"hotspot {chain}{position} has no CA atom in {path}")
        selected.append(ca[key])
    return np.asarray(selected, dtype=float)


def validate_contiguous_records(records: Sequence[dict[str, Any]], label: str) -> int:
    """Require one CA-bearing residue per consecutive 1-based binder/target index."""
    order: list[tuple[int, str]] = []
    ca_seen: set[tuple[int, str]] = set()
    ca_count: dict[tuple[int, str], int] = {}
    for record in records:
        key = (int(record["resseq"]), str(record["icode"]))
        if key not in order:
            order.append(key)
        if record["atom"] == "CA":
            ca_seen.add(key)
            ca_count[key] = ca_count.get(key, 0) + 1
    if not order or len(ca_seen) != len(order):
        raise MsdError(f"{label} has residues without CA atoms")
    if any(count != 1 for count in ca_count.values()):
        raise MsdError(f"{label} has duplicate CA atoms or alternate CA conformations")
    if [key[0] for key in order] != list(range(1, len(order) + 1)) or any(key[1] for key in order):
        raise MsdError(f"{label} residue numbering must be contiguous 1..N without insertion codes")
    return len(order)


def state_geometry_summary(
    target_records: Sequence[dict[str, Any]],
    binder_records: Sequence[dict[str, Any]],
    hotspot_positions: Sequence[int],
    *,
    label: str,
    contact_cutoff_angstrom: float = 12.0,
    ca_clash_cutoff_angstrom: float = 2.5,
) -> dict[str, Any]:
    """Validate state indexing and record a hotspot-to-binder contact diagnostic."""
    target_length = validate_contiguous_records(target_records, f"{label} target")
    binder_length = validate_contiguous_records(binder_records, f"{label} binder")
    target_ca = hotspot_ca_coordinates_from_records(target_records, hotspot_positions, f"{label} hotspot")
    binder_ca = np.asarray([record["xyz"] for record in binder_records if record["atom"] == "CA"], dtype=float)
    distances = np.sqrt(((target_ca[:, None, :] - binder_ca[None, :, :]) ** 2).sum(axis=2))
    min_distance = float(np.min(distances))
    if not np.isfinite(min_distance):
        raise MsdError(f"{label} hotspot contact distance is non-finite")
    all_target_ca = np.asarray(
        [record["xyz"] for record in target_records if record["atom"] == "CA"], dtype=float
    )
    all_distances = np.sqrt(
        ((all_target_ca[:, None, :] - binder_ca[None, :, :]) ** 2).sum(axis=2)
    )
    min_all_distance = float(np.min(all_distances))
    if not np.isfinite(min_all_distance):
        raise MsdError(f"{label} target-binder CA distance is non-finite")
    clash_pass = bool(min_all_distance >= ca_clash_cutoff_angstrom)
    contact_pass = bool(min_distance <= contact_cutoff_angstrom)
    return {
        "label": label,
        "target_length": target_length,
        "binder_length": binder_length,
        "hotspot_binder_ca_min_angstrom": min_distance,
        "hotspot_contact_cutoff_angstrom": float(contact_cutoff_angstrom),
        "hotspot_ca_contact_pass": contact_pass,
        "target_binder_ca_min_angstrom": min_all_distance,
        "target_binder_ca_clash_cutoff_angstrom": float(ca_clash_cutoff_angstrom),
        "target_binder_ca_clash_free": clash_pass,
    }


def hotspot_ca_coordinates_from_records(
    records: Sequence[dict[str, Any]], positions_1_based: Sequence[int], label: str,
) -> np.ndarray:
    order: list[tuple[int, str]] = []
    ca: dict[tuple[int, str], np.ndarray] = {}
    for record in records:
        key = (int(record["resseq"]), str(record["icode"]))
        if key not in order:
            order.append(key)
        if record["atom"] == "CA":
            ca[key] = np.asarray(record["xyz"], dtype=float)
    selected: list[np.ndarray] = []
    for position in positions_1_based:
        if not 1 <= int(position) <= len(order):
            raise MsdError(f"{label} position {position} is outside chain length {len(order)}")
        key = order[int(position) - 1]
        if key not in ca:
            raise MsdError(f"{label} position {position} has no CA atom")
        selected.append(ca[key])
    return np.asarray(selected, dtype=float)


def hotspot_alignment_transform(
    target1: Path,
    target1_chain: str,
    target2: Path,
    target2_chain: str,
    paired_hotspots: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return row-vector transform x2 @ R + t = x1 from hotspot CA pairs."""
    if len(paired_hotspots) != 2 or any(len(item) == 0 for item in paired_hotspots):
        raise MsdError("paired_hotspots must be [target1_positions, target2_positions]")
    if len(paired_hotspots[0]) != len(paired_hotspots[1]):
        raise MsdError("target1/target2 hotspot correspondence lengths differ")
    moving = hotspot_ca_coordinates(target2, target2_chain, paired_hotspots[1])
    target = hotspot_ca_coordinates(target1, target1_chain, paired_hotspots[0])
    if len(target) < 3:
        raise MsdError("at least three hotspot pairs are required for a rigid transform")
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - moving_center @ rotation
    aligned = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    if not np.isfinite(rmsd):
        raise MsdError("hotspot alignment RMSD is non-finite")
    return rotation, translation, {
        "target1_chain": target1_chain,
        "target2_chain": target2_chain,
        "target1_hotspots_1_based": list(paired_hotspots[0]),
        "target2_hotspots_1_based": list(paired_hotspots[1]),
        "paired_hotspot_count": len(target),
        "ca_rmsd_angstrom": rmsd,
        "rotation_row_vector": rotation.tolist(),
        "translation_row_vector": translation.tolist(),
        "transform_definition": "target2_native @ rotation + translation = target1_frame",
    }


def transformed_pdb_records(records: Iterable[dict[str, Any]], output_chain: str, rotation: np.ndarray, translation: np.ndarray) -> list[str]:
    if len(output_chain) != 1:
        raise MsdError(f"output chain must be one character: {output_chain!r}")
    lines: list[str] = []
    serial = 1
    for record in records:
        xyz = np.asarray(record["xyz"], dtype=float) @ rotation + translation
        line = record["line"]
        line = f"{line[:6]}{serial:5d}{line[11:21]}{output_chain}{line[22:30]}"
        line = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
        lines.append(line)
        serial += 1
    return lines


def identity_transform() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3, dtype=float), np.zeros(3, dtype=float)


def write_chain_pdb(path: Path, chains: Sequence[tuple[list[dict[str, Any]], str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for records, chain, rotation, translation in chains:
        transformed = transformed_pdb_records(records, chain, rotation, translation)
        for line in transformed:
            lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
        lines.append(f"TER   {serial:5d}")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def build_rf_command(config: Mapping[str, Any], target: Mapping[str, Any], output_prefix: Path, seed: int) -> list[str]:
    rf = config["rfdiffusion"]
    binder_length = int(config["design"]["binder_length"])
    hotspots = ",".join(f"{target['chain']}{int(item)}" for item in target["hotspots_1_based"])
    argv = [
        str(Path(rf["python"]).resolve()), str(Path(rf["source"]).resolve() / "scripts" / "run_inference.py"),
        f"inference.output_prefix={output_prefix}", f"inference.input_pdb={target['pdb']}",
        f"contigmap.contigs=[{binder_length}-{binder_length}/0 {target['chain']}1-{target['length']}]",
        f"ppi.hotspot_res=[{hotspots}]", "inference.num_designs=1",
        f"inference.ckpt_override_path={rf['checkpoint']}", f"inference.deterministic={str(bool(rf.get('deterministic', True)))}",
        f"inference.design_startnum={seed}",
    ]
    if rf.get("model_directory_path"):
        argv.append(f"inference.model_directory_path={rf['model_directory_path']}")
    argv.extend(str(item) for item in rf.get("extra_overrides", []))
    return argv


def build_msd_generate_command(
    config: Mapping[str, Any],
    output_specs: Path,
    binder_length: int,
) -> list[str]:
    """Build the maintained ProteinMPNN-MSD parser command.

    ``generate_json.py`` combines the state PDBs, remaps their chains, and
    materializes the positive MSD constraints consumed by
    ``run_protein_mpnn.py``.  Calling vanilla ``protein_mpnn_run.py`` with
    chain/tied JSON would silently change this baseline protocol.
    """
    mpnn = config["proteinmpnn"]
    source = Path(mpnn["source"]).resolve()
    designable = f"state1:B1-B{binder_length};state2:B1-B{binder_length}"
    constraints = f"state1:B1-B{binder_length}:1,state2:B1-B{binder_length}:1"
    return [
        str(Path(mpnn["python"]).resolve()), str(source / "run" / "generate_json.py"),
        "--pdb_dir", ".",
        "--designable_res", designable,
        "--constraints", constraints,
        "--multi_state",
        "--validation_tries", "0",
        "--out_path", str(output_specs),
    ]


def build_mpnn_command(config: Mapping[str, Any], states_dir: Path, out_dir: Path, specs: Path, seed: int) -> list[str]:
    """Build the maintained multi-state ProteinMPNN inference command."""
    mpnn = config["proteinmpnn"]
    source = Path(mpnn["source"]).resolve()
    model_weights = mpnn.get("model_weights") or mpnn.get("path_to_model_weights")
    if model_weights:
        weights_path = Path(str(model_weights)).expanduser()
        if not weights_path.is_absolute():
            weights_path = (source / weights_path).resolve()
        if weights_path.suffix == ".pt":
            weights_path = weights_path.parent
        weight_arg = ["--path_to_model_weights", str(weights_path)]
    else:
        weight_arg = []
    return [
        str(Path(mpnn["python"]).resolve()), str(source / "run" / "run_protein_mpnn.py"),
        "--pdb_dir", str(states_dir / "msd"),
        "--design_specs_json", str(specs),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(MSD_SEQUENCE_COUNT), "--batch_size", "1",
        "--sampling_temp", str(mpnn.get("sampling_temperature", 0.1)),
        "--model_name", str(mpnn.get("model_name", "v_48_020")),
        "--backbone_noise", str(mpnn.get("backbone_noise", 0.0)),
        "--global_omit_AAs", str(mpnn.get("global_omit_AAs", "X")),
        "--seed", str(int(mpnn.get("seed", 0)) + int(seed)),
        *weight_arg,
    ]


def run_logged(argv: Sequence[str], cwd: Path, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(list(argv)) + "\n")
        result = subprocess.run(list(argv), cwd=str(cwd), env={**os.environ, **env}, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise MsdError(f"command exited {result.returncode}: {shlex.join(list(argv))}")


def extract_msd_sequences(fasta: Path, binder_length: int) -> list[str]:
    """Read maintained MSD FASTA and return the shared designed chains.

    Depending on the ProteinMPNN-MSD revision, the combined FASTA contains
    either the two designed chains (B/D) or all four chains (A/B/C/D).  Both
    are valid only when the two designed sequences are present and identical;
    silently taking the first chain would make a fixed target look like a
    binder.
    """
    require_file(fasta, "ProteinMPNN-MSD FASTA")
    sequences: list[str] = []
    for line in fasta.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        parts = [part.upper() for part in line.split("/")]
        if len(parts) == 2:
            designed = parts
        elif len(parts) == 4:
            designed = [parts[1], parts[3]]
        else:
            raise MsdError(
                f"MSD output must contain two or four chain fields, got {line[:80]}"
            )
        if any(len(part) != binder_length for part in designed):
            raise MsdError(
                f"MSD output designed chains must each be {binder_length} aa, got {line[:80]}"
            )
        if any(char not in AA_ALPHABET for char in "".join(designed)):
            raise MsdError(f"ProteinMPNN-MSD FASTA contains illegal amino acid: {line[:80]}")
        if designed[0] != designed[1]:
            raise MsdError("ProteinMPNN-MSD did not enforce shared B/D sequence")
        sequences.append(designed[0])
    # run_protein_mpnn.py writes one native sequence followed by samples.
    if len(sequences) != MSD_SEQUENCE_COUNT + 1:
        raise MsdError(f"expected native + {MSD_SEQUENCE_COUNT} sampled sequences, observed {len(sequences)}")
    # Duplicate samples are a method output, not an infrastructure error.  A
    # low-temperature run may legitimately emit repeated sequences; retaining
    # all 50 preserves the pre-registered denominator instead of silently
    # filtering a method after generation.
    return sequences[1:]


def validate_msd_specs(path: Path, binder_length: int) -> dict[str, Any]:
    """Verify the parser output before loading the expensive MPNN model."""
    require_file(path, "ProteinMPNN-MSD design specification")
    contents = path.read_text(encoding="utf-8")
    try:
        # The maintained generate_json.py writes one JSON object with
        # indentation (and therefore multiple physical lines).  The
        # validator must parse the document, not each physical line.  Retain
        # a JSONL fallback for older generated specs.
        document = json.loads(contents)
        records = [document] if isinstance(document, dict) else document
        if not isinstance(records, list):
            records = [records]
    except json.JSONDecodeError as exc:
        try:
            records = [json.loads(line) for line in contents.splitlines() if line.strip()]
        except json.JSONDecodeError:
            raise MsdError(f"invalid ProteinMPNN-MSD design specification: {path}") from exc
    if len(records) != 1 or not isinstance(records[0], dict):
        raise MsdError(f"expected one MSD design-spec record, observed {len(records)}")
    record = records[0]
    chain_key = record.get("chain_key")
    if chain_key != {
        "state1": {"A": "A", "B": "B"},
        "state2": {"A": "C", "B": "D"},
    }:
        raise MsdError(f"MSD chain remapping is not A/B + C/D: {chain_key!r}")
    designable = record.get("designable", [])
    observed_pairs = [(item.get("chain"), item.get("resid")) for item in designable]
    expected_pairs = [("B", i) for i in range(1, binder_length + 1)] + [
        ("D", i) for i in range(1, binder_length + 1)
    ]
    # The upstream parser may order entries by chain; compare as sets while
    # retaining an exact count so a missing/extra position cannot pass.
    if len(designable) != 2 * binder_length or sorted(observed_pairs) != sorted(expected_pairs):
        raise MsdError(
            f"MSD designable positions must be B/D 1..{binder_length}; observed {observed_pairs[:8]}..."
        )
    symmetric = record.get("symmetric", [])
    if len(symmetric) != binder_length:
        raise MsdError(f"MSD must contain {binder_length} B/D ties, observed {len(symmetric)}")
    if any(list(item) != [f"B{index}", f"D{index}"] for index, item in enumerate(symmetric, start=1)):
        raise MsdError("MSD ties are not ordered one-to-one B1:D1 through BL:DL")
    return {
        "chain_key": chain_key,
        "designable_count": len(designable),
        "symmetric_count": len(symmetric),
        "first_tie": symmetric[0],
        "last_tie": symmetric[-1],
        "tied_betas": record.get("tied_betas", {}),
    }


def run_unit(
    config_path: Path,
    output_root: Path,
    unit_id: str,
    seed: int,
    gpu: str,
    dry_run: bool = False,
    pair: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_identity = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    config = materialize_pair_config(load_yaml(config_path), pair)
    if config.get("method") != "rfdiffusion_proteinmpnn_msd":
        raise MsdError("config.method must be 'rfdiffusion_proteinmpnn_msd'")
    require_unit_id(unit_id)
    # Resolve all repository-relative runtime paths against the benchmark root,
    # not against whichever directory the scheduler happens to use as cwd.
    for section_name in ("rfdiffusion", "proteinmpnn"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise MsdError(f"{section_name} block is required")
        if section.get("source"):
            section["source"] = str(resolve_path(section["source"], f"{section_name}.source", config_path))
        if section.get("python"):
            section["python"] = str(Path(str(section["python"])).expanduser().resolve())
    if config["rfdiffusion"].get("checkpoint"):
        config["rfdiffusion"]["checkpoint"] = str(resolve_path(config["rfdiffusion"]["checkpoint"], "rfdiffusion.checkpoint", config_path))
    if config["rfdiffusion"].get("model_directory_path"):
        config["rfdiffusion"]["model_directory_path"] = str(resolve_path(config["rfdiffusion"]["model_directory_path"], "rfdiffusion.model_directory_path", config_path))
    # The central input manifest and every target file are part of the
    # scientific identity.  Fail before loading either model when they drift.
    input_manifest = config.get("input_manifest")
    input_manifest_path: Path | None = None
    input_manifest_sha256: str | None = None
    if input_manifest:
        input_manifest_path = resolve_path(input_manifest, "input_manifest", config_path)
        input_manifest_sha256 = verify_declared_sha(
            input_manifest_path,
            config.get("input_manifest_sha256"),
            "RF-MSD input manifest",
        )

    hotspot_selection = config.get("hotspot_selection", {})
    if not isinstance(hotspot_selection, Mapping):
        raise MsdError("hotspot_selection must be a mapping")
    hotspot_evidence_path: Path | None = None
    hotspot_evidence_sha256: str | None = None
    if hotspot_selection.get("evidence_json"):
        hotspot_evidence_path = resolve_path(
            hotspot_selection["evidence_json"], "hotspot_selection.evidence_json", config_path
        )
        hotspot_evidence_sha256 = verify_declared_sha(
            hotspot_evidence_path,
            hotspot_selection.get("evidence_sha256"),
            "RF-MSD hotspot-correspondence evidence",
        )

    design = config.get("design")
    if not isinstance(design, Mapping):
        raise MsdError("design block is required")
    binder_length = int(design["binder_length"])
    geometry = config.get("geometry", {})
    if not isinstance(geometry, Mapping):
        raise MsdError("geometry block must be a mapping when provided")
    contact_cutoff = float(geometry.get("hotspot_contact_cutoff_angstrom", 12.0))
    require_hotspot_contact = bool(geometry.get("require_hotspot_contact", False))
    clash_cutoff = float(geometry.get("target_binder_ca_clash_cutoff_angstrom", 2.5))
    require_clash_free = bool(geometry.get("require_target_binder_ca_clash_free", False))
    if not np.isfinite(contact_cutoff) or contact_cutoff <= 0:
        raise MsdError("geometry.hotspot_contact_cutoff_angstrom must be positive")
    if not np.isfinite(clash_cutoff) or clash_cutoff <= 0:
        raise MsdError("geometry.target_binder_ca_clash_cutoff_angstrom must be positive")
    targets = config.get("targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise MsdError("targets must contain exactly two entries")
    t1, t2 = (dict(item) for item in targets)
    for index, target in enumerate((t1, t2), 1):
        target["pdb"] = resolve_path(target.get("pdb"), f"targets[{index}].pdb", config_path)
        verify_declared_sha(target["pdb"], target.get("sha256"), f"target {index} PDB")
        target["chain"] = str(target.get("chain", ""))
        actual_length = len(residue_ca_map(target["pdb"], target["chain"])[0])
        target["length"] = int(target.get("length", actual_length))
        if target["length"] != actual_length:
            raise MsdError(
                f"targets[{index}].length={target['length']} does not match "
                f"PDB chain length {actual_length}"
            )
        target["hotspots_1_based"] = [int(item) for item in target.get("hotspots_1_based", [])]
    target1_records = parse_pdb_atoms(t1["pdb"], t1["chain"])
    target2_records = parse_pdb_atoms(t2["pdb"], t2["chain"])
    validate_contiguous_records(target1_records, "target1")
    validate_contiguous_records(target2_records, "target2")
    paired = config.get("paired_hotspots")
    if not isinstance(paired, list) or len(paired) != 2:
        raise MsdError("paired_hotspots must be [target1_positions, target2_positions]")
    paired = [[int(item) for item in side] for side in paired]
    for index, target in enumerate((t1, t2)):
        declared = set(int(item) for item in target["hotspots_1_based"])
        observed = set(paired[index])
        if declared != observed:
            raise MsdError(
                f"paired_hotspots[{index}] must contain exactly target{index + 1} hotspots; "
                f"declared={sorted(declared)}, paired={sorted(observed)}"
            )
    rotation_21, translation_21, alignment = hotspot_alignment_transform(t1["pdb"], t1["chain"], t2["pdb"], t2["chain"], paired)
    runtime = runtime_identity(config)
    root = output_root.resolve()
    if dry_run:
        rf_prefix = root / "rfdiffusion" / "design"
        rf_command = build_rf_command(config, t1, rf_prefix, seed)
        generate_command = build_msd_generate_command(config, root / "proteinmpnn_res_specs.json", binder_length)
        mpnn_command = build_mpnn_command(config, root / "states", root / "proteinmpnn", root / "proteinmpnn_res_specs.json", seed)
        return {
            "dry_run": True, "method": "rfdiffusion_proteinmpnn_msd", "pair": config.get("pair"),
            "unit_id": unit_id, "seed": seed, "gpu": gpu, "profile": profile,
            "binder_length": binder_length,
            "configuration": config_identity,
            "counts": {"design_units": 1, "backbones": 1, "binder_sequences": MSD_SEQUENCE_COUNT, "af3_binary_jobs": 2 * MSD_SEQUENCE_COUNT},
            "alignment": alignment,
            "hotspot_selection_evidence": None if hotspot_evidence_path is None else {
                "path": str(hotspot_evidence_path), "sha256": hotspot_evidence_sha256,
            },
            "runtime_identity": runtime,
            "commands": {"rfdiffusion": rf_command, "proteinmpnn_generate_json": generate_command, "proteinmpnn": mpnn_command},
            "protocol": {"backbone_generation": "target1_only", "target_alignment": "paired_hotspot_ca_kabsch", "binder_placement": "inverse_transform_to_target2_native_frame", "state_coordinate_contract": "target_and_binder_share_frame_within_each_state", "sequence_design": "ProteinMPNN-MSD --multi_state positive tie", "sequences_per_backbone": MSD_SEQUENCE_COUNT, "target_input_contract": "frozen five-arm feature PDBs", "geometry_guard_policy": "record_only"},
        }
    if root.exists() and any(root.iterdir()):
        raise MsdError(f"refusing to overwrite non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        rf_dir = root / "rfdiffusion"
        rf_prefix = rf_dir / "design"
        rf_command = build_rf_command(config, {**t1, "pdb": str(t1["pdb"])}, rf_prefix, seed)
        rf_source = Path(config["rfdiffusion"]["source"]).resolve()
        run_logged(rf_command, rf_source, {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(rf_source)}, root / "rfdiffusion.log")
        rf_pdb = Path(f"{rf_prefix}_{seed}.pdb")
        require_file(rf_pdb, "RFdiffusion backbone PDB")
        chain_ids = pdb_chain_ids(rf_pdb)
        if "A" not in chain_ids or "B" not in chain_ids:
            raise MsdError(
                "RFdiffusion output must contain binder chain A and fixed target chain B; "
                f"observed chains={chain_ids}"
            )
        binder_records = parse_pdb_atoms(rf_pdb, "A")
        generated_target_records = parse_pdb_atoms(rf_pdb, "B")
        generated_binder_length = len(residue_ca_map(rf_pdb, "A")[0])
        if generated_binder_length != binder_length:
            raise MsdError(
                f"RFdiffusion binder chain A length mismatch: expected {binder_length}, "
                f"observed {generated_binder_length}"
            )
        generated_target_length = len(residue_ca_map(rf_pdb, "B")[0])
        if generated_target_length != int(t1["length"]):
            raise MsdError(
                f"RFdiffusion fixed target chain B length mismatch: expected {t1['length']}, "
                f"observed {generated_target_length}"
            )
        # Inverse of x2 @ R21 + t21 = x1, for row vectors.
        inv_rotation = rotation_21.T
        inv_translation = -translation_21 @ rotation_21.T
        geometry1 = state_geometry_summary(
            target1_records, binder_records, paired[0], label="state1",
            contact_cutoff_angstrom=contact_cutoff, ca_clash_cutoff_angstrom=clash_cutoff,
        )
        transformed_binder_records = [
            {**record, "xyz": np.asarray(record["xyz"], dtype=float) @ inv_rotation + inv_translation}
            for record in binder_records
        ]
        geometry2 = state_geometry_summary(
            target2_records, transformed_binder_records, paired[1], label="state2",
            contact_cutoff_angstrom=contact_cutoff, ca_clash_cutoff_angstrom=clash_cutoff,
        )
        if require_hotspot_contact and not geometry1["hotspot_ca_contact_pass"]:
            raise MsdError(
                f"state1 has no hotspot-to-binder CA contact within {contact_cutoff:.2f} A"
            )
        if require_hotspot_contact and not geometry2["hotspot_ca_contact_pass"]:
            raise MsdError(
                f"state2 has no hotspot-to-binder CA contact within {contact_cutoff:.2f} A"
            )
        if require_clash_free and not geometry1["target_binder_ca_clash_free"]:
            raise MsdError(
                f"state1 has a target-binder CA clash within {clash_cutoff:.2f} A"
            )
        if require_clash_free and not geometry2["target_binder_ca_clash_free"]:
            raise MsdError(
                f"state2 has a target-binder CA clash within {clash_cutoff:.2f} A"
            )
        states_dir = root / "states"
        state1 = states_dir / "state1.pdb"
        state2 = states_dir / "state2.pdb"
        # Each state must contain a target and its binder in the *same* frame.
        # State 1 is the RF design frame; state 2 uses the inverse hotspot
        # transform to place that binder back into target2's native frame.
        write_chain_pdb(state1, [
            (target1_records, "A", *identity_transform()),
            (binder_records, "B", *identity_transform()),
        ])
        write_chain_pdb(state2, [
            (target2_records, "A", *identity_transform()),
            (binder_records, "B", inv_rotation, inv_translation),
        ])
        msd_records = root / "msd_pair.pdb"
        write_chain_pdb(msd_records, [
            (target1_records, "A", *identity_transform()),
            (binder_records, "B", *identity_transform()),
            (target2_records, "C", *identity_transform()),
            (binder_records, "D", inv_rotation, inv_translation),
        ])
        context1 = root / "af3_context_1_targetA_binderB.pdb"
        context2 = root / "af3_context_2_targetA_binderB.pdb"
        write_chain_pdb(context1, [(target1_records, "A", *identity_transform()), (binder_records, "B", *identity_transform())])
        write_chain_pdb(context2, [(target2_records, "A", *identity_transform()), (binder_records, "B", inv_rotation, inv_translation)])
        mpnn_dir = root / "proteinmpnn"
        specs = root / "proteinmpnn_res_specs.json"
        mpnn_source = Path(config["proteinmpnn"]["source"]).resolve()
        generate_command = build_msd_generate_command(config, specs, binder_length)
        run_logged(generate_command, states_dir, {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(mpnn_source / "run")}, root / "proteinmpnn_generate_json.log")
        specs_validation = validate_msd_specs(specs, binder_length)
        generated_msd = states_dir / "msd" / "msd.pdb"
        require_file(generated_msd, "ProteinMPNN-MSD combined PDB")
        mpnn_command = build_mpnn_command(config, states_dir, mpnn_dir, specs, seed)
        run_logged(mpnn_command, mpnn_source / "run", {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(mpnn_source / "run")}, root / "proteinmpnn.log")
        raw_fasta = mpnn_dir / "msd.fasta"
        sequences = extract_msd_sequences(raw_fasta, binder_length)
        fasta = root / "binder_sequences.fasta"
        with fasta.open("w", encoding="utf-8") as handle:
            for index, sequence in enumerate(sequences, 1):
                handle.write(f">{unit_id}_backbone1_msd_{index}\n{sequence}\n")
        provenance = {
            "schema_version": 1, "status": "completed", "scientific_status": "success", "method": "rfdiffusion_proteinmpnn_msd", "pair": config.get("pair"), "unit_id": unit_id, "seed": seed, "gpu": gpu, "profile": profile,
            "counts": {"design_units": 1, "backbones": 1, "binder_sequences": len(sequences), "unique_binder_sequences": len(set(sequences)), "af3_binary_jobs": 2 * len(sequences)},
            "protocol": {"backbone_generation": "target1_only", "target_alignment": "paired_hotspot_ca_kabsch", "binder_placement": "inverse_transform_to_target2_native_frame", "state_coordinate_contract": "target_and_binder_share_frame_within_each_state", "sequence_design": "ProteinMPNN-MSD --multi_state positive tie", "designed_chains": ["B", "D"], "fixed_chains": ["A", "C"], "tied_positions": binder_length, "target_input_contract": "frozen five-arm feature PDBs", "geometry_guard_policy": "record_only", "duplicate_sequence_policy": "retain_all_50"},
            "alignment": alignment, "targets": [{"name": t1.get("name", "target1"), "pdb": str(t1["pdb"]), "chain": t1["chain"], "sha256": sha256_file(t1["pdb"]), "length": t1["length"], "hotspots_1_based": t1["hotspots_1_based"]}, {"name": t2.get("name", "target2"), "pdb": str(t2["pdb"]), "chain": t2["chain"], "sha256": sha256_file(t2["pdb"]), "length": t2["length"], "hotspots_1_based": t2["hotspots_1_based"]}],
            "input_manifest": None if input_manifest_path is None else {"path": str(input_manifest_path), "sha256": input_manifest_sha256},
            "hotspot_selection_evidence": None if hotspot_evidence_path is None else {"path": str(hotspot_evidence_path), "sha256": hotspot_evidence_sha256},
            "configuration": config_identity,
            "runtime_identity": runtime,
            "state_geometry": [geometry1, geometry2],
            "design_specs_validation": specs_validation,
            "commands": {"rfdiffusion": rf_command, "proteinmpnn_generate_json": generate_command, "proteinmpnn": mpnn_command},
            "artifacts": {"rf_backbone": {"path": str(rf_pdb), "sha256": sha256_file(rf_pdb)}, "state_pdbs": [{"path": str(state1), "sha256": sha256_file(state1)}, {"path": str(state2), "sha256": sha256_file(state2)}], "msd_combined_pdb": {"path": str(generated_msd), "sha256": sha256_file(generated_msd)}, "msd_pair_native_frames": {"path": str(msd_records), "sha256": sha256_file(msd_records)}, "af3_contexts": [str(context1), str(context2)], "raw_mpnn_fasta": {"path": str(raw_fasta), "sha256": sha256_file(raw_fasta)}, "binder_fasta": {"path": str(fasta), "sha256": sha256_file(fasta)}},
            "sequence_sha256": [hashlib.sha256(seq.encode("ascii")).hexdigest() for seq in sequences],
        }
        write_json(root / "provenance.json", provenance)
        write_json(root / "generation_summary.json", provenance)
        write_json(root / "DONE.json", {"status": "done", "scientific_status": "success", "unit_id": unit_id, "unit_result": str(root / "generation_summary.json"), "counts": provenance["counts"]})
        return provenance
    except Exception as exc:
        failure = {"status": "failed", "method": "rfdiffusion_proteinmpnn_msd", "unit_id": unit_id, "seed": seed, "gpu": gpu, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        write_json(root / "INFRASTRUCTURE_FAILED.json", failure)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pair", default=None, help="Target-pair key when using the central three-pair config.")
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", default=None, help="Scheduler profile label recorded in provenance.")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_unit(
            args.config.resolve(), args.output_root, args.unit_id, args.seed, args.gpu,
            args.dry_run, pair=args.pair, profile=args.profile,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
