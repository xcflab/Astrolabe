#!/usr/bin/env python3
"""BindCraft-ALT adapter with explicit dual-context raw-gradient averaging.

This adapter evaluates two target contexts from the same pre-update binder
sequence, averages their raw gradients, performs one shared update, then applies
an explicitly constructed cross-context endpoint selector. This joint-gradient
protocol and selector are not author-published BindCraft-ALT settings.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
BINDCRAFT_DIR = Path(os.environ.get("ASTROLABE_BINDCRAFT_DIR", PROJECT_DIR / "sources" / "BindCraft"))
STAGE_NAMES = ("logits", "softmax", "one_hot", "joint_semi_greedy")
OPTIMIZATION_PROTOCOL = "joint_gradient_mean_with_joint_semigreedy"
# ColabDesign logits use AlphaFold residue_constants.restype_order.
AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
ENDPOINT_STRATEGY = "joint_cross_context_minimax_mean_full_design_loss"
ENDPOINT_EVALUATION_MODELS = (0, 1, 2, 3, 4)
ENDPOINT_CANDIDATE_SOURCES = ("final_shared_current", "each_context_private_best")
PDB_AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class BindCraftAltError(RuntimeError):
    pass


class ConfigError(BindCraftAltError):
    pass


class ExistingRunOutputError(ConfigError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class TargetContext:
    name: str
    pdb: Path
    chain: str
    hotspot: Any
    normalized_hotspot: Tuple[str, ...]
    target_secondary_structure: Any = False
    target_secondary_structure_mask: Any = False
    target_helicity_weight: float = 0.0


@dataclass(frozen=True)
class StageSpec:
    name: str
    steps: int
    temperature_start: float
    temperature_end: float
    soft_end: float
    tries: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_path(value: Any, field: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_DIR / path).resolve()


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def require_list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{field} must be a list")
    return value


def require_int(value: Any, field: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return int(value)


def require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    return float(value)


def require_false(value: Any, field: str) -> None:
    if value is not False:
        raise ConfigError(f"{field} must be false")


def disabled_value(value: Any) -> bool:
    return value in (False, None, "") or value == [] or value == {}


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def bindcraft_worktree_status() -> Dict[str, Any]:
    if not (BINDCRAFT_DIR / ".git").exists():
        return {
            "path": str(BINDCRAFT_DIR.resolve()),
            "is_git_worktree": False,
            "clean": None,
            "status_porcelain_version": None,
            "status_porcelain_line_count": None,
            "status_porcelain_sha256": None,
            "status_counts": {},
            "dirty_paths": [],
        }
    completed = subprocess.run(
        ["git", "-C", str(BINDCRAFT_DIR), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    status_counts: Dict[str, int] = {}
    dirty_paths = []
    for line in lines:
        code = line[:2] if len(line) >= 2 else line
        path_text = line[3:] if len(line) > 3 else ""
        status_counts[code] = status_counts.get(code, 0) + 1
        if " -> " in path_text:
            dirty_paths.extend(path_text.split(" -> "))
        elif path_text:
            dirty_paths.append(path_text)
    stable_status = "\n".join(lines)
    if stable_status:
        stable_status += "\n"
    return {
        "path": str(BINDCRAFT_DIR.resolve()),
        "is_git_worktree": True,
        "clean": len(lines) == 0,
        "status_porcelain_version": "v1",
        "status_porcelain_line_count": len(lines),
        "status_porcelain_sha256": hashlib.sha256(stable_status.encode("utf-8")).hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
        "dirty_paths": sorted(set(dirty_paths)),
    }


def parse_pdb_chain_features(path: Path, chain: str) -> Dict[str, Any]:
    residues: List[Dict[str, Any]] = []
    residue_lookup: Dict[Tuple[str, str], int] = {}
    atom_positions = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or line[21].strip() != chain:
                continue
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            insertion_code = line[26].strip()
            key = (resseq, insertion_code)
            if key not in residue_lookup:
                residue_lookup[key] = len(residues)
                residues.append(
                    {
                        "resname": resname,
                        "aa": PDB_AA3_TO_AA1.get(resname, "X"),
                        "resseq": resseq,
                        "insertion_code": insertion_code,
                        "ca_coordinate": None,
                        "ca_mask": 0,
                    }
                )
            atom_name = line[12:16].strip()
            xyz = [
                round(float(line[30:38]), 3),
                round(float(line[38:46]), 3),
                round(float(line[46:54]), 3),
            ]
            atom_positions.append(
                {
                    "atom": atom_name,
                    "resname": resname,
                    "resseq": resseq,
                    "insertion_code": insertion_code,
                    "xyz": xyz,
                }
            )
            if atom_name == "CA":
                residue = residues[residue_lookup[key]]
                residue["ca_coordinate"] = xyz
                residue["ca_mask"] = 1
    if not residues:
        raise ConfigError(f"No ATOM/HETATM records for chain {chain} in {path}")
    return {
        "chain": chain,
        "target_length": len(residues),
        "target_sequence": "".join(residue["aa"] for residue in residues),
        "residue_ids": [
            {
                "chain": chain,
                "resseq": residue["resseq"],
                "insertion_code": residue["insertion_code"],
            }
            for residue in residues
        ],
        "residue_id_set": {
            (chain, residue["resseq"], residue["insertion_code"]) for residue in residues
        },
        "ca_coordinates": [residue["ca_coordinate"] for residue in residues],
        "ca_mask": [residue["ca_mask"] for residue in residues],
        "atom_positions": atom_positions,
    }


def parse_hotspot_token(value: Any, chain: str, field: str) -> Tuple[str, str, str]:
    if isinstance(value, bool):
        raise ConfigError(f"{field} hotspot residue must be an integer or chain-qualified string")
    if isinstance(value, int):
        return (chain, str(value), "")
    if isinstance(value, str):
        token = value.strip()
        match = re.match(r"^(?:(?P<chain>[A-Za-z]))?(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)$", token)
        if not match:
            raise ConfigError(f"{field} hotspot residue has unsupported format: {value!r}")
        observed_chain = match.group("chain") or chain
        if observed_chain != chain:
            raise ConfigError(
                f"{field} hotspot residue {value!r} is on chain {observed_chain}, expected chain {chain}"
            )
        return (chain, match.group("resseq"), match.group("icode") or "")
    raise ConfigError(f"{field} hotspot residue must be an integer or chain-qualified string")


def normalize_hotspot(value: Any, chain: str, residue_ids: Optional[set], field: str) -> Tuple[str, ...]:
    if disabled_value(value):
        return tuple()
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    normalized = []
    for index, item in enumerate(raw_items):
        residue = parse_hotspot_token(item, chain, f"{field}[{index}]")
        if residue_ids is not None and residue not in residue_ids:
            chain_id, resseq, insertion_code = residue
            display = f"{chain_id}{resseq}{insertion_code}"
            raise ConfigError(f"{field} hotspot residue does not exist in target PDB chain: {display}")
        normalized.append(f"{residue[0]}{residue[1]}{residue[2]}")
    return tuple(normalized)


def validate_stage_transition_policy(raw: Any, selection_policy: str) -> Dict[str, Any]:
    policy = dict(require_mapping(raw, "stage_transition_policy"))
    required = {
        "source",
        "author_published_values_available",
        "stage_order",
        "transition_rule",
        "gate_policy",
        "pair_min_gate",
        "endpoint_selection",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ConfigError(f"stage_transition_policy is missing required fields: {missing}")
    if policy["source"] != "constructed_reproduction_choice":
        raise ConfigError("stage_transition_policy.source must be 'constructed_reproduction_choice'")
    require_false(policy["author_published_values_available"], "stage_transition_policy.author_published_values_available")
    if policy["stage_order"] != list(STAGE_NAMES):
        raise ConfigError(f"stage_transition_policy.stage_order must be {list(STAGE_NAMES)}")
    if policy["transition_rule"] != "fixed_steps_no_metric_gate":
        raise ConfigError("stage_transition_policy.transition_rule must be 'fixed_steps_no_metric_gate'")
    if policy["gate_policy"] != "none":
        raise ConfigError("stage_transition_policy.gate_policy must be 'none'")
    require_false(policy["pair_min_gate"], "stage_transition_policy.pair_min_gate")
    if policy["endpoint_selection"] != selection_policy:
        raise ConfigError("stage_transition_policy.endpoint_selection must match selection_policy")
    return policy


def validate_endpoint_selection(raw: Any, model_list: Sequence[int]) -> Dict[str, Any]:
    selection = dict(require_mapping(raw, "endpoint_selection"))
    expected = {
        "source": "constructed_reproduction_choice",
        "author_published_selector_available": False,
        "strategy": ENDPOINT_STRATEGY,
        "candidate_sources": list(ENDPOINT_CANDIDATE_SOURCES),
        "evaluation_model_ids": list(ENDPOINT_EVALUATION_MODELS),
        "sample_models": False,
        "per_model_score": "full_design_loss",
        "full_design_loss_key": "loss",
        "within_context_reducer": "mean",
        "cross_context_reducer": "minimize_worst_context_mean",
        "tie_break": ["minimize_overall_context_mean", "lexicographic_sequence"],
    }
    if selection != expected:
        raise ConfigError(f"endpoint_selection must equal the frozen constructed selector: {expected}")
    if list(model_list) != list(ENDPOINT_EVALUATION_MODELS):
        raise ConfigError(
            f"af.model_list must be exactly {list(ENDPOINT_EVALUATION_MODELS)} for joint endpoint selection"
        )
    return selection


def validate_provenance(raw: Any, config_path: Path, check_existing_inputs: bool) -> Dict[str, Any]:
    provenance = dict(require_mapping(raw, "provenance"))
    for field in ("status", "paper_pdf", "paper_pdf_sha256", "bindcraft_commit"):
        if field not in provenance:
            raise ConfigError(f"provenance.{field} is required")
    if provenance["status"] != "constructed_reproduction_not_official":
        raise ConfigError("provenance.status must be 'constructed_reproduction_not_official'")
    paper_pdf = resolve_path(provenance["paper_pdf"], "provenance.paper_pdf", config_path)
    if check_existing_inputs and not paper_pdf.is_file():
        raise ConfigError(f"provenance.paper_pdf does not exist: {paper_pdf}")
    observed_pdf_sha = sha256_file(paper_pdf) if paper_pdf.is_file() else None
    if observed_pdf_sha is not None and provenance["paper_pdf_sha256"] != observed_pdf_sha:
        raise ConfigError(
            "provenance.paper_pdf_sha256 mismatch: "
            f"expected {provenance['paper_pdf_sha256']}, observed {observed_pdf_sha}"
        )
    observed_bindcraft_commit = git_head(BINDCRAFT_DIR) if (BINDCRAFT_DIR / ".git").exists() else None
    if observed_bindcraft_commit is not None and provenance["bindcraft_commit"] != observed_bindcraft_commit:
        raise ConfigError(
            "provenance.bindcraft_commit mismatch: "
            f"expected {provenance['bindcraft_commit']}, observed {observed_bindcraft_commit}"
        )
    source_worktree = bindcraft_worktree_status()
    return {
        "status": provenance["status"],
        "paper_pdf": paper_pdf,
        "paper_pdf_sha256": provenance["paper_pdf_sha256"],
        "paper_pdf_sha256_observed": observed_pdf_sha,
        "bindcraft_commit": provenance["bindcraft_commit"],
        "bindcraft_commit_observed": observed_bindcraft_commit,
        "bindcraft_source_worktree": source_worktree,
        "note": provenance.get("note"),
    }


def validate_config(raw: Mapping[str, Any], config_path: Path, check_existing_inputs: bool = False) -> Dict[str, Any]:
    required = [
        "schema_version",
        "method",
        "official_author_config_available",
        "num_binders",
        "binder_length",
        "trajectory_seeds",
        "advanced_settings_json",
        "advanced_settings_sha256",
        "advanced_settings_overrides",
        "af",
        "optimization_protocol",
        "stages",
        "selection_policy",
        "endpoint_selection",
        "stage_transition_policy",
        "provenance",
        "targets",
        "output",
    ]
    for field in required:
        if field not in raw:
            raise ConfigError(f"Missing required config field: {field}")
    if raw["method"] != "bindcraft_alt":
        raise ConfigError("method must be 'bindcraft_alt'")
    if raw["optimization_protocol"] != OPTIMIZATION_PROTOCOL:
        raise ConfigError(f"optimization_protocol must be '{OPTIMIZATION_PROTOCOL}'")
    if "alternation" in raw:
        raise ConfigError("alternation is invalid for the joint raw-gradient-mean protocol")
    require_false(raw["official_author_config_available"], "official_author_config_available")
    diagnostic_mode = bool(raw.get("diagnostic_mode", False))
    expected_num_binders = 1 if diagnostic_mode else 4
    if require_int(raw["num_binders"], "num_binders", 1) != expected_num_binders:
        raise ConfigError(f"num_binders must be exactly {expected_num_binders}")
    binder_length = require_int(raw["binder_length"], "binder_length", 1)

    seeds = [require_int(seed, f"trajectory_seeds[{i}]", 0) for i, seed in enumerate(require_list(raw["trajectory_seeds"], "trajectory_seeds"))]
    expected_seed_count = 1 if diagnostic_mode else 4
    if len(seeds) != expected_seed_count or len(set(seeds)) != expected_seed_count:
        raise ConfigError(f"trajectory_seeds must contain exactly {expected_seed_count} unique seeds")

    advanced_json = resolve_path(raw["advanced_settings_json"], "advanced_settings_json", config_path)
    if check_existing_inputs and not advanced_json.is_file():
        raise ConfigError(f"advanced_settings_json does not exist: {advanced_json}")
    observed_sha = sha256_file(advanced_json) if advanced_json.is_file() else None
    if observed_sha is not None and raw["advanced_settings_sha256"] != observed_sha:
        raise ConfigError(
            "advanced_settings_sha256 mismatch: "
            f"expected {raw['advanced_settings_sha256']}, observed {observed_sha}"
        )
    advanced_overrides = dict(
        require_mapping(raw["advanced_settings_overrides"], "advanced_settings_overrides")
    )
    expected_overrides = {
        "optimise_beta": False,
        "enable_mpnn": False,
        "use_termini_distance_loss": False,
    }
    if advanced_overrides != expected_overrides:
        raise ConfigError(
            "advanced_settings_overrides must explicitly disable unsupported beta adaptation, "
            "ProteinMPNN, and termini-distance loss: "
            f"expected {expected_overrides}, got {advanced_overrides}"
        )

    af = dict(require_mapping(raw["af"], "af"))
    if "params_dir" not in af:
        raise ConfigError("af.params_dir is required")
    af["params_dir"] = resolve_path(af["params_dir"], "af.params_dir", config_path)
    if check_existing_inputs and not af["params_dir"].exists():
        raise ConfigError(f"af.params_dir does not exist: {af['params_dir']}")
    model_list = require_list(af.get("model_list"), "af.model_list")
    if not model_list:
        raise ConfigError("af.model_list must not be empty")
    af["model_list"] = [int(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in model_list]

    stages_raw = require_mapping(raw["stages"], "stages")
    if "semi_greedy" in stages_raw:
        raise ConfigError(
            "stages.semi_greedy is invalid for the joint protocol; use stages.joint_semi_greedy"
        )
    if "hard_gradient_extension" in stages_raw:
        raise ConfigError(
            "stages.hard_gradient_extension is not part of the joint-gradient-plus-joint-semigreedy protocol"
        )
    stages = []
    for name in STAGE_NAMES:
        if name not in stages_raw:
            raise ConfigError(f"Missing stages.{name}")
        spec = require_mapping(stages_raw[name], f"stages.{name}")
        stages.append(
            StageSpec(
                name=name,
                steps=require_int(spec.get("steps"), f"stages.{name}.steps", 0),
                temperature_start=require_number(
                    spec.get("temperature_start"), f"stages.{name}.temperature_start"
                ),
                temperature_end=require_number(
                    spec.get("temperature_end"), f"stages.{name}.temperature_end"
                ),
                soft_end=require_number(spec.get("soft_end"), f"stages.{name}.soft_end"),
                tries=require_int(spec.get("tries"), f"stages.{name}.tries", 0),
            )
        )
    expected_stage_steps = {
        "logits": 75,
        "softmax": 45,
        "one_hot": 5,
        "joint_semi_greedy": 15,
    }
    observed_stage_steps = {stage.name: stage.steps for stage in stages}
    if observed_stage_steps != expected_stage_steps:
        raise ConfigError(
            f"joint protocol stage steps must be exactly {expected_stage_steps}, got {observed_stage_steps}"
        )
    if any(stage.tries != 0 for stage in stages[:-1]):
        raise ConfigError("Gradient stages must set tries=0")
    joint_semigreedy = stages[-1]
    if (
        joint_semigreedy.soft_end != 1.0
        or joint_semigreedy.temperature_start != 0.01
        or joint_semigreedy.temperature_end != 0.01
        or joint_semigreedy.tries != 1
    ):
        raise ConfigError(
            "stages.joint_semi_greedy must set soft_end=1, "
            "temperature_start=temperature_end=0.01, and tries=1"
        )

    if raw["selection_policy"] != "shared_endpoint":
        raise ConfigError("selection_policy must be 'shared_endpoint'")
    endpoint_selection = validate_endpoint_selection(raw["endpoint_selection"], af["model_list"])
    stage_transition_policy = validate_stage_transition_policy(
        raw["stage_transition_policy"], raw["selection_policy"]
    )
    provenance = validate_provenance(raw["provenance"], config_path, check_existing_inputs)

    targets_raw = require_list(raw["targets"], "targets")
    if len(targets_raw) != 2:
        raise ConfigError("targets must contain exactly 2 target contexts")
    targets: List[TargetContext] = []
    names = set()
    for index, item in enumerate(targets_raw):
        target = require_mapping(item, f"targets[{index}]")
        for field in ("name", "pdb", "chain", "hotspot"):
            if field not in target:
                raise ConfigError(f"targets[{index}].{field} is required")
        name = str(target["name"])
        if not name:
            raise ConfigError(f"targets[{index}].name must be non-empty")
        if name in names:
            raise ConfigError(f"Duplicate target context name: {name}")
        names.add(name)
        pdb = resolve_path(target["pdb"], f"targets[{index}].pdb", config_path)
        if check_existing_inputs and not pdb.is_file():
            raise ConfigError(f"targets[{index}].pdb does not exist: {pdb}")
        chain = str(target["chain"])
        if not chain:
            raise ConfigError(f"targets[{index}].chain must be non-empty")
        target_secondary_structure = target.get("target_secondary_structure", False)
        target_secondary_structure_mask = target.get("target_secondary_structure_mask", False)
        if not disabled_value(target_secondary_structure):
            raise ConfigError(
                f"targets[{index}].target_secondary_structure is not implemented by this adapter"
            )
        if not disabled_value(target_secondary_structure_mask):
            raise ConfigError(
                f"targets[{index}].target_secondary_structure_mask is not implemented by this adapter"
            )
        target_helicity_weight = (
            0.0
            if disabled_value(target.get("target_helicity_weight", 0.0))
            else require_number(target.get("target_helicity_weight"), f"targets[{index}].target_helicity_weight")
        )
        if target_helicity_weight != 0.0:
            raise ConfigError(
                f"targets[{index}].target_helicity_weight is not implemented by this adapter"
            )
        residue_ids = None
        if pdb.is_file():
            residue_ids = parse_pdb_chain_features(pdb, chain)["residue_id_set"]
        normalized_hotspot = normalize_hotspot(
            target["hotspot"], chain, residue_ids, f"targets[{index}].hotspot"
        )
        targets.append(
            TargetContext(
                name=name,
                pdb=pdb,
                chain=chain,
                hotspot=target["hotspot"],
                normalized_hotspot=normalized_hotspot,
                target_secondary_structure=target_secondary_structure,
                target_secondary_structure_mask=target_secondary_structure_mask,
                target_helicity_weight=target_helicity_weight,
            )
        )
    output = dict(require_mapping(raw["output"], "output"))
    for field in ("root", "fasta", "trace_jsonl", "summary_json", "failure_json"):
        if field not in output:
            raise ConfigError(f"output.{field} is required")
        output[field] = resolve_path(output[field], f"output.{field}", config_path)

    return {
        "schema_version": raw["schema_version"],
        "method": raw["method"],
        "official_author_config_available": bool(raw.get("official_author_config_available", False)),
        "diagnostic_mode": diagnostic_mode,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "num_binders": expected_num_binders,
        "binder_length": binder_length,
        "trajectory_seeds": seeds,
        "advanced_settings_json": advanced_json,
        "advanced_settings_sha256": raw["advanced_settings_sha256"],
        "advanced_settings_sha256_observed": observed_sha,
        "advanced_settings_note": raw.get("advanced_settings_note"),
        "advanced_settings_overrides": advanced_overrides,
        "af": af,
        "stages": stages,
        "selection_policy": raw["selection_policy"],
        "endpoint_selection": endpoint_selection,
        "stage_transition_policy": stage_transition_policy,
        "provenance": provenance,
        "targets": targets,
        "output": output,
    }


def load_and_validate_config(config_path: Path, check_existing_inputs: bool = False) -> Dict[str, Any]:
    return validate_config(load_yaml(config_path), config_path.resolve(), check_existing_inputs=check_existing_inputs)


def build_schedule(config: Mapping[str, Any]) -> Dict[str, Any]:
    shared_gradient_updates = 0
    shared_discrete_updates = 0
    stages = []
    gradient_evaluations = {ctx.name: 0 for ctx in config["targets"]}
    mutation_forward_evaluations = {ctx.name: 0 for ctx in config["targets"]}
    for stage in config["stages"]:
        record = {
            "name": stage.name,
            "shared_updates": stage.steps,
            "temperature_start": stage.temperature_start,
            "temperature_end": stage.temperature_end,
            "soft_end": stage.soft_end,
            "tries": stage.tries,
        }
        if stage.name == "joint_semi_greedy":
            shared_discrete_updates += stage.steps
            counts = {ctx.name: stage.steps for ctx in config["targets"]}
            for name in counts:
                mutation_forward_evaluations[name] += stage.steps
            record.update(
                {
                    "update_type": "single_shared_mutation_proposal",
                    "joint_mutation_proposals": stage.steps,
                    "context_mutation_forward_evaluations": counts,
                }
            )
        else:
            shared_gradient_updates += stage.steps
            counts = {ctx.name: stage.steps for ctx in config["targets"]}
            for name in counts:
                gradient_evaluations[name] += stage.steps
            record.update(
                {
                    "update_type": "raw_gradient_mean",
                    "context_gradient_evaluations": counts,
                }
            )
        stages.append(record)
    return {
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "gradient_aggregation": "arithmetic_mean_of_two_raw_context_gradients",
        "gradient_normalization": "once_after_raw_gradient_mean",
        "optimizer_updates_per_shared_step": 1,
        "joint_semigreedy_decision": "accept_sole_proposal_no_incumbent_comparison_mean_full_loss_recorded_only",
        "stage_transition_policy": config["stage_transition_policy"],
        "total_shared_steps": shared_gradient_updates + shared_discrete_updates,
        "total_shared_gradient_updates": shared_gradient_updates,
        "total_context_gradient_evaluations": sum(gradient_evaluations.values()),
        "per_context_gradient_evaluations": gradient_evaluations,
        "total_joint_mutation_proposals": shared_discrete_updates,
        "total_context_mutation_forward_evaluations": sum(mutation_forward_evaluations.values()),
        "per_context_mutation_forward_evaluations": mutation_forward_evaluations,
        "stages": stages,
    }


def provenance_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = config["provenance"]
    return {
        "status": provenance["status"],
        "official_author_config_available": False,
        "bindcraft_commit": provenance["bindcraft_commit"],
        "bindcraft_commit_observed": provenance["bindcraft_commit_observed"],
        "bindcraft_source_worktree": provenance["bindcraft_source_worktree"],
        "paper_pdf": str(provenance["paper_pdf"]),
        "paper_pdf_sha256": provenance["paper_pdf_sha256"],
        "paper_pdf_sha256_observed": provenance["paper_pdf_sha256_observed"],
        "advanced_settings_json": str(config["advanced_settings_json"]),
        "advanced_settings_sha256": config["advanced_settings_sha256"],
        "advanced_settings_sha256_observed": config["advanced_settings_sha256_observed"],
        "advanced_settings_note": config.get("advanced_settings_note"),
        "advanced_settings_overrides": config["advanced_settings_overrides"],
        "optimization_protocol": {
            "name": OPTIMIZATION_PROTOCOL,
            "author_published": False,
            "gradient_contexts_per_shared_update": 2,
            "gradient_aggregation": "arithmetic_mean_of_raw_gradients",
            "gradient_normalization": "once_after_raw_gradient_mean",
            "optimizer_updates_per_shared_step": 1,
            "gradient_update_count_per_trajectory": 125,
            "final_stage": "joint_semi_greedy",
            "joint_semigreedy_update_count_per_trajectory": 15,
            "mutation_proposals_per_joint_semigreedy_step": 1,
            "mutation_context_forwards_per_joint_semigreedy_step": 2,
            "joint_semigreedy_incumbent_compared": False,
            "joint_semigreedy_decision": "accept_sole_proposal_no_incumbent_comparison_mean_full_loss_recorded_only",
            "standard_bindcraft_semigreedy_used": False,
        },
        "endpoint_selection": {
            "policy": config["selection_policy"],
            **config["endpoint_selection"],
            "candidate_pool_size_per_trajectory": 3,
            "context_private_best_used": "candidate_only",
            "candidate_label_used_as_score": False,
            "get_seq_called": False,
            "posthoc_prediction_pdb_get_best": False,
            "posthoc_prediction_pdb_source": "fresh_model_0_predict_aux_for_joint_selected_sequence",
        },
        "note": provenance.get("note"),
    }


def dry_run_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    resolved_targets = [
        {
            "name": ctx.name,
            "pdb": str(ctx.pdb),
            "pdb_exists": ctx.pdb.is_file(),
            "chain": ctx.chain,
            "hotspot": ctx.hotspot,
            "normalized_hotspot": list(ctx.normalized_hotspot),
            "static_target_fingerprint": static_target_fingerprint(ctx),
        }
        for ctx in config["targets"]
    ]
    return {
        "mode": "dry_run",
        "executed": False,
        "method": "bindcraft_alt",
        "official_author_config_available": False,
        "diagnostic_mode": bool(config.get("diagnostic_mode")),
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "num_binders": config["num_binders"],
        "binder_length": config["binder_length"],
        "trajectory_seeds": config["trajectory_seeds"],
        "advanced_settings_json": str(config["advanced_settings_json"]),
        "advanced_settings_sha256": config["advanced_settings_sha256"],
        "advanced_settings_sha256_observed": config["advanced_settings_sha256_observed"],
        "advanced_settings_note": config.get("advanced_settings_note"),
        "advanced_settings_overrides": config["advanced_settings_overrides"],
        "stage_transition_policy": config["stage_transition_policy"],
        "selection_policy": config["selection_policy"],
        "endpoint_selection": {
            "policy": config["selection_policy"],
            **config["endpoint_selection"],
            "candidate_pool_size_per_trajectory": 3,
            "context_private_best_used": "candidate_only",
            "candidate_label_used_as_score": False,
            "get_seq_called": False,
            "posthoc_prediction_pdb_get_best": False,
            "posthoc_prediction_pdb_source": "fresh_model_0_predict_aux_for_joint_selected_sequence",
        },
        "counts": {
            "design_units": 1,
            "backbones": config["num_binders"],
            "binder_sequences": config["num_binders"],
            "af3_binary_jobs": 2 * config["num_binders"],
            "target_contexts": len(config["targets"]),
            "optimization_trajectories": len(config["trajectory_seeds"]),
            "target_specific_posthoc_predictions": 2 * config["num_binders"],
            "endpoint_candidates_per_trajectory": 3,
            "candidate_cross_evaluation_predictions": (
                config["num_binders"] * 3 * len(config["targets"]) * len(config["af"]["model_list"])
            ),
            "shared_optimizer_updates": config["num_binders"] * sum(
                stage.steps for stage in config["stages"] if stage.name != "joint_semi_greedy"
            ),
            "context_gradient_evaluations": (
                config["num_binders"]
                * len(config["targets"])
                * sum(stage.steps for stage in config["stages"] if stage.name != "joint_semi_greedy")
            ),
            "joint_mutation_proposals": config["num_binders"] * config["stages"][-1].steps,
            "context_mutation_forward_evaluations": (
                config["num_binders"] * len(config["targets"]) * config["stages"][-1].steps
            ),
            "total_shared_steps": config["num_binders"] * sum(stage.steps for stage in config["stages"]),
        },
        "backbone_count_semantics": (
            "one joint-selected shared endpoint from a fixed three-candidate pool per declared trajectory; "
            "the paper does not disclose whether the four binders are independent trajectories"
        ),
        "provenance": provenance_payload(config),
        "resolved_paths": {
            "bindcraft_source": str(BINDCRAFT_DIR.resolve()),
            "af_params_dir": str(config["af"]["params_dir"]),
            "output_root": str(config["output"]["root"]),
            "fasta": str(config["output"]["fasta"]),
            "trace_jsonl": str(config["output"]["trace_jsonl"]),
            "summary_json": str(config["output"]["summary_json"]),
            "failure_json": str(config["output"]["failure_json"]),
        },
        "targets": resolved_targets,
        "trajectories": [
            {
                "index": index,
                "seed": seed,
                "joint_gradient_schedule": build_schedule(config),
            }
            for index, seed in enumerate(config["trajectory_seeds"], start=1)
        ],
    }


def load_advanced_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    with config["advanced_settings_json"].open(encoding="utf-8") as handle:
        advanced = json.load(handle)
    if not isinstance(advanced, dict):
        raise ConfigError("advanced_settings_json must contain a JSON object")
    advanced = copy.deepcopy(advanced)
    advanced.update(config["advanced_settings_overrides"])
    if advanced.get("af_params_dir") in [None, ""]:
        advanced["af_params_dir"] = str(config["af"]["params_dir"])
    return advanced


def import_bindcraft_runtime() -> Dict[str, Any]:
    if not BINDCRAFT_DIR.is_dir():
        raise BindCraftAltError(f"BindCraft source directory is missing: {BINDCRAFT_DIR}")
    try:
        from colabdesign import clear_mem, mk_afdesign_model
    except Exception as exc:
        raise BindCraftAltError("Failed to import ColabDesign for --run") from exc
    return {
        "clear_mem": clear_mem,
        "mk_afdesign_model": mk_afdesign_model,
        "add_rg_loss": add_rg_loss,
        "add_i_ptm_loss": add_i_ptm_loss,
        "add_helix_loss": add_helix_loss,
    }


def add_rg_loss(model: Any, weight: float) -> None:
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    def loss_fn(_inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-model._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_threshold = 2.38 * ca.shape[0] ** 0.365
        return {"rg": jax.nn.elu(rg - rg_threshold)}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["rg"] = float(weight)


def add_i_ptm_loss(model: Any, weight: float) -> None:
    from colabdesign.af.loss import get_ptm, mask_loss

    def loss_fn(inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        return {"i_ptm": mask_loss(1 - get_ptm(inputs, outputs, interface=True))}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["i_ptm"] = float(weight)


def add_helix_loss(model: Any, weight: float) -> None:
    import jax.numpy as jnp
    from colabdesign.af.loss import _get_con_loss, get_dgram_bins

    def loss_fn(inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            residue_index = inputs["residue_index"].flatten()
            offset = residue_index[:, None] - residue_index[None, :]
        dgram = outputs["distogram"]["logits"]
        contacts = _get_con_loss(dgram, get_dgram_bins(outputs), cutoff=6.0, binary=True)
        binder_mask = np.append(np.zeros(model._target_len), np.ones(model._binder_len))
        mask_2d = np.outer(binder_mask, binder_mask)
        offset_mask = offset == 3
        offset_mask = jnp.where(mask_2d, offset_mask, 0)
        helix_loss = jnp.where(offset_mask, contacts, 0.0).sum() / (offset_mask.sum() + 1e-8)
        return {"helix": helix_loss}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["helix"] = float(weight)


def hotspot_to_bindcraft(value: Any, chain: str) -> Any:
    normalized = normalize_hotspot(value, chain, None, "hotspot")
    return ",".join(normalized) if normalized else None


def configure_colabdesign_model(model: Any, ctx: TargetContext, config: Mapping[str, Any], advanced: Mapping[str, Any], seed: int, runtime: Mapping[str, Any]) -> Any:
    model.prep_inputs(
        pdb_filename=str(ctx.pdb),
        chain=ctx.chain,
        binder_len=config["binder_length"],
        hotspot=hotspot_to_bindcraft(ctx.hotspot, ctx.chain),
        seed=seed,
        rm_aa=advanced.get("omit_AAs"),
        rm_target=False,
        rm_target_seq=advanced["rm_template_seq_design"],
        rm_target_sc=advanced["rm_template_sc_design"],
    )
    model.opt["weights"].update(
        {
            "pae": advanced["weights_pae_intra"],
            "plddt": advanced["weights_plddt"],
            "i_pae": advanced["weights_pae_inter"],
            "con": advanced["weights_con_intra"],
            "i_con": advanced["weights_con_inter"],
        }
    )
    model.opt["con"].update(
        {
            "num": advanced["intra_contact_number"],
            "cutoff": advanced["intra_contact_distance"],
            "binary": False,
            "seqsep": 9,
        }
    )
    model.opt["i_con"].update(
        {
            "num": advanced["inter_contact_number"],
            "cutoff": advanced["inter_contact_distance"],
            "binary": False,
        }
    )
    if advanced.get("use_rg_loss"):
        runtime["add_rg_loss"](model, advanced["weights_rg"])
    if advanced.get("use_i_ptm_loss"):
        runtime["add_i_ptm_loss"](model, advanced["weights_iptm"])
    helicity = 0.0 if advanced.get("weights_helicity") in [None, False, ""] else float(advanced["weights_helicity"])
    runtime["add_helix_loss"](model, helicity)
    if not disabled_value(ctx.target_secondary_structure):
        raise BindCraftAltError(
            "Target secondary-structure conditioning is outside this BindCraft-ALT adapter; "
            "use an unconditioned fixed target or implement and validate that method explicitly."
        )
    if not disabled_value(ctx.target_secondary_structure_mask):
        raise BindCraftAltError(
            "Target secondary-structure masks are outside this BindCraft-ALT adapter; "
            "use an unconditioned fixed target or implement and validate that method explicitly."
        )
    if ctx.target_helicity_weight != 0.0:
        raise BindCraftAltError(
            "Target helicity conditioning is outside this BindCraft-ALT adapter; "
            "use an unconditioned fixed target or implement and validate that method explicitly."
        )
    return model


def build_colab_models(config: Mapping[str, Any], advanced: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    runtime = import_bindcraft_runtime()
    runtime["clear_mem"]()
    models = {}
    for ctx in config["targets"]:
        model = runtime["mk_afdesign_model"](
            protocol="binder",
            debug=False,
            data_dir=str(config["af"]["params_dir"]),
            use_multimer=advanced["use_multimer_design"],
            num_recycles=advanced["num_recycles_design"],
            best_metric="loss",
        )
        models[ctx.name] = configure_colabdesign_model(model, ctx, config, advanced, seed, runtime)
    return models


def get_seq_param(model: Any) -> Any:
    if not hasattr(model, "_params") or not isinstance(model._params, dict) or "seq" not in model._params:
        raise BindCraftAltError("Model is missing _params['seq']")
    return model._params["seq"]


def get_optimizer_state(model: Any) -> Any:
    if not hasattr(model, "_state"):
        raise BindCraftAltError("Model is missing real optimizer _state; refusing sequence-only transfer")
    return model._state


def array_signature(value: Any) -> Dict[str, Any]:
    array = np.asarray(value)
    return {"kind": "array", "shape": [int(dim) for dim in array.shape], "dtype": str(array.dtype)}


def is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def pytree_signature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "keys": [str(key) for key in sorted(value.keys(), key=str)],
            "children": {
                str(key): pytree_signature(value[key]) for key in sorted(value.keys(), key=str)
            },
        }
    if is_namedtuple(value):
        return {
            "kind": "namedtuple",
            "type": type(value).__name__,
            "fields": list(value._fields),
            "children": {field: pytree_signature(getattr(value, field)) for field in value._fields},
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "length": len(value),
            "children": [pytree_signature(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "length": len(value),
            "children": [pytree_signature(item) for item in value],
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return array_signature(value)
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return {"kind": "scalar", "type": type(value).__name__}
    raise BindCraftAltError(f"Unsupported optimizer-state pytree leaf type: {type(value).__name__}")


def update_pytree_digest(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value.keys(), key=str):
            digest.update(str(key).encode())
            update_pytree_digest(digest, value[key])
        return
    if is_namedtuple(value):
        digest.update(f"namedtuple:{type(value).__name__}".encode())
        for field in value._fields:
            digest.update(field.encode())
            update_pytree_digest(digest, getattr(value, field))
        return
    if isinstance(value, tuple):
        digest.update(b"tuple")
        for item in value:
            update_pytree_digest(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"list")
        for item in value:
            update_pytree_digest(digest, item)
        return
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        digest.update(b"array")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
        return
    if value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        digest.update(json.dumps(to_jsonable(value), sort_keys=True).encode())
        return
    raise BindCraftAltError(f"Unsupported optimizer-state pytree leaf type: {type(value).__name__}")


def pytree_digest(value: Any) -> str:
    digest = hashlib.sha256()
    update_pytree_digest(digest, value)
    return digest.hexdigest()


def shared_contract(model: Any) -> Dict[str, Any]:
    seq = get_seq_param(model)
    state = get_optimizer_state(model)
    return {
        "seq_signature": array_signature(seq),
        "seq_digest": array_digest(seq),
        "state_signature": pytree_signature(state),
        "state_digest": pytree_digest(state),
    }


def shared_contract_report(models: Mapping[str, Any]) -> Dict[str, Any]:
    contexts = {name: shared_contract(model) for name, model in models.items()}
    seq_signatures = {json_digest(contract["seq_signature"]) for contract in contexts.values()}
    state_signatures = {json_digest(contract["state_signature"]) for contract in contexts.values()}
    return {
        "contexts": contexts,
        "seq_signature_compatible": len(seq_signatures) == 1,
        "state_signature_compatible": len(state_signatures) == 1,
    }


def assert_initial_shared_contract(models: Mapping[str, Any]) -> Dict[str, Any]:
    report = shared_contract_report(models)
    if not report["seq_signature_compatible"]:
        raise BindCraftAltError("Binder _params['seq'] signatures are not compatible across contexts")
    if not report["state_signature_compatible"]:
        raise BindCraftAltError("Optimizer _state pytree signatures are not compatible across contexts")
    return report


def seq_shape(model: Any) -> List[int]:
    return [int(dim) for dim in np.asarray(get_seq_param(model)).shape]


def capture_shared(model: Any) -> Dict[str, Any]:
    contract = shared_contract(model)
    return {
        "seq": get_seq_param(model),
        "state": get_optimizer_state(model),
        **contract,
    }


def seq_signatures_transfer_compatible(recipient: Mapping[str, Any], shared: Mapping[str, Any]) -> bool:
    if recipient.get("kind") != "array" or shared.get("kind") != "array":
        return False
    if recipient.get("shape") != shared.get("shape"):
        return False
    recipient_dtype = np.dtype(recipient.get("dtype"))
    shared_dtype = np.dtype(shared.get("dtype"))
    return recipient_dtype == shared_dtype or (
        np.issubdtype(recipient_dtype, np.floating) and np.issubdtype(shared_dtype, np.floating)
    )


def install_shared(model: Any, shared: Mapping[str, Any]) -> None:
    contract = shared_contract(model)
    if not seq_signatures_transfer_compatible(contract["seq_signature"], shared["seq_signature"]):
        raise BindCraftAltError(
            "Refusing to transfer binder _params['seq'] into incompatible context: "
            f"recipient={contract['seq_signature']}, shared={shared['seq_signature']}"
        )
    if contract["state_signature"] != shared["state_signature"]:
        raise BindCraftAltError(
            "Refusing to transfer optimizer _state into incompatible context: "
            f"recipient={contract['state_signature']}, shared={shared['state_signature']}"
        )
    model._params["seq"] = shared["seq"]
    model._state = shared["state"]


def feature_fingerprint(value: Any) -> Dict[str, Any]:
    return {**array_signature(value), "digest": array_digest(value)}


def static_target_fingerprint(ctx: TargetContext) -> Dict[str, Any]:
    features = parse_pdb_chain_features(ctx.pdb, ctx.chain)
    payload = {
        "pdb_sha256": sha256_file(ctx.pdb),
        "chain": ctx.chain,
        "normalized_hotspot": list(ctx.normalized_hotspot),
        "target_length": features["target_length"],
        "target_sequence": features["target_sequence"],
        "pdb_residue_index": features["residue_ids"],
        "pdb_ca_coordinate_digest": json_digest(features["ca_coordinates"]),
        "pdb_ca_mask_digest": json_digest(features["ca_mask"]),
        "pdb_atom_position_digest": json_digest(features["atom_positions"]),
    }
    return {**payload, "digest": json_digest(payload)}


def model_target_fingerprint(model: Any, ctx: TargetContext) -> Dict[str, Any]:
    static = static_target_fingerprint(ctx)
    target_len = int(getattr(model, "_target_len", 0) or 0)
    if target_len <= 0:
        raise BindCraftAltError(f"Model target length is unavailable for context {ctx.name}")
    if target_len != static["target_length"]:
        raise BindCraftAltError(
            f"Model target length mismatch for {ctx.name}: model={target_len}, pdb={static['target_length']}"
        )
    inputs = getattr(model, "_inputs", None)
    batch = inputs.get("batch") if isinstance(inputs, Mapping) else None
    if not isinstance(batch, Mapping):
        raise BindCraftAltError(f"Model _inputs['batch'] is unavailable for target fingerprint: {ctx.name}")
    required_batch_features = {
        "aatype": "target_sequence_aatype",
        "all_atom_positions": "target_coordinates",
        "all_atom_mask": "target_mask",
    }
    model_features = {}
    for source_key, output_key in required_batch_features.items():
        if source_key not in batch:
            raise BindCraftAltError(f"Model target feature is missing for {ctx.name}: {source_key}")
        array = np.asarray(batch[source_key])[:target_len]
        if array.shape[0] != target_len:
            raise BindCraftAltError(
                f"Model target feature length mismatch for {ctx.name}.{source_key}: "
                f"observed={array.shape[0]}, expected={target_len}"
            )
        model_features[output_key] = feature_fingerprint(array)
    if "residue_index" not in inputs:
        raise BindCraftAltError(f"Model target feature is missing for {ctx.name}: residue_index")
    residue_index = np.asarray(inputs["residue_index"])[:target_len]
    if residue_index.shape[0] != target_len:
        raise BindCraftAltError(
            f"Model target feature length mismatch for {ctx.name}.residue_index: "
            f"observed={residue_index.shape[0]}, expected={target_len}"
        )
    model_features["target_residue_index"] = feature_fingerprint(residue_index)
    payload = {
        **static,
        "model_target_length": target_len,
        "model_features": model_features,
    }
    payload["digest"] = json_digest(payload)
    return payload


def target_digest(model: Any, ctx: TargetContext) -> str:
    return model_target_fingerprint(model, ctx)["digest"]


def stage_step_options(stage: StageSpec, stage_step: int) -> Dict[str, Any]:
    if stage.steps <= 0 or stage_step < 1 or stage_step > stage.steps:
        raise BindCraftAltError(
            f"Invalid stage progress for {stage.name}: step={stage_step}, steps={stage.steps}"
        )
    progress = stage_step / stage.steps
    if stage.name == "logits":
        soft = stage.soft_end * progress
        return {
            "soft": soft,
            "hard": 0.0,
            "temp": stage.temperature_end,
            "dropout": True,
        }
    if stage.name == "softmax":
        temp = stage.temperature_end + (
            stage.temperature_start - stage.temperature_end
        ) * (1.0 - progress) ** 2
        return {"soft": stage.soft_end, "hard": 0.0, "temp": temp, "dropout": True}
    if stage.name == "one_hot":
        return {
            "soft": stage.soft_end,
            "hard": 1.0,
            "temp": stage.temperature_end,
            "dropout": False,
        }
    raise BindCraftAltError(f"stage_step_options does not apply to {stage.name}")


def raw_sequence_gradient(model: Any) -> Any:
    aux = getattr(model, "aux", None)
    gradient = aux.get("grad") if isinstance(aux, Mapping) else None
    if not isinstance(gradient, Mapping) or set(gradient) != {"seq"}:
        raise BindCraftAltError("Each context must produce exactly aux['grad']['seq']")
    value = gradient["seq"]
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise BindCraftAltError("Context raw sequence gradient contains non-finite values")
    if array.shape != np.asarray(get_seq_param(model)).shape:
        raise BindCraftAltError("Context raw sequence gradient shape does not match _params['seq']")
    return value


def gradient_l2_norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(-1)))


def joint_gradient_step(
    models: Mapping[str, Any],
    stage: StageSpec,
    stage_step: int,
    config: Mapping[str, Any],
    advanced: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    opts = stage_step_options(stage, stage_step)
    context_records = {}
    gradients = []
    ordered_names = [ctx.name for ctx in config["targets"]]
    if len(ordered_names) != 2:
        raise BindCraftAltError("Joint raw-gradient mean requires exactly two contexts")

    for name in ordered_names:
        model = models[name]
        install_shared(model, shared)
        installed = shared_contract(model)
        if installed["seq_digest"] != shared["seq_digest"] or installed["state_digest"] != shared["state_digest"]:
            raise BindCraftAltError(f"Context {name} did not receive the exact shared pre-update state")
        if set(getattr(model, "_params", {})) != {"seq"}:
            raise BindCraftAltError(f"Context {name} has design parameters other than binder sequence")
        model.set_opt({k: opts[k] for k in ("soft", "hard", "temp")}, dropout=opts["dropout"])
        model.run(
            num_recycles=advanced["num_recycles_design"],
            num_models=1,
            sample_models=advanced["sample_models"],
            models=config["af"]["model_list"],
            backprop=True,
        )
        gradient = raw_sequence_gradient(model)
        gradients.append(gradient)
        context_records[name] = {
            "pre_update_seq_digest": installed["seq_digest"],
            "pre_update_state_digest": installed["state_digest"],
            "raw_gradient_digest": array_digest(gradient),
            "raw_gradient_l2_norm": gradient_l2_norm(gradient),
            "metrics": step_metrics(model, config["binder_length"]),
        }
        model._save_results(save_best=True, verbose=0)

    mean_raw_gradient = (gradients[0] + gradients[1]) / 2.0
    owner = models[ordered_names[0]]
    owner.aux["grad"] = {"seq": mean_raw_gradient}
    if owner.opt.get("norm_seq_grad"):
        owner._norm_seq_grad()
    normalized_gradient = raw_sequence_gradient(owner)
    shared_params = {"seq": shared["seq"]}
    new_state, optimizer_gradient = owner._optimizer(shared["state"], {"seq": normalized_gradient}, shared_params)
    if not isinstance(optimizer_gradient, Mapping) or set(optimizer_gradient) != {"seq"}:
        raise BindCraftAltError("Shared optimizer must return exactly one binder sequence update")
    learning_rates = {float(models[name].opt["learning_rate"]) for name in ordered_names}
    if len(learning_rates) != 1:
        raise BindCraftAltError("Context learning rates differ in a joint update")
    soft = float(owner.opt.get("soft", opts["soft"]))
    temp = float(owner.opt.get("temp", opts["temp"]))
    lr_scale = (1.0 - soft) + (soft * temp)
    learning_rate = learning_rates.pop() * lr_scale
    new_seq = shared["seq"] - learning_rate * optimizer_gradient["seq"]
    new_shared = {
        "seq": new_seq,
        "state": new_state,
        "seq_signature": array_signature(new_seq),
        "seq_digest": array_digest(new_seq),
        "state_signature": pytree_signature(new_state),
        "state_digest": pytree_digest(new_state),
    }
    post_install_digests = {}
    for name in ordered_names:
        install_shared(models[name], new_shared)
        models[name]._k += 1
        post_install_digests[name] = shared_contract(models[name])["seq_digest"]
    if set(post_install_digests.values()) != {new_shared["seq_digest"]}:
        raise BindCraftAltError("Joint post-update sequence was not installed identically in both contexts")

    evidence = {
        "update_type": "joint_raw_gradient_mean",
        "pre_update_seq_digest": shared["seq_digest"],
        "pre_update_state_digest": shared["state_digest"],
        "all_contexts_same_pre_update": all(
            item["pre_update_seq_digest"] == shared["seq_digest"] for item in context_records.values()
        ),
        "contexts": context_records,
        "mean_raw_gradient_digest": array_digest(mean_raw_gradient),
        "mean_raw_gradient_l2_norm": gradient_l2_norm(mean_raw_gradient),
        "normalized_mean_gradient_digest": array_digest(normalized_gradient),
        "normalized_mean_gradient_l2_norm": gradient_l2_norm(normalized_gradient),
        "gradient_normalization": "once_after_raw_gradient_mean",
        "learning_rate_scale": lr_scale,
        "effective_learning_rate": learning_rate,
        "optimizer_update_count": 1,
        "post_update_seq_digest": new_shared["seq_digest"],
        "post_update_state_digest": new_shared["state_digest"],
        "post_install_seq_digests": post_install_digests,
    }
    return new_shared, evidence


def discrete_sequence_from_model(model: Any) -> Any:
    return np.asarray(get_seq_param(model)).argmax(-1)


def binder_plddt_vector(model: Any, binder_length: int) -> np.ndarray:
    aux = getattr(model, "aux", None)
    if not isinstance(aux, Mapping) or "plddt" not in aux:
        raise BindCraftAltError("Joint semi-greedy requires context pLDDT from the preceding evaluation")
    values = np.asarray(aux["plddt"])[-binder_length:]
    if values.shape != (binder_length,) or not np.all(np.isfinite(values)):
        raise BindCraftAltError("Context binder pLDDT vector is invalid for joint mutation proposal")
    return values


def joint_semigreedy_step(
    models: Mapping[str, Any],
    stage: StageSpec,
    config: Mapping[str, Any],
    advanced: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ordered_names = [ctx.name for ctx in config["targets"]]
    if len(ordered_names) != 2 or stage.tries != 1:
        raise BindCraftAltError("Joint semi-greedy requires two contexts and exactly one proposal")
    for name in ordered_names:
        install_shared(models[name], shared)
    pre_contracts = {name: shared_contract(models[name]) for name in ordered_names}
    if any(item["seq_digest"] != shared["seq_digest"] for item in pre_contracts.values()):
        raise BindCraftAltError("Joint semi-greedy contexts do not share the same pre-mutation sequence")

    bias_digests = {name: array_digest(models[name]._inputs["bias"]) for name in ordered_names}
    if len(set(bias_digests.values())) != 1:
        raise BindCraftAltError("Joint semi-greedy requires identical live sequence biases across contexts")
    mean_plddt = np.mean(
        np.stack([binder_plddt_vector(models[name], config["binder_length"]) for name in ordered_names]),
        axis=0,
    )
    current_seq = discrete_sequence_from_model(models[ordered_names[0]])
    proposal_model = models[ordered_names[0]]
    mutant = proposal_model._mutate(
        seq=current_seq,
        plddt=mean_plddt,
        logits=np.asarray(proposal_model._inputs["bias"]),
    )
    mutant = np.asarray(mutant)
    changed = np.argwhere(mutant != np.asarray(current_seq))
    if changed.shape[0] != 1:
        raise BindCraftAltError(f"Joint semi-greedy proposal must contain exactly one mutation, got {changed.shape[0]}")
    mutant_digest = array_digest(mutant)

    context_records = {}
    losses = []
    for name in ordered_names:
        model = models[name]
        aux = model.predict(
            seq=mutant,
            num_models=1,
            sample_models=advanced["sample_models"],
            models=config["af"]["model_list"],
            num_recycles=advanced["num_recycles_design"],
            dropout=False,
            return_aux=True,
            verbose=False,
        )
        loss = aux.get("loss") if isinstance(aux, Mapping) else None
        if loss is None and isinstance(aux, Mapping) and isinstance(aux.get("log"), Mapping):
            loss = aux["log"].get("loss")
        if loss is None or not np.isfinite(float(loss)):
            raise BindCraftAltError(f"Joint semi-greedy context {name} produced invalid full loss")
        losses.append(float(loss))
        context_records[name] = {
            "mutant_digest": mutant_digest,
            "full_design_loss": float(loss),
            "metrics": step_metrics(model, config["binder_length"]),
        }
        model._save_results(save_best=True, verbose=0)

    for name in ordered_names:
        model = models[name]
        model.set_seq(seq=mutant, bias=model._inputs["bias"])
        model._k += 1
    new_shared = capture_shared(models[ordered_names[0]])
    for name in ordered_names[1:]:
        install_shared(models[name], new_shared)
    post_digests = {name: shared_contract(models[name])["seq_digest"] for name in ordered_names}
    if set(post_digests.values()) != {new_shared["seq_digest"]}:
        raise BindCraftAltError("Joint mutant was not installed identically in both contexts")

    row, position = map(int, changed[0])
    from_index = int(np.asarray(current_seq)[row, position])
    to_index = int(mutant[row, position])
    evidence = {
        "update_type": "joint_semi_greedy_single_proposal",
        "pre_update_seq_digest": shared["seq_digest"],
        "pre_update_state_digest": shared["state_digest"],
        "proposal_call_count": 1,
        "proposal_tries": 1,
        "proposal_position": position,
        "proposal_from_aa": AA_ALPHABET[from_index],
        "proposal_to_aa": AA_ALPHABET[to_index],
        "proposal_plddt_source": "mean_cached_binder_plddt_across_two_contexts",
        "proposal_logits_source": "identical_live_sequence_bias",
        "mutant_digest": mutant_digest,
        "mutant_sequence": sequence_from_logits(np.eye(len(AA_ALPHABET))[mutant]),
        "contexts": context_records,
        "same_mutant_both_contexts": all(
            item["mutant_digest"] == mutant_digest for item in context_records.values()
        ),
        "mean_full_design_loss": float(np.mean(losses)),
        "incumbent_compared": False,
        "decision": "accept_sole_standard_tries_1_proposal",
        "shared_decision_count": 1,
        "post_update_seq_digest": new_shared["seq_digest"],
        "post_update_state_digest": new_shared["state_digest"],
        "post_install_seq_digests": post_digests,
    }
    return new_shared, evidence


def endpoint_sequence(shared: Mapping[str, Any]) -> str:
    return sequence_from_logits(shared["seq"])


def sequence_from_logits(value: Any) -> str:
    indices = np.asarray(value).argmax(-1)
    if indices.ndim == 2:
        indices = indices[0]
    return "".join(AA_ALPHABET[int(i) % len(AA_ALPHABET)] for i in indices)


def private_best_sequence(model: Any) -> str:
    """Read one context-private best as a candidate, never as an automatic winner."""

    temporary = getattr(model, "_tmp", None)
    best = temporary.get("best") if isinstance(temporary, Mapping) else None
    aux = best.get("aux") if isinstance(best, Mapping) else None
    seq = aux.get("seq") if isinstance(aux, Mapping) else None
    hard = seq.get("hard") if isinstance(seq, Mapping) else None
    if hard is None:
        raise BindCraftAltError("Context model has no private _tmp.best.aux.seq.hard candidate")
    return sequence_from_logits(hard)


def write_fasta(records: Iterable[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def write_trace_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
    handle.flush()


def predict_endpoint_for_context(model: Any, sequence: str, ctx: TargetContext, trajectory_dir: Path, config: Mapping[str, Any], advanced: Mapping[str, Any]) -> Dict[str, Any]:
    context_dir = trajectory_dir / ctx.name
    context_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = context_dir / "shared_endpoint.pdb"
    model.predict(
        seq=sequence,
        models=[config["af"]["model_list"][0]],
        num_models=1,
        sample_models=False,
        num_recycles=advanced["num_recycles_validation"],
        verbose=False,
    )
    # ``predict`` above has just produced the structure for the selected
    # shared sequence.  Save that fresh prediction, rather than the stale
    # optimization snapshot in ``_tmp['best']``; ``get_best`` applies to the
    # design trajectory, not to this post-hoc endpoint prediction.
    model.save_pdb(str(pdb_path), get_best=False)
    metrics = copy.deepcopy(getattr(model, "aux", {}).get("log", {}))
    metrics_path = context_dir / "shared_endpoint_metrics.json"
    metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"context": ctx.name, "pdb": str(pdb_path), "metrics_json": str(metrics_path), "metrics": to_jsonable(metrics)}


def evaluate_joint_candidate(
    models: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trajectory_dir: Path,
    config: Mapping[str, Any],
    advanced: Mapping[str, Any],
) -> Dict[str, Any]:
    """Cross-evaluate one candidate on both contexts and all five fixed AF models."""

    sequence = str(candidate["sequence"])
    if len(sequence) != config["binder_length"] or set(sequence) - set(AA_ALPHABET):
        raise BindCraftAltError(f"Invalid joint endpoint candidate sequence: {candidate['candidate']}")
    candidate_dir = trajectory_dir / "candidate_evaluations" / str(candidate["candidate"])
    context_results = []
    for ctx in config["targets"]:
        context_dir = candidate_dir / ctx.name
        context_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for model_id in config["af"]["model_list"]:
            if not isinstance(model_id, int) or isinstance(model_id, bool):
                raise BindCraftAltError("Joint endpoint model list must contain integer model ids")
            model = models[ctx.name]
            model.predict(
                seq=sequence,
                models=[model_id],
                num_models=1,
                sample_models=False,
                num_recycles=advanced["num_recycles_validation"],
                verbose=False,
            )
            pdb_path = context_dir / f"model_{model_id}.pdb"
            model.save_pdb(str(pdb_path), get_best=False)
            metrics = copy.deepcopy(getattr(model, "aux", {}).get("log", {}))
            full_design_loss = metrics.get(config["endpoint_selection"]["full_design_loss_key"])
            if (
                isinstance(full_design_loss, bool)
                or not isinstance(full_design_loss, (int, float, np.integer, np.floating))
                or not np.isfinite(float(full_design_loss))
            ):
                raise BindCraftAltError(
                    f"Candidate {candidate['candidate']} context {ctx.name} model {model_id} "
                    "has no finite full design loss"
                )
            binder_metrics = step_metrics(model, config["binder_length"])
            metrics["model_id"] = model_id
            metrics["binder_plddt"] = binder_metrics["binder_plddt"]
            metrics_path = context_dir / f"model_{model_id}_metrics.json"
            metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            model_results.append(
                {
                    "model_id": model_id,
                    "pdb": str(pdb_path),
                    "metrics_json": str(metrics_path),
                    "metrics": to_jsonable(metrics),
                    "full_design_loss": float(full_design_loss),
                    "binder_plddt": binder_metrics["binder_plddt"],
                }
            )
        full_losses = [item["full_design_loss"] for item in model_results]
        binder_values = [item["binder_plddt"] for item in model_results if item["binder_plddt"] is not None]
        complex_values = [item["metrics"].get("plddt") for item in model_results if item["metrics"].get("plddt") is not None]
        context_results.append(
            {
                "context": ctx.name,
                "sequence": sequence,
                "models": model_results,
                "aggregate": {
                    "model_count": len(model_results),
                    "mean_full_design_loss": float(np.mean(full_losses)),
                    "binder_plddt_mean": None if not binder_values else float(np.mean(binder_values)),
                    "complex_plddt_mean": None if not complex_values else float(np.mean(complex_values)),
                },
            }
        )
    return {**dict(candidate), "sequence": sequence, "contexts": context_results}


def score_joint_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    contexts = candidate.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 2:
        raise BindCraftAltError("Joint endpoint candidate must contain exactly two context evaluations")
    context_means = {}
    for context in contexts:
        name = context.get("context") if isinstance(context, Mapping) else None
        aggregate = context.get("aggregate") if isinstance(context, Mapping) else None
        value = aggregate.get("mean_full_design_loss") if isinstance(aggregate, Mapping) else None
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
        ):
            raise BindCraftAltError("Joint endpoint candidate has an invalid context mean full design loss")
        context_means[name] = float(value)
    if len(context_means) != 2:
        raise BindCraftAltError("Joint endpoint candidate context names must be unique")
    worst = max(context_means.values())
    return {
        "context_mean_full_design_loss": context_means,
        "worst_context_mean_full_design_loss": worst,
        "overall_context_mean_full_design_loss": float(np.mean(list(context_means.values()))),
        "worst_contexts": sorted(name for name, value in context_means.items() if value == worst),
    }


def select_joint_endpoint(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(candidates) != 3:
        raise BindCraftAltError("Joint endpoint selector requires exactly three fixed-pool candidates")
    ranked = []
    for candidate in candidates:
        sequence = candidate.get("sequence")
        pool_index = candidate.get("pool_index")
        if not isinstance(sequence, str) or not isinstance(pool_index, int) or isinstance(pool_index, bool):
            raise BindCraftAltError("Joint endpoint candidate lacks deterministic sequence/pool metadata")
        score = score_joint_candidate(candidate)
        ranked.append(
            {
                "candidate": candidate.get("candidate"),
                "source": candidate.get("source"),
                "source_context": candidate.get("source_context"),
                "pool_index": pool_index,
                "sequence": sequence,
                "score": score,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["score"]["worst_context_mean_full_design_loss"],
            item["score"]["overall_context_mean_full_design_loss"],
            item["sequence"],
            item["pool_index"],
        )
    )
    selected = ranked[0]
    return {
        "strategy": ENDPOINT_STRATEGY,
        "primary_reducer": "minimize worst-context mean full design loss",
        "tie_break": [
            "minimize overall context mean full design loss",
            "lexicographic sequence",
            "pool index only for identical duplicate sequences",
        ],
        "candidate_label_used_as_score": False,
        "selected_candidate": selected["candidate"],
        "selected_source": selected["source"],
        "selected_source_context": selected["source_context"],
        "selected_sequence": selected["sequence"],
        "selected_score": selected["score"],
        "ranking": ranked,
    }


def array_digest(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def step_metrics(model: Any, binder_length: int) -> Dict[str, Any]:
    aux = getattr(model, "aux", {})
    log = aux.get("log", {}) if isinstance(aux, Mapping) else {}
    loss = log.get("loss") if isinstance(log, Mapping) else None
    binder_plddt = None
    if isinstance(aux, Mapping) and "plddt" in aux:
        values = np.asarray(aux["plddt"])
        binder_plddt = float(values[..., -binder_length:].mean())
    elif isinstance(log, Mapping) and "plddt" in log:
        binder_plddt = float(log["plddt"])
    return {
        "loss": None if loss is None else float(loss),
        "binder_plddt": binder_plddt,
        "model_log": to_jsonable(log),
    }


def run_config(
    config: Mapping[str, Any],
    advanced: Optional[Mapping[str, Any]] = None,
    model_factory: Optional[Callable[[TargetContext, Mapping[str, Any], Mapping[str, Any], int], Any]] = None,
    predict: bool = True,
) -> Dict[str, Any]:
    advanced = dict(advanced or load_advanced_settings(config))
    out_root = config["output"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)
    trace_path = config["output"]["trace_jsonl"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_handle = trace_path.open("w", encoding="utf-8")
    fasta_records = []
    results = []
    static_target_fingerprints = {ctx.name: static_target_fingerprint(ctx) for ctx in config["targets"]}
    try:
        for traj_index, seed in enumerate(config["trajectory_seeds"], start=1):
            trajectory_dir = out_root / f"trajectory_{traj_index:02d}_seed_{seed}"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            if model_factory is None:
                models = build_colab_models(config, advanced, seed)
            else:
                models = {ctx.name: model_factory(ctx, config, advanced, seed) for ctx in config["targets"]}
            contexts = {ctx.name: ctx for ctx in config["targets"]}
            try:
                contract_report = shared_contract_report(models)
            except BindCraftAltError as exc:
                write_trace_record(
                    trace_handle,
                    {
                        "event": "initial_shared_contract_error",
                        "trajectory_index": traj_index,
                        "seed": seed,
                        "error": str(exc),
                    },
                )
                raise
            write_trace_record(
                trace_handle,
                {
                    "event": "initial_shared_contract",
                    "trajectory_index": traj_index,
                    "seed": seed,
                    **contract_report,
                },
            )
            if not contract_report["seq_signature_compatible"]:
                raise BindCraftAltError("Binder _params['seq'] signatures are not compatible across contexts")
            if not contract_report["state_signature_compatible"]:
                raise BindCraftAltError("Optimizer _state pytree signatures are not compatible across contexts")
            initial_fingerprints = {
                name: model_target_fingerprint(model, contexts[name]) for name, model in models.items()
            }
            initial_target_digest_map = {
                name: fingerprint["digest"] for name, fingerprint in initial_fingerprints.items()
            }
            shapes = {name: seq_shape(model) for name, model in models.items()}
            if len({tuple(shape) for shape in shapes.values()}) != 1:
                raise BindCraftAltError(f"Binder logits shape mismatch after initialization: {shapes}")
            ordered_names = [ctx.name for ctx in config["targets"]]
            shared = capture_shared(models[ordered_names[0]])
            gradient_evaluation_counts = {name: 0 for name in ordered_names}
            mutation_forward_counts = {name: 0 for name in ordered_names}
            shared_gradient_updates = 0
            joint_mutation_updates = 0
            global_step = 0
            for stage in config["stages"]:
                for stage_step in range(1, stage.steps + 1):
                    if stage.name == "joint_semi_greedy":
                        shared, update_evidence = joint_semigreedy_step(
                            models, stage, config, advanced, shared
                        )
                        joint_mutation_updates += 1
                        for name in ordered_names:
                            mutation_forward_counts[name] += 1
                        event = "joint_semigreedy_update"
                    else:
                        shared, update_evidence = joint_gradient_step(
                            models, stage, stage_step, config, advanced, shared
                        )
                        shared_gradient_updates += 1
                        for name in ordered_names:
                            gradient_evaluation_counts[name] += 1
                        event = "joint_gradient_update"
                    fingerprints = {
                        name: model_target_fingerprint(model, contexts[name]) for name, model in models.items()
                    }
                    target_digest_map = {
                        name: fingerprint["digest"] for name, fingerprint in fingerprints.items()
                    }
                    digest_ok = target_digest_map == initial_target_digest_map
                    current_shapes = {name: seq_shape(model) for name, model in models.items()}
                    shapes_equal = len({tuple(shape) for shape in current_shapes.values()}) == 1
                    record = {
                        "event": event,
                        "trajectory_index": traj_index,
                        "seed": seed,
                        "stage": stage.name,
                        "stage_step": stage_step,
                        "global_step": global_step + 1,
                        "target_digest_ok": digest_ok,
                        "target_fingerprint_ok": digest_ok,
                        "target_digests": target_digest_map,
                        "target_fingerprint_digests": target_digest_map,
                        "binder_logits_shapes": current_shapes,
                        "binder_logits_shapes_equal": shapes_equal,
                        "shared_gradient_updates": shared_gradient_updates,
                        "joint_mutation_updates": joint_mutation_updates,
                        "gradient_evaluation_counts": dict(gradient_evaluation_counts),
                        "mutation_forward_counts": dict(mutation_forward_counts),
                        "shared_seq_signature": shared["seq_signature"],
                        "shared_seq_digest": shared["seq_digest"],
                        "shared_binder_digest": shared["seq_digest"],
                        "shared_state_signature": shared["state_signature"],
                        "shared_state_digest": shared["state_digest"],
                        **update_evidence,
                    }
                    write_trace_record(trace_handle, record)
                    if not digest_ok:
                        raise BindCraftAltError(f"Target digest changed at trajectory {traj_index} step {global_step + 1}")
                    if not shapes_equal:
                        raise BindCraftAltError(f"Binder logits shape mismatch at trajectory {traj_index} step {global_step + 1}")
                    global_step += 1
            schedule = build_schedule(config)
            if shared_gradient_updates != schedule["total_shared_gradient_updates"]:
                raise BindCraftAltError("Shared gradient-update count does not match the declared schedule")
            if joint_mutation_updates != schedule["total_joint_mutation_proposals"]:
                raise BindCraftAltError("Joint mutation-update count does not match the declared schedule")
            if gradient_evaluation_counts != schedule["per_context_gradient_evaluations"]:
                raise BindCraftAltError("Context gradient-evaluation counts do not match the declared schedule")
            if mutation_forward_counts != schedule["per_context_mutation_forward_evaluations"]:
                raise BindCraftAltError("Context mutation-forward counts do not match the declared schedule")
            final_model = models[ordered_names[0]]
            private_best_sequences = {
                name: private_best_sequence(model) for name, model in models.items()
            }
            install_shared(final_model, shared)
            transferred_shared = capture_shared(final_model)
            if transferred_shared["seq_digest"] != shared["seq_digest"]:
                raise BindCraftAltError("Final transferred shared binder sequence digest mismatch")
            shared_current_sequence = endpoint_sequence(transferred_shared)
            candidate_pool = [
                {
                    "candidate": "final_shared_current",
                    "source": "final_shared_current",
                    "source_context": None,
                    "pool_index": 0,
                    "sequence": shared_current_sequence,
                }
            ]
            for pool_index, owner_name in enumerate(ordered_names, start=1):
                candidate_pool.append(
                    {
                        "candidate": f"private_best_{owner_name}",
                        "source": "context_private_best",
                        "source_context": owner_name,
                        "pool_index": pool_index,
                        "sequence": private_best_sequences[owner_name],
                    }
                )
            candidate_evaluations = [
                evaluate_joint_candidate(models, candidate, trajectory_dir, config, advanced)
                for candidate in candidate_pool
            ]
            joint_selection = select_joint_endpoint(candidate_evaluations)
            sequence = joint_selection["selected_sequence"]
            write_trace_record(
                trace_handle,
                {
                    "event": "joint_endpoint_selection",
                    "trajectory_index": traj_index,
                    "seed": seed,
                    **joint_selection,
                },
            )
            design_name = f"bindcraft_alt_seed_{seed}_shared_endpoint"
            fasta_records.append((design_name, sequence))
            predictions = []
            if predict:
                for ctx in config["targets"]:
                    install_shared(models[ctx.name], shared)
                    predictions.append(predict_endpoint_for_context(models[ctx.name], sequence, ctx, trajectory_dir, config, advanced))
            results.append(
                {
                    "trajectory_index": traj_index,
                    "seed": seed,
                    "sequence_name": design_name,
                    "sequence": sequence,
                    "shared_current_sequence": shared_current_sequence,
                    "private_best_sequences": private_best_sequences,
                    "candidate_pool": candidate_pool,
                    "candidate_evaluations": candidate_evaluations,
                    "joint_endpoint_selection": joint_selection,
                    "shared_gradient_updates": shared_gradient_updates,
                    "joint_mutation_updates": joint_mutation_updates,
                    "gradient_evaluation_counts": gradient_evaluation_counts,
                    "mutation_forward_counts": mutation_forward_counts,
                    "target_fingerprints": initial_fingerprints,
                    "target_digests": initial_target_digest_map,
                    "predictions": predictions,
                    "diagnostics": candidate_evaluations if config.get("diagnostic_mode") else [],
                }
            )
    finally:
        trace_handle.close()
    write_fasta(fasta_records, config["output"]["fasta"])
    schedule = build_schedule(config)
    trajectory_count = len(config["trajectory_seeds"])
    summary = {
        "status": "completed",
        "method": "bindcraft_alt",
        "official_author_config_available": False,
        "diagnostic_mode": bool(config.get("diagnostic_mode")),
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "selection_policy": "shared_endpoint",
        "endpoint_selection": config["endpoint_selection"],
        "stage_transition_policy": config["stage_transition_policy"],
        "counts": {
            "design_units": 1,
            "backbones": config["num_binders"],
            "binder_sequences": len(fasta_records),
            "af3_binary_jobs": 2 * len(fasta_records),
            "target_contexts": len(config["targets"]),
            "optimization_trajectories": len(config["trajectory_seeds"]),
            "target_specific_posthoc_predictions": 2 * len(fasta_records),
            "endpoint_candidates_per_trajectory": 3,
            "candidate_cross_evaluation_predictions": (
                len(fasta_records) * 3 * len(config["targets"]) * len(config["af"]["model_list"])
            ),
            "total_shared_steps": trajectory_count * schedule["total_shared_steps"],
            "shared_optimizer_updates": trajectory_count * schedule["total_shared_gradient_updates"],
            "context_gradient_evaluations": trajectory_count * schedule["total_context_gradient_evaluations"],
            "joint_mutation_proposals": trajectory_count * schedule["total_joint_mutation_proposals"],
            "context_mutation_forward_evaluations": (
                trajectory_count * schedule["total_context_mutation_forward_evaluations"]
            ),
        },
        "backbone_count_semantics": (
            "one joint-selected shared endpoint from a fixed three-candidate pool per declared trajectory; "
            "the paper does not disclose whether the four binders are independent trajectories"
        ),
        "provenance": provenance_payload(config),
        "advanced_settings_json": str(config["advanced_settings_json"]),
        "advanced_settings_sha256": config["advanced_settings_sha256"],
        "advanced_settings_sha256_observed": config["advanced_settings_sha256_observed"],
        "advanced_settings_overrides": config["advanced_settings_overrides"],
        "joint_optimization_schedule": schedule,
        "target_fingerprints": static_target_fingerprints,
        "key_config": {
            "binder_length": config["binder_length"],
            "trajectory_seeds": config["trajectory_seeds"],
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "stages": [stage.__dict__ for stage in config["stages"]],
            "selection_policy": config["selection_policy"],
            "endpoint_selection": config["endpoint_selection"],
            "stage_transition_policy": config["stage_transition_policy"],
            "af_model_list": config["af"]["model_list"],
        },
        "num_fasta_records": len(fasta_records),
        "fasta": str(config["output"]["fasta"]),
        "trace_jsonl": str(trace_path),
        "results": results,
    }
    atomic_write_json(config["output"]["summary_json"], summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run the explicit BindCraft-ALT joint-optimization adapter.")
    parser.add_argument("--config", type=Path, required=True, help="BindCraft-ALT YAML config.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate config and print schedules without importing JAX or ColabDesign.")
    mode.add_argument("--run", action="store_true", help="Run the adapter in an environment with ColabDesign and BindCraft dependencies.")
    return parser.parse_args(argv)


def configured_failure_path(config_path: Path) -> Optional[Path]:
    try:
        raw = load_yaml(config_path)
        output = require_mapping(raw.get("output"), "output")
        value = output.get("failure_json")
        return None if value is None else resolve_path(value, "output.failure_json", config_path.resolve())
    except Exception:
        return None


def existing_run_outputs(config: Mapping[str, Any]) -> List[Dict[str, str]]:
    output = config["output"]
    existing = []
    for kind in ("summary_json", "fasta", "trace_jsonl"):
        path = output[kind]
        if path.exists():
            existing.append({"kind": kind, "path": str(path)})
    root = output["root"]
    if root.is_dir():
        for path in sorted(root.glob("trajectory_*")):
            if path.is_dir():
                existing.append({"kind": "trajectory_dir", "path": str(path)})
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name in {"shared_endpoint.pdb", "shared_endpoint_metrics.json"}:
                existing.append({"kind": "endpoint_output", "path": str(path)})
    return existing


def assert_no_existing_run_outputs(config: Mapping[str, Any]) -> None:
    existing = existing_run_outputs(config)
    if not existing:
        return
    preview = ", ".join(f"{item['kind']}={item['path']}" for item in existing[:10])
    if len(existing) > 10:
        preview += f", ... {len(existing)} total"
    raise ExistingRunOutputError(
        "Refusing to reuse output with existing BindCraft-ALT terminal or partial artifacts: "
        f"{preview}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config: Optional[Dict[str, Any]] = None
    failure_path = configured_failure_path(args.config)
    try:
        config = load_and_validate_config(args.config, check_existing_inputs=True)
        failure_path = config["output"]["failure_json"]
        if args.dry_run:
            print(json.dumps(to_jsonable(dry_run_summary(config)), indent=2, sort_keys=True))
            return 0
        if failure_path.exists():
            raise ConfigError(f"Refusing to reuse output with an existing failure marker: {failure_path}")
        assert_no_existing_run_outputs(config)
        advanced = load_advanced_settings(config)
        summary = run_config(config, advanced=advanced, model_factory=None, predict=True)
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if (
            args.run
            and failure_path is not None
            and not failure_path.exists()
            and not isinstance(exc, ExistingRunOutputError)
        ):
            atomic_write_json(
                failure_path,
                {
                    "status": "failed",
                    "method": "bindcraft_alt",
                    "failed_at": utc_now(),
                    "config": str(args.config.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
