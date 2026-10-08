#!/usr/bin/env python3
"""Paper-based ComplexDesign AlphaFold-Multimer reproduction.

Only the two target-pair feature masks require a small ColabDesign source
patch.  All other integration uses ColabDesign v1.1.3 interfaces that exist in
the pinned upstream commit.  ``--dry-run`` does not import JAX or ColabDesign.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import traceback
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from complexdesign_contract import (  # noqa: E402
    assert_paper_loss_weights,
    build_complexdesign_masks,
    build_multimer_chain_ids,
    ca_clash_count,
    contract_summary,
    loss_weights_from_config,
    per_target_contact_counts_masked,
    stage_plan_from_config,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "benchmark_contract.yaml"
PATCH_FILE = PROJECT_DIR / "sources" / "colabdesign_v1_1_3_target_pair_mask.patch"
PATCH_MARKER = ".complexdesign_patch.json"
MULTIMER_WEIGHT_NAMES = tuple(f"params_model_{idx}_multimer_v3.npz" for idx in range(1, 6))
AF_MULTIMER_MODEL_NAMES = tuple(f"model_{idx}_multimer_v3" for idx in range(1, 6))
DEFAULT_COLABDESIGN_OPTIMIZER_NAME = "sgd"
DEFAULT_COLABDESIGN_LEARNING_RATE = 0.1
EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS = {
    "seq_ent": 0.0,
    "plddt": 0.1,
    "pae": 0.4,
    "exp_res": 0.0,
    "helix": 0.0,
    "con": 1.0,
    "i_con": 1.0,
    "i_pae": 0.1,
    "i_ptm": 0.05,
    "rg": 0.3,
}


class RuntimeContractError(RuntimeError):
    pass


class ScientificDesignFailure(RuntimeContractError):
    """A deterministic design-quality gate miss, not a runtime failure."""


def stage_gate_policy(config: Mapping[str, Any]) -> str:
    """Return the declared handling of the stage pLDDT threshold.

    ``record_only`` is the benchmark policy: the threshold is serialized for
    audit but never suppresses the final structure/sequence or AF3 handoff.
    The legacy ``fail_fast`` behavior remains available for explicitly
    constructed non-benchmark callers.
    """

    value = config.get("complexdesign_reproduction", {}).get("stage_gate_policy", "fail_fast")
    if value not in {"record_only", "fail_fast"}:
        raise RuntimeContractError(
            "complexdesign_reproduction.stage_gate_policy must be 'record_only' or 'fail_fast'"
        )
    return str(value)


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read benchmark_contract.yaml.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise RuntimeContractError(f"YAML root must be a mapping: {path}")
    return data


def resolve_pair(config: Mapping[str, Any], pair_name: str, binder_length: Optional[int]) -> Dict[str, Any]:
    pairs = config.get("target_pairs", {})
    if pair_name not in pairs:
        raise RuntimeContractError(f"Unknown target pair {pair_name!r}; available={sorted(pairs)}")
    pair = pairs[pair_name]
    low, high = pair["binder_length"]
    length = int(binder_length) if binder_length is not None else int(low)
    if length < int(low) or length > int(high):
        raise RuntimeContractError(f"binder length {length} is outside configured range [{low}, {high}] for {pair_name}")
    return {
        "pair": pair_name,
        "binder_length": length,
        "binder_length_range": [int(low), int(high)],
        "target_1": pair.get("target_1", {}),
        "target_2": pair.get("target_2", {}),
    }


def pdb_chain_lengths(path: Path, required_chains: Sequence[str] = ("B", "C")) -> Dict[str, int]:
    residues = {chain: set() for chain in required_chains}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                continue
            chain = line[21]
            if chain in residues:
                residues[chain].add((line[22:26], line[26]))
    missing = [chain for chain, seen in residues.items() if not seen]
    if missing:
        raise RuntimeContractError(f"target PDB must contain residue-bearing chains {required_chains}; missing={missing}: {path}")
    return {chain: len(seen) for chain, seen in residues.items()}


def validate_multimer_v3_weights(params_dir: Path) -> Dict[str, Any]:
    root = Path(params_dir).expanduser().resolve()
    missing = [name for name in MULTIMER_WEIGHT_NAMES if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise RuntimeContractError(f"Missing or empty AF-Multimer v3 weight files in {root}: {missing}")
    return {"af_params_dir": str(root), "multimer_v3_weights": list(MULTIMER_WEIGHT_NAMES)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path, label: str) -> Dict[str, Any]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeContractError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise RuntimeContractError(f"{label} must be a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise RuntimeContractError(f"{label} must be a non-empty file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(size),
    }


def build_input_provenance(target_pdb: Path, benchmark_config: Path, chain_lengths: Mapping[str, int]) -> Dict[str, Any]:
    target_lengths: Dict[str, int] = {}
    for chain in ("B", "C"):
        if chain not in chain_lengths:
            raise RuntimeContractError(f"--target-pdb chain length provenance lacks chain {chain!r}")
        length = int(chain_lengths[chain])
        if length <= 0:
            raise RuntimeContractError(f"--target-pdb chain {chain!r} length must be positive: {length}")
        target_lengths[chain] = length

    target_info = _file_provenance(target_pdb, "--target-pdb")
    target_info["chain_lengths"] = target_lengths
    return {
        "target_pdb": target_info,
        "benchmark_config": _file_provenance(benchmark_config, "--config"),
    }


def _colabdesign_root() -> Path:
    spec = importlib.util.find_spec("colabdesign")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeContractError("colabdesign is not importable in this Python environment")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _expected_patch_sha256() -> str:
    if not PATCH_FILE.is_file():
        raise RuntimeContractError(f"ComplexDesign ColabDesign patch is missing: {PATCH_FILE}")
    return hashlib.sha256(PATCH_FILE.read_bytes()).hexdigest()


def validate_colabdesign_patch(expected_commit: str) -> Dict[str, Any]:
    root = _colabdesign_root()
    marker_path = root / PATCH_MARKER
    if not marker_path.is_file():
        raise RuntimeContractError(f"Patched ColabDesign marker is missing: {marker_path}")
    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)

    required_true = [
        "relative_encoding_target_pair_mask",
        "template_target_pair_mask",
        "binder_target_entries_preserved",
    ]
    missing = [name for name in required_true if marker.get(name) is not True]
    if missing:
        raise RuntimeContractError(f"ColabDesign patch marker lacks required true flags: {missing}")
    if marker.get("colabdesign_commit") != expected_commit:
        raise RuntimeContractError(
            f"ColabDesign commit mismatch: observed={marker.get('colabdesign_commit')!r}, expected={expected_commit!r}"
        )
    expected_digest = _expected_patch_sha256()
    if marker.get("patch_sha256") != expected_digest:
        raise RuntimeContractError(
            f"ColabDesign patch digest mismatch: observed={marker.get('patch_sha256')!r}, expected={expected_digest!r}"
        )

    module_path = root / "af" / "alphafold" / "model" / "modules_multimer.py"
    if not module_path.is_file():
        raise RuntimeContractError(f"Patched modules_multimer.py is missing: {module_path}")
    module_text = module_path.read_text(encoding="utf-8")
    for token in (
        "COMPLEXDESIGN_TARGET_PAIR_MASK_V1",
        "complexdesign_relpos_pair_mask",
        "complexdesign_template_pair_mask",
    ):
        if token not in module_text:
            raise RuntimeContractError(f"Patched modules_multimer.py lacks {token!r}: {module_path}")
    return {
        "colabdesign_root": str(root),
        "patch_marker": str(marker_path),
        "patch_marker_payload": marker,
    }


def paper_weights_to_colabdesign(weights: Mapping[str, float]) -> Dict[str, float]:
    """Map the seven named paper losses to ColabDesign's real loss keys."""

    assert_paper_loss_weights(weights)
    return {
        "plddt": float(weights["plddt"]),
        "pae": float(weights["pae"]),
        "con": float(weights["contact"]),
        "i_pae": float(weights["interface_pae"]),
        "i_con": float(weights["interface_contact"]),
        "i_ptm": float(weights["interface_ptm"]),
        "rg": float(weights["binder_radius_of_gyration"]),
    }


def _model_loss_weights_mapping(model: Any, root_name: str) -> MutableMapping[str, Any]:
    root = getattr(model, root_name, None)
    if not isinstance(root, Mapping):
        raise RuntimeContractError(f"ColabDesign model.{root_name} must be a mapping")
    weights = root.get("weights")
    if not isinstance(weights, MutableMapping):
        raise RuntimeContractError(f"ColabDesign model.{root_name}['weights'] must be a mutable mapping")
    return weights


def _seed_complexdesign_loss_weight_keys(model: Any) -> None:
    for root_name in ("opt", "_opt"):
        weights = _model_loss_weights_mapping(model, root_name)
        weights["i_ptm"] = 0.0
        weights["rg"] = 0.0


def assert_effective_colabdesign_loss_weights(model: Any) -> Dict[str, float]:
    expected_keys = set(EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS)
    observed_by_root: Dict[str, Dict[str, float]] = {}
    for root_name in ("opt", "_opt"):
        weights = _model_loss_weights_mapping(model, root_name)
        observed: Dict[str, float] = {}
        for key, value in weights.items():
            if isinstance(key, str):
                observed[key] = float(value)
            else:
                observed[str(key)] = float(value)
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        mismatched = {
            key: {"observed": observed[key], "expected": expected}
            for key, expected in EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS.items()
            if key in observed and observed[key] != expected
        }
        if missing or extra or mismatched:
            raise RuntimeContractError(
                f"ColabDesign model.{root_name}['weights'] mismatch: "
                f"missing={missing}, extra={extra}, mismatched={mismatched}"
            )
        observed_by_root[root_name] = observed
    if observed_by_root["opt"] != observed_by_root["_opt"]:
        raise RuntimeContractError(
            "ColabDesign model.opt['weights'] and model._opt['weights'] differ: "
            f"opt={observed_by_root['opt']}, _opt={observed_by_root['_opt']}"
        )
    return dict(EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS)


def _finite_json_value(value: Any, label: str) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeContractError(f"{label} must be finite, observed={value}")
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _finite_json_value(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    return value


def _sanitize_contact_options(options: Mapping[str, Any], label: str) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in options.items():
        if key == "num_pos":
            numeric = float(value)
            if math.isinf(numeric) and numeric > 0:
                sanitized["num_pos_unbounded"] = True
                continue
            if not math.isfinite(numeric):
                raise RuntimeContractError(f"{label}.num_pos must be finite or +inf, observed={numeric}")
            sanitized["num_pos"] = int(value) if isinstance(value, int) and not isinstance(value, bool) else numeric
            continue
        sanitized[str(key)] = _finite_json_value(value, f"{label}.{key}")
    return sanitized


def validate_complexdesign_runtime_masks(model: Any, masks: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify that the injected per-run masks are exactly the paper allow mask."""

    import numpy as np

    if not isinstance(getattr(model, "_inputs", None), Mapping):
        raise RuntimeContractError("ColabDesign model._inputs must be a mapping before mask validation")

    slices = masks["chain_slices"]
    target_1_length = int(slices.target_1.stop - slices.target_1.start)
    target_2_length = int(slices.target_2.stop - slices.target_2.start)
    binder_length = int(slices.binder.stop - slices.binder.start)
    expected_blocked_entries = 2 * target_1_length * target_2_length
    expected_target_self_entries = target_1_length * target_1_length + target_2_length * target_2_length
    expected_binder_target_entries = 2 * binder_length * (target_1_length + target_2_length)
    expected_binder_binder_entries = binder_length * binder_length

    observed_by_label = {}
    for label, input_name, expected_name in (
        ("relative_encoding", "complexdesign_relpos_pair_mask", "relative_encoding_allow_mask"),
        ("template_pair", "complexdesign_template_pair_mask", "template_pair_allow_mask"),
    ):
        if input_name not in model._inputs:
            raise RuntimeContractError(f"ColabDesign model._inputs lacks {input_name}")
        expected = np.asarray(masks[expected_name])
        observed = np.asarray(model._inputs[input_name])
        if observed.shape != expected.shape:
            raise RuntimeContractError(f"{input_name} shape mismatch: observed={observed.shape}, expected={expected.shape}")
        if observed.dtype != expected.dtype:
            raise RuntimeContractError(f"{input_name} dtype mismatch: observed={observed.dtype}, expected={expected.dtype}")

        blocked_forward = int(np.count_nonzero(observed[slices.target_1, slices.target_2] == 0))
        blocked_reverse = int(np.count_nonzero(observed[slices.target_2, slices.target_1] == 0))
        if blocked_forward != target_1_length * target_2_length or blocked_reverse != target_2_length * target_1_length:
            raise RuntimeContractError(
                f"{input_name} must block every target1-target2 entry in both directions: "
                f"observed_forward={blocked_forward}, observed_reverse={blocked_reverse}, "
                f"expected_each_direction={target_1_length * target_2_length}"
            )

        target_self_entries = int(
            np.count_nonzero(observed[slices.target_1, slices.target_1] == 1)
            + np.count_nonzero(observed[slices.target_2, slices.target_2] == 1)
        )
        binder_target_entries = int(
            np.count_nonzero(observed[slices.binder, slices.target_1] == 1)
            + np.count_nonzero(observed[slices.target_1, slices.binder] == 1)
            + np.count_nonzero(observed[slices.binder, slices.target_2] == 1)
            + np.count_nonzero(observed[slices.target_2, slices.binder] == 1)
        )
        binder_binder_entries = int(np.count_nonzero(observed[slices.binder, slices.binder] == 1))
        if target_self_entries != expected_target_self_entries:
            raise RuntimeContractError(
                f"{input_name} must allow target self entries: "
                f"observed={target_self_entries}, expected={expected_target_self_entries}"
            )
        if binder_target_entries != expected_binder_target_entries:
            raise RuntimeContractError(
                f"{input_name} must allow binder-target entries: "
                f"observed={binder_target_entries}, expected={expected_binder_target_entries}"
            )
        if binder_binder_entries != expected_binder_binder_entries:
            raise RuntimeContractError(
                f"{input_name} must allow binder-binder entries: "
                f"observed={binder_binder_entries}, expected={expected_binder_binder_entries}"
            )

        if not np.array_equal(observed, expected):
            raise RuntimeContractError(
                f"{input_name} must exactly equal the ComplexDesign allow mask: "
                f"mismatches={int(np.count_nonzero(observed != expected))}"
            )
        blocked_entries = int(np.count_nonzero(observed == 0))
        if blocked_entries != expected_blocked_entries:
            raise RuntimeContractError(
                f"{input_name} blocked entry count mismatch: "
                f"observed={blocked_entries}, expected={expected_blocked_entries}"
            )

        observed_by_label[label] = {
            "shape": [int(value) for value in observed.shape],
            "dtype": str(observed.dtype),
            "blocked_entries": blocked_entries,
            "expected_blocked_entries": expected_blocked_entries,
            "target_self_allowed_entries": target_self_entries,
            "expected_target_self_allowed_entries": expected_target_self_entries,
            "binder_target_allowed_entries": binder_target_entries,
            "expected_binder_target_allowed_entries": expected_binder_target_entries,
            "binder_binder_allowed_entries": binder_binder_entries,
            "expected_binder_binder_allowed_entries": expected_binder_binder_entries,
        }

    relative_shape = observed_by_label["relative_encoding"]["shape"]
    template_shape = observed_by_label["template_pair"]["shape"]
    if relative_shape != template_shape:
        raise RuntimeContractError(f"ComplexDesign runtime mask shapes differ: relative={relative_shape}, template={template_shape}")
    return {
        "relative_encoding": True,
        "template_pair": True,
        "shape": relative_shape,
        "expected_blocked_entries": expected_blocked_entries,
        "relative_encoding_blocked_entries": observed_by_label["relative_encoding"]["blocked_entries"],
        "template_pair_blocked_entries": observed_by_label["template_pair"]["blocked_entries"],
        "details": observed_by_label,
    }


def inject_complexdesign_inputs(model: Any, chain_lengths: Mapping[str, int], binder_length: int) -> Dict[str, Any]:
    """Make the binder protocol a true target-1/target-2/binder trimer."""

    import numpy as np

    target_1_length = int(chain_lengths["B"])
    target_2_length = int(chain_lengths["C"])
    target_total = target_1_length + target_2_length
    if int(getattr(model, "_target_len", -1)) != target_total:
        raise RuntimeContractError(
            f"ColabDesign target length mismatch: observed={getattr(model, '_target_len', None)}, expected={target_total}"
        )
    if int(getattr(model, "_binder_len", -1)) != int(binder_length):
        raise RuntimeContractError(
            f"ColabDesign binder length mismatch: observed={getattr(model, '_binder_len', None)}, expected={binder_length}"
        )
    if not isinstance(getattr(model, "_inputs", None), Mapping):
        raise RuntimeContractError("ColabDesign model._inputs must be a mapping after prep_inputs")
    if not isinstance(model._inputs.get("batch"), Mapping) or "aatype" not in model._inputs["batch"]:
        raise RuntimeContractError("ColabDesign binder inputs lack batch.aatype")

    params = getattr(model, "_params", {})
    if "seq" not in params:
        raise RuntimeContractError("ColabDesign binder protocol did not create _params['seq']")
    binder_logits = np.asarray(params["seq"])
    alphabet_size = 20
    model_args = getattr(model, "_args", {})
    if isinstance(model_args, Mapping):
        alphabet_size = int(model_args.get("alphabet_size", alphabet_size))
    if (
        binder_logits.ndim != 3
        or binder_logits.shape[0] <= 0
        or binder_logits.shape[1] != int(binder_length)
        or binder_logits.shape[2] != alphabet_size
    ):
        raise RuntimeContractError(
            "Only binder logits may be designable; "
            f"observed _params['seq'] shape={binder_logits.shape}, expected=(num_seq,{binder_length},{alphabet_size})"
        )
    batch_aatype = np.asarray(model._inputs["batch"]["aatype"])
    expected_total = target_total + int(binder_length)
    if batch_aatype.shape[0] != expected_total:
        raise RuntimeContractError(
            f"ColabDesign batch.aatype length mismatch: observed={batch_aatype.shape}, expected first axis={expected_total}"
        )

    masks = build_complexdesign_masks(target_1_length, target_2_length, int(binder_length))
    chain_ids = build_multimer_chain_ids(target_1_length, target_2_length, int(binder_length))
    model._inputs["complexdesign_relpos_pair_mask"] = masks["relative_encoding_allow_mask"]
    model._inputs["complexdesign_template_pair_mask"] = masks["template_pair_allow_mask"]
    target_target_masks = validate_complexdesign_runtime_masks(model, masks)
    for name, values in chain_ids.items():
        ids = np.asarray(values, dtype=np.int32)
        if ids.shape != (expected_total,):
            raise RuntimeContractError(f"{name} must have shape {(expected_total,)}, observed={ids.shape}")
        model._inputs[name] = ids

    # ColabDesign's binder protocol normally collapses all target chains into
    # one asym_id.  Preserve the two targets as separate AF-Multimer chains and
    # keep _lengths consistent with PDB output chain boundaries.
    model._lengths = [target_1_length, target_2_length, int(binder_length)]
    model._complexdesign_fixed_target_aatype = batch_aatype[:target_total].copy()
    model._complexdesign_initial_binder_logits = binder_logits.copy()
    return {
        "chain_ranges": masks["chain_ranges"],
        "target_aatype_fixed": True,
        "binder_logits_shape": list(binder_logits.shape),
        "three_chain_asym_id": True,
        "asym_id_segments": [int(x) for x in np.unique(model._inputs["asym_id"])],
        "sym_id_segments": [int(x) for x in np.unique(model._inputs["sym_id"])],
        "entity_id_segments": [int(x) for x in np.unique(model._inputs["entity_id"])],
        "target_target_masks": target_target_masks,
    }


def configure_complexdesign_losses(model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Install the seven paper losses through real ColabDesign callbacks."""

    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants
    from colabdesign.af.loss import get_ptm, mask_loss

    root = config["complexdesign_reproduction"]
    assumptions = root["reproduction_assumptions"]
    paper_weights = loss_weights_from_config(config)
    mapped_weights = paper_weights_to_colabdesign(paper_weights)

    if not isinstance(getattr(model, "opt", None), Mapping):
        raise RuntimeContractError("ColabDesign model.opt is unavailable after prep_inputs")
    model.opt["con"].update(
        {
            "num": int(assumptions["intra_contact_number"]),
            "cutoff": float(assumptions["intra_contact_cutoff_angstrom"]),
            "binary": False,
            "seqsep": int(assumptions["intra_contact_sequence_separation"]),
        }
    )
    model.opt["i_con"].update(
        {
            "num": int(assumptions["inter_contact_number"]),
            "cutoff": float(assumptions["inter_contact_cutoff_angstrom"]),
            "binary": False,
        }
    )

    binder_length = int(model._binder_len)

    def binder_radius_of_gyration(outputs):
        xyz = outputs["structure_module"]["final_atom_positions"]
        ca = xyz[-binder_length:, residue_constants.atom_order["CA"]]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        threshold = 2.38 * ca.shape[0] ** 0.365
        return {"rg": jax.nn.elu(rg - threshold)}

    def all_interchain_iptm(inputs, outputs):
        value = 1 - get_ptm(inputs, outputs, interface=True)
        return {"i_ptm": mask_loss(value)}

    callbacks = getattr(model, "_callbacks", {}).get("model", {}).get("loss")
    if not isinstance(callbacks, list):
        raise RuntimeContractError("ColabDesign model loss callback list is unavailable")
    callbacks.extend([binder_radius_of_gyration, all_interchain_iptm])
    _seed_complexdesign_loss_weight_keys(model)
    model.set_weights(**EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS, set_defaults=True)
    effective_weights = assert_effective_colabdesign_loss_weights(model)
    model.set_opt(
        {
            "con": dict(model.opt["con"]),
            "i_con": dict(model.opt["i_con"]),
            "num_recycles": int(assumptions["design_recycles"]),
        },
        set_defaults=True,
    )
    effective_weights = assert_effective_colabdesign_loss_weights(model)
    return {
        "paper_loss_weights": dict(paper_weights),
        "colabdesign_loss_weights": mapped_weights,
        "effective_colabdesign_loss_weights": effective_weights,
        "contact_options": {
            "con": _sanitize_contact_options(model.opt["con"], "contact_options.con"),
            "i_con": _sanitize_contact_options(model.opt["i_con"], "contact_options.i_con"),
        },
        "loss_callbacks": ["binder_radius_of_gyration", "all_interchain_iptm"],
    }


def stage_runtime_parameters(config: Mapping[str, Any], binder_length: int) -> Dict[str, Dict[str, Any]]:
    root = config["complexdesign_reproduction"]
    assumptions = root["reproduction_assumptions"]
    stages = {stage.name: int(stage.steps) for stage in stage_plan_from_config(config)}
    early_logits_steps = int(assumptions["early_logits_plddt_screen_steps"])
    semi_greedy_tries = max(1, math.ceil(int(binder_length) * float(assumptions["semi_greedy_candidate_percent"]) / 100.0))
    return {
        "logits": {
            "method": "design_logits",
            "steps": stages["logits"],
            "temperature_e_soft_semantics": (
                "ColabDesign design() linearly updates opt['soft'] from soft_start to e_soft in each call; "
                "temperature remains 1.0 because e_temp is not supplied."
            ),
            "segments": [
                {
                    "name": "early_logits_confidence_screen",
                    "steps": early_logits_steps,
                    "soft_start": 0.0,
                    "e_soft": 0.9,
                    "temperature_start": 1.0,
                    "e_temp": None,
                    "temperature_schedule": "constant",
                    "dropout": True,
                },
                {
                    "name": "remaining_logits_after_gate",
                    "steps": int(stages["logits"]) - early_logits_steps,
                    "soft_start": 0.0,
                    "e_soft": 1.0,
                    "temperature_start": 1.0,
                    "e_temp": None,
                    "temperature_schedule": "constant",
                    "dropout": True,
                },
            ],
            "expected_model_sampling_events_per_step": 1,
        },
        "softmax": {
            "method": "design_soft",
            "steps": stages["softmax"],
            "temperature_e_soft_semantics": (
                "ColabDesign design_soft() fixes opt['soft']=1.0 and quadratically decays opt['temp'] "
                "from 1.0 toward e_temp."
            ),
            "soft": 1.0,
            "temperature_start": 1.0,
            "e_temp": 1e-2,
            "temperature_schedule": "quadratic_decay_to_e_temp",
            "dropout": True,
            "expected_model_sampling_events_per_step": 1,
        },
        "one_hot": {
            "method": "design_hard",
            "steps": stages["one_hot"],
            "temperature_e_soft_semantics": (
                "ColabDesign design_hard() fixes opt['soft']=1.0 and opt['hard']=1.0; "
                "temperature is constant because e_temp is not supplied."
            ),
            "soft": 1.0,
            "hard": 1.0,
            "temperature_start": 1e-2,
            "e_temp": None,
            "temperature_schedule": "constant",
            "dropout": False,
            "expected_model_sampling_events_per_step": 1,
        },
        "semi_greedy": {
            "method": "design_pssm_semigreedy",
            "steps": stages["semi_greedy"],
            "temperature_e_soft_semantics": (
                "This wrapper calls design_pssm_semigreedy with soft_iters=0 and hard_iters=1; "
                "no gradient temperature or e_soft schedule is applied in this stage."
            ),
            "soft_iters": 0,
            "hard_iters_per_call": 1,
            "candidate_sequences_per_step": semi_greedy_tries,
            "ramp_models": False,
            "expected_model_sampling_events_per_step": 2,
        },
    }


def build_runtime_serialization_info(
    config: Mapping[str, Any],
    model: Optional[Any],
    trajectory_seed: Optional[int],
    binder_length: int,
) -> Dict[str, Any]:
    assumptions = config["complexdesign_reproduction"]["reproduction_assumptions"]
    model_args = getattr(model, "_args", {}) if model is not None else {}
    model_opt = getattr(model, "opt", {}) if model is not None else {}
    if not isinstance(model_args, Mapping):
        model_args = {}
    if not isinstance(model_opt, Mapping):
        model_opt = {}
    model_names = list(getattr(model, "_model_names", AF_MULTIMER_MODEL_NAMES)) if model is not None else list(AF_MULTIMER_MODEL_NAMES)
    return {
        "schema_version": 1,
        "optimizer": {
            "name": str(model_args.get("optimizer", DEFAULT_COLABDESIGN_OPTIMIZER_NAME)),
            "learning_rate": float(model_opt.get("learning_rate", DEFAULT_COLABDESIGN_LEARNING_RATE)),
            "observed_from_model": model is not None and "optimizer" in model_args and "learning_rate" in model_opt,
        },
        "design_recycles": {
            "num_recycles": int(model_opt.get("num_recycles", assumptions["design_recycles"])),
            "ramp_recycles": False,
            "recycle_mode": str(model_args.get("recycle_mode", "last")),
            "observed_from_model": model is not None and "num_recycles" in model_opt,
        },
        "af_multimer_model_sampling": {
            "source": "colabdesign._get_model_nums return value",
            "api_observation": "model_indices_returned",
            "model_names": [str(name) for name in model_names],
            "model_names_observed_from_model": model is not None and bool(model_names),
            "num_models_per_step": 1,
            "sample_models": True,
            "trajectory_seed": int(trajectory_seed) if trajectory_seed is not None else None,
            "per_selection_seed": None,
            "per_selection_seed_reason": (
                "ColabDesign v1.1.3 samples model indices with numpy.random.choice inside _get_model_nums "
                "and returns only the selected model index list; no per-selection seed is exposed."
            ),
            "jax_prng_key": None,
            "jax_prng_key_reason": (
                "The JAX PRNG key used by the AF model is consumed inside ColabDesign run/_single and is not returned "
                "to the design callback."
            ),
        },
        "stage_best_model_selection_rule": {
            "save_best": True,
            "best_metric": str(model_args.get("best_metric", "loss")),
            "metric_higher_better": False,
            "reset_best_each_stage": True,
            "source": "ColabDesign _save_results(save_best=True) after model.clear_best() at each stage start",
        },
        "stage_parameters": stage_runtime_parameters(config, binder_length),
    }


def make_af_model(config: Mapping[str, Any], args: argparse.Namespace, chain_lengths: Mapping[str, int]):
    input_provenance = build_input_provenance(args.target_pdb, args.config, chain_lengths)
    patch_info = validate_colabdesign_patch(config["complexdesign_reproduction"]["colabdesign_commit"])
    weights_info = validate_multimer_v3_weights(args.af_params_dir)
    af_module = importlib.import_module("colabdesign.af")
    mk_afdesign_model = getattr(af_module, "mk_afdesign_model", None)
    if not callable(mk_afdesign_model):
        raise RuntimeContractError("colabdesign.af.mk_afdesign_model is not callable")

    assumptions = config["complexdesign_reproduction"]["reproduction_assumptions"]
    random.seed(args.seed)
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=True,
        data_dir=str(Path(args.af_params_dir).resolve()),
        num_recycles=int(assumptions["design_recycles"]),
        best_metric="loss",
    )
    model.prep_inputs(
        pdb_filename=str(Path(args.target_pdb).resolve()),
        chain="B,C",
        binder_len=args.binder_length,
        seed=args.seed,
    )
    input_info = inject_complexdesign_inputs(model, chain_lengths, args.binder_length)
    loss_info = configure_complexdesign_losses(model, config)
    return model, {
        "patch": patch_info,
        "weights": weights_info,
        "input_provenance": input_provenance,
        "inputs": input_info,
        "losses": loss_info,
        "runtime_serialization": build_runtime_serialization_info(config, model, args.seed, args.binder_length),
    }


def _float_scalar(value: Any, label: str) -> float:
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeContractError(f"{label} must be scalar, observed shape={array.shape}")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise RuntimeContractError(f"{label} must be finite, observed={result}")
    return result


def _mean_pair(matrix: Any, rows: slice, columns: slice) -> float:
    import numpy as np

    array = np.asarray(matrix, dtype=float)
    forward = array[rows, columns].reshape(-1)
    reverse = array[columns, rows].reshape(-1)
    return float(np.concatenate([forward, reverse]).mean())


def metrics_from_aux(aux: Mapping[str, Any], target_1_length: int, target_2_length: int, binder_length: int) -> Dict[str, Any]:
    import numpy as np

    target_total = int(target_1_length) + int(target_2_length)
    binder_slice = slice(target_total, target_total + int(binder_length))
    target_1_slice = slice(0, int(target_1_length))
    target_2_slice = slice(int(target_1_length), target_total)
    plddt = np.asarray(aux["plddt"], dtype=float)
    if plddt.shape[-1] != target_total + int(binder_length):
        raise RuntimeContractError(
            f"pLDDT length mismatch: observed={plddt.shape}, expected last axis={target_total + int(binder_length)}"
        )
    losses = {name: _float_scalar(value, f"losses.{name}") for name, value in aux.get("losses", {}).items()}
    for name, weight in EFFECTIVE_COLABDESIGN_LOSS_WEIGHTS.items():
        if name not in losses and float(weight) == 0.0:
            losses[name] = 0.0
    result = {
        "total_loss": _float_scalar(aux["loss"], "loss"),
        "binder_plddt": float(plddt[..., binder_slice].mean()),
        "ptm": _float_scalar(aux["ptm"], "ptm"),
        "i_ptm": _float_scalar(aux["i_ptm"], "i_ptm"),
        "losses": losses,
    }
    if "pae" in aux:
        result["target_1_binder_mean_pae_angstrom"] = _mean_pair(aux["pae"], target_1_slice, binder_slice)
        result["target_2_binder_mean_pae_angstrom"] = _mean_pair(aux["pae"], target_2_slice, binder_slice)
    return result


class MetricsWriter:
    def __init__(
        self,
        handle,
        target_1_length: int,
        target_2_length: int,
        binder_length: int,
        runtime_serialization: Mapping[str, Any],
        model_sampling_recorder: "ModelSamplingRecorder",
    ):
        self.handle = handle
        self.target_1_length = int(target_1_length)
        self.target_2_length = int(target_2_length)
        self.binder_length = int(binder_length)
        self.runtime_serialization = runtime_serialization
        self.model_sampling_recorder = model_sampling_recorder
        self.model_sampling_by_step = []
        self.stage = ""
        self.stage_step = 0
        self.global_step = 0

    def start_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_step = 0

    def runtime_parameters_for_stage(self, stage: str, model_sampling: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "optimizer_name": self.runtime_serialization["optimizer"]["name"],
            "learning_rate": self.runtime_serialization["optimizer"]["learning_rate"],
            "design_recycles": self.runtime_serialization["design_recycles"]["num_recycles"],
            "stage_temperature_e_soft": self.runtime_serialization["stage_parameters"][stage],
            "stage_best_model_selection_rule": self.runtime_serialization["stage_best_model_selection_rule"],
        }
        if model_sampling is not None:
            payload["af_multimer_model_sampling"] = model_sampling
        return payload

    def __call__(self, model: Any) -> None:
        self.stage_step += 1
        self.global_step += 1
        model_sampling = self.model_sampling_recorder.pop_step_sampling()
        record = {
            "record_type": "step",
            "stage": self.stage,
            "stage_step": self.stage_step,
            "global_step": self.global_step,
            "runtime_parameters": self.runtime_parameters_for_stage(self.stage, model_sampling),
            "metrics": metrics_from_aux(
                model.aux,
                self.target_1_length,
                self.target_2_length,
                self.binder_length,
            ),
        }
        self.model_sampling_by_step.append(
            {
                "stage": self.stage,
                "stage_step": self.stage_step,
                "global_step": self.global_step,
                "af_multimer_model_sampling": model_sampling,
            }
        )
        self.handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()

    def write_stage_gate(self, stage: str, gate: str, steps: int, threshold: float, metrics: Mapping[str, Any]) -> None:
        record = {
            "record_type": "stage_gate",
            "stage": stage,
            "gate": gate,
            "steps": int(steps),
            "binder_plddt_gate_threshold": float(threshold),
            "binder_plddt_gate_pass": float(metrics["binder_plddt"]) > float(threshold),
            "best_metrics": dict(metrics),
            "runtime_parameters": self.runtime_parameters_for_stage(stage),
        }
        self.handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()

    def write_stage_summary(self, stage: str, steps: int, metrics: Mapping[str, Any]) -> None:
        record = {
            "record_type": "stage_summary",
            "stage": stage,
            "steps": int(steps),
            "best_metrics": dict(metrics),
            "runtime_parameters": self.runtime_parameters_for_stage(stage),
        }
        self.handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()


class ModelSamplingRecorder:
    def __init__(self, model: Any, runtime_serialization: Mapping[str, Any]):
        self.model = model
        self.runtime_serialization = runtime_serialization
        self.model_names = list(runtime_serialization["af_multimer_model_sampling"]["model_names"])
        self._original_get_model_nums = None
        self._pending_events = []
        self._event_index = 0

    def __enter__(self) -> "ModelSamplingRecorder":
        original = getattr(self.model, "_get_model_nums", None)
        if callable(original):
            self._original_get_model_nums = original

            def wrapped_get_model_nums(*args, **kwargs):
                model_nums = original(*args, **kwargs)
                self._pending_events.append(self._event_from_model_nums(model_nums, args, kwargs))
                return model_nums

            setattr(self.model, "_get_model_nums", wrapped_get_model_nums)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_get_model_nums is not None:
            setattr(self.model, "_get_model_nums", self._original_get_model_nums)

    def _event_from_model_nums(self, model_nums: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        import numpy as np

        self._event_index += 1
        indices = [int(value) for value in np.asarray(model_nums).reshape(-1)]
        names = []
        for index in indices:
            if index < 0 or index >= len(self.model_names):
                raise RuntimeContractError(f"ColabDesign returned out-of-range AF-Multimer model index {index}")
            names.append(str(self.model_names[index]))
        num_models_arg = kwargs.get("num_models", args[0] if len(args) > 0 else None)
        sample_models_arg = kwargs.get("sample_models", args[1] if len(args) > 1 else None)
        models_arg = kwargs.get("models", args[2] if len(args) > 2 else None)
        return {
            "event_index": self._event_index,
            "source": "colabdesign._get_model_nums",
            "api_observation": "model_indices_returned",
            "num_models_argument": int(num_models_arg) if isinstance(num_models_arg, int) and not isinstance(num_models_arg, bool) else None,
            "sample_models_argument": bool(sample_models_arg) if isinstance(sample_models_arg, bool) else None,
            "models_argument": _finite_json_value(models_arg, "models_argument") if models_arg is not None else None,
            "model_indices_zero_based": indices,
            "model_names": names,
            "model_ids": names,
        }

    def pop_step_sampling(self) -> Dict[str, Any]:
        template = self.runtime_serialization["af_multimer_model_sampling"]
        events = self._pending_events
        self._pending_events = []
        return {
            "source": template["source"],
            "api_observation": template["api_observation"],
            "trajectory_seed": template["trajectory_seed"],
            "per_selection_seed": template["per_selection_seed"],
            "per_selection_seed_reason": template["per_selection_seed_reason"],
            "jax_prng_key": template["jax_prng_key"],
            "jax_prng_key_reason": template["jax_prng_key_reason"],
            "observed_event_count": len(events),
            "events": events,
        }


def _best_aux(model: Any) -> Mapping[str, Any]:
    best = getattr(model, "_tmp", {}).get("best", {})
    aux = best.get("aux")
    if not isinstance(aux, Mapping):
        raise RuntimeContractError("ColabDesign stage did not save a lowest-loss best aux result")
    return aux


def _run_gradient_stage(
    model: Any,
    stage_name: str,
    steps: int,
    writer: MetricsWriter,
    *,
    logits_end_soft: float = 1.0,
) -> None:
    common = {
        "iters": int(steps),
        "num_models": 1,
        "sample_models": True,
        "ramp_recycles": False,
        "save_best": True,
        "callback": writer,
        "verbose": 0,
    }
    assert_effective_colabdesign_loss_weights(model)
    if stage_name == "logits":
        model.design_logits(e_soft=float(logits_end_soft), **common)
    elif stage_name == "softmax":
        model.design_soft(e_temp=1e-2, **common)
    elif stage_name == "one_hot":
        model.design_hard(temp=1e-2, dropout=False, **common)
    else:
        raise RuntimeContractError(f"Unknown gradient stage: {stage_name}")


def _run_semigreedy_stage(model: Any, steps: int, tries: int, writer: MetricsWriter) -> None:
    for _ in range(int(steps)):
        assert_effective_colabdesign_loss_weights(model)
        model.design_pssm_semigreedy(
            soft_iters=0,
            hard_iters=1,
            tries=int(tries),
            num_models=1,
            sample_models=True,
            ramp_models=False,
            save_best=True,
            verbose=0,
        )
        writer(model)


def run_stages(
    model: Any,
    config: Mapping[str, Any],
    metrics_jsonl: Path,
    chain_lengths: Mapping[str, int],
    binder_length: int,
    runtime_serialization: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    threshold = float(config["complexdesign_reproduction"]["stage_plddt_threshold"])
    mutation_percent = float(
        config["complexdesign_reproduction"]["reproduction_assumptions"]["semi_greedy_candidate_percent"]
    )
    early_logits_steps = int(
        config["complexdesign_reproduction"]["reproduction_assumptions"]["early_logits_plddt_screen_steps"]
    )
    semigreedy_tries = max(1, math.ceil(int(binder_length) * mutation_percent / 100.0))
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stages = stage_plan_from_config(config)
    stage_records = []
    if runtime_serialization is None:
        runtime_serialization = build_runtime_serialization_info(config, model, None, binder_length)

    with ModelSamplingRecorder(model, runtime_serialization) as recorder, metrics_jsonl.open("w", encoding="utf-8") as handle:
        writer = MetricsWriter(handle, chain_lengths["B"], chain_lengths["C"], binder_length, runtime_serialization, recorder)
        for index, stage in enumerate(stages):
            model.clear_best()
            writer.start_stage(stage.name)
            early_gate = None
            if stage.name == "logits":
                if early_logits_steps <= 0 or early_logits_steps >= int(stage.steps):
                    raise ScientificDesignFailure(
                        "early_logits_plddt_screen_steps must be positive and smaller than the logits stage: "
                        f"screen={early_logits_steps}, logits={stage.steps}"
                    )
                _run_gradient_stage(
                    model,
                    stage.name,
                    early_logits_steps,
                    writer,
                    logits_end_soft=0.9,
                )
                early_metrics = metrics_from_aux(
                    _best_aux(model),
                    chain_lengths["B"],
                    chain_lengths["C"],
                    binder_length,
                )
                early_gate = {
                    "gate": "early_logits_confidence_screen",
                    "steps": early_logits_steps,
                    "best_metrics": early_metrics,
                    "binder_plddt_gate_threshold": threshold,
                    "binder_plddt_gate_pass": float(early_metrics["binder_plddt"]) > threshold,
                    "runtime_parameters": writer.runtime_parameters_for_stage(stage.name),
                }
                writer.write_stage_gate(
                    stage.name,
                    early_gate["gate"],
                    early_logits_steps,
                    threshold,
                    early_metrics,
                )
                if not early_gate["binder_plddt_gate_pass"] and stage_gate_policy(config) == "fail_fast":
                    raise ScientificDesignFailure(
                        f"Early logits screen binder_plddt={early_metrics['binder_plddt']} did not pass >{threshold}; "
                        "stopping before the remaining logits steps."
                    )
                model.clear_best()
                _run_gradient_stage(
                    model,
                    stage.name,
                    int(stage.steps) - early_logits_steps,
                    writer,
                    logits_end_soft=1.0,
                )
            elif stage.name == "semi_greedy":
                _run_semigreedy_stage(model, stage.steps, semigreedy_tries, writer)
            else:
                _run_gradient_stage(model, stage.name, stage.steps, writer)

            best_metrics = metrics_from_aux(
                _best_aux(model),
                chain_lengths["B"],
                chain_lengths["C"],
                binder_length,
            )
            record = {
                "stage": stage.name,
                "steps": stage.steps,
                "best_metrics": best_metrics,
                "binder_plddt_gate_threshold": threshold,
                "binder_plddt_gate_pass": float(best_metrics["binder_plddt"]) > threshold,
                "runtime_parameters": writer.runtime_parameters_for_stage(stage.name),
            }
            if stage.name == "semi_greedy":
                record["candidate_sequences_per_step"] = semigreedy_tries
            if early_gate is not None:
                record["early_logits_confidence_screen"] = early_gate
            writer.write_stage_summary(stage.name, stage.steps, best_metrics)
            stage_records.append(record)
            if index < len(stages) - 1 and not record["binder_plddt_gate_pass"] and stage_gate_policy(config) == "fail_fast":
                raise ScientificDesignFailure(
                    f"Stage {stage.name} binder_plddt={best_metrics['binder_plddt']} did not pass >{threshold}; stopping before next stage."
                )
    return {
        "stage_records": stage_records,
        "metrics_jsonl": str(metrics_jsonl),
        "total_optimization_steps": sum(stage.steps for stage in stages),
        "early_logits_plddt_screen_steps": early_logits_steps,
        "semi_greedy_candidate_sequences_per_step": semigreedy_tries,
        "runtime_serialization": dict(runtime_serialization),
        "af_multimer_model_sampling_by_step": writer.model_sampling_by_step,
    }


def final_structure_metrics(
    aux: Mapping[str, Any],
    config: Mapping[str, Any],
    chain_lengths: Mapping[str, int],
    binder_length: int,
) -> Dict[str, Any]:
    import numpy as np

    root = config["complexdesign_reproduction"]
    method_filter = root["method_filter"]
    extra_filter = root["benchmark_extra_filter"]
    target_1_length = int(chain_lengths["B"])
    target_2_length = int(chain_lengths["C"])
    target_total = target_1_length + target_2_length
    positions = np.asarray(aux["atom_positions"], dtype=float)
    masks = np.asarray(aux["atom_mask"], dtype=float) > 0.5
    expected_length = target_total + int(binder_length)
    if positions.shape[:2] != (expected_length, 37) or masks.shape != (expected_length, 37):
        raise RuntimeContractError(
            f"final atom arrays must be [{expected_length},37,3] and [{expected_length},37]; observed={positions.shape},{masks.shape}"
        )

    t1 = slice(0, target_1_length)
    t2 = slice(target_1_length, target_total)
    binder = slice(target_total, expected_length)
    contact_counts = per_target_contact_counts_masked(
        positions[binder],
        masks[binder],
        positions[t1],
        masks[t1],
        positions[t2],
        masks[t2],
        cutoff_angstrom=float(method_filter["interface_atom_distance_angstrom"]),
    )
    clashes = ca_clash_count(
        positions[:, 1],
        masks[:, 1],
        [target_1_length, target_2_length, int(binder_length)],
        cutoff_angstrom=float(method_filter["ca_clash_distance_angstrom"]),
    )
    binder_plddt = metrics_from_aux(aux, target_1_length, target_2_length, binder_length)["binder_plddt"]
    method_checks = {
        "binder_plddt": binder_plddt > float(method_filter["final_binder_plddt_min"]),
        "ca_clashes": clashes == 0,
        "total_interface_binder_residues": contact_counts["any_target"]
        >= int(method_filter["minimum_total_interface_binder_residues"]),
    }
    benchmark_checks = {
        "target_1_interface_binder_residues": contact_counts["target_1"]
        >= int(extra_filter["minimum_interface_binder_residues_per_target"]),
        "target_2_interface_binder_residues": contact_counts["target_2"]
        >= int(extra_filter["minimum_interface_binder_residues_per_target"]),
    }
    return {
        "binder_plddt": binder_plddt,
        "ca_clash_count": clashes,
        "interface_binder_residue_counts": contact_counts,
        "method_filter_checks": method_checks,
        "method_filter_pass": all(method_checks.values()),
        "benchmark_extra_filter_checks": benchmark_checks,
        "benchmark_extra_filter_pass": all(benchmark_checks.values()),
    }


def runtime_state_checks(model: Any) -> Dict[str, Any]:
    import numpy as np

    target_total = int(model._target_len)
    target_now = np.asarray(model._inputs["batch"]["aatype"][:target_total])
    target_fixed = np.asarray(model._complexdesign_fixed_target_aatype)
    target_unchanged = bool(np.array_equal(target_now, target_fixed))
    binder_now = np.asarray(model._params["seq"])
    binder_initial = np.asarray(model._complexdesign_initial_binder_logits)
    if binder_now.shape != binder_initial.shape:
        raise RuntimeContractError(
            f"binder logits shape changed during optimization: initial={binder_initial.shape}, final={binder_now.shape}"
        )
    if np.isnan(binder_initial).any() or np.isnan(binder_now).any():
        raise RuntimeContractError("Binder logits contain NaN values after optimization")
    finite_now = np.isfinite(binder_now)
    nonfinite_count = int(binder_now.size - np.count_nonzero(finite_now))
    posinf_count = int(np.count_nonzero(np.isposinf(binder_now)))
    neginf_count = int(np.count_nonzero(np.isneginf(binder_now)))
    nan_count = int(np.count_nonzero(np.isnan(binder_now)))
    finite_pair = np.isfinite(binder_initial) & finite_now
    finite_comparable_entry_count = int(np.count_nonzero(finite_pair))
    finite_deltas = np.abs(binder_now[finite_pair] - binder_initial[finite_pair])
    finite_entry_changed_count = int(np.count_nonzero(finite_deltas > 0.0))
    current_argmax = np.argmax(binder_now, axis=-1)
    initial_argmax = np.argmax(binder_initial, axis=-1)
    argmax_changed_positions = int(np.count_nonzero(current_argmax != initial_argmax))
    binder_updated = argmax_changed_positions > 0 or finite_entry_changed_count > 0
    if not target_unchanged:
        raise RuntimeContractError("Fixed target sequence features changed during binder optimization")
    if not binder_updated:
        raise RuntimeContractError("Binder logits did not change during optimization")
    result = {
        "target_sequences_unchanged": target_unchanged,
        "binder_logits_shape": [int(value) for value in binder_now.shape],
        "all_finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nan_count": nan_count,
        "argmax_changed_positions": argmax_changed_positions,
        "finite_comparable_entry_count": finite_comparable_entry_count,
        "finite_entry_changed_count": finite_entry_changed_count,
        "finite_max_abs_change_available": finite_comparable_entry_count > 0,
        "binder_logits_updated": binder_updated,
    }
    if finite_comparable_entry_count > 0:
        result["finite_max_abs_change"] = float(finite_deltas.max(initial=0.0))
    else:
        result["finite_max_abs_change_reason"] = "no_finite_comparable_entries"
    return result


def write_outputs(
    model: Any,
    output_dir: Path,
    pair: str,
    seed: int,
    binder_length: int,
    stage_result: Mapping[str, Any],
    config: Mapping[str, Any],
    chain_lengths: Mapping[str, int],
    runtime_info: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{pair}_seed{seed}_L{binder_length}"
    fasta_path = output_dir / f"{prefix}_binder.fasta"
    pdb_path = output_dir / f"{prefix}_complex.pdb"
    summary_path = output_dir / f"{prefix}_summary.json"

    sequences = model.get_seq(get_best=True)
    if not isinstance(sequences, list) or len(sequences) != 1 or not isinstance(sequences[0], str):
        raise RuntimeContractError(f"ColabDesign get_seq(get_best=True) returned unexpected value: {sequences!r}")
    sequence = sequences[0]
    if len(sequence) != int(binder_length):
        raise RuntimeContractError(f"Binder sequence length mismatch: observed={len(sequence)}, expected={binder_length}")
    fasta_path.write_text(f">{prefix}\n{sequence}\n", encoding="utf-8")
    model.save_pdb(str(pdb_path), get_best=True)
    if not pdb_path.is_file() or pdb_path.stat().st_size == 0:
        raise RuntimeContractError(f"save_pdb did not create a non-empty PDB: {pdb_path}")

    best_aux = _best_aux(model)
    state_checks = runtime_state_checks(model)
    structure_metrics = final_structure_metrics(best_aux, config, chain_lengths, binder_length)
    summary = {
        "status": "completed",
        "pair": pair,
        "seed": seed,
        "binder_length": binder_length,
        "stage_gate_policy": stage_gate_policy(config),
        "stage_gate_semantics": (
            "pLDDT threshold is recorded for audit only and does not block downstream AF3"
            if stage_gate_policy(config) == "record_only"
            else "legacy fail-fast continuation policy"
        ),
        "binder_sequence": sequence,
        "binder_fasta": str(fasta_path),
        "complex_pdb": str(pdb_path),
        "complex_pdb_chain_roles": {"A": "target_1", "B": "target_2", "C": "binder"},
        "metrics_jsonl": stage_result["metrics_jsonl"],
        "stage_records": stage_result["stage_records"],
        "final_metrics": stage_result["stage_records"][-1]["best_metrics"],
        "runtime_state_checks": state_checks,
        "final_structure_filter": structure_metrics,
        "runtime_info": dict(runtime_info),
        "runtime_serialization": dict(stage_result["runtime_serialization"]),
        "af_multimer_model_sampling_by_step": stage_result["af_multimer_model_sampling_by_step"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def dry_run_payload(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pair = resolve_pair(config, args.pair, args.binder_length)
    chain_lengths = None
    if args.target_pdb is not None and Path(args.target_pdb).exists():
        chain_lengths = pdb_chain_lengths(args.target_pdb)
    return {
        "dry_run": True,
        "pair": pair,
        "target_pdb": str(Path(args.target_pdb).resolve()) if args.target_pdb else None,
        "af_params_dir": str(Path(args.af_params_dir).resolve()) if args.af_params_dir else None,
        "output_dir": str(Path(args.output_dir).resolve()),
        "seed": args.seed,
        "contract": contract_summary(config),
        "stage_gate_policy": stage_gate_policy(config),
        "chain_lengths_from_target_pdb": chain_lengths,
        "run_mode_gates": [
            ".complexdesign_patch.json with matching commit and patch digest",
            "params_model_[1-5]_multimer_v3.npz",
            "three explicit AF-Multimer asym_id segments",
            "real ColabDesign loss callbacks for rg and i_ptm",
            "per-step JSONL with actual _get_model_nums AF-Multimer model selections",
            "per-stage lowest-loss aux with serialized best-model rule",
        ],
        "runtime_serialization": build_runtime_serialization_info(config, None, args.seed, args.binder_length),
        "heavy_modules_imported": {
            "jax": any(name == "jax" or name.startswith("jax.") for name in sys.modules),
            "colabdesign": any(name == "colabdesign" or name.startswith("colabdesign.") for name in sys.modules),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-based ComplexDesign AF-Multimer trajectory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to benchmark_contract.yaml.")
    parser.add_argument("--pair", required=True, help="Target pair key from benchmark_contract.yaml.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the trajectory wrapper.")
    parser.add_argument("--binder-length", type=int, default=None, help="Binder length. Defaults to the configured lower bound.")
    parser.add_argument("--target-pdb", type=Path, default=None, help="Prepared target-only PDB containing chains B and C.")
    parser.add_argument("--af-params-dir", type=Path, default=None, help="Directory containing AF-Multimer v3 parameter files.")
    parser.add_argument("--output-dir", type=Path, default=Path("complexdesign_afm_out"), help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved contract without importing JAX or ColabDesign.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    pair = resolve_pair(config, args.pair, args.binder_length)
    args.binder_length = pair["binder_length"]
    if args.dry_run:
        json.dump(dry_run_payload(config, args), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.target_pdb is None:
        raise RuntimeContractError("--target-pdb is required outside --dry-run")
    if args.af_params_dir is None:
        raise RuntimeContractError("--af-params-dir is required outside --dry-run")

    chain_lengths = pdb_chain_lengths(args.target_pdb)
    metrics_jsonl = Path(args.output_dir) / f"{args.pair}_seed{args.seed}_L{args.binder_length}_metrics.jsonl"
    runtime_info: Dict[str, Any] = {
        "runtime_serialization": build_runtime_serialization_info(config, None, args.seed, args.binder_length)
    }
    try:
        model, runtime_info = make_af_model(config, args, chain_lengths)
        stage_result = run_stages(
            model,
            config,
            metrics_jsonl,
            chain_lengths,
            args.binder_length,
            runtime_info["runtime_serialization"],
        )
        summary = write_outputs(
            model,
            args.output_dir,
            args.pair,
            args.seed,
            args.binder_length,
            stage_result,
            config,
            chain_lengths,
            runtime_info,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except ScientificDesignFailure as exc:
        failed = Path(args.output_dir) / f"{args.pair}_seed{args.seed}_L{args.binder_length}_FAILED.json"
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(
            json.dumps(
                {
                    "status": "scientific_failed",
                    "failure_class": "design_quality_gate",
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                    "metrics_jsonl": str(metrics_jsonl),
                    "runtime_info": runtime_info,
                    "runtime_serialization": runtime_info.get("runtime_serialization"),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(str(exc), file=sys.stderr)
        return 10
    except Exception:
        failed = Path(args.output_dir) / f"{args.pair}_seed{args.seed}_L{args.binder_length}_FAILED.json"
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "traceback": traceback.format_exc(),
                    "metrics_jsonl": str(metrics_jsonl),
                    "runtime_info": runtime_info,
                    "runtime_serialization": runtime_info.get("runtime_serialization"),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
