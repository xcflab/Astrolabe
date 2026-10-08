#!/usr/bin/env python3
"""Paper-level ComplexDesign contract helpers.

The helpers in this file are intentionally free of ColabDesign, JAX, and file
I/O.  They define the reproducibility contract that the runtime wrapper must
prove before it is allowed to launch a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class ContractError(ValueError):
    """Raised when an input violates the ComplexDesign reproduction contract."""


def _np():
    try:
        import numpy as numpy
    except ImportError as exc:
        raise ImportError("NumPy is required for ComplexDesign masks and contact counts.") from exc
    return numpy


STAGE_ORDER = ("logits", "softmax", "one_hot", "semi_greedy")
DEFAULT_STAGE_STEPS = {
    "logits": 75,
    "softmax": 45,
    "one_hot": 5,
    "semi_greedy": 15,
}
LOSS_WEIGHT_ORDER = (
    "plddt",
    "pae",
    "contact",
    "interface_pae",
    "interface_contact",
    "interface_ptm",
    "binder_radius_of_gyration",
)
DEFAULT_LOSS_WEIGHTS = {
    "plddt": 0.1,
    "pae": 0.4,
    "contact": 1.0,
    "interface_pae": 0.1,
    "interface_contact": 1.0,
    "interface_ptm": 0.05,
    "binder_radius_of_gyration": 0.3,
}


@dataclass(frozen=True)
class ChainSlices:
    """Sequence-axis slices for target-1, target-2, and binder order."""

    target_1: slice
    target_2: slice
    binder: slice
    total_length: int

    def as_ranges(self) -> Dict[str, Tuple[int, int]]:
        return {
            "target_1": (self.target_1.start, self.target_1.stop),
            "target_2": (self.target_2.start, self.target_2.stop),
            "binder": (self.binder.start, self.binder.stop),
        }


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer, got {value!r}")
    return value


def build_chain_slices(target_1_length: int, target_2_length: int, binder_length: int) -> ChainSlices:
    t1 = _positive_int(target_1_length, "target_1_length")
    t2 = _positive_int(target_2_length, "target_2_length")
    b = _positive_int(binder_length, "binder_length")
    target_1 = slice(0, t1)
    target_2 = slice(t1, t1 + t2)
    binder = slice(t1 + t2, t1 + t2 + b)
    return ChainSlices(target_1=target_1, target_2=target_2, binder=binder, total_length=t1 + t2 + b)


def build_target_pair_allow_mask(
    target_1_length: int,
    target_2_length: int,
    binder_length: int,
    dtype: Any = bool,
):
    """Return an [L,L] mask that deletes only target1-target2 pair features."""

    numpy = _np()
    slices = build_chain_slices(target_1_length, target_2_length, binder_length)
    mask = numpy.ones((slices.total_length, slices.total_length), dtype=dtype)
    mask[slices.target_1, slices.target_2] = 0
    mask[slices.target_2, slices.target_1] = 0
    return mask, slices


def build_complexdesign_masks(target_1_length: int, target_2_length: int, binder_length: int) -> Dict[str, Any]:
    allow, slices = build_target_pair_allow_mask(target_1_length, target_2_length, binder_length, dtype=bool)
    return {
        "relative_encoding_allow_mask": allow.copy(),
        "template_pair_allow_mask": allow.copy(),
        "chain_slices": slices,
        "chain_ranges": slices.as_ranges(),
    }


def build_multimer_chain_ids(target_1_length: int, target_2_length: int, binder_length: int) -> Dict[str, Any]:
    """Return explicit heteromer identifiers for target 1, target 2, binder."""

    numpy = _np()
    slices = build_chain_slices(target_1_length, target_2_length, binder_length)
    chain_ids = numpy.concatenate(
        [
            numpy.full(slices.target_1.stop - slices.target_1.start, 0, dtype=int),
            numpy.full(slices.target_2.stop - slices.target_2.start, 1, dtype=int),
            numpy.full(slices.binder.stop - slices.binder.start, 2, dtype=int),
        ]
    )
    return {
        "asym_id": chain_ids.copy(),
        "sym_id": numpy.zeros(slices.total_length, dtype=int),
        "entity_id": chain_ids.copy(),
    }


def _complexdesign_root(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("complexdesign_reproduction", config)


def stage_plan_from_config(config: Mapping[str, Any]) -> Tuple[Stage, ...]:
    root = _complexdesign_root(config)
    stages = root.get("stages", DEFAULT_STAGE_STEPS)
    return tuple(Stage(name=name, steps=_positive_int(stages.get(name), f"stages.{name}")) for name in STAGE_ORDER)


def loss_weights_from_config(config: Mapping[str, Any]) -> Dict[str, float]:
    root = _complexdesign_root(config)
    weights = root.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
    missing = [name for name in LOSS_WEIGHT_ORDER if name not in weights]
    extra = sorted(set(weights) - set(LOSS_WEIGHT_ORDER))
    if missing or extra:
        raise ContractError(f"loss_weights keys mismatch: missing={missing}, extra={extra}")
    return {name: float(weights[name]) for name in LOSS_WEIGHT_ORDER}


def assert_paper_loss_weights(weights: Mapping[str, float]) -> None:
    observed = {name: float(weights[name]) for name in LOSS_WEIGHT_ORDER}
    expected = {name: float(DEFAULT_LOSS_WEIGHTS[name]) for name in LOSS_WEIGHT_ORDER}
    if observed != expected:
        raise ContractError(f"ComplexDesign paper loss weights changed: observed={observed}, expected={expected}")


def validate_fixed_targets_and_binder_logits(
    target_1_sequence: str,
    target_2_sequence: str,
    binder_length: int,
    binder_logits: Any,
    alphabet_size: int = 20,
) -> Dict[str, Any]:
    """Prove that only binder logits are designable and targets remain fixed."""

    if not isinstance(target_1_sequence, str) or not target_1_sequence:
        raise ContractError("target_1_sequence must be a non-empty fixed sequence")
    if not isinstance(target_2_sequence, str) or not target_2_sequence:
        raise ContractError("target_2_sequence must be a non-empty fixed sequence")
    binder_len = _positive_int(binder_length, "binder_length")
    alphabet = _positive_int(alphabet_size, "alphabet_size")
    logits = _np().asarray(binder_logits)
    valid_shapes = ((binder_len, alphabet),)
    if logits.ndim == 3 and logits.shape[0] > 0:
        valid = logits.shape[1:] == (binder_len, alphabet)
    else:
        valid = logits.shape == valid_shapes[0]
    if not valid:
        raise ContractError(
            "binder_logits must contain binder residues only: "
            f"observed shape {tuple(logits.shape)}, expected {(binder_len, alphabet)} or (num_seq, {binder_len}, {alphabet})"
        )
    return {
        "target_sequences_fixed": True,
        "binder_logits_designable": True,
        "target_1_length": len(target_1_sequence),
        "target_2_length": len(target_2_sequence),
        "binder_length": binder_len,
        "binder_logits_shape": tuple(logits.shape),
    }


def pair_min(target_1_value: float, target_2_value: float) -> float:
    return min(float(target_1_value), float(target_2_value))


def pair_min_metrics(target_1_metrics: Mapping[str, float], target_2_metrics: Mapping[str, float]) -> Dict[str, float]:
    common = sorted(set(target_1_metrics) & set(target_2_metrics))
    return {name: pair_min(target_1_metrics[name], target_2_metrics[name]) for name in common}


def _as_residue_atom_array(coords: Any, field: str):
    numpy = _np()
    arr = numpy.asarray(coords, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ContractError(f"{field} must have shape [residue, atom, xyz] or [residue, xyz]")
    return arr


def _binder_contact_mask(binder_atoms: Any, target_atoms: Any, cutoff_angstrom: float):
    numpy = _np()
    binder = _as_residue_atom_array(binder_atoms, "binder_atoms")
    target = _as_residue_atom_array(target_atoms, "target_atoms")
    cutoff2 = float(cutoff_angstrom) ** 2
    d2 = numpy.sum((binder[:, :, None, None, :] - target[None, None, :, :, :]) ** 2, axis=-1)
    return numpy.any(d2 <= cutoff2, axis=(1, 2, 3))


def per_target_contact_counts(
    binder_atoms: Any,
    target_1_atoms: Any,
    target_2_atoms: Any,
    cutoff_angstrom: float = 3.5,
) -> Dict[str, int]:
    target_1_mask = _binder_contact_mask(binder_atoms, target_1_atoms, cutoff_angstrom)
    target_2_mask = _binder_contact_mask(binder_atoms, target_2_atoms, cutoff_angstrom)
    union = _np().logical_or(target_1_mask, target_2_mask)
    return {
        "target_1": int(target_1_mask.sum()),
        "target_2": int(target_2_mask.sum()),
        "any_target": int(union.sum()),
    }


def _masked_binder_contact_mask(
    binder_atoms: Any,
    binder_atom_mask: Any,
    target_atoms: Any,
    target_atom_mask: Any,
    cutoff_angstrom: float,
):
    numpy = _np()
    binder = _as_residue_atom_array(binder_atoms, "binder_atoms")
    target = _as_residue_atom_array(target_atoms, "target_atoms")
    binder_mask = numpy.asarray(binder_atom_mask, dtype=bool)
    target_mask = numpy.asarray(target_atom_mask, dtype=bool)
    if binder_mask.shape != binder.shape[:2]:
        raise ContractError(
            f"binder_atom_mask must have shape {binder.shape[:2]}, observed {binder_mask.shape}"
        )
    if target_mask.shape != target.shape[:2]:
        raise ContractError(
            f"target_atom_mask must have shape {target.shape[:2]}, observed {target_mask.shape}"
        )
    cutoff2 = float(cutoff_angstrom) ** 2
    d2 = numpy.sum((binder[:, :, None, None, :] - target[None, None, :, :, :]) ** 2, axis=-1)
    valid = binder_mask[:, :, None, None] & target_mask[None, None, :, :]
    return numpy.any(valid & (d2 <= cutoff2), axis=(1, 2, 3))


def per_target_contact_counts_masked(
    binder_atoms: Any,
    binder_atom_mask: Any,
    target_1_atoms: Any,
    target_1_atom_mask: Any,
    target_2_atoms: Any,
    target_2_atom_mask: Any,
    cutoff_angstrom: float = 3.5,
) -> Dict[str, int]:
    """Count binder interface residues while ignoring absent atom37 entries."""

    target_1_mask = _masked_binder_contact_mask(
        binder_atoms,
        binder_atom_mask,
        target_1_atoms,
        target_1_atom_mask,
        cutoff_angstrom,
    )
    target_2_mask = _masked_binder_contact_mask(
        binder_atoms,
        binder_atom_mask,
        target_2_atoms,
        target_2_atom_mask,
        cutoff_angstrom,
    )
    union = _np().logical_or(target_1_mask, target_2_mask)
    return {
        "target_1": int(target_1_mask.sum()),
        "target_2": int(target_2_mask.sum()),
        "any_target": int(union.sum()),
    }


def ca_clash_count(
    ca_coordinates: Any,
    ca_mask: Any,
    chain_lengths: Sequence[int],
    cutoff_angstrom: float = 2.5,
) -> int:
    """Count unique non-adjacent C-alpha pairs below the paper cutoff."""

    numpy = _np()
    coordinates = numpy.asarray(ca_coordinates, dtype=float)
    mask = numpy.asarray(ca_mask, dtype=bool)
    lengths = tuple(_positive_int(int(length), "chain_length") for length in chain_lengths)
    if coordinates.shape != (sum(lengths), 3):
        raise ContractError(
            f"ca_coordinates must have shape {(sum(lengths), 3)}, observed {coordinates.shape}"
        )
    if mask.shape != (sum(lengths),):
        raise ContractError(f"ca_mask must have shape {(sum(lengths),)}, observed {mask.shape}")

    chain_id = numpy.concatenate([numpy.full(length, index, dtype=int) for index, length in enumerate(lengths)])
    chain_position = numpy.concatenate([numpy.arange(length, dtype=int) for length in lengths])
    d2 = numpy.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=-1)
    valid = mask[:, None] & mask[None, :]
    upper = numpy.triu(numpy.ones_like(valid, dtype=bool), k=1)
    adjacent_same_chain = (chain_id[:, None] == chain_id[None, :]) & (
        numpy.abs(chain_position[:, None] - chain_position[None, :]) == 1
    )
    clashes = valid & upper & ~adjacent_same_chain & (d2 < float(cutoff_angstrom) ** 2)
    return int(clashes.sum())


def benchmark_pair_acceptance(
    target_1_metrics: Mapping[str, float],
    target_2_metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    interface_counts: Mapping[str, int],
    minimum_interface_binder_residues_per_target: int = 3,
) -> Dict[str, Any]:
    reduced = pair_min_metrics(target_1_metrics, target_2_metrics)
    threshold_pass = {
        name: reduced[name] >= float(threshold)
        for name, threshold in thresholds.items()
        if name in reduced
    }
    missing = [name for name in thresholds if name not in reduced]
    if missing:
        raise ContractError(f"missing pair metrics for thresholds: {missing}")
    min_interface = _positive_int(
        minimum_interface_binder_residues_per_target,
        "minimum_interface_binder_residues_per_target",
    )
    target_1_contacts = int(interface_counts.get("target_1", 0))
    target_2_contacts = int(interface_counts.get("target_2", 0))
    both_interface_pass = target_1_contacts >= min_interface and target_2_contacts >= min_interface
    return {
        "pair_min": reduced,
        "threshold_pass": threshold_pass,
        "both_interface_pass": both_interface_pass,
        "accepted": all(threshold_pass.values()) and both_interface_pass,
        "interface_counts": {
            "target_1": target_1_contacts,
            "target_2": target_2_contacts,
            "any_target": int(interface_counts.get("any_target", max(target_1_contacts, target_2_contacts))),
        },
    }


def contract_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    weights = loss_weights_from_config(config)
    assert_paper_loss_weights(weights)
    return {
        "stages": {stage.name: stage.steps for stage in stage_plan_from_config(config)},
        "loss_weights": weights,
        "stage_gate_policy": config.get("complexdesign_reproduction", {}).get("stage_gate_policy", "fail_fast"),
        "primary_pair_reducer": config.get("benchmark", {}).get("primary_pair_reducer", "min"),
        "formal_batches_authorized": bool(config.get("benchmark", {}).get("formal_batches_authorized", False)),
    }
