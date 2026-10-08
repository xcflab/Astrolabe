from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from astrolabe.core.repeat_symmetry import (
    apply_transform,
    assign_atom_bundle_coordinates,
    AtomBundle,
    average_rotation_matrices,
    build_internal_repeat_movable_groups,
    build_internal_repeat_residue_groups,
    build_shared_core_residue_groups,
    canonicalize_exact_cyclic_cn,
    collect_atom_bundle,
    compose_transforms,
    cyclic_group_transforms_canonical,
    cyclic_oligomer_atom_bundles,
    dihedral_group_transforms_canonical,
    harmonize_atom_bundles,
    kabsch_transform,
    load_structure,
    protein_residues,
    residue_ca_coordinates,
    select_residues,
    rotation_axis,
    save_structure,
    unique_protomer_chain_map,
    validate_compatible_atom_bundles,
)
from astrolabe.structure_processors import StructureProcessor, register_structure_processor


def _case_suffix(input_file: str | Path) -> str:
    name = Path(input_file).name.lower()
    return ".cif" if any(ext in name for ext in [".cif", ".mmcif"]) else ".pdb"


def _parse_manual_interface_residue_indices(
    raw: str | None,
    *,
    residue_count: int,
) -> set[int]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("manual interface selection requires non-empty residue ranges.")
    out: set[int] = set()
    for token in text.split(","):
        part = token.strip()
        if not part:
            raise ValueError(f"Invalid empty interface range in {text!r}.")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end < start or end > residue_count:
                raise ValueError(
                    f"Invalid interface range {part!r} for {residue_count} residues."
                )
            out.update(range(start - 1, end))
        else:
            pos = int(part)
            if pos <= 0 or pos > residue_count:
                raise ValueError(
                    f"Invalid interface residue {part!r} for {residue_count} residues."
                )
            out.add(pos - 1)
    if not out:
        raise ValueError("manual interface selection resolved to no residues.")
    return out


def _subset_structure_with_atom_layout(
    structure,
    *,
    chain_id: str,
    residue_indices: np.ndarray,
    residue_atom_names: tuple[tuple[str, ...], ...],
):
    subset = copy.deepcopy(structure)
    model = next(subset.get_models())
    keep_ids = set()
    source_residues = protein_residues(structure, chain_id)
    for residue in [source_residues[i] for i in np.asarray(residue_indices, dtype=int).tolist()]:
        keep_ids.add(residue.id)
    for candidate in list(model):
        if str(candidate.id) != str(chain_id):
            model.detach_child(candidate.id)
            continue
        residue_idx = 0
        for residue in list(candidate):
            if residue.id not in keep_ids:
                candidate.detach_child(residue.id)
                continue
            keep_atoms = set(residue_atom_names[residue_idx])
            residue_idx += 1
            for atom in list(residue):
                if str(atom.id) not in keep_atoms:
                    residue.detach_child(atom.id)
    return subset


def _write_case_structure(
    structure,
    output_path: Path,
    *,
    like_input: str | Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_structure(structure, output_path, like_input=like_input)
    return str(output_path)


def _build_rigid_template_structure(
    *,
    source_structure,
    chain_id: str,
    residue_indices: np.ndarray,
    target_ca_coords: np.ndarray,
):
    template_structure = copy.deepcopy(source_structure)
    template_structure = _subset_structure_with_atom_layout(
        template_structure,
        chain_id=chain_id,
        residue_indices=residue_indices,
        residue_atom_names=tuple(
            tuple(str(atom.id) for atom in residue.get_atoms())
            for residue in select_residues(protein_residues(source_structure, chain_id), residue_indices)
        ),
    )
    residues = protein_residues(template_structure, chain_id)
    bundle = collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
    source_ca = residue_ca_coordinates(residues, np.arange(len(residues), dtype=int))
    corr_R, corr_t = kabsch_transform(source_ca, target_ca_coords)
    assign_atom_bundle_coordinates(bundle, apply_transform(bundle.coords, corr_R, corr_t))
    return template_structure


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _principal_axis(coords: np.ndarray) -> np.ndarray:
    centered = np.asarray(coords, dtype=np.float64) - np.mean(np.asarray(coords, dtype=np.float64), axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-8:
        unit = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        unit = unit / norm
    ux, uy, uz = unit.tolist()
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + ux * ux * one_minus_c, ux * uy * one_minus_c - uz * s, ux * uz * one_minus_c + uy * s],
            [uy * ux * one_minus_c + uz * s, c + uy * uy * one_minus_c, uy * uz * one_minus_c - ux * s],
            [uz * ux * one_minus_c - uy * s, uz * uy * one_minus_c + ux * s, c + uz * uz * one_minus_c],
        ],
        dtype=np.float64,
    )


def _align_vector_to_z_rotation(axis: np.ndarray) -> np.ndarray:
    src = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(src))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float64)
    src = src / norm
    dst = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        return _rotation_matrix_from_axis_angle(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            math.pi,
        )
    rot_axis = np.cross(src, dst)
    return _rotation_matrix_from_axis_angle(rot_axis, math.acos(dot))


def _canonicalize_asu_by_pca(
    coords: np.ndarray,
    *,
    target_centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(coords, dtype=np.float64)
    centroid = np.mean(arr, axis=0, keepdims=True)
    principal = _principal_axis(arr)
    if float(np.dot(principal, centroid[0])) < 0.0:
        principal = -principal
    align_R = _align_vector_to_z_rotation(principal)
    aligned = apply_transform(arr - centroid, align_R, np.zeros(3, dtype=np.float64))
    aligned_centroid = apply_transform(centroid, align_R, np.zeros(3, dtype=np.float64))[0]
    xy_norm = float(np.linalg.norm(aligned_centroid[:2]))
    around_z_angle = math.atan2(float(aligned_centroid[1]), float(aligned_centroid[0])) if xy_norm > 1e-8 else 0.0
    around_z = _rotation_matrix_z(around_z_angle)
    if target_centroid is None:
        final_centroid = aligned_centroid @ around_z
    else:
        final_centroid = np.asarray(target_centroid, dtype=np.float64)
    canonical = apply_transform(aligned, around_z, final_centroid)
    canonical_centroid = np.mean(canonical, axis=0)
    return canonical, {
        "axis_source": "monomer_pca",
        "pca_axis": [float(x) for x in principal.tolist()],
        "centroid": [float(x) for x in np.mean(arr, axis=0).tolist()],
        "target_centroid": [float(x) for x in np.asarray(final_centroid, dtype=np.float64).tolist()],
        "canonical_centroid": [float(x) for x in canonical_centroid.tolist()],
        "around_z_angle_rad": float(around_z_angle),
    }


def _fit_rpxdock_geometry_kwargs(
    canonical_average_coords: np.ndarray,
    *,
    point_group: str = "C",
) -> dict[str, Any]:
    centroid = np.mean(np.asarray(canonical_average_coords, dtype=np.float64), axis=0)
    radius = float(np.linalg.norm(centroid[:2]))
    radius_pad = max(5.0, 0.15 * max(radius, 1.0))
    principal = _principal_axis(canonical_average_coords)
    longaxis_dot_z = abs(float(principal[2]))
    group = str(point_group or "C").strip().upper()
    if group.startswith("D"):
        z_height = abs(float(centroid[2]))
        return {
            "cart_bounds": (0.0, 100.0),
            "max_longaxis_dot_z": None,
            "components_aligned_to_sym_axes": True,
            "limit_rotation_to_z": False,
            "fitted_radius": radius,
            "fitted_z_height": z_height,
            "fitted_longaxis_dot_z": longaxis_dot_z,
        }
    return {
        "cart_bounds": (max(0.0, radius - radius_pad), radius + radius_pad),
        # Reduced cyclic search keeps orientation about non-z axes fixed, so a
        # fitted long-axis/z limit would gate the prepared ASU rather than
        # meaningfully shape the sampled search space.
        "max_longaxis_dot_z": None,
        "components_aligned_to_sym_axes": True,
        "limit_rotation_to_z": True,
        "fitted_radius": radius,
        "fitted_longaxis_dot_z": longaxis_dot_z,
    }


def _center_coords_at_origin(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    return arr - np.mean(arr, axis=0, keepdims=True)


def _subset_bundle_coords_by_layout(
    *,
    full_residue_atom_names: tuple[tuple[str, ...], ...],
    full_coords: np.ndarray,
    subset_residue_atom_names: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    coords = np.asarray(full_coords, dtype=np.float64)
    if len(full_residue_atom_names) != len(subset_residue_atom_names):
        raise ValueError("Full and subset residue layouts must have the same residue count.")
    selected: list[np.ndarray] = []
    offset = 0
    for full_layout, subset_layout in zip(
        full_residue_atom_names,
        subset_residue_atom_names,
        strict=True,
    ):
        residue_size = len(full_layout)
        residue_coords = coords[offset : offset + residue_size]
        atom_index = {name: idx for idx, name in enumerate(full_layout)}
        for atom_name in subset_layout:
            idx = atom_index.get(atom_name)
            if idx is None:
                raise ValueError(f"Atom {atom_name!r} is missing from full residue layout.")
            selected.append(np.asarray(residue_coords[idx], dtype=np.float64))
        offset += residue_size
    return np.asarray(selected, dtype=np.float64)


def _apply_fit_targets_to_movable_bundles(
    *,
    fit_bundles: list[AtomBundle],
    movable_bundles: list[AtomBundle],
    target_fit_coords: list[np.ndarray],
    pin_fit_atoms: bool = False,
) -> None:
    if not (len(fit_bundles) == len(movable_bundles) == len(target_fit_coords)):
        raise ValueError("Fit bundles, movable bundles, and target coordinates must have the same length.")
    for fit_bundle, movable_bundle, target_coords in zip(
        fit_bundles,
        movable_bundles,
        target_fit_coords,
        strict=True,
    ):
        corr_R, corr_t = kabsch_transform(fit_bundle.coords, target_coords)
        transformed = apply_transform(movable_bundle.coords, corr_R, corr_t)
        if pin_fit_atoms:
            movable_atom_indices = {
                id(atom): idx for idx, atom in enumerate(movable_bundle.atoms)
            }
            for idx, atom in enumerate(fit_bundle.atoms):
                movable_idx = movable_atom_indices.get(id(atom))
                if movable_idx is None:
                    raise ValueError("Fit bundle atoms must be a subset of movable bundle atoms.")
                transformed[movable_idx] = np.asarray(target_coords[idx], dtype=np.float64)
        assign_atom_bundle_coordinates(movable_bundle, transformed)
        movable_bundle.coords = np.asarray(transformed, dtype=np.float64)


def _atom_element(atom) -> str:
    return str(getattr(atom, "element", "") or str(atom.id)[:1]).strip().upper()


def _dihedral_pair_class(left: int, right: int, order: int) -> str:
    order = int(order)
    if order < 2:
        return "all"
    a, b = sorted((int(left), int(right)))
    if a < order and b < order:
        return "cn_same_layer"
    if a >= order and b >= order:
        return "cn_same_layer"
    if (a % order) == (b % order):
        return "c2_same_index"
    return "cross_other"


def _empty_pair_metrics() -> dict[str, float]:
    return {
        "min_heavy_distance": float("inf"),
        "heavy_contact_count": 0.0,
    }


def _record_pair_metrics(
    metrics: dict[str, float],
    *,
    min_distance: float,
    contact_count: int,
) -> None:
    metrics["min_heavy_distance"] = min(
        float(metrics["min_heavy_distance"]),
        float(min_distance),
    )
    metrics["heavy_contact_count"] = float(
        int(metrics["heavy_contact_count"]) + int(contact_count)
    )


def _heavy_pair_distance_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    contact_cutoff: float,
) -> tuple[float, int]:
    left_tree = cKDTree(np.asarray(left, dtype=np.float64))
    right_arr = np.asarray(right, dtype=np.float64)
    distances, _ = left_tree.query(right_arr, k=1)
    min_distance = float(np.min(distances)) if np.size(distances) else float("inf")
    pair_count = left_tree.count_neighbors(
        cKDTree(right_arr),
        float(contact_cutoff),
        cumulative=True,
    )
    return min_distance, int(pair_count)


def _bundle_chain_metrics(
    bundles: list[AtomBundle],
    *,
    contact_cutoff: float = 5.0,
    dihedral_order: int | None = None,
) -> dict[str, Any]:
    if len(bundles) < 2:
        return _empty_pair_metrics()
    aggregate = _empty_pair_metrics()
    by_pair_class: dict[str, dict[str, float]] = {}
    if dihedral_order is not None:
        by_pair_class = {
            "cn_same_layer": _empty_pair_metrics(),
            "c2_same_index": _empty_pair_metrics(),
            "cross_other": _empty_pair_metrics(),
        }
    heavy_sets: list[np.ndarray] = []
    for bundle in bundles:
        coords = [
            np.asarray(atom.coord, dtype=np.float64)
            for atom in bundle.atoms
            if _atom_element(atom) != "H"
        ]
        heavy_sets.append(np.asarray(coords, dtype=np.float64))
    for idx, left in enumerate(heavy_sets):
        if left.size == 0:
            continue
        for right_idx in range(idx + 1, len(heavy_sets)):
            right = heavy_sets[right_idx]
            if right.size == 0:
                continue
            pair_min, pair_contacts = _heavy_pair_distance_metrics(
                left,
                right,
                contact_cutoff=contact_cutoff,
            )
            _record_pair_metrics(
                aggregate,
                min_distance=pair_min,
                contact_count=pair_contacts,
            )
            if dihedral_order is not None:
                pair_class = _dihedral_pair_class(idx, right_idx, int(dihedral_order))
                _record_pair_metrics(
                    by_pair_class[pair_class],
                    min_distance=pair_min,
                    contact_count=pair_contacts,
                )
    if by_pair_class:
        aggregate["by_pair_class"] = by_pair_class
    return aggregate


def _radially_scaled_coords(
    coords: np.ndarray,
    scale: float,
    *,
    z_scale: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    center = np.mean(arr, axis=0, keepdims=True)
    shifted = arr - center
    scaled_center = np.asarray(center, dtype=np.float64).copy()
    scaled_center[:, :2] *= float(scale)
    scaled_center[:, 2] *= float(z_scale)
    return shifted + scaled_center


def _compact_coords_preserving_dihedral_symmetry(
    coords_by_chain: list[np.ndarray],
    *,
    order: int,
    scale: float,
    z_scale: float,
) -> list[np.ndarray]:
    transforms = dihedral_group_transforms_canonical(int(order))
    if len(coords_by_chain) != len(transforms):
        raise ValueError(
            f"D{order} compaction needs {len(transforms)} coordinate sets; got {len(coords_by_chain)}."
        )
    reference = _radially_scaled_coords(
        coords_by_chain[0],
        float(scale),
        z_scale=float(z_scale),
    )
    return [
        apply_transform(reference, R, t)
        for R, t in transforms
    ]


def _compact_projected_bundles_if_dispersed(
    bundles: list[AtomBundle],
    *,
    dihedral_order: int | None = None,
    min_contacts: int = 1,
    contact_cutoff: float = 5.0,
    min_heavy_distance: float = 2.6,
    max_heavy_distance: float = 5.5,
    min_scale: float = 0.65,
    scale_step: float = 0.025,
) -> dict[str, Any]:
    before = _bundle_chain_metrics(
        bundles,
        contact_cutoff=contact_cutoff,
        dihedral_order=dihedral_order,
    )
    required_classes: tuple[str, ...] = ()
    if dihedral_order is not None:
        required_classes = ("cn_same_layer", "c2_same_index")

    def _metrics_are_acceptable(metrics: dict[str, Any]) -> bool:
        if not (
            float(metrics["min_heavy_distance"]) >= float(min_heavy_distance)
            and float(metrics["min_heavy_distance"]) <= float(max_heavy_distance)
            and int(metrics["heavy_contact_count"]) >= int(min_contacts)
        ):
            return False
        by_pair_class = metrics.get("by_pair_class", {})
        for pair_class in required_classes:
            class_metrics = by_pair_class.get(pair_class, {})
            if int(class_metrics.get("heavy_contact_count", 0.0)) < int(min_contacts):
                return False
            if float(class_metrics.get("min_heavy_distance", float("inf"))) > float(max_heavy_distance):
                return False
        return True

    if (
        _metrics_are_acceptable(before)
    ):
        return {
            "applied": False,
            "failed": False,
            "scale": 1.0,
            "z_scale": 1.0,
            "before": before,
            "after": before,
        }

    original_coords = [np.asarray(bundle.coords, dtype=np.float64).copy() for bundle in bundles]
    best_scale = 1.0
    best_z_scale = 1.0
    best_coords = original_coords
    best_metrics = before
    accepted = False
    scales = np.arange(
        1.0 - float(scale_step),
        float(min_scale) - 0.5 * float(scale_step),
        -float(scale_step),
        dtype=np.float64,
    )
    z_scales = np.asarray([1.0], dtype=np.float64)
    if dihedral_order is not None:
        z_scales = np.arange(
            1.0,
            float(min_scale) - 0.5 * float(scale_step),
            -float(scale_step),
            dtype=np.float64,
        )
    stop = False
    for scale in scales:
        for z_scale in z_scales:
            if dihedral_order is None:
                candidate_coords = [
                    _radially_scaled_coords(coords, float(scale), z_scale=float(z_scale))
                    for coords in original_coords
                ]
            else:
                candidate_coords = _compact_coords_preserving_dihedral_symmetry(
                    original_coords,
                    order=int(dihedral_order),
                    scale=float(scale),
                    z_scale=float(z_scale),
                )
            for bundle, coords in zip(bundles, candidate_coords, strict=True):
                assign_atom_bundle_coordinates(bundle, coords)
                bundle.coords = np.asarray(coords, dtype=np.float64)
            metrics = _bundle_chain_metrics(
                bundles,
                contact_cutoff=contact_cutoff,
                dihedral_order=dihedral_order,
            )
            if float(metrics["min_heavy_distance"]) >= float(min_heavy_distance):
                best_scale = float(scale)
                best_z_scale = float(z_scale)
                best_coords = candidate_coords
                best_metrics = metrics
            if _metrics_are_acceptable(metrics):
                accepted = True
                stop = True
                break
        if stop:
            break
    if not accepted:
        best_scale = 1.0
        best_z_scale = 1.0
        best_coords = original_coords
        best_metrics = before

    for bundle, coords in zip(bundles, best_coords, strict=True):
        assign_atom_bundle_coordinates(bundle, coords)
        bundle.coords = np.asarray(coords, dtype=np.float64)
    return {
        "applied": accepted,
        "failed": not accepted,
        "scale": best_scale,
        "z_scale": best_z_scale,
        "before": before,
        "after": best_metrics,
    }


_DIHEDRAL_RPXDOCK_CONTACT_TARGET = 12


def _dihedral_pair_contact_summary(
    compaction: dict[str, Any],
    *,
    target_contacts: int = _DIHEDRAL_RPXDOCK_CONTACT_TARGET,
) -> dict[str, Any]:
    after = compaction.get("after", {})
    by_pair_class = after.get("by_pair_class", {})
    c2 = by_pair_class.get("c2_same_index", {})
    cn = by_pair_class.get("cn_same_layer", {})
    cross = by_pair_class.get("cross_other", {})
    c2_contacts = int(c2.get("heavy_contact_count", 0.0))
    cn_contacts = int(cn.get("heavy_contact_count", 0.0))
    cross_contacts = int(cross.get("heavy_contact_count", 0.0))
    target = max(1, int(target_contacts))
    return {
        "c2_same_index_contacts": c2_contacts,
        "cn_same_layer_contacts": cn_contacts,
        "cross_other_contacts": cross_contacts,
        "c2_same_index_min_heavy_distance": float(c2.get("min_heavy_distance", float("inf"))),
        "cn_same_layer_min_heavy_distance": float(cn.get("min_heavy_distance", float("inf"))),
        "min_heavy_distance": float(after.get("min_heavy_distance", float("inf"))),
        "heavy_contact_count": int(after.get("heavy_contact_count", 0.0)),
        "applied": bool(compaction.get("applied", False)),
        "failed": bool(compaction.get("failed", False)),
        "contact_target": target,
        "c2_target_fraction": min(float(c2_contacts) / float(target), 1.0),
        "cn_target_fraction": min(float(cn_contacts) / float(target), 1.0),
    }


def _dihedral_pair_contact_selection_score(
    summary: dict[str, Any],
    *,
    rank: int,
    candidate_index: int,
) -> tuple[float, ...]:
    c2_contacts = int(summary.get("c2_same_index_contacts", 0))
    cn_contacts = int(summary.get("cn_same_layer_contacts", 0))
    target = max(1, int(summary.get("contact_target", _DIHEDRAL_RPXDOCK_CONTACT_TARGET)))
    failed = bool(summary.get("failed", False))
    return (
        1.0 if c2_contacts > 0 else 0.0,
        float(min(c2_contacts, target)),
        float(c2_contacts),
        1.0 if not failed else 0.0,
        float(min(cn_contacts, target)),
        float(cn_contacts),
        float(summary.get("heavy_contact_count", 0)),
        -float(rank),
        -float(candidate_index),
    )


_DIHEDRAL_RPXDOCK_PRIMARY_SELECTION_CRITERION = (
    "prefer candidates with postprocess c2_same_index contacts; rank by "
    "capped c2 contacts first, then raw c2 contacts, nonfailed status, "
    "cn_same_layer contacts, total contacts, and rpxdock rank"
)


def _protein_chain_ids(structure) -> list[str]:
    model = next(structure.get_models())
    out = []
    for chain in model:
        residues = [res for res in chain if "CA" in res]
        if residues:
            out.append(str(chain.id))
    if not out:
        raise ValueError("Structure contains no protein chains with CA atoms.")
    return out


def _distribute_rpxdock_first_pose(
    *,
    monomer_path: str | Path,
    group_transforms: list[tuple[np.ndarray, np.ndarray]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[np.ndarray]:
    monomer_structure = load_structure(monomer_path)
    first_chain_id = _protein_chain_ids(monomer_structure)[0]
    residues = protein_residues(monomer_structure, first_chain_id)
    bundle = collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
    first_coords = apply_transform(bundle.coords, rotation, translation)
    return [
        apply_transform(first_coords, Rk, tk)
        for Rk, tk in group_transforms
    ]


def _aligned_average_in_reference_frame(
    canonical_coord_sets: list[np.ndarray],
    inverse_group_rotations: list[np.ndarray],
) -> np.ndarray:
    if not canonical_coord_sets:
        raise ValueError("Need at least one coordinate set to average.")
    degrouped = [
        apply_transform(coords, Rinv, np.zeros(3, dtype=np.float64))
        for coords, Rinv in zip(canonical_coord_sets, inverse_group_rotations, strict=True)
    ]
    consensus = np.asarray(degrouped[0], dtype=np.float64)
    for _ in range(2):
        aligned = []
        for coords in degrouped:
            R, t = kabsch_transform(coords, consensus)
            aligned.append(apply_transform(coords, R, t))
        consensus = np.mean(np.stack(aligned, axis=0), axis=0)
    return np.asarray(consensus, dtype=np.float64)


def _project_cn_bundle_layout(
    coord_sets: list[np.ndarray],
    bundle_coords: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any], Any]:
    canonicalization = canonicalize_exact_cyclic_cn(coord_sets)
    canonical_bundles = [
        apply_transform(
            coords,
            canonicalization.to_canonical_rotation,
            canonicalization.to_canonical_translation,
        )
        for coords in bundle_coords
    ]
    group_transforms = cyclic_group_transforms_canonical(canonicalization.order)
    inverse_rotations = [Rk.T for Rk, _ in group_transforms]
    average_bundle = _aligned_average_in_reference_frame(canonical_bundles, inverse_rotations)
    desired = [
        apply_transform(average_bundle, Rk, tk) for Rk, tk in group_transforms
    ]
    return average_bundle, desired, {
        "order": canonicalization.order,
        "target_rotation_deg": 360.0 / float(canonicalization.order),
        "axis_point": [float(x) for x in canonicalization.axis_point.tolist()],
        "axis_dir": [float(x) for x in canonicalization.axis_direction.tolist()],
    }, canonicalization


def _project_dn_bundle_layout(
    coord_sets: list[np.ndarray],
    bundle_coords: list[np.ndarray],
    *,
    order: int,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any], Any]:
    order = int(order)
    if order < 2:
        raise ValueError(f"Dn order must be >= 2, got {order}.")
    if len(coord_sets) != 2 * order:
        raise ValueError(
            f"Dn projection requires 2*order coordinate sets; got {len(coord_sets)} for D{order}."
        )
    canonicalization = canonicalize_exact_cyclic_cn(coord_sets[:order], order=order)
    canonical_bundles = [
        apply_transform(
            coords,
            canonicalization.to_canonical_rotation,
            canonicalization.to_canonical_translation,
        )
        for coords in bundle_coords
    ]
    group_transforms = dihedral_group_transforms_canonical(order)
    inverse_rotations = [Rk.T for Rk, _ in group_transforms]
    average_bundle = _aligned_average_in_reference_frame(canonical_bundles, inverse_rotations)
    desired = [
        apply_transform(average_bundle, Rk, tk) for Rk, tk in group_transforms
    ]
    return average_bundle, desired, {
        "point_group": f"D{order}",
        "order": order,
        "subunit_count": 2 * order,
        "target_rotation_deg": 360.0 / float(order),
        "axis_point": [float(x) for x in canonicalization.axis_point.tolist()],
        "axis_dir": [float(x) for x in canonicalization.axis_direction.tolist()],
    }, canonicalization


@register_structure_processor("symmetry_canonicalizer")
class SymmetryRescueProcessor(StructureProcessor):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.mode = str(kwargs.get("mode", "")).strip()
        self.oligo = str(kwargs.get("oligo", "")).strip()
        self._allows_runtime_oligo = self.mode in {"cyclic_oligomer", "dihedral_oligomer"} and not self.oligo
        self.included_protomer = kwargs.get("included_protomer", None)
        raw_backend = kwargs.get("backend", None)
        if raw_backend is None and self.mode in {"cyclic_oligomer", "dihedral_oligomer"}:
            self.backend = "rfdiffusion"
        else:
            self.backend = str(raw_backend if raw_backend is not None else "legacy").strip().lower()
        self.order = kwargs.get("order", None)
        self.reference_chain_index = int(kwargs.get("reference_chain_index", 1))
        self.rpxdock_python_bin = kwargs.get("rpxdock_python_bin", None)
        self.rpxdock_vendor_repo = kwargs.get("rpxdock_vendor_repo", None)
        self.rpxdock_willutil_repo = kwargs.get("rpxdock_willutil_repo", None)
        self.rpxdock_timeout_s = int(kwargs.get("rpxdock_timeout_s", 900))
        self.rpxdock_failure_fallback = str(
            kwargs.get("rpxdock_failure_fallback", "rfdiffusion")
        ).strip().lower()
        raw_rpxdock_extra_args = kwargs.get("rpxdock_extra_args", ())
        if isinstance(raw_rpxdock_extra_args, str):
            self.rpxdock_extra_args = tuple(shlex.split(raw_rpxdock_extra_args))
        else:
            self.rpxdock_extra_args = tuple(str(x) for x in (raw_rpxdock_extra_args or ()))
        self.repeat_canonicalizer_mode = str(
            kwargs.get("repeat_canonicalizer_mode", "helical")
        ).strip()
        raw_interface_selection = kwargs.get("interface_selection", None)
        self.interface_selection = (
            str(raw_interface_selection).strip().lower()
            if raw_interface_selection is not None
            else None
        )
        raw_interface = kwargs.get("interface", None)
        self.interface = (
            str(raw_interface).strip() if raw_interface is not None else None
        )
        if self.mode not in {
            "pseudocycle",
            "tandem_repeat",
            "heterooligomer_shared_core",
            "cyclic_oligomer",
            "dihedral_oligomer",
        }:
            raise ValueError(
                f"Unsupported symmetry canonicalizer mode {self.mode!r}."
            )
        if not self.oligo and not self._allows_runtime_oligo:
            raise ValueError("symmetry_canonicalizer requires a non-empty 'oligo'.")
        if self.backend not in {"legacy", "rfdiffusion", "rpxdock"}:
            raise ValueError("backend must be one of: legacy, rfdiffusion, rpxdock.")
        if self.rpxdock_failure_fallback not in {"rfdiffusion", "input", "raise"}:
            raise ValueError("rpxdock_failure_fallback must be one of: rfdiffusion, input, raise.")
        if self.repeat_canonicalizer_mode not in {"helical", "translation"}:
            raise ValueError(
                "repeat_canonicalizer_mode must be one of: helical, translation."
            )
        if self.interface_selection not in {None, "auto", "manual"}:
            raise ValueError("interface_selection must be one of: auto, manual.")
        if self.interface_selection == "manual" and not self.interface:
            raise ValueError("interface_selection='manual' requires interface.")
        if self.order is not None and self.backend == "legacy":
            raise ValueError("Symmetry order is not supported for backend='legacy'.")
        if self.mode == "dihedral_oligomer" and self.backend == "legacy":
            raise ValueError("backend='legacy' is not supported for mode='dihedral_oligomer'.")
        if (
            self.mode == "pseudocycle"
            and self.backend in {"rfdiffusion", "rpxdock"}
            and self.repeat_canonicalizer_mode != "helical"
        ):
            raise ValueError(
                "repeat_canonicalizer_mode='translation' is not supported for exact pseudocycle backends."
            )

    def runtime_clone(self, **overrides: Any):
        clone = copy.copy(self)
        runtime_oligo = overrides.get("oligo", None)
        if runtime_oligo is not None:
            clone.oligo = str(runtime_oligo).strip()
            clone._allows_runtime_oligo = False
        return clone

    def _generate_cache_key(self, input_file) -> str:
        payload = {
            "input_file": str(input_file),
            "class_name": self.__class__.__name__,
            "mode": self.mode,
            "backend": self.backend,
            "oligo": self.oligo,
            "order": self.order,
            "reference_chain_index": self.reference_chain_index,
            "included_protomer": self.included_protomer,
            "repeat_canonicalizer_mode": self.repeat_canonicalizer_mode,
            "interface_selection": self.interface_selection,
            "interface": self.interface,
            "rpxdock_python_bin": None if self.rpxdock_python_bin is None else str(self.rpxdock_python_bin),
            "rpxdock_vendor_repo": None if self.rpxdock_vendor_repo is None else str(self.rpxdock_vendor_repo),
            "rpxdock_willutil_repo": None if self.rpxdock_willutil_repo is None else str(self.rpxdock_willutil_repo),
            "rpxdock_timeout_s": self.rpxdock_timeout_s,
            "rpxdock_extra_args": self.rpxdock_extra_args,
            "rpxdock_failure_fallback": self.rpxdock_failure_fallback,
        }
        return hashlib.md5(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _internal_repeat_fit_groups(
        self,
        protomer_name: str,
        *,
        residue_count: int,
    ) -> list[np.ndarray]:
        groups = build_internal_repeat_residue_groups(protomer_name)
        if self.interface_selection != "manual":
            return groups

        selected = _parse_manual_interface_residue_indices(
            self.interface,
            residue_count=int(residue_count),
        )
        filtered: list[np.ndarray] = []
        for group in groups:
            group_selected = [int(idx) for idx in group if int(idx) in selected]
            if not group_selected:
                raise ValueError(
                    "manual interface selection must include at least one residue "
                    "from every repeat copy."
                )
            filtered.append(np.asarray(group_selected, dtype=int))
        return filtered

    def _write_output(self, structure, input_file: str | Path) -> str:
        input_path = Path(input_file)
        name = input_path.name.lower()
        if any(ext in name for ext in [".cif", ".mmcif"]):
            suffix = ".cif"
        else:
            suffix = ".pdb"
        fd, output_path = tempfile.mkstemp(prefix="symmetry_", suffix=suffix)
        os.close(fd)
        save_structure(structure, output_path, like_input=input_path)
        return output_path

    def _prepare_cached_result(self, cache_key, input_file, result):
        result_path = Path(str(result)).resolve()
        case_root = (self.debug_dir / "symmetry_cases").resolve()
        if result_path.is_relative_to(case_root):
            return str(result_path)
        return super()._prepare_cached_result(cache_key, input_file, result)

    def _after_cache_store(self, cache_key, input_file, result):
        manifest_path = Path(str(result)).resolve().with_name("03_manifest.json")
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates = payload.get("output_candidates")
            if isinstance(candidates, list):
                resolved = [
                    str(Path(candidate).expanduser().resolve())
                    for candidate in candidates
                    if Path(candidate).expanduser().exists()
                ]
                self._set_cache_candidates(cache_key, resolved)
        super()._after_cache_store(cache_key, input_file, result)

    def _run_impl(self, input_file) -> Any:
        if self.backend == "legacy" and self.mode in {"cyclic_oligomer", "dihedral_oligomer"}:
            raise ValueError(f"backend='legacy' is not supported for mode={self.mode!r}.")
        if self.backend != "legacy" and self.mode in {"tandem_repeat", "heterooligomer_shared_core"}:
            raise ValueError(
                f"backend={self.backend!r} is not supported for mode={self.mode!r}."
            )
        if self.backend == "rfdiffusion":
            return self._run_exact_projection(input_file)
        if self.backend == "rpxdock":
            return self._run_rpxdock_projection_with_fallback(input_file)
        if self.mode in {"pseudocycle", "tandem_repeat"}:
            return self._run_internal_repeat(input_file)
        return self._run_shared_core(input_file)

    def _annotate_manifest(self, output_path: str | Path, payload: dict[str, Any]) -> None:
        manifest_path = Path(str(output_path)).resolve().with_name("03_manifest.json")
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(payload)
        _write_manifest(manifest_path, manifest)

    def _run_rpxdock_projection_with_fallback(self, input_file: str | Path) -> str:
        from astrolabe.structure_processors.wrappers.rpxdock import (
            RpxDockDependencyError,
            RpxDockExecutionError,
            RpxDockOutputError,
        )

        try:
            return self._run_rpxdock_projection(input_file)
        except (RpxDockDependencyError, RpxDockExecutionError, RpxDockOutputError) as exc:
            if self.rpxdock_failure_fallback == "raise":
                raise
            reason = f"{exc.__class__.__name__}: {exc}"
            logging.warning(
                "RPXDock symmetry predocker failed; applying %s fallback. %s",
                self.rpxdock_failure_fallback,
                reason,
            )
            if self.rpxdock_failure_fallback == "rfdiffusion":
                fallback = copy.copy(self)
                fallback.backend = "rfdiffusion"
                output_path = fallback._run_exact_projection(input_file)
                fallback._annotate_manifest(
                    output_path,
                    {
                        "fallback_from_backend": "rpxdock",
                        "fallback_policy": "rfdiffusion",
                        "fallback_reason": reason,
                    },
                )
                return output_path
            return self._run_input_fallback(input_file, reason=reason)

    def _run_input_fallback(self, input_file: str | Path, *, reason: str) -> str:
        structure = load_structure(input_file)
        return self._write_case_outputs(
            input_file=input_file,
            initial_structure=copy.deepcopy(structure),
            projected_structure=copy.deepcopy(structure),
            average_structure=copy.deepcopy(structure),
            manifest={
                "backend": "input",
                "mode": self.mode,
                "oligo": self.oligo,
                "reference_chain_index": self.reference_chain_index,
                "included_protomer": self.included_protomer,
                "interface_selection": self.interface_selection,
                "interface": self.interface,
                "fallback_from_backend": "rpxdock",
                "fallback_policy": "input",
                "fallback_reason": reason,
            },
        )

    def _case_dir(self, input_file: str | Path) -> Path:
        digest = hashlib.md5(
            json.dumps(
                {
                    "input_file": str(input_file),
                    "backend": self.backend,
                    "mode": self.mode,
                    "oligo": self.oligo,
                    "order": self.order,
                    "reference_chain_index": self.reference_chain_index,
                    "included_protomer": self.included_protomer,
                    "interface_selection": self.interface_selection,
                    "interface": self.interface,
                    "rpxdock_python_bin": None if self.rpxdock_python_bin is None else str(self.rpxdock_python_bin),
                    "rpxdock_vendor_repo": None if self.rpxdock_vendor_repo is None else str(self.rpxdock_vendor_repo),
                    "rpxdock_willutil_repo": None if self.rpxdock_willutil_repo is None else str(self.rpxdock_willutil_repo),
                    "rpxdock_timeout_s": self.rpxdock_timeout_s,
                    "rpxdock_extra_args": self.rpxdock_extra_args,
                    "rpxdock_failure_fallback": self.rpxdock_failure_fallback,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        case_name = f"{Path(input_file).stem}_{self.backend}_{self.mode}_{digest}"
        case_dir = self.debug_dir / "symmetry_cases" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _resolve_order(self, observed_count: int) -> int:
        if self.order is None:
            return int(observed_count)
        if int(self.order) != int(observed_count):
            raise ValueError(
                f"Explicit order={self.order} does not match observed unit count {observed_count}."
            )
        return int(self.order)

    def _resolve_dihedral_order(self, observed_count: int) -> int:
        if int(observed_count) < 4 or int(observed_count) % 2:
            raise ValueError(
                f"Dihedral oligomer canonicalization requires an even chain count >= 4, got {observed_count}."
            )
        inferred = int(observed_count) // 2
        if self.order is None:
            return inferred
        if int(self.order) != inferred:
            raise ValueError(
                f"Explicit order={self.order} does not match Dn principal order {inferred} from observed unit count {observed_count}."
            )
        return int(self.order)

    def _resolve_reference_occurrence(self, occurrences):
        if not 1 <= self.reference_chain_index <= len(occurrences):
            raise ValueError(
                f"reference_chain_index must be in [1, {len(occurrences)}], got {self.reference_chain_index}."
            )
        return occurrences[self.reference_chain_index - 1]

    def _reorder_cyclic_inputs_by_reference(self, occurrences, *parallel_lists):
        ref_index = self.reference_chain_index - 1
        if ref_index == 0:
            return (list(occurrences), *[list(items) for items in parallel_lists])
        self._resolve_reference_occurrence(occurrences)
        order = list(range(ref_index, len(occurrences))) + list(range(ref_index))
        reordered_occurrences = [occurrences[idx] for idx in order]
        reordered_parallel = [
            [items[idx] for idx in order] for items in parallel_lists
        ]
        return (reordered_occurrences, *reordered_parallel)

    def _reorder_dihedral_inputs_by_reference(self, occurrences, *parallel_lists):
        order = self._resolve_dihedral_order(len(occurrences))
        ref_index = self.reference_chain_index - 1
        if not 0 <= ref_index < order:
            raise ValueError(
                f"reference_chain_index for dihedral_oligomer must select a chain in the first Dn ring [1, {order}], got {self.reference_chain_index}."
            )
        if ref_index == 0:
            return (list(occurrences), *[list(items) for items in parallel_lists])
        reordered_indices = (
            list(range(ref_index, order))
            + list(range(0, ref_index))
            + list(range(order + ref_index, 2 * order))
            + list(range(order, order + ref_index))
        )
        reordered_occurrences = [occurrences[idx] for idx in reordered_indices]
        reordered_parallel = [
            [items[idx] for idx in reordered_indices] for items in parallel_lists
        ]
        return (reordered_occurrences, *reordered_parallel)

    def _write_case_outputs(
        self,
        *,
        input_file: str | Path,
        initial_structure,
        projected_structure,
        average_structure,
        manifest: dict[str, Any],
        projected_candidates: list[tuple[int, Any]] | None = None,
    ) -> str:
        case_dir = self._case_dir(input_file)
        suffix = _case_suffix(input_file)
        initial_path = case_dir / f"00_initial_structure{suffix}"
        average_path = case_dir / f"01_average_asu_structure{suffix}"
        output_path = case_dir / f"02_redock_result{suffix}"
        _write_case_structure(initial_structure, initial_path, like_input=input_file)
        _write_case_structure(average_structure, average_path, like_input=input_file)
        _write_case_structure(projected_structure, output_path, like_input=input_file)
        output_candidates = [str(output_path)]
        for rank, candidate_structure in projected_candidates or []:
            candidate_path = case_dir / f"02_redock_result__top{int(rank)}{suffix}"
            _write_case_structure(candidate_structure, candidate_path, like_input=input_file)
            output_candidates.append(str(candidate_path))
        payload = dict(manifest)
        payload.update(
            {
                "input_file": str(input_file),
                "case_dir": str(case_dir),
                "initial_structure": str(initial_path),
                "average_structure": str(average_path),
                "output_structure": str(output_path),
                "output_candidates": output_candidates,
            }
        )
        _write_manifest(case_dir / "03_manifest.json", payload)
        return str(output_path)

    def _build_average_structure(
        self,
        *,
        source_structure,
        chain_id: str,
        residue_indices: np.ndarray,
        residue_atom_names: tuple[tuple[str, ...], ...],
        average_coords: np.ndarray,
    ):
        average_structure = _subset_structure_with_atom_layout(
            source_structure,
            chain_id=chain_id,
            residue_indices=residue_indices,
            residue_atom_names=residue_atom_names,
        )
        residues = protein_residues(average_structure, chain_id)
        bundle = collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
        assign_atom_bundle_coordinates(bundle, average_coords)
        return average_structure

    def _run_exact_projection(self, input_file: str | Path) -> str:
        if self.mode == "pseudocycle":
            return self._run_pseudocycle_exact(input_file)
        if self.mode == "cyclic_oligomer":
            return self._run_cyclic_oligomer_exact(input_file)
        if self.mode == "dihedral_oligomer":
            return self._run_dihedral_oligomer_exact(input_file)
        raise ValueError(f"Unsupported exact projection mode {self.mode!r}.")

    def _run_rpxdock_projection(self, input_file: str | Path) -> str:
        from astrolabe.structure_processors.wrappers.rpxdock import (
            RpxDockRunConfig,
            run_local_cyclic_dock,
        )
        if self.mode == "pseudocycle":
            structure = load_structure(input_file)
            chain_map = unique_protomer_chain_map(self.oligo)
            proto = (
                str(self.included_protomer)
                if self.included_protomer is not None
                else next(iter(chain_map.keys()))
            )
            chain_id = chain_map[proto]
            residues = protein_residues(structure, chain_id)
            movable_groups = build_internal_repeat_movable_groups(proto)
            groups = self._internal_repeat_fit_groups(
                proto,
                residue_count=len(residues),
            )
            fit_bundles = harmonize_atom_bundles(
                [collect_atom_bundle(residues, group) for group in groups]
            )
            movable_bundles = [collect_atom_bundle(residues, group) for group in movable_groups]
            movable_common_bundles = harmonize_atom_bundles(movable_bundles)
            validate_compatible_atom_bundles(fit_bundles)
            order = self._resolve_order(len(fit_bundles))
            ca_sets = [residue_ca_coordinates(residues, group) for group in groups]
            average_fit_coords, _, meta, _ = _project_cn_bundle_layout(
                ca_sets,
                [bundle.coords for bundle in fit_bundles],
            )
            average_ca_coords, _, _, _ = _project_cn_bundle_layout(
                ca_sets,
                ca_sets,
            )
            average_movable_coords, _, _, _ = _project_cn_bundle_layout(
                ca_sets,
                [bundle.coords for bundle in movable_common_bundles],
            )
            average_structure = self._build_average_structure(
                source_structure=structure,
                chain_id=chain_id,
                residue_indices=movable_groups[0],
                residue_atom_names=movable_common_bundles[0].residue_atom_names,
                average_coords=average_movable_coords,
            )
            case_dir = self._case_dir(input_file)
            average_path = case_dir / f"01_average_asu_structure{_case_suffix(input_file)}"
            monomer_path = case_dir / "01_dock_input_structure.pdb"
            _write_case_structure(average_structure, average_path, like_input=input_file)
            dock_input_structure = _build_rigid_template_structure(
                source_structure=structure,
                chain_id=chain_id,
                residue_indices=groups[0],
                target_ca_coords=_center_coords_at_origin(average_ca_coords),
            )
            _write_case_structure(dock_input_structure, monomer_path, like_input=monomer_path)
            dock_fit = _fit_rpxdock_geometry_kwargs(average_ca_coords)
            dock_result = run_local_cyclic_dock(
                monomer_path,
                RpxDockRunConfig(
                    nfold=order,
                    output_dir=case_dir / "rpxdock_work",
                    python_bin=str(self.rpxdock_python_bin or "python"),
                    vendor_repo=(
                        None if self.rpxdock_vendor_repo in (None, "") else Path(str(self.rpxdock_vendor_repo))
                    ),
                    willutil_repo=(
                        None if self.rpxdock_willutil_repo in (None, "") else Path(str(self.rpxdock_willutil_repo))
                    ),
                    timeout_s=self.rpxdock_timeout_s,
                    extra_args=self.rpxdock_extra_args,
                    cart_bounds=dock_fit["cart_bounds"],
                    max_longaxis_dot_z=dock_fit["max_longaxis_dot_z"],
                    components_aligned_to_sym_axes=bool(dock_fit["components_aligned_to_sym_axes"]),
                    limit_rotation_to_z=bool(dock_fit["limit_rotation_to_z"]),
                ),
            )
            dock_candidates = list(dock_result.candidates) or [
                type("_PrimaryDockCandidate", (), {
                    "rank": 0,
                    "rotation": dock_result.rotation,
                    "translation": dock_result.translation,
                    "docked_structure_path": dock_result.docked_structure_path,
                })()
            ]
            projected_primary = None
            projected_candidates: list[tuple[int, Any]] = []
            for candidate_index, candidate in enumerate(dock_candidates):
                target_full_coords = _distribute_rpxdock_first_pose(
                    monomer_path=monomer_path,
                    group_transforms=cyclic_group_transforms_canonical(order),
                    rotation=candidate.rotation,
                    translation=candidate.translation,
                )
                projected = copy.deepcopy(structure)
                projected_residues = protein_residues(projected, chain_id)
                projected_fit = harmonize_atom_bundles(
                    [collect_atom_bundle(projected_residues, group) for group in groups]
                )
                projected_movable = [
                    collect_atom_bundle(projected_residues, group) for group in movable_groups
                ]
                _apply_fit_targets_to_movable_bundles(
                    fit_bundles=projected_fit,
                    movable_bundles=projected_movable,
                    target_fit_coords=target_full_coords,
                )
                if candidate_index == 0:
                    projected_primary = projected
                else:
                    projected_candidates.append((candidate.rank, projected))
            if projected_primary is None:
                raise ValueError("rpxdock did not yield any candidates for pseudocycle projection.")
            return self._write_case_outputs(
                input_file=input_file,
                initial_structure=copy.deepcopy(structure),
                projected_structure=projected_primary,
                average_structure=average_structure,
                projected_candidates=projected_candidates,
                manifest={
                    "backend": self.backend,
                    "mode": self.mode,
                    "order": order,
                    "asu_kind": "repeat_unit",
                    "oligo": self.oligo,
                    "included_protomer": proto,
                    "interface_selection": self.interface_selection,
                    "interface": self.interface,
                    "rpxdock_output": str(dock_result.docked_structure_path),
                    "rpxdock_output_candidates": [
                        str(candidate.docked_structure_path) for candidate in dock_candidates
                    ],
                    "rpxdock_command": list(dock_result.command),
                    "rpxdock_stdout": str(dock_result.stdout_path),
                    "rpxdock_stderr": str(dock_result.stderr_path),
                    "rpxdock_input": str(monomer_path),
                    "rpxdock_fit": dock_fit,
                    **meta,
                },
            )
        if self.mode == "cyclic_oligomer":
            structure = load_structure(input_file)
            occurrences, ca_sets, fit_bundles, _ = cyclic_oligomer_atom_bundles(
                structure,
                self.oligo,
            )
            occurrences, ca_sets, fit_bundles = self._reorder_cyclic_inputs_by_reference(
                occurrences,
                ca_sets,
                fit_bundles,
            )
            order = self._resolve_order(len(occurrences))
            average_fit_coords, _, meta, _ = _project_cn_bundle_layout(
                ca_sets,
                [bundle.coords for bundle in fit_bundles],
            )
            average_ca_coords, _, _, _ = _project_cn_bundle_layout(
                ca_sets,
                ca_sets,
            )
            ref_occurrence = occurrences[0]
            ref_residues = protein_residues(structure, ref_occurrence.chain_id)
            average_structure = self._build_average_structure(
                source_structure=structure,
                chain_id=ref_occurrence.chain_id,
                residue_indices=np.arange(len(ref_residues), dtype=int),
                residue_atom_names=fit_bundles[0].residue_atom_names,
                average_coords=average_fit_coords,
            )
            case_dir = self._case_dir(input_file)
            average_path = case_dir / f"01_average_asu_structure{_case_suffix(input_file)}"
            monomer_path = case_dir / "01_dock_input_structure.pdb"
            _write_case_structure(average_structure, average_path, like_input=input_file)
            dock_input_structure = _build_rigid_template_structure(
                source_structure=structure,
                chain_id=ref_occurrence.chain_id,
                residue_indices=np.arange(len(ref_residues), dtype=int),
                target_ca_coords=_center_coords_at_origin(average_ca_coords),
            )
            _write_case_structure(dock_input_structure, monomer_path, like_input=monomer_path)
            dock_fit = _fit_rpxdock_geometry_kwargs(average_ca_coords)
            dock_result = run_local_cyclic_dock(
                monomer_path,
                RpxDockRunConfig(
                    nfold=order,
                    output_dir=case_dir / "rpxdock_work",
                    python_bin=str(self.rpxdock_python_bin or "python"),
                    vendor_repo=(
                        None if self.rpxdock_vendor_repo in (None, "") else Path(str(self.rpxdock_vendor_repo))
                    ),
                    willutil_repo=(
                        None if self.rpxdock_willutil_repo in (None, "") else Path(str(self.rpxdock_willutil_repo))
                    ),
                    timeout_s=self.rpxdock_timeout_s,
                    extra_args=self.rpxdock_extra_args,
                    cart_bounds=dock_fit["cart_bounds"],
                    max_longaxis_dot_z=dock_fit["max_longaxis_dot_z"],
                    components_aligned_to_sym_axes=bool(dock_fit["components_aligned_to_sym_axes"]),
                    limit_rotation_to_z=bool(dock_fit["limit_rotation_to_z"]),
                ),
            )
            dock_candidates = list(dock_result.candidates) or [
                type("_PrimaryDockCandidate", (), {
                    "rank": 0,
                    "rotation": dock_result.rotation,
                    "translation": dock_result.translation,
                    "docked_structure_path": dock_result.docked_structure_path,
                })()
            ]
            projected_primary = None
            projected_candidates: list[tuple[int, Any]] = []
            for candidate_index, candidate in enumerate(dock_candidates):
                target_full_coords = _distribute_rpxdock_first_pose(
                    monomer_path=monomer_path,
                    group_transforms=cyclic_group_transforms_canonical(order),
                    rotation=candidate.rotation,
                    translation=candidate.translation,
                )
                projected = copy.deepcopy(structure)
                projected_bundles: list[AtomBundle] = []
                for occurrence in occurrences:
                    residues = protein_residues(projected, occurrence.chain_id)
                    projected_bundles.append(
                        collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
                    )
                projected_fit = harmonize_atom_bundles(projected_bundles)
                target_fit_coords = [
                    _subset_bundle_coords_by_layout(
                        full_residue_atom_names=projected_bundles[0].residue_atom_names,
                        full_coords=coords,
                        subset_residue_atom_names=projected_fit[0].residue_atom_names,
                    )
                    for coords in target_full_coords
                ]
                _apply_fit_targets_to_movable_bundles(
                    fit_bundles=projected_fit,
                    movable_bundles=projected_bundles,
                    target_fit_coords=target_fit_coords,
                    pin_fit_atoms=True,
                )
                if candidate_index == 0:
                    projected_primary = projected
                else:
                    projected_candidates.append((candidate.rank, projected))
            if projected_primary is None:
                raise ValueError("rpxdock did not yield any candidates for cyclic oligomer projection.")
            return self._write_case_outputs(
                input_file=input_file,
                initial_structure=copy.deepcopy(structure),
                projected_structure=projected_primary,
                average_structure=average_structure,
                projected_candidates=projected_candidates,
                manifest={
                    "backend": self.backend,
                    "mode": self.mode,
                    "order": order,
                    "asu_kind": "protomer",
                    "oligo": self.oligo,
                    "reference_chain_index": self.reference_chain_index,
                    "rpxdock_output": str(dock_result.docked_structure_path),
                    "rpxdock_output_candidates": [
                        str(candidate.docked_structure_path) for candidate in dock_candidates
                    ],
                    "rpxdock_command": list(dock_result.command),
                    "rpxdock_stdout": str(dock_result.stdout_path),
                    "rpxdock_stderr": str(dock_result.stderr_path),
                    "rpxdock_input": str(monomer_path),
                    "rpxdock_fit": dock_fit,
                    **meta,
                },
            )
        if self.mode == "dihedral_oligomer":
            structure = load_structure(input_file)
            occurrences, ca_sets, fit_bundles, _ = cyclic_oligomer_atom_bundles(
                structure,
                self.oligo,
            )
            occurrences, ca_sets, fit_bundles = self._reorder_dihedral_inputs_by_reference(
                occurrences,
                ca_sets,
                fit_bundles,
            )
            order = self._resolve_dihedral_order(len(occurrences))
            average_fit_coords, _, meta, _ = _project_dn_bundle_layout(
                ca_sets,
                [bundle.coords for bundle in fit_bundles],
                order=order,
            )
            average_ca_coords, _, _, _ = _project_dn_bundle_layout(
                ca_sets,
                ca_sets,
                order=order,
            )
            dock_average_ca_coords, rpxdock_axis_meta = _canonicalize_asu_by_pca(
                average_ca_coords,
                target_centroid=np.mean(average_ca_coords, axis=0),
            )
            ref_occurrence = occurrences[0]
            ref_residues = protein_residues(structure, ref_occurrence.chain_id)
            average_structure = self._build_average_structure(
                source_structure=structure,
                chain_id=ref_occurrence.chain_id,
                residue_indices=np.arange(len(ref_residues), dtype=int),
                residue_atom_names=fit_bundles[0].residue_atom_names,
                average_coords=average_fit_coords,
            )
            case_dir = self._case_dir(input_file)
            average_path = case_dir / f"01_average_asu_structure{_case_suffix(input_file)}"
            monomer_path = case_dir / "01_dock_input_structure.pdb"
            _write_case_structure(average_structure, average_path, like_input=input_file)
            dock_input_structure = _build_rigid_template_structure(
                source_structure=structure,
                chain_id=ref_occurrence.chain_id,
                residue_indices=np.arange(len(ref_residues), dtype=int),
                target_ca_coords=dock_average_ca_coords,
            )
            _write_case_structure(dock_input_structure, monomer_path, like_input=monomer_path)
            dock_fit = _fit_rpxdock_geometry_kwargs(dock_average_ca_coords, point_group="D")
            dock_result = run_local_cyclic_dock(
                monomer_path,
                RpxDockRunConfig(
                    nfold=order,
                    output_dir=case_dir / "rpxdock_work",
                    python_bin=str(self.rpxdock_python_bin or "python"),
                    vendor_repo=(
                        None if self.rpxdock_vendor_repo in (None, "") else Path(str(self.rpxdock_vendor_repo))
                    ),
                    willutil_repo=(
                        None if self.rpxdock_willutil_repo in (None, "") else Path(str(self.rpxdock_willutil_repo))
                    ),
                    timeout_s=self.rpxdock_timeout_s,
                    extra_args=self.rpxdock_extra_args,
                    architecture=f"D{order}_{order}",
                    cart_bounds=dock_fit["cart_bounds"],
                    max_longaxis_dot_z=dock_fit["max_longaxis_dot_z"],
                    components_aligned_to_sym_axes=bool(dock_fit["components_aligned_to_sym_axes"]),
                    limit_rotation_to_z=bool(dock_fit["limit_rotation_to_z"]),
                ),
            )
            dock_candidates = list(dock_result.candidates) or [
                type("_PrimaryDockCandidate", (), {
                    "rank": 0,
                    "rotation": dock_result.rotation,
                    "translation": dock_result.translation,
                    "docked_structure_path": dock_result.docked_structure_path,
                })()
            ]
            projected_records: list[dict[str, Any]] = []
            candidate_compactions: list[dict[str, Any]] = []
            group_transforms = dihedral_group_transforms_canonical(order)
            for candidate_index, candidate in enumerate(dock_candidates):
                target_full_coords = _distribute_rpxdock_first_pose(
                    monomer_path=monomer_path,
                    group_transforms=group_transforms,
                    rotation=candidate.rotation,
                    translation=candidate.translation,
                )
                projected = copy.deepcopy(structure)
                projected_bundles: list[AtomBundle] = []
                for occurrence in occurrences:
                    residues = protein_residues(projected, occurrence.chain_id)
                    projected_bundles.append(
                        collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
                    )
                projected_fit = harmonize_atom_bundles(projected_bundles)
                target_fit_coords = [
                    _subset_bundle_coords_by_layout(
                        full_residue_atom_names=projected_bundles[0].residue_atom_names,
                        full_coords=coords,
                        subset_residue_atom_names=projected_fit[0].residue_atom_names,
                    )
                    for coords in target_full_coords
                ]
                _apply_fit_targets_to_movable_bundles(
                    fit_bundles=projected_fit,
                    movable_bundles=projected_bundles,
                    target_fit_coords=target_fit_coords,
                    pin_fit_atoms=True,
                )
                compaction = _compact_projected_bundles_if_dispersed(
                    projected_bundles,
                    dihedral_order=order,
                )
                candidate_compactions.append(
                    {
                        "rank": int(candidate.rank),
                        **compaction,
                    }
                )
                contact_summary = _dihedral_pair_contact_summary(compaction)
                selection_score = _dihedral_pair_contact_selection_score(
                    contact_summary,
                    rank=int(candidate.rank),
                    candidate_index=int(candidate_index),
                )
                projected_records.append(
                    {
                        "candidate_index": int(candidate_index),
                        "rank": int(candidate.rank),
                        "structure": projected,
                        "contact_summary": contact_summary,
                        "selection_score": selection_score,
                    }
                )
            if not projected_records:
                raise ValueError("rpxdock did not yield any candidates for dihedral oligomer projection.")
            selected_record = max(
                projected_records,
                key=lambda record: record["selection_score"],
            )
            projected_primary = selected_record["structure"]
            projected_candidates = [
                (int(record["rank"]), record["structure"])
                for record in projected_records
                if record is not selected_record
            ]
            selected_summary = dict(selected_record["contact_summary"])
            selected_summary["selection_score"] = [
                float(value) for value in selected_record["selection_score"]
            ]
            return self._write_case_outputs(
                input_file=input_file,
                initial_structure=copy.deepcopy(structure),
                projected_structure=projected_primary,
                average_structure=average_structure,
                projected_candidates=projected_candidates,
                manifest={
                    "backend": self.backend,
                    "mode": self.mode,
                    "order": order,
                    "asu_kind": "protomer",
                    "oligo": self.oligo,
                    "reference_chain_index": self.reference_chain_index,
                    "rpxdock_architecture": f"D{order}_{order}",
                    "rpxdock_output": str(dock_result.docked_structure_path),
                    "rpxdock_output_candidates": [
                        str(candidate.docked_structure_path) for candidate in dock_candidates
                    ],
                    "rpxdock_command": list(dock_result.command),
                    "rpxdock_stdout": str(dock_result.stdout_path),
                    "rpxdock_stderr": str(dock_result.stderr_path),
                    "rpxdock_input": str(monomer_path),
                    "rpxdock_fit": dock_fit,
                    "rpxdock_axis": rpxdock_axis_meta,
                    "rpxdock_postprocess": {
                        "dihedral_radial_compaction": candidate_compactions,
                        "selected_primary_rank": int(selected_record["rank"]),
                        "selected_primary_candidate_index": int(selected_record["candidate_index"]),
                        "selected_primary_contact_summary": selected_summary,
                        "primary_selection_criterion": _DIHEDRAL_RPXDOCK_PRIMARY_SELECTION_CRITERION,
                    },
                    **meta,
                },
            )
        raise ValueError(f"backend={self.backend!r} is not supported for mode={self.mode!r}.")

    def _run_pseudocycle_exact(self, input_file: str | Path) -> str:
        structure = load_structure(input_file)
        chain_map = unique_protomer_chain_map(self.oligo)
        proto = (
            str(self.included_protomer)
            if self.included_protomer is not None
            else next(iter(chain_map.keys()))
        )
        chain_id = chain_map[proto]
        residues = protein_residues(structure, chain_id)
        movable_groups = build_internal_repeat_movable_groups(proto)
        groups = self._internal_repeat_fit_groups(
            proto,
            residue_count=len(residues),
        )
        fit_bundles = harmonize_atom_bundles(
            [collect_atom_bundle(residues, group) for group in groups]
        )
        movable_bundles = [collect_atom_bundle(residues, group) for group in movable_groups]
        validate_compatible_atom_bundles(fit_bundles)
        order = self._resolve_order(len(fit_bundles))
        if order != len(fit_bundles):
            raise ValueError("Pseudocycle exact projection requires one bundle per repeat copy.")
        ca_sets = [residue_ca_coordinates(residues, group) for group in groups]
        average_fit_coords, desired_fit_coords, meta, _ = _project_cn_bundle_layout(
            ca_sets,
            [bundle.coords for bundle in fit_bundles],
        )
        movable_common_bundles = harmonize_atom_bundles(movable_bundles)
        average_movable_coords, desired_movable_coords, _, _ = _project_cn_bundle_layout(
            ca_sets,
            [bundle.coords for bundle in movable_common_bundles],
        )
        projected = copy.deepcopy(structure)
        projected_residues = protein_residues(projected, chain_id)
        projected_movable = [
            collect_atom_bundle(projected_residues, group) for group in movable_groups
        ]
        projected_movable_common = harmonize_atom_bundles(projected_movable)
        _apply_fit_targets_to_movable_bundles(
            fit_bundles=projected_movable_common,
            movable_bundles=projected_movable,
            target_fit_coords=desired_movable_coords,
            pin_fit_atoms=True,
        )
        average_structure = self._build_average_structure(
            source_structure=structure,
            chain_id=chain_id,
            residue_indices=movable_groups[0],
            residue_atom_names=movable_common_bundles[0].residue_atom_names,
            average_coords=average_movable_coords,
        )
        return self._write_case_outputs(
            input_file=input_file,
            initial_structure=copy.deepcopy(structure),
            projected_structure=projected,
            average_structure=average_structure,
            manifest={
                "backend": self.backend,
                "mode": self.mode,
                "order": order,
                "asu_kind": "repeat_unit",
                "oligo": self.oligo,
                "included_protomer": proto,
                "interface_selection": self.interface_selection,
                "interface": self.interface,
                **meta,
            },
        )

    def _run_cyclic_oligomer_exact(self, input_file: str | Path) -> str:
        structure = load_structure(input_file)
        occurrences, ca_sets, fit_bundles, _ = cyclic_oligomer_atom_bundles(
            structure,
            self.oligo,
        )
        occurrences, ca_sets, fit_bundles = self._reorder_cyclic_inputs_by_reference(
            occurrences,
            ca_sets,
            fit_bundles,
        )
        order = self._resolve_order(len(occurrences))
        average_chain_coords, desired_chain_coords, meta, _ = _project_cn_bundle_layout(
            ca_sets,
            [bundle.coords for bundle in fit_bundles],
        )
        projected = copy.deepcopy(structure)
        projected_bundles: list[AtomBundle] = []
        for occurrence in occurrences:
            residues = protein_residues(projected, occurrence.chain_id)
            projected_bundles.append(
                    collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
                )
        projected_fit = harmonize_atom_bundles(projected_bundles)
        _apply_fit_targets_to_movable_bundles(
            fit_bundles=projected_fit,
            movable_bundles=projected_bundles,
            target_fit_coords=desired_chain_coords,
            pin_fit_atoms=True,
        )
        ref_occurrence = occurrences[0]
        first_chain_residues = protein_residues(structure, ref_occurrence.chain_id)
        average_structure = self._build_average_structure(
            source_structure=structure,
            chain_id=ref_occurrence.chain_id,
            residue_indices=np.arange(len(first_chain_residues), dtype=int),
            residue_atom_names=fit_bundles[0].residue_atom_names,
            average_coords=average_chain_coords,
        )
        return self._write_case_outputs(
            input_file=input_file,
            initial_structure=copy.deepcopy(structure),
            projected_structure=projected,
            average_structure=average_structure,
            manifest={
                "backend": self.backend,
                "mode": self.mode,
                "order": order,
                "asu_kind": "protomer",
                "oligo": self.oligo,
                "reference_chain_index": self.reference_chain_index,
                **meta,
            },
        )

    def _run_dihedral_oligomer_exact(self, input_file: str | Path) -> str:
        structure = load_structure(input_file)
        occurrences, ca_sets, fit_bundles, _ = cyclic_oligomer_atom_bundles(
            structure,
            self.oligo,
        )
        occurrences, ca_sets, fit_bundles = self._reorder_dihedral_inputs_by_reference(
            occurrences,
            ca_sets,
            fit_bundles,
        )
        order = self._resolve_dihedral_order(len(occurrences))
        average_chain_coords, desired_chain_coords, meta, _ = _project_dn_bundle_layout(
            ca_sets,
            [bundle.coords for bundle in fit_bundles],
            order=order,
        )
        projected = copy.deepcopy(structure)
        projected_bundles: list[AtomBundle] = []
        for occurrence in occurrences:
            residues = protein_residues(projected, occurrence.chain_id)
            projected_bundles.append(
                collect_atom_bundle(residues, np.arange(len(residues), dtype=int))
            )
        projected_fit = harmonize_atom_bundles(projected_bundles)
        _apply_fit_targets_to_movable_bundles(
            fit_bundles=projected_fit,
            movable_bundles=projected_bundles,
            target_fit_coords=desired_chain_coords,
            pin_fit_atoms=True,
        )
        ref_occurrence = occurrences[0]
        first_chain_residues = protein_residues(structure, ref_occurrence.chain_id)
        average_structure = self._build_average_structure(
            source_structure=structure,
            chain_id=ref_occurrence.chain_id,
            residue_indices=np.arange(len(first_chain_residues), dtype=int),
            residue_atom_names=fit_bundles[0].residue_atom_names,
            average_coords=average_chain_coords,
        )
        return self._write_case_outputs(
            input_file=input_file,
            initial_structure=copy.deepcopy(structure),
            projected_structure=projected,
            average_structure=average_structure,
            manifest={
                "backend": self.backend,
                "mode": self.mode,
                "order": order,
                "asu_kind": "protomer",
                "oligo": self.oligo,
                "reference_chain_index": self.reference_chain_index,
                **meta,
            },
        )

    def _run_internal_repeat(self, input_file: str | Path) -> str:
        structure = load_structure(input_file)
        chain_map = unique_protomer_chain_map(self.oligo)
        proto = (
            str(self.included_protomer)
            if self.included_protomer is not None
            else next(iter(chain_map.keys()))
        )
        chain_id = chain_map[proto]
        residues = protein_residues(structure, chain_id)
        movable_groups = build_internal_repeat_movable_groups(proto)
        groups = self._internal_repeat_fit_groups(
            proto,
            residue_count=len(residues),
        )
        ca_sets = [residue_ca_coordinates(residues, group) for group in groups]
        fit_bundles = harmonize_atom_bundles(
            [collect_atom_bundle(residues, group) for group in groups]
        )
        movable_bundles = [collect_atom_bundle(residues, group) for group in movable_groups]
        validate_compatible_atom_bundles(fit_bundles)

        ref_ca = ca_sets[0]
        aligned_atom_coords: list[np.ndarray] = []
        for ca_coords, atom_bundle in zip(ca_sets, fit_bundles, strict=True):
            R, t = kabsch_transform(ca_coords, ref_ca)
            aligned_atom_coords.append(apply_transform(atom_bundle.coords, R, t))
        canonical_atom_coords = np.mean(np.stack(aligned_atom_coords, axis=0), axis=0)

        pair_indices = list(range(len(ca_sets) - 1))
        if self.mode == "pseudocycle":
            pair_indices = list(range(len(ca_sets)))

        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        for idx in pair_indices:
            left = ca_sets[idx]
            right = ca_sets[(idx + 1) % len(ca_sets)]
            R, t = kabsch_transform(left, right)
            rotations.append(R)
            translations.append(t)

        step_R = average_rotation_matrices(rotations)
        step_t = np.mean(np.stack(translations, axis=0), axis=0)
        if self.repeat_canonicalizer_mode == "translation":
            step_R = np.eye(3, dtype=np.float64)
        if self.mode == "pseudocycle":
            axis = rotation_axis(step_R, step_t)
            step_t = step_t - axis * float(np.dot(step_t, axis))
        elif self.repeat_canonicalizer_mode == "translation":
            canonical_centroids: list[np.ndarray] = []
            for ca_coords in ca_sets:
                R, t = kabsch_transform(ca_coords, ref_ca)
                canonical_centroids.append(np.mean(apply_transform(ca_coords, R, t), axis=0))
            deltas = [
                canonical_centroids[idx + 1] - canonical_centroids[idx]
                for idx in range(len(canonical_centroids) - 1)
            ]
            if deltas:
                step_t = np.mean(np.stack(deltas, axis=0), axis=0)

        placed_transforms: list[tuple[np.ndarray, np.ndarray]] = [
            (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        ]
        R_acc = np.eye(3, dtype=np.float64)
        t_acc = np.zeros(3, dtype=np.float64)
        for _ in range(1, len(fit_bundles)):
            R_acc, t_acc = compose_transforms(R_acc, t_acc, step_R, step_t)
            placed_transforms.append((R_acc, t_acc))

        desired_fit_coords = [
            apply_transform(canonical_atom_coords, Rk, tk)
            for Rk, tk in placed_transforms
        ]

        for fit_bundle, movable_bundle, target_coords in zip(
            fit_bundles, movable_bundles, desired_fit_coords, strict=True
        ):
            corr_R, corr_t = kabsch_transform(fit_bundle.coords, target_coords)
            assign_atom_bundle_coordinates(
                movable_bundle,
                apply_transform(movable_bundle.coords, corr_R, corr_t),
            )
        return self._write_output(structure, input_file)

    def _run_shared_core(self, input_file: str | Path) -> str:
        structure = load_structure(input_file)
        groups = build_shared_core_residue_groups(self.oligo)
        chain_map = unique_protomer_chain_map(self.oligo)
        protomer_order = list(groups.keys())

        ref_proto = protomer_order[0]
        ref_residues = protein_residues(structure, chain_map[ref_proto])
        ref_group = groups[ref_proto]
        ref_ca = residue_ca_coordinates(ref_residues, ref_group)

        core_bundles = []
        full_bundles = []
        align_transforms: list[tuple[np.ndarray, np.ndarray]] = []
        for proto in protomer_order:
            residues = protein_residues(structure, chain_map[proto])
            residue_indices = groups[proto]
            ca_coords = residue_ca_coordinates(residues, residue_indices)
            atom_bundle = collect_atom_bundle(residues, residue_indices)
            core_bundles.append(atom_bundle)
            full_bundles.append(collect_atom_bundle(residues, np.arange(len(residues), dtype=int)))
            R, t = kabsch_transform(ca_coords, ref_ca)
            align_transforms.append((R, t))

        core_bundles = harmonize_atom_bundles(core_bundles)
        aligned_atom_coords = [
            apply_transform(core_bundle.coords, R, t)
            for core_bundle, (R, t) in zip(core_bundles, align_transforms, strict=True)
        ]
        validate_compatible_atom_bundles(core_bundles)
        canonical_atom_coords = np.mean(np.stack(aligned_atom_coords, axis=0), axis=0)
        for core_bundle, full_bundle in zip(core_bundles, full_bundles, strict=True):
            corr_R, corr_t = kabsch_transform(core_bundle.coords, canonical_atom_coords)
            assign_atom_bundle_coordinates(
                full_bundle,
                apply_transform(full_bundle.coords, corr_R, corr_t),
            )
        return self._write_output(structure, input_file)
