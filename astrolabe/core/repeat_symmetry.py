from __future__ import annotations

import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astrolabe.core import global_config
from astrolabe.utils.mmcif_ids import int_id_to_str_id


@dataclass(frozen=True)
class ComponentOccurrence:
    protomer: str
    component_name: str
    segment_id: int
    component_index: int
    start: int
    stop: int


@dataclass
class AtomBundle:
    atoms: tuple[Any, ...]
    residue_atom_names: tuple[tuple[str, ...], ...]
    coords: np.ndarray


@dataclass(frozen=True)
class ChainOccurrence:
    protomer: str
    chain_id: str
    chain_index: int
    occurrence_index: int


@dataclass(frozen=True)
class ExactCnCanonicalization:
    order: int
    axis_point: np.ndarray
    axis_direction: np.ndarray
    step_rotation: np.ndarray
    step_translation: np.ndarray
    to_canonical_rotation: np.ndarray
    to_canonical_translation: np.ndarray
    from_canonical_rotation: np.ndarray
    from_canonical_translation: np.ndarray


def _structure_protein_chain_ids(structure) -> list[str]:
    try:
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("biopython is required for symmetry structure parsing.") from exc

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError("Structure contains no models.")

    chain_ids: list[str] = []
    for chain in model:
        residues = [res for res in chain if is_aa(res, standard=True) and "CA" in res]
        if residues:
            chain_ids.append(str(chain.id))
    if not chain_ids:
        raise ValueError("Structure contains no protein chains with standard amino-acid CA atoms.")
    return chain_ids


def oligo_chain_occurrences(
    oligo_name: str,
    *,
    chain_ids: list[str] | tuple[str, ...] | None = None,
) -> list[ChainOccurrence]:
    oligo = str(oligo_name or "")
    if not oligo:
        raise ValueError("Oligomer name must be non-empty.")
    if chain_ids is not None and len(chain_ids) != len(oligo):
        raise ValueError(
            f"Provided chain_ids length ({len(chain_ids)}) must match oligomer length ({len(oligo)})."
        )
    proto_counts: dict[str, int] = {}
    out: list[ChainOccurrence] = []
    for chain_index, proto in enumerate(oligo):
        proto_counts[proto] = proto_counts.get(proto, 0) + 1
        out.append(
            ChainOccurrence(
                protomer=proto,
                chain_id=(
                    str(chain_ids[chain_index])
                    if chain_ids is not None
                    else int_id_to_str_id(chain_index + 1)
                ),
                chain_index=chain_index,
                occurrence_index=proto_counts[proto] - 1,
            )
        )
    return out


def cyclic_oligomer_chain_occurrences(
    oligo_name: str,
    included_protomer: str | None = None,
    *,
    chain_ids: list[str] | tuple[str, ...] | None = None,
) -> list[ChainOccurrence]:
    occurrences = oligo_chain_occurrences(oligo_name, chain_ids=chain_ids)
    if included_protomer is None:
        return occurrences
    proto = str(included_protomer)
    selected = [item for item in occurrences if item.protomer == proto]
    if not selected:
        raise ValueError(
            f"Included protomer {proto!r} is not present in oligomer {oligo_name!r}."
        )
    return selected


def cyclic_oligomer_atom_bundles(
    structure,
    oligo_name: str,
    *,
    included_protomer: str | None = None,
    fit_residue_indices: np.ndarray | list[int] | None = None,
    movable_residue_indices: np.ndarray | list[int] | None = None,
) -> tuple[list[ChainOccurrence], list[np.ndarray], list[AtomBundle], list[AtomBundle]]:
    chain_ids = _structure_protein_chain_ids(structure)
    occurrences = cyclic_oligomer_chain_occurrences(
        oligo_name,
        included_protomer=included_protomer,
        chain_ids=chain_ids,
    )
    if len(occurrences) < 2:
        raise ValueError(
            f"Oligomer {oligo_name!r} needs at least 2 chains for cyclic canonicalization."
        )

    coord_sets: list[np.ndarray] = []
    fit_bundles_raw: list[AtomBundle] = []
    movable_bundles: list[AtomBundle] = []
    for occurrence in occurrences:
        residues = protein_residues(structure, occurrence.chain_id)
        if fit_residue_indices is None:
            fit_idx = np.arange(len(residues), dtype=int)
        else:
            fit_idx = np.asarray(fit_residue_indices, dtype=int)
        if movable_residue_indices is None:
            movable_idx = np.arange(len(residues), dtype=int)
        else:
            movable_idx = np.asarray(movable_residue_indices, dtype=int)
        coord_sets.append(residue_ca_coordinates(residues, fit_idx))
        fit_bundles_raw.append(collect_atom_bundle(residues, fit_idx))
        movable_bundles.append(collect_atom_bundle(residues, movable_idx))

    fit_bundles = harmonize_atom_bundles(fit_bundles_raw)
    validate_compatible_atom_bundles(fit_bundles)
    return occurrences, coord_sets, fit_bundles, movable_bundles


def protomer_component_occurrences(protomer_name: str) -> list[ComponentOccurrence]:
    protomer = global_config.protomers[protomer_name]
    occurrences: list[ComponentOccurrence] = []
    for idx, (name, segment_id, seg_slice) in enumerate(
        zip(
            protomer.components,
            protomer.component_ids,
            protomer.components_slices,
            strict=True,
        )
    ):
        occurrences.append(
            ComponentOccurrence(
                protomer=protomer_name,
                component_name=str(name),
                segment_id=int(segment_id),
                component_index=int(idx),
                start=int(seg_slice.start),
                stop=int(seg_slice.stop),
            )
        )
    return occurrences


def build_internal_repeat_residue_groups(protomer_name: str) -> list[np.ndarray]:
    occurrences = protomer_component_occurrences(protomer_name)
    families: dict[int, list[ComponentOccurrence]] = {}
    for occurrence in occurrences:
        families.setdefault(int(occurrence.segment_id), []).append(occurrence)

    repeated_families = [members for members in families.values() if len(members) > 1]
    if not repeated_families:
        raise ValueError(f"Protomer {protomer_name!r} does not contain repeated segments.")

    copy_counts = {len(members) for members in repeated_families}
    if len(copy_counts) != 1:
        raise ValueError(
            f"Protomer {protomer_name!r} has inconsistent repeat copy counts: {sorted(copy_counts)}"
        )
    copy_count = copy_counts.pop()

    groups: list[np.ndarray] = []
    for copy_idx in range(copy_count):
        residues: list[int] = []
        for family in repeated_families:
            occurrence = family[copy_idx]
            residues.extend(range(int(occurrence.start), int(occurrence.stop)))
        groups.append(np.asarray(residues, dtype=int))
    return groups


def build_internal_repeat_movable_groups(protomer_name: str) -> list[np.ndarray]:
    occurrences = protomer_component_occurrences(protomer_name)
    families: dict[int, list[ComponentOccurrence]] = {}
    for occurrence in occurrences:
        families.setdefault(int(occurrence.segment_id), []).append(occurrence)

    repeated_families = {
        segment_id: members for segment_id, members in families.items() if len(members) > 1
    }
    if not repeated_families:
        raise ValueError(f"Protomer {protomer_name!r} does not contain repeated segments.")

    copy_counts = {len(members) for members in repeated_families.values()}
    if len(copy_counts) != 1:
        raise ValueError(
            f"Protomer {protomer_name!r} has inconsistent repeat copy counts: {sorted(copy_counts)}"
        )
    copy_count = copy_counts.pop()
    groups: list[list[int]] = [[] for _ in range(copy_count)]
    family_progress = {segment_id: 0 for segment_id in repeated_families}
    current_copy = 0
    seen_anchor = False

    for occurrence in occurrences:
        segment_id = int(occurrence.segment_id)
        if segment_id in repeated_families:
            current_copy = family_progress[segment_id]
            family_progress[segment_id] += 1
            seen_anchor = True
        elif not seen_anchor:
            current_copy = 0
        groups[current_copy].extend(range(int(occurrence.start), int(occurrence.stop)))

    return [np.asarray(indices, dtype=int) for indices in groups]


def build_shared_core_residue_groups(oligo_name: str) -> dict[str, np.ndarray]:
    protomer_order = list(dict.fromkeys(str(oligo_name or "")))
    if len(protomer_order) < 2:
        raise ValueError("Shared-core groups require at least two distinct protomers.")

    all_occurrences: list[ComponentOccurrence] = []
    for proto in protomer_order:
        all_occurrences.extend(protomer_component_occurrences(proto))

    family_map: dict[int, list[ComponentOccurrence]] = defaultdict(list)
    for occurrence in all_occurrences:
        family_map[int(occurrence.segment_id)].append(occurrence)

    groups: dict[str, list[int]] = {proto: [] for proto in protomer_order}
    reference_occurrences = protomer_component_occurrences(protomer_order[0])
    protomer_set = set(protomer_order)
    seen_segment_ids: set[int] = set()
    for occurrence in reference_occurrences:
        segment_id = int(occurrence.segment_id)
        if segment_id in seen_segment_ids:
            continue
        seen_segment_ids.add(segment_id)
        family = family_map.get(segment_id, [])
        by_proto: dict[str, list[ComponentOccurrence]] = defaultdict(list)
        for member in family:
            by_proto[str(member.protomer)].append(member)
        if set(by_proto.keys()) != protomer_set:
            continue
        if any(len(by_proto[proto]) != 1 for proto in protomer_order):
            raise ValueError(
                f"Shared-core family {segment_id} is ambiguous across protomers in {oligo_name!r}."
            )
        for proto in protomer_order:
            member = by_proto[proto][0]
            groups[proto].extend(range(int(member.start), int(member.stop)))

    if not any(groups.values()):
        raise ValueError(f"Oligomer {oligo_name!r} has no shared-core alias families.")

    return {proto: np.asarray(indices, dtype=int) for proto, indices in groups.items()}


def unique_protomer_chain_map(oligo_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx, proto in enumerate(str(oligo_name or "")):
        if proto in mapping:
            raise ValueError(
                f"Oligomer {oligo_name!r} repeats protomer {proto!r}; unique mapping is required here."
            )
        mapping[proto] = int_id_to_str_id(idx + 1)
    return mapping


def load_structure(path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    structure_path = Path(path)
    name = structure_path.name.lower()
    is_cif = any(ext in name for ext in [".cif", ".mmcif"])
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)

    handle = None
    try:
        if name.endswith(".gz"):
            handle = gzip.open(str(structure_path), "rt")
            structure = parser.get_structure("repeat_symmetry", handle)
        else:
            structure = parser.get_structure("repeat_symmetry", str(structure_path))
    finally:
        if handle is not None:
            handle.close()
    return structure


def save_structure(structure, output_path: str | Path, *, like_input: str | Path) -> None:
    from Bio.PDB import MMCIFIO, PDBIO  # type: ignore

    output_path = Path(output_path)
    input_name = Path(like_input).name.lower()
    is_cif = any(ext in input_name for ext in [".cif", ".mmcif"])
    io = MMCIFIO() if is_cif else PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def protein_residues(structure, chain_id: str) -> list[Any]:
    try:
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("biopython is required for symmetry structure parsing.") from exc

    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError("Structure contains no models.")

    chain = None
    for candidate in model:
        if str(candidate.id) == str(chain_id):
            chain = candidate
            break
    if chain is None:
        available = sorted({str(candidate.id) for candidate in model})
        raise ValueError(f"Chain {chain_id!r} not found. Available chains: {available}")

    residues = [res for res in chain if is_aa(res, standard=True) and "CA" in res]
    if not residues:
        raise ValueError(f"Chain {chain_id!r} has no standard amino-acid residues with CA atoms.")
    return residues


def select_residues(residues: list[Any], residue_indices: np.ndarray | list[int]) -> list[Any]:
    idx = np.asarray(residue_indices, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("Residue indices must be a non-empty 1D array.")
    if int(np.min(idx)) < 0 or int(np.max(idx)) >= len(residues):
        raise ValueError(
            f"Residue selection {idx.tolist()} is out of bounds for chain length {len(residues)}."
        )
    return [residues[i] for i in idx.tolist()]


def residue_ca_coordinates(residues: list[Any], residue_indices: np.ndarray | list[int]) -> np.ndarray:
    selected = select_residues(residues, residue_indices)
    return np.asarray([np.asarray(res["CA"].coord, dtype=np.float64) for res in selected], dtype=np.float64)


def collect_atom_bundle(
    residues: list[Any], residue_indices: np.ndarray | list[int]
) -> AtomBundle:
    selected = select_residues(residues, residue_indices)
    atoms: list[Any] = []
    residue_atom_names: list[tuple[str, ...]] = []
    coords: list[np.ndarray] = []
    for residue in selected:
        residue_atoms = list(residue.get_atoms())
        residue_atom_names.append(tuple(str(atom.id) for atom in residue_atoms))
        atoms.extend(residue_atoms)
        coords.extend(np.asarray(atom.coord, dtype=np.float64) for atom in residue_atoms)
    return AtomBundle(
        atoms=tuple(atoms),
        residue_atom_names=tuple(residue_atom_names),
        coords=np.asarray(coords, dtype=np.float64),
    )


def harmonize_atom_bundles(bundles: list[AtomBundle]) -> list[AtomBundle]:
    if not bundles:
        raise ValueError("No atom bundles provided.")

    residue_count = len(bundles[0].residue_atom_names)
    if residue_count == 0:
        raise ValueError("Atom bundles must contain at least one residue.")
    for idx, bundle in enumerate(bundles[1:], start=1):
        if len(bundle.residue_atom_names) != residue_count:
            raise ValueError(
                f"Atom bundle {idx} has {len(bundle.residue_atom_names)} residues; expected {residue_count}."
            )

    common_layout: list[tuple[str, ...]] = []
    for res_idx in range(residue_count):
        residue_layouts = [bundle.residue_atom_names[res_idx] for bundle in bundles]
        ref_layout = residue_layouts[0]
        shared = tuple(
            atom_name
            for atom_name in ref_layout
            if all(atom_name in layout for layout in residue_layouts[1:])
        )
        if not shared:
            raise ValueError(
                f"Residue {res_idx} has no common atom names across tied bundles: {residue_layouts}"
            )
        common_layout.append(shared)

    harmonized: list[AtomBundle] = []
    for bundle in bundles:
        atoms: list[Any] = []
        coords: list[np.ndarray] = []
        offset = 0
        for residue_layout, shared_layout in zip(bundle.residue_atom_names, common_layout, strict=True):
            residue_size = len(residue_layout)
            residue_atoms = bundle.atoms[offset : offset + residue_size]
            residue_coords = bundle.coords[offset : offset + residue_size]
            atom_index = {name: idx for idx, name in enumerate(residue_layout)}
            for atom_name in shared_layout:
                idx = atom_index[atom_name]
                atoms.append(residue_atoms[idx])
                coords.append(np.asarray(residue_coords[idx], dtype=np.float64))
            offset += residue_size
        harmonized.append(
            AtomBundle(
                atoms=tuple(atoms),
                residue_atom_names=tuple(common_layout),
                coords=np.asarray(coords, dtype=np.float64),
            )
        )
    return harmonized


def assign_atom_bundle_coordinates(bundle: AtomBundle, coords: np.ndarray) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != bundle.coords.shape:
        raise ValueError(
            f"Atom coordinate shape mismatch: got {coords.shape}, expected {bundle.coords.shape}."
        )
    for atom, xyz in zip(bundle.atoms, coords, strict=True):
        atom.set_coord(np.asarray(xyz, dtype=np.float64))


def validate_compatible_atom_bundles(bundles: list[AtomBundle]) -> None:
    if not bundles:
        raise ValueError("No atom bundles provided.")
    ref = bundles[0].residue_atom_names
    for idx, bundle in enumerate(bundles[1:], start=1):
        if bundle.residue_atom_names != ref:
            raise ValueError(
                f"Atom bundle {idx} is incompatible with bundle 0; tied residues must have matching atom layouts."
            )


def kabsch_transform(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Kabsch alignment requires matching (N,3) arrays, got {P.shape} and {Q.shape}.")
    Pc = P - np.mean(P, axis=0, keepdims=True)
    Qc = Q - np.mean(Q, axis=0, keepdims=True)
    cov = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = np.mean(Q, axis=0) - np.mean(P, axis=0) @ R
    return R, t


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(R, dtype=np.float64) + np.asarray(t, dtype=np.float64)


def invert_transform(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    inv_R = R.T
    inv_t = (-t) @ inv_R
    return inv_R, inv_t


def compose_transforms(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R1 = np.asarray(R1, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    return R1 @ R2, t1 @ R2 + t2


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    delta = np.asarray(P, dtype=np.float64) - np.asarray(Q, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def mean_pairwise_aligned_rmsd(coord_sets: list[np.ndarray]) -> float:
    if len(coord_sets) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(coord_sets)):
        for j in range(i + 1, len(coord_sets)):
            R, t = kabsch_transform(coord_sets[i], coord_sets[j])
            scores.append(rmsd(apply_transform(coord_sets[i], R, t), coord_sets[j]))
    return float(np.mean(scores))


def average_rotation_matrices(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        return np.eye(3, dtype=np.float64)
    mean_rot = np.mean(np.stack(rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(mean_rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def rotation_angle_degrees(R: np.ndarray) -> float:
    trace = float(np.trace(np.asarray(R, dtype=np.float64)))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_theta)))


def rotation_axis(R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    angle = math.radians(rotation_angle_degrees(R))
    if abs(math.sin(angle)) > 1e-6:
        axis = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * math.sin(angle))
    else:
        eigvals, eigvecs = np.linalg.eig(R)
        candidates = eigvecs[:, np.isclose(eigvals.real, 1.0)]
        if candidates.size:
            axis = np.asarray(candidates[:, 0].real, dtype=np.float64)
        elif t is not None and np.linalg.norm(t) > 1e-8:
            axis = np.asarray(t, dtype=np.float64)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _normalize_vector(v: np.ndarray, *, default: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.asarray(default, dtype=np.float64)
    return vec / norm


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit_axis = _normalize_vector(
        np.asarray(axis, dtype=np.float64),
        default=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    ux, uy, uz = unit_axis.tolist()
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


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _align_vector_to_z_rotation(axis: np.ndarray) -> np.ndarray:
    src = _normalize_vector(
        np.asarray(axis, dtype=np.float64),
        default=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    dst = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        return _rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=np.float64), math.pi
        )
    rot_axis = _normalize_vector(
        np.cross(src, dst),
        default=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    return _rotation_matrix_from_axis_angle(rot_axis, math.acos(dot))


def estimate_cyclic_step_transform(
    coord_sets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(coord_sets) < 2:
        raise ValueError("At least two coordinate sets are required.")
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for idx in range(len(coord_sets)):
        left = coord_sets[idx]
        right = coord_sets[(idx + 1) % len(coord_sets)]
        R, t = kabsch_transform(left, right)
        rotations.append(R)
        translations.append(t)
    step_R = average_rotation_matrices(rotations)
    step_t = np.mean(np.stack(translations, axis=0), axis=0)
    return step_R, step_t


def canonicalize_exact_cyclic_cn(
    coord_sets: list[np.ndarray],
    *,
    order: int | None = None,
) -> ExactCnCanonicalization:
    if len(coord_sets) < 2:
        raise ValueError("At least two coordinate sets are required.")
    inferred_order = len(coord_sets)
    if order is None:
        order = inferred_order
    order = int(order)
    if order < 2:
        raise ValueError(f"Cn order must be >= 2, got {order}.")
    if inferred_order != order:
        raise ValueError(
            f"Coordinate-set count ({inferred_order}) does not match requested order ({order})."
        )

    step_R_est, step_t_est = estimate_cyclic_step_transform(coord_sets)
    axis_samples = [rotation_axis(step_R_est, step_t_est)]
    for idx in range(len(coord_sets)):
        R, t = kabsch_transform(coord_sets[idx], coord_sets[(idx + 1) % len(coord_sets)])
        axis_samples.append(rotation_axis(R, t))
    axis_direction = _normalize_vector(
        np.mean(np.stack(axis_samples, axis=0), axis=0),
        default=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    theta = 2.0 * math.pi / float(order)
    step_rotation = _rotation_matrix_from_axis_angle(axis_direction, theta)
    step_translation_seed = step_t_est - axis_direction * float(
        np.dot(step_t_est, axis_direction)
    )
    A = (np.eye(3, dtype=np.float64) - step_rotation).T
    b = np.asarray(step_translation_seed, dtype=np.float64).T
    axis_point = np.linalg.lstsq(A, b, rcond=None)[0].T
    axis_point = axis_point - axis_direction * float(np.dot(axis_point, axis_direction))
    step_translation = axis_point - axis_point @ step_rotation

    align_R = _align_vector_to_z_rotation(axis_direction)
    align_t = (-axis_point) @ align_R
    first_centroid = np.mean(np.asarray(coord_sets[0], dtype=np.float64), axis=0, keepdims=True)
    first_centroid_can = apply_transform(first_centroid, align_R, align_t)[0]
    xy = np.asarray([first_centroid_can[0], first_centroid_can[1]], dtype=np.float64)
    if float(np.linalg.norm(xy)) > 1e-8:
        around_z = _rotation_matrix_z(math.atan2(float(xy[1]), float(xy[0])))
        to_can_R, to_can_t = compose_transforms(
            align_R,
            align_t,
            around_z,
            np.zeros(3, dtype=np.float64),
        )
    else:
        to_can_R, to_can_t = align_R, align_t
    from_can_R, from_can_t = invert_transform(to_can_R, to_can_t)

    return ExactCnCanonicalization(
        order=order,
        axis_point=np.asarray(axis_point, dtype=np.float64),
        axis_direction=np.asarray(axis_direction, dtype=np.float64),
        step_rotation=np.asarray(step_rotation, dtype=np.float64),
        step_translation=np.asarray(step_translation, dtype=np.float64),
        to_canonical_rotation=np.asarray(to_can_R, dtype=np.float64),
        to_canonical_translation=np.asarray(to_can_t, dtype=np.float64),
        from_canonical_rotation=np.asarray(from_can_R, dtype=np.float64),
        from_canonical_translation=np.asarray(from_can_t, dtype=np.float64),
    )


def cyclic_group_transforms_canonical(order: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = int(order)
    if order < 2:
        raise ValueError(f"Cn order must be >= 2, got {order}.")
    theta = 2.0 * math.pi / float(order)
    return [
        (_rotation_matrix_z(theta * idx), np.zeros(3, dtype=np.float64))
        for idx in range(order)
    ]


def dihedral_group_transforms_canonical(order: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = int(order)
    if order < 2:
        raise ValueError(f"Dn order must be >= 2, got {order}.")
    theta = 2.0 * math.pi / float(order)
    half_turn_x = _rotation_matrix_x(math.pi)
    transforms: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(order):
        transforms.append(
            (_rotation_matrix_z(theta * idx), np.zeros(3, dtype=np.float64))
        )
    for idx in range(order):
        transforms.append(
            (half_turn_x @ _rotation_matrix_z(theta * idx), np.zeros(3, dtype=np.float64))
        )
    return transforms


def point_group_transforms_canonical(
    point_group: str,
    order: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    group = str(point_group or "").strip().upper()
    if group.startswith("C"):
        return cyclic_group_transforms_canonical(order)
    if group.startswith("D"):
        return dihedral_group_transforms_canonical(order)
    raise ValueError(f"Unsupported point group {point_group!r}; expected Cn or Dn.")


def distribute_template_by_cyclic_group(
    template_coords: np.ndarray,
    canonicalization: ExactCnCanonicalization,
    *,
    canonical_template: bool = False,
) -> list[np.ndarray]:
    template = np.asarray(template_coords, dtype=np.float64)
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError(
            f"Template coordinates must have shape (N,3), got {template.shape}."
        )
    if canonical_template:
        template_can = template
    else:
        template_can = apply_transform(
            template,
            canonicalization.to_canonical_rotation,
            canonicalization.to_canonical_translation,
        )
    out: list[np.ndarray] = []
    for Rk, tk in cyclic_group_transforms_canonical(canonicalization.order):
        placed_can = apply_transform(template_can, Rk, tk)
        placed = apply_transform(
            placed_can,
            canonicalization.from_canonical_rotation,
            canonicalization.from_canonical_translation,
        )
        out.append(placed)
    return out


def assign_cyclic_group_to_bundles(
    *,
    template_bundle: AtomBundle,
    target_bundles: list[AtomBundle],
    canonicalization: ExactCnCanonicalization,
    canonical_template: bool = False,
) -> None:
    if len(target_bundles) != canonicalization.order:
        raise ValueError(
            f"Bundle count ({len(target_bundles)}) must match symmetry order ({canonicalization.order})."
        )
    template_layout = template_bundle.residue_atom_names
    for idx, bundle in enumerate(target_bundles):
        if bundle.residue_atom_names != template_layout:
            raise ValueError(
                f"Target bundle {idx} is incompatible with template bundle atom layout."
            )
    distributed = distribute_template_by_cyclic_group(
        template_bundle.coords,
        canonicalization,
        canonical_template=canonical_template,
    )
    for bundle, coords in zip(target_bundles, distributed, strict=True):
        assign_atom_bundle_coordinates(bundle, coords)
