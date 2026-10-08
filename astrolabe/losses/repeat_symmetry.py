from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from astrolabe.core.repeat_symmetry import (
    apply_transform,
    build_internal_repeat_residue_groups,
    build_shared_core_residue_groups,
    canonicalize_exact_cyclic_cn,
    cyclic_oligomer_atom_bundles,
    dihedral_group_transforms_canonical,
    kabsch_transform,
    load_structure,
    mean_pairwise_aligned_rmsd,
    protein_residues,
    point_group_transforms_canonical,
    residue_ca_coordinates,
    rotation_angle_degrees,
    rotation_axis,
    unique_protomer_chain_map,
)
from astrolabe.core import global_config
from astrolabe.losses.base import loss_registry


_ANANAS_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _normalize_good(raw_value: float, *, m: float, s: float) -> float:
    if s <= 0:
        raise ValueError("Normalization scale 's' must be > 0.")
    bad = 1.0 / (1.0 + math.exp(-((float(raw_value) - float(m)) / float(s))))
    return float(1.0 - bad)


def _normalize_bounded_good(raw_value: float, *, best: float, worst: float) -> float:
    best = float(best)
    worst = float(worst)
    raw_value = float(raw_value)
    if worst <= best:
        raise ValueError("Normalization bound 'worst' must be > 'best'.")
    if raw_value <= best:
        return 1.0
    if raw_value >= worst:
        return 0.0
    return float((worst - raw_value) / (worst - best))


def _tail(text: str, *, max_chars: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _normalize_point_group(point_group: str | None, order: int | None, chain_count: int) -> tuple[str, int]:
    raw = str(point_group or "C").strip().upper()
    if raw in {"AUTO", ""}:
        raw = "C"
    if raw[0] not in {"C", "D"}:
        raise ValueError(f"Unsupported point_group {point_group!r}; expected Cn or Dn.")
    if len(raw) > 1:
        parsed = int(raw[1:])
        if order is not None and int(order) != parsed:
            raise ValueError(f"point_group {raw!r} conflicts with order={order}.")
        order = parsed
    if raw[0] == "C":
        resolved = int(order) if order is not None else int(chain_count)
        if chain_count != resolved:
            raise ValueError(
                f"Cn point-group loss expected {resolved} chains, got {chain_count}."
            )
    else:
        if order is None:
            if chain_count % 2:
                raise ValueError(
                    f"Dn point-group loss requires an even chain count, got {chain_count}."
                )
            resolved = chain_count // 2
        else:
            resolved = int(order)
        if chain_count != 2 * resolved:
            raise ValueError(
                f"Dn point-group loss expected {2 * resolved} chains for D{resolved}, got {chain_count}."
            )
    if resolved < 2:
        raise ValueError(f"Point-group order must be >= 2, got {resolved}.")
    return raw[0], resolved


def _ananas_group_arg(point_group: str | None, order: int | None, chain_count: int) -> str:
    group, resolved_order = _normalize_point_group(point_group, order, chain_count)
    return f"{group.lower()}{resolved_order}"


def _resolve_ananas_path(ananas_path: str | os.PathLike[str] | None = None) -> str:
    raw = None if ananas_path in (None, "") else str(ananas_path)
    if raw is None:
        try:
            configured = getattr(global_config, "ananas_path", None)
        except RuntimeError:
            configured = None
        if configured not in (None, ""):
            raw = str(configured)
        else:
            env_path = os.environ.get("GH_ANANAS")
            if env_path not in (None, ""):
                raw = str(env_path)

    if raw:
        expanded = Path(os.path.expanduser(os.path.expandvars(raw)))
        if expanded.exists():
            return str(expanded.resolve())
        repo_relative = Path(__file__).resolve().parents[2] / expanded
        if repo_relative.exists():
            return str(repo_relative.resolve())
        found = shutil.which(raw)
        if found:
            return found
        raise FileNotFoundError(
            f"AnAnaS executable not found at {raw!r}. Set loss param `ananas_path` "
            "or global_config.ananas_path/GH_ANANAS to an executable path."
        )

    for candidate in ("AnAnaS", "ananas", "ananas-linux"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError(
        "AnAnaS executable not found. Set loss param `ananas_path`, "
        "global_config.ananas_path, or GH_ANANAS."
    )


def _parse_ananas_rmsd(stdout: str) -> float:
    averages = [
        float(match.group(1))
        for match in re.finditer(
            rf"^\s*Average\s+RMSD\s*:\s*({_ANANAS_FLOAT_RE.pattern})\b",
            stdout or "",
            flags=re.IGNORECASE | re.MULTILINE,
        )
    ]
    if averages:
        return averages[-1]

    row_rmsds: list[float] = []
    for line in (stdout or "").splitlines():
        if not re.match(r"^\s*[-+]?(?:\d|\.)", line):
            continue
        values = [float(token) for token in _ANANAS_FLOAT_RE.findall(line)]
        if len(values) >= 6:
            row_rmsds.append(values[0])
    if row_rmsds:
        return float(np.mean(row_rmsds))

    raise ValueError("Could not parse AnAnaS RMSD from stdout.")


def ananas_rmsd_wrapper(
    structure_path: str | os.PathLike[str],
    symmetry_group: str,
    *,
    ananas_path: str | os.PathLike[str] | None = None,
    timeout_s: float = 30.0,
    missing_rmsd: float | None = None,
) -> float:
    executable = _resolve_ananas_path(ananas_path)
    cmd = [executable, str(structure_path), str(symmetry_group)]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"AnAnaS timed out after {timeout_s}s for {structure_path} with group {symmetry_group}."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "AnAnaS failed with exit code "
            f"{result.returncode} for {structure_path} with group {symmetry_group}. "
            f"stderr: {_tail(result.stderr)} stdout: {_tail(result.stdout)}"
        )

    try:
        return _parse_ananas_rmsd(result.stdout or "")
    except ValueError as exc:
        if missing_rmsd is not None:
            warnings.warn(
                "AnAnaS did not report a parseable RMSD for "
                f"{structure_path} with group {symmetry_group}; using {missing_rmsd}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float(missing_rmsd)
        raise RuntimeError(
            f"AnAnaS output did not contain a parseable RMSD for {structure_path} "
            f"with group {symmetry_group}. stdout: {_tail(result.stdout)}"
        ) from exc


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


def _point_group_projection(
    coord_sets: list[np.ndarray],
    *,
    point_group: str | None,
    order: int | None,
) -> tuple[str, int, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    group, resolved_order = _normalize_point_group(point_group, order, len(coord_sets))
    if group == "C":
        canonicalization = canonicalize_exact_cyclic_cn(coord_sets, order=resolved_order)
    else:
        canonicalization = canonicalize_exact_cyclic_cn(
            coord_sets[:resolved_order],
            order=resolved_order,
        )
    canonical_sets = [
        apply_transform(
            coords,
            canonicalization.to_canonical_rotation,
            canonicalization.to_canonical_translation,
        )
        for coords in coord_sets
    ]
    transforms = point_group_transforms_canonical(group, resolved_order)
    inverse_rotations = [Rk.T for Rk, _ in transforms]
    average = _aligned_average_in_reference_frame(canonical_sets, inverse_rotations)
    desired = [apply_transform(average, Rk, tk) for Rk, tk in transforms]
    return group, resolved_order, canonical_sets, average, desired, inverse_rotations


def _oligomer_coord_sets(eval_result, oligo_name: str) -> list[np.ndarray]:
    structure = load_structure(Path(eval_result.complex_paths[oligo_name]))
    _, coord_sets, _, _ = cyclic_oligomer_atom_bundles(structure, str(oligo_name))
    return coord_sets


def _assembly_radius(coord_sets: list[np.ndarray]) -> float:
    coords = np.concatenate([np.asarray(c, dtype=np.float64) for c in coord_sets], axis=0)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _point_group_orbit_metrics(
    coord_sets: list[np.ndarray],
    *,
    point_group: str | None,
    order: int | None,
) -> tuple[float, float, float]:
    _group, _order, canonical_sets, average, desired, _inverse_rotations = _point_group_projection(
        coord_sets,
        point_group=point_group,
        order=order,
    )
    deltas = [
        np.asarray(observed, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        for observed, target in zip(canonical_sets, desired, strict=True)
    ]
    squared = np.concatenate([np.sum(delta * delta, axis=1) for delta in deltas], axis=0)
    orbit_rmsd = float(np.sqrt(np.mean(squared)))
    radius = max(_assembly_radius(canonical_sets), 1e-8)
    rotation_errors: list[float] = []
    for observed, target in zip(canonical_sets, desired, strict=True):
        R_obs, _ = kabsch_transform(average, observed)
        R_target, _ = kabsch_transform(average, target)
        rotation_errors.append(rotation_angle_degrees(R_obs @ R_target.T))
    return orbit_rmsd, float(orbit_rmsd / radius), float(np.mean(rotation_errors))


def _chain_centers(coord_sets: list[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [np.mean(np.asarray(coords, dtype=np.float64), axis=0) for coords in coord_sets],
        dtype=np.float64,
    )


def _coefficient_of_variation(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-8:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _residue_contact_coordinates(residue, *, atom_mode: str) -> list[np.ndarray]:
    mode = str(atom_mode or "heavy").strip().lower()
    if mode in {"ca", "ca_ca", "calpha", "c_alpha"}:
        return [np.asarray(residue["CA"].coord, dtype=np.float64)] if "CA" in residue else []
    if mode not in {"heavy", "all_heavy", "sidechain_or_ca"}:
        raise ValueError("atom_mode must be 'heavy' or 'ca'.")

    coords: list[np.ndarray] = []
    for atom in residue.get_atoms():
        element = str(getattr(atom, "element", "") or atom.get_name()[:1]).strip().upper()
        if element == "H":
            continue
        coords.append(np.asarray(atom.coord, dtype=np.float64))
    return coords


def _chain_contact_metric_inputs(residues, *, atom_mode: str) -> tuple[np.ndarray, np.ndarray]:
    coords: list[np.ndarray] = []
    residue_indices: list[int] = []
    for res_idx, residue in enumerate(residues):
        residue_coords = _residue_contact_coordinates(residue, atom_mode=atom_mode)
        coords.extend(residue_coords)
        residue_indices.extend([res_idx] * len(residue_coords))
    if not coords:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=int)
    return np.asarray(coords, dtype=np.float64), np.asarray(residue_indices, dtype=int)


def _residue_pair_contact_count(
    residues_a,
    residues_b,
    *,
    cutoff: float,
    atom_mode: str,
) -> int:
    coords_a, residue_idx_a = _chain_contact_metric_inputs(residues_a, atom_mode=atom_mode)
    coords_b, residue_idx_b = _chain_contact_metric_inputs(residues_b, atom_mode=atom_mode)
    if coords_a.size == 0 or coords_b.size == 0:
        return 0
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    raw_pairs = tree_a.query_ball_tree(tree_b, r=float(cutoff))
    residue_pairs: set[tuple[int, int]] = set()
    for atom_idx_a, atom_indices_b in enumerate(raw_pairs):
        for atom_idx_b in atom_indices_b:
            residue_pairs.add((int(residue_idx_a[atom_idx_a]), int(residue_idx_b[atom_idx_b])))
    return len(residue_pairs)


def _minimum_interchain_atom_distance(
    residues_a,
    residues_b,
    *,
    atom_mode: str,
) -> float:
    coords_a, _ = _chain_contact_metric_inputs(residues_a, atom_mode=atom_mode)
    coords_b, _ = _chain_contact_metric_inputs(residues_b, atom_mode=atom_mode)
    if coords_a.size == 0 or coords_b.size == 0:
        return math.inf
    tree_a = cKDTree(coords_a)
    dists, _ = tree_a.query(coords_b, k=1)
    return float(np.min(np.asarray(dists, dtype=np.float64)))


def _dihedral_pair_contact_metrics(
    eval_result,
    oligo_name: str,
    *,
    order: int | None,
    contact_cutoff: float,
    atom_mode: str,
) -> tuple[list[int], list[float]]:
    structure = load_structure(Path(eval_result.complex_paths[oligo_name]))
    occurrences, _coord_sets, _fit_bundles, _movable_bundles = cyclic_oligomer_atom_bundles(
        structure,
        str(oligo_name),
    )
    _group, resolved_order = _normalize_point_group(
        "D",
        order,
        len(occurrences),
    )
    counts: list[int] = []
    min_distances: list[float] = []
    for idx in range(resolved_order):
        chain_a = occurrences[idx].chain_id
        chain_b = occurrences[idx + resolved_order].chain_id
        residues_a = protein_residues(structure, chain_a)
        residues_b = protein_residues(structure, chain_b)
        counts.append(
            _residue_pair_contact_count(
                residues_a,
                residues_b,
                cutoff=float(contact_cutoff),
                atom_mode=atom_mode,
            )
        )
        min_distances.append(
            _minimum_interchain_atom_distance(
                residues_a,
                residues_b,
                atom_mode=atom_mode,
            )
        )
    return counts, min_distances


def _single_repeat_protomer_name(oligo_name: str, included_protomer: str | None) -> str:
    if included_protomer:
        if included_protomer not in str(oligo_name):
            raise ValueError(
                f"Included protomer {included_protomer!r} is not present in oligomer {oligo_name!r}."
            )
        return str(included_protomer)
    unique = list(dict.fromkeys(str(oligo_name)))
    if len(unique) != 1:
        raise ValueError(
            f"Oligomer {oligo_name!r} contains multiple protomers; specify included_protomer explicitly."
        )
    return unique[0]


def _internal_repeat_coord_sets(
    eval_result, oligo_name: str, included_protomer: str | None
) -> list[np.ndarray]:
    proto = _single_repeat_protomer_name(oligo_name, included_protomer)
    chain_map = unique_protomer_chain_map(str(oligo_name))
    structure = load_structure(Path(eval_result.complex_paths[oligo_name]))
    residues = protein_residues(structure, chain_map[proto])
    groups = build_internal_repeat_residue_groups(proto)
    return [residue_ca_coordinates(residues, group) for group in groups]


def _pseudocycle_metrics(
    eval_result, oligo_name: str, included_protomer: str | None
) -> tuple[float, float]:
    coord_sets = _internal_repeat_coord_sets(eval_result, oligo_name, included_protomer)
    if len(coord_sets) < 2:
        return 0.0, 0.0
    target_rotation = 360.0 / float(len(coord_sets))
    rotation_errors: list[float] = []
    rise_errors: list[float] = []
    for idx in range(len(coord_sets)):
        left = coord_sets[idx]
        right = coord_sets[(idx + 1) % len(coord_sets)]
        R, t = kabsch_transform(left, right)
        axis = rotation_axis(R, t)
        rotation_errors.append(abs(rotation_angle_degrees(R) - target_rotation))
        rise_errors.append(abs(float(np.dot(np.asarray(t, dtype=float), axis))))
    return float(np.mean(rotation_errors)), float(np.mean(rise_errors))


@loss_registry.register(
    name="repeat_unit_rmsd_raw",
    optional_params=["included_protomer"],
    description="Mean pairwise CA RMSD between internally repeated units after Kabsch alignment.",
    supports_oligomer_specific=True,
)
def repeat_unit_rmsd_raw(eval_result, included_protomer=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        coord_sets = _internal_repeat_coord_sets(eval_result, oligo_name, included_protomer)
        scores[oligo_name] = mean_pairwise_aligned_rmsd(coord_sets)
    return scores


@loss_registry.register(
    name="repeat_unit_rmsd_norm",
    optional_params=["included_protomer", "best", "worst"],
    description="Normalized internal repeat quality in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def repeat_unit_rmsd_norm(eval_result, included_protomer=None, best=0.0, worst=1.0):
    raw = repeat_unit_rmsd_raw(eval_result, included_protomer=included_protomer)
    return {
        oligo_name: _normalize_bounded_good(value, best=best, worst=worst)
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="pseudocycle_rotation_raw",
    optional_params=["included_protomer"],
    description="Absolute deviation (degrees) from ideal 360/N repeat rotation for pseudocycles.",
    supports_oligomer_specific=True,
)
def pseudocycle_rotation_raw(eval_result, included_protomer=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        rotation_error, _ = _pseudocycle_metrics(eval_result, oligo_name, included_protomer)
        scores[oligo_name] = rotation_error
    return scores


@loss_registry.register(
    name="pseudocycle_rotation_norm",
    optional_params=["included_protomer", "best", "worst"],
    description="Normalized pseudocycle rotation quality in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def pseudocycle_rotation_norm(
    eval_result,
    included_protomer=None,
    best=0.0,
    worst=20.0,
):
    raw = pseudocycle_rotation_raw(eval_result, included_protomer=included_protomer)
    return {
        oligo_name: _normalize_bounded_good(value, best=best, worst=worst)
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="pseudocycle_rise_raw",
    optional_params=["included_protomer"],
    description="Absolute screw-axis rise (Angstrom) for pseudocycle closure.",
    supports_oligomer_specific=True,
)
def pseudocycle_rise_raw(eval_result, included_protomer=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        _, rise_error = _pseudocycle_metrics(eval_result, oligo_name, included_protomer)
        scores[oligo_name] = rise_error
    return scores


@loss_registry.register(
    name="pseudocycle_rise_norm",
    optional_params=["included_protomer", "best", "worst"],
    description="Normalized pseudocycle rise quality in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def pseudocycle_rise_norm(
    eval_result,
    included_protomer=None,
    best=0.0,
    worst=1.0,
):
    raw = pseudocycle_rise_raw(eval_result, included_protomer=included_protomer)
    return {
        oligo_name: _normalize_bounded_good(value, best=best, worst=worst)
        for oligo_name, value in raw.items()
    }


def _pseudocycle_closure_bad(
    rotation_error: float,
    rise_error: float,
    *,
    rotation_best: float,
    rotation_worst: float,
    rise_best: float,
    rise_worst: float,
) -> float:
    rot_bad = 1.0 - _normalize_bounded_good(
        rotation_error,
        best=rotation_best,
        worst=rotation_worst,
    )
    rise_bad = 1.0 - _normalize_bounded_good(
        rise_error,
        best=rise_best,
        worst=rise_worst,
    )
    return float(0.5 * (rot_bad + rise_bad))


@loss_registry.register(
    name="pseudocycle_closure_raw",
    optional_params=[
        "included_protomer",
        "rotation_best",
        "rotation_worst",
        "rise_best",
        "rise_worst",
    ],
    description="Paper-style pseudocycle closure score where 0 is best.",
    supports_oligomer_specific=True,
)
def pseudocycle_closure_raw(
    eval_result,
    included_protomer=None,
    rotation_best=0.0,
    rotation_worst=20.0,
    rise_best=0.0,
    rise_worst=1.0,
):
    rot = pseudocycle_rotation_raw(eval_result, included_protomer=included_protomer)
    rise = pseudocycle_rise_raw(eval_result, included_protomer=included_protomer)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        scores[oligo_name] = _pseudocycle_closure_bad(
            rot[oligo_name],
            rise[oligo_name],
            rotation_best=float(rotation_best),
            rotation_worst=float(rotation_worst),
            rise_best=float(rise_best),
            rise_worst=float(rise_worst),
        )
    return scores


@loss_registry.register(
    name="pseudocycle_closure_norm",
    optional_params=[
        "included_protomer",
        "rotation_best",
        "rotation_worst",
        "rise_best",
        "rise_worst",
    ],
    description="Normalized pseudocycle closure quality in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def pseudocycle_closure_norm(
    eval_result,
    included_protomer=None,
    rotation_best=0.0,
    rotation_worst=20.0,
    rise_best=0.0,
    rise_worst=1.0,
):
    raw = pseudocycle_closure_raw(
        eval_result,
        included_protomer=included_protomer,
        rotation_best=rotation_best,
        rotation_worst=rotation_worst,
        rise_best=rise_best,
        rise_worst=rise_worst,
    )
    return {oligo_name: float(1.0 - value) for oligo_name, value in raw.items()}


@loss_registry.register(
    name="point_group_orbit_rmsd_raw",
    optional_params=["point_group", "order"],
    description="Point-group orbit CA RMSD in Angstrom after fitting a Cn/Dn ideal operator layout.",
    supports_oligomer_specific=True,
)
def point_group_orbit_rmsd_raw(eval_result, point_group="C", order=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        orbit_rmsd, _ratio, _rotation = _point_group_orbit_metrics(
            _oligomer_coord_sets(eval_result, oligo_name),
            point_group=point_group,
            order=order,
        )
        scores[oligo_name] = orbit_rmsd
    return scores


@loss_registry.register(
    name="point_group_orbit_rmsd_ratio",
    optional_params=["point_group", "order"],
    description="Point-group orbit CA RMSD divided by assembly radius for Cn/Dn assemblies.",
    supports_oligomer_specific=True,
)
def point_group_orbit_rmsd_ratio(eval_result, point_group="C", order=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        _orbit_rmsd, ratio, _rotation = _point_group_orbit_metrics(
            _oligomer_coord_sets(eval_result, oligo_name),
            point_group=point_group,
            order=order,
        )
        scores[oligo_name] = ratio
    return scores


@loss_registry.register(
    name="point_group_rotation_raw",
    optional_params=["point_group", "order"],
    description="Mean protomer orientation residual in degrees against fitted Cn/Dn operators.",
    supports_oligomer_specific=True,
)
def point_group_rotation_raw(eval_result, point_group="C", order=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        _orbit_rmsd, _ratio, rotation = _point_group_orbit_metrics(
            _oligomer_coord_sets(eval_result, oligo_name),
            point_group=point_group,
            order=order,
        )
        scores[oligo_name] = rotation
    return scores


@loss_registry.register(
    name="point_group_symmetry_norm",
    optional_params=[
        "point_group",
        "order",
        "orbit_best",
        "orbit_worst",
        "rotation_best",
        "rotation_worst",
        "rotation_weight",
    ],
    description="Combined Cn/Dn point-group quality in [0,1] from orbit RMSD ratio and orientation residual.",
    supports_oligomer_specific=True,
)
def point_group_symmetry_norm(
    eval_result,
    point_group="C",
    order=None,
    orbit_best=0.0,
    orbit_worst=0.20,
    rotation_best=0.0,
    rotation_worst=25.0,
    rotation_weight=0.5,
):
    weight = float(rotation_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("rotation_weight must be in [0,1].")
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        _orbit_rmsd, ratio, rotation = _point_group_orbit_metrics(
            _oligomer_coord_sets(eval_result, oligo_name),
            point_group=point_group,
            order=order,
        )
        orbit_good = _normalize_bounded_good(
            ratio,
            best=float(orbit_best),
            worst=float(orbit_worst),
        )
        rotation_good = _normalize_bounded_good(
            rotation,
            best=float(rotation_best),
            worst=float(rotation_worst),
        )
        scores[oligo_name] = float((1.0 - weight) * orbit_good + weight * rotation_good)
    return scores


@loss_registry.register(
    name="ananas_rmsd_raw",
    optional_params=["point_group", "order", "ananas_path", "timeout_s", "missing_rmsd"],
    description="AnAnaS symmetry RMSD for a requested Cn/Dn point group.",
    supports_oligomer_specific=True,
)
def ananas_rmsd_raw(
    eval_result,
    point_group="C",
    order=None,
    ananas_path=None,
    timeout_s=30.0,
    missing_rmsd=math.inf,
):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        symmetry_group = _ananas_group_arg(
            point_group,
            order,
            len(str(oligo_name)),
        )
        scores[oligo_name] = ananas_rmsd_wrapper(
            eval_result.complex_paths[oligo_name],
            symmetry_group,
            ananas_path=ananas_path,
            timeout_s=float(timeout_s),
            missing_rmsd=missing_rmsd,
        )
    return scores


@loss_registry.register(
    name="ananas_rmsd_norm",
    optional_params=["point_group", "order", "ananas_path", "timeout_s", "best", "worst", "missing_rmsd"],
    description="Normalized AnAnaS symmetry RMSD in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def ananas_rmsd_norm(
    eval_result,
    point_group="C",
    order=None,
    ananas_path=None,
    timeout_s=30.0,
    best=0.0,
    worst=2.0,
    missing_rmsd=math.inf,
):
    raw = ananas_rmsd_raw(
        eval_result,
        point_group=point_group,
        order=order,
        ananas_path=ananas_path,
        timeout_s=timeout_s,
        missing_rmsd=missing_rmsd,
    )
    return {
        oligo_name: _normalize_bounded_good(value, best=best, worst=worst)
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="dihedral_ring_sep_cv_raw",
    optional_params=["order"],
    description="Dn adjacent-ring COM distance coefficient of variation.",
    supports_oligomer_specific=True,
)
def dihedral_ring_sep_cv_raw(eval_result, order=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        coord_sets = _oligomer_coord_sets(eval_result, oligo_name)
        group, resolved_order, canonical_sets, *_ = _point_group_projection(
            coord_sets,
            point_group="D",
            order=order,
        )
        if group != "D":  # pragma: no cover
            raise AssertionError("Expected D point group.")
        centers = _chain_centers(canonical_sets)
        distances: list[float] = []
        for offset in (0, resolved_order):
            ring = centers[offset : offset + resolved_order]
            for idx in range(resolved_order):
                distances.append(float(np.linalg.norm(ring[(idx + 1) % resolved_order] - ring[idx])))
        scores[oligo_name] = _coefficient_of_variation(distances)
    return scores


@loss_registry.register(
    name="dihedral_pair_sep_cv_raw",
    optional_params=["order"],
    description="Dn C2-paired COM distance coefficient of variation.",
    supports_oligomer_specific=True,
)
def dihedral_pair_sep_cv_raw(eval_result, order=None):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        coord_sets = _oligomer_coord_sets(eval_result, oligo_name)
        _group, resolved_order, canonical_sets, *_ = _point_group_projection(
            coord_sets,
            point_group="D",
            order=order,
        )
        centers = _chain_centers(canonical_sets)
        distances = [
            float(np.linalg.norm(centers[idx + resolved_order] - centers[idx]))
            for idx in range(resolved_order)
        ]
        scores[oligo_name] = _coefficient_of_variation(distances)
    return scores


@loss_registry.register(
    name="dihedral_pair_contact_count_raw",
    optional_params=["order", "contact_cutoff", "atom_mode", "reduction"],
    description="Residue-pair contact count across Dn C2-paired protomers on evaluator-predicted structures.",
    supports_oligomer_specific=True,
)
def dihedral_pair_contact_count_raw(
    eval_result,
    order=None,
    contact_cutoff=5.0,
    atom_mode="heavy",
    reduction="mean",
):
    scores: dict[str, float] = {}
    reduce_mode = str(reduction or "mean").strip().lower()
    for oligo_name in eval_result.names:
        counts, _min_distances = _dihedral_pair_contact_metrics(
            eval_result,
            oligo_name,
            order=order,
            contact_cutoff=float(contact_cutoff),
            atom_mode=atom_mode,
        )
        if not counts:
            scores[oligo_name] = 0.0
        elif reduce_mode == "mean":
            scores[oligo_name] = float(np.mean(np.asarray(counts, dtype=np.float64)))
        elif reduce_mode in {"min", "minimum"}:
            scores[oligo_name] = float(np.min(np.asarray(counts, dtype=np.float64)))
        elif reduce_mode == "sum":
            scores[oligo_name] = float(np.sum(np.asarray(counts, dtype=np.float64)))
        else:
            raise ValueError("reduction must be one of mean, min, or sum.")
    return scores


@loss_registry.register(
    name="dihedral_pair_contact_count_norm",
    optional_params=["order", "contact_cutoff", "atom_mode", "target", "reduction"],
    description="Normalized Dn C2-pair contact count in [0,1]; higher rewards paired-chain contact.",
    supports_oligomer_specific=True,
)
def dihedral_pair_contact_count_norm(
    eval_result,
    order=None,
    contact_cutoff=5.0,
    atom_mode="heavy",
    target=12.0,
    reduction="mean",
):
    target = float(target)
    if target <= 0.0:
        raise ValueError("target must be > 0.")
    raw = dihedral_pair_contact_count_raw(
        eval_result,
        order=order,
        contact_cutoff=contact_cutoff,
        atom_mode=atom_mode,
        reduction=reduction,
    )
    return {
        oligo_name: float(min(max(value / target, 0.0), 1.0))
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="dihedral_pair_contact_fraction_raw",
    optional_params=["order", "contact_cutoff", "atom_mode", "min_contacts"],
    description="Fraction of Dn C2-paired protomer pairs with at least min_contacts residue contacts.",
    supports_oligomer_specific=True,
)
def dihedral_pair_contact_fraction_raw(
    eval_result,
    order=None,
    contact_cutoff=5.0,
    atom_mode="heavy",
    min_contacts=1,
):
    threshold = int(min_contacts)
    if threshold < 1:
        raise ValueError("min_contacts must be >= 1.")
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        counts, _min_distances = _dihedral_pair_contact_metrics(
            eval_result,
            oligo_name,
            order=order,
            contact_cutoff=float(contact_cutoff),
            atom_mode=atom_mode,
        )
        if not counts:
            scores[oligo_name] = 0.0
            continue
        paired = sum(1 for value in counts if int(value) >= threshold)
        scores[oligo_name] = float(paired / len(counts))
    return scores


@loss_registry.register(
    name="dihedral_pair_contact_fraction_norm",
    optional_params=["order", "contact_cutoff", "atom_mode", "min_contacts", "target"],
    description="Normalized Dn C2-pair interface coverage in [0,1]; higher means more paired interfaces are present.",
    supports_oligomer_specific=True,
)
def dihedral_pair_contact_fraction_norm(
    eval_result,
    order=None,
    contact_cutoff=5.0,
    atom_mode="heavy",
    min_contacts=1,
    target=1.0,
):
    target = float(target)
    if target <= 0.0:
        raise ValueError("target must be > 0.")
    raw = dihedral_pair_contact_fraction_raw(
        eval_result,
        order=order,
        contact_cutoff=contact_cutoff,
        atom_mode=atom_mode,
        min_contacts=min_contacts,
    )
    return {
        oligo_name: float(min(max(value / target, 0.0), 1.0))
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="dihedral_pair_min_distance_raw",
    optional_params=["order", "atom_mode", "reduction"],
    description="Minimum atom distance across Dn C2-paired protomers on evaluator-predicted structures.",
    supports_oligomer_specific=True,
)
def dihedral_pair_min_distance_raw(
    eval_result,
    order=None,
    atom_mode="heavy",
    reduction="mean",
):
    scores: dict[str, float] = {}
    reduce_mode = str(reduction or "mean").strip().lower()
    for oligo_name in eval_result.names:
        _counts, min_distances = _dihedral_pair_contact_metrics(
            eval_result,
            oligo_name,
            order=order,
            contact_cutoff=5.0,
            atom_mode=atom_mode,
        )
        finite = [float(v) for v in min_distances if math.isfinite(float(v))]
        if not finite:
            scores[oligo_name] = math.inf
        elif reduce_mode == "mean":
            scores[oligo_name] = float(np.mean(np.asarray(finite, dtype=np.float64)))
        elif reduce_mode in {"max", "worst"}:
            scores[oligo_name] = float(np.max(np.asarray(finite, dtype=np.float64)))
        elif reduce_mode in {"min", "best"}:
            scores[oligo_name] = float(np.min(np.asarray(finite, dtype=np.float64)))
        else:
            raise ValueError("reduction must be one of mean, max, or min.")
    return scores


@loss_registry.register(
    name="dihedral_pair_min_distance_norm",
    optional_params=["order", "atom_mode", "best", "worst", "reduction"],
    description="Normalized Dn C2-pair atom-distance compactness in [0,1]; higher is closer contact.",
    supports_oligomer_specific=True,
)
def dihedral_pair_min_distance_norm(
    eval_result,
    order=None,
    atom_mode="heavy",
    best=3.5,
    worst=14.0,
    reduction="mean",
):
    raw = dihedral_pair_min_distance_raw(
        eval_result,
        order=order,
        atom_mode=atom_mode,
        reduction=reduction,
    )
    return {
        oligo_name: _normalize_bounded_good(value, best=float(best), worst=float(worst))
        for oligo_name, value in raw.items()
    }


@loss_registry.register(
    name="protomer_rmsd_raw",
    description="Mean pairwise CA RMSD across heterooligomer shared-core regions after Kabsch alignment.",
    supports_oligomer_specific=True,
)
def protomer_rmsd_raw(eval_result):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        groups = build_shared_core_residue_groups(str(oligo_name))
        chain_map = unique_protomer_chain_map(str(oligo_name))
        structure = load_structure(Path(eval_result.complex_paths[oligo_name]))
        coord_sets: list[np.ndarray] = []
        for proto, residue_indices in groups.items():
            residues = protein_residues(structure, chain_map[proto])
            coord_sets.append(residue_ca_coordinates(residues, residue_indices))
        scores[oligo_name] = mean_pairwise_aligned_rmsd(coord_sets)
    return scores


@loss_registry.register(
    name="protomer_rmsd_norm",
    optional_params=["m", "s", "best", "worst"],
    description="Normalized shared-core protomer agreement in [0,1]; higher is better.",
    supports_oligomer_specific=True,
)
def protomer_rmsd_norm(eval_result, m=2.0, s=2.0, best=None, worst=None):
    raw = protomer_rmsd_raw(eval_result)
    if best is None and worst is None:
        return {
            oligo_name: _normalize_good(value, m=float(m), s=float(s))
            for oligo_name, value in raw.items()
        }
    if best is None or worst is None:
        raise ValueError("protomer_rmsd_norm requires both `best` and `worst` when using bounded normalization.")
    return {
        oligo_name: _normalize_bounded_good(value, best=best, worst=worst)
        for oligo_name, value in raw.items()
    }
