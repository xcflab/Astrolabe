from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from astrolabe.losses.base import loss_registry


DEFAULT_OCCLUSION_TARGET = 0.35
DEFAULT_GRID_SPACING = 0.75
DEFAULT_ATOM_RADIUS = 1.7
DEFAULT_PORE_RADIUS_PERCENTILE = 5.0
DEFAULT_PORE_RADIUS_MIN = 6.0
DEFAULT_PORE_RADIUS_MAX = 14.0
DEFAULT_MOUTH_FRACTION = 0.5
DEFAULT_TANGENT_SCALE = 2.0


def _parser_for_path(path: Path):
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _polymer_residues(chain):
    return [residue for residue in chain.get_residues() if str(residue.id[0]).strip() == ""]


def _heavy_atom_coords(residues) -> np.ndarray:
    coords = []
    for residue in residues:
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H" or (not element and atom.get_name().strip().upper().startswith("H")):
                continue
            coords.append(np.asarray(atom.coord, dtype=float))
    if not coords:
        raise ValueError("No heavy atoms found.")
    return np.stack(coords, axis=0)


def _ca_coords(residues) -> np.ndarray:
    coords = [
        np.asarray(residue["CA"].coord, dtype=float)
        for residue in residues
        if "CA" in residue
    ]
    if not coords:
        raise ValueError("No CA atoms found.")
    return np.stack(coords, axis=0)


def _normalize_chain_ids(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(token for token in value.replace(",", " ").split() if token)
    return tuple(str(token) for token in value if str(token).strip())


def _pore_frame(target_ca_by_chain: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_target_ca = np.concatenate(target_ca_by_chain, axis=0)
    center = np.mean(all_target_ca, axis=0)
    chain_centers = np.stack(
        [np.mean(coords, axis=0) for coords in target_ca_by_chain],
        axis=0,
    )
    if len(chain_centers) >= 3:
        centered = chain_centers - np.mean(chain_centers, axis=0)
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, 0]
    else:
        centered = all_target_ca - center
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, -1]
    axis = axis / np.linalg.norm(axis)
    major_component = int(np.argmax(np.abs(axis)))
    if axis[major_component] < 0:
        axis = -axis
    return center, axis


def _radial_and_axial(
    coords: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relative = coords - center
    axial = relative @ axis
    radial_vectors = relative - np.outer(axial, axis)
    return np.linalg.norm(radial_vectors, axis=1), axial


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, axis))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    first = trial - float(np.dot(trial, axis)) * axis
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def _occlusion_ratio(
    binder_coords: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    grid_spacing: float,
    atom_radius: float,
) -> float:
    first, second = _perpendicular_basis(axis)
    values = np.arange(-radius, radius + grid_spacing * 0.5, grid_spacing)
    grid = np.array(
        [
            (x, y)
            for x in values
            for y in values
            if x * x + y * y <= radius * radius
        ],
        dtype=float,
    )
    relative = binder_coords - center
    projected = np.column_stack((relative @ first, relative @ second))
    squared_distances = np.sum(
        (grid[:, None, :] - projected[None, :, :]) ** 2,
        axis=2,
    )
    return float(np.mean(np.any(squared_distances <= atom_radius**2, axis=1)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def compute_peptide_plug_metrics_for_structure(
    structure_path: str | Path,
    *,
    binder_chain: str,
    target_chains: str | Sequence[str] | None = None,
    occlusion_target: float = DEFAULT_OCCLUSION_TARGET,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    atom_radius: float = DEFAULT_ATOM_RADIUS,
    pore_radius_percentile: float = DEFAULT_PORE_RADIUS_PERCENTILE,
    pore_radius_min: float = DEFAULT_PORE_RADIUS_MIN,
    pore_radius_max: float = DEFAULT_PORE_RADIUS_MAX,
    mouth_fraction: float = DEFAULT_MOUTH_FRACTION,
    tangent_scale: float = DEFAULT_TANGENT_SCALE,
) -> dict[str, float]:
    path = Path(structure_path).expanduser().resolve()
    structure = _parser_for_path(path).get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chains = {str(chain.id): chain for chain in model.get_chains()}
    if binder_chain not in chains:
        raise ValueError(f"Binder chain {binder_chain!r} is missing from {path}.")

    target_chain_ids = _normalize_chain_ids(target_chains)
    if not target_chain_ids:
        target_chain_ids = tuple(chain_id for chain_id in chains if chain_id != binder_chain)
    missing = [chain_id for chain_id in target_chain_ids if chain_id not in chains]
    if missing:
        raise ValueError(f"Target chains {missing!r} are missing from {path}.")
    if len(target_chain_ids) < 2:
        raise ValueError("At least two target chains are required to define a pore frame.")

    binder_residues = _polymer_residues(chains[binder_chain])
    if len(binder_residues) < 3:
        raise ValueError("Peptide plug metrics require at least three binder residues.")
    target_residues_by_chain = [
        _polymer_residues(chains[chain_id]) for chain_id in target_chain_ids
    ]
    binder_heavy = _heavy_atom_coords(binder_residues)
    binder_ca = _ca_coords(binder_residues)
    target_heavy = np.concatenate(
        [_heavy_atom_coords(residues) for residues in target_residues_by_chain],
        axis=0,
    )
    target_ca_by_chain = [_ca_coords(residues) for residues in target_residues_by_chain]

    center, axis = _pore_frame(target_ca_by_chain)
    target_radial, target_axial = _radial_and_axial(target_heavy, center, axis)
    pore_radius = float(
        np.clip(
            np.percentile(target_radial, float(pore_radius_percentile)),
            float(pore_radius_min),
            float(pore_radius_max),
        )
    )
    lower_axial, lower_mouth, upper_mouth, upper_axial = np.percentile(
        target_axial,
        [5.0, 10.0, 90.0, 95.0],
    )

    binder_radial, binder_axial = _radial_and_axial(binder_heavy, center, axis)
    within_channel = (binder_axial >= lower_axial) & (binder_axial <= upper_axial)
    if np.any(within_channel):
        pore_occlusion = _occlusion_ratio(
            binder_heavy[within_channel],
            center,
            axis,
            pore_radius,
            grid_spacing=float(grid_spacing),
            atom_radius=float(atom_radius),
        )
    else:
        pore_occlusion = 0.0
    pore_occlusion_score = float(
        np.clip(pore_occlusion / max(float(occlusion_target), 1e-6), 0.0, 1.0)
    )

    binder_ca_radial, binder_ca_axial = _radial_and_axial(binder_ca, center, axis)
    terminal_n_axial = float(binder_ca_axial[0])
    terminal_c_axial = float(binder_ca_axial[-1])
    terminal_same_side = float(terminal_n_axial * terminal_c_axial > 0.0)
    side_sign = 1.0 if terminal_n_axial + terminal_c_axial >= 0.0 else -1.0
    outward_axis = side_sign * axis
    terminal_n_outward = float(np.dot(binder_ca[0] - binder_ca[1], outward_axis))
    terminal_c_outward = float(np.dot(binder_ca[-1] - binder_ca[-2], outward_axis))
    terminal_outward_score = 0.5 * (
        _sigmoid(terminal_n_outward / max(float(tangent_scale), 1e-6))
        + _sigmoid(terminal_c_outward / max(float(tangent_scale), 1e-6))
    )
    mouth_depth = max(
        float(mouth_fraction) * min(abs(float(lower_mouth)), abs(float(upper_mouth))),
        1e-6,
    )
    terminal_mouth_score = float(
        np.clip(
            min(abs(terminal_n_axial), abs(terminal_c_axial)) / mouth_depth,
            0.0,
            1.0,
        )
    )
    scaffoldability = (
        terminal_same_side * terminal_mouth_score * terminal_outward_score
    )
    plug_score = pore_occlusion_score * scaffoldability

    return {
        "pore_occlusion": pore_occlusion,
        "pore_occlusion_score": pore_occlusion_score,
        "terminal_same_side": terminal_same_side,
        "terminal_mouth_score": terminal_mouth_score,
        "terminal_outward_score": terminal_outward_score,
        "scaffoldability": scaffoldability,
        "plug_score": plug_score,
        "terminal_distance": float(np.linalg.norm(binder_ca[0] - binder_ca[-1])),
        "terminal_n_axial": terminal_n_axial,
        "terminal_c_axial": terminal_c_axial,
        "terminal_n_radial": float(binder_ca_radial[0]),
        "terminal_c_radial": float(binder_ca_radial[-1]),
        "terminal_n_outward": terminal_n_outward,
        "terminal_c_outward": terminal_c_outward,
        "peptide_axial_min": float(np.min(binder_ca_axial)),
        "peptide_axial_max": float(np.max(binder_ca_axial)),
        "peptide_min_radial": float(np.min(binder_radial)),
        "pore_radius": pore_radius,
    }


def _file_signature(path_like: str | Path) -> tuple[str, int, int]:
    path = Path(path_like).expanduser().resolve()
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


@functools.lru_cache(maxsize=4096)
def _compute_cached(
    path_signature: tuple[str, int, int],
    binder_chain: str,
    target_chains: tuple[str, ...],
    occlusion_target: float,
    grid_spacing: float,
    atom_radius: float,
    pore_radius_percentile: float,
    pore_radius_min: float,
    pore_radius_max: float,
    mouth_fraction: float,
    tangent_scale: float,
) -> dict[str, float]:
    return compute_peptide_plug_metrics_for_structure(
        path_signature[0],
        binder_chain=binder_chain,
        target_chains=target_chains,
        occlusion_target=occlusion_target,
        grid_spacing=grid_spacing,
        atom_radius=atom_radius,
        pore_radius_percentile=pore_radius_percentile,
        pore_radius_min=pore_radius_min,
        pore_radius_max=pore_radius_max,
        mouth_fraction=mouth_fraction,
        tangent_scale=tangent_scale,
    )


_OPTIONAL_PARAMS = [
    "binder_chain",
    "target_chains",
    "occlusion_target",
    "grid_spacing",
    "atom_radius",
    "pore_radius_percentile",
    "pore_radius_min",
    "pore_radius_max",
    "mouth_fraction",
    "tangent_scale",
]


def _metrics_by_oligo(
    eval_result,
    *,
    binder_chain: str = "A",
    target_chains: str | Sequence[str] | None = None,
    occlusion_target: float = DEFAULT_OCCLUSION_TARGET,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    atom_radius: float = DEFAULT_ATOM_RADIUS,
    pore_radius_percentile: float = DEFAULT_PORE_RADIUS_PERCENTILE,
    pore_radius_min: float = DEFAULT_PORE_RADIUS_MIN,
    pore_radius_max: float = DEFAULT_PORE_RADIUS_MAX,
    mouth_fraction: float = DEFAULT_MOUTH_FRACTION,
    tangent_scale: float = DEFAULT_TANGENT_SCALE,
) -> dict[str, dict[str, float]]:
    target_chain_ids = _normalize_chain_ids(target_chains)
    return {
        str(oligo_name): _compute_cached(
            _file_signature(path_like),
            str(binder_chain),
            target_chain_ids,
            float(occlusion_target),
            float(grid_spacing),
            float(atom_radius),
            float(pore_radius_percentile),
            float(pore_radius_min),
            float(pore_radius_max),
            float(mouth_fraction),
            float(tangent_scale),
        )
        for oligo_name, path_like in (getattr(eval_result, "complex_paths", {}) or {}).items()
    }


def _metric_loss(metric_name: str):
    def loss(eval_result, **kwargs):
        return {
            oligo_name: metrics[metric_name]
            for oligo_name, metrics in _metrics_by_oligo(eval_result, **kwargs).items()
        }

    return loss


for _loss_name, _metric_name, _description in [
    (
        "peptide_pore_occlusion",
        "pore_occlusion",
        "Fraction of the target pore cross-section occupied by peptide heavy atoms.",
    ),
    (
        "peptide_pore_occlusion_score",
        "pore_occlusion_score",
        "Peptide pore occlusion normalized to a configurable target.",
    ),
    (
        "peptide_terminal_same_side",
        "terminal_same_side",
        "Binary score requiring peptide termini to lie on the same side of the pore center.",
    ),
    (
        "peptide_terminal_mouth_score",
        "terminal_mouth_score",
        "Score for keeping both peptide termini in a target pore-mouth half-space.",
    ),
    (
        "peptide_terminal_outward_score",
        "terminal_outward_score",
        "Score for peptide terminal tangents pointing toward the shared pore-mouth side.",
    ),
    (
        "peptide_scaffoldability",
        "scaffoldability",
        "Same-side, mouth-accessible, outward-terminal peptide scaffoldability score.",
    ),
    (
        "peptide_plug_score",
        "plug_score",
        "Combined peptide pore-occlusion and scaffoldability score.",
    ),
    (
        "peptide_terminal_distance",
        "terminal_distance",
        "Distance between peptide terminal CA atoms.",
    ),
]:
    _registered_loss = loss_registry.register(
        name=_loss_name,
        optional_params=_OPTIONAL_PARAMS,
        description=_description,
        supports_oligomer_specific=True,
    )(_metric_loss(_metric_name))
    _registered_loss.__name__ = _loss_name
    globals()[_loss_name] = _registered_loss
