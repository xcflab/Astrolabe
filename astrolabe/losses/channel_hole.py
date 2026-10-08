from __future__ import annotations

import csv
import functools
import os
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser

from astrolabe.core import global_config
from astrolabe.core.repeat_symmetry import (
    apply_transform,
    canonicalize_exact_cyclic_cn,
    cyclic_oligomer_atom_bundles,
    kabsch_transform,
)
from astrolabe.losses.base import loss_registry


DEFAULT_EFFECTIVE_RADIUS_CUTOFF = 4.0
DEFAULT_END_RADIUS = 5.0
DEFAULT_LATERAL_STEP = 2.0
DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD = 40.0
DEFAULT_SOFTPLUS_RADIUS_TARGET = 1.2
DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE = 0.1
FRAME_SOURCE_REFERENCE_SPHERES = "reference_spheres"
FRAME_SOURCE_EXACT_CYCLIC = "exact_cyclic"
_CHANNEL_STEP_DIR_RE = re.compile(r"^step_(\d{3})_(.+)$")
_PARENT_SCAN_ID_RE = re.compile(r"(x\d+_y\d+)")


@dataclass(frozen=True)
class ChannelScanInput:
    scan_id: str
    parent_scan_id: str
    center: tuple[float, float, float]
    axis: tuple[float, float, float]
    ix: int
    iy: int


@dataclass(frozen=True)
class PreparedChannelScanInputs:
    aligned_structure_path: pathlib.Path
    scans: tuple[ChannelScanInput, ...]


def _as_path(path_like: str | pathlib.Path) -> pathlib.Path:
    if isinstance(path_like, pathlib.Path):
        return path_like
    return pathlib.Path(str(path_like))


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return arr / norm


def _parser_for_path(path: pathlib.Path):
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _load_structure(path_like: str | pathlib.Path):
    path = _as_path(path_like)
    parser = _parser_for_path(path)
    return parser.get_structure(path.stem, str(path))


def _polymer_residues(chain) -> list[Any]:
    return [res for res in chain.get_residues() if str(res.id[0]).strip() == ""]


def _resolve_alignment_chain_ids(
    reference_model,
    structure_model,
    alignment_chains: str | Sequence[str] | None,
) -> list[str]:
    if alignment_chains is None:
        ref_ids = [chain.id for chain in reference_model.get_chains()]
        structure_ids = {chain.id for chain in structure_model.get_chains()}
        chain_ids = [chain_id for chain_id in ref_ids if chain_id in structure_ids]
    elif isinstance(alignment_chains, str):
        chain_ids = [token for token in re.split(r"[,\s]+", alignment_chains) if token]
    else:
        chain_ids = [str(token) for token in alignment_chains if str(token).strip()]
    if not chain_ids:
        raise ValueError("No shared chains available for channel_hole alignment.")
    return chain_ids


def _collect_alignment_ca_coords(
    *,
    structure_path: pathlib.Path,
    reference_structure_path: pathlib.Path,
    alignment_chains: str | Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    structure = _load_structure(structure_path)
    reference = _load_structure(reference_structure_path)
    structure_model = next(structure.get_models())
    reference_model = next(reference.get_models())
    chain_ids = _resolve_alignment_chain_ids(
        reference_model,
        structure_model,
        alignment_chains,
    )

    structure_coords: list[np.ndarray] = []
    reference_coords: list[np.ndarray] = []
    for chain_id in chain_ids:
        try:
            structure_chain = structure_model[chain_id]
            reference_chain = reference_model[chain_id]
        except KeyError as exc:
            raise ValueError(
                f"Alignment chain {chain_id!r} is missing from one of the structures."
            ) from exc
        structure_residues = _polymer_residues(structure_chain)
        reference_residues = _polymer_residues(reference_chain)
        if len(structure_residues) != len(reference_residues):
            raise ValueError(
                f"Alignment chain {chain_id!r} residue count mismatch: "
                f"{len(structure_residues)} vs {len(reference_residues)}."
            )
        for structure_residue, reference_residue in zip(structure_residues, reference_residues):
            if "CA" not in structure_residue or "CA" not in reference_residue:
                continue
            structure_coords.append(np.asarray(structure_residue["CA"].coord, dtype=float))
            reference_coords.append(np.asarray(reference_residue["CA"].coord, dtype=float))
    if not structure_coords or len(structure_coords) != len(reference_coords):
        raise ValueError("No matched CA coordinates available for channel_hole alignment.")
    return np.stack(structure_coords, axis=0), np.stack(reference_coords, axis=0)


def _write_aligned_structure_as_pdb(
    *,
    structure_path: pathlib.Path,
    reference_structure_path: pathlib.Path,
    output_path: pathlib.Path,
    alignment_chains: str | Sequence[str] | None,
) -> pathlib.Path:
    structure = _load_structure(structure_path)
    structure_coords, reference_coords = _collect_alignment_ca_coords(
        structure_path=structure_path,
        reference_structure_path=reference_structure_path,
        alignment_chains=alignment_chains,
    )
    rotation, translation = kabsch_transform(structure_coords, reference_coords)

    for atom in structure.get_atoms():
        coord = np.asarray(atom.coord, dtype=float)
        atom.coord = apply_transform(coord[None, :], rotation, translation)[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    return output_path


def _write_structure_as_pdb(
    *,
    structure_path: pathlib.Path,
    output_path: pathlib.Path,
) -> pathlib.Path:
    structure = _load_structure(structure_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    return output_path


def _load_reference_spheres(reference_spheres_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    coords: list[list[float]] = []
    radii: list[float] = []
    for line in reference_spheres_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        coords.append(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
        )
        radii.append(float(line[54:60]))
    if not coords:
        raise ValueError(f"No sphere records found in {reference_spheres_path}")
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float)


def _lateral_axes_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize_vector(axis)
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    seed = min(candidates, key=lambda cand: abs(float(np.dot(cand, axis))))
    east = seed - float(np.dot(seed, axis)) * axis
    east = _normalize_vector(east)
    north = _normalize_vector(np.cross(axis, east))
    return east, north


def _parent_scan_id_from_reference_spheres(reference_spheres_path: pathlib.Path) -> str:
    stem = reference_spheres_path.stem
    m = _PARENT_SCAN_ID_RE.search(stem)
    if m:
        return m.group(1)
    if stem.endswith("_spheres"):
        return stem[: -len("_spheres")]
    return stem


def _reference_frame_from_spheres(
    reference_spheres_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, radii = _load_reference_spheres(reference_spheres_path)
    axis = _normalize_vector(centers[-1] - centers[0])
    if float(axis[2]) < 0.0:
        axis = -axis
    center = centers[int(np.argmin(radii))]
    east, north = _lateral_axes_from_axis(axis)
    return center, axis, east, north


def _normalize_frame_source(frame_source: str | None) -> str:
    value = str(frame_source or FRAME_SOURCE_REFERENCE_SPHERES).strip().lower()
    if value in {"reference", "spheres", "reference_sphere"}:
        return FRAME_SOURCE_REFERENCE_SPHERES
    if value in {"cyclic", "exact_cn", "exact-cyclic"}:
        return FRAME_SOURCE_EXACT_CYCLIC
    if value not in {FRAME_SOURCE_REFERENCE_SPHERES, FRAME_SOURCE_EXACT_CYCLIC}:
        raise ValueError(
            "frame_source must be 'reference_spheres' or 'exact_cyclic', "
            f"got {frame_source!r}."
        )
    return value


def _exact_cyclic_frame_from_structure(
    *,
    structure_path: pathlib.Path,
    oligo_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = _load_structure(structure_path)
    _occurrences, coord_sets, _fit_bundles, _movable_bundles = (
        cyclic_oligomer_atom_bundles(structure, oligo_name)
    )
    canonicalization = canonicalize_exact_cyclic_cn(coord_sets)
    axis = _normalize_vector(canonicalization.axis_direction)
    if float(axis[2]) < 0.0:
        axis = -axis

    all_coords = np.concatenate(
        [np.asarray(coords, dtype=float) for coords in coord_sets],
        axis=0,
    )
    projections = (all_coords - canonicalization.axis_point[None, :]) @ axis
    center_s = 0.5 * (float(np.min(projections)) + float(np.max(projections)))
    center = canonicalization.axis_point + center_s * axis
    east, north = _lateral_axes_from_axis(axis)
    return center, axis, east, north


def build_channel_scan_inputs(
    *,
    structure_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    reference_structure_path: str | pathlib.Path | None = None,
    reference_spheres_path: str | pathlib.Path | None = None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    oligo_name: str | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    alignment_chains: str | Sequence[str] | None = None,
) -> PreparedChannelScanInputs:
    structure_path = _as_path(structure_path).expanduser().resolve()
    output_dir = _as_path(output_dir).expanduser().resolve()
    frame_source_norm = _normalize_frame_source(frame_source)

    if frame_source_norm == FRAME_SOURCE_REFERENCE_SPHERES:
        if reference_structure_path is None or reference_spheres_path is None:
            raise ValueError(
                "reference_spheres frame_source requires reference_structure_path "
                "and reference_spheres_path."
            )
        reference_structure_path = _as_path(reference_structure_path).expanduser().resolve()
        reference_spheres_path = _as_path(reference_spheres_path).expanduser().resolve()
        aligned_structure_path = output_dir / f"{structure_path.stem}__channel_hole_aligned.pdb"
        _write_aligned_structure_as_pdb(
            structure_path=structure_path,
            reference_structure_path=reference_structure_path,
            output_path=aligned_structure_path,
            alignment_chains=alignment_chains,
        )
        center, axis, east, north = _reference_frame_from_spheres(reference_spheres_path)
        parent_scan_id = _parent_scan_id_from_reference_spheres(reference_spheres_path)
    else:
        if not oligo_name:
            raise ValueError("exact_cyclic frame_source requires oligo_name.")
        aligned_structure_path = output_dir / f"{structure_path.stem}__channel_hole_exact_cyclic.pdb"
        _write_structure_as_pdb(
            structure_path=structure_path,
            output_path=aligned_structure_path,
        )
        center, axis, east, north = _exact_cyclic_frame_from_structure(
            structure_path=structure_path,
            oligo_name=str(oligo_name),
        )
        parent_scan_id = f"{oligo_name}__exact_cyclic"

    scans: list[ChannelScanInput] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            scan_center = center + float(dx) * lateral_step * east + float(dy) * lateral_step * north
            scans.append(
                ChannelScanInput(
                    scan_id=f"{parent_scan_id}__dx{dx:+d}_dy{dy:+d}",
                    parent_scan_id=parent_scan_id,
                    ix=dx + 3,
                    iy=dy + 4,
                    center=tuple(float(x) for x in scan_center),
                    axis=tuple(float(x) for x in axis),
                )
            )
    return PreparedChannelScanInputs(
        aligned_structure_path=aligned_structure_path,
        scans=tuple(scans),
    )


def radius_score_from_min_radius(min_radius: float) -> float:
    value = max(0.0, float(min_radius))
    first = np.clip(value / 1.5, 0.0, 1.0) ** 2
    second = np.clip((value - 1.5) / 0.3, 0.0, 1.0)
    return float(0.9 * first + 0.1 * second)


def _path_gate_from_effective_length(
    effective_path_len: float,
    radius_gate_path_len_threshold: float,
) -> float:
    return float(float(effective_path_len) > float(radius_gate_path_len_threshold))


def channel_hole_score(
    *,
    effective_path_len: float,
    min_radius: float,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
) -> float:
    radius_score = radius_score_from_min_radius(min_radius)
    radius_gate = _path_gate_from_effective_length(
        effective_path_len,
        radius_gate_path_len_threshold,
    )
    return float(radius_gate * radius_score)


def softplus_radius_penalty_from_min_radii(
    min_radii: Iterable[float],
    *,
    target_min_radius: float = DEFAULT_SOFTPLUS_RADIUS_TARGET,
    temperature: float = DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE,
) -> float:
    """Return a smooth negative penalty over all supplied channel radii."""
    values = [float(value) for value in min_radii]
    if not values:
        return 0.0
    temp = max(float(temperature), 1e-6)
    target = float(target_min_radius)
    deficits = (target - np.asarray(values, dtype=float)) / temp
    penalties = temp * np.logaddexp(0.0, deficits)
    return float(-np.mean(penalties))


def load_channel_scan_summary(scan_summary_path: str | pathlib.Path) -> list[dict[str, Any]]:
    scan_summary_path = _as_path(scan_summary_path)
    with scan_summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def channel_hole_metrics_from_scan_rows(
    rows: Iterable[dict[str, Any]],
    *,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    softplus_target_min_radius: float = DEFAULT_SOFTPLUS_RADIUS_TARGET,
    softplus_temperature: float = DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE,
) -> dict[str, Any]:
    parsed_rows = []
    softplus_radii: list[float] = []
    for row in rows:
        status = str(row.get("status", "ok")).strip().lower() or "ok"
        if status not in {"ok", "success"}:
            continue
        effective_path_len = float(row["effective_path_len"])
        min_radius = float(row["min_radius"])
        softplus_radii.append(min_radius)
        parsed_rows.append(
            (
                channel_hole_score(
                    effective_path_len=effective_path_len,
                    min_radius=min_radius,
                    radius_gate_path_len_threshold=radius_gate_path_len_threshold,
                ),
                str(row["scan_id"]),
                effective_path_len,
                min_radius,
            )
        )
    if not parsed_rows:
        return {
            "best_channel_scan_id": "",
            "best_channel_effective_path_len": 0.0,
            "best_channel_min_radius": 0.0,
            "channel_hole": 0.0,
            "channel_softplus_radius": 0.0,
            "max_geometric_channel_scan_id": "",
            "max_geometric_channel_effective_path_len": 0.0,
            "max_geometric_channel_min_radius": 0.0,
        }
    best_score, best_scan_id, best_path_len, best_min_radius = max(
        parsed_rows,
        key=lambda item: (item[0], item[3], item[2], item[1]),
    )
    geometric_best_score, geometric_best_scan_id, geometric_best_path_len, geometric_best_min_radius = max(
        parsed_rows,
        key=lambda item: (item[3], item[2], item[0], item[1]),
    )
    return {
        "best_channel_scan_id": best_scan_id,
        "best_channel_effective_path_len": float(best_path_len),
        "best_channel_min_radius": float(best_min_radius),
        "channel_hole": float(best_score),
        "channel_softplus_radius": softplus_radius_penalty_from_min_radii(
            softplus_radii,
            target_min_radius=softplus_target_min_radius,
            temperature=softplus_temperature,
        ),
        "max_geometric_channel_scan_id": geometric_best_scan_id,
        "max_geometric_channel_effective_path_len": float(geometric_best_path_len),
        "max_geometric_channel_min_radius": float(geometric_best_min_radius),
    }


def parse_hole_profile(
    hole_output_path: str | pathlib.Path,
    *,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
) -> dict[str, Any]:
    hole_output_path = _as_path(hole_output_path)
    profile_rows: list[tuple[float, float]] = []
    started = False
    for raw_line in hole_output_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith(" cenxyz.cvec"):
            started = True
            continue
        if not started:
            continue
        if line.startswith(" Minimum radius found:"):
            break
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coord_along_axis = float(parts[0])
            radius = float(parts[1])
        except ValueError:
            continue
        profile_rows.append((coord_along_axis, radius))

    if not profile_rows:
        raise ValueError(f"No HOLE profile rows found in {hole_output_path}")

    s_values = np.asarray([item[0] for item in profile_rows], dtype=float)
    radii = np.asarray([item[1] for item in profile_rows], dtype=float)
    effective_mask = radii < float(effective_radius_cutoff)
    if np.any(effective_mask):
        effective_s = s_values[effective_mask]
        effective_start_s = float(effective_s[0])
        effective_end_s = float(effective_s[-1])
        effective_path_len = float(effective_end_s - effective_start_s)
    else:
        effective_start_s = 0.0
        effective_end_s = 0.0
        effective_path_len = 0.0
    min_idx = int(np.argmin(radii))
    return {
        "profile_length": float(np.max(s_values) - np.min(s_values)),
        "effective_path_len": effective_path_len,
        "effective_start_s": effective_start_s,
        "effective_end_s": effective_end_s,
        "n_profile_points": int(len(profile_rows)),
        "min_radius": float(radii[min_idx]),
        "min_radius_profile": float(s_values[min_idx]),
    }


def _write_channel_scan_summary(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: pathlib.Path,
) -> pathlib.Path:
    fieldnames = [
        "scan_id",
        "parent_scan_id",
        "ix",
        "iy",
        "cx",
        "cy",
        "cz",
        "vx",
        "vy",
        "vz",
        "status",
        "profile_length",
        "effective_path_len",
        "effective_start_s",
        "effective_end_s",
        "n_profile_points",
        "min_radius",
        "min_radius_profile",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output_path


def _resolve_hole_executable(hole_executable: str | None) -> str:
    if hole_executable:
        return str(pathlib.Path(str(hole_executable)).expanduser())
    configured = getattr(global_config, "hole_path", None)
    if configured:
        return str(pathlib.Path(str(configured)).expanduser())
    discovered = shutil.which("hole")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "channel_hole requires a HOLE executable. Set `hole_executable` or `global_config.hole_path`."
    )


def _resolve_hole_radius_file(hole_radius_file: str | None) -> str:
    if hole_radius_file:
        return str(pathlib.Path(str(hole_radius_file)).expanduser())
    configured = getattr(global_config, "hole_radius_file", None)
    if configured:
        return str(pathlib.Path(str(configured)).expanduser())
    env_path = os.environ.get("GH_HOLE_RADIUS_FILE", "").strip()
    if env_path:
        return str(pathlib.Path(env_path).expanduser())
    raise FileNotFoundError(
        "channel_hole requires a HOLE radius file. Set `hole_radius_file`, `global_config.hole_radius_file`, or GH_HOLE_RADIUS_FILE."
    )


def _run_single_hole_scan(
    *,
    aligned_structure_path: pathlib.Path,
    scan: ChannelScanInput,
    output_dir: pathlib.Path,
    hole_executable: str,
    hole_radius_file: str,
    end_radius: float,
    effective_radius_cutoff: float,
) -> dict[str, Any]:
    scan_dir = output_dir / scan.scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    local_coord_path = scan_dir / "input.pdb"
    if not local_coord_path.exists():
        local_coord_path.hardlink_to(aligned_structure_path)
    local_radius_path = scan_dir / "radius.rad"
    if not local_radius_path.exists():
        try:
            local_radius_path.hardlink_to(pathlib.Path(hole_radius_file).expanduser())
        except OSError:
            shutil.copy2(pathlib.Path(hole_radius_file).expanduser(), local_radius_path)
    input_text = "\n".join(
        [
            f"COORD {local_coord_path.name}",
            f"RADIUS {local_radius_path.name}",
            "",
            f"CPOINT {scan.center[0]} {scan.center[1]} {scan.center[2]}",
            f"CVECT  {scan.axis[0]} {scan.axis[1]} {scan.axis[2]}",
            "",
            f"ENDRAD {end_radius}",
            "SPHPDB spheres.pdb",
            "",
        ]
    )
    result = subprocess.run(
        [hole_executable],
        input=input_text,
        text=True,
        capture_output=True,
        cwd=scan_dir,
        check=False,
    )
    hole_txt_path = scan_dir / f"{scan.scan_id}.hole.txt"
    hole_txt_path.write_text(result.stdout or "", encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"HOLE scan {scan.scan_id} failed with code {result.returncode}: {result.stderr.strip()}"
        )
    generated_spheres = scan_dir / "spheres.pdb"
    if generated_spheres.exists():
        generated_spheres.rename(scan_dir / f"{scan.scan_id}_spheres.pdb")

    metrics = parse_hole_profile(
        hole_txt_path,
        effective_radius_cutoff=effective_radius_cutoff,
    )
    return {
        "scan_id": scan.scan_id,
        "parent_scan_id": scan.parent_scan_id,
        "ix": scan.ix,
        "iy": scan.iy,
        "cx": scan.center[0],
        "cy": scan.center[1],
        "cz": scan.center[2],
        "vx": scan.axis[0],
        "vy": scan.axis[1],
        "vz": scan.axis[2],
        "status": "ok",
        **metrics,
    }


def _file_signature(path: pathlib.Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (
        str(path.resolve()),
        int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        int(stat.st_size),
    )


@functools.lru_cache(maxsize=64)
def _compute_channel_hole_metrics_cached(
    *,
    structure_sig: tuple[str, int, int],
    reference_structure_sig: tuple[str, int, int],
    reference_spheres_sig: tuple[str, int, int],
    output_root: str,
    lateral_step: float,
    end_radius: float,
    effective_radius_cutoff: float,
    radius_gate_path_len_threshold: float,
    softplus_target_min_radius: float,
    softplus_temperature: float,
    alignment_chains_key: tuple[str, ...],
    hole_executable: str,
    hole_radius_file: str,
) -> dict[str, Any]:
    structure_path = pathlib.Path(structure_sig[0])
    reference_structure_path = pathlib.Path(reference_structure_sig[0])
    reference_spheres_path = pathlib.Path(reference_spheres_sig[0])
    output_dir = pathlib.Path(output_root)
    prepared = build_channel_scan_inputs(
        structure_path=structure_path,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        output_dir=output_dir,
        lateral_step=lateral_step,
        alignment_chains=list(alignment_chains_key) or None,
    )
    rows = [
        _run_single_hole_scan(
            aligned_structure_path=prepared.aligned_structure_path,
            scan=scan,
            output_dir=output_dir,
            hole_executable=hole_executable,
            hole_radius_file=hole_radius_file,
            end_radius=end_radius,
            effective_radius_cutoff=effective_radius_cutoff,
        )
        for scan in prepared.scans
    ]
    _write_channel_scan_summary(rows, output_path=output_dir / "scan_summary.tsv")
    return channel_hole_metrics_from_scan_rows(
        rows,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        softplus_target_min_radius=softplus_target_min_radius,
        softplus_temperature=softplus_temperature,
    )


@functools.lru_cache(maxsize=64)
def _compute_channel_hole_metrics_exact_cyclic_cached(
    *,
    structure_sig: tuple[str, int, int],
    oligo_name: str,
    output_root: str,
    lateral_step: float,
    end_radius: float,
    effective_radius_cutoff: float,
    radius_gate_path_len_threshold: float,
    softplus_target_min_radius: float,
    softplus_temperature: float,
    hole_executable: str,
    hole_radius_file: str,
) -> dict[str, Any]:
    structure_path = pathlib.Path(structure_sig[0])
    output_dir = pathlib.Path(output_root)
    prepared = build_channel_scan_inputs(
        structure_path=structure_path,
        output_dir=output_dir,
        frame_source=FRAME_SOURCE_EXACT_CYCLIC,
        oligo_name=oligo_name,
        lateral_step=lateral_step,
    )
    rows = [
        _run_single_hole_scan(
            aligned_structure_path=prepared.aligned_structure_path,
            scan=scan,
            output_dir=output_dir,
            hole_executable=hole_executable,
            hole_radius_file=hole_radius_file,
            end_radius=end_radius,
            effective_radius_cutoff=effective_radius_cutoff,
        )
        for scan in prepared.scans
    ]
    _write_channel_scan_summary(rows, output_path=output_dir / "scan_summary.tsv")
    return channel_hole_metrics_from_scan_rows(
        rows,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        softplus_target_min_radius=softplus_target_min_radius,
        softplus_temperature=softplus_temperature,
    )


def _resolve_path_param(
    value: str | pathlib.Path | dict[str, str] | dict[str, pathlib.Path] | None,
    *,
    oligo_name: str,
    param_name: str,
) -> pathlib.Path:
    if value is None:
        raise ValueError(f"{param_name} is required for oligo {oligo_name!r}.")
    if isinstance(value, dict):
        selected = value.get(oligo_name)
        if selected is None:
            raise KeyError(f"Missing {param_name} entry for oligo {oligo_name!r}.")
        return _as_path(selected).expanduser().resolve()
    return _as_path(value).expanduser().resolve()


def _channel_hole_output_dir(oligo_name: str) -> pathlib.Path:
    step = int(getattr(global_config, "step", 0) or 0)
    root = getattr(global_config, "raw_model_results_dir", None)
    if root is None:
        root = pathlib.Path(getattr(global_config, "out_dir")) / "raw_model_results"
    return pathlib.Path(root).expanduser().resolve() / "channel_hole" / f"step_{step:03d}_{oligo_name}"


def compute_channel_hole_metrics_for_structure(
    *,
    structure_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    reference_structure_path: (
        str | pathlib.Path | dict[str, str] | dict[str, pathlib.Path] | None
    ) = None,
    reference_spheres_path: (
        str | pathlib.Path | dict[str, str] | dict[str, pathlib.Path] | None
    ) = None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    oligo_name: str | None = None,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    softplus_target_min_radius: float = DEFAULT_SOFTPLUS_RADIUS_TARGET,
    softplus_temperature: float = DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
) -> dict[str, Any]:
    hole_exec = _resolve_hole_executable(hole_executable)
    hole_radius = _resolve_hole_radius_file(hole_radius_file)
    if alignment_chains is None:
        alignment_chain_tokens: tuple[str, ...] = ()
    elif isinstance(alignment_chains, str):
        alignment_chain_tokens = tuple(
            token for token in re.split(r"[,\s]+", alignment_chains) if token
        )
    else:
        alignment_chain_tokens = tuple(
            str(token) for token in alignment_chains if str(token).strip()
        )
    structure = _as_path(structure_path).expanduser().resolve()
    output_root = _as_path(output_dir).expanduser().resolve()
    frame_source_norm = _normalize_frame_source(frame_source)
    if frame_source_norm == FRAME_SOURCE_EXACT_CYCLIC:
        if not oligo_name:
            raise ValueError("exact_cyclic channel_hole scoring requires oligo_name.")
        return _compute_channel_hole_metrics_exact_cyclic_cached(
            structure_sig=_file_signature(structure),
            oligo_name=str(oligo_name),
            output_root=str(output_root),
            lateral_step=float(lateral_step),
            end_radius=float(end_radius),
            effective_radius_cutoff=float(effective_radius_cutoff),
            radius_gate_path_len_threshold=float(radius_gate_path_len_threshold),
            softplus_target_min_radius=float(softplus_target_min_radius),
            softplus_temperature=float(softplus_temperature),
            hole_executable=hole_exec,
            hole_radius_file=hole_radius,
        )

    if reference_structure_path is None or reference_spheres_path is None:
        raise ValueError(
            "reference_spheres channel_hole scoring requires "
            "reference_structure_path and reference_spheres_path."
        )
    if isinstance(reference_structure_path, dict) or isinstance(reference_spheres_path, dict):
        if not oligo_name:
            raise ValueError(
                "Dict-valued reference paths require oligo_name when scoring a single structure."
            )
        reference_structure = _resolve_path_param(
            reference_structure_path,
            oligo_name=str(oligo_name),
            param_name="reference_structure_path",
        )
        reference_spheres = _resolve_path_param(
            reference_spheres_path,
            oligo_name=str(oligo_name),
            param_name="reference_spheres_path",
        )
    else:
        reference_structure = _as_path(reference_structure_path).expanduser().resolve()
        reference_spheres = _as_path(reference_spheres_path).expanduser().resolve()
    return _compute_channel_hole_metrics_cached(
        structure_sig=_file_signature(structure),
        reference_structure_sig=_file_signature(reference_structure),
        reference_spheres_sig=_file_signature(reference_spheres),
        output_root=str(output_root),
        lateral_step=float(lateral_step),
        end_radius=float(end_radius),
        effective_radius_cutoff=float(effective_radius_cutoff),
        radius_gate_path_len_threshold=float(radius_gate_path_len_threshold),
        softplus_target_min_radius=float(softplus_target_min_radius),
        softplus_temperature=float(softplus_temperature),
        alignment_chains_key=alignment_chain_tokens,
        hole_executable=hole_exec,
        hole_radius_file=hole_radius,
    )


def _channel_hole_metrics_by_oligo(
    eval_result,
    *,
    reference_structure_path: (
        str | pathlib.Path | dict[str, str] | dict[str, pathlib.Path] | None
    ) = None,
    reference_spheres_path: (
        str | pathlib.Path | dict[str, str] | dict[str, pathlib.Path] | None
    ) = None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    softplus_target_min_radius: float = DEFAULT_SOFTPLUS_RADIUS_TARGET,
    softplus_temperature: float = DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
) -> dict[str, dict[str, Any]]:
    hole_exec = _resolve_hole_executable(hole_executable)
    hole_radius = _resolve_hole_radius_file(hole_radius_file)
    alignment_chain_tokens = ()
    if alignment_chains is None:
        alignment_chain_tokens = ()
    elif isinstance(alignment_chains, str):
        alignment_chain_tokens = tuple(
            token for token in re.split(r"[,\s]+", alignment_chains) if token
        )
    else:
        alignment_chain_tokens = tuple(
            str(token) for token in alignment_chains if str(token).strip()
        )

    frame_source_norm = _normalize_frame_source(frame_source)
    out: dict[str, dict[str, Any]] = {}
    for oligo_name, structure_path_like in eval_result.complex_paths.items():
        structure_path = _as_path(structure_path_like).expanduser().resolve()
        output_dir = _channel_hole_output_dir(oligo_name)
        if frame_source_norm == FRAME_SOURCE_EXACT_CYCLIC:
            out[oligo_name] = _compute_channel_hole_metrics_exact_cyclic_cached(
                structure_sig=_file_signature(structure_path),
                oligo_name=str(oligo_name),
                output_root=str(output_dir),
                lateral_step=float(lateral_step),
                end_radius=float(end_radius),
                effective_radius_cutoff=float(effective_radius_cutoff),
                radius_gate_path_len_threshold=float(radius_gate_path_len_threshold),
                softplus_target_min_radius=float(softplus_target_min_radius),
                softplus_temperature=float(softplus_temperature),
                hole_executable=hole_exec,
                hole_radius_file=hole_radius,
            )
        else:
            reference_structure = _resolve_path_param(
                reference_structure_path,
                oligo_name=oligo_name,
                param_name="reference_structure_path",
            )
            reference_spheres = _resolve_path_param(
                reference_spheres_path,
                oligo_name=oligo_name,
                param_name="reference_spheres_path",
            )
            out[oligo_name] = _compute_channel_hole_metrics_cached(
                structure_sig=_file_signature(structure_path),
                reference_structure_sig=_file_signature(reference_structure),
                reference_spheres_sig=_file_signature(reference_spheres),
                output_root=str(output_dir),
                lateral_step=float(lateral_step),
                end_radius=float(end_radius),
                effective_radius_cutoff=float(effective_radius_cutoff),
                radius_gate_path_len_threshold=float(radius_gate_path_len_threshold),
                softplus_target_min_radius=float(softplus_target_min_radius),
                softplus_temperature=float(softplus_temperature),
                alignment_chains_key=alignment_chain_tokens,
                hole_executable=hole_exec,
                hole_radius_file=hole_radius,
            )
    return out


@loss_registry.register(
    required_params=[],
    optional_params=[
        "reference_structure_path",
        "reference_spheres_path",
        "frame_source",
        "alignment_chains",
        "lateral_step",
        "end_radius",
        "effective_radius_cutoff",
        "radius_gate_path_len_threshold",
        "hole_executable",
        "hole_radius_file",
    ],
    description="Best-of-9 HOLE score using a reference-spheres frame or the current exact-cyclic axis.",
)
def channel_hole(
    eval_result,
    *,
    reference_structure_path=None,
    reference_spheres_path=None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
):
    metrics = _channel_hole_metrics_by_oligo(
        eval_result,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        frame_source=frame_source,
        alignment_chains=alignment_chains,
        lateral_step=lateral_step,
        end_radius=end_radius,
        effective_radius_cutoff=effective_radius_cutoff,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        hole_executable=hole_executable,
        hole_radius_file=hole_radius_file,
    )
    return {oligo_name: values["channel_hole"] for oligo_name, values in metrics.items()}


@loss_registry.register(
    required_params=[],
    optional_params=[
        "reference_structure_path",
        "reference_spheres_path",
        "frame_source",
        "alignment_chains",
        "lateral_step",
        "end_radius",
        "effective_radius_cutoff",
        "radius_gate_path_len_threshold",
        "hole_executable",
        "hole_radius_file",
    ],
    description="Effective path length of the best channel_hole scan.",
)
def best_channel_effective_path_len(
    eval_result,
    *,
    reference_structure_path=None,
    reference_spheres_path=None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
):
    metrics = _channel_hole_metrics_by_oligo(
        eval_result,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        frame_source=frame_source,
        alignment_chains=alignment_chains,
        lateral_step=lateral_step,
        end_radius=end_radius,
        effective_radius_cutoff=effective_radius_cutoff,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        hole_executable=hole_executable,
        hole_radius_file=hole_radius_file,
    )
    return {
        oligo_name: values["best_channel_effective_path_len"]
        for oligo_name, values in metrics.items()
    }


@loss_registry.register(
    required_params=[],
    optional_params=[
        "reference_structure_path",
        "reference_spheres_path",
        "frame_source",
        "alignment_chains",
        "lateral_step",
        "end_radius",
        "effective_radius_cutoff",
        "radius_gate_path_len_threshold",
        "hole_executable",
        "hole_radius_file",
    ],
    description="Minimum radius of the best channel_hole scan.",
)
def best_channel_min_radius(
    eval_result,
    *,
    reference_structure_path=None,
    reference_spheres_path=None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
):
    metrics = _channel_hole_metrics_by_oligo(
        eval_result,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        frame_source=frame_source,
        alignment_chains=alignment_chains,
        lateral_step=lateral_step,
        end_radius=end_radius,
        effective_radius_cutoff=effective_radius_cutoff,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        hole_executable=hole_executable,
        hole_radius_file=hole_radius_file,
    )
    return {
        oligo_name: values["best_channel_min_radius"]
        for oligo_name, values in metrics.items()
    }


@loss_registry.register(
    required_params=[],
    optional_params=[
        "reference_structure_path",
        "reference_spheres_path",
        "frame_source",
        "alignment_chains",
        "lateral_step",
        "end_radius",
        "effective_radius_cutoff",
        "radius_gate_path_len_threshold",
        "target_min_radius",
        "temperature",
        "hole_executable",
        "hole_radius_file",
    ],
    description="Smooth negative softplus penalty over all valid HOLE scan minimum radii.",
)
def channel_softplus_radius(
    eval_result,
    *,
    reference_structure_path=None,
    reference_spheres_path=None,
    frame_source: str = FRAME_SOURCE_REFERENCE_SPHERES,
    alignment_chains: str | Sequence[str] | None = None,
    lateral_step: float = DEFAULT_LATERAL_STEP,
    end_radius: float = DEFAULT_END_RADIUS,
    effective_radius_cutoff: float = DEFAULT_EFFECTIVE_RADIUS_CUTOFF,
    radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    target_min_radius: float = DEFAULT_SOFTPLUS_RADIUS_TARGET,
    temperature: float = DEFAULT_SOFTPLUS_RADIUS_TEMPERATURE,
    hole_executable: str | None = None,
    hole_radius_file: str | None = None,
):
    metrics = _channel_hole_metrics_by_oligo(
        eval_result,
        reference_structure_path=reference_structure_path,
        reference_spheres_path=reference_spheres_path,
        frame_source=frame_source,
        alignment_chains=alignment_chains,
        lateral_step=lateral_step,
        end_radius=end_radius,
        effective_radius_cutoff=effective_radius_cutoff,
        radius_gate_path_len_threshold=radius_gate_path_len_threshold,
        softplus_target_min_radius=target_min_radius,
        softplus_temperature=temperature,
        hole_executable=hole_executable,
        hole_radius_file=hole_radius_file,
    )
    return {oligo_name: values["channel_softplus_radius"] for oligo_name, values in metrics.items()}
