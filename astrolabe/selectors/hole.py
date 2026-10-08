from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from Bio.PDB import PDBParser

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.losses.channel_hole import (
    DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
    _channel_hole_output_dir,
    _load_reference_spheres,
    _polymer_residues,
    channel_hole_metrics_from_scan_rows,
    compute_channel_hole_metrics_for_structure,
    load_channel_scan_summary,
)
from astrolabe.selectors.base import PositionSelector

_VDW_BY_ELEMENT = {
    "C": 1.85,
    "O": 1.65,
    "S": 2.00,
    "N": 1.75,
    "H": 1.00,
    "P": 2.10,
}


def _vdw_radius_for_atom(atom) -> float:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return _VDW_BY_ELEMENT.get(element, 1.85)
    atom_name = str(atom.get_name()).strip().upper()
    return _VDW_BY_ELEMENT.get(atom_name[:1], 1.85)


class HoleSurfacePositionSelector(PositionSelector):
    """
    Select protein residues close to the HOLE centerline surface.

    The selector reuses the current step's `channel_hole` artifacts when present.
    When the artifacts are missing, it exports the current pose and computes the
    HOLE scans with the same helper used by the loss pipeline.
    """

    def __init__(
        self,
        complex_poses: ComplexPoses,
        reference_structure_path: str,
        reference_spheres_path: str,
        alignment_chains: str | Sequence[str] | None = None,
        scan_kind: str = "best_channel",
        surface_gap_min: float = -0.25,
        surface_gap_max: float = 1.5,
        radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
        hole_executable: str | None = None,
        hole_radius_file: str | None = None,
        merge: str = "any",
    ):
        self.reference_structure_path = str(reference_structure_path)
        self.reference_spheres_path = str(reference_spheres_path)
        self.alignment_chains = alignment_chains
        self.scan_kind = str(scan_kind).strip() or "best_channel"
        self.surface_gap_min = float(surface_gap_min)
        self.surface_gap_max = float(surface_gap_max)
        self.radius_gate_path_len_threshold = float(radius_gate_path_len_threshold)
        self.hole_executable = hole_executable
        self.hole_radius_file = hole_radius_file
        if self.scan_kind not in {"best_channel", "max_geometric"}:
            raise ValueError(
                f"scan_kind must be 'best_channel' or 'max_geometric', got {self.scan_kind!r}"
            )
        if self.surface_gap_min > self.surface_gap_max:
            raise ValueError("surface_gap_min must be <= surface_gap_max")
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._compute_hole_surface_masks()
        return self.sites

    def _compute_hole_surface_masks(self) -> Mapping[str, np.ndarray]:
        proto2masks: dict[str, list[np.ndarray]] = defaultdict(list)
        for oligo_name, pose in self.complex_poses.poses.items():
            step_dir, selected_scan_id = self._ensure_scan_artifacts(oligo_name, pose)
            aligned_structure_path = self._resolve_aligned_structure_path(step_dir)
            spheres_path = step_dir / selected_scan_id / f"{selected_scan_id}_spheres.pdb"
            if not spheres_path.exists():
                raise FileNotFoundError(
                    f"Missing HOLE spheres file for selector scan {selected_scan_id!r}: {spheres_path}"
                )
            per_proto = self._select_from_aligned_structure(
                oligo_name=oligo_name,
                aligned_structure_path=aligned_structure_path,
                spheres_path=spheres_path,
            )
            for proto, mask in per_proto.items():
                proto2masks[proto].append(mask)

        selected_sites: dict[str, np.ndarray] = {}
        for proto, protomer in global_config.protomers.items():
            if protomer.type != ProtomerTypes.PROTEIN:
                selected_sites[proto] = np.zeros(protomer.length, dtype=bool)
                continue
            if proto not in proto2masks:
                selected_sites[proto] = np.zeros(protomer.length, dtype=bool)
                continue
            selected_sites[proto] = self.merge_func(proto2masks[proto], axis=0)
        return selected_sites

    def _ensure_scan_artifacts(self, oligo_name: str, pose) -> tuple[Path, str]:
        step_dir = _channel_hole_output_dir(oligo_name)
        scan_summary_path = step_dir / "scan_summary.tsv"
        if not scan_summary_path.exists():
            step_dir.mkdir(parents=True, exist_ok=True)
            export_path = step_dir / f"selector_step_{int(getattr(global_config, 'step', 0) or 0):03d}_{oligo_name}.pdb"
            pose.dump_pdb(str(export_path))
            compute_channel_hole_metrics_for_structure(
                structure_path=export_path,
                reference_structure_path=self.reference_structure_path,
                reference_spheres_path=self.reference_spheres_path,
                output_dir=step_dir,
                alignment_chains=self.alignment_chains,
                radius_gate_path_len_threshold=self.radius_gate_path_len_threshold,
                hole_executable=self.hole_executable,
                hole_radius_file=self.hole_radius_file,
            )

        rows = load_channel_scan_summary(scan_summary_path)
        metrics = channel_hole_metrics_from_scan_rows(
            rows,
            radius_gate_path_len_threshold=self.radius_gate_path_len_threshold,
        )
        key = (
            "best_channel_scan_id"
            if self.scan_kind == "best_channel"
            else "max_geometric_channel_scan_id"
        )
        selected_scan_id = str(metrics.get(key, "")).strip()
        if not selected_scan_id:
            raise ValueError(
                f"HoleSurfacePositionSelector could not resolve scan id from {scan_summary_path}"
            )
        return step_dir, selected_scan_id

    @staticmethod
    def _resolve_aligned_structure_path(step_dir: Path) -> Path:
        matches = sorted(step_dir.glob("*__channel_hole_aligned.pdb"))
        if not matches:
            raise FileNotFoundError(
                f"Missing aligned structure artifact in HOLE selector directory: {step_dir}"
            )
        return matches[0]

    def _select_from_aligned_structure(
        self,
        *,
        oligo_name: str,
        aligned_structure_path: Path,
        spheres_path: Path,
    ) -> Mapping[str, np.ndarray]:
        structure = PDBParser(QUIET=True).get_structure(aligned_structure_path.stem, str(aligned_structure_path))
        model = next(structure.get_models())
        sphere_centers, sphere_radii = _load_reference_spheres(spheres_path)
        chain_entries = list(model.get_chains())
        per_proto: dict[str, np.ndarray] = {}
        for chain, proto in zip(chain_entries, oligo_name):
            protomer = global_config.protomers.get(proto)
            if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
                continue
            mask = np.zeros(protomer.length, dtype=bool)
            residues = _polymer_residues(chain)
            for idx, residue in enumerate(residues):
                if idx >= mask.shape[0]:
                    break
                min_surface_gap = self._residue_min_surface_gap(
                    residue=residue,
                    sphere_centers=sphere_centers,
                    sphere_radii=sphere_radii,
                )
                if self.surface_gap_min <= min_surface_gap <= self.surface_gap_max:
                    mask[idx] = True
            per_proto[proto] = mask
        return per_proto

    @staticmethod
    def _residue_min_surface_gap(
        *,
        residue,
        sphere_centers: np.ndarray,
        sphere_radii: np.ndarray,
    ) -> float:
        min_gap = None
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H":
                continue
            coord = np.asarray(atom.coord, dtype=float)
            center_dist = np.linalg.norm(sphere_centers - coord[None, :], axis=1)
            gaps = center_dist - sphere_radii - float(_vdw_radius_for_atom(atom))
            atom_gap = float(np.min(gaps))
            if min_gap is None or atom_gap < min_gap:
                min_gap = atom_gap
        if min_gap is None:
            return float("inf")
        return float(min_gap)


class PoreBottleneckPositionSelector(PositionSelector):
    """
    Select protein residues that are closest to current HOLE bottlenecks.

    Unlike :class:`HoleSurfacePositionSelector`, which selects a shell around the
    channel surface, this selector ranks residues by side-chain surface distance
    to all near-minimum-radius bottleneck regions in the selected scan. It is
    intended for small direct mutations of local blockers. ``top_n`` is applied
    per bottleneck region so separate radius-limiting regions are not dropped by
    a single global ranking; use ``max_total_positions`` only when an explicit
    global cap is needed. For failed channels, all spheres below
    ``target_min_radius`` are treated as limiting regions because any of them can
    become the next rmin blocker after the deepest point is mutated.
    """

    def __init__(
        self,
        complex_poses: ComplexPoses,
        reference_structure_path: str,
        reference_spheres_path: str,
        alignment_chains: str | Sequence[str] | None = None,
        scan_kind: str = "best_channel",
        top_n: int = 8,
        max_total_positions: int | None = None,
        surface_gap_max: float = 1.5,
        min_radius_window: float = 0.05,
        target_min_radius: float | None = 1.2,
        bottleneck_region_gap: int = 3,
        region_stride: int | None = None,
        region_offset: int = 0,
        region_offset_source: str = "static",
        radius_gate_path_len_threshold: float = DEFAULT_RADIUS_GATE_PATH_LEN_THRESHOLD,
        hole_executable: str | None = None,
        hole_radius_file: str | None = None,
        sidechain_only: bool = True,
        merge: str = "any",
    ):
        self.reference_structure_path = str(reference_structure_path)
        self.reference_spheres_path = str(reference_spheres_path)
        self.alignment_chains = alignment_chains
        self.scan_kind = str(scan_kind).strip() or "best_channel"
        self.top_n = max(1, int(top_n))
        if max_total_positions in (None, "", 0, "0", "None"):
            self.max_total_positions = None
        else:
            self.max_total_positions = max(1, int(max_total_positions))
        self.surface_gap_max = float(surface_gap_max)
        self.min_radius_window = max(0.0, float(min_radius_window))
        if target_min_radius in (None, "", "None"):
            self.target_min_radius = None
        else:
            self.target_min_radius = float(target_min_radius)
        self.bottleneck_region_gap = max(0, int(bottleneck_region_gap))
        if region_stride in (None, "", 0, "0", "None"):
            self.region_stride = None
        else:
            self.region_stride = max(1, int(region_stride))
        self.region_offset = int(region_offset or 0)
        self.region_offset_source = str(region_offset_source or "static").strip().lower()
        if self.region_offset_source not in {"static", "step", "activation"}:
            raise ValueError(
                "region_offset_source must be 'static', 'step', or 'activation'."
            )
        self.radius_gate_path_len_threshold = float(radius_gate_path_len_threshold)
        self.hole_executable = hole_executable
        self.hole_radius_file = hole_radius_file
        self.sidechain_only = bool(sidechain_only)
        if self.scan_kind not in {"best_channel", "max_geometric"}:
            raise ValueError(
                f"scan_kind must be 'best_channel' or 'max_geometric', got {self.scan_kind!r}"
            )
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._compute_bottleneck_masks()
        return self.sites

    def _compute_bottleneck_masks(self) -> Mapping[str, np.ndarray]:
        proto2masks: dict[str, list[np.ndarray]] = defaultdict(list)
        for oligo_name, pose in self.complex_poses.poses.items():
            step_dir, selected_scan_id, min_radius = self._ensure_scan_artifacts(
                oligo_name, pose
            )
            aligned_structure_path = HoleSurfacePositionSelector._resolve_aligned_structure_path(
                step_dir
            )
            spheres_path = step_dir / selected_scan_id / f"{selected_scan_id}_spheres.pdb"
            if not spheres_path.exists():
                raise FileNotFoundError(
                    f"Missing HOLE spheres file for bottleneck scan {selected_scan_id!r}: {spheres_path}"
                )
            per_proto = self._select_from_aligned_structure(
                oligo_name=oligo_name,
                aligned_structure_path=aligned_structure_path,
                spheres_path=spheres_path,
                min_radius=min_radius,
            )
            for proto, mask in per_proto.items():
                proto2masks[proto].append(mask)

        selected_sites: dict[str, np.ndarray] = {}
        for proto, protomer in global_config.protomers.items():
            if protomer.type != ProtomerTypes.PROTEIN:
                selected_sites[proto] = np.zeros(protomer.length, dtype=bool)
                continue
            if proto not in proto2masks:
                selected_sites[proto] = np.zeros(protomer.length, dtype=bool)
                continue
            selected_sites[proto] = self.merge_func(proto2masks[proto], axis=0)
        return selected_sites

    def _ensure_scan_artifacts(self, oligo_name: str, pose) -> tuple[Path, str, float]:
        step_dir = _channel_hole_output_dir(oligo_name)
        scan_summary_path = step_dir / "scan_summary.tsv"
        if not scan_summary_path.exists():
            step_dir.mkdir(parents=True, exist_ok=True)
            export_path = step_dir / f"selector_step_{int(getattr(global_config, 'step', 0) or 0):03d}_{oligo_name}.pdb"
            pose.dump_pdb(str(export_path))
            compute_channel_hole_metrics_for_structure(
                structure_path=export_path,
                reference_structure_path=self.reference_structure_path,
                reference_spheres_path=self.reference_spheres_path,
                output_dir=step_dir,
                alignment_chains=self.alignment_chains,
                radius_gate_path_len_threshold=self.radius_gate_path_len_threshold,
                hole_executable=self.hole_executable,
                hole_radius_file=self.hole_radius_file,
            )

        rows = load_channel_scan_summary(scan_summary_path)
        metrics = channel_hole_metrics_from_scan_rows(
            rows,
            radius_gate_path_len_threshold=self.radius_gate_path_len_threshold,
        )
        key = (
            "best_channel_scan_id"
            if self.scan_kind == "best_channel"
            else "max_geometric_channel_scan_id"
        )
        min_radius_key = (
            "best_channel_min_radius"
            if self.scan_kind == "best_channel"
            else "max_geometric_channel_min_radius"
        )
        selected_scan_id = str(metrics.get(key, "")).strip()
        if not selected_scan_id:
            raise ValueError(
                f"PoreBottleneckPositionSelector could not resolve scan id from {scan_summary_path}"
            )
        return step_dir, selected_scan_id, float(metrics.get(min_radius_key, 0.0) or 0.0)

    def _select_from_aligned_structure(
        self,
        *,
        oligo_name: str,
        aligned_structure_path: Path,
        spheres_path: Path,
        min_radius: float,
    ) -> Mapping[str, np.ndarray]:
        structure = PDBParser(QUIET=True).get_structure(
            aligned_structure_path.stem,
            str(aligned_structure_path),
        )
        model = next(structure.get_models())
        sphere_centers, sphere_radii = _load_reference_spheres(spheres_path)
        bottleneck_regions = self._bottleneck_regions(
            sphere_radii=sphere_radii,
            min_radius=min_radius,
        )

        chain_entries = list(model.get_chains())
        per_proto: dict[str, np.ndarray] = {}
        for chain, proto in zip(chain_entries, oligo_name):
            protomer = global_config.protomers.get(proto)
            if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
                continue
            mask = np.zeros(protomer.length, dtype=bool)
            residues = _polymer_residues(chain)
            selected_indices = self._select_residue_indices_for_regions(
                residues=residues,
                mask_length=mask.shape[0],
                sphere_centers=sphere_centers,
                sphere_radii=sphere_radii,
                bottleneck_regions=bottleneck_regions,
            )
            for idx in selected_indices:
                mask[idx] = True
            per_proto[proto] = mask
        return per_proto

    def _bottleneck_regions(
        self,
        *,
        sphere_radii: np.ndarray,
        min_radius: float,
    ) -> list[np.ndarray]:
        threshold = float(min_radius) + self.min_radius_window
        if self.target_min_radius is not None:
            threshold = max(threshold, float(self.target_min_radius))
        bottleneck_indices = np.where(sphere_radii <= threshold)[0]
        if bottleneck_indices.size == 0:
            bottleneck_indices = np.asarray([int(np.argmin(sphere_radii))], dtype=int)

        regions: list[list[int]] = []
        current: list[int] = []
        previous = None
        for raw_idx in bottleneck_indices.tolist():
            idx = int(raw_idx)
            if previous is None or idx - previous <= self.bottleneck_region_gap + 1:
                current.append(idx)
            else:
                regions.append(current)
                current = [idx]
            previous = idx
        if current:
            regions.append(current)
        return [np.asarray(region, dtype=int) for region in regions]

    def _staged_bottleneck_regions(
        self, bottleneck_regions: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        regions = list(bottleneck_regions)
        if not regions or self.region_stride is None or self.region_stride <= 1:
            return regions

        if self.region_offset_source == "step":
            dynamic_offset = int(getattr(global_config, "step", 0) or 0)
        elif self.region_offset_source == "activation":
            dynamic_offset = int(
                getattr(global_config, "pore_bottleneck_activation_index", 0) or 0
            )
        else:
            dynamic_offset = 0

        effective_stride = min(int(self.region_stride), len(regions))
        offset = (int(self.region_offset) + dynamic_offset) % effective_stride
        staged = [
            region
            for idx, region in enumerate(regions)
            if idx % effective_stride == offset
        ]
        return staged or [regions[offset % len(regions)]]

    def _select_residue_indices_for_regions(
        self,
        *,
        residues,
        mask_length: int,
        sphere_centers: np.ndarray,
        sphere_radii: np.ndarray,
        bottleneck_regions: Sequence[np.ndarray],
    ) -> list[int]:
        ranked_by_region: list[list[tuple[float, int]]] = []
        all_ranked_by_idx: dict[int, float] = {}
        for region_indices in self._staged_bottleneck_regions(bottleneck_regions):
            region_centers = sphere_centers[region_indices]
            region_radii = sphere_radii[region_indices]
            ranked: list[tuple[float, int]] = []
            for idx, residue in enumerate(residues):
                if idx >= mask_length:
                    break
                min_surface_gap = self._residue_min_bottleneck_gap(
                    residue=residue,
                    sphere_centers=region_centers,
                    sphere_radii=region_radii,
                )
                if min_surface_gap <= self.surface_gap_max:
                    ranked.append((float(min_surface_gap), idx))
                    if (
                        idx not in all_ranked_by_idx
                        or min_surface_gap < all_ranked_by_idx[idx]
                    ):
                        all_ranked_by_idx[idx] = float(min_surface_gap)
            ranked_by_region.append(sorted(ranked, key=lambda item: (item[0], item[1])))

        selected: list[int] = []
        seen: set[int] = set()
        for ranked in ranked_by_region:
            added_for_region = 0
            for _, idx in ranked:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                added_for_region += 1
                if added_for_region >= self.top_n:
                    break

        max_total_positions = self.max_total_positions
        if max_total_positions is not None and len(selected) < max_total_positions:
            globally_ranked = sorted(
                all_ranked_by_idx.items(),
                key=lambda item: (item[1], item[0]),
            )
            for idx, _gap in globally_ranked:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= max_total_positions:
                    break

        if max_total_positions is None:
            return selected
        return selected[:max_total_positions]

    def _residue_min_bottleneck_gap(
        self,
        *,
        residue,
        sphere_centers: np.ndarray,
        sphere_radii: np.ndarray,
    ) -> float:
        min_gap = None
        for atom in residue.get_atoms():
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H":
                continue
            atom_name = str(atom.get_name()).strip().upper()
            if self.sidechain_only and atom_name in {"N", "CA", "C", "O", "OXT"}:
                continue
            coord = np.asarray(atom.coord, dtype=float)
            center_dist = np.linalg.norm(sphere_centers - coord[None, :], axis=1)
            gaps = center_dist - sphere_radii - float(_vdw_radius_for_atom(atom))
            atom_gap = float(np.min(gaps))
            if min_gap is None or atom_gap < min_gap:
                min_gap = atom_gap
        if min_gap is None:
            return float("inf")
        return float(min_gap)
