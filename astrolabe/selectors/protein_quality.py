from __future__ import annotations

from collections import defaultdict
import logging
from typing import Mapping

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector
from astrolabe.utils.mmcif_ids import int_id_to_str_id

logger = logging.getLogger(__name__)


def _chain2proto_for_oligo(oligo_name: str) -> dict[str, str]:
    return {
        int_id_to_str_id(i + 1): proto
        for i, proto in enumerate(str(oligo_name))
        if proto in global_config.protomers
    }


def _empty_masks() -> dict[str, np.ndarray]:
    return {
        proto: np.zeros(global_config.protomers[proto].length, dtype=bool)
        for proto in global_config.protomers
    }


def _merge_proto_masks(
    proto2masks: Mapping[str, list[np.ndarray]],
    merge_func,
) -> dict[str, np.ndarray]:
    selected = _empty_masks()
    for proto, masks in proto2masks.items():
        if not masks:
            continue
        selected[proto] = merge_func(masks, axis=0)
    return selected


def _residue_subset_to_chain_masks(pose, subset, chain2proto: Mapping[str, str]):
    proto2masks = {}
    subset = np.asarray(list(subset), dtype=bool)
    for chain_index in range(1, pose.num_chains() + 1):
        chain_id = int_id_to_str_id(chain_index)
        proto = chain2proto.get(chain_id)
        if proto is None:
            continue
        begin = pose.chain_begin(chain_index)
        end = pose.chain_end(chain_index)
        mask = np.asarray(subset[begin - 1 : end], dtype=bool)
        if proto in proto2masks:
            if proto2masks[proto].shape != mask.shape:
                raise ValueError(
                    f"Repeated protomer {proto!r} maps to chains with different lengths."
                )
            proto2masks[proto] = np.logical_or(proto2masks[proto], mask)
        else:
            proto2masks[proto] = mask
    return proto2masks


def _pose_total_residue(pose) -> int:
    total = getattr(pose, "total_residue", None)
    if callable(total):
        return int(total())
    if total is not None:
        return int(total)
    return int(pose.size())


def _map_get(mapping, key, default=0.0):
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    try:
        return mapping[key]
    except Exception:
        return default


def _vector1_to_float_array(values, total_residue: int) -> np.ndarray:
    return np.asarray([float(values[idx]) for idx in range(1, total_residue + 1)])


def _compute_residue_packstat_scores(pose, oversample: int = 1):
    from pyrosetta.rosetta.core.scoring import packstat  # type: ignore

    return packstat.compute_residue_packing_scores(pose, int(oversample))


def _packstat_cutoff_from_holes_threshold(holes_threshold: float) -> float:
    return float(np.clip(0.45 - (float(holes_threshold) / 200.0), 0.05, 0.75))


def _is_dalphaball_runtime_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        token in text
        for token in (
            "DAlphaBall",
            "DALPHABALL",
            "libgfortran",
            "surf_vol",
            "compute_holes_score",
        )
    )


def _is_rosetta_filter_backend_error(exc: Exception) -> bool:
    text = str(exc)
    return _is_dalphaball_runtime_error(exc) or "unrecognized atom_type_name" in text


def _residue_name1(residue) -> str:
    name1 = getattr(residue, "name1", None)
    if callable(name1):
        try:
            return str(name1()).upper()
        except Exception:
            pass
    name = getattr(residue, "name", None)
    if callable(name):
        try:
            return str(name()).strip()[:1].upper()
        except Exception:
            pass
    return ""


def _surface_hydrophobic_proxy_scores(pose, total_residue: int) -> dict[int, float]:
    weights = {
        "F": 3.0,
        "W": 3.0,
        "Y": 2.7,
        "I": 2.7,
        "L": 2.7,
        "V": 2.4,
        "M": 2.4,
        "C": 2.0,
        "P": 1.6,
        "A": 1.4,
    }
    scores = {}
    for idx in range(1, total_residue + 1):
        residue = pose.residue(idx)
        if residue.is_protein():
            scores[idx] = weights.get(_residue_name1(residue), 0.0)
    return scores


def _per_residue_sap_scores(pose):
    from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import (  # type: ignore
        PerResidueSapScoreMetric,
    )

    return PerResidueSapScoreMetric().calculate(pose)


class SurfaceHighSAPPositionSelector(PositionSelector):
    def __init__(
        self,
        complex_poses: ComplexPoses,
        threshold: float = 1.0,
        merge: str = "any",
    ):
        self.threshold = float(threshold)
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._select()
        return self.sites

    def _select(self) -> dict[str, np.ndarray]:
        import pyrosetta  # type: ignore

        proto2masks = defaultdict(list)
        layer_selector = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_selector.set_layers(False, False, True)
        layer_selector.set_use_sc_neighbors(False)

        for oligo_name, pose in self.complex_poses.poses.items():
            chain2proto = _chain2proto_for_oligo(oligo_name)
            surface_subset = np.asarray(layer_selector.apply(pose), dtype=bool)
            total_residue = _pose_total_residue(pose)
            try:
                per_res_sap = _per_residue_sap_scores(pose)
            except RuntimeError as exc:
                if not _is_dalphaball_runtime_error(exc):
                    raise
                logger.warning(
                    "surface_high_sap SAP metric failed for %s (%s); using surface hydrophobic proxy.",
                    oligo_name,
                    exc,
                )
                per_res_sap = _surface_hydrophobic_proxy_scores(pose, total_residue)
            full_mask = np.zeros(total_residue, dtype=bool)
            for idx in range(1, total_residue + 1):
                full_mask[idx - 1] = bool(surface_subset[idx - 1]) and (
                    float(_map_get(per_res_sap, idx, 0.0)) >= self.threshold
                )
            for proto, mask in _residue_subset_to_chain_masks(
                pose,
                full_mask,
                chain2proto,
            ).items():
                proto2masks[proto].append(mask)
        return _merge_proto_masks(proto2masks, self.merge_func)


class BuriedUnsatPositionSelector(PositionSelector):
    def __init__(
        self,
        complex_poses: ComplexPoses,
        threshold: float = 0.5,
        ignore_surface_res: bool = True,
        method: str = "filter",
        max_filter_residues: int = 120,
        merge: str = "any",
    ):
        self.threshold = float(threshold)
        self.ignore_surface_res = bool(ignore_surface_res)
        self.method = str(method)
        self.max_filter_residues = int(max_filter_residues)
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._select()
        return self.sites

    def _residue_unsat_score(self, pose, residue_index: int) -> float:
        from pyrosetta.rosetta.core.select.residue_selector import (  # type: ignore
            ResidueIndexSelector,
        )
        from pyrosetta.rosetta.protocols.simple_filters import (  # type: ignore
            BuriedUnsatHbondFilter,
        )

        filt = BuriedUnsatHbondFilter()
        filt.set_report_all_heavy_atom_unsats(True)
        filt.set_ignore_surface_res(self.ignore_surface_res)
        filt.set_residue_selector(ResidueIndexSelector(str(residue_index)))
        return float(filt.report_sm(pose))

    def _buried_polar_proxy_mask(self, pose, total_residue: int) -> np.ndarray:
        import pyrosetta  # type: ignore

        layer_selector = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_selector.set_layers(True, True, False)
        layer_selector.set_use_sc_neighbors(False)
        buried_subset = np.asarray(layer_selector.apply(pose), dtype=bool)
        polar_or_charged = set("DEHKNQRSTY")
        full_mask = np.zeros(total_residue, dtype=bool)
        for idx in range(1, total_residue + 1):
            residue = pose.residue(idx)
            if not residue.is_protein():
                continue
            full_mask[idx - 1] = bool(buried_subset[idx - 1]) and (
                _residue_name1(residue) in polar_or_charged
            )
        return full_mask

    def _select(self) -> dict[str, np.ndarray]:
        if self.method not in {"filter", "buried_polar_proxy", "auto"}:
            raise ValueError(
                "buried_unsat method must be one of 'filter', 'buried_polar_proxy', or 'auto'."
            )
        proto2masks = defaultdict(list)
        for oligo_name, pose in self.complex_poses.poses.items():
            chain2proto = _chain2proto_for_oligo(oligo_name)
            total_residue = _pose_total_residue(pose)
            full_mask = np.zeros(total_residue, dtype=bool)
            use_proxy = self.method == "buried_polar_proxy" or (
                self.method == "auto" and total_residue > self.max_filter_residues
            )
            if use_proxy:
                if self.method == "auto":
                    logger.warning(
                        "buried_unsat uses buried_polar_proxy for %s because pose has %d residues "
                        "(max_filter_residues=%d).",
                        oligo_name,
                        total_residue,
                        self.max_filter_residues,
                    )
                full_mask = self._buried_polar_proxy_mask(pose, total_residue)
            else:
                try:
                    for idx in range(1, total_residue + 1):
                        residue = pose.residue(idx)
                        if not residue.is_protein():
                            continue
                        full_mask[idx - 1] = (
                            self._residue_unsat_score(pose, idx) >= self.threshold
                        )
                except RuntimeError as exc:
                    if not _is_rosetta_filter_backend_error(exc):
                        raise
                    logger.warning(
                        "buried_unsat filter backend failed for %s (%s); using buried_polar_proxy.",
                        oligo_name,
                        exc,
                    )
                    full_mask = self._buried_polar_proxy_mask(pose, total_residue)
            for proto, mask in _residue_subset_to_chain_masks(
                pose,
                full_mask,
                chain2proto,
            ).items():
                proto2masks[proto].append(mask)
        return _merge_proto_masks(proto2masks, self.merge_func)


class CavityNeighborPositionSelector(PositionSelector):
    def __init__(
        self,
        complex_poses: ComplexPoses,
        holes_threshold: float = 25.0,
        neighbor_distance: float = 6.0,
        method: str = "auto",
        max_holes_filter_residues: int = 200,
        packstat_cutoff: float | None = None,
        merge: str = "any",
    ):
        self.holes_threshold = float(holes_threshold)
        self.neighbor_distance = float(neighbor_distance)
        self.method = str(method)
        self.max_holes_filter_residues = int(max_holes_filter_residues)
        self.packstat_cutoff = (
            None if packstat_cutoff is None else float(packstat_cutoff)
        )
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._select()
        return self.sites

    def _protein_focus_candidates(self, pose, chain2proto: Mapping[str, str]):
        total_residue = _pose_total_residue(pose)
        for idx in range(1, total_residue + 1):
            residue = pose.residue(idx)
            chain_id = int_id_to_str_id(pose.chain(idx))
            proto = chain2proto.get(chain_id)
            protomer = global_config.protomers.get(proto)
            if (
                protomer is None
                or protomer.type != ProtomerTypes.PROTEIN
                or not residue.is_protein()
            ):
                continue
            yield idx

    def _holes_filter_focus_indices(self, pose, chain2proto: Mapping[str, str]):
        from pyrosetta.rosetta.core.select.residue_selector import (  # type: ignore
            ResidueIndexSelector,
        )
        from pyrosetta.rosetta.protocols.simple_filters import HolesFilter  # type: ignore

        focus_indices = []
        for idx in self._protein_focus_candidates(pose, chain2proto):
            holes = HolesFilter()
            holes.set_residue_selector(ResidueIndexSelector(str(idx)))
            score = float(holes.report_sm(pose))
            if score >= self.holes_threshold:
                focus_indices.append(str(idx))
        return focus_indices

    def _packstat_focus_indices(self, pose, chain2proto: Mapping[str, str]):
        total_residue = _pose_total_residue(pose)
        scores = _vector1_to_float_array(
            _compute_residue_packstat_scores(pose),
            total_residue,
        )
        cutoff = (
            self.packstat_cutoff
            if self.packstat_cutoff is not None
            else _packstat_cutoff_from_holes_threshold(self.holes_threshold)
        )
        focus_indices = []
        for idx in self._protein_focus_candidates(pose, chain2proto):
            if scores[idx - 1] <= cutoff:
                focus_indices.append(str(idx))
        return focus_indices

    def _focus_indices(self, oligo_name: str, pose, chain2proto: Mapping[str, str]):
        total_residue = _pose_total_residue(pose)
        method = self.method
        if method not in {"auto", "holes_filter", "packstat"}:
            raise ValueError(
                "cavity_neighbor method must be one of 'auto', 'holes_filter', or 'packstat'."
            )
        if method == "packstat":
            return self._packstat_focus_indices(pose, chain2proto)
        use_holes_filter = (
            method == "holes_filter" or total_residue <= self.max_holes_filter_residues
        )
        if use_holes_filter:
            try:
                return self._holes_filter_focus_indices(pose, chain2proto)
            except RuntimeError as exc:
                if method == "holes_filter":
                    raise
                logger.warning(
                    "cavity_neighbor HolesFilter failed for %s (%s); using packstat proxy.",
                    oligo_name,
                    exc,
                )
        else:
            logger.warning(
                "cavity_neighbor uses packstat proxy for %s because pose has %d residues "
                "(max_holes_filter_residues=%d).",
                oligo_name,
                total_residue,
                self.max_holes_filter_residues,
            )
        return self._packstat_focus_indices(pose, chain2proto)

    def _select(self) -> dict[str, np.ndarray]:
        from pyrosetta.rosetta.core.select.residue_selector import (  # type: ignore
            NeighborhoodResidueSelector,
            ResidueIndexSelector,
        )

        proto2masks = defaultdict(list)
        for oligo_name, pose in self.complex_poses.poses.items():
            chain2proto = _chain2proto_for_oligo(oligo_name)
            focus_indices = self._focus_indices(oligo_name, pose, chain2proto)
            if not focus_indices:
                continue
            focus = ResidueIndexSelector(",".join(focus_indices))
            neighborhood = NeighborhoodResidueSelector(
                focus,
                self.neighbor_distance,
                True,
            )
            full_mask = np.asarray(neighborhood.apply(pose), dtype=bool)
            for proto, mask in _residue_subset_to_chain_masks(
                pose,
                full_mask,
                chain2proto,
            ).items():
                proto2masks[proto].append(mask)
        return _merge_proto_masks(proto2masks, self.merge_func)
