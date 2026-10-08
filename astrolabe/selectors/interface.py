from collections import defaultdict

import numpy as np
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id
from pyrosetta.rosetta.protocols.interface import select_interface_residues

from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class InterfacePositionSelector(PositionSelector):
    _SUPPORTED_MODES = (
        "heavy_atom",
        "buried_sasa",
        "rosetta",
        "ca",
        "sidechain_heavy",
        "sasa",
    )

    def __init__(
        self,
        complex_poses: ComplexPoses,
        merge="any",
        interface_distance: float = 8.0,
        mode: str = "heavy_atom",
        buried_sasa_cutoff: float = 10.0,
        sasa_cutoff: float = 0.1,
        sasa_sidechain_only: bool = True,
    ):
        """Select interface residues within a cutoff distance (Å)."""
        self.interface_distance = interface_distance
        self.buried_sasa_cutoff = buried_sasa_cutoff
        self.mode = self._normalize_mode(mode)
        self.sasa_cutoff = float(sasa_cutoff)
        if self.mode == "sasa" and not 0.0 <= self.sasa_cutoff <= 1.0:
            raise ValueError(
                "sasa_cutoff must be between 0 and 1 for relative SASA mode."
            )
        self.sasa_sidechain_only = bool(sasa_sidechain_only)
        super().__init__(complex_poses, merge=merge)

    def select_positions(self):
        self.sites = self.interface_selector(
            self.complex_poses.poses, self.merge_func
        )
        return self.sites

    def interface_selector(self, poses, merge_func=np.any):
        proto2interface = defaultdict(list)
        # todo: handle multiple chains interface selection
        for oligo_name, pose in poses.items():
            # Get all chain pairs to check interfaces between
            chain2proto = {int_id_to_str_id(i+1):proto for i, proto in enumerate(oligo_name)}
            n_chains = pose.num_chains()
            for i in range(1, n_chains):
                for j in range(i + 1, n_chains + 1):
                    chain_i = get_chain_from_chain_id(i, pose)
                    chain_j = get_chain_from_chain_id(j, pose)
                    interface_sele = self._select_pair_interface(
                        pose,
                        chain_i,
                        chain_j,
                        chain_i_index=i,
                        chain_j_index=j,
                    )

                    # Map interface residues back to each chain
                    res_idx = 0
                    for chain_id in range(1, n_chains + 1):
                        chain_name = chain2proto[
                            get_chain_from_chain_id(chain_id, pose)
                        ]
                        chain_length = pose.chain_end(chain_id) - (
                            pose.chain_begin(chain_id) - 1
                        )
                        proto2interface[chain_name].append(
                            interface_sele[res_idx : res_idx + chain_length]
                        )
                        res_idx += chain_length
        selected_sites = {}
        for proto, value in proto2interface.items():
            selected_sites[proto] = merge_func(value, axis=0)
        return selected_sites

    @classmethod
    def _normalize_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in {"heavy", "atom", "all_atom", "heavyatom"}:
            normalized = "heavy_atom"
        if normalized in {"buried", "buried_surface", "buried_surface_area", "delta_sasa"}:
            normalized = "buried_sasa"
        if normalized in {"pyrosetta", "legacy"}:
            normalized = "rosetta"
        if normalized in {"ca_ca", "c_alpha", "calpha"}:
            normalized = "ca"
        if normalized in {
            "sc",
            "sidechain",
            "side_chain",
            "sidechain_heavy_atom",
            "sidechain_heavy_atoms",
        }:
            normalized = "sidechain_heavy"
        if normalized in {"relative_sasa", "rel_sasa"}:
            normalized = "sasa"
        if normalized not in cls._SUPPORTED_MODES:
            supported = ", ".join(cls._SUPPORTED_MODES)
            raise ValueError(f"interface selector mode must be one of: {supported}")
        return normalized

    def _select_pair_interface(
        self,
        pose,
        chain_i: str,
        chain_j: str,
        *,
        chain_i_index: int,
        chain_j_index: int,
    ) -> np.ndarray:
        if self.mode == "heavy_atom":
            return self._select_heavy_atom_interface(
                pose, chain_i_index, chain_j_index
            )
        if self.mode == "buried_sasa":
            return self._select_buried_sasa_interface(
                pose, chain_i_index, chain_j_index
            )
        if self.mode == "rosetta":
            return self._select_rosetta_interface(pose, chain_i, chain_j)
        if self.mode == "ca":
            return self._select_ca_interface(pose, chain_i_index, chain_j_index)
        if self.mode == "sidechain_heavy":
            return self._select_sidechain_heavy_interface(
                pose, chain_i_index, chain_j_index
            )
        if self.mode == "sasa":
            return self._select_sasa_interface(pose, chain_i_index, chain_j_index)
        raise AssertionError(f"Unhandled interface selector mode: {self.mode}")

    def _select_rosetta_interface(self, pose, chain_i: str, chain_j: str) -> np.ndarray:
        interface_cutoff = max(1, int(round(self.interface_distance)))
        return np.array(
            select_interface_residues(
                pose, f"{chain_i}_{chain_j}", interface_cutoff
            ),
            dtype=bool,
        )

    def _select_ca_interface(
        self, pose, chain_i_index: int, chain_j_index: int
    ) -> np.ndarray:
        selected = np.zeros(self._pose_residue_count(pose), dtype=bool)
        residues_i, coords_i = self._collect_chain_ca_coords(pose, chain_i_index)
        residues_j, coords_j = self._collect_chain_ca_coords(pose, chain_j_index)
        if coords_i.size == 0 or coords_j.size == 0:
            return selected

        distances = np.linalg.norm(
            coords_i[:, None, :] - coords_j[None, :, :], axis=-1
        )
        contacts = distances <= float(self.interface_distance)
        if not np.any(contacts):
            return selected

        selected[np.array(residues_i, dtype=int)[np.any(contacts, axis=1)] - 1] = True
        selected[np.array(residues_j, dtype=int)[np.any(contacts, axis=0)] - 1] = True
        return selected

    def _select_heavy_atom_interface(
        self, pose, chain_i_index: int, chain_j_index: int
    ) -> np.ndarray:
        selected = np.zeros(self._pose_residue_count(pose), dtype=bool)
        residues_i = self._collect_chain_heavy_atom_coords(pose, chain_i_index)
        residues_j = self._collect_chain_heavy_atom_coords(pose, chain_j_index)
        if not residues_i or not residues_j:
            return selected

        cutoff = float(self.interface_distance)
        for residue_i_index, coords_i in residues_i:
            for residue_j_index, coords_j in residues_j:
                distances = np.linalg.norm(
                    coords_i[:, None, :] - coords_j[None, :, :], axis=-1
                )
                if np.any(distances <= cutoff):
                    selected[residue_i_index - 1] = True
                    selected[residue_j_index - 1] = True
        return selected

    def _select_buried_sasa_interface(
        self, pose, chain_i_index: int, chain_j_index: int
    ) -> np.ndarray:
        selected = np.zeros(self._pose_residue_count(pose), dtype=bool)
        delta_i, delta_j = self._pair_buried_sasa_values(
            pose, chain_i_index, chain_j_index
        )
        cutoff = float(self.buried_sasa_cutoff)

        begin_i = pose.chain_begin(chain_i_index) - 1
        end_i = pose.chain_end(chain_i_index)
        begin_j = pose.chain_begin(chain_j_index) - 1
        end_j = pose.chain_end(chain_j_index)
        selected[begin_i:end_i] = delta_i > cutoff
        selected[begin_j:end_j] = delta_j > cutoff
        return selected

    def _select_sidechain_heavy_interface(
        self, pose, chain_i_index: int, chain_j_index: int
    ) -> np.ndarray:
        selected = np.zeros(self._pose_residue_count(pose), dtype=bool)
        residues_i, coords_i, coord_to_res_i = self._collect_chain_sidechain_heavy_coords(
            pose, chain_i_index
        )
        residues_j, coords_j, coord_to_res_j = self._collect_chain_sidechain_heavy_coords(
            pose, chain_j_index
        )
        if coords_i.size == 0 or coords_j.size == 0:
            return selected

        distances = np.linalg.norm(
            coords_i[:, None, :] - coords_j[None, :, :], axis=-1
        )
        contacts = distances <= float(self.interface_distance)
        if not np.any(contacts):
            return selected

        contact_i = np.unique(coord_to_res_i[np.any(contacts, axis=1)])
        contact_j = np.unique(coord_to_res_j[np.any(contacts, axis=0)])
        selected[contact_i - 1] = True
        selected[contact_j - 1] = True
        return selected

    def _select_sasa_interface(
        self, pose, chain_i_index: int, chain_j_index: int
    ) -> np.ndarray:
        selected = np.zeros(self._pose_residue_count(pose), dtype=bool)
        sasa_bound, sasa_unbound = self._bound_and_unbound_residue_sasa(
            pose,
            sidechain_only=self.sasa_sidechain_only,
        )
        if sasa_bound.shape != selected.shape or sasa_unbound.shape != selected.shape:
            raise ValueError(
                "Bound/unbound SASA vectors do not align with pose residues."
            )

        if self.sasa_sidechain_only:
            relative_buried_sasa = np.maximum(0.0, sasa_unbound - sasa_bound)
        else:
            delta_sasa = np.maximum(0.0, sasa_unbound - sasa_bound)
            relative_buried_sasa = np.divide(
                delta_sasa,
                sasa_unbound,
                out=np.zeros_like(delta_sasa, dtype=float),
                where=sasa_unbound > 0.0,
            )
        buried = relative_buried_sasa >= self.sasa_cutoff
        for chain_index in (chain_i_index, chain_j_index):
            start = pose.chain_begin(chain_index) - 1
            end = pose.chain_end(chain_index)
            selected[start:end] = buried[start:end]
        return selected

    @staticmethod
    def _bound_and_unbound_residue_sasa(
        pose,
        *,
        sidechain_only: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        sasa_calculator = InterfacePositionSelector._per_residue_sasa_calculator(
            sidechain_only
        )
        sasa_bound = np.asarray(sasa_calculator(pose), dtype=float)
        sasa_unbound_parts = []
        for chain_pose in pose.split_by_chain():
            chain_sasa = np.asarray(sasa_calculator(chain_pose), dtype=float)
            sasa_unbound_parts.append(chain_sasa)
        sasa_unbound = (
            np.concatenate(sasa_unbound_parts)
            if sasa_unbound_parts
            else np.zeros_like(sasa_bound)
        )
        return sasa_bound, sasa_unbound

    @staticmethod
    def _per_residue_sasa_calculator(sidechain_only: bool):
        if sidechain_only:
            from pyrosetta.rosetta.core.scoring.sasa import rel_per_res_sc_sasa

            return rel_per_res_sc_sasa

        from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import (
            PerResidueSasaMetric,
        )

        sasa_metric = PerResidueSasaMetric()

        def calculate_pose_sasa(pose):
            per_res_sasa_dict = sasa_metric.calculate(pose)
            result = np.zeros(len(per_res_sasa_dict))
            for i, val in per_res_sasa_dict.items():
                result[i - 1] = val
            return result

        return calculate_pose_sasa

    @classmethod
    def _pair_buried_sasa_values(
        cls, pose, chain_i_index: int, chain_j_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        chain_poses = list(pose.split_by_chain())
        chain_i_pose = chain_poses[chain_i_index - 1]
        chain_j_pose = chain_poses[chain_j_index - 1]
        return cls._chain_pair_buried_sasa_values(chain_i_pose, chain_j_pose)

    @classmethod
    def _chain_pair_buried_sasa_values(cls, chain_i_pose, chain_j_pose):
        unbound_i = cls._pose_per_residue_sasa(chain_i_pose)
        unbound_j = cls._pose_per_residue_sasa(chain_j_pose)

        pair_pose = cls._make_pair_pose(chain_i_pose, chain_j_pose)
        bound_pair = cls._pose_per_residue_sasa(pair_pose)

        len_i = cls._pose_residue_count(chain_i_pose)
        len_j = cls._pose_residue_count(chain_j_pose)
        expected_len = len_i + len_j
        if bound_pair.shape != (expected_len,):
            raise ValueError(
                "Bound pair SASA vector does not align with chain lengths: "
                f"{bound_pair.shape} vs {(expected_len,)}."
            )
        if unbound_i.shape != (len_i,) or unbound_j.shape != (len_j,):
            raise ValueError(
                "Unbound chain SASA vectors do not align with chain lengths: "
                f"{unbound_i.shape}, {unbound_j.shape} vs {(len_i,)}, {(len_j,)}."
            )

        bound_i = bound_pair[:len_i]
        bound_j = bound_pair[len_i:]
        return (
            np.maximum(0.0, unbound_i - bound_i),
            np.maximum(0.0, unbound_j - bound_j),
        )

    @staticmethod
    def _pose_per_residue_sasa(pose) -> np.ndarray:
        from astrolabe.per_residue_scores import _per_residue_sasa

        return _per_residue_sasa(pose)

    @staticmethod
    def _make_pair_pose(chain_i_pose, chain_j_pose):
        from pyrosetta.rosetta.core.pose import append_pose_to_pose

        pair_pose = chain_i_pose.clone()
        append_pose_to_pose(pair_pose, chain_j_pose, True)
        return pair_pose

    @classmethod
    def _collect_chain_heavy_atom_coords(
        cls, pose, chain_index: int
    ) -> list[tuple[int, np.ndarray]]:
        residues = []
        for residue_index in range(
            pose.chain_begin(chain_index), pose.chain_end(chain_index) + 1
        ):
            coords = cls._residue_heavy_atom_coords(pose.residue(residue_index))
            if coords.size == 0:
                continue
            residues.append((residue_index, coords))
        return residues

    @classmethod
    def _collect_chain_sidechain_heavy_coords(
        cls, pose, chain_index: int
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        residue_indices = []
        coord_to_residue = []
        coords = []
        for residue_index in range(
            pose.chain_begin(chain_index), pose.chain_end(chain_index) + 1
        ):
            residue = pose.residue(residue_index)
            residue_coords = cls._residue_sidechain_heavy_coords(residue)
            if not residue_coords:
                continue
            residue_indices.append(residue_index)
            for coord in residue_coords:
                coords.append(coord)
                coord_to_residue.append(residue_index)
        if not coords:
            return residue_indices, np.zeros((0, 3), dtype=float), np.array([], dtype=int)
        return (
            residue_indices,
            np.stack(coords, axis=0),
            np.array(coord_to_residue, dtype=int),
        )

    @classmethod
    def _residue_sidechain_heavy_coords(cls, residue) -> list[np.ndarray]:
        if hasattr(residue, "is_protein") and not residue.is_protein():
            return []
        if not hasattr(residue, "natoms") or not hasattr(residue, "atom_name"):
            return []
        coords = []
        backbone_atoms = {"N", "CA", "C", "O", "OXT"}
        for atom_index in range(1, int(residue.natoms()) + 1):
            atom_name = str(residue.atom_name(atom_index)).strip()
            if atom_name in backbone_atoms:
                continue
            if cls._is_hydrogen_atom(residue, atom_index, atom_name):
                continue
            coords.append(cls._xyz_to_numpy(residue.xyz(atom_index)))
        return coords

    @classmethod
    def _residue_heavy_atom_coords(cls, residue) -> np.ndarray:
        if hasattr(residue, "is_protein") and not residue.is_protein():
            return np.zeros((0, 3), dtype=float)
        if not hasattr(residue, "natoms") or not hasattr(residue, "atom_name"):
            return np.zeros((0, 3), dtype=float)

        coords = []
        for atom_index in range(1, residue.natoms() + 1):
            atom_name = str(residue.atom_name(atom_index)).strip()
            if cls._is_hydrogen_atom(residue, atom_index, atom_name):
                continue
            coords.append(cls._xyz_to_numpy(residue.xyz(atom_index)))
        if not coords:
            return np.zeros((0, 3), dtype=float)
        return np.stack(coords, axis=0)

    @staticmethod
    def _is_hydrogen_atom(residue, atom_index: int, atom_name: str) -> bool:
        if hasattr(residue, "atom_is_hydrogen"):
            try:
                return bool(residue.atom_is_hydrogen(atom_index))
            except (AttributeError, TypeError):
                pass
        if hasattr(residue, "atom_type"):
            try:
                atom_type = residue.atom_type(atom_index)
                element = atom_type.element()
                if str(element).strip().upper() == "H":
                    return True
            except (AttributeError, TypeError):
                pass
        for character in atom_name.upper():
            if character.isalpha():
                return character == "H"
        return False

    @classmethod
    def _collect_chain_ca_coords(
        cls, pose, chain_index: int
    ) -> tuple[list[int], np.ndarray]:
        residue_indices = []
        coords = []
        for residue_index in range(
            pose.chain_begin(chain_index), pose.chain_end(chain_index) + 1
        ):
            ca_coord = cls._residue_ca_coord(pose.residue(residue_index))
            if ca_coord is None:
                continue
            residue_indices.append(residue_index)
            coords.append(ca_coord)
        if not coords:
            return residue_indices, np.zeros((0, 3), dtype=float)
        return residue_indices, np.stack(coords, axis=0)

    @classmethod
    def _residue_ca_coord(cls, residue):
        if hasattr(residue, "is_protein") and not residue.is_protein():
            return None
        if hasattr(residue, "has"):
            try:
                if residue.has("CA"):
                    return cls._xyz_to_numpy(residue.xyz("CA"))
                return None
            except TypeError:
                pass

        if not hasattr(residue, "natoms") or not hasattr(residue, "atom_name"):
            return None
        for atom_index in range(1, residue.natoms() + 1):
            if str(residue.atom_name(atom_index)).strip() == "CA":
                return cls._xyz_to_numpy(residue.xyz(atom_index))
        return None

    @staticmethod
    def _pose_residue_count(pose) -> int:
        if hasattr(pose, "total_residue"):
            return int(pose.total_residue())
        return int(pose.chain_end(pose.num_chains()))

    @staticmethod
    def _xyz_to_numpy(xyz) -> np.ndarray:
        if isinstance(xyz, np.ndarray):
            return xyz.astype(float, copy=False)
        if hasattr(xyz, "x") and hasattr(xyz, "y") and hasattr(xyz, "z"):
            x = xyz.x() if callable(xyz.x) else xyz.x
            y = xyz.y() if callable(xyz.y) else xyz.y
            z = xyz.z() if callable(xyz.z) else xyz.z
            return np.array([float(x), float(y), float(z)], dtype=float)
        if hasattr(xyz, "__iter__"):
            return np.array(list(xyz), dtype=float)
        raise TypeError(f"Unsupported coordinate object: {type(xyz)!r}")
