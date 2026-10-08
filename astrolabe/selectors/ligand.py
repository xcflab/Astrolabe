from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Union

import numpy as np
from astrolabe.utils.mmcif_ids import int_id_to_str_id, str_id_to_int_id

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.core.repeat_symmetry import (
    canonicalize_exact_cyclic_cn,
    cyclic_oligomer_atom_bundles,
    load_structure,
    protein_residues,
)
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector
from resources.ligand_hbond_filter import LigandHBondFilter


class _LigandSelectionConfig(object):
    """Container describing how ligand chains should be identified."""

    def __init__(
        self,
        protomer_names: Set[str],
        chain_ids: Set[str],
        default_types: Iterable[ProtomerTypes],
    ):
        self.protomer_names = protomer_names
        self.chain_ids = chain_ids
        self.default_types = frozenset(default_types)

    @classmethod
    def from_user_input(
        cls,
        user_targets: Optional[Union[str, Sequence[str]]],
        default_types: Iterable[ProtomerTypes],
    ):
        if user_targets is None:
            return cls(set(), set(), default_types)

        if isinstance(user_targets, str):
            iterable_targets = [user_targets]
        else:
            iterable_targets = list(user_targets)

        protomer_names: Set[str] = set()
        chain_ids: Set[str] = set()

        for target in iterable_targets:
            if target in global_config.protomers:
                protomer_names.add(target)
            else:
                chain_ids.add(target)

        return cls(protomer_names, chain_ids, default_types)


class LigandUnsatisfiedPolarPositionSelector(PositionSelector):
    """
    Selects protein residues surrounding unsatisfied polar atoms in ligand chains.

    The selector analyses ligand hydrogen-bond satisfaction using ``LigandHBondFilter``
    and marks protein residues within a configurable distance of polar heavy atoms
    that are not currently engaged in hydrogen bonds.
    """

    def __init__(
        self,
        complex_poses: ComplexPoses,
        ligand_targets: Optional[Union[str, Sequence[str]]] = None,
        distance_min: float = 4.5,
        distance_max: float = 7.0,
        use_cb: bool = True,
        exclude_hbond_residues: bool = True,
        polar_elements: Sequence[str] = ("N", "O", "S"),
        merge: str = "any",
    ):
        """
        Args:
            complex_poses: Poses produced by the evaluator for each oligomer key.
            ligand_targets: Optional protomer or chain identifiers to force the selector
                to treat them as ligand chains. Entries matching protomer names are
                interpreted as protomers; any other string is assumed to be a pose
                chain identifier (e.g., 'C'). When omitted, the selector defaults to
                protomers whose type is SMILES or CCD.
            distance_min: Minimum distance threshold (Å) for neighbouring residues.
            distance_max: Maximum distance threshold (Å) for neighbouring residues.
            use_cb: Whether the underlying filter should use CB atoms (GLY uses CA).
            exclude_hbond_residues: If True, residues already forming a hydrogen bond
                with the ligand are ignored.
            polar_elements: Elements that should be considered polar heavy atoms.
            merge: Merge strategy inherited from :class:`PositionSelector`.
        """
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.use_cb = use_cb
        self.exclude_hbond_residues = exclude_hbond_residues
        self.polar_elements = tuple(polar_elements)
        self.ligand_config = _LigandSelectionConfig.from_user_input(
            ligand_targets,
            default_types=(ProtomerTypes.SMILES, ProtomerTypes.CCD),
        )

        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._compute_unsatisfied_environment_masks()
        return self.sites

    def _compute_unsatisfied_environment_masks(self) -> Dict[str, np.ndarray]:
        selected_sites: Dict[str, np.ndarray] = {
            proto: np.zeros(
                global_config.protomers[proto].length, dtype=bool
            )
            for proto in global_config.protomers
        }

        for oligo_name, pose in self.complex_poses.poses.items():
            chain2proto = {
                int_id_to_str_id(i + 1): proto
                for i, proto in enumerate(oligo_name)
                if proto in global_config.protomers
            }

            ligand_chains = self._resolve_ligand_chains(chain2proto)
            if not ligand_chains:
                continue

            for chain_id in ligand_chains:
                self._accumulate_unsatisfied_residue_masks(
                    pose, chain_id, chain2proto, selected_sites
                )

        return selected_sites

    def _resolve_ligand_chains(self, chain2proto: Mapping[str, str]) -> Set[str]:
        if self.ligand_config.protomer_names or self.ligand_config.chain_ids:
            return {
                chain_id
                for chain_id, proto in chain2proto.items()
                if proto in self.ligand_config.protomer_names
                or chain_id in self.ligand_config.chain_ids
            }

        return {
            chain_id
            for chain_id, proto in chain2proto.items()
            if global_config.protomers[proto].type
            in self.ligand_config.default_types
        }

    def _accumulate_unsatisfied_residue_masks(
        self,
        pose,
        ligand_chain: str,
        chain2proto: Mapping[str, str],
        selected_sites: Mapping[str, np.ndarray],
    ) -> None:
        ligand_filter = LigandHBondFilter(
            pose=pose,
            ligand_chain=ligand_chain,
            distance_min=self.distance_min,
            distance_max=self.distance_max,
            use_cb=self.use_cb,
            exclude_hbond_residues=self.exclude_hbond_residues,
        )

        if not ligand_filter.ligand_residues:
            return

        unsatisfied_atoms = self._get_unsatisfied_polar_atoms(pose, ligand_filter)

        for ligand_res_idx, atom_name in unsatisfied_atoms:
            eligible_residues = ligand_filter.get_eligible_residues_for_group(
                atom_name, ligand_res_idx=ligand_res_idx
            )
            self._mark_selected_sites(
                pose, eligible_residues, chain2proto, selected_sites
            )

    def _get_unsatisfied_polar_atoms(
        self, pose, ligand_filter: LigandHBondFilter
    ) -> Set[tuple]:
        unsatisfied: Set[tuple] = set()
        satisfied_atoms = ligand_filter.hbond_satisfied_atoms

        for res_idx in ligand_filter.ligand_residues:
            residue = pose.residue(res_idx)
            for atom_idx in range(1, residue.natoms() + 1):
                atom_type = residue.atom_type(atom_idx)
                if atom_type.element() == "H":
                    continue
                if atom_type.element() not in self.polar_elements:
                    continue

                atom_name = residue.atom_name(atom_idx).strip()
                if (res_idx, atom_name) in satisfied_atoms:
                    continue
                unsatisfied.add((res_idx, atom_name))

        return unsatisfied

    def _mark_selected_sites(
        self,
        pose,
        eligible_residues: Sequence[Mapping[str, Union[float, int, str]]],
        chain2proto: Mapping[str, str],
        selected_sites: Mapping[str, np.ndarray],
    ) -> None:
        if not eligible_residues:
            return

        for residue_info in eligible_residues:
            residue_index = int(residue_info["residue_num"])
            chain_idx = pose.chain(residue_index)
            chain_id = int_id_to_str_id(chain_idx)

            proto = chain2proto.get(chain_id)
            if proto is None:
                continue

            protomer = global_config.protomers.get(proto)
            if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
                continue

            relative_index = residue_index - pose.chain_begin(chain_idx)
            if not 0 <= relative_index < selected_sites[proto].shape[0]:
                continue

            selected_sites[proto][relative_index] = True


class LigandContactPositionSelector(PositionSelector):
    """Select protein residues currently contacting ligand chains."""

    def __init__(
        self,
        complex_poses: ComplexPoses,
        ligand_targets: Optional[Union[str, Sequence[str]]] = None,
        distance_min: float = 0.0,
        distance_max: float = 5.0,
        protein_elements: Sequence[str] = ("C", "N", "O", "S", "P"),
        ligand_elements: Sequence[str] = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I"),
        merge: str = "any",
    ):
        if distance_min < 0:
            raise ValueError("distance_min must be >= 0")
        if distance_max <= 0:
            raise ValueError("distance_max must be > 0")
        if distance_min > distance_max:
            raise ValueError("distance_min must be <= distance_max")

        self.distance_min = float(distance_min)
        self.distance_max = float(distance_max)
        self.protein_elements = frozenset(str(x) for x in protein_elements)
        self.ligand_elements = frozenset(str(x) for x in ligand_elements)
        self.ligand_config = _LigandSelectionConfig.from_user_input(
            ligand_targets,
            default_types=(ProtomerTypes.SMILES, ProtomerTypes.CCD),
        )
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._compute_contact_masks()
        return self.sites

    def _compute_contact_masks(self) -> Dict[str, np.ndarray]:
        selected_sites: Dict[str, np.ndarray] = {
            proto: np.zeros(global_config.protomers[proto].length, dtype=bool)
            for proto in global_config.protomers
        }

        for oligo_name, pose in self.complex_poses.poses.items():
            chain2proto = {
                int_id_to_str_id(i + 1): proto
                for i, proto in enumerate(oligo_name)
                if proto in global_config.protomers
            }
            ligand_chains = self._resolve_ligand_chains(chain2proto)
            if not ligand_chains:
                continue
            ligand_residues = self._collect_chain_residue_indices(pose, ligand_chains)
            ligand_coords = self._collect_residue_heavy_atom_coords(
                pose,
                ligand_residues,
                allowed_elements=self.ligand_elements,
            )
            if ligand_coords.size == 0:
                continue

            for chain_id, proto in chain2proto.items():
                protomer = global_config.protomers.get(proto)
                if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
                    continue
                chain_index = self._chain_id_to_index(chain_id)
                chain_begin = pose.chain_begin(chain_index)
                chain_end = pose.chain_end(chain_index)
                for residue_index in range(chain_begin, chain_end + 1):
                    residue = pose.residue(residue_index)
                    if not residue.is_protein():
                        continue
                    residue_coords = self._collect_residue_heavy_atom_coords(
                        pose,
                        [residue_index],
                        allowed_elements=self.protein_elements,
                    )
                    if residue_coords.size == 0:
                        continue
                    min_distance = self._min_pairwise_distance(residue_coords, ligand_coords)
                    if self.distance_min <= min_distance <= self.distance_max:
                        rel_index = residue_index - chain_begin
                        if 0 <= rel_index < selected_sites[proto].shape[0]:
                            selected_sites[proto][rel_index] = True

        return selected_sites

    def _resolve_ligand_chains(self, chain2proto: Mapping[str, str]) -> Set[str]:
        if self.ligand_config.protomer_names or self.ligand_config.chain_ids:
            return {
                chain_id
                for chain_id, proto in chain2proto.items()
                if proto in self.ligand_config.protomer_names
                or chain_id in self.ligand_config.chain_ids
            }
        return {
            chain_id
            for chain_id, proto in chain2proto.items()
            if global_config.protomers[proto].type in self.ligand_config.default_types
        }

    @staticmethod
    def _chain_id_to_index(chain_id: str) -> int:
        return str_id_to_int_id(str(chain_id).strip().upper())

    @staticmethod
    def _collect_chain_residue_indices(pose, chain_ids: Iterable[str]) -> list[int]:
        residue_indices: list[int] = []
        for chain_id in chain_ids:
            chain_index = LigandContactPositionSelector._chain_id_to_index(chain_id)
            residue_indices.extend(
                range(pose.chain_begin(chain_index), pose.chain_end(chain_index) + 1)
            )
        return residue_indices

    @staticmethod
    def _xyz_to_numpy(xyz) -> np.ndarray:
        if isinstance(xyz, np.ndarray):
            return xyz.astype(float, copy=False)
        if hasattr(xyz, "x") and hasattr(xyz, "y") and hasattr(xyz, "z"):
            return np.array([float(xyz.x), float(xyz.y), float(xyz.z)], dtype=float)
        if hasattr(xyz, "__iter__"):
            return np.array(list(xyz), dtype=float)
        raise TypeError(f"Unsupported coordinate object: {type(xyz)!r}")

    @classmethod
    def _collect_residue_heavy_atom_coords(
        cls,
        pose,
        residue_indices: Iterable[int],
        *,
        allowed_elements: frozenset[str],
    ) -> np.ndarray:
        coords = []
        for residue_index in residue_indices:
            residue = pose.residue(residue_index)
            for atom_index in range(1, residue.natoms() + 1):
                element = str(residue.atom_type(atom_index).element()).strip()
                if element == "H" or element not in allowed_elements:
                    continue
                coords.append(cls._xyz_to_numpy(residue.xyz(atom_index)))
        if not coords:
            return np.zeros((0, 3), dtype=float)
        return np.stack(coords, axis=0)

    @staticmethod
    def _min_pairwise_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
        deltas = coords_a[:, None, :] - coords_b[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        return float(np.min(distances))


class MembraneOuterSurfacePositionSelector(PositionSelector):
    """
    Select lipid-facing residues in a membrane slab from the current cyclic oligomer.
    """

    def __init__(
        self,
        complex_poses: ComplexPoses,
        membrane_thickness: float = 30.0,
        outer_surface_depth: float = 5.0,
        axial_bin_size: float = 4.0,
        radial_atom: str = "CA",
        merge: str = "any",
    ):
        self.membrane_thickness = float(membrane_thickness)
        self.outer_surface_depth = float(outer_surface_depth)
        self.axial_bin_size = float(axial_bin_size)
        self.radial_atom = str(radial_atom or "CA").strip().upper()
        if self.membrane_thickness <= 0.0:
            raise ValueError("membrane_thickness must be positive.")
        if self.outer_surface_depth < 0.0:
            raise ValueError("outer_surface_depth must be non-negative.")
        if self.axial_bin_size <= 0.0:
            raise ValueError("axial_bin_size must be positive.")
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **_) -> Mapping[str, np.ndarray]:
        self.sites = self._compute_membrane_outer_surface_masks()
        return self.sites

    def _compute_membrane_outer_surface_masks(self) -> Mapping[str, np.ndarray]:
        proto2masks: dict[str, list[np.ndarray]] = defaultdict(list)
        complex_paths = getattr(self.complex_poses, "complex_paths", None) or {}
        oligo_names = list(self.complex_poses.poses.keys())
        for oligo_name in complex_paths:
            if oligo_name not in oligo_names:
                oligo_names.append(oligo_name)
        for oligo_name in oligo_names:
            structure_path = complex_paths.get(oligo_name)
            if structure_path is None:
                pose = self.complex_poses.poses.get(oligo_name)
                if pose is None:
                    raise ValueError(
                        f"No pose or structure path available for oligomer {oligo_name!r}."
                    )
                structure_path = self._dump_pose_for_selector(str(oligo_name), pose)
            per_proto = self._select_from_structure_path(
                oligo_name=str(oligo_name),
                structure_path=Path(structure_path),
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

    @staticmethod
    def _dump_pose_for_selector(oligo_name: str, pose) -> Path:
        output_dir = Path(global_config.out_dir) / "raw_model_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_path = (
            output_dir
            / f"membrane_selector_step_{int(getattr(global_config, 'step', 0) or 0):03d}_{oligo_name}.pdb"
        )
        pose.dump_pdb(str(structure_path))
        return structure_path

    def _select_from_structure_path(
        self,
        *,
        oligo_name: str,
        structure_path: Path,
    ) -> Mapping[str, np.ndarray]:
        structure = load_structure(structure_path)
        occurrences, coord_sets, _fit_bundles, _movable_bundles = (
            cyclic_oligomer_atom_bundles(structure, oligo_name)
        )
        canonicalization = canonicalize_exact_cyclic_cn(coord_sets)
        axis = np.asarray(canonicalization.axis_direction, dtype=float)
        axis = axis / float(np.linalg.norm(axis))
        all_fit_coords = np.concatenate(
            [np.asarray(coords, dtype=float) for coords in coord_sets],
            axis=0,
        )
        all_s = (all_fit_coords - canonicalization.axis_point[None, :]) @ axis
        axis_mid_s = 0.5 * (float(np.min(all_s)) + float(np.max(all_s)))
        half_thickness = 0.5 * self.membrane_thickness

        residues_by_occurrence = [
            protein_residues(structure, occurrence.chain_id)
            for occurrence in occurrences
        ]
        residue_records: list[tuple[str, int, float, float]] = []
        for occurrence, residues in zip(occurrences, residues_by_occurrence, strict=True):
            for idx, residue in enumerate(residues):
                coord = self._residue_radial_coord(residue)
                axial_s = float(np.dot(coord - canonicalization.axis_point, axis))
                rel_s = axial_s - axis_mid_s
                if abs(rel_s) > half_thickness:
                    continue
                radial_vec = coord - (canonicalization.axis_point + axial_s * axis)
                radial = float(np.linalg.norm(radial_vec))
                residue_records.append((occurrence.protomer, idx, rel_s, radial))

        if not residue_records:
            return {
                proto: np.zeros(protomer.length, dtype=bool)
                for proto, protomer in global_config.protomers.items()
                if protomer.type == ProtomerTypes.PROTEIN
            }

        bin_to_max_radius: dict[int, float] = {}
        for _proto, _idx, rel_s, radial in residue_records:
            bin_id = int(np.floor((rel_s + half_thickness) / self.axial_bin_size))
            current = bin_to_max_radius.get(bin_id)
            if current is None or radial > current:
                bin_to_max_radius[bin_id] = radial

        per_proto: dict[str, np.ndarray] = {}
        for occurrence in occurrences:
            protomer = global_config.protomers.get(occurrence.protomer)
            if protomer is not None and protomer.type == ProtomerTypes.PROTEIN:
                per_proto.setdefault(
                    occurrence.protomer,
                    np.zeros(protomer.length, dtype=bool),
                )

        for proto, idx, rel_s, radial in residue_records:
            if proto not in per_proto or idx >= per_proto[proto].shape[0]:
                continue
            bin_id = int(np.floor((rel_s + half_thickness) / self.axial_bin_size))
            outer_radius = float(bin_to_max_radius[bin_id])
            if radial >= outer_radius - self.outer_surface_depth:
                per_proto[proto][idx] = True
        return per_proto

    def _residue_radial_coord(self, residue) -> np.ndarray:
        if self.radial_atom in residue:
            return np.asarray(residue[self.radial_atom].coord, dtype=float)
        raise ValueError(
            f"Residue {residue.get_full_id()} is missing radial_atom {self.radial_atom!r}."
        )
