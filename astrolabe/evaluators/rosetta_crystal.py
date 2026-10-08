from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from astrolabe.core import global_config
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores


_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return dict(value or {})


class RosettaCrystalEvaluator(Evaluator):
    """Evaluate candidate sequences on immutable crystal Fab-ligand references."""

    def __init__(
        self,
        *,
        states: Mapping[str, Mapping[str, Any]],
        raw_model_results_dir: str | Path,
        primary_state: str = "minus2",
        protein_chains: list[str] | tuple[str, ...] = ("A", "B"),
        ligand_chain: str = "X",
        design_positions: Mapping[str, list[int]] | None = None,
        neighbor_distance: float = 8.0,
        replicates: int = 3,
        random_seed: int = 701,
        coordinate_constraint_weight: float = 1.0,
        protein_ca_sd: float = 0.3,
        ligand_heavy_atom_sd: float = 0.25,
        packer_nloop: int = 3,
        minimize_tolerance: float = 1e-4,
        minimize_max_iter: int = 200,
    ) -> None:
        self.states = {str(k): _plain_mapping(v) for k, v in dict(states).items()}
        if not self.states:
            raise ValueError("rosetta_crystal.states must define at least one state")
        self.primary_state = str(primary_state)
        if self.primary_state not in self.states:
            raise ValueError(
                f"primary_state {self.primary_state!r} is not in states {sorted(self.states)}"
            )
        self.output_dir = Path(raw_model_results_dir)
        self.protein_chains = tuple(str(c) for c in protein_chains)
        self.ligand_chain = str(ligand_chain)
        self.design_positions = {
            str(chain): tuple(int(pos) for pos in positions)
            for chain, positions in dict(design_positions or {}).items()
        }
        self.neighbor_distance = float(neighbor_distance)
        self.replicates = int(replicates)
        if self.replicates < 1:
            raise ValueError("rosetta_crystal.replicates must be positive")
        self.random_seed = int(random_seed)
        self.constraint_weight = float(coordinate_constraint_weight)
        self.protein_ca_sd = float(protein_ca_sd)
        self.ligand_heavy_atom_sd = float(ligand_heavy_atom_sd)
        self.packer_nloop = int(packer_nloop)
        if self.packer_nloop < 1:
            raise ValueError("rosetta_crystal.packer_nloop must be positive")
        self.minimize_tolerance = float(minimize_tolerance)
        self.minimize_max_iter = int(minimize_max_iter)
        self._pyrosetta = None
        self._reference_poses: dict[tuple[str, str], Any] = {}
        self._reference_metrics: dict[tuple[str, str], dict[str, Any]] = {}

    def _ensure_pyrosetta(self):
        if self._pyrosetta is not None:
            return self._pyrosetta
        import pyrosetta  # type: ignore

        params = []
        for state, cfg in self.states.items():
            path = Path(str(cfg.get("params", ""))).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"rosetta_crystal state {state} params missing: {path}")
            params.append(path)
        initialized = bool(
            hasattr(pyrosetta, "is_initialized") and pyrosetta.is_initialized()
        )
        if not initialized:
            opts = (
                "-ignore_zero_occupancy false -mute all -constant_seed "
                f"-jran {self.random_seed} -extra_res_fa "
                + " ".join(str(path) for path in params)
            )
            pyrosetta.init(opts)
        self._pyrosetta = pyrosetta
        return pyrosetta

    def _reference_path(self, state: str, oligo_name: str) -> Path:
        cfg = self.states[state]
        references = _plain_mapping(cfg.get("references"))
        raw = references.get(oligo_name, cfg.get("reference"))
        if raw is None:
            raise ValueError(
                f"rosetta_crystal state {state} lacks reference for {oligo_name}"
            )
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Crystal reference missing: {path}")
        return path

    def _reference_pose(self, state: str, oligo_name: str):
        key = (state, oligo_name)
        cached = self._reference_poses.get(key)
        if cached is not None:
            return cached.clone()
        pyrosetta = self._ensure_pyrosetta()
        pose = pyrosetta.pose_from_file(str(self._reference_path(state, oligo_name)))
        self._validate_pose(pose, state=state, oligo_name=oligo_name)
        self._reference_poses[key] = pose.clone()
        return pose

    def _validate_pose(self, pose, *, state: str, oligo_name: str) -> None:
        pdb_info = pose.pdb_info()
        protein_seen = {
            pdb_info.chain(i)
            for i in range(1, pose.total_residue() + 1)
            if pose.residue(i).is_protein()
        }
        missing = set(self.protein_chains).difference(protein_seen)
        if missing:
            raise ValueError(
                f"{state}/{oligo_name} reference misses protein chains {sorted(missing)}"
            )
        ligands = self._ligand_indices(pose)
        if len(ligands) != 1:
            raise ValueError(
                f"{state}/{oligo_name} must contain one ligand residue on chain "
                f"{self.ligand_chain}; found {ligands}"
            )
        if int(pose.residue(ligands[0]).nheavyatoms()) != 85:
            raise ValueError(
                f"{state}/{oligo_name} ligand must contain 85 heavy atoms"
            )

    def _ligand_indices(self, pose) -> list[int]:
        pdb_info = pose.pdb_info()
        return [
            i
            for i in range(1, pose.total_residue() + 1)
            if pdb_info.chain(i) == self.ligand_chain
            and not pose.residue(i).is_protein()
        ]

    @staticmethod
    def _chain_protein_indices(pose, chain_id: str) -> list[int]:
        pdb_info = pose.pdb_info()
        return [
            i
            for i in range(1, pose.total_residue() + 1)
            if pdb_info.chain(i) == chain_id and pose.residue(i).is_protein()
        ]

    @staticmethod
    def _vector1_length(values: Any) -> int:
        try:
            return int(len(values))
        except TypeError:
            return int(values.size())

    def _chain_interface_dg(self, pose, per_residue_data) -> dict[str, np.ndarray]:
        dg_values = per_residue_data.dG
        interface_residues = per_residue_data.interface_residues
        total_residue = int(pose.total_residue())
        if self._vector1_length(dg_values) != total_residue:
            raise ValueError(
                f"InterfaceAnalyzerMover returned {self._vector1_length(dg_values)} dG values "
                f"for pose with {total_residue} residues"
            )
        if self._vector1_length(interface_residues) != total_residue:
            raise ValueError(
                "InterfaceAnalyzerMover interface_residues length "
                f"{self._vector1_length(interface_residues)} != pose residues {total_residue}"
            )

        by_chain: dict[str, np.ndarray] = {}
        for chain_id in self.protein_chains:
            indices = self._chain_protein_indices(pose, chain_id)
            values = np.full(len(indices), np.nan, dtype=float)
            for local_index, pose_index in enumerate(indices):
                if not bool(interface_residues[pose_index]):
                    continue
                value = float(dg_values[pose_index])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite Rosetta per-residue interface dG at "
                        f"{chain_id}:{local_index + 1}"
                    )
                values[local_index] = value
            by_chain[chain_id] = values
        return by_chain

    def _target_sequences(self, input_obj: Any) -> dict[str, str]:
        sequences = {
            str(chain.id): str(chain.sequence)
            for chain in list(getattr(input_obj, "protein_chains", []) or [])
        }
        missing = set(self.protein_chains).difference(sequences)
        if missing:
            raise ValueError(f"Evaluator input misses protein chains {sorted(missing)}")
        return sequences

    def _apply_sequence(self, pose, target_sequences: Mapping[str, str]) -> list[int]:
        pyrosetta = self._ensure_pyrosetta()
        mutated: list[int] = []
        for chain_id in self.protein_chains:
            indices = self._chain_protein_indices(pose, chain_id)
            target = str(target_sequences[chain_id])
            if len(indices) != len(target):
                raise ValueError(
                    f"Chain {chain_id} reference length {len(indices)} != target length {len(target)}"
                )
            for index, amino_acid in zip(indices, target):
                current = str(pose.residue(index).name1())
                if current == amino_acid:
                    continue
                if amino_acid not in _ONE_TO_THREE:
                    raise ValueError(
                        f"Rosetta crystal evaluator only supports canonical mutations; "
                        f"got {amino_acid!r} at {chain_id}:{index}"
                    )
                mover = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                    int(index), _ONE_TO_THREE[amino_acid]
                )
                mover.apply(pose)
                mutated.append(index)
        return mutated

    def _fixed_pack_positions(self, pose) -> set[int]:
        positions: set[int] = set()
        for chain_id, local_positions in self.design_positions.items():
            indices = self._chain_protein_indices(pose, chain_id)
            for position in local_positions:
                if position < 1 or position > len(indices):
                    raise ValueError(
                        f"design position {chain_id}:{position} outside reference length {len(indices)}"
                    )
                positions.add(indices[position - 1])

        ligand_index = self._ligand_indices(pose)[0]
        ligand = pose.residue(ligand_index)
        threshold2 = self.neighbor_distance * self.neighbor_distance
        for index in range(1, pose.total_residue() + 1):
            residue = pose.residue(index)
            if not residue.is_protein():
                continue
            close = False
            for atom_index in range(1, residue.nheavyatoms() + 1):
                xyz = residue.xyz(atom_index)
                for ligand_atom in range(1, ligand.nheavyatoms() + 1):
                    delta = xyz - ligand.xyz(ligand_atom)
                    if float(delta.length_squared()) <= threshold2:
                        close = True
                        break
                if close:
                    break
            if close:
                positions.add(index)
        if not positions:
            raise ValueError("rosetta_crystal selected no protein positions for repacking")
        return positions

    def _add_coordinate_constraints(self, pose, reference_pose) -> None:
        pyrosetta = self._ensure_pyrosetta()
        AtomID = pyrosetta.rosetta.core.id.AtomID
        CoordinateConstraint = (
            pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint
        )
        HarmonicFunc = pyrosetta.rosetta.core.scoring.func.HarmonicFunc
        anchor_residue = self._chain_protein_indices(pose, self.protein_chains[0])[0]
        anchor_atom = AtomID(pose.residue(anchor_residue).atom_index("CA"), anchor_residue)
        for index in range(1, pose.total_residue() + 1):
            residue = pose.residue(index)
            reference = reference_pose.residue(index)
            if residue.is_protein():
                atom_names = ["CA"]
                sd = self.protein_ca_sd
            elif pose.pdb_info().chain(index) == self.ligand_chain:
                atom_names = [
                    residue.atom_name(atom).strip()
                    for atom in range(1, residue.nheavyatoms() + 1)
                ]
                sd = self.ligand_heavy_atom_sd
            else:
                continue
            for atom_name in atom_names:
                if not reference.has(atom_name) or not residue.has(atom_name):
                    raise ValueError(
                        f"Coordinate constraint atom {atom_name} missing at residue {index}"
                    )
                atom_id = AtomID(residue.atom_index(atom_name), index)
                pose.add_constraint(
                    CoordinateConstraint(
                        atom_id,
                        anchor_atom,
                        reference.xyz(atom_name),
                        HarmonicFunc(0.0, sd),
                    )
                )

    def _repack_and_minimize(self, pose, reference_pose, *, seed: int) -> None:
        pyrosetta = self._ensure_pyrosetta()
        pyrosetta.rosetta.numeric.random.rg().set_seed(int(seed))
        packable = self._fixed_pack_positions(reference_pose)
        pack_scorefxn = pyrosetta.create_score_function("ref2015")
        task = pyrosetta.standard_packer_task(pose)
        task.restrict_to_repacking()
        task.or_include_current(True)
        for index in range(1, pose.total_residue() + 1):
            if index not in packable:
                task.nonconst_residue_task(index).prevent_repacking()
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(
            pack_scorefxn, task
        )
        packer.nloop(self.packer_nloop)
        packer.apply(pose)

        self._add_coordinate_constraints(pose, reference_pose)
        move_map = pyrosetta.MoveMap()
        for index in packable:
            move_map.set_chi(index, True)
            move_map.set_bb(index, True)
        ligand_index = self._ligand_indices(pose)[0]
        move_map.set_chi(ligand_index, True)
        ligand_jump = int(pose.fold_tree().get_jump_that_builds_residue(ligand_index))
        if ligand_jump > 0:
            move_map.set_jump(ligand_jump, True)

        minimize_scorefxn = pyrosetta.create_score_function("ref2015_cart")
        minimize_scorefxn.set_weight(
            pyrosetta.rosetta.core.scoring.coordinate_constraint,
            self.constraint_weight,
        )
        minimizer = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
            move_map,
            minimize_scorefxn,
            "lbfgs_armijo_nonmonotone",
            self.minimize_tolerance,
            True,
        )
        minimizer.cartesian(True)
        minimizer.max_iter(self.minimize_max_iter)
        minimizer.apply(pose)
        pose.remove_constraints()

    @staticmethod
    def _rmsd(coords_a: list[list[float]], coords_b: list[list[float]]) -> float:
        a = np.asarray(coords_a, dtype=float)
        b = np.asarray(coords_b, dtype=float)
        if a.shape != b.shape or not a.size:
            raise ValueError(f"RMSD coordinate mismatch: {a.shape} versus {b.shape}")
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    def _geometry_metrics(self, pose, reference_pose) -> dict[str, float]:
        protein_now: list[list[float]] = []
        protein_ref: list[list[float]] = []
        for chain_id in self.protein_chains:
            for index in self._chain_protein_indices(pose, chain_id):
                protein_now.append(list(pose.residue(index).xyz("CA")))
                protein_ref.append(list(reference_pose.residue(index).xyz("CA")))
        ligand_index = self._ligand_indices(pose)[0]
        ligand_ref_index = self._ligand_indices(reference_pose)[0]
        ligand_now = pose.residue(ligand_index)
        ligand_ref = reference_pose.residue(ligand_ref_index)
        ligand_coords_now: list[list[float]] = []
        ligand_coords_ref: list[list[float]] = []
        for atom in range(1, ligand_now.nheavyatoms() + 1):
            atom_name = ligand_now.atom_name(atom).strip()
            ligand_coords_now.append(list(ligand_now.xyz(atom_name)))
            ligand_coords_ref.append(list(ligand_ref.xyz(atom_name)))
        return {
            "protein_ca_rmsd": self._rmsd(protein_now, protein_ref),
            "ligand_heavy_rmsd": self._rmsd(ligand_coords_now, ligand_coords_ref),
            "ligand_heavy_atoms": float(ligand_now.nheavyatoms()),
        }

    def _interaction_energy(self, pose, scorefxn) -> float:
        complex_energy = float(scorefxn(pose))
        ligand_index = self._ligand_indices(pose)[0]
        ligand_chain_index = int(pose.chain(ligand_index))
        ligand_pose = pose.split_by_chain(ligand_chain_index)
        protein_pose = pose.clone()
        protein_pose.delete_residue_slow(ligand_index)
        return complex_energy - float(scorefxn(protein_pose)) - float(scorefxn(ligand_pose))

    def _interface_metrics(
        self, pose, scorefxn
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        pyrosetta = self._ensure_pyrosetta()
        mover = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
        mover.set_interface(f"{''.join(self.protein_chains)}_{self.ligand_chain}")
        mover.set_scorefunction(scorefxn)
        mover.set_compute_packstat(True)
        mover.set_compute_interface_energy(True)
        mover.set_calc_dSASA(True)
        mover.set_calc_hbond_sasaE(True)
        mover.set_compute_interface_sc(True)
        mover.set_pack_separated(True)
        mover.apply(pose)
        data = mover.get_all_data()
        protein_chains = set(self.protein_chains)
        pdb_info = pose.pdb_info()
        protein_indices = [
            index
            for index in range(1, pose.total_residue() + 1)
            if str(pdb_info.chain(index)) in protein_chains
        ]
        selector_ns = pyrosetta.rosetta.core.select.residue_selector
        cms_filter = (
            pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
        )
        cms_filter.selector1(
            selector_ns.ResidueIndexSelector(",".join(map(str, protein_indices)))
        )
        cms_filter.selector2(selector_ns.ChainSelector(self.ligand_chain))
        cms_filter.distance_weight(0.5)
        cms_filter.use_rosetta_radii(True)
        metrics = {
            "interface_dg": float(mover.get_interface_dG()),
            "interface_dsasa": float(mover.get_interface_delta_sasa()),
            "shape_complementarity": float(data.sc_value),
            "contact_molecular_surface": float(cms_filter.report_sm(pose)),
            "interface_hbonds": float(data.interface_hbonds),
            "buried_unsat_hbonds": float(data.delta_unsat_hbonds),
            "packstat": float(data.packstat),
        }
        return metrics, self._chain_interface_dg(
            pose, mover.get_all_per_residue_data()
        )

    def _replicate_metrics(
        self, pose, reference_pose, *, state: str, oligo_name: str, replicate: int
    ) -> tuple[Any, dict[str, float]]:
        seed = self.random_seed + 1009 * replicate
        self._repack_and_minimize(pose, reference_pose, seed=seed)
        scorefxn = self._ensure_pyrosetta().create_score_function("ref2015")
        metrics = self._geometry_metrics(pose, reference_pose)
        metrics.update(
            {
                "total_score": float(scorefxn(pose)),
                "interaction_energy": self._interaction_energy(pose, scorefxn),
                "replicate": float(replicate),
                "seed": float(seed),
            }
        )
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite Rosetta metric for {state}/{oligo_name}")
        return pose, metrics

    def _evaluate_state(
        self,
        *,
        state: str,
        oligo_name: str,
        target_sequences: Mapping[str, str],
    ) -> dict[str, Any]:
        reference = self._reference_pose(state, oligo_name)
        replicate_rows: list[dict[str, float]] = []
        replicate_poses: list[Any] = []
        for replicate in range(self.replicates):
            pose = reference.clone()
            self._apply_sequence(pose, target_sequences)
            pose, metrics = self._replicate_metrics(
                pose,
                reference,
                state=state,
                oligo_name=oligo_name,
                replicate=replicate,
            )
            replicate_poses.append(pose)
            replicate_rows.append(metrics)

        reference_key = (state, oligo_name)
        reference_summary = self._reference_metrics.get(reference_key)
        if reference_summary is None:
            reference_rows: list[dict[str, float]] = []
            for replicate in range(self.replicates):
                reference_pose = reference.clone()
                _, metrics = self._replicate_metrics(
                    reference_pose,
                    reference,
                    state=state,
                    oligo_name=oligo_name,
                    replicate=replicate,
                )
                reference_rows.append(metrics)
            reference_summary = {
                "interaction_energy": float(
                    np.median([row["interaction_energy"] for row in reference_rows])
                ),
                "total_score": float(np.median([row["total_score"] for row in reference_rows])),
            }
            self._reference_metrics[reference_key] = reference_summary

        order = np.argsort([row["interaction_energy"] for row in replicate_rows])
        chosen_index = int(order[len(order) // 2])
        chosen_pose = replicate_poses[chosen_index]
        chosen_metrics = dict(replicate_rows[chosen_index])
        scorefxn = self._ensure_pyrosetta().create_score_function("ref2015")
        interface_metrics, interface_dg = self._interface_metrics(
            chosen_pose.clone(), scorefxn
        )
        chosen_metrics.update(interface_metrics)
        chosen_metrics["interaction_energy_median"] = float(
            np.median([row["interaction_energy"] for row in replicate_rows])
        )
        chosen_metrics["interaction_energy_std"] = float(
            np.std([row["interaction_energy"] for row in replicate_rows])
        )
        chosen_metrics["paired_interaction_dde"] = (
            chosen_metrics["interaction_energy_median"]
            - float(reference_summary["interaction_energy"])
        )
        chosen_metrics["paired_total_de"] = float(
            np.median([row["total_score"] for row in replicate_rows])
        ) - float(reference_summary["total_score"])
        chosen_metrics["chosen_replicate"] = float(chosen_index)
        if any(
            not math.isfinite(float(value))
            for value in chosen_metrics.values()
            if isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"Non-finite Rosetta interface metric for {state}/{oligo_name}")
        return {
            "pose": chosen_pose,
            "metrics": chosen_metrics,
            "replicates": replicate_rows,
            "per_chain_interface_dg": interface_dg,
        }

    def prepare_input(self, inputs, runtime_config):
        return {
            str(oligo_name): self._target_sequences(input_obj)
            for oligo_name, input_obj in inputs.items()
        }

    def run_function(self, input_feat):
        output: dict[str, Any] = {}
        for oligo_name, target_sequences in input_feat.items():
            states = {
                state: self._evaluate_state(
                    state=state,
                    oligo_name=oligo_name,
                    target_sequences=target_sequences,
                )
                for state in self.states
            }
            output[oligo_name] = states
        return output

    @staticmethod
    def _aggregate_state_metrics(
        state_metrics: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        rows = list(state_metrics.values())
        return {
            "robust_interaction_dde": float(
                max(row["paired_interaction_dde"] for row in rows)
            ),
            "robust_total_de": float(max(row["paired_total_de"] for row in rows)),
            "robust_interface_dg": float(max(row["interface_dg"] for row in rows)),
            "min_shape_complementarity": float(
                min(row["shape_complementarity"] for row in rows)
            ),
            "min_contact_molecular_surface": float(
                min(row["contact_molecular_surface"] for row in rows)
            ),
            "min_interface_hbonds": float(
                min(row["interface_hbonds"] for row in rows)
            ),
            "max_buried_unsat_hbonds": float(
                max(row["buried_unsat_hbonds"] for row in rows)
            ),
            "min_packstat": float(min(row["packstat"] for row in rows)),
            "min_interface_dsasa": float(
                min(row["interface_dsasa"] for row in rows)
            ),
            "max_ligand_heavy_rmsd": float(
                max(row["ligand_heavy_rmsd"] for row in rows)
            ),
            "max_protein_ca_rmsd": float(
                max(row["protein_ca_rmsd"] for row in rows)
            ),
            "max_interaction_energy_std": float(
                max(row["interaction_energy_std"] for row in rows)
            ),
        }

    @staticmethod
    def _aggregate_interface_dg_terms(
        state_terms: Mapping[str, Mapping[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        rows = list(state_terms.values())
        if not rows:
            raise ValueError("rosetta_crystal has no states to aggregate interface dG")

        chain_ids = set(rows[0])
        for row in rows[1:]:
            if set(row) != chain_ids:
                raise ValueError("rosetta_crystal state interface dG chains do not match")

        aggregated: dict[str, np.ndarray] = {}
        for chain_id in chain_ids:
            arrays = [np.asarray(row[chain_id], dtype=float) for row in rows]
            first_shape = arrays[0].shape
            if any(arr.shape != first_shape for arr in arrays):
                raise ValueError(
                    f"rosetta_crystal state interface dG shapes do not match for chain {chain_id}"
                )
            stacked = np.stack(arrays, axis=0)
            if np.isinf(stacked).any():
                raise ValueError(
                    f"rosetta_crystal state interface dG has infinite values for chain {chain_id}"
                )
            finite = np.isfinite(stacked)
            values = np.full(first_shape, np.nan, dtype=float)
            has_finite = np.any(finite, axis=0)
            if np.any(has_finite):
                values[has_finite] = np.max(
                    np.where(finite, stacked, -np.inf), axis=0
                )[has_finite]
            aggregated[chain_id] = values
        return aggregated

    def _chain_to_proto_names(self, oligo_name: str) -> dict[str, str]:
        oligo = str(oligo_name)
        if len(oligo) != len(self.protein_chains):
            raise ValueError(
                f"rosetta_crystal oligo {oligo!r} has {len(oligo)} protomers but "
                f"protein_chains has {len(self.protein_chains)} chains"
            )
        return dict(zip(self.protein_chains, oligo, strict=True))

    def prepare_output(self, output, inputs, runtime_config):
        step = int((runtime_config or {}).get("step", getattr(global_config, "step", 0)) or 0)
        structure_dir = Path(global_config.out_dir) / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)
        complex_paths: dict[str, Path] = {}
        metrics_by_oligo: dict[str, dict[str, Any]] = {}
        per_proto_score_terms: dict[str, dict[str, list[np.ndarray]]] = {}
        for oligo_name, states in output.items():
            primary = states[self.primary_state]
            path = structure_dir / f"step_{step:03d}_{oligo_name}_rosetta_crystal.pdb"
            primary["pose"].dump_pdb(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Rosetta evaluator did not write {path}")
            complex_paths[oligo_name] = path
            state_metrics = {
                state: dict(result["metrics"]) for state, result in states.items()
            }
            metrics_by_oligo[oligo_name] = self._aggregate_state_metrics(state_metrics)
            metrics_by_oligo[oligo_name]["states"] = state_metrics
            per_chain_interface_dg = self._aggregate_interface_dg_terms(
                {
                    state: result["per_chain_interface_dg"]
                    for state, result in states.items()
                }
            )
            chain_to_proto = self._chain_to_proto_names(str(oligo_name))
            for chain_id in self.protein_chains:
                proto_name = chain_to_proto[chain_id]
                per_proto_score_terms.setdefault(proto_name, {}).setdefault(
                    "rosetta_interface_dg",
                    [],
                ).append(np.asarray(per_chain_interface_dg[chain_id], dtype=float))
        scores = StandardScores(
            names=list(output),
            plddt={name: 0.0 for name in output},
            complex_paths=complex_paths,
            per_proto_score_terms=per_proto_score_terms,
            ptm={name: 0.0 for name in output},
            iptm={name: 0.0 for name in output},
        )
        scores.rosetta_metrics = metrics_by_oligo
        return scores
