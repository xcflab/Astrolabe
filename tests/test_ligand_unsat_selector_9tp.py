"""Optional integration test for `ligand_unsat` selector on 9TP.

This test requires PyRosetta + license. It is skipped automatically when
PyRosetta is unavailable.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.ligand import LigandUnsatisfiedPolarPositionSelector
from astrolabe.utils.mmcif_ids import int_id_to_str_id
from resources.ligand_hbond_filter import LigandHBondFilter


class _SimpleProtomer:
    """Minimal protomer wrapper providing the attributes used by selectors."""

    def __init__(self, length: int, proto_type: ProtomerTypes):
        self.length = int(length)
        self.type = proto_type
        self._mutable_mask = np.ones(self.length, dtype=bool)

    @property
    def mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def aggregate_bool(self, mask: np.ndarray) -> np.ndarray:
        return mask


class TestLigandUnsatSelector9TP(unittest.TestCase):
    def _try_import_pyrosetta(self):
        try:
            import pyrosetta  # type: ignore
        except ImportError:
            self.skipTest("PyRosetta not available")
        return pyrosetta

    def _ensure_pyrosetta_initialized(self, pyrosetta) -> None:
        if hasattr(pyrosetta, "is_initialized") and pyrosetta.is_initialized():
            return
        pyrosetta.init("-mute all")

    def _load_pose(self, pyrosetta):
        cif_path = Path(__file__).resolve().parents[1] / "examples" / "input" / "9TP_cid.cif"
        if not cif_path.exists():
            self.skipTest(f"Missing example structure: {cif_path}")
        return pyrosetta.rosetta.core.import_pose.pose_from_file(str(cif_path))

    def _configure_protomers(
        self, pose
    ) -> Tuple[Dict[str, _SimpleProtomer], List[str], List[str]]:
        protomers: Dict[str, _SimpleProtomer] = {}
        chain_order: List[str] = []
        ligand_chains: List[str] = []

        for chain_idx in range(1, pose.num_chains() + 1):
            chain_name = int_id_to_str_id(chain_idx)
            begin = pose.chain_begin(chain_idx)
            end = pose.chain_end(chain_idx)
            length = end - begin + 1
            representative = pose.residue(begin)

            if representative.is_protein():
                proto_type = ProtomerTypes.PROTEIN
            elif representative.is_ligand():
                proto_type = ProtomerTypes.CCD
                ligand_chains.append(chain_name)
            else:
                proto_type = ProtomerTypes.SMILES
                ligand_chains.append(chain_name)

            protomers[chain_name] = _SimpleProtomer(length, proto_type)
            chain_order.append(chain_name)

        # Best-effort global_config injection (selector expects it).
        try:
            global_config.init_const("protomers", protomers)
        except RuntimeError:
            global_config._protomers = protomers  # type: ignore[attr-defined]
            global_config._initialized_constants["protomers"] = True  # type: ignore[attr-defined]

        return protomers, chain_order, ligand_chains

    def test_selector_runs(self) -> None:
        if os.environ.get("GH_RUN_PYROSETTA_TESTS") != "1":
            self.skipTest("Set GH_RUN_PYROSETTA_TESTS=1 to enable PyRosetta integration tests")

        pyrosetta = self._try_import_pyrosetta()
        self._ensure_pyrosetta_initialized(pyrosetta)

        pose = self._load_pose(pyrosetta)
        _, chain_order, ligand_chains = self._configure_protomers(pose)
        oligo_name = "".join(chain_order)
        complex_poses = ComplexPoses({oligo_name: pose})

        selector = LigandUnsatisfiedPolarPositionSelector(
            complex_poses=complex_poses,
            ligand_targets=ligand_chains,
            distance_min=4.5,
            distance_max=7.0,
        )
        selection = selector.select_positions()
        self.assertIsInstance(selection, dict)

        # Spot-check we can traverse unsatisfied polar atoms without crashing.
        pdb_info = pose.pdb_info()
        for chain_id in ligand_chains:
            filter_ = LigandHBondFilter(
                pose=pose,
                ligand_chain=chain_id,
                distance_min=4.5,
                distance_max=7.0,
            )
            unsatisfied = selector._get_unsatisfied_polar_atoms(pose, filter_)
            for res_idx, atom_name in list(sorted(unsatisfied))[:5]:
                residues = filter_.get_eligible_residues_for_group(
                    atom_name, ligand_res_idx=res_idx
                )
                _ = [
                    f"{res['residue_name']}{pdb_info.number(res['residue_num'])}:{res['chain']}"
                    for res in residues
                ]
