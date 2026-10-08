"""AutoDock molecular docking implementation with GPU and Vina support."""

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Optional
from io import StringIO

import numpy as np
import prody as pr
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Structure
from Bio.SeqUtils import seq1

# Optional: only needed for the (currently unused) Meeko-based ligand preparation path.
try:
    from meeko import (
        MoleculePreparation,
        ResidueChemTemplates,
        Polymer,
        PDBQTWriterLegacy,
    )
    from meeko.gridbox import get_gpf_string
except Exception:  # pragma: no cover - optional dependency
    MoleculePreparation = None  # type: ignore[assignment]
    ResidueChemTemplates = None  # type: ignore[assignment]
    Polymer = None  # type: ignore[assignment]
    PDBQTWriterLegacy = None  # type: ignore[assignment]
    get_gpf_string = None  # type: ignore[assignment]

from astrolabe.core import global_config
from astrolabe.core.chain_resolution import chain_ids_for_oligo, chain_to_protomer
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.structure_processors import StructureProcessor, register_structure_processor
from astrolabe.utils.extra_residues import resolve_ligand_resname_for_protomer

# Constants for atom types used in AutoDock
LIGAND_ATOM_TYPES = [
    "C",
    "A",
    "N",
    "O",
    "S",
    "H",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "HD",
    "NA",
    "OA",
    "SA",
]
RECEPTOR_ATOM_TYPES = [
    "HD",
    "C",
    "A",
    "N",
    "NA",
    "OA",
    "F",
    "P",
    "SA",
    "S",
    "Cl",
    "Br",
    "I",
    "Mg",
    "Ca",
    "Mn",
    "Fe",
    "Zn",
]

# Default binding site parameters
DEFAULT_BINDING_SITE = {
    "center_x": 0.0,
    "center_y": 0.0,
    "center_z": 0.0,
    "size_x": 20.0,
    "size_y": 20.0,
    "size_z": 20.0,
}
MIN_BOX_SIZE = 5.0  # Minimum binding box size in Angstroms
DEFAULT_PADDING = 1.0  # Default padding for binding site calculation


class DockingPenaltyError(RuntimeError):
    """Soft docking failure that should become a score penalty instead of aborting the run."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.reason = str(reason).strip() or "docking_penalty"
        self.details = dict(details or {})


class AutoDockProcessorBase(StructureProcessor):
    """Base class for AutoDock molecular docking implementations."""

    def __init__(self, **kwargs):
        """Initialize AutoDock predocker with configuration parameters.

        Args:
            **kwargs: Configuration parameters including:
                - num_modes (int): Number of docking modes to generate
                - exhaustiveness (int): Search exhaustiveness
                - autodock_gpu_path (str): Path to AutoDock-GPU executable
                - vina_path (str): Path to AutoDock Vina executable
                - obabel_path (str): Path to OpenBabel executable
                - autogrid4_path (str): Path to AutoGrid4 executable
        """
        super().__init__(**kwargs)

        self.num_docking_modes = kwargs.get("num_modes", 5)
        self.search_exhaustiveness = kwargs.get("exhaustiveness", 8)
        self.autodock_gpu_executable = kwargs.get("autodock_gpu_path")
        self.autodock_vina_executable = kwargs.get("vina_path")
        self.obabel_executable = kwargs.get("obabel_path")
        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        cores = 0
        if callable(sched_getaffinity):
            try:
                cores = len(sched_getaffinity(0))
            except (OSError, NotImplementedError):
                cores = 0
        if not cores:
            cpu_count = os.cpu_count() or 1
            cores = max(1, cpu_count // 2)
        self.num_cpu_cores = int(cores)
        self.autogrid4_executable = kwargs.get("autogrid4_path")
        self.mk_prepare_receptor = kwargs.get("mk_prepare_receptor") or kwargs.get(
            "mk_prepare_receptor_path"
        )
        self.mk_prepare_ligand = kwargs.get("mk_prepare_ligand") or kwargs.get(
            "mk_prepare_ligand_path"
        )
        self.mk_prepare_python_bin = kwargs.get("mk_prepare_python_bin") or kwargs.get(
            "mk_prepare_python"
        )
        self.mk_prepare_pythonpath = kwargs.get("mk_prepare_pythonpath")
        self.binding_site_strategy = kwargs.get(
            "binding_site_strategy", "interface"
        ).lower()
        self.binding_site_coverage = float(kwargs.get("binding_site_coverage", 80.0))
        self.binding_site_padding = float(kwargs.get("binding_site_padding", DEFAULT_PADDING))
        self.reference_structure_path = kwargs.get("reference_structure_path")

        self.reference_structure_chain = kwargs.get("reference_structure_chain")
        self.current_structure_chain = kwargs.get("current_structure_chain")
        self.current_structure_sequence = kwargs.get("current_structure_sequence") or kwargs.get(
            "current_design_sequence"
        )
        raw_positions = kwargs.get("reference_pocket_positions") or []
        raw_positions_by_chain = kwargs.get("reference_pocket_positions_by_chain")
        if raw_positions_by_chain is None and hasattr(raw_positions, "items"):
            raw_positions_by_chain = raw_positions
            raw_positions = []
        self.reference_pocket_positions = self._normalize_position_list(raw_positions)
        self.reference_pocket_positions_by_chain = self._normalize_positions_by_chain(
            raw_positions_by_chain
        )
        self.reference_box_padding = float(
            kwargs.get("reference_box_padding", self.binding_site_padding)
        )
        self.reference_min_box_size = float(kwargs.get("reference_min_box_size", MIN_BOX_SIZE))
        self.reference_box_geometry = str(
            kwargs.get("reference_box_geometry", "residue_heavy_atoms")
        ).strip().lower()
        self.reference_alignment_mode = str(
            kwargs.get("reference_alignment_mode", "rigid")
        ).strip().lower()
        self.reference_max_local_rmsd = float(
            kwargs.get("reference_max_local_rmsd", 3.0)
        )
        self.rank_buried_fraction = kwargs.get("rank_buried_fraction", True)
        self.rank_com_diff = kwargs.get("rank_com_diff", False)
        self.rank_buried_contrib = kwargs.get("rank_buried_contribution", False)
        self.rank_vina_order = kwargs.get("rank_vina_order", False)
        self.cache_first_pose_only = bool(kwargs.get("cache_first_pose_only", False))
        self.fail_on_error = bool(kwargs.get("fail_on_error", False))
        self.rank_alpha_shape_occupancy = bool(
            kwargs.get("rank_alpha_shape_occupancy", False)
        )
        self.rank_alpha_shape_min_occupancy = float(
            kwargs.get("rank_alpha_shape_min_occupancy", 0.8)
        )
        self.rank_alpha_shape_tolerance = float(kwargs.get("rank_alpha_shape_tolerance", 1e-8))
        self.rank_alpha_shape_retry_attempts = max(
            0, int(kwargs.get("rank_alpha_shape_retry_attempts", 0))
        )
        self.rank_alpha_shape_retry_exhaustiveness = kwargs.get(
            "rank_alpha_shape_retry_exhaustiveness"
        )
        if self.rank_alpha_shape_retry_exhaustiveness is not None:
            self.rank_alpha_shape_retry_exhaustiveness = int(
                self.rank_alpha_shape_retry_exhaustiveness
            )
        self.rank_alpha_shape_retry_num_modes = kwargs.get(
            "rank_alpha_shape_retry_num_modes"
        )
        if self.rank_alpha_shape_retry_num_modes is not None:
            self.rank_alpha_shape_retry_num_modes = int(
                self.rank_alpha_shape_retry_num_modes
            )
        self.alpha_radius = kwargs.get("alpha_radius")
        self.alpha_radius_scale = float(kwargs.get("alpha_radius_scale", 1.8))
        self.alpha_min_radius = float(kwargs.get("alpha_min_radius", 3.0))
        self.alpha_max_radius = float(kwargs.get("alpha_max_radius", 8.0))
        self.rank_tolerances = {
            "buried_fraction": float(kwargs.get("rank_buried_fraction_tolerance", 0.1)),
            "com_diff": float(kwargs.get("rank_com_diff_tolerance", 1.0)),
            "buried_contribution_diff": float(
                kwargs.get("rank_buried_contribution_tolerance", 0.1)
            ),
        }
        self.penalty_metadata_prefix = (
            str(kwargs.get("penalty_metadata_prefix", "autodock_penalty")).strip()
            or "autodock_penalty"
        )
        self.binding_site_metadata_prefix = (
            str(kwargs.get("binding_site_metadata_prefix", "autodock_binding_site")).strip()
            or "autodock_binding_site"
        )
        self._last_binding_site_debug: Optional[Dict[str, Any]] = None
        self._last_binding_site_box_debug: Optional[Dict[str, Any]] = None
        self._last_binding_site_transform: Optional[Dict[str, Any]] = None

        # Pose sampling metadata for cache-aware random selection.
        self._last_pose_candidates: List[str] = []
        self.rng = np.random.default_rng()

        if not self.obabel_executable:
            raise ValueError("AutoDock predocker requires 'obabel_path'.")
        if not self.autodock_gpu_executable and not self.autodock_vina_executable:
            logging.warning("AutoDock predocker missing docking executable paths.")

        # AutoDock-GPU workflow requirements
        if self.autodock_gpu_executable:
            if not self.autogrid4_executable:
                logging.warning("AutoDock-GPU predocker missing 'autogrid4_path'.")
            if not self.mk_prepare_receptor:
                logging.warning("AutoDock-GPU predocker missing 'mk_prepare_receptor'.")

    def _mk_prepare_env(self) -> dict[str, str] | None:
        if not self.mk_prepare_pythonpath:
            return None
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.mk_prepare_pythonpath)
            if not existing
            else str(self.mk_prepare_pythonpath) + os.pathsep + existing
        )
        return env

    @staticmethod
    def _infer_oligo_name(input_file_path: str) -> Optional[str]:
        stem = Path(str(input_file_path)).stem
        match = re.search(r"step_\d+_([^._]+)_model", stem)
        if match:
            return str(match.group(1))
        return None

    @staticmethod
    def _is_ligand_protomer_type(protomer_type: Any) -> bool:
        if protomer_type in {ProtomerTypes.SMILES, ProtomerTypes.CCD}:
            return True
        value = getattr(protomer_type, "value", protomer_type)
        return str(value).strip().lower() in {"s", "smiles", "c", "ccd"}

    def _configured_ligand_protomer_for_oligo(self, oligo_name: str):
        chain_to_proto = chain_to_protomer(oligo_name)
        ordered_chain_ids = chain_ids_for_oligo(oligo_name)
        try:
            protomers = global_config.get_const("protomers")
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot infer ligand chain for oligo {oligo_name!r}: global protomers are unavailable"
            ) from exc
        ligand_chain_ids = []
        for chain_id in ordered_chain_ids:
            proto = chain_to_proto.get(chain_id)
            protomer = protomers.get(proto) if isinstance(protomers, dict) else None
            if protomer is None:
                continue
            if self._is_ligand_protomer_type(getattr(protomer, "type", None)):
                ligand_chain_ids.append((chain_id, proto, protomer))
        if len(ligand_chain_ids) != 1:
            raise ValueError(
                f"Expected exactly one ligand protomer for oligo {oligo_name!r}, got {ligand_chain_ids!r}"
            )
        return ligand_chain_ids[0]

    def _configured_ligand_chain_id_for_oligo(self, oligo_name: str) -> str:
        return self._configured_ligand_protomer_for_oligo(oligo_name)[0]

    def _configured_ligand_resname_for_oligo(self, oligo_name: str) -> str:
        _chain_id, proto, protomer = self._configured_ligand_protomer_for_oligo(
            oligo_name
        )
        ccd_code = None
        if getattr(protomer, "type", None) == ProtomerTypes.CCD:
            ccd_codes = [
                token.strip().upper()
                for token in str(getattr(protomer, "sequence", "")).split(",")
                if token.strip()
            ]
            if len(ccd_codes) == 1:
                ccd_code = ccd_codes[0]
        return resolve_ligand_resname_for_protomer(
            protomer_key=proto,
            protomer_name=str(getattr(protomer, "name", proto)),
            ccd_code=ccd_code,
        )

    @staticmethod
    def _rename_chain_residues(chain, resname: Optional[str]) -> None:
        if not resname:
            return
        for residue in chain:
            residue.resname = str(resname)

    @staticmethod
    def _normalize_position_list(raw_positions: Any) -> List[int]:
        if raw_positions is None:
            return []
        if isinstance(raw_positions, str):
            raw_positions = [
                token.strip()
                for chunk in raw_positions.split(",")
                for token in chunk.split()
                if token.strip()
            ]
        return [int(pos) for pos in raw_positions]

    @classmethod
    def _normalize_positions_by_chain(cls, raw_positions_by_chain: Any) -> Dict[str, List[int]]:
        if not raw_positions_by_chain:
            return {}
        if not hasattr(raw_positions_by_chain, "items"):
            raise ValueError(
                "reference_pocket_positions_by_chain must be a mapping of chain selector to positions"
            )
        out: Dict[str, List[int]] = {}
        for chain_selector, positions in raw_positions_by_chain.items():
            key = str(chain_selector).strip()
            if not key:
                continue
            parsed = cls._normalize_position_list(positions)
            if parsed:
                out[key] = parsed
        return out

    def _penalty_metadata_path(self, *, step: int, oligo_name: str) -> Path:
        return self.debug_dir / f"{self.penalty_metadata_prefix}_step_{int(step):03d}_{oligo_name}.json"
    def _binding_site_metadata_path(self, *, step: int, oligo_name: str) -> Path:
        return self.debug_dir / f"{self.binding_site_metadata_prefix}_step_{int(step):03d}_{oligo_name}.json"


    def _write_penalty_metadata(
        self,
        *,
        input_file_path: str,
        penalty_active: bool,
        reason: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        oligo_name = self._infer_oligo_name(input_file_path)
        if not oligo_name:
            logging.debug(
                "Skipping autodock penalty metadata write because oligo name could not be inferred: %s",
                input_file_path,
            )
            return
        try:
            step_value = int(global_config.step or 0)
        except Exception:
            step_value = 0
        payload = {
            "penalty_active": bool(penalty_active),
            "reason": str(reason).strip() or "docking_penalty",
            "message": str(message).strip(),
            "step": int(step_value),
            "oligo_name": oligo_name,
            "input_file_path": str(input_file_path),
            "details": dict(details or {}),
        }
        metadata_path = self._penalty_metadata_path(
            step=step_value, oligo_name=oligo_name
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_binding_site_metadata(
        self,
        *,
        input_file_path: str,
        binding_site: Dict[str, float],
    ) -> None:
        oligo_name = self._infer_oligo_name(input_file_path)
        if not oligo_name:
            logging.debug(
                "Skipping binding-site metadata write because oligo name could not be inferred: %s",
                input_file_path,
            )
            return
        details = dict(self._last_binding_site_debug or {})
        if not details:
            logging.debug(
                "Skipping binding-site metadata write because no binding-site debug payload was captured"
            )
            return
        try:
            step_value = int(global_config.step or 0)
        except Exception:
            step_value = 0
        payload = {
            "step": int(step_value),
            "oligo_name": oligo_name,
            "input_file_path": str(input_file_path),
            "strategy": str(self.binding_site_strategy or "").strip().lower(),
            "binding_site": {
                "center_x": float(binding_site["center_x"]),
                "center_y": float(binding_site["center_y"]),
                "center_z": float(binding_site["center_z"]),
                "size_x": float(binding_site["size_x"]),
                "size_y": float(binding_site["size_y"]),
                "size_z": float(binding_site["size_z"]),
            },
            "details": details,
        }
        metadata_path = self._binding_site_metadata_path(
            step=step_value, oligo_name=oligo_name
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _materialize_penalty_artifact(self, input_file_path: str) -> str:
        """Create a cacheable predocker artifact for soft docking failures."""
        input_path = Path(str(input_file_path))
        output_path = Path(
            self._get_output_path(f"autodock_penalty_{input_path.name}")
        )
        if output_path.exists():
            output_path.unlink()
        try:
            os.link(input_path, output_path)
        except OSError as exc:
            logging.warning(
                "Could not hardlink penalty artifact %s -> %s (%s); copying instead.",
                input_path,
                output_path,
                exc,
            )
            shutil.copy2(input_path, output_path)
        return str(output_path)

    def _convert_cif_to_pdb(self, input_path: str) -> str:
        """Convert CIF format to PDB format using OpenBabel.

        Args:
            input_path: Path to input CIF file

        Returns:
            Path to converted PDB file

        Raises:
            subprocess.CalledProcessError: If conversion fails
        """
        output_pdb_path = self._get_output_path(f"autodock_{Path(input_path).stem}.pdb")
        conversion_cmd = [
            self.obabel_executable,
            str(input_path),
            "-O",
            str(output_pdb_path),
        ]

        try:
            subprocess.run(conversion_cmd, check=True, capture_output=True, text=True)
            logging.debug(f"Successfully converted CIF to PDB: {output_pdb_path}")
            return str(output_pdb_path)
        except subprocess.CalledProcessError as e:
            logging.error(f"CIF conversion failed: {e.stderr}")
            raise

    def _separate_protein_ligand_with_prody(
        self, pdb_file_path: str
    ) -> Tuple[str, str]:
        """Separate protein and ligand structures using ProDy.

        Args:
            pdb_file_path: Path to input PDB file

        Returns:
            Tuple of (protein_pdb_path, ligand_pdb_path)
        """
        try:
            structure = pr.parsePDB(pdb_file_path)

            # Select protein (standard amino acids)
            protein_selection = structure.select("protein")

            # Select ligand (HETATM excluding water and ions)
            ligand_selection = structure.select(
                "(not protein) and not water and not ion"
            )

            # Generate output file paths
            protein_output_path = self._get_output_path(
                f"atd_{Path(pdb_file_path).stem}_protein.pdb"
            )
            ligand_output_path = self._get_output_path(
                f"atd_{Path(pdb_file_path).stem}_ligand.pdb"
            )

            # Save protein structure
            if protein_selection is not None:
                pr.writePDB(str(protein_output_path), protein_selection)
            else:
                logging.warning("No protein structure found")
                self._write_empty_pdb(protein_output_path, "No protein found")

            # Save ligand structure
            if ligand_selection is not None and len(ligand_selection) > 0:
                pr.writePDB(str(ligand_output_path), ligand_selection)
                ligand_residue_names = list(set(ligand_selection.getResnames()))
                logging.info(f"Identified ligand residues: {ligand_residue_names}")
            else:
                logging.warning("No ligands found")
                self._write_empty_pdb(ligand_output_path, "No ligands found")

            logging.info(
                f"ProDy separation completed - Protein: {protein_output_path}, Ligand: {ligand_output_path}"
            )
            return str(protein_output_path), str(ligand_output_path)

        except Exception as e:
            logging.error(f"ProDy processing failed: {e}")
            if self.fail_on_error:
                raise
            return self._separate_protein_ligand_fallback(pdb_file_path)

    def _separate_protein_ligand_fallback(self, pdb_file_path: str) -> Tuple[str, str]:
        """Fallback text-based method for separating protein and ligand.

        Args:
            pdb_file_path: Path to input PDB file

        Returns:
            Tuple of (protein_pdb_path, ligand_pdb_path)
        """
        logging.info("Using text parsing fallback method for protein-ligand separation")

        protein_output_path = self._get_output_path(
            f"{Path(pdb_file_path).stem}_protein.pdb"
        )
        ligand_output_path = self._get_output_path(
            f"{Path(pdb_file_path).stem}_ligand.pdb"
        )

        with open(pdb_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        protein_lines = []
        ligand_lines = []
        excluded_residues = {"HOH", "WAT"}  # Water molecules to exclude

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                if line.startswith("ATOM"):
                    protein_lines.append(line)
                elif (
                    line.startswith("HETATM")
                    and line[17:20].strip() not in excluded_residues
                ):
                    ligand_lines.append(line)
            elif line.startswith(("CONECT", "END")):
                continue

        # Write separated structures
        self._write_pdb_lines(protein_output_path, protein_lines)
        self._write_pdb_lines(ligand_output_path, ligand_lines)

        logging.info(
            f"Text parsing separation completed - Protein: {protein_output_path}, Ligand: {ligand_output_path}"
        )
        return str(protein_output_path), str(ligand_output_path)

    def _write_empty_pdb(self, file_path: str, remark: str) -> None:
        """Write an empty PDB file with a remark."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"REMARK   {remark}\nEND\n")

    def _write_pdb_lines(self, file_path: str, lines: List[str]) -> None:
        """Write PDB lines to file with proper ending."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
            file.write("END\n")

    def _prepare_receptor_with_obabel(
        self, protein_pdb_path: str, output_pdbqt_path: str
    ) -> None:
        """Prepare receptor PDBQT file using OpenBabel.

        Args:
            protein_pdb_path: Path to protein PDB file
            output_pdbqt_path: Path for output PDBQT file

        Raises:
            subprocess.CalledProcessError: If preparation fails
        """
        preparation_cmd = [
            self.obabel_executable,
            protein_pdb_path,
            "-O",
            output_pdbqt_path,
            "-h",  # Add hydrogens
            "-xr",  # Remove redundant atoms
        ]

        try:
            subprocess.run(preparation_cmd, check=True, capture_output=True, text=True)
            logging.info(f"Receptor prepared with OpenBabel: {output_pdbqt_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"OpenBabel receptor preparation failed: {e.stderr}")
            raise

    def _prepare_ligand_with_obabel(
        self, ligand_pdb_path: str, output_pdbqt_path: str
    ) -> None:
        """Prepare ligand PDBQT file using OpenBabel.

        Args:
            ligand_pdb_path: Path to ligand PDB file
            output_pdbqt_path: Path for output PDBQT file

        Raises:
            subprocess.CalledProcessError: If preparation fails
        """
        preparation_cmd = [
            self.obabel_executable,
            ligand_pdb_path,
            "-O",
            output_pdbqt_path,
            "-h",  # Add hydrogens
        ]

        logging.debug(
            f"Preparing ligand with OpenBabel, command: {' '.join(preparation_cmd)}"
        )

        try:
            subprocess.run(preparation_cmd, check=True, capture_output=True, text=True)
            logging.info(f"Ligand prepared with OpenBabel: {output_pdbqt_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"OpenBabel ligand preparation failed: {e.stderr}")
            raise

    def _prepare_ligand_with_meeko(
        self, ligand_pdb_path: str, ligand_pdbqt_path: str
    ) -> None:
        """Prepare ligand PDBQT file using Meeko."""
        if MoleculePreparation is None:
            raise ImportError(
                "Meeko is required for _prepare_ligand_with_meeko but is not importable. "
                "Install `meeko` in the active environment or switch to the OpenBabel path."
            )
        if not self.mk_prepare_ligand:
            raise ValueError(
                "Missing mk_prepare_ligand path; set autodock.mk_prepare_ligand in env.yaml."
            )
        mol2_file = self._get_output_path(f"{Path(ligand_pdb_path).stem}.mol2")
        preparation_cmd = [
            self.obabel_executable,
            ligand_pdb_path,
            "-O",
            mol2_file,
            "-h",  # Add hydrogens
        ]
        try:
            subprocess.run(preparation_cmd, check=True, capture_output=True, text=True)
            logging.info(f"Ligand converted to MOL2 with OpenBabel: {mol2_file}")
        except subprocess.CalledProcessError as e:
            logging.error(f"OpenBabel ligand conversion to MOL2 failed: {e.stderr}")
            raise

        try:
            if self.mk_prepare_python_bin:
                cmd = [
                    str(self.mk_prepare_python_bin),
                    str(self.mk_prepare_ligand),
                    "-i",
                    str(mol2_file),
                    "-o",
                    str(ligand_pdbqt_path),
                    "--rigid_macrocycles",
                ]
            else:
                cmd = [
                    str(self.mk_prepare_ligand),
                    "-i",
                    str(mol2_file),
                    "-o",
                    str(ligand_pdbqt_path),
                    "--rigid_macrocycles",
                ]
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=self._mk_prepare_env(),
            )
            logging.info(f"Ligand prepared with Meeko: {ligand_pdbqt_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Meeko ligand preparation failed: {e.stderr}")
            raise

    def _determine_binding_site(
        self,
        protein_pdb_path: str,
        ligand_pdb_path: Optional[str] = None,
        oligo_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Return binding site parameters using the configured strategy."""
        strategy = (self.binding_site_strategy or "interface").lower()
        if strategy == "interface":
            return self._calculate_binding_site_from_protein_interface(
                protein_pdb_path,
                padding=self.binding_site_padding,
                coverage_percent=self.binding_site_coverage,
            )
        if strategy in {"residue", "residue_percentile", "percentile"}:
            return self._calculate_binding_site_covering_residues(
                protein_pdb_path,
                coverage_percent=self.binding_site_coverage,
                padding=self.binding_site_padding,
            )
        if strategy in {"aligned_reference_pocket", "reference_pocket_aligned"}:
            return self._calculate_binding_site_from_aligned_reference_pocket(
                protein_pdb_path,
                oligo_name=oligo_name,
            )
        if strategy == "ligand" and ligand_pdb_path:
            return self._calculate_binding_site_from_ligand(
                ligand_pdb_path, padding=self.binding_site_padding
            )
        logging.warning(
            "Unknown binding site strategy '%s', falling back to interface-based calculation",
            strategy,
        )
        return self._calculate_binding_site_from_protein_interface(
            protein_pdb_path,
            padding=self.binding_site_padding,
            coverage_percent=self.binding_site_coverage,
        )

    def _calculate_binding_site_covering_residues(
        self,
        protein_pdb_path: str,
        coverage_percent: float = 90.0,
        padding: float = DEFAULT_PADDING,
    ) -> Dict[str, float]:
        """Calculate a bounding box that encloses N%% of protein CA atoms."""
        try:
            structure = pr.parsePDB(protein_pdb_path)
            ca_atoms = structure.select("protein and name CA")
            if ca_atoms is None or ca_atoms.numAtoms() == 0:
                logging.warning(
                    "No CA atoms found for residue coverage calculation, using default binding site"
                )
                return DEFAULT_BINDING_SITE.copy()

            coords = ca_atoms.getCoords()
            lower_percentile = max(0.0, (100.0 - coverage_percent) / 2.0)
            upper_percentile = 100.0 - lower_percentile

            lower_bounds = np.percentile(coords, lower_percentile, axis=0)
            upper_bounds = np.percentile(coords, upper_percentile, axis=0)

            center_coords = (lower_bounds + upper_bounds) / 2.0
            box_sizes = np.maximum(
                upper_bounds - lower_bounds + 2 * padding, MIN_BOX_SIZE
            )

            binding_site = {
                "center_x": float(center_coords[0]),
                "center_y": float(center_coords[1]),
                "center_z": float(center_coords[2]),
                "size_x": float(box_sizes[0]),
                "size_y": float(box_sizes[1]),
                "size_z": float(box_sizes[2]),
            }
            logging.info(
                "Residue coverage binding site calculated (%.1f%% coverage)",
                coverage_percent,
            )
            logging.debug(
                "Residue coverage binding site center=(%.2f, %.2f, %.2f) size=(%.2f, %.2f, %.2f)",
                center_coords[0],
                center_coords[1],
                center_coords[2],
                binding_site["size_x"],
                binding_site["size_y"],
                binding_site["size_z"],
            )
            return binding_site
        except Exception as exc:
            logging.error(
                "Residue coverage binding site calculation failed: %s", exc
            )
            return DEFAULT_BINDING_SITE.copy()

    def _calculate_binding_site_from_protein_interface(
        self,
        protein_pdb_path: str,
        padding: float = DEFAULT_PADDING,
        coverage_percent=90,
    ) -> Dict[str, float]:
        """protein_pdb_path中包含两条链，通过计算刚好能包裹住全部界面残Cα原子的最小长方体来定义AutoDock对接盒子
        使用主成分分析(PCA)确定界面残基的主轴方向，旋转蛋白使主轴与坐标轴对齐，并使用80%覆盖率阈值"""
        def unavailable(message: str, **details: Any) -> None:
            payload = {
                "strategy": "interface",
                "current_structure_path": str(protein_pdb_path),
                "binding_site_coverage": float(coverage_percent),
                "binding_site_padding": float(padding),
                "reason": "binding_site_unavailable",
                "message": str(message),
                **details,
            }
            self._last_binding_site_debug = payload
            raise DockingPenaltyError(
                str(message),
                reason="binding_site_unavailable",
                details=payload,
            )

        try:
            structure = pr.parsePDB(protein_pdb_path)

            # Get all chains
            chain_ids = sorted(set(structure.getChids()))
            if len(chain_ids) < 2:
                unavailable(
                    "Interface binding site requires at least two chains",
                    chain_ids=[str(chain_id) for chain_id in chain_ids],
                )

            # Take first two chains
            chain_a = structure.select(f"chain {chain_ids[0]} and name CA")
            chain_b = structure.select(f"chain {chain_ids[1]} and name CA")

            if chain_a is None or chain_b is None:
                unavailable(
                    "Could not select CA atoms from interface chains",
                    chain_ids=[str(chain_ids[0]), str(chain_ids[1])],
                )

            # Get coordinates
            coords_a = chain_a.getCoords()
            coords_b = chain_b.getCoords()

            from astrolabe.losses.interaction import protein_interface_ca_box

            box = protein_interface_ca_box(
                coords_a, coords_b, interface_distance=8.0,
                coverage_percent=coverage_percent, padding=padding,
                min_box_size=MIN_BOX_SIZE,
            )
            # The existing docking transform applies to both receptor and ligand,
            # and its inverse restores the selected pose to the input frame.
            self._last_binding_site_transform = {
                "origin": box["origin"].tolist(),
                "axes": box["axes"].tolist(),
            }
            binding_site = {
                **{f"center_{axis}": float(box["center"][i]) for i, axis in enumerate("xyz")},
                **{f"size_{axis}": float(box["size"][i]) for i, axis in enumerate("xyz")},
            }
            self._last_binding_site_debug = {
                "strategy": "interface",
                "current_structure_path": str(protein_pdb_path),
                "chain_ids": [str(chain_ids[0]), str(chain_ids[1])],
                "interface_distance": 8.0,
                "binding_site_coverage": float(coverage_percent),
                "binding_site_padding": float(padding),
                "interface_coordinate_count": len(box["points"]),
                "pca_eigenvalues": box["eigenvalues"].tolist(),
                "pca_center": box["center"].tolist(),
                "rotation_axes": box["axes"].tolist(),
                "box_bounds_pca": np.stack([box["lower"], box["upper"]], axis=1).tolist(),
            }
            logging.info(
                "Interface binding site calculated from %d CA atoms at %.1f%% coverage",
                len(box["points"]), coverage_percent,
            )
            return binding_site

        except DockingPenaltyError:
            raise
        except Exception as e:
            logging.error(f"Interface binding site calculation failed: {e}")
            unavailable(f"Interface binding site calculation failed: {e}")

    def _calculate_binding_site_from_ligand(
        self, ligand_pdb_path: str, padding: float = DEFAULT_PADDING
    ) -> Dict[str, float]:
        """Calculate binding site based on original ligand position.

        Args:
            ligand_pdb_path: Path to ligand PDB file
            padding: Padding to add around ligand (default: 5.0 Angstroms)

        Returns:
            Dictionary containing binding site center and size coordinates
        """
        coordinates = []

        try:
            with open(ligand_pdb_path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith(("ATOM", "HETATM")):
                        try:
                            x_coord = float(line[30:38])
                            y_coord = float(line[38:46])
                            z_coord = float(line[46:54])
                            coordinates.append([x_coord, y_coord, z_coord])
                        except ValueError:
                            continue
        except FileNotFoundError:
            logging.error(f"Ligand file not found: {ligand_pdb_path}")
            return DEFAULT_BINDING_SITE.copy()

        if not coordinates:
            logging.warning(
                "No valid coordinates found in ligand file, using default binding site"
            )
            return DEFAULT_BINDING_SITE.copy()

        # Calculate bounding box
        coordinates_array = np.array(coordinates)
        min_coords = np.min(coordinates_array, axis=0)
        max_coords = np.max(coordinates_array, axis=0)

        # Calculate center and size
        center_coords = (min_coords + max_coords) / 2
        box_sizes = max_coords - min_coords + 2 * padding

        binding_site = {
            "center_x": float(center_coords[0]),
            "center_y": float(center_coords[1]),
            "center_z": float(center_coords[2]),
            "size_x": max(float(box_sizes[0]), MIN_BOX_SIZE),
            "size_y": max(float(box_sizes[1]), MIN_BOX_SIZE),
            "size_z": max(float(box_sizes[2]), MIN_BOX_SIZE),
        }

        logging.info(
            f"Binding site calculated: center=({center_coords[0]:.2f}, {center_coords[1]:.2f}, {center_coords[2]:.2f}), "
            f"size=({binding_site['size_x']:.2f}, {binding_site['size_y']:.2f}, {binding_site['size_z']:.2f})"
        )

        return binding_site

    @staticmethod
    def _load_alignment_model(structure_path: str):
        path = Path(structure_path)
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if ".cif" in suffixes or path.suffix.lower() in {".cif", ".mmcif"}:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure(path.stem, str(path))
        return next(structure.get_models())

    @staticmethod
    def _resolve_chain_id(model, preferred_chain: Optional[str]) -> str:
        if preferred_chain:
            chain_id = str(preferred_chain).strip()
            if chain_id in [chain.id for chain in model]:
                return chain_id
            raise ValueError(f"Chain {chain_id!r} not found in structure model")

        for chain in model:
            if any(residue.id[0] == " " for residue in chain.get_residues()):
                return str(chain.id)
        raise ValueError("No polymer chain found in structure model")

    @staticmethod
    def _collect_chain_residues(model, chain_id: str) -> Dict[int, Any]:
        chain = model[chain_id]
        return {
            int(residue.id[1]): residue
            for residue in chain.get_residues()
            if residue.id[0] == " "
        }

    @staticmethod
    def _collect_chain_residue_list(model, chain_id: str) -> list[Any]:
        chain = model[chain_id]
        return [residue for residue in chain.get_residues() if residue.id[0] == " "]

    @staticmethod
    def _clean_protein_sequence(sequence: Any) -> str:
        return re.sub(r"\s+", "", str(sequence or "").replace("+", "")).upper()

    @classmethod
    def _sequence_from_residue_list(cls, residues: list[Any]) -> str:
        return "".join(
            seq1(str(residue.get_resname()).strip().upper(), undef_code="X").upper()
            for residue in residues
        )

    def _current_design_sequence_for_chain(
        self,
        chain_id: str,
        current_residues: list[Any],
        oligo_name: Optional[str] = None,
    ) -> tuple[str, str]:
        explicit_sequence = self._clean_protein_sequence(self.current_structure_sequence)
        if explicit_sequence:
            if len(explicit_sequence) != len(current_residues):
                raise DockingPenaltyError(
                    "Configured current design sequence length does not match current chain residues",
                    reason="binding_site_unavailable",
                    details={
                        "current_chain_id": chain_id,
                        "current_sequence_source": "predocker.current_structure_sequence",
                        "current_sequence_length": len(explicit_sequence),
                        "current_polymer_residue_count": len(current_residues),
                    },
                )
            return explicit_sequence, "predocker.current_structure_sequence"

        try:
            protomers = global_config.get_const("protomers")
        except RuntimeError:
            protomers = {}
        proto_id = chain_id
        if oligo_name:
            proto_id = chain_to_protomer(oligo_name).get(chain_id, chain_id)
        protomer = protomers.get(proto_id) if isinstance(protomers, dict) else None
        protomer_sequence = self._clean_protein_sequence(
            getattr(protomer, "sequence", "") if protomer is not None else ""
        )
        if protomer_sequence:
            if len(protomer_sequence) != len(current_residues):
                raise DockingPenaltyError(
                    "Configured protomer sequence length does not match current chain residues",
                    reason="binding_site_unavailable",
                    details={
                        "current_chain_id": chain_id,
                        "current_sequence_source": f"global_config.protomers[{proto_id!r}].sequence",
                        "current_sequence_length": len(protomer_sequence),
                        "current_polymer_residue_count": len(current_residues),
                    },
                )
            return protomer_sequence, f"global_config.protomers[{proto_id!r}].sequence"

        return (
            self._sequence_from_residue_list(current_residues),
            "current_structure_resnames",
        )

    @staticmethod
    def _align_reference_sequence_to_current(
        reference_sequence: str,
        current_sequence: str,
        *,
        allow_mismatches: bool = False,
        ordinal: bool = False,
    ) -> tuple[dict[int, int], list[tuple[int, int]], list[tuple[int, int, str, str]]]:
        if len(reference_sequence) != len(current_sequence):
            raise DockingPenaltyError(
                "Reference/current chain sequence lengths do not match",
                reason="binding_site_unavailable",
                details={
                    "reference_sequence_length": len(reference_sequence),
                    "current_sequence_length": len(current_sequence),
                },
            )
        if ordinal:
            return (
                {idx: idx for idx in range(len(reference_sequence))},
                [(0, len(reference_sequence))],
                [(idx + 1, idx + 1, ref, cur)
                 for idx, (ref, cur) in enumerate(zip(reference_sequence, current_sequence))
                 if ref != cur],
            )
        if reference_sequence == current_sequence:
            return (
                {idx: idx for idx in range(len(reference_sequence))},
                [(0, len(reference_sequence))],
                [],
            )

        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -2
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        alignment = aligner.align(reference_sequence, current_sequence)[0]

        current_idx_by_reference_idx: dict[int, int] = {}
        aligned_blocks: list[tuple[int, int]] = []
        mismatches: list[tuple[int, int, str, str]] = []
        for (ref_start, ref_end), (cur_start, cur_end) in zip(
            alignment.aligned[0],
            alignment.aligned[1],
        ):
            ref_span = int(ref_end) - int(ref_start)
            cur_span = int(cur_end) - int(cur_start)
            if ref_span != cur_span:
                raise DockingPenaltyError(
                    "Reference/current alignment block lengths do not match",
                    reason="binding_site_unavailable",
                    details={"alignment_aligned": repr(alignment.aligned)},
                )
            aligned_blocks.append((int(ref_start), int(ref_end)))
            for offset in range(ref_span):
                ref_idx = int(ref_start) + offset
                cur_idx = int(cur_start) + offset
                ref_aa = reference_sequence[ref_idx]
                cur_aa = current_sequence[cur_idx]
                if ref_aa != cur_aa:
                    mismatches.append((ref_idx + 1, cur_idx + 1, ref_aa, cur_aa))
                    if not allow_mismatches:
                        continue
                current_idx_by_reference_idx[ref_idx] = cur_idx
        return current_idx_by_reference_idx, aligned_blocks, mismatches

    def _allow_reference_sequence_mismatch_mapping(self) -> bool:
        return self.reference_alignment_mode in {
            "ordinal",
            "position",
            "author_position",
            "allow_mismatch",
            "allow_mismatches",
            "allow_sequence_mismatch",
            "allow_sequence_mismatches",
        }

    @classmethod
    def _author_position_to_ordinal_map(cls, model, chain_id: str) -> Dict[int, int]:
        ordinal_by_author: Dict[int, int] = {}
        duplicate_positions: list[int] = []
        for ordinal_idx, residue in enumerate(cls._collect_chain_residue_list(model, chain_id)):
            author_pos = int(residue.id[1])
            if author_pos in ordinal_by_author:
                duplicate_positions.append(author_pos)
                continue
            ordinal_by_author[author_pos] = ordinal_idx
        if duplicate_positions:
            dupes = sorted(set(duplicate_positions))
            raise ValueError(
                f"Duplicate author residue positions in chain {chain_id!r}: {dupes}"
            )
        return ordinal_by_author

    @staticmethod
    def _residue_heavy_atom_coordinates(residue) -> np.ndarray:
        coords = [
            np.asarray(atom.coord, dtype=float)
            for atom in residue.get_atoms()
            if str(getattr(atom, "element", "")).strip().upper() != "H"
        ]
        if not coords:
            raise ValueError(f"Residue {residue} has no heavy atoms")
        return np.stack(coords, axis=0)

    @staticmethod
    def _residue_cb_or_ca_coordinate(residue) -> np.ndarray:
        resname = str(residue.get_resname()).strip().upper()
        preferred_atom_name = "CA" if resname == "GLY" else "CB"
        atom_name = preferred_atom_name if preferred_atom_name in residue else "CA"
        if atom_name not in residue:
            raise ValueError(
                f"Residue {resname} {residue.id[1]} is missing {preferred_atom_name}/CA."
            )
        coord = residue[atom_name].get_coord()
        return np.asarray([float(coord[0]), float(coord[1]), float(coord[2])], dtype=float)

    def _use_cb_alpha_shape_box(self) -> bool:
        return self.reference_box_geometry in {
            "cb_alpha_shape",
            "alpha_shape",
            "pocket_cb_alpha_shape",
            "cb",
        }

    @staticmethod
    def _box_volume(box_sizes: np.ndarray) -> float:
        return float(np.prod(np.asarray(box_sizes, dtype=float)))

    def _axis_aligned_box_from_coordinates(
        self,
        coords: np.ndarray,
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        lower_bounds = np.min(coords, axis=0)
        upper_bounds = np.max(coords, axis=0)
        center_coords = (lower_bounds + upper_bounds) / 2.0
        min_box_size = max(float(self.reference_min_box_size), 0.0)
        box_sizes = np.maximum(
            upper_bounds - lower_bounds + 2.0 * self.reference_box_padding,
            min_box_size,
        )
        return (
            {
                "center_x": float(center_coords[0]),
                "center_y": float(center_coords[1]),
                "center_z": float(center_coords[2]),
                "size_x": float(box_sizes[0]),
                "size_y": float(box_sizes[1]),
                "size_z": float(box_sizes[2]),
            },
            lower_bounds,
            upper_bounds,
            box_sizes,
        )

    def _pca_oriented_box_from_coordinates(
        self,
        coords: np.ndarray,
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        origin = np.mean(coords, axis=0)
        centered_coords = coords - origin
        covariance = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        axes = eigenvectors[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
        local_coords = centered_coords @ axes
        binding_site, lower_bounds, upper_bounds, box_sizes = (
            self._axis_aligned_box_from_coordinates(local_coords)
        )
        debug = {
            "reference_box_orientation": "pca",
            "reference_box_transform_origin": [float(x) for x in origin],
            "reference_box_transform_axes": [
                [float(value) for value in row] for row in axes.tolist()
            ],
            "reference_box_pca_eigenvalues": [float(x) for x in eigenvalues],
        }
        return binding_site, lower_bounds, upper_bounds, box_sizes, debug

    def _choose_binding_site_box_for_coordinates(
        self,
        coords: np.ndarray,
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
        (
            axis_box,
            axis_lower,
            axis_upper,
            axis_sizes,
        ) = self._axis_aligned_box_from_coordinates(coords)
        axis_volume = self._box_volume(axis_sizes)
        self._last_binding_site_transform = None
        self._last_binding_site_box_debug = {
            "reference_box_orientation": "axis_aligned",
            "reference_box_coordinate_frame": "input_xyz",
            "reference_box_axis_aligned_size": [float(x) for x in axis_sizes],
            "reference_box_axis_aligned_volume": axis_volume,
            "reference_box_selected_size": [float(x) for x in axis_sizes],
            "reference_box_selected_volume": axis_volume,
        }

        if not self._use_cb_alpha_shape_box():
            return axis_box, axis_lower, axis_upper

        try:
            (
                pca_box,
                pca_lower,
                pca_upper,
                pca_sizes,
                pca_debug,
            ) = self._pca_oriented_box_from_coordinates(coords)
        except np.linalg.LinAlgError as exc:
            if self.fail_on_error:
                raise
            logging.warning("PCA-oriented pocket box calculation failed: %s", exc)
            return axis_box, axis_lower, axis_upper

        pca_volume = self._box_volume(pca_sizes)
        self._last_binding_site_box_debug.update(
            {
                "reference_box_pca_size": [float(x) for x in pca_sizes],
                "reference_box_pca_volume": pca_volume,
            }
        )
        if pca_volume >= axis_volume - 1e-8:
            return axis_box, axis_lower, axis_upper

        self._last_binding_site_box_debug.update(
            {
                **pca_debug,
                "reference_box_coordinate_frame": "pca_local",
                "reference_box_selected_size": [float(x) for x in pca_sizes],
                "reference_box_selected_volume": pca_volume,
            }
        )
        self._last_binding_site_transform = {
            "origin": pca_debug["reference_box_transform_origin"],
            "axes": pca_debug["reference_box_transform_axes"],
        }
        logging.info(
            "Using PCA-oriented cb_alpha_shape docking box; volume %.3f -> %.3f",
            axis_volume,
            pca_volume,
        )
        return pca_box, pca_lower, pca_upper

    def _box_coordinates_for_residue(self, residue) -> np.ndarray:
        if self._use_cb_alpha_shape_box():
            return self._residue_cb_or_ca_coordinate(residue)[None, :]
        return self._residue_heavy_atom_coordinates(residue)

    def _binding_site_from_box_coordinates(
        self,
        coords: np.ndarray,
        *,
        context: str,
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
        if coords.size == 0:
            raise DockingPenaltyError(
                f"No selected pocket coordinates were found for {context}",
                reason="binding_site_unavailable",
            )
        if self._use_cb_alpha_shape_box() and len(np.unique(coords, axis=0)) < 4:
            raise DockingPenaltyError(
                "CB alpha-shape box geometry requires at least 4 unique pocket C-beta/C-alpha points",
                reason="binding_site_unavailable",
                details={"reference_box_geometry": self.reference_box_geometry},
            )
        return self._choose_binding_site_box_for_coordinates(coords)

    def _calculate_binding_site_from_aligned_reference_pocket(
        self,
        current_structure_path: str,
        oligo_name: Optional[str] = None,
    ) -> Dict[str, float]:
        if not self.reference_pocket_positions and not self.reference_pocket_positions_by_chain:
            raise ValueError("aligned_reference_pocket strategy requires reference_pocket_positions")
        if not self.reference_structure_path:
            raise ValueError(
                "aligned_reference_pocket strategy requires reference_structure_path"
            )

        reference_model = self._load_alignment_model(str(self.reference_structure_path))
        current_model = self._load_alignment_model(str(current_structure_path))
        if self.reference_pocket_positions_by_chain:
            return self._calculate_binding_site_from_aligned_reference_pocket_by_chain(
                reference_model,
                current_model,
                current_structure_path=current_structure_path,
                oligo_name=oligo_name,
            )

        reference_chain_selector = self.reference_structure_chain or self.current_structure_chain
        if not reference_chain_selector:
            raise ValueError(
                "aligned_reference_pocket strategy requires reference_structure_chain or current_structure_chain"
            )
        reference_chain_id = self._resolve_chain_id(reference_model, reference_chain_selector)
        current_chain_selector = self.current_structure_chain or self.reference_structure_chain
        if not current_chain_selector:
            raise ValueError(
                "aligned_reference_pocket strategy requires current_structure_chain or reference_structure_chain"
            )
        current_chain_id = self._resolve_chain_id(current_model, current_chain_selector)
        reference_residues = self._collect_chain_residue_list(reference_model, reference_chain_id)
        current_residues = self._collect_chain_residue_list(current_model, current_chain_id)
        if len(reference_residues) != len(current_residues):
            raise DockingPenaltyError(
                "Reference/current chain polymer residue counts do not match",
                reason="binding_site_unavailable",
                details={
                    "reference_chain_id": reference_chain_id,
                    "current_chain_id": current_chain_id,
                    "reference_polymer_residue_count": len(reference_residues),
                    "current_polymer_residue_count": len(current_residues),
                },
            )
        reference_sequence = self._sequence_from_residue_list(reference_residues)
        current_sequence, current_sequence_source = self._current_design_sequence_for_chain(
            current_chain_id,
            current_residues,
            oligo_name=oligo_name,
        )
        (
            current_idx_by_reference_idx,
            sequence_alignment_blocks,
            sequence_alignment_mismatches,
        ) = self._align_reference_sequence_to_current(
            reference_sequence,
            current_sequence,
            allow_mismatches=self._allow_reference_sequence_mismatch_mapping(),
            ordinal=self.reference_alignment_mode == "ordinal",
        )
        if sequence_alignment_mismatches and not self._allow_reference_sequence_mismatch_mapping():
            raise DockingPenaltyError(
                "Reference/current sequence alignment has residue mismatches",
                reason="binding_site_unavailable",
                details={
                    "current_sequence_source": current_sequence_source,
                    "mismatches": sequence_alignment_mismatches[:20],
                    "mismatch_count": len(sequence_alignment_mismatches),
                },
            )
        try:
            reference_ordinal_by_author = self._author_position_to_ordinal_map(
                reference_model, reference_chain_id
            )
        except ValueError as exc:
            raise DockingPenaltyError(
                str(exc),
                reason="binding_site_unavailable",
                details={"reference_chain_id": reference_chain_id},
            ) from exc

        current_box_coords = []
        missing_positions: list[int] = []
        mapped_current_positions: list[int] = []
        residue_mappings: list[Dict[str, Any]] = []
        for pos in self.reference_pocket_positions:
            ordinal_idx = reference_ordinal_by_author.get(int(pos))
            if ordinal_idx is None:
                missing_positions.append(int(pos))
                continue
            current_idx = current_idx_by_reference_idx.get(int(ordinal_idx))
            if current_idx is None:
                missing_positions.append(int(pos))
                continue
            reference_residue = reference_residues[ordinal_idx]
            residue = current_residues[current_idx]
            current_box_coords.append(self._box_coordinates_for_residue(residue))
            mapped_current_positions.append(int(residue.id[1]))
            residue_mappings.append(
                {
                    "reference_author_position": int(pos),
                    "reference_ordinal_position": int(ordinal_idx + 1),
                    "reference_resname": str(reference_residue.get_resname()).strip(),
                    "current_seqid": int(residue.id[1]),
                    "current_ordinal_position": int(current_idx + 1),
                    "current_resname": str(residue.get_resname()).strip(),
                }
            )
        if self.fail_on_error and missing_positions:
            raise DockingPenaltyError(
                "Not all selected pocket residues map to the current structure",
                reason="binding_site_unavailable",
                details={"missing_positions": missing_positions},
            )
        if not current_box_coords:
            raise DockingPenaltyError(
                "No selected pocket residues were found in the current structure",
                reason="binding_site_unavailable",
                details={
                    "missing_positions": missing_positions,
                    "mapped_current_positions": mapped_current_positions,
                },
            )
        current_box_coords = np.concatenate(current_box_coords, axis=0)

        binding_site, lower_bounds, upper_bounds = self._binding_site_from_box_coordinates(
            current_box_coords,
            context="aligned_reference_pocket",
        )
        self._last_binding_site_debug = {
            "reference_structure_path": str(self.reference_structure_path),
            "current_structure_path": str(current_structure_path),
            "reference_box_geometry": self.reference_box_geometry,
            "reference_box_padding": float(self.reference_box_padding),
            "reference_min_box_size": float(self.reference_min_box_size),
            "reference_chain_id": reference_chain_id,
            "current_chain_id": current_chain_id,
            "reference_polymer_residue_count": len(reference_residues),
            "current_polymer_residue_count": len(current_residues),
            "current_sequence_source": current_sequence_source,
            "sequence_alignment_blocks": [
                [int(start), int(end)] for start, end in sequence_alignment_blocks
            ],
            "sequence_alignment_mismatches": [
                [int(ref_idx), int(cur_idx), str(ref_aa), str(cur_aa)]
                for ref_idx, cur_idx, ref_aa, cur_aa in sequence_alignment_mismatches[:20]
            ],
            "sequence_alignment_mismatch_count": len(sequence_alignment_mismatches),
            "reference_alignment_mode": self.reference_alignment_mode,
            "reference_pocket_positions": [int(pos) for pos in self.reference_pocket_positions],
            "missing_reference_positions": missing_positions,
            "mapped_current_positions": mapped_current_positions,
            "residue_mappings": residue_mappings,
            "current_box_bounds": {
                "min": [float(x) for x in lower_bounds],
                "max": [float(x) for x in upper_bounds],
            },
            "current_box_coordinate_count": int(len(current_box_coords)),
            **dict(self._last_binding_site_box_debug or {}),
        }
        logging.info(
            "Current selected-residue pocket binding site calculated using %d residues%s; mapping pairs=%s",
            len(self.reference_pocket_positions) - len(missing_positions),
            "" if not missing_positions else f" (missing {len(missing_positions)})",
            [
                (row["reference_author_position"], row["current_seqid"])
                for row in residue_mappings
            ],
        )
        return binding_site

    @staticmethod
    def _selector_mapping(raw_selector: Any) -> Dict[str, str]:
        if raw_selector is None or isinstance(raw_selector, str):
            return {}
        if hasattr(raw_selector, "items"):
            return {
                str(key).strip(): str(value).strip()
                for key, value in raw_selector.items()
                if str(key).strip() and str(value).strip()
            }
        return {}

    @staticmethod
    def _single_selector(raw_selector: Any) -> Optional[str]:
        if raw_selector is None:
            return None
        if isinstance(raw_selector, str):
            token = raw_selector.strip()
            return token or None
        return None

    def _chain_selector_for_pocket_key(
        self,
        raw_selector: Any,
        pocket_key: str,
        *,
        allow_single_for_multichain: bool,
    ) -> str:
        mapping = self._selector_mapping(raw_selector)
        if mapping:
            return mapping.get(pocket_key, pocket_key)
        single = self._single_selector(raw_selector)
        if single and allow_single_for_multichain:
            return single
        return pocket_key

    def _calculate_binding_site_from_aligned_reference_pocket_by_chain(
        self,
        reference_model,
        current_model,
        *,
        current_structure_path: str,
        oligo_name: Optional[str],
    ) -> Dict[str, float]:
        if not self.reference_pocket_positions_by_chain:
            raise ValueError(
                "aligned_reference_pocket strategy requires nonempty reference_pocket_positions_by_chain"
            )
        single_chain_mode = len(self.reference_pocket_positions_by_chain) == 1
        current_box_coords = []
        missing_positions_by_chain: Dict[str, List[int]] = {}
        mapped_current_positions_by_chain: Dict[str, List[int]] = {}
        residue_mappings: List[Dict[str, Any]] = []
        sequence_alignment_blocks_by_chain: Dict[str, list[list[int]]] = {}
        sequence_alignment_mismatches_by_chain: Dict[str, list[list[Any]]] = {}
        sequence_alignment_mismatch_counts_by_chain: Dict[str, int] = {}
        current_sequence_sources: Dict[str, str] = {}
        chain_mappings: List[Dict[str, str]] = []

        for pocket_key, positions in self.reference_pocket_positions_by_chain.items():
            reference_chain_selector = self._chain_selector_for_pocket_key(
                self.reference_structure_chain,
                pocket_key,
                allow_single_for_multichain=single_chain_mode,
            )
            current_chain_selector = self._chain_selector_for_pocket_key(
                self.current_structure_chain,
                pocket_key,
                allow_single_for_multichain=single_chain_mode,
            )
            reference_chain_id = self._resolve_chain_id(reference_model, reference_chain_selector)
            current_chain_id = self._resolve_chain_id(current_model, current_chain_selector)
            chain_mappings.append(
                {
                    "pocket_key": pocket_key,
                    "reference_chain_id": reference_chain_id,
                    "current_chain_id": current_chain_id,
                }
            )

            reference_residues = self._collect_chain_residue_list(
                reference_model, reference_chain_id
            )
            current_residues = self._collect_chain_residue_list(
                current_model, current_chain_id
            )
            if len(reference_residues) != len(current_residues):
                raise DockingPenaltyError(
                    "Reference/current chain polymer residue counts do not match",
                    reason="binding_site_unavailable",
                    details={
                        "pocket_key": pocket_key,
                        "reference_chain_id": reference_chain_id,
                        "current_chain_id": current_chain_id,
                        "reference_polymer_residue_count": len(reference_residues),
                        "current_polymer_residue_count": len(current_residues),
                    },
                )
            reference_sequence = self._sequence_from_residue_list(reference_residues)
            current_sequence, current_sequence_source = self._current_design_sequence_for_chain(
                current_chain_id,
                current_residues,
                oligo_name=oligo_name,
            )
            current_sequence_sources[pocket_key] = current_sequence_source
            (
                current_idx_by_reference_idx,
                sequence_alignment_blocks,
                sequence_alignment_mismatches,
            ) = self._align_reference_sequence_to_current(
                reference_sequence,
                current_sequence,
                allow_mismatches=self._allow_reference_sequence_mismatch_mapping(),
                ordinal=self.reference_alignment_mode == "ordinal",
            )
            sequence_alignment_blocks_by_chain[pocket_key] = [
                [int(start), int(end)] for start, end in sequence_alignment_blocks
            ]
            sequence_alignment_mismatches_by_chain[pocket_key] = [
                [int(ref_idx), int(cur_idx), str(ref_aa), str(cur_aa)]
                for ref_idx, cur_idx, ref_aa, cur_aa in sequence_alignment_mismatches[:20]
            ]
            sequence_alignment_mismatch_counts_by_chain[pocket_key] = len(
                sequence_alignment_mismatches
            )
            if sequence_alignment_mismatches and not self._allow_reference_sequence_mismatch_mapping():
                raise DockingPenaltyError(
                    "Reference/current sequence alignment has residue mismatches",
                    reason="binding_site_unavailable",
                    details={
                        "pocket_key": pocket_key,
                        "current_sequence_source": current_sequence_source,
                        "mismatches": sequence_alignment_mismatches[:20],
                        "mismatch_count": len(sequence_alignment_mismatches),
                    },
                )
            try:
                reference_ordinal_by_author = self._author_position_to_ordinal_map(
                    reference_model, reference_chain_id
                )
            except ValueError as exc:
                raise DockingPenaltyError(
                    str(exc),
                    reason="binding_site_unavailable",
                    details={
                        "pocket_key": pocket_key,
                        "reference_chain_id": reference_chain_id,
                    },
                ) from exc

            missing_positions: List[int] = []
            mapped_current_positions: List[int] = []
            for pos in positions:
                ordinal_idx = reference_ordinal_by_author.get(int(pos))
                if ordinal_idx is None:
                    missing_positions.append(int(pos))
                    continue
                current_idx = current_idx_by_reference_idx.get(int(ordinal_idx))
                if current_idx is None:
                    missing_positions.append(int(pos))
                    continue
                reference_residue = reference_residues[ordinal_idx]
                residue = current_residues[current_idx]
                current_box_coords.append(self._box_coordinates_for_residue(residue))
                mapped_current_positions.append(int(residue.id[1]))
                residue_mappings.append(
                    {
                        "pocket_key": pocket_key,
                        "reference_chain_id": reference_chain_id,
                        "current_chain_id": current_chain_id,
                        "reference_author_position": int(pos),
                        "reference_ordinal_position": int(ordinal_idx + 1),
                        "reference_resname": str(reference_residue.get_resname()).strip(),
                        "current_seqid": int(residue.id[1]),
                        "current_ordinal_position": int(current_idx + 1),
                        "current_resname": str(residue.get_resname()).strip(),
                    }
                )
            missing_positions_by_chain[pocket_key] = missing_positions
            mapped_current_positions_by_chain[pocket_key] = mapped_current_positions

        if self.fail_on_error and any(missing_positions_by_chain.values()):
            raise DockingPenaltyError(
                "Not all selected pocket residues map to the current structure",
                reason="binding_site_unavailable",
                details={"missing_positions_by_chain": missing_positions_by_chain},
            )
        if not current_box_coords:
            raise DockingPenaltyError(
                "No selected pocket residues were found in the current structure",
                reason="binding_site_unavailable",
                details={
                    "missing_positions_by_chain": missing_positions_by_chain,
                    "mapped_current_positions_by_chain": mapped_current_positions_by_chain,
                },
            )

        all_box_coords = np.concatenate(current_box_coords, axis=0)
        binding_site, lower_bounds, upper_bounds = self._binding_site_from_box_coordinates(
            all_box_coords,
            context="aligned_reference_pocket multichain",
        )
        self._last_binding_site_debug = {
            "reference_structure_path": str(self.reference_structure_path),
            "current_structure_path": str(current_structure_path),
            "reference_box_geometry": self.reference_box_geometry,
            "reference_box_padding": float(self.reference_box_padding),
            "reference_min_box_size": float(self.reference_min_box_size),
            "chain_mappings": chain_mappings,
            "current_sequence_sources": current_sequence_sources,
            "sequence_alignment_blocks_by_chain": sequence_alignment_blocks_by_chain,
            "sequence_alignment_mismatches_by_chain": sequence_alignment_mismatches_by_chain,
            "sequence_alignment_mismatch_counts_by_chain": (
                sequence_alignment_mismatch_counts_by_chain
            ),
            "reference_alignment_mode": self.reference_alignment_mode,
            "reference_pocket_positions_by_chain": {
                str(chain): [int(pos) for pos in positions]
                for chain, positions in self.reference_pocket_positions_by_chain.items()
            },
            "missing_reference_positions_by_chain": missing_positions_by_chain,
            "mapped_current_positions_by_chain": mapped_current_positions_by_chain,
            "residue_mappings": residue_mappings,
            "current_box_bounds": {
                "min": [float(x) for x in lower_bounds],
                "max": [float(x) for x in upper_bounds],
            },
            "current_box_coordinate_count": int(len(all_box_coords)),
            **dict(self._last_binding_site_box_debug or {}),
        }
        logging.info(
            "Current selected-residue multichain pocket binding site calculated using %d residues; chain mappings=%s",
            len(residue_mappings),
            [
                (row["pocket_key"], row["reference_chain_id"], row["current_chain_id"])
                for row in chain_mappings
            ],
        )
        return binding_site


    @staticmethod
    def _extract_atom_coordinates(pdb_path: str) -> List[np.ndarray]:
        coords: List[np.ndarray] = []
        try:
            with open(pdb_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(("ATOM", "HETATM")):
                        try:
                            x_coord = float(line[30:38])
                            y_coord = float(line[38:46])
                            z_coord = float(line[46:54])
                            coords.append(np.array([x_coord, y_coord, z_coord], dtype=float))
                        except ValueError:
                            continue
        except FileNotFoundError:
            logging.error("Structure file not found when extracting coordinates: %s", pdb_path)
        return coords

    def _compute_structure_centroid(self, pdb_path: str) -> np.ndarray:
        coords = self._extract_atom_coordinates(pdb_path)
        if not coords:
            raise ValueError(f"No atoms found in {pdb_path} for centroid calculation")
        return np.mean(np.stack(coords, axis=0), axis=0)

    def _apply_translation_to_pdb(
        self, pdb_path: str, translation: np.ndarray, output_path: str
    ) -> str:
        tx, ty, tz = translation.astype(float)
        with open(pdb_path, "r", encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x_coord = float(line[30:38]) + tx
                        y_coord = float(line[38:46]) + ty
                        z_coord = float(line[46:54]) + tz
                        line = (
                            f"{line[:30]}{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}{line[54:]}"
                        )
                    except ValueError:
                        pass
                fout.write(line)
        return output_path

    @staticmethod
    def _transform_coordinate(
        coord: np.ndarray,
        transform: Mapping[str, Any],
        *,
        inverse: bool = False,
    ) -> np.ndarray:
        origin = np.asarray(transform["origin"], dtype=float)
        axes = np.asarray(transform["axes"], dtype=float)
        if inverse:
            return np.asarray(coord, dtype=float) @ axes.T + origin
        return (np.asarray(coord, dtype=float) - origin) @ axes

    def _apply_coordinate_transform_to_pdb(
        self,
        pdb_path: str,
        output_path: str,
        transform: Mapping[str, Any],
        *,
        inverse: bool = False,
    ) -> str:
        with open(pdb_path, "r", encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        coord = np.array(
                            [
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54]),
                            ],
                            dtype=float,
                        )
                        x_coord, y_coord, z_coord = self._transform_coordinate(
                            coord, transform, inverse=inverse
                        )
                        line = (
                            f"{line[:30]}{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}{line[54:]}"
                        )
                    except ValueError:
                        pass
                fout.write(line)
        return output_path

    def _coordinate_transformed_docking_inputs(
        self,
        protein_pdb_path: str,
        ligand_pdb_path: str,
    ) -> tuple[str, str]:
        transform = self._last_binding_site_transform
        if not transform:
            return protein_pdb_path, ligand_pdb_path
        protein_out = self._get_output_path(
            f"{Path(protein_pdb_path).stem}_oriented_box.pdb"
        )
        ligand_out = self._get_output_path(
            f"{Path(ligand_pdb_path).stem}_oriented_box.pdb"
        )
        self._apply_coordinate_transform_to_pdb(
            protein_pdb_path,
            str(protein_out),
            transform,
            inverse=False,
        )
        self._apply_coordinate_transform_to_pdb(
            ligand_pdb_path,
            str(ligand_out),
            transform,
            inverse=False,
        )
        return str(protein_out), str(ligand_out)

    def _merge_protein_ligand_files(
        self, protein_pdb_path: str, ligand_pdb_path: str, output_path: str
    ) -> str:
        with open(output_path, "w", encoding="utf-8") as fout:
            for path in (protein_pdb_path, ligand_pdb_path):
                with open(path, "r", encoding="utf-8") as fin:
                    content = fin.read().rstrip()
                    if content:
                        fout.write(content + "\n")
            fout.write("END\n")
        return output_path

    def _align_ligand_to_target(
        self,
        protein_pdb_path: str,
        ligand_pdb_path: str,
        target_center: np.ndarray,
        output_prefix: str,
    ) -> str:
        ligand_center = self._compute_structure_centroid(ligand_pdb_path)
        translation = np.asarray(target_center, dtype=float) - ligand_center
        translated_ligand_path = self._get_output_path(
            f"{output_prefix}_translated_ligand.pdb"
        )
        self._apply_translation_to_pdb(
            ligand_pdb_path, translation, translated_ligand_path
        )
        merged_output = self._get_output_path(f"{output_prefix}_aligned_complex.pdb")
        return self._merge_protein_ligand_files(
            protein_pdb_path, translated_ligand_path, merged_output
        )

    def _run_fpocket(self, protein_pdb_path: str, fpocket_exec: str) -> Path:
        base = Path(protein_pdb_path)
        output_dir_path = base.with_suffix("").with_name(base.stem + "_out")
        if output_dir_path.exists():
            shutil.rmtree(output_dir_path)
        cmd = [fpocket_exec, "-f", protein_pdb_path]
        logging.info("Running fpocket: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_dir_path

    def _get_fpocket_pocket_center(
        self, fpocket_dir: Path, pocket_rank: int
    ) -> np.ndarray:
        pocket_file = (
            fpocket_dir / "pockets" / f"pocket{max(1, pocket_rank)}_atm.pdb"
        )
        if not pocket_file.exists():
            raise FileNotFoundError(f"Fpocket pocket file missing: {pocket_file}")
        coords = self._extract_atom_coordinates(str(pocket_file))
        if not coords:
            raise ValueError(f"No atoms found in {pocket_file}")
        return np.mean(np.stack(coords, axis=0), axis=0)

    def convert_pdbqt_results_to_pdb(self, result_pdbqt_path: str) -> str:
        """Convert docking results from PDBQT to PDB format.

        Args:
            result_pdbqt_path: Path to PDBQT result file

        Returns:
            Path to converted PDB file
        """
        result_pdb_path = self._get_output_path(f"{Path(result_pdbqt_path).stem}.pdb")
        conversion_cmd = [
            self.obabel_executable,
            result_pdbqt_path,
            "-O",
            str(result_pdb_path),
        ]

        try:
            subprocess.run(conversion_cmd, check=True, capture_output=True, text=True)
            logging.info(f"Results converted to PDB format: {result_pdb_path}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"PDB conversion failed: {e.stderr}")

        return str(result_pdb_path)

    def preprocess(self, input_file) -> Any:
        """Preprocess input arguments for AutoDock.

        This method can be extended for specific preprocessing needs.

        Returns:
            Preprocessed arguments and keyword arguments
        """
        return input_file

    def _after_cache_store(self, cache_key, input_file, result):
        """Persist pose bank metadata for cache hits."""
        pose_bank = getattr(self, "_last_pose_candidates", None) or []
        if not pose_bank:
            pose_bank = [result]
        else:
            pose_bank = [p for p in pose_bank if os.path.exists(p)] or [result]
        self._set_cache_candidates(cache_key, pose_bank)
        super()._after_cache_store(cache_key, input_file, result)


@register_structure_processor("autodock_gpu")
class AutoDockGPUProcessor(AutoDockProcessorBase):
    """AutoDock-GPU implementation for molecular docking."""

    # todo not implemented
    def _generate_grid_parameter_file(
        self, protein_pdb_path, ligand_pdbqt_path, binding_site, gpf_output_path
    ) -> Tuple[str, str, str]:
        """Generate grid parameter file for AutoDock-GPU.

        Args:
            input_file_path: Path to input structure file

        Returns:
            Tuple of (gpf_file_path, ligand_pdbqt_path, binding_site)
        """

        return gpf_output_path, ligand_pdbqt_path, binding_site

    def run_autodock_gpu_docking(
        self,
        input_file_path,
    ) -> str:
        """Run AutoDock-GPU docking simulation.

        Args:
            protein_pdb_path: Path to protein PDB file
            ligand_pdbqt_path: Path to ligand PDBQT file
            binding_site: Binding site parameters

        Returns:
            Path to docking results file

        Raises:
            subprocess.CalledProcessError: If docking fails
        """
        input_file_path = str(input_file_path)
        # Generate grid files
        gpf_output_path = self._get_output_path(
            f"{Path(input_file_path).stem}_grid.gpf"
        )

        if input_file_path.endswith(".cif"):
            input_file_path = self._convert_cif_to_pdb(input_file_path)

        protein_pdb_path, ligand_pdb_path = self._separate_protein_ligand_with_prody(
            input_file_path
        )
        oligo_name = self._infer_oligo_name(input_file_path)
        binding_site = self._determine_binding_site(
            protein_pdb_path, ligand_pdb_path=ligand_pdb_path, oligo_name=oligo_name
        )
        self._write_binding_site_metadata(
            input_file_path=input_file_path,
            binding_site=binding_site,
        )
        protein_pdb_path, ligand_pdb_path = self._coordinate_transformed_docking_inputs(
            protein_pdb_path,
            ligand_pdb_path,
        )

        center_coords = [
            binding_site["center_x"],
            binding_site["center_y"],
            binding_site["center_z"],
        ]
        box_sizes = [
            binding_site["size_x"],
            binding_site["size_y"],
            binding_site["size_z"],
        ]

        ligand_pdbqt_path = ligand_pdb_path.replace(".pdb", ".pdbqt")
        self._prepare_ligand_with_obabel(ligand_pdb_path, ligand_pdbqt_path)
        identi = Path(input_file_path).stem
        if not self.mk_prepare_receptor:
            raise ValueError(
                "Missing mk_prepare_receptor path; set autodock.mk_prepare_receptor in env.yaml."
            )
        workdir = self._get_output_path("")
        receptor_cmd = [
            str(self.mk_prepare_receptor),
            "--read_pdb",
            str(protein_pdb_path),
            "-o",
            str(identi),
            "-g",
            "-p",
            "--box_center",
            str(center_coords[0]),
            str(center_coords[1]),
            str(center_coords[2]),
            "--box_size",
            str(box_sizes[0]),
            str(box_sizes[1]),
            str(box_sizes[2]),
        ]
        if self.mk_prepare_python_bin:
            receptor_cmd = [str(self.mk_prepare_python_bin)] + receptor_cmd
        subprocess.run(receptor_cmd, cwd=workdir, check=True, env=self._mk_prepare_env())

        autogrid_cmd = [
            str(self.autogrid4_executable),
            "-p",
            f"{identi}.gpf",
            "-l",
            f"{identi}_grid.glg",
        ]
        subprocess.run(autogrid_cmd, cwd=workdir, check=True)

        cmd = [
            str(self.autodock_gpu_executable),
            "--lfile",
            str(ligand_pdbqt_path),
            "--ffile",
            f"{identi}.maps.fld",
            "-n",
            str(self.num_docking_modes),
        ]
        subprocess.run(cmd, cwd=workdir, check=True)

    def _run_impl(self, input_file_path: str) -> str:
        """Implementation of the docking workflow."""
        return self._run_complete_autodock_gpu_workflow(input_file_path)

    def _run_complete_autodock_gpu_workflow(self, input_file_path: str) -> str:
        """Run complete AutoDock-GPU docking workflow.

        Args:
            input_file_path: Path to input structure file

        Returns:
            Path to final docking results in PDB format

        Raises:
            Exception: If any step in the workflow fails
        """
        logging.info("Starting complete AutoDock-GPU docking workflow")
        self._last_binding_site_debug = None
        self._last_binding_site_box_debug = None
        self._last_binding_site_transform = None

        try:
            # Step 4: Run docking
            result_pdbqt_path = self.run_autodock_gpu_docking(input_file_path)

            # Step 5: Convert results to PDB
            result_pdb_path = self.convert_pdbqt_results_to_pdb(result_pdbqt_path)

            return result_pdb_path

        except Exception as e:
            logging.error(f"Docking workflow failed: {str(e)}")
            raise


@register_structure_processor("autodock_vina")
class AutoDockVinaLigandPoseRescue(AutoDockProcessorBase):
    """AutoDock Vina implementation for molecular docking."""

    def _run_impl(self, input_file_path: str) -> str:
        """Implementation of the docking workflow."""
        return self._run_complete_vina_workflow(input_file_path)

    def _run_complete_vina_workflow(self, input_file_path: str) -> str:
        """Run complete AutoDock Vina docking workflow.

        Args:
            input_file_path: Path to input structure file

        Returns:
            Path to final docking results in PDB format

        Raises:
            Exception: If any step in the workflow fails
        """
        input_file_path = str(input_file_path)
        logging.info("Starting complete AutoDock Vina docking workflow")
        self._last_binding_site_debug = None
        self._last_binding_site_box_debug = None
        self._last_binding_site_transform = None

        try:
            # Step 1: Extract and separate structures
            if input_file_path.endswith(".cif"):
                input_file_path = self._convert_cif_to_pdb(input_file_path)
            protein_pdb_path, ligand_pdb_path = (
                self._separate_protein_ligand_with_prody(input_file_path)
            )
            # Step 3: Calculate binding site
            oligo_name = self._infer_oligo_name(input_file_path)
            binding_site = self._determine_binding_site(
                protein_pdb_path, ligand_pdb_path=ligand_pdb_path, oligo_name=oligo_name
            )
            self._write_binding_site_metadata(
                input_file_path=input_file_path,
                binding_site=binding_site,
            )
            protein_pdb_path, ligand_pdb_path = self._coordinate_transformed_docking_inputs(
                protein_pdb_path,
                ligand_pdb_path,
            )
            # Step 2: Prepare receptor and ligand
            receptor_pdbqt_path = protein_pdb_path.replace(".pdb", ".pdbqt")
            ligand_pdbqt_path = ligand_pdb_path.replace(".pdb", ".pdbqt")

            self._prepare_receptor_with_obabel(protein_pdb_path, receptor_pdbqt_path)
            self._prepare_ligand_with_obabel(ligand_pdb_path, ligand_pdbqt_path)

            # Step 4: Run Vina docking
            result_pdbqt_path = self._run_vina_docking(
                receptor_pdbqt_path,
                ligand_pdbqt_path,
                binding_site,
            )

            # Step 5: Convert results and merge with receptor
            merged_pdb_path = self._merge_ligand_with_receptor_with_retry(
                result_pdbqt_path,
                protein_pdb_path,
                receptor_pdbqt_path=receptor_pdbqt_path,
                ligand_pdbqt_path=ligand_pdbqt_path,
                binding_site=binding_site,
                inverse_transform=self._last_binding_site_transform,
            )
            self._write_penalty_metadata(
                input_file_path=input_file_path,
                penalty_active=False,
                reason="ok",
                message="",
            )

            return merged_pdb_path

        except DockingPenaltyError as e:
            if self.fail_on_error:
                raise
            logging.warning("Treating docking issue as score penalty: %s", str(e))
            self._write_penalty_metadata(
                input_file_path=input_file_path,
                penalty_active=True,
                reason=e.reason,
                message=str(e),
                details=e.details,
            )
            return self._materialize_penalty_artifact(input_file_path)
        except Exception as e:
            logging.error(f"Vina docking workflow failed: {str(e)}")
            raise

    def _run_vina_docking(
        self,
        receptor_pdbqt_path: str,
        ligand_pdbqt_path: str,
        binding_site: Dict[str, float],
        *,
        attempt: int = 0,
        exhaustiveness: Optional[int] = None,
        num_modes: Optional[int] = None,
    ) -> str:
        """Run AutoDock Vina docking simulation.

        Args:
            receptor_pdbqt_path: Path to receptor PDBQT file
            ligand_pdbqt_path: Path to ligand PDBQT file
            binding_site: Binding site parameters

        Returns:
            Path to docking results file

        Raises:
            subprocess.CalledProcessError: If docking fails
        """
        receptor_path = Path(receptor_pdbqt_path)
        if "_protein" not in receptor_path.name:
            raise ValueError(
                "Expected '_protein' in receptor filename for Vina output naming: "
                f"{receptor_path}"
            )
        output_name = receptor_path.name.replace("_protein", "_vina_result")
        if attempt > 0:
            output_name = output_name.replace(".pdbqt", f"_retry{attempt}.pdbqt")
        output_pdbqt_path = receptor_path.with_name(output_name)

        vina_cmd = [
            self.autodock_vina_executable,
            "--receptor",
            receptor_pdbqt_path,
            "--cpu",
            str(self.num_cpu_cores),
            "--ligand",
            ligand_pdbqt_path,
            "--out",
            str(output_pdbqt_path),
            "--center_x",
            str(binding_site["center_x"]),
            "--center_y",
            str(binding_site["center_y"]),
            "--center_z",
            str(binding_site["center_z"]),
            "--size_x",
            str(binding_site["size_x"]),
            "--size_y",
            str(binding_site["size_y"]),
            "--size_z",
            str(binding_site["size_z"]),
            "--exhaustiveness",
            str(exhaustiveness if exhaustiveness is not None else self.search_exhaustiveness),
            "--num_modes",
            str(num_modes if num_modes is not None else self.num_docking_modes),
        ]

        logging.info("Starting AutoDock Vina docking...")
        logging.info(f"Command: {' '.join(vina_cmd)}")

        try:
            result = subprocess.run(
                vina_cmd, check=True, capture_output=True, text=True
            )
            logging.info(result.stdout)
            return str(output_pdbqt_path)
        except subprocess.CalledProcessError as e:
            logging.error(f"Vina docking failed: {e.stderr}")
            raise

    def _merge_ligand_with_receptor_with_retry(
        self,
        docked_ligand_pdbqt_path: str,
        receptor_pdb_path: str,
        *,
        receptor_pdbqt_path: str,
        ligand_pdbqt_path: str,
        binding_site: Dict[str, float],
        inverse_transform: Optional[Mapping[str, Any]] = None,
    ) -> str:
        try:
            return self._merge_ligand_with_receptor(
                docked_ligand_pdbqt_path,
                receptor_pdb_path,
                inverse_transform=inverse_transform,
            )
        except DockingPenaltyError as exc:
            if not self._should_retry_alpha_shape_filter(exc):
                raise

            retry_results = self._collect_merged_ligand_results(
                docked_ligand_pdbqt_path,
                receptor_pdb_path,
                inverse_transform=inverse_transform,
            )
            for attempt in range(1, self.rank_alpha_shape_retry_attempts + 1):
                retry_pdbqt_path = self._run_vina_docking(
                    receptor_pdbqt_path,
                    ligand_pdbqt_path,
                    binding_site,
                    attempt=attempt,
                    exhaustiveness=self.rank_alpha_shape_retry_exhaustiveness,
                    num_modes=self.rank_alpha_shape_retry_num_modes,
                )
                retry_results.update(
                    self._collect_merged_ligand_results(
                        retry_pdbqt_path,
                        receptor_pdb_path,
                        inverse_transform=inverse_transform,
                        sample_id_offset=len(retry_results),
                    )
                )

            try:
                return self._filter_ranking(retry_results)
            except DockingPenaltyError as retry_exc:
                if retry_exc.reason != "alpha_shape_pose_filter_failed":
                    raise
                retry_exc.details = {
                    **retry_exc.details,
                    "initial_alpha_shape_occupancy_by_pose": exc.details.get(
                        "alpha_shape_occupancy_by_pose", {}
                    ),
                    "retry_attempts": int(self.rank_alpha_shape_retry_attempts),
                    "retry_exhaustiveness": (
                        self.rank_alpha_shape_retry_exhaustiveness
                        if self.rank_alpha_shape_retry_exhaustiveness is not None
                        else self.search_exhaustiveness
                    ),
                    "retry_num_modes": (
                        self.rank_alpha_shape_retry_num_modes
                        if self.rank_alpha_shape_retry_num_modes is not None
                        else self.num_docking_modes
                    ),
                }
                raise retry_exc

    def _should_retry_alpha_shape_filter(self, exc: DockingPenaltyError) -> bool:
        return (
            exc.reason == "alpha_shape_pose_filter_failed"
            and self.rank_alpha_shape_occupancy
            and self.rank_alpha_shape_retry_attempts > 0
        )

    def _merge_ligand_with_receptor(
        self,
        docked_ligand_pdbqt_path: str,
        receptor_pdb_path: str,
        inverse_transform: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Merge docking results with receptor structure into a single PDB file.

        Args:
            docked_ligand_pdbqt_path: Path to docked ligand PDBQT file
            receptor_pdb_path: Path to receptor PDB file

        Returns:
            Path to merged PDB file
        """

        return self._filter_ranking(
            self._collect_merged_ligand_results(
                docked_ligand_pdbqt_path,
                receptor_pdb_path,
                inverse_transform=inverse_transform,
            )
        )

    def _collect_merged_ligand_results(
        self,
        docked_ligand_pdbqt_path: str,
        receptor_pdb_path: str,
        *,
        inverse_transform: Optional[Mapping[str, Any]] = None,
        sample_id_offset: int = 0,
    ) -> Dict[int, Tuple[str, str]]:
        parser = PDBParser(QUIET=True)

        # Convert ligand PDBQT to PDB for parsing
        ligand_pdb_path = self.convert_pdbqt_results_to_pdb(
            docked_ligand_pdbqt_path
        )

        ligand_structure = parser.get_structure("ligand", ligand_pdb_path)
        receptor_structure = parser.get_structure("receptor", receptor_pdb_path)
        oligo_name = self._infer_oligo_name(receptor_pdb_path)
        configured_ligand_chain_id = (
            self._configured_ligand_chain_id_for_oligo(oligo_name)
            if oligo_name
            else None
        )
        configured_ligand_resname = (
            self._configured_ligand_resname_for_oligo(oligo_name)
            if oligo_name
            else None
        )
        results: Dict[int, Tuple[str, str]] = {}
        valid_pose_idx = 0
        for raw_model_idx, model in enumerate(ligand_structure):
            ligand_chains = list(model)
            if not ligand_chains:
                logging.warning(
                    "Skipping empty Vina PDB model %d from %s",
                    raw_model_idx,
                    ligand_pdb_path,
                )
                continue

            sample_id = sample_id_offset + valid_pose_idx
            merged_pdb_path = self._get_output_path(
                f"{Path(docked_ligand_pdbqt_path).stem}_merged_{valid_pose_idx}.pdb"
            )
            # Create merged structure
            merged_structure = Structure.Structure("merged")
            # Add receptor chains
            merged_model = receptor_structure[0].copy()

            # Add ligand chains
            merged_ligand_chain_ids = []
            for chain in ligand_chains:
                ligand_chain = chain.copy()
                if configured_ligand_chain_id is not None:
                    ligand_chain.id = configured_ligand_chain_id
                self._rename_chain_residues(ligand_chain, configured_ligand_resname)
                ligand_chain_id = str(ligand_chain.id).strip()
                if not ligand_chain_id:
                    raise ValueError(
                        f"Could not determine ligand chain ID for Vina pose {valid_pose_idx} "
                        f"(source model {raw_model_idx}) from {ligand_pdb_path}"
                    )
                merged_model.add(ligand_chain)
                merged_ligand_chain_ids.append(ligand_chain_id)

            if len(set(merged_ligand_chain_ids)) != 1:
                raise ValueError(
                    f"Expected one ligand chain in Vina pose {valid_pose_idx} "
                    f"(source model {raw_model_idx}) from {ligand_pdb_path}, "
                    f"got {merged_ligand_chain_ids!r}"
                )

            results[sample_id] = (merged_pdb_path, merged_ligand_chain_ids[0])

            merged_structure.add(merged_model)
            # Save merged structure
            io = PDBIO()
            io.set_structure(merged_structure)
            io.save(merged_pdb_path)
            if inverse_transform:
                merged_pdb_path = Path(merged_pdb_path)
                original_frame_path = self._get_output_path(
                    f"{merged_pdb_path.stem}_original_frame.pdb"
                )
                merged_pdb_path = Path(
                    self._apply_coordinate_transform_to_pdb(
                        str(merged_pdb_path),
                        str(original_frame_path),
                        inverse_transform,
                        inverse=True,
                    )
                )
                results[sample_id] = (str(merged_pdb_path), merged_ligand_chain_ids[0])
            valid_pose_idx += 1
        if not results:
            raise ValueError(
                f"No ligand chains found in converted Vina result {ligand_pdb_path} "
                f"from {docked_ligand_pdbqt_path}"
            )
        return results

    def _filter_ranking(self, results):
        # fixme currently hardcoded for CID sensor design util came up with a better idea to define function through expressions.
        """
        If multiple ligands are docked, filter and rank them based on defined criteria.
        if none of the ligands meet the criteria, return the input file.
        :return:
        """
        from astrolabe.losses.interaction import (
            calculate_buried_fraction_allala,
            calculate_buried_contribution_diff,
            calculate_com_diff,
        )

        structure_files = {k: v[0] for k, v in results.items()}
        buried_chain = {k: v[1] for k, v in results.items()}
        if self.rank_alpha_shape_occupancy:
            alpha_values = self._alpha_shape_occupancy_for_pose_files(
                structure_files=structure_files,
                ligand_chain_by_id=buried_chain,
            )
            passing_ids = [
                sample_id
                for sample_id, value in alpha_values.items()
                if value >= self.rank_alpha_shape_min_occupancy
            ]
            logging.info(
                "Alpha-shape pose filter retained %d/%d Vina poses at occupancy >= %.3f: %s",
                len(passing_ids),
                len(structure_files),
                self.rank_alpha_shape_min_occupancy,
                {int(k): float(v) for k, v in sorted(alpha_values.items())},
            )
            if not passing_ids:
                raise DockingPenaltyError(
                    "No Vina pose satisfied the alpha-shape pocket occupancy filter",
                    reason="alpha_shape_pose_filter_failed",
                    details={
                        "rank_alpha_shape_min_occupancy": float(
                            self.rank_alpha_shape_min_occupancy
                        ),
                        "alpha_shape_occupancy_by_pose": {
                            str(k): float(v) for k, v in sorted(alpha_values.items())
                        },
                    },
                )
            structure_files = {
                sample_id: structure_files[sample_id] for sample_id in passing_ids
            }
            buried_chain = {sample_id: buried_chain[sample_id] for sample_id in passing_ids}
        if self.rank_vina_order:
            return self._first_pose(structure_files, sorted(structure_files.keys()))

        metric_specs = []
        if self.rank_buried_fraction:
            metric_specs.append(
                ("buried_fraction", calculate_buried_fraction_allala, True)
            )
        if self.rank_com_diff:
            metric_specs.append(("com_diff", calculate_com_diff, False))
        if self.rank_buried_contrib:
            metric_specs.append(
                (
                    "buried_contribution_diff",
                    calculate_buried_contribution_diff,
                    True,
                )
            )

        if not metric_specs:
            return self._sample_pose(structure_files, list(structure_files.keys()))

        metric_values = {}
        criteria = {}
        for name, func, maximize in metric_specs:
            values = func(structure_files=structure_files, buried_chain=buried_chain)
            metric_values[name] = values
            if not values:
                continue
            target = max(values.values()) if maximize else min(values.values())
            criteria[name] = {
                "target": target,
                "tolerance": self.rank_tolerances.get(name, 0.1),
                "maximize": maximize,
            }

        if not criteria:
            return self._sample_pose(structure_files, list(structure_files.keys()))

        # Calculate excess scores for each sample and metric
        excess_scores = {}
        for sample_id in structure_files.keys():
            excess_scores[sample_id] = {}

            for metric_name, meta in criteria.items():
                metric_dict = metric_values.get(metric_name, {})
                value = metric_dict.get(
                    sample_id,
                    float("-inf") if meta["maximize"] else float("inf"),
                )
                target = meta["target"]
                tolerance = meta["tolerance"]
                if meta["maximize"]:
                    excess = max(0.0, target - value - tolerance)
                else:
                    excess = max(0.0, value - target - tolerance)
                excess_scores[sample_id][metric_name] = excess

        # Rank samples for each metric (ascending order - lower excess score is better)
        metric_rankings = {}
        for metric in criteria.keys():
            sorted_samples = sorted(
                structure_files.keys(), key=lambda x: excess_scores[x][metric]
            )
            metric_rankings[metric] = {
                sample_id: rank + 1 for rank, sample_id in enumerate(sorted_samples)
            }

        # Calculate total ranking score for each sample
        total_scores = {}
        for sample_id in structure_files.keys():
            total_scores[sample_id] = sum(
                metric_rankings[metric][sample_id] for metric in criteria.keys()
            )

        ordered_ids = sorted(total_scores.keys(), key=lambda x: total_scores[x])
        return self._sample_pose(structure_files, ordered_ids, weighted=True)

    def _alpha_shape_occupancy_for_pose_files(
        self,
        *,
        structure_files: Dict[int, str],
        ligand_chain_by_id: Dict[int, str],
    ) -> Dict[int, float]:
        if not self.reference_structure_path:
            raise DockingPenaltyError(
                "rank_alpha_shape_occupancy requires reference_structure_path",
                reason="alpha_shape_pose_filter_unavailable",
            )
        if not self.reference_pocket_positions and not self.reference_pocket_positions_by_chain:
            raise DockingPenaltyError(
                "rank_alpha_shape_occupancy requires reference pocket positions",
                reason="alpha_shape_pose_filter_unavailable",
            )
        from astrolabe.losses.interaction import (
            _current_pocket_cb_coords_from_structure_model,
            _fraction_inside_cb_alpha_shape,
            _heavy_atom_coordinates_for_model_chains,
            _reference_pocket_specs,
            _structure_model_from_file,
        )

        try:
            reference_model = _structure_model_from_file(str(self.reference_structure_path))
            pocket_specs = _reference_pocket_specs(
                reference_pocket_positions=self.reference_pocket_positions,
                reference_pocket_positions_by_chain=(
                    self.reference_pocket_positions_by_chain or None
                ),
                reference_structure_chain=self.reference_structure_chain,
                current_structure_chain=self.current_structure_chain,
            )
        except Exception as exc:
            raise DockingPenaltyError(
                f"Could not initialize alpha-shape pose filter: {exc}",
                reason="alpha_shape_pose_filter_unavailable",
            ) from exc

        scores: Dict[int, float] = {}
        for sample_id, structure_path in structure_files.items():
            try:
                current_model = _structure_model_from_file(structure_path)
                pocket_cb_coords = _current_pocket_cb_coords_from_structure_model(
                    reference_model=reference_model,
                    current_model=current_model,
                    pocket_specs=pocket_specs,
                    allow_length_mismatch=False,
                )
                ligand_coords = _heavy_atom_coordinates_for_model_chains(
                    current_model,
                    [ligand_chain_by_id[sample_id]],
                )
                scores[sample_id] = _fraction_inside_cb_alpha_shape(
                    ligand_coords,
                    pocket_cb_coords,
                    alpha_radius=self.alpha_radius,
                    alpha_radius_scale=self.alpha_radius_scale,
                    alpha_min_radius=self.alpha_min_radius,
                    alpha_max_radius=self.alpha_max_radius,
                    tolerance=self.rank_alpha_shape_tolerance,
                )
            except Exception as exc:
                raise DockingPenaltyError(
                    f"Could not compute alpha-shape occupancy for Vina pose {sample_id}: {exc}",
                    reason="alpha_shape_pose_filter_unavailable",
                    details={
                        "pose_id": int(sample_id),
                        "pose_path": str(structure_path),
                    },
                ) from exc
        return scores

    def _sample_pose(
        self,
        structure_files: Dict[int, str],
        ordered_ids: List[int],
        weighted: bool = False,
    ) -> str:
        if not ordered_ids:
            raise ValueError("No docking poses available for sampling")

        pose_paths = [structure_files[idx] for idx in ordered_ids]

        weights = None
        if weighted and len(ordered_ids) > 1:
            weights = [1.0 / (rank + 1) for rank in range(len(ordered_ids))]
            weights_sum = sum(weights)
            if weights_sum > 0:
                weights = [w / weights_sum for w in weights]
            else:
                weights = None

        self._last_pose_candidates = pose_paths

        if weights is None:
            return str(self.rng.choice(pose_paths))

        idx = self.rng.choice(len(pose_paths), p=weights)
        return pose_paths[idx]

    def _first_pose(
        self,
        structure_files: Dict[int, str],
        ordered_ids: List[int],
    ) -> str:
        if not ordered_ids:
            raise ValueError("No docking poses available for selection")
        pose_paths = [structure_files[idx] for idx in ordered_ids]
        self._last_pose_candidates = pose_paths[:1] if self.cache_first_pose_only else pose_paths
        return str(pose_paths[0])


@register_structure_processor("ligand_fpocket_alignment")
class LigandPocketAlignmentProcessor(AutoDockProcessorBase):
    """Align ligand centroid to an fpocket pocket center."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fpocket_executable = kwargs.get("fpocket_path", "fpocket")
        self.pocket_rank = int(kwargs.get("pocket_rank", 1))

    def _run_impl(self, input_file_path: str) -> str:
        input_file_path = str(input_file_path)
        if input_file_path.endswith(".cif"):
            input_file_path = self._convert_cif_to_pdb(input_file_path)

        protein_pdb_path, ligand_pdb_path = self._separate_protein_ligand_with_prody(
            input_file_path
        )

        try:
            fpocket_dir = self._run_fpocket(protein_pdb_path, self.fpocket_executable)
            pocket_center = self._get_fpocket_pocket_center(
                fpocket_dir, self.pocket_rank
            )
            logging.info(
                "Using fpocket pocket %d center at (%.2f, %.2f, %.2f)",
                self.pocket_rank,
                pocket_center[0],
                pocket_center[1],
                pocket_center[2],
            )
        except Exception as exc:
            logging.warning(
                "Fpocket pocket extraction failed (%s); using protein centroid instead",
                exc,
            )
            pocket_center = self._compute_structure_centroid(protein_pdb_path)

        output_prefix = f"fpocket_align_{Path(input_file_path).stem}"
        return self._align_ligand_to_target(
            protein_pdb_path, ligand_pdb_path, pocket_center, output_prefix
        )


@register_structure_processor("ligand_centroid_alignment")
class LigandCentroidAlignmentProcessor(AutoDockProcessorBase):
    """Translate ligand so its centroid matches the protein centroid."""

    def _run_impl(self, input_file_path: str) -> str:
        input_file_path = str(input_file_path)
        if input_file_path.endswith(".cif"):
            input_file_path = self._convert_cif_to_pdb(input_file_path)

        protein_pdb_path, ligand_pdb_path = self._separate_protein_ligand_with_prody(
            input_file_path
        )
        protein_center = self._compute_structure_centroid(protein_pdb_path)
        logging.info(
            "Protein centroid located at (%.2f, %.2f, %.2f)",
            protein_center[0],
            protein_center[1],
            protein_center[2],
        )
        output_prefix = f"centroid_align_{Path(input_file_path).stem}"
        return self._align_ligand_to_target(
            protein_pdb_path, ligand_pdb_path, protein_center, output_prefix
        )
