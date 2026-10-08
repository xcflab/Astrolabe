import os
import re
import logging
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from Bio import PDB
from omegaconf import DictConfig
import Bio.PDB
from Bio.PDB import PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser

from astrolabe.core import global_config
from astrolabe.structure_processors import StructureProcessor, register_structure_processor
from astrolabe.structure_processors.wrappers import RAPiDockRunner


class PeptideDockingError(Exception):
    """Custom exception for peptide docking operations."""

    pass


def load_pose_from_file(file_path: str, pyrosetta):
    """Load PyRosetta pose from file."""
    file_path_str = str(file_path)

    if file_path_str.endswith(".pdb"):
        return pyrosetta.pose_from_pdb(file_path_str)
    elif file_path_str.endswith(".cif"):
        return pyrosetta.pose_from_file(
            file_path_str,
            type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file,
            read_fold_tree=True,
        )
    else:
        return pyrosetta.pose_from_file(file_path_str)


def compute_delta_sasa(complex_path: str, hotspot_mask: np.ndarray) -> float:
    """Calculate delta SASA for hotspot residues."""
    import pyrosetta
    from pyrosetta.rosetta.core.scoring.sasa import (
        rel_per_res_sc_sasa as sasa_calculator,
    )

    try:
        pose = load_pose_from_file(complex_path, pyrosetta)

        # Calculate SASA for the complex
        complex_sasa = sasa_calculator(pose)
        sasa_bound = list(complex_sasa)

        # Calculate SASA for individual chains
        sasa_unbound = []
        split_poses = pose.split_by_chain()
        for chain_pose in split_poses:
            chain_sasa = sasa_calculator(chain_pose)
            sasa_unbound.extend(chain_sasa)

        # Calculate delta SASA for hotspot residues
        delta_sasa_per_residue = np.array(sasa_unbound) - np.array(sasa_bound)
        return float(np.sum(delta_sasa_per_residue * hotspot_mask))

    except Exception as e:
        logging.error(f"SASA calculation failed for {complex_path}: {e}")
        return 0.0


def merge_pdb_structures(
    peptide_file: str, protein_file: str, output_file: str, peptide_chain_index: int
) -> str:
    """Merge peptide and protein structures into a single complex."""

    try:
        parser = PDB.PDBParser(QUIET=True)
        protein_structure = parser.get_structure("protein", protein_file)
        peptide_structure = parser.get_structure("peptide", peptide_file)

        # Create merged structure
        merged_structure = PDB.Structure.Structure("merged")
        model = PDB.Model.Model(0)
        merged_structure.add(model)

        # Add protein chains
        for chain in protein_structure[0]:
            model.add(chain)

        # Insert peptide chains at specified index
        for chain in peptide_structure[0]:
            model.insert(peptide_chain_index, chain)

        # Save merged structure
        io = PDBIO()
        io.set_structure(merged_structure)
        io.save(output_file)

        return output_file

    except Exception as e:
        raise PeptideDockingError(f"Structure merging failed: {e}") from e


def parse_structure_file(input_file: str):
    """Parse structure file (PDB or CIF)."""

    if input_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)

    return parser.get_structure("complex", input_file)


def analyze_chain_lengths(structure) -> dict:
    """Analyze chain lengths in the structure."""
    chain_lengths = {}
    chain_order = []

    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_lengths[chain_id] = len(list(chain.get_residues()))
            if chain_id not in chain_order:
                chain_order.append(chain_id)

    return {"lengths": chain_lengths, "order": chain_order}


def identify_peptide_chain(
    chain_info: dict, max_peptide_length: int = 50
) -> Tuple[str, int]:
    """Identify the shortest chain as peptide."""
    chain_lengths = chain_info["lengths"]
    chain_order = chain_info["order"]

    peptide_chain_id = min(chain_lengths, key=chain_lengths.get)
    peptide_length = chain_lengths[peptide_chain_id]

    if peptide_length > max_peptide_length:
        raise PeptideDockingError(
            f"No chain shorter than {max_peptide_length} residues found. "
            f"Shortest chain has {peptide_length} residues."
        )

    peptide_chain_index = chain_order.index(peptide_chain_id)
    return peptide_chain_id, peptide_chain_index


def save_structure_chains(
    structure, chain_id_to_exclude: str, output_file: str, exclude_mode: bool = True
):
    """Save structure chains, either excluding or including specified chain."""

    output_structure = structure.copy()

    for model in output_structure:
        if exclude_mode:
            # Remove specified chain (for protein output)
            chains_to_remove = [
                chain for chain in model if chain.id == chain_id_to_exclude
            ]
        else:
            # Keep only specified chain (for peptide output)
            chains_to_remove = [
                chain for chain in model if chain.id != chain_id_to_exclude
            ]

        for chain in chains_to_remove:
            model.detach_child(chain.id)

    io = PDBIO()
    io.set_structure(output_structure)
    io.save(str(output_file))


def extract_oligomer_name_from_filename(input_file: str) -> str:
    """Extract oligomer name from filename using regex."""
    pattern = r".*step_\d+_(\w+)_model.*"
    match = re.match(pattern, Path(input_file).stem)

    if not match:
        raise PeptideDockingError(f"Could not extract oligomer name from {input_file}")

    return match.group(1)


def create_hotspot_mask(oligo_name: str, hotspot_alias: str) -> np.ndarray:
    """Create boolean mask for hotspot residues."""
    try:
        hotspot_masks = []

        for protomer_id in oligo_name:
            if protomer_id in global_config.protomers:
                mask = global_config.protomers[protomer_id].select_residues_by_alias(
                    hotspot_alias
                )
                hotspot_masks.append(mask)
            else:
                logging.warning(f"Protomer {protomer_id} not found in global config")

        if not hotspot_masks:
            raise PeptideDockingError(f"No valid protomers found for {oligo_name}")

        return np.concatenate(hotspot_masks).astype(bool)

    except Exception as e:
        raise PeptideDockingError(f"Failed to create hotspot mask: {e}") from e


@register_structure_processor("RAPiDock")
class RAPiDockPeptideDockingProcessor(StructureProcessor):
    """Predocker for peptide-protein docking using RAPiDock."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Configuration parameters
        self.hotspot = kwargs.get("hotspot", None)
        self.max_peptide_length = kwargs.get("max_peptide_length", 50)

        # Validate required parameters
        model_dir = kwargs.get("model_dir")
        checkpoint = kwargs.get("ckpt")
        if not model_dir or not checkpoint:
            raise PeptideDockingError("model_dir and ckpt parameters are required")

        # Initialize long-lived RAPiDock runner.
        self.runner = RAPiDockRunner(kwargs)

        # Set output directory
        self.output_dir = global_config.out_dir / "predocker"

    def preprocess(self, input_file) -> Any:
        """Preprocess input file."""
        # TODO: Handle PTM modifications that might interfere with docking
        return input_file

    def _run_impl(self, input_file: str) -> str:
        """Main docking implementation."""
        try:
            # Split complex into protein and peptide
            protein_file, peptide_file, peptide_chain_index = (
                self._split_complex_chains(input_file)
            )

            # Run RAPiDock docking
            candidate_paths = self._run_rapidock_inference(
                input_file, protein_file, peptide_file
            )
            materialized_candidates = [
                self._materialize_candidate_path(
                    input_file=input_file,
                    candidate_path=candidate_path,
                )
                for candidate_path in candidate_paths
            ]

            peptide_files = [Path(path) for path in materialized_candidates]
            peptide_files.append(Path(peptide_file))
            # Extract oligomer name and evaluate results
            oligo_name = extract_oligomer_name_from_filename(input_file)
            best_result = self._evaluate_docking_results(
                peptide_files, protein_file, peptide_chain_index, oligo_name
            )

            return best_result

        except Exception as e:
            logging.error(
                f"Peptide docking failed for {input_file}: {e}", exc_info=True
            )
            raise PeptideDockingError(f"Docking pipeline failed: {e}") from e

    def _materialize_candidate_path(self, *, input_file: str, candidate_path: str) -> str:
        """Copy RAPiDock bank outputs into the predocker debug dir before downstream use."""

        src = Path(candidate_path)
        if not src.exists():
            raise PeptideDockingError(
                f"RAPiDock candidate path does not exist: {candidate_path}"
            )
        dst = Path(
            self._get_output_path(f"{Path(input_file).stem}__{src.name}")
        )
        if src.resolve() == dst.resolve():
            return str(dst)
        shutil.copy2(src, dst)
        return str(dst)

    def _run_rapidock_inference(
        self, input_file: str, protein_file: str, peptide_file: str
    ) -> List[str]:
        """Request the cached RAPiDock candidate bank for this protein/peptide pair."""
        return self.runner.dock_candidates(
            name=Path(input_file).stem,
            protein_pdb=protein_file,
            peptide_pdb=peptide_file,
        )

    def _split_complex_chains(self, input_file: str) -> Tuple[str, str, int]:
        """Split complex into protein and peptide files."""
        try:
            # Parse structure
            structure = parse_structure_file(input_file)

            # Analyze chains and identify peptide
            chain_info = analyze_chain_lengths(structure)
            peptide_chain_id, peptide_chain_index = identify_peptide_chain(
                chain_info, self.max_peptide_length
            )

            # Generate output paths
            base_name = Path(input_file).stem
            protein_file = self._get_output_path(f"{base_name}_protein.pdb")
            peptide_file = self._get_output_path(f"{base_name}_peptide.pdb")

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save protein chains (exclude peptide)
            save_structure_chains(
                structure, peptide_chain_id, protein_file, exclude_mode=True
            )

            # Save peptide chain only
            save_structure_chains(
                structure, peptide_chain_id, peptide_file, exclude_mode=False
            )

            return str(protein_file), str(peptide_file), peptide_chain_index

        except Exception as e:
            raise PeptideDockingError(f"Chain splitting failed for {input_file}: {e}") from e

    def _evaluate_docking_results(
        self,
        peptide_files: List[Path],
        protein_file: str,
        peptide_chain_index: int,
        oligo_name: str,
    ) -> str:
        """Evaluate docking results and select the best one."""

        if not self.hotspot:
            logging.warning("No hotspot specified, returning first merged result")
            output_file = self._get_output_path(f"{peptide_files[0].stem}_merged.pdb")
            return merge_pdb_structures(
                str(peptide_files[0]),
                protein_file,
                output_file,
                peptide_chain_index,
            )

        # Create merged complexes
        merged_files = []
        for peptide_file in peptide_files:
            output_file = self._get_output_path(f"{peptide_file.stem}_merged.pdb")
            merged_file = merge_pdb_structures(
                str(peptide_file), protein_file, output_file, peptide_chain_index
            )
            merged_files.append(merged_file)

        # Calculate buried surface areas
        hotspot_mask = create_hotspot_mask(oligo_name, self.hotspot)
        sasa_scores = {}

        for merged_file in merged_files:
            try:
                dsasa = compute_delta_sasa(merged_file, hotspot_mask)
                sasa_scores[merged_file] = dsasa
            except Exception as e:
                logging.warning(f"SASA calculation failed for {merged_file}: {e}")
                sasa_scores[merged_file] = 0.0

        if not sasa_scores:
            raise PeptideDockingError("No valid SASA scores calculated")

        # Select best result
        best_file = max(sasa_scores, key=sasa_scores.get)
        best_score = sasa_scores[best_file]

        logging.info(f"Selected best result: {best_file} (SASA: {best_score:.3f})")
        return best_file
