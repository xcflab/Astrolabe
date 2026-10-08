import logging
import os
from pathlib import Path
from typing import Any
import importlib
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO, Structure, Chain

if importlib.util.find_spec("PLACER") is not None:
    import PLACER
    from PLACER import runner

from astrolabe.structure_processors import StructureProcessor, register_structure_processor


@register_structure_processor("dimer_interface_placer")
class DimerInterfacePlacementProcessor(StructureProcessor):
    """
    A utility class for analyzing protein dimer interfaces, finding cavity centers,
    and moving specified molecular chains to that center.

    This class encapsulates all steps from loading structures to analysis,
    manipulation, and saving.

    Usage:
    1. Instantiate the class with a PDB/CIF file path.
    2. Call the analyze_and_place() method to execute the complete workflow.

    Example:
        placer = DimerInterfacePlacementProcessor("my_dimer.pdb")
        results = placer.analyze_and_place(
            target_chain_id="L",
            output_file="result.pdb"
        )
        print(f"Cavity center coordinates: {results['cavity_center']}")
    """

    def __init__(self, **kwargs):
        """
        Initialize DimerInterfacePlacementProcessor.

        Parameters:
        -----------
        pdb_cif_file : str
            Path to the input PDB or CIF file.
        """
        super().__init__(**kwargs)

        self.radius_cutoff = kwargs.get("radius_cutoff", 5.0)
        self.window_size = kwargs.get("window_size", 3.0)
        self.step_size = kwargs.get("step_size", 0.5)
        self.density_threshold = kwargs.get("density_threshold", None)

    def _run_impl(
        self,
        input_file: str,
    ) -> dict:
        """
        Implementation of the predocker run method.
        """
        output_file = self._get_output_path(f"dip_{Path(input_file).stem}.pdb")
        self.chain1, self.chain2, target_chain = self._get_target_chain(input_file)
        return self.analyze_and_place(
            target_chain=target_chain,
            input_file=input_file,
            output_file=output_file,
            radius_cutoff=self.radius_cutoff,
            window_size=self.window_size,
            step_size=self.step_size,
            density_threshold=self.density_threshold,
        )

    def preprocess(self, input_file) -> Any:
        """Preprocess input arguments for AutoDock.

        This method can be extended for specific preprocessing needs.

        Returns:
            Preprocessed arguments and keyword arguments
        """
        return input_file

    def _get_target_chain(self, input_file):
        self.structure = self._load_structure(input_file)

        # Get the first two protein chains as chain1 and chain2
        # Get the first ligand chain as target_chain
        flag = False
        for model in self.structure:
            chains = list(model.get_chains())
            if len(chains) < 2:
                error_msg = "Structure must contain at least two chains to define dimer interface."

                raise ValueError(error_msg)

            chain1 = chains[0]
            chain2 = chains[1]
            # Find the first chain that is not a protein (i.e., not a CA atom)
            for chain in chains:
                if all(res.id[0] != " " for res in chain.get_residues()):
                    target_chain = chain
                    flag = True
                    break
        if not flag:
            error_msg = "No suitable target chain found in the structure."

            raise ValueError(error_msg)
        else:
            return chain1, chain2, target_chain

    def _load_structure(self, input_file) -> Structure.Structure:
        """Load PDB or CIF file based on file extension."""

        if input_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)

        else:
            parser = PDBParser(QUIET=True)

        try:
            structure = parser.get_structure("protein_dimer", input_file)
            num_chains = len(list(structure.get_chains()))
            if num_chains < 2:
                error_msg = "Structure must contain at least two chains to define dimer interface."

                raise ValueError(error_msg)
            return structure
        except Exception as e:
            raise

    def _get_chain_centroid(self, chain: Chain.Chain) -> np.ndarray:
        """
        Calculate the centroid of a chain, intelligently adapting to proteins or small molecules.

        First attempts to use CA atoms (suitable for proteins), if that fails,
        uses all heavy atoms.
        """

        coords = []

        # Try CA atoms first
        ca_count = 0
        for res in chain:
            if res.has_id("CA") and len(res.get_resname()) <= 3:
                coords.append(res["CA"].get_coord())
                ca_count += 1

        if coords:

            logging.debug(
                f"Using {len(coords)} CA atoms to calculate centroid for chain {chain.id}."
            )
            centroid = np.mean(coords, axis=0)

            return centroid

        # Fall back to heavy atoms if no CA atoms found

        heavy_atom_count = 0
        for residue in chain:
            for atom in residue:
                if not atom.element.startswith("H") and atom.element.strip() != "":
                    coords.append(atom.get_coord())
                    heavy_atom_count += 1

        if coords:
            logging.debug(
                f"Using {len(coords)} heavy atoms to calculate centroid for chain {chain.id}."
            )
            centroid = np.mean(coords, axis=0)

            return centroid

        error_msg = (
            f"No usable atoms found in chain {chain.id} for centroid calculation."
        )

        raise ValueError(error_msg)

    def _find_low_density_region(
        self,
        projected_coords: np.ndarray,
        window_size: float,
        step_size: float,
        axis_length: float,
        density_threshold: float = None,
    ) -> float:
        """
        Internal method: Use sliding window to find the center of the lowest density region
        in projected coordinates.
        """

        if len(projected_coords) == 0:

            logging.warning("No CA atoms found near axis, returning axis midpoint.")
            return axis_length / 2.0

        # Limit analysis range to between the two centroids
        valid_mask = (projected_coords >= 0.0) & (projected_coords <= axis_length)
        projected_coords = projected_coords[valid_mask]

        if len(projected_coords) == 0:

            logging.warning("No CA atoms between centroids, returning axis midpoint.")
            return axis_length / 2.0

        min_proj, max_proj = 0.0, axis_length
        if max_proj - min_proj < window_size:

            logging.warning(
                f"Axis length {max_proj - min_proj:.2f}Å smaller than window size {window_size:.2f}Å, returning axis midpoint."
            )
            return (min_proj + max_proj) / 2.0

        # Generate sliding window center points
        window_centers = np.arange(
            min_proj + window_size / 2, max_proj - window_size / 2, step_size
        )

        if len(window_centers) == 0:

            return (min_proj + max_proj) / 2.0

        # Calculate density for each window
        densities = []
        for i, center in enumerate(window_centers):
            win_start = center - window_size / 2
            win_end = center + window_size / 2
            atoms_in_window = np.sum(
                (projected_coords >= win_start) & (projected_coords < win_end)
            )
            density = atoms_in_window / window_size
            densities.append(density)
            # if i < 5 or i % 10 == 0:
            #     print(f"DEBUG: Window {i}: center={center:.2f}, atoms={atoms_in_window}, density={density:.3f}")

        densities = np.array(densities)
        min_density = np.min(densities)

        # Find all windows with minimum density
        if density_threshold is not None and min_density <= density_threshold:
            candidate_indices = np.where(densities <= density_threshold)[0]

        else:
            candidate_indices = np.where(densities == min_density)[0]

        # Find the center of the longest continuous low-density region
        # (Logic consistent with original script, selecting center of longest continuous region)
        if len(candidate_indices) > 1:

            # Find continuous regions and select the longest one
            longest_region_center = window_centers[candidate_indices[0]]
            max_len = 0
            current_len = 0

            # Simplified method to find the center of the longest continuous region
            for i in range(len(candidate_indices)):
                current_len += 1
                if (
                    i == len(candidate_indices) - 1
                    or candidate_indices[i + 1] != candidate_indices[i] + 1
                ):
                    if current_len > max_len:
                        max_len = current_len
                        # Calculate the center of this region
                        start_idx = candidate_indices[i - current_len + 1]
                        end_idx = candidate_indices[i]
                        center_pos = (
                            window_centers[start_idx] + window_centers[end_idx]
                        ) / 2.0
                        longest_region_center = center_pos

                    current_len = 0
            best_proj = longest_region_center
        else:
            best_proj = window_centers[candidate_indices[0]]

        logging.debug(
            f"Found lowest density region at projection position {best_proj:.2f} Å."
        )
        return best_proj

    def _calculate_cavity_center(
        self,
        radius_cutoff: float,
        window_size: float,
        step_size: float,
        density_threshold: float = None,
    ) -> dict:
        """
        Internal method: Execute cavity center calculation.
        """

        logging.debug("=== Step 1: Calculate cavity center ===")

        # 1. Calculate centroids and axis

        centroid1 = self._get_chain_centroid(self.chain1)
        centroid2 = self._get_chain_centroid(self.chain2)
        axis_vector = centroid2 - centroid1
        axis_length = np.linalg.norm(axis_vector)
        axis_point = centroid1
        axis_unit = axis_vector / axis_length

        logging.debug(f"Chain {self.chain1.id} centroid: {np.round(centroid1, 2)}")
        logging.debug(f"Chain {self.chain2.id} centroid: {np.round(centroid2, 2)}")
        logging.debug(f"Interface axis length: {axis_length:.2f} Å")

        # 2. Collect all CA atoms and filter

        all_ca_coords = np.array(
            [
                res["CA"].get_coord()
                for model in self.structure
                for chain in model
                for res in chain
                if res.has_id("CA") and len(res.get_resname()) <= 3
            ]
        )

        vectors_to_points = all_ca_coords - axis_point
        proj_points = (
            axis_point + np.dot(vectors_to_points, axis_unit)[:, np.newaxis] * axis_unit
        )
        distances = np.linalg.norm(all_ca_coords - proj_points, axis=1)

        ca_near_axis = all_ca_coords[distances < radius_cutoff]

        logging.debug(
            f"Found {len(ca_near_axis)} CA atoms within {radius_cutoff}Å radius of axis."
        )

        # 3. Project and find low density region

        vectors_to_points_near = ca_near_axis - axis_point
        projected_coords = np.dot(vectors_to_points_near, axis_unit)

        cavity_proj = self._find_low_density_region(
            projected_coords, window_size, step_size, axis_length, density_threshold
        )

        # 4. Calculate 3D cavity center coordinates
        cavity_center = axis_point + cavity_proj * axis_unit

        logging.debug(f"✓ Calculated cavity center: {np.round(cavity_center, 2)}")

        return {
            "cavity_center": cavity_center,
            "axis_point": axis_point,
            "axis_vector": axis_vector,
            "axis_length": axis_length,
        }

    def _move_chain_to_cavity(self, target_chain, cavity_center: np.ndarray) -> dict:
        """
        Internal method: Move the specified chain to the cavity center.
        """

        original_center = self._get_chain_centroid(target_chain)
        translation_vector = cavity_center - original_center

        logging.debug(f"Target chain original centroid: {np.round(original_center, 2)}")
        logging.debug(f"Translation vector: {np.round(translation_vector, 2)}")
        logging.debug(
            f"Translation distance: {np.linalg.norm(translation_vector):.2f} Å"
        )

        # Apply translation

        atom_count = 0
        for atom in target_chain.get_atoms():
            atom.set_coord(atom.get_coord() + translation_vector)
            atom_count += 1

        new_center = self._get_chain_centroid(target_chain)

        logging.debug(
            f"✓ Moved {atom_count} atoms. New centroid after movement: {np.round(new_center, 2)}"
        )

        return {
            "original_center": original_center,
            "new_center": new_center,
            "translation_distance": np.linalg.norm(translation_vector),
        }

    def save_structure(self, output_file: str):
        """
        Save the current (possibly modified) structure to a file.

        Parameters:
        -----------
        output_file : str
            Path to the output PDB or CIF file.
        """

        if not (output_file.endswith(".pdb") or output_file.endswith(".cif")):
            error_msg = "Output file must be in .pdb or .cif format."

            raise ValueError(error_msg)

        if output_file.endswith(".cif"):
            io = MMCIFIO()

        else:
            io = PDBIO()

        io.set_structure(self.structure)
        io.save(output_file)

        logging.debug(f"✓ Successfully saved modified structure to '{output_file}'.")

    def analyze_and_place(
        self,
        target_chain,
        input_file: str,
        output_file: str,
        radius_cutoff: float = 5.0,
        window_size: float = 3.0,
        step_size: float = 0.5,
        density_threshold: float = None,
    ) -> dict:
        """
        Execute the complete analysis and placement workflow.

        This method first calculates the cavity center of the dimer interface,
        then moves the specified target chain to that center, and finally
        saves the result to an output file.

        Parameters:
        -----------
        target_chain_id : str
            ID of the target chain to be moved to the cavity center.
        output_file : str
            Path to save the modified structure (PDB or CIF file).
        radius_cutoff : float, optional
            Maximum radius from axis when defining interface region (Å), default 5.0.
        window_size : float, optional
            Size of sliding window for finding low-density regions (Å), default 3.0.
        step_size : float, optional
            Step size of sliding window (Å), default 0.5.
        density_threshold : float or None, optional
            Density threshold. If the minimum density found is below this value,
            all regions below this value will be considered, not just the lowest
            density region. Default is None.

        Returns:
        --------
        dict:
            A dictionary containing all analysis and operation results.
        """

        # Step 1: Analyze cavity

        cavity_results = self._calculate_cavity_center(
            radius_cutoff, window_size, step_size, density_threshold
        )

        # Step 2: Move chain

        move_results = self._move_chain_to_cavity(
            target_chain, cavity_results["cavity_center"]
        )

        # Step 3: Save file

        self.save_structure(output_file)

        return output_file


@register_structure_processor("chemnet_placer")
class ChemNetPlacementProcessor(StructureProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if importlib.util.find_spec("PLACER") is None:
            raise ImportError(
                "PLACER module not found. Please install PLACER to use ChemNetPlacementProcessor."
            )
        self.num_samples = kwargs.get("num_samples", 5)
        self.placer = runner.PLACER()
        self.pl_inp = runner.PLACERinput()

    def _get_target_chain(self, input_file):
        self.structure = self._load_structure(input_file)

        # Get the first two protein chains as chain1 and chain2
        # Get the first ligand chain as target_chain
        flag = False
        for model in self.structure:
            chains = list(model.get_chains())
            if len(chains) < 2:
                error_msg = "Structure must contain at least two chains to define dimer interface."

                raise ValueError(error_msg)

            chain1 = chains[0]
            chain2 = chains[1]
            # Find the first chain that is not a protein (i.e., not a CA atom)
            for chain in chains:
                if all(res.id[0] != " " for res in chain.get_residues()):
                    target_chain = chain
                    flag = True
                    break
        if not flag:
            error_msg = "No suitable target chain found in the structure."

            raise ValueError(error_msg)
        else:
            return chain1, chain2, target_chain

    def _load_structure(self, input_file) -> Structure.Structure:
        """Load PDB or CIF file based on file extension."""

        if input_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)

        else:
            parser = PDBParser(QUIET=True)

        try:
            structure = parser.get_structure("protein_dimer", input_file)
            num_chains = len(list(structure.get_chains()))
            if num_chains < 2:
                error_msg = "Structure must contain at least two chains to define dimer interface."

                raise ValueError(error_msg)
            return structure
        except Exception as e:
            raise

    def preprocess(self, input_file) -> Any:
        """Preprocess input arguments for AutoDock.

        This method can be extended for specific preprocessing needs.

        Returns:
            Preprocessed arguments and keyword arguments
        """
        # todo if ptm shown in input_file, ChemNet will view it as a ligand and throw an error because it cannot handle multiple ligands
        return input_file

    def _run_impl(self, input_file: str) -> str:
        """
        Implementation of the predocker run method.
        """
        self.target_chain_id = self._get_target_chain(input_file)[2].id
        output_file = self._get_output_path(f"chemnet_{Path(input_file).stem}.pdb")
        return self.resample_chain(input_file, self.target_chain_id, output_file)

    def resample_chain(self, input_file, target_chain_id, output_file):
        """
        Resample the third chain from the dimer structure.
        """

        if target_chain_id is None:
            raise ValueError("third_chain must be specified")
        # import importlib.util
        #

        if input_file.endswith(".pdb"):
            self.pl_inp.pdb(input_file)
        elif input_file.endswith(".cif"):
            self.pl_inp.cif(input_file)
        self.pl_inp.name(os.path.basename(input_file))
        # todo continuously sampling, not just one time, as long as the inputs not changed, this function should return one of previously sampled results
        # get the ligand name by taking the first three characters of the third chain
        outputs = self.placer.run(self.pl_inp, self.num_samples)

        prmsd_index = [(i, output["prmsd"]) for i, output in outputs.items()]
        lowest_prmsd_index = sorted(prmsd_index, key=lambda x: x[1])[0][0]
        to_path = input_file.replace(".pdb", "_resampled.pdb")

        with open(to_path, "w") as f:
            f.write(outputs[lowest_prmsd_index]["model"])

        # take third chain coordinates from the resampled structure, replacing the original third chain
        from Bio import PDB

        # Parse original structure
        parser = PDB.PDBParser(QUIET=True)
        original_structure = parser.get_structure("original", input_file)

        # Parse resampled structure
        resampled_structure = parser.get_structure("resampled", to_path)

        # Get the third chain from resampled structure
        resampled_chain = None
        for model in resampled_structure:
            for chain in model:
                if chain.id == target_chain_id:
                    resampled_chain = chain
                    break
            if resampled_chain:
                break

        if resampled_chain is None:
            raise ValueError(
                f"Chain {target_chain_id} not found in resampled structure"
            )

        # Replace the third chain in original structure
        for model in original_structure:
            if target_chain_id in [chain.id for chain in model]:
                model.detach_child(target_chain_id)
            model.add(resampled_chain.copy())

        # Write the modified structure
        io = PDB.PDBIO()
        io.set_structure(original_structure)
        io.save(output_file)
        os.remove(to_path)  # Clean up the resampled file

        return output_file
