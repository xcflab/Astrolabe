from pathlib import Path
from typing import Any

import numpy as np
import biotite.structure.io as structure_io
import biotite.structure as structure

from astrolabe.structure_processors import register_structure_processor, StructureProcessor

# Constants for backbone and alanine atoms
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
ALANINE_ATOMS = BACKBONE_ATOMS | {"CB"}
ALANINE_RESIDUE = "ALA"


@register_structure_processor("all_alanine")
class AllAlanineStructureProcessor(StructureProcessor):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        included_chains = kwargs.get("included_chains")
        if isinstance(included_chains, str):
            included_chains = [chain.strip() for chain in included_chains.split(",")]
        self.included_chains = included_chains

    def _run_impl(self, input_file_path: str) -> str:
        """Read PDB or CIF file, mutate all residues to alanine, and write to output file."""
        input_structure = structure_io.load_structure(input_file_path)

        amino_acid_atoms, non_amino_acid_atoms = self._separate_amino_acids(
            input_structure
        )
        mutated_amino_acids = self._mutate_to_alanine(amino_acid_atoms)
        final_structure = structure.concatenate(
            [mutated_amino_acids, non_amino_acid_atoms]
        )

        output_file_path = self._save_mutated_structure(
            final_structure, input_file_path
        )
        return output_file_path

    def _separate_amino_acids(self, input_structure: Any) -> tuple[Any, Any]:
        """Separate amino acid atoms from non-amino acid atoms."""
        is_amino_acid = structure.filter_amino_acids(input_structure)
        if hasattr(input_structure, "hetero"):
            is_amino_acid = is_amino_acid & ~input_structure.hetero
        if self.included_chains is not None:
            is_amino_acid &= np.isin(input_structure.chain_id, self.included_chains)
        amino_acid_atoms = input_structure[is_amino_acid]
        non_amino_acid_atoms = input_structure[~is_amino_acid]
        return amino_acid_atoms, non_amino_acid_atoms

    def _mutate_to_alanine(self, amino_acid_atoms: Any) -> Any:
        """Mutate all amino acid residues to alanine, keeping only backbone and CB atoms."""
        atoms_to_keep = np.ones(len(amino_acid_atoms), dtype=bool)

        for atom_index, atom in enumerate(amino_acid_atoms):
            if atom.res_name != ALANINE_RESIDUE:
                if atom.atom_name in ALANINE_ATOMS:
                    amino_acid_atoms.res_name[atom_index] = ALANINE_RESIDUE
                else:
                    # Remove side chain atoms beyond CB
                    atoms_to_keep[atom_index] = False

        return amino_acid_atoms[atoms_to_keep]

    def _save_mutated_structure(
        self, mutated_structure: Any, original_file_path: str
    ) -> str:
        """Save the mutated structure to a PDB file."""
        input_stem = Path(original_file_path).stem
        output_file_path = self._get_output_path(f"allala_{input_stem}.pdb")
        structure_io.save_structure(output_file_path, mutated_structure)
        return output_file_path
