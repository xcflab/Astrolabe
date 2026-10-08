import gzip
import io
import logging

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser, PDBIO
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem

from astrolabe.utils import chemicals
from astrolabe.utils.chemicals import (
    rna_nucleotides,
    dna_nucleotides,
    nucleotides_ent,
)
from astrolabe.utils.constant import restype_3to1


# from rdkit import AllChem, rdmolops


def parse_mmcif(file_path, parse_hetatom=False, ignore_het_h=True):
    parser = MMCIFParser(QUIET=True)
    filehandle = (
        gzip.open(file_path, "rt")
        if file_path.endswith(".gz")
        else open(file_path, "rt")
    )
    structure = parser.get_structure("mmcif_structure", filehandle)

    res, pdb_idx = [], []
    res_idx = 0
    for model in structure:
        for chain in model:
            for resi in chain:
                if "CA" in resi:
                    res.append((resi.get_id()[1], resi.get_resname()))
                    pdb_idx.append((chain.get_id(), resi.get_id()[1]))
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for model in structure:
        for chain in model:
            for resi in chain:
                if resi.resname not in chemicals.aa2num:
                    continue
                for idx, atom in enumerate(
                    chemicals.aa2long[chemicals.aa2num[resi.resname]][:14]
                ):
                    if atom is not None and atom.strip() in resi:
                        # print(f"{idx} {resi[atom.strip()]}")
                        xyz[res_idx, idx, :] = resi[atom.strip()].get_coord()
                res_idx += 1
    # Convert residue name to numeric sequence using chemicals.aa2num
    seq = [chemicals.aa2num.get(r[1], 20) for r in res]

    # Create mask for existing atoms
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # Remove duplicated residues
    new_idx, i_unique = [], []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # Cartesian coordinates, [Lx14]
        "mask": mask,  # Atom mask showing which atoms are present, [Lx14]
        "idx": np.array([i[1] for i in pdb_idx]),  # Residue numbers, [L]
        "seq": np.array(seq),  # Amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # List of (chain letter, residue number), [L]
        "seq_str": "".join(
            [chemicals.num2aa[s] for s in seq]
        ),  # Amino acid sequence as string, [L]
        "type": "p",  # Type of the structure
    }

    # Parsing heteroatoms (ligands, etc.)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for model in structure:
            for chain in model:
                for resi in chain:
                    if resi.id[0].startswith("H"):  # Heteroatom residues
                        for atom in resi:
                            if not (ignore_het_h and atom.element == "H"):
                                info_het.append(
                                    {
                                        "idx": resi.get_id()[1],
                                        "atom_id": atom.get_name(),
                                        "atom_type": atom.element,
                                        "name": resi.get_resname(),
                                    }
                                )
                                xyz_het.append(atom.get_coord())

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out


def parse_pdb_zip(filename, **kwargs):
    """extract xyz coords for all heavy atoms"""
    with gzip.open(filename, "rt") as f:
        lines = f.readlines()
    return parse_pdb_lines(lines, **kwargs)


def parse_pdb(filename, **kwargs):
    """extract xyz coords for all heavy atoms"""
    with open(filename, "r") as f:
        lines = f.readlines()
    return parse_pdb_lines(lines, **kwargs)


def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res, pdb_idx = [], []
    for l in lines:
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            res.append((l[22:26], l[17:20]))
            # chain letter, res num
            pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
    seq = [
        chemicals.aa2num[r[1]] if r[1] in chemicals.aa2num.keys() else 20 for r in res
    ]
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num
    pdb_idx_lookup = {}
    for i, key in enumerate(pdb_idx):
        pdb_idx_lookup.setdefault(key, i)

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22].strip(),
            int(l[22:26]),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        idx = pdb_idx_lookup.get((chain, resNo))
        if idx is not None:
            # for i_atm, tgtatm in enumerate(chemicals.aa2long[chemicals.aa2num[aa]]):
            for (
                i_atm,
                tgtatm,
            ) in enumerate(  # aa: 三字母缩写 -> aa2num: 20种氨基酸的编号 -> aa2long: 20种氨基酸的原子组成
                chemicals.aa2long[chemicals.aa2num[aa]][:14]
            ):
                if (
                    tgtatm is not None and tgtatm.strip() == atom.strip()
                ):  # ignore whitespace
                    xyz[idx, i_atm, :] = [
                        float(l[30:38]),
                        float(l[38:46]),
                        float(l[46:54]),
                    ]
                    break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx14]
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
        "seq_str": "".join(
            [chemicals.num2aa[s] for s in seq]
        ),  # Amino acid sequence as string, [L]
        "type": "p",  # Type of the structure
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                info_het.append(
                    dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[16:20],
                    )
                )
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out


def read_fasta(file_path):
    from Bio import SeqIO

    fasta_sequences = SeqIO.parse(open(file_path), "fasta")
    rslt = {}
    for fasta in fasta_sequences:
        entity_str, *desc_parts = fasta.id.split("|")
        type = {
            "protein": "p",
            "smiles": "s",
            "ccdcodes": "c",
            "RNA": "r",
            "DNA": "d",
        }.get(entity_str)
        name = [d for d in desc_parts if d.startswith("name=")][0].split("=")[1]

        rslt[name] = {"seq_str": str(fasta.seq), "type": type}
    return rslt


def parse_structure(file_path):
    # Choose parser based on file extension
    if file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format. Please use mmCIF or PDB.")

    structure = parser.get_structure("structure", file_path)
    return structure


def extract_ligands(structure):
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if (
                    not is_aa(residue)
                    and residue.get_resname() not in nucleotides_ent
                    and residue.get_resname().strip() != "HOH"
                ):  # Skip standard amino acids
                    ligands.append(residue)
    return ligands


# def convert_to_smiles(residue):
#     # Get the atoms from the residue (ligand)
#     atoms = [atom for atom in residue.get_atoms()]
#     mol = Chem.RWMol()

#     atom_indices = {}

#     # Step 1: Add atoms to the molecule
#     for atom in atoms:
#         element = atom.element.capitalize()

#         try:
#             rd_atom = Chem.Atom(element)
#         except ValueError:
#             print(f"Element {element} not supported in RDKit")
#             continue

#         mol_idx = mol.AddAtom(rd_atom)
#         atom_indices[atom] = mol_idx

#     # Step 2: Add bonds based on Biopython or distance criteria
#     added_bonds = set()
#     # Prefer using Biopython bonds if available
#     if hasattr(residue, "get_bonds"):
#         for bond in residue.get_bonds():
#             atom1, atom2 = bond
#             if atom1 in atom_indices and atom2 in atom_indices:
#                 bond_type = Chem.BondType.SINGLE  # Default to single bond
#                 # Optionally, infer bond type from Biopython or other criteria
#                 if bond.is_double():
#                     bond_type = Chem.BondType.DOUBLE
#                 elif bond.is_triple():
#                     bond_type = Chem.BondType.TRIPLE

#                 mol.AddBond(atom_indices[atom1], atom_indices[atom2], bond_type)
#                 added_bonds.add((atom1, atom2))
#     else:
#         # Fallback: Use distance-based bonding
#         for i, atom1 in enumerate(atoms):
#             for j, atom2 in enumerate(atoms):
#                 if (
#                     atom1 - atom2 < 1.9  # Use distance cutoff for bond formation
#                     and atom1 in atom_indices
#                     and atom2 in atom_indices
#                     and atom1 != atom2
#                     and mol.GetBondBetweenAtoms(
#                         atom_indices[atom1], atom_indices[atom2]
#                     )
#                     is None
#                 ):
#                     print(f"Adding bond between {atom1} and {atom2}")
#                     mol.AddBond(
#                         atom_indices[atom1], atom_indices[atom2], Chem.BondType.SINGLE
#                     )
#                     added_bonds.add((atom1, atom2))

#     # Step 3: Generate SMILES if the molecule has atoms
#     if mol.GetNumAtoms() > 0:
#         rdmolops.SanitizeMol(mol)  # Ensure valid molecule representation
#         smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
#     else:
#         smiles = None
#     return smiles


def convert_to_smiles(residue):
    """
    Convert a Biopython residue to a SMILES string using RDKit's MolFromPDBBlock.
    Args:
        residue (Bio.PDB.Residue): The ligand residue to convert.
    Returns:
        str: The SMILES string for the ligand, or None if conversion fails.
    """

    pdb_io = PDBIO()
    pdb_io.set_structure(residue)

    pdb_buffer = io.StringIO()
    pdb_io.save(pdb_buffer, write_end=0)
    pdb_block = pdb_buffer.getvalue()
    pdb_buffer.close()

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)

    if mol is not None:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    else:
        logging.info(f"Failed to convert residue {residue} to SMILES.")
        return None


def extract_heteroatoms(structure):
    heteroatoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == "HETATM":
                    heteroatoms.append(residue)
    return heteroatoms


def extract_rna_dna(structure):
    nucleotides = []
    for model in structure:
        for chain in model:
            nucleotide_seq = []
            for residue in chain:
                resname = residue.get_resname().strip()

                # Check if the residue is a nucleotide (RNA or DNA)
                if resname in dna_nucleotides or resname in rna_nucleotides:
                    nucleotide_seq.append(residue)

            # Store the nucleotide sequence for the chain if any were found
            if nucleotide_seq:
                nucleotides.append(nucleotide_seq)

    return nucleotides


def extract_proteins(structure):
    proteins = []
    for model in structure:
        for chain in model:
            protein_seq = []
            for residue in chain:
                resname = residue.get_resname().strip()

                if is_aa(residue, standard=True):
                    protein_seq.append(residue)

            if protein_seq:
                proteins.append(protein_seq)

    return proteins


def parse_PDB_seq(file_path):
    structure = parse_structure(file_path)

    # Proteins
    proteins = extract_proteins(structure)

    # Ligands and their SMILES strings
    ligands = extract_ligands(structure)
    ligand_smiles = [convert_to_smiles(lig) for lig in ligands]

    # RNA and DNA
    nucleotides = extract_rna_dna(structure)

    # Heteroatoms
    heteroatoms = extract_heteroatoms(structure)

    # Organize into a dataframe for easy handling
    data = {
        "Proteins": [
            [
                [restype_3to1[res.get_resname().strip()] for res in chain]
                for chain in proteins
            ]
        ],
        "Ligand SMILES": [ligand_smiles],
        "RNA/DNA Count": [
            [[res.get_resname() for res in chain] for chain in nucleotides]
        ],
        "Heteroatom Count": [len(heteroatoms)],
    }

    df = pd.DataFrame(data)
    return df
