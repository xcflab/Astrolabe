# Nonstandard Residue Contracts

This document records the current nonstandard-residue contract across config,
folding inputs, evaluator wrappers, and Rosetta params generation. It is an
audit baseline for Phase 7, not a syntax migration.

## Parameter Organization

`nonstandard_residues.aliases` is the user-facing alias map. Each alias key is
uppercased and limited to three characters so it can be used as a compact PTM
token in protein sequence statements.

PTM-like aliases are selected by `parent_aa`. They must define a source
(`source.format` plus `source.value/path`, or a direct `smiles`, `sdf`, `mol`,
`mmcif`, or `pdb` field) and atom names containing the peptide backbone atoms
`N`, `CA`, `C`, and `O`. They generate a `user_ccd` block and resolve sequence
tokens such as `[S5]` to a CCD id plus parent amino acid.

Ligand-like aliases omit `parent_aa`. They also generate `user_ccd`, but are
not valid inside protein PTM sequence positions. They are intended for ligand
CCD/SMILES paths and Rosetta ligand params generation.

`cyclization` owns covalent topology. It can express head-to-tail cyclization,
disulfides, stapled side-chain bonds, lactone/thiolactone bonds, cross-chain
bonds, bond order, target distance, signal mode, and relative-position
encoding. It is intentionally separate from `nonstandard_residues`: one field
defines chemical components, the other defines bonds between atoms in the
folding input.

`ligand_resnames` is Rosetta-only naming control for params files. It does not
change CCD or SMILES chemistry parsing.

## Backend Support Matrix

| Backend | PTM/user CCD | CCD ligand | SMILES ligand | Covalent bond/postbond | Cyclic topology signal | Rosetta params |
|---|---|---|---|---|---|---|
| AF3 | Uses `ProteinChain.ptms` and `Input.user_ccd` through `chemical_components.Ccd(user_ccd=...)`. | Uses `Ligand.ccd_ids`. | Uses `Ligand.smiles`. | Uses `bonded_atom_pairs`. | Uses `cyclic_asym_ids` and `cyclic_relative_patches` kwargs. | Not applicable. |
| Protenix | Converts protein/RNA/DNA modifications to `CCD_<id>`. User CCD cache is a backend environment concern. | Converts each CCD id to `CCD_<id>` ligand entries. | Uses ligand SMILES. | Converts `bonded_atom_pairs` to `covalent_bonds`. | Injects `cyclic_asym_ids` and `cyclic_relative_patches` into feature dicts. | Not applicable. |
| Boltz | Converts protein PTMs to `modifications: [{position, ccd}]`; custom CCD availability depends on Boltz CCD cache. | Emits ligand `ccd`. | Emits ligand `smiles`. | Emits bond constraints. | Emits `protein.cyclic` and `inference_options.cyclic_relative_entries`. | Not applicable. |
| Chai | Current wrapper primarily preserves protein sequence and covalent topology; PTM/user CCD support is not a confirmed contract. | Ligand CCD support is not a confirmed wrapper contract in this audit. | Ligand SMILES support is not a confirmed wrapper contract in this audit. | Writes Chai covalent restraint rows. | Passes `cyclic_subchains` and `cyclic_relative_patches`. | Not applicable. |
| Rosetta/PyRosetta | Generates polymer params for PTM-like nonstandard aliases from CCD/user CCD entries. | Generates params for non-default CCD ligands. | Generates params from SMILES or structure-derived mol2. | Consumed through Rosetta params and PyRosetta selectors/losses rather than folding-input bond fields. | Not a folding backend; cyclic topology may still affect upstream folded structures. | Required when PyRosetta sees non-built-in residue names. Failures must mention missing dependency or params contract explicitly. |

## Representative Cases

The minimal fixtures are stored in
`tests/fixtures/nonstandard_residue_cases.yaml` and are exercised by
`tests/test_nonstandard_residue_contracts.py`.

- `cyclic_peptide`: head-to-tail peptide using `cyclization.head_to_tail`.
- `stapled_peptide`: same-chain noncanonical side-chain bond with sparse
  relative-position patches.
- `lactone`: side-chain oxygen to carboxyl carbon bond with explicit order and
  target distance.
- `thiolactone`: side-chain sulfur to carboxyl carbon bond with explicit order
  and target distance.
- `csd_direct_ccd_ligand`: direct CCD ligand input representing CSD-derived or
  curated CCD chemistry.

## Compatibility Notes

- Existing sequence statement syntax is unchanged. `[ALIAS]` or `(CCD)` PTM
  tokens still resolve through `parse_ptm_sequence()`.
- Existing peptide CLI options `--nonstd-map`, `--cyclize-head-to-tail`,
  `--cyclize-disulfide`, `--cyclize-bond`, `--cyclize-bond-typed`, and
  `--cyclic-encoding` remain the active compatibility surface.
- Rosetta support is explicit. If `molfile_to_params`, RDKit/OpenBabel,
  PyRosetta, or CCD-derived polymer tags are missing or inconsistent, the code
  should fail visibly with the dependency or params contract in the message.

