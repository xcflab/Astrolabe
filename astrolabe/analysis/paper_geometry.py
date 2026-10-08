"""Reference-domain centroid geometry from the Astrolabe Cre experiment.

Reference domain ranges 1–128 and 139–343 are canonical Cre residue positions.
By default these are author residue numbers in both structures; no ordinal or
sequence-based remapping is inferred. If either structure uses other numbering,
provide ``residue_mapping={cre_position: (reference_id, prediction_id), ...}``.
Each author ID is an integer or ``(integer, insertion_code)``. Include every
reference residue with CA in an explicit mapping. Every reference CA requires
a corresponding predicted CA, matching the original analysis. All atoms of the
reference residues participating in the fit define the unweighted centroid.
Only the CAs fit the transform; predicted side chains never define the centroid.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, is_aa


def _residue_id(value) -> tuple[int, str]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("An author residue identifier is (number, insertion_code)")
        return int(value[0]), str(value[1]).strip()
    return int(value), ""


def _chain(path: str | Path, chain: str) -> dict:
    path = Path(path).expanduser()
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Expected PDB or mmCIF: {path}")
    model = next(parser.get_structure("structure", str(path)).get_models())
    if chain not in model:
        raise ValueError(f"Chain {chain!r} absent in {path}")
    residues = {}
    for residue in model[chain]:
        if is_aa(residue, standard=False):
            key = int(residue.id[1]), residue.id[2].strip()
            if key in residues:
                raise ValueError(f"Ambiguous author residue {key} in {path}")
            residues[key] = residue
    return residues


def cre_domain_distance(
    reference: str | Path,
    prediction: str | Path,
    reference_chain: str = "B",
    predicted_chain: str = "B",
    *,
    residue_mapping: Mapping | None = None,
) -> dict:
    """Return transformed reference-domain distance and independent fit details.

    Distances and RMSDs are in Angstrom. At least three non-collinear CA pairs
    per domain are required. Explicit mappings use canonical Cre positions as
    keys and author residue IDs in both files as values; JSON lists are accepted.
    """
    ref, pred = _chain(reference, reference_chain), _chain(prediction, predicted_chain)
    if residue_mapping is None:
        if any(insertion for _, insertion in ref):
            raise ValueError("Reference insertion codes require explicit residue_mapping to canonical Cre positions")
        mapping = {number: ((number, ""), (number, "")) for number, _ in ref}
    else:
        mapping = {}
        for position, pair in residue_mapping.items():
            if len(pair) != 2:
                raise ValueError("residue_mapping values must pair reference and predicted author IDs")
            canonical = int(position)
            if not 1 <= canonical <= 343 or canonical in mapping:
                raise ValueError("residue_mapping keys must be unique canonical Cre positions 1–343")
            mapping[canonical] = _residue_id(pair[0]), _residue_id(pair[1])
        for side in (0, 1):
            if len({pair[side] for pair in mapping.values()}) != len(mapping):
                raise ValueError("residue_mapping must not reuse an author residue ID")
    domains = []
    for start, stop in ((1, 128), (139, 343)):
        matched_ref, matched_pred, all_atoms = [], [], []
        positions = []
        for position in sorted(p for p in mapping if start <= p <= stop):
            ref_id, pred_id = mapping[position]
            if ref_id not in ref:
                raise ValueError(f"Reference residue {ref_id} mapped from Cre {position} is absent")
            r = ref[ref_id]
            if "CA" not in r:
                continue
            if pred_id not in pred or "CA" not in pred[pred_id]:
                raise ValueError(f"Missing predicted CA for reference residue {ref_id} (Cre {position}); check residue_mapping")
            all_atoms.extend(atom.coord for atom in r.get_atoms())
            if r.resname != pred[pred_id].resname:
                raise ValueError(f"Residue mismatch at Cre {position}: {r.resname} vs {pred[pred_id].resname}; check residue_mapping")
            matched_ref.append(r["CA"].coord)
            matched_pred.append(pred[pred_id]["CA"].coord)
            positions.append(position)
        if len(matched_ref) < 3:
            raise ValueError(f"Cre domain {start}–{stop} has fewer than three corresponding CA atoms; check residue_mapping")
        x, y, atoms = np.asarray(matched_ref, dtype=float), np.asarray(matched_pred, dtype=float), np.asarray(all_atoms, dtype=float)
        if not all(np.isfinite(values).all() for values in (x, y, atoms)):
            raise ValueError(f"Non-finite coordinates in Cre domain {start}–{stop}")
        xc, yc = x.mean(axis=0), y.mean(axis=0)
        if np.linalg.matrix_rank(x - xc, tol=1e-8) < 2 or np.linalg.matrix_rank(y - yc, tol=1e-8) < 2:
            raise ValueError(f"Cre domain {start}–{stop} has collinear CA anchors")
        u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
        u[:, -1] *= np.sign(np.linalg.det(u @ vt))
        rotation = u @ vt
        translation = yc - xc @ rotation
        reference_centroid = atoms.mean(axis=0)
        centroid = reference_centroid @ rotation + translation
        domains.append({"start": start, "stop": stop, "ca_pairs": len(x),
                        "reference_atoms": len(atoms), "matched_cre_positions": positions,
                        "fit_rmsd": float(np.sqrt(np.mean(np.sum((x @ rotation + translation - y) ** 2, axis=1)))),
                        "reference_centroid": reference_centroid.tolist(),
                        "transformed_centroid": centroid.tolist(),
                        "rotation": rotation.tolist(), "translation": translation.tolist()})
    return {"distance_angstrom": float(np.linalg.norm(np.array(domains[0]["transformed_centroid"]) -
                                                       np.array(domains[1]["transformed_centroid"]))),
            "reference": str(reference), "prediction": str(prediction),
            "reference_chain": reference_chain, "predicted_chain": predicted_chain,
            "mapping_mode": "canonical_to_author_explicit" if residue_mapping is not None else "author_identity",
            "domains": domains}
