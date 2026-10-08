"""Physical RNA review metrics, independent of model execution.

Distances are in angstroms. Archived contacts use <=4.5; clashes use <1.5.
Hydrogens and deuteriums are excluded; residue indices follow polymer order,
so a numbering gap does not change the definition of sequence adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from pathlib import Path
import json

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.kdtrees import KDTree
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


@dataclass(frozen=True)
class RNAAtoms:
    coordinates: np.ndarray
    chain_ids: np.ndarray
    residue_indices: np.ndarray
    atom_names: np.ndarray


def load_rna_atoms(structure_path) -> RNAAtoms:
    """Read the first PDB/mmCIF model's polymer heavy atoms using Biopython."""
    path = Path(structure_path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("RNA geometry requires a PDB or mmCIF structure")
    model = next(parser.get_structure("rna", str(path)).get_models())
    coordinates, chains, residues, names = [], [], [], []
    for chain in model:
        polymers = [residue for residue in chain if residue.id[0] == " "]
        for residue_index, residue in enumerate(polymers):
            for atom in residue.get_atoms():
                if atom.element.upper() in {"H", "D"}:
                    continue
                coordinates.append(atom.coord)
                chains.append(str(chain.id))
                residues.append(residue_index)
                names.append(atom.name.replace("*", "'"))
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.size == 0 or not np.isfinite(xyz).all():
        raise ValueError("Structure must contain finite polymer heavy-atom coordinates")
    return RNAAtoms(xyz, np.asarray(chains), np.asarray(residues), np.asarray(names))


def _close_pairs(atoms: RNAAtoms, cutoff: float = 4.5, *, inclusive: bool = False):
    pairs = KDTree(np.asarray(atoms.coordinates, dtype=np.float64), 10).neighbor_search(cutoff)
    pairs = [pair for pair in pairs if pair.radius <= cutoff] if inclusive else [pair for pair in pairs if pair.radius < cutoff]
    left = np.asarray([pair.index1 for pair in pairs], dtype=int)
    right = np.asarray([pair.index2 for pair in pairs], dtype=int)
    distance = np.asarray([pair.radius for pair in pairs], dtype=float)
    return left, right, distance


def bivalent_geometry(atoms: RNAAtoms) -> dict:
    """Review A(40nt)+B/C(protein), including both kinds of forbidden overlap.

    Archived repeat contacts are distinct residue pairs with heavy-atom
    distance <=4.5 A and sequence separation >4. PAE is required to compute
    the actual repeat-cohesion objective; this geometry-only view reports
    contact counts and coverage without substituting a different score.
    """
    if set(atoms.chain_ids) != {"A", "B", "C"}:
        raise ValueError("Bivalent review requires RNA chain A and protein chains B/C")
    rna = atoms.chain_ids == "A"
    if set(atoms.residue_indices[rna]) != set(range(40)):
        raise ValueError("Bivalent RNA chain A must contain exactly 40 polymer residues")
    left, right, distance = _close_pairs(atoms, inclusive=True)
    interchain = atoms.chain_ids[left] != atoms.chain_ids[right]
    rna_pair = rna[left] & rna[right]
    nonadjacent = np.abs(atoms.residue_indices[left] - atoms.residue_indices[right]) > 1
    different_repeat = (atoms.residue_indices[left] < 20) != (atoms.residue_indices[right] < 20)
    cross_repeat = rna_pair & different_repeat & (np.abs(atoms.residue_indices[left] - atoms.residue_indices[right]) > 4)
    residue_pairs = {
        tuple(sorted((int(atoms.residue_indices[i]), int(atoms.residue_indices[j]))))
        for i, j in zip(left[cross_repeat], right[cross_repeat])
    }
    interchain_clashes = int(np.count_nonzero(interchain & (distance < 1.5)))
    intra_clashes = int(np.count_nonzero(rna_pair & nonadjacent & (distance < 1.5)))
    return {
        "repeat_contact_pairs": len(residue_pairs),
        "repeat_contact_nt1": len({pair[0] for pair in residue_pairs}),
        "repeat_contact_nt2": len({pair[1] for pair in residue_pairs}),
        "interchain_clashes": interchain_clashes,
        "rna_nonadjacent_intrachain_clashes": intra_clashes,
        "structure_review_pass": interchain_clashes == 0 and intra_clashes == 0,
        "contact_cutoff_angstrom": 4.5,
        "clash_cutoff_angstrom": 1.5,
        "repeat_contact_definition": "distinct residue pairs <=4.5 A; RNA sequence separation >4",
    }


def analyze_bivalent_rna_geometry(structure_path) -> dict:
    return bivalent_geometry(load_rna_atoms(structure_path))


def cyclic_chain_rmsd(c4_coordinates: dict[str, np.ndarray]) -> dict:
    """Best proper-rotation Kabsch RMSD under a single cyclic chain permutation.

    All chains must have matching C4' residue positions. Fix the first chain
    and enumerate the remaining (n-1)! cycle orders; each fit maps the entire
    assembly to its one-step chain permutation with one shared rigid motion.
    No reflection, independent chain alignment or sequence remapping is used.
    """
    chains = list(c4_coordinates)
    if len(chains) not in (4, 5):
        raise ValueError("Cyclic RNA review requires four or five chains")
    arrays = {chain: np.asarray(c4_coordinates[chain], dtype=float) for chain in chains}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or arrays[chains[0]].ndim != 2 or arrays[chains[0]].shape[1] != 3 or not len(arrays[chains[0]]):
        raise ValueError("C4' coordinates must be nonempty, matching Nx3 arrays")
    source = np.concatenate([arrays[chain] for chain in chains])
    if not np.isfinite(source).all():
        raise ValueError("C4' coordinates must be finite")
    source = source - source.mean(axis=0)
    best_rmsd, best_cycle = float("inf"), None
    for suffix in permutations(chains[1:]):
        cycle = (chains[0], *suffix)
        successor = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
        target = np.concatenate([arrays[successor[chain]] for chain in chains])
        target = target - target.mean(axis=0)
        u, _, vt = np.linalg.svd(source.T @ target)
        correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        rotation = u @ correction @ vt
        rmsd = float(np.sqrt(np.mean(np.sum((source @ rotation - target) ** 2, axis=1))))
        if rmsd < best_rmsd:
            best_rmsd, best_cycle = rmsd, cycle
    return {"c4_cyclic_rmsd_angstrom": best_rmsd, "best_chain_cycle": list(best_cycle),
            "c4_atoms_per_chain": len(arrays[chains[0]]),
            "cycle_orders_evaluated": factorial(len(chains) - 1)}


def oligomer_geometry(atoms: RNAAtoms) -> dict:
    chains = list(dict.fromkeys(atoms.chain_ids.tolist()))
    c4 = {}
    positions = None
    for chain in chains:
        selected = (atoms.chain_ids == chain) & (atoms.atom_names == "C4'")
        chain_positions = atoms.residue_indices[selected]
        if positions is not None and not np.array_equal(positions, chain_positions):
            raise ValueError("Every RNA chain must have C4' atoms at matching residue positions")
        positions = chain_positions
        if len(chain_positions) != len(set(atoms.residue_indices[atoms.chain_ids == chain])):
            raise ValueError("Every RNA residue must have one C4' atom")
        c4[chain] = atoms.coordinates[selected]
    result = cyclic_chain_rmsd(c4)
    left, right, _ = _close_pairs(atoms, inclusive=True)
    edges = {}
    adjacency = {chain: set() for chain in chains}
    for i, j in zip(left, right):
        a, b = str(atoms.chain_ids[i]), str(atoms.chain_ids[j])
        if a == b:
            continue
        a, b = sorted((a, b))
        edges[f"{a}:{b}"] = edges.get(f"{a}:{b}", 0) + 1
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(chains)
    components = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            chain = pending.pop()
            if chain in component:
                continue
            component.add(chain)
            pending.extend(adjacency[chain] - component)
        remaining -= component
        components.append(sorted(component))
    result.update(chain_contact_atom_pairs=edges, chain_connected_components=components,
                  chain_graph_connected=len(components) == 1, contact_cutoff_angstrom=4.5)
    return result


def analyze_rna_oligomer(structure_path) -> dict:
    return oligomer_geometry(load_rna_atoms(structure_path))


# Scientific formulas ported from the archived bivalent_af3_metrics.py.
def _pair_pae(rna_key, target_key, token_indices, pae):
    rna_indices = token_indices.get(rna_key)
    target_indices = token_indices.get(target_key)
    if not rna_indices or not target_indices:
        raise ValueError(f"contact residue missing from AF3 token mapping: {rna_key}, {target_key}")
    forward = float(np.max(pae[np.ix_(rna_indices, target_indices)]))
    reverse = float(np.max(pae[np.ix_(target_indices, rna_indices)]))
    return max(forward, reverse)

def _interface_metrics(contact_distances, token_indices, pae):
    rna_residues = {rna for rna, _ in contact_distances}
    target_residues = {target for _, target in contact_distances}
    pair_pae = [
        _pair_pae(rna, target, token_indices, pae)
        for rna, target in contact_distances
    ]
    if pair_pae:
        confidence = float(np.mean(np.clip(1.0 - np.asarray(pair_pae) / 12.0, 0.0, 1.0)))
        quality = min(len(rna_residues) / 6.0, 1.0) * (0.25 + 0.75 * confidence)
        mean_pae = float(np.mean(pair_pae))
    else:
        quality = 0.0
        mean_pae = None
    return {
        "contact_nt": len(rna_residues),
        "contact_aa": len(target_residues),
        "contact_pairs": len(contact_distances),
        "min_distance": min(contact_distances.values()) if contact_distances else None,
        "contact_pae": mean_pae,
        "interface_quality": float(quality),
    }

def _repeat_metrics(contact_distances, token_indices, pae):
    pair_pae = [
        _pair_pae(repeat1, repeat2, token_indices, pae)
        for repeat1, repeat2 in contact_distances
    ]
    repeat1_nt = len({repeat1 for repeat1, _ in contact_distances})
    repeat2_nt = len({repeat2 for _, repeat2 in contact_distances})
    if pair_pae:
        clipped_confidence = np.clip(1.0 - np.asarray(pair_pae) / 12.0, 0.0, 1.0)
        cohesion = min(min(repeat1_nt, repeat2_nt) / 4.0, 1.0) * float(
            np.mean(clipped_confidence)
        )
        mean_pae = float(np.mean(pair_pae))
        min_pae = min(pair_pae)
        max_pae = max(pair_pae)
    else:
        cohesion = 0.0
        mean_pae = min_pae = max_pae = None
    return {
        "repeat_contact_pairs": len(contact_distances),
        "repeat_contact_nt1": repeat1_nt,
        "repeat_contact_nt2": repeat2_nt,
        "repeat_contact_pae": mean_pae,
        "repeat_contact_pae_min": min_pae,
        "repeat_contact_pae_max": max_pae,
        "repeat_cohesion": float(cohesion),
    }

def _choose_assignment(bc_qualities, cb_qualities):
    bc_rank = (min(bc_qualities), sum(bc_qualities))
    cb_rank = (min(cb_qualities), sum(cb_qualities))
    return "BC" if bc_rank >= cb_rank else "CB"


def bivalent_contact_metrics(atoms: RNAAtoms, token_indices: dict, pae, residue_plddt: dict) -> dict:
    """Archived two-arm contact/PAE objective, including B/C assignment symmetry.

    Residue keys are (chain, zero-based polymer position). For each contact,
    confidence uses the larger of the two directional PAE values (and the
    maximum over tokens if a residue has several). Interface quality is
    min(contact_nt/6,1)*(0.25+0.75*mean(clip(1-PAE/12,0,1))). Repeat cohesion
    is min(min(contact_nt1,contact_nt2)/4,1)*mean(clip(1-PAE/12,0,1)).
    """
    metrics = bivalent_geometry(atoms)
    pae = np.asarray(pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or not np.isfinite(pae).all() or np.any(pae < 0):
        raise ValueError("AF3 PAE must be a finite, nonnegative square matrix")
    contacts = {(arm, chain): {} for arm in ("arm1", "arm2") for chain in "BC"}
    repeats = {}
    left, right, distance = _close_pairs(atoms, inclusive=True)
    for i, j, dist in zip(left, right, distance):
        a = (str(atoms.chain_ids[i]), int(atoms.residue_indices[i]))
        b = (str(atoms.chain_ids[j]), int(atoms.residue_indices[j]))
        if b[0] == "A" and a[0] != "A":
            a, b = b, a
        if a[0] == "A" and b[0] in "BC":
            table = contacts[("arm1" if a[1] < 20 else "arm2", b[0])]
        elif a[0] == b[0] == "A" and (a[1] < 20) != (b[1] < 20) and abs(a[1] - b[1]) > 4:
            if a[1] > b[1]:
                a, b = b, a
            table = repeats
        else:
            continue
        table[(a, b)] = min(float(dist), table.get((a, b), float("inf")))

    interfaces = {}
    for (arm, chain), pairs in contacts.items():
        summary = _interface_metrics(pairs, token_indices, pae)
        interfaces[(arm, chain)] = summary
        metrics.update({f"{arm}_{chain}_{key}": value for key, value in summary.items()})
    metrics.update(_repeat_metrics(repeats, token_indices, pae))
    assignment = _choose_assignment(
        (interfaces[("arm1", "B")]["interface_quality"], interfaces[("arm2", "C")]["interface_quality"]),
        (interfaces[("arm1", "C")]["interface_quality"], interfaces[("arm2", "B")]["interface_quality"]),
    )
    selected = [interfaces[("arm1", assignment[0])], interfaces[("arm2", assignment[1])]]
    arm_plddt = [float(np.mean([residue_plddt[("A", index)] for index in range(start, start + 20)]))
                 for start in (0, 20)]
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in arm_plddt):
        raise ValueError("Selected-model residue pLDDT must be in [0,1]")
    target_sites = [{key[1][1] for key in contacts[(arm, chain)]}
                    for arm, chain in zip(("arm1", "arm2"), assignment)]
    site_union = target_sites[0] | target_sites[1]
    selected_pae = [item["contact_pae"] for item in selected]
    metrics.update(
        chosen_assignment=assignment,
        dual_interface=min(item["interface_quality"] for item in selected),
        arm1_plddt=arm_plddt[0], arm2_plddt=arm_plddt[1], weak_arm_plddt=min(arm_plddt),
        weak_contact_nt=min(item["contact_nt"] for item in selected),
        weak_contact_aa=min(item["contact_aa"] for item in selected),
        worst_contact_pae=None if any(value is None for value in selected_pae) else max(selected_pae),
        contact_target_ordinal_jaccard=len(target_sites[0] & target_sites[1]) / len(site_union) if site_union else None,
        intra_rna_clashes=metrics["rna_nonadjacent_intrachain_clashes"],
    )
    metrics["total_clashes"] = metrics["interchain_clashes"] + metrics["intra_rna_clashes"]
    metrics["objective"] = (2 * metrics["dual_interface"] + metrics["repeat_cohesion"]
                            + 0.5 * metrics["weak_arm_plddt"] - 0.5 * min(metrics["interchain_clashes"], 1))
    return metrics


def measure_bivalent_model(model_path, confidence_path) -> dict:
    """Measure the exact AF3 selected CIF and its paired confidences JSON.

    mmCIF auth/label chain and residue aliases map AF3 tokens to coordinates;
    pLDDT comes from selected-model heavy-atom B factors, not the five-sample
    mean returned by the evaluator. Uses the archived scientific formulas.
    """
    model_path, confidence_path = Path(model_path), Path(confidence_path)
    if (not model_path.name.endswith("_model.cif") or not confidence_path.name.endswith("_confidences.json")
            or model_path.name[:-len("_model.cif")] != confidence_path.name[:-len("_confidences.json")]):
        raise ValueError("Selected AF3 model and confidences must share the same output stem")
    data = MMCIF2Dict(str(model_path))
    if len(set(data["_atom_site.pdbx_PDB_model_num"])) != 1:
        raise ValueError("Expected exactly one selected AF3 model")
    aliases, ordinals, coordinates, atom_names, atom_chains, atom_residues = {}, {}, [], [], [], []
    residue_names = {}
    b_factors = {}
    for i, element in enumerate(data["_atom_site.type_symbol"]):
        auth_chain = data["_atom_site.auth_asym_id"][i]
        label_chain = data["_atom_site.label_asym_id"][i]
        auth_residue = int(data["_atom_site.auth_seq_id"][i])
        label_residue = int(data["_atom_site.label_seq_id"][i])
        residue_identity = (auth_residue, data["_atom_site.pdbx_PDB_ins_code"][i], label_chain, label_residue)
        chain_ordinals = ordinals.setdefault(auth_chain, {})
        ordinal = chain_ordinals.setdefault(residue_identity, len(chain_ordinals))
        key = (auth_chain, ordinal)
        residue_names[key] = data["_atom_site.label_comp_id"][i].upper()
        for chain_alias in {auth_chain, label_chain}:
            for residue_alias in {auth_residue, label_residue}:
                aliases.setdefault((chain_alias, residue_alias), set()).add(key)
        if element.upper() in {"H", "D"}:
            continue
        coordinates.append([float(data[f"_atom_site.Cartn_{axis}"][i]) for axis in "xyz"])
        atom_names.append(data["_atom_site.label_atom_id"][i])
        atom_chains.append(auth_chain)
        atom_residues.append(ordinal)
        b_factors.setdefault(key, []).append(float(data["_atom_site.B_iso_or_equiv"][i]) / 100)
    if set(ordinals) != set("ABC") or [len(ordinals[c]) for c in "ABC"] != [40, 96, 96]:
        raise ValueError("Selected bivalent model must have 40nt RNA A and 96-residue proteins B/C")
    rna_names = [residue_names[("A", i)] for i in range(40)]
    rna_map = {"A": "A", "C": "C", "G": "G", "U": "U", "RA": "A", "RC": "C", "RG": "G", "RU": "U"}
    rna_sequence = "".join(rna_map[name] for name in rna_names)
    if rna_sequence[:20] != rna_sequence[20:] or [residue_names[("B", i)] for i in range(96)] != [residue_names[("C", i)] for i in range(96)]:
        raise ValueError("Selected model must preserve synchronized RNA repeats and identical target copies")
    atoms = RNAAtoms(np.asarray(coordinates), np.asarray(atom_chains), np.asarray(atom_residues), np.asarray(atom_names))
    confidence = json.loads(confidence_path.read_text())
    token_chains, token_residues = confidence["token_chain_ids"], confidence["token_res_ids"]
    if len(token_chains) != len(token_residues):
        raise ValueError("AF3 token chain and residue arrays must have equal length")
    token_indices = {}
    for index, (chain, residue) in enumerate(zip(token_chains, token_residues)):
        keys = aliases.get((str(chain), int(residue)), set())
        if len(keys) != 1:
            raise ValueError(f"AF3 token does not map uniquely to the selected structure: {chain}/{residue}")
        token_indices.setdefault(next(iter(keys)), []).append(index)
    pae = np.asarray(confidence["pae"], dtype=float)
    if pae.shape != (len(token_chains), len(token_chains)):
        raise ValueError("AF3 PAE size differs from its token map")
    residue_plddt = {key: float(np.mean(values)) for key, values in b_factors.items()}
    result = bivalent_contact_metrics(atoms, token_indices, pae, residue_plddt)
    result["rna_sequence"] = rna_sequence
    result["n_target_residues"] = 96
    result["confidence_path"] = str(confidence_path)
    result["confidence_aggregation"] = "AF3 ranking-selected sample: paired CIF and confidences JSON"
    return result
