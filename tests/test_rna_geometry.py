"""Analytic coordinate fixtures test geometry, without simulated predictions."""

import numpy as np
import pytest

from astrolabe.analysis.rna_geometry import (
    RNAAtoms,
    _interface_metrics,
    _repeat_metrics,
    _choose_assignment,
    bivalent_geometry,
    cyclic_chain_rmsd,
    load_rna_atoms,
    oligomer_geometry,
)


def _cyclic_coordinates(n):
    local = np.array([[2.0, 0.0, 0.0], [2.3, 0.2, 0.6], [2.5, 0.5, 1.2]])
    coordinates = {}
    for index, chain in enumerate("ABCDE"[:n]):
        theta = 2 * np.pi * index / n
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        coordinates[chain] = local @ rotation.T
    return coordinates


@pytest.mark.parametrize("n,orders", [(4, 6), (5, 24)])
def test_cyclic_rmsd_finds_chain_cycle_and_detects_distortion(n, orders):
    coordinates = _cyclic_coordinates(n)
    coordinates["B"], coordinates["C"] = coordinates["C"], coordinates["B"]
    result = cyclic_chain_rmsd(coordinates)
    assert result["c4_cyclic_rmsd_angstrom"] < 1e-12
    assert result["cycle_orders_evaluated"] == orders
    assert result["best_chain_cycle"][0] == "A"
    coordinates["B"] = coordinates["B"] + np.array([1.0, 0, 0])
    assert cyclic_chain_rmsd(coordinates)["c4_cyclic_rmsd_angstrom"] > 0.1


def test_oligomer_heavy_atom_contact_graph_detects_an_isolated_chain():
    coordinates = _cyclic_coordinates(4)
    atoms = RNAAtoms(np.concatenate(list(coordinates.values())),
                     np.repeat(list(coordinates), 3), np.tile(np.arange(3), 4),
                     np.full(12, "C4'"))
    result = oligomer_geometry(atoms)
    assert result["chain_graph_connected"] is True
    assert result["chain_connected_components"] == [["A", "B", "C", "D"]]
    shifted = atoms.coordinates.copy()
    shifted[atoms.chain_ids == "D"] += 50
    result = oligomer_geometry(RNAAtoms(shifted, atoms.chain_ids, atoms.residue_indices, atoms.atom_names))
    assert result["chain_graph_connected"] is False
    assert result["chain_connected_components"] == [["A", "B", "C"], ["D"]]


def _bivalent_atoms():
    coordinates = np.column_stack((np.arange(40) * 10.0, np.zeros(40), np.zeros(40)))
    coordinates[20] = [2, 0, 0]
    coordinates[21] = [3, 0, 0]
    return RNAAtoms(np.vstack((coordinates, [[0.5, 0, 0], [1000, 0, 0]])),
                    np.array(["A"] * 40 + ["B", "C"]), np.array(list(range(40)) + [0, 0]),
                    np.array(["C4'"] * 40 + ["CA", "CA"]))


def test_bivalent_contact_counts_and_two_distinct_clash_categories():
    atoms = _bivalent_atoms()
    result = bivalent_geometry(atoms)
    assert result["repeat_contact_pairs"] == 2
    assert result["repeat_contact_nt1"] == 1
    assert result["repeat_contact_nt2"] == 2
    assert result["interchain_clashes"] == 1  # A20--B is exactly 1.5 A, excluded.
    assert result["rna_nonadjacent_intrachain_clashes"] == 0
    assert result["structure_review_pass"] is False
    atoms.coordinates[22] = [0.2, 0.1, 0]
    result = bivalent_geometry(atoms)
    assert result["interchain_clashes"] == 2
    assert result["rna_nonadjacent_intrachain_clashes"] == 1
    assert result["repeat_contact_pairs"] == 3


def test_bivalent_contacts_count_residue_pairs_once_and_exclude_local_positions():
    atoms = _bivalent_atoms()
    doubled = RNAAtoms(np.vstack((atoms.coordinates, [[0.1, 0, 0]])),
                       np.append(atoms.chain_ids, "A"), np.append(atoms.residue_indices, 0),
                       np.append(atoms.atom_names, "C3'"))
    assert bivalent_geometry(doubled)["repeat_contact_pairs"] == 2
    atoms.coordinates[20] = [191, 0, 0]  # Only the directly joined pair 19/20 contacts.
    atoms.coordinates[21] = [210, 0, 0]
    assert bivalent_geometry(atoms)["repeat_contact_pairs"] == 0
    atoms.coordinates[20] = atoms.coordinates[17] + [1, 0, 0]
    assert bivalent_geometry(atoms)["repeat_contact_pairs"] == 0


def test_archived_interface_and_repeat_confidence_formulas():
    keys = [("A", i) for i in range(6)] + [("B", i) for i in range(6)]
    token_indices = {key: [index] for index, key in enumerate(keys)}
    pae = np.zeros((12, 12))
    for i in range(6):
        pae[i, 6 + i], pae[6 + i, i] = 3, 9
    contacts = {(("A", i), ("B", i)): 3.0 for i in range(6)}
    metrics = _interface_metrics(contacts, token_indices, pae)
    assert metrics["contact_pae"] == 9
    assert metrics["interface_quality"] == pytest.approx(0.25 + 0.75 * 0.25)
    repeat = _repeat_metrics(contacts, token_indices, pae)
    assert repeat["repeat_cohesion"] == pytest.approx(0.25)
    assert _choose_assignment((0.9, 0.2), (0.5, 0.5)) == "CB"
    assert _choose_assignment((0.5, 0.6), (0.5, 0.5)) == "BC"


@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_structure_reader_excludes_hydrogens_and_uses_polymer_order(tmp_path, suffix):
    from Bio.PDB import MMCIFIO, PDBIO
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    structure = Structure("geometry")
    model = Model(0)
    chain = Chain("A")
    structure.add(model)
    model.add(chain)
    for index, resnum in enumerate([10, 12]):
        residue = Residue((" ", resnum, " "), "A", " ")
        chain.add(residue)
        residue.add(Atom("C4'", np.array([index * 10.0, 0, 0]), 80, 1, " ", " C4'", index * 2, element="C"))
        residue.add(Atom("H4'", np.array([index * 10.0 + 1, 0, 0]), 80, 1, " ", " H4'", index * 2 + 1, element="H"))
    output = tmp_path / ("geometry" + suffix)
    writer = PDBIO() if suffix == ".pdb" else MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(output))
    atoms = load_rna_atoms(output)
    assert atoms.coordinates.shape == (2, 3)
    assert atoms.residue_indices.tolist() == [0, 1]
    assert atoms.atom_names.tolist() == ["C4'", "C4'"]
