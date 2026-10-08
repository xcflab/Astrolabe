from pathlib import Path

import numpy as np
import pytest

from astrolabe.analysis.paper_geometry import cre_domain_distance


def _write(path, records, chain="B"):
    lines = []
    for serial, (number, name, xyz) in enumerate(records, 1):
        x, y, z = xyz
        lines.append(f"ATOM  {serial:5d} {name:>4s} ALA {chain}{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
    path.write_text("".join(lines) + "END\n")
    return path


def test_cre_distance_transforms_reference_all_atom_centroids(tmp_path):
    base = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    reference, prediction, expected = [], [], []
    for indices, offset, rotation, translation in [
        ([1, 2, 3, 128], np.zeros(3), rz, np.array([5, 0, 3])),
        ([139, 140, 141, 343], np.array([10, 0, 0]), np.eye(3), np.array([-1, 4, 0])),
    ]:
        all_ref = []
        for index, xyz in zip(indices, base + offset):
            for atom, delta in [("CA", np.zeros(3)), ("CB", np.array([0, 0, 2]))]:
                point = xyz + delta
                reference.append((index, atom, point))
                all_ref.append(point)
                # Prediction side chains differ; they must not define the centroid.
                predicted = point @ rotation + translation
                if atom == "CB":
                    predicted = predicted + [100, 100, 100]
                prediction.append((index, atom, predicted))
        expected.append(np.mean(all_ref, axis=0) @ rotation + translation)
    result = cre_domain_distance(_write(tmp_path / "reference.pdb", reference),
                                 _write(tmp_path / "prediction.pdb", prediction))
    assert result["distance_angstrom"] == pytest.approx(np.linalg.norm(expected[0] - expected[1]))
    assert [domain["ca_pairs"] for domain in result["domains"]] == [4, 4]
    assert [domain["reference_atoms"] for domain in result["domains"]] == [8, 8]
    for domain, centroid in zip(result["domains"], expected):
        assert domain["fit_rmsd"] == pytest.approx(0, abs=1e-12)
        assert domain["transformed_centroid"] == pytest.approx(centroid)


def test_cre_renumbered_prediction_requires_explicit_mapping_and_reads_mmcif(tmp_path):
    from Bio.PDB import MMCIFIO, PDBParser

    coords = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    records = [(i, "CA", x + offset) for indices, offset in [([1, 2, 128], 0), ([139, 140, 343], 10)]
               for i, x in zip(indices, coords)]
    reference = _write(tmp_path / "reference.pdb", records)
    prediction = _write(tmp_path / "prediction.pdb", [(i + 1000, name, x) for i, name, x in records], chain="A")
    with pytest.raises(ValueError, match="Missing predicted CA"):
        cre_domain_distance(reference, prediction, predicted_chain="A")
    io = MMCIFIO()
    io.set_structure(PDBParser(QUIET=True).get_structure("prediction", str(prediction)))
    io.save(str(tmp_path / "prediction.cif"))
    result = cre_domain_distance(reference, tmp_path / "prediction.cif", predicted_chain="A",
                                 residue_mapping={str(i): [i, i + 1000] for i, _, _ in records})
    assert result["mapping_mode"] == "canonical_to_author_explicit"
    assert result["distance_angstrom"] == pytest.approx(np.sqrt(300))


def test_cre_collinear_domain_has_no_defined_transform(tmp_path):
    records = [(i, "CA", [i, 0, 0]) for i in (1, 2, 128, 139, 140, 343)]
    reference = _write(tmp_path / "reference.pdb", records)
    with pytest.raises(ValueError, match="collinear CA"):
        cre_domain_distance(reference, reference)


def test_cre_missing_predicted_reference_ca_is_not_silently_skipped(tmp_path):
    records = [(i, "CA", xyz) for i, xyz in [(1, [0, 0, 0]), (2, [2, 0, 0]), (128, [0, 2, 0]), (139, [10, 0, 0]), (140, [12, 0, 0]), (343, [10, 2, 0])]]
    reference = _write(tmp_path / "reference.pdb", records)
    prediction = _write(tmp_path / "prediction.pdb", records[1:])
    with pytest.raises(ValueError, match="Missing predicted CA"):
        cre_domain_distance(reference, prediction)
