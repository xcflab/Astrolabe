import numpy as np
import pytest

from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


class _FakeRunner:
    def __init__(self, probs_by_path):
        self.probs_by_path = probs_by_path
        self.calls = []

    def design_sequences(self, **input_feat):
        self.calls.append(dict(input_feat))
        pdb_path = input_feat["pdb_path"]
        return {"sampling_probs": np.asarray(self.probs_by_path[pdb_path], dtype=float)}


def _probs(a_prob):
    return np.asarray([[[a_prob, 1.0 - a_prob]]], dtype=float)


def test_ligandmpnn_blends_primary_with_each_supplemental_probability():
    runner = _FakeRunner(
        {
            "primary.pdb": _probs(0.8),
            "supp_a.pdb": _probs(0.2),
            "supp_b.pdb": _probs(0.4),
        }
    )
    mutator = LigandMPNNMutator(
        runner,
        supplemental_structure_weight=1.0,
    )

    output = mutator.compute_mutation(
        {
            "A": {
                "pdb_path": "primary.pdb",
                "supplemental_inputs": [
                    {"source": "ref_a", "features": {"pdb_path": "supp_a.pdb"}},
                    {"source": "ref_b", "features": {"pdb_path": "supp_b.pdb"}},
                ],
            }
        }
    )

    assert [call["pdb_path"] for call in runner.calls] == [
        "primary.pdb",
        "supp_a.pdb",
        "supp_b.pdb",
    ]
    assert np.allclose(output["A"]["sampling_probs"], _probs((0.8 + 0.2 + 0.4) / 3.0))
    assert output["A"]["supplemental_structure_outputs"] == [
        {"source": "ref_a", "feature_dict": {"pdb_path": "supp_a.pdb"}},
        {"source": "ref_b", "feature_dict": {"pdb_path": "supp_b.pdb"}},
    ]


def test_ligandmpnn_applies_supplemental_structure_weight():
    runner = _FakeRunner(
        {
            "primary.pdb": _probs(0.8),
            "supp_a.pdb": _probs(0.2),
            "supp_b.pdb": _probs(0.4),
        }
    )
    mutator = LigandMPNNMutator(
        runner,
        supplemental_structure_weight=0.5,
    )

    output = mutator.compute_mutation(
        {
            "A": {
                "pdb_path": "primary.pdb",
                "supplemental_inputs": [
                    {"source": "ref_a", "features": {"pdb_path": "supp_a.pdb"}},
                    {"source": "ref_b", "features": {"pdb_path": "supp_b.pdb"}},
                ],
            }
        }
    )

    expected = (0.8 + 0.5 * 0.2 + 0.5 * 0.4) / 2.0
    assert np.allclose(output["A"]["sampling_probs"], _probs(expected))


def test_ligandmpnn_rejects_negative_supplemental_structure_weight():
    with pytest.raises(ValueError, match="supplemental_structure_weight"):
        LigandMPNNMutator(
            _FakeRunner({}),
            supplemental_structure_weight=-0.1,
        )


def test_ligandmpnn_rejects_supplemental_probability_shape_mismatch():
    runner = _FakeRunner(
        {
            "primary.pdb": _probs(0.8),
            "supp_a.pdb": np.asarray([[[0.2, 0.8], [0.4, 0.6]]], dtype=float),
        }
    )
    mutator = LigandMPNNMutator(runner)

    with pytest.raises(ValueError, match="shape mismatch"):
        mutator.compute_mutation(
            {
                "A": {
                    "pdb_path": "primary.pdb",
                    "supplemental_inputs": [
                        {"source": "ref_a", "features": {"pdb_path": "supp_a.pdb"}},
                    ],
                }
            }
        )
