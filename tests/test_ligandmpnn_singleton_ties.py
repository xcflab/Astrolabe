from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


def test_prepare_mutation_input_skips_singleton_symmetry_groups(monkeypatch) -> None:
    global_config.reset()

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [], "B": []},
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
    )
    protomers = {
        "A": SimpleNamespace(tied_positions=[(0,), (1,)]),
        "B": SimpleNamespace(tied_positions=[(0,), (1,)]),
    }
    scores = SimpleNamespace(
        names=["AB"],
        complex_paths={"AB": "/tmp/dummy.cif"},
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert result["AB"]["symmetry_residues"] == ""
    assert result["AB"]["symmetry_weights"] == ""


def test_prepare_mutation_input_normalizes_c8_symmetry_weights(monkeypatch) -> None:
    global_config.reset()

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": []},
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
    )
    protomers = {"A": SimpleNamespace(tied_positions=[(0,)])}
    scores = SimpleNamespace(
        names=["AAAAAAAA"],
        complex_paths={"AAAAAAAA": "/tmp/dummy.cif"},
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert result["AAAAAAAA"]["symmetry_residues"] == "A1,B1,C1,D1,E1,F1,G1,H1"
    weights = [
        float(value)
        for value in result["AAAAAAAA"]["symmetry_weights"].split(",")
    ]
    assert weights == [0.125] * 8
    assert sum(weights) == pytest.approx(1.0)


def test_prepare_mutation_input_can_disable_selector_empty_fallback(monkeypatch) -> None:
    global_config.reset()

    def fixed_only_selector(_position_selector, _evaluation_result, protomers):
        return {
            proto: np.array([True, False, False], dtype=bool)
            for proto in protomers
        }

    monkeypatch.setattr(
        "astrolabe.mutators.utils.construct_position_selector",
        fixed_only_selector,
    )

    protomers = {
        "A": SimpleNamespace(
            type=ProtomerTypes.PROTEIN,
            length=3,
            mutable_mask=np.array([False, True, True], dtype=bool),
            tied_positions=[],
            aggregate_bool=lambda mask: mask,
        )
    }
    scores = SimpleNamespace(
        names=["A"],
        complex_paths={"A": "/tmp/dummy.cif"},
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector="dummy_selector()",
        fallback_to_all_mutable_if_selector_empty=False,
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert result["A"]["fixed_residues"] == "A1 A2 A3"


def _scores(tmp_path: Path, prefix: str, names: list[str]) -> StandardScores:
    return StandardScores(
        names=names,
        plddt={name: [0.9] for name in names},
        complex_paths={name: tmp_path / f"{prefix}_{name}.pdb" for name in names},
    )


def _protomers(names: str):
    return {
        name: SimpleNamespace(tied_positions=[], aggregate_bool=lambda mask: mask)
        for name in names
    }


def test_prepare_mutation_input_uses_current_backbone_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [], "B": [], "C": []},
    )
    scores = _scores(tmp_path, "current", ["A", "B", "C"])
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
    )

    result = mutator.prepare_mutation_input(_protomers("ABC"), scores, runtime_config={})

    assert result["A"]["pdb_path"] == str(tmp_path / "current_A.pdb")
    assert result["B"]["pdb_path"] == str(tmp_path / "current_B.pdb")
    assert result["C"]["pdb_path"] == str(tmp_path / "current_C.pdb")


def test_prepare_mutation_input_keeps_current_and_adds_reference_supplement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [0], "B": [], "C": [1]},
    )
    current_scores = _scores(tmp_path, "current", ["A", "B", "C"])
    reference_scores = _scores(tmp_path, "reference", ["A", "B", "C"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=reference_scores),
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
        supplemental_structures=["structure_start_reference"],
    )

    result = mutator.prepare_mutation_input(
        _protomers("ABC"),
        current_scores,
        runtime_config={},
    )

    assert result["A"]["pdb_path"] == str(tmp_path / "current_A.pdb")
    assert result["B"]["pdb_path"] == str(tmp_path / "current_B.pdb")
    assert result["C"]["pdb_path"] == str(tmp_path / "current_C.pdb")
    assert result["C"]["supplemental_inputs"] == [
        {
            "source": "structure_start_reference",
            "features": {
                "pdb_path": str(tmp_path / "reference_C.pdb"),
                "fixed_residues": "A2",
                "symmetry_residues": "",
                "symmetry_weights": "",
            },
        }
    ]
    assert result["A"]["fixed_residues"] == "A1"
    assert result["C"]["fixed_residues"] == "A2"


def test_prepare_mutation_input_reference_backbone_requires_complete_references(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [], "B": [], "C": []},
    )
    current_scores = _scores(tmp_path, "current", ["A", "B", "C"])
    reference_scores = _scores(tmp_path, "reference", ["A", "B"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=reference_scores),
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
        supplemental_structures=["structure_start_reference"],
    )

    with pytest.raises(ValueError, match="missing structure_start reference"):
        mutator.prepare_mutation_input(_protomers("ABC"), current_scores, runtime_config={})


@pytest.mark.parametrize("runtime_key", ["ignored_complexes", "negative_design_complexes"])
def test_prepare_mutation_input_skips_excluded_complexes(
    monkeypatch,
    tmp_path: Path,
    runtime_key: str,
) -> None:
    global_config.reset()
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [], "B": [], "C": []},
    )
    current_scores = _scores(tmp_path, "current", ["AC", "BC"])
    reference_scores = _scores(tmp_path, "reference", ["AC"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=reference_scores),
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector=None,
        supplemental_structures=["structure_start_reference"],
    )

    result = mutator.prepare_mutation_input(
        _protomers("ABC"),
        current_scores,
        runtime_config={runtime_key: ["BC"]},
    )

    assert sorted(result) == ["AC"]
    assert result["AC"]["pdb_path"] == str(tmp_path / "current_AC.pdb")
    assert result["AC"]["supplemental_inputs"][0]["features"]["pdb_path"] == str(
        tmp_path / "reference_AC.pdb"
    )


def test_ligandmpnn_rejects_unknown_supplemental_structure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": []},
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        supplemental_structures=["latest_reference"],
    )

    with pytest.raises(ValueError, match="supplemental_structures"):
        mutator.prepare_mutation_input(
            _protomers("A"),
            _scores(tmp_path, "current", ["A"]),
            runtime_config={},
        )


def test_compute_mutation_merges_current_and_supplemental_probabilities() -> None:
    class _Runner:
        def __init__(self):
            self.calls = []

        def design_sequences(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["pdb_path"] == "current.pdb":
                probs = np.asarray([[[0.8, 0.2]]], dtype=float)
            elif kwargs["pdb_path"] == "reference.pdb":
                probs = np.asarray([[[0.2, 0.8]]], dtype=float)
            else:
                raise AssertionError(kwargs)
            return {"sampling_probs": probs}

    runner = _Runner()
    mutator = LigandMPNNMutator(
        sequence_model_runner=runner,
        supplemental_structure_weight=1.0,
    )

    output = mutator.compute_mutation(
        {
            "A": {
                "pdb_path": "current.pdb",
                "fixed_residues": "",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "supplemental_inputs": [
                    {
                        "source": "structure_start_reference",
                        "features": {
                            "pdb_path": "reference.pdb",
                            "fixed_residues": "",
                            "symmetry_residues": "",
                            "symmetry_weights": "",
                        },
                    }
                ],
            }
        }
    )

    assert [call["pdb_path"] for call in runner.calls] == [
        "current.pdb",
        "reference.pdb",
    ]
    np.testing.assert_allclose(
        output["A"]["sampling_probs"],
        np.asarray([[[0.5, 0.5]]], dtype=float),
    )
