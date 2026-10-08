from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
AF2_VENDOR = REPO_ROOT / "vendor" / "alphafold2"
if str(AF2_VENDOR) not in sys.path:
    sys.path.insert(0, str(AF2_VENDOR))

from alphafold.common import confidence, protein, residue_constants
from alphafold.model import model

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.alphafold2 import AlphaFold2Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.confidence import binder_iptm, binder_ptm, chain_pair_iptm


def test_af2_chain_pair_iptm_uses_each_pair_from_one_multimer_prediction() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(7, 7, 4))
    breaks = np.array([1.0, 2.0, 3.0], dtype=float)
    asym_id = np.array([1, 1, 2, 3, 3, 4, 4], dtype=np.int32)

    matrix = model._compute_chain_pair_iptm(logits, breaks, asym_id)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)

    for left in range(4):
        for right in range(left + 1, 4):
            selected = np.isin(asym_id, [left + 1, right + 1])
            expected = confidence.predicted_tm_score(
                logits=logits[selected][:, selected, :],
                breaks=breaks,
                asym_id=asym_id[selected],
                interface=True,
            )
            assert matrix[left, right] == np.float32(expected)

    scores = StandardScores(
        names=["ABCD"],
        plddt={"ABCD": [np.ones(7)]},
        complex_paths={"ABCD": Path("ABCD.pdb")},
        chain_pair_scores={"ABCD": {"chain_pair_iptm": [matrix]}},
    )
    expected_binder_iptm = np.mean(matrix[0, 1:])
    assert binder_iptm(scores, binder_chain="A")["ABCD"] == expected_binder_iptm


def test_af2_chain_ptm_uses_each_chain_from_one_multimer_prediction() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(7, 7, 4))
    breaks = np.array([1.0, 2.0, 3.0], dtype=float)
    asym_id = np.array([1, 1, 2, 3, 3, 4, 4], dtype=np.int32)

    chain_scores = model._compute_chain_ptm(logits, breaks, asym_id)

    assert chain_scores.shape == (4,)
    for chain_idx, chain_id in enumerate([1, 2, 3, 4]):
        selected = asym_id == chain_id
        expected = confidence.predicted_tm_score(
            logits=logits[selected][:, selected, :],
            breaks=breaks,
        )
        assert chain_scores[chain_idx] == np.float32(expected)


def test_af2_confidence_keeps_global_iptm_and_adds_chain_pair_matrix() -> None:
    logits = np.zeros((4, 4, 4), dtype=float)
    breaks = np.array([1.0, 2.0, 3.0], dtype=float)
    asym_id = np.array([1, 2, 3, 4], dtype=np.int32)
    prediction = {
        "predicted_lddt": {"logits": np.zeros((4, 50), dtype=float)},
        "predicted_aligned_error": {
            "logits": logits,
            "breaks": breaks,
            "asym_id": asym_id,
        },
    }

    metrics = model.get_confidence_metrics(prediction, multimer_mode=True)

    expected_global_iptm = confidence.predicted_tm_score(
        logits=logits,
        breaks=breaks,
        asym_id=asym_id,
        interface=True,
    )
    assert metrics["iptm"] == expected_global_iptm
    assert metrics["chain_pair_iptm"].shape == (4, 4)
    assert metrics["chain_ptm"].shape == (4,)


def test_af2_confidence_does_not_invent_chain_pair_scores_without_pae() -> None:
    prediction = {
        "predicted_lddt": {"logits": np.zeros((2, 50), dtype=float)},
    }

    metrics = model.get_confidence_metrics(prediction, multimer_mode=True)

    assert "chain_pair_iptm" not in metrics
    assert "chain_ptm" not in metrics
    assert "iptm" not in metrics


def test_af2_evaluator_exposes_chain_pair_scores_without_changing_global_iptm(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers",
        {
            name: SimpleNamespace(type=ProtomerTypes.PROTEIN)
            for name in "ABCD"
        },
    )
    global_config.set_variable("step", 0)
    atom_count = residue_constants.atom_type_num
    predicted_protein = protein.Protein(
        atom_positions=np.zeros((4, atom_count, 3), dtype=np.float32),
        atom_mask=np.zeros((4, atom_count), dtype=np.float32),
        aatype=np.zeros(4, dtype=np.int32),
        residue_index=np.arange(1, 5, dtype=np.int32),
        chain_index=np.arange(4, dtype=np.int32),
        b_factors=np.zeros((4, atom_count), dtype=np.float32),
    )
    matrix = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.2, 0.0, 0.3, 0.5],
            [0.4, 0.3, 0.0, 0.7],
            [0.6, 0.5, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    chain_ptm = np.array([0.61, 0.72, 0.83, 0.94], dtype=np.float32)
    prediction = {
        "plddt": np.full(4, 80.0, dtype=np.float32),
        "ptm": 0.71,
        "iptm": 0.83,
        "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
        "chain_pair_iptm": matrix,
        "chain_ptm": chain_ptm,
        "ranking_confidence": 0.8,
    }
    inputs = {
        "ABCD": SimpleNamespace(
            name="ABCD",
            chains=[SimpleNamespace(sequence="A") for _ in range(4)],
        )
    }
    outputs = {
        "ABCD": [
            {"seed": 1, "prediction": prediction, "protein": predicted_protein}
        ]
    }

    scores = AlphaFold2Evaluator(model_runners={}).prepare_output(
        outputs, inputs, runtime_config={}
    )

    assert scores.iptm["ABCD"] == [0.83]
    assert np.allclose(
        scores.chain_pair_scores["ABCD"]["chain_pair_iptm"][0], matrix
    )
    assert np.allclose(scores.chain_pair_scores["ABCD"]["chain_ptm"][0], chain_ptm)
    assert binder_ptm(scores, binder_chain="A")["ABCD"] == pytest.approx(0.61)
    assert binder_iptm(scores, binder_chain="A")["ABCD"] == pytest.approx(
        np.mean(matrix[0, 1:])
    )
    assert chain_pair_iptm(scores, chain_a="A", chain_b="B")["ABCD"] == pytest.approx(0.2)
    assert chain_pair_iptm(scores, chain_a="A", chain_b="C")["ABCD"] == pytest.approx(0.4)
    assert chain_pair_iptm(scores, chain_a="A", chain_b="D") == {
        "ABCD": pytest.approx(0.6)
    }
    global_config.reset()


def test_af2_evaluator_does_not_invent_chain_pair_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction = {
        "plddt": np.full(2, 80.0, dtype=np.float32),
        "iptm": 0.72,
        "ranking_confidence": 0.72,
    }
    predicted_protein = SimpleNamespace(
        aatype=np.zeros(2),
        b_factors=np.zeros((2, residue_constants.atom_type_num)),
    )
    evaluator = AlphaFold2Evaluator(model_runners={})
    evaluator._apply_plddt_to_b_factors = lambda prot, _prediction: prot
    inputs = {
        "AB": SimpleNamespace(
            name="AB", chains=[SimpleNamespace(sequence="A") for _ in range(2)]
        )
    }

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers",
        {name: SimpleNamespace(type=ProtomerTypes.PROTEIN) for name in "AB"},
    )
    global_config.set_variable("step", 0)
    monkeypatch.setattr(protein, "to_pdb", lambda _prot: "END\n")
    scores = evaluator.prepare_output(
        {
            "AB": [
                {
                    "seed": 1,
                    "prediction": prediction,
                    "protein": predicted_protein,
                }
            ]
        },
        inputs,
        runtime_config={},
    )

    assert scores.iptm["AB"] == [0.72]
    assert scores.chain_pair_scores is None
    global_config.reset()
