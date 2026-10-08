from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.boltz import BoltzEvaluator
from astrolabe.evaluators.chai1 import Chai1Evaluator


def _protein_chain(*, chain_id: str, sequence: str) -> ProteinChain:
    return ProteinChain(
        id=chain_id,
        sequence=sequence,
        ptms=[],
        unpaired_msa="",
        paired_msa="",
        templates=[],
    )


def test_chai_prepare_output_exposes_chain_pair_scores(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.set_variable("step", 0)
    (tmp_path / "structures").mkdir(parents=True, exist_ok=True)

    evaluator = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai_out")
    inp = Input(
        name="AB",
        chains=[
            _protein_chain(chain_id="A", sequence="AC"),
            _protein_chain(chain_id="B", sequence="DE"),
        ],
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    best_cif = tmp_path / "best.cif"
    best_cif.write_text("data_dummy\n#\n", encoding="utf-8")
    ranking = SimpleNamespace(
        aggregate_score=0.9,
        ptm_scores=SimpleNamespace(
            complex_ptm=0.61,
            interface_ptm=0.42,
            per_chain_ptm=np.array([0.31, 0.71], dtype=float),
            per_chain_pair_iptm=np.array([[0.0, 0.42], [0.42, 0.0]], dtype=float),
        ),
        plddt_scores=SimpleNamespace(
            per_chain_plddt=np.array([0.81, 0.91], dtype=float)
        ),
    )
    output = {
        "AB": SimpleNamespace(
            plddt=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float),
            pae=[np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)],
            pde=[np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)],
            ranking_data=[ranking],
            cif_paths=[best_cif],
        )
    }

    scores = evaluator.prepare_output(output, {"AB": inp}, runtime_config={})

    assert np.allclose(scores.chain_pair_scores["AB"]["chain_ptm"][0], [0.31, 0.71])
    assert np.allclose(
        scores.chain_pair_scores["AB"]["chain_pair_iptm"][0],
        [[0.0, 0.42], [0.42, 0.0]],
    )
    assert np.allclose(scores.chain_pair_scores["AB"]["chain_plddt"][0], [0.81, 0.91])
    global_config.reset()


def test_boltz_prepare_output_exposes_chain_pair_scores(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.set_variable("step", 0)
    (tmp_path / "structures").mkdir(parents=True, exist_ok=True)

    evaluator = BoltzEvaluator(model_runner=None, output_dir=tmp_path / "boltz_out")
    inp = SimpleNamespace(protein_chains=[SimpleNamespace(id="A", sequence="AC")])
    best_cif = tmp_path / "best_model.cif"
    best_cif.write_text("data_dummy\n#\n", encoding="utf-8")
    score_json = tmp_path / "confidence_model_0.json"
    score_json.write_text(
        """
        {
          "complex_plddt": 0.75,
          "ptm": 0.66,
          "iptm": 0.44,
          "complex_pde": 2.5,
          "chains_ptm": {"0": 0.31, "1": 0.73},
          "pair_chains_iptm": {"0": {"0": 0.31, "1": 0.44}, "1": {"0": 0.44, "1": 0.73}}
        }
        """,
        encoding="utf-8",
    )

    evaluator.rechain_structure = lambda *args, **kwargs: (np.array([0.2, 0.3]), best_cif)
    output = {"AB": {"out_dir": tmp_path}}
    runtime_config = {"chain2input_chain": {"AB": {"A": "A"}}, "chain2proto": {"A": "A"}}

    import glob

    orig_glob = glob.glob

    def _fake_glob(pattern: str):
        if pattern.endswith("*_model_0.cif"):
            return [str(best_cif)]
        if pattern.endswith("*_model_*.json"):
            return [str(score_json)]
        return orig_glob(pattern)

    glob.glob = _fake_glob
    try:
        scores = evaluator.prepare_output(output, {"AB": inp}, runtime_config)
    finally:
        glob.glob = orig_glob
        global_config.reset()

    assert np.allclose(scores.chain_pair_scores["AB"]["chain_ptm"][0], [0.31, 0.73])
    assert np.allclose(
        scores.chain_pair_scores["AB"]["chain_pair_iptm"][0],
        [[0.31, 0.44], [0.44, 0.73]],
    )
