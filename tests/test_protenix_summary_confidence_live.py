from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch

from astrolabe.core import global_config
from astrolabe.evaluators.protenix import ProtenixEvaluator
from astrolabe.losses.confidence import binder_ptm, scaled_pae_min


class _TensorLike:
    """Tensor-like wrapper that only supports detach/cpu/numpy conversion."""

    def __init__(self, value):
        self._value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


def _build_prediction(
    summary_confidence: dict | list[dict],
    *,
    atom_plddt=None,
) -> dict:
    summaries = summary_confidence if isinstance(summary_confidence, list) else [summary_confidence]
    full_data = []
    for sample_idx, _summary in enumerate(summaries):
        offset = float(sample_idx) * 0.01
        sample_atom_plddt = (
            atom_plddt
            if atom_plddt is not None
            else torch.tensor(
                [0.8 + offset, 0.7 + offset, 0.9 + offset, 0.85 + offset],
                dtype=torch.float32,
            )
        )
        full_data.append(
            {
                "atom_plddt": sample_atom_plddt,
                "token_pair_pae": torch.tensor(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [1.0, 0.0, 0.5, 4.0],
                        [2.0, 0.5, 0.0, 1.5],
                        [3.0, 4.0, 1.5, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "token_pair_pde": torch.tensor(
                    [
                        [0.0, 2.0, 3.0, 4.0],
                        [2.0, 0.0, 1.0, 5.0],
                        [3.0, 1.0, 0.0, 2.5],
                        [4.0, 5.0, 2.5, 0.0],
                    ],
                    dtype=torch.float32,
                ),
            }
        )
    return {
        "summary_confidence": summaries,
        "atom_to_residue": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "full_data": full_data,
    }


def _run_prepare_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    summary_confidence: dict,
    atom_plddt=None,
):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path / "out")
    global_config.set_variable("step", 0)

    raw_cif = tmp_path / "input_model.cif"
    raw_cif.write_text("data_dummy\n#\n", encoding="utf-8")

    evaluator = ProtenixEvaluator(
        model_runner=types.SimpleNamespace(
            configs=types.SimpleNamespace(seed=7, dump_dir=tmp_path / "dump")
        )
    )
    monkeypatch.setattr(
        evaluator,
        "_find_raw_prediction_cif",
        lambda **_kwargs: raw_cif,
    )
    output = {"AB": _build_prediction(summary_confidence, atom_plddt=atom_plddt)}
    inputs = {"AB": types.SimpleNamespace(name="AB", chains=[])}
    try:
        return evaluator.prepare_output(output, inputs, runtime_config={})
    finally:
        global_config.reset()


def test_prepare_output_extracts_tensor_like_summary_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_prepare_output(
        tmp_path,
        monkeypatch,
        summary_confidence={
            "ptm": _TensorLike(0.71),
            "iptm": _TensorLike(0.31),
            "chain_ptm": _TensorLike([0.2, 0.8]),
            "chain_iptm": _TensorLike([0.31, 0.31]),
            "chain_plddt": _TensorLike([0.82, 0.85]),
            "chain_gpde": _TensorLike([2.6, 0.4]),
            "chain_pair_iptm": _TensorLike([[0.0, 0.31], [0.31, 0.0]]),
            "chain_pair_iptm_global": _TensorLike([[0.0, 0.32], [0.32, 0.0]]),
            "chain_pair_plddt": _TensorLike([[0.0, 0.84], [0.84, 0.0]]),
            "chain_pair_gpde": _TensorLike([[0.0, 7.1], [7.1, 0.0]]),
            "chain_pair_pae_min": _TensorLike([[0.0, 2.5], [2.5, 0.0]]),
        },
    )

    scores = result.chain_pair_scores["AB"]
    assert result.ptm["AB"] == pytest.approx([0.71])
    assert result.iptm["AB"] == pytest.approx([0.31])
    assert binder_ptm(result, binder_chain="B")["AB"] == pytest.approx(0.8)
    assert scaled_pae_min(result, chain_a="A", chain_b="B", cutoff=10.0)["AB"] == pytest.approx(0.75)
    assert np.allclose(scores["chain_ptm"][0], np.array([0.2, 0.8]))
    assert np.allclose(scores["chain_iptm"][0], np.array([0.31, 0.31]))
    assert np.allclose(scores["chain_plddt"][0], np.array([0.82, 0.85]))
    assert np.allclose(scores["chain_gpde"][0], np.array([2.6, 0.4]))
    assert np.allclose(scores["chain_pair_iptm"][0], np.array([[0.0, 0.31], [0.31, 0.0]]))
    assert np.allclose(
        scores["chain_pair_iptm_global"][0], np.array([[0.0, 0.32], [0.32, 0.0]])
    )
    assert np.allclose(scores["chain_pair_plddt"][0], np.array([[0.0, 0.84], [0.84, 0.0]]))
    assert np.allclose(scores["chain_pair_gpde"][0], np.array([[0.0, 7.1], [7.1, 0.0]]))
    assert np.allclose(scores["chain_pair_pae_min"][0], np.array([[0.0, 2.5], [2.5, 0.0]]))


@pytest.mark.parametrize(
    "atom_plddt",
    [
        np.array([0.8, 0.7, 0.9, 0.85], dtype=np.float32),
        [0.8, 0.7, 0.9, 0.85],
    ],
)
def test_prepare_output_normalizes_atom_plddt_array_like_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atom_plddt,
) -> None:
    result = _run_prepare_output(
        tmp_path,
        monkeypatch,
        summary_confidence={"ptm": 0.71, "iptm": 0.31},
        atom_plddt=atom_plddt,
    )

    assert np.allclose(result.plddt["AB"][0], np.array([0.8, 0.7, 0.9, 0.85]))


def test_prepare_output_exposes_structure_start_soft_select_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path / "out")
    global_config.init_const(
        "structure_start",
        types.SimpleNamespace(config=types.SimpleNamespace(mode="soft_select")),
    )
    global_config.set_variable("step", 0)
    raw_cifs = []
    for sample_idx in range(2):
        raw_cif = tmp_path / f"sample_{sample_idx}.cif"
        raw_cif.write_text(f"data_sample_{sample_idx}\n#\n", encoding="utf-8")
        raw_cifs.append(raw_cif)
    evaluator = ProtenixEvaluator(
        model_runner=types.SimpleNamespace(
            configs=types.SimpleNamespace(seed=7, dump_dir=tmp_path / "dump")
        )
    )

    def _fake_find_raw_prediction_cif(**kwargs):
        return raw_cifs[int(kwargs.get("sample_idx", 0))]

    monkeypatch.setattr(evaluator, "_find_raw_prediction_cif", _fake_find_raw_prediction_cif)
    output = {
        "AB": _build_prediction(
            [
                {"ptm": 0.71, "iptm": 0.31},
                {"ptm": 0.82, "iptm": 0.44},
            ]
        )
    }
    inputs = {"AB": types.SimpleNamespace(name="AB", chains=[])}
    try:
        result = evaluator.prepare_output(output, inputs, runtime_config={})
    finally:
        global_config.reset()

    candidates = getattr(result, "candidate_scores", None)
    assert candidates is not None
    assert len(candidates) == 2
    assert candidates[0].ptm["AB"] == pytest.approx([0.71])
    assert candidates[1].iptm["AB"] == pytest.approx([0.44])
    assert candidates[1].complex_paths["AB"].name.endswith("candidate_001_model.cif")
    assert candidates[1].complex_paths["AB"].exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prepare_output_extracts_cuda_summary_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_prepare_output(
        tmp_path,
        monkeypatch,
        summary_confidence={
            "ptm": 0.72,
            "iptm": 0.33,
            "chain_ptm": torch.tensor([0.25, 0.9], device="cuda"),
            "chain_iptm": torch.tensor([0.33, 0.33], device="cuda"),
            "chain_plddt": torch.tensor([0.81, 0.86], device="cuda"),
            "chain_gpde": torch.tensor([2.7, 0.5], device="cuda"),
            "chain_pair_iptm": torch.tensor([[0.0, 0.33], [0.33, 0.0]], device="cuda"),
            "chain_pair_iptm_global": torch.tensor(
                [[0.0, 0.34], [0.34, 0.0]], device="cuda"
            ),
            "chain_pair_plddt": torch.tensor([[0.0, 0.85], [0.85, 0.0]], device="cuda"),
            "chain_pair_gpde": torch.tensor([[0.0, 7.2], [7.2, 0.0]], device="cuda"),
        },
    )

    scores = result.chain_pair_scores["AB"]
    assert binder_ptm(result, binder_chain="B")["AB"] == pytest.approx(0.9)
    assert np.allclose(scores["chain_pair_iptm"][0], np.array([[0.0, 0.33], [0.33, 0.0]]))
    assert np.allclose(scores["chain_pair_gpde"][0], np.array([[0.0, 7.2], [7.2, 0.0]]))
