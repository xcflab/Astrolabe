from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

from omegaconf import OmegaConf

from astrolabe.evaluators.esmfold import (
    ESMFoldEvaluator,
    _adapt_esmfold_state_dict_for_openfold,
    _load_esmfold_model_class,
)
from astrolabe.evaluators.esmfold2 import ESMFold2Evaluator
from astrolabe.evaluators.utils import construct_evaluators


def test_construct_evaluators_merges_esmfold_base_and_term_defaults(tmp_path: Path) -> None:
    class _EvalConf:
        esmfold = {"model_name": "/tmp/base_model.pt", "device": "cpu"}
        evaluator_terms = {"esmfold": {"_type": "ESMFold", "device": "cuda"}}

    evaluators = construct_evaluators(_EvalConf(), raw_model_results_dir=tmp_path)
    evaluator = evaluators["esmfold"]
    assert isinstance(evaluator, ESMFoldEvaluator)
    assert evaluator._model_name == "/tmp/base_model.pt"
    assert evaluator._device == "cuda"


def test_construct_evaluators_merges_esmfold2_base_and_term_defaults(tmp_path: Path) -> None:
    class _EvalConf:
        esmfold = {"model_name": "/tmp/esmfold_v1.pt", "device": "cpu"}
        esmfold2 = {
            "model_name": "biohub/ESMFold2-Fast",
            "python_bin": "/tmp/esmfold2/bin/python",
            "esm_repo_path": "/tmp/biohub_esm",
            "transformers_repo_path": "/tmp/biohub_transformers",
            "device": "cpu",
        }
        evaluator_terms = {"esmfold2": {"_type": "esmfold2", "device": "cuda"}}

    evaluators = construct_evaluators(_EvalConf(), raw_model_results_dir=tmp_path)
    evaluator = evaluators["esmfold2"]
    assert isinstance(evaluator, ESMFold2Evaluator)
    assert evaluator._model_name == "biohub/ESMFold2-Fast"
    assert evaluator._device == "cuda"
    assert evaluator.runner.python_bin == "/tmp/esmfold2/bin/python"
    assert evaluator.runner.esm_repo_path == "/tmp/biohub_esm"
    assert evaluator.runner.transformers_repo_path == "/tmp/biohub_transformers"


def test_base_config_declares_esmfold2_defaults() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "base.yaml")
    esmfold2_cfg = OmegaConf.to_container(cfg.evaluator.esmfold2, resolve=False)
    assert esmfold2_cfg["model_name"] == "biohub/ESMFold2"
    assert esmfold2_cfg["python_bin"] == "${subprocess_envs.esmfold2.python_bin}"
    assert esmfold2_cfg["esm_repo_path"] == "${subprocess_envs.esmfold2.esm_repo_path}"
    assert esmfold2_cfg["transformers_repo_path"] == "${subprocess_envs.esmfold2.transformers_repo_path}"
    assert esmfold2_cfg["device"] is None


def test_esmfold2_adapts_openfold_point_projection_state_keys() -> None:
    q_weight = object()
    q_bias = object()
    kv_weight = object()
    kv_bias = object()
    state = {
        "trunk.structure_module.ipa.linear_q_points.weight": q_weight,
        "trunk.structure_module.ipa.linear_q_points.bias": q_bias,
        "trunk.structure_module.ipa.linear_kv_points.weight": kv_weight,
        "trunk.structure_module.ipa.linear_kv_points.bias": kv_bias,
    }

    adapted = _adapt_esmfold_state_dict_for_openfold(state)

    assert adapted["trunk.structure_module.ipa.linear_q_points.linear.weight"] is q_weight
    assert adapted["trunk.structure_module.ipa.linear_q_points.linear.bias"] is q_bias
    assert adapted["trunk.structure_module.ipa.linear_kv_points.linear.weight"] is kv_weight
    assert adapted["trunk.structure_module.ipa.linear_kv_points.linear.bias"] is kv_bias


def test_esmfold_model_class_import_falls_back_to_upstream_esm(monkeypatch) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "esm2.esmfold.v1.esmfold":
            raise ImportError("renamed esm2 package is unavailable")
        if name == "esm.esmfold.v1.esmfold":
            return SimpleNamespace(ESMFold=sentinel)
        raise AssertionError(name)

    monkeypatch.setattr("astrolabe.evaluators.esmfold.importlib.import_module", fake_import_module)
    assert _load_esmfold_model_class() is sentinel
    assert calls == [
        "esm2.esmfold.v1.esmfold",
        "esm2.esmfold.v1.esmfold",
        "esm.esmfold.v1.esmfold",
    ]


def test_esmfold_model_class_retries_renamed_package_after_vendor_paths(
    monkeypatch,
) -> None:
    sentinel = object()
    calls = 0

    def fake_import_module(name: str):
        nonlocal calls
        assert name == "esm2.esmfold.v1.esmfold"
        calls += 1
        if calls == 1:
            raise ImportError("openfold path is not active yet")
        return SimpleNamespace(ESMFold=sentinel)

    monkeypatch.setattr(
        "astrolabe.evaluators.esmfold.importlib.import_module",
        fake_import_module,
    )
    assert _load_esmfold_model_class() is sentinel
    assert calls == 2


def test_esmfold2_uses_biohub_subprocess_runner(tmp_path: Path) -> None:
    evaluator = ESMFold2Evaluator(
        model_name="biohub/ESMFold2",
        python_bin="/tmp/esmfold2/bin/python",
        esm_repo_path="/tmp/biohub_esm",
        transformers_repo_path="/tmp/biohub_transformers",
        device="cpu",
        raw_model_results_dir=tmp_path,
    )

    assert evaluator.runner.model_name == "biohub/ESMFold2"
    assert evaluator.runner.python_bin == "/tmp/esmfold2/bin/python"
    assert evaluator.runner.esm_repo_path == "/tmp/biohub_esm"
    assert evaluator.runner.transformers_repo_path == "/tmp/biohub_transformers"
    assert evaluator.raw_model_results_dir == tmp_path
