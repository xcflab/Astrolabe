from __future__ import annotations

import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import construct_evaluators


def test_uses_alphafold3_evaluator_ignores_unused_unresolved_defaults() -> None:
    import run as run_mod

    class _StageUnit:
        evaluator = "Dummy"

    class _StageManager:
        default_setting = _StageUnit()
        stage_args = {"default": _StageUnit()}

    args = types.SimpleNamespace(
        evaluator=OmegaConf.create(
            {
                "alphafold3": {
                    "model_dir": "${weight_path.alphafold3}",
                    "datapipeline": {
                        "jackhmmer_binary_path": "${hmmer.jackhmmer}",
                    },
                },
                "rhofold": {
                    "checkpoint": "${rhofold.checkpoint}",
                },
                "evaluator_terms": {
                    "Dummy": {"_type": "dummy"},
                },
            }
        )
    )

    assert run_mod._uses_alphafold3_evaluator(args, _StageManager()) is False


def test_uses_alphafold3_evaluator_includes_transition_gate_evaluator() -> None:
    import run as run_mod

    class _DefaultStage:
        evaluator = "Protenix"

    class _StageUnit:
        evaluator = "Protenix"
        transition_consistency_gate = types.SimpleNamespace(evaluator="AF3")

    class _StageManager:
        default_setting = _DefaultStage()
        stage_args = {"default": _DefaultStage(), "stage_1": _StageUnit()}

    args = types.SimpleNamespace(
        evaluator=OmegaConf.create(
            {
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                    "AF3": {"_type": "alphafold3"},
                },
            }
        )
    )

    assert run_mod._uses_alphafold3_evaluator(args, _StageManager()) is True


def test_construct_evaluators_does_not_use_global_stage_manager_for_filtering(
    tmp_path: Path,
) -> None:
    """Without explicit enabled_aliases, invalid evaluator_terms should fail fast."""
    global_config.reset()

    class _StageUnit:
        evaluator = "Dummy"

    class _StageManager:
        default_setting = _StageUnit()
        stage_args = {"default": _StageUnit()}

    global_config.init_const("stage_manager", _StageManager())

    evaluator_conf = types.SimpleNamespace(
        evaluator_terms={
            "Dummy": {"_type": "dummy"},
            "Unused": {"_type": "this_is_not_a_real_evaluator"},
        }
    )

    with pytest.raises(ValueError, match="Unknown evaluator name"):
        construct_evaluators(evaluator_conf, raw_model_results_dir=tmp_path, oligo_names=None)


def test_initialize_pipeline_components_passes_enabled_aliases_to_construct_evaluators(
    tmp_path: Path, monkeypatch
) -> None:
    global_config.reset()

    import run as run_mod

    calls: dict[str, object] = {}

    def _fake_construct_evaluators(
        evaluator_conf,
        raw_model_results_dir,
        oligo_names=None,
        *,
        enabled_aliases=None,
    ):
        calls["enabled_aliases"] = enabled_aliases
        return {}

    monkeypatch.setattr(run_mod, "construct_evaluators", _fake_construct_evaluators)
    monkeypatch.setattr(run_mod, "initialize_global_config", lambda _args: None)
    monkeypatch.setattr(run_mod, "setup_protomers", lambda _args: None)
    monkeypatch.setattr(run_mod, "apply_initializer", lambda _args: None)
    monkeypatch.setattr(run_mod, "build_mutators", lambda _conf, _dir: {})
    monkeypatch.setattr(run_mod, "construct_standard_input", lambda *_a, **_k: {})
    monkeypatch.setattr(run_mod, "initialize_predockers", lambda _p: None)
    monkeypatch.setattr(run_mod, "_uses_alphafold3_evaluator", lambda *_a, **_k: False)

    args = types.SimpleNamespace(
        raw_model_results_dir=tmp_path,
        mutator=types.SimpleNamespace(mutator_terms={}, enabled_aliases=[]),
        loss={"loss_terms": {}},
        stages=OmegaConf.create(
            {
                "default": {
                    "oligo": ["AB"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "evaluator": "Dummy",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                }
            }
        ),
        evaluator=types.SimpleNamespace(
            evaluator_terms={"Dummy": {"_type": "dummy"}},
            enabled_aliases=["Dummy"],
        ),
        seed=1,
        predocker=None,
        obabel_path=None,
        molfile_to_params=None,
        dssp_path=None,
        usalign_path=None,
    )

    run_mod.initialize_pipeline_components(args)

    assert calls["enabled_aliases"] == ["Dummy"]
