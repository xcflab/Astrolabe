from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.argparser import (
    _resolve_repo_relative_path,
    create_auto_managed_raw_model_results_dir,
    parse_arguments,
)
from astrolabe.core import global_config


def _base_args(tmp_path: Path):
    return {
        "task_name": "t",
        "out_dir": str(tmp_path),
        "raw_model_results_dir": str(tmp_path / "raw"),
        "seed": 1,
        "debug": False,
        "named_files": None,
        "seqs": {"A": "AAA"},
        "types": None,
        "dssp_path": None,
        "usalign_path": None,
        # Required structured sections
        "loss": {"loss_terms": {}},
    }


def test_resolve_repo_relative_path_checks_parent_checkout_for_worktree_vendor(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    worktree = repo / ".worktrees" / "feature"
    executable = repo / "vendor" / "usalign" / "USalign"
    executable.parent.mkdir(parents=True)
    worktree.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    resolved = _resolve_repo_relative_path(
        "vendor/usalign/USalign",
        repo_root=worktree,
    )

    assert resolved == str(executable)


def test_parse_arguments_sets_evaluator_enabled_aliases_from_stages(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                },
                "stage_1": {
                    "_entrance": "plddt > 0.0",
                    "oligo": ["A"],
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_override",
                },
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {
                    "E_default": {"_type": "dummy"},
                    "E_override": {"_type": "dummy"},
                }
            },
        }
    )

    resolved = parse_arguments(args)
    assert resolved.evaluator.enabled_aliases == ["E_default", "E_override"]


def test_parse_arguments_keeps_structure_initializer_mutator_alias(
    tmp_path: Path,
) -> None:
    global_config.reset()
    source = tmp_path / "source.pdb"
    source.write_text("ATOM\n", encoding="utf-8")
    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "initializer": {
                "mode": "structure",
                "sequence_designer": "InitMPNN",
                "structure": {
                    "source": str(source),
                    "chain_map": {"A": "A"},
                },
            },
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {
                    "MCMC": {"_type": "MCMC"},
                    "InitMPNN": {"_type": "LigandMPNN"},
                },
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "LigandMPNN": {},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {"E_default": {"_type": "dummy"}}
            },
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert resolved.mutator.enabled_aliases == ["MCMC", "InitMPNN"]
    assert set(saved_cfg.mutator.mutator_terms) == {"MCMC", "InitMPNN"}


def test_parse_arguments_persists_rfd3_initializer_runtime_paths(
    tmp_path: Path,
) -> None:
    global_config.reset()
    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "initializer": {
                "mode": "rfd3",
                "sequence_designer": "InitMPNN",
                "rfd3": {
                    "repo_path": "/tmp/foundry",
                    "python_bin": "/tmp/foundry-python",
                    "ckpt_path": "/tmp/rfd3.ckpt",
                    "specification": {"length": "3"},
                    "chain_map": {"A": "A"},
                },
            },
            "rfd3": {
                "repo_path": "/tmp/foundry",
                "ckpt_path": "/tmp/rfd3.ckpt",
            },
            "weight_path": {
                "ligandmpnn": "/tmp/ligandmpnn",
                "rfd3": "/tmp/rfd3.ckpt",
            },
            "subprocess_envs": {
                "rf3": {"python_bin": "/tmp/foundry-python"},
            },
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {
                    "MCMC": {"_type": "MCMC"},
                    "InitMPNN": {"_type": "LigandMPNN"},
                },
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "LigandMPNN": {},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {"E_default": {"_type": "dummy"}}
            },
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert saved_cfg.rfd3.repo_path == "/tmp/foundry"
    assert saved_cfg.rfd3.ckpt_path == "/tmp/rfd3.ckpt"
    assert saved_cfg.subprocess_envs.rf3.python_bin == "/tmp/foundry-python"
    assert saved_cfg.weight_path.ligandmpnn == "/tmp/ligandmpnn"
    assert saved_cfg.weight_path.rfd3 == "/tmp/rfd3.ckpt"


def test_parse_arguments_sets_evaluator_enabled_aliases_with_default_fallback(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                },
                "stage_1": {
                    "_entrance": "plddt > 0.0",
                    "oligo": ["A"],
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    # evaluator intentionally omitted -> should fall back to default evaluator
                },
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {
                    "E_default": {"_type": "dummy"},
                    "Unused": {"_type": "dummy"},
                }
            },
        }
    )

    resolved = parse_arguments(args)
    assert resolved.evaluator.enabled_aliases == ["E_default"]


def test_parse_arguments_rejects_disabled_evaluator_alias(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {"evaluator_terms": {"E_default": None}},
        }
    )

    with pytest.raises(ValueError, match="Stages reference evaluators"):
        parse_arguments(args)


def test_parse_arguments_rejects_disabled_mutator_alias(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": None},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {"evaluator_terms": {"E_default": {"_type": "dummy"}}},
        }
    )

    with pytest.raises(ValueError, match="Stages reference mutators"):
        parse_arguments(args)


def test_parse_arguments_rejects_non_mapping_stages(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": ["default"],
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {"evaluator_terms": {"E_default": {"_type": "dummy"}}},
        }
    )

    with pytest.raises(ValueError, match="stages"):
        parse_arguments(args)


def test_create_auto_managed_raw_model_results_dir_avoids_name_collisions(
    tmp_path: Path,
) -> None:
    stale_dir = tmp_path / "gh_demo_001_run"
    stale_dir.mkdir()

    raw_dir, marker = create_auto_managed_raw_model_results_dir(
        run_name="demo_001_run",
        tmp_root=tmp_path,
        fallback_out_dir=tmp_path / "fallback",
    )

    assert raw_dir.name == "gh_demo_001_run_2"
    assert raw_dir.exists()
    assert marker == raw_dir / ".gh_auto_managed"
    assert marker.exists()


def test_parse_arguments_prunes_persisted_config_but_keeps_live_args(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "helper_expr",
                    "mutators": ["MUsed"],
                    "mutator_weights": [1.0],
                    "evaluator": "EUsed",
                },
                "stage_1": {
                    "_entrance": "enter_expr",
                    "oligo": ["A"],
                    "mutators": ["MUsed"],
                    "mutator_weights": [1.0],
                },
            },
            "mutator": {
                "mutator_terms": {
                    "MUsed": {"_type": "MCMC"},
                    "MUnused": {"_type": "NAMPNN"},
                },
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "NAMPNN": {"repo_path": "${oc.env:GH_UNUSED_NAMPNN_REPO}"},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {
                    "EUsed": {"_type": "UsedEval"},
                    "EUnused": {"_type": "UnusedEval"},
                },
                "scheduling": {
                    "overlap_cpu_postprocess": True,
                    "cpu_postprocess_workers": 2,
                },
                "UsedEval": {"path": "/tmp/used"},
                "UnusedEval": {"path": "${oc.env:GH_UNUSED_EVAL_PATH}"},
            },
            "loss": {
                "loss_terms": {
                    "used_loss": {"_type": "ptm"},
                    "unused_loss": {"_type": "ptm"},
                },
                "expression_terms": {
                    "helper_expr": "used_loss",
                    "enter_expr": "helper_expr > 0",
                    "unused_expr": "unused_loss",
                },
            },
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert "EUnused" in resolved.evaluator.evaluator_terms
    assert "MUnused" in resolved.mutator.mutator_terms
    assert "unused_loss" in resolved.loss.loss_terms
    assert resolved.evaluator.scheduling.overlap_cpu_postprocess is True
    assert set(saved_cfg.evaluator.evaluator_terms.keys()) == {"EUsed"}
    assert saved_cfg.evaluator.scheduling.overlap_cpu_postprocess is True
    assert saved_cfg.evaluator.scheduling.cpu_postprocess_workers == 2
    assert hasattr(saved_cfg.evaluator, "UsedEval")
    assert not hasattr(saved_cfg.evaluator, "UnusedEval")
    assert set(saved_cfg.mutator.mutator_terms.keys()) == {"MUsed"}
    assert hasattr(saved_cfg.mutator, "MCMC")
    assert not hasattr(saved_cfg.mutator, "NAMPNN")
    assert set(saved_cfg.mutator.selectors.keys()) == {"MUsed"}
    assert set(saved_cfg.loss.loss_terms.keys()) == {"used_loss"}
    assert set(saved_cfg.loss.expression_terms.keys()) == {"helper_expr", "enter_expr"}
    assert "unused_expr" not in saved_cfg.loss.expression_terms


def test_parse_arguments_persisted_config_hides_inactive_legacy_root_fields(
    tmp_path: Path,
) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "initial_temperature": 0.05,
            "use_metropolis_acceptance": False,
            "mutation_num": 1,
            "reinitialize_random_segments_with_dplm": False,
            "initializer": {"mode": "random"},
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
                "mutation_num": None,
            },
            "evaluator": {
                "evaluator_terms": {"E_default": {"_type": "dummy"}},
            },
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert resolved.initial_temperature == 0.05
    assert resolved.mutation_num == 1
    assert resolved.reinitialize_random_segments_with_dplm is False
    assert not hasattr(saved_cfg, "initial_temperature")
    assert not hasattr(saved_cfg, "mutation_num")
    assert not hasattr(saved_cfg, "reinitialize_random_segments_with_dplm")
    assert saved_cfg.stages.default.mutation_num == 1
    assert saved_cfg.initializer.mode == "random"


def test_parse_arguments_persisted_config_migrates_legacy_dplm_initializer(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "use_metropolis_acceptance": False,
            "reinitialize_random_segments_with_dplm": True,
            "initializer": {"mode": "random"},
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {"E_default": {"_type": "dummy"}},
            },
        }
    )

    with caplog.at_level("WARNING"):
        resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert resolved.reinitialize_random_segments_with_dplm is True
    assert saved_cfg.initializer.mode == "dplm"
    assert not hasattr(saved_cfg, "reinitialize_random_segments_with_dplm")
    assert any("initializer.mode: dplm" in record.getMessage() for record in caplog.records)


def test_parse_arguments_rejects_legacy_mutator_mutation_num_root(
    tmp_path: Path,
) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "E_default",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
                "mutation_num": 5,
            },
            "evaluator": {
                "evaluator_terms": {"E_default": {"_type": "dummy"}},
            },
        }
    )

    with pytest.raises(ValueError, match="mutator.mutation_num"):
        parse_arguments(args)


def test_parse_arguments_persisted_config_keeps_refold_relevant_backend_for_used_alias(
    tmp_path: Path,
) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "stages": {
                "default": {
                    "oligo": ["A"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                    "evaluator": "StrictPX",
                }
            },
            "mutator": {
                "mutator_terms": {"MCMC": {"_type": "MCMC"}},
                "MCMC": {"mutation_method": "random", "select_positions": "random"},
                "selector_expression": None,
            },
            "evaluator": {
                "evaluator_terms": {
                    "StrictPX": {"_type": "protenix"},
                    "UnusedAF3": {"_type": "alphafold3"},
                },
                "protenix": {"repo_path": "/tmp/protenix"},
                "alphafold3": {"repo_path": "/tmp/af3"},
            },
            "weight_path": {
                "protenix": "/tmp/protenix_weights",
                "alphafold3": "/tmp/af3_weights",
            },
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert set(saved_cfg.evaluator.evaluator_terms.keys()) == {"StrictPX"}
    assert hasattr(saved_cfg.evaluator, "protenix")
    assert hasattr(saved_cfg.evaluator, "alphafold3")
    assert saved_cfg.weight_path.protenix == "/tmp/protenix_weights"
    assert saved_cfg.weight_path.alphafold3 == "/tmp/af3_weights"


def test_parse_arguments_prunes_unused_predocker_and_env_sections(tmp_path: Path) -> None:
    global_config.reset()

    args = OmegaConf.create(
        {
            **_base_args(tmp_path),
            "seqs": {"A": "AAAA", "B": "BBBB", "C": "CCCC"},
            "stages": {
                "default": {
                    "oligo": ["ABC"],
                    "mutation_num": 1,
                    "weighted_score_expression": "used_loss",
                    "mutators": ["LMPNNUsed"],
                    "mutator_weights": [1.0],
                    "evaluator": "EUsed",
                }
            },
            "mutator": {
                "mutator_terms": {
                    "LMPNNUsed": {
                        "_type": "LigandMPNN",
                        "predocker": ["dock_used"],
                    },
                    "MUnused": {"_type": "NAMPNN"},
                },
                "LigandMPNN": {"checkpoint_path": "/tmp/ligandmpnn"},
                "NAMPNN": {"repo_path": "/tmp/nampnn"},
                "selector_expression": None,
            },
            "predocker": {
                "predocker_terms": {
                    "dock_used": {"_type": "autodock_vina"},
                    "dock_unused": {"_type": "RAPiDock"},
                },
                "autodock_vina": {"vina_path": "/tmp/vina", "obabel_path": "/tmp/obabel"},
                "RAPiDock": {"repo_path": "/tmp/rapidock", "python_bin": "/tmp/rapidock_py"},
            },
            "evaluator": {
                "evaluator_terms": {
                    "EUsed": {"_type": "dummy"},
                },
                "dummy": {"path": "/tmp/dummy"},
            },
            "loss": {
                "loss_terms": {
                    "used_loss": {"_type": "ptm"},
                },
            },
            "weight_path": {
                "ligandmpnn": "/tmp/ligandmpnn_weights",
                "rapidock": "/tmp/rapidock_weights",
            },
            "subprocess_envs": {
                "dplm": {"python_bin": "/tmp/dplm_py", "repo_path": "/tmp/dplm_repo"},
                "rapidock": {"python_bin": "/tmp/rapidock_py", "repo_path": "/tmp/rapidock_repo"},
            },
            "autodock": {
                "vina_path": "/tmp/vina",
                "autodock_gpu_path": "/tmp/adgpu",
                "autogrid4_path": "/tmp/autogrid4",
                "mk_prepare_receptor": "/tmp/mk_prepare_receptor",
            },
            "obabel_path": "/tmp/obabel",
        }
    )

    resolved = parse_arguments(args)
    saved_cfg = OmegaConf.load(Path(resolved.out_dir) / "config.yaml")

    assert set(saved_cfg.predocker.predocker_terms.keys()) == {"dock_used"}
    assert hasattr(saved_cfg.predocker, "autodock_vina")
    assert not hasattr(saved_cfg.predocker, "RAPiDock")
    assert saved_cfg.weight_path.ligandmpnn == "/tmp/ligandmpnn_weights"
    assert not hasattr(saved_cfg.weight_path, "rapidock")
    assert saved_cfg.autodock.vina_path == "/tmp/vina"
    assert not hasattr(saved_cfg.autodock, "autodock_gpu_path")
    assert saved_cfg.obabel_path == "/tmp/obabel"
    assert not hasattr(saved_cfg, "subprocess_envs") or not hasattr(saved_cfg.subprocess_envs, "rapidock")
    assert not hasattr(saved_cfg, "subprocess_envs") or not hasattr(saved_cfg.subprocess_envs, "dplm")
