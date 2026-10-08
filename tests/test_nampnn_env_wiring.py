from __future__ import annotations

import importlib
from pathlib import Path

from omegaconf import OmegaConf

import astrolabe.cli as cli
from astrolabe.runtime_requirements import task_requirements
from astrolabe.runtime_usage import analyze_runtime_usage


def test_cli_task_requirements_shim_matches_runtime_requirements() -> None:
    cfg = {
        "mutator": {
            "mutator_terms": {
                "na_mut": {
                    "_type": "nampnn",
                }
            }
        }
    }

    assert cli._task_requirements(cfg) == task_requirements(cfg)


def test_task_requirements_for_nampnn_include_subprocess_env_keys() -> None:
    req = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "na_mut": {
                        "_type": "nampnn",
                    }
                }
            }
        }
    )

    assert "subprocess_envs.nampnn.python_bin" in req["required_env_keys"]
    assert "subprocess_envs.nampnn.repo_path" in req["required_env_keys"]
    assert req["env_key_reasons"]["subprocess_envs.nampnn.python_bin"].startswith(
        "required by NAMPNN"
    )


def test_task_requirements_ignore_unused_nampnn_aliases() -> None:
    req = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "ligand": {"_type": "ligandmpnn"},
                    "na_mut": {"_type": "nampnn"},
                }
            },
            "stages": {
                "default": {
                    "mutators": ["ligand"],
                }
            },
        }
    )

    assert "weight_path.ligandmpnn" in req["required_env_keys"]
    assert "subprocess_envs.nampnn.python_bin" not in req["required_env_keys"]
    assert "subprocess_envs.nampnn.repo_path" not in req["required_env_keys"]


def test_task_requirements_include_ligandmpnn_weight_for_dynamic_mpnn() -> None:
    req = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "dynamic": {"_type": "DynamicMPNN"},
                }
            },
            "stages": {"default": {"mutators": ["dynamic"]}},
        }
    )

    assert "weight_path.ligandmpnn" in req["required_env_keys"]


def test_task_requirements_include_ligandmpnn_weight_for_multistate_proteinmpnn() -> None:
    req = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "far_apart": {
                        "_type": "MultistateProteinMPNN",
                    },
                }
            },
            "stages": {"default": {"mutators": ["far_apart"]}},
        }
    )

    assert "weight_path.ligandmpnn" in req["required_env_keys"]
    assert "prody" in req["required_imports"]


def test_task_requirements_ignore_unused_unresolved_evaluator_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "evaluator": {
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
                    "UnusedAF3": {"_type": "alphafold3"},
                },
            },
            "stages": {
                "default": {
                    "evaluator": "Dummy",
                }
            },
        }
    )

    req = task_requirements(OmegaConf.to_container(cfg, resolve=False))

    assert "weight_path.alphafold3" not in req["required_env_keys"]
    assert "hmmer.jackhmmer" not in req["required_env_keys"]
    assert "rhofold.checkpoint" not in req["required_env_keys"]


def test_task_requirements_include_transition_gate_evaluator_dependencies() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                    "AF3": {"_type": "alphafold3"},
                },
            },
            "stages": {
                "default": {
                    "evaluator": "Protenix",
                },
                "stage_1": {
                    "evaluator": "Protenix",
                    "transition_consistency_gate": {
                        "evaluator": "AF3",
                    },
                },
            },
        }
    )

    assert "weight_path.alphafold3" in req["required_env_keys"]
    assert "hmmer.jackhmmer" not in req["required_env_keys"]
    assert "weight_path.protenix" in req["required_env_keys"]


def test_task_requirements_for_alphafold2_include_tensorflow_import() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "AF2": {"_type": "alphafold2"},
                },
            },
            "stages": {
                "default": {
                    "evaluator": "AF2",
                }
            },
        }
    )

    assert "weight_path.alphafold" in req["required_env_keys"]
    assert "tensorflow" in req["required_imports"]


def test_task_requirements_include_hmmer_only_for_explicit_af3_data_pipeline() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "AF3": {"_type": "alphafold3", "run_data_pipeline": True},
                },
            },
            "stages": {"default": {"evaluator": "AF3"}},
        }
    )

    assert "weight_path.alphafold3" in req["required_env_keys"]
    assert "hmmer.jackhmmer" in req["required_env_keys"]
    assert req["env_key_reasons"]["hmmer.jackhmmer"].startswith(
        "required by explicit Alphafold3 MSA/data pipeline"
    )


def test_task_requirements_include_rhofold_and_evaluator_imports() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "RhoFold": {"_type": "rhofold"},
                    "Chai": {"_type": "chai1"},
                    "ESMFold": {"_type": "ESMFold"},
                }
            }
        }
    )

    assert "rhofold.checkpoint" in req["required_env_keys"]
    assert "rhofold.binary_dpath" not in req["required_env_keys"]
    assert "rhofold.database_dpath" not in req["required_env_keys"]
    assert "rhofold" in req["required_imports"]
    assert "chai_lab" in req["required_imports"]
    assert "esm2" in req["required_imports"]


def test_task_requirements_include_esmfold2_subprocess_runtime() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "ESMFold2": {"_type": "esmfold2"},
                }
            }
        }
    )

    assert "esm2" not in req["required_imports"]
    assert "torch" not in req["required_imports"]
    assert "subprocess_envs.esmfold2.python_bin" in req["required_env_keys"]
    assert "subprocess_envs.esmfold2.esm_repo_path" in req["required_env_keys"]
    assert "subprocess_envs.esmfold2.transformers_repo_path" in req["required_env_keys"]
    assert "weight_path.esmfold2" not in req["required_env_keys"]
    assert req["env_key_reasons"]["subprocess_envs.esmfold2.python_bin"].startswith(
        "required by Biohub ESMFold2"
    )
    assert req["required_env_repo_files"] == []


def test_task_requirements_include_opendde_repo_and_assets() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "OpenDDE": {"_type": "OpenDDE"},
                }
            },
            "stages": {"default": {"evaluator": "OpenDDE"}},
        }
    )

    assert "weight_path.opendde" in req["required_env_keys"]
    assert "subprocess_envs.opendde.python_bin" not in req["required_env_keys"]
    assert "subprocess_envs.opendde.repo_path" in req["required_env_keys"]
    assert "opendde" not in req["required_imports"]
    assert {
        "base_env_key": "weight_path.opendde",
        "path": "checkpoint/opendde.pt",
        "label": "opendde_checkpoint",
    } in req["required_env_repo_files"]


def test_task_requirements_include_opendde_preprocessing_tools_only_when_enabled() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "evaluator_terms": {
                    "OpenDDE": {
                        "_type": "OpenDDE",
                        "use_template": True,
                        "use_rna_msa": True,
                    },
                }
            },
        }
    )

    assert "hmmer.hmmsearch" in req["required_env_keys"]
    assert "hmmer.hmmbuild" in req["required_env_keys"]
    assert "hmmer.nhmmer" in req["required_env_keys"]
    assert "hmmer.hmmalign" in req["required_env_keys"]


def test_task_requirements_include_protenix_constraint_checkpoint_for_binding_site() -> None:
    req = task_requirements(
        {
            "binding_site": {"binder": "A", "target": "B:10"},
            "evaluator": {
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                }
            },
        }
    )

    assert "weight_path.protenix" in req["required_env_keys"]
    assert "weight_path.protenix_constraint" in req["required_env_keys"]


def test_task_requirements_include_cutlass_for_protenix_deepspeed_attention() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "protenix": {
                    "triangle_attention": "deepspeed",
                },
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                },
            },
        }
    )

    assert "CUTLASS_PATH" in req["required_env_keys"]
    assert "deepspeed" in req["required_imports"]


def test_task_requirements_include_cuequivariance_for_protenix_optin_backend() -> None:
    req = task_requirements(
        {
            "evaluator": {
                "protenix": {
                    "triangle_attention": "cuequivariance",
                    "triangle_multiplicative": "cuequivariance",
                },
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                },
            },
        }
    )

    assert "cuequivariance_torch" in req["required_imports"]


def test_task_requirements_include_autodock_ligand_prepare_and_rpxdock_backend() -> None:
    req = task_requirements(
        {
            "predocker": {
                "predocker_terms": {
                    "vina": {"_type": "autodock_vina"},
                    "symmetry": {
                        "_type": "symmetry_canonicalizer",
                        "backend": "rpxdock",
                    },
                }
            },
        }
    )

    assert "autodock.vina_path" in req["required_env_keys"]
    assert "autodock.mk_prepare_ligand" in req["required_env_keys"]
    assert "subprocess_envs.meeko.python_bin" in req["required_env_keys"]
    assert "subprocess_envs.rpxdock.python_bin" in req["required_env_keys"]
    assert "subprocess_envs.rpxdock.repo_path" in req["required_env_keys"]
    assert "subprocess_envs.rpxdock.willutil_repo_path" in req["required_env_keys"]
    assert "rpxdock" not in req["required_imports"]


def test_task_requirements_include_haddock3_for_used_nucleic_predocker() -> None:
    req = task_requirements(
        {
            "predocker": {
                "predocker_terms": {
                    "haddock": {"_type": "haddock3_nucleic"},
                    "unused_vina": {"_type": "autodock_vina"},
                }
            },
            "mutator": {
                "mutator_terms": {
                    "na": {"_type": "NAMPNN", "predocker": ["haddock"]},
                }
            },
            "stages": {"complex": {"mutators": ["na"]}},
        }
    )

    assert "subprocess_envs.haddock3.haddock3_bin" in req["required_env_keys"]
    assert "autodock.vina_path" not in req["required_env_keys"]


def test_task_requirements_follow_used_mutator_predocker_aliases_only() -> None:
    req = task_requirements(
        {
            "predocker": {
                "predocker_terms": {
                    "used_vina": {"_type": "autodock_vina"},
                    "unused_gpu": {"_type": "autodock_gpu"},
                }
            },
            "mutator": {
                "mutator_terms": {
                    "ligand": {"_type": "LigandMPNN", "predocker": ["used_vina"]},
                }
            },
            "stages": {"default": {"mutators": ["ligand"]}},
        }
    )

    assert "autodock.vina_path" in req["required_env_keys"]
    assert "autodock.autodock_gpu_path" not in req["required_env_keys"]


def test_runtime_usage_reports_stage_used_terms_and_predockers() -> None:
    usage = analyze_runtime_usage(
        {
            "evaluator": {
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                    "AF3": {"_type": "alphafold3"},
                    "Unused": {"_type": "rhofold"},
                }
            },
            "mutator": {
                "mutator_terms": {
                    "ligand": {"_type": "LigandMPNN", "predocker": ["used_vina"]},
                    "na": {"_type": "NAMPNN", "predocker": ["unused_dock"]},
                }
            },
            "predocker": {
                "predocker_terms": {
                    "used_vina": {"_type": "autodock_vina"},
                    "unused_dock": {"_type": "RAPiDock"},
                }
            },
            "stages": {
                "default": {"evaluator": "Protenix", "mutators": ["ligand"]},
                "gate": {"transition_consistency_gate": {"evaluator": "AF3"}},
            },
        }
    )

    assert usage.evaluator_types == {"protenix", "alphafold3"}
    assert usage.mutator_types == {"LigandMPNN"}
    assert usage.used_predocker_aliases == ["used_vina"]
    assert usage.predocker_types == {"autodock_vina"}


def test_runtime_usage_includes_structure_initializer_sequence_designer() -> None:
    usage = analyze_runtime_usage(
        {
            "initializer": {
                "mode": "structure",
                "sequence_designer": "init_mpnn",
            },
            "mutator": {
                "mutator_terms": {
                    "mcmc": {"_type": "MCMC"},
                    "init_mpnn": {"_type": "LigandMPNN"},
                }
            },
            "stages": {"default": {"mutators": ["mcmc"]}},
        }
    )

    assert usage.mutator_types == {"MCMC", "LigandMPNN"}


def test_task_requirements_include_rfd3_initializer_runtime() -> None:
    req = task_requirements(
        {
            "initializer": {
                "mode": "rfd3",
                "sequence_designer": "init_mpnn",
            },
            "mutator": {
                "mutator_terms": {
                    "init_mpnn": {"_type": "LigandMPNN"},
                }
            },
            "stages": {},
        }
    )

    assert {
        "rfd3.repo_path",
        "rfd3.ckpt_path",
        "weight_path.rfd3",
        "subprocess_envs.rf3.python_bin",
        "weight_path.ligandmpnn",
    }.issubset(req["required_env_keys"])


def test_task_requirements_auto_enable_pyrosetta_for_explicit_pyrosetta_loss() -> None:
    req = task_requirements(
        {
            "enable_pyrosetta": False,
            "loss": {
                "loss_terms": {
                    "hotspot": {
                        "_type": "hotspot_dsasa",
                        "select_expression": "B_1:9",
                    }
                }
            },
        }
    )

    assert "pyrosetta" in req["required_imports"]
    assert "__CONFIG_ERROR__enable_pyrosetta_false_but_required__" not in req[
        "required_imports"
    ]


def test_task_requirements_detect_design_screening_pyrosetta_loss() -> None:
    req = task_requirements(
        {
            "loss": {
                "loss_terms": {
                    "sap": {"_type": "sap_score"},
                }
            },
        }
    )

    assert "pyrosetta" in req["required_imports"]


def test_task_requirements_detect_rosetta_minimize_hook() -> None:
    req = task_requirements(
        {
            "post_eval_hooks": {
                "rosetta_minimize": {
                    "enabled": True,
                    "mode": "repack",
                }
            },
        }
    )

    assert "pyrosetta" in req["required_imports"]


def test_run_repeats_does_not_resolve_unused_env_interpolations(monkeypatch) -> None:
    run_mod = importlib.import_module("run")
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "repeats": 1,
            "mutator": {
                "LigandMPNN": {
                    "checkpoint_ligand_mpnn": "/tmp/ligand.pt",
                },
                "NAMPNN": {
                    "python_bin": "${oc.env:GH_NAMPNN_PYTHON,???}",
                    "repo_path": "${oc.env:GH_NAMPNN_REPO,???}",
                },
            },
        }
    )

    called = []
    monkeypatch.setattr(run_mod, "install_handlers", lambda: None)
    monkeypatch.setattr(run_mod, "stop_requested", lambda: False)
    monkeypatch.setattr(run_mod, "_run_pipeline", lambda run_cfg: called.append(run_cfg))

    run_mod.run_repeats(cfg)

    assert len(called) == 1


def test_base_and_env_configs_expose_nampnn_runner_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_text = (repo_root / "config" / "base.yaml").read_text(encoding="utf-8")
    env_text = (repo_root / "config" / "env.yaml").read_text(encoding="utf-8")
    env_container_text = (repo_root / "config" / "env_container.yaml").read_text(
        encoding="utf-8"
    )
    third_party_text = (
        repo_root / "config" / "third_party_envs.yaml"
    ).read_text(encoding="utf-8")

    assert "python_bin: ${subprocess_envs.nampnn.python_bin}" in base_text
    assert "repo_path: ${subprocess_envs.nampnn.repo_path}" in base_text

    assert "nampnn:" in env_text
    assert "GH_NAMPNN_PYTHON" in env_text
    assert "GH_NAMPNN_REPO" in env_text
    assert "dssp_path: ${oc.env:GH_DSSP,./vendor/mkdssp}" in env_text

    assert "nampnn:" in env_container_text
    assert "GH_NAMPNN_PYTHON" in env_container_text
    assert "GH_NAMPNN_REPO" in env_container_text
    assert "GH_MEEKO_PYTHONPATH" in env_container_text

    assert "subprocess_envs.nampnn.python_bin" in third_party_text
    assert "subprocess_envs.nampnn.repo_path" in third_party_text
    assert "subprocess_envs.meeko.pythonpath" in third_party_text


def test_validate_path_value_resolves_relative_to_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / "config" / "env" / "env_local.yaml"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path = tmp_path / "bin" / "tool.sh"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)

    ok, reason = cli._validate_path_value(
        "subprocess_envs.meeko.python_bin",
        "../../bin/tool.sh",
        env_path=env_path,
    )

    assert ok is True
    assert reason == "ok"


def test_classify_env_key_kind_treats_protenix_constraint_weight_as_file() -> None:
    assert cli._classify_env_key_kind("weight_path.protenix_constraint") == "file"


def test_task_requirements_include_nampnn_default_checkpoint_repo_file() -> None:
    req = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "na_mut": {
                        "_type": "nampnn",
                    }
                }
            }
        }
    )

    assert not any(
        path.endswith("vendor/na_mpnn/models/design_model/s_19137.pt")
        for path in req["required_repo_files"]
    )
    assert {
        "base_env_key": "subprocess_envs.nampnn.repo_path",
        "path": "models/design_model/s_19137.pt",
        "label": "nampnn_checkpoint",
    } in req["required_env_repo_files"]


def test_task_requirements_include_nampnn_explicit_checkpoint_repo_file() -> None:
    req = task_requirements(
        {
            "mutator": {
                "NAMPNN": {
                    "checkpoint": "checkpoints/custom.pt",
                },
                "mutator_terms": {
                    "na_mut": {
                        "_type": "nampnn",
                    }
                },
            }
        }
    )

    assert {
        "base_env_key": "subprocess_envs.nampnn.repo_path",
        "path": "checkpoints/custom.pt",
        "label": "nampnn_checkpoint",
    } in req["required_env_repo_files"]


def test_doctor_checks_nampnn_checkpoint_under_configured_repo_path(tmp_path: Path) -> None:
    repo = tmp_path / "external_na_mpnn"
    checkpoint = repo / "models" / "design_model" / "s_19137.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    env_path = tmp_path / "config" / "env.yaml"
    env_path.parent.mkdir()
    req = {
        "required_env_repo_files": [
            {
                "base_env_key": "subprocess_envs.nampnn.repo_path",
                "path": "models/design_model/s_19137.pt",
                "label": "nampnn_checkpoint",
            }
        ]
    }
    env_obj = {
        "subprocess_envs": {
            "nampnn": {
                "repo_path": str(repo),
            }
        }
    }

    items = cli._doctor_env_repo_file_items(req, env_obj, env_path=env_path)

    assert items == [
        cli.DoctorItem(
            key="repo_file::nampnn_checkpoint",
            value=str(checkpoint),
            exists=True,
            kind="file",
        )
    ]


def test_doctor_env_value_reads_process_env_for_uppercase_path(monkeypatch, tmp_path: Path) -> None:
    cutlass = tmp_path / "cutlass"
    cutlass.mkdir()
    monkeypatch.setenv("CUTLASS_PATH", str(cutlass))

    assert cli._get_doctor_env_value({}, "CUTLASS_PATH") == str(cutlass)
