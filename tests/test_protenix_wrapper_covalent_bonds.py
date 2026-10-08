from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.wrappers.protenix import (
    _normalize_protenix_configs,
    from_af3_to_protenix,
)


def test_normalize_protenix_v2_preserves_model_specific_architecture() -> None:
    cfg = OmegaConf.create(
        {
            "model_name": "protenix-v2",
            "load_checkpoint_path": "/tmp/protenix-v2.pt",
            "model": {"N_cycle": 1, "pairformer": {"c_z": 128}},
            "sample_diffusion": {"N_sample": 1, "N_step": 5},
            "use_deepspeed_evo_attention": False,
            "data": {
                "ccd_components_file": "/tmp/components.cif",
                "ccd_components_rdkit_mol_file": "/tmp/components.pkl",
            },
        }
    )

    normalized = _normalize_protenix_configs(cfg)

    assert normalized.c_z == 256
    assert normalized.model.pairformer.c_z == 256
    assert normalized.model.diffusion_module.c_z == 256
    assert normalized.model.N_cycle == 1
    assert normalized.sample_diffusion.N_step == 5
    assert normalized.use_deepspeed_evo_attention is False


def test_normalize_protenix_v2_accepts_pathlike_runtime_dump_dir(
    tmp_path: Path,
) -> None:
    normalized = _normalize_protenix_configs(
        {
            "model_name": "protenix-v2",
            "dump_dir": tmp_path / "protenix",
            "load_checkpoint_path": tmp_path / "protenix-v2.pt",
        }
    )

    assert normalized.dump_dir == str(tmp_path / "protenix")
    assert normalized.load_checkpoint_path == str(tmp_path / "protenix-v2.pt")


def test_from_af3_to_protenix_emits_covalent_bonds(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const("binding_site", None)
    global_config.init_const("dynamic_binding_site", None)

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDEFGHIK",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=[(("A", 1, "N"), ("A", 9, "C"))],
        user_ccd=None,
    )
    data = from_af3_to_protenix(inp, "A")
    chain = data["sequences"][0]["proteinChain"]
    assert chain["id"] == ["A"]
    assert chain["unpairedMsaPath"].endswith(
        "/sequences/protenix_msa/A/1/non_pairing.a3m"
    )
    assert chain["pairedMsaPath"].endswith(
        "/sequences/protenix_msa/A/1/pairing.a3m"
    )
    assert data["covalent_bonds"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 1,
            "atom1": "N",
            "entity2": 1,
            "copy2": 1,
            "position2": 9,
            "atom2": "C",
        }
    ]


def test_from_af3_to_protenix_can_disable_binding_site_constraints(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const(
        "binding_site",
        {
            "AB": {
                "binder": "A",
                "target": "B:2",
                "mode": "pocket",
                "max_distance": 8.0,
            }
        },
    )
    global_config.init_const("dynamic_binding_site", None)

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDEFGHIK",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            ProteinChain(
                id="B",
                sequence="LMNPQRSTV",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
        ],
        name="AB",
        rng_seeds=[0],
        bonded_atom_pairs=[(("A", 1, "N"), ("B", 2, "C"))],
        user_ccd=None,
    )

    constrained = from_af3_to_protenix(inp, "AB")
    unconstrained = from_af3_to_protenix(inp, "AB", use_constraints=False)

    assert "constraint" in constrained
    assert "constraint" not in unconstrained
    assert unconstrained["covalent_bonds"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 1,
            "atom1": "N",
            "entity2": 2,
            "copy2": 1,
            "position2": 2,
            "atom2": "C",
        }
    ]


def test_from_af3_to_protenix_uses_dynamic_binding_site_when_static_missing(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const("binding_site", None)
    global_config.init_const(
        "dynamic_binding_site",
        {
            "AB": {
                "binder": "A:1,3",
                "target": "B:2",
                "mode": "both",
                "max_distance": 5.5,
                "min_distance": 1.0,
            }
        },
    )

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDEFGHIK",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            ProteinChain(
                id="B",
                sequence="LMNPQRSTV",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
        ],
        name="AB",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )

    data = from_af3_to_protenix(inp, "AB")

    assert data["constraint"]["pocket"] == {
        "binder_chain": {"entity": 1, "copy": 1},
        "contact_residues": [{"entity": 2, "copy": 1, "position": 2}],
        "max_distance": 5.5,
    }
    assert data["constraint"]["contact"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 1,
            "entity2": 2,
            "copy2": 1,
            "position2": 2,
            "max_distance": 5.5,
            "min_distance": 1.0,
        },
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 3,
            "entity2": 2,
            "copy2": 1,
            "position2": 2,
            "max_distance": 5.5,
            "min_distance": 1.0,
        },
    ]


def test_construct_evaluators_uses_constraint_checkpoint_for_raw_protenix_constraint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.evaluators.utils import construct_evaluators

    default_ckpt = tmp_path / "default.pt"
    constraint_ckpt = tmp_path / "constraint.pt"
    default_ckpt.write_text("default\n", encoding="utf-8")
    constraint_ckpt.write_text("constraint\n", encoding="utf-8")

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protenix_constraint",
        {
            "pocket_constraints": [],
            "contact_constraints": [],
            "contact_atom_constraints": [],
        },
    )
    global_config.init_const("binding_site", None)
    global_config.init_const("dynamic_binding_site_enabled", False)

    evaluator_conf = OmegaConf.create(
        {
            "evaluator_terms": {"Protenix": {"_type": "protenix"}},
            "protenix": {
                "load_checkpoint_path": str(default_ckpt),
                "model": {
                    "constraint_embedder": {
                        "pocket_embedder": {"enable": False},
                        "contact_embedder": {"enable": False},
                        "substructure_embedder": {"enable": False},
                        "contact_atom_embedder": {"enable": False},
                    }
                },
            },
            "weight_path": {"protenix_constraint": str(constraint_ckpt)},
        }
    )

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, conf):
            captured["load_checkpoint_path"] = OmegaConf.to_container(conf, resolve=True)[
                "load_checkpoint_path"
            ]

    monkeypatch.setattr("astrolabe.evaluators.wrappers.protenix.ProtenixRunner", _FakeRunner)
    monkeypatch.setattr(
        "astrolabe.evaluators.protenix.ProtenixEvaluator",
        lambda runner, **_: runner,
    )

    construct_evaluators(
        evaluator_conf,
        raw_model_results_dir=tmp_path / "raw_model_results",
        oligo_names=["AB"],
    )

    assert captured["load_checkpoint_path"] == str(constraint_ckpt)


def test_construct_evaluators_uses_constraint_checkpoint_for_dynamic_binding_site(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.evaluators.utils import construct_evaluators

    default_ckpt = tmp_path / "default.pt"
    constraint_ckpt = tmp_path / "constraint.pt"
    default_ckpt.write_text("default\n", encoding="utf-8")
    constraint_ckpt.write_text("constraint\n", encoding="utf-8")

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const("binding_site", None)
    global_config.init_const("dynamic_binding_site_enabled", False)

    evaluator_conf = OmegaConf.create(
        {
            "evaluator_terms": {
                "Protenix": {
                    "_type": "protenix",
                    "dynamic_binding_site": True,
                }
            },
            "protenix": {
                "load_checkpoint_path": str(default_ckpt),
                "model": {
                    "constraint_embedder": {
                        "pocket_embedder": {"enable": False},
                        "contact_embedder": {"enable": False},
                        "substructure_embedder": {"enable": False},
                        "contact_atom_embedder": {"enable": False},
                    }
                },
            },
            "weight_path": {"protenix_constraint": str(constraint_ckpt)},
        }
    )

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, conf):
            resolved = OmegaConf.to_container(conf, resolve=True)
            captured["load_checkpoint_path"] = resolved["load_checkpoint_path"]
            captured["dynamic_binding_site"] = resolved.get("dynamic_binding_site")
            captured["pocket_enabled"] = resolved["model"]["constraint_embedder"][
                "pocket_embedder"
            ]["enable"]

    monkeypatch.setattr("astrolabe.evaluators.wrappers.protenix.ProtenixRunner", _FakeRunner)
    monkeypatch.setattr(
        "astrolabe.evaluators.protenix.ProtenixEvaluator",
        lambda runner, **_: runner,
    )

    construct_evaluators(
        evaluator_conf,
        raw_model_results_dir=tmp_path / "raw_model_results",
        oligo_names=["AB"],
    )

    assert captured == {
        "load_checkpoint_path": str(constraint_ckpt),
        "dynamic_binding_site": None,
        "pocket_enabled": True,
    }


def test_construct_evaluators_use_constraints_false_keeps_base_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from astrolabe.evaluators.utils import construct_evaluators

    default_ckpt = tmp_path / "default.pt"
    constraint_ckpt = tmp_path / "constraint.pt"
    default_ckpt.write_text("default\n", encoding="utf-8")
    constraint_ckpt.write_text("constraint\n", encoding="utf-8")

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protenix_constraint", None)
    global_config.init_const(
        "binding_site",
        {
            "AB": {
                "binder": "A",
                "target": "B:2",
                "mode": "pocket",
                "max_distance": 8.0,
            }
        },
    )
    global_config.init_const("dynamic_binding_site_enabled", False)

    evaluator_conf = OmegaConf.create(
        {
            "evaluator_terms": {
                "ProtenixBaseNoConstraint": {
                    "_type": "protenix",
                    "use_constraints": False,
                }
            },
            "protenix": {
                "load_checkpoint_path": str(default_ckpt),
                "model": {
                    "constraint_embedder": {
                        "pocket_embedder": {"enable": False},
                        "contact_embedder": {"enable": False},
                        "substructure_embedder": {"enable": False},
                        "contact_atom_embedder": {"enable": False},
                    }
                },
            },
            "weight_path": {"protenix_constraint": str(constraint_ckpt)},
        }
    )

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, conf):
            captured["load_checkpoint_path"] = OmegaConf.to_container(conf, resolve=True)[
                "load_checkpoint_path"
            ]

    class _FakeEvaluator:
        def __init__(self, runner, *, use_constraints: bool = True):
            captured["use_constraints"] = use_constraints
            self.runner = runner

    monkeypatch.setattr("astrolabe.evaluators.wrappers.protenix.ProtenixRunner", _FakeRunner)
    monkeypatch.setattr("astrolabe.evaluators.protenix.ProtenixEvaluator", _FakeEvaluator)

    construct_evaluators(
        evaluator_conf,
        raw_model_results_dir=tmp_path / "raw_model_results",
        oligo_names=["AB"],
    )

    assert captured == {
        "load_checkpoint_path": str(default_ckpt),
        "use_constraints": False,
    }
