from __future__ import annotations

import hashlib

import json

from pathlib import Path

from types import SimpleNamespace

import pytest

from astrolabe.core.folding_input import Input, ProteinChain, Template

from astrolabe.pre_eval_hooks.af3_antibody_msa_cache import (
    apply_af3_antibody_msa_cache_hook,
)

from omegaconf import OmegaConf, open_dict
from astrolabe.cli import _compose_hydra_config
from astrolabe.core import global_config
from astrolabe.losses.base import LossEngine


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = [
    "protein_binder_af3", "protein_binder_protenix", "protein_binder_sap",
    "protein_binder_af3_archived", "protein_binder_protenix_archived",
    "protein_binder_sap_archived", "idr_binder", "idr_binder_archived",
    "cre_binder_af3", "cre_binder_protenix2", "dual_target_af3",
    "dual_target_af3_archived", "dual_target_protenix2", "dual_target_esmfold2",
    "single_target_control", "single_target_control_archived", "antibody_framework",
    "cross_reactive_vhh", "cross_reactive_vhh_archived", "single_target_vhh_control",
    "single_target_vhh_control_archived", "solublempnn_control",
]


@pytest.mark.parametrize("name", PROTOCOLS)
def test_paper_protein_config_constructs_score_and_stage_contract(name):
    cfg = _compose_hydra_config(
        config_dir=ROOT / "config", config_name=f"paper/{name}", overrides=["env=env"]
    )
    global_config.reset()
    try:
        loss = OmegaConf.to_container(cfg.loss, resolve=False)
        if name == "single_target_control_archived":
            loss["loss_terms"]["target_hotspot"]["select_expression"] = "B_1:106,98,39"
        engine = LossEngine(loss)
        for stage in cfg.stages.values():
            assert stage.evaluator in cfg.evaluator.evaluator_terms
            assert set(stage.mutators) <= set(cfg.mutator.mutator_terms)
            engine.setup(stage.weighted_score_expression, stage.get("early_stopping"), stage.get("acceptance_threshold"))
    finally:
        global_config.reset()


@pytest.mark.parametrize("name", ["cre_binder_protenix2", "dual_target_protenix2"])
def test_paper_protenix_sampling_reaches_wrapper_owned_fields(name):
    cfg = _compose_hydra_config(
        config_dir=ROOT / "config", config_name=f"paper/{name}", overrides=["env=env"]
    )
    assert cfg.evaluator.protenix.model.N_cycle == 10
    assert cfg.evaluator.protenix.sample_diffusion.N_step == 200
    assert cfg.evaluator.protenix.sample_diffusion.N_sample == 1
    assert not {"N_cycle", "N_step", "N_sample"} & set(cfg.evaluator.protenix)
    assert cfg.evaluator.protenix.load_checkpoint_path is None


def _protocol(name):
    return _compose_hydra_config(
        config_dir=ROOT / "config", config_name=f"paper/{name}", overrides=["env=env"]
    )


def test_formal_benchmark_and_vhh_routes_follow_methods_after_inheritance():
    from astrolabe.antibody_framework_presets import get_antibody_framework_preset

    for name in ("protein_binder_af3", "protein_binder_protenix", "single_target_control", "single_target_vhh_control"):
        cfg = _protocol(name)
        assert list(cfg.stages.default.oligo) == ["A"]
        assert list(cfg.stages.complex.oligo) == ["AB"]
        assert list(cfg.stages.default.mutators) == ["dplm"]
        for alias in cfg.stages.complex.mutators:
            term = cfg.mutator.mutator_terms[alias]
            assert term._type == "LigandMPNN"
            assert term.get("model_type", cfg.mutator.LigandMPNN.model_type) == "ligand_mpnn"
            assert term.fallback_to_all_mutable_if_selector_empty is False
        assert "esm3" not in cfg.mutator.mutator_terms
    vhh = _protocol("single_target_vhh_control")
    assert vhh.initializer.mode == "x_mix" and vhh.initializer.x_fraction == 1.0
    assert vhh.seqs.A == get_antibody_framework_preset("vhh_7eow").binder_seq()
    assert vhh.pre_eval_hooks.af3_antibody_msa_cache.enabled
    assert vhh.steps == 100
    assert vhh.paper_protocol.representative == "last_accepted"

    cross = _protocol("cross_reactive_vhh")
    assert list(cross.stages.default.oligo) == ["AB", "AC"]
    alias = cross.stages.default.mutators[0]
    assert cross.mutator.mutator_terms[alias]._type == "LigandMPNN"
    assert cross.mutator.mutator_terms[alias].model_type == "protein_mpnn"
    assert cross.mutator.mutator_terms[alias].temperature == .15
    archived = _protocol("cross_reactive_vhh_archived")
    assert archived.mutator.mutator_terms[archived.stages.default.mutators[0]]._type == "FarApartTiedMultitargetLigandMPNN"
    for name in ("protein_binder_af3_archived", "protein_binder_protenix_archived"):
        assert "esm3" in _protocol(name).stages.complex.mutators


def test_methods_formula_images_match_effective_scores_and_target_lengths():
    context = {"A_length": 100, "binder_rog": .395 * 100 ** .6 + 7.257,
               "binder_loop_frac": .2, "iptm": .7, "ptm": .8, "plddt": .9,
               "interface_sc": .5, "binder_sap": 35, "binder_pair_tmscore": .8,
               "AB": SimpleNamespace(iptm=.8), "AC": SimpleNamespace(iptm=.6)}
    for name, stage, expected in [
        ("protein_binder_af3", "complex", 1.75),
        ("protein_binder_protenix", "complex", 1.75),
        ("single_target_vhh_control", "default", -.32),
        ("single_target_vhh_control", "complex", 1.75),
        ("protein_binder_sap", "complex", .93),
        ("cre_binder_af3", "complex", 2.02),
        ("dual_target_af3", "default", 2.075),
        ("cross_reactive_vhh", "default", 1.9),
    ]:
        cfg = _protocol(name)
        engine = LossEngine(OmegaConf.to_container(cfg.loss, resolve=False))
        assert engine.weighted_score(context, cfg.stages[stage].weighted_score_expression) == pytest.approx(expected)
    for name, length in [("dual_il4_il13", "100-120"), ("dual_pdl1_pdl2", "75-90"), ("dual_il12_il23", "75-90")]:
        cfg = _protocol("dual_target_af3")
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT / f"examples/paper/inputs/protein/{name}.yaml"))
        assert cfg.seqs.A == length
        assert cfg.seqs.B.startswith("+") and cfg.seqs.C.startswith("+")


def test_paper_data_bundle_preserves_declared_files_and_target_inputs():
    import tarfile
    import yaml

    manifest = json.loads((ROOT / "resources/paper_inputs_protein.json").read_text())
    with tarfile.open(ROOT / "resources" / manifest["archive"]) as archive:
        assert set(archive.getnames()) == {entry["path"] for entry in manifest["files"]}
        for entry in manifest["files"]:
            data = archive.extractfile(entry["path"]).read()
            assert len(data) == entry["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        for path in (ROOT / "examples/paper/inputs/protein").glob("*.yaml"):
            overlay = yaml.safe_load(path.read_text())
            for chain_id, ref in overlay.get("homologies", {}).items():
                if not isinstance(ref, str):
                    continue
                member, source_chain = ref.replace("${oc.env:ASTROLABE_PAPER_DATA}/", "").rsplit(":", 1)
                record = json.loads(archive.extractfile(member).read())
                protein = next(item["protein"] for item in record["sequences"] if item.get("protein", {}).get("id") == source_chain)
                assert overlay["seqs"][chain_id] == "+" + protein["sequence"]

ANTIBODY_SEQUENCE = "AAAAAA"

AB_TARGET_SEQUENCE = "M" * 129

AC_TARGET_SEQUENCE = "K" * 112

FRAMEWORK_MSA = f">query\n{ANTIBODY_SEQUENCE}\n>hit1\nAAAATA\n"

AB_TARGET_MSA = f">target\n{AB_TARGET_SEQUENCE}\n>target_hit\n{AB_TARGET_SEQUENCE}\n"

AC_TARGET_MSA = f">target\n{AC_TARGET_SEQUENCE}\n>target_hit\n{AC_TARGET_SEQUENCE}\n"

def _chain(
    chain_id: str,
    sequence: str,
    *,
    unpaired_msa: str = "",
    paired_msa: str = "",
    templates=None,
) -> ProteinChain:
    return ProteinChain(
        id=chain_id,
        sequence=sequence,
        ptms=[],
        unpaired_msa=unpaired_msa,
        paired_msa=paired_msa,
        templates=templates or [],
    )

def _input(name: str, chains: list[ProteinChain]) -> Input:
    return Input(name=name, chains=chains, rng_seeds=[0])

def _sha256_text(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

def _target_b_templates():
    return [Template(mmcif="target_b", query_to_template_map={0: 0})]

def _target_c_templates():
    return [Template(mmcif="target_c", query_to_template_map={0: 0})]

def _semantic_protomers(
    *,
    antibody_sequence: str = ANTIBODY_SEQUENCE,
    b_sequence: str = AB_TARGET_SEQUENCE,
    c_sequence: str = AC_TARGET_SEQUENCE,
    b_msa: str = AB_TARGET_MSA,
    c_msa: str = AC_TARGET_MSA,
    b_templates=None,
    c_templates=None,
) -> dict[str, SimpleNamespace]:
    b_templates = _target_b_templates() if b_templates is None else b_templates
    c_templates = _target_c_templates() if c_templates is None else c_templates
    return {
        "A": SimpleNamespace(
            sequence=antibody_sequence,
            mutated_sequence=antibody_sequence,
            homology={},
        ),
        "B": SimpleNamespace(
            sequence=b_sequence,
            mutated_sequence=b_sequence,
            homology={"unpairedMsa": b_msa, "templates": b_templates},
        ),
        "C": SimpleNamespace(
            sequence=c_sequence,
            mutated_sequence=c_sequence,
            homology={"unpairedMsa": c_msa, "templates": c_templates},
        ),
    }

def _ab_ac_inputs():
    ab_target = _chain(
        "B",
        AB_TARGET_SEQUENCE,
        unpaired_msa=AB_TARGET_MSA,
        paired_msa="target-paired",
        templates=_target_b_templates(),
    )
    ac_target = _chain(
        "B",
        AC_TARGET_SEQUENCE,
        unpaired_msa=AC_TARGET_MSA,
        paired_msa="target-paired-c",
        templates=_target_c_templates(),
    )
    return {
        "AB": _input(
            "AB",
            [
                _chain(
                    "A",
                    ANTIBODY_SEQUENCE,
                    unpaired_msa=FRAMEWORK_MSA,
                    paired_msa="seed-paired",
                    templates=[
                        Template(mmcif="data_template", query_to_template_map={0: 10, 1: 11})
                    ],
                ),
                ab_target,
            ],
        ),
        "AC": _input("AC", [_chain("A", ANTIBODY_SEQUENCE), ac_target]),
    }, ab_target, ac_target

def _cfg(cache_dir: Path) -> dict:
    return {
        "enabled": True,
        "chain_ids": ["A"],
        "max_residue_differences": 4,
        "cache_dir": str(cache_dir),
    }


def test_antibody_monomer_hook_preserves_framework_and_reuses_cache_in_complexes(tmp_path):
    initial = "AAAXXX"
    mutated = "AAACCC"
    seed_template = Template(mmcif="fixed_framework", query_to_template_map={0: 10, 1: 11})
    current_template = Template(mmcif="current_framework", query_to_template_map={0: 10, 2: 12})
    cfg = {**_cfg(tmp_path / "cache"), "preserve_existing_templates": True,
           "expected_template_count": 1, "minimum_unpaired_msa_sequences": 2}
    monomer = _input("A", [_chain("A", initial, unpaired_msa=">query\nAAAXXX\n>hit1\nAAAGGG\n",
                                   paired_msa="paired", templates=[seed_template])])
    updated, reports, cache = apply_af3_antibody_msa_cache_hook(
        {"A": monomer}, cfg, output_dir=tmp_path / "raw" / "alphafold3",
        semantic_protomers=_semantic_protomers(antibody_sequence=initial),
    )
    assert updated["A"].chains[0].sequence == initial
    assert updated["A"].chains[0].templates == [seed_template]
    assert updated["A"].chains[0].paired_msa == ""
    assert reports["A"][-1]["physical_to_semantic_mapping"] == {"A": "A"}
    complexes, _, _ = _ab_ac_inputs()
    complexes = {name: _input(name, [_chain("A", mutated, templates=[current_template]), value.chains[1]])
                 for name, value in complexes.items()}
    updated, reports, _ = apply_af3_antibody_msa_cache_hook(
        complexes, cfg, output_dir=tmp_path / "raw" / "alphafold3", cache=cache,
        semantic_protomers=_semantic_protomers(antibody_sequence=mutated),
    )
    for name in ("AB", "AC"):
        assert reports[name][0]["action"] == "hit"
        assert updated[name].chains[0].unpaired_msa == ">query\nAAACCC\n>hit1\nAAAGGG\n"
        assert updated[name].chains[0].templates == [current_template]
    assert reports["AB"][-1]["semantic_target_id"] == "B"
    assert reports["AC"][-1]["semantic_target_id"] == "C"

def test_ab_ac_inputs_share_chain_a_cache_and_keep_targets_by_object(tmp_path: Path):
    inputs, ab_target, ac_target = _ab_ac_inputs()

    updated, reports, _cache = apply_af3_antibody_msa_cache_hook(
        inputs,
        _cfg(tmp_path / "cache"),
        output_dir=tmp_path / "raw" / "alphafold3",
        semantic_protomers=_semantic_protomers(),
    )

    assert updated["AB"].chains[1].id == "B"
    assert updated["AC"].chains[1].id == "B"
    assert updated["AB"].chains[1].unpaired_msa == ab_target.unpaired_msa
    assert updated["AC"].chains[1].unpaired_msa == ac_target.unpaired_msa
    assert updated["AB"].chains[1].templates == ab_target.templates
    assert updated["AC"].chains[1].templates == ac_target.templates
    assert updated["AB"].chains[1].paired_msa == ""
    assert updated["AC"].chains[1].paired_msa == ""
    hit_chain = updated["AC"].chains[0]
    assert isinstance(hit_chain, ProteinChain)
    assert hit_chain.unpaired_msa == ">query\nAAAAAA\n>hit1\nAAAATA\n"
    assert hit_chain.paired_msa == ""
    assert hit_chain.templates
    assert reports["AB"][0]["action"] == "seed"
    assert reports["AC"][0]["action"] == "hit"
    assert reports["AC"][0]["distance"] == 0
    assert reports["AB"][-1]["semantic_state"] == "AB"
    assert reports["AB"][-1]["semantic_chain_ids"] == ["A", "B"]
    assert reports["AB"][-1]["physical_chain_ids"] == ["A", "B"]
    assert reports["AB"][-1]["physical_to_semantic_mapping"] == {"A": "A", "B": "B"}
    assert reports["AB"][-1]["target_physical_chain_id"] == "B"
    assert reports["AB"][-1]["target_length"] == 129
    assert reports["AB"][-1]["target_sequence_sha256"] == _sha256_text(AB_TARGET_SEQUENCE)
    assert reports["AC"][-1]["semantic_state"] == "AC"
    assert reports["AC"][-1]["semantic_chain_ids"] == ["A", "C"]
    assert reports["AC"][-1]["physical_chain_ids"] == ["A", "B"]
    assert reports["AC"][-1]["physical_to_semantic_mapping"] == {"A": "A", "B": "C"}
    assert reports["AC"][-1]["semantic_target_id"] == "C"
    assert reports["AC"][-1]["target_physical_chain_id"] == "B"
    assert reports["AC"][-1]["target_length"] == 112
    assert reports["AC"][-1]["target_sequence_sha256"] == _sha256_text(AC_TARGET_SEQUENCE)
    assert (
        reports["AB"][-1]["target_unpaired_msa_sha256"]
        != reports["AC"][-1]["target_unpaired_msa_sha256"]
    )

def test_strict_payload_contract_preserves_framework_templates(tmp_path: Path):
    inputs, _ab_target, _ac_target = _ab_ac_inputs()
    ac_framework_template = Template(
        mmcif="ac_framework", query_to_template_map={0: 20, 1: 21}
    )
    inputs["AC"] = _input(
        "AC",
        [
            _chain("A", ANTIBODY_SEQUENCE, templates=[ac_framework_template]),
            inputs["AC"].chains[1],
        ],
    )
    cfg = _cfg(tmp_path / "cache")
    cfg.update(
        {
            "clear_paired_msa": True,
            "preserve_existing_templates": True,
            "expected_template_count": 1,
            "minimum_unpaired_msa_sequences": 2,
        }
    )

    updated, _reports, _cache = apply_af3_antibody_msa_cache_hook(
        inputs,
        cfg,
        output_dir=tmp_path / "raw" / "alphafold3",
        semantic_protomers=_semantic_protomers(),
    )

    assert updated["AC"].chains[0].templates == [ac_framework_template]
    for input_obj in updated.values():
        for chain in input_obj.chains:
            assert chain.paired_msa == ""
            assert chain.unpaired_msa.count(">") >= 2
            assert len(chain.templates) == 1

def test_af3_antibody_msa_cache_rejects_semantic_target_payload_swap(
    tmp_path: Path,
):
    inputs, _ab_target, _ac_target = _ab_ac_inputs()
    inputs["AC"] = _input(
        "AC",
        [
            inputs["AC"].chains[0],
            _chain(
                "B",
                AC_TARGET_SEQUENCE,
                unpaired_msa=AB_TARGET_MSA,
                templates=_target_b_templates(),
            ),
        ],
    )

    with pytest.raises(ValueError, match="semantic target C unpaired MSA SHA256 mismatch"):
        apply_af3_antibody_msa_cache_hook(
            inputs,
            _cfg(tmp_path / "cache"),
            output_dir=tmp_path / "raw" / "alphafold3",
            semantic_protomers=_semantic_protomers(),
        )
