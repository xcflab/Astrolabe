from pathlib import Path
from types import SimpleNamespace

import csv

import numpy as np
import pytest
from omegaconf import OmegaConf

import run as run_mod
from astrolabe.cli import _compose_hydra_config
from astrolabe.core import global_config
from astrolabe.core.repeat_symmetry import build_internal_repeat_residue_groups, build_shared_core_residue_groups
from astrolabe.losses.sequence import sequence_region_identity


_CONFIG = Path(__file__).resolve().parents[1] / "config"


def _compose(name):
    return _compose_hydra_config(config_dir=_CONFIG, config_name=f"paper/{name}", overrides=["env=env"])


@pytest.mark.parametrize("symmetry", ["c3", "c4", "c5", "c6", "d2", "d3"])
def test_paper_rescue_control_changes_only_structural_intervention(symmetry):
    assisted = _compose(f"symmetry_{symmetry}")
    control = _compose(f"symmetry_{symmetry}_no_rescue")
    assert assisted.steps == control.steps == 50
    assert assisted.paper_protocol.planned_trajectories_per_protocol == 50
    assert assisted.stages.oligomer.evaluator == "alphafold2"
    count = int(symmetry[1:]) * (1 if symmetry.startswith("c") else 2)
    assert list(assisted.stages.oligomer.oligo) == ["A" * count]
    assert assisted.predocker.predocker_terms.symmetry.backend == "rpxdock"
    assert assisted.predocker.predocker_terms.symmetry.rpxdock_failure_fallback == "raise"
    assert assisted.stages.default.evaluator == ("alphafold2" if symmetry.startswith("c") else "Protenix")
    assert assisted.mutator.mutator_terms.dplm._type == "DPLM"
    if symmetry.startswith("c"):
        assert assisted.mutator.mutator_terms.mpnn_interface.model_type == "protein_mpnn"
        assert assisted.mutator.mutator_terms.mpnn_surface.model_type == "soluble_mpnn"
    else:
        assert assisted.mutator.mutator_terms.mpnn_dn.model_type == "soluble_mpnn"
        assert list(control.mutator.mutator_terms.mpnn_dn.predocker) == ["allala"]
    left = OmegaConf.to_container(assisted, resolve=False)
    right = OmegaConf.to_container(control, resolve=False)
    left.pop("paper_protocol"); right.pop("paper_protocol")
    left["predocker"]["predocker_terms"].pop("symmetry")
    for term in left["mutator"]["mutator_terms"].values():
        if "predocker" in term:
            term["predocker"] = [name for name in term["predocker"] if name != "symmetry"]
    assert left == right


def _setup(tmp_path, seqs):
    global_config.reset()
    args = SimpleNamespace(exclude_AA=[], initial_temperature=0.05, out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw", use_metropolis_acceptance=False,
        enable_pyrosetta=False, skip_complex_poses=True, protenix_constraint=None,
        binding_site=None, cyclization=None, ligand_resnames=None, seqs=seqs,
        types={proto: "protein" for proto in seqs}, homologies={proto: None for proto in seqs})
    run_mod.initialize_global_config(args)
    run_mod.setup_protomers(args)


@pytest.mark.parametrize("count", [3, 4, 5])
def test_tandem_config_uses_native_synchronized_repeat_and_paper_backends(tmp_path, count):
    cfg = _compose(f"tandem_repeat_{count}")
    assert cfg.steps == 100
    assert list(cfg.stages) == ["default", "refine"]
    assert cfg.stages.default.evaluator == "Alphafold2"
    assert list(cfg.stages.default.mutators) == ["ligand_mpnn_refine"]
    assert list(cfg.stages.refine.mutators) == ["ligand_mpnn_refine"]
    assert list(cfg.stages.default.mutator_weights) == [1.0]
    assert list(cfg.stages.refine.mutator_weights) == [1.0]
    assert cfg.mutator.mutator_terms.ligand_mpnn_refine.model_type == "protein_mpnn"
    assert cfg.paper_protocol.recipe_kind == "manuscript"
    assert cfg.paper_protocol.trajectory_analysis.reported_selected_trajectories == 80
    assert cfg.paper_protocol.trajectory_analysis.reported_cohorts.initial.selected == 43
    assert cfg.paper_protocol.trajectory_analysis.reported_cohorts.additional.selected == 37
    try:
        _setup(tmp_path, {"A": cfg.seqs.A})
        groups = build_internal_repeat_residue_groups("A")
        assert len(groups) == count
        unit_length = len(groups[0])
        assert 25 <= unit_length <= 35
        assert all(np.array_equal(group, np.arange(i * unit_length, (i + 1) * unit_length)) for i, group in enumerate(groups))
    finally:
        global_config.reset()


@pytest.mark.parametrize("count", [3, 4, 5])
def test_tandem_archived_recipe_retains_original_proposals_and_geometry(count):
    formal = _compose(f"tandem_repeat_{count}")
    archived = _compose(f"tandem_repeat_{count}_archived")
    assert archived.paper_protocol.recipe_kind == "archived"
    assert list(archived.stages.default.mutators) == ["mcmc_symmetry"]
    assert list(archived.stages.refine.mutators) == ["mcmc_loop_refine", "ligand_mpnn_refine"]
    assert list(archived.stages.refine.mutator_weights) == [0.25, 0.75]
    assert archived.paper_protocol.trajectory_analysis.selected_cohort == 43
    assert formal.seqs == archived.seqs
    assert formal.loss == archived.loss
    assert formal.initializer == archived.initializer
    for stage in formal.stages:
        assert formal.stages[stage].weighted_score_expression == archived.stages[stage].weighted_score_expression
        assert formal.stages[stage].mutation_num == archived.stages[stage].mutation_num
        assert formal.stages[stage].evaluator == archived.stages[stage].evaluator == "Alphafold2"


def test_pseudo_c3_recovered_inputs_and_shared_core_contract(tmp_path):
    cfg = _compose("pseudo_c3")
    assert cfg.steps == 40
    assert cfg.stages.default.evaluator == "Alphafold2"
    assert list(cfg.stages.default.oligo) == ["ABC"]
    assert list(cfg.stages.default.mutator_weights) == [1.0, 0.5, 0.5]
    assert cfg.mutator.mutator_terms.ligand_mpnn_sync.model_type == "protein_mpnn"
    assert cfg.mutator.mutator_terms.ligand_mpnn_patch.temperature == 0.5
    assert cfg.mutator.mutator_terms.mcmc_patch.max_mutations == 10
    assert cfg.loss.expression_terms.ABC_reference_tmscore_penalty == "100.0 * (ABC_reference_tmscore < 0.75)"
    inputs = _CONFIG.parent / "examples/paper/inputs/assemblies"
    with (inputs / "manifest.csv").open() as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 20
    seeds = [seed for row in manifest for seed in row["seeds"].split(";")]
    assert len(seeds) == len(set(seeds)) == 100
    for row in manifest:
        overlay = OmegaConf.load(inputs / row["input_yaml"])
        assert (inputs / overlay.paper_protocol.reference_structure).is_file()
        assert overlay.paper_protocol.reference_structure == row["reference_cif"]
        OmegaConf.set_struct(cfg, False)
        composed = OmegaConf.merge(cfg, overlay)
        try:
            _setup(tmp_path, OmegaConf.to_container(composed.seqs, resolve=True))
            groups = build_shared_core_residue_groups("ABC")
            assert set(groups) == {"A", "B", "C"}
            assert len(groups["A"]) > 0
            assert np.array_equal(groups["A"], groups["B"])
            assert np.array_equal(groups["A"], groups["C"])
            assert composed.homologies.A == composed.homologies.B == composed.homologies.C
            positions = composed.loss.loss_terms.ABC_patch_barcode_identity.positions
            assert sequence_region_identity(None, "A,B,C", positions) == 1.0
        finally:
            global_config.reset()
