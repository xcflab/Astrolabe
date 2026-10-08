"""Numerical search bookkeeping and real shared-segment contract checks.

These tests do not stand in for NA-MPNN or AlphaFold 3 inference.
"""

import math

import pytest

from astrolabe.rna_beam import (
    BeamSearchState,
    _expand_repeated_protomer,
    _normalize_bivalent_sequence_statement,
    _restore_sequence,
    _sequence_key,
    _validate_bivalent_rna,
    _validate_sequence_statements,
    _validate_shared_rna,
    score_bivalent_rna,
    validate_bivalent_objective_weights,
    proposal_seed,
    beam_expansion_order,
)


def test_initial_prediction_consumes_budget_and_duplicates_do_not():
    state = BeamSearchState(beam_width=2, max_evaluations=3)
    initial_id = state.reserve("ACGU", generation=0, parent_id=None)
    initial = state.complete(initial_id, score=0.70, structure_path="initial.cif")
    state.select([initial])
    assert initial_id == 0
    assert state.remaining == 2
    assert state.reserve("ACGU", generation=1, parent_id=initial_id) is None
    assert state.remaining == 2

    child_id = state.reserve("ACGA", generation=1, parent_id=initial_id)
    child = state.complete(child_id, score=0.90, structure_path="child.cif")
    other_id = state.reserve("ACGC", generation=1, parent_id=initial_id)
    other = state.complete(other_id, score=0.60, structure_path="other.cif")
    assert state.remaining == 0
    assert state.reserve("ACGG", generation=1, parent_id=initial_id) is None
    assert state.select([child, other]) == [child, initial]
    assert state.records[other_id]["entered_beam"] is False
    assert state.records[initial_id]["entered_beam"] is True
    assert state.records[initial_id]["in_final_beam"] is True


def test_elitist_selection_stable_ties_and_global_seen_after_eviction():
    state = BeamSearchState(beam_width=1, max_evaluations=4)
    first_id = state.reserve("A", generation=0, parent_id=None)
    first = state.complete(first_id, score=0.5, structure_path="0.cif")
    state.select([first])
    tied_id = state.reserve("C", generation=1, parent_id=first_id)
    tied = state.complete(tied_id, score=0.5, structure_path="1.cif")
    assert state.select([tied]) == [first]
    best_id = state.reserve("G", generation=2, parent_id=first_id)
    best = state.complete(best_id, score=0.8, structure_path="2.cif")
    assert state.select([best]) == [best]
    assert state.reserve("A", generation=3, parent_id=best_id) is None
    assert state.reserve("C", generation=3, parent_id=best_id) is None
    assert state.records[first_id]["entered_beam"] is True
    assert state.records[first_id]["in_final_beam"] is False


def test_failed_or_invalid_prediction_is_not_a_success_or_free_retry():
    state = BeamSearchState(max_evaluations=1)
    evaluation_id = state.reserve("ACGU", generation=0, parent_id=None)
    with pytest.raises(ValueError, match="Non-finite"):
        state.complete(evaluation_id, score=math.nan, structure_path="invalid.cif")
    assert state.remaining == 0
    assert state.records[evaluation_id]["status"] == "running"
    assert state.reserve("ACGU", generation=0, parent_id=None) is None
    assert state.beam == []


@pytest.mark.parametrize("key,value", [("beam_width", 0), ("max_evaluations", -1), ("beam_width", True)])
def test_invalid_budget_or_width(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        BeamSearchState(**{key: value})


def test_random_rna_initialization_is_rejected_before_parsing():
    for statement in ("87", "80-90@shared", "+87", "ACGU 83"):
        with pytest.raises(ValueError, match="supplied candidate"):
            _validate_sequence_statements({"A": statement})
    _validate_sequence_statements({"A": "ACGU@shared", "B": "shared"})


@pytest.mark.parametrize("chains", ["ABCD", "ABCDE", "AAAA", "AAAAA"])
def test_native_segments_share_queued_changes_and_restore_parent(chains):
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager

    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("named_files", {})
    sequence = "ACGU" * 21 + "ACG"
    try:
        for name in dict.fromkeys(chains):
            statement = sequence + "@shared" if name == "A" else "shared"
            global_config.protomers[name] = Protomer(name, statement, "rna")
        assert _validate_shared_rna(global_config.protomers, global_config.segments, chains) == chains
        physical_chains = _expand_repeated_protomer(chains)
        assert physical_chains == "ABCDE"[:len(chains)]
        assert _validate_shared_rna(global_config.protomers, global_config.segments) == physical_chains
        shared = global_config.segments["shared"]
        shared.add_mutations((0, "A", "U", 0.0, 0.0))
        assert {p.mutated_sequence for p in global_config.protomers.values()} == {"U" + sequence[1:]}
        global_config.segment_manager.apply_mutations()
        assert {p.sequence for p in global_config.protomers.values()} == {"U" + sequence[1:]}
        _restore_sequence(sequence, global_config.protomers["A"], global_config.segments)
        assert {p.sequence for p in global_config.protomers.values()} == {sequence}
        assert shared.mutations == []
        # Equal strings in independent segments do not implement synchronization.
        global_config.protomers["B"] = Protomer("B_independent", sequence, "rna")
        with pytest.raises(ValueError, match="same ordered segments"):
            _validate_shared_rna(global_config.protomers, global_config.segments)
    finally:
        global_config.reset()


def test_native_bivalent_repeats_share_initialization_mutation_and_restore():
    import numpy as np
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager

    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("named_files", {})
    try:
        np.random.seed(29)
        rna = Protomer("A", _normalize_bivalent_sequence_statement("20@repeat repeat"), "rna")
        target = "ACDEFGHIKLMNPQRSTVWY" * 4 + "ACDEFGHIKLMNPQRS"
        global_config.protomers.update(A=rna, B=Protomer("B", "+" + target, "protein"),
                                      C=Protomer("C", "+" + target, "protein"))
        assert _validate_bivalent_rna(global_config.protomers, global_config.segments, "ABC") == "ABC"
        original = _sequence_key(rna, global_config.segments)
        assert len(original) == 20
        assert rna.sequence == original * 2
        replacement = "A" if original[0] != "A" else "C"
        global_config.segments["repeat"].add_mutations((0, original[0], replacement, 0.0, 0.0))
        changed = replacement + original[1:]
        assert _sequence_key(rna, global_config.segments, mutated=True) == changed
        assert rna.mutated_sequence == changed * 2
        _restore_sequence(changed, rna, global_config.segments)
        assert rna.sequence == changed * 2
        _restore_sequence(original, rna, global_config.segments)
        assert rna.sequence == original * 2
        assert global_config.protomers["B"].sequence == target
        assert global_config.protomers["C"].sequence == target
    finally:
        global_config.reset()


def test_bivalent_objective_requires_explicit_weights_and_saturates_clash_penalty():
    weights = {"weak_interface": 2, "repeat_contacts": 3, "repeat_plddt": 4, "clashes": 0.25}
    metrics = {"dual_interface": 0.6, "repeat_cohesion": 0.5, "weak_arm_plddt": 0.4, "interchain_clashes": 1}
    assert score_bivalent_rna(metrics, weights) == pytest.approx(1.2 + 1.5 + 1.6 - 0.25)
    metrics["interchain_clashes"] = 100
    assert score_bivalent_rna(metrics, weights) == pytest.approx(1.2 + 1.5 + 1.6 - 0.25)
    for invalid in (None, {}, {**weights, "extra": 1}, {**weights, "clashes": -1}, {**weights, "clashes": math.nan}):
        with pytest.raises(ValueError):
            validate_bivalent_objective_weights(invalid)


def test_raw_metrics_are_retained_with_generic_beam_scores():
    state = BeamSearchState(max_evaluations=1)
    index = state.reserve("A" * 20, generation=0, parent_id=None)
    metrics = {"weak_interface": 0.4, "repeat_plddt": 0.8, "interchain_clashes": 2}
    candidate = state.complete(index, score=-0.6, structure_path="prediction.cif", metrics=metrics)
    state.select([candidate])
    assert candidate.score == -0.6
    assert candidate.metrics == metrics
    assert state.records[index]["score"] == -0.6
    assert state.records[index]["metrics"] == metrics


def test_archived_proposal_seed_and_balanced_partial_generation():
    assert proposal_seed(67101, 1) == 267850363
    assert proposal_seed(67101, 2) == 267850364
    assert beam_expansion_order(["parent0", "parent1", "parent2"], 4)[:5] == [
        ("parent0", 0), ("parent1", 0), ("parent2", 0), ("parent0", 1), ("parent1", 1),
    ]


def _compose_paper_recipe(name, input_name=None):
    from pathlib import Path
    from omegaconf import OmegaConf
    from astrolabe import cli

    root = Path(__file__).resolve().parents[1]
    cfg = cli._compose_hydra_config(
        config_dir=root / "config", config_name="paper/" + name, overrides=["env=env"],
    )
    if input_name:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "examples/paper/inputs/nucleic" / (input_name + ".yaml")))
    return cfg


@pytest.mark.parametrize("polymer,target", [("rna", "4kqy"), ("dna", "1tup")])
def test_methods_recipe_uses_dplm_initialization_and_conditional_haddock(polymer, target):
    name = "protein_binder_" + polymer
    cfg = _compose_paper_recipe(name, name + "_" + target)
    assert cfg.initializer.mode == "dplm"
    assert cfg.paper_protocol.initial_sequence_generator is None
    assert list(cfg.mutator.mutator_terms.NAMPNN.predocker) == ["haddock_na"]
    docking = cfg.predocker.predocker_terms.haddock_na
    assert docking._type == "haddock3_nucleic"
    assert docking.skip_if_contact is True and docking.require_contact is True
    assert docking.contact_cutoff == 4.0
    assert list(docking.target_chains) == (["B"] if polymer == "rna" else ["B", "C"])
    assert cfg.evaluator.protenix.model_name == "protenix_base_default_v1.0.0"
    assert cfg.stages.complex.weighted_score_expression == "iptm"
    assert cfg.paper_protocol.representative == "accepted_complex_max_iptm"
    assert cfg.steps == 20 and cfg.paper_protocol.planned_runs_per_target == 80
    alias = _compose_paper_recipe(name + "_haddock")
    assert alias.initializer.mode == "dplm"
    assert list(alias.mutator.mutator_terms.NAMPNN.predocker) == ["haddock_na"]


@pytest.mark.parametrize("polymer", ["rna", "dna"])
def test_archived_recipe_preserves_original_nucleic_binder_initialization(polymer):
    from omegaconf import OmegaConf

    cfg = _compose_paper_recipe("protein_binder_" + polymer + "_archived")
    assert cfg.initializer.mode == "random"
    assert cfg.paper_protocol.initial_sequence_generator == "python_random"
    assert cfg.paper_protocol.initial_sequence_alphabet == "ADEFGHIKLMNPQRSTVWY"
    assert list(cfg.paper_protocol.initial_sequence_length) == [60, 80]
    assert OmegaConf.select(cfg, "mutator.mutator_terms.NAMPNN.predocker") is None
    assert cfg.paper_protocol.variant == "archived"


def test_methods_recipe_bivalent_defaults_to_random_and_preserves_archived_starts():
    cfg = _compose_paper_recipe("bivalent_rna")
    assert cfg.beam_search.initialization == "random"
    assert cfg.seqs.A == "20@repeat repeat"
    assert cfg.seqs.B == cfg.seqs.C and len(cfg.seqs.B.lstrip("+")) == 96
    assert cfg.beam_search.max_evaluations == 200
    assert cfg.paper_protocol.planned_starts == 3
    archived = _compose_paper_recipe("bivalent_rna_archived")
    assert archived.beam_search.initialization == "archived"
    assert archived.seqs.A == "UCGGCAGCUGCCAUCUGACU@repeat repeat"
    assert [entry.source_step for entry in archived.beam_search.archived_starts] == [269, 613, 774]
    assert archived.beam_search.objective_weights == cfg.beam_search.objective_weights
    assert archived.evaluator.alphafold3.num_samples == cfg.evaluator.alphafold3.num_samples == 5


@pytest.mark.parametrize("polymer,steps", [("rna", 20), ("dna", 40)])
def test_methods_recipe_aptamer_budget_and_same_state_objective(polymer, steps):
    cfg = _compose_paper_recipe(polymer + "_aptamer", polymer + "_aptamer_1urn_u1a")
    assert cfg.seqs.A == "28-36" and cfg.seqs.B.startswith("+")
    assert cfg.initializer.mode == "random"
    assert cfg.steps == steps and cfg.paper_protocol.planned_runs_per_target == 80
    assert cfg.stages.default.mutation_num == "100%"
    assert cfg.stages.default.weighted_score_expression == "aptamer_iptm + aptamer_plddt + 0.4 * aptamer_ptm"
    assert cfg.loss.loss_terms.aptamer_plddt.binder_chain == "A"
    assert cfg.paper_protocol.representative == "max_objective_first_20"
    assert list(cfg.paper_protocol.analysis_step_range) == [0, 19]


@pytest.mark.parametrize("chains", [4, 5])
def test_methods_recipe_oligomer_keeps_supplied_synchronized_start_and_budget(chains):
    cfg = _compose_paper_recipe(f"rna_oligomer_{chains}")
    assert len(cfg.seqs.A) == 87 and set(cfg.seqs.A) <= set("ACGU")
    assert list(cfg.oligo) == ["A" * chains]
    assert cfg.beam_search.beam_width == 8
    assert cfg.beam_search.proposals_per_parent == 4
    assert cfg.beam_search.max_evaluations == 1000
