from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import (
    Protomer,
    SegmentManager,
    register_files,
)
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.utils import StandardScores
from astrolabe.structure_start import (
    HotspotGateConfig,
    ReferenceGateResult,
    ReferenceTmScoreConfig,
    StructureStartConfig,
    StructureStartReference,
    StructureStartState,
    build_structure_start_state,
    evaluate_structure_start_gate,
    reference_gate_score_context,
    select_structure_start_candidate,
)


def _write_ala_pdb(path: Path, chain_lengths: dict[str, int]) -> Path:
    lines: list[str] = []
    serial = 1
    x = 0.0
    for chain_id, length in chain_lengths.items():
        for res_i in range(1, int(length) + 1):
            x += 1.0
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _init_parser_runtime(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("out_dir", tmp_path)


def _scores(name: str, path: Path | None = None) -> StandardScores:
    return StandardScores(
        names=[name],
        plddt={name: [1.0]},
        complex_paths={name: path or Path(f"{name.lower()}.pdb")},
    )


def test_structure_backed_segments_support_colon_ranges_and_fixed_prefix(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"A": 4})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})

    protomer = Protomer("A", "src:A:1-2 +src:A:3-4", "protein")

    assert protomer.sequence == "AAAA"
    np.testing.assert_array_equal(
        protomer.mutable_mask,
        np.asarray([True, True, False, False]),
    )


def test_structure_backed_segments_reject_non_protein_protomers(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"A": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})

    with pytest.raises(NotImplementedError, match="protein protomers only"):
        Protomer("R", "src:A", "rna")


def test_structure_start_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        StructureStartConfig.from_raw(
            {
                "references": {
                    "AB": {
                        "structure_alias": "src",
                        "chain_map": {"A": "A", "B": "B"},
                    }
                },
                "unexpected": True,
            }
        )


def test_structure_start_config_requires_references() -> None:
    with pytest.raises(ValueError, match="references must be a non-empty mapping"):
        StructureStartConfig.from_raw({"mode": "repredict_gate"})


def test_structure_start_rejects_chain_map_outside_oligo(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"A": 2, "B": 2, "C": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    protomers = {
        "A": Protomer("A", "src:A", "protein"),
        "B": Protomer("B", "+src:B", "protein"),
    }
    global_config.init_const("protomers", protomers)

    with pytest.raises(ValueError, match="not present in the oligo"):
        build_structure_start_state(
            {
                "references": {
                    "AB": {
                        "structure_alias": "src",
                        "chain_map": {"A": "A", "B": "B", "C": "C"},
                    }
                }
            }
        )


def test_structure_start_builds_normalized_reference_manifest(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 2, "Y": 3})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    protomers = {
        "A": Protomer("A", "src:X", "protein"),
        "B": Protomer("B", "+src:Y", "protein"),
    }
    global_config.init_const("protomers", protomers)

    state = build_structure_start_state(
        {
            "mode": "repredict_gate",
            "references": {
                "AB": {
                    "structure_alias": "src",
                    "chain_map": {"X": "A", "Y": "B"},
                }
            },
            "reference_tmscore": {"enabled": True, "threshold": 0.85},
        }
    )

    assert state is not None
    assert state.reference_scores.names == ["AB"]
    snapshot = state.reference_scores.complex_paths["AB"]
    assert snapshot.exists()
    manifest = state.manifest_path.read_text(encoding="utf-8")
    assert '"ordered_input_chains": [' in manifest
    assert '"X"' in manifest
    assert '"Y"' in manifest


def test_structure_start_rejects_mismatched_sequence_by_default(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    global_config.init_const("protomers", {"A": Protomer("A", "GG", "protein")})

    with pytest.raises(ValueError, match="reference sequence mismatch"):
        build_structure_start_state(
            {
                "references": {
                    "A": {
                        "structure_alias": "src",
                        "chain_map": {"X": "A"},
                    }
                }
            }
        )


def test_structure_start_length_only_allows_structure_reference_sequence_mismatch(
    tmp_path: Path,
) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    global_config.init_const("protomers", {"A": Protomer("A", "GG", "protein")})

    state = build_structure_start_state(
        {
            "sequence_policy": "length_only",
            "references": {
                "A": {
                    "structure_alias": "src",
                    "chain_map": {"X": "A"},
                }
            },
        }
    )

    assert state is not None
    manifest = state.manifest_path.read_text(encoding="utf-8")
    assert '"sequence_policy": "length_only"' in manifest
    assert state.reference_scores.names == ["A"]


def test_structure_start_orders_repeated_protomer_chain_map(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 2, "Y": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    global_config.init_const("protomers", {"A": Protomer("A", "src:X", "protein")})

    state = build_structure_start_state(
        {
            "references": {
                "AA": {
                    "structure_alias": "src",
                    "chain_map": {"X": "A", "Y": "A"},
                }
            }
        }
    )

    manifest = state.manifest_path.read_text(encoding="utf-8")
    assert '"ordered_input_chains": [' in manifest
    assert '"X"' in manifest
    assert '"Y"' in manifest


def test_reference_gate_adds_context_and_per_oligo_tmscore(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"A": 2, "B": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    protomers = {
        "A": Protomer("A", "src:A", "protein"),
        "B": Protomer("B", "+src:B", "protein"),
    }
    global_config.init_const("protomers", protomers)
    state = build_structure_start_state(
        {
            "references": {
                "AB": {
                    "structure_alias": "src",
                    "chain_map": {"A": "A", "B": "B"},
                }
            },
            "reference_tmscore": {"enabled": True, "threshold": 0.85},
        }
    )
    global_config.init_const("structure_start", state)

    result = evaluate_structure_start_gate(
        _scores("AB", tmp_path / "candidate.pdb"),
        timing="pre_accept",
        compare_fn=lambda **kwargs: 0.8,
    )

    assert result is not None
    assert result.gate_active is True
    assert result.passed is False
    assert result.per_oligo_tmscore == {"AB": 0.8}
    context = reference_gate_score_context(result)
    assert context["reference_tmscore"] == 0.8
    assert context["reference_gate_passed"] == 0.0
    assert context["AB"]["reference_tmscore"] == 0.8


def test_reference_gate_rejects_even_when_weighted_score_improves(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(test_reference_gate_rejects_even_when_weighted_score_improves, "_applied", True),
            clear_mutations=lambda: setattr(test_reference_gate_rejects_even_when_weighted_score_improves, "_cleared", True),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="default"))
    global_config.init_const("use_metropolis_acceptance", False)
    test_reference_gate_rejects_even_when_weighted_score_improves._applied = False
    test_reference_gate_rejects_even_when_weighted_score_improves._cleared = False

    decisions = Decisions.create(
        _scores("AB"),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (10.0, {}),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (0.0, {}),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
    )

    _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        1,
        _scores("AB"),
        loss,
        {
            "reference_gate_result": ReferenceGateResult(
                gate_active=True,
                score=0.4,
                threshold=0.85,
                passed=False,
                comparisons={"AB~AB": 0.4, "min~AB": 0.4},
                per_oligo_tmscore={"AB": 0.4},
            )
        },
    )

    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert test_reference_gate_rejects_even_when_weighted_score_improves._applied is False
    assert test_reference_gate_rejects_even_when_weighted_score_improves._cleared is True
    assert decisions.row_history[-1]["reference_gate_passed"] == 0.0


def test_acceptance_gate_rejects_even_when_weighted_score_improves(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(test_acceptance_gate_rejects_even_when_weighted_score_improves, "_applied", True),
            clear_mutations=lambda: setattr(test_acceptance_gate_rejects_even_when_weighted_score_improves, "_cleared", True),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="default"))
    global_config.init_const("use_metropolis_acceptance", False)
    test_acceptance_gate_rejects_even_when_weighted_score_improves._applied = False
    test_acceptance_gate_rejects_even_when_weighted_score_improves._cleared = False

    decisions = Decisions.create(
        _scores("AB"),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (10.0, {}),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (0.0, {"target_ligand_in_target_pocket": 0.7}),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: scores_detail["target_ligand_in_target_pocket"] >= 0.8,
    )

    _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        1,
        _scores("AB"),
        loss,
        {},
    )

    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert test_acceptance_gate_rejects_even_when_weighted_score_improves._applied is False
    assert test_acceptance_gate_rejects_even_when_weighted_score_improves._cleared is True
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 0.0


def test_acceptance_gate_can_use_previous_score_deltas(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(
                test_acceptance_gate_can_use_previous_score_deltas,
                "_applied",
                True,
            ),
            clear_mutations=lambda: setattr(
                test_acceptance_gate_can_use_previous_score_deltas,
                "_cleared",
                True,
            ),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="complex"))
    global_config.init_const("use_metropolis_acceptance", False)
    test_acceptance_gate_can_use_previous_score_deltas._applied = False
    test_acceptance_gate_can_use_previous_score_deltas._cleared = False

    decisions = Decisions.create(
        _scores("AB"),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (
            1.0,
            {"iptm": 0.70, "cross_contact_lis": 0.40},
        ),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            1.1,
            {"iptm": 0.60, "cross_contact_lis": 0.45},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: (
            scores_detail["previous_iptm"] < 0.5
            or scores_detail["delta_iptm"] >= -0.02
        ),
    )

    _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        1,
        _scores("AB"),
        loss,
        {},
    )

    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert decisions.scores_detail[1]["previous_iptm"] == pytest.approx(0.70)
    assert decisions.scores_detail[1]["delta_iptm"] == pytest.approx(-0.10)
    assert decisions.scores_detail[1]["previous_weighted_score"] == pytest.approx(1.0)
    assert decisions.scores_detail[1]["delta_weighted_score"] == pytest.approx(0.1)
    assert decisions.scores_detail[1]["stage_entry"] == 0.0
    assert test_acceptance_gate_can_use_previous_score_deltas._applied is False
    assert test_acceptance_gate_can_use_previous_score_deltas._cleared is True
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 0.0


def test_acceptance_gate_rejects_force_accept_stage_change(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(
                test_acceptance_gate_rejects_force_accept_stage_change,
                "_applied",
                True,
            ),
            clear_mutations=lambda: setattr(
                test_acceptance_gate_rejects_force_accept_stage_change,
                "_cleared",
                True,
            ),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="default"))
    global_config.init_const("use_metropolis_acceptance", False)
    test_acceptance_gate_rejects_force_accept_stage_change._applied = False
    test_acceptance_gate_rejects_force_accept_stage_change._cleared = False

    decisions = Decisions.create(
        _scores("AB"),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (0.0, {}),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            10.0,
            {"pair_hotspot_support": 0.0},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: scores_detail[
            "pair_hotspot_support"
        ]
        >= 0.14,
    )

    _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        1,
        _scores("AB"),
        loss,
        {},
        force_accept=True,
    )

    assert accepted is False
    assert accepted_eval is decisions.evaluation_results[0]
    assert test_acceptance_gate_rejects_force_accept_stage_change._applied is False
    assert test_acceptance_gate_rejects_force_accept_stage_change._cleared is True
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 0.0


def test_acceptance_gate_allows_marked_stage_entry_force_accept(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("initial_temperature", 0.05)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {})
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(
            apply_mutations=lambda: setattr(
                test_acceptance_gate_allows_marked_stage_entry_force_accept,
                "_applied",
                True,
            ),
            clear_mutations=lambda: setattr(
                test_acceptance_gate_allows_marked_stage_entry_force_accept,
                "_cleared",
                True,
            ),
        ),
    )
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="complex"))
    global_config.init_const("use_metropolis_acceptance", False)
    test_acceptance_gate_allows_marked_stage_entry_force_accept._applied = False
    test_acceptance_gate_allows_marked_stage_entry_force_accept._cleared = False

    decisions = Decisions.create(
        _scores("AB"),
        runtime_config={},
        score_func=lambda eval_result, runtime_config: (
            0.0,
            {"iptm": 0.0, "cross_contact_lis": 0.0},
        ),
    )
    loss = SimpleNamespace(
        compute_loss=lambda eval_result, runtime_config: (
            10.0,
            {"iptm": 0.1, "cross_contact_lis": 0.0},
        ),
        is_early_stopping=lambda scores_detail: False,
        is_soft_accepted=lambda scores_detail: False,
        passes_acceptance_gate=lambda scores_detail: scores_detail["stage_entry"]
        > 0,
    )

    _early_stop, accepted_eval, accepted = decisions.make_mutation_decision(
        1,
        _scores("AB"),
        loss,
        {},
        force_accept=True,
        stage_entry=1.0,
    )

    assert accepted is True
    assert accepted_eval is decisions.evaluation_results[1]
    assert decisions.scores_detail[1]["stage_entry"] == 1.0
    assert test_acceptance_gate_allows_marked_stage_entry_force_accept._applied is True
    assert test_acceptance_gate_allows_marked_stage_entry_force_accept._cleared is False
    assert decisions.row_history[-1]["acceptance_gate_passed"] == 1.0


def test_soft_select_chooses_candidate_with_best_reference_similarity(tmp_path: Path) -> None:
    global_config.reset()
    reference_scores = _scores("AB", tmp_path / "reference.pdb")
    state = StructureStartState(
        config=StructureStartConfig(
            mode="soft_select",
            gate_timing=("pre_accept",),
            aggregate="min",
            references={
                "AB": StructureStartReference(
                    oligo="AB",
                    structure_alias="src",
                    chain_map={"A": "A", "B": "B"},
                )
            },
            reference_tmscore=ReferenceTmScoreConfig(enabled=True, threshold=0.85),
            hotspot_gate=HotspotGateConfig(enabled=False),
        ),
        reference_scores=reference_scores,
        manifest_path=tmp_path / "manifest.json",
    )
    global_config.init_const("structure_start", state)

    bad = _scores("AB", tmp_path / "bad.pdb")
    good = _scores("AB", tmp_path / "good.pdb")
    wrapper = _scores("AB", tmp_path / "wrapper.pdb")
    wrapper.candidate_scores = [bad, good]

    selected, result = select_structure_start_candidate(
        wrapper,
        timing="pre_accept",
        compare_fn=lambda current_structure, **kwargs: (
            0.9 if Path(current_structure).name == "good.pdb" else 0.3
        ),
    )

    assert selected is good
    assert result is not None
    assert result.gate_active is False
    assert result.passed is True
    assert result.candidate_count == 2
    assert result.selected_candidate_index == 1


def test_soft_select_reports_threshold_failure_without_hard_gate(tmp_path: Path) -> None:
    global_config.reset()
    reference_scores = _scores("AB", tmp_path / "reference.pdb")
    state = StructureStartState(
        config=StructureStartConfig(
            mode="soft_select",
            gate_timing=("pre_accept",),
            aggregate="min",
            references={
                "AB": StructureStartReference(
                    oligo="AB",
                    structure_alias="src",
                    chain_map={"A": "A", "B": "B"},
                )
            },
            reference_tmscore=ReferenceTmScoreConfig(enabled=True, threshold=0.85),
            hotspot_gate=HotspotGateConfig(enabled=False),
        ),
        reference_scores=reference_scores,
        manifest_path=tmp_path / "manifest.json",
    )
    global_config.init_const("structure_start", state)

    selected, result = select_structure_start_candidate(
        _scores("AB", tmp_path / "candidate.pdb"),
        timing="pre_accept",
        compare_fn=lambda **kwargs: 0.8,
    )

    assert selected is not None
    assert result is not None
    assert result.gate_active is False
    assert result.passed is False
    context = reference_gate_score_context(result)
    assert context["reference_tmscore"] == 0.8
    assert context["reference_gate_passed"] == 0.0


def test_hotspot_gate_applies_reference_tolerance(
    tmp_path: Path, monkeypatch
) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"A": 2, "B": 2})
    _init_parser_runtime(tmp_path)
    register_files({"src": str(pdb)})
    protomers = {
        "A": Protomer("A", "src:A", "protein"),
        "B": Protomer("B", "+src:B", "protein"),
    }
    global_config.init_const("protomers", protomers)
    constructed = []

    def _fake_construct_complex_poses(scores):
        constructed.append(tuple(scores.names))
        return SimpleNamespace(poses={name: object() for name in scores.names})

    monkeypatch.setattr(
        "astrolabe.structure_start.ComplexPoses.construct_complex_poses",
        _fake_construct_complex_poses,
    )

    class _Loss:
        def score_context_for_expressions(self, scores, expressions):
            path = next(iter(scores.complex_paths.values()))
            value = 0.75 if Path(path).name.startswith("AB_reference") else 0.5
            return {"hot": value}

        def weighted_score(self, env, expression):
            return env[expression]

    state = build_structure_start_state(
        {
            "references": {
                "AB": {
                    "structure_alias": "src",
                    "chain_map": {"A": "A", "B": "B"},
                }
            },
            "hotspot_gate": {
                "enabled": True,
                "expression": "hot",
                "floor": 0.25,
                "reference_fraction": 0.8,
                "tolerance": 0.15,
            },
        },
        loss_computer=_Loss(),
    )
    global_config.init_const("structure_start", state)

    result = evaluate_structure_start_gate(
        _scores("AB", tmp_path / "candidate.pdb"),
        timing="pre_accept",
        loss_computer=_Loss(),
        compare_fn=lambda **kwargs: 0.9,
    )

    assert result is not None
    assert constructed == [("AB",)]
    assert result.hotspot_score == pytest.approx(0.5)
    assert state.reference_hotspot_score == pytest.approx(0.75)
    assert result.hotspot_threshold == pytest.approx(0.45)
    assert result.hotspot_passed is True
    assert result.passed is True
