from __future__ import annotations

import pathlib

import numpy as np

from astrolabe.evaluators.utils import StandardScores
from astrolabe.decision_context import attach_reference_gate_details
from astrolabe.decision_maker import Decisions
from astrolabe.losses.base import LossEngine
from astrolabe.structure_start import ReferenceGateResult


def test_expression_terms_can_reference_prior_expression_terms() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "globin_rog": "0.395 * (A_length ** 0.6) + 7.257",
                "shape_penalty": "globin_rog + binder_rog",
            },
        },
        validate_config=False,
    )

    values = engine._evaluate_expression_terms(
        {"A_length": 100.0, "binder_rog": 8.0},
        engine._basic_operations(),
    )

    assert values["globin_rog"] > 0.0
    assert values["shape_penalty"] == values["globin_rog"] + 8.0


def test_eval_condition_handles_python_and_torch_bool_results() -> None:
    engine = LossEngine({"loss_terms": {}}, validate_config=False)

    assert engine.eval_condition({}, "True") is True
    assert engine.eval_condition({}, "np.bool_(True)") is True

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        engine.weighted_score = lambda _env, _expr: torch.tensor(True)  # type: ignore[method-assign]
        assert engine.eval_condition({}, "torch_bool") is True


def test_compute_loss_does_not_evaluate_unused_expression_terms() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "globin_rog": "0.395 * (A_length ** 0.6) + 7.257",
                "shape_penalty": "globin_rog + binder_rog",
                "pair_min_iptm": "min(AB.iptm, AC.iptm)",
            },
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="-shape_penalty",
        early_stopping=None,
        acceptance_threshold="binder_rog <= globin_rog + 3",
        validate_config=False,
    )
    engine._extract_scores_with_registry = lambda _eval_result: ({}, {"binder_rog": 8.0})  # type: ignore[method-assign]
    engine._default_expression_symbols = lambda: {"A_length": 100.0}  # type: ignore[method-assign]

    class _EvalResult:
        names = ["A"]
        complex_paths = {"A": pathlib.Path("pred.cif")}

    score, detail = engine.compute_loss(_EvalResult(), runtime_config=None)

    assert isinstance(score, float)
    assert "shape_penalty" in detail
    assert "globin_rog" in detail
    assert "pair_min_iptm" not in detail


def test_acceptance_gate_requires_terms_outside_weighted_score() -> None:
    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    engine.setup(
        weighted_score_expression="iptm + plddt",
        early_stopping=None,
        acceptance_threshold=None,
        acceptance_gate="target_ligand_in_target_pocket >= 0.8",
        validate_config=False,
    )
    engine._extract_scores_with_registry = lambda _eval_result: (  # type: ignore[method-assign]
        {},
        {
            "iptm": 0.7,
            "plddt": 0.8,
            "target_ligand_in_target_pocket": 0.75,
        },
    )

    class _EvalResult:
        names = ["AB"]
        complex_paths = {"AB": pathlib.Path("pred.cif")}

    score, detail = engine.compute_loss(_EvalResult(), runtime_config=None)

    assert score == 1.5
    assert detail["target_ligand_in_target_pocket"] == 0.75
    assert engine.passes_acceptance_gate(detail) is False
    detail["target_ligand_in_target_pocket"] = 0.8
    assert engine.passes_acceptance_gate(detail) is True


def test_compute_loss_evaluates_loss_terms_referenced_inside_expression_terms() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "pair_min_iptm": "min(AB.iptm, AC.iptm)",
            },
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="pair_min_iptm",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )

    eval_result = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.8], "AC": [0.7]},
        iptm={"AB": [0.4], "AC": [0.6]},
        complex_paths={
            "AB": pathlib.Path("ab.cif"),
            "AC": pathlib.Path("ac.cif"),
        },
    )

    score, detail = engine.compute_loss(eval_result, runtime_config=None)

    assert score == 0.4
    assert detail["pair_min_iptm"] == 0.4
    assert detail["AB"]["iptm"] == 0.4
    assert detail["AC"]["iptm"] == 0.6


def test_omni_refine_stability_terms_are_not_saturated_on_cre_scale() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "sap_per_residue": "sap_score / max(A_length, 1)",
                "binder_energy_per_residue": "binder_energy / max(A_length, 1)",
                "quality_confidence": (
                    "0.30*iptm + 0.20*ptm + 0.25*plddt + 0.25*binder_ptm"
                ),
                "interface_geometry": (
                    "0.30*interface_sc + 0.30*ipsae"
                    " + 0.15*max(0, min(1, (10 - ipae_min) / 10))"
                    " + 0.15*sigmoid((interface_dSASA - 900) / 250)"
                    " + 0.10*sigmoid((interface_hbonds - 2) / 2)"
                ),
                "bindcraft_energy_gate": (
                    "0.06*sigmoid((2200 - interface_dg) / 600)"
                    " + 0.04*sigmoid((1.50 - binder_energy_per_residue) / 0.80)"
                ),
                "stability_penalty": (
                    "0.14*sigmoid((sap_per_residue - 0.75) / 0.18)"
                    " + 0.08*sigmoid((surface_hydrophobicity - 0.42) / 0.10)"
                    " + 0.12*sigmoid((interface_unsat - 16) / 6)"
                    " + 0.06*sigmoid((binder_energy_per_residue - 1.50) / 0.80)"
                ),
                "reference_penalty": (
                    "0.40*(max(0, min(1, (reference_tmscore_threshold - reference_tmscore) / 0.10)) ** 2)"
                ),
            },
        },
        validate_config=False,
    )
    operations = engine._basic_operations()

    baseline = engine._evaluate_expression_terms(
        {
            "A_length": 160.0,
            "sap_score": 145.0,
            "surface_hydrophobicity": 0.48,
            "interface_unsat": 20.0,
            "interface_dg": 650.0,
            "binder_energy": 190.0,
            "iptm": 0.82,
            "ptm": 0.80,
            "plddt": 0.84,
            "binder_ptm": 0.78,
            "interface_sc": 0.60,
            "ipsae": 0.72,
            "ipae_min": 1.2,
            "interface_dSASA": 7500.0,
            "interface_hbonds": 12.0,
            "reference_tmscore": 0.92,
            "reference_tmscore_threshold": 0.87,
        },
        operations,
    )
    repaired = engine._evaluate_expression_terms(
        {
            "A_length": 160.0,
            "sap_score": 133.0,
            "surface_hydrophobicity": 0.47,
            "interface_unsat": 10.0,
            "interface_dg": 650.0,
            "binder_energy": 205.0,
            "iptm": 0.82,
            "ptm": 0.80,
            "plddt": 0.84,
            "binder_ptm": 0.78,
            "interface_sc": 0.60,
            "ipsae": 0.72,
            "ipae_min": 1.2,
            "interface_dSASA": 7500.0,
            "interface_hbonds": 12.0,
            "reference_tmscore": 0.80,
            "reference_tmscore_threshold": 0.87,
        },
        operations,
    )

    assert baseline["sap_per_residue"] == 0.90625
    assert repaired["stability_penalty"] < baseline["stability_penalty"] - 0.05
    assert abs(repaired["bindcraft_energy_gate"] - baseline["bindcraft_energy_gate"]) < 0.02
    assert 0.7 < baseline["quality_confidence"] < 0.9
    assert 0.7 < baseline["interface_geometry"] < 0.9
    assert 0.2 < baseline["stability_penalty"] < 0.4
    assert baseline["reference_penalty"] == 0.0
    assert repaired["reference_penalty"] > 0.15


def test_ligand_omni_refine_terms_are_balanced_on_poor_case_scale() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "A_globular_target": "0.395 * (A_length ** 0.6) + 7.257",
                "binder_rg_ratio": "A_single_chain_rog / max(2.2 * (A_length ** 0.38), 1)",
                "sap_per_residue": "sap_score / max(A_length, 1)",
                "confidence_quality": "0.58*iptm + 0.18*ptm + 0.24*plddt",
                "pocket_quality": "0.38*ligand_burial + 0.34*hbonds + 0.28*(1 - holes)",
                "protein_quality_penalty": (
                    "0.15*sigmoid((binder_rg_ratio - 1.20) / 0.08)"
                    " + 0.12*sigmoid((A_loop_frac - 0.32) / 0.08)"
                    " + 0.13*sigmoid((sap_per_residue - 0.75) / 0.18)"
                ),
                "pocket_quality_penalty": (
                    "0.12*sigmoid((0.72 - ligand_burial) / 0.12)"
                    " + 0.10*holes"
                    " + 0.08*sigmoid((0.30 - hbonds) / 0.10)"
                ),
            },
        },
        validate_config=False,
    )
    operations = engine._basic_operations()

    poor = engine._evaluate_expression_terms(
        {
            "A_length": 165.0,
            "A_single_chain_rog": 18.4,
            "A_loop_frac": 0.38,
            "sap_score": 142.0,
            "iptm": 0.45,
            "ptm": 0.62,
            "plddt": 0.73,
            "ligand_burial": 0.54,
            "hbonds": 0.20,
            "holes": 0.55,
        },
        operations,
    )
    repaired = engine._evaluate_expression_terms(
        {
            "A_length": 165.0,
            "A_single_chain_rog": 17.0,
            "A_loop_frac": 0.30,
            "sap_score": 124.0,
            "iptm": 0.50,
            "ptm": 0.64,
            "plddt": 0.75,
            "ligand_burial": 0.74,
            "hbonds": 0.42,
            "holes": 0.28,
        },
        operations,
    )

    assert 1.1 < poor["binder_rg_ratio"] < 1.4
    assert 0.50 < poor["confidence_quality"] < 0.60
    assert 0.3 < poor["pocket_quality"] < 0.45
    assert 0.15 < poor["protein_quality_penalty"] < 0.35
    assert 0.12 < poor["pocket_quality_penalty"] < 0.28
    assert repaired["pocket_quality"] > poor["pocket_quality"] + 0.15
    assert repaired["protein_quality_penalty"] < poor["protein_quality_penalty"]
    assert repaired["pocket_quality_penalty"] < poor["pocket_quality_penalty"] - 0.05


def test_compute_loss_exposes_standard_scores_pde() -> None:
    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    engine.setup(
        weighted_score_expression="pde",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )

    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.8]},
        pde={"AB": [np.array([[2.0, 4.0], [4.0, 2.0]], dtype=float)]},
        complex_paths={"AB": pathlib.Path("ab.cif")},
    )

    score, detail = engine.compute_loss(eval_result, runtime_config=None)

    assert score == 0.15
    assert detail["pde"] == 0.15
    assert detail["AB"]["pde"] == 0.15


def test_reference_gate_context_preserves_oligo_attribute_access() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "pair_min_iptm": "min(AB.iptm, AC.iptm)",
                "min_reference_tmscore": "min(AB.reference_tmscore, AC.reference_tmscore)",
            },
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="pair_min_iptm + min_reference_tmscore",
        early_stopping=None,
        acceptance_threshold=None,
        validate_config=False,
    )

    eval_result = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.8], "AC": [0.7]},
        iptm={"AB": [0.4], "AC": [0.6]},
        complex_paths={
            "AB": pathlib.Path("ab.cif"),
            "AC": pathlib.Path("ac.cif"),
        },
    )
    runtime_config = {
        "reference_gate_result": ReferenceGateResult(
            gate_active=True,
            score=0.88,
            threshold=0.85,
            passed=True,
            comparisons={},
            per_oligo_tmscore={"AB": 0.91, "AC": 0.88},
        )
    }

    score, detail = engine.compute_loss(eval_result, runtime_config=runtime_config)

    assert score == 1.28
    assert detail["pair_min_iptm"] == 0.4
    assert detail["min_reference_tmscore"] == 0.88
    assert detail["AB"]["iptm"] == 0.4
    assert detail["AB"]["reference_tmscore"] == 0.91
    assert detail["AC"]["iptm"] == 0.6
    assert detail["AC"]["reference_tmscore"] == 0.88


def test_reference_gate_details_preserve_early_stopping_dotted_oligos() -> None:
    engine = LossEngine(
        {
            "loss_terms": {},
            "expression_terms": {
                "pair_min_iptm": "min(AB.iptm, AC.iptm)",
                "min_reference_tmscore": "min(AB.reference_tmscore, AC.reference_tmscore)",
            },
        },
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression="pair_min_iptm + min_reference_tmscore",
        early_stopping="pair_min_iptm >= 0.4 and min_reference_tmscore >= 0.85",
        acceptance_threshold=None,
        validate_config=False,
    )

    eval_result = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.8], "AC": [0.7]},
        iptm={"AB": [0.4], "AC": [0.6]},
        complex_paths={
            "AB": pathlib.Path("ab.cif"),
            "AC": pathlib.Path("ac.cif"),
        },
    )
    runtime_config = {
        "reference_gate_result": ReferenceGateResult(
            gate_active=True,
            score=0.88,
            threshold=0.85,
            passed=True,
            comparisons={},
            per_oligo_tmscore={"AB": 0.91, "AC": 0.88},
        )
    }

    _score, detail = engine.compute_loss(eval_result, runtime_config=runtime_config)
    merged = attach_reference_gate_details(detail, runtime_config)
    merged = Decisions._merge_default_score_details(eval_result, merged)

    assert engine.is_early_stopping(merged) is True
    assert merged["AB"].iptm == 0.4
    assert merged["AB"].reference_tmscore == 0.91


def test_reference_gate_details_override_stale_reference_scores() -> None:
    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.8]},
        complex_paths={"AB": pathlib.Path("ab.cif")},
    )
    detail = {
        "reference_tmscore": 0.81,
        "reference_gate_passed": 1.0,
        "AB": {"reference_tmscore": 0.81},
    }
    runtime_config = {
        "reference_gate_result": ReferenceGateResult(
            gate_active=True,
            score=0.79,
            threshold=0.87,
            passed=False,
            comparisons={},
            per_oligo_tmscore={"AB": 0.79},
        )
    }

    merged = attach_reference_gate_details(detail, runtime_config)
    merged = Decisions._merge_default_score_details(eval_result, merged)

    assert merged["reference_tmscore"] == 0.79
    assert merged["reference_gate_passed"] == 0.0
    assert merged["reference_tmscore_threshold"] == 0.87
    assert merged["AB"]["reference_tmscore"] == 0.79
