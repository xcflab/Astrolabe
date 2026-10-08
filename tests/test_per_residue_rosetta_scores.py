from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.score_snapshot import build_standard_score_snapshot
from astrolabe.evaluators.utils import ComplexPoses, StandardScores
from astrolabe.mutators.dplm import DPLMMutator
from astrolabe.per_residue_scores import (
    PerResidueScoreRegistry,
    attach_per_residue_scores,
    build_mutator_selector_score_context,
    combine_per_proto_mutator_scores,
    mutator_site_score,
    mutator_selector_score_delta_details,
)
from astrolabe.mutators.utils import prepare_exclude_sites


class _Protomer:
    type = ProtomerTypes.PROTEIN

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.mutable_mask = np.ones(self.length, dtype=bool)

    def aggregate_bool(self, mask):
        return np.asarray(mask, dtype=bool)


def setup_function() -> None:
    global_config.reset()


def teardown_function() -> None:
    global_config.reset()


def _try_pose():
    try:
        import pyrosetta  # type: ignore
    except ImportError:
        pytest.skip("PyRosetta not available")
    is_initialized = getattr(pyrosetta, "is_initialized", lambda: False)
    if not is_initialized():
        dalpha = pathlib.Path("astrolabe/assets/DAlphaBall.gcc").resolve()
        pyrosetta.init(f"-mute all -holes:dalphaball {dalpha}")
    return pyrosetta.pose_from_sequence("ACDEFGHIK")


def test_registry_lists_default_rosetta_metric_terms() -> None:
    available = set(PerResidueScoreRegistry.list_available())

    assert {"sap", "sasa", "buried_sasa", "buns", "holes", "packstat"} <= available


def test_attach_per_residue_scores_maps_registered_metric_to_protomer(monkeypatch) -> None:
    pose = _try_pose()
    global_config.init_const("protomers", {"A": _Protomer(pose.total_residue())})

    @PerResidueScoreRegistry.register("unit_test_linear")
    def _linear_metric(metric_pose, _context):
        return np.arange(1, metric_pose.total_residue() + 1, dtype=float)

    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.full(pose.total_residue(), 0.5, dtype=float)},
        complex_poses=ComplexPoses({"A": pose}),
    )

    out = attach_per_residue_scores(
        scores,
        {
            "enabled": True,
            "terms": {
                "linear": {"_type": "unit_test_linear"},
            },
        },
    )

    assert np.allclose(
        out.per_proto_score_terms["A"]["linear"][0],
        np.arange(1, pose.total_residue() + 1, dtype=float),
    )
    assert np.allclose(
        out.per_proto_score_terms["A"]["plddt"][0],
        np.full(pose.total_residue(), 0.5, dtype=float),
    )


def test_combine_per_proto_mutator_scores_uses_weights_and_fallback() -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.array([0.6, 0.5, 0.4], dtype=float)},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([0.6, 0.5, 0.4], dtype=float),
                "sap": np.array([1.0, 2.0, 3.0], dtype=float),
                "buns": np.array([0.0, 1.0, 0.0], dtype=float),
            }
        },
    )

    combined = combine_per_proto_mutator_scores(
        scores,
        {
            "per_residue_scores": {
                "mutator_score_terms": {
                    "plddt": 1.0,
                    "sap": {"weight": -0.1},
                    "buns": {"weight": -0.5},
                }
            }
        },
    )

    assert np.allclose(combined["A"], np.array([0.5, -0.2, 0.1], dtype=float))
    assert mutator_site_score(
        scores,
        proto_name="A",
        site_index=1,
        runtime_config={
            "per_residue_scores": {
                "mutator_score_terms": {
                    "plddt": 1.0,
                    "sap": {"weight": -0.1},
                    "buns": {"weight": -0.5},
                }
            }
        },
    ) == pytest.approx(-0.2)


def test_dplm_local_site_score_uses_configured_named_terms() -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.array([0.2, 0.2], dtype=float)},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([0.2, 0.2], dtype=float),
                "sap": np.array([1.0, 4.0], dtype=float),
            }
        },
    )

    score = DPLMMutator._per_proto_site_score(
        scores,
        proto_name="A",
        site_index=1,
        runtime_config={
            "per_residue_scores": {
                "mutator_score_terms": {
                    "plddt": 1.0,
                    "sap": -0.25,
                }
            }
        },
    )

    assert score == pytest.approx(-0.8)


def test_score_snapshot_exports_per_residue_term_means() -> None:
    scores = SimpleNamespace(
        names=["A"],
        plddt={"A": [0.5]},
        ptm={},
        iptm={},
        pde={},
        chain_pair_scores={},
        pae={},
        per_proto_score_terms={
            "A": {
                "sap": [np.array([1.0, 3.0], dtype=float)],
            }
        },
    )

    snapshot = build_standard_score_snapshot(scores)

    assert snapshot["per_residue_sap_mean"] == pytest.approx(2.0)
    assert snapshot["pA_per_residue_sap_mean"] == pytest.approx(2.0)


def test_mutator_selector_delta_uses_fixed_before_mutation_positions() -> None:
    before_scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.array([0.2, 0.8, 0.4], dtype=float)},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([0.2, 0.8, 0.4], dtype=float),
                "sap": np.array([3.0, 1.0, 5.0], dtype=float),
            }
        },
    )
    after_scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.array([0.7, 0.1, 0.6], dtype=float)},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([0.7, 0.1, 0.6], dtype=float),
                "sap": np.array([1.0, 9.0, 2.0], dtype=float),
            }
        },
    )
    runtime_config = {
        "per_residue_scores": {
            "mutator_score_terms": {
                "plddt": 1.0,
                "sap": -0.1,
            }
        }
    }
    context = build_mutator_selector_score_context(
        mutator_alias="LigandMPNN_pocket",
        selector_expression="unit_test_selector",
        evaluation_result=before_scores,
        selected_masks={"A": np.array([True, False, True])},
        runtime_config=runtime_config,
    )
    runtime_config["mutator_selector_score_contexts"] = [context]

    details = mutator_selector_score_delta_details(after_scores, runtime_config)

    assert details["mutator_selector_LigandMPNN_pocket_selected_count"] == 2.0
    assert details["mutator_selector_LigandMPNN_pocket_plddt_before_mean"] == pytest.approx(0.3)
    assert details["mutator_selector_LigandMPNN_pocket_plddt_after_mean"] == pytest.approx(0.65)
    assert details["mutator_selector_LigandMPNN_pocket_plddt_delta"] == pytest.approx(0.35)
    assert details["mutator_selector_LigandMPNN_pocket_sap_before_mean"] == pytest.approx(4.0)
    assert details["mutator_selector_LigandMPNN_pocket_sap_after_mean"] == pytest.approx(1.5)
    assert details["mutator_selector_LigandMPNN_pocket_sap_delta"] == pytest.approx(-2.5)
    assert details["mutator_selector_LigandMPNN_pocket_combined_delta"] == pytest.approx(0.6)


def test_prepare_exclude_sites_records_selector_score_context() -> None:
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.5]},
        complex_paths={"A": pathlib.Path("unused.pdb")},
        per_proto_scores={"A": np.array([0.2, 0.8, 0.4], dtype=float)},
        per_proto_score_terms={
            "A": {
                "plddt": np.array([0.2, 0.8, 0.4], dtype=float),
                "sap": np.array([3.0, 1.0, 5.0], dtype=float),
            }
        },
    )
    protomers = {"A": _Protomer(3)}
    runtime_config = {
        "record_mutator_selector_scores": True,
        "current_mutator_alias": "MCMC_local",
        "per_residue_scores": {
            "mutator_score_terms": {
                "plddt": 1.0,
                "sap": -0.1,
            }
        },
    }

    excluded = prepare_exclude_sites(
        "True",
        scores,
        protomers,
        runtime_config=runtime_config,
    )

    assert excluded == {"A": []}
    contexts = runtime_config["mutator_selector_score_contexts"]
    assert len(contexts) == 1
    assert contexts[0]["mutator"] == "MCMC_local"
    assert contexts[0]["positions"] == {"A": [0, 1, 2]}
    assert contexts[0]["before"]["terms"]["sap"] == pytest.approx(3.0)


def test_dummy_evaluator_can_emit_structure_for_rosetta_terms(tmp_path) -> None:
    pyrosetta = _try_pose().__class__.__module__.split(".")[0]
    assert pyrosetta == "pyrosetta"

    from astrolabe.evaluators.dummy import DummyEvaluator
    from astrolabe.core.folding_input import Input, ProteinChain

    evaluator = DummyEvaluator(write_structure=True, structure_dir=tmp_path)
    scores = evaluator.evaluate(
        {
            "A": Input(
                name="A",
                chains=[
                    ProteinChain(id="A", sequence="ACDE", ptms=[]),
                ],
                rng_seeds=[1],
            )
        },
        runtime_config={},
    )

    assert scores.complex_paths["A"].exists()
    assert "\nATOM" in scores.complex_paths["A"].read_text(encoding="utf-8")
