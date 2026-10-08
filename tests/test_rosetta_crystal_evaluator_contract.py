from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.rosetta_crystal import (
    _ONE_TO_THREE,
    RosettaCrystalEvaluator,
)
from astrolabe.losses.rosetta_crystal import rosetta_crystal_metric


class _Vector1:
    def __init__(self, values) -> None:
        self._values = [None, *values]

    def __len__(self) -> int:
        return len(self._values) - 1

    def __getitem__(self, index: int):
        return self._values[index]


class _FakePDBInfo:
    def __init__(self, chains: list[str]) -> None:
        self._chains = chains

    def chain(self, index: int) -> str:
        return self._chains[index - 1]


class _FakeResidue:
    def __init__(self, *, protein: bool) -> None:
        self._protein = bool(protein)

    def is_protein(self) -> bool:
        return self._protein


class _FakePose:
    def __init__(
        self,
        chains: list[str] | None = None,
        protein_mask: list[bool] | None = None,
    ) -> None:
        self._chains = chains or ["A"]
        self._protein_mask = protein_mask or [True for _ in self._chains]

    def total_residue(self) -> int:
        return len(self._chains)

    def pdb_info(self):
        return _FakePDBInfo(self._chains)

    def residue(self, index: int):
        return _FakeResidue(protein=self._protein_mask[index - 1])

    def dump_pdb(self, path: str) -> None:
        Path(path).write_text("ATOM      1  CA  ALA A   1\n", encoding="utf-8")


def setup_function() -> None:
    global_config.reset()


def teardown_function() -> None:
    global_config.reset()


def test_requires_primary_state(tmp_path) -> None:
    with pytest.raises(ValueError, match="primary_state"):
        RosettaCrystalEvaluator(
            states={"neutral": {"params": tmp_path / "x.params"}},
            primary_state="minus2",
            raw_model_results_dir=tmp_path,
        )


def test_target_sequences_are_chain_keyed(tmp_path) -> None:
    evaluator = RosettaCrystalEvaluator(
        states={"neutral": {"params": tmp_path / "x.params"}},
        primary_state="neutral",
        raw_model_results_dir=tmp_path,
    )
    input_obj = SimpleNamespace(
        protein_chains=[
            SimpleNamespace(id="A", sequence="ACD"),
            SimpleNamespace(id="B", sequence="EFG"),
        ]
    )
    assert evaluator.prepare_input({"AB": input_obj}, {}) == {
        "AB": {"A": "ACD", "B": "EFG"}
    }


def test_rosetta_metric_loss_reads_emitted_metric() -> None:
    scores = SimpleNamespace(
        names=["AB"],
        rosetta_metrics={"AB": {"robust_interaction_dde": -1.25}},
    )
    assert rosetta_crystal_metric(scores, "robust_interaction_dde") == {"AB": -1.25}


def test_rosetta_metric_loss_rejects_missing_metric() -> None:
    scores = SimpleNamespace(names=["AB"], rosetta_metrics={"AB": {}})
    with pytest.raises(ValueError, match="missing"):
        rosetta_crystal_metric(scores, "robust_interaction_dde")


def test_canonical_mutations_use_rosetta_residue_names() -> None:
    assert len(_ONE_TO_THREE) == 20
    assert _ONE_TO_THREE["D"] == "ASP"
    assert _ONE_TO_THREE["E"] == "GLU"


def test_per_residue_interface_dg_keeps_chain_protein_order_and_noninterface_nan(tmp_path) -> None:
    evaluator = RosettaCrystalEvaluator(
        states={"neutral": {"params": tmp_path / "x.params"}},
        primary_state="neutral",
        raw_model_results_dir=tmp_path,
    )
    pose = _FakePose(
        chains=["A", "A", "A", "B", "B", "B", "X"],
        protein_mask=[True, True, True, True, True, True, False],
    )
    data = SimpleNamespace(
        dG=_Vector1([-10.0, -2.0, 4.0, -1.0, 0.5, 2.0, 99.0]),
        interface_residues=_Vector1([False, True, True, False, True, False, True]),
    )

    by_chain = evaluator._chain_interface_dg(pose, data)

    assert np.allclose(
        by_chain["A"],
        np.array([np.nan, -2.0, 4.0], dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        by_chain["B"],
        np.array([np.nan, 0.5, np.nan], dtype=float),
        equal_nan=True,
    )


def test_packer_nloop_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="packer_nloop"):
        RosettaCrystalEvaluator(
            states={"neutral": {"params": tmp_path / "x.params"}},
            primary_state="neutral",
            raw_model_results_dir=tmp_path,
            packer_nloop=0,
        )


def test_state_aggregation_uses_worst_state_directions() -> None:
    metrics = RosettaCrystalEvaluator._aggregate_state_metrics(
        {
            "neutral": {
                "paired_interaction_dde": -2.0,
                "paired_total_de": -1.0,
                "interface_dg": -12.0,
                "shape_complementarity": 0.70,
                "contact_molecular_surface": 410.0,
                "interface_hbonds": 5.0,
                "buried_unsat_hbonds": 3.0,
                "packstat": 0.80,
                "interface_dsasa": 510.0,
                "ligand_heavy_rmsd": 0.10,
                "protein_ca_rmsd": 0.05,
                "interaction_energy_std": 0.2,
            },
            "minus2": {
                "paired_interaction_dde": 0.5,
                "paired_total_de": 1.0,
                "interface_dg": -8.0,
                "shape_complementarity": 0.60,
                "contact_molecular_surface": 360.0,
                "interface_hbonds": 3.0,
                "buried_unsat_hbonds": 6.0,
                "packstat": 0.75,
                "interface_dsasa": 480.0,
                "ligand_heavy_rmsd": 0.20,
                "protein_ca_rmsd": 0.07,
                "interaction_energy_std": 0.4,
            },
        }
    )
    assert metrics["robust_interaction_dde"] == 0.5
    assert metrics["robust_interface_dg"] == -8.0
    assert metrics["min_shape_complementarity"] == 0.60
    assert metrics["min_contact_molecular_surface"] == 360.0
    assert metrics["min_interface_hbonds"] == 3.0
    assert metrics["max_buried_unsat_hbonds"] == 6.0
    assert metrics["min_packstat"] == 0.75


def test_per_residue_interface_dg_state_aggregation_uses_max_with_nan_semantics() -> None:
    terms = RosettaCrystalEvaluator._aggregate_interface_dg_terms(
        {
            "neutral": {
                "A": np.array([np.nan, -1.0, -4.0], dtype=float),
                "B": np.array([0.0, np.nan], dtype=float),
            },
            "minus2": {
                "A": np.array([-2.0, np.nan, -3.0], dtype=float),
                "B": np.array([np.nan, np.nan], dtype=float),
            },
        }
    )

    assert np.allclose(
        terms["A"],
        np.array([-2.0, -1.0, -3.0], dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        terms["B"],
        np.array([0.0, np.nan], dtype=float),
        equal_nan=True,
    )


def _metric_row(interface_dg: float, shape_complementarity: float) -> dict[str, float]:
    return {
        "paired_interaction_dde": interface_dg / 10.0,
        "paired_total_de": interface_dg / 20.0,
        "interface_dg": interface_dg,
        "shape_complementarity": shape_complementarity,
        "contact_molecular_surface": 375.0,
        "interface_hbonds": 3.0,
        "buried_unsat_hbonds": 1.0,
        "packstat": 0.7,
        "interface_dsasa": 400.0,
        "ligand_heavy_rmsd": 0.2,
        "protein_ca_rmsd": 0.1,
        "interaction_energy_std": 0.3,
    }


def test_prepare_output_emits_rosetta_interface_dg_without_changing_scalar_metrics(
    tmp_path,
) -> None:
    global_config.init_const("out_dir", tmp_path)
    evaluator = RosettaCrystalEvaluator(
        states={
            "neutral": {"params": tmp_path / "neutral.params"},
            "minus2": {"params": tmp_path / "minus2.params"},
        },
        primary_state="minus2",
        raw_model_results_dir=tmp_path,
    )
    output = {
        "AB": {
            "neutral": {
                "pose": _FakePose(),
                "metrics": _metric_row(-12.0, 0.70),
                "per_chain_interface_dg": {
                    "A": np.array([np.nan, -1.0], dtype=float),
                    "B": np.array([2.0, np.nan], dtype=float),
                },
            },
            "minus2": {
                "pose": _FakePose(),
                "metrics": _metric_row(-8.0, 0.60),
                "per_chain_interface_dg": {
                    "A": np.array([-2.0, -0.5], dtype=float),
                    "B": np.array([1.0, 3.0], dtype=float),
                },
            },
        }
    }

    scores = evaluator.prepare_output(output, inputs={}, runtime_config={"step": 7})

    assert scores.complex_paths["AB"].name == "step_007_AB_rosetta_crystal.pdb"
    assert scores.rosetta_metrics["AB"]["robust_interface_dg"] == -8.0
    assert scores.rosetta_metrics["AB"]["min_shape_complementarity"] == 0.60
    assert scores.rosetta_metrics["AB"]["min_contact_molecular_surface"] == 375.0
    assert np.allclose(
        scores.per_proto_score_terms["A"]["rosetta_interface_dg"][0],
        np.array([-2.0, -0.5], dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        scores.per_proto_score_terms["B"]["rosetta_interface_dg"][0],
        np.array([2.0, 3.0], dtype=float),
        equal_nan=True,
    )
