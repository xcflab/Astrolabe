from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.mutators.channel_bias import build_channel_aa_bias

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_channel_aa_bias_uses_default_channel_profiles() -> None:
    payload = build_channel_aa_bias(
        pore_lining_residues=["A10", "A11"],
        extracellular_mouth_residues=["A12"],
    )

    assert payload["mode"] == "multiplicative"
    assert payload["bias_AA"] == ""
    assert payload["omit_AA_per_residue"] == {
        "A10": "ACGMPW",
        "A11": "ACGMPW",
        "A12": "ACGMPW",
    }

    pore_bias = payload["bias_AA_per_residue"]["A10"]
    mouth_bias = payload["bias_AA_per_residue"]["A12"]

    assert pore_bias["S"] == pytest.approx(math.exp(1.0 / 0.3))
    assert pore_bias["T"] == pytest.approx(math.exp(1.0 / 0.3))
    assert pore_bias["N"] == pytest.approx(math.exp(1.0 / 0.3))
    assert pore_bias["Q"] == pytest.approx(math.exp(1.0 / 0.3))
    assert pore_bias["Y"] == pytest.approx(math.exp(-0.6 / 0.3))
    assert pore_bias["V"] == pytest.approx(math.exp(-0.3 / 0.3))
    assert pore_bias["I"] == pytest.approx(math.exp(-0.3 / 0.3))
    assert pore_bias["L"] == pytest.approx(math.exp(-0.3 / 0.3))
    assert pore_bias["F"] == pytest.approx(math.exp(-0.3 / 0.3))
    assert pore_bias["K"] == pytest.approx(math.exp(-0.6 / 0.3))
    assert pore_bias["R"] == pytest.approx(math.exp(-0.6 / 0.3))
    assert pore_bias["H"] == pytest.approx(math.exp(-0.6 / 0.3))
    assert pore_bias["D"] == pytest.approx(math.exp(-0.6 / 0.3))
    assert pore_bias["E"] == pytest.approx(math.exp(-0.6 / 0.3))

    assert mouth_bias == pore_bias


def test_build_channel_aa_bias_can_wrap_payload_for_ligandmpnn_multi_inputs() -> None:
    payload = build_channel_aa_bias(
        pore_lining_residues=["A10"],
        extracellular_mouth_residues=["A12"],
        pdb_path="/tmp/example_model.pdb",
        omit_aas_by_residue={"A10": "CG", "A12": ["P"]},
    )

    assert payload["bias_AA_per_residue_multi"] == {
        "/tmp/example_model.pdb": payload["bias_AA_per_residue"]
    }
    assert payload["omit_AA_per_residue_multi"] == {
        "/tmp/example_model.pdb": {"A10": "ACGMPW", "A12": "ACGMPW"}
    }


def test_base_config_routes_root_aa_bias_to_ligandmpnn_defaults() -> None:
    payload = build_channel_aa_bias(
        pore_lining_residues=["A10"],
        extracellular_mouth_residues=["A12"],
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "config" / "base.yaml"),
        OmegaConf.create({"aa_bias": payload}),
    )

    assert cfg.mutator.LigandMPNN.bias_mode == "multiplicative"
    assert cfg.mutator.LigandMPNN.bias_AA_per_residue["A10"]["S"] == pytest.approx(
        math.exp(1.0 / 0.3)
    )
    assert cfg.mutator.LigandMPNN.omit_AA_per_residue["A10"] == "ACGMPW"


def test_base_config_sets_channel_ligandmpnn_runtime_defaults() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "base.yaml")

    assert cfg.mutator.LigandMPNN.temperature == 0.3
    assert cfg.mutator.LigandMPNN.ligand_mpnn_use_side_chain_context == 1
