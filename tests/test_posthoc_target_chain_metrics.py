from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrolabe.posthoc.derive import derive_af3


def test_derive_af3_extracts_target_chain_confidence_metrics(tmp_path: Path) -> None:
    step_dir = tmp_path / "raw_model_results" / "alphafold3" / "step_000_AB"
    step_dir.mkdir(parents=True)

    summary = {
        "ptm": 0.88,
        "iptm": 0.77,
        "chain_ptm": [0.91, 0.42],
        "chain_iptm": [0.11, 0.82],
        "chain_pair_iptm": [[0.0, 0.33], [0.33, 0.0]],
        "chain_pair_pae_min": [[0.0, 2.5], [2.5, 0.0]],
        "chain_pair_lis": [[0.0, 0.44], [0.44, 0.0]],
        "chain_pair_contact_lis": [[0.0, 0.55], [0.55, 0.0]],
    }
    conf = {
        "token_chain_ids": ["A", "A", "B"],
        "token_res_ids": [1, 2, 1],
        "pae": [[0.0, 1.0, 5.0], [1.0, 0.0, 6.0], [5.0, 6.0, 0.0]],
        "atom_chain_ids": ["A", "A", "A", "B", "B"],
        "atom_plddts": [90.0, 80.0, 70.0, 60.0, 50.0],
    }
    (step_dir / "step_000_AB_summary_confidences.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (step_dir / "step_000_AB_confidences.json").write_text(
        json.dumps(conf),
        encoding="utf-8",
    )

    metrics = derive_af3(tmp_path, binder_chain="B", target_chain="A")[(0, "AB")]

    assert metrics["binder_ptm"] == 0.42
    assert metrics["binder_iptm"] == 0.82
    assert metrics["target_chain_ptm"] == 0.91
    assert metrics["target_chain_iptm"] == 0.11
    assert metrics["target_chain_plddt_mean"] == 0.8
    assert metrics["chain_pair_lis_ab"] == 0.44
    assert metrics["chain_pair_contact_lis_ab"] == 0.55
    assert metrics["ipsae_ab"] == 1.0 / 26.0


def test_derive_af3_supports_binder_against_target_chain_group(
    tmp_path: Path,
) -> None:
    step_dir = tmp_path / "raw_model_results" / "alphafold3" / "step_000_ABCD"
    step_dir.mkdir(parents=True)
    summary = {
        "ptm": 0.88,
        "iptm": 0.77,
        "chain_ptm": [0.91, 0.42, 0.45, 0.48],
        "chain_iptm": [0.50, 0.20, 0.2666666667, 0.3333333333],
        "chain_pair_iptm": [
            [0.0, 0.30, 0.50, 0.70],
            [0.40, 0.0, 0.10, 0.10],
            [0.60, 0.10, 0.0, 0.10],
            [0.80, 0.10, 0.10, 0.0],
        ],
        "chain_pair_pae_min": [
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 0.0, 8.0, 8.0],
            [4.0, 8.0, 0.0, 8.0],
            [6.0, 8.0, 8.0, 0.0],
        ],
        "chain_pair_lis": [
            [0.0, 0.20, 0.40, 0.60],
            [0.20, 0.0, 0.10, 0.10],
            [0.40, 0.10, 0.0, 0.10],
            [0.60, 0.10, 0.10, 0.0],
        ],
        "chain_pair_contact_lis": [
            [0.0, 0.30, 0.50, 0.70],
            [0.30, 0.0, 0.10, 0.10],
            [0.50, 0.10, 0.0, 0.10],
            [0.70, 0.10, 0.10, 0.0],
        ],
    }
    conf = {
        "token_chain_ids": ["A", "B", "C", "D"],
        "pae": [
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 0.0, 8.0, 8.0],
            [4.0, 8.0, 0.0, 8.0],
            [6.0, 8.0, 8.0, 0.0],
        ],
        "atom_chain_ids": ["A", "B", "C", "D"],
        "atom_plddts": [90.0, 60.0, 70.0, 80.0],
    }
    (step_dir / "step_000_ABCD_summary_confidences.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (step_dir / "step_000_ABCD_confidences.json").write_text(
        json.dumps(conf),
        encoding="utf-8",
    )

    metrics = derive_af3(
        tmp_path,
        binder_chain="A",
        target_chain="others",
    )[(0, "ABCD")]

    assert metrics["binder_ptm"] == 0.91
    assert metrics["binder_iptm"] == 0.50
    assert metrics["target_chain_ptm"] == 0.45
    assert metrics["target_chain_iptm"] == pytest.approx(
        (0.20 + 0.2666666667 + 0.3333333333) / 3.0
    )
    assert metrics["target_chain_plddt_mean"] == 0.7
    assert metrics["halludesign_af3_iptm"] == pytest.approx(0.50)
    assert metrics["halludesign_af3_ipae"] == 4.0
    assert metrics["chain_pair_lis_ab"] == pytest.approx(0.4)
    assert metrics["chain_pair_contact_lis_ab"] == pytest.approx(0.5)

    binder_c_metrics = derive_af3(
        tmp_path,
        binder_chain="C",
        target_chain="others",
    )[(0, "ABCD")]
    assert binder_c_metrics["binder_ptm"] == 0.45
    assert binder_c_metrics["binder_iptm"] == pytest.approx(0.2666666667)
    assert binder_c_metrics["halludesign_af3_iptm"] == pytest.approx(
        (0.60 + 0.10 + 0.10) / 3.0
    )
    assert binder_c_metrics["target_chain_ptm"] == pytest.approx(
        (0.91 + 0.42 + 0.48) / 3.0
    )
