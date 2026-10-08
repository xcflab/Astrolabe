from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from astrolabe.evaluators.wrappers import alphafold3 as af3_wrapper


def test_aggregate_matrix_averages_duplicate_labels() -> None:
    matrix = np.array(
        [
            [1.0, 3.0, 10.0],
            [5.0, 7.0, 20.0],
            [30.0, 40.0, 9.0],
        ]
    )
    labels = np.array(["A_1", "A_1", "B_1"], dtype=object)

    aggregated = af3_wrapper.aggregate_matrix(matrix, labels)

    expected = np.array(
        [
            [4.0, 15.0],
            [35.0, 9.0],
        ]
    )
    np.testing.assert_allclose(aggregated, expected)


def _af3_result(label: str, ranking_score: float) -> SimpleNamespace:
    return SimpleNamespace(label=label, metadata={"ranking_score": ranking_score})


def _seed_result(seed: int, *results: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(seed=seed, inference_results=list(results))


def test_write_outputs_default_writes_only_best_canonical(
    tmp_path: Path, monkeypatch
) -> None:
    low = _af3_result("low", 0.2)
    high = _af3_result("high", 0.9)
    calls = []

    def fake_write_output(*, inference_result, output_dir, name, **_kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_model.cif").write_text(
            f"data_{inference_result.label}\n", encoding="utf-8"
        )
        calls.append(
            {
                "result": inference_result,
                "output_dir": output_dir,
                "name": name,
            }
        )

    monkeypatch.setattr(
        af3_wrapper.post_processing, "write_output", fake_write_output
    )

    af3_wrapper.write_outputs(
        all_inference_results=[
            _seed_result(101, low),
            _seed_result(202, high),
        ],
        output_dir=tmp_path,
        job_name="af3_job",
    )

    assert calls == [
        {
            "result": high,
            "output_dir": tmp_path,
            "name": "af3_job",
        }
    ]
    assert (
        tmp_path / "af3_job_model.cif"
    ).read_text(encoding="utf-8") == "data_high\n"
    assert not (tmp_path / "all_samples").exists()


def test_write_outputs_save_all_samples_preserves_best_canonical(
    tmp_path: Path, monkeypatch
) -> None:
    low = _af3_result("low", 0.2)
    high = _af3_result("high", 0.9)
    mid = _af3_result("mid", 0.5)
    calls = []

    def fake_write_output(*, inference_result, output_dir, name, **_kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_model.cif").write_text(
            f"data_{inference_result.label}\n", encoding="utf-8"
        )
        calls.append(
            {
                "result": inference_result,
                "output_dir": output_dir,
                "name": name,
            }
        )

    monkeypatch.setattr(
        af3_wrapper.post_processing, "write_output", fake_write_output
    )

    af3_wrapper.write_outputs(
        all_inference_results=[
            _seed_result(101, low, high),
            _seed_result(202, mid),
        ],
        output_dir=tmp_path,
        job_name="af3_job",
        save_all_samples=True,
    )

    sample_calls = [
        call for call in calls if "all_samples" in call["output_dir"].parts
    ]
    assert {
        call["output_dir"].relative_to(tmp_path).as_posix(): call["name"]
        for call in sample_calls
    } == {
        "all_samples/seed-101_sample-0": "af3_job_seed-101_sample-0",
        "all_samples/seed-101_sample-1": "af3_job_seed-101_sample-1",
        "all_samples/seed-202_sample-0": "af3_job_seed-202_sample-0",
    }
    assert (
        tmp_path
        / "all_samples"
        / "seed-101_sample-0"
        / "af3_job_seed-101_sample-0_model.cif"
    ).read_text(encoding="utf-8") == "data_low\n"
    assert (
        tmp_path
        / "all_samples"
        / "seed-101_sample-1"
        / "af3_job_seed-101_sample-1_model.cif"
    ).read_text(encoding="utf-8") == "data_high\n"
    assert (
        tmp_path
        / "all_samples"
        / "seed-202_sample-0"
        / "af3_job_seed-202_sample-0_model.cif"
    ).read_text(encoding="utf-8") == "data_mid\n"
    canonical_calls = [call for call in calls if call["output_dir"] == tmp_path]
    assert canonical_calls == [
        {
            "result": high,
            "output_dir": tmp_path,
            "name": "af3_job",
        }
    ]


class TestAlphafold3ScoresChainPtm(unittest.TestCase):
    def test_get_af3_scores_extracts_chain_ptm_and_chain_iptm(self) -> None:
        # `get_af3_scores` always calls `get_LIS_scores` (needs heavy AF3 objects),
        # so patch it to a trivial deterministic output.
        chain_pair_lis = np.zeros((2, 2), dtype=float)
        chain_pair_contact_lis = np.zeros((2, 2), dtype=float)

        predicted_structure = SimpleNamespace(
            # Two residues, two atoms each.
            res_auth_seq_id=np.array([0, 0, 1, 1], dtype=int),
            atom_b_factor=np.array([50.0, 50.0, 60.0, 60.0], dtype=float),
        )

        infer_rslt = SimpleNamespace(
            metadata={
                "ranking_score": 1.0,
                # AF3 stores these under `iptm_ichain`/`iptm_xchain`, and the wrapper
                # should expose them as `chain_ptm`/`chain_iptm`.
                "iptm_ichain": [0.1, 0.9],
                "iptm_xchain": [0.2, 0.8],
            },
            predicted_structure=predicted_structure,
        )

        results_for_seed = SimpleNamespace(inference_results=[infer_rslt])

        with mock.patch.object(
            af3_wrapper,
            "get_LIS_scores",
            return_value=(chain_pair_lis, chain_pair_contact_lis),
        ):
            scores = af3_wrapper.get_af3_scores([results_for_seed])

        self.assertIn("chain_ptm", scores["chain_pair_scores"])
        self.assertIn("chain_iptm", scores["chain_pair_scores"])
        self.assertEqual(scores["chain_pair_scores"]["chain_ptm"][0], [0.1, 0.9])
        self.assertEqual(scores["chain_pair_scores"]["chain_iptm"][0], [0.2, 0.8])
