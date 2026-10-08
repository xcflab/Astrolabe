from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from astrolabe.core import global_config
from astrolabe.losses.confidence import ipsae
from astrolabe.losses.confidence import pae_min
from astrolabe.losses.confidence import scaled_pae_min


class _EvalResult:
    def __init__(self, name: str, *, pae_mats: list[np.ndarray], structure_path: pathlib.Path):
        self.names = [name]
        self.pae = {name: pae_mats}
        self.complex_paths = {name: structure_path}
        self.chain_pair_scores = {}

    def __getitem__(self, item):
        return getattr(self, item)


class _TensorLike:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class TestConfidenceIpsae(unittest.TestCase):
    def test_ipsae_symmetric_max(self) -> None:
        # Order is A residues then B residues.
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  ALA B   2       1.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        pae = np.array(
            [
                [0.0, 0.0, 5.0, 15.0],
                [0.0, 0.0, 8.0, 9.0],
                [3.0, 4.0, 0.0, 0.0],
                [20.0, 20.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)

            out = ipsae(eval_result, chain_a="A", chain_b="B", pae_cutoff=10.0)["AB"]

        # With d0(min)=1.0 for small n0res, best directional score is from B->A:
        # mean(1/(1+(3/1)^2), 1/(1+(4/1)^2)) = (0.1 + 0.0588235)/2 = 0.0794117...
        self.assertAlmostEqual(out, 0.0794117647, places=8)

    def test_ipsae_multiple_matrices_mean(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        # 2 residues total: A(0), B(1).
        pae1 = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=float)
        pae2 = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy2.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae1, pae2], structure_path=path)

            out = ipsae(eval_result, chain_a="A", chain_b="B", pae_cutoff=10.0)["AB"]

        # pae=5 passes cutoff -> ptm=1/(1+25)=1/26; pae=10 does not pass (< cutoff) -> 0.
        self.assertAlmostEqual(out, (1.0 / 26.0) / 2.0, places=10)

    def test_ipsae_and_pae_min_fallback_to_af3_confidence_full_pae(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        # Simulate AF3 runtime `pae_xchain`: not a residue-level square matrix, so direct
        # `_stack_pae_matrices` fallback must fail and the implementation must load
        # full PAE from `*_confidences.json`.
        chain_summary_only = np.array([4.0, 5.0], dtype=float)
        confidence = {
            "token_chain_ids": ["A", "B"],
            "token_res_ids": [1, 1],
            "pae": [[0.0, 5.0], [3.0, 0.0]],
        }

        global_config.reset()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = pathlib.Path(tmpdir)
                path = root / "toy_af3.pdb"
                path.write_text(pdb, encoding="utf-8")
                step_dir = root / "raw" / "alphafold3" / "step_000_AB"
                step_dir.mkdir(parents=True)
                (step_dir / "step_000_AB_confidences.json").write_text(
                    json.dumps(confidence),
                    encoding="utf-8",
                )
                global_config.init_const("raw_model_results_dir", root / "raw")
                global_config.set_variable("step", 0)
                eval_result = _EvalResult(
                    "AB",
                    pae_mats=[chain_summary_only],
                    structure_path=path,
                )

                ipsae_out = ipsae(
                    eval_result,
                    chain_a="A",
                    chain_b="B",
                    pae_cutoff=10.0,
                )["AB"]
                pae_min_out = pae_min(eval_result, chain_a="A", chain_b="B")["AB"]

            self.assertAlmostEqual(ipsae_out, 0.1, places=8)
            self.assertAlmostEqual(pae_min_out, 3.0, places=8)
        finally:
            global_config.reset()

    def test_pae_min_prefers_chain_pair_scores(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        # Residue-level PAE would give min=3.0 (B->A).
        pae = np.array([[0.0, 5.0], [3.0, 0.0]], dtype=float)
        # Chain-pair summary matrix should be preferred; min(7, 9) = 7.
        chain_pair_pae_min = np.array([[0.0, 7.0], [9.0, 0.0]], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy3.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)
            eval_result.chain_pair_scores = {"AB": {"chain_pair_pae_min": [chain_pair_pae_min]}}

            out = pae_min(eval_result, chain_a="A", chain_b="B")["AB"]

        self.assertAlmostEqual(out, 7.0, places=8)

    def test_pae_min_fallbacks_to_residue_pae(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        # A has 2 residues, B has 1 residue. Cross-chain mins are:
        # AB submatrix: [8, 2] -> min 2; BA submatrix: [9, 7] -> min 7 => overall min 2.
        pae = np.array(
            [
                [0.0, 0.0, 8.0],
                [0.0, 0.0, 2.0],
                [9.0, 7.0, 0.0],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy4.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)

            out = pae_min(eval_result, chain_a="A", chain_b="B")["AB"]

        self.assertAlmostEqual(out, 2.0, places=8)

    def test_scaled_pae_min_linear_clipped(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        pae = np.array([[0.0, 2.5], [2.5, 0.0]], dtype=float)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy5.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)

            out = scaled_pae_min(eval_result, chain_a="A", chain_b="B", cutoff=10.0)["AB"]

        self.assertAlmostEqual(out, 0.75, places=8)

    def test_pae_min_fallback_accepts_tensor_like_pae_matrix(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        pae = _TensorLike([[0.0, 2.5], [2.5, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_tensorlike.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)

            out = pae_min(eval_result, chain_a="A", chain_b="B")["AB"]

        self.assertAlmostEqual(out, 2.5, places=8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_pae_min_fallback_accepts_cuda_tensor_pae_matrix(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       0.000   1.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        pae = torch.tensor([[0.0, 3.5], [3.5, 0.0]], device="cuda")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_cuda.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", pae_mats=[pae], structure_path=path)

            out = pae_min(eval_result, chain_a="A", chain_b="B")["AB"]

        self.assertAlmostEqual(out, 3.5, places=8)

    def test_pae_min_uses_chain_first_with_explicit_protomer_override(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1      10.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA C   1       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        # AAB oligomer:
        #  - protomer A -> chains A/B (indices 0/1)
        #  - protomer B -> chain C (index 2)
        # Pairwise mins:
        #  - A(0) vs C(2): min(7, 5) = 5
        #  - B(1) vs C(2): min(6, 4) = 4
        # => protomer-level min = 4
        chain_pair_pae_min = np.array(
            [
                [0.0, 8.0, 7.0],
                [9.0, 0.0, 6.0],
                [5.0, 4.0, 0.0],
            ],
            dtype=float,
        )
        dummy_pae = np.eye(3, dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy6.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AAB", pae_mats=[dummy_pae], structure_path=path)
            eval_result.chain_pair_scores = {"AAB": {"chain_pair_pae_min": [chain_pair_pae_min]}}

            out = pae_min(eval_result, chain_a="A", chain_b="B")["AAB"]
            out_chain = pae_min(eval_result, chain_a="chain:A", chain_b="chain:B")["AAB"]
            out_protomer = pae_min(
                eval_result, chain_a="protomer:A", chain_b="protomer:B"
            )["AAB"]
            scaled = scaled_pae_min(
                eval_result,
                chain_a="protomer:A",
                chain_b="protomer:B",
                cutoff=10.0,
            )["AAB"]

        self.assertAlmostEqual(out, 8.0, places=8)
        self.assertAlmostEqual(out_chain, 8.0, places=8)
        self.assertAlmostEqual(out_protomer, 4.0, places=8)
        self.assertAlmostEqual(scaled, 0.6, places=8)
