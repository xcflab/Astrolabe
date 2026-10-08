from __future__ import annotations

import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

from astrolabe.core import global_config
from astrolabe.losses import structure as structure_losses


class _EvalResult:
    def __init__(self, name: str = "AB"):
        self.names = [name]
        self.complex_paths = {name: pathlib.Path("pred.cif")}


def _write_complex_pdb(path: pathlib.Path, *, binder_shift: float = 0.0) -> None:
    records = [
        (1, "ALA", "A", 1, 0.0 + binder_shift, 0.0, 0.0),
        (2, "GLY", "A", 2, 1.0 + binder_shift, 0.0, 0.0),
        (3, "TYR", "B", 1, 9.0, 0.0, 0.0),
    ]
    lines = [
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        for serial, resname, chain, resseq, x, y, z in records
    ]
    lines.extend(["TER", "END", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


class TestUSalignTMscore(unittest.TestCase):
    def setUp(self) -> None:
        global_config.reset()

    def test_parse_usalign_output_prefers_reference_normalization(self) -> None:
        out = "\n".join(
            [
                "Aligned length= 10, RMSD=   1.23, Seq_ID=n_identical/n_aligned= 1.000",
                "TM-score= 0.11111 (normalized by length of Structure_1: L=10, d0=1.0)",
                "TM-score= 0.22222 (normalized by length of Structure_2: L=20, d0=1.0)",
            ]
        )
        rmsd, tm = structure_losses._parse_usalign_output(out, tm_normalize_by="reference")
        self.assertAlmostEqual(rmsd, 1.23, places=6)
        self.assertAlmostEqual(tm, 0.22222, places=6)

    def test_tmscore_can_resolve_template_file_alias(self) -> None:
        global_config.init_const(
            "named_files",
            {"initial": SimpleNamespace(file_path="/tmp/initial.cif")},
        )

        fake_stdout = "\n".join(
            [
                "Aligned length= 10, RMSD=   0.00, Seq_ID=n_identical/n_aligned= 1.000",
                "TM-score= 0.50000 (normalized by length of Structure_1: L=10, d0=1.0)",
                "TM-score= 0.90000 (normalized by length of Structure_2: L=10, d0=1.0)",
            ]
        )

        def _fake_run(cmd, check, capture_output, text):
            self.assertIn("-mm", cmd)
            self.assertIn("1", cmd)
            self.assertIn("/tmp/initial.cif", cmd)
            return SimpleNamespace(stdout=fake_stdout)

        with mock.patch.object(structure_losses.subprocess, "run", side_effect=_fake_run):
            scores = structure_losses.tmscore(_EvalResult(), template_file_alias="initial")

        self.assertAlmostEqual(scores["AB"], 0.9, places=6)

    def test_tmscore_can_compare_selected_query_and_reference_chains(self) -> None:
        tmp = pathlib.Path(self.id().replace("__main__.", "")).with_suffix("")
        workdir = pathlib.Path("/tmp") / f"target_chain_tmscore_{tmp.name}"
        workdir.mkdir(parents=True, exist_ok=True)
        pred = workdir / "pred.pdb"
        ref = workdir / "ref.pdb"
        _write_complex_pdb(pred, binder_shift=0.0)
        _write_complex_pdb(ref, binder_shift=1.0)

        class _ChainEvalResult:
            names = ["AB"]
            complex_paths = {"AB": pred}

        def _fake_usalign(query, reference, force_alignment=None, *, multimer=True, tm_normalize_by="reference"):
            self.assertTrue(multimer)
            self.assertIsNone(force_alignment)
            q_text = pathlib.Path(query).read_text(encoding="utf-8")
            r_text = pathlib.Path(reference).read_text(encoding="utf-8")
            self.assertNotIn(" A ", q_text)
            self.assertIn(" B ", q_text)
            self.assertIn(" A ", r_text)
            self.assertNotIn(" B ", r_text)
            return (0.4, 0.6)

        with mock.patch.object(structure_losses, "usalign_wrapper", side_effect=_fake_usalign):
            scores = structure_losses.tmscore(
                _ChainEvalResult(),
                template_files=[str(ref)],
                query_chain="B",
                reference_chain="A",
            )

        self.assertAlmostEqual(scores["AB"], 0.6, places=6)

    def test_binder_pair_tmscore_averages_query_and_reference_scores(self) -> None:
        tmp = pathlib.Path(self.id().replace("__main__.", "")).with_suffix("")
        workdir = pathlib.Path("/tmp") / f"binder_pair_tmscore_{tmp.name}"
        workdir.mkdir(parents=True, exist_ok=True)
        ab = workdir / "ab.pdb"
        ac = workdir / "ac.pdb"
        _write_complex_pdb(ab, binder_shift=0.0)
        _write_complex_pdb(ac, binder_shift=0.5)

        class _PairEvalResult:
            names = ["AB", "AC"]
            complex_paths = {"AB": ab, "AC": ac}

        def _fake_usalign(query, reference, force_alignment=None, *, multimer=True, tm_normalize_by="reference"):
            self.assertTrue(multimer)
            self.assertIsNone(force_alignment)
            q_text = pathlib.Path(query).read_text(encoding="utf-8")
            r_text = pathlib.Path(reference).read_text(encoding="utf-8")
            self.assertIn(" A ", q_text)
            self.assertNotIn(" B ", q_text)
            self.assertIn(" A ", r_text)
            self.assertNotIn(" B ", r_text)
            return (0.4, 0.8 if tm_normalize_by == "query" else 0.6)

        with mock.patch.object(structure_losses, "usalign_wrapper", side_effect=_fake_usalign):
            scores = structure_losses.binder_pair_tmscore(
                _PairEvalResult(),
                oligo_a="AB",
                oligo_b="AC",
                binder_chain="A",
            )

        self.assertAlmostEqual(scores["AB"], 0.7, places=6)
        self.assertAlmostEqual(scores["AC"], 0.7, places=6)

    def test_binder_pair_tmscore_returns_zero_when_requested_oligos_are_absent(self) -> None:
        class _MonomerEvalResult:
            names = ["A"]
            complex_paths = {"A": pathlib.Path("pred.cif")}

        scores = structure_losses.binder_pair_tmscore(
            _MonomerEvalResult(),
            oligo_a="AB",
            oligo_b="AC",
            binder_chain="A",
        )

        self.assertEqual(scores, {"A": 0.0})
