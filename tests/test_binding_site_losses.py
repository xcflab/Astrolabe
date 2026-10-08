from __future__ import annotations

import pathlib
import tempfile
import unittest
import math
from pathlib import Path

from astrolabe.core import global_config
from astrolabe.losses.base import LossEngine
from astrolabe.losses.binding_site import (
    binder_target_min_dist,
    binding_site_allowed_min_dist,
    binding_site_framework_min_dist,
    dynamic_binding_site_allowed_min_dist,
)
from astrolabe.evaluators.utils import StandardScores


class _EvalResult:
    def __init__(self, name: str, structure_path: str):
        self.names = [name]
        self.complex_paths = {name: pathlib.Path(structure_path)}


class TestBindingSiteLosses(unittest.TestCase):
    def setUp(self) -> None:
        global_config.reset()

    def test_allowed_vs_framework_min_dist(self) -> None:
        global_config.init_const(
            "binding_site",
            {
                "binder": ["A:1-2"],
                "target": "B:258",
                "mode": "pocket",
                "max_distance": 10.0,
            },
        )
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       7.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  THR B 258       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", str(path))

            allowed = binding_site_allowed_min_dist(eval_result, use_ca=True)["AB"]
            framework = binding_site_framework_min_dist(eval_result, use_ca=True)["AB"]

            self.assertAlmostEqual(allowed, 5.0, places=4)
            self.assertAlmostEqual(framework, 2.0, places=4)

    def test_paratope_only_uses_target_whole_chain(self) -> None:
        global_config.init_const(
            "binding_site",
            {
                "binder": ["A:1-2"],
                "target": "B",
            },
        )

        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       7.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  THR B 258       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      5  CA  THR B 259      20.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_paratope_only.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", str(path))

            allowed = binding_site_allowed_min_dist(eval_result, use_ca=True)["AB"]
            framework = binding_site_framework_min_dist(eval_result, use_ca=True)["AB"]

            self.assertAlmostEqual(allowed, 5.0, places=4)
            self.assertAlmostEqual(framework, 2.0, places=4)

    def test_dynamic_binding_site_allowed_min_dist_returns_inf_before_publication(
        self,
    ) -> None:
        global_config.init_const("dynamic_binding_site", None)
        eval_result = _EvalResult("AB", "/tmp/not_needed_before_dynamic_binding_site.pdb")

        allowed = dynamic_binding_site_allowed_min_dist(eval_result, use_ca=True)["AB"]

        self.assertTrue(math.isinf(allowed))

    def test_dynamic_binding_site_allowed_min_dist_uses_runtime_spec(self) -> None:
        global_config.init_const(
            "dynamic_binding_site",
            {
                "AB": {
                    "binder": "A:1",
                    "target": "B:10",
                    "mode": "both",
                    "max_distance": 10.0,
                }
            },
        )
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  THR B  10       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_dynamic.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AB", str(path))

            allowed = dynamic_binding_site_allowed_min_dist(eval_result, use_ca=True)["AB"]

            self.assertAlmostEqual(allowed, 5.0, places=4)

    def test_dynamic_binding_site_loss_engine_cache_tracks_runtime_spec(self) -> None:
        global_config.init_const("dynamic_binding_site", None)
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  THR B  10       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_dynamic_cache.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = StandardScores(
                names=["AB"],
                plddt={"AB": [0.9]},
                complex_paths={"AB": path},
            )
            loss = LossEngine(
                {
                    "loss_terms": {
                        "dynamic_binding_site_allowed_min_dist": {
                            "_type": "dynamic_binding_site_allowed_min_dist",
                            "use_ca": True,
                        },
                    },
                },
                validate_config=False,
            )
            loss.setup(
                "dynamic_binding_site_allowed_min_dist",
                early_stopping=None,
                acceptance_threshold=None,
                validate_config=True,
            )

            _weighted, before_scores = loss.compute_loss(eval_result, {})
            global_config.update_const(
                "dynamic_binding_site",
                {
                    "AB": {
                        "binder": "A:1",
                        "target": "B:10",
                        "mode": "both",
                        "max_distance": 10.0,
                    }
                },
            )
            _weighted, after_scores = loss.compute_loss(eval_result, {})

            self.assertTrue(
                math.isinf(before_scores["dynamic_binding_site_allowed_min_dist"])
            )
            self.assertAlmostEqual(
                after_scores["dynamic_binding_site_allowed_min_dist"],
                5.0,
                places=4,
            )

    def test_binder_target_min_dist_uses_nearest_target_chain(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA   DA B  10      20.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA   DA C  11       1.500   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_binder_target.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("ABC", str(path))

            distance = binder_target_min_dist(
                eval_result,
                binder_chain="A",
                target_chains=["B", "C"],
                use_ca=True,
            )["ABC"]

            self.assertAlmostEqual(distance, 3.5, places=4)

    def test_allowed_min_dist_uses_chain_first_with_explicit_override(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1      10.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA B   1       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  THR C   1       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy_aab.pdb"
            path.write_text(pdb, encoding="utf-8")
            eval_result = _EvalResult("AAB", str(path))

            global_config.init_const(
                "binding_site",
                {
                    "binder": "A",
                    "target": "B:1",
                    "mode": "pocket",
                    "max_distance": 10.0,
                },
            )
            chain_default = binding_site_allowed_min_dist(eval_result, use_ca=True)["AAB"]
            self.assertAlmostEqual(chain_default, 8.0, places=4)

            global_config.reset()
            global_config.init_const(
                "binding_site",
                {
                    "binder": "chain:A",
                    "target": "chain:B:1",
                    "mode": "pocket",
                    "max_distance": 10.0,
                },
            )
            chain_override = binding_site_allowed_min_dist(eval_result, use_ca=True)["AAB"]
            self.assertAlmostEqual(chain_override, 8.0, places=4)

    def test_allowed_min_dist_supports_per_oligo_binding_site_mapping(self) -> None:
        pdb_ab = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  THR B  10       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        pdb_ac = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       8.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  THR B  20       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ab_path = Path(tmpdir) / "ab.pdb"
            ac_path = Path(tmpdir) / "ac.pdb"
            ab_path.write_text(pdb_ab, encoding="utf-8")
            ac_path.write_text(pdb_ac, encoding="utf-8")

            global_config.init_const(
                "binding_site",
                {
                    "AB": {"binder": "A", "target": "B:10"},
                    "AC": {"binder": "A", "target": "C:20"},
                },
            )
            eval_result = StandardScores(
                names=["AB", "AC"],
                plddt={"AB": [0.9], "AC": [0.9]},
                complex_paths={"AB": ab_path, "AC": ac_path},
            )

            allowed = binding_site_allowed_min_dist(eval_result, use_ca=True)
            self.assertAlmostEqual(allowed["AB"], 5.0, places=4)
            self.assertAlmostEqual(allowed["AC"], 8.0, places=4)
