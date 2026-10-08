from __future__ import annotations

import pathlib
import tempfile
import unittest

from astrolabe.analysis.constraint_compliance import compute_geometry_metrics
from astrolabe.core.binding_site import parse_binding_site


class TestConstraintComplianceAnalysis(unittest.TestCase):
    def test_compute_geometry_metrics_contact_fraction(self) -> None:
        pdb = "\n".join(
            [
                "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       7.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  THR B  64       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      5  CA  THR B  65      10.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
        spec = parse_binding_site(
            {
                "binder": ["A:1-2"],
                "target": "B:64-65",
                "mode": "contact",
                "max_distance": 6.0,
            }
        )
        assert spec is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "toy.pdb"
            path.write_text(pdb, encoding="utf-8")
            out = compute_geometry_metrics(path, spec)

        self.assertAlmostEqual(out["ca_any_binder_min_dist"], 2.0, places=6)
        self.assertAlmostEqual(out["ca_allowed_min_dist"], 3.0, places=6)
        self.assertAlmostEqual(out["ca_framework_min_dist"], 2.0, places=6)
        self.assertEqual(out["ca_any_binder_pass"], 1.0)
        self.assertEqual(out["ca_allowed_pass"], 1.0)
        self.assertEqual(out["ca_contact_pair_count"], 4.0)
        self.assertAlmostEqual(out["ca_contact_satisfied_fraction"], 0.75, places=6)

