from __future__ import annotations

import unittest


class TestAf3FeaturizationProfile(unittest.TestCase):
    def test_profile_featurization_returns_step_timings(self) -> None:
        from astrolabe.utils.af3_featurization_profile import (
            profile_featurization_steps,
        )

        seq = "A" * 10
        result = profile_featurization_steps(
            sequence=seq,
            deterministic_frames=False,
            msa_crop_size=32,
            max_templates=0,
            pad_num_chains=1,
            runs=1,
            warmup=0,
        )

        self.assertIn("total_s", result)
        self.assertIn("times_s", result)
        self.assertIn("bucket_tokens", result)
        self.assertGreater(result["total_s"], 0.0)

        times = result["times_s"]
        self.assertIn("to_structure", times)
        self.assertIn("clean_structure", times)
        self.assertIn("tokenizer", times)
        self.assertIn("msa_features", times)
        self.assertIn("ref_structure", times)

        for name, value in times.items():
            self.assertIsInstance(value, float, msg=f"{name} is not float")
            self.assertGreaterEqual(value, 0.0, msg=f"{name} is negative")

