from __future__ import annotations

import inspect
import importlib.util
import unittest


class TestChaiConstraintSupport(unittest.TestCase):
    def test_chai1_run_inference_supports_constraint_path(self) -> None:
        if importlib.util.find_spec("pandera") is None or importlib.util.find_spec("numba") is None:
            raise unittest.SkipTest("vendored chai_lab requires pandera+numba for constraint support")

        from astrolabe.evaluators import chai1 as chai_lab

        chai1_mod = chai_lab._load_chai1_module()
        sig = inspect.signature(chai1_mod.run_inference)
        self.assertIn("constraint_path", sig.parameters)
