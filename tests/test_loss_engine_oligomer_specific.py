from __future__ import annotations

import unittest

from astrolabe.losses.base import LossEngine, loss_registry


@loss_registry.register(name="__unittest_dummy_loss__", optional_params=[], supports_oligomer_specific=True)
def _dummy_loss(eval_result):
    return {name: 1.0 for name in eval_result.names}


@loss_registry.register(
    name="__unittest_no_empty_oligomer_call__",
    optional_params=[],
    supports_oligomer_specific=True,
)
def _no_empty_oligomer_call(eval_result):
    raise AssertionError("loss must not run without a matching oligomer")


@loss_registry.register(name="__unittest_cache_loss_a__", optional_params=[])
def _dummy_cache_loss_a(eval_result):
    eval_result._counter["a"] += 1  # type: ignore[attr-defined]
    return {name: 1.0 for name in eval_result.names}


@loss_registry.register(name="__unittest_cache_loss_b__", optional_params=[])
def _dummy_cache_loss_b(eval_result):
    eval_result._counter["b"] += 1  # type: ignore[attr-defined]
    return {name: 2.0 for name in eval_result.names}


class _EvalResult:
    def __init__(self, names: list[str], *, counter: dict[str, int]):
        self.names = list(names)
        self._counter = counter

    def filter_by_oligomer(self, oligomer_specific):
        # Simulate the real API: return a new object with filtered oligomers.
        self._counter["calls"] += 1
        if isinstance(oligomer_specific, str):
            allowed = {oligomer_specific}
        else:
            allowed = set(oligomer_specific or [])
        return _EvalResult([n for n in self.names if n in allowed], counter=self._counter)


class TestLossEngineOligomerSpecific(unittest.TestCase):
    def test_oligomer_specific_params_not_mutated(self) -> None:
        counter = {"calls": 0}
        engine = LossEngine(
            loss_config={
                "loss_terms": {
                    "dummy": {
                        "_type": "__unittest_dummy_loss__",
                        "_for": "AB",
                    }
                }
            },
            validate_config=False,
        )
        engine.setup(
            weighted_score_expression="dummy",
            early_stopping=None,
            acceptance_threshold=None,
            validate_config=False,
        )

        eval1 = _EvalResult(["AB", "AC"], counter=counter)
        eval2 = _EvalResult(["AB", "AC"], counter=counter)

        per_oligo_scores_1, _ = engine._extract_scores_with_registry(eval1)
        per_oligo_scores_2, _ = engine._extract_scores_with_registry(eval2)

        # Must filter on both calls (two distinct eval_result objects => cache miss).
        self.assertEqual(counter["calls"], 2)
        self.assertEqual(set(per_oligo_scores_1.keys()), {"AB"})
        self.assertEqual(set(per_oligo_scores_2.keys()), {"AB"})

    def test_setup_invalidation_clears_cached_eval_results(self) -> None:
        counter = {"a": 0, "b": 0}
        engine = LossEngine(
            loss_config={"loss_terms": {}},
            validate_config=False,
        )

        eval_result = _EvalResult(["AB"], counter=counter)

        engine.setup(
            weighted_score_expression="__unittest_cache_loss_a__",
            early_stopping=None,
            acceptance_threshold=None,
            validate_config=False,
        )
        _, per_terms_a = engine._extract_scores_with_registry(eval_result)
        self.assertEqual(counter["a"], 1)
        self.assertIn("__unittest_cache_loss_a__", per_terms_a)
        self.assertNotIn("__unittest_cache_loss_b__", per_terms_a)

        engine.setup(
            weighted_score_expression="__unittest_cache_loss_b__",
            early_stopping=None,
            acceptance_threshold=None,
            validate_config=False,
        )
        _, per_terms_b = engine._extract_scores_with_registry(eval_result)
        self.assertEqual(counter["b"], 1)
        self.assertIn("__unittest_cache_loss_b__", per_terms_b)
        self.assertNotIn("__unittest_cache_loss_a__", per_terms_b)

    def test_empty_oligomer_specific_dict_loss_has_zero_mean(self) -> None:
        counter = {"calls": 0}
        engine = LossEngine(
            loss_config={
                "loss_terms": {
                    "dummy": {
                        "_type": "__unittest_no_empty_oligomer_call__",
                        "_for": "AB",
                    }
                }
            },
            validate_config=False,
        )
        engine.setup(
            weighted_score_expression="dummy",
            early_stopping=None,
            acceptance_threshold=None,
            validate_config=False,
        )

        _, per_terms = engine._extract_scores_with_registry(
            _EvalResult(["A"], counter=counter)
        )

        self.assertEqual(per_terms["dummy"], 0.0)
