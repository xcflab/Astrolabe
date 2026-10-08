from __future__ import annotations

from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
from astrolabe.evaluators.wrappers.alphafold3 import BUCKETS


def _bucket_policy(tolerance: int = 50) -> AlphaFold3Evaluator:
    evaluator = object.__new__(AlphaFold3Evaluator)
    evaluator._bucket_pad_tolerance = tolerance
    evaluator._default_buckets = BUCKETS
    return evaluator


def test_af3_bucket_policy_uses_default_bucket_only_near_boundary() -> None:
    evaluator = _bucket_policy(tolerance=50)

    assert evaluator._should_use_default_buckets(180) is False
    assert evaluator._should_use_default_buckets(220) is True
    assert evaluator._should_use_default_buckets(185) is False
    assert evaluator._should_use_default_buckets(225) is True


def test_af3_bucket_policy_handles_unknown_and_oversized_tokens() -> None:
    evaluator = _bucket_policy(tolerance=50)

    assert evaluator._should_use_default_buckets(None) is True
    assert evaluator._should_use_default_buckets(max(BUCKETS) + 1) is False
