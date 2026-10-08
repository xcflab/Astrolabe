from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.evaluators.utils import construct_evaluators


def test_construct_evaluators_errors_include_alias_on_unresolved_interpolation(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create({"_type": "chai1", "foo": "${does_not_exist}"})

    class _EvalConf:
        evaluator_terms = {"BadAlias": cfg}

    with pytest.raises(ValueError, match="BadAlias"):
        construct_evaluators(_EvalConf(), raw_model_results_dir=tmp_path)

