from __future__ import annotations

from pathlib import Path
import types

import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.structure_processors.base import StructureProcessor


class _DummyPredocker(StructureProcessor):
    def _run_impl(self, input_file):  # pragma: no cover - not used in this unit test
        return input_file


def test_predocker_eval_condition_uses_loss_engine_score_context(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("protomers", {})
    global_config.init_const("stage_manager", None)

    loss = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss)

    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )
    global_config.init_const(
        "decisions",
        types.SimpleNamespace(
            evaluation_results={0: eval_result},
            last_accepted=0,
        ),
    )

    predocker = _DummyPredocker(cache_enabled=False, _condition="plddt > 0.0")

    def _boom(*_a, **_k):
        raise RuntimeError("do not call private extractor")

    loss._extract_scores_with_registry = _boom  # type: ignore[method-assign]
    loss.score_context = lambda _eval_result: {"plddt": 1.0}  # type: ignore[attr-defined]

    assert predocker.eval_condition() is True


def test_predocker_condition_missing_defaults_are_used_for_absent_scores(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("protomers", {})
    global_config.init_const("stage_manager", None)

    loss = LossEngine({"loss_terms": {}})
    loss.score_context = lambda _eval_result: {}  # type: ignore[attr-defined]
    global_config.init_const("loss_computer", loss)

    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [0.9]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )
    global_config.init_const(
        "decisions",
        types.SimpleNamespace(
            evaluation_results={0: eval_result},
            last_accepted=0,
        ),
    )

    predocker = _DummyPredocker(
        cache_enabled=False,
        _condition="d3_ananas == 0.0",
        _condition_missing_defaults={"d3_ananas": 0.0},
    )

    assert predocker.eval_condition() is True


def test_predocker_condition_real_scores_override_missing_defaults(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("protomers", {})
    global_config.init_const("stage_manager", None)

    loss = LossEngine({"loss_terms": {}})
    loss.score_context = lambda _eval_result: {"d3_ananas": 0.25}  # type: ignore[attr-defined]
    global_config.init_const("loss_computer", loss)

    eval_result = StandardScores(
        names=["AAAAAA"],
        plddt={"AAAAAA": [0.9]},
        complex_paths={"AAAAAA": tmp_path / "d3.pdb"},
    )
    global_config.init_const(
        "decisions",
        types.SimpleNamespace(
            evaluation_results={0: eval_result},
            last_accepted=0,
        ),
    )

    predocker = _DummyPredocker(
        cache_enabled=False,
        _condition="d3_ananas == 0.0",
        _condition_missing_defaults={"d3_ananas": 0.0},
    )

    assert predocker.eval_condition() is False
