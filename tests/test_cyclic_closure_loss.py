from __future__ import annotations

from pathlib import Path

import pytest

from astrolabe.core import global_config
from astrolabe.losses.cyclic_closure import (
    cyclic_all_bonds_closed,
    cyclic_closure_distance_score,
    cyclic_closure_max_excess,
    cyclic_closure_norm,
)


class _Protomer:
    def __init__(self, sequence: str):
        self.sequence = sequence


class _EvalResult:
    def __init__(self, path: Path):
        self.names = ["A"]
        self.complex_paths = {"A": path}


def _write_pdb(path: Path) -> Path:
    lines = [
        "ATOM      1    N ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2   CA ALA A   2       0.000   4.000   0.000  1.00 20.00           C",
        "ATOM      3   CA ALA A   3       0.000   9.000   0.000  1.00 20.00           C",
        "ATOM      4    C ALA A   4       1.330   0.000   0.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _set_global_for_test(*, cyclization: dict) -> None:
    global_config.reset()
    global_config.init_const("cyclization", cyclization)
    global_config.init_const("protomers", {"A": _Protomer("AAAA")})


def test_cyclic_closure_norm_head_to_tail_near_target(tmp_path: Path) -> None:
    pdb_path = _write_pdb(tmp_path / "near_target.pdb")
    _set_global_for_test(cyclization={"head_to_tail": True})
    score = cyclic_closure_norm(_EvalResult(pdb_path), sigma=0.20, aggregate="min")
    assert score["A"] == pytest.approx(1.0, abs=1e-6)


def test_cyclic_closure_norm_respects_encoding_scope(tmp_path: Path) -> None:
    pdb_path = _write_pdb(tmp_path / "scope_test.pdb")
    _set_global_for_test(
        cyclization={
            "head_to_tail": True,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 3, "atom": "CA"},
                    "order": "single",
                    "target_distance": 1.54,
                }
            ],
            "encoding": {"scope": "head_to_tail_only"},
        }
    )
    score_head_to_tail_only = cyclic_closure_norm(
        _EvalResult(pdb_path), sigma=0.20, aggregate="min"
    )
    assert score_head_to_tail_only["A"] == pytest.approx(1.0, abs=1e-6)

    _set_global_for_test(
        cyclization={
            "head_to_tail": True,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 3, "atom": "CA"},
                    "order": "single",
                    "target_distance": 1.54,
                }
            ],
            "encoding": {"scope": "all_bonds"},
        }
    )
    score_all_bonds = cyclic_closure_norm(_EvalResult(pdb_path), sigma=0.20, aggregate="min")
    assert score_all_bonds["A"] < 1e-4


@pytest.mark.parametrize("signal_mode", ["none", "index_only", "bond_only", "both"])
def test_cyclic_closure_norm_is_independent_of_signal_mode(
    tmp_path: Path,
    signal_mode: str,
) -> None:
    pdb_path = _write_pdb(tmp_path / f"signal_mode_{signal_mode}.pdb")
    _set_global_for_test(
        cyclization={
            "head_to_tail": True,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 3, "atom": "CA"},
                    "order": "single",
                    "target_distance": 1.54,
                }
            ],
            "encoding": {"scope": "all_bonds"},
            "signal_mode": signal_mode,
        }
    )

    score = cyclic_closure_norm(_EvalResult(pdb_path), sigma=0.20, aggregate="min")

    assert score["A"] < 1e-4


def test_direct_closure_metric_accepts_distance_within_margin(tmp_path: Path) -> None:
    pdb_path = _write_pdb(tmp_path / "direct_margin_pass.pdb")
    _set_global_for_test(
        cyclization={
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 3, "atom": "CA"},
                    "order": "single",
                    "target_distance": 4.80,
                }
            ],
            "encoding": {"scope": "all_bonds"},
        }
    )

    max_excess = cyclic_closure_max_excess(_EvalResult(pdb_path))
    closed = cyclic_all_bonds_closed(_EvalResult(pdb_path), margin=0.30)
    score = cyclic_closure_distance_score(_EvalResult(pdb_path), margin=0.30)
    gaussian = cyclic_closure_norm(_EvalResult(pdb_path), sigma=0.20, aggregate="min")

    assert max_excess["A"] == pytest.approx(0.20, abs=1e-6)
    assert closed["A"] == pytest.approx(1.0, abs=1e-6)
    assert score["A"] == pytest.approx(1.0 - 0.20 / 0.30, abs=1e-6)
    assert gaussian["A"] < 0.90


def test_direct_closure_metric_rejects_distance_outside_margin(tmp_path: Path) -> None:
    pdb_path = _write_pdb(tmp_path / "direct_margin_fail.pdb")
    _set_global_for_test(
        cyclization={
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 3, "atom": "CA"},
                    "order": "single",
                    "target_distance": 4.60,
                }
            ],
            "encoding": {"scope": "all_bonds"},
        }
    )

    max_excess = cyclic_closure_max_excess(_EvalResult(pdb_path))
    closed = cyclic_all_bonds_closed(_EvalResult(pdb_path), margin=0.30)
    score = cyclic_closure_distance_score(_EvalResult(pdb_path), margin=0.30)

    assert max_excess["A"] == pytest.approx(0.40, abs=1e-6)
    assert closed["A"] == pytest.approx(0.0, abs=1e-6)
    assert score["A"] == pytest.approx(0.0, abs=1e-6)
