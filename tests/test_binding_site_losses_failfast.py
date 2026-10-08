from __future__ import annotations

import pathlib

import pytest

from astrolabe.core import global_config
from astrolabe.losses.binding_site import (
    BindingSiteLossError,
    binding_site_allowed_min_dist,
)


class _EvalResult:
    def __init__(self, name: str, structure_path: pathlib.Path):
        self.names = [name]
        self.complex_paths = {name: structure_path}


@pytest.fixture(autouse=True)
def _reset_global_config():
    global_config.reset()
    yield
    global_config.reset()


def test_binding_site_loss_requires_config(tmp_path: pathlib.Path) -> None:
    eval_result = _EvalResult("AB", tmp_path / "dummy.pdb")
    with pytest.raises(BindingSiteLossError, match="require `binding_site` config"):
        binding_site_allowed_min_dist(eval_result, use_ca=True)


def test_binding_site_loss_fails_when_selector_is_unresolved(tmp_path: pathlib.Path) -> None:
    pdb = "\n".join(
        [
            "ATOM      1  CA  ALA A   1       5.000   0.000   0.000  1.00 20.00           C",
            "ATOM      2  CA  THR B 258       0.000   0.000   0.000  1.00 20.00           C",
            "TER",
            "END",
            "",
        ]
    )
    path = tmp_path / "toy.pdb"
    path.write_text(pdb, encoding="utf-8")
    eval_result = _EvalResult("AB", path)
    global_config.init_const(
        "binding_site",
        {
            "binder": ["C:1-2"],
            "target": "B:258",
            "mode": "pocket",
            "max_distance": 10.0,
        },
    )

    with pytest.raises(BindingSiteLossError, match="did not resolve to any chains"):
        binding_site_allowed_min_dist(eval_result, use_ca=True)


def test_binding_site_loss_fails_when_selector_matches_multiple_chains(
    tmp_path: pathlib.Path,
) -> None:
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
    path = tmp_path / "toy_aab.pdb"
    path.write_text(pdb, encoding="utf-8")
    eval_result = _EvalResult("AAB", path)
    global_config.init_const(
        "binding_site",
        {
            "binder": "protomer:A",
            "target": "protomer:B:1",
            "mode": "pocket",
            "max_distance": 10.0,
        },
    )

    with pytest.raises(BindingSiteLossError, match="matches multiple chains"):
        binding_site_allowed_min_dist(eval_result, use_ca=True)
