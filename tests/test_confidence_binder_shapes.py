from __future__ import annotations

import numpy as np
import pytest

from astrolabe.losses.confidence import binder_iptm, binder_ptm


class _EvalResultHigherRank:
    def __init__(self):
        self.names = ["AB"]
        self.chain_pair_scores = {
            "AB": {
                "chain_ptm": [
                    np.array([[0.25, 0.75]], dtype=float),
                ],
                "chain_pair_iptm": [
                    np.array(
                        [[[0.0, 0.31], [0.31, 0.0]]],
                        dtype=float,
                    )
                ],
            }
        }


def test_binder_ptm_accepts_singleton_sample_axis() -> None:
    scores = binder_ptm(_EvalResultHigherRank(), binder_chain="A")
    assert scores["AB"] == pytest.approx(0.25)


def test_binder_iptm_accepts_singleton_sample_axis() -> None:
    scores = binder_iptm(_EvalResultHigherRank(), binder_chain="A")
    assert scores["AB"] == pytest.approx(0.31)
