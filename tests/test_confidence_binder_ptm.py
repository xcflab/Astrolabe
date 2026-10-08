from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.losses.confidence import binder_iptm, binder_ptm, target_hotspot_plddt


class _EvalResult:
    def __init__(self):
        self.names = ["AB"]
        self.chain_pair_scores = {
            "AB": {
                "chain_ptm": [
                    np.array([0.25, 0.75], dtype=float),
                ]
            }
        }


class _EvalResultRepeated:
    def __init__(self):
        self.names = ["AAB"]
        self.chain_pair_scores = {
            "AAB": {
                "chain_ptm": [
                    np.array([0.10, 0.20, 0.90], dtype=float),
                ],
                "chain_pair_iptm": [
                    np.array(
                        [
                            [0.0, 0.2, 0.4],
                            [0.2, 0.0, 0.6],
                            [0.4, 0.6, 0.0],
                        ],
                        dtype=float,
                    )
                ],
            }
        }


class _EvalResultHotspot:
    def __init__(self):
        self.names = ["AB"]
        self.per_proto_scores = {
            "A": [np.array([0.11, 0.22, 0.33], dtype=float)],
            "B": [np.array([0.61, 0.72, 0.83, 0.94], dtype=float)],
        }


class _EvalResultHotspotRepeated:
    def __init__(self):
        self.names = ["AAB"]
        self.per_proto_scores = {
            "A": [
                np.array([0.10, 0.20, 0.30], dtype=float),
                np.array([0.40, 0.50, 0.60], dtype=float),
            ],
            "B": [np.array([0.70, 0.80, 0.90], dtype=float)],
        }


class TestConfidenceBinderPtm(unittest.TestCase):
    def tearDown(self) -> None:
        global_config.reset()

    def test_binder_ptm_defaults_to_chain_B(self) -> None:
        with mock.patch("astrolabe.core.binding_site.parse_binding_site", return_value=None):
            out = binder_ptm(_EvalResult())["AB"]
        self.assertAlmostEqual(out, 0.75, places=8)

    def test_binder_ptm_can_select_chain_A(self) -> None:
        out = binder_ptm(_EvalResult(), binder_chain="A")["AB"]
        self.assertAlmostEqual(out, 0.25, places=8)

    def test_binder_ptm_uses_chain_first_for_repeated_oligo(self) -> None:
        out = binder_ptm(_EvalResultRepeated(), binder_chain="A")["AAB"]
        self.assertAlmostEqual(out, 0.10, places=8)
        out_b = binder_ptm(_EvalResultRepeated(), binder_chain="B")["AAB"]
        self.assertAlmostEqual(out_b, 0.20, places=8)

    def test_binder_ptm_supports_explicit_chain_override(self) -> None:
        out = binder_ptm(_EvalResultRepeated(), binder_chain="chain:B")["AAB"]
        self.assertAlmostEqual(out, 0.20, places=8)

    def test_binder_ptm_supports_explicit_protomer_override(self) -> None:
        out = binder_ptm(_EvalResultRepeated(), binder_chain="protomer:A")["AAB"]
        self.assertAlmostEqual(out, 0.15, places=8)
        out_b = binder_ptm(_EvalResultRepeated(), binder_chain="protomer:B")["AAB"]
        self.assertAlmostEqual(out_b, 0.90, places=8)

    def test_binder_iptm_uses_chain_first_for_repeated_oligo(self) -> None:
        out = binder_iptm(_EvalResultRepeated(), binder_chain="A")["AAB"]
        self.assertAlmostEqual(out, 0.30, places=8)
        out_b = binder_iptm(_EvalResultRepeated(), binder_chain="B")["AAB"]
        self.assertAlmostEqual(out_b, 0.40, places=8)

    def test_binder_iptm_supports_explicit_chain_override(self) -> None:
        out = binder_iptm(_EvalResultRepeated(), binder_chain="chain:B")["AAB"]
        self.assertAlmostEqual(out, 0.40, places=8)

    def test_binder_iptm_supports_explicit_protomer_override(self) -> None:
        out = binder_iptm(_EvalResultRepeated(), binder_chain="protomer:A")["AAB"]
        self.assertAlmostEqual(out, 0.35, places=8)
        out_b = binder_iptm(_EvalResultRepeated(), binder_chain="protomer:B")["AAB"]
        self.assertAlmostEqual(out_b, 0.50, places=8)

    def test_binder_metrics_infer_binding_site_per_oligo(self) -> None:
        class _EvalResultPerOligo:
            def __init__(self):
                self.names = ["AB", "AC"]
                self.chain_pair_scores = {
                    "AB": {
                        "chain_ptm": [np.array([0.2, 0.8], dtype=float)],
                        "chain_pair_iptm": [np.array([[0.0, 0.25], [0.25, 0.0]], dtype=float)],
                    },
                    "AC": {
                        "chain_ptm": [np.array([0.6, 0.4], dtype=float)],
                        "chain_pair_iptm": [np.array([[0.0, 0.75], [0.75, 0.0]], dtype=float)],
                    },
                }

        def _parse_binding_site_side_effect(_cfg, *, oligo_name=None):
            binder = {"AB": "A", "AC": "B"}[str(oligo_name)]
            return mock.Mock(binder=mock.Mock(chain=binder))

        global_config.init_const("binding_site", {"AB": {}, "AC": {}})
        eval_result = _EvalResultPerOligo()
        with mock.patch(
            "astrolabe.core.binding_site.parse_binding_site",
            side_effect=_parse_binding_site_side_effect,
        ):
            ptm_scores = binder_ptm(eval_result)
            iptm_scores = binder_iptm(eval_result)

        self.assertEqual(ptm_scores, {"AB": 0.2, "AC": 0.4})
        self.assertEqual(iptm_scores, {"AB": 0.25, "AC": 0.75})

    def test_target_hotspot_plddt_uses_explicit_target_field(self) -> None:
        global_config.init_const(
            "binding_site",
            {
                "binder": "A",
                "target": "B:1",
            },
        )
        out = target_hotspot_plddt(_EvalResultHotspot(), target="B:2-4")["AB"]
        self.assertAlmostEqual(out, (0.72 + 0.83 + 0.94) / 3.0, places=8)

    def test_target_hotspot_plddt_merges_list_specs_on_same_chain(self) -> None:
        out = target_hotspot_plddt(
            _EvalResultHotspot(),
            target=["B:1-2", "B:4"],
        )["AB"]
        self.assertAlmostEqual(out, (0.61 + 0.72 + 0.94) / 3.0, places=8)

    def test_target_hotspot_plddt_accepts_omegaconf_listconfig(self) -> None:
        out = target_hotspot_plddt(
            _EvalResultHotspot(),
            target=OmegaConf.create(["B:1-2", "B:4"]),
        )["AB"]
        self.assertAlmostEqual(out, (0.61 + 0.72 + 0.94) / 3.0, places=8)

    def test_target_hotspot_plddt_respects_chain_first_and_explicit_protomer(self) -> None:
        out_chain = target_hotspot_plddt(
            _EvalResultHotspotRepeated(),
            target="chain:B:1-2",
        )["AAB"]
        self.assertAlmostEqual(out_chain, (0.40 + 0.50) / 2.0, places=8)

        out_proto = target_hotspot_plddt(
            _EvalResultHotspotRepeated(),
            target="protomer:A:1-2",
        )["AAB"]
        self.assertAlmostEqual(out_proto, (0.10 + 0.20 + 0.40 + 0.50) / 4.0, places=8)
