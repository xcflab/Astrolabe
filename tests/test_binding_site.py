from __future__ import annotations

import unittest

from omegaconf import OmegaConf

from astrolabe.core.binding_site import (
    has_binding_site_config,
    parse_binding_site,
    parse_chain_positions,
)


class TestBindingSiteParsing(unittest.TestCase):
    def test_parse_chain_positions_chain_only(self) -> None:
        parsed = parse_chain_positions("B")
        self.assertEqual(parsed.chain, "B")
        self.assertEqual(parsed.positions, ())

    def test_parse_chain_positions_single(self) -> None:
        parsed = parse_chain_positions("B:258")
        self.assertEqual(parsed.chain, "B")
        self.assertEqual(parsed.positions, (258,))

    def test_parse_chain_positions_ranges(self) -> None:
        parsed = parse_chain_positions("A:1-3,5")
        self.assertEqual(parsed.chain, "A")
        self.assertEqual(parsed.positions, (1, 2, 3, 5))

    def test_parse_chain_positions_chain_selector_prefix(self) -> None:
        parsed = parse_chain_positions("chain:B:1-2")
        self.assertEqual(parsed.chain, "chain:B")
        self.assertEqual(parsed.positions, (1, 2))

    def test_parse_binding_site_basic(self) -> None:
        spec = parse_binding_site({"binder": "A", "target": "B:258"})
        assert spec is not None
        self.assertEqual(spec.binder.chain, "A")
        self.assertEqual(spec.binder.positions, ())
        self.assertEqual(spec.target.chain, "B")
        self.assertEqual(spec.target.positions, (258,))
        self.assertEqual(spec.mode, "pocket")
        self.assertEqual(spec.max_distance, 6.0)
        self.assertEqual(spec.min_distance, 0.0)

    def test_parse_binding_site_paratope_only(self) -> None:
        spec = parse_binding_site({"binder": "A:2-3", "target": "B"})
        assert spec is not None
        self.assertEqual(spec.binder.chain, "A")
        self.assertEqual(spec.binder.positions, (2, 3))
        self.assertEqual(spec.target.chain, "B")
        self.assertEqual(spec.target.positions, ())
        self.assertEqual(spec.mode, "pocket")

    def test_parse_binding_site_target_list_merges(self) -> None:
        spec = parse_binding_site({"binder": "A", "target": ["B:1", "B:2-3", "B:2"]})
        assert spec is not None
        self.assertEqual(spec.target.chain, "B")
        # Unique + preserve first-seen order.
        self.assertEqual(spec.target.positions, (1, 2, 3))

    def test_parse_binding_site_binder_list_merges(self) -> None:
        spec = parse_binding_site(
            {"binder": ["A:1", "A:3-4", "A:3"], "target": "B:258", "mode": "pocket"}
        )
        assert spec is not None
        self.assertEqual(spec.binder.chain, "A")
        # Unique + preserve first-seen order.
        self.assertEqual(spec.binder.positions, (1, 3, 4))

    def test_parse_binding_site_binder_listconfig(self) -> None:
        cfg = OmegaConf.create(
            {"binder": ["A:1", "A:3-4", "A:3"], "target": "B:258", "mode": "pocket"}
        )
        spec = parse_binding_site(cfg)
        assert spec is not None
        self.assertEqual(spec.binder.chain, "A")
        self.assertEqual(spec.binder.positions, (1, 3, 4))

    def test_parse_binding_site_two_sided_infers_both(self) -> None:
        spec = parse_binding_site({"binder": "A:1-2", "target": "B:258-259"})
        assert spec is not None
        self.assertEqual(spec.mode, "both")

    def test_parse_binding_site_requires_at_least_one_side_positions(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A", "target": "B"})

    def test_parse_binding_site_explicit_contact_requires_two_sided_positions(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A:1-2", "target": "B", "mode": "contact"})
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A", "target": "B:1-2", "mode": "contact"})

    def test_parse_binding_site_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A", "target": "B:1", "mode": "bad"})

    def test_parse_binding_site_distance_override(self) -> None:
        spec = parse_binding_site(
            {"binder": "A", "target": "B:1", "max_distance": 10, "min_distance": 2}
        )
        assert spec is not None
        self.assertEqual(spec.max_distance, 10.0)
        self.assertEqual(spec.min_distance, 2.0)

    def test_parse_binding_site_distance_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A", "target": "B:1", "max_distance": 0})
        with self.assertRaises(ValueError):
            parse_binding_site({"binder": "A", "target": "B:1", "min_distance": -1})
        with self.assertRaises(ValueError):
            parse_binding_site(
                {"binder": "A", "target": "B:1", "min_distance": 9, "max_distance": 8}
            )

    def test_parse_binding_site_per_oligo_mapping(self) -> None:
        cfg = {
            "AB": {"binder": "A", "target": "B:10-11"},
            "AC": {"binder": "A", "target": "C:20-21"},
        }
        spec_ab = parse_binding_site(cfg, oligo_name="AB")
        spec_ac = parse_binding_site(cfg, oligo_name="AC")
        assert spec_ab is not None
        assert spec_ac is not None
        self.assertEqual(spec_ab.target.chain, "B")
        self.assertEqual(spec_ab.target.positions, (10, 11))
        self.assertEqual(spec_ac.target.chain, "C")
        self.assertEqual(spec_ac.target.positions, (20, 21))

    def test_parse_binding_site_per_oligo_mapping_missing_returns_none(self) -> None:
        cfg = {"AB": {"binder": "A", "target": "B:10-11"}}
        spec = parse_binding_site(cfg, oligo_name="AC")
        self.assertIsNone(spec)

    def test_has_binding_site_config_accepts_per_oligo_mapping(self) -> None:
        cfg = {
            "AB": {"binder": "A", "target": "B:10-11"},
            "AC": {"binder": "A", "target": "C:20-21"},
        }
        self.assertTrue(has_binding_site_config(cfg))
