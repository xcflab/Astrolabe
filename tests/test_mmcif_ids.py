from __future__ import annotations

import unittest

from astrolabe.utils.mmcif_ids import int_id_to_str_id, str_id_to_int_id


class TestMmcifIds(unittest.TestCase):
    def test_known_mappings(self) -> None:
        self.assertEqual(int_id_to_str_id(1), "A")
        self.assertEqual(int_id_to_str_id(26), "Z")
        self.assertEqual(int_id_to_str_id(27), "AA")
        # NOTE: alphafold3 uses a little-endian variant: 28 -> "BA" (not "AB").
        self.assertEqual(int_id_to_str_id(28), "BA")
        self.assertEqual(int_id_to_str_id(52), "ZA")
        self.assertEqual(int_id_to_str_id(53), "AB")

    def test_roundtrip(self) -> None:
        for i in range(1, 2000):
            self.assertEqual(str_id_to_int_id(int_id_to_str_id(i)), i)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            int_id_to_str_id(0)
        with self.assertRaises(ValueError):
            str_id_to_int_id("")
        with self.assertRaises(ValueError):
            str_id_to_int_id("a")

