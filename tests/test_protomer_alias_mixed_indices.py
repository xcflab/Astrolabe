from __future__ import annotations

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager


def test_select_residues_by_alias_supports_mixed_singletons_and_ranges() -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    protomer = Protomer("B", "A" * 200, "protein")

    mask = protomer.select_residues_by_alias("B_1:44,56-58,60,139,152,153,156")

    expected = np.zeros(200)
    expected[[43, 55, 56, 57, 59, 138, 151, 152, 155]] = 1.0
    np.testing.assert_array_equal(mask, expected)
