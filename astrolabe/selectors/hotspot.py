from __future__ import annotations

from typing import Optional

import numpy as np

from astrolabe.core import global_config
from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class HotspotAliasPositionSelector(PositionSelector):
    """Select positions using protomer alias expressions.

    This selector enables using protomer-defined segment aliases (from `seqs`
    statements with `@alias` and implicit segment names like `A_1`) inside
    `position_selector` expressions via `hotspot(...)`.

    Example:
      - position_selector: \"hotspot('A_1:10-30;B_2:5-8')\"
    """

    def __init__(self, complex_poses: Optional[ComplexPoses], alias: str, merge: str = "any"):
        self.alias = str(alias)
        super().__init__(complex_poses, merge=merge)

    def select_positions(self, **kwargs):
        if not self.alias or not self.alias.strip():
            raise ValueError("hotspot(alias=...) must be a non-empty string")

        sites = {}
        protomers = getattr(global_config, "protomers", {}) or {}
        for proto_name, protomer in protomers.items():
            weights = protomer.select_residues_by_alias(self.alias)
            weights = np.asarray(weights)
            if weights.shape[-1] != protomer.length:
                raise ValueError(
                    f"hotspot selector produced shape {weights.shape} for protomer '{proto_name}' "
                    f"(expected length {protomer.length})"
                )
            sites[proto_name] = weights > 0

        self.sites = sites
        return self.sites

