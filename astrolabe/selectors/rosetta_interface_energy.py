from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from astrolabe.selectors.base import PositionSelector


_TERM_NAME = "rosetta_interface_dg"
logger = logging.getLogger(__name__)


class RosettaInterfaceEnergyPositionSelector(PositionSelector):
    def __init__(self, evaluation_result, protomers, top_k: int = 5, merge: str = "any"):
        self.evaluation_result = evaluation_result
        self.protomers = protomers
        self.top_k = self._validate_top_k(top_k)
        super().__init__(None, merge=merge)

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
            raise ValueError("rosetta_interface_energy top_k must be a positive integer")
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("rosetta_interface_energy top_k must be a positive integer")
        return top_k

    @staticmethod
    def _as_score_arrays(raw_values: Any) -> list[np.ndarray]:
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if not values:
            raise ValueError(f"{_TERM_NAME} has no score arrays")
        return [np.asarray(value, dtype=float) for value in values]

    @staticmethod
    def _max_finite_score(arrays: list[np.ndarray], *, proto: str, length: int) -> np.ndarray:
        for arr in arrays:
            if arr.shape != (length,):
                raise ValueError(
                    f"{_TERM_NAME} length mismatch for protomer {proto!r}: "
                    f"expected {length}, got shape {arr.shape}"
                )
            if np.isinf(arr).any():
                raise ValueError(f"{_TERM_NAME} has infinite values for protomer {proto!r}")

        stacked = np.stack(arrays, axis=0)
        finite = np.isfinite(stacked)
        score = np.full(length, np.nan, dtype=float)
        has_finite = np.any(finite, axis=0)
        if np.any(has_finite):
            score[has_finite] = np.max(np.where(finite, stacked, -np.inf), axis=0)[
                has_finite
            ]
        return score

    def _score_for_proto(self, proto: str, protomer) -> np.ndarray:
        terms = getattr(self.evaluation_result, "per_proto_score_terms", None)
        if not isinstance(terms, Mapping):
            raise KeyError(f"Missing {_TERM_NAME} in evaluation_result.per_proto_score_terms")

        metric_map = terms.get(proto)
        if not isinstance(metric_map, Mapping) or _TERM_NAME not in metric_map:
            raise KeyError(f"Missing {_TERM_NAME} for protomer {proto!r}")

        return self._max_finite_score(
            self._as_score_arrays(metric_map[_TERM_NAME]),
            proto=proto,
            length=int(protomer.length),
        )

    def select_positions(self, **kwargs):
        sites = {
            proto: np.zeros(int(protomer.length), dtype=bool)
            for proto, protomer in self.protomers.items()
        }
        candidates: list[tuple[float, str, int]] = []

        for proto, protomer in self.protomers.items():
            score = self._score_for_proto(str(proto), protomer)
            mutable_mask = np.asarray(protomer.mutable_mask, dtype=bool)
            if mutable_mask.shape != (int(protomer.length),):
                raise ValueError(
                    f"mutable_mask length mismatch for protomer {proto!r}: "
                    f"expected {int(protomer.length)}, got shape {mutable_mask.shape}"
                )
            for position, value in enumerate(score):
                if bool(mutable_mask[position]) and np.isfinite(value):
                    candidates.append((float(value), str(proto), int(position)))

        if not candidates:
            raise ValueError("rosetta_interface_energy selected no finite mutable candidates")

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = candidates[: self.top_k]
        for _, proto, position in selected:
            sites[proto][position] = True
        logger.info(
            "rosetta_interface_energy selected %s",
            ", ".join(
                f"{proto}{position + 1}={value:.6f}" for value, proto, position in selected
            ),
        )

        self.sites = sites
        return self.sites
