from __future__ import annotations

import pytest

from astrolabe.core import global_config
from astrolabe.core import protomer_config_parser


def test_parse_ptm_sequence_uses_alias_parent_aa_fallback(monkeypatch) -> None:
    ccd = {
        "S5A": {
            "_chem_comp.one_letter_code": ["?"],
        }
    }

    def _fake_cached_ccd(*args, **kwargs):
        return ccd

    monkeypatch.setattr(protomer_config_parser, "cached_ccd", _fake_cached_ccd)

    global_config.reset()
    global_config.init_const(
        "nonstandard_aliases",
        {"S5": {"ccd_id": "S5A", "parent_aa": "S"}},
    )
    global_config.init_const("user_ccd", None)

    seq, ptm_dict = protomer_config_parser.parse_ptm_sequence("A[S5]G")
    assert seq == "ASG"
    assert ptm_dict == {1: "S5A"}


def test_parse_ptm_sequence_raises_without_valid_fallback(monkeypatch) -> None:
    ccd = {
        "MSE": {
            "_chem_comp.one_letter_code": ["?"],
        }
    }

    def _fake_cached_ccd(*args, **kwargs):
        return ccd

    monkeypatch.setattr(protomer_config_parser, "cached_ccd", _fake_cached_ccd)

    global_config.reset()
    global_config.init_const("nonstandard_aliases", {})
    global_config.init_const("user_ccd", None)

    with pytest.raises(ValueError, match="Invalid PTM one-letter code"):
        protomer_config_parser.parse_ptm_sequence("A[MSE]G")


def test_parse_ptm_sequence_rejects_empty_one_letter_code(monkeypatch) -> None:
    ccd = {
        "BAD": {
            "_chem_comp.one_letter_code": [""],
        }
    }

    def _fake_cached_ccd(*args, **kwargs):
        return ccd

    monkeypatch.setattr(protomer_config_parser, "cached_ccd", _fake_cached_ccd)

    global_config.reset()
    global_config.init_const("nonstandard_aliases", {})
    global_config.init_const("user_ccd", None)

    with pytest.raises(ValueError, match="Invalid PTM one-letter code"):
        protomer_config_parser.parse_ptm_sequence("A[BAD]G")
