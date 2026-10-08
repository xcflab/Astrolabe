from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import parse_predefined_statement


def test_predefined_statement_does_not_leak_across_global_config_reset() -> None:
    global_config.reset()

    cfg = OmegaConf.create({"predefined_statement": {"x": "AAAA"}})
    parse_predefined_statement(cfg)
    assert "x" in global_config.predefined_statement

    global_config.reset()
    with pytest.raises(RuntimeError):
        _ = global_config.predefined_statement

