from __future__ import annotations

import pytest

from astrolabe.core.chain_resolution import (
    chain_ids_for_oligo,
    protomer_to_chain_ids,
    resolve_selector_chain_ids,
    resolve_single_chain_id,
)


def test_protomer_to_chain_ids_keeps_all_repeated_copies() -> None:
    assert chain_ids_for_oligo("AAB") == ["A", "B", "C"]
    assert protomer_to_chain_ids("AAB") == {"A": ["A", "B"], "B": ["C"]}


def test_selector_resolution_is_chain_first_with_explicit_protomer_override() -> None:
    assert resolve_selector_chain_ids("AAB", "A") == ["A"]
    assert resolve_selector_chain_ids("AAB", "B") == ["B"]
    assert resolve_selector_chain_ids("AAB", "chain:B") == ["B"]
    assert resolve_selector_chain_ids("AAB", "C") == ["C"]
    assert resolve_selector_chain_ids("AAB", "protomer:A") == ["A", "B"]
    assert resolve_selector_chain_ids("AAB", "protomer:B") == ["C"]


def test_resolve_single_chain_rejects_ambiguous_explicit_protomer_selector() -> None:
    with pytest.raises(ValueError, match="matches multiple chains"):
        _ = resolve_single_chain_id("AAB", "protomer:A", context="unit-test")
