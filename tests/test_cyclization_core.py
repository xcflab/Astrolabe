from __future__ import annotations

import pytest

from astrolabe.core.cyclization import (
    expand_cyclization_bonds_for_oligo,
    filter_expanded_bonds_for_encoding,
    parse_cyclization,
    resolve_pos,
    resolve_cyclization_bond_specs,
)


def test_parse_cyclization_signal_mode_defaults_to_both() -> None:
    spec = parse_cyclization({"head_to_tail": True})
    assert spec is not None
    assert spec.signal_mode == "both"


@pytest.mark.parametrize("signal_mode", ["none", "index_only", "bond_only", "both"])
def test_parse_cyclization_signal_mode_accepts_supported_values(
    signal_mode: str,
) -> None:
    spec = parse_cyclization(
        {
            "head_to_tail": True,
            "signal_mode": signal_mode,
        }
    )
    assert spec is not None
    assert spec.signal_mode == signal_mode


def test_parse_cyclization_signal_mode_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid cyclization.signal_mode='bad'",
    ):
        parse_cyclization({"head_to_tail": True, "signal_mode": "bad"})


def test_resolve_cyclization_bond_specs_requires_order_for_generic() -> None:
    spec = parse_cyclization(
        {
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                }
            ]
        }
    )
    assert spec is not None
    with pytest.raises(ValueError, match="must define `order` explicitly"):
        resolve_cyclization_bond_specs(
            spec,
            protomer_lengths={"A": 8},
            require_order_for_generic=True,
        )


def test_resolve_expand_filter_defaults_for_head_to_tail_scope() -> None:
    spec = parse_cyclization(
        {
            "head_to_tail": True,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                    "order": "double",
                }
            ],
        }
    )
    assert spec is not None

    resolved = resolve_cyclization_bond_specs(
        spec,
        protomer_lengths={"A": 8},
        require_order_for_generic=True,
    )
    assert len(resolved) == 2
    generic = [bond for bond in resolved if not bond.is_head_to_tail][0]
    assert generic.order == "double"
    assert generic.target_distance == pytest.approx(1.34)

    expanded = expand_cyclization_bonds_for_oligo(resolved, oligo_name="AA")
    assert len(expanded) == 4
    filtered = filter_expanded_bonds_for_encoding(expanded, encoding=spec.encoding)
    assert len(filtered) == 2
    assert all(b.is_head_to_tail for b in filtered)


def test_filter_expanded_bonds_for_all_bonds_scope() -> None:
    spec = parse_cyclization(
        {
            "head_to_tail": True,
            "encoding": {"scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                    "order": "single",
                }
            ],
        }
    )
    assert spec is not None
    resolved = resolve_cyclization_bond_specs(
        spec,
        protomer_lengths={"A": 8},
        require_order_for_generic=True,
    )
    expanded = expand_cyclization_bonds_for_oligo(resolved, oligo_name="AA")
    filtered = filter_expanded_bonds_for_encoding(expanded, encoding=spec.encoding)
    assert len(filtered) == 4


@pytest.mark.parametrize(
    ("raw_pos", "length", "expected"),
    [
        ("nterm+4", 12, 5),
        ("NTERM+4", 12, 5),
        ("cterm-4", 12, 8),
        ("c_term-4", 12, 8),
        (" begin +2 ", 12, 3),
        ("last-3", 12, 9),
    ],
)
def test_resolve_pos_accepts_terminal_anchor_offsets(
    raw_pos: str,
    length: int,
    expected: int,
) -> None:
    assert resolve_pos(raw_pos, length=length) == expected
