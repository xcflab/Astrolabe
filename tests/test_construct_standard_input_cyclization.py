from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.utils.base import construct_standard_input


@dataclass
class _ProtomerStub:
    sequence: str
    mutated_sequence: str
    type: ProtomerTypes = ProtomerTypes.PROTEIN
    ptm_dict: dict[int, str] = field(default_factory=dict)
    homology: dict[str, object] = field(default_factory=dict)


def _init_test_state(*, cyclization: dict, protomers: dict[str, _ProtomerStub]) -> None:
    global_config.reset()
    global_config.init_const("protomers", protomers)
    global_config.init_const("cyclization", cyclization)


def _init_test_state_with_covalent_bonds(
    *,
    protomers: dict[str, _ProtomerStub],
    covalent_bonds: list[dict[str, object]],
) -> None:
    global_config.reset()
    global_config.init_const("protomers", protomers)
    global_config.init_const("cyclization", None)
    global_config.init_const("covalent_bonds", covalent_bonds)


@pytest.mark.parametrize(
    ("signal_mode", "expect_bonds", "expect_chain_ids"),
    [
        ("none", False, False),
        ("index_only", False, True),
        ("bond_only", True, False),
        ("both", True, True),
    ],
)
def test_construct_standard_input_emits_head_to_tail_signals_by_signal_mode(
    signal_mode: str,
    expect_bonds: bool,
    expect_chain_ids: bool,
) -> None:
    _init_test_state(
        cyclization={"head_to_tail": True, "signal_mode": signal_mode},
        protomers={"A": _ProtomerStub(sequence="ACDE", mutated_sequence="ACDE")},
    )

    result = construct_standard_input(["A"], seed=7)["A"]

    if expect_bonds:
        assert result.bonded_atom_pairs == [(("A", 1, "N"), ("A", 4, "C"))]
        assert result.bonded_atom_pair_meta == [
            {
                "order": "single",
                "target_distance": pytest.approx(1.33),
                "is_head_to_tail": True,
            }
        ]
    else:
        assert result.bonded_atom_pairs is None
        assert result.bonded_atom_pair_meta is None

    if expect_chain_ids:
        assert result.cyclic_chain_ids == ("A",)
    else:
        assert result.cyclic_chain_ids is None

    assert result.cyclic_relative_patches is None


@pytest.mark.parametrize(
    ("signal_mode", "expect_bonds", "expect_patches"),
    [
        ("none", False, False),
        ("index_only", False, True),
        ("bond_only", True, False),
        ("both", True, True),
    ],
)
def test_construct_standard_input_emits_same_chain_endpoint_patch_signals_by_signal_mode(
    signal_mode: str,
    expect_bonds: bool,
    expect_patches: bool,
) -> None:
    _init_test_state(
        cyclization={
            "signal_mode": signal_mode,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                    "order": "single",
                    "offset_window": 2,
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGH", mutated_sequence="ABCDEFGH")},
    )

    result = construct_standard_input(["A"], seed=11)["A"]

    if expect_bonds:
        assert result.bonded_atom_pairs == [(("A", 2, "CA"), ("A", 7, "CB"))]
    else:
        assert result.bonded_atom_pairs is None

    assert result.cyclic_chain_ids is None

    if expect_patches:
        assert result.cyclic_relative_patches is not None
        assert len(result.cyclic_relative_patches) == 1
        patch = result.cyclic_relative_patches[0]
        assert patch.chain_id == "A"
        assert [(entry.residue_i, entry.residue_j, entry.offset) for entry in patch.offsets] == [
            (2, 6, 2),
            (2, 7, 1),
            (3, 6, -3),
            (3, 7, 2),
            (6, 2, -2),
            (6, 3, 3),
            (7, 2, -1),
            (7, 3, -2),
        ]
    else:
        assert result.cyclic_relative_patches is None


@pytest.mark.parametrize("signal_mode", ["index_only", "both"])
def test_construct_standard_input_never_emits_relative_patches_for_cross_chain_bonds(
    signal_mode: str,
) -> None:
    _init_test_state(
        cyclization={
            "signal_mode": signal_mode,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "B", "pos": 3, "atom": "CB"},
                    "order": "single",
                }
            ],
        },
        protomers={
            "A": _ProtomerStub(sequence="ACDE", mutated_sequence="ACDE"),
            "B": _ProtomerStub(sequence="FGHI", mutated_sequence="FGHI"),
        },
    )

    result = construct_standard_input(["AB"], seed=13)["AB"]

    assert result.cyclic_chain_ids is None
    assert result.cyclic_relative_patches is None
    if signal_mode == "both":
        assert result.bonded_atom_pairs == [(("A", 2, "CA"), ("B", 3, "CB"))]
    else:
        assert result.bonded_atom_pairs is None


def test_construct_standard_input_can_emit_index_from_explicit_canonical_bond_with_generic_bond_only(
) -> None:
    _init_test_state(
        cyclization={
            "signal_mode": "bond_only",
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 1, "atom": "N"},
                    "b": {"proto": "A", "pos": 5, "atom": "C"},
                    "order": "single",
                    "target_distance": 1.33,
                    "signal_mode": "both",
                },
                {
                    "a": {"proto": "A", "pos": 2, "atom": "SG"},
                    "b": {"proto": "A", "pos": 5, "atom": "SG"},
                    "order": "single",
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ACDEC", mutated_sequence="ACDEC")},
    )

    result = construct_standard_input(["A"], seed=23)["A"]

    assert result.cyclic_chain_ids == ("A",)
    assert result.cyclic_relative_patches is None
    assert result.bonded_atom_pairs == [
        (("A", 1, "N"), ("A", 5, "C")),
        (("A", 2, "SG"), ("A", 5, "SG")),
    ]
    assert result.bonded_atom_pair_meta == [
        {
            "order": "single",
            "target_distance": pytest.approx(1.33),
            "is_head_to_tail": True,
        },
        {
            "order": "single",
            "target_distance": pytest.approx(2.03),
            "is_head_to_tail": False,
        },
    ]


@pytest.mark.parametrize("signal_mode", ["none", "index_only"])
def test_construct_standard_input_skips_generic_order_validation_without_bond_signals(
    signal_mode: str,
) -> None:
    _init_test_state(
        cyclization={
            "signal_mode": signal_mode,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGH", mutated_sequence="ABCDEFGH")},
    )

    result = construct_standard_input(["A"], seed=17)["A"]

    assert result.bonded_atom_pairs is None
    assert result.bonded_atom_pair_meta is None
    assert result.cyclic_chain_ids is None
    if signal_mode == "index_only":
        assert result.cyclic_relative_patches is not None
        assert len(result.cyclic_relative_patches) == 1
    else:
        assert result.cyclic_relative_patches is None


@pytest.mark.parametrize("signal_mode", ["bond_only", "both"])
def test_construct_standard_input_requires_generic_order_when_bond_signals_enabled(
    signal_mode: str,
) -> None:
    _init_test_state(
        cyclization={
            "signal_mode": signal_mode,
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "CA"},
                    "b": {"proto": "A", "pos": 7, "atom": "CB"},
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGH", mutated_sequence="ABCDEFGH")},
    )

    with pytest.raises(ValueError, match="must define `order` explicitly"):
        construct_standard_input(["A"], seed=19)


def test_construct_standard_input_supports_anchor_offset_positions() -> None:
    _init_test_state(
        cyclization={
            "signal_mode": "both",
            "bonds": [
                {
                    "a": {"proto": "A", "pos": "nterm+4", "atom": "NZ"},
                    "b": {"proto": "A", "pos": "cterm-3", "atom": "CG"},
                    "order": "single",
                    "offset_window": 1,
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGHIJKL", mutated_sequence="ABCDEFGHIJKL")},
    )

    result = construct_standard_input(["A"], seed=23)["A"]

    assert result.bonded_atom_pairs == [(("A", 5, "NZ"), ("A", 9, "CG"))]
    assert result.cyclic_relative_patches is not None
    assert len(result.cyclic_relative_patches) == 1


def test_construct_standard_input_emits_top_level_covalent_bonds() -> None:
    _init_test_state_with_covalent_bonds(
        protomers={
            "A": _ProtomerStub(sequence="ACDE", mutated_sequence="ACDE"),
            "B": _ProtomerStub(
                sequence="R13",
                mutated_sequence="R13",
                type=ProtomerTypes.CCD,
            ),
        },
        covalent_bonds=[
            {
                "atom1": {"chain_id": "A", "resseq": 2, "atom_name": "SG"},
                "atom2": {"chain_id": "B", "resseq": 1, "atom_name": "C1"},
            }
        ],
    )

    result = construct_standard_input(["AB"], seed=29)["AB"]

    assert result.bonded_atom_pairs == [(("A", 2, "SG"), ("B", 1, "C1"))]


def test_construct_standard_input_highfold_cycpoem_encoding_emits_full_matrix_patches() -> None:
    _init_test_state(
        cyclization={
            "signal_mode": "both",
            "encoding": {"mode": "highfold_cycpoem", "scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "SG"},
                    "b": {"proto": "A", "pos": 7, "atom": "SG"},
                    "order": "single",
                },
                {
                    "a": {"proto": "A", "pos": 4, "atom": "SG"},
                    "b": {"proto": "A", "pos": 8, "atom": "SG"},
                    "order": "single",
                },
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGH", mutated_sequence="ABCDEFGH")},
    )

    result = construct_standard_input(["A"], seed=29)["A"]

    assert result.bonded_atom_pairs == [
        (("A", 2, "SG"), ("A", 7, "SG")),
        (("A", 4, "SG"), ("A", 8, "SG")),
    ]
    assert result.cyclic_relative_patches is not None
    assert len(result.cyclic_relative_patches) == 1
    patch = result.cyclic_relative_patches[0]
    assert len(patch.offsets) == 64
    by_pair = {
        (entry.residue_i, entry.residue_j): entry.offset for entry in patch.offsets
    }
    assert by_pair[(1, 8)] == -1
    assert by_pair[(2, 7)] == -1
    assert by_pair[(7, 2)] == 1


def test_construct_standard_input_rfpeptide_like_encoding_emits_global_interval_patch() -> None:
    _init_test_state(
        cyclization={
            "signal_mode": "both",
            "encoding": {"mode": "rfpeptide_like", "scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": 2, "atom": "SG"},
                    "b": {"proto": "A", "pos": 7, "atom": "SG"},
                    "order": "single",
                    "offset_window": 2,
                }
            ],
        },
        protomers={"A": _ProtomerStub(sequence="ABCDEFGH", mutated_sequence="ABCDEFGH")},
    )

    result = construct_standard_input(["A"], seed=31)["A"]

    assert result.bonded_atom_pairs == [(("A", 2, "SG"), ("A", 7, "SG"))]
    assert result.cyclic_relative_patches is not None
    assert len(result.cyclic_relative_patches) == 1
    patch = result.cyclic_relative_patches[0]
    assert len(patch.offsets) == 36
    by_pair = {
        (entry.residue_i, entry.residue_j): entry.offset for entry in patch.offsets
    }
    assert by_pair[(2, 7)] == 1
    assert by_pair[(7, 2)] == -1
    assert by_pair[(2, 5)] == -3
    assert by_pair[(5, 2)] == 3
    assert (1, 8) not in by_pair
