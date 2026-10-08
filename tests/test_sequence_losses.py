from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes


def _monkeypatch_global_const(monkeypatch, name: str, value) -> None:
    monkeypatch.setattr(global_config, f"_{name}", value, raising=True)
    monkeypatch.setitem(global_config._initialized_constants, name, True)


def _protein_protomer(*, sequence: str, mutated_sequence) -> SimpleNamespace:
    return SimpleNamespace(
        type=ProtomerTypes.PROTEIN,
        sequence=sequence,
        mutated_sequence=mutated_sequence,
    )


class _HashableEvalResult:
    names = ("ABC",)


def test_sequence_residue_fraction_counts_x_from_current_protomer_sequence(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAAA", mutated_sequence="AXXXX"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    score = sequence_residue_fraction(
        eval_result={},
        included_protomers="A",
        residues="X",
    )

    assert score == 0.8


def test_sequence_residue_fraction_supports_multiple_residues(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="WAAA"),
            "B": _protein_protomer(sequence="BBBB", mutated_sequence="XXAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    score = sequence_residue_fraction(
        eval_result={},
        included_protomers="A, B",
        residues="W,X",
    )

    assert score == 0.375


def test_sequence_residue_fraction_rejects_non_protein_protomer(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.SMILES,
                sequence="CCO",
                mutated_sequence="CCO",
            ),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="protein protomer"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


@pytest.mark.parametrize("mutated_sequence", [None, 123, "", "   "])
def test_sequence_residue_fraction_rejects_invalid_mutated_sequence(
    monkeypatch, mutated_sequence
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence=mutated_sequence),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="mutated_sequence"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


def test_sequence_residue_fraction_rejects_missing_mutated_sequence_attribute(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                sequence="AAAA",
            ),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="mutated_sequence"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


@pytest.mark.parametrize("residues", ["ALA", "X,ALA", ""])
def test_sequence_residue_fraction_rejects_malformed_residue_tokens(
    monkeypatch, residues: str
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AXXX"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="residue"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues=residues,
        )


def test_sequence_region_identity_returns_one_for_identical_region(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B,C",
        positions="1-4",
    )

    assert score == 1.0


def test_sequence_region_identity_uses_mean_pairwise_identity(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AACC"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="BBAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B,C",
        positions="1-4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_region_identity_parses_ranges_and_singletons(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AXCY"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B",
        positions="2-3,4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_edge_region_identity_compares_directed_edge_barcodes(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="WXYZ"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="ABYZ"),
        },
    )

    from astrolabe.losses.sequence import sequence_edge_region_identity

    score = sequence_edge_region_identity(
        eval_result={},
        edges="A>B,B>C,C>A",
        m_positions="1-2",
        n_positions="3-4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_edge_region_identity_rejects_duplicate_edges(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="BBBB"),
        },
    )

    from astrolabe.losses.sequence import sequence_edge_region_identity

    with pytest.raises(ValueError, match="duplicates"):
        sequence_edge_region_identity(
            eval_result={},
            edges="A>B,A>B",
            m_positions="1",
            n_positions="2",
        )


@pytest.mark.parametrize("positions", ["", "0", "2-1", "A", "1-"])
def test_sequence_region_identity_rejects_invalid_positions(
    monkeypatch, positions: str
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="position"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions=positions,
        )


def test_sequence_region_identity_rejects_out_of_range_positions(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="outside protomer length"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions="1,5",
        )


def test_sequence_region_identity_rejects_unequal_protomer_lengths(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAA", mutated_sequence="AAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="equal-length"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions="1-3",
        )


def test_sequence_region_identity_works_through_loss_engine(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AACC"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="BBAA"),
        },
    )

    from astrolabe.losses.base import LossEngine

    loss_config = {
        "loss_terms": {
            "ABC_patch_barcode_identity": {
                "_type": "sequence_region_identity",
                "included_protomers": "A,B,C",
                "positions": "1-4",
            },
        },
        "expression_terms": {
            "ABC_patch_barcode_score": (
                "max(0.0, 1.0 - abs(ABC_patch_barcode_identity - 0.35) / 0.35)"
            ),
        },
    }
    engine = LossEngine(loss_config)
    engine.setup(
        weighted_score_expression="ABC_patch_barcode_score",
        early_stopping=None,
        acceptance_threshold=None,
    )

    score, env = engine.compute_loss(
        eval_result=_HashableEvalResult(),
        runtime_config={},
    )

    assert env["ABC_patch_barcode_identity"] == pytest.approx(1.0 / 3.0)
    assert score == pytest.approx(1.0 - abs((1.0 / 3.0) - 0.35) / 0.35)
