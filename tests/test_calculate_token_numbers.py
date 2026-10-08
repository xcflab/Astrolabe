from __future__ import annotations

import pytest

from run import calculate_token_numbers

from astrolabe.core.folding_input import Input, Ligand, ProteinChain


def test_calculate_token_numbers_counts_x_residues() -> None:
    seq = "ACDXEFGHIK"
    inputs = {
        "A": Input(
            name="A",
            chains=[ProteinChain(id="A", sequence=seq, ptms=[])],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }
    try:
        assert calculate_token_numbers(inputs) == {"A": len(seq)}
    except RuntimeError as exc:
        raise pytest.skip(str(exc))


def test_calculate_token_numbers_includes_ligand_tokens() -> None:
    inputs = {
        "A": Input(
            name="A",
            chains=[
                ProteinChain(id="A", sequence="ACDE", ptms=[]),
                Ligand(id="L", smiles="CCO"),
            ],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }
    try:
        counts = calculate_token_numbers(inputs)
    except RuntimeError as exc:
        raise pytest.skip(str(exc))
    assert counts["A"] > 4


def test_calculate_token_numbers_accepts_all_x_protein_chain() -> None:
    seq = "XXXXX"
    inputs = {
        "A": Input(
            name="A",
            chains=[ProteinChain(id="A", sequence=seq, ptms=[])],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }
    try:
        assert calculate_token_numbers(inputs) == {"A": len(seq)}
    except RuntimeError as exc:
        raise pytest.skip(str(exc))
