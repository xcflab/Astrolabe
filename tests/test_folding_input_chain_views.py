from __future__ import annotations

from astrolabe.core.folding_input import (
    DnaChain,
    Input,
    Ligand,
    ProteinChain,
    RnaChain,
)


def test_input_chain_views_partition_chains() -> None:
    inp = Input(
        name="ABCD",
        chains=[
            ProteinChain(id="A", sequence="ACDX", ptms=[]),
            RnaChain(id="B", sequence="ACGU", modifications=[]),
            DnaChain(id="C", sequence="ACGT", modifications=[]),
            Ligand(id="D", smiles="CCO"),
        ],
        rng_seeds=[0],
    )
    assert [c.id for c in inp.protein_chains] == ["A"]
    assert [c.id for c in inp.rna_chains] == ["B"]
    assert [c.id for c in inp.dna_chains] == ["C"]
    assert [c.id for c in inp.ligands] == ["D"]
