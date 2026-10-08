from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from astrolabe.core.folding_input import (
    DnaChain as CoreDnaChain,
    Input as CoreInput,
    Ligand as CoreLigand,
    ProteinChain as CoreProteinChain,
    RnaChain as CoreRnaChain,
    Template as CoreTemplate,
)
from astrolabe.evaluators.x_masking import preserve_unknown_residue_sequence


def _require_af3():
    try:
        from alphafold3.common import folding_input as af3_folding_input  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "AlphaFold3 evaluator features were requested, but the `alphafold3` python "
            "package (and its deps) is not available in this environment."
        ) from exc
    return af3_folding_input


def to_af3_template(t: Any):
    """Convert a lightweight Template (or dict) into an AF3 Template."""
    af3 = _require_af3()
    if isinstance(t, af3.Template):
        return t
    if isinstance(t, CoreTemplate):
        return af3.Template(mmcif=t.mmcif, query_to_template_map=dict(t.query_to_template_map))
    if isinstance(t, dict):
        mmcif = t.get("mmcif")
        q = t.get("queryIndices") or []
        ti = t.get("templateIndices") or []
        q2t = dict(zip(q, ti))
        return af3.Template(mmcif=mmcif, query_to_template_map=q2t)
    raise TypeError(f"Unsupported template type: {type(t)}")


def to_af3_chain(chain: Any):
    """Convert a lightweight chain to an AF3 folding_input chain."""
    af3 = _require_af3()
    if isinstance(
        chain,
        (
            af3.ProteinChain,
            af3.RnaChain,
            af3.DnaChain,
            af3.Ligand,
        ),
    ):
        return chain

    if isinstance(chain, CoreProteinChain):
        templates = chain.templates or []
        return af3.ProteinChain(
            id=chain.id,
            sequence=preserve_unknown_residue_sequence(
                chain.sequence, evaluator="alphafold3"
            ),
            ptms=list(chain.ptms),
            description=chain.description,
            unpaired_msa=chain.unpaired_msa or "",
            paired_msa=chain.paired_msa or "",
            templates=[to_af3_template(t) for t in templates],
        )
    if isinstance(chain, CoreRnaChain):
        return af3.RnaChain(
            id=chain.id,
            sequence=chain.sequence,
            modifications=list(chain.modifications),
            unpaired_msa=chain.unpaired_msa or "",
        )
    if isinstance(chain, CoreDnaChain):
        return af3.DnaChain(
            id=chain.id,
            sequence=chain.sequence,
            modifications=list(chain.modifications),
        )
    if isinstance(chain, CoreLigand):
        if chain.smiles is not None:
            return af3.Ligand(id=chain.id, smiles=chain.smiles)
        if chain.ccd_ids is not None:
            return af3.Ligand(id=chain.id, ccd_ids=list(chain.ccd_ids))
        raise ValueError("Ligand must have either smiles or ccd_ids")
    raise TypeError(f"Unsupported chain type: {type(chain)}")


def to_af3_input(inp: Any):
    """Convert a lightweight Input into an AF3 folding_input.Input."""
    af3 = _require_af3()
    if isinstance(inp, af3.Input):
        return inp
    if not isinstance(inp, CoreInput):
        raise TypeError(f"Unsupported input type: {type(inp)}")
    return af3.Input(
        name=inp.name,
        chains=[to_af3_chain(c) for c in inp.chains],
        rng_seeds=list(inp.rng_seeds),
        bonded_atom_pairs=inp.bonded_atom_pairs,
        user_ccd=inp.user_ccd,
    )


def has_all_x_protein_chains(fold_input: Any) -> bool:
    """Whether an AF3 input contains any protein chain made entirely of `X`."""

    af3 = _require_af3()
    if not isinstance(fold_input, af3.Input):
        raise TypeError(f"Expected an AF3 Input, got {type(fold_input)}")
    for chain in getattr(fold_input, "protein_chains", []) or []:
        seq = (getattr(chain, "sequence", "") or "").upper()
        if seq and set(seq) == {"X"}:
            return True
    return False


@contextmanager
def allow_all_x_protein_chains_during_featurisation(fold_input: Any):
    """Patch AF3 cleaning so intentional all-`X` chains are not dropped.

    AlphaFold3's structure cleaning removes polymer chains whose full sequence is
    entirely unknown (`UNK`), which turns an all-`X` design seed into
    `No chains in structure!`. Protein-Hunter-style design treats all-`X`
    sequences as intentional masked initialization, so for this narrow case we
    keep those chains during featurisation.
    """

    if not has_all_x_protein_chains(fold_input):
        yield
        return

    from alphafold3.model.pipeline import structure_cleaning

    original = structure_cleaning.clean_structure

    def patched_clean_structure(*args, **kwargs):
        kwargs["drop_missing_sequence"] = False
        return original(*args, **kwargs)

    structure_cleaning.clean_structure = patched_clean_structure
    try:
        yield
    finally:
        structure_cleaning.clean_structure = original
