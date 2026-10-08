from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


_SUPPORTED_EVALUATORS = frozenset({"alphafold3", "boltz", "protenix", "chai1", "opendde"})


def preserve_unknown_residue_sequence(sequence: str, *, evaluator: str) -> str:
    """Keep protein sequences with `X` length-stable across evaluator adapters.

    AF3, Boltz, Protenix, Chai, and OpenDDE all have native unknown-residue encodings, so
    the adapter layer should pass the sequence through unchanged instead of
    dropping masked positions.
    """

    if evaluator not in _SUPPORTED_EVALUATORS:
        raise ValueError(
            f"Unsupported evaluator {evaluator!r}; expected one of {sorted(_SUPPORTED_EVALUATORS)!r}."
        )
    return str(sequence or "")


def mask_unknown_residues_for_chai_esm(sequence: str) -> str:
    """Map `X` residues to Chai's ESM `<mask>` token without changing token count."""

    return "".join(
        "<mask>" if residue == "X" else residue for residue in str(sequence or "")
    )


def chai_esm_token_count(sequence: str) -> int:
    from chai_lab.data.dataset.embeddings import esm as esm_mod

    return len(esm_mod.esm_tokenizer.tokenize(str(sequence or "")))


def build_chai_masked_esm_embedding_context(chains: list[Any], device) -> Any:
    """Build Chai ESM embeddings while treating `X` residues as mask tokens.

    Protein-Hunter replaces `X` with ESM's `<mask>` token for Chai embeddings so
    masked positions stay in the sequence length while remaining explicitly
    unknown to the language model. Reimplement that behavior against the current
    chai_lab API without touching the rest of the Chai inference pipeline.
    """

    import torch
    from chai_lab.data.dataset.embeddings import esm as esm_mod
    from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
    from chai_lab.data.parsing.structure.entity_type import EntityType

    protein_sequences = {
        mask_unknown_residues_for_chai_esm(chain.entity_data.sequence)
        for chain in chains
        if getattr(chain.entity_data, "entity_type", None) == EntityType.PROTEIN
    }

    seq_to_embedding_context: dict[str, EmbeddingContext] = {}
    if protein_sequences:
        with torch.no_grad():
            with esm_mod.esm_model(device=device) as model:
                for seq in protein_sequences:
                    token_ids = torch.asarray(
                        esm_mod.esm_tokenizer.tokenize(f"<cls>{seq}<eos>")
                    )[None, :].to(device)
                    last_hidden_state = model(tokens=token_ids)
                    esm_embeddings = last_hidden_state[0, 1:-1].float().to("cpu")
                    expected_tokens = chai_esm_token_count(seq)
                    if esm_embeddings.shape[0] != expected_tokens:
                        raise ValueError(
                            "Chai ESM embedding/token count mismatch for masked sequence "
                            f"{seq!r}: embeddings={esm_embeddings.shape[0]} tokens={expected_tokens}."
                        )
                    seq_to_embedding_context[seq] = EmbeddingContext(
                        esm_embeddings=esm_embeddings
                    )

    chain_embeddings = []
    for chain in chains:
        if getattr(chain.entity_data, "entity_type", None) == EntityType.PROTEIN:
            masked_sequence = mask_unknown_residues_for_chai_esm(
                chain.entity_data.sequence
            )
            chain_embeddings.append(seq_to_embedding_context[masked_sequence])
        else:
            chain_embeddings.append(
                EmbeddingContext.empty(n_tokens=chain.structure_context.num_tokens)
            )

    exploded_embeddings = [
        embedding.esm_embeddings[chain.structure_context.token_residue_index, :]
        for embedding, chain in zip(chain_embeddings, chains, strict=True)
    ]
    merged_embeddings = torch.cat(exploded_embeddings, dim=0)
    return EmbeddingContext(esm_embeddings=merged_embeddings)


@contextmanager
def patch_chai1_get_esm_embedding_context(chai1_mod: Any) -> Iterator[None]:
    """Temporarily patch chai1 to mask `X` residues during ESM embedding."""

    original = getattr(chai1_mod, "get_esm_embedding_context", None)
    if original is None:
        yield
        return

    chai1_mod.get_esm_embedding_context = build_chai_masked_esm_embedding_context
    try:
        yield
    finally:
        chai1_mod.get_esm_embedding_context = original
