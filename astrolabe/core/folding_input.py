from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from astrolabe.core.cyclization import CyclicRelativePatch


@dataclass(frozen=True, slots=True, kw_only=True)
class Template:
    """Lightweight template container (AF3-compatible shape, no AF3 deps)."""

    mmcif: str | None
    query_to_template_map: Mapping[int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProteinChain:
    """Lightweight protein chain container.

    Mirrors the subset of AlphaFold3's `ProteinChain` interface used by this repo.
    """

    id: str
    sequence: str
    ptms: Sequence[tuple[str, int]]
    description: str | None = None
    unpaired_msa: str | None = None
    paired_msa: str | None = None
    templates: Sequence[Template] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RnaChain:
    id: str
    sequence: str
    modifications: Sequence[tuple[str, int]]
    unpaired_msa: str | None = None
    paired_msa: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DnaChain:
    id: str
    sequence: str
    modifications: Sequence[tuple[str, int]]
    unpaired_msa: str | None = None
    paired_msa: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Ligand:
    id: str
    smiles: str | None = None
    ccd_ids: Sequence[str] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Input:
    """Lightweight folding input container (AF3-compatible shape, no AF3 deps)."""

    name: str
    chains: Sequence[ProteinChain | RnaChain | DnaChain | Ligand]
    rng_seeds: Sequence[int]
    bonded_atom_pairs: Sequence[Any] | None = None
    cyclic_chain_ids: Sequence[str] | None = None
    cyclic_relative_patches: Sequence[CyclicRelativePatch] | None = None
    bonded_atom_pair_meta: Sequence[Any] | None = None
    user_ccd: str | None = None

    @property
    def protein_chains(self) -> list[ProteinChain]:
        return [c for c in self.chains if isinstance(c, ProteinChain)]

    @property
    def rna_chains(self) -> list[RnaChain]:
        return [c for c in self.chains if isinstance(c, RnaChain)]

    @property
    def dna_chains(self) -> list[DnaChain]:
        return [c for c in self.chains if isinstance(c, DnaChain)]

    @property
    def ligands(self) -> list[Ligand]:
        return [c for c in self.chains if isinstance(c, Ligand)]
