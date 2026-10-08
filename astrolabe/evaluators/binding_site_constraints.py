from __future__ import annotations

import logging
from collections import defaultdict
from typing import Mapping

from astrolabe.core.binding_site import (
    BindingSiteSpec,
    has_contact_pairs,
    resolve_pocket_constraint,
)
from astrolabe.core.chain_resolution import resolve_selector_chain_ids

LOGGER = logging.getLogger(__name__)


def _resolve_selector_candidates(
    selector: str,
    *,
    present_chains: set[str],
    context: str,
    role: str,
    oligo_name: str | None,
) -> list[str]:
    if oligo_name:
        candidates = [
            chain_id
            for chain_id in resolve_selector_chain_ids(oligo_name, selector)
            if chain_id in present_chains
        ]
    else:
        token = str(selector)
        candidates = [token] if token in present_chains else []

    if not candidates:
        LOGGER.info(
            "Skip binding_site constraint for '%s': unresolved %s selector %s against present chains=%s",
            context,
            role,
            selector,
            sorted(present_chains),
        )
        return []
    return candidates


def _require_single_binding_pair(
    *,
    binder_candidates: list[str],
    target_candidates: list[str],
    context: str,
    binder_selector: str,
    target_selector: str,
) -> tuple[str | None, str | None]:
    if not binder_candidates or not target_candidates:
        return None, None

    if len(binder_candidates) > 1 or len(target_candidates) > 1:
        raise ValueError(
            "binding_site in "
            f"{context!r}: selectors must resolve to exactly one chain for evaluator constraints. "
            f"Got binder={binder_selector!r}->{binder_candidates!r}, "
            f"target={target_selector!r}->{target_candidates!r}. "
            "Use explicit `chain:<ID>` selectors."
        )

    binder_chain = binder_candidates[0]
    target_chain = target_candidates[0]
    if binder_chain == target_chain:
        raise ValueError(
            "binding_site in "
            f"{context!r}: binder and target selectors resolved to the same chain "
            f"{binder_chain!r}. Use explicit `chain:<ID>` selectors."
        )

    return binder_chain, target_chain


def _resolve_pocket_constraint_for_pair(
    spec: BindingSiteSpec,
    *,
    binder_chain: str,
    target_chain: str,
) -> tuple[str, str, tuple[int, ...]]:
    pocket_spec = resolve_pocket_constraint(spec)
    pocket_binder_chain = (
        binder_chain if pocket_spec.binder_chain == spec.binder.chain else target_chain
    )
    pocket_contact_chain = (
        target_chain if pocket_spec.contact_chain == spec.target.chain else binder_chain
    )
    return pocket_binder_chain, pocket_contact_chain, pocket_spec.contact_positions


def invert_chain2input_chain_unique(
    chain2input_chain: Mapping[str, str],
    *,
    context: str,
) -> dict[str, str]:
    """Invert AF3->runner chain map, ensuring original chains map uniquely.

    Some runners (e.g. Boltz/Protenix adapters) may split a single ligand chain into
    multiple 'input' entities (e.g. multiple CCD codes). Inverting a 1->N mapping
    must not silently drop information.
    """
    orig_to_inputs: dict[str, list[str]] = defaultdict(list)
    for input_chain, orig_chain in chain2input_chain.items():
        orig_to_inputs[str(orig_chain)].append(str(input_chain))

    out: dict[str, str] = {}
    for orig_chain, inputs in orig_to_inputs.items():
        if len(inputs) != 1:
            raise ValueError(
                f"binding_site chain mapping is ambiguous for {context!r}: "
                f"original chain {orig_chain!r} maps to multiple input chains {sorted(inputs)!r}"
            )
        out[orig_chain] = inputs[0]
    return out


def _aa_at(*, seq_by_chain: Mapping[str, str], chain_id: str, pos: int) -> str:
    seq = seq_by_chain.get(str(chain_id))
    if seq is None:
        raise ValueError(f"Unknown chain {chain_id!r}")
    if pos <= 0 or pos > len(seq):
        raise ValueError(f"Residue position {pos} out of range (1..{len(seq)}) for chain {chain_id!r}")
    return seq[pos - 1]


def build_boltz_binding_site_constraints(
    spec: BindingSiteSpec,
    *,
    chain2input_chain: Mapping[str, str],
    context: str,
    oligo_name: str | None = None,
) -> list[dict[str, object]]:
    """Translate binding_site -> Boltz-1 input constraints.

    Boltz-1 supports pocket constraints but not explicit contact constraints.
    """
    orig_to_input = invert_chain2input_chain_unique(chain2input_chain, context=context)
    present = set(orig_to_input.keys())
    binder_candidates = _resolve_selector_candidates(
        spec.binder.chain,
        present_chains=present,
        context=context,
        role="binder",
        oligo_name=oligo_name,
    )
    target_candidates = _resolve_selector_candidates(
        spec.target.chain,
        present_chains=present,
        context=context,
        role="target",
        oligo_name=oligo_name,
    )
    binder_chain, target_chain = _require_single_binding_pair(
        binder_candidates=binder_candidates,
        target_candidates=target_candidates,
        context=context,
        binder_selector=spec.binder.chain,
        target_selector=spec.target.chain,
    )
    if binder_chain is None or target_chain is None:
        return []

    pocket_binder_chain, pocket_contact_chain, pocket_positions = (
        _resolve_pocket_constraint_for_pair(
            spec,
            binder_chain=binder_chain,
            target_chain=target_chain,
        )
    )
    binder = orig_to_input[pocket_binder_chain]
    contact_chain = orig_to_input[pocket_contact_chain]

    constraints: list[dict[str, object]] = []
    max_d = float(getattr(spec, "max_distance", 6.0))
    # Boltz-1 schema only supports max_distance == 6.0 for pocket constraints. If a
    # user sets a different value (often shared across evaluators), clamp it to 6.0
    # with an explicit warning rather than failing later with a vendored schema error.
    if max_d != 6.0:
        LOGGER.warning(
            "Boltz-1 pocket constraints only support max_distance=6.0; overriding %.3f -> 6.0 for %s.",
            max_d,
            context,
        )
        max_d = 6.0
    if spec.mode in {"both", "pocket"}:
        constraints.append(
            {
                "pocket": {
                    "binder": binder,
                    "contacts": [[contact_chain, int(pos)] for pos in pocket_positions],
                    "max_distance": max_d,
                    # Boltz defaults `force=false`, which only annotates the pocket
                    # and does not actively enforce it during inference. For
                    # binding_site constraints we want the pocket to take effect.
                    # See vendored docs: vendor/boltz/docs/prediction.md
                    "force": True,
                }
            }
        )

    if spec.mode in {"both", "contact"} and has_contact_pairs(spec):
        LOGGER.warning(
            "Boltz-1 does not support 'contact' constraints; ignoring binder.positions=%s target.positions=%s",
            list(spec.binder.positions),
            list(spec.target.positions),
        )

    return constraints


def build_protenix_binding_site_constraint(
    spec: BindingSiteSpec,
    *,
    chain_id_to_entities: Mapping[str, list[int]],
    context: str,
    oligo_name: str | None = None,
) -> dict[str, object] | None:
    """Translate binding_site -> Protenix input constraint dict."""
    present = set(str(k) for k in chain_id_to_entities.keys())
    binder_candidates = _resolve_selector_candidates(
        spec.binder.chain,
        present_chains=present,
        context=context,
        role="binder",
        oligo_name=oligo_name,
    )
    target_candidates = _resolve_selector_candidates(
        spec.target.chain,
        present_chains=present,
        context=context,
        role="target",
        oligo_name=oligo_name,
    )
    binder_chain, target_chain = _require_single_binding_pair(
        binder_candidates=binder_candidates,
        target_candidates=target_candidates,
        context=context,
        binder_selector=spec.binder.chain,
        target_selector=spec.target.chain,
    )
    if binder_chain is None or target_chain is None:
        return None

    binder_entities = chain_id_to_entities.get(binder_chain) or []
    target_entities = chain_id_to_entities.get(target_chain) or []
    if len(binder_entities) != 1:
        raise ValueError(
            f"binding_site binder chain {binder_chain!r} maps to {len(binder_entities)} entities "
            f"in {context!r}: {binder_entities!r}"
        )
    if len(target_entities) != 1:
        raise ValueError(
            f"binding_site target chain {target_chain!r} maps to {len(target_entities)} entities "
            f"in {context!r}: {target_entities!r}"
        )

    binder_entity = int(binder_entities[0])
    target_entity = int(target_entities[0])
    pocket_binder_chain, pocket_contact_chain, pocket_positions = (
        _resolve_pocket_constraint_for_pair(
            spec,
            binder_chain=binder_chain,
            target_chain=target_chain,
        )
    )
    pocket_binder_entity = binder_entity if pocket_binder_chain == binder_chain else target_entity
    pocket_contact_entity = target_entity if pocket_contact_chain == target_chain else binder_entity

    constraint: dict[str, object] = {}
    max_d = float(getattr(spec, "max_distance", 6.0))
    min_d = float(getattr(spec, "min_distance", 0.0))
    if spec.mode in {"both", "pocket"}:
        constraint["pocket"] = {
            "binder_chain": {"entity": pocket_binder_entity, "copy": 1},
            "contact_residues": [
                {"entity": pocket_contact_entity, "copy": 1, "position": int(pos)}
                for pos in pocket_positions
            ],
            "max_distance": max_d,
        }
    if spec.mode in {"both", "contact"} and has_contact_pairs(spec):
        constraint["contact"] = [
            {
                "entity1": binder_entity,
                "copy1": 1,
                "position1": int(bpos),
                "entity2": target_entity,
                "copy2": 1,
                "position2": int(tpos),
                "max_distance": max_d,
                "min_distance": min_d,
            }
            for bpos in spec.binder.positions
            for tpos in spec.target.positions
        ]

    return constraint or None


def build_chai_binding_site_rows(
    spec: BindingSiteSpec,
    *,
    seq_by_chain: Mapping[str, str],
    context: str,
    comment: str = "auto_binding_site",
    oligo_name: str | None = None,
) -> list[dict[str, object]]:
    """Translate binding_site -> Chai1 CSV restraint rows (without covalent bonds)."""
    present = set(str(k) for k in seq_by_chain.keys())
    binder_candidates = _resolve_selector_candidates(
        spec.binder.chain,
        present_chains=present,
        context=context,
        role="binder",
        oligo_name=oligo_name,
    )
    target_candidates = _resolve_selector_candidates(
        spec.target.chain,
        present_chains=present,
        context=context,
        role="target",
        oligo_name=oligo_name,
    )
    binder_chain, target_chain = _require_single_binding_pair(
        binder_candidates=binder_candidates,
        target_candidates=target_candidates,
        context=context,
        binder_selector=spec.binder.chain,
        target_selector=spec.target.chain,
    )
    if binder_chain is None or target_chain is None:
        return []

    rows: list[dict[str, object]] = []
    max_d = float(getattr(spec, "max_distance", 6.0))
    min_d = float(getattr(spec, "min_distance", 0.0))
    pocket_binder_chain, pocket_contact_chain, pocket_positions = (
        _resolve_pocket_constraint_for_pair(
            spec,
            binder_chain=binder_chain,
            target_chain=target_chain,
        )
    )
    if spec.mode in {"both", "pocket"}:
        for idx, tpos in enumerate(pocket_positions):
            rows.append(
                {
                    "chainA": pocket_binder_chain,
                    "res_idxA": "",
                    "chainB": pocket_contact_chain,
                    "res_idxB": f"{_aa_at(seq_by_chain=seq_by_chain, chain_id=pocket_contact_chain, pos=int(tpos))}{int(tpos)}",
                    "connection_type": "pocket",
                    "confidence": 1.0,
                    "min_distance_angstrom": min_d,
                    "max_distance_angstrom": max_d,
                    "comment": comment,
                    "restraint_id": f"restraint_{idx}",
                }
            )

    if spec.mode in {"both", "contact"} and has_contact_pairs(spec):
        offset = len(rows)
        k = 0
        for bpos in spec.binder.positions:
            for tpos in spec.target.positions:
                rows.append(
                    {
                        "chainA": binder_chain,
                        "res_idxA": f"{_aa_at(seq_by_chain=seq_by_chain, chain_id=binder_chain, pos=int(bpos))}{int(bpos)}",
                        "chainB": target_chain,
                        "res_idxB": f"{_aa_at(seq_by_chain=seq_by_chain, chain_id=target_chain, pos=int(tpos))}{int(tpos)}",
                        "connection_type": "contact",
                        "confidence": 1.0,
                        "min_distance_angstrom": min_d,
                        "max_distance_angstrom": max_d,
                        "comment": comment,
                        "restraint_id": f"restraint_{offset + k}",
                    }
                )
                k += 1

    return rows
