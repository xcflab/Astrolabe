from __future__ import annotations

from collections import defaultdict

from astrolabe.utils.mmcif_ids import int_id_to_str_id


CHAIN_PREFIX = "chain:"
PROTOMER_PREFIX = "protomer:"


def chain_ids_for_oligo(oligo_name: str) -> list[str]:
    """Return ordered mmCIF chain IDs for an oligomer string."""
    oligo = str(oligo_name)
    return [int_id_to_str_id(i + 1) for i in range(len(oligo))]


def chain_to_protomer(oligo_name: str) -> dict[str, str]:
    """Return mapping: complex chain ID -> protomer name."""
    oligo = str(oligo_name)
    return {int_id_to_str_id(i + 1): proto for i, proto in enumerate(oligo)}


def protomer_to_chain_ids(oligo_name: str) -> dict[str, list[str]]:
    """Return mapping: protomer name -> all complex chain IDs (ordered copies)."""
    out: dict[str, list[str]] = defaultdict(list)
    for chain_id, proto in chain_to_protomer(oligo_name).items():
        out[proto].append(chain_id)
    return dict(out)


def _normalize_selector(selector: str | None) -> str:
    return str(selector or "").strip()


def _strip_prefix(token: str, prefix: str) -> str:
    return token[len(prefix) :].strip()


def resolve_selector_chain_ids(
    oligo_name: str,
    selector: str | None,
) -> list[str]:
    """Resolve selector token to one-or-more chain IDs with chain-first semantics.

    Resolution order:
      1) `chain:<ID>` explicit chain selector (strict; no fallback).
      2) `protomer:<P>` explicit protomer selector.
      3) default token as raw chain ID.
      4) backward-compat fallback: token as protomer selector (only if #3 had no match).
    """
    token = _normalize_selector(selector)
    if not token:
        return []

    chain_map = chain_to_protomer(oligo_name)
    prot_map = protomer_to_chain_ids(oligo_name)

    lowered = token.lower()
    if lowered.startswith(CHAIN_PREFIX):
        chain_id = _strip_prefix(token, CHAIN_PREFIX).upper()
        return [chain_id] if chain_id in chain_map else []

    if lowered.startswith(PROTOMER_PREFIX):
        proto = _strip_prefix(token, PROTOMER_PREFIX)
        return list(prot_map.get(proto, []))

    # Default: treat token as raw chain id to preserve historic selector semantics.
    chain_token = token.upper()
    if chain_token in chain_map:
        return [chain_token]

    proto_matches = prot_map.get(token)
    if proto_matches:
        return list(proto_matches)
    return []


def resolve_single_chain_id(
    oligo_name: str,
    selector: str | None,
    *,
    context: str,
) -> str | None:
    """Resolve selector to exactly one chain ID, raising on ambiguity."""
    chain_ids = resolve_selector_chain_ids(oligo_name, selector)
    if not chain_ids:
        return None
    if len(chain_ids) > 1:
        raise ValueError(
            f"{context}: selector {selector!r} matches multiple chains {chain_ids!r} "
            f"for oligo {oligo_name!r}; use an explicit `chain:<ID>` selector."
        )
    return chain_ids[0]
