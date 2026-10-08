from collections.abc import Sequence
import json
import logging

import numpy as np
import torch

from astrolabe.core import global_config
from astrolabe.core.binding_site import ChainPositions, parse_chain_positions
from astrolabe.core.chain_resolution import (
    chain_ids_for_oligo,
    resolve_selector_chain_ids,
)
from astrolabe.losses.base import loss_registry
from astrolabe.losses.utils import pae_residue_aggregation


# todo enable selector for those residue-wise scores, plddt, pae, hotspot_dsasa


@loss_registry.register(description="Mean predicted aligned error (PAE) score")
def pae(eval_result, normalize_factor=20):
    return {
        name: np.mean(eval_result["pae"][name]) / normalize_factor
        for name in eval_result.names
    }


@loss_registry.register(description="Mean predicted distance error (PDE) score")
def pde(eval_result, normalize_factor=20):
    return {
        name: np.mean(eval_result["pde"][name]) / normalize_factor
        for name in eval_result.names
    }


@loss_registry.register(description="Mean predicted TM score")
def ptm(eval_result):
    """Calculate mean predicted TM score."""
    return {name: np.mean(eval_result["ptm"][name]) for name in eval_result.names}


@loss_registry.register(description="Mean predicted LDDT score normalized to [0,1]")
def plddt(eval_result):
    """Calculate mean predicted LDDT score."""
    return {name: np.mean(eval_result["plddt"][name]) for name in eval_result.names}


@loss_registry.register(description="Mean interfacial predicted TM score")
def iptm(eval_result):
    """Calculate mean interfacial predicted TM score."""
    return {name: np.mean(eval_result["iptm"][name]) for name in eval_result.names}


def _infer_binder_selector_for_oligo(oligo_name: str) -> str | None:
    try:
        from astrolabe.core.binding_site import parse_binding_site
        from astrolabe.core import global_config

        spec = parse_binding_site(
            getattr(global_config, "binding_site", None),
            oligo_name=oligo_name,
        )
        if spec is not None:
            return str(spec.binder.chain)
    except Exception:
        return None
    return None


def _parse_target_hotspot_target(raw_target) -> ChainPositions:
    if isinstance(raw_target, Sequence) and not isinstance(raw_target, (str, bytes)):
        parsed = [parse_chain_positions(str(item)) for item in raw_target]
        chains = {item.chain for item in parsed}
        if len(chains) != 1:
            raise ValueError(
                f"target_hotspot_plddt.target must use one chain selector, got {sorted(chains)}"
            )
        positions: list[int] = []
        seen: set[int] = set()
        for item in parsed:
            for pos in item.positions:
                if pos not in seen:
                    seen.add(pos)
                    positions.append(pos)
        target = ChainPositions(chain=parsed[0].chain, positions=tuple(positions))
    else:
        target = parse_chain_positions(str(raw_target))
    if not target.positions:
        raise ValueError("target_hotspot_plddt.target requires explicit residue position(s).")
    return target


def _resolve_plddt_arrays_for_selector(eval_result, *, oligo_name: str, selector: str) -> list[np.ndarray]:
    per_proto_scores = getattr(eval_result, "per_proto_scores", None) or {}
    chain_ids = resolve_selector_chain_ids(oligo_name, selector)
    if not chain_ids:
        return []

    ordered_chain_ids = chain_ids_for_oligo(oligo_name)
    chain_slots: dict[str, tuple[str, int]] = {}
    proto_counts: dict[str, int] = {}
    for chain_id, proto_name in zip(ordered_chain_ids, str(oligo_name)):
        occ = proto_counts.get(proto_name, 0)
        chain_slots[chain_id] = (proto_name, occ)
        proto_counts[proto_name] = occ + 1

    resolved: list[np.ndarray] = []
    for chain_id in chain_ids:
        proto_name, occ = chain_slots.get(chain_id, (None, None))
        if proto_name is None:
            continue
        proto_scores = per_proto_scores.get(proto_name) or []
        if occ is None or occ >= len(proto_scores):
            continue
        resolved.append(np.asarray(proto_scores[occ], dtype=float))
    return resolved


def _mean_summary_array(
    mats,
    *,
    expected_ndim: int,
    context: str,
) -> np.ndarray:
    averaged = np.mean(np.stack([_to_numpy_float_array(m) for m in mats]), axis=0)
    averaged = np.asarray(averaged, dtype=float)
    while averaged.ndim > expected_ndim and averaged.shape[0] == 1:
        averaged = averaged[0]
    if averaged.ndim != expected_ndim:
        raise ValueError(
            f"{context}: expected {expected_ndim}D summary after averaging, got shape {tuple(averaged.shape)}."
        )
    return averaged


def _mean_chain_vector(mats, *, context: str) -> np.ndarray:
    averaged = _mean_summary_array(mats, expected_ndim=1, context=context)
    if averaged.shape[0] < 1:
        raise ValueError(f"{context}: expected at least one chain, got shape {tuple(averaged.shape)}.")
    return averaged


def _mean_chain_pair_matrix(mats, *, context: str) -> np.ndarray:
    averaged = _mean_summary_array(mats, expected_ndim=2, context=context)
    if averaged.shape[0] != averaged.shape[1] or averaged.shape[0] < 2:
        raise ValueError(
            f"{context}: expected square chain-pair matrix with >=2 chains, got shape {tuple(averaged.shape)}."
        )
    return averaged


@loss_registry.register(
    name="binder_iptm",
    description="Binder chain interface pTM (Protenix chain_iptm definition: mean of chain_pair_iptm pairs involving binder chain)",
    optional_params=["binder_chain"],
    supports_oligomer_specific=True,
)
def binder_iptm(eval_result, binder_chain: str | None = None):
    """
    Compute binder-chain iPTM from complex outputs.

    Definition (aligned to Protenix `chain_iptm`):
      - Let M be the per-chain-pair iPTM matrix (`chain_pair_iptm`) for a complex.
      - binder_iptm = mean_j!=b M[b, j]
        (for 2-chain A/B complex, this equals M[A,B] == M[B,A]).

    Notes:
      - This is a *complex* metric; binder-only single-chain folds do not define an interface.
      - If binder_chain is not provided, we try to infer it from global_config.binding_site (binder.chain).
    """
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None)
    if not chain_pair_scores:
        return {name: 0.0 for name in getattr(eval_result, "names", [])}

    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        selector = binder_chain or _infer_binder_selector_for_oligo(oligo_name) or "B"
        mats = (chain_pair_scores.get(oligo_name) or {}).get("chain_pair_iptm")
        if not mats:
            scores[oligo_name] = 0.0
            continue
        averaged = _mean_chain_pair_matrix(
            mats,
            context="binder_iptm.chain_pair_iptm",
        )
        n = int(averaged.shape[0])
        b_indices = _resolve_chain_indices_for_oligo(
            oligo_name,
            str(selector),
            matrix_size=n,
            context="binder_iptm.binder_chain",
        )
        if not b_indices:
            scores[oligo_name] = 0.0
            continue
        by_chain_vals: list[float] = []
        for b_idx in b_indices:
            vals = [float(averaged[b_idx, j]) for j in range(n) if j != b_idx]
            if vals:
                by_chain_vals.append(float(np.mean(vals)))
        scores[oligo_name] = float(np.mean(by_chain_vals)) if by_chain_vals else 0.0
    return scores


@loss_registry.register(
    name="chain_pair_iptm",
    description="Mean chain-pair iPTM for the selected chain_a/chain_b pairs",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def chain_pair_iptm(eval_result, chain_a, chain_b):
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None)
    if not chain_pair_scores:
        return {name: 0.0 for name in getattr(eval_result, "names", [])}

    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        mats = (chain_pair_scores.get(oligo_name) or {}).get("chain_pair_iptm")
        if not mats:
            scores[oligo_name] = 0.0
            continue
        averaged = _mean_chain_pair_matrix(
            mats,
            context="chain_pair_iptm.chain_pair_iptm",
        )
        pairs = _resolve_chain_pair_indices(
            oligo_name,
            chain_a=chain_a,
            chain_b=chain_b,
            matrix_size=int(averaged.shape[0]),
            context="chain_pair_iptm",
        )
        values = [float(averaged[left, right]) for left, right in pairs]
        scores[oligo_name] = float(np.mean(values)) if values else 0.0
    return scores


@loss_registry.register(
    name="binder_ptm",
    description="Binder chain pTM (from per-chain chain_ptm in AF3/Protenix summary_confidence)",
    optional_params=["binder_chain"],
    supports_oligomer_specific=True,
)
def binder_ptm(eval_result, binder_chain: str | None = None):
    """
    Compute binder-chain pTM from model summary outputs when available.

    This aligns to AF3/Protenix-style `chain_ptm` arrays, where `chain_ptm[i]`
    corresponds to chain i (A=0, B=1, ...).
    """
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None)
    if not chain_pair_scores:
        return {name: 0.0 for name in getattr(eval_result, "names", [])}

    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        selector = binder_chain or _infer_binder_selector_for_oligo(oligo_name) or "B"
        mats = (chain_pair_scores.get(oligo_name) or {}).get("chain_ptm")
        if not mats:
            scores[oligo_name] = 0.0
            continue
        averaged = _mean_chain_vector(
            mats,
            context="binder_ptm.chain_ptm",
        )
        n = int(averaged.shape[0])
        b_indices = _resolve_chain_indices_for_oligo(
            oligo_name,
            str(selector),
            matrix_size=n,
            context="binder_ptm.binder_chain",
        )
        if not b_indices:
            scores[oligo_name] = 0.0
            continue
        vals = [float(averaged[b_idx]) for b_idx in b_indices if b_idx < n]
        scores[oligo_name] = float(np.mean(vals)) if vals else 0.0
    return scores


@loss_registry.register(
    name="target_hotspot_plddt",
    description="Mean residue-level pLDDT over an explicitly configured target hotspot window",
    required_params=["target"],
    supports_oligomer_specific=True,
)
def target_hotspot_plddt(eval_result, target):
    spec = _parse_target_hotspot_target(target)
    scores: dict[str, float] = {}
    for oligo_name in getattr(eval_result, "names", []):
        arrays = _resolve_plddt_arrays_for_selector(
            eval_result,
            oligo_name=oligo_name,
            selector=str(spec.chain),
        )
        if not arrays:
            scores[oligo_name] = 0.0
            continue

        selected_vals: list[float] = []
        for arr in arrays:
            if arr.ndim != 1:
                arr = np.asarray(arr, dtype=float).reshape(-1)
            missing = [pos for pos in spec.positions if pos <= 0 or pos > len(arr)]
            if missing:
                raise ValueError(
                    f"target_hotspot_plddt.target positions {missing!r} fall outside selector {spec.chain!r} "
                    f"length {len(arr)} for oligo {oligo_name!r}."
                )
            selected_vals.extend(float(arr[pos - 1]) for pos in spec.positions)
        scores[oligo_name] = float(np.mean(selected_vals)) if selected_vals else 0.0
    return scores


intra_chain_keys = [
    "intra_contact_lis",
    "intra_iptm",
    "intra_lis",
    "intra_pae_min",
]
cross_chain_keys = [
    "cross_contact_lis",
    "cross_iptm",
    "cross_lis",
    "cross_pae_min",
]


for key in intra_chain_keys:
    chain_pair_key = key.replace("intra", "chain_pair")

    @loss_registry.register(
        name=key,
        description=f"Mean intra-chain {key.replace('_', ' ')} score",
        supports_oligomer_specific=True,
    )
    def intra_chain_loss(eval_result, _chain_pair_key=chain_pair_key):
        """Calculate mean intra-chain loss."""
        chain_pair_scores = getattr(eval_result, "chain_pair_scores", None) or {}
        scores = {}
        for oligo_name in eval_result.names:
            score_mats = (chain_pair_scores.get(oligo_name) or {}).get(_chain_pair_key)
            if not score_mats:
                scores[oligo_name] = 0.0
                continue
            first = np.asarray(score_mats[0], dtype=float)
            mask = np.eye(int(first.shape[0]), dtype=bool)
            scores[oligo_name] = float(
                np.mean([np.asarray(s, dtype=float)[mask] for s in score_mats])
            )
        return scores


for key in cross_chain_keys:
    chain_pair_key = key.replace("cross", "chain_pair")

    @loss_registry.register(
        name=key,
        description=f"Mean cross-chain {key.replace('_', ' ')} score",
        supports_oligomer_specific=True,
    )
    def cross_chain_loss(eval_result, _chain_pair_key=chain_pair_key):
        """Calculate mean cross-chain loss."""
        chain_pair_scores = getattr(eval_result, "chain_pair_scores", None) or {}
        scores = {}
        for oligo_name in eval_result.names:
            score_mats = (chain_pair_scores.get(oligo_name) or {}).get(_chain_pair_key)
            if not score_mats:
                scores[oligo_name] = 0.0
                continue
            first = np.asarray(score_mats[0], dtype=float)
            mask = ~np.eye(int(first.shape[0]), dtype=bool)
            scores[oligo_name] = float(
                np.mean([np.asarray(s, dtype=float)[mask] for s in score_mats])
            )
        return scores


def _stack_pae_matrices(value) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, list):
        mats = value
    else:
        mats = [value]
    out: list[np.ndarray] = []
    for m in mats:
        try:
            arr = _to_numpy_float_array(m)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
            out.append(arr)
    return out


def _maybe_get_global_config_value(name: str):
    try:
        return getattr(global_config, name)
    except Exception:
        return None


def _load_af3_residue_level_pae_matrices(oligo_name: str) -> list[np.ndarray]:
    raw_model_results_dir = _maybe_get_global_config_value("raw_model_results_dir")
    if raw_model_results_dir is None:
        return []

    step_value = _maybe_get_global_config_value("step")
    step_value = 0 if step_value is None else int(step_value)
    identifier = f"step_{step_value:03d}_{oligo_name}"
    confidence_path = (
        raw_model_results_dir / "alphafold3" / identifier / f"{identifier}_confidences.json"
    )
    if not confidence_path.exists():
        return []

    try:
        with confidence_path.open("r", encoding="utf-8") as handle:
            confidence_finegrain = json.load(handle)
        pae_residue, _ = pae_residue_aggregation(
            confidence_finegrain["token_res_ids"],
            confidence_finegrain["token_chain_ids"],
            confidence_finegrain["pae"],
        )
    except Exception as exc:
        logging.warning(
            "Failed to load AF3 residue-level PAE from %s for %s: %s",
            confidence_path,
            oligo_name,
            exc,
        )
        return []

    arr = np.asarray(pae_residue, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return []
    return [arr]


def _resolve_residue_level_pae_matrices(eval_result, *, oligo_name: str) -> list[np.ndarray]:
    pae_by_oligo = (
        eval_result["pae"]
        if hasattr(eval_result, "__getitem__")
        else getattr(eval_result, "pae", {})
    )
    mats = _stack_pae_matrices((pae_by_oligo or {}).get(oligo_name))
    if mats:
        return mats
    return _load_af3_residue_level_pae_matrices(str(oligo_name))


def _to_numpy_float_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return np.asarray(value.detach().cpu().float().numpy(), dtype=float)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float)
    if isinstance(value, (list, tuple)):
        normalized_items = [_to_numpy_float_array(item) for item in value]
        return np.asarray(normalized_items, dtype=float)
    if all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
        return np.asarray(value.detach().cpu().numpy(), dtype=float)
    return np.asarray(value, dtype=float)


def _chain_residue_indices(structure_path, chain_id: str) -> np.ndarray:
    from astrolabe.utils.file_parser import parse_mmcif, parse_pdb

    path_str = str(structure_path)
    if path_str.endswith(".pdb") or path_str.endswith(".ent"):
        parsed = parse_pdb(path_str)
    else:
        parsed = parse_mmcif(path_str)
    pdb_idx = parsed.get("pdb_idx", [])
    idx = [i for i, (c, _r) in enumerate(pdb_idx) if str(c) == str(chain_id)]
    return np.asarray(idx, dtype=int)


_OTHER_CHAIN_SELECTORS = {
    "other",
    "others",
    "all-except-binder",
    "all_except_binder",
}


def _selector_tokens(selector) -> list[str]:
    if selector is None:
        return []
    if isinstance(selector, str):
        return [
            token.strip()
            for chunk in selector.split(",")
            for token in chunk.split()
            if token.strip()
        ]
    try:
        return [str(item).strip() for item in selector if str(item).strip()]
    except TypeError:
        token = str(selector).strip()
        return [token] if token else []


def _selector_is_others(selector) -> bool:
    return any(token.lower() in _OTHER_CHAIN_SELECTORS for token in _selector_tokens(selector))


def _selector_is_protomer_group(selector) -> bool:
    return any(
        token.lower().startswith("protomer:")
        for token in _selector_tokens(selector)
    )


def _resolve_chain_ids_for_oligo(
    oligo_name: str,
    selector,
    *,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Resolve a possibly grouped selector to ordered, unique chain IDs."""
    ordered = chain_ids_for_oligo(oligo_name)
    excluded = {str(chain_id) for chain_id in exclude}
    selected: list[str] = []
    for token in _selector_tokens(selector):
        if token.lower() in _OTHER_CHAIN_SELECTORS:
            candidates = ordered
        else:
            candidates = resolve_selector_chain_ids(oligo_name, token)
        for chain_id in candidates:
            chain_id = str(chain_id)
            if chain_id not in excluded and chain_id not in selected:
                selected.append(chain_id)
    return selected


def _resolve_chain_pair_ids(
    oligo_name: str,
    *,
    chain_a,
    chain_b,
) -> tuple[list[str], list[str]]:
    selected_a = _resolve_chain_ids_for_oligo(oligo_name, chain_a)
    selected_b = _resolve_chain_ids_for_oligo(
        oligo_name,
        chain_b,
        exclude=selected_a if _selector_is_others(chain_b) else (),
    )
    if _selector_is_others(chain_a):
        selected_a = _resolve_chain_ids_for_oligo(
            oligo_name,
            chain_a,
            exclude=selected_b,
        )
    return selected_a, selected_b


def _resolve_chain_pair_indices(
    oligo_name: str,
    *,
    chain_a,
    chain_b,
    matrix_size: int,
    context: str,
) -> list[tuple[int, int]]:
    """Resolve selectors into directed chain pairs, excluding target self-pairs."""
    selected_a, selected_b = _resolve_chain_pair_ids(
        oligo_name,
        chain_a=chain_a,
        chain_b=chain_b,
    )
    ordered = chain_ids_for_oligo(oligo_name)[:matrix_size]
    chain_index = {chain_id: idx for idx, chain_id in enumerate(ordered)}
    missing = [
        chain_id
        for chain_id in [*selected_a, *selected_b]
        if chain_id not in chain_index
    ]
    if missing:
        raise ValueError(
            f"{context}: selectors {chain_a!r}/{chain_b!r} resolved to chains {missing!r} "
            f"outside matrix size {matrix_size} for oligo {oligo_name!r}."
        )
    return [
        (chain_index[a_id], chain_index[b_id])
        for a_id in selected_a
        for b_id in selected_b
        if a_id != b_id
    ]


def _resolve_chain_indices_for_oligo(
    oligo_name: str,
    selector,
    *,
    matrix_size: int,
    context: str,
) -> list[int]:
    chain_ids = _resolve_chain_ids_for_oligo(oligo_name, selector)
    if not chain_ids:
        return []
    ordered = chain_ids_for_oligo(oligo_name)[:matrix_size]
    chain_index = {cid: idx for idx, cid in enumerate(ordered)}
    resolved = [chain_index[cid] for cid in chain_ids if cid in chain_index]
    if not resolved and chain_ids:
        raise ValueError(
            f"{context}: selector {selector!r} resolved to chains {chain_ids!r} "
            f"outside matrix size {matrix_size} for oligo {oligo_name!r}."
        )
    return resolved


def _calc_d0_array(L) -> np.ndarray:
    """IPSAE d0(L) with minimum value 1.0.

    Matches DunbrackLab/IPSAE `calc_d0_array` behavior (incl. max(26, L)).
    """
    L = np.array(L, dtype=float)
    L = np.maximum(26, L)
    return np.maximum(1.0, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def _ptm_func(pae: np.ndarray, d0: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


def _ipsae_directional(
    pae: np.ndarray,
    *,
    chain_a_idx: np.ndarray,
    chain_b_idx: np.ndarray,
    pae_cutoff: float,
) -> float:
    if chain_a_idx.size == 0 or chain_b_idx.size == 0:
        return 0.0
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return 0.0
    if int(np.max(chain_a_idx)) >= pae.shape[0] or int(np.max(chain_b_idx)) >= pae.shape[0]:
        return 0.0

    pae_sub = pae[np.ix_(chain_a_idx, chain_b_idx)]
    valid = pae_sub < float(pae_cutoff)
    n0res = np.sum(valid, axis=1)
    d0 = _calc_d0_array(n0res)

    # Mean ptm(pae, d0(i)) over valid pairs for each residue i; 0 if no valid pairs.
    scored = _ptm_func(pae_sub, d0[:, None])
    scores_byres = np.where(valid, scored, 0.0)
    denom = np.where(n0res > 0, n0res, 1.0)
    mean_byres = np.sum(scores_byres, axis=1) / denom
    return float(np.max(mean_byres)) if mean_byres.size else 0.0


@loss_registry.register(
    name="ipsae",
    description="ipSAE (DunbrackLab/IPSAE): equal-weight mean of symmetric per-chain-pair ipSAE values",
    required_params=["chain_a", "chain_b"],
    optional_params=["pae_cutoff"],
    supports_oligomer_specific=True,
)
def ipsae(eval_result, chain_a, chain_b, pae_cutoff: float = 10.0):
    """Compute IPsaE independently for each selected chain pair, then average.

    ``chain_b='others'`` is intentionally a group selector.  It means
    ``(chain_a, B)``, ``(chain_a, C)``, ... and never a concatenated target
    residue block; target-target pairs are therefore excluded.
    """
    scores: dict[str, float] = {}
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}

    for oligo_name in eval_result.names:
        mats = _resolve_residue_level_pae_matrices(eval_result, oligo_name=oligo_name)
        if not mats:
            scores[oligo_name] = 0.0
            continue

        structure_path = complex_paths.get(oligo_name)
        if structure_path is None:
            scores[oligo_name] = 0.0
            continue

        selected_a, selected_b = _resolve_chain_pair_ids(
            oligo_name,
            chain_a=chain_a,
            chain_b=chain_b,
        )

        pair_indices = [
            (
                _chain_residue_indices(structure_path, chain_a_id),
                _chain_residue_indices(structure_path, chain_b_id),
            )
            for chain_a_id in selected_a
            for chain_b_id in selected_b
            if chain_a_id != chain_b_id
        ]
        pair_indices = [
            (a_idx, b_idx)
            for a_idx, b_idx in pair_indices
            if a_idx.size and b_idx.size
        ]
        if not pair_indices:
            scores[oligo_name] = 0.0
            continue

        per_matrix_scores: list[float] = []
        for matrix in mats:
            pair_scores: list[float] = []
            for a_idx, b_idx in pair_indices:
                ab = _ipsae_directional(
                    matrix,
                    chain_a_idx=a_idx,
                    chain_b_idx=b_idx,
                    pae_cutoff=pae_cutoff,
                )
                ba = _ipsae_directional(
                    matrix,
                    chain_a_idx=b_idx,
                    chain_b_idx=a_idx,
                    pae_cutoff=pae_cutoff,
                )
                pair_scores.append(float(max(ab, ba)))
            if pair_scores:
                per_matrix_scores.append(float(np.mean(pair_scores)))
        scores[oligo_name] = float(np.mean(per_matrix_scores)) if per_matrix_scores else 0.0
    return scores


def _pae_min_between_chains(
    pae: np.ndarray, *, chain_a_idx: np.ndarray, chain_b_idx: np.ndarray
) -> float:
    if chain_a_idx.size == 0 or chain_b_idx.size == 0:
        return 0.0
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return 0.0
    if int(np.max(chain_a_idx)) >= pae.shape[0] or int(np.max(chain_b_idx)) >= pae.shape[0]:
        return 0.0
    sub_ab = pae[np.ix_(chain_a_idx, chain_b_idx)]
    sub_ba = pae[np.ix_(chain_b_idx, chain_a_idx)]
    return float(min(float(np.min(sub_ab)), float(np.min(sub_ba))))


@loss_registry.register(
    name="pae_min",
    description="Minimum cross-chain PAE between chain_a and chain_b; target-group selectors are reduced per chain pair then averaged",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def pae_min(eval_result, chain_a: str, chain_b: str):
    """Compute cross-chain PAE minimum for a specified chain pair.

    Priority:
      1) If evaluator provides `chain_pair_scores[oligo]['chain_pair_pae_min']` (chain-level matrix),
         extract the pair value and return min(M[a,b], M[b,a]).
      2) Fallback to full residue-level PAE matrices under `eval_result.pae[oligo]` using structure
         chain-to-residue indexing from `eval_result.complex_paths[oligo]`.
    """
    scores: dict[str, float] = {}
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None) or {}
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}

    for oligo_name in eval_result.names:
        # Preferred: chain-level summary matrix from evaluator outputs.
        mats = (chain_pair_scores.get(oligo_name) or {}).get("chain_pair_pae_min")
        if mats:
            averaged = _mean_chain_pair_matrix(
                mats,
                context="pae_min.chain_pair_pae_min",
            )
            n = int(averaged.shape[0])
            pair_indices = _resolve_chain_pair_indices(
                oligo_name,
                chain_a=chain_a,
                chain_b=chain_b,
                matrix_size=n,
                context="pae_min.chain_pair",
            )
            selected_a_ids, selected_b_ids = _resolve_chain_pair_ids(
                oligo_name,
                chain_a=chain_a,
                chain_b=chain_b,
            )
            pair_values = [
                min(float(averaged[a_idx, b_idx]), float(averaged[b_idx, a_idx]))
                for a_idx, b_idx in pair_indices
            ]
            if pair_values:
                # A multi-chain target is an explicit group and is averaged
                # pairwise.  Preserve the historical protomer-level minimum
                # when only the binder selector expands to multiple copies.
                if (
                    len(selected_b_ids) > 1
                    or len(selected_a_ids) <= 1
                    or not _selector_is_protomer_group(chain_a)
                ):
                    scores[oligo_name] = float(np.mean(pair_values))
                else:
                    scores[oligo_name] = float(min(pair_values))
                continue

        # Fallback: compute from full residue-level PAE matrix.
        structure_path = complex_paths.get(oligo_name)
        mats = _resolve_residue_level_pae_matrices(eval_result, oligo_name=oligo_name)
        if not mats or structure_path is None:
            scores[oligo_name] = 0.0
            continue
        selected_a, selected_b = _resolve_chain_pair_ids(
            oligo_name,
            chain_a=chain_a,
            chain_b=chain_b,
        )
        pair_indices = [
            (
                _chain_residue_indices(structure_path, chain_a_id),
                _chain_residue_indices(structure_path, chain_b_id),
            )
            for chain_a_id in selected_a
            for chain_b_id in selected_b
            if chain_a_id != chain_b_id
        ]
        pair_indices = [
            (a_idx, b_idx)
            for a_idx, b_idx in pair_indices
            if a_idx.size and b_idx.size
        ]
        if not pair_indices:
            scores[oligo_name] = 0.0
            continue
        per_matrix_vals: list[float] = []
        for matrix in mats:
            pair_vals = [
                _pae_min_between_chains(
                    matrix,
                    chain_a_idx=a_idx,
                    chain_b_idx=b_idx,
                )
                for a_idx, b_idx in pair_indices
            ]
            if pair_vals:
                if (
                    len(selected_b) > 1
                    or len(selected_a) <= 1
                    or not _selector_is_protomer_group(chain_a)
                ):
                    per_matrix_vals.append(float(np.mean(pair_vals)))
                else:
                    per_matrix_vals.append(float(min(pair_vals)))
        scores[oligo_name] = float(np.mean(per_matrix_vals)) if per_matrix_vals else 0.0

    return scores


@loss_registry.register(
    name="scaled_pae_min",
    description="Per-chain-pair scaled cross-chain PAE min mapped to [0,1]: max(0, 1 - pae_min/cutoff), then averaged",
    required_params=["chain_a", "chain_b"],
    optional_params=["cutoff"],
    supports_oligomer_specific=True,
)
def scaled_pae_min(eval_result, chain_a: str, chain_b: str, cutoff: float = 10.0):
    """Return a [0,1] score with scaling applied per target chain pair."""
    cutoff_val = float(cutoff)
    if cutoff_val <= 0:
        cutoff_val = 10.0

    scores: dict[str, float] = {}
    for oligo_name in getattr(eval_result, "names", []):
        selected_a, selected_b = _resolve_chain_pair_ids(
            oligo_name,
            chain_a=chain_a,
            chain_b=chain_b,
        )

        # Applying the clipping after averaging would let a high-PAE target
        # chain disappear into a low-PAE partner.  Scale each chain pair first.
        legacy_protomer_group = (
            len(selected_a) > 1
            and len(selected_b) == 1
            and _selector_is_protomer_group(chain_a)
        )
        if (len(selected_a) > 1 or len(selected_b) > 1) and not legacy_protomer_group:
            pair_raw: list[float] = []
            for binder_id in selected_a:
                for target_id in selected_b:
                    if binder_id == target_id:
                        continue
                    one_pair = pae_min(
                        eval_result,
                        chain_a=binder_id,
                        chain_b=target_id,
                    ).get(oligo_name)
                    if one_pair is not None:
                        pair_raw.append(float(one_pair))
            if pair_raw:
                scores[oligo_name] = float(
                    np.mean(
                        [
                            np.clip(1.0 - (value / cutoff_val), 0.0, 1.0)
                            for value in pair_raw
                        ]
                    )
                )
                continue

        raw = pae_min(eval_result, chain_a=chain_a, chain_b=chain_b).get(oligo_name)
        if raw is None:
            scores[oligo_name] = 0.0
        else:
            scores[oligo_name] = float(np.clip(1.0 - (float(raw) / cutoff_val), 0.0, 1.0))
    return scores
