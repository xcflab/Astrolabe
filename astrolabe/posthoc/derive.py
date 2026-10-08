from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from astrolabe.losses.channel_hole import (
    channel_hole_metrics_from_scan_rows,
    load_channel_scan_summary,
)


_AF3_DIR_RE = re.compile(r"^step_(\d{3})_(.+)$")
_AF3_SUMMARY_RE = re.compile(r"^step_(\d{3})_(.+)_summary_confidences\.json$")
_AF3_CONF_RE = re.compile(r"^step_(\d{3})_(.+)_confidences\.json$")
_CHANNEL_HOLE_DIR_RE = re.compile(r"^step_(\d{3})_(.+)$")


def _chain_order_from_token_chain_ids(token_chain_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for c in token_chain_ids:
        c = str(c)
        if c in seen:
            continue
        seen.add(c)
        order.append(c)
    return order


_ALL_OTHER_CHAIN_SELECTORS = {
    "other",
    "others",
    "all-except-binder",
    "all_except_binder",
}


def _is_all_other_selector(selector: Any) -> bool:
    return str(selector).strip().lower() in _ALL_OTHER_CHAIN_SELECTORS


def _chain_ids_from_selector(
    selector: Any,
    *,
    chain_order: list[str],
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = {str(chain_id) for chain_id in exclude}
    if _is_all_other_selector(selector):
        return [chain_id for chain_id in chain_order if chain_id not in excluded]
    if isinstance(selector, (list, tuple, set)):
        raw_ids = [str(item).strip() for item in selector]
    else:
        raw_ids = [part.strip() for part in str(selector).split(",")]
    selected: list[str] = []
    for chain_id in raw_ids:
        if (
            chain_id
            and chain_id in chain_order
            and chain_id not in excluded
            and chain_id not in selected
        ):
            selected.append(chain_id)
    return selected


def _resolve_chain_pair_selectors(
    *,
    chain_a: Any,
    chain_b: Any,
    chain_order: list[str],
) -> tuple[list[str], list[str]]:
    a_is_other = _is_all_other_selector(chain_a)
    b_is_other = _is_all_other_selector(chain_b)
    if a_is_other and b_is_other:
        return [], []
    if a_is_other:
        selected_b = _chain_ids_from_selector(chain_b, chain_order=chain_order)
        selected_a = _chain_ids_from_selector(
            chain_a,
            chain_order=chain_order,
            exclude=selected_b,
        )
        return selected_a, selected_b
    selected_a = _chain_ids_from_selector(chain_a, chain_order=chain_order)
    selected_b = _chain_ids_from_selector(
        chain_b,
        chain_order=chain_order,
        exclude=selected_a if b_is_other else (),
    )
    return selected_a, selected_b


def _binder_metrics_from_chain_vectors(
    *,
    chain_order: list[str],
    binder_chain: str,
    chain_ptm: Any | None,
    chain_iptm: Any | None,
) -> Dict[str, float]:
    if not chain_order or str(binder_chain) not in chain_order:
        return {}
    b_idx = int(chain_order.index(str(binder_chain)))
    out: Dict[str, float] = {}
    if isinstance(chain_ptm, list) and b_idx < len(chain_ptm):
        try:
            out["binder_ptm"] = float(chain_ptm[b_idx])
        except Exception:
            pass
    if isinstance(chain_iptm, list) and b_idx < len(chain_iptm):
        try:
            out["binder_iptm"] = float(chain_iptm[b_idx])
        except Exception:
            pass
    return out


def _selected_chain_metrics_from_chain_vectors(
    *,
    chain_order: list[str],
    chain_id: str,
    chain_ptm: Any | None,
    chain_iptm: Any | None,
    ptm_key: str,
    iptm_key: str,
    exclude_chain_ids: Iterable[str] = (),
) -> Dict[str, float]:
    selected = _chain_ids_from_selector(
        chain_id,
        chain_order=chain_order,
        exclude=exclude_chain_ids,
    )
    if not selected:
        return {}
    indices = [int(chain_order.index(selected_id)) for selected_id in selected]
    out: Dict[str, float] = {}
    if isinstance(chain_ptm, list):
        try:
            vals = [float(chain_ptm[idx]) for idx in indices if idx < len(chain_ptm)]
            if vals:
                out[ptm_key] = float(np.mean(vals))
        except Exception:
            pass
    if isinstance(chain_iptm, list):
        try:
            vals = [float(chain_iptm[idx]) for idx in indices if idx < len(chain_iptm)]
            if vals:
                out[iptm_key] = float(np.mean(vals))
        except Exception:
            pass
    return out


def _normalise_plddt_mean(value: float) -> float:
    if value > 2.0:
        return value / 100.0
    return value


def _chain_plddt_mean(
    *,
    atom_chain_ids: list[str] | None,
    atom_plddts: list[float] | None,
    chain_id: str,
    exclude_chain_ids: Iterable[str] = (),
) -> Dict[str, float]:
    if not isinstance(atom_chain_ids, list) or not isinstance(atom_plddts, list):
        return {}
    selected = set(
        _chain_ids_from_selector(
            chain_id,
            chain_order=_chain_order_from_token_chain_ids(atom_chain_ids),
            exclude=exclude_chain_ids,
        )
    )
    vals = [
        float(plddt)
        for cid, plddt in zip(atom_chain_ids, atom_plddts)
        if str(cid) in selected
    ]
    if not vals:
        return {}
    return {"target_chain_plddt_mean": float(_normalise_plddt_mean(float(np.mean(vals))))}


def _ipae_from_pae_matrix(
    pae: np.ndarray, *, chain_a: str, chain_b: str, token_chain_ids: list[str]
) -> Dict[str, float]:
    """Compute iPAE separately for every selected chain pair, then average.

    A selector such as ``chain_b='others'`` denotes a *group* of target chains.
    Concatenating that group into one residue block would weight long chains more
    heavily and would allow target-target pairs to affect the result.  Keep the
    reduction inside each A/B pair, and only then reduce the pair values.
    """
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return {}
    chain_order = _chain_order_from_token_chain_ids(token_chain_ids)
    resolved_a, resolved_b = _resolve_chain_pair_selectors(
        chain_a=chain_a,
        chain_b=chain_b,
        chain_order=chain_order,
    )
    if not resolved_a or not resolved_b:
        return {}

    chain_ids = np.asarray([str(c) for c in token_chain_ids], dtype=object)
    pair_means: list[float] = []
    pair_mins: list[float] = []
    for chain_a_id in resolved_a:
        a_idx = np.where(chain_ids == str(chain_a_id))[0]
        if a_idx.size == 0:
            continue
        for chain_b_id in resolved_b:
            if chain_a_id == chain_b_id:
                continue
            b_idx = np.where(chain_ids == str(chain_b_id))[0]
            if b_idx.size == 0:
                continue
            if max(int(np.max(a_idx)), int(np.max(b_idx))) >= pae.shape[0]:
                continue
            sub_ab = pae[np.ix_(a_idx, b_idx)]
            sub_ba = pae[np.ix_(b_idx, a_idx)]
            pair_means.append(float(np.mean([np.mean(sub_ab), np.mean(sub_ba)])))
            pair_mins.append(float(min(float(np.min(sub_ab)), float(np.min(sub_ba)))))

    if not pair_means:
        return {}
    return {
        "ipae_mean": float(np.mean(pair_means)),
        "ipae_min": float(np.mean(pair_mins)),
    }


def _calc_d0_array(lengths: np.ndarray) -> np.ndarray:
    values = np.maximum(26.0, np.asarray(lengths, dtype=float))
    return np.maximum(1.0, 1.24 * (values - 15.0) ** (1.0 / 3.0) - 1.8)


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
    pae_sub = pae[np.ix_(chain_a_idx, chain_b_idx)]
    valid = pae_sub < float(pae_cutoff)
    n0res = np.sum(valid, axis=1)
    d0 = _calc_d0_array(n0res)
    scored = _ptm_func(pae_sub, d0[:, None])
    masked = np.where(valid, scored, 0.0)
    denom = np.where(n0res > 0, n0res, 1.0)
    mean_byres = np.sum(masked, axis=1) / denom
    return float(np.max(mean_byres)) if mean_byres.size else 0.0


def _ipsae_from_pae_matrix(
    pae: np.ndarray,
    *,
    chain_a: str,
    chain_b: str,
    token_chain_ids: list[str],
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """Compute IPsaE per selected chain pair and return their equal-weight mean."""
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return {}
    chain_order = _chain_order_from_token_chain_ids(token_chain_ids)
    resolved_a, resolved_b = _resolve_chain_pair_selectors(
        chain_a=chain_a,
        chain_b=chain_b,
        chain_order=chain_order,
    )
    if not resolved_a or not resolved_b:
        return {}

    chain_ids = np.asarray([str(c) for c in token_chain_ids], dtype=object)
    pair_values: list[float] = []
    for chain_a_id in resolved_a:
        a_idx = np.where(chain_ids == str(chain_a_id))[0]
        if a_idx.size == 0:
            continue
        for chain_b_id in resolved_b:
            if chain_a_id == chain_b_id:
                continue
            b_idx = np.where(chain_ids == str(chain_b_id))[0]
            if b_idx.size == 0:
                continue
            if max(int(np.max(a_idx)), int(np.max(b_idx))) >= pae.shape[0]:
                continue
            ab = _ipsae_directional(
                pae,
                chain_a_idx=a_idx,
                chain_b_idx=b_idx,
                pae_cutoff=pae_cutoff,
            )
            ba = _ipsae_directional(
                pae,
                chain_a_idx=b_idx,
                chain_b_idx=a_idx,
                pae_cutoff=pae_cutoff,
            )
            pair_values.append(float(max(ab, ba)))

    return {"ipsae_ab": float(np.mean(pair_values))} if pair_values else {}


def _chain_pair_metric_from_matrix(
    matrix_like: Any | None,
    *,
    chain_order: list[str],
    chain_a: str,
    chain_b: str,
    out_key: str,
    symmetric: bool = True,
) -> Dict[str, float]:
    selected_a, selected_b = _resolve_chain_pair_selectors(
        chain_a=chain_a,
        chain_b=chain_b,
        chain_order=chain_order,
    )
    if not selected_a or not selected_b:
        return {}
    try:
        matrix = np.asarray(matrix_like, dtype=float)
    except Exception:
        return {}
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return {}
    a_indices = [int(chain_order.index(chain_id)) for chain_id in selected_a]
    b_indices = [int(chain_order.index(chain_id)) for chain_id in selected_b]
    if max(a_indices + b_indices) >= matrix.shape[0]:
        return {}
    if symmetric:
        values = [
            float((matrix[a_idx, b_idx] + matrix[b_idx, a_idx]) / 2.0)
            for a_idx in a_indices
            for b_idx in b_indices
            if a_idx != b_idx
        ]
    else:
        values = [
            float(matrix[a_idx, b_idx])
            for a_idx in a_indices
            for b_idx in b_indices
            if a_idx != b_idx
        ]
    return {out_key: float(np.mean(values))} if values else {}


def derive_af3(
    run_dir: Path,
    *,
    binder_chain: str,
    target_chain: str,
) -> Dict[tuple[int, str], Dict[str, Any]]:
    """
    Derive metrics from AF3 raw outputs.

    Returns: {(step, oligo): metrics_dict}
    """
    base = Path(run_dir) / "raw_model_results" / "alphafold3"
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    if not base.exists():
        return out

    # Summaries
    for step_dir in base.iterdir():
        if not step_dir.is_dir():
            continue
        m = _AF3_DIR_RE.match(step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        oligo = m.group(2)
        summary_path = step_dir / f"step_{step:03d}_{oligo}_summary_confidences.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = None
            if isinstance(summary, dict):
                metrics: Dict[str, Any] = {}
                # Scalars we commonly want.
                for k in ["ptm", "iptm", "ranking_score", "fraction_disordered", "has_clash"]:
                    if k in summary:
                        try:
                            metrics[f"af3_{k}"] = float(summary[k])
                        except Exception:
                            pass
                # HalluDesign-aligned derived metrics from summary arrays.
                chain_ptm = summary.get("chain_ptm")
                chain_iptm = summary.get("chain_iptm")
                chain_pair_iptm = summary.get("chain_pair_iptm")
                chain_pair_pae_min = summary.get("chain_pair_pae_min")
                chain_pair_lis = summary.get("chain_pair_lis")
                chain_pair_contact_lis = summary.get("chain_pair_contact_lis")
                if isinstance(chain_ptm, list) and chain_ptm:
                    try:
                        metrics["halludesign_af3_ptm"] = float(chain_ptm[0])
                    except Exception:
                        pass
                # Binder metrics need chain order.
                chain_order: list[str] = []
                conf_path = step_dir / f"step_{step:03d}_{oligo}_confidences.json"
                if conf_path.exists():
                    try:
                        conf = json.loads(conf_path.read_text(encoding="utf-8"))
                    except Exception:
                        conf = None
                    if isinstance(conf, dict):
                        token_chain_ids = conf.get("token_chain_ids")
                        if isinstance(token_chain_ids, list) and token_chain_ids:
                            chain_order = _chain_order_from_token_chain_ids(token_chain_ids)

                        # iPAE from full PAE matrix (available in AF3 confidences).
                        pae = conf.get("pae")
                        if isinstance(pae, list) and pae and isinstance(pae[0], list):
                            try:
                                pae_mat = np.asarray(pae, dtype=float)
                            except Exception:
                                pae_mat = None
                            if pae_mat is not None and pae_mat.ndim == 2:
                                ipae = _ipae_from_pae_matrix(
                                    pae_mat,
                                    chain_a=str(target_chain),
                                    chain_b=str(binder_chain),
                                    token_chain_ids=token_chain_ids or [],
                                )
                                metrics.update(ipae)
                                metrics.update(
                                    _ipsae_from_pae_matrix(
                                        pae_mat,
                                        chain_a=str(target_chain),
                                        chain_b=str(binder_chain),
                                        token_chain_ids=token_chain_ids or [],
                                    )
                                )

                metrics.update(
                    _binder_metrics_from_chain_vectors(
                        chain_order=chain_order,
                        binder_chain=str(binder_chain),
                        chain_ptm=chain_ptm,
                        chain_iptm=chain_iptm,
                    )
                )
                metrics.update(
                    _chain_pair_metric_from_matrix(
                        chain_pair_iptm,
                        chain_order=chain_order,
                        chain_a=str(binder_chain),
                        chain_b=str(target_chain),
                        out_key="halludesign_af3_iptm",
                        symmetric=False,
                    )
                )
                metrics.update(
                    _chain_pair_metric_from_matrix(
                        chain_pair_pae_min,
                        chain_order=chain_order,
                        chain_a=str(binder_chain),
                        chain_b=str(target_chain),
                        out_key="halludesign_af3_ipae",
                        symmetric=False,
                    )
                )
                metrics.update(
                    _selected_chain_metrics_from_chain_vectors(
                        chain_order=chain_order,
                        chain_id=str(target_chain),
                        chain_ptm=chain_ptm,
                        chain_iptm=chain_iptm,
                        ptm_key="target_chain_ptm",
                        iptm_key="target_chain_iptm",
                        exclude_chain_ids=_chain_ids_from_selector(
                            binder_chain,
                            chain_order=chain_order,
                        ),
                    )
                )
                if conf_path.exists():
                    try:
                        conf = json.loads(conf_path.read_text(encoding="utf-8"))
                    except Exception:
                        conf = None
                    if isinstance(conf, dict):
                        metrics.update(
                            _chain_plddt_mean(
                                atom_chain_ids=conf.get("atom_chain_ids"),
                                atom_plddts=conf.get("atom_plddts"),
                                chain_id=str(target_chain),
                                exclude_chain_ids=_chain_ids_from_selector(
                                    binder_chain,
                                    chain_order=chain_order,
                                ),
                            )
                        )
                metrics.update(
                    _chain_pair_metric_from_matrix(
                        chain_pair_lis,
                        chain_order=chain_order,
                        chain_a=str(target_chain),
                        chain_b=str(binder_chain),
                        out_key="chain_pair_lis_ab",
                    )
                )
                metrics.update(
                    _chain_pair_metric_from_matrix(
                        chain_pair_contact_lis,
                        chain_order=chain_order,
                        chain_a=str(target_chain),
                        chain_b=str(binder_chain),
                        out_key="chain_pair_contact_lis_ab",
                    )
                )
                out[(step, oligo)] = metrics
    return out


def derive_protenix(
    run_dir: Path,
    *,
    binder_chain: str,
    target_chain: str = "A",
) -> Dict[tuple[int, str], Dict[str, Any]]:
    base = Path(run_dir) / "raw_model_results" / "protenix"
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    if not base.exists():
        return out

    # Directory layout: raw_model_results/protenix/step_000/A/seed_1/predictions/<...>_summary_confidence_sample_*.json
    for step_dir in base.glob("step_*"):
        if not step_dir.is_dir():
            continue
        m = re.match(r"^step_(\d{3})$", step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        for oligo_dir in step_dir.iterdir():
            if not oligo_dir.is_dir():
                continue
            oligo = oligo_dir.name
            best: tuple[float, Path] | None = None
            for p in oligo_dir.glob("seed_*/predictions/*_summary_confidence_sample_*.json"):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                rs = d.get("ranking_score")
                try:
                    score = float(rs)
                except Exception:
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, p)
            if best is None:
                continue
            try:
                summary = json.loads(best[1].read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(summary, dict):
                continue
            metrics: Dict[str, Any] = {}
            for k in ["plddt", "gpde", "ptm", "iptm", "ranking_score", "disorder"]:
                if k in summary:
                    try:
                        metrics[f"protenix_{k}"] = float(summary[k])
                    except Exception:
                        pass

            # Keep raw vectors for downstream mapping.
            if isinstance(summary.get("chain_ptm"), list):
                metrics["protenix_chain_ptm"] = summary.get("chain_ptm")
            if isinstance(summary.get("chain_iptm"), list):
                metrics["protenix_chain_iptm"] = summary.get("chain_iptm")

            # binder metrics may be refined later using structure chain order; provide a conservative fallback.
            chain_ptm = summary.get("chain_ptm")
            chain_iptm = summary.get("chain_iptm")
            if isinstance(chain_ptm, list):
                chain_order = [chr(ord("A") + i) for i in range(len(chain_ptm))]
                metrics.update(
                    _binder_metrics_from_chain_vectors(
                        chain_order=chain_order,
                        binder_chain=str(binder_chain),
                        chain_ptm=chain_ptm,
                        chain_iptm=chain_iptm,
                    )
                )
                metrics.update(
                    _selected_chain_metrics_from_chain_vectors(
                        chain_order=chain_order,
                        chain_id=str(target_chain),
                        chain_ptm=chain_ptm,
                        chain_iptm=chain_iptm,
                        ptm_key="target_chain_ptm",
                        iptm_key="target_chain_iptm",
                    )
                )
            out[(step, oligo)] = metrics
    return out


def derive_boltz(
    run_dir: Path,
    *,
    binder_chain: str,
    target_chain: str,
) -> Dict[tuple[int, str], Dict[str, Any]]:
    base = Path(run_dir) / "raw_model_results" / "boltz"
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    if not base.exists():
        return out

    for step_dir in base.glob("step_*"):
        if not step_dir.is_dir():
            continue
        m = re.match(r"^step_(\d{3})_(.+)$", step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        oligo = m.group(2)
        # Prefer model_0 json (Boltz uses multiple diffusion samples/models).
        candidates = sorted(step_dir.glob("predictions/*/*_model_0.json"))
        if not candidates:
            candidates = sorted(step_dir.glob("predictions/*/*_model_*.json"))
        if not candidates:
            continue
        p = candidates[0]
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        metrics: Dict[str, Any] = {}
        for k in ["complex_plddt", "ptm", "iptm", "complex_pde"]:
            if k in d:
                try:
                    metrics[f"boltz_{k}"] = float(d[k])
                except Exception:
                    pass

        # Optional pair-wise iptm by chain IDs.
        pci = d.get("pair_chains_iptm")
        if isinstance(pci, dict):
            try:
                v = float(pci.get(str(target_chain), {}).get(str(binder_chain)))
            except Exception:
                v = None
            if v is not None:
                metrics["boltz_target_binder_iptm"] = v
        out[(step, oligo)] = metrics
    return out


def derive_chai1(run_dir: Path) -> Dict[tuple[int, str], Dict[str, Any]]:
    """
    Best-effort parsing for chai1 raw outputs.

    Chai output schemas vary; we keep this conservative and only expose metrics when
    obvious JSON files are present. Most chai metrics should already be in scores.csv.
    """
    base = Path(run_dir) / "raw_model_results" / "chai1"
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    if not base.exists():
        return out
    # Placeholder: no stable on-disk schema enforced in this repo yet.
    return out


def derive_channel_hole(run_dir: Path) -> Dict[tuple[int, str], Dict[str, Any]]:
    base = Path(run_dir) / "raw_model_results" / "channel_hole"
    out: Dict[tuple[int, str], Dict[str, Any]] = {}
    if not base.exists():
        return out

    for step_dir in base.iterdir():
        if not step_dir.is_dir():
            continue
        m = _CHANNEL_HOLE_DIR_RE.match(step_dir.name)
        if not m:
            continue
        scan_summary_path = step_dir / "scan_summary.tsv"
        if not scan_summary_path.exists():
            continue
        step = int(m.group(1))
        oligo = m.group(2)
        try:
            rows = load_channel_scan_summary(scan_summary_path)
        except Exception:
            continue
        out[(step, oligo)] = channel_hole_metrics_from_scan_rows(rows)
    return out


def read_run_evaluator_name(run_dir: Path) -> str | None:
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception:
        return None
    ev = cfg.get("evaluator")
    if isinstance(ev, dict) and "name" in ev:
        return str(ev["name"])
    return None
