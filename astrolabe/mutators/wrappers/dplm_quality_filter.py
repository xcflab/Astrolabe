from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DplmQualityFilterConfig:
    # Matches upstream DPLM analysis/bd_iter_ablation_seq_stats.py defaults.
    entropy_threshold: float = 0.35
    max_res_frac_threshold: float = 0.60
    kmer_k: int = 3
    kmer_unique_frac_threshold: float = 0.30
    repeat_max_period: int = 6
    repeat_match_threshold: float = 0.90
    repeat_min_repeats: int = 5
    repeat_max_unique_aa_in_motif: int = 4
    max_run_threshold: int = 10
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"


def shannon_entropy_bits(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = float(len(seq))
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p, 2)
    return float(h)


def entropy_norm(seq: str, alphabet_size: int = 20) -> float:
    if alphabet_size <= 1:
        return 0.0
    denom = math.log(float(alphabet_size), 2)
    if denom <= 0:
        return 0.0
    return shannon_entropy_bits(seq) / denom


def max_residue_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    return float(max(counts.values())) / float(len(seq))


def max_run_length(seq: str) -> int:
    s = (seq or "").strip().upper()
    if not s:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return int(best)


def kmer_unique_fraction(seq: str, k: int) -> float:
    if not seq:
        return 0.0
    if k <= 0:
        return 0.0
    if len(seq) < k:
        return 0.0
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    if not kmers:
        return 0.0
    return float(len(set(kmers))) / float(len(kmers))


def best_small_period_repeat(seq: str, max_period: int) -> tuple[int, str, float]:
    """Infer most periodic small motif by voting residues at each modulo position.

    Returns (period, motif, match_fraction). period=0 if not applicable.
    """
    s = (seq or "").strip().upper()
    if not s:
        return 0, "", 0.0
    best_p = 0
    best_motif = ""
    best_match = 0.0
    max_p = max(1, int(max_period))
    for p in range(1, max_p + 1):
        buckets = [Counter() for _ in range(p)]
        for i, ch in enumerate(s):
            buckets[i % p][ch] += 1
        motif_chars = []
        for b in buckets:
            motif_chars.append(b.most_common(1)[0][0] if b else "X")
        motif = "".join(motif_chars)
        match = 0
        for i, ch in enumerate(s):
            if ch == motif[i % p]:
                match += 1
        match_frac = float(match) / float(len(s))
        if (match_frac > best_match) or (
            match_frac == best_match and (best_p == 0 or p < best_p)
        ):
            best_p = p
            best_motif = motif
            best_match = match_frac
    return int(best_p), str(best_motif), float(best_match)


def _is_low_quality(seq: str, cfg: DplmQualityFilterConfig) -> tuple[bool, dict[str, float | int | str]]:
    s = (seq or "").strip().upper()
    info: dict[str, float | int | str] = {}
    if not s:
        return True, {"reason": "empty"}
    allowed = set(cfg.alphabet)
    bad = sorted({c for c in s if c not in allowed})
    if bad:
        return True, {"reason": "bad_chars", "bad_chars": "".join(bad)}

    en = entropy_norm(s, 20)
    mr = max_residue_fraction(s)
    mrn = max_run_length(s)
    ku = kmer_unique_fraction(s, int(cfg.kmer_k))
    p, motif, match = best_small_period_repeat(s, int(cfg.repeat_max_period))
    motif_unique = len(set(motif)) if motif else 0

    info.update(
        {
            "entropy_norm": float(en),
            "max_residue_fraction": float(mr),
            "max_run_length": int(mrn),
            "kmer_unique_frac": float(ku),
            "repeat_period": int(p),
            "repeat_motif": motif,
            "repeat_match_frac": float(match),
            "repeat_motif_unique_aa": int(motif_unique),
        }
    )

    low_entropy = en < float(cfg.entropy_threshold)
    low_kmer = ku < float(cfg.kmer_unique_frac_threshold)
    max_res = mr > float(cfg.max_res_frac_threshold)
    simple_repeat = (
        (p > 0)
        and (len(s) / float(p) >= int(cfg.repeat_min_repeats))
        and (match >= float(cfg.repeat_match_threshold))
        and (motif_unique <= int(cfg.repeat_max_unique_aa_in_motif))
    )
    long_run = mrn >= int(cfg.max_run_threshold)

    # Be conservative: reject if any strong low-quality flag triggers.
    if long_run:
        return True, {**info, "reason": "long_homopolymer_run"}
    if simple_repeat:
        return True, {**info, "reason": "simple_repeat"}
    if low_entropy or low_kmer or max_res:
        return True, {**info, "reason": "low_complexity"}
    return False, info


def filter_dplm_sequences(
    sequences: Sequence[str],
    *,
    cfg: DplmQualityFilterConfig,
    expected_length: int | None = None,
) -> tuple[list[str], dict[str, float | int | None]]:
    """Filter DPLM samples. Preserves order, removes duplicates, returns stats."""
    kept: list[str] = []
    stats = {
        "raw": 0,
        "kept": 0,
        "rejected": 0,
        "rejected_bad_length": 0,
        "rejected_duplicate": 0,
        "rejected_low_quality": 0,
        "entropy_norm_mean": None,
        "entropy_norm_min": None,
        "entropy_norm_max": None,
    }
    seen: set[str] = set()
    entropy_values: list[float] = []
    for raw in sequences or ():
        seq = (raw or "").strip().upper()
        if not seq:
            stats["raw"] += 1
            stats["rejected"] += 1
            stats["rejected_low_quality"] += 1
            continue
        stats["raw"] += 1
        if expected_length is not None and len(seq) != int(expected_length):
            stats["rejected"] += 1
            stats["rejected_bad_length"] += 1
            continue
        if seq in seen:
            stats["rejected"] += 1
            stats["rejected_duplicate"] += 1
            continue
        seen.add(seq)
        bad, info = _is_low_quality(seq, cfg)
        entropy = info.get("entropy_norm")
        if isinstance(entropy, (int, float)):
            entropy_values.append(float(entropy))
        if bad:
            stats["rejected"] += 1
            stats["rejected_low_quality"] += 1
            continue
        kept.append(seq)
    stats["kept"] = len(kept)
    if entropy_values:
        stats["entropy_norm_mean"] = float(sum(entropy_values) / len(entropy_values))
        stats["entropy_norm_min"] = float(min(entropy_values))
        stats["entropy_norm_max"] = float(max(entropy_values))
    return kept, stats


def build_quality_filter_config(obj: dict) -> DplmQualityFilterConfig:
    def _get(key: str, default):
        v = obj.get(key, default)
        return default if v is None else v

    return DplmQualityFilterConfig(
        entropy_threshold=float(_get("entropy_threshold", DplmQualityFilterConfig.entropy_threshold)),
        max_res_frac_threshold=float(
            _get("max_res_frac_threshold", DplmQualityFilterConfig.max_res_frac_threshold)
        ),
        kmer_k=int(_get("kmer_k", DplmQualityFilterConfig.kmer_k)),
        kmer_unique_frac_threshold=float(
            _get("kmer_unique_frac_threshold", DplmQualityFilterConfig.kmer_unique_frac_threshold)
        ),
        repeat_max_period=int(_get("repeat_max_period", DplmQualityFilterConfig.repeat_max_period)),
        repeat_match_threshold=float(
            _get("repeat_match_threshold", DplmQualityFilterConfig.repeat_match_threshold)
        ),
        repeat_min_repeats=int(_get("repeat_min_repeats", DplmQualityFilterConfig.repeat_min_repeats)),
        repeat_max_unique_aa_in_motif=int(
            _get("repeat_max_unique_aa_in_motif", DplmQualityFilterConfig.repeat_max_unique_aa_in_motif)
        ),
        max_run_threshold=int(_get("max_run_threshold", DplmQualityFilterConfig.max_run_threshold)),
        alphabet=str(_get("alphabet", DplmQualityFilterConfig.alphabet)),
    )
