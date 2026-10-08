from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CANONICAL_AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_CHANNEL_BIAS_TEMPERATURE = 0.3
DEFAULT_PORE_PROFILE = {
    "S": 1.0,
    "T": 1.0,
    "N": 1.0,
    "Q": 1.0,
    "Y": -0.6,
    "V": -0.3,
    "I": -0.3,
    "L": -0.3,
    "F": -0.3,
    "D": -0.6,
    "E": -0.6,
    "K": -0.6,
    "R": -0.6,
    "H": -0.6,
}
DEFAULT_MOUTH_BONUS: dict[str, float] = {}


def _normalize_residue_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _normalize_omit_mapping(
    omit_aas_by_residue: Mapping[str, str | Iterable[str]] | None,
) -> dict[str, str]:
    if omit_aas_by_residue is None:
        return {}
    out: dict[str, str] = {}
    for residue_name, raw_values in omit_aas_by_residue.items():
        residue_token = str(residue_name or "").strip()
        if not residue_token:
            continue
        if isinstance(raw_values, str):
            aas = [aa for aa in raw_values if aa.strip()]
        else:
            aas = [str(value or "").strip() for value in raw_values]
        compact = "".join(dict.fromkeys(aa.upper() for aa in aas if aa))
        if compact:
            out[residue_token] = compact
    return out


def _merge_omit_values(
    *,
    auto_omit: Iterable[str],
    explicit_omit: str | Iterable[str] | None,
) -> str:
    explicit_set = set()
    if isinstance(explicit_omit, str):
        explicit_set = {aa.upper() for aa in explicit_omit if aa.strip()}
    elif explicit_omit is not None:
        explicit_set = {str(value or "").strip().upper() for value in explicit_omit if str(value or "").strip()}
    merged = explicit_set | {str(aa).strip().upper() for aa in auto_omit if str(aa).strip()}
    return "".join(aa for aa in CANONICAL_AA_ORDER if aa in merged)


def _profile_for_mode(
    *,
    additive_profile: Mapping[str, float],
    mode: str,
    temperature: float,
) -> dict[str, float]:
    normalized = {str(aa): float(weight) for aa, weight in additive_profile.items()}
    if mode == "additive":
        return normalized
    if mode == "multiplicative":
        if temperature <= 0:
            raise ValueError("Channel aa_bias multiplicative mode requires temperature > 0.")
        return {
            aa: float(math.exp(weight / temperature))
            for aa, weight in normalized.items()
        }
    raise ValueError(f"Unsupported aa_bias mode: {mode!r}")


def _merged_bias_for_region(
    *,
    base_profile: Mapping[str, float],
    mouth_bonus: Mapping[str, float],
    is_mouth: bool,
) -> dict[str, float]:
    merged = {str(aa): float(weight) for aa, weight in base_profile.items()}
    if is_mouth:
        for aa, weight in mouth_bonus.items():
            merged[str(aa)] = float(merged.get(str(aa), 0.0)) + float(weight)
    return merged


def build_channel_aa_bias(
    *,
    pore_lining_residues: Sequence[str],
    extracellular_mouth_residues: Sequence[str] | None = None,
    pore_profile: Mapping[str, float] | None = None,
    mouth_bonus: Mapping[str, float] | None = None,
    omit_aas_by_residue: Mapping[str, str | Iterable[str]] | None = None,
    pdb_path: str | None = None,
    mode: str = "multiplicative",
    temperature: float = DEFAULT_CHANNEL_BIAS_TEMPERATURE,
) -> dict[str, Any]:
    pore_residues = _normalize_residue_list(pore_lining_residues)
    mouth_residues = _normalize_residue_list(extracellular_mouth_residues)
    profile = dict(DEFAULT_PORE_PROFILE if pore_profile is None else pore_profile)
    mouth = dict(DEFAULT_MOUTH_BONUS if mouth_bonus is None else mouth_bonus)
    explicit_omit_mapping = _normalize_omit_mapping(omit_aas_by_residue)

    per_residue_bias: dict[str, dict[str, float]] = {}
    omit_mapping: dict[str, str] = {}
    for residue_name in pore_residues:
        additive_profile = _merged_bias_for_region(
            base_profile=profile,
            mouth_bonus=mouth,
            is_mouth=False,
        )
        per_residue_bias[residue_name] = _profile_for_mode(
            additive_profile=additive_profile,
            mode=mode,
            temperature=temperature,
        )
        unspecified = [aa for aa in CANONICAL_AA_ORDER if aa not in additive_profile]
        omit_mapping[residue_name] = _merge_omit_values(
            auto_omit=unspecified,
            explicit_omit=explicit_omit_mapping.get(residue_name, ""),
        )
    for residue_name in mouth_residues:
        additive_profile = _merged_bias_for_region(
            base_profile=profile,
            mouth_bonus=mouth,
            is_mouth=True,
        )
        per_residue_bias[residue_name] = _profile_for_mode(
            additive_profile=additive_profile,
            mode=mode,
            temperature=temperature,
        )
        unspecified = [aa for aa in CANONICAL_AA_ORDER if aa not in additive_profile]
        omit_mapping[residue_name] = _merge_omit_values(
            auto_omit=unspecified,
            explicit_omit=explicit_omit_mapping.get(residue_name, ""),
        )

    payload: dict[str, Any] = {
        "mode": mode,
        "bias_AA": "",
        "bias_AA_per_residue": per_residue_bias,
        "omit_AA_per_residue": omit_mapping,
    }
    if pdb_path:
        payload["bias_AA_per_residue_multi"] = {
            str(pdb_path): per_residue_bias,
        }
        payload["omit_AA_per_residue_multi"] = {
            str(pdb_path): omit_mapping,
        }
    return payload
