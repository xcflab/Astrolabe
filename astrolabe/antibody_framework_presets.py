from __future__ import annotations

from dataclasses import dataclass
import pathlib
import re
from typing import Any, Mapping, Sequence

from astrolabe.core.structure_parser import create_structure_parser


_SCFV_LINKER = "GGGGSGGGGSGGGGS"  # (GGGGS)3

# Chothia loop ranges (1-indexed, inclusive).
_CHOTHIA_CDR_RANGES: dict[str, tuple[int, int]] = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

_VHH_LOOPS: tuple[str, ...] = ("H1", "H2", "H3")
_LIGHT_LOOPS: tuple[str, ...] = ("L1", "L2", "L3")
_SCFV_VHVL_LOOPS: tuple[str, ...] = ("H1", "H2", "H3", "L1", "L2", "L3")
_SCFV_VLVH_LOOPS: tuple[str, ...] = ("L1", "L2", "L3", "H1", "H2", "H3")
_FRAMEWORK_HOMOLOGY_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "framework_homologies"

_LEN_RANGE_RE = re.compile(r"^\d+(?:-\d+)?$")


@dataclass(frozen=True)
class AntibodyFrameworkPreset:
    """
    Sequence-only antibody/scFv binder presets for `gh design antibody`.

    Notes:
    - These are protomer-statement segments: `+` prefixes fixed framework segments; numeric
      `a-b` segments define mutable random-length ranges.
    - `binder_segments` are whitespace-joined when constructing `--binder-seq`.
    """

    name: str
    description: str
    binder_segments: tuple[str, ...]
    source: str
    kind: str
    loop_order: tuple[str, ...]
    source_loop_lengths: Mapping[str, int]
    aliases: tuple[str, ...] = ()

    def binder_seq(self, cdr_lengths: Mapping[str, str] | None = None) -> str:
        merged = _merge_cdr_lengths(
            loop_order=self.loop_order,
            base_ranges={k: None for k in self.loop_order},
            overrides=cdr_lengths,
        )
        return " ".join(_apply_loop_ranges(self.binder_segments, self.loop_order, merged)).strip()

    def fixed_segments(self) -> tuple[str, ...]:
        return _coalesce_fixed_segments(self.binder_segments)

    def homology_layout(self) -> dict[str, Any]:
        return {
            "fixed_segments": list(self.fixed_segments()),
            "loop_order": list(self.loop_order),
            "source_loop_lengths": {
                loop: int(self.source_loop_lengths[loop]) for loop in self.loop_order
            },
        }


_S = AntibodyFrameworkPreset


def _normalize_len_range(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty CDR length range")
    if not _LEN_RANGE_RE.fullmatch(s):
        raise ValueError(
            f"Invalid CDR length range: {raw!r} (expected '<n>' or '<lo>-<hi>')"
        )
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
    else:
        lo = int(s)
        hi = lo
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError(f"Invalid CDR length range: {raw!r}")
    return f"{lo}-{hi}"


def _is_len_range_token(token: str) -> bool:
    return bool(_LEN_RANGE_RE.fullmatch(str(token or "").strip()))


def _strip_fixed_prefix(token: str) -> str:
    text = str(token or "").strip()
    return text[1:] if text.startswith("+") else text


def _coalesce_fixed_segments(tokens: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    current: list[str] = []
    for token in tokens:
        if _is_len_range_token(token):
            if current:
                merged.append("".join(current))
                current = []
            continue
        current.append(_strip_fixed_prefix(token))
    if current:
        merged.append("".join(current))
    return tuple(merged)


def parse_cdr_length_overrides(raw: str | None) -> dict[str, str]:
    """Parse '--cdr-lengths' style input, e.g. 'H1:7-11,H2:6-10'."""
    text = str(raw or "").strip()
    if not text:
        return {}

    out: dict[str, str] = {}
    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid CDR length item {part!r} (expected '<LOOP>:<RANGE>', e.g. H1:7-11)"
            )
        loop_raw, range_raw = part.split(":", 1)
        loop = loop_raw.strip().upper()
        if not loop:
            raise ValueError(f"Invalid CDR loop label in item {part!r}")
        out[loop] = _normalize_len_range(range_raw)
    return out


def _merge_cdr_lengths(
    *,
    loop_order: tuple[str, ...],
    base_ranges: Mapping[str, str | None],
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    allowed = set(loop_order)
    merged: dict[str, str] = {}
    for loop in loop_order:
        base = base_ranges.get(loop)
        if base is None:
            merged[loop] = ""
        else:
            merged[loop] = _normalize_len_range(base)

    overrides = overrides or {}
    bad_keys = sorted(k for k in overrides.keys() if k not in allowed)
    if bad_keys:
        raise ValueError(
            f"Unsupported CDR labels for this framework: {bad_keys}. Allowed: {sorted(allowed)}"
        )
    for loop, raw_range in overrides.items():
        merged[loop] = _normalize_len_range(raw_range)

    return merged


def _apply_loop_ranges(
    binder_segments: tuple[str, ...],
    loop_order: tuple[str, ...],
    merged_ranges: Mapping[str, str],
) -> tuple[str, ...]:
    out: list[str] = []
    loop_idx = 0
    for token in binder_segments:
        tok = str(token).strip()
        if _is_len_range_token(tok):
            if loop_idx >= len(loop_order):
                raise ValueError(
                    "Framework definition has more mutable length segments than loop labels"
                )
            loop = loop_order[loop_idx]
            repl = str(merged_ranges.get(loop, "")).strip() or tok
            out.append(_normalize_len_range(repl))
            loop_idx += 1
        else:
            out.append(tok)

    if loop_idx != len(loop_order):
        raise ValueError(
            "Framework definition loop count mismatch: "
            f"mutable_segments={loop_idx} expected={len(loop_order)}"
        )
    return tuple(out)


ANTIBODY_FRAMEWORK_PRESETS: dict[str, AntibodyFrameworkPreset] = {
    # RFantibody example: hu-4D5-8_Fv.pdb (Chothia-style splitting).
    "scfv_vh_linker_vl__rfantibody_4d5": _S(
        name="scfv_vh_linker_vl__rfantibody_4d5",
        aliases=("scfv4d5_vhvl", "4d5_vhvl"),
        kind="scfv",
        loop_order=_SCFV_VHVL_LOOPS,
        description="scFv (4D5/Herceptin): VH-(GGGGS)3-VL; CDR length ranges from RFantibody examples.",
        binder_segments=(
            "+EVQLVESGGGLVQPGGSLRLSCAAS",
            "7-7",
            "+YIHWVRQAPGKGLEWVARI",
            "6-6",
            "+YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVY",
            "5-13",
            "+MDYWGQGTLVTVSS",
            f"+{_SCFV_LINKER}",
            "+DIQMTQSPSSLSASVGDRVTITC",
            "8-13",
            "+WYQQKPGKAPKLLIY",
            "7-7",
            "+GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC",
            "9-11",
            "+FGQGTKVEIK",
        ),
        source="RosettaCommons/RFantibody (hu-4D5-8_Fv.pdb + example scripts)",
        source_loop_lengths={"H1": 7, "H2": 5, "H3": 8, "L1": 11, "L2": 7, "L3": 9},
    ),
    "scfv_vl_linker_vh__rfantibody_4d5": _S(
        name="scfv_vl_linker_vh__rfantibody_4d5",
        aliases=("scfv4d5_vlvh", "4d5_vlvh"),
        kind="scfv",
        loop_order=_SCFV_VLVH_LOOPS,
        description="scFv (4D5/Herceptin): VL-(GGGGS)3-VH; CDR length ranges from RFantibody examples.",
        binder_segments=(
            "+DIQMTQSPSSLSASVGDRVTITC",
            "8-13",
            "+WYQQKPGKAPKLLIY",
            "7-7",
            "+GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC",
            "9-11",
            "+FGQGTKVEIK",
            f"+{_SCFV_LINKER}",
            "+EVQLVESGGGLVQPGGSLRLSCAAS",
            "7-7",
            "+YIHWVRQAPGKGLEWVARI",
            "6-6",
            "+YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVY",
            "5-13",
            "+MDYWGQGTLVTVSS",
        ),
        source="RosettaCommons/RFantibody (hu-4D5-8_Fv.pdb + example scripts)",
        source_loop_lengths={"L1": 11, "L2": 7, "L3": 9, "H1": 7, "H2": 5, "H3": 8},
    ),
    # boltzgen nanobody scaffolds: 7eow/7xl0/8coh/8z8v.
    "vhh__boltzgen_7eow": _S(
        name="vhh__boltzgen_7eow",
        aliases=("vhh_7eow",),
        kind="vhh",
        loop_order=_VHH_LOOPS,
        description="VHH/nanobody scaffold 7eow; variable CDRs derived from boltzgen scaffold YAML.",
        binder_segments=(
            "+EVQLVESGGGLVQPGGSLRLSCAAS",
            "7-11",
            "+GWFRQAPGKGRELVAAI",
            "6-10",
            "+YPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCA",
            "15-28",
            "+GQGTQVTVSS",
        ),
        source="HannesStark/boltzgen (example/nanobody_scaffolds/7eow.yaml)",
        source_loop_lengths={"H1": 9, "H2": 8, "H3": 21},
    ),
    "vhh__boltzgen_7xl0": _S(
        name="vhh__boltzgen_7xl0",
        aliases=("vhh_7xl0",),
        kind="vhh",
        loop_order=_VHH_LOOPS,
        description="VHH/nanobody scaffold 7xl0; variable CDRs derived from boltzgen scaffold YAML.",
        binder_segments=(
            "+EVQLVESGGGLVQPGGSLRLSCAAS",
            "6-10",
            "+MAWYRQAPGKGRELVAG",
            "5-9",
            "+SYADSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCA",
            "9-20",
            "+WGQGTLVTVSS",
        ),
        source="HannesStark/boltzgen (example/nanobody_scaffolds/7xl0.yaml)",
        source_loop_lengths={"H1": 8, "H2": 7, "H3": 14},
    ),
    "vhh__boltzgen_8coh": _S(
        name="vhh__boltzgen_8coh",
        aliases=("vhh_8coh",),
        kind="vhh",
        loop_order=_VHH_LOOPS,
        description="VHH/nanobody scaffold 8coh; variable CDRs derived from boltzgen scaffold YAML.",
        binder_segments=(
            "+EVQLVESGGGLVQPGGSLRLSCAAS",
            "6-10",
            "+MAWFRQAPGQEREFVAG",
            "6-10",
            "+LYADSVRGRFTNSRDNSKNTLYLQMNSLRAEDTAVYYC",
            "13-26",
            "+WGQGTLVTVSSHHHHHH",
        ),
        source="HannesStark/boltzgen (example/nanobody_scaffolds/8coh.yaml)",
        source_loop_lengths={"H1": 8, "H2": 8, "H3": 19},
    ),
    "vhh__boltzgen_8z8v": _S(
        name="vhh__boltzgen_8z8v",
        aliases=("vhh_8z8v",),
        kind="vhh",
        loop_order=_VHH_LOOPS,
        description="VHH/nanobody scaffold 8z8v; variable CDRs derived from boltzgen scaffold YAML.",
        binder_segments=(
            "+EVQLVESGGGLVQPGNSLRLSCAAS",
            "6-10",
            "+MSWVRQAPGKGLEWVSS",
            "6-10",
            "+LYADSVKGRFTISRDNAKTTLYLQMNSLRPEDTAVYYCT",
            "9-20",
            "+TLVTVSS",
        ),
        source="HannesStark/boltzgen (example/nanobody_scaffolds/8z8v.yaml)",
        source_loop_lengths={"H1": 8, "H2": 8, "H3": 11},
    ),
}


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _preset in ANTIBODY_FRAMEWORK_PRESETS.items():
    _ALIAS_TO_CANONICAL[_canonical] = _canonical
    for _alias in _preset.aliases:
        key = str(_alias).strip()
        if not key:
            continue
        if key in _ALIAS_TO_CANONICAL and _ALIAS_TO_CANONICAL[key] != _canonical:
            raise ValueError(
                f"Duplicate framework alias {key!r}: {_canonical} vs {_ALIAS_TO_CANONICAL[key]}"
            )
        _ALIAS_TO_CANONICAL[key] = _canonical


def _resolve_framework_name(name_or_alias: str) -> str:
    key = str(name_or_alias or "").strip()
    if not key:
        raise ValueError("Empty framework preset name")
    canonical = _ALIAS_TO_CANONICAL.get(key)
    if canonical is None:
        available = ", ".join(sorted(ANTIBODY_FRAMEWORK_PRESETS.keys()))
        aliases = ", ".join(sorted(k for k in _ALIAS_TO_CANONICAL.keys() if k not in ANTIBODY_FRAMEWORK_PRESETS))
        raise ValueError(
            f"Unknown antibody framework preset: {key}. Available names: {available}. "
            f"Available aliases: {aliases}"
        )
    return canonical


def get_antibody_framework_preset(name: str) -> AntibodyFrameworkPreset:
    canonical = _resolve_framework_name(name)
    return ANTIBODY_FRAMEWORK_PRESETS[canonical]


def list_antibody_framework_presets() -> tuple[AntibodyFrameworkPreset, ...]:
    return tuple(ANTIBODY_FRAMEWORK_PRESETS[k] for k in sorted(ANTIBODY_FRAMEWORK_PRESETS.keys()))


def _default_framework_homology_source(framework_name: str) -> str:
    return f"{_FRAMEWORK_HOMOLOGY_DIR / f'{framework_name}.json'}:A"


def build_framework_homology_config(
    *,
    framework: str | None = None,
    framework_name: str | None = None,
    framework_homology: Any | None = None,
    fixed_segments: Sequence[str] | None = None,
    loop_order: Sequence[str] | None = None,
    source_loop_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if framework is not None:
        preset = get_antibody_framework_preset(framework)
        framework_name = preset.name
        fixed_segments = preset.fixed_segments()
        loop_order = preset.loop_order
        source_loop_lengths = preset.source_loop_lengths
        source_value = framework_homology or _default_framework_homology_source(
            preset.name
        )
        source_path = pathlib.Path(str(source_value).split(":", 1)[0])
        if framework_homology is None and not source_path.is_file():
            raise FileNotFoundError(
                f"Missing built-in framework homology resource for {preset.name}: {source_value}"
            )
    else:
        if not framework_name:
            raise ValueError("framework_name is required when framework is not provided")
        if not framework_homology:
            raise ValueError("framework_homology is required for custom framework layout")
        if fixed_segments is None or loop_order is None or source_loop_lengths is None:
            raise ValueError("fixed_segments, loop_order, and source_loop_lengths are required")
        source_value = framework_homology

    return {
        "_type": "antibody_framework_homology",
        "framework_name": str(framework_name),
        "source": source_value,
        "layout": {
            "fixed_segments": [str(part) for part in fixed_segments or ()],
            "loop_order": [str(loop) for loop in loop_order or ()],
            "source_loop_lengths": {
                str(loop): int(source_loop_lengths[loop]) for loop in loop_order or ()
            },
        },
    }


def _normalize_kind(raw: str | None) -> str:
    kind = str(raw or "auto").strip().lower()
    if kind not in {"auto", "vhh", "scfv"}:
        raise ValueError(f"Invalid framework kind {raw!r} (expected: auto|vhh|scfv)")
    return kind


def _normalize_scfv_order(raw: str | None) -> str:
    order = str(raw or "vh_linker_vl").strip().lower().replace("-", "_")
    if order not in {"vh_linker_vl", "vl_linker_vh"}:
        raise ValueError(
            f"Invalid scFv order {raw!r} (expected: vh_linker_vl|vl_linker_vh)"
        )
    return order


def _load_structure_chain_data(
    structure_path: pathlib.Path,
) -> dict[str, tuple[list[int], str]]:
    parser = create_structure_parser(str(structure_path))
    parsed = parser.parse()
    out: dict[str, tuple[list[int], str]] = {}
    for chain_id, (indices, seq) in parsed.items():
        out[str(chain_id)] = (list(indices), str(seq))
    return out


def _require_chain(
    parsed: Mapping[str, tuple[list[int], str]],
    chain_id: str,
    *,
    role: str,
) -> tuple[list[int], str]:
    key = str(chain_id or "").strip()
    if not key:
        raise ValueError(f"Missing {role} chain id")
    if key not in parsed:
        available = sorted(parsed.keys())
        raise ValueError(
            f"{role} chain '{key}' not found in structure. Available chains: {available}"
        )
    indices, seq = parsed[key]
    if not seq:
        raise ValueError(f"{role} chain '{key}' has empty sequence")
    return indices, seq


def _split_framework_and_cdr(
    indices: list[int],
    sequence: str,
    loop_names: tuple[str, ...],
) -> tuple[list[str], dict[str, int]]:
    if len(indices) != len(sequence):
        raise ValueError(
            f"Structure chain parse mismatch: len(indices)={len(indices)} len(sequence)={len(sequence)}"
        )
    residues = list(zip(indices, sequence, strict=True))
    if not residues:
        raise ValueError("Empty chain sequence")

    # Structure parser already yields residue-index order, but keep this robust.
    residues.sort(key=lambda x: x[0])

    framework_parts: list[str] = []
    cdr_lengths: dict[str, int] = {}

    prev_end = residues[0][0] - 1
    cursor = 0
    for loop in loop_names:
        start, end = _CHOTHIA_CDR_RANGES[loop]

        fr_chars: list[str] = []
        while cursor < len(residues):
            idx, aa = residues[cursor]
            if idx >= start:
                break
            if idx > prev_end:
                fr_chars.append(aa)
            cursor += 1

        cdr_chars: list[str] = []
        while cursor < len(residues):
            idx, aa = residues[cursor]
            if idx > end:
                break
            if idx >= start:
                cdr_chars.append(aa)
            cursor += 1

        fr_seq = "".join(fr_chars)
        cdr_seq = "".join(cdr_chars)
        framework_parts.append(fr_seq)
        cdr_len = len(cdr_seq)
        if cdr_len <= 0:
            raise ValueError(
                f"Failed to infer {loop} residues from structure numbering range {start}-{end}. "
                "Use a Chothia-numbered H/L input, or provide a matching framework preset instead."
            )
        cdr_lengths[loop] = cdr_len
        prev_end = end

    tail = "".join(aa for idx, aa in residues[cursor:] if idx > prev_end)
    framework_parts.append(tail)
    return framework_parts, cdr_lengths


def _build_chain_tokens(
    *,
    framework_parts: list[str],
    loop_names: tuple[str, ...],
    merged_cdr_ranges: Mapping[str, str],
) -> list[str]:
    tokens: list[str] = []
    for idx, loop in enumerate(loop_names):
        fr = framework_parts[idx]
        if fr:
            tokens.append(f"+{fr}")
        cdr_range = str(merged_cdr_ranges[loop]).strip()
        if not cdr_range:
            raise ValueError(f"Missing CDR length range for {loop}")
        tokens.append(cdr_range)
    tail = framework_parts[-1]
    if tail:
        tokens.append(f"+{tail}")
    return tokens


def _build_framework_design_from_structure(
    *,
    structure: str,
    heavy_chain: str | None = None,
    light_chain: str | None = None,
    kind: str = "auto",
    scfv_order: str = "vh_linker_vl",
    scfv_linker: str = _SCFV_LINKER,
    cdr_lengths: Mapping[str, str] | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, int], dict[str, Any]]:
    structure_path = pathlib.Path(str(structure)).expanduser().resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"Framework structure not found: {structure_path}")

    parsed = _load_structure_chain_data(structure_path)
    available = sorted(parsed.keys())
    if not available:
        raise ValueError(f"No chains parsed from structure: {structure_path}")

    kind_norm = _normalize_kind(kind)
    heavy = str(heavy_chain or "").strip() or None
    light = str(light_chain or "").strip() or None

    if kind_norm == "auto":
        if heavy and light:
            kind_norm = "scfv"
        elif heavy and not light:
            kind_norm = "vhh"
        elif not heavy and not light:
            if "H" in parsed and "L" in parsed:
                heavy = "H"
                light = "L"
                kind_norm = "scfv"
            elif "H" in parsed:
                heavy = "H"
                kind_norm = "vhh"
            elif len(available) == 1:
                heavy = available[0]
                kind_norm = "vhh"
            else:
                raise ValueError(
                    "Cannot auto-detect framework kind/chains. Provide --framework-kind and chain ids "
                    f"(available chains: {available})."
                )
        else:
            raise ValueError(
                "Cannot infer framework from light chain only. Provide --framework-heavy-chain "
                "(and optionally --framework-light-chain for scFv)."
            )

    cdr_overrides = {
        k.upper(): _normalize_len_range(v) for k, v in (cdr_lengths or {}).items()
    }

    if kind_norm == "vhh":
        if not heavy:
            raise ValueError("--framework-heavy-chain is required for vhh mode")
        if light:
            raise ValueError("--framework-light-chain is not allowed for vhh mode")

        heavy_indices, heavy_seq = _require_chain(parsed, heavy, role="heavy")
        heavy_fr, heavy_default_lengths = _split_framework_and_cdr(
            heavy_indices,
            heavy_seq,
            _VHH_LOOPS,
        )
        merged = _merge_cdr_lengths(
            loop_order=_VHH_LOOPS,
            base_ranges={k: f"{n}-{n}" for k, n in heavy_default_lengths.items()},
            overrides=cdr_overrides,
        )
        tokens = _build_chain_tokens(
            framework_parts=heavy_fr,
            loop_names=_VHH_LOOPS,
            merged_cdr_ranges=merged,
        )
        return (
            " ".join(tokens).strip(),
            _coalesce_fixed_segments(tokens),
            _VHH_LOOPS,
            dict(heavy_default_lengths),
            {
                "structure_path": str(structure_path),
                "kind": "vhh",
                "heavy_chain": heavy,
                "light_chain": None,
                "scfv_order": None,
                "scfv_linker": None,
                "source_chain_sequences": {"heavy": heavy_seq},
            },
        )

    if kind_norm != "scfv":
        raise ValueError(f"Unsupported framework kind: {kind_norm}")

    if not heavy or not light:
        raise ValueError(
            "scFv mode requires both --framework-heavy-chain and --framework-light-chain"
        )

    heavy_indices, heavy_seq = _require_chain(parsed, heavy, role="heavy")
    light_indices, light_seq = _require_chain(parsed, light, role="light")

    heavy_fr, heavy_default_lengths = _split_framework_and_cdr(
        heavy_indices,
        heavy_seq,
        _VHH_LOOPS,
    )
    light_fr, light_default_lengths = _split_framework_and_cdr(
        light_indices,
        light_seq,
        _LIGHT_LOOPS,
    )

    order = _normalize_scfv_order(scfv_order)
    if order == "vh_linker_vl":
        full_loop_order = _SCFV_VHVL_LOOPS
    else:
        full_loop_order = _SCFV_VLVH_LOOPS

    base_ranges = {
        "H1": f"{heavy_default_lengths['H1']}-{heavy_default_lengths['H1']}",
        "H2": f"{heavy_default_lengths['H2']}-{heavy_default_lengths['H2']}",
        "H3": f"{heavy_default_lengths['H3']}-{heavy_default_lengths['H3']}",
        "L1": f"{light_default_lengths['L1']}-{light_default_lengths['L1']}",
        "L2": f"{light_default_lengths['L2']}-{light_default_lengths['L2']}",
        "L3": f"{light_default_lengths['L3']}-{light_default_lengths['L3']}",
    }
    merged = _merge_cdr_lengths(
        loop_order=full_loop_order,
        base_ranges=base_ranges,
        overrides=cdr_overrides,
    )

    heavy_tokens = _build_chain_tokens(
        framework_parts=heavy_fr,
        loop_names=_VHH_LOOPS,
        merged_cdr_ranges=merged,
    )
    light_tokens = _build_chain_tokens(
        framework_parts=light_fr,
        loop_names=_LIGHT_LOOPS,
        merged_cdr_ranges=merged,
    )

    linker_seq = str(scfv_linker or "").strip()
    if not linker_seq:
        raise ValueError("Empty scFv linker sequence")
    linker_token = f"+{linker_seq}"

    if order == "vh_linker_vl":
        tokens = [*heavy_tokens, linker_token, *light_tokens]
    else:
        tokens = [*light_tokens, linker_token, *heavy_tokens]

    source_lengths = {
        "H1": int(heavy_default_lengths["H1"]),
        "H2": int(heavy_default_lengths["H2"]),
        "H3": int(heavy_default_lengths["H3"]),
        "L1": int(light_default_lengths["L1"]),
        "L2": int(light_default_lengths["L2"]),
        "L3": int(light_default_lengths["L3"]),
    }
    return (
        " ".join(tokens).strip(),
        _coalesce_fixed_segments(tokens),
        full_loop_order,
        source_lengths,
        {
            "structure_path": str(structure_path),
            "kind": "scfv",
            "heavy_chain": heavy,
            "light_chain": light,
            "scfv_order": order,
            "scfv_linker": linker_seq,
            "source_chain_sequences": {
                "heavy": heavy_seq,
                "light": light_seq,
            },
        },
    )


def build_framework_layout_from_structure(
    *,
    structure: str,
    heavy_chain: str | None = None,
    light_chain: str | None = None,
    kind: str = "auto",
    scfv_order: str = "vh_linker_vl",
    scfv_linker: str = _SCFV_LINKER,
    cdr_lengths: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    (
        binder_seq,
        fixed_segments,
        loop_order,
        source_loop_lengths,
        source_meta,
    ) = _build_framework_design_from_structure(
        structure=structure,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        kind=kind,
        scfv_order=scfv_order,
        scfv_linker=scfv_linker,
        cdr_lengths=cdr_lengths,
    )
    framework_name = f"framework_pdb::{pathlib.Path(str(structure)).expanduser().resolve().stem}"
    return (
        binder_seq,
        {
            "framework_name": framework_name,
            "fixed_segments": fixed_segments,
            "loop_order": loop_order,
            "source_loop_lengths": source_loop_lengths,
            **source_meta,
        },
    )


def build_framework_binder_design_from_structure(
    *,
    structure: str,
    framework_homology: str,
    heavy_chain: str | None = None,
    light_chain: str | None = None,
    kind: str = "auto",
    scfv_order: str = "vh_linker_vl",
    scfv_linker: str = _SCFV_LINKER,
    cdr_lengths: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    binder_seq, framework_layout = build_framework_layout_from_structure(
        structure=structure,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        kind=kind,
        scfv_order=scfv_order,
        scfv_linker=scfv_linker,
        cdr_lengths=cdr_lengths,
    )
    return (
        binder_seq,
        build_framework_homology_config(
            framework_name=str(framework_layout["framework_name"]),
            framework_homology=framework_homology,
            fixed_segments=framework_layout["fixed_segments"],
            loop_order=framework_layout["loop_order"],
            source_loop_lengths=framework_layout["source_loop_lengths"],
        ),
    )


def build_framework_binder_seq_from_structure(
    *,
    structure: str,
    heavy_chain: str | None = None,
    light_chain: str | None = None,
    kind: str = "auto",
    scfv_order: str = "vh_linker_vl",
    scfv_linker: str = _SCFV_LINKER,
    cdr_lengths: Mapping[str, str] | None = None,
) -> str:
    """
    Build a protomer statement from an antibody framework structure.

    `heavy_chain`/`light_chain` should point to H/L-like chains in a Chothia-numbered structure.
    """
    binder_seq, _ = build_framework_layout_from_structure(
        structure=structure,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        kind=kind,
        scfv_order=scfv_order,
        scfv_linker=scfv_linker,
        cdr_lengths=cdr_lengths,
    )
    return binder_seq
