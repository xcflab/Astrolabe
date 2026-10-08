from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

VALID_BOND_ORDERS = frozenset({"single", "double", "triple", "aromatic"})
VALID_CYCLIZATION_SIGNAL_MODES = frozenset(
    {"none", "index_only", "bond_only", "both"}
)
DEFAULT_GENERIC_TARGET_DISTANCE_BY_ORDER = {
    "single": 1.54,
    "double": 1.34,
    "triple": 1.20,
    "aromatic": 1.40,
}
HEAD_TO_TAIL_TARGET_DISTANCE = 1.33
DISULFIDE_TARGET_DISTANCE = 2.03

import numpy as np

from astrolabe.utils.mmcif_ids import int_id_to_str_id


@dataclass(frozen=True)
class CyclizationAtomRef:
    proto: str
    pos: int | str
    atom: str
    aa: str | None = None


@dataclass(frozen=True)
class CyclizationBond:
    a: CyclizationAtomRef
    b: CyclizationAtomRef
    offset_window: int | None = None
    order: str | None = None
    target_distance: float | None = None
    signal_mode: str | None = None


@dataclass(frozen=True)
class CyclicEncodingSpec:
    enabled: bool = True
    mode: str = "equivalent"
    scope: str = "head_to_tail_only"


@dataclass(frozen=True)
class CyclizationSpec:
    bonds: tuple[CyclizationBond, ...] = ()
    offset_window: int | None = None
    encoding: CyclicEncodingSpec = CyclicEncodingSpec()
    signal_mode: str = "both"

    @property
    def emits_index_signals(self) -> bool:
        return self.signal_mode in {"index_only", "both"}

    @property
    def emits_bond_signals(self) -> bool:
        return self.signal_mode in {"bond_only", "both"}


@dataclass(frozen=True)
class CyclicRelativeOffset:
    residue_i: int  # 1-indexed within chain
    residue_j: int  # 1-indexed within chain
    offset: int


@dataclass(frozen=True)
class CyclicRelativePatch:
    chain_id: str
    offsets: tuple[CyclicRelativeOffset, ...]


def flatten_cyclic_relative_patches(
    patches: Iterable[CyclicRelativePatch] | None,
) -> tuple[tuple[str, int, int, int], ...]:
    flattened: list[tuple[str, int, int, int]] = []
    for patch in patches or ():
        chain_id = str(patch.chain_id)
        for rel in patch.offsets:
            flattened.append(
                (chain_id, int(rel.residue_i), int(rel.residue_j), int(rel.offset))
            )
    return tuple(flattened)


def serialize_cyclic_relative_patches(
    patches: Iterable[CyclicRelativePatch] | None,
    *,
    chain_ids: Iterable[str],
    asym_id_start: int = 1,
) -> tuple[tuple[int, int, int, int], ...]:
    chain_map = {
        str(chain_id): idx + int(asym_id_start)
        for idx, chain_id in enumerate(chain_ids)
    }
    serialized: list[tuple[int, int, int, int]] = []
    for chain_id, residue_i, residue_j, offset in flatten_cyclic_relative_patches(
        patches
    ):
        if chain_id not in chain_map:
            raise ValueError(f"Unknown chain_id in cyclic relative patch: {chain_id!r}")
        serialized.append((chain_map[chain_id], residue_i, residue_j, offset))
    return tuple(serialized)


def _as_mapping(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "keys"):
        return {str(k): cfg[k] for k in cfg.keys()}
    raise TypeError(f"Unsupported cyclization config type: {type(cfg)}")


def _normalize_offset_window(value: Any, *, context: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        window = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer, got {value!r}") from exc
    if window <= 0:
        raise ValueError(f"{context} must be a positive integer, got {window}")
    return window


def parse_cyclization(cyclization_cfg: Any) -> CyclizationSpec | None:
    if cyclization_cfg is None:
        return None
    cfg = _as_mapping(cyclization_cfg)
    if not cfg:
        return None

    head_to_tail = bool(cfg.get("head_to_tail", False))
    offset_window = _normalize_offset_window(
        cfg.get("offset_window"), context="cyclization.offset_window"
    )
    encoding = _parse_cyclic_encoding(cfg.get("encoding"))
    signal_mode = _parse_cyclization_signal_mode(cfg.get("signal_mode"))
    head_to_tail_signal_mode_raw = cfg.get("head_to_tail_signal_mode")
    head_to_tail_signal_mode = (
        _parse_cyclization_signal_mode(head_to_tail_signal_mode_raw)
        if head_to_tail_signal_mode_raw is not None
        else None
    )
    bonds_raw = cfg.get("bonds") or []
    if bonds_raw is None:
        bonds_raw = []

    bonds: list[CyclizationBond] = []
    for idx, item in enumerate(bonds_raw):
        bcfg = _as_mapping(item)
        if not bcfg:
            raise ValueError(f"cyclization.bonds[{idx}] is empty")
        if "a" not in bcfg or "b" not in bcfg:
            raise ValueError(f"cyclization.bonds[{idx}] must have keys: a, b")
        a = _as_mapping(bcfg["a"])
        b = _as_mapping(bcfg["b"])
        bonds.append(
            CyclizationBond(
                a=CyclizationAtomRef(
                    proto=str(a.get("proto", "")).strip(),
                    pos=a.get("pos"),
                    atom=str(a.get("atom", "")).strip(),
                    aa=(str(a["aa"]).strip() if a.get("aa") is not None else None),
                ),
                b=CyclizationAtomRef(
                    proto=str(b.get("proto", "")).strip(),
                    pos=b.get("pos"),
                    atom=str(b.get("atom", "")).strip(),
                    aa=(str(b["aa"]).strip() if b.get("aa") is not None else None),
                ),
                offset_window=_normalize_offset_window(
                    bcfg.get("offset_window"),
                    context=f"cyclization.bonds[{idx}].offset_window",
                ),
                order=(
                    str(bcfg.get("order")).strip()
                    if bcfg.get("order") is not None
                    else None
                ),
                target_distance=(
                    float(bcfg.get("target_distance"))
                    if bcfg.get("target_distance") is not None
                    else None
                ),
                signal_mode=(
                    _parse_cyclization_signal_mode(bcfg.get("signal_mode"))
                    if bcfg.get("signal_mode") is not None
                    else None
                ),
            )
        )

    if head_to_tail:
        bonds.insert(
            0,
            CyclizationBond(
                a=CyclizationAtomRef(proto="A", pos=1, atom="N"),
                b=CyclizationAtomRef(proto="A", pos=-1, atom="C"),
                signal_mode=str(head_to_tail_signal_mode or signal_mode),
            ),
        )

    spec = CyclizationSpec(
        bonds=tuple(bonds),
        offset_window=offset_window,
        encoding=encoding,
        signal_mode=signal_mode,
    )
    if not spec.bonds:
        return None
    return spec


def _parse_cyclization_signal_mode(raw_value: Any) -> str:
    if raw_value is None:
        return "both"
    mode = str(raw_value).strip().lower()
    if mode not in VALID_CYCLIZATION_SIGNAL_MODES:
        expected = "|".join(sorted(VALID_CYCLIZATION_SIGNAL_MODES))
        raise ValueError(
            f"Invalid cyclization.signal_mode={raw_value!r} (expected {expected})"
        )
    return mode


def _signal_mode_emits_index(signal_mode: str | None) -> bool:
    return str(signal_mode or "both").strip().lower() in {"index_only", "both"}


def _signal_mode_emits_bond(signal_mode: str | None) -> bool:
    return str(signal_mode or "both").strip().lower() in {"bond_only", "both"}


def _parse_cyclic_encoding(raw_cfg: Any) -> CyclicEncodingSpec:
    if raw_cfg is None:
        return CyclicEncodingSpec()
    if isinstance(raw_cfg, bool):
        return CyclicEncodingSpec(enabled=bool(raw_cfg))

    cfg = _as_mapping(raw_cfg)
    enabled = bool(cfg.get("enabled", True))
    mode = str(cfg.get("mode", "equivalent")).strip().lower()
    scope = str(cfg.get("scope", "head_to_tail_only")).strip().lower()

    scope_alias = {
        "head_to_tail": "head_to_tail_only",
        "head_to_tail_only": "head_to_tail_only",
        "all": "all_bonds",
        "all_bonds": "all_bonds",
    }
    scope = scope_alias.get(scope, scope)
    if scope not in {"head_to_tail_only", "all_bonds"}:
        raise ValueError(
            f"Invalid cyclization.encoding.scope={scope!r} (expected head_to_tail_only|all_bonds)"
        )

    if mode in {"off", "disabled", "none"}:
        enabled = False
        mode = "equivalent"
    mode_alias = {
        "equivalent": "equivalent",
        "generic": "equivalent",
        "generic_relative_patch": "equivalent",
        "global": "global_equivalent",
        "global_equivalent": "global_equivalent",
        "global-generic": "global_equivalent",
        "global_generic": "global_equivalent",
        "global_generic_relative_patch": "global_equivalent",
        "rfpeptide": "global_equivalent",
        "rfpeptide_like": "global_equivalent",
        "rfpeptide-like": "global_equivalent",
        "highfold": "highfold_cycpoem",
        "highfold_cycpoem": "highfold_cycpoem",
        "highfold-cycpoem": "highfold_cycpoem",
        "cycpoem": "highfold_cycpoem",
    }
    mode = mode_alias.get(mode, mode)
    if mode not in {"equivalent", "global_equivalent", "highfold_cycpoem"}:
        raise ValueError(
            f"Invalid cyclization.encoding.mode={mode!r} "
            "(expected equivalent|global_equivalent|highfold_cycpoem|off)"
        )
    return CyclicEncodingSpec(enabled=enabled, mode=mode, scope=scope)


def resolve_pos(pos: int | str, *, length: int) -> int:
    if length <= 0:
        raise ValueError(f"Invalid protomer length: {length}")
    if isinstance(pos, int):
        if pos == -1:
            return length
        if pos <= 0:
            raise ValueError(f"Residue positions must be 1-indexed (>0) or -1, got {pos}")
        if pos > length:
            raise ValueError(f"Residue position {pos} out of range (1..{length})")
        return pos
    if isinstance(pos, str):
        s = pos.strip().lower().replace(" ", "")
        if s in {"nterm", "nt", "start", "begin"}:
            return 1
        if s in {"cterm", "c_term", "end", "last"}:
            return length
        if s == "-1":
            return length
        if s.isdigit():
            return resolve_pos(int(s), length=length)
        anchor_match = re.fullmatch(
            r"(nterm|nt|start|begin|cterm|c_term|end|last)([+-]\d+)?",
            s,
        )
        if anchor_match is not None:
            anchor = str(anchor_match.group(1))
            raw_delta = anchor_match.group(2)
            base = 1 if anchor in {"nterm", "nt", "start", "begin"} else length
            delta = int(raw_delta) if raw_delta is not None else 0
            return resolve_pos(base + delta, length=length)
    raise ValueError(f"Invalid residue position spec: {pos!r}")


@dataclass(frozen=True)
class ResolvedCyclizationAtomRef:
    proto: str
    pos: int  # 1-indexed
    atom: str
    aa: str | None = None


@dataclass(frozen=True)
class ResolvedCyclizationBond:
    a: ResolvedCyclizationAtomRef
    b: ResolvedCyclizationAtomRef
    offset_window: int | None = None
    signal_mode: str = "both"


@dataclass(frozen=True)
class ResolvedCyclizationBondSpec:
    a: ResolvedCyclizationAtomRef
    b: ResolvedCyclizationAtomRef
    order: str
    target_distance: float
    is_head_to_tail: bool
    signal_mode: str = "both"


@dataclass(frozen=True)
class ExpandedCyclizationBond:
    atom1: tuple[str, int, str]
    atom2: tuple[str, int, str]
    order: str
    target_distance: float
    is_head_to_tail: bool
    signal_mode: str = "both"


def resolve_cyclization_bonds(
    spec: CyclizationSpec, *, protomer_lengths: dict[str, int]
) -> tuple[ResolvedCyclizationBond, ...]:
    out: list[ResolvedCyclizationBond] = []

    def _resolve_one(ref: CyclizationAtomRef) -> ResolvedCyclizationAtomRef:
        proto = str(ref.proto or "").strip()
        if not proto:
            raise ValueError("cyclization bond endpoint missing proto")
        if proto not in protomer_lengths:
            raise ValueError(f"cyclization references unknown protomer {proto!r}")
        atom = str(ref.atom or "").strip()
        if not atom:
            raise ValueError(f"cyclization endpoint for {proto!r} missing atom")
        pos1 = resolve_pos(ref.pos, length=int(protomer_lengths[proto]))
        aa = ref.aa
        if aa is not None:
            aa = str(aa).strip()
            if len(aa) != 1 or not aa.isalpha():
                raise ValueError(f"cyclization endpoint aa must be 1-letter residue, got {ref.aa!r}")
            aa = aa.upper()
        return ResolvedCyclizationAtomRef(proto=proto, pos=pos1, atom=atom, aa=aa)

    for bond in spec.bonds:
        out.append(
            ResolvedCyclizationBond(
                a=_resolve_one(bond.a),
                b=_resolve_one(bond.b),
                offset_window=(
                    bond.offset_window
                    if bond.offset_window is not None
                    else spec.offset_window
                ),
                signal_mode=str(bond.signal_mode or spec.signal_mode),
            )
        )

    return tuple(out)


def _normalize_bond_order(order: str | None) -> str:
    if order is None:
        return "single"
    norm = str(order).strip().lower()
    alias = {
        "s": "single",
        "single": "single",
        "d": "double",
        "double": "double",
        "t": "triple",
        "triple": "triple",
        "a": "aromatic",
        "aromatic": "aromatic",
    }
    out = alias.get(norm, norm)
    if out not in VALID_BOND_ORDERS:
        raise ValueError(
            f"Invalid cyclization bond order {order!r} (expected single|double|triple|aromatic)"
        )
    return out


def _is_head_to_tail_resolved(
    bond: ResolvedCyclizationBond | ResolvedCyclizationBondSpec,
    *,
    protomer_lengths: dict[str, int],
) -> bool:
    a = bond.a
    b = bond.b
    if a.proto != b.proto:
        return False
    length = int(protomer_lengths.get(a.proto, 0))
    if length <= 0:
        return False

    ends = {int(a.pos), int(b.pos)}
    atoms = {str(a.atom).upper(), str(b.atom).upper()}
    return ends == {1, length} and atoms == {"N", "C"}


def _is_disulfide_resolved(
    bond: ResolvedCyclizationBond | ResolvedCyclizationBondSpec,
) -> bool:
    atoms = {str(bond.a.atom).upper(), str(bond.b.atom).upper()}
    return atoms == {"SG"}


def resolve_cyclization_bond_specs(
    spec: CyclizationSpec,
    *,
    protomer_lengths: dict[str, int],
    require_order_for_generic: bool = False,
) -> tuple[ResolvedCyclizationBondSpec, ...]:
    resolved = resolve_cyclization_bonds(spec, protomer_lengths=protomer_lengths)
    source_bonds: list[CyclizationBond] = list(spec.bonds)

    if len(resolved) != len(source_bonds):
        raise RuntimeError(
            "Internal error: resolved cyclization bond count does not match source specification"
        )

    out: list[ResolvedCyclizationBondSpec] = []
    for rbond, sbond in zip(resolved, source_bonds):
        is_head_to_tail = _is_head_to_tail_resolved(
            rbond, protomer_lengths=protomer_lengths
        )
        is_disulfide = _is_disulfide_resolved(rbond)
        order = sbond.order
        distance = sbond.target_distance

        if is_head_to_tail:
            order_norm = _normalize_bond_order(order)
            target = float(distance if distance is not None else HEAD_TO_TAIL_TARGET_DISTANCE)
        elif is_disulfide:
            order_norm = _normalize_bond_order(order)
            target = float(distance if distance is not None else DISULFIDE_TARGET_DISTANCE)
        else:
            if order is None and require_order_for_generic:
                raise ValueError(
                    "Generic cyclization bonds must define `order` explicitly "
                    "(single|double|triple|aromatic)."
                )
            order_norm = _normalize_bond_order(order)
            target = float(
                distance
                if distance is not None
                else DEFAULT_GENERIC_TARGET_DISTANCE_BY_ORDER[order_norm]
            )

        if target <= 0:
            raise ValueError(
                f"Cyclization target_distance must be > 0, got {target} for bond {rbond}"
            )

        out.append(
            ResolvedCyclizationBondSpec(
                a=rbond.a,
                b=rbond.b,
                order=order_norm,
                target_distance=target,
                is_head_to_tail=is_head_to_tail,
                signal_mode=str(rbond.signal_mode),
            )
        )
    return tuple(out)


def expand_cyclization_bonds_for_oligo(
    resolved_specs: Sequence[ResolvedCyclizationBondSpec],
    *,
    oligo_name: str,
) -> tuple[ExpandedCyclizationBond, ...]:
    if not resolved_specs:
        return ()

    from astrolabe.utils.mmcif_ids import int_id_to_str_id

    out: list[ExpandedCyclizationBond] = []
    oligo = str(oligo_name)

    def _occurrences(proto: str) -> list[int]:
        return [i for i, c in enumerate(oligo) if c == proto]

    for bond in resolved_specs:
        a_idxs = _occurrences(bond.a.proto)
        b_idxs = _occurrences(bond.b.proto)
        if not a_idxs or not b_idxs:
            continue

        if bond.a.proto == bond.b.proto:
            pairs = [(i, i) for i in a_idxs]
        else:
            if len(a_idxs) != len(b_idxs):
                raise ValueError(
                    "cyclization bond requires equal copy counts when protomers repeat "
                    f"(a.proto={bond.a.proto!r} copies={len(a_idxs)}, "
                    f"b.proto={bond.b.proto!r} copies={len(b_idxs)}) for oligo {oligo!r}"
                )
            pairs = list(zip(a_idxs, b_idxs))

        for a_i, b_i in pairs:
            atom1 = (int_id_to_str_id(a_i + 1), int(bond.a.pos), str(bond.a.atom))
            atom2 = (int_id_to_str_id(b_i + 1), int(bond.b.pos), str(bond.b.atom))
            out.append(
                ExpandedCyclizationBond(
                    atom1=atom1,
                    atom2=atom2,
                    order=bond.order,
                    target_distance=float(bond.target_distance),
                    is_head_to_tail=bool(bond.is_head_to_tail),
                    signal_mode=str(bond.signal_mode),
                )
            )
    return tuple(out)


def filter_expanded_bonds_for_encoding(
    expanded: Sequence[ExpandedCyclizationBond],
    *,
    encoding: CyclicEncodingSpec,
) -> tuple[ExpandedCyclizationBond, ...]:
    if not encoding.enabled:
        return ()
    if encoding.scope == "head_to_tail_only":
        return tuple(b for b in expanded if b.is_head_to_tail)
    return tuple(expanded)


def filter_expanded_bonds_for_signal_modes(
    expanded: Sequence[ExpandedCyclizationBond],
) -> tuple[ExpandedCyclizationBond, ...]:
    out: list[ExpandedCyclizationBond] = []
    for bond in expanded:
        if _signal_mode_emits_bond(bond.signal_mode):
            out.append(bond)
    return tuple(out)


def derive_locked_positions(
    resolved: Iterable[ResolvedCyclizationBond | ResolvedCyclizationBondSpec],
) -> dict[str, set[int]]:
    locked: dict[str, set[int]] = {}
    for bond in resolved:
        for ref in (bond.a, bond.b):
            locked.setdefault(ref.proto, set()).add(int(ref.pos) - 1)
    return locked


def derive_aa_constraints(
    resolved: Iterable[ResolvedCyclizationBond | ResolvedCyclizationBondSpec],
) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for bond in resolved:
        for ref in (bond.a, bond.b):
            if ref.aa is None:
                continue
            per_proto = out.setdefault(ref.proto, {})
            pos1 = int(ref.pos)
            aa = str(ref.aa)
            prev = per_proto.get(pos1)
            if prev is not None and prev != aa:
                raise ValueError(
                    f"Conflicting cyclization aa constraints for {ref.proto}:{pos1}: {prev!r} vs {aa!r}"
                )
            per_proto[pos1] = aa
    return out


def is_canonical_head_to_tail_bond(
    bond: ResolvedCyclizationBond, *, protomer_lengths: dict[str, int]
) -> bool:
    if bond.a.proto != bond.b.proto:
        return False
    proto = str(bond.a.proto)
    length = int(protomer_lengths.get(proto, 0))
    if length <= 0:
        return False

    refs = {
        (int(bond.a.pos), str(bond.a.atom).strip().upper()),
        (int(bond.b.pos), str(bond.b.atom).strip().upper()),
    }
    return refs == {(1, "N"), (length, "C")}


def derive_cyclic_chain_ids(
    resolved: Iterable[ResolvedCyclizationBond],
    *,
    oligo_name: str,
    protomer_lengths: dict[str, int],
) -> tuple[str, ...]:
    chain_ids: list[str] = []
    seen: set[str] = set()

    for bond in resolved:
        if not _signal_mode_emits_index(getattr(bond, "signal_mode", "both")):
            continue
        if not is_canonical_head_to_tail_bond(
            bond, protomer_lengths=protomer_lengths
        ):
            continue
        proto = str(bond.a.proto)
        for idx0, proto_name in enumerate(str(oligo_name)):
            if proto_name != proto:
                continue
            chain_id = int_id_to_str_id(idx0 + 1)
            if chain_id in seen:
                continue
            seen.add(chain_id)
            chain_ids.append(chain_id)

    return tuple(chain_ids)


def _endpoint_position_groups(
    pos_a: int, pos_b: int, *, window: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lo = min(int(pos_a), int(pos_b))
    hi = max(int(pos_a), int(pos_b))
    left = tuple(range(lo, min(hi, lo + window - 1) + 1))
    right = tuple(range(max(lo, hi - window + 1), hi + 1))
    return left, right


def compute_cyclic_local_offset(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    idx = np.arange(length, dtype=np.int32)
    offset = idx[:, None] - idx[None, :]
    half = length // 2
    offset = np.where(offset > half, offset - length, offset)
    offset = np.where(offset < -half, offset + length, offset)
    return offset.astype(np.int32, copy=False)


def _build_patch_offsets(
    pos_a: int, pos_b: int, *, window: int
) -> tuple[CyclicRelativeOffset, ...]:
    lo = min(int(pos_a), int(pos_b))
    hi = max(int(pos_a), int(pos_b))
    length = hi - lo + 1
    local_offset = compute_cyclic_local_offset(length)
    left_group, right_group = _endpoint_position_groups(lo, hi, window=window)
    offsets: dict[tuple[int, int], int] = {}
    for i_pos in left_group:
        for j_pos in right_group:
            local_i = i_pos - lo
            local_j = j_pos - lo
            offsets[(i_pos, j_pos)] = int(local_offset[local_i, local_j])
            offsets[(j_pos, i_pos)] = int(local_offset[local_j, local_i])
    return tuple(
        CyclicRelativeOffset(residue_i=i_pos, residue_j=j_pos, offset=offset)
        for (i_pos, j_pos), offset in sorted(offsets.items())
    )


def _build_interval_patch_offsets(
    pos_a: int, pos_b: int
) -> tuple[CyclicRelativeOffset, ...]:
    lo = min(int(pos_a), int(pos_b))
    hi = max(int(pos_a), int(pos_b))
    length = hi - lo + 1
    local_offset = compute_cyclic_local_offset(length)
    offsets: list[CyclicRelativeOffset] = []
    for i_pos in range(lo, hi + 1):
        for j_pos in range(lo, hi + 1):
            offsets.append(
                CyclicRelativeOffset(
                    residue_i=i_pos,
                    residue_j=j_pos,
                    offset=int(local_offset[i_pos - lo, j_pos - lo]),
                )
            )
    return tuple(offsets)


def compute_highfold_cycpoem_offset(
    length: int,
    bonds: Iterable[tuple[int, int]] = (),
    *,
    include_head_to_tail: bool = True,
) -> np.ndarray:
    """Build a HighFold-style CycPOEM shortest-path offset matrix.

    Residue positions in ``bonds`` are 1-indexed. HighFold constructs an
    undirected graph with linear peptide edges, an optional N-C edge, and
    disulfide/cyclization edges, then uses shortest-path distances as relative
    position magnitudes. Its released implementation assigns negative signs to
    the upper triangle before AlphaFold relative-position one-hot embedding.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    matrix = np.zeros((length, length), dtype=np.int32) + int(length)
    np.fill_diagonal(matrix, 0)

    for idx in range(length - 1):
        matrix[idx, idx + 1] = 1
        matrix[idx + 1, idx] = 1

    if include_head_to_tail and length > 1:
        matrix[0, length - 1] = 1
        matrix[length - 1, 0] = 1

    for pos_a, pos_b in bonds:
        i = int(pos_a) - 1
        j = int(pos_b) - 1
        if not (0 <= i < length):
            raise ValueError(f"CycPOEM bond residue_i out of range: {pos_a}")
        if not (0 <= j < length):
            raise ValueError(f"CycPOEM bond residue_j out of range: {pos_b}")
        if i == j:
            continue
        matrix[i, j] = 1
        matrix[j, i] = 1

    for k in range(length):
        via_k = matrix[:, k, None] + matrix[k, None, :]
        matrix = np.minimum(matrix, via_k).astype(np.int32, copy=False)

    upper = np.triu_indices(length, k=1)
    matrix[upper] *= -1
    return matrix.astype(np.int32, copy=False)


def _build_full_matrix_patch_offsets(matrix: np.ndarray) -> tuple[CyclicRelativeOffset, ...]:
    matrix = np.asarray(matrix, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Cyclic offset matrix must be square, got {matrix.shape}")
    offsets: list[CyclicRelativeOffset] = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            offsets.append(
                CyclicRelativeOffset(
                    residue_i=i + 1,
                    residue_j=j + 1,
                    offset=int(matrix[i, j]),
                )
            )
    return tuple(offsets)


def derive_highfold_cycpoem_relative_patches(
    resolved: Iterable[ResolvedCyclizationBond],
    *,
    oligo_name: str,
    protomer_lengths: dict[str, int],
    include_head_to_tail: bool = True,
) -> tuple[CyclicRelativePatch, ...]:
    bonds_by_proto: dict[str, list[tuple[int, int]]] = {}
    for bond in resolved:
        if not _signal_mode_emits_index(getattr(bond, "signal_mode", "both")):
            continue
        if bond.a.proto != bond.b.proto:
            continue
        proto = str(bond.a.proto)
        if proto not in protomer_lengths:
            raise ValueError(f"cyclization references unknown protomer {proto!r}")
        bonds_by_proto.setdefault(proto, []).append((int(bond.a.pos), int(bond.b.pos)))

    patches: list[CyclicRelativePatch] = []

    def _occurrences(proto: str) -> list[int]:
        return [i for i, c in enumerate(str(oligo_name)) if c == proto]

    for proto, bonds in sorted(bonds_by_proto.items()):
        length = int(protomer_lengths[proto])
        matrix = compute_highfold_cycpoem_offset(
            length,
            bonds,
            include_head_to_tail=include_head_to_tail,
        )
        offsets = _build_full_matrix_patch_offsets(matrix)
        for idx0 in _occurrences(proto):
            patches.append(
                CyclicRelativePatch(
                    chain_id=int_id_to_str_id(idx0 + 1),
                    offsets=offsets,
                )
            )
    return tuple(patches)


def derive_cyclic_relative_patches(
    resolved: Iterable[ResolvedCyclizationBond],
    *,
    oligo_name: str,
    protomer_lengths: dict[str, int],
    default_offset_window: int = 3,
) -> tuple[CyclicRelativePatch, ...]:
    if default_offset_window <= 0:
        raise ValueError(
            f"default_offset_window must be positive, got {default_offset_window}"
        )

    patches: list[CyclicRelativePatch] = []
    emitted_offsets_by_chain: dict[str, dict[tuple[int, int], int]] = {}

    def _occurrences(proto: str) -> list[int]:
        return [i for i, c in enumerate(str(oligo_name)) if c == proto]

    for bond_idx, bond in enumerate(resolved, start=1):
        if not _signal_mode_emits_index(getattr(bond, "signal_mode", "both")):
            continue
        if bond.a.proto != bond.b.proto:
            continue
        if is_canonical_head_to_tail_bond(bond, protomer_lengths=protomer_lengths):
            if bond.offset_window is not None:
                raise ValueError(
                    "canonical head-to-tail bond cannot set offset_window"
                )
            continue

        window = int(
            bond.offset_window
            if bond.offset_window is not None
            else default_offset_window
        )
        if window <= 0:
            raise ValueError(f"offset_window must be positive, got {window}")

        proto = str(bond.a.proto)
        offsets = _build_patch_offsets(int(bond.a.pos), int(bond.b.pos), window=window)

        for idx0 in _occurrences(proto):
            chain_id = int_id_to_str_id(idx0 + 1)
            emitted = emitted_offsets_by_chain.setdefault(chain_id, {})
            unique_offsets: list[CyclicRelativeOffset] = []
            for entry in offsets:
                key = (int(entry.residue_i), int(entry.residue_j))
                value = int(entry.offset)
                previous = emitted.get(key)
                if previous is None:
                    emitted[key] = value
                    unique_offsets.append(entry)
                    continue
                if int(previous) != value:
                    raise ValueError(
                        "cyclic relative matrix cell conflict on chain "
                        f"{chain_id}: residue_i={key[0]} residue_j={key[1]} "
                        f"previous_offset={previous} new_offset={value} "
                        f"from bond_index={bond_idx}"
                    )
            if unique_offsets:
                patches.append(
                    CyclicRelativePatch(
                        chain_id=chain_id,
                        offsets=tuple(unique_offsets),
                    )
                )

    return tuple(patches)


def derive_global_cyclic_relative_patches(
    resolved: Iterable[ResolvedCyclizationBond],
    *,
    oligo_name: str,
    protomer_lengths: dict[str, int],
) -> tuple[CyclicRelativePatch, ...]:
    patches: list[CyclicRelativePatch] = []
    emitted_offsets_by_chain: dict[str, dict[tuple[int, int], int]] = {}

    def _occurrences(proto: str) -> list[int]:
        return [i for i, c in enumerate(str(oligo_name)) if c == proto]

    for bond_idx, bond in enumerate(resolved, start=1):
        if not _signal_mode_emits_index(getattr(bond, "signal_mode", "both")):
            continue
        if bond.a.proto != bond.b.proto:
            continue
        if is_canonical_head_to_tail_bond(bond, protomer_lengths=protomer_lengths):
            if bond.offset_window is not None:
                raise ValueError(
                    "canonical head-to-tail bond cannot set offset_window"
                )
            continue

        proto = str(bond.a.proto)
        if proto not in protomer_lengths:
            raise ValueError(f"cyclization references unknown protomer {proto!r}")
        offsets = _build_interval_patch_offsets(int(bond.a.pos), int(bond.b.pos))

        for idx0 in _occurrences(proto):
            chain_id = int_id_to_str_id(idx0 + 1)
            emitted = emitted_offsets_by_chain.setdefault(chain_id, {})
            unique_offsets: list[CyclicRelativeOffset] = []
            for entry in offsets:
                key = (int(entry.residue_i), int(entry.residue_j))
                value = int(entry.offset)
                previous = emitted.get(key)
                if previous is None:
                    emitted[key] = value
                    unique_offsets.append(entry)
                    continue
                if int(previous) != value:
                    raise ValueError(
                        "cyclic relative matrix cell conflict on chain "
                        f"{chain_id}: residue_i={key[0]} residue_j={key[1]} "
                        f"previous_offset={previous} new_offset={value} "
                        f"from bond_index={bond_idx}"
                    )
            if unique_offsets:
                patches.append(
                    CyclicRelativePatch(
                        chain_id=chain_id,
                        offsets=tuple(unique_offsets),
                    )
                )

    return tuple(patches)


def build_cyclic_offset_matrix(
    residue_index: np.ndarray,
    *,
    chain_lengths: Iterable[int],
    cyclic_chain_ids: Iterable[str],
    chain_ids: Iterable[str] | None = None,
    cyclic_relative_patches: Iterable[CyclicRelativePatch] | None = None,
) -> np.ndarray:
    residue_index = np.asarray(residue_index, dtype=np.int32).reshape(-1)
    offset = residue_index[:, None] - residue_index[None, :]

    chain_lengths_list = [int(length) for length in chain_lengths]
    if chain_ids is None:
        chain_ids_list = [
            int_id_to_str_id(i + 1) for i in range(len(chain_lengths_list))
        ]
    else:
        chain_ids_list = [str(chain_id) for chain_id in chain_ids]

    cyclic_chain_id_set = {str(chain_id) for chain_id in cyclic_chain_ids}
    chain_ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for chain_id, chain_length in zip(chain_ids_list, chain_lengths_list, strict=True):
        stop = start + chain_length
        chain_ranges[str(chain_id)] = (start, stop)
        if chain_id in cyclic_chain_id_set and chain_length > 2:
            offset[start:stop, start:stop] = compute_cyclic_local_offset(chain_length)
        start = stop

    emitted_patch_offsets: dict[str, dict[tuple[int, int], int]] = {}
    for patch in cyclic_relative_patches or ():
        chain_id = str(patch.chain_id)
        if chain_id not in chain_ranges:
            raise ValueError(f"Unknown chain_id in cyclic_relative_patches: {chain_id!r}")
        chain_start, chain_stop = chain_ranges[chain_id]
        for entry in patch.offsets:
            i_idx = chain_start + int(entry.residue_i) - 1
            j_idx = chain_start + int(entry.residue_j) - 1
            if not (chain_start <= i_idx < chain_stop):
                raise ValueError(
                    f"cyclic patch residue_i out of range for chain {chain_id}: {entry.residue_i}"
                )
            if not (chain_start <= j_idx < chain_stop):
                raise ValueError(
                    f"cyclic patch residue_j out of range for chain {chain_id}: {entry.residue_j}"
                )
            key = (int(entry.residue_i), int(entry.residue_j))
            value = int(entry.offset)
            emitted = emitted_patch_offsets.setdefault(chain_id, {})
            previous = emitted.get(key)
            if previous is not None and int(previous) != value:
                raise ValueError(
                    "cyclic relative matrix cell conflict on chain "
                    f"{chain_id}: residue_i={key[0]} residue_j={key[1]} "
                    f"previous_offset={previous} new_offset={value}"
                )
            emitted[key] = value
            offset[i_idx, j_idx] = value

    return offset.astype(np.int32, copy=False)
