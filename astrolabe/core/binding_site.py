from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChainPositions:
    chain: str
    positions: tuple[int, ...]  # 1-indexed residue positions


@dataclass(frozen=True)
class BindingSiteSpec:
    binder: ChainPositions
    target: ChainPositions
    mode: str = "both"  # pocket/contact/both
    max_distance: float = 6.0  # Angstrom
    min_distance: float = 0.0  # Angstrom (used for contact constraints)


@dataclass(frozen=True)
class PocketConstraintSpec:
    binder_chain: str
    contact_chain: str
    contact_positions: tuple[int, ...]


def _parse_positions(expr: str) -> tuple[int, ...]:
    expr = str(expr).strip()
    if not expr:
        return ()
    out: list[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid range '{part}' (end < start)")
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    for pos in out:
        if pos <= 0:
            raise ValueError(f"Residue positions must be 1-indexed (>0), got {pos}")
    # de-dup, stable
    seen = set()
    uniq: list[int] = []
    for pos in out:
        if pos not in seen:
            seen.add(pos)
            uniq.append(pos)
    return tuple(uniq)


def parse_chain_positions(spec: str) -> ChainPositions:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("Empty chain/position spec")
    lowered = raw.lower()
    if lowered.startswith("chain:") or lowered.startswith("protomer:"):
        # Support explicit selector prefixes while keeping the historic
        # `CHAIN:pos` syntax.
        pieces = raw.split(":")
        if len(pieces) >= 3:
            chain = f"{pieces[0]}:{pieces[1]}".strip()
            positions = _parse_positions(":".join(pieces[2:]))
        else:
            chain = raw.strip()
            positions = ()
    elif ":" in raw:
        chain, pos_expr = raw.split(":", 1)
        chain = chain.strip()
        positions = _parse_positions(pos_expr)
    else:
        chain = raw.strip()
        positions = ()
    if not chain:
        raise ValueError(f"Invalid chain spec '{spec}'")
    return ChainPositions(chain=chain, positions=positions)


def has_explicit_positions(chain_positions: ChainPositions) -> bool:
    return bool(chain_positions.positions)


def has_contact_pairs(spec: BindingSiteSpec) -> bool:
    return has_explicit_positions(spec.binder) and has_explicit_positions(spec.target)


def resolve_pocket_constraint(spec: BindingSiteSpec) -> PocketConstraintSpec:
    binder_has = has_explicit_positions(spec.binder)
    target_has = has_explicit_positions(spec.target)
    if binder_has and not target_has:
        return PocketConstraintSpec(
            binder_chain=spec.target.chain,
            contact_chain=spec.binder.chain,
            contact_positions=spec.binder.positions,
        )
    if target_has:
        return PocketConstraintSpec(
            binder_chain=spec.binder.chain,
            contact_chain=spec.target.chain,
            contact_positions=spec.target.positions,
        )
    raise ValueError(
        "binding_site requires residue position(s) on binder or target; both chain-only specs are invalid"
    )


def _parse_chain_positions_field(raw, *, field_name: str) -> ChainPositions:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        parsed = [parse_chain_positions(str(x)) for x in raw]
        chains = {p.chain for p in parsed}
        if len(chains) != 1:
            raise ValueError(
                f"binding_site.{field_name} must use one chain, got chains={sorted(chains)}"
            )
        positions: list[int] = []
        for part in parsed:
            positions.extend(list(part.positions))
        return ChainPositions(chain=parsed[0].chain, positions=tuple(dict.fromkeys(positions)))
    return parse_chain_positions(str(raw))


def _normalize_binding_site_cfg(binding_site_cfg):
    if binding_site_cfg is None:
        return None

    cfg = binding_site_cfg
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        cfg = {k: cfg[k] for k in cfg.keys()}
    return cfg


def _is_direct_binding_site_cfg(cfg) -> bool:
    return isinstance(cfg, Mapping) and ("binder" in cfg or "target" in cfg)


def has_binding_site_config(binding_site_cfg) -> bool:
    cfg = _normalize_binding_site_cfg(binding_site_cfg)
    if cfg is None:
        return False
    if _is_direct_binding_site_cfg(cfg):
        return True
    if not isinstance(cfg, Mapping):
        return False
    return any(_is_direct_binding_site_cfg(value) for value in cfg.values())


def parse_binding_site(binding_site_cfg, *, oligo_name: str | None = None) -> BindingSiteSpec | None:
    """Parse `binding_site` config (OmegaConf dict or plain dict).

    Supported:
      binding_site:
        binder: "A" or "A:10,12-14" or ["A:26-32", "A:52-56", ...]
        target: "B" or "B:258" or ["B:258", "B:260"]
        mode: "pocket"/"contact"/"both"  # optional; inferred when omitted
    """
    cfg = _normalize_binding_site_cfg(binding_site_cfg)
    if cfg is None:
        return None

    if not _is_direct_binding_site_cfg(cfg):
        if oligo_name is None:
            raise ValueError(
                "binding_site per-oligomer mapping requires `oligo_name` to select a sub-config"
            )
        subcfg = cfg.get(str(oligo_name))
        if subcfg is None:
            return None
        return parse_binding_site(subcfg)

    binder_raw = cfg.get("binder")
    target_raw = cfg.get("target")
    if not binder_raw or not target_raw:
        raise ValueError("binding_site requires both 'binder' and 'target'")

    max_distance = cfg.get("max_distance", cfg.get("max_distance_angstrom", 6.0))
    min_distance = cfg.get("min_distance", cfg.get("min_distance_angstrom", 0.0))
    try:
        max_distance_f = float(max_distance)
        min_distance_f = float(min_distance)
    except (TypeError, ValueError) as exc:
        raise ValueError("binding_site max_distance/min_distance must be numeric") from exc
    if max_distance_f <= 0:
        raise ValueError(f"binding_site.max_distance must be > 0, got {max_distance_f}")
    if min_distance_f < 0:
        raise ValueError(f"binding_site.min_distance must be >= 0, got {min_distance_f}")
    if min_distance_f > max_distance_f:
        raise ValueError(
            f"binding_site.min_distance must be <= max_distance, got {min_distance_f} > {max_distance_f}"
        )

    binder = _parse_chain_positions_field(binder_raw, field_name="binder")
    target = _parse_chain_positions_field(target_raw, field_name="target")

    binder_has = has_explicit_positions(binder)
    target_has = has_explicit_positions(target)
    if not binder_has and not target_has:
        raise ValueError(
            "binding_site requires residue position(s) on binder or target; both chain-only specs are invalid"
        )

    raw_mode = cfg.get("mode")
    if raw_mode is None or str(raw_mode).strip() == "":
        mode = "both" if (binder_has and target_has) else "pocket"
    else:
        mode = str(raw_mode).strip().lower()
        if mode not in {"both", "pocket", "contact"}:
            raise ValueError(f"binding_site.mode must be pocket/contact/both, got '{mode}'")
        if mode == "contact" and not (binder_has and target_has):
            raise ValueError(
                "binding_site.mode='contact' requires explicit residue position(s) on both binder and target"
            )

    return BindingSiteSpec(
        binder=binder,
        target=target,
        mode=mode,
        max_distance=max_distance_f,
        min_distance=min_distance_f,
    )
