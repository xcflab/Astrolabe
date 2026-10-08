from __future__ import annotations

import functools
import math
import pathlib
from typing import Iterable

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.binding_site import BindingSiteSpec, ChainPositions, parse_binding_site
from astrolabe.core.chain_resolution import chain_ids_for_oligo, resolve_selector_chain_ids
from astrolabe.losses.base import loss_registry


class BindingSiteLossError(RuntimeError):
    """Raised when binding-site loss inputs are invalid or unavailable."""


def _as_path(path_like) -> pathlib.Path:
    if isinstance(path_like, pathlib.Path):
        return path_like
    return pathlib.Path(str(path_like))


@functools.lru_cache(maxsize=8)
def _load_biopython_structure_cached(path_str: str, mtime_ns: int, size: int):
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    path = pathlib.Path(path_str)
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("complex", str(path))


def _load_biopython_structure(path: pathlib.Path):
    try:
        st = path.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size = int(st.st_size)
    except OSError:
        mtime_ns = -1
        size = -1
    return _load_biopython_structure_cached(str(path), mtime_ns, size)


def _iter_residues_for_positions(chain, positions: Iterable[int]):
    want = set(int(p) for p in positions)
    for residue in chain.get_residues():
        hetflag, resseq, _icode = residue.id
        if hetflag.strip():
            continue
        if int(resseq) in want:
            yield residue


def _iter_polymer_resseq(chain) -> set[int]:
    resseqs: set[int] = set()
    for residue in chain.get_residues():
        hetflag, resseq, _icode = residue.id
        if hetflag.strip():
            continue
        try:
            resseqs.add(int(resseq))
        except (TypeError, ValueError):
            continue
    return resseqs


def _atom_coords(residue, *, atom_mode: str) -> np.ndarray:
    atom_mode_l = atom_mode.lower().strip()
    if atom_mode_l in {"any"}:
        atom_mode_l = "all"
    if atom_mode_l not in {"all", "ca", "c-alpha", "c_alpha", "cb", "c-beta", "c_beta"}:
        raise ValueError(f"Unknown atom_mode={atom_mode!r} (expected: all|ca|cb)")

    coords = []
    for atom in residue.get_atoms():
        name = atom.get_name()
        if atom_mode_l in {"ca", "c-alpha", "c_alpha"} and name != "CA":
            continue
        if atom_mode_l in {"cb", "c-beta", "c_beta"} and name != "CB":
            continue
        c = atom.get_coord()
        if c is None:
            continue
        c = np.asarray(c, dtype=float)
        if not np.isfinite(c).all():
            continue
        coords.append(c)
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.stack(coords, axis=0)


def _min_distance_between_coords(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        raise BindingSiteLossError("No valid atom coordinates available for distance calculation.")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return float(np.sqrt(d2.min()))


def _min_distance_between_residue_sets(
    structure_path: pathlib.Path,
    left: ChainPositions,
    right: ChainPositions,
    *,
    atom_mode: str,
) -> float:
    structure = _load_biopython_structure(structure_path)
    model = next(structure.get_models())
    try:
        left_chain = model[left.chain]
        right_chain = model[right.chain]
    except KeyError as exc:
        raise BindingSiteLossError(
            f"Chain not found in structure {structure_path}: left={left.chain!r}, right={right.chain!r}"
        ) from exc

    left_coords = []
    for residue in _iter_residues_for_positions(left_chain, left.positions):
        coords = _atom_coords(residue, atom_mode=atom_mode)
        if coords.size:
            left_coords.append(coords)
    if not left_coords:
        raise BindingSiteLossError(
            f"No valid atoms found for chain {left.chain!r} at positions {tuple(left.positions)!r} in {structure_path}"
        )

    right_coords = []
    for residue in _iter_residues_for_positions(right_chain, right.positions):
        coords = _atom_coords(residue, atom_mode=atom_mode)
        if coords.size:
            right_coords.append(coords)
    if not right_coords:
        raise BindingSiteLossError(
            f"No valid atoms found for chain {right.chain!r} at positions {tuple(right.positions)!r} in {structure_path}"
        )
    left_arr = np.concatenate(left_coords, axis=0) if len(left_coords) > 1 else left_coords[0]
    right_arr = (
        np.concatenate(right_coords, axis=0) if len(right_coords) > 1 else right_coords[0]
    )
    return _min_distance_between_coords(left_arr, right_arr)


def _get_binding_site_spec(*, oligo_name: str) -> BindingSiteSpec:
    try:
        raw = getattr(global_config, "binding_site", None)
    except RuntimeError as exc:
        raise BindingSiteLossError(
            "binding_site losses require `binding_site` config, but it is not configured."
        ) from exc
    if raw is None:
        raise BindingSiteLossError(
            "binding_site losses require `binding_site` config, but it is not configured."
        )
    spec = parse_binding_site(raw, oligo_name=oligo_name)
    if spec is None:
        raise BindingSiteLossError(
            f"Failed to parse binding_site config for oligo {oligo_name!r}."
        )
    return spec


def _get_dynamic_binding_site_config():
    try:
        return getattr(global_config, "dynamic_binding_site", None)
    except RuntimeError:
        return None


def _get_static_binding_site_config():
    try:
        return getattr(global_config, "binding_site", None)
    except RuntimeError:
        return None


def _resolve_chain_positions_for_oligo(
    *,
    oligo_name: str,
    selector: ChainPositions,
) -> list[ChainPositions]:
    resolved = resolve_selector_chain_ids(oligo_name, selector.chain)
    return [ChainPositions(chain=chain_id, positions=selector.positions) for chain_id in resolved]


def _require_resolved_chain_positions(
    *,
    oligo_name: str,
    selector: ChainPositions,
    role: str,
) -> list[ChainPositions]:
    resolved = _resolve_chain_positions_for_oligo(oligo_name=oligo_name, selector=selector)
    if not resolved:
        raise BindingSiteLossError(
            f"{role} selector {selector.chain!r} did not resolve to any chains for oligo {oligo_name!r}."
        )
    if len(resolved) > 1:
        raise BindingSiteLossError(
            f"{role} selector {selector.chain!r} matches multiple chains "
            f"{[item.chain for item in resolved]!r} for oligo {oligo_name!r}. "
            "Use explicit `chain:<ID>` selectors."
        )
    return resolved


def _atom_mode_from_kwargs(atom_mode: str | None, use_ca: bool | None) -> str:
    if use_ca is None:
        return atom_mode or "all"
    return "ca" if use_ca else (atom_mode or "all")


def _normalize_chain_selector_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _resolve_chain_selector_list(
    *,
    oligo_name: str,
    selectors: Iterable[str],
    role: str,
) -> list[str]:
    out: list[str] = []
    for selector in selectors:
        resolved = resolve_selector_chain_ids(oligo_name, selector)
        if not resolved:
            raise BindingSiteLossError(
                f"{role} selector {selector!r} did not resolve to any chains for oligo {oligo_name!r}."
            )
        for chain_id in resolved:
            if chain_id not in out:
                out.append(chain_id)
    return out


def _chain_heavy_atom_coordinates(
    structure_path: pathlib.Path,
    chain_ids: Iterable[str],
    *,
    atom_mode: str,
    context: str,
) -> np.ndarray:
    atom_mode_l = atom_mode.lower().strip()
    if atom_mode_l in {"any"}:
        atom_mode_l = "all"
    if atom_mode_l not in {"all", "ca", "c-alpha", "c_alpha", "cb", "c-beta", "c_beta"}:
        raise ValueError(f"Unknown atom_mode={atom_mode!r} (expected: all|ca|cb)")

    wanted = set(str(chain_id) for chain_id in chain_ids)
    structure = _load_biopython_structure(structure_path)
    model = next(structure.get_models())
    coords: list[np.ndarray] = []
    for chain in model:
        if str(chain.id) not in wanted:
            continue
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                name = atom.get_name()
                if atom_mode_l in {"ca", "c-alpha", "c_alpha"} and name != "CA":
                    continue
                if atom_mode_l in {"cb", "c-beta", "c_beta"} and name != "CB":
                    continue
                if atom_mode_l == "all":
                    element = str(getattr(atom, "element", "") or name[:1]).strip().upper()
                    if element == "H" or name.strip().upper().startswith("H"):
                        continue
                c = atom.get_coord()
                if c is None:
                    continue
                c = np.asarray(c, dtype=float)
                if np.isfinite(c).all():
                    coords.append(c)
    if not coords:
        raise BindingSiteLossError(f"No valid atoms found for {context} in {structure_path}")
    return np.stack(coords, axis=0)


def _expand_chain_positions_if_empty(
    structure_path: pathlib.Path,
    chain_positions: ChainPositions,
) -> ChainPositions | None:
    if chain_positions.positions:
        return chain_positions

    structure = _load_biopython_structure(structure_path)
    model = next(structure.get_models())
    try:
        chain = model[chain_positions.chain]
    except Exception:
        return None
    all_pos = sorted(_iter_polymer_resseq(chain))
    return ChainPositions(chain=chain_positions.chain, positions=tuple(all_pos))


def _expand_chain_positions_or_raise(
    structure_path: pathlib.Path,
    chain_positions: ChainPositions,
    *,
    role: str,
) -> ChainPositions:
    expanded = _expand_chain_positions_if_empty(structure_path, chain_positions)
    if expanded is None:
        raise BindingSiteLossError(
            f"{role} chain {chain_positions.chain!r} not found in {structure_path}"
        )
    return expanded


def _binding_site_allowed_min_dist_from_config(
    eval_result,
    *,
    binding_site_cfg,
    atom_mode: str,
    context_name: str,
    missing_value: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for oligo_name, path_like in eval_result.complex_paths.items():
        if binding_site_cfg is None:
            if missing_value is not None:
                out[oligo_name] = float(missing_value)
                continue
            raise BindingSiteLossError(
                f"{context_name} losses require `{context_name}` config, but it is not configured."
            )
        path = _as_path(path_like)
        try:
            spec = parse_binding_site(binding_site_cfg, oligo_name=oligo_name)
            if spec is None:
                if missing_value is not None:
                    out[oligo_name] = float(missing_value)
                    continue
                raise BindingSiteLossError(
                    f"Failed to parse {context_name} config for oligo {oligo_name!r}."
                )
            targets = _require_resolved_chain_positions(
                oligo_name=oligo_name,
                selector=spec.target,
                role="target",
            )
            binders = _require_resolved_chain_positions(
                oligo_name=oligo_name,
                selector=spec.binder,
                role="binder",
            )
            expanded_targets = [
                _expand_chain_positions_or_raise(path, target, role="target")
                for target in targets
            ]
            expanded_binders = [
                _expand_chain_positions_or_raise(path, binder, role="binder")
                for binder in binders
            ]
            distances = [
                _min_distance_between_residue_sets(path, target, binder, atom_mode=atom_mode)
                for target in expanded_targets
                for binder in expanded_binders
            ]
            if not distances:
                raise BindingSiteLossError(f"No valid {context_name} distances available.")
            out[oligo_name] = min(distances)
        except BindingSiteLossError as exc:
            raise BindingSiteLossError(f"Oligo {oligo_name}: {exc}") from exc
    return out


@loss_registry.register(
    description="Min atom-atom distance (Å) between binding_site-selected residues and allowed residues on the opposite chain; when one side omits positions, that side expands to the whole chain."
)
def binding_site_allowed_min_dist(eval_result, atom_mode: str | None = None, use_ca: bool | None = None):
    mode = _atom_mode_from_kwargs(atom_mode, use_ca)
    return _binding_site_allowed_min_dist_from_config(
        eval_result,
        binding_site_cfg=_get_static_binding_site_config(),
        atom_mode=mode,
        context_name="binding_site",
    )


@loss_registry.register(
    description=(
        "Min atom-atom distance (Å) using runtime dynamic_binding_site residues; "
        "returns missing_value before a predocker publishes dynamic contacts."
    )
)
def dynamic_binding_site_allowed_min_dist(
    eval_result,
    atom_mode: str | None = None,
    use_ca: bool | None = None,
    missing_value: float = math.inf,
):
    mode = _atom_mode_from_kwargs(atom_mode, use_ca)
    return _binding_site_allowed_min_dist_from_config(
        eval_result,
        binding_site_cfg=_get_dynamic_binding_site_config(),
        atom_mode=mode,
        context_name="dynamic_binding_site",
        missing_value=float(missing_value),
    )


@loss_registry.register(
    description=(
        "Min heavy-atom distance (Å) between binder chain(s) and target chain(s); "
        "target defaults to all non-binder chains in the evaluated oligo."
    ),
    supports_oligomer_specific=True,
)
def binder_target_min_dist(
    eval_result,
    binder_chain: str = "A",
    target_chains=None,
    atom_mode: str | None = None,
    use_ca: bool | None = None,
):
    mode = _atom_mode_from_kwargs(atom_mode, use_ca)
    out: dict[str, float] = {}
    for oligo_name, path_like in eval_result.complex_paths.items():
        binder_ids = _resolve_chain_selector_list(
            oligo_name=oligo_name,
            selectors=[binder_chain],
            role="binder",
        )
        target_selectors = _normalize_chain_selector_list(target_chains)
        if target_selectors:
            target_ids = _resolve_chain_selector_list(
                oligo_name=oligo_name,
                selectors=target_selectors,
                role="target",
            )
        else:
            target_ids = [
                chain_id for chain_id in chain_ids_for_oligo(oligo_name) if chain_id not in binder_ids
            ]
        if not target_ids:
            raise BindingSiteLossError(
                f"No target chains available for oligo {oligo_name!r}; binder chains={binder_ids!r}."
            )

        path = _as_path(path_like)
        binder_coords = _chain_heavy_atom_coordinates(
            path,
            binder_ids,
            atom_mode=mode,
            context=f"binder chains {binder_ids!r}",
        )
        target_coords = _chain_heavy_atom_coordinates(
            path,
            target_ids,
            atom_mode=mode,
            context=f"target chains {target_ids!r}",
        )
        out[oligo_name] = _min_distance_between_coords(binder_coords, target_coords)
    return out


@loss_registry.register(
    description="Min atom-atom distance (Å) between the target-side binding site (or full target chain) and binder framework residues."
)
def binding_site_framework_min_dist(eval_result, atom_mode: str | None = None, use_ca: bool | None = None):
    mode = _atom_mode_from_kwargs(atom_mode, use_ca)
    out: dict[str, float] = {}
    for oligo_name, path_like in eval_result.complex_paths.items():
        path = _as_path(path_like)
        try:
            spec = _get_binding_site_spec(oligo_name=oligo_name)
            if not spec.binder.positions:
                raise BindingSiteLossError(
                    "binding_site_framework_min_dist requires explicit binder hotspot positions."
                )
            targets = _require_resolved_chain_positions(
                oligo_name=oligo_name,
                selector=spec.target,
                role="target",
            )
            binders = _require_resolved_chain_positions(
                oligo_name=oligo_name,
                selector=spec.binder,
                role="binder",
            )
            expanded_targets = [
                _expand_chain_positions_or_raise(path, target, role="target")
                for target in targets
            ]
            structure = _load_biopython_structure(path)
            model = next(structure.get_models())
            allowed = set(int(p) for p in spec.binder.positions)
            framework_sets: list[ChainPositions] = []
            for binder in binders:
                try:
                    binder_chain = model[binder.chain]
                except KeyError as exc:
                    raise BindingSiteLossError(
                        f"binder chain {binder.chain!r} not found in {path}"
                    ) from exc
                all_positions = _iter_polymer_resseq(binder_chain)
                framework_positions = sorted(p for p in all_positions if p not in allowed)
                if not framework_positions:
                    raise BindingSiteLossError(
                        f"framework residue set is empty for binder chain {binder.chain!r}."
                    )
                framework_sets.append(
                    ChainPositions(chain=binder.chain, positions=tuple(framework_positions))
                )
            distances = [
                _min_distance_between_residue_sets(path, target, framework, atom_mode=mode)
                for target in expanded_targets
                for framework in framework_sets
            ]
            if not distances:
                raise BindingSiteLossError("No valid binding_site framework distances available.")
            out[oligo_name] = min(distances)
        except BindingSiteLossError as exc:
            raise BindingSiteLossError(f"Oligo {oligo_name}: {exc}") from exc
    return out
