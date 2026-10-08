from __future__ import annotations

import functools
import math
import pathlib

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.cyclization import (
    expand_cyclization_bonds_for_oligo,
    filter_expanded_bonds_for_encoding,
    parse_cyclization,
    resolve_cyclization_bond_specs,
)
from astrolabe.losses.base import loss_registry


def _as_path(path_like) -> pathlib.Path:
    if isinstance(path_like, pathlib.Path):
        return path_like
    return pathlib.Path(str(path_like))


@functools.lru_cache(maxsize=8)
def _load_atom_index_cached(path_str: str, mtime_ns: int, size: int):
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore

    path = pathlib.Path(path_str)
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("complex", str(path))
    model = next(structure.get_models())
    index: dict[tuple[str, int, str], np.ndarray] = {}
    for chain in model:
        chain_id = str(chain.id)
        for residue in chain:
            try:
                res_id = int(residue.id[1])
            except Exception:
                continue
            for atom in residue.get_atoms():
                atom_name = str(atom.get_name()).strip()
                coord = atom.get_coord()
                if coord is None:
                    continue
                arr = np.asarray(coord, dtype=float)
                if not np.isfinite(arr).all():
                    continue
                index[(chain_id, res_id, atom_name)] = arr
    return index


def _load_atom_index(path: pathlib.Path):
    try:
        st = path.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size = int(st.st_size)
    except Exception:
        mtime_ns = -1
        size = -1
    return _load_atom_index_cached(str(path), mtime_ns, size)


def _distance(index, atom1: tuple[str, int, str], atom2: tuple[str, int, str]) -> float | None:
    p1 = index.get((str(atom1[0]), int(atom1[1]), str(atom1[2])))
    p2 = index.get((str(atom2[0]), int(atom2[1]), str(atom2[2])))
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))


def _cyclization_context():
    try:
        cyc_cfg = getattr(global_config, "cyclization")
    except Exception:
        cyc_cfg = None
    cyc_spec = parse_cyclization(cyc_cfg)
    if cyc_spec is None:
        return None, []

    protomers = getattr(global_config, "protomers", {}) or {}
    protomer_lengths = {name: len(p.sequence) for name, p in protomers.items()}
    resolved_specs = resolve_cyclization_bond_specs(
        cyc_spec,
        protomer_lengths=protomer_lengths,
        require_order_for_generic=True,
    )
    return cyc_spec, resolved_specs


def _selected_distances(
    eval_result, oligo_name: str, *, all_configured: bool = False
) -> list[tuple[float, float | None]]:
    cyc_spec, resolved_specs = _cyclization_context()
    if cyc_spec is None:
        return []
    expanded = expand_cyclization_bonds_for_oligo(
        resolved_specs,
        oligo_name=str(oligo_name),
    )
    selected = expanded if all_configured else filter_expanded_bonds_for_encoding(
        expanded, encoding=cyc_spec.encoding
    )
    if not selected:
        return []

    path_like = (getattr(eval_result, "complex_paths", {}) or {}).get(str(oligo_name))
    if path_like is None:
        return [(float(bond.target_distance), None) for bond in selected]
    atom_index = _load_atom_index(_as_path(path_like))
    return [
        (
            float(bond.target_distance),
            _distance(atom_index, bond.atom1, bond.atom2),
        )
        for bond in selected
    ]


def _max_excess(distances: list[tuple[float, float | None]]) -> float:
    if not distances:
        return 0.0
    excesses: list[float] = []
    for target_distance, observed_distance in distances:
        if observed_distance is None:
            return math.inf
        excesses.append(float(observed_distance) - float(target_distance))
    return float(max(excesses)) if excesses else 0.0


@loss_registry.register(
    optional_params=["sigma", "aggregate"],
    description="Normalized cyclic closure quality in [0,1] using Gaussian distance penalty on configured cyclization bonds.",
)
def cyclic_closure_norm(
    eval_result,
    sigma: float = 0.20,
    aggregate: str = "min",
):
    if float(sigma) <= 0:
        raise ValueError(f"cyclic_closure_norm sigma must be > 0, got {sigma}")
    aggregate_norm = str(aggregate or "min").strip().lower()
    if aggregate_norm not in {"min", "mean"}:
        raise ValueError(
            f"cyclic_closure_norm aggregate must be 'min' or 'mean', got {aggregate!r}"
        )

    cyc_spec, resolved_specs = _cyclization_context()
    if cyc_spec is None:
        return {name: 1.0 for name in eval_result.names}

    out: dict[str, float] = {}
    for oligo_name in eval_result.names:
        expanded = expand_cyclization_bonds_for_oligo(
            resolved_specs,
            oligo_name=str(oligo_name),
        )
        selected = filter_expanded_bonds_for_encoding(
            expanded,
            encoding=cyc_spec.encoding,
        )
        if not selected:
            out[str(oligo_name)] = 1.0
            continue

        path_like = (getattr(eval_result, "complex_paths", {}) or {}).get(str(oligo_name))
        if path_like is None:
            out[str(oligo_name)] = 0.0
            continue
        atom_index = _load_atom_index(_as_path(path_like))

        q_values: list[float] = []
        for bond in selected:
            dist = _distance(atom_index, bond.atom1, bond.atom2)
            if dist is None:
                q_values.append(0.0)
                continue
            delta = float(dist) - float(bond.target_distance)
            q = math.exp(-(delta * delta) / (2.0 * float(sigma) * float(sigma)))
            q_values.append(float(max(0.0, min(1.0, q))))

        if not q_values:
            out[str(oligo_name)] = 1.0
        elif aggregate_norm == "min":
            out[str(oligo_name)] = float(min(q_values))
        else:
            out[str(oligo_name)] = float(sum(q_values) / len(q_values))
    return out


@loss_registry.register(
    description="Maximum direct distance excess over configured cyclization bonds in Angstrom; lower is better.",
)
def cyclic_closure_max_excess(eval_result):
    out: dict[str, float] = {}
    for oligo_name in eval_result.names:
        out[str(oligo_name)] = _max_excess(
            _selected_distances(eval_result, str(oligo_name))
        )
    return out


@loss_registry.register(
    optional_params=["margin"],
    description="Binary direct cyclic closure status; 1 when every configured bond has observed distance <= target_distance + margin.",
)
def cyclic_all_bonds_closed(eval_result, margin: float = 0.30):
    margin_value = float(margin)
    if margin_value < 0:
        raise ValueError(f"cyclic_all_bonds_closed margin must be >= 0, got {margin}")
    out: dict[str, float] = {}
    for oligo_name in eval_result.names:
        max_excess = _max_excess(_selected_distances(eval_result, str(oligo_name)))
        out[str(oligo_name)] = float(math.isfinite(max_excess) and max_excess <= margin_value)
    return out


@loss_registry.register(
    description="Maximum absolute bond-length error in Angstrom across every configured cyclization bond, independent of encoding scope; missing endpoint atoms give infinity.",
)
def cyclic_max_bond_error(eval_result):
    out: dict[str, float] = {}
    for oligo_name in eval_result.names:
        distances = _selected_distances(
            eval_result, str(oligo_name), all_configured=True
        )
        out[str(oligo_name)] = max(
            (
                math.inf if observed is None else abs(observed - target)
                for target, observed in distances
            ),
            default=0.0,
        )
    return out


@loss_registry.register(
    optional_params=["margin"],
    description="Publication closure endpoint: 1 when all configured bonds have endpoint atoms and absolute bond-length error <= margin, independent of encoding scope.",
)
def cyclic_all_bonds_within_error(eval_result, margin: float = 0.30):
    margin_value = float(margin)
    if margin_value < 0:
        raise ValueError(
            f"cyclic_all_bonds_within_error margin must be >= 0, got {margin}"
        )
    return {
        name: float(math.isfinite(error) and error <= margin_value)
        for name, error in cyclic_max_bond_error(eval_result).items()
    }


@loss_registry.register(
    optional_params=["margin"],
    description="Direct cyclic closure quality in [0,1] from distance excess; 1 means no bond exceeds target distance, 0 means at least one bond exceeds target distance by margin or more.",
)
def cyclic_closure_distance_score(eval_result, margin: float = 0.30):
    margin_value = float(margin)
    if margin_value <= 0:
        raise ValueError(
            f"cyclic_closure_distance_score margin must be > 0, got {margin}"
        )
    out: dict[str, float] = {}
    for oligo_name in eval_result.names:
        max_excess = _max_excess(_selected_distances(eval_result, str(oligo_name)))
        if not math.isfinite(max_excess):
            out[str(oligo_name)] = 0.0
        else:
            score = 1.0 - max(0.0, max_excess) / margin_value
            out[str(oligo_name)] = float(max(0.0, min(1.0, score)))
    return out
