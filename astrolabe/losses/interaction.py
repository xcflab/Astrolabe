import itertools
import logging
import math
import pathlib
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

_PYROSETTA_IMPORT_ERROR = None
try:
    import pyrosetta  # type: ignore
    from pyrosetta.rosetta.core.scoring import hbonds as rosetta_hbonds  # type: ignore
    from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import (  # type: ignore
        PerResidueSasaMetric,
    )
    from pyrosetta.rosetta.protocols.simple_filters import (  # type: ignore
        ContactMolecularSurfaceFilter,
        HolesFilter,
        ShapeComplementarityFilter,
    )
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue  # type: ignore
except Exception as exc:  # pragma: no cover - optional/proprietary dependency
    pyrosetta = None  # type: ignore
    rosetta_hbonds = None  # type: ignore
    PerResidueSasaMetric = None  # type: ignore
    ContactMolecularSurfaceFilter = None  # type: ignore
    HolesFilter = None  # type: ignore
    ShapeComplementarityFilter = None  # type: ignore
    MutateResidue = None  # type: ignore
    _PYROSETTA_IMPORT_ERROR = exc
from scipy.spatial import Delaunay, QhullError, cKDTree

from astrolabe.core.chain_resolution import (
    chain_ids_for_oligo,
    chain_to_protomer,
    protomer_to_chain_ids,
    resolve_selector_chain_ids,
    resolve_single_chain_id,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.base import loss_registry
from astrolabe.utils.file_parser import parse_mmcif, parse_pdb
from astrolabe.utils.mmcif_fallback import (
    ensure_centered_pdb_fallback,
    load_pose_with_optional_mmcif_pdb_fallback,
)


LIGAND_TYPES = (ProtomerTypes.SMILES, ProtomerTypes.CCD)
AUTODOCK_MIN_BOX_SIZE = 5.0
CD47_ANTIBODY_CONSENSUS_RESIDUES = (
    33,
    34,
    35,
    36,
    38,
    45,
    50,
    52,
    53,
    96,
    98,
    99,
    100,
    101,
    102,
    103,
)


def _require_pyrosetta():
    if pyrosetta is None:
        raise ImportError(
            "PyRosetta is required for these losses/selectors but is not available. "
            "Install it (e.g. via `pyrosetta-installer`) or remove PyRosetta-based terms."
        ) from _PYROSETTA_IMPORT_ERROR


def sasa_calculator(pose):
    _require_pyrosetta()
    sasa_metric = PerResidueSasaMetric()
    per_res_sasa_dict = sasa_metric.calculate(pose)
    result = np.zeros(len(per_res_sasa_dict))
    for i, val in per_res_sasa_dict.items():
        result[i - 1] = val
    return result


def _parse_residue_structure(model_path):
    path = str(model_path)
    if path.endswith(".pdb") or path.endswith(".ent"):
        return parse_pdb(path)
    return parse_mmcif(path)


def _chain_residue_indices(parsed: dict, chain_id: str | None = None) -> np.ndarray:
    pdb_idx = parsed.get("pdb_idx", [])
    if chain_id is None:
        return np.arange(len(pdb_idx), dtype=int)
    return np.asarray(
        [i for i, (cid, _resnum) in enumerate(pdb_idx) if str(cid) == str(chain_id)],
        dtype=int,
    )


def _ca_contact_pairs(
    parsed: dict,
    chain_a: str | None = None,
    chain_b: str | None = None,
    *,
    cutoff: float = 8.0,
) -> list[tuple[int, int]]:
    xyz = np.asarray(parsed.get("xyz"), dtype=float)
    mask = np.asarray(parsed.get("mask"), dtype=bool)
    if xyz.ndim != 3 or xyz.shape[1] <= 1 or mask.shape[:2] != xyz.shape[:2]:
        return []

    a_idx = _chain_residue_indices(parsed, chain_a)
    b_idx = _chain_residue_indices(parsed, chain_b)
    if chain_a is None and chain_b is None:
        chains = np.asarray([str(cid) for cid, _ in parsed.get("pdb_idx", [])], dtype=object)
        pairs = []
        ca_ok = mask[:, 1]
        for i in np.where(ca_ok)[0]:
            for j in np.where(ca_ok)[0]:
                if i >= j or chains[i] == chains[j]:
                    continue
                if float(np.linalg.norm(xyz[i, 1] - xyz[j, 1])) < float(cutoff):
                    pairs.append((int(i), int(j)))
        return pairs
    if a_idx.size == 0 or b_idx.size == 0:
        return []

    a_idx = a_idx[mask[a_idx, 1]]
    b_idx = b_idx[mask[b_idx, 1]]
    if a_idx.size == 0 or b_idx.size == 0:
        return []
    tree_a = cKDTree(xyz[a_idx, 1])
    tree_b = cKDTree(xyz[b_idx, 1])
    raw_pairs = tree_a.query_ball_tree(tree_b, r=float(cutoff))
    pairs: list[tuple[int, int]] = []
    for ia, neighbors in enumerate(raw_pairs):
        for ib in neighbors:
            i = int(a_idx[ia])
            j = int(b_idx[ib])
            if float(np.linalg.norm(xyz[i, 1] - xyz[j, 1])) < float(cutoff):
                pairs.append((i, j))
    return pairs


def _sc_contact_pairs(
    parsed: dict,
    chain_a: str | None = None,
    chain_b: str | None = None,
    *,
    cutoff: float = 5.0,
) -> list[tuple[int, int]]:
    xyz = np.asarray(parsed.get("xyz"), dtype=float)
    mask = np.asarray(parsed.get("mask"), dtype=bool)
    if xyz.ndim != 3 or mask.shape[:2] != xyz.shape[:2]:
        return []

    pdb_idx = parsed.get("pdb_idx", [])
    chains = np.asarray([str(cid) for cid, _ in pdb_idx], dtype=object)
    if chain_a is None and chain_b is None:
        a_idx = np.arange(len(pdb_idx), dtype=int)
        b_idx = a_idx
    else:
        a_idx = _chain_residue_indices(parsed, chain_a)
        b_idx = _chain_residue_indices(parsed, chain_b)
    if a_idx.size == 0 or b_idx.size == 0:
        return []

    a_atom_coords: list[np.ndarray] = []
    a_atom_res: list[int] = []
    b_atom_coords: list[np.ndarray] = []
    b_atom_res: list[int] = []
    for i in a_idx:
        atom_ids = np.where(mask[i])[0]
        a_atom_coords.extend(xyz[i, atom_ids])
        a_atom_res.extend([int(i)] * len(atom_ids))
    for j in b_idx:
        atom_ids = np.where(mask[j])[0]
        b_atom_coords.extend(xyz[j, atom_ids])
        b_atom_res.extend([int(j)] * len(atom_ids))
    if not a_atom_coords or not b_atom_coords:
        return []

    tree_a = cKDTree(np.asarray(a_atom_coords, dtype=float))
    tree_b = cKDTree(np.asarray(b_atom_coords, dtype=float))
    raw_pairs = tree_a.query_ball_tree(tree_b, r=float(cutoff))
    pairs: set[tuple[int, int]] = set()
    b_coords = np.asarray(b_atom_coords, dtype=float)
    for ia, neighbors in enumerate(raw_pairs):
        i = int(a_atom_res[ia])
        for ib in neighbors:
            j = int(b_atom_res[ib])
            if i == j or chains[i] == chains[j]:
                continue
            if float(np.linalg.norm(a_atom_coords[ia] - b_coords[ib])) <= float(cutoff):
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def find_interchain_contact_pairs(
    model_path,
    chain_a: str | None = None,
    chain_b: str | None = None,
    *,
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
) -> list[tuple[int, int]]:
    """Return residue-index contact pairs across chains."""
    parsed = _parse_residue_structure(model_path)
    method = str(contact_method).lower().replace("-", "_")
    if method == "sc":
        return _sc_contact_pairs(
            parsed,
            chain_a=chain_a,
            chain_b=chain_b,
            cutoff=float(sc_cutoff),
        )
    if method in {"ca", "ca_ca", "c_alpha", "calpha"}:
        return _ca_contact_pairs(
            parsed,
            chain_a=chain_a,
            chain_b=chain_b,
            cutoff=float(ca_cutoff),
        )
    raise ValueError(
        "contact_method must be one of sc or ca/ca_ca"
    )


def _split_chain_selectors(value: str | Sequence[str] | None, *, default: str) -> list[str]:
    if value is None or value == "":
        value = default
    if isinstance(value, str):
        return [
            token.strip()
            for chunk in value.split(",")
            for token in chunk.split()
            if token.strip()
        ]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _resolve_interface_chain_pairs(
    parsed: dict,
    *,
    chain_a: str | Sequence[str] | None,
    chain_b: str | Sequence[str] | None,
) -> list[tuple[str, str]]:
    """Resolve selectors into distinct chain pairs for interface metrics.

    Group selectors are expanded before any residue/contact calculation.  This
    keeps ``A`` versus ``others`` as ``(A,B)``, ``(A,C)``, ... rather than one
    concatenated A-versus-BCD block.  With no selectors, the legacy
    all-interchain contact reduction is retained.
    """
    chain_order: list[str] = []
    for chain_id, _ in parsed.get("pdb_idx", []):
        chain_id = str(chain_id)
        if chain_id not in chain_order:
            chain_order.append(chain_id)

    def tokens(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _split_chain_selectors(value, default="")
        try:
            return [str(item).strip() for item in value if str(item).strip()]
        except TypeError:
            token = str(value).strip()
            return [token] if token else []

    other_names = {"other", "others", "all-except-binder", "all_except_binder"}

    def is_other(value) -> bool:
        return any(token.lower() in other_names for token in tokens(value))

    def resolve(value, *, exclude: set[str] | None = None) -> list[str]:
        excluded = exclude or set()
        selected: list[str] = []
        for token in tokens(value):
            candidates = chain_order if token.lower() in other_names else [token]
            for chain_id in candidates:
                if (
                    chain_id in chain_order
                    and chain_id not in excluded
                    and chain_id not in selected
                ):
                    selected.append(chain_id)
        return selected

    if chain_a is None and chain_b is None:
        return [
            (chain_order[i], chain_order[j])
            for i in range(len(chain_order))
            for j in range(i + 1, len(chain_order))
        ]

    selected_b = resolve(chain_b)
    selected_a = resolve(chain_a, exclude=set(selected_b) if is_other(chain_a) else set())
    if is_other(chain_b):
        selected_b = resolve(chain_b, exclude=set(selected_a))

    return [
        (a_id, b_id)
        for a_id in selected_a
        for b_id in selected_b
        if a_id != b_id
    ]


def _resolve_interface_chain_pairs_from_structure(
    structure_path,
    *,
    chain_a: str | Sequence[str] | None,
    chain_b: str | Sequence[str] | None,
) -> list[tuple[str, str]]:
    parsed = _parse_residue_structure(structure_path)
    return _resolve_interface_chain_pairs(parsed, chain_a=chain_a, chain_b=chain_b)


def _target_contact_residue_numbers(
    parsed: dict,
    *,
    binder_chains: Sequence[str],
    target_chains: Sequence[str],
    contact_method: str,
    sc_cutoff: float,
    ca_cutoff: float,
) -> set[int]:
    method = str(contact_method).lower().replace("-", "_")
    pairs: set[tuple[int, int]] = set()
    for binder_chain in binder_chains:
        for target_chain in target_chains:
            if method == "sc":
                pairs.update(
                    _sc_contact_pairs(
                        parsed,
                        chain_a=binder_chain,
                        chain_b=target_chain,
                        cutoff=float(sc_cutoff),
                    )
                )
            elif method in {"ca", "ca_ca", "c_alpha", "calpha"}:
                pairs.update(
                    _ca_contact_pairs(
                        parsed,
                        chain_a=binder_chain,
                        chain_b=target_chain,
                        cutoff=float(ca_cutoff),
                    )
                )
            else:
                raise ValueError("contact_method must be one of sc or ca/ca_ca")

    pdb_idx = parsed.get("pdb_idx", [])
    target_set = {str(chain) for chain in target_chains}
    contacts: set[int] = set()
    for i, j in pairs:
        for idx in (int(i), int(j)):
            if idx >= len(pdb_idx):
                continue
            chain_id, resnum = pdb_idx[idx]
            if str(chain_id) in target_set:
                contacts.add(int(resnum))
    return contacts


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


@loss_registry.register(
    name="cd47_antibody_like_epitope_score",
    description=(
        "CD47 antibody-like target-epitope score: consensus-contact Jaccard, "
        "zeroed when overlap is below the antibody-like floor or the contact set is N-terminal-dominant."
    ),
    supports_oligomer_specific=True,
)
def cd47_antibody_like_epitope_score(
    eval_result,
    binder_chain: str = "A",
    target_chain: str = "B",
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    min_overlap: int = 6,
    zero_if_nterm_dominant: bool = True,
    nterm_max_resi: int = 30,
    nterm_fraction: float = 0.50,
):
    consensus = set(CD47_ANTIBODY_CONSENSUS_RESIDUES)
    binder_chains = _split_chain_selectors(binder_chain, default="A")
    target_chains = _split_chain_selectors(target_chain, default="B")
    out = {}
    for oligo_name, path_like in (getattr(eval_result, "complex_paths", {}) or {}).items():
        parsed = _parse_residue_structure(path_like)
        contacts = _target_contact_residue_numbers(
            parsed,
            binder_chains=binder_chains,
            target_chains=target_chains,
            contact_method=contact_method,
            sc_cutoff=float(sc_cutoff),
            ca_cutoff=float(ca_cutoff),
        )
        overlap = len(contacts & consensus)
        if overlap < int(min_overlap):
            out[oligo_name] = 0.0
            continue
        if contacts and _as_bool_value(zero_if_nterm_dominant):
            nterm_count = len([resnum for resnum in contacts if resnum <= int(nterm_max_resi)])
            nterm_contact_fraction = nterm_count / len(contacts)
            nterm_dominant = (
                nterm_contact_fraction > float(nterm_fraction)
                or max(contacts) <= int(nterm_max_resi)
            )
            if nterm_dominant:
                out[oligo_name] = 0.0
                continue
        out[oligo_name] = float(_jaccard(contacts, consensus))
    return out


def _compute_contact_pair_pae(
    pae: np.ndarray,
    contact_pairs: Sequence[tuple[int, int]],
    *,
    symmetric: str = "mean",
    reduction: str = "mean",
) -> float:
    vals: list[float] = []
    matrix = np.asarray(pae, dtype=float)
    mode = str(symmetric).lower()
    for i, j in contact_pairs:
        if i >= matrix.shape[0] or j >= matrix.shape[0]:
            continue
        a = float(matrix[i, j])
        b = float(matrix[j, i])
        if mode == "mean":
            vals.append((a + b) / 2.0)
        elif mode == "max":
            vals.append(max(a, b))
        elif mode == "min":
            vals.append(min(a, b))
        else:
            raise ValueError("symmetric must be mean, max, or min")
    if not vals:
        return float("nan")
    reduce_mode = str(reduction).lower()
    if reduce_mode == "mean":
        return float(np.mean(vals))
    if reduce_mode in {"minimal", "min"}:
        return float(np.min(vals))
    raise ValueError("reduction must be mean or minimal")


def compute_interface_pae(
    eval_result,
    *,
    symmetric: str = "mean",
    reduction: str = "mean",
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    chain_a: str | Sequence[str] | None = None,
    chain_b: str | Sequence[str] | None = None,
):
    """Calculate contact-pair interfacial PAE by selected chain pair.

    When a selector names multiple chains, each chain pair is reduced first and
    the resulting values are averaged equally.  Thus ``chain_a='A'`` and
    ``chain_b='others'`` excludes B-C/B-D/C-D target-target contacts.
    """
    from astrolabe.losses.confidence import _resolve_residue_level_pae_matrices

    ipae = {}
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    for oligo_name in eval_result.names:
        if len(str(oligo_name)) < 2:
            continue
        structure_path = complex_paths.get(oligo_name)
        mats = _resolve_residue_level_pae_matrices(eval_result, oligo_name=oligo_name)
        if structure_path is None or not mats:
            ipae[oligo_name] = float("nan")
            continue

        if chain_a is None and chain_b is None:
            contact_pairs = find_interchain_contact_pairs(
                structure_path,
                contact_method=contact_method,
                sc_cutoff=sc_cutoff,
                ca_cutoff=ca_cutoff,
            )
            values = [
                _compute_contact_pair_pae(
                    matrix,
                    contact_pairs,
                    symmetric=symmetric,
                    reduction=reduction,
                )
                for matrix in mats
            ]
            finite = [value for value in values if not math.isnan(float(value))]
            ipae[oligo_name] = float(np.mean(finite)) if finite else float("nan")
            continue

        chain_pairs = _resolve_interface_chain_pairs_from_structure(
            structure_path,
            chain_a=chain_a,
            chain_b=chain_b,
        )
        pair_values: list[float] = []
        missing_pair = False
        for pair_a, pair_b in chain_pairs:
            contact_pairs = find_interchain_contact_pairs(
                structure_path,
                chain_a=pair_a,
                chain_b=pair_b,
                contact_method=contact_method,
                sc_cutoff=sc_cutoff,
                ca_cutoff=ca_cutoff,
            )
            vals = [
                _compute_contact_pair_pae(
                    matrix,
                    contact_pairs,
                    symmetric=symmetric,
                    reduction=reduction,
                )
                for matrix in mats
            ]
            finite = [value for value in vals if not math.isnan(float(value))]
            if finite:
                pair_values.append(float(np.mean(finite)))
            else:
                missing_pair = True
        ipae[oligo_name] = (
            float(np.mean(pair_values))
            if pair_values and not missing_pair
            else float("nan")
        )
    return ipae


def _normalize_ligand_targets(
    ligand_targets: str | Sequence[str] | None,
) -> tuple[set[str], set[str]]:
    if ligand_targets is None:
        return set(), set()
    if isinstance(ligand_targets, str):
        ligand_targets = [ligand_targets]
    proto_targets = set()
    chain_targets = set()
    for target in ligand_targets:
        token = str(target).strip()
        if token in global_config.protomers:
            proto_targets.add(token)
        else:
            chain_targets.add(token)
    return proto_targets, chain_targets


def _resolve_ligand_chains(
    oligo_name: str,
    ligand_targets: str | Sequence[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    proto_targets, chain_targets = _normalize_ligand_targets(ligand_targets)
    prot2chain = _protomer_chain_map(oligo_name)
    forced_chain_targets = {
        chain_id
        for selector in chain_targets
        for chain_id in resolve_selector_chain_ids(oligo_name, selector)
    }
    ligands: list[tuple[str, str]] = []
    seen = set()
    for proto, chain_ids in prot2chain.items():
        protomer = global_config.protomers.get(proto)
        if protomer is None:
            continue
        default_ligand = protomer.type in LIGAND_TYPES
        proto_forced = proto in proto_targets
        for chain_id in chain_ids:
            forced = proto_forced or chain_id in forced_chain_targets
            if forced or (not proto_targets and not chain_targets and default_ligand):
                if chain_id not in seen:
                    ligands.append((proto, chain_id))
                    seen.add(chain_id)
    return ligands, prot2chain


def _protomer_chain_map(oligo_name: str) -> dict[str, list[str]]:
    """Map protomer identifiers in an oligomer name to ordered chain IDs."""
    return protomer_to_chain_ids(oligo_name)


def _protein_chains_for_oligo(oligo_name: str) -> list[str]:
    prot2chain = _protomer_chain_map(oligo_name)
    protein_chains: list[str] = []
    for proto, chain_ids in prot2chain.items():
        protomer = global_config.protomers.get(proto)
        if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
            continue
        protein_chains.extend(chain_ids)
    return protein_chains


def _residue_indices_for_chains(pose, chain_ids: Iterable[str]) -> list[int]:
    target = set(chain_ids)
    pdb_info = pose.pdb_info()
    return [
        i
        for i in range(1, pose.total_residue() + 1)
        if pdb_info.chain(i) in target
    ]


def _normalize_residue_numbers(raw: int | str | Sequence[int | str]) -> set[int]:
    if raw is None:
        return set()
    if isinstance(raw, (int, np.integer)):
        return {int(raw)}
    if isinstance(raw, str):
        tokens = [
            token.strip()
            for chunk in raw.split(",")
            for token in chunk.split()
            if token.strip()
        ]
    else:
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    return {int(token) for token in tokens}


def _is_mapping_like(raw) -> bool:
    return isinstance(raw, Mapping) or hasattr(raw, "items")


def _normalize_chain_selectors(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [
            token.strip()
            for chunk in raw.split(",")
            for token in chunk.split()
            if token.strip()
        ]
    return [str(item).strip() for item in raw if str(item).strip()]


def _normalize_chain_residue_specs(
    *,
    pocket_chain: str | Sequence[str] | None,
    pocket_residues: int | str | Sequence[int | str] | Mapping[str, int | str | Sequence[int | str]],
    pocket_residues_by_chain: Mapping[str, int | str | Sequence[int | str]] | None = None,
) -> list[tuple[str, set[int]]]:
    raw = pocket_residues_by_chain if pocket_residues_by_chain is not None else pocket_residues
    if _is_mapping_like(raw):
        specs: list[tuple[str, set[int]]] = []
        for selector, residues in raw.items():
            chain_selector = str(selector).strip()
            if not chain_selector:
                continue
            numbers = _normalize_residue_numbers(residues)
            if numbers:
                specs.append((chain_selector, numbers))
        return specs

    target_residue_numbers = _normalize_residue_numbers(raw)
    return [
        (selector, set(target_residue_numbers))
        for selector in _normalize_chain_selectors(pocket_chain)
        if target_residue_numbers
    ]


def _resolve_pocket_chain_number_specs(
    oligo_name: str,
    raw_specs: Iterable[tuple[str, set[int]]],
) -> list[tuple[str, set[int]]]:
    resolved: list[tuple[str, set[int]]] = []
    for selector, residue_numbers in raw_specs:
        chain_ids = resolve_selector_chain_ids(oligo_name, selector)
        if not chain_ids:
            raise ValueError(
                f"ligand_pocket_occupancy pocket selector={selector!r} does not match oligo {oligo_name!r}."
            )
        for chain_id in chain_ids:
            resolved.append((chain_id, set(residue_numbers)))
    return resolved


def _residue_indices_for_chain_numbers(
    pose,
    chain_ids: Iterable[str],
    residue_numbers: Iterable[int],
) -> list[int]:
    target_chains = set(chain_ids)
    target_numbers = set(int(num) for num in residue_numbers)
    pdb_info = pose.pdb_info()
    return [
        idx
        for idx in range(1, pose.total_residue() + 1)
        if pdb_info.chain(idx) in target_chains and int(pdb_info.number(idx)) in target_numbers
    ]


def _residue_indices_for_chain_number_specs(
    pose,
    chain_number_specs: Iterable[tuple[str, set[int]]],
) -> list[int]:
    numbers_by_chain: dict[str, set[int]] = defaultdict(set)
    for chain_id, residue_numbers in chain_number_specs:
        numbers_by_chain[str(chain_id)].update(int(num) for num in residue_numbers)
    pdb_info = pose.pdb_info()
    return [
        idx
        for idx in range(1, pose.total_residue() + 1)
        if int(pdb_info.number(idx)) in numbers_by_chain.get(pdb_info.chain(idx), set())
    ]


def _heavy_atom_coordinates(pose, residue_indices: Iterable[int]) -> np.ndarray:
    coords = []
    for res_idx in residue_indices:
        residue = pose.residue(res_idx)
        for atom_idx in range(1, residue.natoms() + 1):
            atom_type = residue.atom_type(atom_idx)
            if atom_type.element() == "H":
                continue
            atom_xyz = residue.xyz(atom_idx)
            coords.append([atom_xyz.x, atom_xyz.y, atom_xyz.z])
    return np.asarray(coords)


def _require_nonempty_coords(coords: np.ndarray, context: str) -> np.ndarray:
    if coords.size == 0:
        raise ValueError(f"{context} has no heavy atom coordinates.")
    return coords


def _fraction_inside_pocket_box(
    ligand_coords: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_padding: float,
) -> float:
    ligand_coords = _require_nonempty_coords(ligand_coords, "ligand")
    pocket_coords = _require_nonempty_coords(pocket_coords, "pocket")
    lower = np.min(pocket_coords, axis=0) - float(pocket_padding)
    upper = np.max(pocket_coords, axis=0) + float(pocket_padding)
    inside = np.logical_and(ligand_coords >= lower, ligand_coords <= upper).all(axis=1)
    return float(np.mean(inside))


def _structure_model_from_file(path):
    path = str(path)
    parser = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure("complex", path)[0]


def _residue_heavy_atom_coordinates_biopdb(residue) -> list[list[float]]:
    coords = []
    for atom in residue.get_atoms():
        element = str(getattr(atom, "element", "") or atom.get_name()[:1]).strip().upper()
        if element == "H":
            continue
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return coords


def _ligand_pocket_occupancy_from_structure_file(
    structure_path,
    *,
    pocket_chain_number_specs: Iterable[tuple[str, set[int]]],
    ligand_chain_ids: Iterable[str],
    pocket_padding: float,
) -> float:
    model = _structure_model_from_file(structure_path)
    numbers_by_chain: dict[str, set[int]] = defaultdict(set)
    for chain_id, residue_numbers in pocket_chain_number_specs:
        numbers_by_chain[str(chain_id)].update(int(num) for num in residue_numbers)
    ligand_chains = set(ligand_chain_ids)
    pocket_coords = []
    ligand_coords = []
    for chain in model:
        chain_id = str(chain.id).strip()
        for residue in chain.get_residues():
            if int(residue.id[1]) in numbers_by_chain.get(chain_id, set()):
                pocket_coords.extend(_residue_heavy_atom_coordinates_biopdb(residue))
            if chain_id in ligand_chains:
                ligand_coords.extend(_residue_heavy_atom_coordinates_biopdb(residue))
    if not pocket_coords:
        raise ValueError(
            f"ligand_pocket_occupancy selected no pocket atoms from {structure_path}."
        )
    if not ligand_coords:
        raise ValueError(
            f"ligand_pocket_occupancy selected no ligand atoms from {structure_path}."
        )
    return _fraction_inside_pocket_box(
        np.asarray(ligand_coords, dtype=float),
        np.asarray(pocket_coords, dtype=float),
        pocket_padding,
    )


def _resolve_dynamic_interface_chain_ids(
    *,
    model,
    oligo_name: str,
    interface_chains: str | Sequence[str] | None,
    ligand_chain_ids: Iterable[str],
    context: str,
) -> list[str]:
    available = set(_model_chain_by_id(model))
    ligand_set = {str(chain_id).strip() for chain_id in ligand_chain_ids}
    selectors = _normalize_chain_selectors(interface_chains)
    chain_ids: list[str] = []
    if selectors:
        for selector in selectors:
            candidates = resolve_selector_chain_ids(oligo_name, selector)
            token = str(selector).strip()
            if token.lower().startswith("chain:"):
                token = token.split(":", 1)[1].strip().upper()
            candidates.append(token)
            for chain_id in candidates:
                if chain_id in available and chain_id not in chain_ids:
                    chain_ids.append(chain_id)
                    break
            else:
                raise ValueError(
                    f"{context}: interface chain selector {selector!r} did not match "
                    f"chains {sorted(available)} for oligo {oligo_name!r}."
                )
    else:
        chain_map = chain_to_protomer(oligo_name)
        for chain_id in chain_ids_for_oligo(oligo_name):
            protomer = global_config.protomers.get(chain_map.get(chain_id))
            if protomer is None or protomer.type in LIGAND_TYPES:
                continue
            if chain_id in available and chain_id not in ligand_set:
                chain_ids.append(chain_id)

    if len(chain_ids) < 2:
        raise ValueError(
            f"{context} requires at least two protein interface chains, got {chain_ids!r}."
        )
    return chain_ids


def _interface_residues_from_structure_model(
    *,
    model,
    chain_ids: Sequence[str],
    interface_distance: float,
    context: str,
) -> list[Any]:
    residues_by_chain: dict[str, list[Any]] = {}
    ca_by_chain: dict[str, np.ndarray] = {}
    for chain_id in chain_ids:
        chain = _model_chain_by_id(model).get(str(chain_id))
        if chain is None:
            raise ValueError(f"{context}: chain {chain_id!r} is missing from structure.")
        residues = [residue for residue in chain.get_residues() if "CA" in residue]
        if not residues:
            raise ValueError(f"{context}: chain {chain_id!r} has no polymer CA atoms.")
        residues_by_chain[str(chain_id)] = residues
        ca_by_chain[str(chain_id)] = np.asarray(
            [
                [float(coord) for coord in residue["CA"].get_coord()]
                for residue in residues
            ],
            dtype=float,
        )

    selected: set[tuple[str, int]] = set()
    cutoff = float(interface_distance)
    for chain_a, chain_b in itertools.combinations([str(c) for c in chain_ids], 2):
        coords_a = ca_by_chain[chain_a]
        coords_b = ca_by_chain[chain_b]
        raw_pairs = cKDTree(coords_a).query_ball_tree(cKDTree(coords_b), r=cutoff)
        for idx_a, neighbors in enumerate(raw_pairs):
            if not neighbors:
                continue
            selected.add((chain_a, int(idx_a)))
            for idx_b in neighbors:
                selected.add((chain_b, int(idx_b)))

    if not selected:
        raise ValueError(
            f"{context}: no interface residues found within {cutoff:.3f} A for "
            f"chains {list(chain_ids)!r}."
        )

    return [
        residue
        for chain_id in chain_ids
        for idx, residue in enumerate(residues_by_chain[str(chain_id)])
        if (str(chain_id), idx) in selected
    ]


def _heavy_atom_coordinates_for_residues_biopdb(
    residues: Iterable[Any],
    *,
    context: str,
) -> np.ndarray:
    coords = []
    for residue in residues:
        coords.extend(_residue_heavy_atom_coordinates_biopdb(residue))
    return _require_nonempty_coords(np.asarray(coords, dtype=float), context)


def _cb_or_ca_coordinates_for_residues_biopdb(
    residues: Iterable[Any],
    *,
    context: str,
) -> np.ndarray:
    coords = [
        _residue_cb_or_ca_coordinate_biopdb(residue, context=context)
        for residue in residues
    ]
    return _require_alpha_shape_points(np.asarray(coords, dtype=float), context)


def _model_chain_by_id(model) -> dict[str, Any]:
    return {str(chain.id).strip(): chain for chain in model}


def _chain_id_from_model_selector(model, selector: str, *, context: str) -> str:
    token = str(selector).strip()
    if token.lower().startswith("chain:"):
        token = token.split(":", 1)[1].strip().upper()
    chains = _model_chain_by_id(model)
    if token in chains:
        return token
    raise ValueError(
        f"{context}: chain selector {selector!r} did not match chains {sorted(chains)}."
    )


def _chain_ids_from_pose(pose) -> set[str]:
    pdb_info = pose.pdb_info()
    return {
        str(pdb_info.chain(idx)).strip()
        for idx in range(1, pose.total_residue() + 1)
        if str(pdb_info.chain(idx)).strip()
    }


def _chain_id_from_pose_selector(
    pose,
    selector: str,
    *,
    oligo_name: str,
    context: str,
) -> str:
    existing = _chain_ids_from_pose(pose)
    candidates = resolve_selector_chain_ids(oligo_name, selector)
    token = str(selector).strip()
    if token.lower().startswith("chain:"):
        token = token.split(":", 1)[1].strip().upper()
    candidates.append(token)
    for chain_id in candidates:
        if chain_id in existing:
            return chain_id
    raise ValueError(
        f"{context}: chain selector {selector!r} did not match chains {sorted(existing)}."
    )


def _selector_mapping(raw_selector: Any) -> dict[str, str]:
    if raw_selector is None or isinstance(raw_selector, str):
        return {}
    if hasattr(raw_selector, "items"):
        return {
            str(key).strip(): str(value).strip()
            for key, value in raw_selector.items()
            if str(key).strip() and str(value).strip()
        }
    return {}


def _selector_list(raw_selector: Any) -> list[str]:
    if raw_selector is None or _is_mapping_like(raw_selector):
        return []
    if isinstance(raw_selector, str):
        return [
            token.strip()
            for chunk in raw_selector.split(",")
            for token in chunk.split()
            if token.strip()
        ]
    return [str(item).strip() for item in raw_selector if str(item).strip()]


def _alignment_chain_pairs(
    reference_structure_chain: Any,
    current_structure_chain: Any,
) -> list[tuple[str, str]]:
    reference_map = _selector_mapping(reference_structure_chain)
    current_map = _selector_mapping(current_structure_chain)
    if reference_map or current_map:
        keys: list[str] = []
        for mapping in (reference_map, current_map):
            for key in mapping:
                if key not in keys:
                    keys.append(key)
        if not keys:
            raise ValueError("ligand_aligned_reference_pose_distance requires chain selectors.")
        return [
            (reference_map.get(key, key), current_map.get(key, key))
            for key in keys
        ]

    reference_selectors = _selector_list(reference_structure_chain)
    current_selectors = _selector_list(current_structure_chain)
    if not reference_selectors and not current_selectors:
        raise ValueError(
            "ligand_aligned_reference_pose_distance requires reference_structure_chain "
            "or current_structure_chain."
        )
    if not reference_selectors:
        reference_selectors = list(current_selectors)
    if not current_selectors:
        current_selectors = list(reference_selectors)
    if len(reference_selectors) != len(current_selectors):
        raise ValueError(
            "ligand_aligned_reference_pose_distance requires equal numbers of "
            "reference/current chain selectors."
        )
    return list(zip(reference_selectors, current_selectors))


def _ca_coordinates_from_model_chain(model, selector: str) -> np.ndarray:
    chain_id = _chain_id_from_model_selector(
        model,
        selector,
        context="ligand_aligned_reference_pose_distance reference",
    )
    coords = []
    for residue in _model_chain_by_id(model)[chain_id].get_residues():
        if "CA" not in residue:
            continue
        atom = residue["CA"]
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return np.asarray(coords, dtype=float)


def _ca_coordinates_from_pose_chain(pose, selector: str, *, oligo_name: str) -> np.ndarray:
    chain_id = _chain_id_from_pose_selector(
        pose,
        selector,
        oligo_name=oligo_name,
        context="ligand_aligned_reference_pose_distance current",
    )
    coords = []
    pdb_info = pose.pdb_info()
    for idx in range(1, pose.total_residue() + 1):
        if pdb_info.chain(idx) != chain_id:
            continue
        residue = pose.residue(idx)
        try:
            has_ca = bool(residue.has("CA"))
        except Exception:
            has_ca = False
        if not has_ca:
            continue
        atom_xyz = residue.xyz("CA")
        coords.append([atom_xyz.x, atom_xyz.y, atom_xyz.z])
    return np.asarray(coords, dtype=float)


def _heavy_atom_coordinates_for_model_chains(model, chain_ids: Iterable[str]) -> np.ndarray:
    chain_set = {str(chain_id).strip() for chain_id in chain_ids}
    coords = []
    for chain in model:
        if str(chain.id).strip() not in chain_set:
            continue
        for residue in chain.get_residues():
            coords.extend(_residue_heavy_atom_coordinates_biopdb(residue))
    return np.asarray(coords, dtype=float)


def _polymer_residues_from_model_chain(model, selector: str) -> list[Any]:
    chain_id = _chain_id_from_model_selector(
        model,
        selector,
        context="ligand_aligned_reference_pocket_box_occupancy",
    )
    return [
        residue
        for residue in _model_chain_by_id(model)[chain_id].get_residues()
        if "CA" in residue
    ]


def _polymer_residue_indices_from_pose_chain(
    pose,
    selector: str,
    *,
    oligo_name: str,
) -> list[int]:
    chain_id = _chain_id_from_pose_selector(
        pose,
        selector,
        oligo_name=oligo_name,
        context="ligand_aligned_reference_pocket_box_occupancy",
    )
    indices = []
    pdb_info = pose.pdb_info()
    for idx in range(1, pose.total_residue() + 1):
        if pdb_info.chain(idx) != chain_id:
            continue
        residue = pose.residue(idx)
        try:
            has_ca = bool(residue.has("CA"))
        except Exception:
            has_ca = False
        if has_ca:
            indices.append(idx)
    return indices


def _author_position_to_ordinal(residues: Sequence[Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, residue in enumerate(residues):
        try:
            author_position = int(residue.id[1])
        except Exception:
            continue
        mapping.setdefault(author_position, idx)
    return mapping


def _chain_selector_for_pocket_key(
    raw_selector: Any,
    pocket_key: str,
    *,
    allow_single_for_multichain: bool,
) -> str:
    mapping = _selector_mapping(raw_selector)
    if mapping:
        return mapping.get(pocket_key, pocket_key)
    selectors = _selector_list(raw_selector)
    if len(selectors) == 1 and allow_single_for_multichain:
        return selectors[0]
    return pocket_key


def _reference_pocket_specs(
    *,
    reference_pocket_positions: int | str | Sequence[int | str] | None = None,
    reference_pocket_positions_by_chain: Mapping[str, int | str | Sequence[int | str]] | None = None,
    reference_structure_chain: Any = None,
    current_structure_chain: Any = None,
) -> list[tuple[str, str, set[int]]]:
    if reference_pocket_positions_by_chain is not None:
        single_chain_mode = len(reference_pocket_positions_by_chain) == 1
        specs = []
        for pocket_key, positions in reference_pocket_positions_by_chain.items():
            ref_selector = _chain_selector_for_pocket_key(
                reference_structure_chain,
                str(pocket_key),
                allow_single_for_multichain=single_chain_mode,
            )
            cur_selector = _chain_selector_for_pocket_key(
                current_structure_chain,
                str(pocket_key),
                allow_single_for_multichain=single_chain_mode,
            )
            numbers = _normalize_residue_numbers(positions)
            if numbers:
                specs.append((ref_selector, cur_selector, numbers))
        return specs

    positions = _normalize_residue_numbers(reference_pocket_positions)
    if not positions:
        raise ValueError(
            "ligand_aligned_reference_pocket_box_occupancy requires reference_pocket_positions "
            "or reference_pocket_positions_by_chain."
        )
    chain_pairs = _alignment_chain_pairs(reference_structure_chain, current_structure_chain)
    if len(chain_pairs) != 1:
        raise ValueError(
            "ligand_aligned_reference_pocket_box_occupancy requires "
            "reference_pocket_positions_by_chain for multi-chain pockets."
        )
    ref_selector, cur_selector = chain_pairs[0]
    return [(ref_selector, cur_selector, positions)]


def _current_pocket_coords_from_structure_model(
    *,
    reference_model,
    current_model,
    pocket_specs: Sequence[tuple[str, str, set[int]]],
    allow_length_mismatch: bool,
) -> np.ndarray:
    coords = []
    for ref_selector, cur_selector, positions in pocket_specs:
        reference_residues = _polymer_residues_from_model_chain(reference_model, ref_selector)
        current_residues = _polymer_residues_from_model_chain(current_model, cur_selector)
        if len(reference_residues) != len(current_residues) and not allow_length_mismatch:
            raise ValueError(
                "ligand_aligned_reference_pocket_box_occupancy reference/current chain "
                f"lengths do not match for reference={ref_selector!r}, current={cur_selector!r}: "
                f"{len(reference_residues)} vs {len(current_residues)}."
            )
        reference_ordinal = _author_position_to_ordinal(reference_residues)
        for pos in sorted(int(p) for p in positions):
            ordinal = reference_ordinal.get(pos)
            if ordinal is None or ordinal >= len(current_residues):
                continue
            coords.extend(_residue_heavy_atom_coordinates_biopdb(current_residues[ordinal]))
    return _require_nonempty_coords(
        np.asarray(coords, dtype=float),
        "ligand_aligned_reference_pocket_box_occupancy pocket",
    )


def _current_pocket_coords_from_pose(
    *,
    reference_model,
    pose,
    oligo_name: str,
    pocket_specs: Sequence[tuple[str, str, set[int]]],
    allow_length_mismatch: bool,
) -> np.ndarray:
    residue_indices = []
    for ref_selector, cur_selector, positions in pocket_specs:
        reference_residues = _polymer_residues_from_model_chain(reference_model, ref_selector)
        current_indices = _polymer_residue_indices_from_pose_chain(
            pose,
            cur_selector,
            oligo_name=oligo_name,
        )
        if len(reference_residues) != len(current_indices) and not allow_length_mismatch:
            raise ValueError(
                "ligand_aligned_reference_pocket_box_occupancy reference/current chain "
                f"lengths do not match for reference={ref_selector!r}, current={cur_selector!r}: "
                f"{len(reference_residues)} vs {len(current_indices)}."
            )
        reference_ordinal = _author_position_to_ordinal(reference_residues)
        for pos in sorted(int(p) for p in positions):
            ordinal = reference_ordinal.get(pos)
            if ordinal is None or ordinal >= len(current_indices):
                continue
            residue_indices.append(current_indices[ordinal])
    return _require_nonempty_coords(
        _heavy_atom_coordinates(pose, residue_indices),
        "ligand_aligned_reference_pocket_box_occupancy pocket",
    )


def _residue_cb_or_ca_coordinate_biopdb(residue, *, context: str) -> np.ndarray:
    resname = str(residue.get_resname()).strip().upper()
    preferred_atom_name = "CA" if resname == "GLY" else "CB"
    atom_name = preferred_atom_name if preferred_atom_name in residue else "CA"
    if atom_name not in residue:
        raise ValueError(
            f"{context}: residue {resname} {residue.id[1]} is missing {preferred_atom_name}/CA."
        )
    coord = residue[atom_name].get_coord()
    return np.asarray([float(coord[0]), float(coord[1]), float(coord[2])], dtype=float)


def _residue_cb_or_ca_coordinate_pose(residue, *, context: str) -> np.ndarray:
    try:
        resname = str(residue.name3()).strip().upper()
    except Exception:
        resname = ""
    preferred_atom_name = "CA" if resname == "GLY" else "CB"
    atom_name = preferred_atom_name
    try:
        has_atom = bool(residue.has(atom_name))
    except Exception:
        has_atom = False
    if not has_atom and preferred_atom_name != "CA":
        atom_name = "CA"
        try:
            has_atom = bool(residue.has(atom_name))
        except Exception:
            has_atom = False
    if not has_atom:
        raise ValueError(
            f"{context}: residue {resname or '?'} is missing {preferred_atom_name}/CA."
        )
    atom_xyz = residue.xyz(atom_name)
    return np.asarray([atom_xyz.x, atom_xyz.y, atom_xyz.z], dtype=float)


def _require_alpha_shape_points(points: np.ndarray, context: str) -> np.ndarray:
    pts = _require_nonempty_coords(np.asarray(points, dtype=float), context)
    pts = np.unique(pts, axis=0)
    if len(pts) < 4:
        raise ValueError(f"{context} needs at least 4 unique C-beta points, got {len(pts)}.")
    return pts


def _current_pocket_cb_coords_from_structure_model(
    *,
    reference_model,
    current_model,
    pocket_specs: Sequence[tuple[str, str, set[int]]],
    allow_length_mismatch: bool,
) -> np.ndarray:
    coords = []
    for ref_selector, cur_selector, positions in pocket_specs:
        reference_residues = _polymer_residues_from_model_chain(reference_model, ref_selector)
        current_residues = _polymer_residues_from_model_chain(current_model, cur_selector)
        if len(reference_residues) != len(current_residues) and not allow_length_mismatch:
            raise ValueError(
                "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy "
                "reference/current chain lengths do not match for "
                f"reference={ref_selector!r}, current={cur_selector!r}: "
                f"{len(reference_residues)} vs {len(current_residues)}."
            )
        reference_ordinal = _author_position_to_ordinal(reference_residues)
        for pos in sorted(int(p) for p in positions):
            ordinal = reference_ordinal.get(pos)
            if ordinal is None or ordinal >= len(current_residues):
                continue
            coords.append(
                _residue_cb_or_ca_coordinate_biopdb(
                    current_residues[ordinal],
                    context="ligand_aligned_reference_pocket_cb_alpha_shape_occupancy pocket",
                )
            )
    return _require_alpha_shape_points(
        np.asarray(coords, dtype=float),
        "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy pocket",
    )


def _current_pocket_cb_coords_from_pose(
    *,
    reference_model,
    pose,
    oligo_name: str,
    pocket_specs: Sequence[tuple[str, str, set[int]]],
    allow_length_mismatch: bool,
) -> np.ndarray:
    coords = []
    for ref_selector, cur_selector, positions in pocket_specs:
        reference_residues = _polymer_residues_from_model_chain(reference_model, ref_selector)
        current_indices = _polymer_residue_indices_from_pose_chain(
            pose,
            cur_selector,
            oligo_name=oligo_name,
        )
        if len(reference_residues) != len(current_indices) and not allow_length_mismatch:
            raise ValueError(
                "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy "
                "reference/current chain lengths do not match for "
                f"reference={ref_selector!r}, current={cur_selector!r}: "
                f"{len(reference_residues)} vs {len(current_indices)}."
            )
        reference_ordinal = _author_position_to_ordinal(reference_residues)
        for pos in sorted(int(p) for p in positions):
            ordinal = reference_ordinal.get(pos)
            if ordinal is None or ordinal >= len(current_indices):
                continue
            coords.append(
                _residue_cb_or_ca_coordinate_pose(
                    pose.residue(current_indices[ordinal]),
                    context="ligand_aligned_reference_pocket_cb_alpha_shape_occupancy pocket",
                )
            )
    return _require_alpha_shape_points(
        np.asarray(coords, dtype=float),
        "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy pocket",
    )


def _circumsphere_radius(points: np.ndarray) -> float:
    p1, p2, p3, p4 = points
    matrix = 2.0 * np.vstack([p2 - p1, p3 - p1, p4 - p1])
    rhs = np.asarray(
        [
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1),
            np.dot(p4, p4) - np.dot(p1, p1),
        ],
        dtype=float,
    )
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return math.inf
    return float(np.linalg.norm(center - p1))


def _positive_finite_float(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}.")
    return parsed


def _adaptive_alpha_radius(
    points: np.ndarray,
    *,
    scale: float = 1.8,
    min_radius: float = 3.0,
    max_radius: float = 8.0,
) -> float:
    pts = _require_alpha_shape_points(points, "alpha shape")
    scale = _positive_finite_float(scale, "alpha_radius_scale")
    min_radius = _positive_finite_float(min_radius, "alpha_min_radius")
    max_radius = _positive_finite_float(max_radius, "alpha_max_radius")
    if max_radius < min_radius:
        raise ValueError(
            "alpha_max_radius must be greater than or equal to alpha_min_radius, "
            f"got {max_radius!r} < {min_radius!r}."
        )
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    kth_index = min(3, len(pts) - 2)
    kth = np.partition(dist, kth_index, axis=1)[:, kth_index]
    finite = kth[np.isfinite(kth)]
    if finite.size == 0:
        raise ValueError("alpha shape cannot infer radius from pocket C-beta points.")
    median_k = float(np.median(finite))
    return float(np.clip(median_k * scale, min_radius, max_radius))


def _alpha_shape_kept_simplex_indices(
    points: np.ndarray,
    *,
    alpha_radius: float | None = None,
    alpha_radius_scale: float = 1.8,
    alpha_min_radius: float = 3.0,
    alpha_max_radius: float = 8.0,
) -> tuple[Delaunay, set[int], float]:
    pts = _require_alpha_shape_points(
        points,
        "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy pocket",
    )
    radius_cutoff = (
        _positive_finite_float(alpha_radius, "alpha_radius")
        if alpha_radius is not None
        else _adaptive_alpha_radius(
            pts,
            scale=alpha_radius_scale,
            min_radius=alpha_min_radius,
            max_radius=alpha_max_radius,
        )
    )
    try:
        triangulation = Delaunay(pts)
    except QhullError:
        triangulation = Delaunay(pts, qhull_options="QJ")
    kept = {
        int(idx)
        for idx, simplex in enumerate(triangulation.simplices)
        if _circumsphere_radius(pts[simplex]) <= radius_cutoff
    }
    if not kept:
        raise ValueError(
            "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy produced no "
            f"alpha-shape tetrahedra with radius cutoff {radius_cutoff:.3f}."
        )
    return triangulation, kept, radius_cutoff


def _fraction_inside_cb_alpha_shape(
    ligand_coords: np.ndarray,
    pocket_cb_coords: np.ndarray,
    *,
    alpha_radius: float | None = None,
    alpha_radius_scale: float = 1.8,
    alpha_min_radius: float = 3.0,
    alpha_max_radius: float = 8.0,
    tolerance: float = 1e-8,
) -> float:
    ligand_coords = _require_nonempty_coords(
        ligand_coords,
        "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy ligand",
    )
    triangulation, kept, _radius_cutoff = _alpha_shape_kept_simplex_indices(
        pocket_cb_coords,
        alpha_radius=alpha_radius,
        alpha_radius_scale=alpha_radius_scale,
        alpha_min_radius=alpha_min_radius,
        alpha_max_radius=alpha_max_radius,
    )
    simplex_indices = triangulation.find_simplex(ligand_coords, tol=float(tolerance))
    inside = np.asarray([int(idx) in kept for idx in simplex_indices], dtype=bool)
    return float(np.mean(inside))


def _fraction_inside_autodock_box(
    ligand_coords: np.ndarray,
    pocket_coords: np.ndarray,
    *,
    box_padding: float,
    min_box_size: float = AUTODOCK_MIN_BOX_SIZE,
) -> float:
    ligand_coords = _require_nonempty_coords(
        ligand_coords,
        "ligand_aligned_reference_pocket_box_occupancy ligand",
    )
    pocket_coords = _require_nonempty_coords(
        pocket_coords,
        "ligand_aligned_reference_pocket_box_occupancy pocket",
    )
    lower_raw = np.min(pocket_coords, axis=0)
    upper_raw = np.max(pocket_coords, axis=0)
    center = (lower_raw + upper_raw) / 2.0
    size = np.maximum(upper_raw - lower_raw + 2.0 * float(box_padding), float(min_box_size))
    lower = center - size / 2.0
    upper = center + size / 2.0
    inside = np.logical_and(ligand_coords >= lower, ligand_coords <= upper).all(axis=1)
    return float(np.mean(inside))


def _all_heavy_atom_coordinates_from_structure_file(path) -> np.ndarray:
    model = _structure_model_from_file(path)
    coords = []
    for chain in model:
        for residue in chain.get_residues():
            coords.extend(_residue_heavy_atom_coordinates_biopdb(residue))
    return _require_nonempty_coords(
        np.asarray(coords, dtype=float),
        f"ligand_aligned_reference_pose_distance reference ligand {path}",
    )


def _append_ordered_ca_pairs(
    *,
    reference_coords: np.ndarray,
    current_coords: np.ndarray,
    reference_selector: str,
    current_selector: str,
    target_reference: list[np.ndarray],
    mobile_current: list[np.ndarray],
    min_ca_atoms: int,
    allow_length_mismatch: bool,
) -> None:
    if reference_coords.size == 0 or current_coords.size == 0:
        raise ValueError(
            "ligand_aligned_reference_pose_distance selected an empty CA set "
            f"for reference={reference_selector!r}, current={current_selector!r}."
        )
    if len(reference_coords) != len(current_coords) and not allow_length_mismatch:
        raise ValueError(
            "ligand_aligned_reference_pose_distance reference/current chain lengths "
            f"do not match for reference={reference_selector!r}, current={current_selector!r}: "
            f"{len(reference_coords)} vs {len(current_coords)}."
        )
    count = min(len(reference_coords), len(current_coords))
    if count < int(min_ca_atoms):
        raise ValueError(
            "ligand_aligned_reference_pose_distance has too few matched CA atoms "
            f"for reference={reference_selector!r}, current={current_selector!r}: "
            f"{count} < {int(min_ca_atoms)}."
        )
    target_reference.extend(reference_coords[:count])
    mobile_current.extend(current_coords[:count])


def _protein_alignment_arrays_from_structure_file(
    *,
    reference_model,
    current_model,
    chain_pairs: Sequence[tuple[str, str]],
    min_ca_atoms: int,
    allow_length_mismatch: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_all = []
    current_all = []
    for reference_selector, current_selector in chain_pairs:
        _append_ordered_ca_pairs(
            reference_coords=_ca_coordinates_from_model_chain(
                reference_model, reference_selector
            ),
            current_coords=_ca_coordinates_from_model_chain(
                current_model, current_selector
            ),
            reference_selector=reference_selector,
            current_selector=current_selector,
            target_reference=reference_all,
            mobile_current=current_all,
            min_ca_atoms=min_ca_atoms,
            allow_length_mismatch=allow_length_mismatch,
        )
    return np.vstack(current_all), np.vstack(reference_all)


def _protein_alignment_arrays_from_pose(
    *,
    reference_model,
    current_pose,
    oligo_name: str,
    chain_pairs: Sequence[tuple[str, str]],
    min_ca_atoms: int,
    allow_length_mismatch: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_all = []
    current_all = []
    for reference_selector, current_selector in chain_pairs:
        _append_ordered_ca_pairs(
            reference_coords=_ca_coordinates_from_model_chain(
                reference_model, reference_selector
            ),
            current_coords=_ca_coordinates_from_pose_chain(
                current_pose,
                current_selector,
                oligo_name=oligo_name,
            ),
            reference_selector=reference_selector,
            current_selector=current_selector,
            target_reference=reference_all,
            mobile_current=current_all,
            min_ca_atoms=min_ca_atoms,
            allow_length_mismatch=allow_length_mismatch,
        )
    return np.vstack(current_all), np.vstack(reference_all)


def _kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid
    covariance = mobile_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = target_centroid - mobile_centroid @ rotation
    return rotation, translation


def _aligned_ligand_centroid_distance(
    *,
    current_ligand_coords: np.ndarray,
    reference_ligand_coords: np.ndarray,
    mobile_ca_coords: np.ndarray,
    reference_ca_coords: np.ndarray,
) -> float:
    rotation, translation = _kabsch_transform(mobile_ca_coords, reference_ca_coords)
    aligned_ligand = current_ligand_coords @ rotation + translation
    return float(
        np.linalg.norm(
            aligned_ligand.mean(axis=0) - reference_ligand_coords.mean(axis=0)
        )
    )


def _ligand_reference_pose_distance_from_structure_file(
    structure_path,
    *,
    reference_model,
    reference_ligand_coords: np.ndarray,
    chain_pairs: Sequence[tuple[str, str]],
    ligand_chain_ids: Iterable[str],
    min_ca_atoms: int,
    allow_length_mismatch: bool,
) -> float:
    current_model = _structure_model_from_file(structure_path)
    mobile_ca, reference_ca = _protein_alignment_arrays_from_structure_file(
        reference_model=reference_model,
        current_model=current_model,
        chain_pairs=chain_pairs,
        min_ca_atoms=min_ca_atoms,
        allow_length_mismatch=allow_length_mismatch,
    )
    current_ligand_coords = _require_nonempty_coords(
        _heavy_atom_coordinates_for_model_chains(current_model, ligand_chain_ids),
        f"ligand_aligned_reference_pose_distance ligand in {structure_path}",
    )
    return _aligned_ligand_centroid_distance(
        current_ligand_coords=current_ligand_coords,
        reference_ligand_coords=reference_ligand_coords,
        mobile_ca_coords=mobile_ca,
        reference_ca_coords=reference_ca,
    )


def _ligand_reference_pose_distance_from_pose(
    pose,
    *,
    oligo_name: str,
    reference_model,
    reference_ligand_coords: np.ndarray,
    chain_pairs: Sequence[tuple[str, str]],
    ligand_chain_ids: Iterable[str],
    min_ca_atoms: int,
    allow_length_mismatch: bool,
) -> float:
    mobile_ca, reference_ca = _protein_alignment_arrays_from_pose(
        reference_model=reference_model,
        current_pose=pose,
        oligo_name=oligo_name,
        chain_pairs=chain_pairs,
        min_ca_atoms=min_ca_atoms,
        allow_length_mismatch=allow_length_mismatch,
    )
    ligand_residue_indices = _residue_indices_for_chains(pose, ligand_chain_ids)
    if not ligand_residue_indices:
        raise ValueError(
            f"ligand_aligned_reference_pose_distance found no ligand residues for oligo {oligo_name!r}."
        )
    current_ligand_coords = _require_nonempty_coords(
        _heavy_atom_coordinates(pose, ligand_residue_indices),
        f"ligand_aligned_reference_pose_distance ligand for oligo {oligo_name!r}",
    )
    return _aligned_ligand_centroid_distance(
        current_ligand_coords=current_ligand_coords,
        reference_ligand_coords=reference_ligand_coords,
        mobile_ca_coords=mobile_ca,
        reference_ca_coords=reference_ca,
    )


def _extract_subpose(pose, keep_residue_indices: Sequence[int]):
    keep = set(keep_residue_indices)
    if not keep:
        return None
    new_pose = pose.clone()
    for idx in range(new_pose.total_residue(), 0, -1):
        if idx not in keep:
            new_pose.delete_residue_slow(idx)
    return new_pose


def _saturating_fraction(value: float, midpoint: float) -> float:
    value = max(0.0, float(value))
    midpoint = max(1e-6, float(midpoint))
    return value / (value + midpoint)


def _build_ligand_environment_pose(pose, ligand_chain: str, neighbor_distance: float):
    ligand_residues = _residue_indices_for_chains(pose, [ligand_chain])
    if not ligand_residues:
        return None
    ligand_coords = _heavy_atom_coordinates(pose, ligand_residues)
    if ligand_coords.size == 0:
        return None
    ligand_tree = cKDTree(ligand_coords)
    neighbor_residues = []
    for res_idx in range(1, pose.total_residue() + 1):
        if res_idx in ligand_residues:
            continue
        residue = pose.residue(res_idx)
        if not residue.is_protein():
            continue
        coords = _heavy_atom_coordinates(pose, [res_idx])
        if coords.size == 0:
            continue
        nearest, _ = ligand_tree.query(coords, k=1)
        if np.min(nearest) <= neighbor_distance:
            neighbor_residues.append(res_idx)
    return _extract_subpose(pose, ligand_residues + neighbor_residues)


def _isolate_chains(pose, keep_chains: Iterable[str]):
    keep = set(keep_chains)
    if not keep:
        return None
    new_pose = pose.clone()
    pdb_info = new_pose.pdb_info()
    for idx in range(new_pose.total_residue(), 0, -1):
        if pdb_info.chain(idx) not in keep:
            new_pose.delete_residue_slow(idx)
    return new_pose


def _compute_shape_complementarity_between_chains(pose, chain_a: str, chain_b: str):
    _require_pyrosetta()
    if not chain_a or not chain_b or chain_a == chain_b:
        return None
    sub_pose = _isolate_chains(pose, [chain_a, chain_b])
    if sub_pose is None or sub_pose.total_residue() < 2 or sub_pose.num_chains() < 2:
        return None
    sc_filter = ShapeComplementarityFilter()
    sc_filter.jump_id(1)
    sc_filter.use_rosetta_radii(True)
    try:
        return float(sc_filter.report_sm(sub_pose))
    except RuntimeError as exc:  # pragma: no cover - PyRosetta runtime errors
        logging.warning(
            "Shape complementarity calculation failed for chains %s-%s: %s",
            chain_a,
            chain_b,
            exc,
        )
        return None


def _compute_contact_molecular_surface_between_chains(
    pose,
    chain_a: str | Sequence[str],
    chain_b: str | Sequence[str],
    *,
    distance_weight: float = 0.5,
    use_rosetta_radii: bool = True,
    protomer_a_residues: str | None = None,
    protomer_b_residues: str | None = None,
):
    _require_pyrosetta()
    chains_a = [
        str(chain) for chain in ([chain_a] if isinstance(chain_a, str) else chain_a)
    ]
    chains_b = [
        str(chain) for chain in ([chain_b] if isinstance(chain_b, str) else chain_b)
    ]
    if not chains_a or not chains_b or set(chains_a) & set(chains_b):
        return None

    cms_filter = ContactMolecularSurfaceFilter()
    cms_filter.selector1(
        _chain_group_region_selector(pose, chains_a, protomer_a_residues)
    )
    cms_filter.selector2(
        _chain_group_region_selector(pose, chains_b, protomer_b_residues)
    )
    cms_filter.distance_weight(float(distance_weight))
    cms_filter.use_rosetta_radii(bool(use_rosetta_radii))
    try:
        return float(cms_filter.report_sm(pose))
    except RuntimeError as exc:  # pragma: no cover - PyRosetta runtime errors
        logging.warning(
            "Contact molecular surface calculation failed for chains %s-%s: %s",
            chain_a,
            chain_b,
            exc,
        )
        return None


def _chain_relative_residue_indices(pose, chain: str, residues: str) -> list[int]:
    pdb_info = pose.pdb_info()
    chain_indices = [
        idx
        for idx in range(1, pose.total_residue() + 1)
        if str(pdb_info.chain(idx)) == str(chain)
    ]
    if not chain_indices:
        raise ValueError(f"Chain {chain!r} not found in pose.")

    selected: list[int] = []
    for part in str(residues).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
        else:
            start = end = int(token)
        if start < 1 or end < start or end > len(chain_indices):
            raise ValueError(
                f"Invalid chain-relative residue range {token!r} for chain {chain!r} "
                f"with length {len(chain_indices)}."
            )
        selected.extend(chain_indices[start - 1 : end])
    return sorted(set(selected))


def _chain_region_selector(pose, chain: str, residues: str | None = None):
    selector_ns = pyrosetta.rosetta.core.select.residue_selector
    if residues is None or str(residues).strip() == "":
        return selector_ns.ChainSelector(chain)
    indices = _chain_relative_residue_indices(pose, chain, residues)
    if not indices:
        raise ValueError(f"No residues selected for chain {chain!r} range {residues!r}.")
    return selector_ns.ResidueIndexSelector(",".join(str(i) for i in indices))


def _chain_group_region_selector(
    pose,
    chains: Sequence[str],
    residues: str | None = None,
):
    chain_ids = [str(chain) for chain in chains]
    if len(chain_ids) == 1:
        return _chain_region_selector(pose, chain_ids[0], residues)

    if residues is None or str(residues).strip() == "":
        wanted = set(chain_ids)
        pdb_info = pose.pdb_info()
        indices = [
            idx
            for idx in range(1, pose.total_residue() + 1)
            if str(pdb_info.chain(idx)) in wanted
        ]
    else:
        indices = sorted(
            {
                idx
                for chain in chain_ids
                for idx in _chain_relative_residue_indices(pose, chain, residues)
            }
        )
    if not indices:
        raise ValueError(
            f"No residues selected for chain group {chain_ids!r} range {residues!r}."
        )
    selector_ns = pyrosetta.rosetta.core.select.residue_selector
    return selector_ns.ResidueIndexSelector(",".join(str(i) for i in indices))


def _compute_interface_buried_unsatisfied_hbonds(
    pose,
    binder_chains: Sequence[str],
    target_chains: Sequence[str],
    *,
    separation_distance: float = 1000.0,
    ignore_surface_res: bool = False,
) -> float:
    _require_pyrosetta()
    binder_chain_ids = {str(chain) for chain in binder_chains}
    target_chain_ids = {str(chain) for chain in target_chains}
    if not binder_chain_ids or not target_chain_ids:
        raise ValueError("Interface BUNS requires non-empty binder and target groups.")
    if binder_chain_ids & target_chain_ids:
        raise ValueError("Interface BUNS binder and target groups must be disjoint.")

    pose_chain_ids = {
        str(pose.pdb_info().chain(idx))
        for idx in range(1, pose.total_residue() + 1)
    }
    selected_chain_ids = binder_chain_ids | target_chain_ids
    if pose_chain_ids != selected_chain_ids:
        raise ValueError(
            "Interface BUNS chain groups must cover the full pose; "
            f"pose={sorted(pose_chain_ids)!r} selected={sorted(selected_chain_ids)!r}."
        )

    xml = (
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" '
        f'ignore_surface_res="{str(bool(ignore_surface_res)).lower()}" '
        'use_ddG_style="false" dalphaball_sasa="1" probe_radius="1.1" '
        'burial_cutoff_apo="0.2" confidence="0" />'
    )
    buns_filter = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.static_get_filter(
        xml
    )
    bound_buns = float(buns_filter.report_sm(pose))

    separated_pose = pose.clone()
    atom_id_type = pyrosetta.rosetta.core.id.AtomID
    translation = pyrosetta.rosetta.numeric.xyzVector_double_t(
        float(separation_distance), 0.0, 0.0
    )
    for residue_index in range(1, separated_pose.total_residue() + 1):
        if str(separated_pose.pdb_info().chain(residue_index)) not in binder_chain_ids:
            continue
        residue = separated_pose.residue(residue_index)
        for atom_index in range(1, residue.natoms() + 1):
            atom_id = atom_id_type(atom_index, residue_index)
            separated_pose.set_xyz(
                atom_id,
                separated_pose.xyz(atom_id) + translation,
            )
    separated_pose.update_residue_neighbors()
    unbound_buns = float(buns_filter.report_sm(separated_pose))
    return max(0.0, bound_buns - unbound_buns)


def normalize_contact_molecular_surface(
    raw_value: float | None,
    *,
    midpoint: float = 400.0,
    midpoint_value: float = 0.8,
    lower_scale: float = 150.0,
    upper_scale: float = 75.0,
) -> float:
    if raw_value is None:
        return 0.0

    value = max(0.0, float(raw_value))
    midpoint = max(1e-6, float(midpoint))
    midpoint_value = min(max(float(midpoint_value), 0.0), 1.0)
    lower_scale = max(1e-6, float(lower_scale))
    upper_scale = max(1e-6, float(upper_scale))

    if value <= midpoint:
        return float(midpoint_value * math.exp((value - midpoint) / lower_scale))
    return float(
        midpoint_value
        + (1.0 - midpoint_value) * (1.0 - math.exp(-(value - midpoint) / upper_scale))
    )


@loss_registry.register(
    description="Interfacial predicted aligned error (iPAE) score",
    supports_oligomer_specific=True,
)
def ipae(
    eval_result,
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    chain_a: str | Sequence[str] | None = None,
    chain_b: str | Sequence[str] | None = None,
):
    """Calculate iPAE, averaging independently reduced selected chain pairs."""
    return compute_interface_pae(
        eval_result,
        symmetric="mean",
        reduction="mean",
        contact_method=contact_method,
        sc_cutoff=sc_cutoff,
        ca_cutoff=ca_cutoff,
        chain_a=chain_a,
        chain_b=chain_b,
    )


@loss_registry.register(
    name="ipae_min",
    description="Mean contact-pair iPAE using min(PAE[i,j], PAE[j,i]) for each cross-chain contact pair",
    optional_params=["contact_method", "sc_cutoff", "ca_cutoff", "chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def ipae_min(
    eval_result,
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    chain_a: str | Sequence[str] | None = None,
    chain_b: str | Sequence[str] | None = None,
):
    return compute_interface_pae(
        eval_result,
        symmetric="min",
        reduction="mean",
        contact_method=contact_method,
        sc_cutoff=sc_cutoff,
        ca_cutoff=ca_cutoff,
        chain_a=chain_a,
        chain_b=chain_b,
    )


@loss_registry.register(
    name="ipae_minimal",
    description="Minimal contact-pair iPAE over min(PAE[i,j], PAE[j,i]) values",
    optional_params=["contact_method", "sc_cutoff", "ca_cutoff", "chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def ipae_minimal(
    eval_result,
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    chain_a: str | Sequence[str] | None = None,
    chain_b: str | Sequence[str] | None = None,
):
    return compute_interface_pae(
        eval_result,
        symmetric="min",
        reduction="minimal",
        contact_method=contact_method,
        sc_cutoff=sc_cutoff,
        ca_cutoff=ca_cutoff,
        chain_a=chain_a,
        chain_b=chain_b,
    )


@loss_registry.register(
    name="ipae_max",
    description="Mean contact-pair iPAE using max(PAE[i,j], PAE[j,i]) for each cross-chain contact pair",
    optional_params=["contact_method", "sc_cutoff", "ca_cutoff", "chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def ipae_max(
    eval_result,
    contact_method: str = "sc",
    sc_cutoff: float = 5.0,
    ca_cutoff: float = 8.0,
    chain_a: str | Sequence[str] | None = None,
    chain_b: str | Sequence[str] | None = None,
):
    return compute_interface_pae(
        eval_result,
        symmetric="max",
        reduction="mean",
        contact_method=contact_method,
        sc_cutoff=sc_cutoff,
        ca_cutoff=ca_cutoff,
        chain_a=chain_a,
        chain_b=chain_b,
    )


def find_interface_residues_by_ca(model_path, cutoff=8.0):
    parsed_str = parse_mmcif(model_path)
    coords = parsed_str["xyz"][:, 1]
    chains = [chain for chain, _ in parsed_str["pdb_idx"]]
    unique_chains = list(set(chains))
    interface_residues = set()
    for i in range(len(unique_chains)):
        for j in range(i + 1, len(unique_chains)):
            # Get coordinates for each chain
            chain1_mask = np.array(chains) == unique_chains[i]
            chain2_mask = np.array(chains) == unique_chains[j]

            # Create KDTrees for efficient distance calculation
            tree1 = cKDTree(coords[chain1_mask])
            tree2 = cKDTree(coords[chain2_mask])

            # Find pairs of points within 8A
            pairs = tree1.query_ball_tree(tree2, r=cutoff)

            # Get residue indices for interface residues
            chain1_res_indices = np.where(chain1_mask)[0]
            chain2_res_indices = np.where(chain2_mask)[0]

            for idx1, neighbors in enumerate(pairs):
                if neighbors:  # If there are any neighbors within cutoff
                    interface_residues.add(chain1_res_indices[idx1])
                    interface_residues.update(chain2_res_indices[neighbors])
    interface_residues = np.array(sorted(interface_residues))
    return interface_residues


def calculate_ddg_loss(complex_poses, protomer, normalize_factor=5.0):
    """Calculates the DDG loss for mutations using PyRosetta.

    Args:
        complex_poses: Dictionary of PyRosetta poses for each oligomer
        protomer: Protomer object containing mutation information
        normalize_factor: Factor to normalize DDG values (default: 5.0)

    Returns:
        Dictionary mapping oligomer names to normalized DDG scores
    """
    _require_pyrosetta()
    # Create scoring function
    scorefxn = pyrosetta.create_score_function("ref2015")

    ddg_scores = {}

    for oligo_name, pose in complex_poses.poses.items():
        if len(oligo_name) < 2:
            continue

        # Get initial score
        initial_energy = scorefxn(pose)

        # Make a copy of the pose for mutation
        mutant_pose = pose.clone()

        # Track total DDG for this oligomer
        total_ddg = 0.0
        mutation_count = 0

        # Calculate DDG for each chain in oligomer
        for chain_idx, chain in enumerate(oligo_name):
            if chain in protomer.mutations:
                chain_offset = sum(
                    len(protomer.sequences[c]) for c in oligo_name[:chain_idx]
                )

                # Calculate DDG for each mutation in this chain
                for mutation in protomer.mutations[chain]:
                    res_id = chain_offset + mutation.position + 1
                    wt_aa = mutation.original
                    mut_aa = mutation.mutated

                    # Create mutant
                    mutator = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                        res_id,
                        pyrosetta.rosetta.core.chemical.aa_from_oneletter_code(mut_aa),
                    )
                    mutator.apply(mutant_pose)

                    # Minimize around mutation site
                    movemap = pyrosetta.MoveMap()
                    movemap.set_bb(False)
                    movemap.set_chi(False)
                    movemap.set_jump(False)

                    for i in range(
                        max(1, res_id - 5), min(mutant_pose.total_residue(), res_id + 6)
                    ):
                        movemap.set_chi(i, True)

                    minmover = (
                        pyrosetta.rosetta.protocols.minimization_packing.MinMover(
                            movemap, scorefxn, "lbfgs_armijo_nonmonotone", 0.001, True
                        )
                    )
                    minmover.apply(mutant_pose)

                    # Calculate DDG
                    mutant_energy = scorefxn(mutant_pose)
                    ddg = mutant_energy - initial_energy
                    total_ddg += ddg
                    mutation_count += 1

                    # Reset pose for next mutation
                    mutant_pose = pose.clone()

        # Calculate average normalized DDG for this oligomer
        if mutation_count > 0:
            avg_ddg = total_ddg / mutation_count
            ddg_scores[oligo_name] = avg_ddg / normalize_factor
        else:
            ddg_scores[oligo_name] = 0.0

    return ddg_scores


@loss_registry.register(
    name="protomer_shape_complementarity",
    description="Shape complementarity penalty between two protomers",
    optional_params=[
        "protomer_a",
        "protomer_b",
        # Back-compat alias used by some configs.
        "included_protomers",
        "target_sc",
        "saturation_constant",
        "return_raw",
    ],
    supports_oligomer_specific=True,
)
def protomer_shape_complementarity(
    eval_result,
    protomer_a: str | None = None,
    protomer_b: str | None = None,
    included_protomers: str | None = None,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Shape complementarity loss requires complex poses.")

    if protomer_a is None and included_protomers:
        # `included_protomers` may contain comma-separated protomer names; use the first as protomer_a.
        protomer_a = str(included_protomers).split(",")[0].strip() or None

    scores = {}
    for oligo_name, pose in complex_poses.poses.items():
        chain_a, chain_b = _resolve_chain_pair(
            oligo_name, protomer_a, protomer_b
        )
        if chain_a is None or chain_b is None or chain_a == chain_b:
            scores[oligo_name] = 0.0
            continue

        sc_value = _compute_shape_complementarity_between_chains(pose, chain_a, chain_b)
        if sc_value is None or sc_value < 0.0:
            scores[oligo_name] = 0.0
            continue

        scores[oligo_name] = sc_value

    return scores


@loss_registry.register(
    name="ligand_shape_complementarity",
    description="Mean Rosetta shape complementarity between ligand chain(s) and protein chain(s)",
    optional_params=["ligand_targets"],
    supports_oligomer_specific=True,
)
def ligand_shape_complementarity(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Ligand shape-complementarity loss requires complex poses.")

    scores = {}
    for oligo_name, pose in complex_poses.poses.items():
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        protein_chains = _protein_chains_for_oligo(oligo_name)
        if not ligands or not protein_chains:
            scores[oligo_name] = 0.0
            continue

        values = []
        for _proto, ligand_chain in ligands:
            for protein_chain in protein_chains:
                value = _compute_shape_complementarity_between_chains(
                    pose, protein_chain, ligand_chain
                )
                if value is not None and value >= 0.0:
                    values.append(float(value))
        scores[oligo_name] = float(np.mean(values)) if values else 0.0
    return scores


@loss_registry.register(
    name="contact_molecular_surface",
    description="Rosetta contact molecular surface between two protomer/chain groups",
    optional_params=[
        "protomer_a",
        "protomer_b",
        "protomer_a_residues",
        "protomer_b_residues",
        "included_protomers",
        "distance_weight",
        "use_rosetta_radii",
    ],
    supports_oligomer_specific=True,
)
def contact_molecular_surface(
    eval_result,
    protomer_a: str | Sequence[str] | None = None,
    protomer_b: str | Sequence[str] | None = None,
    protomer_a_residues: str | None = None,
    protomer_b_residues: str | None = None,
    included_protomers: str | None = None,
    distance_weight: float = 0.5,
    use_rosetta_radii: bool = True,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    pose_map = validate_load_poses(
        pose_map,
        getattr(eval_result, "complex_paths", None),
    )

    if protomer_a is None and included_protomers:
        protomer_a = str(included_protomers).split(",")[0].strip() or None

    scores = {}
    for oligo_name, pose in pose_map.items():
        chains_a, chains_b = _resolve_contact_chain_groups(
            oligo_name, protomer_a, protomer_b
        )
        if not chains_a or not chains_b:
            scores[oligo_name] = 0.0
            continue

        cms_value = _compute_contact_molecular_surface_between_chains(
            pose,
            chains_a,
            chains_b,
            distance_weight=distance_weight,
            use_rosetta_radii=use_rosetta_radii,
            protomer_a_residues=protomer_a_residues,
            protomer_b_residues=protomer_b_residues,
        )
        scores[oligo_name] = max(0.0, float(cms_value)) if cms_value is not None else 0.0

    return scores


@loss_registry.register(
    name="contact_molecular_surface_norm",
    description="Normalized Rosetta contact molecular surface score with cms=400 mapped to 0.8",
    optional_params=[
        "protomer_a",
        "protomer_b",
        "protomer_a_residues",
        "protomer_b_residues",
        "included_protomers",
        "distance_weight",
        "use_rosetta_radii",
        "midpoint",
        "midpoint_value",
        "lower_scale",
        "upper_scale",
    ],
    supports_oligomer_specific=True,
)
def contact_molecular_surface_norm(
    eval_result,
    protomer_a: str | Sequence[str] | None = None,
    protomer_b: str | Sequence[str] | None = None,
    protomer_a_residues: str | None = None,
    protomer_b_residues: str | None = None,
    included_protomers: str | None = None,
    distance_weight: float = 0.5,
    use_rosetta_radii: bool = True,
    midpoint: float = 400.0,
    midpoint_value: float = 0.8,
    lower_scale: float = 150.0,
    upper_scale: float = 75.0,
):
    raw_scores = contact_molecular_surface(
        eval_result,
        protomer_a=protomer_a,
        protomer_b=protomer_b,
        protomer_a_residues=protomer_a_residues,
        protomer_b_residues=protomer_b_residues,
        included_protomers=included_protomers,
        distance_weight=distance_weight,
        use_rosetta_radii=use_rosetta_radii,
    )
    return {
        oligo_name: normalize_contact_molecular_surface(
            raw_value,
            midpoint=midpoint,
            midpoint_value=midpoint_value,
            lower_scale=lower_scale,
            upper_scale=upper_scale,
        )
        for oligo_name, raw_value in raw_scores.items()
    }


@loss_registry.register(
    name="interface_buried_unsatisfied_hbonds",
    description=(
        "Interface-induced buried unsatisfied H-bond count between two "
        "protomer/chain groups"
    ),
    optional_params=[
        "protomer_a",
        "protomer_b",
        "included_protomers",
        "separation_distance",
        "ignore_surface_res",
    ],
    supports_oligomer_specific=True,
)
def interface_buried_unsatisfied_hbonds(
    eval_result,
    protomer_a: str | Sequence[str] | None = None,
    protomer_b: str | Sequence[str] | None = None,
    included_protomers: str | None = None,
    separation_distance: float = 1000.0,
    ignore_surface_res: bool = False,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    pose_map = validate_load_poses(
        pose_map,
        getattr(eval_result, "complex_paths", None),
    )
    if protomer_a is None and included_protomers:
        protomer_a = str(included_protomers).split(",")[0].strip() or None

    scores = {}
    for oligo_name, pose in pose_map.items():
        chains_a, chains_b = _resolve_contact_chain_groups(
            oligo_name, protomer_a, protomer_b
        )
        scores[oligo_name] = _compute_interface_buried_unsatisfied_hbonds(
            pose,
            chains_a,
            chains_b,
            separation_distance=separation_distance,
            ignore_surface_res=ignore_surface_res,
        )
    return scores


def _resolve_contact_chain_group(
    oligo_name: str,
    selectors: str | Sequence[str] | None,
    *,
    default_chains: Sequence[str],
) -> list[str]:
    selector_tokens = _split_chain_selectors(selectors, default="")
    if not selector_tokens:
        return list(default_chains)

    resolved: list[str] = []
    for selector in selector_tokens:
        chain_ids = resolve_selector_chain_ids(oligo_name, selector)
        if not chain_ids:
            raise ValueError(
                f"Contact molecular surface selector {selector!r} does not match "
                f"oligo {oligo_name!r}."
            )
        for chain_id in chain_ids:
            if chain_id not in resolved:
                resolved.append(chain_id)
    return resolved


def _resolve_contact_chain_groups(
    oligo_name: str,
    protomer_a: str | Sequence[str] | None,
    protomer_b: str | Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    ordered_chains = chain_ids_for_oligo(oligo_name)
    if not ordered_chains:
        return [], []

    chains_a = _resolve_contact_chain_group(
        oligo_name,
        protomer_a,
        default_chains=ordered_chains[:1],
    )
    chains_b = _resolve_contact_chain_group(
        oligo_name,
        protomer_b,
        default_chains=[chain for chain in ordered_chains if chain not in chains_a],
    )
    overlap = set(chains_a) & set(chains_b)
    if overlap:
        raise ValueError(
            "Contact molecular surface chain groups must be disjoint; "
            f"overlap={sorted(overlap)!r}."
        )
    return chains_a, chains_b


def _resolve_chain_pair(
    oligo_name: str,
    protomer_a: str | None,
    protomer_b: str | None,
):
    ordered_chains = chain_ids_for_oligo(oligo_name)
    if not ordered_chains:
        return None, None

    if protomer_a:
        candidates_a = resolve_selector_chain_ids(oligo_name, protomer_a)
    else:
        candidates_a = [ordered_chains[0]]
    if not candidates_a:
        return None, None

    if protomer_b:
        candidates_b = resolve_selector_chain_ids(oligo_name, protomer_b)
    else:
        candidates_b = ordered_chains
    if not candidates_b:
        return None, None

    for chain_a in candidates_a:
        for chain_b in candidates_b:
            if chain_b != chain_a:
                return chain_a, chain_b
    return None, None


@loss_registry.register(
    name="ligand_protein_hbond",
    description="Saturated count of ligand–protein hydrogen bonds",
    optional_params=["ligand_targets", "saturation_constant"],
    supports_oligomer_specific=True,
)
def ligand_protein_hbond(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
    saturation_constant: float = 4.0,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Ligand hydrogen-bond loss requires complex poses.")

    scores = {}
    for oligo_name, pose in complex_poses.poses.items():
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            continue
        ligand_chains = [chain for _, chain in ligands]
        ligand_res_indices = _residue_indices_for_chains(pose, ligand_chains)
        if not ligand_res_indices:
            scores[oligo_name] = 0.0
            continue

        ligand_residue_set = set(ligand_res_indices)
        pose.update_residue_neighbors()
        hb_set = rosetta_hbonds.HBondSet()
        rosetta_hbonds.fill_hbond_set(pose, False, hb_set)

        interaction_count = 0.0
        for idx in range(1, hb_set.nhbonds() + 1):
            hbond = hb_set.hbond(idx)
            don = hbond.don_res()
            acc = hbond.acc_res()
            don_is_ligand = don in ligand_residue_set
            acc_is_ligand = acc in ligand_residue_set
            if don_is_ligand == acc_is_ligand:
                continue
            partner = acc if don_is_ligand else don
            if pose.residue(partner).is_protein():
                interaction_count += 1.0

        averaged = interaction_count / max(1, len(ligand_chains))
        scores[oligo_name] = _saturating_fraction(averaged, saturation_constant)
    return scores


def compute_hotspot_delta_sasa(
    eval_result, select_expression, normalized_by=20, sidechain_only=True
):
    """
    Calculate the change in solvent accessible surface area (SASA) for hotspot residues in a protein complex.

    Args:
        eval_result: Evaluation result object containing complex poses.
        chain2proto: Dictionary mapping chain IDs to protomer IDs.
        normalized_by: Factor to normalize the delta SASA values (default: 20).

    Returns:
        Dictionary mapping oligomer names to normalized delta SASA values.
    """
    oligo_names = list(getattr(eval_result, "names", []) or [])
    if not oligo_names:
        return {}

    complex_poses = eval_result.complex_poses

    hotspot_weights = _normalized_hotspot_selector_values(
        oligo_names,
        select_expression,
    )

    delta_sasa = {}

    for oligo_name, pose in complex_poses.poses.items():
        if oligo_name not in oligo_names:
            continue
        if len(oligo_name) < 2:
            # Single-chain: no binding event -> delta SASA defined as 0.
            delta_sasa[oligo_name] = 0.0
            continue
        weights = hotspot_weights.get(oligo_name)
        if weights is None:
            delta_sasa[oligo_name] = 0.0
            continue

        sasa_bound, sasa_unbound = _bound_and_unbound_residue_sasa(
            pose,
            sidechain_only=sidechain_only,
        )
        delta_sasa_pose = sasa_unbound - sasa_bound
        # where the hotspot index are used, whether the ligand is present or not?
        hotspot_dsasa = delta_sasa_pose * weights
        delta_sasa[oligo_name] = float(np.sum(weights * hotspot_dsasa))

    # Normalize delta SASA values
    delta_sasa = {k: float(v) / normalized_by for k, v in delta_sasa.items()}

    return delta_sasa


@loss_registry.register(
    required_params=["select_expression"],
    optional_params=["normalize_factor"],
    description="Hotspot delta SASA calculation",
)
def hotspot_dsasa(eval_result, select_expression, normalize_factor: int = 20):
    """Calculate hotspot delta SASA."""
    _require_pyrosetta()

    return compute_hotspot_delta_sasa(eval_result, select_expression)


def _hotspot_selector_values(
    oligo_names: Sequence[str],
    select_expression: str,
) -> dict[str, np.ndarray | None]:
    if not oligo_names:
        return {}
    protomers = global_config.protomers
    selectors: dict[str, np.ndarray | None] = {}
    for oligo_name in oligo_names:
        selector_parts = []
        for proto in oligo_name:
            protomer = protomers.get(proto)
            if protomer is None:
                continue
            selector_parts.append(
                np.asarray(
                    protomer.select_residues_by_alias(select_expression),
                    dtype=float,
                )
            )
        if not selector_parts:
            selectors[oligo_name] = None
            continue
        selector = np.concatenate(selector_parts)
        selectors[oligo_name] = selector if np.any(selector > 0.0) else None
    return selectors


def _normalized_hotspot_selector_values(
    oligo_names: Sequence[str],
    select_expression: str,
) -> dict[str, np.ndarray | None]:
    selectors = _hotspot_selector_values(oligo_names, select_expression)
    normalized = {}
    for oligo_name, selector in selectors.items():
        if selector is None:
            normalized[oligo_name] = None
            continue
        total = float(np.sum(selector))
        normalized[oligo_name] = None if total <= 0.0 else selector / total
    return normalized


def _per_residue_sasa_calculator(sidechain_only: bool):
    if sidechain_only:
        from pyrosetta.rosetta.core.scoring.sasa import per_res_sc_sasa

        return per_res_sc_sasa

    sasa_metric = PerResidueSasaMetric()

    def calculate_pose_sasa(pose):
        per_res_sasa_dict = sasa_metric.calculate(pose)
        result = np.zeros(len(per_res_sasa_dict))
        for i, val in per_res_sasa_dict.items():
            result[i - 1] = val
        return result

    return calculate_pose_sasa


def _bound_and_unbound_residue_sasa(pose, *, sidechain_only: bool) -> tuple[np.ndarray, np.ndarray]:
    sasa_calculator = _per_residue_sasa_calculator(sidechain_only)
    sasa_bound = np.asarray(sasa_calculator(pose), dtype=float)
    sasa_unbound_parts = []
    for chain_pose in pose.split_by_chain():
        chain_sasa = np.asarray(sasa_calculator(chain_pose), dtype=float)
        sasa_unbound_parts.append(chain_sasa)
    if sasa_unbound_parts:
        sasa_unbound = np.concatenate(sasa_unbound_parts)
    else:
        sasa_unbound = np.zeros_like(sasa_bound)
    if sasa_bound.shape != sasa_unbound.shape:
        raise ValueError(
            "Bound/unbound SASA vectors do not align for hotspot exposure scoring."
        )
    return sasa_bound, sasa_unbound


def _selected_bound_and_unbound_exposed_sasa(
    pose,
    selector,
    *,
    sidechain_only: bool,
) -> tuple[float, float]:
    selector_array = np.asarray(selector, dtype=float)
    sasa_bound, sasa_unbound = _bound_and_unbound_residue_sasa(
        pose,
        sidechain_only=sidechain_only,
    )
    if selector_array.shape != sasa_bound.shape:
        raise ValueError(
            f"Hotspot selector length {selector_array.shape} does not match SASA vector {sasa_bound.shape}."
        )
    selected = np.clip(selector_array, 0.0, None)
    bound_exposed = float(np.sum(sasa_bound * selected))
    unbound_exposed = float(np.sum(sasa_unbound * selected))
    return bound_exposed, unbound_exposed


@loss_registry.register(
    name="hotspot_buried_ratio",
    required_params=["select_expression"],
    optional_params=["sidechain_only"],
    description="Selected hotspot buried ratio defined as 1 - bound_exposed_sasa / unbound_exposed_sasa",
)
def hotspot_buried_ratio(
    eval_result,
    select_expression,
    sidechain_only: bool = True,
):
    """Return selected hotspot buried ratio defined as 1 - bound/unbound exposure."""
    _require_pyrosetta()

    selectors = _hotspot_selector_values(eval_result.names, select_expression)
    buried_ratios = {}
    for oligo_name, pose in eval_result.complex_poses.poses.items():
        selector = selectors.get(oligo_name)
        if selector is None:
            buried_ratios[oligo_name] = 0.0
            continue
        if len(oligo_name) < 2:
            buried_ratios[oligo_name] = 0.0
            continue
        bound_exposed, unbound_exposed = _selected_bound_and_unbound_exposed_sasa(
            pose,
            selector,
            sidechain_only=sidechain_only,
        )
        if unbound_exposed <= 0.0:
            buried_ratios[oligo_name] = 0.0
            continue
        buried_ratios[oligo_name] = float(1.0 - (bound_exposed / unbound_exposed))
    return buried_ratios


@loss_registry.register(
    required_params=["buried_protomer"],
    description="Calculate buried contribution difference for a specific protomer",
    supports_oligomer_specific=True,
)
def buried_contribution_diff(eval_result, buried_protomer: str):
    """Calculate buried contribution difference for specified protomer."""
    _require_pyrosetta()
    buried_chain = {}
    for oligo_name in eval_result.names:
        if len(oligo_name) != 3:
            continue
        chain_id = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="buried_contribution_diff.buried_protomer",
        )
        if chain_id is None:
            continue
        buried_chain[oligo_name] = chain_id
    return calculate_buried_contribution_diff(
        eval_result.complex_poses.poses, buried_chain
    )


def calculate_buried_contribution_diff(
    complex_poses=None, buried_chain=None, structure_files=None
):
    result = {}
    complex_poses = validate_load_poses(complex_poses, structure_files)
    for oligo_name, pose in dict(complex_poses).items():

        chain_ids = set(pose.pdb_info().chain(i) for i in range(1, pose.size() + 1))

        chainA, chainB, *other = [c for c in chain_ids if c != buried_chain[oligo_name]]
        if other:
            raise NotImplementedError("Only support triplet oligomer for now")

        chainA_contribution = get_buried_contribution(
            pose, chainA, buried_chain[oligo_name]
        )
        chainB_contribution = get_buried_contribution(
            pose, chainB, buried_chain[oligo_name]
        )
        contribution_diff_fraction = 1 - abs(
            chainA_contribution - chainB_contribution
        ) / (chainA_contribution + chainB_contribution)
        result[oligo_name] = contribution_diff_fraction
    return result


@lru_cache(maxsize=10)
def get_chain_sasa(pose, chain_id):
    """Calculate the SASA for a specific chain in the pose."""
    original_sasa = sasa_calculator(pose)

    according_chain = original_sasa[
        [
            i
            for i in range(pose.total_residue())
            if pose.pdb_info().chain(i + 1) in chain_id
        ]
    ]
    return according_chain


def get_buried_contribution(pose, chain_id, buried_chain: str):
    """Calculate the contribution of a buried chain to the total SASA."""
    according_buried_chain = get_chain_sasa(pose, buried_chain)

    # remove chain `chain_id` from pose
    new_pose = remove_chain_from_pose(pose, chain_id)

    according_unbound_chain = get_chain_sasa(new_pose, buried_chain)
    return np.sum(according_unbound_chain) - np.sum(according_buried_chain)


@loss_registry.register(
    required_params=["buried_protomer"],
    description="Calculate holes difference for a specific protomer",
    supports_oligomer_specific=True,
)
def chain_holes_diff(eval_result, buried_protomer: str):
    """Calculate holes difference for specified protomer."""
    _require_pyrosetta()
    buried_chain = {}
    for oligo_name in eval_result.names:
        if len(oligo_name) != 3:
            continue
        chain_id = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="chain_holes_diff.buried_protomer",
        )
        if chain_id is None:
            continue
        buried_chain[oligo_name] = chain_id
    return calculate_holes_diff(eval_result.complex_poses.poses, buried_chain)


def calculate_holes_diff(complex_poses=None, buried_chain=None, structure_files=None):
    result = {}
    holes_filter = HolesFilter()

    complex_poses = validate_load_poses(complex_poses, structure_files)
    for oligo_name, pose in dict(complex_poses).items():

        new_pose = remove_chain_from_pose(pose, buried_chain[oligo_name])
        wt_buried_chain = holes_filter.report_sm(pose)
        wo_buried_chain = holes_filter.report_sm(new_pose)
        result[oligo_name] = wo_buried_chain - wt_buried_chain
    return result


@loss_registry.register(
    name="ligand_delta_sasa_fraction",
    description="Fraction of ligand SASA buried upon binding",
    optional_params=["ligand_targets", "epsilon"],
    supports_oligomer_specific=True,
)
def ligand_delta_sasa_fraction(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
    epsilon: float = 1e-3,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Ligand SASA loss requires complex poses.")

    scores = {}
    for oligo_name, pose in complex_poses.poses.items():
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            continue
        per_res_sasa = sasa_calculator(pose)
        fractions = []
        for _, chain_id in ligands:
            ligand_res = _residue_indices_for_chains(pose, [chain_id])
            if not ligand_res:
                continue
            bound_sasa = per_res_sasa[[idx - 1 for idx in ligand_res]].sum()
            ligand_pose = _isolate_chains(pose, [chain_id])
            if ligand_pose is None or ligand_pose.total_residue() == 0:
                continue
            ligand_unbound_sasa = sasa_calculator(ligand_pose).sum()
            buried = max(0.0, ligand_unbound_sasa - bound_sasa)
            denom = max(ligand_unbound_sasa, epsilon)
            fractions.append(min(1.0, buried / denom))
        if fractions:
            scores[oligo_name] = float(np.mean(fractions))
        else:
            scores[oligo_name] = 0.0
    return scores


@loss_registry.register(
    name="ligand_pocket_occupancy",
    required_params=["pocket_chain"],
    optional_params=[
        "pocket_residues",
        "pocket_residues_by_chain",
        "ligand_targets",
        "pocket_padding",
    ],
    description="Fraction of ligand heavy atoms inside the selected pocket residue box",
    supports_oligomer_specific=True,
)
def ligand_pocket_occupancy(
    eval_result,
    pocket_chain: str | Sequence[str] | None,
    pocket_residues: int | str | Sequence[int | str] | Mapping[str, int | str | Sequence[int | str]] | None = None,
    pocket_residues_by_chain: Mapping[str, int | str | Sequence[int | str]] | None = None,
    ligand_targets: str | Sequence[str] | None = None,
    pocket_padding: float = 1.0,
):
    raw_pocket_specs = _normalize_chain_residue_specs(
        pocket_chain=pocket_chain,
        pocket_residues=pocket_residues,
        pocket_residues_by_chain=pocket_residues_by_chain,
    )
    if not raw_pocket_specs:
        raise ValueError("ligand_pocket_occupancy requires nonempty pocket residues.")

    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names and pose_map is not None:
        names = list(pose_map.keys())
    if not names:
        raise ValueError("Ligand pocket occupancy loss requires complex paths or complex poses.")

    for oligo_name in names:
        pocket_chain_number_specs = _resolve_pocket_chain_number_specs(
            oligo_name, raw_pocket_specs
        )
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                f"ligand_pocket_occupancy found no ligand chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]

        if isinstance(complex_paths, dict) and oligo_name in complex_paths:
            scores[oligo_name] = _ligand_pocket_occupancy_from_structure_file(
                complex_paths[oligo_name],
                pocket_chain_number_specs=pocket_chain_number_specs,
                ligand_chain_ids=ligand_chain_ids,
                pocket_padding=pocket_padding,
            )
            continue

        if pose_map is None or oligo_name not in pose_map:
            raise ValueError(
                f"ligand_pocket_occupancy has no structure file or pose for oligo {oligo_name!r}."
            )
        pose = pose_map[oligo_name]
        pocket_residue_indices = _residue_indices_for_chain_number_specs(
            pose,
            pocket_chain_number_specs,
        )
        if not pocket_residue_indices:
            raise ValueError(
                "ligand_pocket_occupancy selected no residues for "
                f"oligo={oligo_name!r}, pocket_chain={pocket_chain!r}, "
                f"pocket_residues={[(chain, sorted(nums)) for chain, nums in pocket_chain_number_specs]!r}."
            )
        pocket_coords = _require_nonempty_coords(
            _heavy_atom_coordinates(pose, pocket_residue_indices),
            "ligand_pocket_occupancy pocket",
        )

        ligand_residue_indices = _residue_indices_for_chains(
            pose, ligand_chain_ids
        )
        if not ligand_residue_indices:
            raise ValueError(
                f"ligand_pocket_occupancy found no ligand residues for oligo {oligo_name!r}."
            )
        ligand_coords = _require_nonempty_coords(
            _heavy_atom_coordinates(pose, ligand_residue_indices),
            "ligand_pocket_occupancy ligand",
        )
        scores[oligo_name] = _fraction_inside_pocket_box(
            ligand_coords,
            pocket_coords,
            pocket_padding,
        )
    return scores


def protein_interface_ca_box(
    coordinates_a: np.ndarray,
    coordinates_b: np.ndarray,
    *,
    interface_distance: float = 8.0,
    coverage_percent: float = 90.0,
    padding: float = 1.0,
    min_box_size: float = 5.0,
) -> dict[str, np.ndarray]:
    """PCA percentile box shared by CID docking and its CA90 endpoint.

    Row-vector world coordinates map to box coordinates as ``(xyz-origin) @ axes``.
    The box center is expressed in that local frame, never in world coordinates.
    """
    coordinates_a = np.asarray(coordinates_a, dtype=float)
    coordinates_b = np.asarray(coordinates_b, dtype=float)
    if not len(coordinates_a) or not len(coordinates_b):
        raise ValueError("Interface CA box requires CA atoms in both protein chains")
    distances = np.linalg.norm(coordinates_a[:, None] - coordinates_b[None, :], axis=-1)
    points = np.concatenate([
        coordinates_a[np.min(distances, axis=1) <= interface_distance],
        coordinates_b[np.min(distances, axis=0) <= interface_distance],
    ])
    if len(points) < 2:
        raise ValueError("No protein-interface CA atoms found within distance cutoff")
    origin = points.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov((points - origin).T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, axes = eigenvalues[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local_points = (points - origin) @ axes
    low = (100.0 - coverage_percent) / 2.0
    lower, upper = np.percentile(local_points, [low, 100.0 - low], axis=0)
    return {
        "origin": origin,
        "axes": axes,
        "center": (lower + upper) / 2.0,
        "size": np.maximum(upper - lower + 2.0 * padding, min_box_size),
        "lower": lower,
        "upper": upper,
        "eigenvalues": eigenvalues,
        "points": points,
    }


@loss_registry.register(
    name="ligand_protein_interface_ca90",
    optional_params=[
        "interface_chains", "ligand_targets", "interface_distance",
        "coverage_percent", "padding", "min_box_size",
    ],
    description="Ligand heavy-atom fraction inside the protein-interface C-alpha PCA percentile box",
    supports_oligomer_specific=True,
)
def ligand_protein_interface_ca90(
    eval_result,
    interface_chains: str | Sequence[str] | None = None,
    ligand_targets: str | Sequence[str] | None = None,
    interface_distance: float = 8.0,
    coverage_percent: float = 90.0,
    padding: float = 1.0,
    min_box_size: float = 5.0,
):
    scores = {}
    for name in eval_result.names:
        model = _structure_model_from_file(eval_result.complex_paths[name])
        ligands, _ = _resolve_ligand_chains(name, ligand_targets)
        ligand_chain_ids = [chain for _, chain in ligands]
        chains = _resolve_dynamic_interface_chain_ids(
            model=model, oligo_name=name, interface_chains=interface_chains,
            ligand_chain_ids=ligand_chain_ids, context="ligand_protein_interface_ca90",
        )
        if len(chains) != 2:
            raise ValueError("ligand_protein_interface_ca90 requires exactly two protein chains")
        ca_coordinates = [
            np.asarray([residue["CA"].coord for residue in model[chain]
                        if residue.id[0] == " " and "CA" in residue], dtype=float)
            for chain in chains
        ]
        box = protein_interface_ca_box(
            *ca_coordinates, interface_distance=interface_distance,
            coverage_percent=coverage_percent, padding=padding, min_box_size=min_box_size,
        )
        ligand_coordinates = _heavy_atom_coordinates_for_model_chains(model, ligand_chain_ids)
        if not len(ligand_coordinates):
            raise ValueError(f"ligand_protein_interface_ca90 found no ligand heavy atoms in {name}")
        local = (ligand_coordinates - box["origin"]) @ box["axes"]
        inside = np.all(np.abs(local - box["center"]) <= box["size"] / 2.0, axis=1)
        scores[name] = float(np.mean(inside))
    return scores


@loss_registry.register(
    name="ligand_protein_interface_occupancy",
    optional_params=[
        "interface_chains",
        "ligand_targets",
        "interface_distance",
        "interface_padding",
        "min_box_size",
    ],
    description=(
        "Fraction of ligand heavy atoms inside the AutoDock-style box built from "
        "the dynamically computed protein-protein interface residues"
    ),
    supports_oligomer_specific=True,
)
def ligand_protein_interface_occupancy(
    eval_result,
    interface_chains: str | Sequence[str] | None = None,
    ligand_targets: str | Sequence[str] | None = None,
    interface_distance: float = 8.0,
    interface_padding: float = 1.0,
    min_box_size: float = AUTODOCK_MIN_BOX_SIZE,
):
    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names:
        raise ValueError("ligand_protein_interface_occupancy requires complex paths.")

    for oligo_name in names:
        if not isinstance(complex_paths, dict) or oligo_name not in complex_paths:
            raise ValueError(
                "ligand_protein_interface_occupancy has no structure file for "
                f"oligo {oligo_name!r}."
            )
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                f"ligand_protein_interface_occupancy found no ligand chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]
        current_model = _structure_model_from_file(complex_paths[oligo_name])
        resolved_interface_chains = _resolve_dynamic_interface_chain_ids(
            model=current_model,
            oligo_name=oligo_name,
            interface_chains=interface_chains,
            ligand_chain_ids=ligand_chain_ids,
            context="ligand_protein_interface_occupancy",
        )
        interface_residues = _interface_residues_from_structure_model(
            model=current_model,
            chain_ids=resolved_interface_chains,
            interface_distance=float(interface_distance),
            context="ligand_protein_interface_occupancy",
        )
        interface_coords = _heavy_atom_coordinates_for_residues_biopdb(
            interface_residues,
            context="ligand_protein_interface_occupancy interface",
        )
        ligand_coords = _heavy_atom_coordinates_for_model_chains(
            current_model,
            ligand_chain_ids,
        )
        scores[oligo_name] = _fraction_inside_autodock_box(
            ligand_coords,
            interface_coords,
            box_padding=float(interface_padding),
            min_box_size=float(min_box_size),
        )
    return scores


@loss_registry.register(
    name="ligand_protein_interface_cb_alpha_shape_occupancy",
    optional_params=[
        "interface_chains",
        "ligand_targets",
        "interface_distance",
        "alpha_radius",
        "alpha_radius_scale",
        "alpha_min_radius",
        "alpha_max_radius",
        "alpha_tolerance",
    ],
    description=(
        "Fraction of ligand heavy atoms inside the alpha shape built from C-beta "
        "or glycine C-alpha points of dynamically computed protein-protein "
        "interface residues"
    ),
    supports_oligomer_specific=True,
)
def ligand_protein_interface_cb_alpha_shape_occupancy(
    eval_result,
    interface_chains: str | Sequence[str] | None = None,
    ligand_targets: str | Sequence[str] | None = None,
    interface_distance: float = 8.0,
    alpha_radius: float | None = None,
    alpha_radius_scale: float = 1.8,
    alpha_min_radius: float = 3.0,
    alpha_max_radius: float = 8.0,
    alpha_tolerance: float = 1e-8,
):
    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names:
        raise ValueError(
            "ligand_protein_interface_cb_alpha_shape_occupancy requires complex paths."
        )

    for oligo_name in names:
        if not isinstance(complex_paths, dict) or oligo_name not in complex_paths:
            raise ValueError(
                "ligand_protein_interface_cb_alpha_shape_occupancy has no structure "
                f"file for oligo {oligo_name!r}."
            )
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                "ligand_protein_interface_cb_alpha_shape_occupancy found no ligand "
                f"chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]
        current_model = _structure_model_from_file(complex_paths[oligo_name])
        resolved_interface_chains = _resolve_dynamic_interface_chain_ids(
            model=current_model,
            oligo_name=oligo_name,
            interface_chains=interface_chains,
            ligand_chain_ids=ligand_chain_ids,
            context="ligand_protein_interface_cb_alpha_shape_occupancy",
        )
        interface_residues = _interface_residues_from_structure_model(
            model=current_model,
            chain_ids=resolved_interface_chains,
            interface_distance=float(interface_distance),
            context="ligand_protein_interface_cb_alpha_shape_occupancy",
        )
        interface_cb_coords = _cb_or_ca_coordinates_for_residues_biopdb(
            interface_residues,
            context="ligand_protein_interface_cb_alpha_shape_occupancy interface",
        )
        ligand_coords = _heavy_atom_coordinates_for_model_chains(
            current_model,
            ligand_chain_ids,
        )
        scores[oligo_name] = _fraction_inside_cb_alpha_shape(
            ligand_coords,
            interface_cb_coords,
            alpha_radius=alpha_radius,
            alpha_radius_scale=alpha_radius_scale,
            alpha_min_radius=alpha_min_radius,
            alpha_max_radius=alpha_max_radius,
            tolerance=alpha_tolerance,
        )
    return scores


@loss_registry.register(
    name="ligand_aligned_reference_pose_distance",
    required_params=["reference_structure_path", "reference_ligand_path"],
    optional_params=[
        "reference_structure_chain",
        "current_structure_chain",
        "ligand_targets",
        "min_ca_atoms",
        "allow_length_mismatch",
    ],
    description=(
        "Centroid distance between the current ligand and a reference ligand pose "
        "after aligning current protein chains to the reference structure"
    ),
    supports_oligomer_specific=True,
)
def ligand_aligned_reference_pose_distance(
    eval_result,
    reference_structure_path: str,
    reference_ligand_path: str,
    reference_structure_chain: Any = None,
    current_structure_chain: Any = None,
    ligand_targets: str | Sequence[str] | None = None,
    min_ca_atoms: int = 20,
    allow_length_mismatch: bool = False,
):
    chain_pairs = _alignment_chain_pairs(
        reference_structure_chain,
        current_structure_chain,
    )
    reference_model = _structure_model_from_file(reference_structure_path)
    reference_ligand_coords = _all_heavy_atom_coordinates_from_structure_file(
        reference_ligand_path
    )

    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names and pose_map is not None:
        names = list(pose_map.keys())
    if not names:
        raise ValueError(
            "ligand_aligned_reference_pose_distance requires complex paths or complex poses."
        )

    for oligo_name in names:
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                f"ligand_aligned_reference_pose_distance found no ligand chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]

        if isinstance(complex_paths, dict) and oligo_name in complex_paths:
            scores[oligo_name] = _ligand_reference_pose_distance_from_structure_file(
                complex_paths[oligo_name],
                reference_model=reference_model,
                reference_ligand_coords=reference_ligand_coords,
                chain_pairs=chain_pairs,
                ligand_chain_ids=ligand_chain_ids,
                min_ca_atoms=min_ca_atoms,
                allow_length_mismatch=allow_length_mismatch,
            )
            continue

        if pose_map is None or oligo_name not in pose_map:
            raise ValueError(
                "ligand_aligned_reference_pose_distance has no structure file or "
                f"pose for oligo {oligo_name!r}."
            )
        scores[oligo_name] = _ligand_reference_pose_distance_from_pose(
            pose_map[oligo_name],
            oligo_name=oligo_name,
            reference_model=reference_model,
            reference_ligand_coords=reference_ligand_coords,
            chain_pairs=chain_pairs,
            ligand_chain_ids=ligand_chain_ids,
            min_ca_atoms=min_ca_atoms,
            allow_length_mismatch=allow_length_mismatch,
        )
    return scores


@loss_registry.register(
    name="ligand_aligned_reference_pocket_box_occupancy",
    required_params=["reference_structure_path"],
    optional_params=[
        "reference_structure_chain",
        "current_structure_chain",
        "reference_pocket_positions",
        "reference_pocket_positions_by_chain",
        "reference_box_padding",
        "binding_site_padding",
        "ligand_targets",
        "allow_length_mismatch",
        "min_box_size",
    ],
    description=(
        "Fraction of ligand heavy atoms inside the AutoDock aligned_reference_pocket "
        "box computed from the mapped current pocket residues"
    ),
    supports_oligomer_specific=True,
)
def ligand_aligned_reference_pocket_box_occupancy(
    eval_result,
    reference_structure_path: str,
    reference_structure_chain: Any = None,
    current_structure_chain: Any = None,
    reference_pocket_positions: int | str | Sequence[int | str] | None = None,
    reference_pocket_positions_by_chain: Mapping[str, int | str | Sequence[int | str]] | None = None,
    reference_box_padding: float | None = None,
    binding_site_padding: float | None = None,
    ligand_targets: str | Sequence[str] | None = None,
    allow_length_mismatch: bool = False,
    min_box_size: float = AUTODOCK_MIN_BOX_SIZE,
):
    box_padding = (
        float(reference_box_padding)
        if reference_box_padding is not None
        else float(binding_site_padding if binding_site_padding is not None else 1.0)
    )
    reference_model = _structure_model_from_file(reference_structure_path)
    pocket_specs = _reference_pocket_specs(
        reference_pocket_positions=reference_pocket_positions,
        reference_pocket_positions_by_chain=reference_pocket_positions_by_chain,
        reference_structure_chain=reference_structure_chain,
        current_structure_chain=current_structure_chain,
    )

    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names and pose_map is not None:
        names = list(pose_map.keys())
    if not names:
        raise ValueError(
            "ligand_aligned_reference_pocket_box_occupancy requires complex paths or complex poses."
        )

    for oligo_name in names:
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                f"ligand_aligned_reference_pocket_box_occupancy found no ligand chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]

        if isinstance(complex_paths, dict) and oligo_name in complex_paths:
            current_model = _structure_model_from_file(complex_paths[oligo_name])
            pocket_coords = _current_pocket_coords_from_structure_model(
                reference_model=reference_model,
                current_model=current_model,
                pocket_specs=pocket_specs,
                allow_length_mismatch=allow_length_mismatch,
            )
            ligand_coords = _heavy_atom_coordinates_for_model_chains(
                current_model,
                ligand_chain_ids,
            )
        else:
            if pose_map is None or oligo_name not in pose_map:
                raise ValueError(
                    "ligand_aligned_reference_pocket_box_occupancy has no structure "
                    f"file or pose for oligo {oligo_name!r}."
                )
            pose = pose_map[oligo_name]
            pocket_coords = _current_pocket_coords_from_pose(
                reference_model=reference_model,
                pose=pose,
                oligo_name=oligo_name,
                pocket_specs=pocket_specs,
                allow_length_mismatch=allow_length_mismatch,
            )
            ligand_residue_indices = _residue_indices_for_chains(pose, ligand_chain_ids)
            if not ligand_residue_indices:
                raise ValueError(
                    "ligand_aligned_reference_pocket_box_occupancy found no ligand "
                    f"residues for oligo {oligo_name!r}."
                )
            ligand_coords = _heavy_atom_coordinates(pose, ligand_residue_indices)
        scores[oligo_name] = _fraction_inside_autodock_box(
            ligand_coords,
            pocket_coords,
            box_padding=box_padding,
            min_box_size=min_box_size,
        )
    return scores


@loss_registry.register(
    name="ligand_aligned_reference_pocket_cb_alpha_shape_occupancy",
    required_params=["reference_structure_path"],
    optional_params=[
        "reference_structure_chain",
        "current_structure_chain",
        "reference_pocket_positions",
        "reference_pocket_positions_by_chain",
        "alpha_radius",
        "alpha_radius_scale",
        "alpha_min_radius",
        "alpha_max_radius",
        "alpha_tolerance",
        "ligand_targets",
        "allow_length_mismatch",
    ],
    description=(
        "Fraction of ligand heavy atoms inside the alpha shape built from mapped "
        "current target-pocket residue C-beta points; glycine uses C-alpha"
    ),
    supports_oligomer_specific=True,
)
def ligand_aligned_reference_pocket_cb_alpha_shape_occupancy(
    eval_result,
    reference_structure_path: str,
    reference_structure_chain: Any = None,
    current_structure_chain: Any = None,
    reference_pocket_positions: int | str | Sequence[int | str] | None = None,
    reference_pocket_positions_by_chain: Mapping[str, int | str | Sequence[int | str]] | None = None,
    alpha_radius: float | None = None,
    alpha_radius_scale: float = 1.8,
    alpha_min_radius: float = 3.0,
    alpha_max_radius: float = 8.0,
    alpha_tolerance: float = 1e-8,
    ligand_targets: str | Sequence[str] | None = None,
    allow_length_mismatch: bool = False,
):
    reference_model = _structure_model_from_file(reference_structure_path)
    pocket_specs = _reference_pocket_specs(
        reference_pocket_positions=reference_pocket_positions,
        reference_pocket_positions_by_chain=reference_pocket_positions_by_chain,
        reference_structure_chain=reference_structure_chain,
        current_structure_chain=current_structure_chain,
    )

    scores = {}
    names = list(getattr(eval_result, "names", []) or [])
    complex_paths = getattr(eval_result, "complex_paths", {}) or {}
    complex_poses = getattr(eval_result, "complex_poses", None)
    pose_map = getattr(complex_poses, "poses", None) if complex_poses is not None else None
    if not names and isinstance(complex_paths, dict):
        names = list(complex_paths.keys())
    if not names and pose_map is not None:
        names = list(pose_map.keys())
    if not names:
        raise ValueError(
            "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy requires "
            "complex paths or complex poses."
        )

    for oligo_name in names:
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            raise ValueError(
                "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy found "
                f"no ligand chains for oligo {oligo_name!r}."
            )
        ligand_chain_ids = [chain for _, chain in ligands]

        if isinstance(complex_paths, dict) and oligo_name in complex_paths:
            current_model = _structure_model_from_file(complex_paths[oligo_name])
            pocket_cb_coords = _current_pocket_cb_coords_from_structure_model(
                reference_model=reference_model,
                current_model=current_model,
                pocket_specs=pocket_specs,
                allow_length_mismatch=allow_length_mismatch,
            )
            ligand_coords = _heavy_atom_coordinates_for_model_chains(
                current_model,
                ligand_chain_ids,
            )
        else:
            if pose_map is None or oligo_name not in pose_map:
                raise ValueError(
                    "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy "
                    f"has no structure file or pose for oligo {oligo_name!r}."
                )
            pose = pose_map[oligo_name]
            pocket_cb_coords = _current_pocket_cb_coords_from_pose(
                reference_model=reference_model,
                pose=pose,
                oligo_name=oligo_name,
                pocket_specs=pocket_specs,
                allow_length_mismatch=allow_length_mismatch,
            )
            ligand_residue_indices = _residue_indices_for_chains(pose, ligand_chain_ids)
            if not ligand_residue_indices:
                raise ValueError(
                    "ligand_aligned_reference_pocket_cb_alpha_shape_occupancy found "
                    f"no ligand residues for oligo {oligo_name!r}."
                )
            ligand_coords = _heavy_atom_coordinates(pose, ligand_residue_indices)
        scores[oligo_name] = _fraction_inside_cb_alpha_shape(
            ligand_coords,
            pocket_cb_coords,
            alpha_radius=alpha_radius,
            alpha_radius_scale=alpha_radius_scale,
            alpha_min_radius=alpha_min_radius,
            alpha_max_radius=alpha_max_radius,
            tolerance=alpha_tolerance,
        )
    return scores


@loss_registry.register(
    name="ligand_protein_iptm",
    description="Mean iPTM between ligand and protein chains",
    optional_params=["ligand_targets"],
    supports_oligomer_specific=True,
)
def ligand_protein_iptm(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
):
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None)
    if not chain_pair_scores:
        return {name: 0.0 for name in getattr(eval_result, "names", [])}

    scores = {}
    for oligo_name in eval_result.names:
        if oligo_name not in chain_pair_scores:
            continue
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            continue

        iptm_mats = chain_pair_scores[oligo_name].get("chain_pair_iptm")
        if not iptm_mats:
            continue
        averaged_matrix = np.mean(np.stack(iptm_mats), axis=0)
        chain_ids_ordered = chain_ids_for_oligo(oligo_name)[: int(averaged_matrix.shape[0])]
        chain_index = {cid: idx for idx, cid in enumerate(chain_ids_ordered)}
        ligand_indices = [chain_index.get(chain) for _, chain in ligands]
        chain2proto = chain_to_protomer(oligo_name)
        protein_indices = [
            chain_index[chain]
            for chain in chain_ids_ordered
            if chain in chain_index
            for proto in [chain2proto.get(chain)]
            if global_config.protomers.get(proto)
            and global_config.protomers[proto].type == ProtomerTypes.PROTEIN
        ]
        pair_values = []
        for lig_idx in ligand_indices:
            if lig_idx is None:
                continue
            for prot_idx in protein_indices:
                if prot_idx is None or prot_idx == lig_idx:
                    continue
                pair_values.append(
                    float(averaged_matrix[lig_idx, prot_idx])
                )
        if pair_values:
            scores[oligo_name] = float(np.mean(pair_values))
        else:
            scores[oligo_name] = 0.0
    return scores


@loss_registry.register(
    name="ligand_local_holes",
    description="Local Holes/packstat penalty in ligand environment",
    optional_params=["ligand_targets", "neighbor_distance", "saturation_constant"],
    supports_oligomer_specific=True,
)
def ligand_local_holes(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
    neighbor_distance: float = 8.0,
    saturation_constant: float = 10.0,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Ligand Holes loss requires complex poses.")

    scores = {}
    holes_filter = HolesFilter()
    for oligo_name, pose in complex_poses.poses.items():
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            continue

        local_scores = []
        for _, chain_id in ligands:
            sub_pose = _build_ligand_environment_pose(pose, chain_id, neighbor_distance)
            if sub_pose is None or sub_pose.total_residue() < 2:
                continue
            raw_score = max(0.0, holes_filter.report_sm(sub_pose))
            local_scores.append(raw_score)

        if local_scores:
            avg_raw = float(np.mean(local_scores))
            scores[oligo_name] = _saturating_fraction(avg_raw, saturation_constant)
        else:
            scores[oligo_name] = 0.0
    return scores


@loss_registry.register(
    name="ligand_buried_unsat_hbonds",
    description="Buried unsatisfied polar groups in the ligand local environment",
    optional_params=["ligand_targets", "neighbor_distance", "saturation_constant"],
    supports_oligomer_specific=True,
)
def ligand_buried_unsat_hbonds(
    eval_result,
    ligand_targets: str | Sequence[str] | None = None,
    neighbor_distance: float = 8.0,
    saturation_constant: float = 6.0,
):
    _require_pyrosetta()
    complex_poses = getattr(eval_result, "complex_poses", None)
    if complex_poses is None or complex_poses.poses is None:
        raise ValueError("Ligand buried-unsat loss requires complex poses.")

    XmlObjects = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" '
        'ignore_surface_res="false" use_ddG_style="true" '
        'dalphaball_sasa="1" probe_radius="1.1" '
        'burial_cutoff_apo="0.2" confidence="0" />'
    )

    scores = {}
    for oligo_name, pose in complex_poses.poses.items():
        ligands, _ = _resolve_ligand_chains(oligo_name, ligand_targets)
        if not ligands:
            scores[oligo_name] = 0.0
            continue

        local_scores = []
        for _proto, chain_id in ligands:
            sub_pose = _build_ligand_environment_pose(pose, chain_id, neighbor_distance)
            if sub_pose is None or sub_pose.total_residue() < 2:
                continue
            try:
                raw_score = max(0.0, float(buns_filter.report_sm(sub_pose)))
            except RuntimeError as exc:  # pragma: no cover - PyRosetta runtime errors
                raise RuntimeError(
                    f"Ligand buried-unsat calculation failed for {oligo_name} "
                    f"chain {chain_id}: {exc}"
                ) from exc
            local_scores.append(raw_score)

        if local_scores:
            avg_raw = float(np.mean(local_scores))
            scores[oligo_name] = _saturating_fraction(avg_raw, saturation_constant)
        else:
            scores[oligo_name] = 0.0
    return scores


def validate_load_poses(complex_poses, structure_files):

    if complex_poses is None and structure_files is not None:
        complex_poses = {}
        for oligo_name, complex_path in structure_files.items():
            pose = load_pose(complex_path)
            complex_poses[oligo_name] = pose
    if not complex_poses:
        raise ValueError("No complex poses or structure files provided.")
    return complex_poses


def _current_raw_model_extra_residue_params() -> tuple[tuple[str, float | None], ...]:
    try:
        raw_dir = pathlib.Path(global_config.out_dir) / "raw_model_results"
    except RuntimeError:
        return ()
    params = []
    for path in sorted(raw_dir.glob("*.params")):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        params.append((str(path), mtime))
    return tuple(params)


def load_pose(complex_path):
    path = pathlib.Path(str(complex_path))
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    extra_residue_params = _current_raw_model_extra_residue_params()
    return _load_pose_cached(str(path), mtime, extra_residue_params)


@lru_cache(maxsize=16)
def _load_pose_cached(
    path: str,
    _mtime: float | None,
    extra_residue_params: tuple[tuple[str, float | None], ...],
):
    try:
        allow_mmcif_pdb_fallback = bool(global_config.allow_mmcif_pdb_fallback)
    except RuntimeError:
        allow_mmcif_pdb_fallback = False

    if not path.endswith((".pdb", ".cif")):
        logging.warning(f"Unsupported file format: {path}, loading as PDB")
    return load_pose_with_optional_mmcif_pdb_fallback(
        pyrosetta,
        path,
        allow_mmcif_pdb_fallback=allow_mmcif_pdb_fallback,
        fallback_builder=ensure_centered_pdb_fallback,
        extra_residue_params=[path for path, _mtime in extra_residue_params],
        unsupported_format_as_pdb=True,
    )


@lru_cache(maxsize=4)
def remove_chain_from_pose(pose, removed_chain):
    print("get_chain_removed called")
    new_pose = pose.clone()
    for i in range(new_pose.total_residue(), 0, -1):
        if new_pose.pdb_info().chain(i) == removed_chain:
            new_pose.delete_residue_slow(i)
    return new_pose


@loss_registry.register(
    required_params=["buried_protomer"],
    description="Calculate buried fraction for a specific protomer",
    supports_oligomer_specific=True,
)
def chain_buried_fraction(eval_result, buried_protomer: str):
    """Calculate buried fraction for specified protomer."""
    _require_pyrosetta()
    buried_chain = {}
    for oligo_name in eval_result.names:
        if len(oligo_name) != 3:
            continue
        chain_id = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="chain_buried_fraction.buried_protomer",
        )
        if chain_id is None:
            continue
        buried_chain[oligo_name] = chain_id
    return calculate_buried_fraction(eval_result.complex_poses.poses, buried_chain)


def calculate_buried_fraction(
    complex_poses=None, buried_chain=None, structure_files=None
):

    result = {}

    complex_poses = validate_load_poses(complex_poses, structure_files)
    for oligo_name, pose in dict(complex_poses).items():
        if buried_chain is None or oligo_name not in buried_chain or not buried_chain[oligo_name]:
            raise ValueError(f"Missing buried chain for oligo {oligo_name!r}.")
        chain_id_to_index, chains = split_pose_by_chain(pose)
        chain_idx = chain_id_to_index.get(buried_chain[oligo_name], None)
        if chain_idx is None:
            raise ValueError(
                f"Buried chain {buried_chain[oligo_name]!r} not present in oligo {oligo_name!r}."
            )

        bound_chain_sasa = get_chain_sasa(pose, buried_chain[oligo_name])
        unbound_chain_sasa = get_chain_sasa(chains[chain_idx], buried_chain[oligo_name])
        assert len(unbound_chain_sasa) == len(
            bound_chain_sasa
        ), "SASA calculation mismatch with residue count"
        buried_ratio = (np.sum(unbound_chain_sasa) - np.sum(bound_chain_sasa)) / np.sum(
            unbound_chain_sasa
        )
        result[oligo_name] = buried_ratio
    return result


@loss_registry.register(
    required_params=["buried_protomer"],
    description="Calculate buried fraction using all-alanine reference",
    supports_oligomer_specific=True,
)
def buried_fraction_allala(eval_result, buried_protomer: str):
    """Calculate buried fraction using backbone-only reference."""
    _require_pyrosetta()
    buried_chain = {}
    for oligo_name in eval_result.names:
        if len(oligo_name) != 3:
            continue
        chain_id = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="buried_fraction_allala.buried_protomer",
        )
        if chain_id is None:
            continue
        buried_chain[oligo_name] = chain_id
    return calculate_buried_fraction_allala(
        eval_result.complex_poses.poses, buried_chain
    )


def calculate_buried_fraction_allala(
    complex_poses=None, buried_chain=None, structure_files=None
):

    result = {}
    complex_poses = validate_load_poses(complex_poses, structure_files)
    for oligo_name, pose in dict(complex_poses).items():
        if buried_chain is None or oligo_name not in buried_chain or not buried_chain[oligo_name]:
            raise ValueError(f"Missing buried chain for oligo {oligo_name!r}.")

        new_pose = mutate_pose_to_all_ala(pose)

        chain_id_to_index, chains = split_pose_by_chain(new_pose)
        chain_idx = chain_id_to_index.get(buried_chain[oligo_name], None)
        if chain_idx is None:
            raise ValueError(
                f"Buried chain {buried_chain[oligo_name]!r} not present in oligo {oligo_name!r}."
            )

        bound_chain_sasa = get_chain_sasa(new_pose, buried_chain[oligo_name])
        unbound_chain_sasa = get_chain_sasa(chains[chain_idx], buried_chain[oligo_name])
        assert len(unbound_chain_sasa) == len(
            bound_chain_sasa
        ), "SASA calculation mismatch with residue count"
        buried_ratio = (np.sum(unbound_chain_sasa) - np.sum(bound_chain_sasa)) / np.sum(
            unbound_chain_sasa
        )
        result[oligo_name] = buried_ratio
    return result


@lru_cache(maxsize=4)
def mutate_pose_to_all_ala(pose):
    print("get_allala_pose called")
    new_pose = pose.clone()
    for i in range(1, new_pose.size() + 1):
        if pose.residue(i).is_protein():  # 仅对蛋白质残基处理
            mutator = MutateResidue(i, "ALA")
            mutator.set_preserve_atom_coords(True)
            mutator.apply(new_pose)
    return new_pose


@lru_cache(maxsize=4)
def split_pose_by_chain(pose):
    print("get_chain_poses called")
    chains = pose.split_by_chain()
    chain_id_to_index = {}
    for i in range(1, len(chains) + 1):  # split_by_chain 是 1-based
        sub_pose = chains[i]
        chain_id = sub_pose.pdb_info().chain(1)  # 每个子 pose 都只有一条链
        chain_id_to_index[chain_id] = i
    return chain_id_to_index, chains


def get_residue_center(res):
    if res.has("CA"):
        ca = res.xyz("CA")
        return np.array([ca.x, ca.y, ca.z])
    else:
        # 取所有原子的几何中心
        coords = np.array(
            [
                [res.xyz(i).x, res.xyz(i).y, res.xyz(i).z]
                for i in range(1, res.natoms() + 1)
            ]
        )
        return coords.mean(axis=0)


@loss_registry.register(
    required_params=["buried_protomer"],
    description="Calculate center of mass distance for a specific protomer",
    supports_oligomer_specific=True,
)
def com_distance(eval_result, buried_protomer: str):
    """Calculate center of mass distance for specified protomer."""
    _require_pyrosetta()

    buried_chain = {}
    for oligo_name in eval_result.names:
        chain_id = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="com_distance.buried_protomer",
        )
        if chain_id is None:
            continue
        buried_chain[oligo_name] = chain_id
    return calculate_com_diff(eval_result.complex_poses.poses, buried_chain)


def calculate_com_diff(complex_poses=None, buried_chain=None, structure_files=None):
    # todo specified for triplet

    result = {}

    complex_poses = validate_load_poses(complex_poses, structure_files)
    for oligo_name, pose in dict(complex_poses).items():
        if oligo_name not in buried_chain:
            continue

        # split the pose by chains
        group1_coords = []
        group2_coords = []

        for i in range(1, pose.size() + 1):
            chain_id = pose.pdb_info().chain(i)
            res = pose.residue(i)

            center = get_residue_center(res)
            if chain_id in buried_chain[oligo_name]:
                group1_coords.append(center)
            else:
                group2_coords.append(center)

        compute_group_centroid = lambda coords: np.mean(np.vstack(coords), axis=0)

        centroid1 = compute_group_centroid(group1_coords)
        centroid2 = compute_group_centroid(group2_coords)
        distance = np.linalg.norm(centroid1 - centroid2)
        result[oligo_name] = distance
    return result


def calculate_com_line_distance(eval_result, buried_protomer):
    """计算buried chain到其他任意两条链的质心连线的距离"""

    def get_residue_center(res):
        if res.has("CA"):
            ca = res.xyz("CA")
            return np.array([ca.x, ca.y, ca.z])
        else:
            # 取所有原子的几何中心
            coords = np.array(
                [
                    [res.xyz(i).x, res.xyz(i).y, res.xyz(i).z]
                    for i in range(1, res.natoms() + 1)
                ]
            )
            return coords.mean(axis=0)

    def point_to_line_distance(A, B, C):
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        AB = B - A
        AC = C - A

        cross = np.cross(AB, AC)
        distance = np.linalg.norm(cross) / np.linalg.norm(AB)
        return distance

    result = {}
    for oligo_name, pose in eval_result.complex_poses.poses.items():
        if len(oligo_name) != 3:
            continue
        chain_c = resolve_single_chain_id(
            oligo_name,
            buried_protomer,
            context="calculate_com_line_distance.buried_protomer",
        )
        if chain_c is None:
            continue
        other_chains = [c for c in chain_ids_for_oligo(oligo_name) if c != chain_c]
        if len(other_chains) != 2:
            raise NotImplementedError("Only support triplet oligomer for now")
        chainA, chainB = other_chains

        # split the pose by chains
        chain2coords = defaultdict(list)

        for i in range(1, pose.size() + 1):
            chain_id = pose.pdb_info().chain(i)
            res = pose.residue(i)

            center = get_residue_center(res)
            chain2coords[chain_id].append(center)

        compute_group_centroid = lambda coords: np.mean(np.vstack(coords), axis=0)
        chain2com = {
            chain: compute_group_centroid(coords)
            for chain, coords in chain2coords.items()
        }

        result[oligo_name] = point_to_line_distance(
            chain2com[chainA], chain2com[chainB], chain2com[chain_c]
        )

    return result


def get_hotspot_interaction_number(eval_result, protomer):
    # -------------
    # temporary obj
    hotspot = protomer.hotspots

    # split the complex into chains
    complex_str = load_structure(eval_result.complex_paths)

    proto_hotspot_num_interactions = defaultdict(lambda: defaultdict(list))

    for complex_name, complex in complex_str.items():
        atom_coord, res_index, chain_name = [
            complex[key] for key in ["atom_coord", "res_index", "chain_name"]
        ]
        hotspot_coordinates = []
        exclude_coord = []
        # split the complex by chains
        for chain, complex_ in itertools.groupby(
            zip(atom_coord, res_index, chain_name), key=lambda x: x[2]
        ):
            complex_ = list(complex_)
            chain_atom_coord, chain_res_index, _ = zip(*complex_)
            chain_atom_coord = np.array(chain_atom_coord)
            chain_res_index = np.array(chain_res_index)

            if chain2proto[chain] in hotspot:
                hotspot_index = np.isin(
                    chain_res_index, hotspot[chain2proto[chain]], kind="table"
                )
                hotspot_coord = chain_atom_coord[hotspot_index]
                hotspot_residue_index = chain_res_index[hotspot_index]
                hotspot_coordinates.append(hotspot_coord)
                exclude_coord.append(atom_coord[chain_name != chain])
                target_kdt = cKDTree(atom_coord[chain_name != chain])
                hotspot_kdt = cKDTree(hotspot_coord)
                neighbors = hotspot_kdt.query_ball_tree(target_kdt, r=4.0)
                # dist = np.unique(np.array([j for i in dist for j in i]))

                for res, grp in itertools.groupby(
                    zip(hotspot_residue_index, neighbors), key=lambda x: x[0]
                ):
                    _, res_neighbor = zip(*grp)
                    num_interactions = sum(len(i) for i in res_neighbor)
                    print(res, res_neighbor)
                    proto_hotspot_num_interactions[chain2proto[chain]][res].append(
                        num_interactions
                    )
    return proto_hotspot_num_interactions


def load_structure(complex_paths):
    result = {name: {} for name in complex_paths}
    for name, str_path in complex_paths.items():
        parser = MMCIFParser()
        structure = parser.get_structure("sample", str_path)

        res_index = []
        atom_coord = []
        atom_name = []
        chain_name = []
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        res_index.append(residue.id[1])
                        atom_coord.append(atom.get_coord())
                        atom_name.append(atom.name)
                        chain_name.append(chain.id)
        atom_coord = np.array(atom_coord)
        res_index = np.array(res_index)
        chain_name = np.array(chain_name)
        result[name] = {
            "atom_coord": atom_coord,
            "res_index": res_index,
            "chain_name": chain_name,
        }
    return result
