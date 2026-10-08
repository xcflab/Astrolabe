from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from astrolabe.losses.base import loss_registry
from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized


try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None  # type: ignore


def _require_ckdtree():
    if cKDTree is None:
        raise ImportError(
            "scipy is required for BindCraft-aligned clash/interface-residue losses."
        )


def _to_pdb_path(structure_path: Path) -> Path:
    """
    Some downstream tools (Bio.PDB DSSP, PyRosetta) are most reliable on PDB.
    Convert mmCIF to a temp PDB when needed.
    """
    structure_path = Path(structure_path)
    if structure_path.suffix.lower() in {".pdb", ".ent"}:
        return structure_path
    import gemmi

    st = gemmi.read_structure(str(structure_path))
    f = tempfile.NamedTemporaryFile("w+", suffix=".pdb", delete=False)
    f.close()
    st.write_pdb(f.name)
    return Path(f.name)


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _all_chains_in_structure(path: Path) -> list[str]:
    st = _load_biopython_structure(path)
    for model in st:
        return [str(ch.id) for ch in model.get_chains()]
    return []


def _mean_bfactor(path: Path, *, chain_ids: Optional[set[str]] = None) -> float:
    st = _load_biopython_structure(path)
    b: list[float] = []
    for model in st:
        for chain in model.get_chains():
            if chain_ids is not None and str(chain.id) not in chain_ids:
                continue
            for atom in chain.get_atoms():
                try:
                    b.append(float(atom.bfactor))
                except Exception:
                    continue
        break
    return _mean_or_zero(b)


@loss_registry.register(
    name="bfactor_mean",
    description="Mean B-factor over all atoms (common pLDDT proxy for structure files)",
    supports_oligomer_specific=True,
)
def bfactor_mean(eval_result):
    return {name: _mean_bfactor(Path(eval_result.complex_paths[name])) for name in eval_result.names}


@loss_registry.register(
    name="bfactor_mean_norm",
    description="Mean B-factor normalized to [0,1] assuming 0..100 scale",
    supports_oligomer_specific=True,
)
def bfactor_mean_norm(eval_result):
    scores = {}
    for name in eval_result.names:
        mean = _mean_bfactor(Path(eval_result.complex_paths[name]))
        scores[name] = float(mean / 100.0) if mean > 1.5 else float(mean)
    return scores


@loss_registry.register(
    name="plddt_mean_norm",
    description="Alias for mean pLDDT in [0,1] (model-native per-residue pLDDT)",
    supports_oligomer_specific=True,
)
def plddt_mean_norm(eval_result):
    # StandardScores.plddt is assumed to already be in [0,1] in this codebase.
    return {name: float(np.mean(eval_result['plddt'][name])) for name in eval_result.names}


@loss_registry.register(
    name="plddt_mean",
    description="Alias for mean pLDDT (same as plddt_mean_norm in this codebase)",
    supports_oligomer_specific=True,
)
def plddt_mean(eval_result):
    return {name: float(np.mean(eval_result['plddt'][name])) for name in eval_result.names}


# BindCraft uses average binder-chain pLDDT >= 0.8 (as B-factor-derived proxy in many pipelines).
@loss_registry.register(
    name="bindcraft_binder_plddt_norm",
    description="BindCraft-aligned binder pLDDT normalized to [0,1] from B-factors",
    required_params=["binder_chain"],
    supports_oligomer_specific=True,
)
def bindcraft_binder_plddt_norm(eval_result, binder_chain: str):
    scores = {}
    for name in eval_result.names:
        mean = _mean_bfactor(Path(eval_result.complex_paths[name]), chain_ids={str(binder_chain)})
        scores[name] = float(mean / 100.0) if mean > 1.5 else float(mean)
    return scores


# ---------------------------------------------------------------------------
# HalluDesign-aligned B-factor (pLDDT proxy) aggregations
# ---------------------------------------------------------------------------


@loss_registry.register(
    name="halludesign_plddt",
    description="HalluDesign-aligned average B-factor over all atoms (pLDDT proxy)",
    supports_oligomer_specific=True,
)
def halludesign_plddt(eval_result):
    return {name: _mean_bfactor(Path(eval_result.complex_paths[name])) for name in eval_result.names}


@loss_registry.register(
    name="halludesign_plddt_chain_A",
    description="HalluDesign-aligned average B-factor over chain A (pLDDT proxy)",
    supports_oligomer_specific=True,
)
def halludesign_plddt_chain_A(eval_result):
    return {name: _mean_bfactor(Path(eval_result.complex_paths[name]), chain_ids={"A"}) for name in eval_result.names}


@loss_registry.register(
    name="halludesign_plddt_chain_B",
    description="HalluDesign-aligned average B-factor over chain B (pLDDT proxy)",
    supports_oligomer_specific=True,
)
def halludesign_plddt_chain_B(eval_result):
    return {name: _mean_bfactor(Path(eval_result.complex_paths[name]), chain_ids={"B"}) for name in eval_result.names}


@loss_registry.register(
    name="halludesign_key_res_plddt",
    description="HalluDesign-aligned key-region pLDDT (mean protein atom B-factors within cutoff of ligand/binder atoms; duplicates allowed)",
    required_params=["protein_chain"],
    optional_params=["ligand_chains", "cutoff"],
    supports_oligomer_specific=True,
)
def halludesign_key_res_plddt(
    eval_result,
    protein_chain: str,
    ligand_chains: str | None = None,
    cutoff: float = 5.0,
):
    _require_ckdtree()
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        pdb_path = _to_pdb_path(path)
        st = _load_biopython_structure(pdb_path)

        prot_atoms = []
        prot_b = []
        lig_atoms = []

        target_chain_ids = {str(protein_chain)}
        if ligand_chains is None:
            ligand_chain_ids = set(c for c in _all_chains_in_structure(pdb_path) if c != str(protein_chain))
        else:
            ligand_chain_ids = {c.strip() for c in str(ligand_chains).split(",") if c.strip()}

        for model in st:
            for chain in model.get_chains():
                cid = str(chain.id)
                if cid in target_chain_ids:
                    for atom in chain.get_atoms():
                        prot_atoms.append(np.asarray(atom.coord, dtype=np.float64))
                        prot_b.append(float(getattr(atom, "bfactor", 0.0)))
                elif cid in ligand_chain_ids:
                    for atom in chain.get_atoms():
                        lig_atoms.append(np.asarray(atom.coord, dtype=np.float64))
            break

        if not prot_atoms or not lig_atoms:
            scores[oligo_name] = 0.0
            continue

        prot_tree = cKDTree(np.stack(prot_atoms, axis=0))
        b_factors_near = []
        for lig in lig_atoms:
            idxs = prot_tree.query_ball_point(lig, r=float(cutoff))
            # duplicates allowed across ligand atoms
            b_factors_near.extend(prot_b[i] for i in idxs)
        scores[oligo_name] = _mean_or_zero(b_factors_near)
    return scores


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) that best aligns P to Q: P @ R + t ~= Q (least squares).
    P, Q: (N,3)
    """
    Pc = P - np.mean(P, axis=0, keepdims=True)
    Qc = Q - np.mean(Q, axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = np.mean(Q, axis=0) - np.mean(P, axis=0) @ R
    return R, t


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# ODesign TMscore.cpp (exact parity by shelling out to upstream implementation)
# ---------------------------------------------------------------------------


_TM_RE = re.compile(r"TM-score\\s*=\\s*([0-9.]+)")
_RMSD_RE = re.compile(r"RMSD\\s*of\\s*the\\s*common\\s*residues\\s*=\\s*([0-9.]+)")


def _ensure_odesign_tmscore_binary(tmscore_cpp: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[3]
    build_dir = root / ".cache" / "odesign_tmscore"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "TMscore"
    if exe.exists():
        return exe

    candidates = []
    if tmscore_cpp is not None:
        candidates.append(Path(tmscore_cpp))
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "ODesign TMscore.cpp not found. Set `tmscore_cpp` to the upstream source file."
        )

    subprocess.run(
        ["g++", "-O3", "-ffast-math", "-o", str(exe), str(src)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return exe


def _run_odesign_tmscore(structure_a: Path, structure_b: Path, *, tmscore_bin: Path) -> tuple[float, float | None]:
    proc = subprocess.run(
        [str(tmscore_bin), str(structure_a), str(structure_b)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout or ""
    m_tm = _TM_RE.search(out)
    m_rmsd = _RMSD_RE.search(out)
    if not m_tm:
        raise RuntimeError(
            f"Failed to parse TM-score from TMscore output (exit={proc.returncode})."
        )
    tm = float(m_tm.group(1))
    rmsd = float(m_rmsd.group(1)) if m_rmsd else None
    return tm, rmsd


@loss_registry.register(
    name="odesign_tmscore",
    description="ODesign TM-score (shells out to upstream TMscore.cpp; requires reference_structure)",
    required_params=["reference_structure"],
    optional_params=["tmscore_cpp", "tmscore_bin"],
    supports_oligomer_specific=True,
)
def odesign_tmscore(
    eval_result,
    reference_structure: str,
    tmscore_cpp: str | None = None,
    tmscore_bin: str | None = None,
):
    ref = Path(reference_structure)
    bin_path = Path(tmscore_bin) if tmscore_bin else _ensure_odesign_tmscore_binary(Path(tmscore_cpp) if tmscore_cpp else None)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        pred = _to_pdb_path(Path(eval_result.complex_paths[oligo_name]))
        ref_pdb = _to_pdb_path(ref)
        tm, _ = _run_odesign_tmscore(pred, ref_pdb, tmscore_bin=bin_path)
        scores[oligo_name] = float(tm)
    return scores


def _load_biopython_structure(path: Path):
    try:
        from Bio.PDB import MMCIFParser, PDBParser  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("biopython is required for structure parsing.") from exc

    path = Path(path)
    if path.suffix.lower() in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    return parser.get_structure("model", str(path))


def _extract_ca_coords(path: Path, *, chain_id: Optional[str]) -> np.ndarray:
    """
    Extract CA coordinates from a PDB/mmCIF.

    NOTE: PXDesignBench iterates residues in file order; we keep the same behavior.
    """
    try:
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("biopython is required for CA extraction.") from exc

    st = _load_biopython_structure(path)
    coords = []
    for model in st:
        for chain in model.get_chains():
            if chain_id is not None and str(chain.id) != str(chain_id):
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                if "CA" not in res:
                    continue
                coords.append(np.asarray(res["CA"].coord, dtype=np.float64))
        break  # first model only
    return np.asarray(coords, dtype=np.float64)


def _collect_ca_coords_by_key(path: Path, *, chain_id: Optional[str]) -> dict[tuple[str, int, str], np.ndarray]:
    """
    PXDesignBench fallback when CA counts differ: residue-key matching on (chain_id, resseq, icode).
    """
    try:
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("biopython is required for CA extraction.") from exc

    st = _load_biopython_structure(path)
    idx: dict[tuple[str, int, str], np.ndarray] = {}
    for model in st:
        for chain in model.get_chains():
            if chain_id is not None and str(chain.id) != str(chain_id):
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                if "CA" not in res:
                    continue
                het, resseq, icode = res.get_id()
                if het != " ":
                    continue
                key = (str(chain.id), int(resseq), str(icode or "").strip())
                idx[key] = np.asarray(res["CA"].coord, dtype=np.float64)
        break
    return idx


# ---------------------------------------------------------------------------
# PXDesignBench-aligned RMSD losses (CA-only, Kabsch)
# ---------------------------------------------------------------------------


def _pxdesign_align_and_calculate_rmsd(file1: Path, file2: Path) -> Optional[float]:
    P = _extract_ca_coords(file1, chain_id=None)
    Q = _extract_ca_coords(file2, chain_id=None)
    if P.shape[0] != Q.shape[0]:
        idx1 = _collect_ca_coords_by_key(file1, chain_id=None)
        idx2 = _collect_ca_coords_by_key(file2, chain_id=None)
        common = sorted(set(idx1.keys()) & set(idx2.keys()))
        if len(common) < 3:
            return None
        P = np.vstack([idx1[k] for k in common])
        Q = np.vstack([idx2[k] for k in common])
    if P.shape[0] < 3:
        return None
    R, t = _kabsch(P, Q)
    P_aligned = P @ R + t
    return _rmsd(P_aligned, Q)


def _pxdesign_binder_align_and_calculate_rmsd(
    file1: Path, file2: Path, *, chain_id: str
) -> Optional[float]:
    P = _extract_ca_coords(file1, chain_id=None)
    Q = _extract_ca_coords(file2, chain_id=str(chain_id))
    if P.shape[0] != Q.shape[0] or P.shape[0] < 3:
        return None
    R, t = _kabsch(P, Q)
    P_aligned = P @ R + t
    return _rmsd(P_aligned, Q)


@loss_registry.register(
    name="complex_rmsd",
    description="PXDesignBench-style CA-only Kabsch RMSD over all chains (requires reference_complex)",
    required_params=["reference_complex"],
    supports_oligomer_specific=True,
)
def complex_rmsd(eval_result, reference_complex: str):
    ref = Path(reference_complex)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        pred = Path(eval_result.complex_paths[oligo_name])
        v = _pxdesign_align_and_calculate_rmsd(pred, ref)
        scores[oligo_name] = float(v) if v is not None else 1e6
    return scores


@loss_registry.register(
    name="binder_rmsd",
    description="PXDesignBench-style CA-only RMSD for binder_unbound_structure vs binder chain in reference_complex",
    required_params=["binder_unbound_structure", "reference_complex", "binder_chain"],
    supports_oligomer_specific=True,
)
def binder_rmsd(eval_result, binder_unbound_structure: str, reference_complex: str, binder_chain: str):
    binder_unbound = Path(binder_unbound_structure)
    ref = Path(reference_complex)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        v = _pxdesign_binder_align_and_calculate_rmsd(binder_unbound, ref, chain_id=str(binder_chain))
        scores[oligo_name] = float(v) if v is not None else 1e6
    return scores


@loss_registry.register(
    name="binder_bound_unbound_binder_rmsd",
    description="PXDesignBench-style CA-only RMSD for binder_unbound_structure vs binder chain in predicted complex",
    required_params=["binder_unbound_structure", "binder_chain"],
    supports_oligomer_specific=True,
)
def binder_bound_unbound_binder_rmsd(eval_result, binder_unbound_structure: str, binder_chain: str):
    binder_unbound = Path(binder_unbound_structure)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        pred = Path(eval_result.complex_paths[oligo_name])
        v = _pxdesign_binder_align_and_calculate_rmsd(binder_unbound, pred, chain_id=str(binder_chain))
        scores[oligo_name] = float(v) if v is not None else 1e6
    return scores


# ---------------------------------------------------------------------------
# BindCraft-aligned clash + interface residue identification
# ---------------------------------------------------------------------------


_THREE_TO_ONE_MAP: dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


@dataclass(frozen=True)
class _Atom:
    chain: str
    resseq: int
    resname: str
    atom: str
    element: str
    coord: np.ndarray


def _iter_atoms(path: Path) -> list[_Atom]:
    st = _load_biopython_structure(path)
    atoms: list[_Atom] = []
    for model in st:
        for chain in model:
            for res in chain:
                resseq = int(res.id[1])
                resname = str(getattr(res, "resname", "") or "").upper()
                for atom in res.get_atoms():
                    element = str(getattr(atom, "element", "") or "").upper()
                    name = str(getattr(atom, "name", "") or "").strip()
                    coord = np.asarray(atom.coord, dtype=np.float64)
                    atoms.append(
                        _Atom(
                            chain=str(chain.id),
                            resseq=resseq,
                            resname=resname,
                            atom=name,
                            element=element,
                            coord=coord,
                        )
                    )
        break
    return atoms


def _calculate_bindcraft_clash_score(
    structure_path: Path, *, threshold: float = 2.4, only_ca: bool = False
) -> int:
    """
    BindCraft-aligned clash counting:
      - skip hydrogen atoms
      - exclude same-residue pairs
      - exclude sequential residues (|i-j|==1) within same chain
      - when only_ca=False: only count inter-chain clashes (skip same-chain pairs)
    """
    _require_ckdtree()
    atoms = _iter_atoms(structure_path)
    filtered: list[_Atom] = []
    coords: list[np.ndarray] = []
    for a in atoms:
        if a.element == "H":
            continue
        if only_ca and a.atom != "CA":
            continue
        filtered.append(a)
        coords.append(a.coord)

    if len(coords) < 2:
        return 0

    tree = cKDTree(np.stack(coords, axis=0))
    pairs = tree.query_pairs(threshold)

    valid = 0
    for i, j in pairs:
        ai = filtered[i]
        aj = filtered[j]

        if ai.chain == aj.chain and ai.resseq == aj.resseq:
            continue
        if ai.chain == aj.chain and abs(ai.resseq - aj.resseq) == 1:
            continue
        if (not only_ca) and ai.chain == aj.chain:
            continue
        valid += 1
    return int(valid)


def _bindcraft_hotspot_residues(
    structure_path: Path,
    *,
    binder_chain: str = "B",
    target_chain: str = "A",
    atom_distance_cutoff: float = 4.0,
) -> dict[int, str]:
    """
    BindCraft-aligned interface residue identification.

    Returns: {binder_resseq -> binder_aa_1letter} for binder residues with >=1 atom within cutoff
    of any target atom.
    """
    _require_ckdtree()
    atoms = _iter_atoms(structure_path)
    binder_atoms = [a for a in atoms if a.chain == str(binder_chain)]
    target_atoms = [a for a in atoms if a.chain == str(target_chain)]
    if not binder_atoms or not target_atoms:
        return {}

    binder_coords = np.stack([a.coord for a in binder_atoms], axis=0)
    target_coords = np.stack([a.coord for a in target_atoms], axis=0)

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    interacting: dict[int, str] = {}
    for binder_idx, close_indices in enumerate(pairs):
        if not close_indices:
            continue
        resname3 = binder_atoms[binder_idx].resname.upper()
        if resname3 not in _THREE_TO_ONE_MAP:
            continue
        interacting[int(binder_atoms[binder_idx].resseq)] = _THREE_TO_ONE_MAP[resname3]
    return interacting


@lru_cache(maxsize=256)
def _bindcraft_interface_aa_summary(
    structure_path_str: str,
    *,
    binder_chain: str,
    target_chain: str,
    atom_distance_cutoff: float = 4.0,
) -> dict[str, float]:
    """
    BindCraft-aligned interface residue AA composition summary.

    Returns numeric fields used by filtering:
      - bindcraft_interface_nres
      - bindcraft_interface_hydrophobicity (0..100)
      - bindcraft_interface_aa_count_<AA> for AA in ACDEFGHIKLMNPQRSTVWY
    """
    structure_path = Path(structure_path_str)
    iface = _bindcraft_hotspot_residues(
        structure_path,
        binder_chain=binder_chain,
        target_chain=target_chain,
        atom_distance_cutoff=atom_distance_cutoff,
    )
    counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for aa in iface.values():
        if aa in counts:
            counts[aa] += 1

    interface_nres = float(len(iface))
    hydrophobic = set("ACFILMPVWY")
    hydrophobic_count = float(sum(counts.get(aa, 0) for aa in hydrophobic))
    interface_hydrophobicity = (
        (hydrophobic_count / interface_nres) * 100.0 if interface_nres > 0 else 0.0
    )

    out: dict[str, float] = {
        "bindcraft_interface_nres": interface_nres,
        "bindcraft_interface_hydrophobicity": float(interface_hydrophobicity),
    }
    for aa, n in counts.items():
        out[f"bindcraft_interface_aa_count_{aa}"] = float(n)
    return out


@loss_registry.register(
    name="clash_pairs",
    description="BindCraft-aligned heavy-atom inter-chain clash count (<=2.4Å)",
    supports_oligomer_specific=True,
)
def clash_pairs(eval_result):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        scores[oligo_name] = float(
            _calculate_bindcraft_clash_score(path, threshold=2.4, only_ca=False)
        )
    return scores


@loss_registry.register(
    name="clash_pairs_ca",
    description="BindCraft-aligned Cα clash count (<=2.5Å)",
    supports_oligomer_specific=True,
)
def clash_pairs_ca(eval_result):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        scores[oligo_name] = float(
            _calculate_bindcraft_clash_score(path, threshold=2.5, only_ca=True)
        )
    return scores


# ---------------------------------------------------------------------------
# BindCraft-aligned RMSD metrics (PyRosetta/Bio.PDB; require reference inputs)
# ---------------------------------------------------------------------------


def _bindcraft_align_pdbs(
    *,
    reference_pdb: Path,
    align_pdb: Path,
    reference_chain_id: str,
    align_chain_id: str,
    pyrosetta_init: str,
) -> Path:
    _ensure_pyrosetta(pyrosetta_init)
    import pyrosetta as pr  # type: ignore
    from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover  # type: ignore

    reference_pdb = _to_pdb_path(reference_pdb)
    align_pdb = _to_pdb_path(align_pdb)

    reference_pose = pr.pose_from_pdb(str(reference_pdb))
    align_pose = pr.pose_from_pdb(str(align_pdb))

    mover = AlignChainMover()
    mover.pose(reference_pose)

    reference_chain_id = str(reference_chain_id).split(",")[0]
    align_chain_id = str(align_chain_id).split(",")[0]

    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        reference_chain_id, reference_pose
    )
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        align_chain_id, align_pose
    )

    mover.source_chain(align_chain)
    mover.target_chain(reference_chain)
    mover.apply(align_pose)

    out = tempfile.NamedTemporaryFile("w+", suffix=".pdb", delete=False)
    out.close()
    align_pose.dump_pdb(out.name)
    return Path(out.name)


def _bindcraft_unaligned_rmsd(
    *,
    reference_pdb: Path,
    align_pdb: Path,
    reference_chain_id: str,
    align_chain_id: str,
    pyrosetta_init: str,
) -> float:
    _ensure_pyrosetta(pyrosetta_init)
    import pyrosetta as pr  # type: ignore
    from pyrosetta.rosetta.core.io import pose_from_pose  # type: ignore
    from pyrosetta.rosetta.core.select import get_residues_from_subset  # type: ignore
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector  # type: ignore
    from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric  # type: ignore

    reference_pdb = _to_pdb_path(reference_pdb)
    align_pdb = _to_pdb_path(align_pdb)

    reference_pose = pr.pose_from_pdb(str(reference_pdb))
    align_pose = pr.pose_from_pdb(str(align_pdb))

    reference_subset = ChainSelector(str(reference_chain_id)).apply(reference_pose)
    align_subset = ChainSelector(str(align_chain_id)).apply(align_pose)

    reference_indices = get_residues_from_subset(reference_subset)
    align_indices = get_residues_from_subset(align_subset)

    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()
    pose_from_pose(reference_chain_pose, reference_pose, reference_indices)
    pose_from_pose(align_chain_pose, align_pose, align_indices)

    metric = RMSDMetric()
    metric.set_comparison_pose(reference_chain_pose)
    return float(round(float(metric.calculate(align_chain_pose)), 2))


def _bindcraft_target_pdb_rmsd(
    *,
    trajectory_pdb: Path,
    starting_pdb: Path,
    chain_ids_string: str,
) -> float:
    try:
        from Bio.PDB import PDBParser, Superimposer  # type: ignore
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("BindCraft target RMSD requires biopython (Bio.PDB).") from exc

    trajectory_pdb = _to_pdb_path(trajectory_pdb)
    starting_pdb = _to_pdb_path(starting_pdb)

    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure("trajectory", str(trajectory_pdb))
    structure_starting = parser.get_structure("starting", str(starting_pdb))

    chain_trajectory = structure_trajectory[0]["A"]

    chain_ids = [c.strip() for c in str(chain_ids_string).split(",") if c.strip()]
    residues_starting = []
    for cid in chain_ids:
        chain = structure_starting[0][cid]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)

    residues_trajectory = [r for r in chain_trajectory if is_aa(r, standard=True)]

    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]

    atoms_starting = [r["CA"] for r in residues_starting if "CA" in r]
    atoms_trajectory = [r["CA"] for r in residues_trajectory if "CA" in r]

    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    return float(round(float(sup.rms), 2))


@loss_registry.register(
    name="bindcraft_hotspot_rmsd",
    description="BindCraft-aligned hotspot RMSD (unaligned PyRosetta RMSDMetric on binder chain; requires bindcraft_trajectory_pdb)",
    required_params=["bindcraft_trajectory_pdb", "binder_chain"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_hotspot_rmsd(
    eval_result,
    bindcraft_trajectory_pdb: str,
    binder_chain: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    traj = Path(bindcraft_trajectory_pdb)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        pred = Path(eval_result.complex_paths[oligo_name])
        scores[oligo_name] = _bindcraft_unaligned_rmsd(
            reference_pdb=traj,
            align_pdb=pred,
            reference_chain_id=str(binder_chain),
            align_chain_id=str(binder_chain),
            pyrosetta_init=pyrosetta_init,
        )
    return scores


@loss_registry.register(
    name="bindcraft_binder_rmsd",
    description="BindCraft-aligned binder RMSD (align binder-alone onto trajectory, then unaligned RMSD; requires bindcraft_trajectory_pdb + bindcraft_binder_alone_pdb)",
    required_params=["bindcraft_trajectory_pdb", "bindcraft_binder_alone_pdb", "binder_chain"],
    optional_params=["bindcraft_binder_alone_chain", "pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_binder_rmsd(
    eval_result,
    bindcraft_trajectory_pdb: str,
    bindcraft_binder_alone_pdb: str,
    binder_chain: str,
    bindcraft_binder_alone_chain: str = "A",
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    traj = Path(bindcraft_trajectory_pdb)
    binder_alone = Path(bindcraft_binder_alone_pdb)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        aligned = _bindcraft_align_pdbs(
            reference_pdb=traj,
            align_pdb=binder_alone,
            reference_chain_id=str(binder_chain),
            align_chain_id=str(bindcraft_binder_alone_chain),
            pyrosetta_init=pyrosetta_init,
        )
        scores[oligo_name] = _bindcraft_unaligned_rmsd(
            reference_pdb=traj,
            align_pdb=aligned,
            reference_chain_id=str(binder_chain),
            align_chain_id=str(bindcraft_binder_alone_chain),
            pyrosetta_init=pyrosetta_init,
        )
    return scores


@loss_registry.register(
    name="bindcraft_target_rmsd",
    description="BindCraft-aligned target RMSD (CA-only; requires bindcraft_starting_pdb + bindcraft_starting_target_chains)",
    required_params=["bindcraft_starting_pdb", "bindcraft_starting_target_chains"],
    supports_oligomer_specific=True,
)
def bindcraft_target_rmsd(
    eval_result,
    bindcraft_starting_pdb: str,
    bindcraft_starting_target_chains: str,
):
    start = Path(bindcraft_starting_pdb)
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        pred = Path(eval_result.complex_paths[oligo_name])
        scores[oligo_name] = _bindcraft_target_pdb_rmsd(
            trajectory_pdb=pred,
            starting_pdb=start,
            chain_ids_string=str(bindcraft_starting_target_chains),
        )
    return scores


# ---------------------------------------------------------------------------
# BindCraft-aligned DSSP loop% + pLDDT aggregation from B-factors
# ---------------------------------------------------------------------------


def _bindcraft_ss_metrics(
    structure_path: Path,
    *,
    binder_chain: str,
    target_chain: str,
    dssp_path: str = "mkdssp",
    atom_distance_cutoff: float = 4.0,
) -> dict[str, float]:
    try:
        from Bio.PDB import DSSP, PDBParser  # type: ignore
        from Bio.PDB.Polypeptide import is_aa  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "BindCraft SS% losses require biopython (Bio.PDB) + DSSP (mkdssp)."
        ) from exc

    pdb_path = _to_pdb_path(structure_path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]
    dssp = DSSP(model, str(pdb_path), dssp=dssp_path)

    chain = model[str(binder_chain)]
    interacting_residues = set(
        _bindcraft_hotspot_residues(
            structure_path,
            binder_chain=binder_chain,
            target_chain=target_chain,
            atom_distance_cutoff=atom_distance_cutoff,
        ).keys()
    )

    ss_counts = {"helix": 0, "sheet": 0, "loop": 0}
    ss_interface_counts = {"helix": 0, "sheet": 0, "loop": 0}
    plddts_interface: list[float] = []
    plddts_ss: list[float] = []

    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        residue_id = int(residue.id[1])
        if (str(binder_chain), residue_id) not in dssp:
            continue
        ss = dssp[(str(binder_chain), residue_id)][2]
        ss_type = "loop"
        if ss in ["H", "G", "I"]:
            ss_type = "helix"
        elif ss == "E":
            ss_type = "sheet"

        ss_counts[ss_type] += 1

        if ss_type != "loop":
            avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
            plddts_ss.append(float(avg_plddt_ss))

        if residue_id in interacting_residues:
            ss_interface_counts[ss_type] += 1
            avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
            plddts_interface.append(float(avg_plddt_residue))

    total_residues = int(sum(ss_counts.values()))
    total_interface_residues = int(sum(ss_interface_counts.values()))

    def _pct(total: int, helix: int, sheet: int) -> tuple[float, float, float]:
        helix_percentage = round((helix / total) * 100, 2) if total > 0 else 0
        sheet_percentage = round((sheet / total) * 100, 2) if total > 0 else 0
        loop_percentage = round(((total - helix - sheet) / total) * 100, 2) if total > 0 else 0
        return float(helix_percentage), float(sheet_percentage), float(loop_percentage)

    helix_pct, sheet_pct, loop_pct = _pct(
        total_residues, ss_counts["helix"], ss_counts["sheet"]
    )
    i_helix_pct, i_sheet_pct, i_loop_pct = _pct(
        total_interface_residues,
        ss_interface_counts["helix"],
        ss_interface_counts["sheet"],
    )

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return {
        "bindcraft_binder_helix_pct": helix_pct,
        "bindcraft_binder_sheet_pct": sheet_pct,
        "bindcraft_binder_loop_pct": loop_pct,
        "bindcraft_interface_helix_pct": i_helix_pct,
        "bindcraft_interface_sheet_pct": i_sheet_pct,
        "bindcraft_interface_loop_pct": i_loop_pct,
        "bindcraft_interface_plddt_norm": float(i_plddt),
        "bindcraft_ss_plddt_norm": float(ss_plddt),
    }


def _bindcraft_ss_field(
    eval_result, *, field: str, binder_chain: str, target_chain: str
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        vals = _bindcraft_ss_metrics(path, binder_chain=binder_chain, target_chain=target_chain)
        scores[oligo_name] = float(vals.get(field, 0.0))
    return scores


@loss_registry.register(
    name="bindcraft_binder_loop_pct",
    description="BindCraft-aligned binder loop% (DSSP; requires mkdssp)",
    required_params=["binder_chain", "target_chain"],
    supports_oligomer_specific=True,
)
def bindcraft_binder_loop_pct(eval_result, binder_chain: str, target_chain: str):
    return _bindcraft_ss_field(eval_result, field="bindcraft_binder_loop_pct", binder_chain=binder_chain, target_chain=target_chain)


@loss_registry.register(
    name="bindcraft_interface_loop_pct",
    description="BindCraft-aligned interface loop% (DSSP; requires mkdssp)",
    required_params=["binder_chain", "target_chain"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_loop_pct(eval_result, binder_chain: str, target_chain: str):
    return _bindcraft_ss_field(eval_result, field="bindcraft_interface_loop_pct", binder_chain=binder_chain, target_chain=target_chain)


@loss_registry.register(
    name="bindcraft_interface_plddt_norm",
    description="BindCraft-aligned interface pLDDT normalized to [0,1] (from B-factors)",
    required_params=["binder_chain", "target_chain"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_plddt_norm(eval_result, binder_chain: str, target_chain: str):
    return _bindcraft_ss_field(eval_result, field="bindcraft_interface_plddt_norm", binder_chain=binder_chain, target_chain=target_chain)


@loss_registry.register(
    name="bindcraft_ss_plddt_norm",
    description="BindCraft-aligned non-loop pLDDT normalized to [0,1] (from B-factors)",
    required_params=["binder_chain", "target_chain"],
    supports_oligomer_specific=True,
)
def bindcraft_ss_plddt_norm(eval_result, binder_chain: str, target_chain: str):
    return _bindcraft_ss_field(eval_result, field="bindcraft_ss_plddt_norm", binder_chain=binder_chain, target_chain=target_chain)


# ---------------------------------------------------------------------------
# HalluDesign AF3 summary-style chain-pair aggregation (when available)
# ---------------------------------------------------------------------------


def _chain_pair_row_mean(
    eval_result,
    *,
    oligo_name: str,
    key: str,
    row: int = 0,
) -> float:
    chain_pair_scores = getattr(eval_result, "chain_pair_scores", None) or {}
    mats = (chain_pair_scores.get(oligo_name) or {}).get(key)
    if not mats:
        return 0.0
    averaged = np.mean(np.stack(mats), axis=0)
    if averaged.ndim != 2 or averaged.shape[0] <= row:
        return 0.0
    vals = [float(x) for x in averaged[row, 1:].tolist()]
    return float(sum(vals) / len(vals)) if vals else 0.0


@loss_registry.register(
    name="halludesign_af3_iptm",
    description="HalluDesign AF3 summary-style ipTM aggregation from chain_pair_iptm (row 0, cols 1..)",
    supports_oligomer_specific=True,
)
def halludesign_af3_iptm(eval_result):
    return {name: _chain_pair_row_mean(eval_result, oligo_name=name, key="chain_pair_iptm", row=0) for name in eval_result.names}


@loss_registry.register(
    name="halludesign_af3_ipae",
    description="HalluDesign AF3 summary-style ipAE aggregation from chain_pair_pae_min (row 0, cols 1..)",
    supports_oligomer_specific=True,
)
def halludesign_af3_ipae(eval_result):
    return {name: _chain_pair_row_mean(eval_result, oligo_name=name, key="chain_pair_pae_min", row=0) for name in eval_result.names}


# ---------------------------------------------------------------------------
# Interface energy metrics (BindCraft-style PyRosetta InterfaceAnalyzerMover)
# ---------------------------------------------------------------------------


_PYROSETTA_INITED = False


def _maybe_add_dalphaball(init_opts: str) -> str:
    """
    BindCraft configures BuriedUnsatHbonds with `dalphaball_sasa=1`, which requires
    `-holes:dalphaball <path-to-DAlphaBall>` at PyRosetta init time.

    Best-effort: try common locations in this environment so users don't have to
    manually inject this flag in every config.
    """
    toks = [t for t in str(init_opts).split() if t.strip()]

    if "-holes:dalphaball" in toks:
        i = toks.index("-holes:dalphaball")
        has_path = (i + 1) < len(toks) and not toks[i + 1].startswith("-")
        if has_path:
            return init_opts

    candidates: list[Path] = []
    dalpha = os.environ.get("GH_DALPHABALL", "").strip()
    if dalpha:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(dalpha))))
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "assets" / "DAlphaBall.gcc",
            Path(__file__).resolve().parents[2] / "assets" / "DAlphaBall.gcc",
        ]
    )
    dalpha = next((p for p in candidates if p.exists()), None)
    if dalpha is None:
        return init_opts

    try:
        mode = dalpha.stat().st_mode
        if (mode & 0o100) == 0:
            dalpha.chmod(mode | 0o100)
    except Exception:
        pass

    if "-holes:dalphaball" in toks:
        i = toks.index("-holes:dalphaball")
        toks.insert(i + 1, str(dalpha))
        return " ".join(toks)
    return f"{init_opts} -holes:dalphaball {dalpha}"


def _ensure_pyrosetta(
    init_opts: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> None:
    global _PYROSETTA_INITED
    if _PYROSETTA_INITED:
        return
    import pyrosetta  # type: ignore

    if pyrosetta_was_initialized(pyrosetta):
        _PYROSETTA_INITED = True
        return
    pyrosetta.init(_maybe_add_dalphaball(init_opts))
    _PYROSETTA_INITED = True


def physical_interface_scores(
    structure_path: Path,
    *,
    chain_a: str,
    chain_b: str,
    init_opts: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> dict[str, float]:
    """
    Minimal InterfaceAnalyzer metrics for physical interface optimization.
    """
    _ensure_pyrosetta(init_opts)
    import pyrosetta  # type: ignore

    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover  # type: ignore

    pdb_path = _to_pdb_path(structure_path)
    pose = pyrosetta.pose_from_file(str(pdb_path))

    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{chain_a}_{chain_b}")
    iam.set_scorefunction(pyrosetta.create_score_function("ref2015"))
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interface_data = iam.get_all_data()
    return {
        "interface_sc": float(interface_data.sc_value),
        "dG_separated": float(iam.get_interface_dG()),
        "dSASA_int": float(iam.get_interface_delta_sasa()),
    }


def _bindcraft_interface_scores(
    structure_path: Path,
    *,
    chain_a: str,
    chain_b: str,
    init_opts: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> dict[str, float]:
    """
    Minimal subset of BindCraft `score_interface` outputs used for filtering.
    """
    _ensure_pyrosetta(init_opts)
    import pyrosetta  # type: ignore

    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector  # type: ignore
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover  # type: ignore
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects  # type: ignore

    pdb_path = _to_pdb_path(structure_path)
    pose = pyrosetta.pose_from_file(str(pdb_path))

    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{chain_a}_{chain_b}")
    scorefxn = pyrosetta.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interfacescore = iam.get_all_data()
    interface_sc = float(interfacescore.sc_value)
    interface_interface_hbonds = float(interfacescore.interface_hbonds)
    interface_dG = float(iam.get_interface_dG())
    interface_dSASA = float(iam.get_interface_delta_sasa())
    interface_nres_raw = getattr(interfacescore, "interface_nres", 0)
    # PyRosetta exposes `interface_nres` as a vector1<unsigned long> with entries like:
    #   [total, chain1, chain2, ...]
    # Use the total count (index 1 in Rosetta's 1-based vector).
    try:
        interface_nres = float(interface_nres_raw[1])  # type: ignore[index]
    except Exception:
        try:
            interface_nres = float(interface_nres_raw)  # type: ignore[arg-type]
        except Exception:
            interface_nres = 0.0

    # Buried unsat H-bonds (same RosettaScripts filter snippet used by BindCraft).
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" '
        'ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" '
        'burial_cutoff_apo="0.2" confidence="0" />'
    )
    try:
        interface_delta_unsat_hbonds = float(buns_filter.report_sm(pose))
    except Exception as exc:
        logging.warning("BindCraft buried-unsat calculation failed: %s", exc)
        interface_delta_unsat_hbonds = 1e6

    chain_design = ChainSelector(chain_b)
    tem = pyrosetta.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = float(tem.calculate(pose))

    # Surface hydrophobicity fraction on binder chain (BindCraft-style).
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }.get(str(chain_b))
    surface_hydrophobicity = float("nan")
    if binder_pose is not None:
        layer_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)
        exp_apol_count = 0
        total_count = 0
        for i in range(1, len(surface_res) + 1):
            if surface_res[i] is True:
                res = binder_pose.residue(i)
                if res.is_apolar() is True or res.name() in {"PHE", "TRP", "TYR"}:
                    exp_apol_count += 1
                total_count += 1
        surface_hydrophobicity = (
            exp_apol_count / total_count if total_count > 0 else float("nan")
        )

    return {
        "bindcraft_interface_sc": float(round(interface_sc, 2)),
        "bindcraft_interface_dG": float(round(interface_dG, 2)),
        "bindcraft_interface_dSASA": float(round(interface_dSASA, 2)),
        "bindcraft_interface_nres": float(round(interface_nres, 2)),
        "bindcraft_interface_interface_hbonds": float(round(interface_interface_hbonds, 2)),
        "bindcraft_interface_delta_unsat_hbonds": float(round(interface_delta_unsat_hbonds, 2)),
        "bindcraft_binder_score": float(round(binder_score, 2)),
        "bindcraft_surface_hydrophobicity": float(round(surface_hydrophobicity, 4))
        if isinstance(surface_hydrophobicity, float)
        else 0.0,
    }


def _bindcraft_interface_field(
    eval_result, *, field: str, chain_a: str, chain_b: str, init_opts: str
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        vals = _bindcraft_interface_scores(path, chain_a=chain_a, chain_b=chain_b, init_opts=init_opts)
        scores[oligo_name] = float(vals.get(field, 0.0))
    return scores


@loss_registry.register(
    name="physical_interface_score",
    description="Minimal physical interface score from InterfaceAnalyzerMover: normalized SC improvement minus normalized dG_separated increase.",
    required_params=["chain_a", "chain_b", "baseline_sc", "baseline_dg", "sc_scale", "dg_scale"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def physical_interface_score(
    eval_result,
    chain_a: str,
    chain_b: str,
    baseline_sc: float,
    baseline_dg: float,
    sc_scale: float,
    dg_scale: float,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> dict[str, float]:
    baseline_sc_f = float(baseline_sc)
    baseline_dg_f = float(baseline_dg)
    sc_scale_f = float(sc_scale)
    dg_scale_f = float(dg_scale)
    if sc_scale_f <= 0.0 or dg_scale_f <= 0.0:
        raise ValueError("physical_interface_score requires sc_scale > 0 and dg_scale > 0.")

    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        vals = physical_interface_scores(
            path,
            chain_a=chain_a,
            chain_b=chain_b,
            init_opts=pyrosetta_init,
        )
        scores[oligo_name] = float(
            (vals["interface_sc"] - baseline_sc_f) / sc_scale_f
            - (vals["dG_separated"] - baseline_dg_f) / dg_scale_f
        )
    return scores


def _sap_score_single(
    structure_path: Path,
    *,
    included_protomers: str | None,
    oligo_name: str,
    init_opts: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> float:
    _ensure_pyrosetta(init_opts)
    import pyrosetta as pr  # type: ignore
    from pyrosetta.rosetta.core.io import pose_from_pose  # type: ignore

    pdb_path = _to_pdb_path(structure_path)
    pose = pr.pose_from_file(str(pdb_path))

    pose_to_score = pose
    include = [p.strip() for p in str(included_protomers or "").split(",") if p.strip()]
    if include:
        from astrolabe.utils.mmcif_ids import int_id_to_str_id

        selected_chains = [
            int_id_to_str_id(i + 1) for i, p in enumerate(str(oligo_name)) if p in include
        ]
        if selected_chains:
            if len(selected_chains) == 1:
                chain_id = selected_chains[0]
                chain_pose = {
                    pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
                    for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
                }.get(str(chain_id))
                if chain_pose is not None:
                    pose_to_score = chain_pose
                else:
                    indices = pr.rosetta.utility.vector1_unsigned_long()
                    pdb_info = pose.pdb_info()
                    for i in range(1, pose.size() + 1):
                        if str(pdb_info.chain(i)) == str(chain_id):
                            indices.append(i)
                    if len(indices) > 0:
                        sub_pose = pr.Pose()
                        pose_from_pose(sub_pose, pose, indices)
                        pose_to_score = sub_pose
            else:
                indices = pr.rosetta.utility.vector1_unsigned_long()
                pdb_info = pose.pdb_info()
                sel = {str(c) for c in selected_chains}
                for i in range(1, pose.size() + 1):
                    if str(pdb_info.chain(i)) in sel:
                        indices.append(i)
                if len(indices) > 0:
                    sub_pose = pr.Pose()
                    pose_from_pose(sub_pose, pose, indices)
                    pose_to_score = sub_pose
        else:
            logging.warning(
                "sap_score: included_protomers=%r did not match oligomer %r; scoring full complex (%s).",
                included_protomers,
                oligo_name,
                structure_path,
            )

    # Prefer the core SapScoreMetric (matches RosettaScripts "SapScoreMetric").
    # API varies across PyRosetta builds, so try a couple import paths.
    metric = None
    try:
        from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import (  # type: ignore
            SapScoreMetric as _SapScoreMetric,
        )

        metric = _SapScoreMetric()
    except Exception:
        try:
            metric = pr.rosetta.core.simple_metrics.metrics.SapScoreMetric()  # type: ignore[attr-defined]
        except Exception:
            metric = None

    if metric is not None:
        return float(metric.calculate(pose_to_score))

    # Last-resort fallback: direct function call (not always available / signature may differ).
    try:
        from pyrosetta.rosetta.core.pack.guidance_scoreterms import sap as rosetta_sap  # type: ignore

        return float(rosetta_sap.calculate_sap(pose_to_score))
    except Exception as exc:
        raise RuntimeError(
            "PyRosetta does not expose SapScoreMetric/calculate_sap in this build."
        ) from exc


@loss_registry.register(
    name="sap_score",
    description="Rosetta SAP score (SapScoreMetric). Set included_protomers to restrict to a subset of protomers; empty/None scores the full complex. Lower is better.",
    optional_params=["included_protomers", "pyrosetta_init"],
)
def sap_score(
    eval_result,
    included_protomers: str | None = None,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        try:
            val = _sap_score_single(
                path,
                included_protomers=included_protomers,
                oligo_name=str(oligo_name),
                init_opts=pyrosetta_init,
            )
        except Exception as exc:
            logging.warning("SAP score failed for %s (%s): %s", oligo_name, path, exc)
            val = 1e6
        scores[oligo_name] = float(round(val, 2))
    return scores


@loss_registry.register(
    name="bindcraft_interface_sc",
    description="BindCraft-style interface shape complementarity (InterfaceAnalyzerMover sc_value)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_sc(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_interface_sc", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_interface_dG",
    description="BindCraft-style interface dG (InterfaceAnalyzerMover get_interface_dG)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_dG(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_interface_dG", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_interface_dSASA",
    description="BindCraft-style interface dSASA (InterfaceAnalyzerMover get_interface_delta_sasa)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_dSASA(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_interface_dSASA", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_interface_nres",
    description="BindCraft-aligned interface residue count on binder chain (hotspot-residue definition)",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_nres(eval_result, chain_a: str, chain_b: str):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        summary = _bindcraft_interface_aa_summary(
            str(path), binder_chain=str(chain_b), target_chain=str(chain_a)
        )
        scores[oligo_name] = float(summary.get("bindcraft_interface_nres", 0.0))
    return scores


@loss_registry.register(
    name="bindcraft_interface_interface_hbonds",
    description="BindCraft-style interface hbonds (InterfaceAnalyzerMover interface_hbonds)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_interface_hbonds(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_interface_interface_hbonds", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_interface_delta_unsat_hbonds",
    description="BindCraft-style buried unsatisfied HBonds (RosettaScripts BuriedUnsatHbonds ddG-style)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_delta_unsat_hbonds(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_interface_delta_unsat_hbonds", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_binder_score",
    description="BindCraft-style binder chain energy score (TotalEnergyMetric)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_binder_score(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_binder_score", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_surface_hydrophobicity",
    description="BindCraft-style binder surface hydrophobicity fraction (LayerSelector surface residues)",
    required_params=["chain_a", "chain_b"],
    optional_params=["pyrosetta_init"],
    supports_oligomer_specific=True,
)
def bindcraft_surface_hydrophobicity(
    eval_result,
    chain_a: str,
    chain_b: str,
    pyrosetta_init: str = "-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
):
    return _bindcraft_interface_field(eval_result, field="bindcraft_surface_hydrophobicity", chain_a=chain_a, chain_b=chain_b, init_opts=pyrosetta_init)


@loss_registry.register(
    name="bindcraft_interface_hydrophobicity",
    description="BindCraft-aligned interface hydrophobicity (percent hydrophobic AA among hotspot residues)",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def bindcraft_interface_hydrophobicity(eval_result, chain_a: str, chain_b: str):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        path = Path(eval_result.complex_paths[oligo_name])
        summary = _bindcraft_interface_aa_summary(
            str(path), binder_chain=str(chain_b), target_chain=str(chain_a)
        )
        scores[oligo_name] = float(summary.get("bindcraft_interface_hydrophobicity", 0.0))
    return scores


for _aa in "ACDEFGHIKLMNPQRSTVWY":

    @loss_registry.register(
        name=f"bindcraft_interface_aa_count_{_aa}",
        description=f"BindCraft-aligned interface AA count for {_aa} (hotspot residues)",
        required_params=["chain_a", "chain_b"],
        supports_oligomer_specific=True,
    )
    def _bindcraft_iface_aa_count(eval_result, chain_a: str, chain_b: str, _aa=_aa):  # noqa: B023
        scores: dict[str, float] = {}
        for oligo_name in eval_result.names:
            path = Path(eval_result.complex_paths[oligo_name])
            summary = _bindcraft_interface_aa_summary(
                str(path), binder_chain=str(chain_b), target_chain=str(chain_a)
            )
            scores[oligo_name] = float(summary.get(f"bindcraft_interface_aa_count_{_aa}", 0.0))
        return scores


# ---------------------------------------------------------------------------
# PAE-derived iPAE metrics (Design-metrics convention: mean/min of cross-chain PAE)
# ---------------------------------------------------------------------------


def _stack_pae_matrices(value) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, list):
        mats = value
    else:
        mats = [value]
    out = []
    for m in mats:
        try:
            arr = np.asarray(m, dtype=float)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
            out.append(arr)
    return out


def _chain_residue_indices(structure_path: Path, chain_id: str) -> np.ndarray:
    from astrolabe.utils.file_parser import parse_mmcif, parse_pdb

    path_str = str(structure_path)
    if path_str.endswith(".pdb") or path_str.endswith(".ent"):
        parsed = parse_pdb(path_str)
    else:
        parsed = parse_mmcif(path_str)
    pdb_idx = parsed.get("pdb_idx", [])
    idx = [i for i, (c, _r) in enumerate(pdb_idx) if str(c) == str(chain_id)]
    return np.asarray(idx, dtype=int)


def _ipae_from_pae_matrix(pae: np.ndarray, *, chain_a_idx: np.ndarray, chain_b_idx: np.ndarray) -> tuple[float, float]:
    if chain_a_idx.size == 0 or chain_b_idx.size == 0:
        return 0.0, 0.0
    sub_ab = pae[np.ix_(chain_a_idx, chain_b_idx)]
    sub_ba = pae[np.ix_(chain_b_idx, chain_a_idx)]
    mean_val = float(np.mean([np.mean(sub_ab), np.mean(sub_ba)]))
    min_val = float(min(float(np.min(sub_ab)), float(np.min(sub_ba))))
    return mean_val, min_val


@loss_registry.register(
    name="ipae_mean",
    description="Mean cross-chain PAE (AB+BA mean) from full PAE matrix",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def ipae_mean(eval_result, chain_a: str, chain_b: str):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        mats = _stack_pae_matrices(getattr(eval_result, "pae", {}).get(oligo_name))
        if not mats:
            scores[oligo_name] = 0.0
            continue
        path = Path(eval_result.complex_paths[oligo_name])
        a_idx = _chain_residue_indices(path, str(chain_a))
        b_idx = _chain_residue_indices(path, str(chain_b))
        vals = [_ipae_from_pae_matrix(m, chain_a_idx=a_idx, chain_b_idx=b_idx)[0] for m in mats]
        scores[oligo_name] = float(np.mean(vals))
    return scores


@loss_registry.register(
    name="ipae_min",
    description="Minimum cross-chain PAE across AB/BA submatrices from full PAE matrix",
    required_params=["chain_a", "chain_b"],
    supports_oligomer_specific=True,
)
def ipae_min(eval_result, chain_a: str, chain_b: str):
    scores: dict[str, float] = {}
    for oligo_name in eval_result.names:
        mats = _stack_pae_matrices(getattr(eval_result, "pae", {}).get(oligo_name))
        if not mats:
            scores[oligo_name] = 0.0
            continue
        path = Path(eval_result.complex_paths[oligo_name])
        a_idx = _chain_residue_indices(path, str(chain_a))
        b_idx = _chain_residue_indices(path, str(chain_b))
        vals = [_ipae_from_pae_matrix(m, chain_a_idx=a_idx, chain_b_idx=b_idx)[1] for m in mats]
        scores[oligo_name] = float(np.mean(vals))
    return scores
