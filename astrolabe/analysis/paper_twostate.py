"""Analyze the paper's fixed-sequence apo/holo prediction ensembles.

``analyze_twostate(manifest_path, out_dir)`` returns paths to ``per_design.json``,
``per_design.csv`` and ``per_residue.csv``. CSV input columns are design_id,
method, ligand, state (apo/holo), seed, sample, structure_path, protein_chain,
ligand_chain and iptm (holo only). Optional predictor separates AF3/Protenix2;
if omitted, results are labelled ``unspecified``. Paths are relative to the
manifest. Each state requires five distinct seeds with five samples each.
Protein residue numbers/insertion codes must correspond across predictions.

Optional protein_plddt supplies a 0–100 protein confidence scalar; otherwise
protein confidence is the mean CA B-factor. Residue confidence always uses CA
B-factors on the 0–100 scale. Optional protein_length supplies the modeled
length, required when the numbers of observed protein residues differ.
PDB and mmCIF use the first model and author chain/residue identifiers.

The analysis follows MainPaper Methods 171–177 and SI 29–32: state-specific
iterative Kabsch, a second fit on initially stable residues (RMSF <= 2 A),
then a cross-state fit on mutually stable residues. Iteration stops at mean
anchor movement < 1e-6 A (100 iterations maximum). Undefined geometry and
incomplete ensembles remain output rows with non-passing displacement gates;
no partial ensemble is substituted for five-by-five evaluation. No planned
units absent from this manifest are invented: join an independent planned-unit
manifest before calculating benchmark denominators.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.PDBExceptions import PDBConstructionException


_GROUP = ("design_id", "method", "ligand", "predictor")
_REQUIRED = set(_GROUP[:-1]) | {
    "state", "seed", "sample", "structure_path", "protein_chain", "ligand_chain", "iptm"
}
_CUMULATIVE = (
    "cumulative_holo_plddt", "cumulative_holo_iptm", "cumulative_displacement",
    "cumulative_holo_compactness", "cumulative_apo_plddt",
)
_RESIDUE_COLUMNS = (*_GROUP, "residue_number", "insertion_code", "residue_name",
                    "apo_rmsf", "holo_rmsf", "apo_stable", "holo_stable",
                    "displacement", "holo_plddt", "consensus_ligand_contact",
                    "ordering_candidate", "candidate_region", "qualifying_region")


class _UndefinedGeometry(ValueError):
    pass


@dataclass
class _Prediction:
    seed: str
    sample: str
    coords: dict
    names: dict
    residue_plddt: dict
    contacts: dict
    protein_plddt: float
    iptm: float | None
    rg: float
    length: int


def _number(value: str, name: str, upper: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= upper:
        raise ValueError(f"{name} must be finite and in [0, {upper}]: {value}")
    return result


def _heavy_coords(residue) -> np.ndarray:
    return np.asarray([a.coord for a in residue.get_atoms() if a.element not in {"H", "D"}], dtype=float).reshape(-1, 3)


def _load(row: dict, base: Path) -> _Prediction:
    path = Path(row["structure_path"]).expanduser()
    if not path.is_absolute():
        path = base / path
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Expected PDB or mmCIF structure: {path}")
    model = next(parser.get_structure("prediction", str(path)).get_models())
    if row["protein_chain"] not in model:
        raise ValueError(f"Protein chain {row['protein_chain']!r} absent in {path}")
    residues = [r for r in model[row["protein_chain"]] if is_aa(r, standard=False)]
    ca_residues = [r for r in residues if "CA" in r]
    if not ca_residues:
        raise ValueError(f"No protein CA atoms in {path}")
    coords, names, confidence, contacts = {}, {}, {}, {}
    ligand_coords = None
    if row["state"] == "holo":
        chain = row["ligand_chain"]
        if not chain or chain == row["protein_chain"] or chain not in model:
            raise ValueError(f"A separate ligand_chain is required for holo structure: {path}")
        ligand_coords = np.concatenate([_heavy_coords(r) for r in model[chain]])
        if not len(ligand_coords):
            raise ValueError(f"No ligand heavy atoms in {path}")
    for residue in ca_residues:
        key = (int(residue.id[1]), residue.id[2].strip())
        if key in coords:
            raise ValueError(f"Ambiguous protein residue identifier {key} in {path}")
        coords[key] = np.asarray(residue["CA"].coord, dtype=float)
        names[key] = residue.resname
        confidence[key] = _number(residue["CA"].bfactor, "CA pLDDT", 100)
        if ligand_coords is not None:
            delta = _heavy_coords(residue)[:, None, :] - ligand_coords[None, :, :]
            contacts[key] = bool(np.any(np.sum(delta * delta, axis=-1) <= 4.5 ** 2))
    ca = np.stack(list(coords.values()))
    if not np.isfinite(ca).all():
        raise ValueError(f"Non-finite CA coordinates in {path}")
    return _Prediction(
        seed=row["seed"], sample=row["sample"], coords=coords, names=names,
        residue_plddt=confidence, contacts=contacts,
        protein_plddt=(_number(row["protein_plddt"], "protein_plddt", 100)
                      if row.get("protein_plddt") else float(np.mean(list(confidence.values())))),
        iptm=_number(row["iptm"], "iptm", 1) if row["state"] == "holo" else None,
        rg=float(np.sqrt(np.mean(np.sum((ca - ca.mean(axis=0)) ** 2, axis=1)))),
        length=len(residues),
    )


def _complete(predictions: list[_Prediction]) -> bool:
    seeds = defaultdict(list)
    for prediction in predictions:
        seeds[prediction.seed].append(prediction)
    return len(seeds) == 5 and all(len(samples) == 5 for samples in seeds.values())


def _seed_median(predictions: list[_Prediction], values: np.ndarray) -> np.ndarray:
    return np.median([values[[i for i, p in enumerate(predictions) if p.seed == seed]].mean(axis=0)
                      for seed in sorted({p.seed for p in predictions})], axis=0)


def _fit(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(moving) < 3:
        raise _UndefinedGeometry("fewer than three non-collinear alignment anchors")
    a, b = moving.mean(axis=0), target.mean(axis=0)
    x, y = moving - a, target - b
    if len(x) < 3 or np.linalg.matrix_rank(x, tol=1e-8) < 2 or np.linalg.matrix_rank(y, tol=1e-8) < 2:
        raise _UndefinedGeometry("fewer than three non-collinear alignment anchors")
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.sign(np.linalg.det(u @ vt))
    rotation = u @ vt
    return rotation, b - a @ rotation


def _align_ensemble(coords: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(anchors) < 3:
        raise _UndefinedGeometry("fewer than three stable alignment anchors")
    mean = coords[0].copy()
    for _ in range(100):
        aligned = np.empty_like(coords)
        for i, sample in enumerate(coords):
            rotation, translation = _fit(sample[anchors], mean[anchors])
            aligned[i] = sample @ rotation + translation
        updated = aligned.mean(axis=0)
        movement = np.sqrt(np.mean(np.sum((updated[anchors] - mean[anchors]) ** 2, axis=1)))
        mean = updated
        if movement < 1e-6:
            rmsf = np.sqrt(np.mean(np.sum((aligned - mean) ** 2, axis=2), axis=0))
            return mean, rmsf
    raise _UndefinedGeometry("iterative Kabsch did not converge in 100 iterations")


def _state_geometry(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, initial_rmsf = _align_ensemble(coords, np.arange(coords.shape[1]))
    return _align_ensemble(coords, np.flatnonzero(initial_rmsf <= 2.0))


def _regions(mean: np.ndarray, candidate: np.ndarray, contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(mean), dtype=int)
    qualifies = np.zeros(len(mean), dtype=bool)
    remaining = set(np.flatnonzero(candidate).tolist())
    region = 0
    while remaining:
        region += 1
        start = min(remaining)
        remaining.remove(start)
        component, pending = [start], [start]
        while pending:
            point = pending.pop()
            neighbours = [i for i in sorted(remaining) if np.linalg.norm(mean[i] - mean[point]) <= 8.0]
            remaining.difference_update(neighbours)
            pending.extend(neighbours)
            component.extend(neighbours)
        labels[component] = region
        if len(component) >= 5 and np.sum(contacts[component]) >= 3:
            qualifies[component] = True
    return labels, qualifies


def _analyze_group(key: tuple, rows: list[dict], base: Path) -> tuple[dict, list[dict]]:
    result = dict(zip(_GROUP, key))
    result.update(status="incomplete_ensemble", reason="", apo_samples=0, holo_samples=0,
                  ensemble_complete=False, geometry_defined=False, protein_length=None,
                  shared_residues=0, mutually_stable_residues=None,
                  apo_plddt=None, holo_plddt=None, holo_iptm=None, holo_rg_ratio=None,
                  apo_scalar_plddt_rows=0, holo_scalar_plddt_rows=0,
                  displacement_count=None, ordering_residue_count=None, ordering_region_count=None)
    result.update({name: False for name in _CUMULATIVE})
    states = {state: [] for state in ("apo", "holo")}
    errors = []
    for row in rows:
        try:
            states[row["state"]].append(_load(row, base))
            result[f"{row['state']}_scalar_plddt_rows"] += int(bool(row.get("protein_plddt")))
        except (OSError, ValueError, KeyError, PDBConstructionException) as exc:
            errors.append(f"{row['state']}/{row['seed']}/{row['sample']}: {exc}")
    complete = {state: _complete(predictions) and len(predictions) == sum(r["state"] == state for r in rows)
                for state, predictions in states.items()}
    for state, predictions in states.items():
        result[f"{state}_samples"] = len(predictions)
        if complete[state]:
            result[f"{state}_plddt"] = float(_seed_median(predictions, np.array([p.protein_plddt for p in predictions])))
    apo, holo = states["apo"], states["holo"]
    if complete["holo"]:
        result["holo_iptm"] = float(_seed_median(holo, np.array([p.iptm for p in holo])))
        result[_CUMULATIVE[0]] = result["holo_plddt"] >= 80
        result[_CUMULATIVE[1]] = result[_CUMULATIVE[0]] and result["holo_iptm"] >= 0.70
    if not (complete["apo"] and complete["holo"]):
        result["reason"] = "; ".join(errors or ["each state requires five seeds with five samples per seed"])
        return result, []
    result["ensemble_complete"] = True
    result["status"] = "geometry_undefined"
    all_predictions = apo + holo
    lengths = {int(r["protein_length"]) for r in rows if r.get("protein_length")}
    if len(lengths) > 1 or any(length <= 0 for length in lengths):
        raise ValueError(f"Inconsistent protein_length for {key}")
    if not lengths:
        lengths = {p.length for p in all_predictions}
    if len(lengths) != 1:
        result["status"] = "length_undefined"
        result["reason"] = "observed residue counts differ; provide modeled protein_length"
        return result, []
    length = lengths.pop()
    if length < max(p.length for p in all_predictions):
        raise ValueError(f"protein_length smaller than observed protein for {key}")
    result["protein_length"] = length
    result["holo_rg_ratio"] = float(np.median([p.rg for p in holo]) / (2.2 * length ** 0.38))
    shared = sorted(set.intersection(*(set(p.coords) for p in all_predictions)))
    result["shared_residues"] = len(shared)
    if len(shared) < 3:
        result["reason"] = "fewer than three shared CA residues"
        return result, []
    if any(len({p.names[k] for p in all_predictions}) != 1 for k in shared):
        result["status"] = "sequence_mismatch"
        result["reason"] = "residue identity differs across fixed-sequence ensembles"
        return result, []
    residue_confidence = _seed_median(holo, np.array([[p.residue_plddt[k] for k in shared] for p in holo]))
    sample_contacts = np.array([[p.contacts[k] for k in shared] for p in holo])
    contact_votes = np.array([sample_contacts[[i for i, p in enumerate(holo) if p.seed == seed]].sum(axis=0) >= 3
                              for seed in sorted({p.seed for p in holo})])
    contacts = contact_votes.sum(axis=0) >= 4
    residue_rows = [dict(zip(_GROUP, key), residue_number=k[0], insertion_code=k[1],
                         residue_name=holo[0].names[k], apo_rmsf=None, holo_rmsf=None,
                         apo_stable=None, holo_stable=None, displacement=None,
                         holo_plddt=float(residue_confidence[i]), consensus_ligand_contact=bool(contacts[i]),
                         ordering_candidate=None, candidate_region=None, qualifying_region=None)
                    for i, k in enumerate(shared)]
    try:
        apo_mean, apo_rmsf = _state_geometry(np.array([[p.coords[k] for k in shared] for p in apo]))
        holo_mean, holo_rmsf = _state_geometry(np.array([[p.coords[k] for k in shared] for p in holo]))
        for i, row in enumerate(residue_rows):
            row.update(apo_rmsf=float(apo_rmsf[i]), holo_rmsf=float(holo_rmsf[i]),
                       apo_stable=bool(apo_rmsf[i] <= 2), holo_stable=bool(holo_rmsf[i] <= 2))
        stable = (apo_rmsf <= 2) & (holo_rmsf <= 2)
        result["mutually_stable_residues"] = int(stable.sum())
        rotation, translation = _fit(apo_mean[stable], holo_mean[stable])
    except _UndefinedGeometry as exc:
        result["reason"] = str(exc)
        return result, residue_rows
    displacement = np.linalg.norm(apo_mean @ rotation + translation - holo_mean, axis=1)
    candidate = (apo_rmsf > 3) & (holo_rmsf <= 2) & (displacement > 3) & (residue_confidence >= 70)
    regions, qualifies = _regions(holo_mean, candidate, contacts)
    for i, row in enumerate(residue_rows):
        row.update(displacement=float(displacement[i]), ordering_candidate=bool(candidate[i]),
                   candidate_region=int(regions[i]), qualifying_region=bool(qualifies[i]))
    result.update(status="ok", geometry_defined=True,
                  displacement_count=int(np.sum(stable & (displacement > 2))),
                  ordering_residue_count=int(qualifies.sum()),
                  ordering_region_count=len(set(regions[qualifies].tolist())))
    result[_CUMULATIVE[2]] = result[_CUMULATIVE[1]] and result["displacement_count"] >= 3
    result[_CUMULATIVE[3]] = result[_CUMULATIVE[2]] and result["holo_rg_ratio"] <= 1.30
    result[_CUMULATIVE[4]] = result[_CUMULATIVE[3]] and result["apo_plddt"] >= 70
    return result, residue_rows


def _write_csv(path: Path, rows: list[dict], columns) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze_twostate(manifest_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write per-design and per-residue results; return output-name/path mapping."""
    manifest_path, out_dir = Path(manifest_path).expanduser().resolve(), Path(out_dir).expanduser()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest columns: {', '.join(sorted(missing))}")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError("The structure manifest contains no prediction rows")
    if any(r.get("predictor") for r in rows) and any(not r.get("predictor") for r in rows):
        raise ValueError("predictor must be specified for every row when used")
    groups, seen = defaultdict(list), set()
    for row in rows:
        row["predictor"] = row.get("predictor") or "unspecified"
        if row["state"] not in {"apo", "holo"}:
            raise ValueError(f"Invalid state: {row['state']!r}; expected apo or holo")
        if any(not row[name] for name in (*_GROUP, "seed", "sample", "structure_path", "protein_chain")):
            raise ValueError("Group identifiers, seed, sample, structure_path and protein_chain must be nonempty")
        key = tuple(row[name] for name in _GROUP)
        sample_key = (*key, row["state"], row["seed"], row["sample"])
        if sample_key in seen:
            raise ValueError(f"Duplicate prediction identity: {sample_key}")
        seen.add(sample_key)
        groups[key].append(row)
    designs, residues = [], []
    for key, group in sorted(groups.items()):
        result, residue_rows = _analyze_group(key, group, manifest_path.parent)
        designs.append(result)
        residues.extend(residue_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"design_json": out_dir / "per_design.json", "design_csv": out_dir / "per_design.csv",
               "residue_csv": out_dir / "per_residue.csv"}
    outputs["design_json"].write_text(json.dumps({"protocol": "MainPaper Methods 171-177; SI 29-32",
        "analysis_version": 1,
        "implementation_conventions": {"kabsch_convergence_angstrom": 1e-6, "kabsch_max_iterations": 100,
            "noncollinearity_rank_tolerance_angstrom": 1e-8,
            "protein_plddt": "protein_plddt CSV scalar when supplied; otherwise mean protein CA B-factor",
            "residue_plddt": "CA B-factor (0-100)",
            "structure_model": "first", "residue_identifiers": "author residue number and insertion code"},
        "manifest": str(manifest_path), "prediction_rows": len(rows), "designs": designs}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(outputs["design_csv"], designs, tuple(designs[0]))
    _write_csv(outputs["residue_csv"], residues, _RESIDUE_COLUMNS)
    return outputs
