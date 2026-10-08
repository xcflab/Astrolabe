"""Fixed-design AF3/Protenix agreement and cross-predictor confidence analysis.

The CSV manifest has one row per (method, group, design_id, side). Required
columns are those four identifiers, target_sequence, binder_sequence,
af3_structure and protenix_structure. Structure paths are relative to the CSV.
Optional comparable=false excludes a design from geometry only; retain an
exclusion_reason, e.g. different target constructs. Missing geometry never
becomes disagreement. Scores must belong to the listed selected structures.

``columns`` maps canonical manifest names to actual headers. ``confidence_columns``
maps predictor labels to per-side ipTM columns; defaults are AF3 -> af3_iptm and
Protenix -> protenix_iptm. Additional predictors, e.g. ESMFold2, need no structures.
AF3 and Protenix labels are required. Empty confidence values remain missing.
All scores are on the 0–1 scale. Diffusion samples are not manifest rows.

Geometry follows 013_paper_writing's 23_multitarget_binding_modes.py and
09_cross_benchmark_binding_modes/01_binding_mode_analysis.py: exact sequences,
untrimmed Kabsch on reference-interface target CAs, heavy-atom residue contacts,
then maximum two-side RMSDs / minimum contact recall. The final source requires
four designs per correlation and two estimable groups per method mean. NumPy
average tied ranks replace scipy.stats.rankdata without adding dependencies.
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, PDBParser


_REQUIRED = ("design_id", "method", "group", "side", "target_sequence", "binder_sequence",
             "af3_structure", "protenix_structure")
_OPTIONAL = ("comparable", "exclusion_reason")
_METRICS = ("pose_rmsd", "contact_recall", "fold_rmsd")
_CONFIDENCE = {"AF3": "af3_iptm", "Protenix": "protenix_iptm"}


def _fit(moving, reference):
    moving, reference = np.asarray(moving, dtype=float), np.asarray(reference, dtype=float)
    cm, cr = moving.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - cm).T @ (reference - cr))
    u[:, -1] *= np.sign(np.linalg.det(u @ vt))
    rotation = u @ vt
    translation = cr - cm @ rotation
    return rotation, translation


def _rmsd(left, right):
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def _structure(path, sequences):
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Expected a PDB or mmCIF structure: {path}")
    model = next(parser.get_structure("complex", str(path)).get_models())
    chains = []
    for chain in model:
        residues = [residue for residue in chain if "CA" in residue]
        sequence = "".join(protein_letters_3to1_extended.get(r.resname.upper(), "X") for r in residues)
        chains.append((chain.id, sequence, residues))
    selected = []
    for sequence in sequences:
        matches = [record for record in chains if record[1] == sequence]
        if len(matches) != 1:
            raise ValueError(f"{path}: exact sequence matches={len(matches)} for length {len(sequence)}; expected one")
        chain_id, _, residues = matches[0]
        cas, heavy_atoms, residue_indices = [], [], []
        for index, residue in enumerate(residues):
            atoms = [atom for atom in residue.get_unpacked_list() if atom.altloc in (" ", "A")]
            ca = next((atom for atom in atoms if atom.name == "CA"), None)
            if ca is None:
                raise ValueError(f"{path}: no primary CA at chain {chain_id} sequence position {index + 1}")
            cas.append(ca.coord)
            for atom in atoms:
                if atom.element.upper() not in {"H", "D"}:
                    heavy_atoms.append(atom.coord)
                    residue_indices.append(index)
        coordinates = np.asarray(heavy_atoms, dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"Non-finite coordinates in {path}, chain {chain_id}")
        selected.append({"chain": chain_id, "ca": np.asarray(cas, dtype=float),
                         "atoms": coordinates, "residues": np.asarray(residue_indices, dtype=int)})
    target, binder = selected
    if target["chain"] == binder["chain"]:
        raise ValueError(f"Target and binder must identify different chains in {path}")
    contacts = {5: set(), 10: set()}
    # Bound the temporary distance matrix while preserving inclusive atom cutoffs.
    for start in range(0, len(target["atoms"]), 256):
        difference = target["atoms"][start:start + 256, None, :] - binder["atoms"][None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", difference, difference)
        for cutoff in contacts:
            a, b = np.where(distance2 <= cutoff ** 2)
            contacts[cutoff].update(zip(target["residues"][start + a].tolist(), binder["residues"][b].tolist()))
    return target, binder, contacts


def compare_binding_modes(
    af3_structure: str | Path,
    protenix_structure: str | Path,
    *,
    target_sequence: str,
    binder_sequence: str,
) -> dict:
    """Compare a fixed binary complex; RMSDs are CA-based and in Angstrom.

AF3 target residues with any heavy atom within 10 A of an AF3 binder heavy atom
define local target anchors. No coordinate outliers are removed. Contact recall
uses AF3 contacts as denominator. A missing local fit or zero reference contacts
returns None for that metric, never an invented disagreement value.
"""
    if not target_sequence or not binder_sequence:
        raise ValueError("Exact target_sequence and binder_sequence are required")
    rt, rb, rc = _structure(af3_structure, (target_sequence, binder_sequence))
    mt, mb, mc = _structure(protenix_structure, (target_sequence, binder_sequence))
    rotation, translation = _fit(mb["ca"], rb["ca"])
    anchors = sorted({target for target, _ in rc[10]})
    result = {"pose_rmsd": None, "fold_rmsd": _rmsd(mb["ca"] @ rotation + translation, rb["ca"]),
              "contact_recall": len(rc[5] & mc[5]) / len(rc[5]) if rc[5] else None,
              "af3_contacts": len(rc[5]), "protenix_contacts": len(mc[5]),
              "shared_contacts": len(rc[5] & mc[5]), "local_target_ca_count": len(anchors),
              "af3_target_chain": rt["chain"], "af3_binder_chain": rb["chain"],
              "protenix_target_chain": mt["chain"], "protenix_binder_chain": mb["chain"]}
    if len(anchors) >= 3 and np.linalg.matrix_rank(rt["ca"][anchors] - rt["ca"][anchors].mean(axis=0)) >= 2:
        rotation, translation = _fit(mt["ca"][anchors], rt["ca"][anchors])
        result["pose_rmsd"] = _rmsd(mb["ca"] @ rotation + translation, rb["ca"])
    return result


def _rank_rows(values):
    """Average tied ranks, independently for every row (equivalent to rankdata)."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    positions = np.broadcast_to(np.arange(values.shape[1]), values.shape)
    starts = np.concatenate((np.ones((len(values), 1), dtype=bool), ordered[:, 1:] != ordered[:, :-1]), axis=1)
    ends = np.concatenate((ordered[:, :-1] != ordered[:, 1:], np.ones((len(values), 1), dtype=bool)), axis=1)
    first = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, positions, values.shape[1] - 1)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty_like(values)
    np.put_along_axis(ranks, order, (first + last) / 2 + 1, axis=1)
    return ranks


def _rank_correlations(x, y):
    x, y = _rank_rows(x), _rank_rows(y)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(np.sum(x * y, axis=1), denominator,
                     out=np.full(len(x), np.nan), where=denominator > 0)


def _confidence(value, field):
    if value == "":
        return None
    score = float(value)
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{field} must be finite ipTM on the 0–1 scale, got {value!r}")
    return score


def _comparable(value):
    if value.lower() in {"", "true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise ValueError(f"comparable must be true/false or 1/0, got {value!r}")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_agreement(
    manifest: str | Path,
    out_dir: str | Path,
    *,
    columns: dict[str, str] | None = None,
    confidence_columns: dict[str, str] | None = None,
    astrolabe_method: str = "Astrolabe",
    expected_sides: int = 2,
    bootstrap_samples: int = 2000,
    seed: int = 20260930,
) -> dict[str, str]:
    """Write side/design geometry, group/method estimates, distributions and provenance.

Bootstrap units are complete designs after side reduction, sampled independently
within method and group. Constant bootstrap draws remain undefined; method CIs
use draws estimable in every included group. Groups receive equal weights.
Methods/groups are visited in first-occurrence CSV order, recorded in metadata.
"""
    manifest, out_dir = Path(manifest).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    if expected_sides < 1 or bootstrap_samples < 1:
        raise ValueError("expected_sides and bootstrap_samples must be positive")
    mapping = {key: key for key in _REQUIRED + _OPTIONAL}
    if set(columns or {}) - set(mapping):
        raise ValueError("Unknown canonical manifest columns: " + ", ".join(sorted(set(columns) - set(mapping))))
    mapping.update(columns or {})
    predictors = {**_CONFIDENCE, **(confidence_columns or {})}
    sides, by_design, methods, groups = [], defaultdict(list), [], []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {mapping[key] for key in _REQUIRED} | set(predictors.values())
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing manifest columns: " + ", ".join(sorted(missing)))
        for number, raw in enumerate(reader, 2):
            row = {key: (raw.get(header) or "").strip() for key, header in mapping.items()}
            if any(not row[key] for key in ("design_id", "method", "group", "side", "target_sequence", "binder_sequence")):
                raise ValueError(f"Manifest row {number} requires identifiers and exact target/binder sequences")
            row["comparable"] = _comparable(row["comparable"])
            for label, header in predictors.items():
                row[f"confidence_{label}"] = _confidence((raw.get(header) or "").strip(), header)
            for key in ("af3_structure", "protenix_structure"):
                if row[key]:
                    path = Path(row[key]).expanduser()
                    row[key] = str(path if path.is_absolute() else manifest.parent / path)
            key = row["method"], row["group"], row["design_id"]
            if any(previous["side"] == row["side"] for previous in by_design[key]):
                raise ValueError(f"Duplicate design side {key + (row['side'],)}; select one native structure per side first")
            for field, order in (("method", methods), ("group", groups)):
                if row[field] not in order:
                    order.append(row[field])
            row.update(dict.fromkeys(_METRICS))
            row["detail"] = ""
            if not row["comparable"]:
                row["geometry_status"] = "not_comparable"
            elif not all(row[key] and Path(row[key]).is_file() for key in ("af3_structure", "protenix_structure")):
                row["geometry_status"] = "missing_structure"
            else:
                try:
                    row.update(compare_binding_modes(row["af3_structure"], row["protenix_structure"],
                               target_sequence=row["target_sequence"], binder_sequence=row["binder_sequence"]))
                    row["geometry_status"] = "complete" if all(row[m] is not None for m in _METRICS) else "undefined_geometry"
                except (OSError, ValueError) as error:
                    row["geometry_status"] = "structure_or_sequence_mismatch"
                    row["detail"] = str(error)
            sides.append(row)
            by_design[key].append(row)
    if not sides:
        raise ValueError("Manifest contains no design sides")

    designs = []
    for (method, group, design_id), records in by_design.items():
        if len({row["binder_sequence"] for row in records}) != 1:
            raise ValueError(f"Different binder sequences across sides of {(method, group, design_id)}")
        if len(records) > expected_sides:
            raise ValueError(f"Too many target sides for {(method, group, design_id)}")
        row = {"method": method, "group": group, "design_id": design_id,
               "binder_sequence": records[0]["binder_sequence"], "source_sides": len(records),
               "expected_sides": expected_sides, "comparable": all(r["comparable"] for r in records),
               "geometry_sides": sum(r["geometry_status"] == "complete" for r in records),
               "exclusion_reason": "; ".join(dict.fromkeys(r["exclusion_reason"] for r in records if r["exclusion_reason"]))}
        row.update(dict.fromkeys(_METRICS))
        if not row["comparable"]:
            row["geometry_status"] = "not_comparable"
        elif len(records) == expected_sides and row["geometry_sides"] == expected_sides:
            row["geometry_status"] = "complete"
            for metric in _METRICS:
                row[metric] = (min if metric == "contact_recall" else max)(r[metric] for r in records)
        else:
            row["geometry_status"] = "incomplete"
        for label in predictors:
            values = [r[f"confidence_{label}"] for r in records]
            row[f"confidence_{label}"] = min(values) if len(values) == expected_sides and all(v is not None for v in values) else None
        designs.append(row)

    rng = np.random.default_rng(seed)
    group_stats, method_stats, strata = [], [], []
    for method in methods:
        all_method = [r for r in designs if r["method"] == method]
        valid = [r for r in all_method if r["geometry_status"] == "complete" and r["confidence_Protenix"] is not None]
        for group in [*groups, "ALL"]:
            members = valid if group == "ALL" else [r for r in valid if r["group"] == group]
            for label in ("low", "intermediate", "high"):
                chosen = [r for r in members if ("low" if r["confidence_Protenix"] <= .6 else
                          "intermediate" if r["confidence_Protenix"] <= .8 else "high") == label]
                strata.append({"method": method, "group": group, "stratum": label, "n": len(chosen),
                               "median_contact_recall": float(np.median([r["contact_recall"] for r in chosen])) if chosen else None})
        for metric in _METRICS:
            estimates, bootstraps = [], []
            for group in groups:
                all_group = [r for r in all_method if r["group"] == group]
                members = [r for r in valid if r["group"] == group]
                x = np.asarray([r["confidence_Protenix"] for r in members])
                y = np.asarray([r[metric] for r in members])
                entry = {"method": method, "group": group, "metric": metric, "n_manifest": len(all_group),
                         "n_not_comparable": sum(not r["comparable"] for r in all_group), "n": len(members),
                         "spearman": None, "status": "not_estimable", "ci_low": None, "ci_high": None,
                         "valid_bootstrap_replicates": 0}
                if len(x) >= 4 and np.unique(x).size > 1 and np.unique(y).size > 1:
                    entry.update(spearman=float(_rank_correlations(x, y)[0]), status="estimated")
                    indices = rng.integers(0, len(x), (bootstrap_samples, len(x)))
                    bootstrap = _rank_correlations(x[indices], y[indices])
                    finite = bootstrap[np.isfinite(bootstrap)]
                    entry.update(ci_low=float(np.quantile(finite, .025)) if len(finite) else None,
                                 ci_high=float(np.quantile(finite, .975)) if len(finite) else None,
                                 valid_bootstrap_replicates=len(finite))
                    estimates.append(entry["spearman"])
                    bootstraps.append(bootstrap)
                group_stats.append(entry)
            combined = np.mean(bootstraps, axis=0) if len(bootstraps) >= 2 else np.asarray([])
            finite = combined[np.isfinite(combined)]
            method_stats.append({"method": method, "metric": metric, "n": len(valid),
                "groups_estimable": len(estimates), "groups_in_manifest": len(groups),
                "spearman": float(np.mean(estimates)) if len(estimates) >= 2 else None,
                "ci_low": float(np.quantile(finite, .025)) if len(finite) else None,
                "ci_high": float(np.quantile(finite, .975)) if len(finite) else None,
                "valid_bootstrap_replicates": len(finite), "status": "estimated" if len(estimates) >= 2 else "not_estimable"})

    wins = []
    for baseline in (m for m in methods if m != astrolabe_method):
        for predictor in predictors:
            probabilities = []
            for group in groups:
                a = np.sort([r[f"confidence_{predictor}"] for r in designs if r["method"] == astrolabe_method and r["group"] == group and r[f"confidence_{predictor}"] is not None])
                b = np.sort([r[f"confidence_{predictor}"] for r in designs if r["method"] == baseline and r["group"] == group and r[f"confidence_{predictor}"] is not None])
                value = None
                if len(a) and len(b):
                    # Mann–Whitney ordering probability, with half weight for exact ties.
                    value = float(np.mean((np.searchsorted(b, a, side="left") + np.searchsorted(b, a, side="right")) / (2 * len(b))))
                    probabilities.append(value)
                wins.append({"astrolabe_method": astrolabe_method, "baseline": baseline, "predictor": predictor,
                             "group": group, "n_astrolabe": len(a), "n_baseline": len(b), "groups_compared": int(value is not None), "win_probability": value})
            wins.append({"astrolabe_method": astrolabe_method, "baseline": baseline, "predictor": predictor,
                         "group": "ALL", "groups_compared": len(probabilities),
                         "win_probability": float(np.mean(probabilities)) if len(probabilities) == len(groups) else None})

    overlap = []
    for method in methods:
        for group in [*groups, "ALL"]:
            members = [r for r in designs if r["method"] == method and (group == "ALL" or r["group"] == group)]
            for first, second in itertools.combinations(predictors, 2):
                matched = [r for r in members if r[f"confidence_{first}"] is not None and r[f"confidence_{second}"] is not None]
                high_first = sum(r[f"confidence_{first}"] > .8 for r in matched)
                high_second = sum(r[f"confidence_{second}"] > .8 for r in matched)
                both = sum(r[f"confidence_{first}"] > .8 and r[f"confidence_{second}"] > .8 for r in matched)
                overlap.append({"method": method, "group": group, "first_predictor": first, "second_predictor": second,
                    "n_manifest": len(members), "n_matched": len(matched), "n_first_high": high_first,
                    "n_second_high": high_second, "n_both_high": both, "n_only_first": high_first - both,
                    "n_only_second": high_second - both, "n_neither": len(matched) - high_first - high_second + both,
                    "retained_of_first_high": both / high_first if high_first else None,
                    "retained_of_second_high": both / high_second if high_second else None})

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = [("sides", "01_sides.csv", sides), ("designs", "02_designs.csv", designs),
              ("group_correlations", "03_group_correlations.csv", group_stats),
              ("method_correlations", "04_method_correlations.csv", method_stats),
              ("confidence_strata", "05_confidence_strata.csv", strata),
              ("win_probabilities", "06_win_probabilities.csv", wins),
              ("confidence_overlap", "07_confidence_overlap.csv", overlap)]
    result = {}
    for key, filename, rows in tables:
        path = out_dir / filename
        fields = list(dict.fromkeys(k for row in rows for k in row))
        if not fields:
            fields = ["astrolabe_method", "baseline", "predictor", "group", "win_probability"]
        _write_csv(path, rows, fields)
        result[key] = str(path)
    metadata = {"manifest": str(manifest), "columns": mapping, "confidence_columns": predictors,
        "astrolabe_method": astrolabe_method, "method_order": methods, "group_order": groups,
        "expected_sides": expected_sides, "side_status_counts": dict(Counter(r["geometry_status"] for r in sides)),
        "design_status_counts": dict(Counter(r["geometry_status"] for r in designs)),
        "excluded_designs": [{k: r[k] for k in ("method", "group", "design_id", "exclusion_reason")} for r in designs if not r["comparable"]],
        "geometry": {"sequence_matching": "exact_full_sequence", "target_alignment": "AF3 heavy-atom interface within inclusive 10 Angstrom; target CA Kabsch; no outlier trimming",
                     "contacts": "any non-H/non-D atom pair <=5 Angstrom, residue-pair recall relative to AF3",
                     "side_reduction": "maximum pose/fold RMSD; minimum contact recall and predictor confidence"},
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "design after side aggregation",
                      "strata": "within method and target group", "minimum_group_n": 4, "minimum_estimable_groups": 2,
                      "method_statistic": "equal-weight mean of estimable group Spearman correlations",
                      "interval": "2.5/97.5 percentiles of draws finite in every included group"},
        "missingness": "Missing geometry is excluded from agreement, never counted as disagreement. comparable=false affects geometry only. Missing score pairs remain outside matched confidence comparisons.",
        "confidence_strata": {"low": "<=0.6", "intermediate": "(0.6,0.8]", "high": ">0.8"},
        "win_probability": "P(Astrolabe > baseline) + 0.5 P(tie), equally weighted across all manifest target groups; undefined if any group lacks scores",
        "overlap_threshold": ">0.8 for each side under each predictor", "dependencies": ["numpy", "biopython"],
        "sources": ["013_paper_writing/analysis/002_final_candidate_reevaluation/01_cross_model_common_cohort/01_common_candidate_panel/23_multitarget_binding_modes.py",
                    "013_paper_writing/figures_workflow/src/manuscript_figures/figures/supp_multitarget_orthogonal.py"],
        "interpretation": "Cross-predictor agreement, not experimental correctness or calibrated predictor equivalence."}
    path = out_dir / "08_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    result["metadata"] = str(path)
    return result
