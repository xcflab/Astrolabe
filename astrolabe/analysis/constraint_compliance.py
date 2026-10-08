from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from astrolabe.core.binding_site import BindingSiteSpec, parse_binding_site


@dataclass(frozen=True)
class ParsedRun:
    run_dir: Path
    evaluator: str
    mode: str
    case_id: str
    task_name: str
    contact_supported: bool
    spec: BindingSiteSpec


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _to_float(value, default=float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _to_int(value, default=0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _load_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return MMCIFParser(QUIET=True).get_structure("x", str(path))
    if suffix == ".pdb":
        return PDBParser(QUIET=True).get_structure("x", str(path))
    raise ValueError(f"Unsupported structure format: {path}")


def _iter_polymer_resseq(chain) -> list[int]:
    out: list[int] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        out.append(int(residue.id[1]))
    return out


def _residue_coords(chain, positions: Iterable[int], atom_mode: str) -> np.ndarray:
    selected = set(int(p) for p in positions)
    coords: list[np.ndarray] = []
    for residue in chain:
        if residue.id[0] != " " or int(residue.id[1]) not in selected:
            continue
        if atom_mode == "ca":
            if "CA" not in residue:
                continue
            c = np.asarray(residue["CA"].get_coord(), dtype=float)
            if np.isfinite(c).all():
                coords.append(c.reshape(1, 3))
            continue
        if atom_mode != "all":
            raise ValueError(f"Unsupported atom mode: {atom_mode}")
        residue_coords = []
        for atom in residue.get_atoms():
            c = np.asarray(atom.get_coord(), dtype=float)
            if np.isfinite(c).all():
                residue_coords.append(c)
        if residue_coords:
            coords.append(np.asarray(residue_coords, dtype=float))
    if not coords:
        return np.zeros((0, 3), dtype=float)
    if len(coords) == 1:
        return coords[0]
    return np.concatenate(coords, axis=0)


def _min_distance_between_coords(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("inf")
    chunk = 2048
    best = float("inf")
    for start in range(0, left.shape[0], chunk):
        a = left[start : start + chunk]
        d = np.linalg.norm(a[:, None, :] - right[None, :, :], axis=-1)
        local = float(np.min(d))
        if local < best:
            best = local
    return best


def _min_distance_between_sets(model, chain_left: str, left_positions: Iterable[int], chain_right: str, right_positions: Iterable[int], atom_mode: str) -> float:
    try:
        left_chain = model[chain_left]
        right_chain = model[chain_right]
    except Exception:
        return float("inf")
    left = _residue_coords(left_chain, left_positions, atom_mode=atom_mode)
    right = _residue_coords(right_chain, right_positions, atom_mode=atom_mode)
    return _min_distance_between_coords(left, right)


def _contact_pair_metrics(model, spec: BindingSiteSpec, atom_mode: str) -> dict[str, float]:
    if not spec.binder.positions:
        return {
            f"{atom_mode}_contact_pair_count": 0.0,
            f"{atom_mode}_contact_satisfied_fraction": float("nan"),
            f"{atom_mode}_contact_missing_pairs": 0.0,
            f"{atom_mode}_contact_min_dist": float("nan"),
            f"{atom_mode}_contact_p10_dist": float("nan"),
            f"{atom_mode}_contact_p50_dist": float("nan"),
        }

    pair_dists: list[float] = []
    satisfied = 0
    lower = float(spec.min_distance)
    upper = float(spec.max_distance)
    for bpos in spec.binder.positions:
        for tpos in spec.target.positions:
            d = _min_distance_between_sets(
                model,
                spec.binder.chain,
                [int(bpos)],
                spec.target.chain,
                [int(tpos)],
                atom_mode=atom_mode,
            )
            pair_dists.append(d)
            if math.isfinite(d) and lower <= d <= upper:
                satisfied += 1

    arr = np.asarray(pair_dists, dtype=float)
    finite = arr[np.isfinite(arr)]
    total = float(arr.size)
    missing = float(total - finite.size)
    sat_frac = float(satisfied / total) if total > 0 else float("nan")
    if finite.size == 0:
        min_d = p10 = p50 = float("nan")
    else:
        min_d = float(np.min(finite))
        p10 = float(np.percentile(finite, 10))
        p50 = float(np.percentile(finite, 50))
    return {
        f"{atom_mode}_contact_pair_count": total,
        f"{atom_mode}_contact_satisfied_fraction": sat_frac,
        f"{atom_mode}_contact_missing_pairs": missing,
        f"{atom_mode}_contact_min_dist": min_d,
        f"{atom_mode}_contact_p10_dist": p10,
        f"{atom_mode}_contact_p50_dist": p50,
    }


def compute_geometry_metrics(structure_path: Path, spec: BindingSiteSpec) -> dict[str, float]:
    structure = _load_structure(structure_path)
    model = next(structure.get_models())
    try:
        binder_chain = model[spec.binder.chain]
    except Exception:
        binder_chain = None

    all_binder_positions = _iter_polymer_resseq(binder_chain) if binder_chain is not None else []
    if spec.binder.positions:
        allowed_positions = list(int(p) for p in spec.binder.positions)
    else:
        allowed_positions = list(all_binder_positions)

    allowed_set = set(allowed_positions)
    framework_positions = [p for p in all_binder_positions if p not in allowed_set]
    target_positions = list(int(p) for p in spec.target.positions)

    out: dict[str, float] = {}
    for atom_mode, prefix in (("ca", "ca"), ("all", "aa")):
        any_min = _min_distance_between_sets(
            model,
            spec.binder.chain,
            all_binder_positions,
            spec.target.chain,
            target_positions,
            atom_mode=atom_mode,
        )
        allowed_min = _min_distance_between_sets(
            model,
            spec.binder.chain,
            allowed_positions,
            spec.target.chain,
            target_positions,
            atom_mode=atom_mode,
        )
        framework_min = _min_distance_between_sets(
            model,
            spec.binder.chain,
            framework_positions,
            spec.target.chain,
            target_positions,
            atom_mode=atom_mode,
        )
        out[f"{prefix}_any_binder_min_dist"] = float(any_min)
        out[f"{prefix}_allowed_min_dist"] = float(allowed_min)
        out[f"{prefix}_framework_min_dist"] = float(framework_min)
        out[f"{prefix}_any_binder_pass"] = 1.0 if any_min <= spec.max_distance else 0.0
        out[f"{prefix}_allowed_pass"] = 1.0 if allowed_min <= spec.max_distance else 0.0
        out.update(_contact_pair_metrics(model, spec, atom_mode=atom_mode))

    return out


def _load_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pick_final_accepted(rows: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    if not rows:
        raise ValueError("scores.csv is empty")
    accepted = [r for r in rows if _to_bool(r.get("accepted"))]
    source = accepted if accepted else rows
    final = max(source, key=lambda r: _to_int(r.get("step"), default=0))
    step = _to_int(final.get("step"), default=0)
    return step, final


def _extract_metric(row: dict[str, str], names: list[str]) -> float:
    for name in names:
        if name in row:
            return _to_float(row.get(name))
    return float("nan")


def _resolve_structure(run_dir: Path, step: int) -> Path | None:
    structures_dir = run_dir / "structures"
    preferred = structures_dir / f"step_{step:03d}_AB_model.cif"
    if preferred.exists():
        return preferred
    fallback = sorted(structures_dir.glob(f"step_{step:03d}_*_model.cif"))
    if fallback:
        return fallback[0]
    return None


def _read_run_success(run_dir: Path) -> bool:
    info_log = run_dir / "logs" / "info.log"
    if not info_log.exists():
        return False
    text = info_log.read_text(encoding="utf-8", errors="ignore")
    return "Pipeline completed successfully" in text


def _parse_manifest_row(row: dict[str, str]) -> ParsedRun:
    binder_field = str(row.get("binder", "")).strip()
    if ";" in binder_field:
        binder = [part.strip() for part in binder_field.split(";") if part.strip()]
    else:
        binder = binder_field
    cfg = {
        "binder": binder,
        "target": row.get("target"),
        "mode": row.get("mode", "pocket"),
        "max_distance": _to_float(row.get("max_distance"), default=12.0),
        "min_distance": _to_float(row.get("min_distance"), default=0.0),
    }
    spec = parse_binding_site(cfg)
    if spec is None:
        raise ValueError(f"Invalid binding_site in manifest row: {row}")
    run_dir_text = str(row.get("run_dir", "")).strip()
    run_dir = (
        Path(run_dir_text).expanduser().resolve()
        if run_dir_text
        else Path("/__missing_run_dir__")
    )
    return ParsedRun(
        run_dir=run_dir,
        evaluator=str(row["evaluator"]).strip(),
        mode=str(row["mode"]).strip(),
        case_id=str(row["case_id"]).strip(),
        task_name=str(row["task_name"]).strip(),
        contact_supported=_to_bool(row.get("contact_supported", "true")),
        spec=spec,
    )


def evaluate_run(parsed: ParsedRun) -> dict[str, object]:
    out: dict[str, object] = {
        "case_id": parsed.case_id,
        "task_name": parsed.task_name,
        "evaluator": parsed.evaluator,
        "mode": parsed.mode,
        "contact_supported": int(parsed.contact_supported),
        "run_dir": str(parsed.run_dir),
    }
    out["run_success"] = int(_read_run_success(parsed.run_dir))

    scores_path = parsed.run_dir / "scores.csv"
    if not scores_path.exists():
        out["status"] = "missing_scores"
        return out

    rows = _load_scores(scores_path)
    step, final_row = _pick_final_accepted(rows)
    out["status"] = "ok"
    out["final_step"] = step
    out["weighted_score"] = _to_float(final_row.get("weighted_score"))
    out["accepted"] = int(_to_bool(final_row.get("accepted")))
    out["stage"] = final_row.get("stage", "")

    out["score_binding_site_allowed_min_dist"] = _extract_metric(
        final_row, ["binding_site_allowed_min_dist", "AB.binding_site_allowed_min_dist"]
    )
    out["score_binding_site_framework_min_dist"] = _extract_metric(
        final_row, ["binding_site_framework_min_dist", "AB.binding_site_framework_min_dist"]
    )
    out["dsasa_target"] = _extract_metric(
        final_row, ["dsasa_target", "AB.dsasa_target", "hotspot_dsasa_target"]
    )
    out["dsasa_paratope"] = _extract_metric(
        final_row, ["dsasa_paratope", "AB.dsasa_paratope", "hotspot_dsasa_paratope"]
    )

    structure_path = _resolve_structure(parsed.run_dir, step)
    if structure_path is None:
        out["status"] = "missing_structure"
        return out
    out["structure_path"] = str(structure_path)
    out.update(compute_geometry_metrics(structure_path, parsed.spec))
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    columns: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in columns:
                columns.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("evaluator")), str(row.get("mode")))].append(row)

    metric_names = [
        "ca_any_binder_min_dist",
        "ca_allowed_min_dist",
        "ca_framework_min_dist",
        "aa_any_binder_min_dist",
        "aa_allowed_min_dist",
        "aa_framework_min_dist",
        "ca_contact_satisfied_fraction",
        "aa_contact_satisfied_fraction",
        "dsasa_target",
        "dsasa_paratope",
    ]
    pass_names = ["ca_any_binder_pass", "ca_allowed_pass", "aa_any_binder_pass", "aa_allowed_pass"]

    summaries: list[dict[str, object]] = []
    for (evaluator, mode), group_rows in sorted(groups.items()):
        ok_rows = [r for r in group_rows if r.get("status") == "ok"]
        row: dict[str, object] = {
            "evaluator": evaluator,
            "mode": mode,
            "runs_total": len(group_rows),
            "runs_ok": len(ok_rows),
            "contact_supported": int(any(_to_bool(r.get("contact_supported", 0)) for r in group_rows)),
        }
        for key in metric_names:
            vals = np.asarray(
                [
                    _to_float(r.get(key))
                    for r in ok_rows
                    if math.isfinite(_to_float(r.get(key)))
                ],
                dtype=float,
            )
            if vals.size == 0:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
            else:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"] = float(np.std(vals))
        for key in pass_names:
            vals = np.asarray(
                [
                    _to_float(r.get(key))
                    for r in ok_rows
                    if math.isfinite(_to_float(r.get(key)))
                ],
                dtype=float,
            )
            row[f"{key}_rate"] = float(np.mean(vals)) if vals.size else float("nan")
        summaries.append(row)
    return summaries


def write_markdown_report(path: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    lines: list[str] = []
    lines.append("# Constraint Compliance Report")
    lines.append("")
    lines.append(f"- Total runs: {len(rows)}")
    lines.append(f"- Successful runs: {sum(1 for r in rows if r.get('status') == 'ok')}")
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    lines.append("| evaluator | mode | runs_ok/total | ca_allowed_pass_rate | aa_allowed_pass_rate | ca_contact_satisfied_fraction_mean | aa_contact_satisfied_fraction_mean | dsasa_target_mean | dsasa_paratope_mean | note |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in summaries:
        note = ""
        if str(s.get("mode")) == "contact" and not _to_bool(s.get("contact_supported")):
            note = "contact unsupported by evaluator backend"
        lines.append(
            "| {e} | {m} | {ok}/{tot} | {ca_p:.4f} | {aa_p:.4f} | {ca_c:.4f} | {aa_c:.4f} | {dt:.4f} | {dp:.4f} | {note} |".format(
                e=s.get("evaluator", ""),
                m=s.get("mode", ""),
                ok=s.get("runs_ok", 0),
                tot=s.get("runs_total", 0),
                ca_p=_to_float(s.get("ca_allowed_pass_rate")),
                aa_p=_to_float(s.get("aa_allowed_pass_rate")),
                ca_c=_to_float(s.get("ca_contact_satisfied_fraction_mean")),
                aa_c=_to_float(s.get("aa_contact_satisfied_fraction_mean")),
                dt=_to_float(s.get("dsasa_target_mean")),
                dp=_to_float(s.get("dsasa_paratope_mean")),
                note=note,
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Contact mode is evaluated by pair satisfaction fraction, not all-pairs-must-pass.")
    lines.append("- Both CA-CA and all-atom distance metrics are reported.")
    lines.append("- dSASA terms are read at each run's final accepted step.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_manifest(manifest_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    parsed_runs = [_parse_manifest_row(r) for r in manifest_rows]
    run_rows = [evaluate_run(parsed) for parsed in parsed_runs]
    summary_rows = summarize(run_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = output_dir / "per_run_metrics.csv"
    per_case_csv = output_dir / "per_case_summary.csv"
    report_md = output_dir / "constraint_compliance_report.md"
    _write_csv(per_run_csv, run_rows)
    _write_csv(per_case_csv, summary_rows)
    write_markdown_report(report_md, run_rows, summary_rows)
    return per_run_csv, per_case_csv, report_md


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze constraint compliance benchmark outputs.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV produced by run script.")
    parser.add_argument("--output-dir", default=None, help="Directory for report artifacts.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest.parent / f"{manifest.stem}_report"
    )
    per_run_csv, per_case_csv, report_md = analyze_manifest(manifest, output_dir)
    print(f"per-run metrics: {per_run_csv}")
    print(f"per-case summary: {per_case_csv}")
    print(f"markdown report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
