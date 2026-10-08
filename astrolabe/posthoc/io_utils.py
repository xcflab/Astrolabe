from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from astrolabe.utils.run_dirs import parse_run_dir_index, run_dir_sort_key

_STRUCT_RE = re.compile(r"^step_(\d{3})_(.+)_model\.(cif|pdb|ent)$")


def sanitize_token(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", ".", "+"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def scan_run_dirs(out_root: Path) -> List[Path]:
    out_root = Path(out_root).expanduser().resolve()
    run_dirs: List[Path] = []
    for p in out_root.rglob("scores.csv"):
        run_dir = p.parent
        if not run_dir.is_dir():
            continue
        if parse_run_dir_index(run_dir.name) is None:
            continue
        run_dirs.append(run_dir)
    run_dirs.sort(key=lambda path: (str(path.parent), run_dir_sort_key(path)))
    return run_dirs


def parse_scores_csv(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Dependency-light CSV parser.

    Returns: {step_int: metrics_dict}.
    """
    rows_by_step: Dict[int, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return rows_by_step
        fieldnames = [str(name).strip() for name in r.fieldnames]
        if any(not name for name in fieldnames):
            raise ValueError(f"scores.csv contains an empty column name: {path}")
        duplicate_names = sorted(
            {name for name in fieldnames if fieldnames.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(
                "scores.csv contains duplicate columns after trimming whitespace: "
                + ", ".join(duplicate_names)
            )
        r.fieldnames = fieldnames
        for row in r:
            raw_step = row.get("step", "")
            try:
                step = int(raw_step)
            except Exception:
                continue

            out: Dict[str, Any] = {}
            for k, v in row.items():
                if k is None:
                    raise ValueError(
                        f"scores.csv row contains more fields than its header: {path}"
                    )
                if v is None:
                    continue
                vv = v.strip()
                if vv == "":
                    continue
                if vv in {"True", "False"}:
                    out[k] = vv == "True"
                    continue
                # Preserve sequences as strings.
                if k.endswith("_sequence"):
                    out[k] = vv
                    continue
                # Best-effort numeric conversion.
                try:
                    if "." in vv or "e" in vv.lower():
                        out[k] = float(vv)
                    else:
                        out[k] = int(vv)
                except Exception:
                    out[k] = vv
            rows_by_step[step] = out
    return rows_by_step


def index_structures(structure_dir: Path) -> Dict[tuple[int, str], Path]:
    mapping: Dict[tuple[int, str], Path] = {}
    if not structure_dir.exists():
        return mapping
    for p in structure_dir.iterdir():
        if not p.is_file():
            continue
        m = _STRUCT_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        oligo = m.group(2)
        mapping[(step, oligo)] = p
    return mapping


def file_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(st.st_mtime_ns), int(st.st_size)


def chain_order_from_structure(path: Path) -> list[str]:
    """
    Best-effort chain ID order from a structure file.

    Uses Biopython parsers (already a core dependency in this repo).
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".pdb" or path.suffix.lower() == ".ent":
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
        else:
            from Bio.PDB import MMCIFParser

            parser = MMCIFParser(QUIET=True)
        st = parser.get_structure("st", str(path))
    except Exception:
        return []
    order: list[str] = []
    seen: set[str] = set()
    try:
        model = next(iter(st))
    except Exception:
        return []
    for chain in model:
        cid = str(chain.id).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        order.append(cid)
    return order
