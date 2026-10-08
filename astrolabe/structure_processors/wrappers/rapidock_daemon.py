#!/usr/bin/env python3
"""JSONL daemon that bridges GH's RAPiDock runner to upstream inference.py."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPiDock JSONL daemon.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--confidence-model-dir")
    parser.add_argument("--confidence-ckpt", default="best_model.pt")
    parser.add_argument("--device")
    return parser.parse_args()


def _write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _rank_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = re.search(r"rank(\d+)", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def _collect_candidates(output_dir: pathlib.Path) -> List[str]:
    preferred = [path for path in output_dir.rglob("rank*_confidence.pdb") if path.is_file()]
    fallback = [
        path
        for path in output_dir.rglob("rank*.pdb")
        if path.is_file() and re.fullmatch(r"rank\d+\.pdb", path.name)
    ]
    candidates = sorted(preferred or fallback, key=_rank_sort_key)
    if not candidates:
        raise RuntimeError(f"No final rank PDB outputs found under '{output_dir}'.")
    return [str(path.resolve()) for path in candidates]


def _build_command(
    *,
    repo_path: pathlib.Path,
    model_dir: pathlib.Path,
    ckpt: str,
    confidence_model_dir: str | None,
    confidence_ckpt: str,
    device: str | None,
    task: Dict[str, Any],
    params: Dict[str, Any],
) -> List[str]:
    name = task.get("name")
    protein_pdb = task.get("protein_pdb")
    peptide_pdb = task.get("peptide_pdb")
    output_dir = task.get("output_dir")
    if not all(
        isinstance(item, str) and item
        for item in (name, protein_pdb, peptide_pdb, output_dir)
    ):
        raise ValueError("Dock task must include non-empty name/protein_pdb/peptide_pdb/output_dir.")

    cmd = [
        sys.executable,
        str(repo_path / "inference.py"),
        "--complex_name",
        name,
        "--protein_description",
        protein_pdb,
        "--peptide_description",
        peptide_pdb,
        "--output_dir",
        output_dir,
        "--model_dir",
        str(model_dir),
        "--ckpt",
        ckpt,
        "--N",
        str(int(params.get("num_samples_bank", 1))),
        "--batch_size",
        str(int(params.get("batch_size", 1))),
        "--inference_steps",
        str(int(params.get("inference_steps", 20))),
        "--cpu",
        str(int(params.get("cpu", 1))),
        "--conformation_type",
        str(params.get("conformation_type", "E")),
    ]

    actual_steps = params.get("actual_steps")
    if actual_steps not in (None, "", "null", "None"):
        cmd.extend(["--actual_steps", str(int(actual_steps))])
    if bool(params.get("no_final_step_noise", False)):
        cmd.append("--no_final_step_noise")
    scoring_function = params.get("scoring_function")
    if scoring_function not in (None, "", "null", "None"):
        cmd.extend(["--scoring_function", str(scoring_function)])
    if bool(params.get("fastrelax", False)):
        cmd.append("--fastrelax")
    if bool(params.get("save_visualisation", False)):
        cmd.append("--save_visualisation")
    conformation_partial = params.get("conformation_partial")
    if conformation_partial not in (None, "", "null", "None"):
        cmd.extend(["--conformation_partial", str(conformation_partial)])
    if confidence_model_dir not in (None, "", "null", "None"):
        cmd.extend(["--confidence_model_dir", str(confidence_model_dir)])
        cmd.extend(["--confidence_ckpt", str(confidence_ckpt)])
    return cmd


def _run_dock(
    *,
    repo_path: pathlib.Path,
    model_dir: pathlib.Path,
    ckpt: str,
    confidence_model_dir: str | None,
    confidence_ckpt: str,
    device: str | None,
    task: Dict[str, Any],
    params: Dict[str, Any],
) -> List[str]:
    output_dir = pathlib.Path(str(task["output_dir"])).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(
        repo_path=repo_path,
        model_dir=model_dir,
        ckpt=ckpt,
        confidence_model_dir=confidence_model_dir,
        confidence_ckpt=confidence_ckpt,
        device=device,
        task=task,
        params=params,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(repo_path),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"RAPiDock inference failed with exit code {proc.returncode}: {detail}"
        )

    return _collect_candidates(output_dir)


def main() -> int:
    args = _parse_args()
    repo_path = pathlib.Path(args.repo_path).expanduser().resolve()
    model_dir = pathlib.Path(args.model_dir).expanduser().resolve()
    inference_path = repo_path / "inference.py"
    ckpt_path = model_dir / args.ckpt

    if not repo_path.exists():
        raise FileNotFoundError(f"RAPiDock repo path does not exist: {repo_path}")
    if not inference_path.exists():
        raise FileNotFoundError(f"RAPiDock inference script not found: {inference_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RAPiDock checkpoint not found: {ckpt_path}")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg: Dict[str, Any] | None = None
        try:
            msg = json.loads(line)
            mtype = msg.get("type")
            if mtype == "ping":
                _write({"type": "pong", "id": msg.get("id")})
                continue
            if mtype == "shutdown":
                _write({"type": "ok", "message": "bye"})
                return 0
            if mtype != "dock":
                raise ValueError(f"unknown type: {mtype}")

            req_id = msg.get("id")
            tasks = msg.get("tasks") or []
            params = msg.get("params") or {}
            results = []
            for task in tasks:
                candidates = _run_dock(
                    repo_path=repo_path,
                    model_dir=model_dir,
                    ckpt=str(args.ckpt),
                    confidence_model_dir=args.confidence_model_dir,
                    confidence_ckpt=str(args.confidence_ckpt),
                    device=args.device,
                    task=task,
                    params=params,
                )
                results.append(
                    {
                        "name": task.get("name"),
                        "candidates": [
                            {"rank": rank, "peptide_pdb": path}
                            for rank, path in enumerate(candidates, start=1)
                        ],
                    }
                )
            _write({"type": "result", "id": req_id, "results": results})
        except Exception as exc:
            _write(
                {
                    "type": "error",
                    "id": msg.get("id") if isinstance(msg, dict) else None,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
