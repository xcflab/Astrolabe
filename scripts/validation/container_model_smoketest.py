#!/usr/bin/env python3
"""Run a weighted model smoke through the generated container GH_PY runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _verify_outputs(run_dir: Path) -> dict[str, object]:
    scores = run_dir / "scores.csv"
    structures = sorted((run_dir / "structures").glob("*"))
    nonempty_structures = [path for path in structures if path.is_file() and path.stat().st_size > 0]
    if not scores.is_file() or scores.stat().st_size == 0:
        raise RuntimeError(f"Missing non-empty scores.csv: {scores}")
    if not nonempty_structures:
        raise RuntimeError(f"No non-empty model structures under: {run_dir / 'structures'}")
    return {
        "run_dir": str(run_dir),
        "scores": str(scores),
        "scores_bytes": scores.stat().st_size,
        "structures": [str(path) for path in nonempty_structures],
    }


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Generated GH_PY container runner.")
    parser.add_argument("--config-name", required=True, help="Config template under config/.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    runner = Path(os.path.expandvars(os.path.expanduser(args.runner))).resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise SystemExit(f"Runner is not executable: {runner}")
    suffix = f"{args.config_name}_{os.getpid()}"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path("/tmp") / f"gh_container_{suffix}"
    task_name = args.task_name or suffix
    run_dir = out_dir / task_name / "001_run"

    env = os.environ.copy()
    env["GH_PY"] = str(runner)
    env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    command = [
        str(repo_root / "scripts" / "gh_dev"),
        "run",
        "--task",
        args.config_name,
        f"out_dir={out_dir}",
        f"task_name={task_name}",
    ]
    started = time.monotonic()
    subprocess.check_call(command, cwd=repo_root, env=env)
    report = _verify_outputs(run_dir)
    report.update(
        {
            "config_name": args.config_name,
            "runner": str(runner),
            "seconds": round(time.monotonic() - started, 3),
        }
    )
    report_path = Path(args.report).resolve() if args.report else out_dir / "smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: {args.config_name} model smoke passed: {run_dir}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
