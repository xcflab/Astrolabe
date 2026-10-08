#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

from astrolabe.utils.run_dirs import parse_run_dir_index, run_dir_sort_key


REPO_ROOT = Path(__file__).resolve().parents[2]

EVALUATOR_ALIASES = {
    "protenix": ("Protenix", True),
    "chai1": ("chai1", True),
    "boltz": ("Boltz", False),
}


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pocket/contact constraint compliance benchmark across evaluators."
    )
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "GH_PY", str(Path.home() / ".gh_envs" / "envs" / "gh" / "bin" / "python")
        ),
        help="Python executable used for `-m astrolabe.cli run`.",
    )
    parser.add_argument(
        "--task",
        default="benchmark_binder_v1_4",
        help="Template task passed to `astrolabe.cli run --task`.",
    )
    parser.add_argument(
        "--config-file",
        "--overlay",
        dest="config_file",
        default=str(
            REPO_ROOT
            / "examples"
            / "hcar1"
            / "hcar1_constraint_compliance_benchmark.overlay.yaml"
        ),
        help="Overlay config file.",
    )
    parser.add_argument(
        "--config-env",
        default=os.environ.get("GH_CONFIG_ENV", "env"),
        help=(
            "Hydra env config group passed as env=<value>. Defaults to env for "
            "portable examples; use env_local for a local config/env/env_local.yaml."
        ),
    )
    parser.add_argument(
        "--target-homology",
        default=os.environ.get("HCAR1_HOMOLOGY"),
        help=(
            "HCAR1 target homology override for homologies.B, e.g. "
            "/path/to/af3_unique_chains_part_1_data.json:A. "
            "Can also be set with HCAR1_HOMOLOGY."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("outputs") / "hcar1_constraint_compliance_benchmark"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--base-task-name",
        default="hcar1_constraint_compliance",
        help="Task name prefix for generated cases.",
    )
    parser.add_argument("--steps", type=int, default=11, help="Total pipeline steps.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per case.")
    parser.add_argument(
        "--evaluators",
        default="protenix,chai1,boltz",
        help="Comma-separated evaluator keys.",
    )
    parser.add_argument(
        "--modes",
        default="pocket,contact",
        help="Comma-separated binding_site modes.",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA_VISIBLE_DEVICES value. Defaults to the current environment or 0.",
    )
    parser.add_argument(
        "--session-tag",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Unique tag appended to task names.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path. Defaults to <out-dir>/<base-task-name>_<tag>_manifest.csv",
    )
    parser.add_argument(
        "--extra-overrides",
        default="",
        help="Extra Hydra overrides appended to each run command.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining cases when one case fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write planned manifest without executing runs.",
    )
    return parser


def _has_env_group_override(overrides: list[str]) -> bool:
    for override in overrides:
        token = str(override).strip()
        if "=" not in token:
            continue
        key = token.split("=", 1)[0].strip()
        if key.startswith("+"):
            key = key[1:]
        if key == "env":
            return True
    return False


def _list_run_dirs(out_dir: Path, task_name: str) -> list[Path]:
    task_dir = out_dir / task_name
    if not task_dir.exists():
        return []
    run_dirs = [
        p
        for p in task_dir.iterdir()
        if p.is_dir() and parse_run_dir_index(p.name) is not None
    ]
    run_dirs.sort(key=run_dir_sort_key)
    return run_dirs


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "session_tag",
        "task_name",
        "evaluator",
        "evaluator_alias",
        "mode",
        "contact_supported",
        "run_index",
        "run_dir",
        "steps",
        "repeats",
        "max_distance",
        "min_distance",
        "binder",
        "target",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = _build_parser().parse_args()
    config_file = Path(args.config_file).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not args.target_homology:
        raise ValueError(
            "HCAR1 target homology is required. Pass --target-homology "
            "or set HCAR1_HOMOLOGY to an AF3 JSON path plus chain, e.g. "
            "/path/to/af3_unique_chains_part_1_data.json:A."
        )
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else out_dir / f"{args.base_task_name}_{args.session_tag}_manifest.csv"
    )

    evaluators = _parse_csv_list(args.evaluators)
    modes = _parse_csv_list(args.modes)
    extra_overrides = _parse_csv_list(args.extra_overrides)

    rows: list[dict[str, object]] = []
    for evaluator_key in evaluators:
        key = evaluator_key.lower()
        if key not in EVALUATOR_ALIASES:
            raise ValueError(
                f"Unknown evaluator '{evaluator_key}'. Choose from: {sorted(EVALUATOR_ALIASES)}"
            )
        alias, contact_supported = EVALUATOR_ALIASES[key]
        for mode in modes:
            case_id = f"{key}_{mode}"
            task_name = f"{args.base_task_name}_{args.session_tag}_{case_id}"
            overrides = [
                f"task_name={task_name}",
                f"out_dir={out_dir}",
                f"steps={int(args.steps)}",
                f"repeats={int(args.repeats)}",
                f"stages.default.evaluator={alias}",
                f"stages.ligand.evaluator={alias}",
                f"binding_site.mode={mode}",
                f"homologies.B={args.target_homology}",
                "enable_pyrosetta=true",
                "skip_complex_poses=false",
            ]
            if args.config_env and not _has_env_group_override(extra_overrides):
                overrides.append(f"env={args.config_env}")
            overrides.extend(extra_overrides)
            cmd = [
                args.python,
                "-m",
                "astrolabe.cli",
                "run",
                "--task",
                args.task,
                "--standalone",
                str(config_file),
                "--no-chdir",
                "--overrides",
                " ".join(overrides),
            ]

            print(f"[case] {case_id} task={task_name}")
            print(" ".join(cmd))
            run_status = "planned"
            if not args.dry_run:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                completed = subprocess.run(cmd, env=env)
                run_status = "ok" if completed.returncode == 0 else f"failed_rc_{completed.returncode}"
                if completed.returncode != 0 and not args.keep_going:
                    run_dirs = _list_run_dirs(out_dir, task_name)
                    if run_dirs:
                        for idx, run_dir in enumerate(run_dirs, start=1):
                            rows.append(
                                {
                                    "case_id": case_id,
                                    "session_tag": args.session_tag,
                                    "task_name": task_name,
                                    "evaluator": key,
                                    "evaluator_alias": alias,
                                    "mode": mode,
                                    "contact_supported": int(contact_supported),
                                    "run_index": idx,
                                    "run_dir": str(run_dir.resolve()),
                                    "steps": int(args.steps),
                                    "repeats": int(args.repeats),
                                    "max_distance": 12.0,
                                    "min_distance": 0.0,
                                    "binder": "A:26-35;A:55-60;A:102-117",
                                    "target": "B:64-68",
                                    "status": run_status,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "case_id": case_id,
                                "session_tag": args.session_tag,
                                "task_name": task_name,
                                "evaluator": key,
                                "evaluator_alias": alias,
                                "mode": mode,
                                "contact_supported": int(contact_supported),
                                "run_index": 0,
                                "run_dir": "",
                                "steps": int(args.steps),
                                "repeats": int(args.repeats),
                                "max_distance": 12.0,
                                "min_distance": 0.0,
                                "binder": "A:26-35;A:55-60;A:102-117",
                                "target": "B:64-68",
                                "status": run_status,
                            }
                        )
                    _write_manifest(manifest, rows)
                    raise SystemExit(completed.returncode)

            run_dirs = _list_run_dirs(out_dir, task_name)
            if run_dirs:
                for idx, run_dir in enumerate(run_dirs, start=1):
                    rows.append(
                        {
                            "case_id": case_id,
                            "session_tag": args.session_tag,
                            "task_name": task_name,
                            "evaluator": key,
                            "evaluator_alias": alias,
                            "mode": mode,
                            "contact_supported": int(contact_supported),
                            "run_index": idx,
                            "run_dir": str(run_dir.resolve()),
                            "steps": int(args.steps),
                            "repeats": int(args.repeats),
                            "max_distance": 12.0,
                            "min_distance": 0.0,
                            "binder": "A:26-35;A:55-60;A:102-117",
                            "target": "B:64-68",
                            "status": run_status,
                        }
                    )
            else:
                rows.append(
                    {
                        "case_id": case_id,
                        "session_tag": args.session_tag,
                        "task_name": task_name,
                        "evaluator": key,
                        "evaluator_alias": alias,
                        "mode": mode,
                        "contact_supported": int(contact_supported),
                        "run_index": 0,
                        "run_dir": "",
                        "steps": int(args.steps),
                        "repeats": int(args.repeats),
                        "max_distance": 12.0,
                        "min_distance": 0.0,
                        "binder": "A:26-35;A:55-60;A:102-117",
                        "target": "B:64-68",
                        "status": run_status,
                    }
                )

    _write_manifest(manifest, rows)
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
