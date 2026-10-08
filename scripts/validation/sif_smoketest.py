#!/usr/bin/env python3
"""
Smoke-test the Apptainer/Singularity SIF image.

Goal: prove the built SIF can run `gh` and that task-scoped
`astrolabe env doctor --resolve` succeeds without inventing placeholder paths for
optional tools.
"""

from __future__ import print_function

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def _detect_singularity():
    for name in ("apptainer", "singularity"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _run(cmd, env=None):
    print("+ " + " ".join([str(x) for x in cmd]))
    subprocess.check_call(cmd, env=env)


def main(argv=None):
    p = argparse.ArgumentParser(description="Smoke test Astrolabe SIF.")
    p.add_argument(
        "--sif",
        default=str(pathlib.Path.home() / ".gh_envs" / "images" / "Astrolabe_af3.sif"),
        help="Path to the built SIF.",
    )
    p.add_argument(
        "--env",
        default="/opt/Astrolabe/config/env_container.yaml",
        help="env yaml path inside the container.",
    )
    p.add_argument(
        "--task",
        default="container_smoke",
        help="Task used for task-scoped env doctor validation.",
    )
    args = p.parse_args(argv)

    engine = _detect_singularity()
    if not engine:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")

    sif = pathlib.Path(os.path.expanduser(os.path.expandvars(args.sif))).resolve()
    if not sif.exists():
        raise SystemExit("SIF not found: %s" % sif)

    # 1) CLI is runnable.
    _run([engine, "exec", str(sif), "astrolabe", "--help"])

    # 2) Run task-scoped env doctor without soft-fail. This should not require
    # optional tools that the selected smoke task does not use.
    _run(
        [
            engine,
            "exec",
            str(sif),
            "astrolabe",
            "env",
            "doctor",
            "--task",
            str(args.task),
            "--env",
            args.env,
            "--resolve",
        ],
        env=os.environ.copy(),
    )

    print("OK: SIF smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
