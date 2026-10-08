#!/usr/bin/env python3
"""
End-to-end smoke test: run the real `astrolabe run` pipeline entrypoint inside the SIF.

This uses the lightweight config `config/container_smoke.yaml` which is designed
to avoid proprietary weights and optional heavy deps (pyrosetta/rdkit/biopython).
"""

from __future__ import print_function

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def _detect_engine():
    for name in ("apptainer", "singularity"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _run(cmd, env=None):
    print("+ " + " ".join([str(x) for x in cmd]))
    subprocess.check_call(cmd, env=env)


def main(argv=None):
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser(description="Run `astrolabe run` inside the SIF (smoke test).")
    p.add_argument(
        "--sif",
        default=str(pathlib.Path.home() / ".gh_envs" / "images" / "Astrolabe_af3.sif"),
        help="Path to the built SIF.",
    )
    p.add_argument(
        "--config-name",
        default="container_smoke",
        help="Hydra config name (under repo config/).",
    )
    args = p.parse_args(argv)

    engine = _detect_engine()
    if not engine:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")

    sif = pathlib.Path(os.path.expanduser(os.path.expandvars(args.sif))).resolve()
    if not sif.exists():
        raise SystemExit("SIF not found: %s" % sif)

    # Bind repo root to match the baked-in container path, so `gh` uses current
    # worktree sources/configs instead of whatever was baked into the image.
    bind = "%s:/opt/Astrolabe" % str(repo_root)

    _run(
        [
            engine,
            "exec",
            "-B",
            bind,
            str(sif),
            "astrolabe",
            "run",
            "--task",
            str(args.config_name),
        ]
    )

    print("OK: pipeline smoke test passed inside SIF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
