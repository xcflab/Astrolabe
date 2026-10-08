#!/usr/bin/env python3
"""
End-to-end smoke test: run AlphaFold3 evaluator inside the SIF.

This requires AF3 model params to be available on the host and mounted to the
container path used by `config/env_container.yaml` (default: /models/alphafold3).
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


def _verify_outputs(run_dir):
    scores = run_dir / "scores.csv"
    structures = [path for path in (run_dir / "structures").glob("*") if path.is_file()]
    if not scores.is_file() or scores.stat().st_size == 0:
        raise RuntimeError("Missing non-empty AF3 scores: %s" % scores)
    if not any(path.stat().st_size > 0 for path in structures):
        raise RuntimeError("Missing non-empty AF3 structures under: %s" % (run_dir / "structures"))


def main(argv=None):
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser(description="Run AF3 smoke test inside the SIF.")
    p.add_argument(
        "--sif",
        default=str(pathlib.Path.home() / ".gh_envs" / "images" / "Astrolabe_af3.sif"),
        help="Path to the built SIF.",
    )
    p.add_argument(
        "--af3-models",
        required=True,
        help="Host path to AF3 model params directory (must contain af3.bin).",
    )
    p.add_argument(
        "--nv",
        action="store_true",
        help="Pass --nv to singularity/apptainer (recommended for GPU inference).",
    )
    p.add_argument(
        "--config-name",
        default="container_af3_smoke",
        help="Hydra config name under repo config/.",
    )
    p.add_argument(
        "--jax-cache-dir",
        default=None,
        help="Container path for JAX compilation cache (default: unique /tmp dir).",
    )
    p.add_argument("--out-dir", default=None, help="Host-visible smoke output root.")
    args = p.parse_args(argv)

    engine = _detect_engine()
    if not engine:
        raise SystemExit("Could not find `apptainer` or `singularity` in PATH.")

    sif = pathlib.Path(os.path.expanduser(os.path.expandvars(args.sif))).resolve()
    if not sif.exists():
        raise SystemExit("SIF not found: %s" % sif)

    af3_models = pathlib.Path(os.path.expanduser(os.path.expandvars(args.af3_models))).resolve()
    if not (af3_models / "af3.bin").exists():
        raise SystemExit("AF3 weights not found at: %s (expected af3.bin)" % af3_models)

    bind_repo = "%s:/opt/Astrolabe" % str(repo_root)
    bind_models = "%s:/models/alphafold3" % str(af3_models)
    jax_cache_dir = args.jax_cache_dir or ("/tmp/jax_cache_smoke_%d" % os.getpid())
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else pathlib.Path(
        "/tmp/gh_af3_container_smoke_%d" % os.getpid()
    )
    task_name = "sif_af3_smoke_%d" % os.getpid()

    cmd = [engine, "exec"]
    if args.nv:
        cmd.append("--nv")
    cmd += ["-B", bind_repo, "-B", bind_models, str(sif)]
    cmd += [
        "astrolabe",
        "run",
        "--task",
        str(args.config_name),
        # Ensure the AF3 weights are found inside the container.
        "weight_path.alphafold3=/models/alphafold3",
        "evaluator.alphafold3.jax_compilation_cache_dir=%s" % jax_cache_dir,
        "out_dir=%s" % out_dir,
        "task_name=%s" % task_name,
    ]

    env = os.environ.copy()
    if args.nv:
        env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
        env.setdefault("GH_CUDA_PREINITIALIZE", "1")
    _run(cmd, env=env)
    _verify_outputs(out_dir / task_name / "001_run")
    print("OK: AF3 smoke test passed inside SIF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
