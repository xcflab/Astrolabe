#!/usr/bin/env python3
"""Validate the consolidated container runtime through its generated GH_PY runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PREFIX = "GH_CONTAINER_RUNTIME_JSON="


def _inner_code(*, expect_pyrosetta: bool) -> str:
    return textwrap.dedent(
        f"""
        import argparse
        import importlib
        import importlib.util
        import json
        import pathlib
        import shutil
        import subprocess
        import sys

        sys.path.insert(0, "/opt/Astrolabe")
        from astrolabe.app import _load_run_module
        from astrolabe.cli import _preinitialize_cuda_driver

        _load_run_module()
        _preinitialize_cuda_driver(argparse.Namespace(_cuda_preinitialize=True, dry_run=False))

        def run_command(candidates, args):
            executable = next((shutil.which(name) for name in candidates if shutil.which(name)), None)
            if executable is None:
                raise RuntimeError("missing command: " + "/".join(candidates))
            proc = subprocess.run(
                [executable, *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = proc.stdout.strip()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"command failed ({{proc.returncode}}): {{executable}} {{' '.join(args)}}\\n{{output}}"
                )
            return {{"path": executable, "output": output}}

        modules = {{}}
        for name in (
            "alphafold",
            "alphafold3.common.folding_input",
            "boltz",
            "chai_lab",
            "esm2",
            "esm3",
            "protenix.data.inference.json_to_feature",
            "rhofold",
        ):
            module = importlib.import_module(name)
            modules[name] = getattr(module, "__file__", None)

        import esm2
        if esm2.__file__ is None or not hasattr(esm2, "data"):
            raise RuntimeError(f"esm2 resolved to an invalid namespace package: {{esm2!r}}")

        import jax
        import tensorflow
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("Torch cannot see a CUDA device")
        jax_devices = [str(device) for device in jax.devices()]
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError(f"JAX cannot see a GPU: {{jax_devices}}")

        pyrosetta_residues = None
        if {expect_pyrosetta!r}:
            import pyrosetta
            pyrosetta.init("-mute all")
            pose = pyrosetta.pose_from_sequence("ACDE")
            pyrosetta_residues = pose.total_residue()
            if pyrosetta_residues != 4:
                raise RuntimeError(f"PyRosetta pose smoke returned {{pyrosetta_residues}} residues")
        elif importlib.util.find_spec("pyrosetta") is not None:
            raise RuntimeError("core image unexpectedly contains PyRosetta")

        commands = {{
            "dssp": run_command(("mkdssp", "dssp"), ("--version",)),
            "nvidia_smi": run_command(("nvidia-smi",), ("--query-gpu=name", "--format=csv,noheader")),
            "obabel": run_command(("obabel",), ("-V",)),
            # US-align returns 1 for -v/--version even after printing its banner.
            "usalign": run_command(("USalign",), ("-h",)),
            "vina": run_command(("vina",), ("--version",)),
        }}

        report = {{
            "commands": commands,
            "esm2_file": esm2.__file__,
            "jax_devices": jax_devices,
            "jax_version": jax.__version__,
            "modules": modules,
            "pyrosetta_residues": pyrosetta_residues,
            "python_version": sys.version.split()[0],
            "tensorflow_version": tensorflow.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_devices": torch.cuda.device_count(),
            "torch_version": torch.__version__,
        }}
        if not report["python_version"].startswith("3.11."):
            raise RuntimeError(f"unexpected Python version: {{report['python_version']}}")
        if not report["torch_version"].startswith("2.7.1"):
            raise RuntimeError(f"unexpected Torch version: {{report['torch_version']}}")
        if report["jax_version"] != "0.4.34":
            raise RuntimeError(f"unexpected JAX version: {{report['jax_version']}}")
        if report["tensorflow_version"] != "2.15.1":
            raise RuntimeError(f"unexpected TensorFlow version: {{report['tensorflow_version']}}")
        print({REPORT_PREFIX!r} + json.dumps(report, sort_keys=True))
        """
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Generated GH_PY container runner.")
    parser.add_argument("--report", required=True, help="JSON report path.")
    pyrosetta = parser.add_mutually_exclusive_group()
    pyrosetta.add_argument(
        "--expect-pyrosetta",
        dest="expect_pyrosetta",
        action="store_true",
        default=True,
        help="Require the default licensed PyRosetta feature (default).",
    )
    pyrosetta.add_argument(
        "--expect-no-pyrosetta",
        dest="expect_pyrosetta",
        action="store_false",
        help="Validate an explicitly reduced image that omits PyRosetta.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Path(args.runner).expanduser().resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise SystemExit(f"Container runner is not executable: {runner}")

    env = os.environ.copy()
    env["GH_CUDA_PREINITIALIZE"] = "1"
    env["NVIDIA_VISIBLE_DEVICES"] = "all"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    proc = subprocess.run(
        [
            str(runner),
            "-c",
            _inner_code(expect_pyrosetta=args.expect_pyrosetta),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(REPORT_PREFIX):
            payload = json.loads(line[len(REPORT_PREFIX) :])
            break
    if payload is None:
        raise SystemExit("Container runtime smoke produced no JSON report")

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: consolidated container runtime passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
