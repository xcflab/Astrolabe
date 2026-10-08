#!/usr/bin/env python3
"""Inspect the pinned Anthropic optimization kits.

Installation is container-only. This manifest helper intentionally has no
host-Python installation path; build a SIF with
``anthropic_kits_apptainer.py`` and let the definition's ``%post`` install
the pinned software inside that image.

Examples:
  python scripts/containers/anthropic_kits.py check
  python scripts/containers/anthropic_kits.py list
  python scripts/containers/anthropic_kits_apptainer.py --kit esm_if1 --force
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "config" / "anthropic_kits.yaml"


def load_manifest() -> dict:
    value = OmegaConf.to_container(OmegaConf.load(str(MANIFEST_PATH)), resolve=True)
    if not isinstance(value, dict) or not isinstance(value.get("kits"), dict):
        raise SystemExit(f"invalid Anthropic kit manifest: {MANIFEST_PATH}")
    return value


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/uplifting-biomolecular-modeling"), *args],
        text=True,
    ).strip()


def selected_kits(manifest: dict, names: list[str] | None, software_ids: list[str] | None) -> list[tuple[str, dict]]:
    kits = manifest["kits"]
    requested = set(names or [])
    requested_software = set(software_ids or [])
    selected = []
    for name, spec in kits.items():
        if requested and name not in requested:
            continue
        ids = set(spec.get("softwareIds", []))
        if requested_software and not (ids & requested_software):
            continue
        selected.append((name, spec))
    unknown = requested - set(kits)
    if unknown:
        raise SystemExit(f"unknown Anthropic kit(s): {', '.join(sorted(unknown))}")
    if not selected:
        raise SystemExit("no Anthropic kits selected")
    return selected


def check_source(manifest: dict) -> list[dict]:
    source = ROOT / manifest["source"]["path"]
    expected = manifest["source"]["revision"]
    if not source.is_dir():
        raise SystemExit(f"Anthropic source is missing: {source}")
    actual = run_git("rev-parse", "HEAD")
    if actual != expected:
        raise SystemExit(f"Anthropic source revision drift: expected {expected}, observed {actual}")
    rows = []
    for name, spec in manifest["kits"].items():
        kit_dir = source / name
        run_sh = kit_dir / "run.sh"
        if not kit_dir.is_dir() or not run_sh.is_file():
            raise SystemExit(f"Anthropic kit is incomplete: {name} ({run_sh})")
        modes = spec.get("modes", [])
        if not isinstance(modes, list) or any(not isinstance(mode, str) for mode in modes):
            raise SystemExit(f"Anthropic kit {name} has non-string modes; quote YAML values such as 'off'")
        if not set(modes).issubset({"off", "exact", "fast", "big"}):
            raise SystemExit(f"Anthropic kit {name} has an unknown mode: {modes}")
        rows.append({
            "kit": name,
            "softwareIds": list(spec.get("softwareIds", [])),
            "role": spec.get("role"),
            "modes": list(spec.get("modes", [])),
            "run": str(run_sh),
        })
    return rows


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "list"))
    parser.add_argument("--kit", action="append", dest="kits", help="Anthropic kit name; repeat to select multiple")
    parser.add_argument("--software", action="append", dest="software_ids", help="Current GH software id; repeat to select multiple")
    args = parser.parse_args()
    manifest = load_manifest()

    if args.command in {"check", "list"}:
        rows = check_source(manifest)
        if args.command == "list":
            print_json({"source": manifest["source"], "kits": rows, "unmapped": manifest.get("unmapped", {})})
        else:
            print_json({"status": "ready", "sourceRevision": manifest["source"]["revision"], "kits": rows})
        return 0

    raise AssertionError("unreachable: install is not a host-Python command")


if __name__ == "__main__":
    sys.exit(main())
