#!/usr/bin/env python3
"""Export current Astrolabe worktree sources without private or run artifacts."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKED_PARTS = {".git", ".venv", ".codex-tasks", ".beads", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", "build", "dist", "runs", "outputs", "logs", "datasets", "generated", "checkpoints", "wandb", "config_tobe_generalized"}
BLOCKED_NAMES = {"env_local.yaml", ".gh_python_path", ".astrolabe_python_path", ".env", ".DS_Store"}
BLOCKED_SUFFIXES = {".pyc", ".pyo", ".pse", ".ckpt", ".pt", ".pth"}

def git_paths(repo: Path) -> list[Path]:
    raw = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout
    paths = [Path(item.decode()) for item in raw.split(b"\0") if item]
    stage = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout
    gitlinks = set()
    for record in stage.split(b"\0"):
        if record:
            metadata, raw_path = record.split(b"\t", 1)
            if metadata.split()[0] == b"160000":
                gitlinks.add(Path(raw_path.decode()))
    return sorted((path for path in paths if path not in gitlinks), key=lambda value: value.as_posix())

def submodule_pins(repo: Path) -> list[dict[str, str]]:
    stage = subprocess.run(
        ["git", "ls-files", "--stage", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout
    pins = []
    for record in stage.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, commit, stage_number = metadata.decode().split()
        if mode == "160000":
            pins.append({"path": raw_path.decode(), "commit": commit, "stage": stage_number})
    return sorted(pins, key=lambda item: item["path"])


def allowed(relative: Path) -> bool:
    pure = PurePosixPath(relative.as_posix())
    if any(part in BLOCKED_PARTS or part.endswith(".egg-info") for part in pure.parts):
        return False
    if pure.name in BLOCKED_NAMES or pure.name.startswith("d.") and pure.name.endswith(".overlay.yaml"):
        return False
    return pure.suffix.lower() not in BLOCKED_SUFFIXES

def copy_source(destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True)
    hashes = {}
    for relative in git_paths(PROJECT_ROOT):
        if not allowed(relative):
            continue
        source = PROJECT_ROOT / relative
        if not source.exists() and not source.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
            hashes[relative.as_posix()] = "symlink:" + os.readlink(source)
        elif source.is_file():
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    pins = submodule_pins(PROJECT_ROOT)
    (destination / "SUBMODULE_PINS.json").write_text(json.dumps({"submodules": pins}, indent=2, sort_keys=True) + "\n")
    scope = {
        "git_history_included": False,
        "task_history_included": False,
        "submodule_sources_included": False,
        "cpu_source_install": True,
        "model_bootstrap": "scripts/install.py initializes pinned vendor sources from this archive",
    }
    (destination / "SOURCE_EXPORT.json").write_text(json.dumps({"scope": scope, "files": hashes}, indent=2, sort_keys=True) + "\n")
    (destination / "SOURCE_ARCHIVE_SCOPE.md").write_text(
        "# Astrolabe source archive scope\n\n"
        "This archive is a clean snapshot of the current working tree. It contains no .git directory, "
        "commit history, local task records, private overlays, or model run outputs.\n\n"
        "## CPU source installation\n\n"
        "The archive supports a normal Python source installation for the public Astrolabe package, "
        "CPU analysis, post-hoc tooling, configuration resources, and the bundled CPU reproduction example.\n\n"
        "## Model bootstrap\n\n"
        "SUBMODULE_PINS.json records the exact dependency commits. Run scripts/install.py as described "
        "in container/README.md: it checks the source manifest, creates a local Git index (without "
        "upstream history or a main-repository remote), and fetches selected vendor sources from the URLs "
        "in .gitmodules. Git and network access to those dependencies are required; access to the Astrolabe "
        "GitHub repository is not. Model weights and restricted-license inputs remain separate.\n"
    )
    return hashes

def write_archive(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(source.rglob("*"), key=lambda value: value.as_posix()):
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(source.parent).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error(f"destination already exists: {destination}")
    hashes = copy_source(destination)
    if args.archive:
        write_archive(destination, args.archive.resolve())
    print(f"exported {len(hashes)} files to {destination}")
    if args.archive:
        print(f"archive {args.archive.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
