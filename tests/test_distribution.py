from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from astrolabe import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = (
    "archive/",
    "build/",
    "container/",
    "docs/",
    "scripts/",
    "tests/",
    "vendor/",
)
CLI_CPU_REQUIREMENTS = {
    "absl-py",
    "biopython",
    "filelock",
    "hydra-core",
    "matplotlib",
    "numpy",
    "omegaconf",
    "prompt-toolkit",
}
MODEL_RUNTIME_REQUIREMENTS = {"torch", "pyrosetta", "alphafold3", "boltz", "protenix", "rdkit"}


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from setuptools import build_meta

    out_dir = tmp_path_factory.mktemp("dist")
    with _chdir(REPO_ROOT):
        return out_dir / build_meta.build_wheel(str(out_dir))


@pytest.fixture(scope="module")
def wheel_names(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        return archive.namelist()


def test_wheel_ships_runner_and_portable_config_resources(wheel_names: list[str]) -> None:
    names = set(wheel_names)
    assert "run.py" in names
    assert "astrolabe/__init__.py" in names
    assert "resources/ligand_hbond_filter.py" in names
    for required in (
        "config/base.yaml",
        "config/monomer.yaml",
        "config/env/env.yaml",
        "config/env/env_container.yaml",
        "data/framework_fastas/framework_sources.fasta",
        "data/framework_fastas/framework_sources.tsv",
        "data/framework_homologies/vhh__boltzgen_7xl0.json",
        "config/paper/rna_aptamer.yaml",
        "examples/paper/baselines/adapters/complexdesign_afm.py",
        "examples/paper/baselines/sources/colabdesign_v1_1_3_target_pair_mask.patch",
        "examples/paper/inputs/nucleic/rna_aptamer_1urn_u1a.yaml",
        "resources/paper_inputs_nucleic.tar.gz",
    ):
        assert required in names


def test_wheel_excludes_vendor_overlays_and_generated_artifacts(wheel_names: list[str]) -> None:
    assert not [name for name in wheel_names if name.startswith(EXCLUDED_ROOTS)]
    assert not [name for name in wheel_names if name.startswith("examples/") and not name.startswith("examples/paper/")]
    assert not [name for name in wheel_names if "__pycache__" in name or ".egg-info" in name or name.endswith((".pyc", ".un~"))]
    assert not [
        name
        for name in wheel_names
        if name.endswith(("env_local.yaml", ".gh_python_path", ".astrolabe_python_path"))
        or (name.startswith("config/d.") and name.endswith(".overlay.yaml"))
    ]


def test_wheel_metadata_declares_distribution_contract(wheel_path: Path, wheel_names: list[str]) -> None:
    metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
    with zipfile.ZipFile(wheel_path) as archive:
        metadata = archive.read(metadata_name).decode("utf-8")
        entry_points = archive.read(metadata_name.replace("METADATA", "entry_points.txt")).decode("utf-8")
    lines = metadata.splitlines()

    def single(prefix: str) -> str:
        return next(line.split(": ", 1)[1] for line in lines if line.startswith(f"{prefix}: "))

    requires = {Requirement(line.split(": ", 1)[1]).name for line in lines if line.startswith("Requires-Dist: ")}
    assert canonicalize_name(single("Name")) == "astrolabe"
    assert single("Version") == __version__
    assert single("Requires-Python") == ">=3.11"
    assert CLI_CPU_REQUIREMENTS <= requires
    assert not MODEL_RUNTIME_REQUIREMENTS & requires
    assert "astrolabe = astrolabe.cli:main" in entry_points
    assert "gh =" not in entry_points


def test_installed_layout_resolves_runner_and_config_resources(wheel_path: Path, tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(extracted)
    code = "\n".join(
        [
            "import astrolabe",
            "import astrolabe.app as app",
            "from astrolabe.antibody_framework_presets import _FRAMEWORK_HOMOLOGY_DIR",
            "print(astrolabe.__version__)",
            "print(astrolabe.install_root())",
            "print(app.run_script_path())",
            "print(astrolabe.config_dir())",
            "print((astrolabe.config_dir() / 'monomer.yaml').is_file())",
            "print((astrolabe.config_dir() / 'env' / 'env.yaml').is_file())",
            "print((_FRAMEWORK_HOMOLOGY_DIR / 'vhh__boltzgen_7xl0.json').is_file())",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    version, root, runner, configs, task_config, env_config, framework_data = proc.stdout.split()
    assert version == __version__
    assert Path(root) == extracted
    assert Path(runner) == extracted / "run.py"
    assert Path(configs) == extracted / "config"
    assert (task_config, env_config, framework_data) == ("True", "True", "True")


def test_public_source_export_has_no_history_and_records_gitlinks(tmp_path: Path) -> None:
    destination = tmp_path / "Astrolabe-0.1.0-public-source"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "release" / "export.py"), str(destination)],
        cwd=REPO_ROOT,
        check=True,
    )
    export = json.loads((destination / "SOURCE_EXPORT.json").read_text())
    pins = json.loads((destination / "SUBMODULE_PINS.json").read_text())["submodules"]
    scope = export["scope"]
    assert scope["git_history_included"] is False
    assert scope["task_history_included"] is False
    assert scope["submodule_sources_included"] is False
    assert scope["cpu_source_install"] is True
    assert len(pins) == 31
    assert any(item["path"] == "vendor/uplifting-biomolecular-modeling" for item in pins)
    assert (destination / "SOURCE_ARCHIVE_SCOPE.md").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / ".codex-tasks").exists()
    assert not (destination / ".beads").exists()
    assert not any(path.name.startswith("d.") and path.name.endswith(".overlay.yaml") for path in destination.rglob("*"))

    # The same archive is a model-install source: no public main-repo clone needed.
    spec = importlib.util.spec_from_file_location("archive_installer", destination / "scripts/install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    installer._prepare_source_archive(dry_run=True)
    assert not (destination / ".git").exists()
    readme = destination / "README.md"
    original = readme.read_bytes()
    readme.write_bytes(original + b"\nlocal edit\n")
    with pytest.raises(SystemExit, match="differs from its manifest: README.md"):
        installer._prepare_source_archive(dry_run=False)
    assert not (destination / ".git").exists()
    readme.write_bytes(original)
    installer._prepare_source_archive(dry_run=False)
    index_before = (destination / ".git/index").read_bytes()
    installer._prepare_source_archive(dry_run=False)
    assert (destination / ".git/index").read_bytes() == index_before
    index = subprocess.check_output(["git", "ls-files", "--stage"], cwd=destination, text=True)
    for pin in pins:
        assert f"160000 {pin['commit']} 0\t{pin['path']}" in index
    assert "\tastrolabe/app.py" in index
    assert not subprocess.check_output(["git", "remote"], cwd=destination, text=True).strip()
    assert subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=destination, capture_output=True).returncode != 0
    subprocess.run(["git", "submodule", "status"], cwd=destination, check=True, capture_output=True)

    spec = importlib.util.spec_from_file_location("archive_builder", destination / "scripts/containers/apptainer_build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert Path("astrolabe/app.py") in builder._tracked_build_context_files(destination)



def test_export_refuses_existing_destination_without_deleting(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve me\n")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "release" / "export.py"), str(destination)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    assert "destination already exists" in proc.stderr
    assert marker.read_text() == "preserve me\n"


def test_scripts_root_contains_only_public_entrypoints() -> None:
    root_files = {path.name for path in (REPO_ROOT / "scripts").iterdir() if path.is_file()}
    assert root_files == {"astrolabe", "gh_dev", "install.py"}
    for removed in ("gh_candidate_batch_run.py", "gh_collect_scores.py", "gh_filter_structures.py", "gh_refold_export.py", "gh_refold_run.py"):
        assert not (REPO_ROOT / "scripts" / removed).exists()
