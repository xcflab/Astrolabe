from __future__ import annotations

from pathlib import Path

from astrolabe.evaluators import utils


def test_resolve_alphafold2_import_root_accepts_repo_root_data_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "af2_repo"
    (repo_root / "alphafold" / "common").mkdir(parents=True)
    (repo_root / "alphafold" / "common" / "protein.py").write_text("", encoding="utf-8")

    resolved = utils._resolve_alphafold2_import_root(repo_root)

    assert resolved == repo_root


def test_resolve_alphafold2_import_root_accepts_package_dir_data_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "af2_repo"
    package_dir = repo_root / "alphafold"
    (package_dir / "common").mkdir(parents=True)
    (package_dir / "common" / "protein.py").write_text("", encoding="utf-8")

    resolved = utils._resolve_alphafold2_import_root(package_dir)

    assert resolved == repo_root


def test_resolve_alphafold2_import_root_prefers_vendor_with_initial_guess_patch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "external_af2_repo"
    (repo_root / "alphafold" / "common").mkdir(parents=True)
    (repo_root / "alphafold" / "common" / "protein.py").write_text(
        "",
        encoding="utf-8",
    )
    (repo_root / "alphafold" / "model").mkdir(parents=True)
    (repo_root / "alphafold" / "model" / "modules.py").write_text(
        "def model_without_initial_guess_patch(): pass\n",
        encoding="utf-8",
    )

    resolved = utils._resolve_alphafold2_import_root(repo_root)

    assert resolved == Path(__file__).resolve().parents[1] / "vendor" / "alphafold2"
