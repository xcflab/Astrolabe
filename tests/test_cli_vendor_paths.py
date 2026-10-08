from __future__ import annotations

import sys
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import astrolabe.cli as cli


def test_ensure_sys_path_adds_vendored_alphafold_roots(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    af2_root = repo_root / "vendor" / "alphafold2"
    af3_src = repo_root / "vendor" / "alphafold3" / "src"
    config_dir.mkdir(parents=True)
    af2_root.mkdir(parents=True)
    af3_src.mkdir(parents=True)

    original_sys_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    cli._ensure_sys_path_has_repo_root(config_dir)

    assert str(repo_root) in sys.path
    assert str(af2_root) in sys.path
    assert str(af3_src) in sys.path


def test_installed_af3_is_not_shadowed_and_doctor_checks_its_data(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    af3_src = repo / "vendor/alphafold3/src"
    af3_src.mkdir(parents=True)
    installed = tmp_path / "site-packages/alphafold3"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(installed)]))
    cli._ensure_sys_path_has_repo_root(repo / "config")
    assert str(af3_src) not in sys.path
    requirements = cli.runtime_requirements.task_requirements({
        "evaluator": {"evaluator_terms": {"AF3": {"_type": "Alphafold3"}}},
        "stages": {"default": {"evaluator": "AF3", "oligo": ["A"], "mutators": []}},
    })
    assert set(requirements["required_repo_files"]) == {
        str(installed / "constants/converters/ccd.pickle"),
        str(installed / "constants/converters/chemical_component_sets.pickle"),
    }
