from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.evaluators import utils as evaluator_utils
from astrolabe.utils.mmcif_fallback import load_pose_with_optional_mmcif_pdb_fallback


@pytest.fixture(autouse=True)
def _reset_global_config():
    global_config.reset()
    yield
    global_config.reset()


class _FakePose:
    pass


class _FakePyRosetta:
    Pose = _FakePose

    class rosetta:
        class core:
            class import_pose:
                class FileType:
                    CIF_file = "CIF_file"

    def __init__(self):
        self.calls = []

    def generate_nonstandard_residue_set(self, pose, params):
        pose.params = getattr(pose, "params", []) + list(params)

    def pose_from_file(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args and isinstance(args[0], _FakePose):
            args[0].loaded_from = args[1]
            args[0].load_kwargs = dict(kwargs)
            return None
        return "plain-pose"

    def pose_from_pdb(self, filename):
        self.calls.append(((filename,), {"pdb": True}))
        return "plain-pdb"


def test_mmcif_loader_loads_extra_residue_params_into_pose(tmp_path: Path) -> None:
    fake = _FakePyRosetta()
    cif_path = tmp_path / "model.cif"
    cif_path.write_text("data_mock\n", encoding="utf-8")
    params_path = tmp_path / "l01.params"
    params_path.write_text("NAME l01\n", encoding="utf-8")

    pose = load_pose_with_optional_mmcif_pdb_fallback(
        fake,
        cif_path,
        extra_residue_params=[params_path],
    )

    assert isinstance(pose, _FakePose)
    assert pose.params == [str(params_path)]
    assert pose.loaded_from == str(cif_path)
    assert pose.load_kwargs["type"] == "CIF_file"
    assert pose.load_kwargs["read_fold_tree"] is True


def test_mmcif_loader_ignores_duplicate_loaded_residue_type(tmp_path: Path) -> None:
    class DuplicateAwarePyRosetta(_FakePyRosetta):
        def generate_nonstandard_residue_set(self, pose, params):
            if Path(params[0]).stem == "l01":
                raise RuntimeError(
                    "ERROR: Error in core::chemical::ResidueTypeSetCache::add_residue_type(): "
                    "Attempting to add a new residue type, but residue type 'l01' already exists in the cache."
                )
            return super().generate_nonstandard_residue_set(pose, params)

    fake = DuplicateAwarePyRosetta()
    cif_path = tmp_path / "model.cif"
    cif_path.write_text("data_mock\n", encoding="utf-8")
    duplicate_params = tmp_path / "l01.params"
    duplicate_params.write_text("NAME l01\n", encoding="utf-8")
    new_params = tmp_path / "x01.params"
    new_params.write_text("NAME x01\n", encoding="utf-8")

    pose = load_pose_with_optional_mmcif_pdb_fallback(
        fake,
        cif_path,
        extra_residue_params=[duplicate_params, new_params],
    )

    assert isinstance(pose, _FakePose)
    assert pose.params == [str(new_params)]
    assert pose.loaded_from == str(cif_path)


def test_complex_poses_passes_current_raw_model_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw_model_results"
    raw_dir.mkdir()
    (raw_dir / "B.params").write_text("NAME B\n", encoding="utf-8")
    (raw_dir / "l01.params").write_text("NAME l01\n", encoding="utf-8")
    complex_path = tmp_path / "step_000_AB_model.cif"
    complex_path.write_text("data_mock\n", encoding="utf-8")
    global_config.init_const("out_dir", tmp_path)

    captured = {}

    def fake_loader(_pyrosetta, path, **kwargs):
        captured["path"] = path
        captured["params"] = [str(p) for p in kwargs["extra_residue_params"]]
        return object()

    monkeypatch.setitem(sys.modules, "pyrosetta", SimpleNamespace())
    monkeypatch.setattr(
        evaluator_utils,
        "load_pose_with_optional_mmcif_pdb_fallback",
        fake_loader,
    )

    result = evaluator_utils.ComplexPoses.construct_complex_poses(
        SimpleNamespace(complex_paths={"AB": complex_path})
    )

    assert set(result.poses) == {"AB"}
    assert captured["path"] == str(complex_path)
    assert captured["params"] == [
        str(raw_dir / "B.params"),
        str(raw_dir / "l01.params"),
    ]


def test_interaction_load_pose_passes_current_raw_model_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astrolabe.losses import interaction

    raw_dir = tmp_path / "raw_model_results"
    raw_dir.mkdir()
    (raw_dir / "l01.params").write_text("NAME l01\n", encoding="utf-8")
    complex_path = tmp_path / "vina_pose.pdb"
    complex_path.write_text("HETATM    1  C1  l01 B   1       0.000   0.000   0.000\nEND\n", encoding="utf-8")
    global_config.init_const("out_dir", tmp_path)

    captured = {}

    def fake_loader(_pyrosetta, path, **kwargs):
        captured["path"] = path
        captured["params"] = [str(p) for p in kwargs["extra_residue_params"]]
        return object()

    monkeypatch.setattr(interaction, "pyrosetta", SimpleNamespace())
    monkeypatch.setattr(
        interaction,
        "load_pose_with_optional_mmcif_pdb_fallback",
        fake_loader,
    )
    interaction._load_pose_cached.cache_clear()

    try:
        interaction.load_pose(complex_path)
    finally:
        interaction._load_pose_cached.cache_clear()

    assert captured["path"] == str(complex_path)
    assert captured["params"] == [str(raw_dir / "l01.params")]
