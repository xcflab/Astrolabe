from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import ComplexPoses, StandardScores, construct_evaluators


def test_construct_evaluators_respects_enabled_aliases() -> None:
    class _EvalConf:
        evaluator_terms = {
            "Dummy": {"_type": "dummy"},
            "Unused": {"_type": "this_is_not_a_real_evaluator"},
        }

    evaluators = construct_evaluators(
        _EvalConf(),
        raw_model_results_dir=Path("/tmp"),
        enabled_aliases=["Dummy"],
    )
    assert set(evaluators.keys()) == {"Dummy"}
    assert isinstance(evaluators["Dummy"], object)


def test_construct_evaluators_passes_dummy_term_config(tmp_path: Path) -> None:
    class _EvalConf:
        evaluator_terms = {
            "Dummy": {
                "_type": "dummy",
                "plddt": 0.42,
                "write_structure": True,
                "structure_dir": str(tmp_path / "dummy_structures"),
            },
        }

    evaluators = construct_evaluators(
        _EvalConf(),
        raw_model_results_dir=tmp_path,
        enabled_aliases=["Dummy"],
    )

    evaluator = evaluators["Dummy"]
    assert evaluator._plddt == 0.42
    assert evaluator._write_structure is True
    assert evaluator._structure_dir == tmp_path / "dummy_structures"


def test_construct_evaluators_defaults_dummy_structure_dir_to_raw_results(tmp_path: Path) -> None:
    class _EvalConf:
        evaluator_terms = {
            "Dummy": {
                "_type": "dummy",
                "write_structure": True,
            },
        }

    evaluators = construct_evaluators(
        _EvalConf(),
        raw_model_results_dir=tmp_path,
        enabled_aliases=["Dummy"],
    )

    evaluator = evaluators["Dummy"]
    assert evaluator._write_structure is True
    assert evaluator._structure_dir == tmp_path / "dummy"


def test_build_mutators_skips_null_term_entries() -> None:
    from omegaconf import OmegaConf

    from astrolabe.mutators.base import build_mutators

    class _MutConf:
        # Minimal defaults required by the RandomSubstitutionMutator constructor.
        MCMC = OmegaConf.create({"mutation_method": "random", "select_positions": "random"})
        mutator_terms = {
            "MCMC": {"_type": "MCMC"},
            "Unused": None,
        }

    mutators = build_mutators(_MutConf(), raw_model_results_dir=Path("/tmp"))
    assert set(mutators.keys()) == {"MCMC"}
    from astrolabe.mutators.random_substitution import RandomSubstitutionMutator

    assert type(mutators["MCMC"]) is RandomSubstitutionMutator
    assert type(mutators["MCMC"]).__name__ == "RandomSubstitutionMutator"
    assert mutators["MCMC"].name == "MCMC"


@pytest.mark.parametrize(
    ("module_name", "class_name", "base_module", "base_name"),
    [
        ("evaluators.alphafold2", "AlphaFold2Evaluator", "evaluators.base", "Evaluator"),
        ("evaluators.alphafold3", "AlphaFold3Evaluator", "evaluators.base", "Evaluator"),
        ("evaluators.chai1", "Chai1Evaluator", "evaluators.base", "Evaluator"),
        ("evaluators.rhofold", "RhoFoldEvaluator", "evaluators.base", "Evaluator"),
        ("evaluators.rf3", "RoseTTAFold3Evaluator", "evaluators.base", "Evaluator"),
        ("mutators.random_substitution", "PoreBlockingSubstitutionMutator", "mutators.random_substitution", "RandomSubstitutionMutator"),
        ("mutators.esm_if1", "ESMIF1Mutator", "mutators.base", "Mutator"),
        ("initializers.builtin", "AminoAcidSetInitializer", "initializers.base", "Initializer"),
        ("selectors.base", "ExplicitPositionSelector", "selectors.base", "PositionSelector"),
        ("selectors.sse", "LongLoopPositionSelector", "selectors.base", "PositionSelector"),
        ("structure_processors.autodock", "AutoDockVinaLigandPoseRescue", "structure_processors.base", "StructureProcessor"),
        ("structure_processors.symmetry", "SymmetryRescueProcessor", "structure_processors.base", "StructureProcessor"),
    ],
)
def test_public_component_names_are_real_subclasses(
    module_name: str, class_name: str, base_module: str, base_name: str
) -> None:
    import importlib

    module = importlib.import_module(f"astrolabe.{module_name}")
    base = importlib.import_module(f"astrolabe.{base_module}")
    component = getattr(module, class_name)
    assert component.__name__ == class_name
    assert component.__module__ == module.__name__
    assert issubclass(component, getattr(base, base_name))


def test_dummy_evaluator_conforms_to_evaluator_interface(monkeypatch) -> None:
    from astrolabe.core import global_config
    from astrolabe.evaluators.dummy import DummyEvaluator

    class _Protomer:
        length = 3
        sequence = "AAA"

    # global_config uses a const/variable registry that raises unless initialized.
    monkeypatch.setattr(global_config, "_protomers", {"A": _Protomer()}, raising=False)
    monkeypatch.setitem(global_config._initialized_constants, "protomers", True)

    evaluator = DummyEvaluator(plddt=0.9)
    assert isinstance(evaluator, Evaluator)

    scores = evaluator.evaluate({"A": object()}, {})
    assert scores.names == ["A"]
    assert scores.plddt["A"] == [0.9]
    assert np.allclose(scores.per_proto_scores["A"][0], np.array([0.9, 0.9, 0.9]))


def test_standard_scores_filter_by_oligomer_preserves_per_proto_scores() -> None:
    scores = StandardScores(
        names=["AB", "AC"],
        plddt={
            "AB": [np.array([0.9, 0.9])],
            "AC": [np.array([0.8, 0.8])],
        },
        complex_paths={"AB": Path("ab.pdb"), "AC": Path("ac.pdb")},
        per_proto_scores={
            "A": [np.array([0.9])],
            "B": [np.array([0.1])],
            "C": [np.array([0.2])],
        },
    )

    filtered = scores.filter_by_oligomer(["AB"])

    assert filtered.names == ["AB"]
    assert set((filtered.plddt or {}).keys()) == {"AB"}
    assert set((filtered.complex_paths or {}).keys()) == {"AB"}

    # per_proto_scores is keyed by protomer, not oligomer; keep protomers present in the filtered oligomers.
    assert set((filtered.per_proto_scores or {}).keys()) == {"A", "B"}


def test_complex_poses_uses_path_str_not_private_attr() -> None:
    class _PathWithoutPrivateStr:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    calls: list[tuple] = []

    def pose_from_pdb(path):
        calls.append(("pdb", path))
        return f"POSE:{path}"

    def pose_from_file(path, **kwargs):
        calls.append(("file", path, kwargs))
        return f"POSE:{path}"

    fake_pyrosetta = types.SimpleNamespace(
        pose_from_pdb=pose_from_pdb,
        pose_from_file=pose_from_file,
        rosetta=types.SimpleNamespace(
            core=types.SimpleNamespace(
                import_pose=types.SimpleNamespace(
                    FileType=types.SimpleNamespace(CIF_file="CIF")
                )
            )
        ),
    )

    prev = sys.modules.get("pyrosetta")
    sys.modules["pyrosetta"] = fake_pyrosetta
    try:
        scores = StandardScores(
            names=["AB", "AC"],
            plddt={
                "AB": [np.array([0.9, 0.9])],
                "AC": [np.array([0.8, 0.8])],
            },
            complex_paths={
                "AB": _PathWithoutPrivateStr("ab.pdb"),
                "AC": _PathWithoutPrivateStr("ac.cif"),
            },
        )
        poses = ComplexPoses.construct_complex_poses(scores)
        assert set(poses.poses.keys()) == {"AB", "AC"}
    finally:
        if prev is None:
            del sys.modules["pyrosetta"]
        else:
            sys.modules["pyrosetta"] = prev

    assert ("pdb", "ab.pdb") in calls
    assert any(call[0] == "file" and call[1] == "ac.cif" for call in calls)


def test_complex_poses_cif_error_does_not_require_private_path_attr() -> None:
    class _PathWithoutPrivateStr:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    def pose_from_file(_path, **_kwargs):
        raise RuntimeError("boom")

    fake_pyrosetta = types.SimpleNamespace(
        pose_from_pdb=lambda _path: None,
        pose_from_file=pose_from_file,
        rosetta=types.SimpleNamespace(
            core=types.SimpleNamespace(
                import_pose=types.SimpleNamespace(
                    FileType=types.SimpleNamespace(CIF_file="CIF")
                )
            )
        ),
    )

    prev = sys.modules.get("pyrosetta")
    sys.modules["pyrosetta"] = fake_pyrosetta
    try:
        scores = StandardScores(
            names=["AC"],
            plddt={"AC": [np.array([0.8, 0.8])]},
            complex_paths={"AC": _PathWithoutPrivateStr("ac.cif")},
        )
        with pytest.raises(RuntimeError) as excinfo:
            ComplexPoses.construct_complex_poses(scores)
        assert "ac.cif" in str(excinfo.value)
    finally:
        if prev is None:
            del sys.modules["pyrosetta"]
        else:
            sys.modules["pyrosetta"] = prev


def test_complex_poses_can_explicitly_fallback_to_centered_pdb(monkeypatch, tmp_path: Path) -> None:
    class _PathWithoutPrivateStr:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    calls: list[tuple] = []

    def pose_from_file(_path, **_kwargs):
        raise RuntimeError("boom")

    def pose_from_pdb(path):
        calls.append(("pdb", path))
        return f"POSE:{path}"

    fake_pyrosetta = types.SimpleNamespace(
        pose_from_pdb=pose_from_pdb,
        pose_from_file=pose_from_file,
        rosetta=types.SimpleNamespace(
            core=types.SimpleNamespace(
                import_pose=types.SimpleNamespace(
                    FileType=types.SimpleNamespace(CIF_file="CIF")
                )
            )
        ),
    )

    fallback_path = tmp_path / "ac.pyrosetta_fallback.pdb"
    fallback_path.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    import astrolabe.evaluators.utils as evaluator_utils

    monkeypatch.setattr(
        evaluator_utils,
        "ensure_centered_pdb_fallback",
        lambda mmcif_path, suffix=".pyrosetta_fallback.pdb": fallback_path,
        raising=False,
    )

    global_config.reset()
    global_config.init_const("allow_mmcif_pdb_fallback", True)

    prev = sys.modules.get("pyrosetta")
    sys.modules["pyrosetta"] = fake_pyrosetta
    try:
        scores = StandardScores(
            names=["AC"],
            plddt={"AC": [np.array([0.8, 0.8])]},
            complex_paths={"AC": _PathWithoutPrivateStr("ac.cif")},
        )
        poses = ComplexPoses.construct_complex_poses(scores)
        assert poses.poses["AC"] == f"POSE:{fallback_path}"
    finally:
        global_config.reset()
        if prev is None:
            del sys.modules["pyrosetta"]
        else:
            sys.modules["pyrosetta"] = prev

    assert calls == [("pdb", str(fallback_path))]


def test_interaction_load_pose_can_explicitly_fallback_to_centered_pdb(
    monkeypatch, tmp_path: Path
) -> None:
    import astrolabe.losses.interaction as interaction

    calls: list[tuple] = []

    def pose_from_file(_path, **_kwargs):
        raise RuntimeError("boom")

    def pose_from_pdb(path):
        calls.append(("pdb", path))
        return f"POSE:{path}"

    fake_pyrosetta = types.SimpleNamespace(
        pose_from_pdb=pose_from_pdb,
        pose_from_file=pose_from_file,
        rosetta=types.SimpleNamespace(
            core=types.SimpleNamespace(
                import_pose=types.SimpleNamespace(
                    FileType=types.SimpleNamespace(CIF_file="CIF")
                )
            )
        ),
    )

    cif_path = tmp_path / "ac.cif"
    cif_path.write_text("data_ac\n", encoding="utf-8")
    fallback_path = tmp_path / "ac.pyrosetta_fallback.pdb"
    fallback_path.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    monkeypatch.setattr(interaction, "pyrosetta", fake_pyrosetta)
    monkeypatch.setattr(
        interaction,
        "ensure_centered_pdb_fallback",
        lambda mmcif_path, suffix=".pyrosetta_fallback.pdb": fallback_path,
        raising=False,
    )

    interaction._load_pose_cached.cache_clear()
    global_config.reset()
    global_config.init_const("allow_mmcif_pdb_fallback", True)
    try:
        pose = interaction.load_pose(cif_path)
        assert pose == f"POSE:{fallback_path}"
    finally:
        interaction._load_pose_cached.cache_clear()
        global_config.reset()

    assert calls == [("pdb", str(fallback_path))]
