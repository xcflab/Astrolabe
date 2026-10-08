from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.losses import design_screening
from astrolabe.losses.base import loss_registry


@pytest.fixture(autouse=True)
def _reset_design_screening_pyrosetta_flag(monkeypatch):
    monkeypatch.setattr(design_screening, "_PYROSETTA_INITED", False)


def _install_fake_pyrosetta(monkeypatch):
    calls = []

    class FakeInterfaceAnalyzerMover:
        def __init__(self) -> None:
            calls.append(("new_iam",))

        def set_interface(self, interface: str) -> None:
            calls.append(("interface", interface))

        def set_scorefunction(self, scorefxn) -> None:
            calls.append(("scorefunction", scorefxn))

        def set_compute_interface_energy(self, enabled: bool) -> None:
            calls.append(("compute_interface_energy", enabled))

        def set_calc_dSASA(self, enabled: bool) -> None:
            calls.append(("calc_dSASA", enabled))

        def set_compute_interface_sc(self, enabled: bool) -> None:
            calls.append(("compute_interface_sc", enabled))

        def set_pack_separated(self, enabled: bool) -> None:
            calls.append(("pack_separated", enabled))

        def apply(self, pose) -> None:
            calls.append(("apply", pose))

        def get_all_data(self):
            calls.append(("get_all_data",))
            return SimpleNamespace(sc_value=0.73)

        def get_interface_dG(self) -> float:
            calls.append(("get_interface_dG",))
            return -14.2

        def get_interface_delta_sasa(self) -> float:
            calls.append(("get_interface_delta_sasa",))
            return 1188.5

    analysis_mod = types.ModuleType("pyrosetta.rosetta.protocols.analysis")
    analysis_mod.InterfaceAnalyzerMover = FakeInterfaceAnalyzerMover

    protocols_mod = types.ModuleType("pyrosetta.rosetta.protocols")
    protocols_mod.__path__ = []
    protocols_mod.analysis = analysis_mod

    rosetta_mod = types.ModuleType("pyrosetta.rosetta")
    rosetta_mod.__path__ = []
    rosetta_mod.basic = SimpleNamespace(was_init_called=lambda: True)
    rosetta_mod.protocols = protocols_mod

    pyrosetta_mod = types.ModuleType("pyrosetta")
    pyrosetta_mod.__path__ = []
    pyrosetta_mod.rosetta = rosetta_mod
    pyrosetta_mod.init = lambda opts: calls.append(("init", opts))
    pyrosetta_mod.create_score_function = lambda name: calls.append(
        ("create_score_function", name)
    ) or f"scorefxn:{name}"
    pyrosetta_mod.pose_from_file = lambda path: calls.append(
        ("pose_from_file", path)
    ) or f"pose:{path}"

    monkeypatch.setitem(sys.modules, "pyrosetta", pyrosetta_mod)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta", rosetta_mod)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols", protocols_mod)
    monkeypatch.setitem(
        sys.modules,
        "pyrosetta.rosetta.protocols.analysis",
        analysis_mod,
    )
    return calls


def test_physical_interface_scores_chain_b_bcd_runs_one_interface_analyzer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = _install_fake_pyrosetta(monkeypatch)
    pdb_path = tmp_path / "abcd.pdb"
    pdb_path.write_text("MODEL\nEND\n", encoding="utf-8")

    scores = design_screening.physical_interface_scores(
        pdb_path,
        chain_a="A",
        chain_b="BCD",
    )

    assert scores == {
        "interface_sc": pytest.approx(0.73),
        "dG_separated": pytest.approx(-14.2),
        "dSASA_int": pytest.approx(1188.5),
    }
    assert calls.count(("new_iam",)) == 1
    assert calls.count(("apply", f"pose:{pdb_path}")) == 1
    assert ("interface", "A_BCD") in calls
    assert ("create_score_function", "ref2015") in calls
    assert ("scorefunction", "scorefxn:ref2015") in calls
    assert ("compute_interface_energy", True) in calls
    assert ("calc_dSASA", True) in calls
    assert ("compute_interface_sc", True) in calls
    assert ("pack_separated", True) in calls


def test_physical_interface_score_formula_rewards_higher_sc_and_lower_dg(
    monkeypatch,
    tmp_path: Path,
) -> None:
    better_path = tmp_path / "better.pdb"
    worse_path = tmp_path / "worse.pdb"
    per_path = {
        better_path: {"interface_sc": 0.7, "dG_separated": -12.0, "dSASA_int": 900.0},
        worse_path: {"interface_sc": 0.4, "dG_separated": -9.0, "dSASA_int": 800.0},
    }
    calls = []

    def fake_physical_interface_scores(path, *, chain_a, chain_b, init_opts):
        calls.append((Path(path), chain_a, chain_b, init_opts))
        return per_path[Path(path)]

    monkeypatch.setattr(
        design_screening,
        "physical_interface_scores",
        fake_physical_interface_scores,
    )
    eval_result = SimpleNamespace(
        names=["better", "worse"],
        complex_paths={"better": better_path, "worse": worse_path},
    )

    scores = design_screening.physical_interface_score(
        eval_result,
        chain_a="A",
        chain_b="BCD",
        baseline_sc=0.5,
        baseline_dg=-10.0,
        sc_scale=0.1,
        dg_scale=2.0,
    )

    assert scores["better"] == pytest.approx(3.0)
    assert scores["worse"] == pytest.approx(-1.5)
    assert scores["better"] > scores["worse"]
    assert calls == [
        (better_path, "A", "BCD", "-ignore_unrecognized_res -ignore_zero_occupancy -mute all"),
        (worse_path, "A", "BCD", "-ignore_unrecognized_res -ignore_zero_occupancy -mute all"),
    ]


@pytest.mark.parametrize(
    ("sc_scale", "dg_scale"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_physical_interface_score_rejects_nonpositive_scales_before_scoring(
    monkeypatch,
    tmp_path: Path,
    sc_scale: float,
    dg_scale: float,
) -> None:
    called = False

    def fake_physical_interface_scores(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("physical_interface_scores should not be called")

    monkeypatch.setattr(
        design_screening,
        "physical_interface_scores",
        fake_physical_interface_scores,
    )
    eval_result = SimpleNamespace(
        names=["ABCD"],
        complex_paths={"ABCD": tmp_path / "abcd.pdb"},
    )

    with pytest.raises(ValueError, match="sc_scale > 0 and dg_scale > 0"):
        design_screening.physical_interface_score(
            eval_result,
            chain_a="A",
            chain_b="BCD",
            baseline_sc=0.5,
            baseline_dg=-10.0,
            sc_scale=sc_scale,
            dg_scale=dg_scale,
        )

    assert called is False


def test_physical_interface_score_registry_contract() -> None:
    metadata = loss_registry.get_metadata("physical_interface_score")

    assert metadata.required_params == [
        "chain_a",
        "chain_b",
        "baseline_sc",
        "baseline_dg",
        "sc_scale",
        "dg_scale",
    ]
    assert metadata.optional_params == ["pyrosetta_init"]
    assert metadata.supports_oligomer_specific is True
