from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


def _protomer(length: int):
    return SimpleNamespace(
        length=length,
        mutable_mask=np.ones(length, dtype=bool),
        tied_positions=[(idx,) for idx in range(length)],
    )


def test_ligandmpnn_membrane_selector_sets_labels_without_redesign_mask(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [1], "B": []},
    )

    def _fake_selector(expression, evaluation_scores, protomers):
        assert expression == "membrane_outer_surface(membrane_thickness=30.0)"
        return {
            "A": np.array([True, True, False], dtype=bool),
            "B": np.array([False, True, True], dtype=bool),
        }

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.construct_position_selector",
        _fake_selector,
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector="surface()",
        membrane_selector="membrane_outer_surface(membrane_thickness=30.0)",
    )
    protomers = {"A": _protomer(3), "B": _protomer(3)}
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert result["AB"]["fixed_residues"] == "A2 "
    assert result["AB"]["redesigned_residues"] == ""
    assert result["AB"]["transmembrane_interface"] == "A1 A2 B2 B3"
    assert result["AB"]["transmembrane_buried"] == ""


def test_ligandmpnn_membrane_label_selectors_split_boundary_interface(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [], "B": []},
    )
    calls: list[str] = []

    def _fake_selector(expression, evaluation_scores, protomers):
        calls.append(expression)
        if expression == "membrane_outer_surface(membrane_thickness=30.0)":
            return {
                "A": np.array([True, True, True, False], dtype=bool),
                "B": np.array([False, True, True, True], dtype=bool),
            }
        if expression == (
            "membrane_outer_surface(membrane_thickness=30.0) "
            "and not membrane_outer_surface(membrane_thickness=24.0)"
        ):
            return {
                "A": np.array([True, False, False, False], dtype=bool),
                "B": np.array([False, False, True, False], dtype=bool),
            }
        raise AssertionError(expression)

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.construct_position_selector",
        _fake_selector,
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        membrane_label_selectors={
            "buried": "membrane_outer_surface(membrane_thickness=30.0)",
            "interface": (
                "membrane_outer_surface(membrane_thickness=30.0) "
                "and not membrane_outer_surface(membrane_thickness=24.0)"
            ),
        },
    )
    protomers = {"A": _protomer(4), "B": _protomer(4)}
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert calls == [
        "membrane_outer_surface(membrane_thickness=30.0)",
        (
            "membrane_outer_surface(membrane_thickness=30.0) "
            "and not membrane_outer_surface(membrane_thickness=24.0)"
        ),
    ]
    assert result["AB"]["redesigned_residues"] == ""
    assert result["AB"]["transmembrane_interface"] == "A1 B3"
    assert result["AB"]["transmembrane_buried"] == "A2 A3 B2 B4"


def test_ligandmpnn_position_selector_controls_design_mask_when_labels_exist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen_position_selectors: list[str] = []

    def _fake_exclude(position_selector, *args, **kwargs):
        seen_position_selectors.append(position_selector)
        return {"A": [0, 3], "B": [1]}

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        _fake_exclude,
    )
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.construct_position_selector",
        lambda *args, **kwargs: {
            "A": np.array([True, True, False, True], dtype=bool),
            "B": np.array([False, True, True, False], dtype=bool),
        },
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        position_selector="design_surface()",
        membrane_label_selectors={
            "buried": "membrane_outer_surface(membrane_thickness=30.0)",
        },
    )
    protomers = {"A": _protomer(4), "B": _protomer(4)}
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )

    result = mutator.prepare_mutation_input(protomers, scores, runtime_config={})

    assert seen_position_selectors == ["design_surface()"]
    assert result["AB"]["fixed_residues"] == "A1 A4 B2"
    assert result["AB"]["redesigned_residues"] == ""
    assert result["AB"]["transmembrane_buried"] == "A1 A2 A4 B2 B3"


def test_ligandmpnn_membrane_selector_empty_selection_is_visible(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": []},
    )
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.construct_position_selector",
        lambda *args, **kwargs: {"A": np.array([False, False], dtype=bool)},
    )

    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        membrane_selector="membrane_outer_surface(membrane_thickness=30.0)",
    )
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": tmp_path / "a.pdb"},
    )

    with pytest.raises(ValueError, match="selected no residues"):
        mutator.prepare_mutation_input({"A": _protomer(2)}, scores, runtime_config={})
