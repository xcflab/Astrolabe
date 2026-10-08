from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.structure_inputs import (
    normalize_supplemental_structures,
    prepare_structure_input_scores,
)


def _scores(tmp_path: Path, prefix: str, names: list[str]) -> StandardScores:
    return StandardScores(
        names=names,
        plddt={name: [0.9] for name in names},
        complex_paths={name: tmp_path / f"{prefix}_{name}.pdb" for name in names},
        ptm={name: [0.1] for name in names},
    )


def test_prepare_structure_input_scores_defaults_to_current_only(
    tmp_path: Path,
) -> None:
    global_config.reset()
    scores = _scores(tmp_path, "current", ["A", "B"])

    resolved = prepare_structure_input_scores(scores)

    assert resolved.primary is scores
    assert resolved.supplemental == ()


def test_prepare_structure_input_scores_adds_structure_start_reference_as_supplemental(
    tmp_path: Path,
) -> None:
    global_config.reset()
    current = _scores(tmp_path, "current", ["A", "C"])
    references = _scores(tmp_path, "reference", ["A", "B", "C"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=references),
    )

    resolved = prepare_structure_input_scores(
        current,
        supplemental_structures=["structure_start_reference"],
        mutator_name="test_mutator",
    )

    assert resolved.primary is current
    assert len(resolved.supplemental) == 1
    assert resolved.supplemental[0].source == "structure_start_reference"
    assert resolved.supplemental[0].scores.names == ["A", "C"]
    assert resolved.supplemental[0].scores.complex_paths == {
        "A": tmp_path / "reference_A.pdb",
        "C": tmp_path / "reference_C.pdb",
    }


def test_prepare_structure_input_scores_requires_complete_supplemental_references(
    tmp_path: Path,
) -> None:
    global_config.reset()
    current = _scores(tmp_path, "current", ["A", "C"])
    references = _scores(tmp_path, "reference", ["A", "B"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=references),
    )

    with pytest.raises(ValueError, match="missing structure_start reference"):
        prepare_structure_input_scores(
            current,
            supplemental_structures=["structure_start_reference"],
            mutator_name="test_mutator",
        )


def test_prepare_structure_input_scores_predocks_primary_and_supplemental(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_config.reset()
    current = _scores(tmp_path, "current", ["AB"])
    references = _scores(tmp_path, "reference", ["AB"])
    global_config.init_const(
        "structure_start",
        SimpleNamespace(reference_scores=references),
    )
    calls = []

    def _fake_run_nested(predocker_names, evaluation_scores):
        calls.append((predocker_names, evaluation_scores))
        predocked = evaluation_scores.clone()
        predocked.complex_paths = {
            name: tmp_path / f"{Path(path).stem}.predocked.pdb"
            for name, path in evaluation_scores.complex_paths.items()
        }
        return predocked

    monkeypatch.setattr(
        "astrolabe.mutators.structure_inputs.run_nested_predocker_on_scores",
        _fake_run_nested,
    )

    resolved = prepare_structure_input_scores(
        current,
        supplemental_structures=["structure_start_reference"],
        predocker=["dock"],
        mutator_name="test_mutator",
    )

    assert resolved.primary.complex_paths == {
        "AB": tmp_path / "current_AB.predocked.pdb"
    }
    assert resolved.supplemental[0].scores.complex_paths == {
        "AB": tmp_path / "reference_AB.predocked.pdb"
    }
    assert len(calls) == 2
    assert calls[0][0] == ["dock"]
    assert calls[0][1] is current
    assert calls[1][1].complex_paths == {"AB": tmp_path / "reference_AB.pdb"}


def test_normalize_supplemental_structures_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="supplemental_structures"):
        normalize_supplemental_structures(["latest_reference"])
