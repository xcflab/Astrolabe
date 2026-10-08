from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator
from astrolabe.structure_processors.base import (
    StructureProcessor,
    StructureProcessorRegistry,
    run_nested_predocker_on_scores,
)


def _pdb_atom(serial: int, chain: str, residue: int, xyz) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain:1s}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
    )


def _write_pdb(path: Path, chain_coords: dict[str, list[tuple[int, tuple[float, float, float]]]]) -> Path:
    lines = []
    serial = 1
    for chain, residues in chain_coords.items():
        for residue, xyz in residues:
            lines.append(_pdb_atom(serial, chain, residue, xyz))
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_ca_coords(path: Path, chain: str, residues: list[int]) -> np.ndarray:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    model = next(structure.get_models())
    coords = []
    for residue in residues:
        coords.append(np.asarray(model[chain][(" ", residue, " ")]["CA"].coord, dtype=float))
    return np.stack(coords, axis=0)


class _ScoreContext:
    def __init__(self, context):
        self._context = context

    def score_context(self, _evaluation_scores):
        return self._context


class _SuffixPredocker(StructureProcessor):
    def runtime_clone(self, *, oligo: str):
        clone = _SuffixPredocker(cache_enabled=False)
        clone.oligo = oligo
        return clone

    def _run_impl(self, input_file):
        return f"{input_file}.{self.oligo}.predocked"


class _Protomer(SimpleNamespace):
    type = ProtomerTypes.PROTEIN
    length = 1
    mutable_mask = np.asarray([True], dtype=bool)
    tied_positions = []

    def aggregate_bool(self, mask):
        return mask


def _init_global(tmp_path: Path, score_context: dict) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("loss_computer", _ScoreContext(score_context))
    global_config.init_const("binding_site", {"binder": "A", "target": "B:1"})
    global_config.set_variable("step", 7)


def _scores(tmp_path: Path) -> StandardScores:
    ab_path = _write_pdb(
        tmp_path / "ab.pdb",
        {
            "A": [(1, (0.0, 5.0, 0.0))],
            "B": [
                (1, (0.0, 0.0, 0.0)),
                (2, (1.0, 0.0, 0.0)),
                (3, (0.0, 1.0, 0.0)),
            ],
        },
    )
    ac_path = _write_pdb(
        tmp_path / "ac.pdb",
        {
            "A": [(1, (10.0, 15.0, 0.0))],
            "B": [
                (11, (10.0, 10.0, 0.0)),
                (12, (10.0, 11.0, 0.0)),
                (13, (9.0, 10.0, 0.0)),
            ],
        },
    )
    return StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.9], "AC": [0.9]},
        complex_paths={"AB": ab_path, "AC": ac_path},
        chain_pair_scores={
            "AB": {"chain_ptm": np.asarray([0.2, 0.9], dtype=float)},
            "AC": {"chain_ptm": np.asarray([0.8, 0.9], dtype=float)},
        },
    )


def _predocker() -> object:
    cls = StructureProcessorRegistry.get("hotspot_reference_align")
    return cls(
        hotspot_scores={
            "AB": "hotspot_il4_global",
            "AC": "hotspot_il13_global",
        },
        target_hotspots={
            "AB": "B:1,2,3",
            "AC": "C:11,12,13",
        },
        binder_chain="A",
        tie_delta=0.05,
        cache_enabled=False,
    )


def test_registry_lazy_imports_hotspot_reference_predocker() -> None:
    cls = StructureProcessorRegistry.get("hotspot_reference_align")
    assert cls.__name__ == "HotspotReferenceAlignmentProcessor"


def test_hotspot_delta_selects_reference_and_aligns_mobile_complex(tmp_path: Path) -> None:
    _init_global(
        tmp_path,
        {
            "AB": {"hotspot_il4_global": 0.60},
            "AC": {"hotspot_il13_global": 0.40},
        },
    )
    scores = _scores(tmp_path)
    predocker = _predocker()

    updated = predocker.run_on_scores(scores)

    assert predocker.last_reference_oligo == "AB"
    assert predocker.last_reference_reason == "hotspot_score"
    assert updated is not scores
    assert updated.complex_paths["AB"] == scores.complex_paths["AB"]
    assert updated.complex_paths["AC"] != scores.complex_paths["AC"]
    assert Path(updated.complex_paths["AC"]).exists()

    ref_coords = _read_ca_coords(Path(updated.complex_paths["AB"]), "B", [1, 2, 3])
    mobile_coords = _read_ca_coords(Path(updated.complex_paths["AC"]), "B", [11, 12, 13])
    rmsd = float(np.sqrt(np.mean(np.sum((ref_coords - mobile_coords) ** 2, axis=1))))
    assert rmsd < 1e-4


def test_binder_ptm_breaks_hotspot_tie(tmp_path: Path) -> None:
    _init_global(
        tmp_path,
        {
            "AB": {"hotspot_il4_global": 0.60},
            "AC": {"hotspot_il13_global": 0.57},
        },
    )
    scores = _scores(tmp_path)
    predocker = _predocker()

    updated = predocker.run_on_scores(scores)

    assert predocker.last_reference_oligo == "AC"
    assert predocker.last_reference_reason == "binder_ptm_tiebreak"
    assert updated.complex_paths["AC"] == scores.complex_paths["AC"]
    assert updated.complex_paths["AB"] != scores.complex_paths["AB"]


def test_single_hotspot_alignment_translates_mobile_complex(tmp_path: Path) -> None:
    _init_global(
        tmp_path,
        {
            "AB": {"hotspot_il4_global": 0.60},
            "AC": {"hotspot_il13_global": 0.40},
        },
    )
    scores = _scores(tmp_path)
    cls = StructureProcessorRegistry.get("hotspot_reference_align")
    predocker = cls(
        hotspot_scores={
            "AB": "hotspot_il4_global",
            "AC": "hotspot_il13_global",
        },
        target_hotspots={
            "AB": "B:1",
            "AC": "C:11",
        },
        binder_chain="A",
        cache_enabled=False,
    )

    updated = predocker.run_on_scores(scores)

    ref_coord = _read_ca_coords(Path(updated.complex_paths["AB"]), "B", [1])
    mobile_coord = _read_ca_coords(Path(updated.complex_paths["AC"]), "B", [11])
    assert np.allclose(ref_coord, mobile_coord, atol=1e-4)


def test_scores_aware_nested_predocker_preserves_legacy_path_predockers(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("predockers", {"suffix": _SuffixPredocker(cache_enabled=False)})
    scores = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.9], "AC": [0.8]},
        complex_paths={"AB": tmp_path / "ab.pdb", "AC": tmp_path / "ac.pdb"},
    )

    updated = run_nested_predocker_on_scores(["suffix"], scores)

    assert updated is not scores
    assert updated.complex_paths == {
        "AB": f"{tmp_path / 'ab.pdb'}.AB.predocked",
        "AC": f"{tmp_path / 'ac.pdb'}.AC.predocked",
    }
    assert scores.complex_paths == {"AB": tmp_path / "ab.pdb", "AC": tmp_path / "ac.pdb"}


def test_ligandmpnn_prepare_uses_scores_aware_predocker(monkeypatch, tmp_path: Path) -> None:
    calls = []
    original_scores = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )
    predocked_scores = original_scores.clone()
    predocked_scores.complex_paths = {"AB": tmp_path / "ab.predocked.pdb"}

    def _fake_run_nested(predocker_names, evaluation_scores):
        calls.append((predocker_names, evaluation_scores))
        return predocked_scores

    import astrolabe.mutators.ligandmpnn as ligandmpnn_module

    monkeypatch.setattr(
        ligandmpnn_module,
        "run_nested_predocker_on_scores",
        _fake_run_nested,
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        predocker=["hotspot_reference_align"],
    )
    protomers = {"A": _Protomer(), "B": _Protomer()}

    prepared = mutator.prepare_mutation_input(protomers, original_scores, runtime_config={})

    assert calls == [(["hotspot_reference_align"], original_scores)]
    assert prepared["AB"]["pdb_path"] == str(tmp_path / "ab.predocked.pdb")
