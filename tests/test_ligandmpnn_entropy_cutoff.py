from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators import utils as mutator_utils
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator

AA_ORDER = [mutator_utils.restype_int_to_str[i] for i in range(20)]


class _FakeSegmentManager:
    def __init__(self, mutations):
        self.mutations = mutations
        self.added = None

    def collect_mutations(self):
        return list(self.mutations)

    def add_mutations(self, mutations, num_mutations):
        self.added = (list(mutations), num_mutations)


def _mutation(site: int, entropy: float):
    return ("A_1", (site, "A", "V", entropy, 0.0))


def _init_protein_runtime(tmp_path, sequence: str) -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("out_dir", tmp_path)
    protomer = Protomer("A", sequence, "protein")
    global_config.init_const("protomers", {"A": protomer})
    global_config.init_const("mutable_protomers", {"A": protomer})
    global_config.set_variable("mutation_num", 0)
    global_config.set_variable("step", 0)
    global_config.set_variable("ignored_complexes", [])


def _sampling_probs(rows: list[dict[str, float]]) -> np.ndarray:
    probs = np.zeros((1, len(rows), len(AA_ORDER)), dtype=float)
    for row_index, aa_probs in enumerate(rows):
        for aa, value in aa_probs.items():
            probs[0, row_index, mutator_utils.restype_str_to_int[aa]] = value
    row_sums = probs.sum(axis=-1, keepdims=True)
    return np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums > 0)


def _log_probs(rows: list[dict[str, float]]) -> np.ndarray:
    values = np.full((1, len(rows), len(AA_ORDER)), -np.inf, dtype=float)
    for row_index, aa_values in enumerate(rows):
        for aa, value in aa_values.items():
            values[0, row_index, mutator_utils.restype_str_to_int[aa]] = value
    return values


def _mutation_output(
    rows: list[dict[str, float]],
    *,
    log_rows: list[dict[str, float]] | None = None,
) -> dict:
    output = {"sampling_probs": _sampling_probs(rows)}
    if log_rows is not None:
        output["log_probs"] = _log_probs(log_rows)
    return {"A": output}


def _protein_protomers(lengths: dict[str, int]) -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(
            length=length,
            tied_positions=[],
            aggregate_bool=lambda mask: mask,
        )
        for name, length in lengths.items()
    }


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    resseq: int,
    x: float,
    icode: str = " ",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4} {residue_name:>3} "
        f"{chain}{resseq:4d}{icode}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}           C\n"
    )


def _write_offset_pdb(
    path: Path,
    chain_starts: dict[str, int],
    residues_per_chain: int = 3,
) -> None:
    lines = []
    serial = 1
    for chain, start in chain_starts.items():
        for offset in range(residues_per_chain):
            lines.append(
                _pdb_atom_line(
                    serial,
                    "CA",
                    "ALA",
                    chain,
                    start + offset,
                    float(serial),
                )
            )
            serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def test_prepare_mutation_input_defaults_to_sequence_number_fixed_residues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "ab_offset.pdb"
    _write_offset_pdb(pdb_path, {"A": 20, "B": 21})
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [0, 2], "B": [1]},
    )
    mutator = LigandMPNNMutator(sequence_model_runner=SimpleNamespace())
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": pdb_path},
    )

    result = mutator.prepare_mutation_input(
        _protein_protomers({"A": 3, "B": 3}),
        scores,
        runtime_config={},
    )

    assert result["AB"]["fixed_residues"] == "A1 A3 B2"


def test_prepare_mutation_input_maps_sequence_positions_to_pdb_residue_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "ab_offset.pdb"
    _write_offset_pdb(pdb_path, {"A": 20, "B": 21})
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [0, 2], "B": [1]},
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        map_sequence_positions_to_pdb_residues=True,
    )
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [1.0]},
        complex_paths={"AB": pdb_path},
    )

    result = mutator.prepare_mutation_input(
        _protein_protomers({"A": 3, "B": 3}),
        scores,
        runtime_config={},
    )

    assert result["AB"]["fixed_residues"] == "A20 A22 B22"


def test_pdb_residue_ids_by_chain_reads_af3_mmcif(tmp_path: Path) -> None:
    from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

    structure = Structure.Structure("af3")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residue_numbers in {"A": (20, 21, 22), "B": (21, 22, 23)}.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for serial_number, residue_number in enumerate(residue_numbers, start=1):
            residue = Residue.Residue((" ", residue_number, " "), "ALA", " ")
            residue.add(
                Atom.Atom(
                    "CA",
                    np.array([float(serial_number), 0.0, 0.0]),
                    1.0,
                    1.0,
                    " ",
                    " CA ",
                    serial_number,
                    element="C",
                )
            )
            chain.add(residue)

    cif_path = tmp_path / "af3_model.cif"
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(cif_path))

    assert LigandMPNNMutator._pdb_residue_ids_by_chain(str(cif_path)) == {
        "A": ["A20", "A21", "A22"],
        "B": ["B21", "B22", "B23"],
    }


def test_prepare_mutation_input_pdb_mapping_rejects_chain_length_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "a_short.pdb"
    _write_offset_pdb(pdb_path, {"A": 20}, residues_per_chain=2)
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": []},
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        map_sequence_positions_to_pdb_residues=True,
    )
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": pdb_path},
    )

    with pytest.raises(ValueError, match="chain A residue count 2"):
        mutator.prepare_mutation_input(
            _protein_protomers({"A": 3}),
            scores,
            runtime_config={},
        )


def test_prepare_mutation_input_pdb_mapping_rejects_unmapped_position(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "a_offset.pdb"
    _write_offset_pdb(pdb_path, {"A": 20})
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": [3]},
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        map_sequence_positions_to_pdb_residues=True,
    )
    scores = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": pdb_path},
    )

    with pytest.raises(ValueError, match="selector position 4"):
        mutator.prepare_mutation_input(
            _protein_protomers({"A": 3}),
            scores,
            runtime_config={},
        )


def test_sample_mutations_by_aa_probs_default_allows_wt_noop(tmp_path) -> None:
    _init_protein_runtime(tmp_path, "AA")
    global_config.set_variable("mutation_num", 2)

    mutator_utils.sample_mutations_by_aa_probs(
        evaluation_scores=SimpleNamespace(names=["A"], per_proto_scores={}),
        mutation_output=_mutation_output([{"A": 1.0}, {"A": 1.0}]),
        aa_order=AA_ORDER,
        mutation_weight=1.0,
        propagate_scores=False,
    )

    assert not global_config.segments["A_1"].mutations


def test_sample_mutations_by_aa_probs_enforced_uses_full_non_wt_budget(tmp_path) -> None:
    _init_protein_runtime(tmp_path, "AA")
    global_config.set_variable("mutation_num", 2)

    mutator_utils.sample_mutations_by_aa_probs(
        evaluation_scores=SimpleNamespace(names=["A"], per_proto_scores={}),
        mutation_output=_mutation_output(
            [
                {"A": 1.0},
                {"A": 1.0},
            ],
            log_rows=[
                {"A": 0.0, "V": -1000.0},
                {"A": 0.0, "G": -1000.0},
            ],
        ),
        aa_order=AA_ORDER,
        mutation_weight=1.0,
        propagate_scores=False,
        enforce_mutation_num=True,
    )

    mutations = global_config.segments["A_1"].mutations
    assert len(mutations) == 2
    assert {(site, from_aa, to_aa) for site, from_aa, to_aa, _, _ in mutations} == {
        (0, "A", "V"),
        (1, "A", "G"),
    }


def test_sample_mutations_by_aa_probs_enforced_rejects_short_budget(tmp_path) -> None:
    _init_protein_runtime(tmp_path, "AA")
    global_config.set_variable("mutation_num", 2)

    with pytest.raises(
        ValueError,
        match="enforce_mutation_num requested 2 mutations but only 1 non-WT",
    ):
        mutator_utils.sample_mutations_by_aa_probs(
            evaluation_scores=SimpleNamespace(names=["A"], per_proto_scores={}),
            mutation_output=_mutation_output(
                [
                    {"A": 1.0},
                    {},
                ],
                log_rows=[
                    {"A": 0.0, "V": -1000.0},
                    {},
                ],
            ),
            aa_order=AA_ORDER,
            mutation_weight=1.0,
            propagate_scores=False,
            enforce_mutation_num=True,
        )


def test_sample_mutations_by_aa_probs_filters_entropy_cutoff(monkeypatch) -> None:
    manager = _FakeSegmentManager(
        [
            _mutation(1, 0.4),
            _mutation(2, 1.2),
            _mutation(3, 1.6),
        ]
    )
    monkeypatch.setattr(
        mutator_utils,
        "global_config",
        SimpleNamespace(
            mutation_num=3,
            segment_manager=manager,
            protomers={},
        ),
    )
    monkeypatch.setattr(
        mutator_utils,
        "collect_blended_proto_probabilities",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        mutator_utils,
        "write_state_conflict_diagnostics",
        lambda *args, **kwargs: None,
    )

    mutator_utils.sample_mutations_by_aa_probs(
        evaluation_scores=SimpleNamespace(per_proto_scores={}),
        mutation_output={},
        aa_order=list("ACDEFGHIKLMNPQRSTVWY"),
        mutation_weight=1.0,
        propagate_scores=False,
        entropy_cutoff=1.2,
    )

    added, num_mutations = manager.added
    assert num_mutations == 2
    assert {mutation[1][0] for mutation in added} == {1, 2}


def test_sample_mutations_by_aa_probs_default_keeps_high_entropy(monkeypatch) -> None:
    manager = _FakeSegmentManager(
        [
            _mutation(1, 0.4),
            _mutation(2, 1.2),
            _mutation(3, 1.6),
        ]
    )
    monkeypatch.setattr(
        mutator_utils,
        "global_config",
        SimpleNamespace(
            mutation_num=3,
            segment_manager=manager,
            protomers={},
        ),
    )
    monkeypatch.setattr(
        mutator_utils,
        "collect_blended_proto_probabilities",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        mutator_utils,
        "write_state_conflict_diagnostics",
        lambda *args, **kwargs: None,
    )

    mutator_utils.sample_mutations_by_aa_probs(
        evaluation_scores=SimpleNamespace(per_proto_scores={}),
        mutation_output={},
        aa_order=list("ACDEFGHIKLMNPQRSTVWY"),
        mutation_weight=1.0,
        propagate_scores=False,
    )

    added, num_mutations = manager.added
    assert num_mutations == 3
    assert {mutation[1][0] for mutation in added} == {1, 2, 3}


def test_ligandmpnn_mutator_passes_sampling_options(monkeypatch) -> None:
    captured = {}

    def fake_sample(*args, **kwargs):
        captured["entropy_cutoff"] = kwargs.get("entropy_cutoff")
        captured["enforce_mutation_num"] = kwargs.get("enforce_mutation_num")

    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.sample_mutations_by_aa_probs",
        fake_sample,
    )
    mutator = LigandMPNNMutator(
        sequence_model_runner=SimpleNamespace(),
        entropy_cutoff=1.2,
        enforce_mutation_num=True,
    )
    mutator.mutation_weight = 1.0

    mutator.prepare_mutation_output(
        mutation_output={},
        protomers={},
        evaluation_scores=SimpleNamespace(),
        runtime_config={},
    )

    assert captured["entropy_cutoff"] == 1.2
    assert captured["enforce_mutation_num"] is True
