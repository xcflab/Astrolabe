from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from prody import parsePDB, writeMMCIF

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.proteinmpnn import (
    MultistateProteinMPNNMutator,
    ProteinMPNNMutator,
)
from astrolabe.mutators.registry import MutatorRegistry


_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def design_sequences(self, **kwargs):
        self.calls.append(dict(kwargs))
        rows = []
        for prob in (0.91, 0.82, 0.73):
            rows.append(_row(prob))
        rows.extend([_row(0.21), _row(0.22)])
        for prob in (0.91, 0.82, 0.73):
            rows.append(_row(prob))
        rows.extend([_row(0.31), _row(0.32), _row(0.33), _row(0.34)])
        return {
            "sampling_probs": np.asarray([rows], dtype=float),
            "chain_letters": tuple("AAABBCCCDDDD"),
        }


def _row(a_prob: float) -> list[float]:
    row = [0.0] * 20
    row[0] = a_prob
    row[1] = 1.0 - a_prob
    return row


def _reset_global(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw")
    global_config.set_variable("step", 11)


def _write_two_chain_pdb(path: Path, binder_seq: str, target_seq: str) -> None:
    lines = []
    serial = 1
    for chain_id, seq, y_base in (("A", binder_seq, 0.0), ("B", target_seq, 30.0)):
        for resi, aa in enumerate(seq, start=1):
            resname = _AA3[aa]
            for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
                x = resi * 3.8 + atom_index * 0.4
                y = y_base + atom_index * 0.2
                z = atom_index * 0.1
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:<4} {resname:>3} {chain_id}"
                    f"{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00 20.00           {element:>2}"
                )
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coords_by_chain(path: Path) -> dict[str, list[np.ndarray]]:
    coords: dict[str, list[np.ndarray]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        coords.setdefault(line[21].strip(), []).append(
            np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=float,
            )
        )
    return coords


def _min_distance(first: np.ndarray, second: np.ndarray) -> float:
    diff = first[:, None, :] - second[None, :, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=-1))))


def test_multistate_proteinmpnn_single_call_and_split(
    tmp_path: Path,
) -> None:
    _reset_global(tmp_path)
    ab_pdb = tmp_path / "AB.pdb"
    ac_pdb = tmp_path / "AC.pdb"
    _write_two_chain_pdb(ab_pdb, binder_seq="ACD", target_seq="EF")
    _write_two_chain_pdb(ac_pdb, binder_seq="ACD", target_seq="GHIK")

    runner = _RecordingRunner()
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=runner,
        translation_x=5000.0,
    )
    mutator.alias = "LigandMPNN_itf"

    output = mutator.compute_mutation(
        {
            "AB": {
                "pdb_path": str(ab_pdb),
                "fixed_residues": "A1 B2",
                "redesigned_residues": "A2 B1",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "transmembrane_interface": "A3 B1",
                "transmembrane_buried": "",
            },
            "AC": {
                "pdb_path": str(ac_pdb),
                "fixed_residues": "B3",
                "redesigned_residues": "B4",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "transmembrane_interface": "",
                "transmembrane_buried": "A2 B4",
            },
        }
    )

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert Path(call["pdb_path"]).exists()
    assert call["parse_these_chains_only"] == "ABCD"
    assert call["fixed_residues"].split() == ["A1", "B2", "D3"]
    assert call["redesigned_residues"].split() == ["A2", "B1", "D4"]
    assert call["transmembrane_interface"].split() == ["A3", "B1"]
    assert call["transmembrane_buried"].split() == ["C2", "D4"]
    assert call["symmetry_residues"].split("|") == [
        "A1,C1",
        "A2,C2",
        "A3,C3",
    ]
    assert call["symmetry_weights"].split("|") == ["0.5,0.5"] * 3
    assert not any(
        token[0] in {"B", "D"}
        for tie in call["symmetry_residues"].split("|")
        for token in tie.split(",")
    )

    coords = _coords_by_chain(Path(call["pdb_path"]))
    assert sorted(coords) == ["A", "B", "C", "D"]
    first_state = np.asarray(coords["A"] + coords["B"], dtype=float)
    second_state = np.asarray(coords["C"] + coords["D"], dtype=float)
    assert _min_distance(first_state, second_state) > 1000.0
    assert np.min(second_state[:, 0]) == pytest.approx(5003.8)
    records = [
        line
        for line in Path(call["pdb_path"]).read_text().splitlines()
        if line.startswith(("ATOM", "HETATM", "TER"))
    ]
    assert sum(line.startswith("TER") for line in records) == 4
    assert [int(line[6:11]) for line in records] == list(range(1, len(records) + 1))
    assert "/MultistateProteinMPNN/LigandMPNN_itf/combined_inputs/" in str(
        call["pdb_path"]
    )

    assert output["AB"]["sampling_probs"].shape == (1, 5, 20)
    assert output["AC"]["sampling_probs"].shape == (1, 7, 20)
    np.testing.assert_array_equal(
        output["AB"]["sampling_probs"][:, :3, :],
        output["AC"]["sampling_probs"][:, :3, :],
    )
    provenance = output["AB"]["multistate_proteinmpnn_provenance"]
    assert provenance["runner_call_count"] == 1
    assert provenance["no_probability_average_fallback"] is True
    assert provenance["tie_count"] == 3
    assert provenance["chain_maps"]["AC"] == {"A": "C", "B": "D"}
    assert Path(provenance["provenance_json"]).exists()


def test_multistate_proteinmpnn_rejects_short_translation() -> None:
    with pytest.raises(ValueError, match="translation_x"):
        MultistateProteinMPNNMutator(
            sequence_model_runner=_RecordingRunner(),
            translation_x=999.0,
        )


def test_multistate_proteinmpnn_applies_explicit_signed_state_betas(
    tmp_path: Path,
) -> None:
    _reset_global(tmp_path)
    ab_pdb = tmp_path / "AB.pdb"
    ac_pdb = tmp_path / "AC.pdb"
    _write_two_chain_pdb(ab_pdb, binder_seq="ACD", target_seq="EF")
    _write_two_chain_pdb(ac_pdb, binder_seq="ACD", target_seq="GHIK")
    runner = _RecordingRunner()
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=runner,
        state_betas={"AB": 1.0, "AC": -2.0},
    )

    output = mutator.compute_mutation(
        {
            "AB": {"pdb_path": str(ab_pdb), "fixed_residues": ""},
            "AC": {"pdb_path": str(ac_pdb), "fixed_residues": ""},
        }
    )

    assert runner.calls[0]["symmetry_weights"].split("|") == ["0.5,-1.0"] * 3
    provenance = output["AB"]["multistate_proteinmpnn_provenance"]
    assert provenance["state_betas"] == {"AB": 1.0, "AC": -2.0}
    assert provenance["effective_state_logit_weights"] == {
        "AB": 0.5,
        "AC": -1.0,
    }
    assert provenance["negative_state_oligos"] == ["AC"]


@pytest.mark.parametrize(
    ("state_betas", "message"),
    [
        ({"AB": 0.0, "AC": 0.0}, "non-zero"),
        ({"AB": 1.0, "AC": float("nan")}, "finite"),
    ],
)
def test_multistate_proteinmpnn_rejects_invalid_state_betas(
    state_betas,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MultistateProteinMPNNMutator(
            sequence_model_runner=_RecordingRunner(),
            state_betas=state_betas,
        )


def test_multistate_proteinmpnn_rejects_state_beta_keys_not_in_stage_oligo() -> None:
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=_RecordingRunner(),
        state_betas={"AB": 1.0},
    )

    with pytest.raises(ValueError, match="non-ignored stage oligos"):
        mutator.compute_mutation({"AB": {}, "AC": {}})


@pytest.mark.parametrize(
    ("runtime_config", "message"),
    [
        (
            {
                "oligo": ["AB", "AC"],
                "ignored_complexes": ["AC"],
            },
            "exactly two distinct non-ignored",
        ),
        (
            {
                "oligo": ["AB", "AC"],
                "negative_design_complexes": [],
            },
            "signs must match",
        ),
    ],
)
def test_multistate_proteinmpnn_rejects_marker_conflicts(
    monkeypatch,
    runtime_config,
    message: str,
) -> None:
    monkeypatch.setattr(
        ProteinMPNNMutator,
        "prepare_mutation_input",
        lambda *_args, **_kwargs: {"AB": {}, "AC": {}},
    )
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=_RecordingRunner(),
        state_betas={"AB": 1.0, "AC": -2.0},
    )

    with pytest.raises(ValueError, match=message):
        mutator.prepare_mutation_input({}, SimpleNamespace(), runtime_config)


def test_multistate_proteinmpnn_infers_beta_from_negative_oligo_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _reset_global(tmp_path)
    ab_pdb = tmp_path / "AB.pdb"
    ac_pdb = tmp_path / "AC.pdb"
    _write_two_chain_pdb(ab_pdb, binder_seq="ACD", target_seq="EF")
    _write_two_chain_pdb(ac_pdb, binder_seq="ACD", target_seq="GHIK")
    monkeypatch.setattr(
        "astrolabe.mutators.ligandmpnn.prepare_exclude_sites",
        lambda *_args, **_kwargs: {"A": [], "B": [], "C": []},
    )
    protomers = {
        name: SimpleNamespace(length=length, tied_positions=[])
        for name, length in {"A": 3, "B": 2, "C": 4}.items()
    }
    scores = StandardScores(
        names=["AB", "AC"],
        plddt={"AB": [0.9] * 5, "AC": [0.9] * 7},
        complex_paths={"AB": ab_pdb, "AC": ac_pdb},
    )
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=_RecordingRunner(),
    )

    runtime_config = {
        "oligo": ["AB", "AC", "AD"],
        "ignored_complexes": ["AD"],
        "negative_design_complexes": ["AC"],
    }
    prepared = mutator.prepare_mutation_input(
        protomers,
        scores,
        runtime_config=runtime_config,
    )

    assert list(prepared) == ["AB", "AC"]
    assert prepared["AC"]["pdb_path"] == str(ac_pdb)
    assert prepared["AB"]["_multistate_beta"] == 1.0
    assert prepared["AC"]["_multistate_beta"] == -1.0
    assert mutator.state_oligos == ["AB", "AC"]
    assert mutator.shared_protomer == "A"
    output = mutator.compute_mutation(prepared)
    provenance = output["AB"]["multistate_proteinmpnn_provenance"]
    assert provenance["state_betas"] == {"AB": 1.0, "AC": -1.0}
    assert provenance["state_oligos"] == ["AB", "AC"]
    assert provenance["shared_protomer"] == "A"
    assert runtime_config == {
        "oligo": ["AB", "AC", "AD"],
        "ignored_complexes": ["AD"],
        "negative_design_complexes": ["AC"],
    }


def test_multistate_proteinmpnn_infers_non_a_shared_protomer_from_stage_oligo(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ProteinMPNNMutator,
        "prepare_mutation_input",
        lambda *_args, **_kwargs: {"AB": {}, "BC": {}},
    )
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=_RecordingRunner(),
    )

    prepared = mutator.prepare_mutation_input(
        {},
        SimpleNamespace(names=["AB", "BC", "AD"]),
        {
            "oligo": ["AB", "BC", "AD"],
            "ignored_complexes": ["AD"],
            "negative_design_complexes": ["BC"],
        },
    )

    assert list(prepared) == ["AB", "BC"]
    assert mutator.state_oligos == ["AB", "BC"]
    assert mutator.shared_protomer == "B"
    assert prepared["AB"]["_multistate_beta"] == 1.0
    assert prepared["BC"]["_multistate_beta"] == -1.0


def test_multistate_proteinmpnn_rejects_ambiguous_shared_protomer() -> None:
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=_RecordingRunner(),
    )

    with pytest.raises(ValueError, match="exactly one shared protomer"):
        mutator.prepare_mutation_input(
            {},
            SimpleNamespace(names=["AB", "ABC"]),
            {
                "oligo": ["AB", "ABC"],
                "ignored_complexes": [],
                "negative_design_complexes": ["ABC"],
            },
        )


def test_multistate_proteinmpnn_rejects_mismatched_shared_binder(
    tmp_path: Path,
) -> None:
    _reset_global(tmp_path)
    ab_pdb = tmp_path / "AB.pdb"
    ac_pdb = tmp_path / "AC.pdb"
    _write_two_chain_pdb(ab_pdb, binder_seq="ACD", target_seq="EF")
    _write_two_chain_pdb(ac_pdb, binder_seq="AGD", target_seq="GHIK")
    runner = _RecordingRunner()
    mutator = MultistateProteinMPNNMutator(
        sequence_model_runner=runner,
        translation_x=5000.0,
    )

    with pytest.raises(ValueError, match="binder sequences differ"):
        mutator.compute_mutation(
            {
                "AB": {"pdb_path": str(ab_pdb), "fixed_residues": ""},
                "AC": {"pdb_path": str(ac_pdb), "fixed_residues": ""},
            }
        )
    assert runner.calls == []


def test_multistate_proteinmpnn_factory_and_compatibility_aliases_are_registered() -> None:
    import astrolabe.mutators.base  # noqa: F401

    factory = MutatorRegistry.get("MultistateProteinMPNN")
    assert factory is MutatorRegistry.get("multistate_proteinmpnn")
    assert factory is MutatorRegistry.get("FarApartTiedMultitargetLigandMPNN")
    assert factory is MutatorRegistry.get("far_apart_tied_multitarget_ligandmpnn")
    assert issubclass(MultistateProteinMPNNMutator, ProteinMPNNMutator)


def test_proteinmpnn_factory_forces_protein_model_and_alias_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import astrolabe.mutators.base as base_module

    captured = {}

    def fake_build(mutator_cls, term_conf, **kwargs):
        captured["mutator_cls"] = mutator_cls
        captured["term_conf"] = term_conf
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(base_module, "_build_mpnn_mutator", fake_build)
    factory = MutatorRegistry.get("ProteinMPNN")
    factory(
        {},
        alias="sequence",
        mutator_conf=object(),
        raw_model_results_dir=tmp_path,
    )

    assert captured["mutator_cls"] is ProteinMPNNMutator
    assert captured["term_conf"]["model_type"] == "protein_mpnn"
    assert captured["output_name"] == "ProteinMPNN_sequence"


def test_multistate_proteinmpnn_factory_forces_protein_model_and_alias_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import astrolabe.mutators.base as base_module

    captured = {}

    def fake_build(mutator_cls, term_conf, **kwargs):
        captured["mutator_cls"] = mutator_cls
        captured["term_conf"] = term_conf
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(base_module, "_build_mpnn_mutator", fake_build)
    factory = MutatorRegistry.get("MultistateProteinMPNN")
    factory(
        {},
        alias="LigandMPNN_itf",
        mutator_conf=object(),
        raw_model_results_dir=tmp_path,
    )

    assert captured["mutator_cls"] is MultistateProteinMPNNMutator
    assert captured["term_conf"]["model_type"] == "protein_mpnn"
    assert captured["output_name"] == "MultistateProteinMPNN_LigandMPNN_itf"


def test_multistate_proteinmpnn_accepts_cif_and_separates_alias_artifacts(
    tmp_path: Path,
) -> None:
    _reset_global(tmp_path)
    ab_pdb = tmp_path / "AB.pdb"
    ac_pdb = tmp_path / "AC.pdb"
    _write_two_chain_pdb(ab_pdb, binder_seq="ACD", target_seq="EF")
    _write_two_chain_pdb(ac_pdb, binder_seq="ACD", target_seq="GHIK")
    ab_cif = Path(writeMMCIF(str(tmp_path / "AB.cif"), parsePDB(str(ab_pdb))))
    ac_cif = Path(writeMMCIF(str(tmp_path / "AC.cif"), parsePDB(str(ac_pdb))))

    paths = []
    for alias in ("LigandMPNN_itf", "LigandMPNN_noitf"):
        runner = _RecordingRunner()
        mutator = MultistateProteinMPNNMutator(
            sequence_model_runner=runner,
            translation_x=5000.0,
        )
        mutator.alias = alias
        mutator.compute_mutation(
            {
                "AB": {"pdb_path": str(ab_cif), "fixed_residues": ""},
                "AC": {"pdb_path": str(ac_cif), "fixed_residues": ""},
            }
        )
        paths.append(Path(runner.calls[0]["pdb_path"]))

    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)
