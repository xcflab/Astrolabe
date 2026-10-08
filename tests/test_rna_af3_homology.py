from __future__ import annotations

import json
from pathlib import Path

from astrolabe.cli import _load_af3_chain_dict_from_homology, _sequence_from_af3_json
from astrolabe.core.msa_utils import parse_a3m_sequences, parse_homology_arg


def _write_rna_af3_json(tmp_path: Path) -> Path:
    path = tmp_path / "rna_input.json"
    path.write_text(
        json.dumps(
            {
                "name": "rna_target",
                "sequences": [
                    {
                        "rna": {
                            "id": "A",
                            "sequence": "AUGC",
                            "unpairedMsa": ">query\nAUGC\n>hit\nAUGU\n",
                        }
                    }
                ],
                "modelSeeds": [1],
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_dna_af3_json(tmp_path: Path) -> Path:
    path = tmp_path / "dna_input.json"
    path.write_text(
        json.dumps(
            {
                "name": "dna_target",
                "sequences": [
                    {
                        "dna": {
                            "id": "B",
                            "sequence": "ATGC",
                        }
                    }
                ],
                "modelSeeds": [1],
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_sequence_from_af3_json_supports_rna_chain(tmp_path: Path) -> None:
    json_path = _write_rna_af3_json(tmp_path)

    sequence = _sequence_from_af3_json(f"{json_path}:A")

    assert sequence == "AUGC"


def test_load_af3_chain_dict_from_homology_supports_rna_chain(tmp_path: Path) -> None:
    json_path = _write_rna_af3_json(tmp_path)

    chain_data = _load_af3_chain_dict_from_homology(f"{json_path}:A")

    assert chain_data["sequence"] == "AUGC"
    assert chain_data["unpairedMsa"].startswith(">query")


def test_parse_homology_arg_supports_rna_af3_json(tmp_path: Path) -> None:
    json_path = _write_rna_af3_json(tmp_path)

    homology = parse_homology_arg(
        sequence="AUGC",
        fixed_residues=[],
        homologies=f"{json_path}:A",
    )

    assert homology["unpairedMsa"].startswith(">query")
    assert homology["pairedMsa"] is None
    assert homology["templates"] is None


def test_sequence_from_af3_json_supports_dna_chain(tmp_path: Path) -> None:
    json_path = _write_dna_af3_json(tmp_path)

    sequence = _sequence_from_af3_json(f"{json_path}:B")

    assert sequence == "ATGC"


def test_load_af3_chain_dict_from_homology_supports_dna_chain(tmp_path: Path) -> None:
    json_path = _write_dna_af3_json(tmp_path)

    chain_data = _load_af3_chain_dict_from_homology(f"{json_path}:B")

    assert chain_data["sequence"] == "ATGC"


def test_parse_homology_arg_supports_dna_af3_json(tmp_path: Path) -> None:
    json_path = _write_dna_af3_json(tmp_path)

    homology = parse_homology_arg(
        sequence="ATGC",
        fixed_residues=[],
        homologies=f"{json_path}:B",
    )

    assert homology["unpairedMsa"] is None
    assert homology["pairedMsa"] is None
    assert homology["templates"] is None


def test_parse_homology_arg_can_keep_mutable_template_residues() -> None:
    raw_template = {
        "mmcif": "data_template\n",
        "queryIndices": [0, 1, 2, 3],
        "templateIndices": [10, 11, 12, 13],
        "includeMutableResidues": True,
    }

    homology = parse_homology_arg(
        sequence="ACDE",
        fixed_residues=[1],
        homologies={"templates": [raw_template]},
    )

    assert homology["templates"][0].query_to_template_map == {
        0: 10,
        1: 11,
        2: 12,
        3: 13,
    }


def test_parse_homology_arg_can_keep_mutable_msa_columns() -> None:
    homology = parse_homology_arg(
        sequence="ACDE",
        fixed_residues=[0, 2],
        homologies={
            "unpairedMsa": ">query\nACDE\n>hit\nWYWF\n",
            "pairedMsa": ">query\nACDE\n>paired\nVVVV\n",
            "maskMutableResidues": False,
        },
    )

    assert parse_a3m_sequences(homology["unpairedMsa"]) == [
        ("query", "ACDE"),
        ("hit", "WYWF"),
    ]
    assert parse_a3m_sequences(homology["pairedMsa"]) == [
        ("query", "ACDE"),
        ("paired", "VVVV"),
    ]
