from __future__ import annotations

import pathlib
import json

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.core.msa_utils import parse_a3m_sequences, parse_homology_arg
from astrolabe.antibody_framework_presets import (
    build_framework_homology_config,
    build_framework_layout_from_structure,
)


def _append_ala_chain(
    *,
    lines: list[str],
    serial_start: int,
    chain_id: str,
    start_res: int,
    end_res: int,
    x0: float,
) -> tuple[int, float]:
    serial = serial_start
    x = x0
    for res_i in range(start_res, end_res + 1):
        x += 1.0
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        serial += 1
    lines.append("TER")
    return serial, x


def _write_framework_pdb(
    path: pathlib.Path,
    *,
    include_h: bool = True,
    include_l: bool = False,
) -> pathlib.Path:
    lines: list[str] = []
    serial = 1
    x = 0.0
    if include_h:
        serial, x = _append_ala_chain(
            lines=lines,
            serial_start=serial,
            chain_id="H",
            start_res=20,
            end_res=110,
            x0=x,
        )
    if include_l:
        serial, x = _append_ala_chain(
            lines=lines,
            serial_start=serial,
            chain_id="L",
            start_res=20,
            end_res=100,
            x0=x,
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _init_protomer_globals() -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("nonstandard_aliases", {})
    global_config.init_const("user_ccd", None)


def test_parse_homology_arg_stretches_antibody_framework_spec_dict():
    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": {
                "unpairedMsa": ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
                "pairedMsa": ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
                "templates": [
                    {
                        "mmcif": "data_test",
                        "queryIndices": [0, 1, 4, 5, 9, 10],
                        "templateIndices": [100, 101, 104, 105, 109, 110],
                    }
                ],
            },
        }
    )

    result = parse_homology_arg(
        "AAQQQBBRCC",
        [0, 1, 5, 6, 8, 9],
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    paired = parse_a3m_sequences(result["pairedMsa"])

    assert unpaired[0][1] == "AAQQQBBRCC"
    assert unpaired[1][1] == "AA---BB-CC"
    assert paired[0][1] == "AAQQQBBRCC"
    assert paired[1][1] == "AA---BB-CC"
    assert result["templates"][0].query_to_template_map == {
        0: 100,
        1: 101,
        5: 104,
        6: 105,
        8: 109,
        9: 110,
    }


def test_parse_homology_arg_stretches_antibody_framework_spec_file(tmp_path: pathlib.Path):
    spec_path = tmp_path / "toy_framework.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "unpairedMsa: |",
                "  >query",
                "  AA--BB---CC",
                "  >hit1",
                "  AA--BB---CC",
                "templates:",
                "  - mmcif: data_test",
                "    queryIndices: [0, 1, 4, 5, 9, 10]",
                "    templateIndices: [10, 11, 14, 15, 19, 20]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": str(spec_path),
        }
    )

    result = parse_homology_arg(
        "AAQQQBBRCC",
        [0, 1, 5, 6, 8, 9],
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    assert unpaired[0][1] == "AAQQQBBRCC"
    assert unpaired[1][1] == "AA---BB-CC"
    assert result["templates"][0].query_to_template_map == {
        0: 10,
        1: 11,
        5: 14,
        6: 15,
        8: 19,
        9: 20,
    }


def test_parse_homology_arg_stretches_antibody_framework_spec_file_with_relative_msa_paths(
    tmp_path: pathlib.Path,
):
    unpaired_path = tmp_path / "toy_framework_unpaired.a3m"
    paired_path = tmp_path / "toy_framework_paired.a3m"
    unpaired_path.write_text(
        ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
        encoding="utf-8",
    )
    paired_path.write_text(
        ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "toy_framework_paths.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "unpairedMsa: toy_framework_unpaired.a3m",
                "pairedMsa: toy_framework_paired.a3m",
                "templates:",
                "  - mmcif: data_test",
                "    queryIndices: [0, 1, 4, 5, 9, 10]",
                "    templateIndices: [30, 31, 34, 35, 39, 40]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": str(spec_path),
        }
    )

    result = parse_homology_arg(
        "AAQQQBBRCC",
        [0, 1, 5, 6, 8, 9],
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    paired = parse_a3m_sequences(result["pairedMsa"])
    assert unpaired[0][1] == "AAQQQBBRCC"
    assert unpaired[1][1] == "AA---BB-CC"
    assert paired[0][1] == "AAQQQBBRCC"
    assert paired[1][1] == "AA---BB-CC"
    assert result["templates"][0].query_to_template_map == {
        0: 30,
        1: 31,
        5: 34,
        6: 35,
        8: 39,
        9: 40,
    }


def test_parse_homology_arg_stretches_antibody_framework_a3m_source_file(tmp_path: pathlib.Path):
    a3m_path = tmp_path / "toy_framework.a3m"
    a3m_path.write_text(
        ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
        encoding="utf-8",
    )
    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": str(a3m_path),
        }
    )

    result = parse_homology_arg(
        "AAQQQBBRCC",
        [0, 1, 5, 6, 8, 9],
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    assert unpaired[0][1] == "AAQQQBBRCC"
    assert unpaired[1][1] == "AA---BB-CC"


def test_parse_homology_arg_stretches_antibody_framework_af3_json_chain(tmp_path: pathlib.Path):
    json_path = tmp_path / "toy_framework.json"
    json_path.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAQQBBRRRCC",
                            "unpairedMsa": ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
                            "pairedMsa": ">query\nAA--BB---CC\n>hit1\nAA--BB---CC\n",
                            "templates": [
                                {
                                    "mmcif": "data_test",
                                    "queryIndices": [0, 1, 4, 5, 9, 10],
                                    "templateIndices": [20, 21, 24, 25, 29, 30],
                                }
                            ],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": f"{json_path}:A",
        }
    )

    result = parse_homology_arg(
        "AAQQQBBRCC",
        [0, 1, 5, 6, 8, 9],
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    assert unpaired[0][1] == "AAQQQBBRCC"
    assert unpaired[1][1] == "AA---BB-CC"
    assert result["templates"][0].query_to_template_map == {
        0: 20,
        1: 21,
        5: 24,
        6: 25,
        8: 29,
        9: 30,
    }


def test_parse_homology_arg_rejects_antibody_framework_source_mismatch():
    spec = OmegaConf.create(
        {
            "_type": "antibody_framework_homology",
            "framework_name": "toy_framework",
            "layout": {
                "fixed_segments": ["AA", "BB", "CC"],
                "loop_order": ["H1", "H2"],
                "source_loop_lengths": {"H1": 2, "H2": 3},
            },
            "source": {
                "unpairedMsa": ">query\nAX--BB---CC\n>hit1\nAX--BB---CC\n",
            },
        }
    )

    with pytest.raises(ValueError, match="fixed framework segment"):
        parse_homology_arg(
            "AAQQQBBRCC",
            [0, 1, 5, 6, 8, 9],
            spec,
        )


def test_parse_homology_arg_accepts_builtin_framework_resource():
    spec = OmegaConf.create(
        build_framework_homology_config(framework="vhh_7eow")
    )
    current_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAAS"
        + "Q" * 7
        + "GWFRQAPGKGRELVAAI"
        + "R" * 6
        + "YPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCA"
        + "S" * 15
        + "GQGTQVTVSS"
    )
    fixed_residues = (
        list(range(0, 25))
        + list(range(32, 49))
        + list(range(55, 93))
        + list(range(108, 118))
    )

    result = parse_homology_arg(
        current_seq,
        fixed_residues,
        spec,
    )

    unpaired = parse_a3m_sequences(result["unpairedMsa"])
    assert unpaired[0][1] == current_seq
    assert len(unpaired[1][1]) == len(current_seq)


def test_prepare_framework_homology_payload_structure_only_templates_are_applied_in_protomer_debug(
    tmp_path: pathlib.Path,
):
    framework_pdb = _write_framework_pdb(tmp_path / "framework_only_h.pdb", include_h=True)
    binder_seq, framework_layout = build_framework_layout_from_structure(
        structure=str(framework_pdb),
        kind="vhh",
        heavy_chain="H",
        cdr_lengths={"H1": "10-10"},
    )
    homology_cfg = cli._prepare_framework_homology_payload(
        binder_seq=binder_seq,
        framework_layout=framework_layout,
        framework_homology_source=None,
    )

    assert sorted(homology_cfg.keys()) == ["templates"]

    _init_protomer_globals()
    protomer = Protomer("A", binder_seq, "protein", homology_cfg)
    homology = protomer.homology

    assert "templates" in homology
    assert len(homology["templates"]) == 1
    template = homology["templates"][0]
    fixed_positions = (~protomer.mutable_mask).nonzero()[0].tolist()
    mutable_positions = protomer.mutable_mask.nonzero()[0].tolist()
    assert template.mmcif is not None
    assert template.mmcif.startswith("data_template")
    assert set(template.query_to_template_map) == set(fixed_positions)
    assert mutable_positions[0] not in template.query_to_template_map
    assert template.query_to_template_map[fixed_positions[0]] == 0
    assert template.query_to_template_map[fixed_positions[-1]] == 90


def test_prepare_framework_homology_payload_explicit_a3m_overrides_structure_templates(
    tmp_path: pathlib.Path,
):
    framework_pdb = _write_framework_pdb(tmp_path / "framework_only_h.pdb", include_h=True)
    binder_seq, framework_layout = build_framework_layout_from_structure(
        structure=str(framework_pdb),
        kind="vhh",
        heavy_chain="H",
        cdr_lengths={"H1": "10-10"},
    )
    source_query_len = sum(len(seg) for seg in framework_layout["fixed_segments"]) + sum(
        int(v) for v in framework_layout["source_loop_lengths"].values()
    )
    homology_a3m = tmp_path / "framework_external.a3m"
    homology_a3m.write_text(
        ">query\n" + ("A" * source_query_len) + "\n>hit1\n" + ("A" * source_query_len) + "\n",
        encoding="utf-8",
    )
    homology_cfg = cli._prepare_framework_homology_payload(
        binder_seq=binder_seq,
        framework_layout=framework_layout,
        framework_homology_source=str(homology_a3m),
    )

    assert sorted(homology_cfg.keys()) == ["unpairedMsa"]
    msa = parse_a3m_sequences(homology_cfg["unpairedMsa"])

    _init_protomer_globals()
    protomer = Protomer("A", binder_seq, "protein", homology_cfg)
    homology = protomer.homology

    assert len(msa[0][1]) == protomer.length
    assert len(msa[1][1]) == protomer.length
    assert "unpairedMsa" in homology
    assert "templates" not in homology


def test_prepare_framework_homology_payload_explicit_template_overrides_structure_templates(
    tmp_path: pathlib.Path,
):
    framework_pdb = _write_framework_pdb(tmp_path / "framework_only_h.pdb", include_h=True)
    binder_seq, framework_layout = build_framework_layout_from_structure(
        structure=str(framework_pdb),
        kind="vhh",
        heavy_chain="H",
        cdr_lengths={"H1": "10-10"},
    )
    source_query_len = sum(len(seg) for seg in framework_layout["fixed_segments"]) + sum(
        int(v) for v in framework_layout["source_loop_lengths"].values()
    )
    homology_json = tmp_path / "framework_external.json"
    homology_json.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "version": 1,
                "name": "framework_external",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "A" * source_query_len,
                            "templates": [
                                {
                                    "mmcif": "data_external",
                                    "queryIndices": list(range(source_query_len)),
                                    "templateIndices": [500 + i for i in range(source_query_len)],
                                }
                            ],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    homology_cfg = cli._prepare_framework_homology_payload(
        binder_seq=binder_seq,
        framework_layout=framework_layout,
        framework_homology_source=f"{homology_json}:A",
    )

    assert sorted(homology_cfg.keys()) == ["templates"]

    _init_protomer_globals()
    protomer = Protomer("A", binder_seq, "protein", homology_cfg)
    homology = protomer.homology

    assert "templates" in homology
    assert len(homology["templates"]) == 1
    template = homology["templates"][0]
    fixed_positions = (~protomer.mutable_mask).nonzero()[0].tolist()
    mutable_positions = protomer.mutable_mask.nonzero()[0].tolist()
    assert template.mmcif == "data_external"
    assert set(template.query_to_template_map) == set(fixed_positions)
    assert mutable_positions[0] not in template.query_to_template_map
    assert template.query_to_template_map[fixed_positions[0]] == 500
    assert template.query_to_template_map[fixed_positions[-1]] == 590
