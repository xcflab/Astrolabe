from __future__ import annotations

import pathlib

import pytest

from astrolabe.antibody_framework_presets import (
    build_framework_homology_config,
    build_framework_binder_seq_from_structure,
    get_antibody_framework_preset,
    parse_cdr_length_overrides,
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
        # Covers H1/H2/H3 Chothia windows (26-32, 52-56, 95-102).
        serial, x = _append_ala_chain(
            lines=lines,
            serial_start=serial,
            chain_id="H",
            start_res=20,
            end_res=110,
            x0=x,
        )
    if include_l:
        # Covers L1/L2/L3 Chothia windows (24-34, 50-56, 89-97).
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


def test_parse_cdr_length_overrides():
    assert parse_cdr_length_overrides(None) == {}
    assert parse_cdr_length_overrides("") == {}
    assert parse_cdr_length_overrides("H1:7-11, H2:6, L3:9-12") == {
        "H1": "7-11",
        "H2": "6-6",
        "L3": "9-12",
    }


def test_parse_cdr_length_overrides_invalid_item():
    with pytest.raises(ValueError, match="expected '<LOOP>:<RANGE>'"):
        parse_cdr_length_overrides("H1-7-11")


def test_framework_alias_resolution_and_override():
    preset = get_antibody_framework_preset("vhh_7eow")
    assert preset.name == "vhh__boltzgen_7eow"
    seq = preset.binder_seq({"H1": "9-9"})
    assert "9-9" in seq
    with pytest.raises(ValueError, match="Unsupported CDR labels"):
        preset.binder_seq({"L1": "8-8"})


def test_build_framework_homology_config_for_builtin_preset():
    cfg = build_framework_homology_config(framework="vhh_7eow")
    assert cfg["_type"] == "antibody_framework_homology"
    assert cfg["framework_name"] == "vhh__boltzgen_7eow"
    assert cfg["source"].endswith(".json:A")
    assert pathlib.Path(cfg["source"].split(":", 1)[0]).is_file()
    assert cfg["layout"]["loop_order"] == ["H1", "H2", "H3"]
    assert cfg["layout"]["source_loop_lengths"] == {"H1": 9, "H2": 8, "H3": 21}


def test_build_framework_homology_config_for_builtin_scfv_preset():
    cfg = build_framework_homology_config(framework="scfv4d5_vhvl")
    assert cfg["_type"] == "antibody_framework_homology"
    assert cfg["framework_name"] == "scfv_vh_linker_vl__rfantibody_4d5"
    assert cfg["layout"]["loop_order"] == ["H1", "H2", "H3", "L1", "L2", "L3"]
    assert len(cfg["layout"]["fixed_segments"]) == len(cfg["layout"]["loop_order"]) + 1
    assert cfg["layout"]["fixed_segments"][3] == (
        "MDYWGQGTLVTVSSGGGGSGGGGSGGGGSDIQMTQSPSSLSASVGDRVTITC"
    )


def test_build_framework_from_structure_vhh_auto(tmp_path):
    pdb_path = _write_framework_pdb(tmp_path / "vhh.pdb", include_h=True, include_l=False)
    seq = build_framework_binder_seq_from_structure(structure=str(pdb_path))
    assert " 7-7 " in f" {seq} "
    assert " 5-5 " in f" {seq} "
    assert " 8-8 " in f" {seq} "
    assert "GGGGSGGGGSGGGGS" not in seq


def test_build_framework_from_structure_scfv_auto_and_partial_overrides(tmp_path):
    pdb_path = _write_framework_pdb(tmp_path / "scfv.pdb", include_h=True, include_l=True)
    seq = build_framework_binder_seq_from_structure(structure=str(pdb_path))
    assert "GGGGSGGGGSGGGGS" in seq
    assert " 11-11 " in f" {seq} "
    assert " 9-9 " in f" {seq} "

    seq2 = build_framework_binder_seq_from_structure(
        structure=str(pdb_path),
        cdr_lengths={"H1": "9-9", "L3": "12"},
    )
    assert " 9-9 " in f" {seq2} "
    assert " 12-12 " in f" {seq2} "


def test_build_framework_from_structure_scfv_requires_both_chains(tmp_path):
    pdb_path = _write_framework_pdb(tmp_path / "h_only.pdb", include_h=True, include_l=False)
    with pytest.raises(ValueError, match="requires both --framework-heavy-chain and --framework-light-chain"):
        build_framework_binder_seq_from_structure(
            structure=str(pdb_path),
            kind="scfv",
            heavy_chain="H",
        )


def test_build_framework_from_structure_rejects_wrong_cdr_labels(tmp_path):
    pdb_path = _write_framework_pdb(tmp_path / "h_only_bad_label.pdb", include_h=True, include_l=False)
    with pytest.raises(ValueError, match="Unsupported CDR labels"):
        build_framework_binder_seq_from_structure(
            structure=str(pdb_path),
            kind="vhh",
            heavy_chain="H",
            cdr_lengths={"L1": "8-8"},
        )
