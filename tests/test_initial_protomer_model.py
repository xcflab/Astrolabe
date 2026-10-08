from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.msa_utils import parse_a3m_sequences
from astrolabe.core.protomer_config_parser import (
    Protomer,
    ProtomerTypes,
    SegmentManager,
    register_files,
)
from astrolabe.structure_start import build_structure_start_state


def _write_ala_pdb(path: Path, chain_lengths: dict[str, int]) -> Path:
    lines: list[str] = []
    serial = 1
    x = 0.0
    for chain_id, length in chain_lengths.items():
        for res_i in range(1, int(length) + 1):
            x += 1.0
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{res_i:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _init_runtime(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("out_dir", tmp_path)


def _initializer_args(
    tmp_path: Path,
    *,
    seq: str = "5-5",
    initializer: dict | None = None,
):
    return SimpleNamespace(
        exclude_AA=[],
        out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw",
        seqs={"A": seq},
        types={"A": "p"},
        homologies={},
        initializer=OmegaConf.create(initializer or {}),
        reinitialize_random_segments_with_dplm=False,
        mutator=OmegaConf.create(
            {
                "DPLM": {
                    "num_samples": 1,
                    "quality_filter_enable": False,
                },
                "mutator_terms": {},
            }
        ),
    )


def test_setup_protomers_owns_types_mutability_and_fixed_segments(
    tmp_path: Path,
) -> None:
    import run as run_mod

    _init_runtime(tmp_path)
    args = SimpleNamespace(
        seqs={"A": "ACD +GG", "R": "AUGU", "L": "LIG"},
        types={"R": "r", "L": "c"},
        homologies={},
    )

    run_mod.setup_protomers(args)

    assert global_config.protomers["A"].type == ProtomerTypes.PROTEIN
    assert global_config.protomers["R"].type == ProtomerTypes.RNA
    assert global_config.protomers["L"].type == ProtomerTypes.CCD
    assert sorted(global_config.mutable_protomers.keys()) == ["A", "R"]
    assert global_config.protomers["A"].sequence == "ACDGG"
    np.testing.assert_array_equal(
        global_config.protomers["A"].mutable_mask,
        np.asarray([True, True, True, False, False]),
    )


def test_protomer_full_msa_updates_only_query_after_mutation(tmp_path: Path) -> None:
    _init_runtime(tmp_path)
    protomer = Protomer(
        "A",
        "AC +DE",
        "protein",
        homologies={
            "unpairedMsa": ">query\nACDE\n>hit\nWYWF\n",
            "pairedMsa": ">query\nACDE\n>paired\nVVVV\n",
            "maskMutableResidues": False,
        },
    )

    protomer.set_residue(2, "G")
    assert parse_a3m_sequences(protomer.homology["unpairedMsa"]) == [
        ("query", "AGDE"),
        ("hit", "WYWF"),
    ]
    assert parse_a3m_sequences(protomer.homology["pairedMsa"]) == [
        ("query", "AGDE"),
        ("paired", "VVVV"),
    ]


def test_structure_start_validates_complex_oligomer_chain_mapping(
    tmp_path: Path,
) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 2, "Y": 3, "Z": 2})
    _init_runtime(tmp_path)
    register_files({"src": str(pdb)})
    global_config.protomers.update(
        {
            "A": Protomer("A", "src:X", "protein"),
            "B": Protomer("B", "+src:Y", "protein"),
        }
    )

    state = build_structure_start_state(
        {
            "references": {
                "ABA": {
                    "structure_alias": "src",
                    "chain_map": {"X": "A", "Y": "B", "Z": "A"},
                }
            }
        }
    )

    assert state is not None
    assert state.reference_scores.names == ["ABA"]
    assert state.reference_scores.complex_paths["ABA"].exists()
    manifest = state.manifest_path.read_text(encoding="utf-8")
    assert '"ordered_input_chains": [' in manifest
    assert '"X"' in manifest
    assert '"Y"' in manifest
    assert '"Z"' in manifest


def test_structure_start_rejects_reference_length_mismatch(tmp_path: Path) -> None:
    pdb = _write_ala_pdb(tmp_path / "src.pdb", {"X": 3})
    _init_runtime(tmp_path)
    register_files({"src": str(pdb)})
    global_config.protomers.update({"A": Protomer("A", "AA", "protein")})

    with pytest.raises(ValueError, match="reference length=3 protomer length=2"):
        build_structure_start_state(
            {
                "references": {
                    "A": {
                        "structure_alias": "src",
                        "chain_map": {"X": "A"},
                    }
                }
            }
        )


def test_dplm_initializer_length_mismatch_preserves_random_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import run as run_mod

    random.seed(5)
    np.random.seed(5)
    args = _initializer_args(
        tmp_path,
        seq="5-5",
        initializer={"mode": "dplm"},
    )
    _init_runtime(tmp_path)
    run_mod.setup_protomers(args)
    before = global_config.protomers["A"].sequence

    class _FakeDPLMRunner:
        def __init__(self, _runner_conf):
            pass

        def design_sequences(self, tasks):
            assert tasks[0]["name"] == "A"
            return {"A": [before + "A"]}

    monkeypatch.setattr(run_mod, "DPLMRunner", _FakeDPLMRunner)

    with caplog.at_level(logging.WARNING):
        run_mod.apply_initializer(args)

    assert global_config.protomers["A"].sequence == before
    assert global_config.segments["A_1"].initialized_from_random is True
    assert "Length mismatch while applying DPLM sequence" in caplog.text
