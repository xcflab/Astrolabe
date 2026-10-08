from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.cli import _compose_validate_basic
from astrolabe.initializers import (
    AminoAcidSetInitializer,
    AllAInitializer,
    DPLMInitializer,
    ExistingStructureInitializer,
    MaskMixInitializer,
    PSSMInitializer,
    RFD3Initializer,
    RandomInitializer,
    XMixInitializer,
)
from astrolabe.initializers.wrappers.rfd3 import RFD3Runner, RFD3StructureOutput
from astrolabe.mutators.capabilities import MutatorCapabilities
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator


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
    "X": "UNK",
}


def _pdb_lines(
    chain: str,
    residue_ids: tuple[tuple[int, str], ...] = (
        (1, ""),
        (2, ""),
        (3, ""),
        (4, ""),
    ),
    sequence: str | None = None,
) -> list[str]:
    sequence = sequence or "A" * len(residue_ids)
    assert len(sequence) == len(residue_ids)
    lines = []
    for serial, ((residue_number, insertion_code), amino_acid) in enumerate(
        zip(residue_ids, sequence, strict=True),
        start=1,
    ):
        lines.append(
            f"ATOM  {serial:5d}  CA  {_AA3[amino_acid]} {chain:1}"
            f"{residue_number:4d}{insertion_code:1}   "
            f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}           C"
        )
    return lines


def _write_pdb(
    path: Path,
    *,
    chain: str,
    residue_ids: tuple[tuple[int, str], ...] = (
        (1, ""),
        (2, ""),
        (3, ""),
        (4, ""),
    ),
    sequence: str | None = None,
) -> None:
    lines = _pdb_lines(chain, residue_ids, sequence)
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _args(tmp_path: Path, *, seq: str, initializer: dict):
    return types.SimpleNamespace(
        exclude_AA=["C"],
        initial_temperature=0.05,
        out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw",
        use_metropolis_acceptance=True,
        seqs={"A": seq},
        types={"A": "p"},
        homologies={},
        cyclization=None,
        initializer=OmegaConf.create(initializer),
        reinitialize_random_segments_with_dplm=False,
        mutator=OmegaConf.create({"mutator_terms": {}}),
    )


def _setup(args):
    import run as run_mod

    global_config.reset()
    run_mod.initialize_global_config(args)
    run_mod.setup_protomers(args)
    return run_mod


class _Designer:
    capabilities = MutatorCapabilities(supports_initialization_from_structure=True)

    def __init__(self, sequences):
        self.sequences = sequences
        self.calls = []

    def design_initial_sequences(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.sequences)


def test_build_initializer_returns_classes_for_existing_modes(tmp_path: Path) -> None:
    run_mod = _setup(_args(tmp_path, seq="4-4", initializer={"mode": "random"}))
    expected_classes = {
        "random": RandomInitializer,
        "dplm": DPLMInitializer,
        "aa_set": AminoAcidSetInitializer,
        "mask_mix": MaskMixInitializer,
        "all_a": AllAInitializer,
        "pssm": PSSMInitializer,
        "xmix": XMixInitializer,
    }
    for mode, expected_class in expected_classes.items():
        args = _args(tmp_path, seq="4-4", initializer={"mode": mode})
        assert isinstance(run_mod.build_initializer(args), expected_class)


def test_structure_initializer_inverse_folds_literal_mutable_sequence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdb"
    _write_pdb(
        source,
        chain="X",
        residue_ids=((10, ""), (11, "A"), (20, ""), (21, "")),
    )
    args = _args(
        tmp_path,
        seq="AAAA",
        initializer={
            "mode": "structure",
            "sequence_designer": "mpnn_init",
            "structure": {
                "source": str(source),
                "chain_map": {"X": "A"},
            },
            "rfd3": {"repo_path": "${missing.value}"},
        },
    )
    run_mod = _setup(args)
    designer = _Designer({"X": "GSTV"})

    initializer = run_mod.build_initializer(
        args,
        mutators={"mpnn_init": designer},
    )
    assert isinstance(initializer, ExistingStructureInitializer)
    result = initializer.apply()

    assert global_config.protomers["A"].sequence == "GSTV"
    assert result.sequences == {"A": "GSTV"}
    assert result.structure_path.read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sequence_designer"] == "mpnn_init"
    assert designer.calls[0]["redesigned_residues"] == "X10 X11A X20 X21"


def test_structure_initializer_preserves_fixed_segment_positions(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    _write_pdb(source, chain="A")
    args = _args(
        tmp_path,
        seq="+AA GG",
        initializer={
            "mode": "start_from_structure",
            "sequence_designer": "mpnn_init",
            "structure": {
                "source": str(source),
                "chain_map": {"A": "A"},
            },
        },
    )
    run_mod = _setup(args)
    result = run_mod.apply_initializer(
        args,
        mutators={"mpnn_init": _Designer({"A": "XXVV"})},
    )

    assert result.mode == "structure"
    assert global_config.protomers["A"].sequence == "AAVV"


def test_structure_initializer_ties_only_mutable_positions_across_chains(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdb"
    source.write_text(
        "\n".join(_pdb_lines("X") + _pdb_lines("Y")) + "\nEND\n",
        encoding="utf-8",
    )
    args = _args(
        tmp_path,
        seq="+AA GG",
        initializer={
            "mode": "structure",
            "sequence_designer": "mpnn_init",
            "structure": {
                "source": str(source),
                "chain_map": {"X": "A", "Y": "A"},
            },
        },
    )
    run_mod = _setup(args)
    designer = _Designer({"X": "XXVV", "Y": "AAVV"})

    result = run_mod.apply_initializer(
        args,
        mutators={"mpnn_init": designer},
    )

    assert result.sequences == {"A": "AAVV"}
    assert designer.calls[0]["symmetry_residues"] == "X3,Y3|X4,Y4"


def test_rfd3_initializer_delegates_generated_structure_to_sequence_designer(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        seq="4-4",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A"},
            },
        },
    )
    run_mod = _setup(args)
    designer = _Designer({"A": "VVVV"})
    runner_calls = []

    class _Runner:
        def __init__(self, config):
            runner_calls.append(("config", dict(config)))

        def run(self, *, specification, output_dir, n_batches):
            output_dir.mkdir(parents=True, exist_ok=True)
            structure = output_dir / "generated.pdb"
            _write_pdb(structure, chain="A")
            runner_calls.append(("run", specification, n_batches))
            return [
                RFD3StructureOutput(
                    example_id="gh_initializer_model_0",
                    structure_path=structure,
                    metadata_path=None,
                )
            ]

    initializer = RFD3Initializer(
        args=args,
        mutators={"mpnn_init": designer},
        segment_plan=run_mod._collect_mutable_protein_segments(),
        runner_cls=_Runner,
    )
    result = initializer.apply()

    assert runner_calls[1] == ("run", {"length": "4"}, 1)
    assert designer.calls[0]["structure_path"].name == "generated.pdb"
    assert global_config.protomers["A"].sequence == "VVVV"
    assert result.mode == "rfd3"


def test_rfd3_translates_fixed_seqs_motif_and_preserves_hotspots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motif_input.pdb"
    _write_pdb(
        source,
        chain="M",
        residue_ids=tuple((number, "") for number in range(10, 18)),
        sequence="QQCDEFQQ",
    )
    args = _args(
        tmp_path,
        seq="2-2 +CDEF 3-3",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A"},
                "specification": {
                    "input": str(source),
                    "length": "8-12",
                    "infer_ori_strategy": "hotspots",
                },
                "select_hotspots": {"M11": "CA"},
            },
        },
    )
    run_mod = _setup(args)
    runner_configs = []

    class _Runner:
        def __init__(self, config):
            runner_configs.append(dict(config))

    initializer = RFD3Initializer(
        args=args,
        mutators={"mpnn_init": _Designer({"A": "V" * 9})},
        segment_plan=run_mod._collect_mutable_protein_segments(),
        runner_cls=_Runner,
    )

    assert initializer.specification == {
        "length": "9",
        "contig": "2,M12-15,3",
        "select_hotspots": {"M11": "CA"},
        "input": str(source.resolve()),
        "infer_ori_strategy": "hotspots",
    }
    assert "select_hotspots" not in runner_configs[0]
    assert initializer.specification_details["motif_assignments"][0]["source"] == (
        "sequence_match"
    )


def test_rfd3_explicit_motif_map_supports_wildcard_fixed_segment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motif_input.pdb"
    _write_pdb(
        source,
        chain="M",
        residue_ids=tuple((number, "") for number in range(30, 37)),
        sequence="ACDEFGH",
    )
    args = _args(
        tmp_path,
        seq="2-2 +XXXX 1-1",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A"},
                "motif_map": {"A": {"2": "M31-34"}},
                "specification": {"input": str(source)},
            },
        },
    )
    run_mod = _setup(args)

    class _Runner:
        def __init__(self, _config):
            pass

    initializer = RFD3Initializer(
        args=args,
        mutators={"mpnn_init": _Designer({"A": "V" * 7})},
        segment_plan=run_mod._collect_mutable_protein_segments(),
        runner_cls=_Runner,
    )

    assert initializer.specification["contig"] == "2,M31-34,1"
    assignment = initializer.specification_details["motif_assignments"][0]
    assert assignment["sequence"] == "XXXX"
    assert assignment["input_sequence"] == "CDEF"
    assert assignment["source"] == "motif_map"


def test_rfd3_partial_diffusion_preserves_native_input_specification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "partial_input.pdb"
    _write_pdb(source, chain="A", sequence="ACDE")
    args = _args(
        tmp_path,
        seq="4-4",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A"},
                "specification": {
                    "input": str(source),
                    "partial_t": 5.0,
                    "select_hotspots": {"A2": "CA"},
                },
            },
        },
    )
    run_mod = _setup(args)

    class _Runner:
        def __init__(self, _config):
            pass

    initializer = RFD3Initializer(
        args=args,
        mutators={"mpnn_init": _Designer({"A": "VVVV"})},
        segment_plan=run_mod._collect_mutable_protein_segments(),
        runner_cls=_Runner,
    )

    assert initializer.specification == {
        "input": str(source.resolve()),
        "partial_t": 5.0,
        "select_hotspots": {"A2": "CA"},
    }
    assert "length" not in initializer.specification
    assert "contig" not in initializer.specification
    assert initializer.specification_details["source"] == "user_partial_diffusion"


def test_rfd3_rejects_unused_input_in_automatically_derived_specification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unused_input.pdb"
    _write_pdb(source, chain="M", sequence="ACDE")
    args = _args(
        tmp_path,
        seq="4-4",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A"},
                "specification": {"input": str(source)},
            },
        },
    )
    run_mod = _setup(args)

    class _Runner:
        def __init__(self, _config):
            pass

    with pytest.raises(ValueError, match="input is not represented"):
        RFD3Initializer(
            args=args,
            mutators={"mpnn_init": _Designer({"A": "VVVV"})},
            segment_plan=run_mod._collect_mutable_protein_segments(),
            runner_cls=_Runner,
        )


def test_structure_initializer_uses_fixed_motif_to_map_longer_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "longer_source.pdb"
    _write_pdb(
        source,
        chain="X",
        residue_ids=tuple((number, "") for number in range(10, 20)),
        sequence="QQCDEFAAAA",
    )
    args = _args(
        tmp_path,
        seq="+CDEF 3-3",
        initializer={
            "mode": "structure",
            "sequence_designer": "mpnn_init",
            "structure": {
                "source": str(source),
                "chain_map": {"X": "A"},
            },
        },
    )
    run_mod = _setup(args)
    designer = _Designer({"X": "V" * 10})

    result = run_mod.apply_initializer(
        args,
        mutators={"mpnn_init": designer},
    )

    assert result.sequences == {"A": "CDEFVVV"}
    assert designer.calls[0]["redesigned_residues"] == "X16 X17 X18"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["resolved_chain_map"][0]["structure_residues"] == [
        f"X{number}" for number in range(12, 19)
    ]


def test_structure_initializer_accepts_explicit_residue_to_position_mapping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "longer_source.pdb"
    _write_pdb(
        source,
        chain="X",
        residue_ids=tuple((number, "") for number in range(10, 20)),
        sequence="A" * 10,
    )
    args = _args(
        tmp_path,
        seq="5-5",
        initializer={
            "mode": "structure",
            "sequence_designer": "mpnn_init",
            "structure": {
                "source": str(source),
                "chain_map": {
                    "X": {
                        "protomer": "A",
                        "structure_residues": "X12-16",
                        "protomer_positions": "1-5",
                    }
                },
            },
        },
    )
    run_mod = _setup(args)
    designer = _Designer({"X": "AACDEFGAAA"})

    result = run_mod.apply_initializer(
        args,
        mutators={"mpnn_init": designer},
    )

    assert result.sequences == {"A": "CDEFG"}
    assert designer.calls[0]["redesigned_residues"] == "X12 X13 X14 X15 X16"


def test_rfd3_fixed_context_chain_is_not_sent_to_sequence_designer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "target.pdb"
    _write_pdb(
        source,
        chain="M",
        residue_ids=tuple((number, "") for number in range(10, 14)),
        sequence="CDEF",
    )
    args = _args(
        tmp_path,
        seq="4-4",
        initializer={
            "mode": "rfd3",
            "sequence_designer": "mpnn_init",
            "rfd3": {
                "repo_path": "/foundry",
                "python_bin": "/foundry/python",
                "ckpt_path": "/foundry/rfd3.ckpt",
                "chain_map": {"A": "A", "B": "B"},
                "specification": {
                    "input": str(source),
                    "select_hotspots": {"M11": "CA"},
                },
            },
        },
    )
    args.seqs = {"A": "4-4", "B": "+CDEF"}
    args.types = {"A": "p", "B": "p"}
    run_mod = _setup(args)
    designer = _Designer({"A": "VVVV"})
    runner_specs = []

    class _Runner:
        def __init__(self, _config):
            pass

        def run(self, *, specification, output_dir, n_batches):
            runner_specs.append(specification)
            output_dir.mkdir(parents=True, exist_ok=True)
            structure = output_dir / "generated.pdb"
            structure.write_text(
                "\n".join(
                    _pdb_lines("A", sequence="AAAA")
                    + _pdb_lines("B", sequence="CDEF")
                )
                + "\nEND\n",
                encoding="utf-8",
            )
            return [
                RFD3StructureOutput(
                    example_id="context_model_0",
                    structure_path=structure,
                    metadata_path=None,
                )
            ]

    initializer = RFD3Initializer(
        args=args,
        mutators={"mpnn_init": designer},
        segment_plan=run_mod._collect_mutable_protein_segments(),
        runner_cls=_Runner,
    )
    result = initializer.apply()

    assert runner_specs[0]["contig"] == "4,/0,M10-13"
    assert runner_specs[0]["select_hotspots"] == {"M11": "CA"}
    assert designer.calls[0]["chains_to_design"] == ["A"]
    assert result.sequences == {"A": "VVVV"}


def test_ligandmpnn_mutator_exposes_structure_initial_sequence_design() -> None:
    class _Runner:
        def design_sequences(self, **kwargs):
            assert kwargs["chains_to_design"] == "A,B"
            assert kwargs["fixed_residues"] == ""
            assert kwargs["redesigned_residues"] == "A1 A2 B1 B2"
            return {
                "S": np.asarray([[20, 1, 2, 3]]),
                "chain_letters": ("A", "A", "B", "B"),
            }

    mutator = LigandMPNNMutator(sequence_model_runner=_Runner())
    sequences = mutator.design_initial_sequences(
        structure_path=Path("input.pdb"),
        chains_to_design=["A", "B"],
        redesigned_residues="A1 A2 B1 B2",
    )

    assert sequences == {"A": "XC", "B": "DE"}


def test_rfd3_runner_uses_configured_foundry_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "foundry"
    (repo / "src").mkdir(parents=True)
    (repo / "models" / "rfd3" / "src").mkdir(parents=True)
    actual_python = repo / "python3.12"
    actual_python.write_text("#!/bin/sh\n", encoding="utf-8")
    actual_python.chmod(0o755)
    python_bin = repo / "python"
    python_bin.symlink_to(actual_python)
    checkpoint = repo / "rfd3.ckpt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    calls = []

    def _run(command, *, cwd, env, check):
        calls.append((command, cwd, env, check))
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        output_dir = Path(request["output_dir"])
        structure = output_dir / "sample.pdb"
        structure.write_text("ATOM\n", encoding="utf-8")
        Path(request["response_path"]).write_text(
            json.dumps(
                {
                    "outputs": [
                        {
                            "example_id": "sample",
                            "structure_path": str(structure),
                            "metadata_path": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "astrolabe.initializers.wrappers.rfd3.subprocess.run",
        _run,
    )
    runner = RFD3Runner(
        {
            "repo_path": str(repo),
            "python_bin": str(python_bin),
            "ckpt_path": str(checkpoint),
            "diffusion_batch_size": 1,
        }
    )
    outputs = runner.run(
        specification={"length": "20"},
        output_dir=tmp_path / "outputs",
        n_batches=1,
    )

    assert outputs[0].structure_path.name == "sample.pdb"
    assert calls[0][0][0] == str(python_bin.absolute())
    assert calls[0][1] == str(repo.resolve())
    assert calls[0][3] is True


def test_compose_validation_checks_structure_initializer_alias_and_chain_map() -> None:
    valid = {
        "initializer": {
            "mode": "structure",
            "sequence_designer": "InitMPNN",
            "structure": {"source": "source", "chain_map": {"A": "A"}},
        },
        "mutator": {
            "mutator_terms": {"InitMPNN": {"_type": "LigandMPNN"}}
        },
    }
    assert _compose_validate_basic(valid) == []

    invalid = {
        "initializer": {
            "mode": "rfd3",
            "sequence_designer": "Missing",
            "rfd3": {"chain_map": {}},
        },
        "mutator": {
            "mutator_terms": {"InitMPNN": {"_type": "LigandMPNN"}}
        },
    }
    errors = _compose_validate_basic(invalid)
    assert "initializer.sequence_designer references unknown mutator alias: Missing" in errors
    assert "initializer.rfd3.chain_map must be a non-empty dict" in errors
