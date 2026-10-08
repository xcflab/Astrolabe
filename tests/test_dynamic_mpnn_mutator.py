from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.ligandmpnn import GroupedLigandMPNNMutator


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def design_sequences(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("pdb_path_multi"):
            paths = json.loads(Path(kwargs["pdb_path_multi"]).read_text(encoding="utf-8"))
            fixed = json.loads(Path(kwargs["fixed_residues_multi"]).read_text(encoding="utf-8"))
            buried = json.loads(
                Path(kwargs["transmembrane_buried_multi"]).read_text(encoding="utf-8")
            )
            interface = json.loads(
                Path(kwargs["transmembrane_interface_multi"]).read_text(encoding="utf-8")
            )
            assert paths == ["AB.pdb", "AC.pdb"]
            assert fixed == {"AB.pdb": ["B1"], "AC.pdb": ["B1"]}
            assert buried == {"AB.pdb": "A1", "AC.pdb": "A2"}
            assert interface == {"AB.pdb": "A3", "AC.pdb": "A4"}
            return {
                "multi_outputs": {
                    "AB.pdb": {
                        "sampling_probs": np.asarray([[[0.8, 0.2]]], dtype=float)
                    },
                    "AC.pdb": {
                        "sampling_probs": np.asarray([[[0.6, 0.4]]], dtype=float)
                    },
                }
            }
        return {"sampling_probs": np.asarray([[[0.3, 0.7]]], dtype=float)}


def _init_global(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw")
    global_config.init_const("segments", {})
    global_config.init_const(
        "protomers",
        {
            name: SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                mutable_length=1,
                length=1,
            )
            for name in "ABCDE"
        },
    )
    global_config.set_variable("step", 7)


def test_dynamic_mpnn_groups_oligomers_that_share_design_protomers(tmp_path: Path) -> None:
    _init_global(tmp_path)
    runner = _RecordingRunner()
    mutator = GroupedLigandMPNNMutator(sequence_model_runner=runner)

    result = mutator.compute_mutation(
        {
            "AB": {
                "pdb_path": "AB.pdb",
                "fixed_residues": "B1",
                "redesigned_residues": "",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "parse_these_chains_only": "AB",
                "transmembrane_buried": "A1",
                "transmembrane_interface": "A3",
            },
            "AC": {
                "pdb_path": "AC.pdb",
                "fixed_residues": "B1",
                "redesigned_residues": "",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "parse_these_chains_only": "AB",
                "transmembrane_buried": "A2",
                "transmembrane_interface": "A4",
            },
            "DE": {
                "pdb_path": "DE.pdb",
                "fixed_residues": "",
                "redesigned_residues": "",
                "symmetry_residues": "",
                "symmetry_weights": "",
                "parse_these_chains_only": "AB",
            },
        }
    )

    assert len(runner.calls) == 2
    assert runner.calls[0]["pdb_path_multi"]
    assert runner.calls[1]["pdb_path"] == "DE.pdb"
    assert result["AB"]["sampling_probs"][0, 0, 0] == 0.8
    assert result["AC"]["sampling_probs"][0, 0, 0] == 0.6
    assert result["DE"]["sampling_probs"][0, 0, 0] == 0.3
    assert result["AB"]["name"] == "AB"
    assert result["AC"]["feature_dict"]["pdb_path"] == "AC.pdb"


def test_dynamic_mpnn_groups_protomers_that_share_alias_segment(tmp_path: Path) -> None:
    _init_global(tmp_path)
    shared_segment = SimpleNamespace(is_fixed=False)
    global_config.segments["A_1"] = shared_segment
    global_config.segments["B_1"] = shared_segment
    global_config.protomers["A"] = SimpleNamespace(
        type=ProtomerTypes.PROTEIN,
        mutable_length=1,
        length=1,
        components=["A_1"],
    )
    global_config.protomers["B"] = SimpleNamespace(
        type=ProtomerTypes.PROTEIN,
        mutable_length=1,
        length=1,
        components=["B_1"],
    )

    class _AliasRunner:
        def __init__(self) -> None:
            self.calls = []

        def design_sequences(self, **kwargs):
            self.calls.append(dict(kwargs))
            paths = json.loads(Path(kwargs["pdb_path_multi"]).read_text(encoding="utf-8"))
            assert paths == ["A.pdb", "B.pdb"]
            return {
                "multi_outputs": {
                    "A.pdb": {"sampling_probs": np.asarray([[[0.9, 0.1]]])},
                    "B.pdb": {"sampling_probs": np.asarray([[[0.4, 0.6]]])},
                }
            }

    runner = _AliasRunner()
    mutator = GroupedLigandMPNNMutator(sequence_model_runner=runner)
    result = mutator.compute_mutation(
        {
            "A": {"pdb_path": "A.pdb", "fixed_residues": ""},
            "B": {"pdb_path": "B.pdb", "fixed_residues": ""},
        }
    )

    assert len(runner.calls) == 1
    assert runner.calls[0]["pdb_path_multi"]
    assert result["A"]["sampling_probs"][0, 0, 0] == 0.9
    assert result["B"]["sampling_probs"][0, 0, 0] == 0.4


def test_dynamic_mpnn_requires_multi_output_for_shared_group(tmp_path: Path) -> None:
    _init_global(tmp_path)

    class _BadRunner:
        def design_sequences(self, **_kwargs):
            return {"sampling_probs": np.asarray([[[1.0, 0.0]]], dtype=float)}

    mutator = GroupedLigandMPNNMutator(sequence_model_runner=_BadRunner())

    try:
        mutator.compute_mutation(
            {
                "AB": {
                    "pdb_path": "AB.pdb",
                    "fixed_residues": "",
                    "parse_these_chains_only": "AB",
                },
                "AC": {
                    "pdb_path": "AC.pdb",
                    "fixed_residues": "",
                    "parse_these_chains_only": "AB",
                },
            }
        )
    except RuntimeError as exc:
        assert "multi_outputs" in str(exc)
    else:
        raise AssertionError("Expected grouped DynamicMPNN call to require multi_outputs")
