from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators import utils as mutator_utils

AA_ORDER = [mutator_utils.restype_int_to_str[i] for i in range(20)]


class _Protomer(SimpleNamespace):
    def propagating_probs(self, probs):
        self.last_probs = probs

    def propagating_scores(self, scores):
        self.last_scores = scores


class _SegmentManager:
    def collect_mutations(self):
        return []

    def add_mutations(self, mutations, n_mutations):
        self.added = (mutations, n_mutations)


def _init_runtime(tmp_path):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers",
        {
            "A": _Protomer(
                type=ProtomerTypes.PROTEIN,
                length=2,
                sequence="AC",
                components=["segA"],
            ),
            "B": _Protomer(
                type=ProtomerTypes.PROTEIN,
                length=1,
                sequence="G",
                components=["segB"],
            ),
            "C": _Protomer(
                type=ProtomerTypes.PROTEIN,
                length=1,
                sequence="S",
                components=["segC"],
            ),
        },
    )
    global_config.init_const("segment_manager", _SegmentManager())
    global_config.set_variable("mutation_num", 0)
    global_config.set_variable("step", 7)
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("negative_design_complexes", [])


def _protein_probs(rows):
    arr = np.full((1, len(rows), 20), 1e-6, dtype=float)
    for i, aa_probs in enumerate(rows):
        for aa, value in aa_probs.items():
            arr[0, i, mutator_utils.restype_str_to_int[aa]] = value
    arr /= arr.sum(axis=-1, keepdims=True)
    return arr


def test_state_conflict_diagnostics_disabled_does_not_write(tmp_path):
    _init_runtime(tmp_path)
    eval_scores = SimpleNamespace(names=["AB", "AC"])
    mutation_output = {
        "AB": {"sampling_probs": _protein_probs([{"A": 0.9}, {"C": 0.8}, {"G": 0.7}])},
        "AC": {"sampling_probs": _protein_probs([{"V": 0.9}, {"D": 0.8}, {"S": 0.7}])},
    }

    mutator_utils.sample_mutations_by_aa_probs(
        eval_scores,
        mutation_output,
        AA_ORDER,
        mutation_weight=1.0,
        runtime_config={"state_conflict_diagnostics": {"enabled": False}},
    )

    assert not (tmp_path / "mutator_state_conflicts").exists()


def test_mutator_runtime_config_carries_current_design_complex_markers(tmp_path):
    _init_runtime(tmp_path)
    global_config.init_const(
        "stage_manager", SimpleNamespace(current_oligo=["AB", "AC", "BC"])
    )
    global_config.set_variable("ignored_complexes", ["BC"])
    global_config.set_variable("negative_design_complexes", ["AC"])

    runtime_config = mutator_utils.build_mutator_runtime_config(
        step=1,
        former_config={"state_conflict_diagnostics": {"enabled": False}},
    )

    assert runtime_config["oligo"] == ["AB", "AC", "BC"]
    assert runtime_config["ignored_complexes"] == ["BC"]
    assert runtime_config["negative_design_complexes"] == ["AC"]
    assert runtime_config["state_conflict_diagnostics"] == {"enabled": False}


def test_state_conflict_diagnostics_requires_sampling_probs_when_enabled(tmp_path):
    _init_runtime(tmp_path)
    eval_scores = SimpleNamespace(names=["AB"])

    with pytest.raises(ValueError, match="State conflict diagnostics require"):
        mutator_utils.sample_mutations_by_aa_probs(
            eval_scores,
            {"AB": {"sequence": "ACG"}},
            AA_ORDER,
            mutation_weight=1.0,
            runtime_config={"state_conflict_diagnostics": {"enabled": True}},
        )


def test_state_conflict_diagnostics_writes_pairwise_proto_report(tmp_path):
    _init_runtime(tmp_path)
    eval_scores = SimpleNamespace(names=["AB", "AC"])
    mutation_output = {
        "AB": {"sampling_probs": _protein_probs([{"A": 0.9}, {"C": 0.8}, {"G": 0.7}])},
        "AC": {"sampling_probs": _protein_probs([{"V": 0.9}, {"D": 0.8}, {"S": 0.7}])},
    }
    output_dir = tmp_path / "diagnostics"

    mutator_utils.sample_mutations_by_aa_probs(
        eval_scores,
        mutation_output,
        AA_ORDER,
        mutation_weight=1.0,
        runtime_config={
            "step": 7,
            "current_mutator_alias": "LigandMPNN_itf",
            "state_conflict_diagnostics": {
                "enabled": True,
                "output_dir": str(output_dir),
            },
        },
    )

    output_path = output_dir / "step_007_LigandMPNN_itf.csv"
    assert output_path.exists()
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert len(rows) == 2
    first = rows[0]
    assert first["protomer"] == "A"
    assert first["position"] == "1"
    assert first["current_aa"] == "A"
    assert first["oligo_a"] == "AB"
    assert first["oligo_b"] == "AC"
    assert first["top_aa_a"] == "A"
    assert first["top_aa_b"] == "V"
    assert first["top_agree"] == "0"
    assert float(first["js_divergence"]) > 0.0
    assert float(first["current_prob_a"]) > float(first["current_prob_b"])

    propagated = global_config.protomers["A"].last_probs
    expected = 0.5 * (
        mutation_output["AB"]["sampling_probs"][0, :2, :]
        + mutation_output["AC"]["sampling_probs"][0, :2, :]
    )
    expected = mutator_utils.sort_probabilities_by_aaorder(
        expected,
        AA_ORDER,
    )
    assert np.allclose(propagated, expected)


def test_state_conflict_diagnostics_reads_proto_sequence_outputs(tmp_path):
    _init_runtime(tmp_path)
    eval_scores = SimpleNamespace(names=["AB", "AC"])
    mutation_output = {
        "AB": {
            "proto_sequences": {
                "A": {"sampling_probs": _protein_probs([{"A": 0.9}, {"C": 0.8}])},
            }
        },
        "AC": {
            "proto_sequences": {
                "A": {"sampling_probs": _protein_probs([{"V": 0.9}, {"D": 0.8}])},
            }
        },
    }
    output_dir = tmp_path / "proto_diagnostics"

    mutator_utils.sample_mutations_by_aa_probs(
        eval_scores,
        mutation_output,
        AA_ORDER,
        mutation_weight=1.0,
        runtime_config={
            "step": 3,
            "current_mutator_alias": "NAMPNN",
            "state_conflict_diagnostics": {
                "enabled": True,
                "output_dir": str(output_dir),
            },
        },
    )

    rows = list(csv.DictReader((output_dir / "step_003_NAMPNN.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["protomer"] == "A"
    assert rows[0]["top_aa_a"] == "A"
    assert rows[0]["top_aa_b"] == "V"


def test_state_conflict_diagnostics_averages_repeated_protomer_occurrences(tmp_path):
    _init_runtime(tmp_path)
    eval_scores = SimpleNamespace(names=["AA", "AB"])
    mutation_output = {
        "AA": {
            "sampling_probs": _protein_probs(
                [{"A": 0.9}, {"C": 0.8}, {"D": 0.9}, {"E": 0.8}]
            )
        },
        "AB": {"sampling_probs": _protein_probs([{"V": 0.9}, {"D": 0.8}, {"G": 0.7}])},
    }
    output_dir = tmp_path / "repeat_diagnostics"

    mutator_utils.sample_mutations_by_aa_probs(
        eval_scores,
        mutation_output,
        AA_ORDER,
        mutation_weight=1.0,
        runtime_config={
            "step": 4,
            "current_mutator_alias": "LigandMPNN",
            "state_conflict_diagnostics": {
                "enabled": True,
                "output_dir": str(output_dir),
            },
        },
    )

    rows = list(csv.DictReader((output_dir / "step_004_LigandMPNN.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["oligo_a"] == "AA"
    assert rows[0]["oligo_b"] == "AB"
    assert rows[0]["top_aa_a"] in {"A", "D"}
