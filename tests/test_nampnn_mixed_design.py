from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.na_mpnn import NAMPNNMutator
from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner
from astrolabe.utils.constant import aa_order

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RunnerStub:
    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)

    def design_sequences(self, **_kwargs):
        raise AssertionError("design_sequences should not be called in this test")


class _RecordingRunner:
    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)
        self.calls = []

    def design_sequences(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "sequence": "ACDE",
            "probabilities": _ppm_bundle(
                ["A1", "A2", "A3", "A4"],
                [
                    {"ALA": 1.0},
                    {"CYS": 1.0},
                    {"ASP": 1.0},
                    {"GLU": 1.0},
                ],
            ),
        }


def _init_runtime(tmp_path: Path, protomer_specs: dict[str, tuple[str, str]]) -> dict[str, Protomer]:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("named_files", {})
    global_config.init_const("predefined_statement", {})
    global_config.init_const("exclude_AA", [])
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw_model_results")
    global_config.init_const("enable_pyrosetta", False)
    global_config.init_const("skip_complex_poses", True)
    protomers: dict[str, Protomer] = {}
    for name, (ptype, seq) in protomer_specs.items():
        protomers[name] = Protomer(name, seq, ptype)
    global_config.init_const("protomers", protomers)
    global_config.init_const("mutable_protomers", protomers)
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("mutation_num", 0)
    global_config.set_variable("step", 0)
    return protomers


def _scores(tmp_path: Path, names: list[str], path_name: str = "input.pdb") -> StandardScores:
    complex_path = tmp_path / path_name
    complex_path.write_text("MODEL\nEND\n", encoding="utf-8")
    return StandardScores(
        names=names,
        plddt={name: [0.9] for name in names},
        complex_paths={name: complex_path for name in names},
        per_proto_scores={},
        ptm={name: [0.0] for name in names},
        iptm={name: [0.0] for name in names},
    )


def _queued_mutations(proto_name: str):
    protomer = global_config.protomers[proto_name]
    queued = []
    for segment_name in protomer.components:
        segment = global_config.segments[segment_name]
        for mutation in segment.mutations or []:
            queued.append((segment_name, mutation))
    return queued


def _ppm_bundle(encoded_residues: list[str], rows: list[dict[str, float]]) -> dict[str, object]:
    restype_order = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "DA",
        "DC",
        "DG",
        "DT",
        "A",
        "C",
        "G",
        "U",
    ]
    restype_to_int = {name: idx for idx, name in enumerate(restype_order)}
    predicted_ppm = np.zeros((len(rows), len(restype_order)), dtype=np.float32)
    for i, probs in enumerate(rows):
        for restype, value in probs.items():
            predicted_ppm[i, restype_to_int[restype]] = float(value)
    return {
        "predicted_ppm": predicted_ppm,
        "restype_to_int": restype_to_int,
        "encoded_residues": np.array(encoded_residues, dtype=object),
    }


def test_base_config_defaults_nampnn_to_mixed() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "base.yaml")
    assert cfg.mutator.NAMPNN.design_na_only == 0
    assert cfg.mutator.NAMPNN.parse_na_only == 0
    assert cfg.mutator.NAMPNN.na_shared_tokens == 0


def test_nampnn_runner_strips_cif_suffix_for_specificity_lookup() -> None:
    assert NAMPNNRunner._structure_basename(Path("step_000_AB_model.cif")) == "step_000_AB_model"
    assert NAMPNNRunner._structure_basename(Path("step_000_AB_model.pdb")) == "step_000_AB_model"
    assert NAMPNNRunner._structure_basename(Path("step_000_AB_model.mmcif")) == "step_000_AB_model"


def test_nampnn_runner_reuses_job_dir_and_reads_current_structure_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "na_mpnn"
    (repo / "inference").mkdir(parents=True)
    (repo / "models" / "design_model").mkdir(parents=True)
    (repo / "inference" / "run.py").write_text("# fake\n", encoding="utf-8")
    (repo / "models" / "design_model" / "s_19137.pt").write_text("fake\n", encoding="utf-8")
    structure_path = tmp_path / "step_017_AB_model.cif"
    structure_path.write_text("data_test\n", encoding="utf-8")

    output_root = tmp_path / "raw" / "NAMPNN"
    job_dir = output_root / "confirmed_target"
    (job_dir / "seqs").mkdir(parents=True)
    (job_dir / "specificity").mkdir()
    (job_dir / "seqs" / "step_016_AB_model.fa").write_text(
        ">old, overall_confidence=0.99\nOLD\n",
        encoding="utf-8",
    )
    (job_dir / "seqs" / "step_017_AB_model.fa").write_text(
        ">current, overall_confidence=0.42\nACDE\n",
        encoding="utf-8",
    )
    np.savez(
        job_dir / "specificity" / "step_017_AB_model.npz",
        predicted_ppm=np.zeros((1, 1), dtype=np.float32),
        restype_to_int={"ALA": 0},
        encoded_residues=np.array(["A1"], dtype=object),
    )

    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )
    runner = NAMPNNRunner({"repo_path": str(repo)}, output_root=output_root)

    result = runner.design_sequences(
        structure_path=structure_path,
        job_name="confirmed_target",
        chains_to_design="A",
    )

    assert result["sequence"] == "ACDE"
    assert result["fasta_path"] == job_dir / "seqs" / "step_017_AB_model.fa"
    assert calls


def test_compute_mutation_uses_external_nampnn_job_name_not_step(tmp_path: Path) -> None:
    _init_runtime(tmp_path, {"A": ("p", "ACDE")})
    global_config.set_variable("step", 17)
    runner = _RecordingRunner(na_shared_tokens=0, job_name="target_confirmed_job")
    mutator = NAMPNNMutator(sequence_model_runner=runner)

    mutator.compute_mutation(
        {
            "A": {
                "structure_path": tmp_path / "step_017_A_model.cif",
                "protos": ["A"],
                "chains_to_design": "A",
                "fixed_residues": [],
                "parse_these_chains_only": "A",
                "excluded": {"A": set()},
            }
        }
    )

    assert runner.calls[0]["job_name"] == "target_confirmed_job"


def test_compute_mutation_requires_external_nampnn_job_name(tmp_path: Path) -> None:
    _init_runtime(tmp_path, {"A": ("p", "ACDE")})
    runner = _RecordingRunner(na_shared_tokens=0)
    mutator = NAMPNNMutator(sequence_model_runner=runner)

    with pytest.raises(ValueError, match="mutator.NAMPNN.job_name"):
        mutator.compute_mutation(
            {
                "A": {
                    "structure_path": tmp_path / "step_017_A_model.cif",
                    "protos": ["A"],
                    "chains_to_design": "A",
                    "fixed_residues": [],
                    "parse_these_chains_only": "A",
                    "excluded": {"A": set()},
                }
            }
        )

    assert runner.calls == []


def test_prepare_mutation_input_supports_fixed_rna_designing_protein(tmp_path: Path) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "ACDE"),
            "B": ("r", "+AUGU"),
        },
    )
    scores = _scores(tmp_path, ["AB"])
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))

    result = mutator.prepare_mutation_input(global_config.protomers, scores, runtime_config={})

    job = result["AB"]
    assert job["protos"] == ["A", "B"]
    assert job["chains_to_design"] == "A"
    assert job["parse_these_chains_only"] == "AB"
    assert job["fixed_residues"] == ["B1", "B2", "B3", "B4"]


def test_prepare_mutation_input_rejects_shared_tokens_for_mixed_rna_and_dna(tmp_path: Path) -> None:
    _init_runtime(
        tmp_path,
        {
            "B": ("r", "AUG"),
            "C": ("d", "ATG"),
        },
    )
    scores = _scores(tmp_path, ["BC"])
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=1))

    with pytest.raises(ValueError, match="na_shared_tokens"):
        mutator.prepare_mutation_input(global_config.protomers, scores, runtime_config={})


def test_prepare_mutation_input_rejects_unsupported_partner_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "ACDE"),
            "L": ("c", "ATP"),
        },
    )
    scores = _scores(tmp_path, ["AL"])
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))
    monkeypatch.setattr("astrolabe.mutators.na_mpnn.prepare_exclude_sites", lambda *_args, **_kwargs: {})

    with pytest.raises(ValueError, match="only supports protein/RNA/DNA"):
        mutator.prepare_mutation_input(global_config.protomers, scores, runtime_config={})


def test_compute_mutation_decodes_mixed_sequences_and_probabilities(tmp_path: Path) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "ACD"),
            "B": ("r", "AUG"),
            "C": ("d", "ATG"),
        },
    )
    mutator = NAMPNNMutator(
        sequence_model_runner=SimpleNamespace(
            na_shared_tokens=0,
            job_name="mixed_design_job",
            design_sequences=lambda **_kwargs: {
                "sequence": "MKT/bdu/acg",
                "probabilities": _ppm_bundle(
                    ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
                    [
                        {"MET": 1.0},
                        {"LYS": 1.0},
                        {"THR": 1.0},
                        {"A": 1.0},
                        {"C": 1.0},
                        {"U": 1.0},
                        {"DA": 1.0},
                        {"DC": 1.0},
                        {"DG": 1.0},
                    ],
                ),
            },
        )
    )

    result = mutator.compute_mutation(
        {
            "ABC": {
                "structure_path": tmp_path / "abc.pdb",
                "protos": ["A", "B", "C"],
                "chains_to_design": "A,B,C",
                "fixed_residues": [],
                "parse_these_chains_only": "A,B,C",
                "excluded": {"A": set(), "B": set(), "C": set()},
            }
        }
    )

    proto_sequences = result["ABC"]["proto_sequences"]
    assert proto_sequences["A"]["sequence"] == "MKT"
    assert proto_sequences["B"]["sequence"] == "ACU"
    assert proto_sequences["C"]["sequence"] == "ACG"
    assert proto_sequences["A"]["sampling_probs"].shape == (1, 3, 20)
    assert proto_sequences["A"]["sampling_probs"][0, 0, aa_order.index("M")] == pytest.approx(1.0)
    assert proto_sequences["A"]["sampling_probs"][0, 1, aa_order.index("K")] == pytest.approx(1.0)
    assert proto_sequences["A"]["sampling_probs"][0, 2, aa_order.index("T")] == pytest.approx(1.0)
    assert proto_sequences["B"]["sampling_probs"].shape == (1, 3, 4)
    assert proto_sequences["C"]["sampling_probs"].shape == (1, 3, 4)


def test_compute_mutation_accepts_partial_output_for_fixed_rna_designing_protein(
    tmp_path: Path,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "ACDE"),
            "B": ("r", "+AUGU"),
        },
    )
    mutator = NAMPNNMutator(
        sequence_model_runner=SimpleNamespace(
            na_shared_tokens=0,
            job_name="protein_design_job",
            design_sequences=lambda **_kwargs: {
                "sequence": "VDKR",
                "probabilities": _ppm_bundle(
                    ["A1", "A2", "A3", "A4"],
                    [
                        {"VAL": 1.0},
                        {"ASP": 1.0},
                        {"LYS": 1.0},
                        {"ARG": 1.0},
                    ],
                ),
            },
        )
    )

    result = mutator.compute_mutation(
        {
            "AB": {
                "structure_path": tmp_path / "ab.pdb",
                "protos": ["A", "B"],
                "chains_to_design": "A",
                "fixed_residues": ["B1", "B2", "B3", "B4"],
                "parse_these_chains_only": "AB",
                "excluded": {"A": set(), "B": set(range(4))},
            }
        }
    )

    proto_sequences = result["AB"]["proto_sequences"]
    assert list(proto_sequences) == ["A"]
    assert proto_sequences["A"]["sequence"] == "VDKR"
    assert proto_sequences["A"]["sampling_probs"].shape == (1, 4, 20)


def test_compute_mutation_accepts_partial_output_for_fixed_protein_designing_rna(
    tmp_path: Path,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDE"),
            "B": ("r", "AUGU"),
        },
    )
    mutator = NAMPNNMutator(
        sequence_model_runner=SimpleNamespace(
            na_shared_tokens=0,
            job_name="rna_design_job",
            design_sequences=lambda **_kwargs: {
                "sequence": "bdhu",
                "probabilities": _ppm_bundle(
                    ["B1", "B2", "B3", "B4"],
                    [
                        {"A": 1.0},
                        {"C": 1.0},
                        {"G": 1.0},
                        {"U": 1.0},
                    ],
                ),
            },
        )
    )

    result = mutator.compute_mutation(
        {
            "AB": {
                "structure_path": tmp_path / "ab.pdb",
                "protos": ["A", "B"],
                "chains_to_design": "B",
                "fixed_residues": ["A1", "A2", "A3", "A4"],
                "parse_these_chains_only": "AB",
                "excluded": {"A": set(range(4)), "B": set()},
            }
        }
    )

    proto_sequences = result["AB"]["proto_sequences"]
    assert list(proto_sequences) == ["B"]
    assert proto_sequences["B"]["sequence"] == "ACGU"
    assert proto_sequences["B"]["sampling_probs"].shape == (1, 4, 4)


def test_compute_mutation_accepts_full_chain_output_for_fixed_protein_designing_rna(
    tmp_path: Path,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDE"),
            "B": ("r", "AUGU"),
        },
    )
    mutator = NAMPNNMutator(
        sequence_model_runner=SimpleNamespace(
            na_shared_tokens=0,
            job_name="rna_design_full_chain_job",
            design_sequences=lambda **_kwargs: {
                "sequence": "ACDE/bdhu",
                "probabilities": _ppm_bundle(
                    ["B1", "B2", "B3", "B4"],
                    [
                        {"A": 1.0},
                        {"C": 1.0},
                        {"G": 1.0},
                        {"U": 1.0},
                    ],
                ),
            },
        )
    )

    result = mutator.compute_mutation(
        {
            "AB": {
                "structure_path": tmp_path / "ab.pdb",
                "protos": ["A", "B"],
                "chains_to_design": "B",
                "fixed_residues": ["A1", "A2", "A3", "A4"],
                "parse_these_chains_only": "AB",
                "excluded": {"A": set(range(4)), "B": set()},
            }
        }
    )

    proto_sequences = result["AB"]["proto_sequences"]
    assert list(proto_sequences) == ["B"]
    assert proto_sequences["B"]["sequence"] == "ACGU"
    assert proto_sequences["B"]["sampling_probs"].shape == (1, 4, 4)


def test_prepare_mutation_output_uses_mutable_length_budget_for_fixed_protein_design_rna(
    tmp_path: Path,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDEFGHIK"),
            "B": ("r", "AUGC"),
        },
    )
    global_config.set_variable("mutation_num", 1)
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))

    mutation_output = {
        "AB": {
            "proto_sequences": {
                "A": {
                    "sequence": "ACDEFGHIK",
                    "excluded": set(range(9)),
                },
                "B": {
                    "sequence": "CUGC",
                    "excluded": set(),
                    "sampling_probs": torch.tensor(
                        [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                },
            }
        }
    }

    mutator.prepare_mutation_output(
        mutation_output,
        global_config.protomers,
        _scores(tmp_path, ["AB"]),
        runtime_config={},
    )

    queued = _queued_mutations("B")
    assert len(queued) == 1
    assert queued[0][1][2] != "A"


def test_prepare_mutation_output_backfills_sequence_when_probability_path_queues_nothing(
    tmp_path: Path,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDE"),
            "B": ("r", "AUGC"),
        },
    )
    global_config.set_variable("mutation_num", 2)
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))

    mutation_output = {
        "AB": {
            "proto_sequences": {
                "A": {
                    "sequence": "ACDE",
                    "excluded": set(range(4)),
                },
                "B": {
                    "sequence": "AUAU",
                    "excluded": set(),
                    "sampling_probs": torch.zeros((1, 0, 4), dtype=torch.float32),
                },
            }
        }
    }

    mutator.prepare_mutation_output(
        mutation_output,
        global_config.protomers,
        _scores(tmp_path, ["AB"]),
        runtime_config={},
    )

    queued = [mutation for _segment, mutation in _queued_mutations("B")]
    assert queued == [
        (2, "G", "A", 0.0, 0.0),
        (3, "C", "U", 0.0, 0.0),
    ]


def test_prepare_mutation_output_respects_mutation_weight(tmp_path: Path) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDE"),
            "B": ("r", "AUGC"),
        },
    )
    global_config.set_variable("mutation_num", 4)
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))
    mutator.mutation_weight = 0.25

    mutation_output = {
        "AB": {
            "proto_sequences": {
                "A": {
                    "sequence": "ACDE",
                    "excluded": set(range(4)),
                },
                "B": {
                    "sequence": "CAUA",
                    "excluded": set(),
                    "sampling_probs": torch.zeros((1, 0, 4), dtype=torch.float32),
                },
            }
        }
    }

    mutator.prepare_mutation_output(
        mutation_output,
        global_config.protomers,
        _scores(tmp_path, ["AB"]),
        runtime_config={},
    )

    queued = [mutation for _segment, mutation in _queued_mutations("B")]
    assert queued == [(0, "A", "C", 0.0, 0.0)]


def test_prepare_mutation_output_respects_max_mutations_per_proto(tmp_path: Path) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "+ACDE"),
            "B": ("r", "AUGC"),
        },
    )
    global_config.set_variable("mutation_num", 4)
    mutator = NAMPNNMutator(
        sequence_model_runner=_RunnerStub(na_shared_tokens=0),
        max_mutations_per_proto={"B": 1},
    )

    mutation_output = {
        "AB": {
            "proto_sequences": {
                "A": {
                    "sequence": "ACDE",
                    "excluded": set(range(4)),
                },
                "B": {
                    "sequence": "CAUA",
                    "excluded": set(),
                    "sampling_probs": torch.zeros((1, 0, 4), dtype=torch.float32),
                },
            }
        }
    }

    mutator.prepare_mutation_output(
        mutation_output,
        global_config.protomers,
        _scores(tmp_path, ["AB"]),
        runtime_config={},
    )

    queued = [mutation for _segment, mutation in _queued_mutations("B")]
    assert queued == [(0, "A", "C", 0.0, 0.0)]


def test_prepare_mutation_input_surfaces_mutable_mask_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runtime(
        tmp_path,
        {
            "A": ("p", "ACDE"),
        },
    )
    scores = _scores(tmp_path, ["A"])
    mutator = NAMPNNMutator(sequence_model_runner=_RunnerStub(na_shared_tokens=0))

    def _raise_mask(self):
        raise RuntimeError("mask boom")

    monkeypatch.setattr(Protomer, "mutable_mask", property(_raise_mask))

    with pytest.raises(RuntimeError, match="mask boom"):
        mutator.prepare_mutation_input(global_config.protomers, scores, runtime_config={})
