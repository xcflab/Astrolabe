from __future__ import annotations

import random
import types
import logging
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import build_mutators


def _build_args(
    tmp_path: Path,
    *,
    seq: str = "5-5",
    initializer: dict | None = None,
    legacy_dplm: bool = False,
    cyclization: dict | None = None,
):
    init_cfg = OmegaConf.create(initializer) if initializer is not None else None
    return types.SimpleNamespace(
        exclude_AA=["C"],
        initial_temperature=0.05,
        out_dir=tmp_path,
        raw_model_results_dir=tmp_path / "raw",
        use_metropolis_acceptance=True,
        seqs={"A": seq},
        types={"A": "p"},
        homologies={},
        cyclization=OmegaConf.create(cyclization) if cyclization is not None else None,
        initializer=init_cfg,
        reinitialize_random_segments_with_dplm=legacy_dplm,
        mutator=OmegaConf.create(
            {
                "DPLM": {"num_samples": 1},
                "mutator_terms": {},
            }
        ),
    )


def _setup_single_random_protomer(args) -> None:
    import run as run_mod

    global_config.reset()
    run_mod.initialize_global_config(args)
    run_mod.setup_protomers(args)


def _tokenize_masked_sequence(masked_sequence: str) -> list[str]:
    tokens: list[str] = []
    cursor = 0
    while cursor < len(masked_sequence):
        if masked_sequence.startswith("<mask>", cursor):
            tokens.append("<mask>")
            cursor += len("<mask>")
            continue
        tokens.append(masked_sequence[cursor])
        cursor += 1
    return tokens


def test_apply_initializer_maps_legacy_dplm_flag_to_dplm(tmp_path: Path, monkeypatch) -> None:
    import run as run_mod

    args = _build_args(tmp_path, initializer=None, legacy_dplm=True)
    called = {"dplm": 0}
    monkeypatch.setattr(
        run_mod,
        "_apply_dplm_initializer",
        lambda _args: called.__setitem__("dplm", called["dplm"] + 1),
    )

    run_mod.apply_initializer(args)
    assert called["dplm"] == 1


def test_apply_initializer_rejects_conflicting_legacy_and_mode(tmp_path: Path) -> None:
    import run as run_mod

    args = _build_args(
        tmp_path,
        initializer={"mode": "aa_set", "aa_set": "AG"},
        legacy_dplm=True,
    )
    with pytest.raises(ValueError, match="Conflicting initializer settings"):
        run_mod.apply_initializer(args)


def test_aa_set_initializer_replaces_random_segments(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(7)
    np.random.seed(7)
    args = _build_args(
        tmp_path,
        seq="6-6",
        initializer={
            "mode": "aa_set",
            "aa_set": "AG",
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    segment = global_config.segments["A_1"]
    assert set(segment.sequence).issubset({"A", "G"})
    assert len(segment.sequence) == 6
    assert segment.initialized_from_random is False


def test_aa_set_initializer_can_target_specific_segment_aliases(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(11)
    np.random.seed(11)
    args = _build_args(
        tmp_path,
        seq="6-6@core_1 4-4@loop_1 core_1 loop_1",
        initializer={
            "mode": "aa_set",
            "aa_set": {
                "segments": {
                    "loop_1": "GSTNQPD",
                }
            },
        },
    )
    _setup_single_random_protomer(args)

    before_core = global_config.segments["core_1"].sequence
    before_loop = global_config.segments["loop_1"].sequence

    run_mod.apply_initializer(args)

    after_core = global_config.segments["core_1"].sequence
    after_loop = global_config.segments["loop_1"].sequence
    assert after_core == before_core
    assert after_loop != before_loop
    assert set(after_loop).issubset(set("GSTNQPD"))
    assert global_config.segments["core_1"].initialized_from_random is True
    assert global_config.segments["loop_1"].initialized_from_random is False


def test_aa_set_initializer_skips_unmatched_segments_without_logging_crash(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import run as run_mod

    random.seed(17)
    np.random.seed(17)
    args = _build_args(
        tmp_path,
        seq="4-4@loop_1 6-6@core_1 loop_1 core_1",
        initializer={
            "mode": "aa_set",
            "aa_set": {
                "segments": {
                    "loop_1": "GSTNQPD",
                }
            },
        },
    )
    _setup_single_random_protomer(args)

    before_core = global_config.segments["core_1"].sequence
    with caplog.at_level(logging.INFO):
        run_mod.apply_initializer(args)

    assert global_config.segments["core_1"].sequence == before_core
    assert global_config.segments["core_1"].initialized_from_random is True
    assert global_config.segments["loop_1"].initialized_from_random is False
    assert "Reinitialized 1 protein segment(s) with aa_set initializer" in caplog.text


def test_pssm_initializer_uses_per_position_weights(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(13)
    np.random.seed(13)
    args = _build_args(
        tmp_path,
        seq="5-5",
        initializer={
            "mode": "pssm",
            "pssm": {
                "A": [
                    {"A": 1.0},
                    {"G": 1.0},
                    {"S": 1.0},
                    {"T": 1.0},
                    {"V": 1.0},
                ]
            },
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    assert global_config.protomers["A"].sequence == "AGSTV"
    assert global_config.segments["A_1"].initialized_from_random is False


def test_pssm_initializer_requires_entries_for_random_protomers(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(23)
    np.random.seed(23)
    args = _build_args(
        tmp_path,
        seq="5-5",
        initializer={
            "mode": "pssm",
            "pssm": {
                "B": [{"A": 1.0}] * 5,
            },
        },
    )
    _setup_single_random_protomer(args)

    with pytest.raises(ValueError, match="missing protomer entries"):
        run_mod.apply_initializer(args)


def test_x_mix_initializer_can_fill_all_positions_with_x(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(31)
    np.random.seed(31)
    args = _build_args(
        tmp_path,
        seq="6-6",
        initializer={
            "mode": "x_mix",
            "x_fraction": 1.0,
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    segment = global_config.segments["A_1"]
    assert segment.sequence == "XXXXXX"
    assert segment.initialized_from_random is False


def test_x_mix_initializer_can_use_custom_aa_set_for_non_x_positions(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(37)
    np.random.seed(37)
    args = _build_args(
        tmp_path,
        seq="8-8",
        initializer={
            "mode": "x_mix",
            "x_fraction": 0.0,
            "aa_set": "AG",
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    segment = global_config.segments["A_1"]
    assert set(segment.sequence).issubset({"A", "G"})
    assert "X" not in segment.sequence
    assert segment.initialized_from_random is False


def test_x_mix_initializer_rejects_invalid_fraction(tmp_path: Path) -> None:
    import run as run_mod

    args = _build_args(
        tmp_path,
        seq="5-5",
        initializer={
            "mode": "x_mix",
            "x_fraction": 1.5,
        },
    )
    _setup_single_random_protomer(args)

    with pytest.raises(ValueError, match="initializer.x_fraction"):
        run_mod.apply_initializer(args)


def test_x_mix_mutator_can_resample_full_sequence_via_stage_mutator(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(43)
    np.random.seed(43)
    args = _build_args(tmp_path, seq="6-6")
    _setup_single_random_protomer(args)
    global_config.set_variable("mutation_num", 6)

    mutators = build_mutators(
        OmegaConf.create(
            {
                "mutator_terms": {
                    "x_mask": {
                        "_type": "XMix",
                        "x_fraction": 1.0,
                    }
                }
            }
        ),
        tmp_path / "raw",
    )

    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": Path("a.pdb")},
    )
    mutators["x_mask"].mutate(global_config.mutable_protomers, eval_result, {})

    assert global_config.protomers["A"].mutated_sequence == "XXXXXX"


def test_x_mix_mutator_preserves_locked_positions(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(47)
    np.random.seed(47)
    args = _build_args(
        tmp_path,
        seq="9-9",
        cyclization={
            "signal_mode": "both",
            "encoding": {"scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": "nterm+3", "atom": "NZ", "aa": "K"},
                    "b": {"proto": "A", "pos": "cterm-3", "atom": "CD", "aa": "E"},
                    "order": "single",
                    "target_distance": 1.33,
                }
            ],
        },
    )
    _setup_single_random_protomer(args)
    run_mod.apply_cyclization(args)
    global_config.set_variable("mutation_num", 9)

    mutators = build_mutators(
        OmegaConf.create(
            {
                "mutator_terms": {
                    "x_mask": {
                        "_type": "XMix",
                        "x_fraction": 1.0,
                    }
                }
            }
        ),
        tmp_path / "raw",
    )

    eval_result = StandardScores(
        names=["A"],
        plddt={"A": [1.0]},
        complex_paths={"A": Path("a.pdb")},
    )
    mutators["x_mask"].mutate(global_config.mutable_protomers, eval_result, {})

    mutated = global_config.protomers["A"].mutated_sequence
    assert mutated[3] == "K"
    assert mutated[5] == "E"
    assert mutated.count("X") == 7


def test_apply_cyclization_enforces_endpoint_identity_and_locking(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(41)
    np.random.seed(41)
    args = _build_args(
        tmp_path,
        seq="9-9",
        cyclization={
            "signal_mode": "both",
            "encoding": {"scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": "nterm+3", "atom": "NZ", "aa": "K"},
                    "b": {"proto": "A", "pos": "cterm-3", "atom": "CD", "aa": "E"},
                    "order": "single",
                    "target_distance": 1.33,
                }
            ],
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_cyclization(args)

    protomer = global_config.protomers["A"]
    assert protomer.sequence[3] == "K"
    assert protomer.sequence[5] == "E"
    assert protomer.locked_positions == frozenset({3, 5})
    assert bool(protomer.mutable_mask[3]) is False
    assert bool(protomer.mutable_mask[5]) is False


def test_dplm_initializer_does_not_mask_locked_cyclization_endpoints(
    tmp_path: Path, monkeypatch
) -> None:
    import run as run_mod

    random.seed(43)
    np.random.seed(43)
    args = _build_args(
        tmp_path,
        seq="9-9",
        cyclization={
            "signal_mode": "both",
            "encoding": {"scope": "all_bonds"},
            "bonds": [
                {
                    "a": {"proto": "A", "pos": "nterm+3", "atom": "NZ", "aa": "K"},
                    "b": {"proto": "A", "pos": "cterm-3", "atom": "CD", "aa": "E"},
                    "order": "single",
                    "target_distance": 1.33,
                }
            ],
        },
    )
    _setup_single_random_protomer(args)
    run_mod.apply_cyclization(args)

    captured: dict[str, object] = {}

    class _FakeDPLMRunner:
        def __init__(self, _runner_conf):
            pass

        def design_sequences(self, tasks):
            captured["tasks"] = list(tasks)
            return {"A": [global_config.protomers["A"].sequence]}

    monkeypatch.setattr(run_mod, "DPLMRunner", _FakeDPLMRunner)

    run_mod._apply_dplm_initializer(args)

    tasks = captured["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    masked_tokens = _tokenize_masked_sequence(str(tasks[0]["masked_sequence"]))
    assert len(masked_tokens) == 9
    assert masked_tokens[3] == "K"
    assert masked_tokens[5] == "E"
    assert masked_tokens.count("<mask>") == 7

def test_mask_mix_initializer_applies_exact_x_quota(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(29)
    np.random.seed(29)
    args = _build_args(
        tmp_path,
        seq="10-10",
        initializer={
            "mode": "mask_mix",
            "mask_fraction": 0.8,
            "aa_set": "AG",
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    seq = global_config.protomers["A"].sequence
    assert seq.count("X") == 8
    assert set(seq.replace("X", "")).issubset({"A", "G"})
    assert global_config.segments["A_1"].initialized_from_random is False


def test_mask_mix_initializer_uses_protomer_wide_x_quota_across_segments(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(30)
    np.random.seed(30)
    args = _build_args(
        tmp_path,
        seq="1-1 1-1",
        initializer={
            "mode": "mask_mix",
            "mask_fraction": 0.5,
            "aa_set": "AG",
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    seq = global_config.protomers["A"].sequence
    assert seq.count("X") == 1
    assert set(seq.replace("X", "")).issubset({"A", "G"})
    assert global_config.segments["A_1"].initialized_from_random is False
    assert global_config.segments["A_2"].initialized_from_random is False


def test_mask_mix_initializer_rejects_invalid_fraction(tmp_path: Path) -> None:
    import run as run_mod

    args = _build_args(
        tmp_path,
        seq="6-6",
        initializer={
            "mode": "mask_mix",
            "mask_fraction": 1.2,
        },
    )
    _setup_single_random_protomer(args)

    with pytest.raises(ValueError, match="mask_fraction"):
        run_mod.apply_initializer(args)


def test_mask_mix_initializer_preserves_fixed_and_locked_positions(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(31)
    np.random.seed(31)
    args = _build_args(
        tmp_path,
        seq="+4-4 6-6",
        initializer={
            "mode": "mask_mix",
            "mask_fraction": 1.0,
            "aa_set": "AG",
        },
    )
    _setup_single_random_protomer(args)
    fixed_segment_before = global_config.segments["A_1"].sequence
    locked_index = 4
    locked_residue_before = global_config.protomers["A"].sequence[locked_index]
    global_config.protomers["A"].add_locked_positions([locked_index])

    run_mod.apply_initializer(args)

    assert global_config.segments["A_1"].sequence == fixed_segment_before
    assert global_config.protomers["A"].sequence[locked_index] == locked_residue_before
    assert global_config.segments["A_2"].sequence[0] == locked_residue_before
    assert global_config.segments["A_2"].sequence[1:] == "XXXXX"
    assert global_config.segments["A_2"].initialized_from_random is False


def test_mask_mix_initializer_default_residue_set_respects_exclude_aa(tmp_path: Path) -> None:
    import run as run_mod

    random.seed(37)
    np.random.seed(37)
    args = _build_args(
        tmp_path,
        seq="12-12",
        initializer={
            "mode": "mask_mix",
            "mask_fraction": 0.0,
        },
    )
    _setup_single_random_protomer(args)

    run_mod.apply_initializer(args)

    seq = global_config.protomers["A"].sequence
    assert "X" not in seq
    assert "C" not in seq
