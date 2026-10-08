from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.random_substitution import RandomSubstitutionMutator, PoreBlockingSubstitutionMutator
from astrolabe.mutators.utils import aggregate_per_proto_scores
from astrolabe.utils.constant import restype_str_to_int


class _RecordingProtomer:
    def __init__(self, sequence: str = "AA") -> None:
        self.sequence = sequence
        self.length = len(sequence)
        self.type = ProtomerTypes.PROTEIN
        self.mutable_mask = np.ones(self.length, dtype=bool)
        self.calls: list[np.ndarray] = []
        self.prob_calls: list[np.ndarray] = []

    def propagating_scores(self, scores) -> None:
        self.calls.append(np.asarray(scores, dtype=float))

    def propagating_probs(self, probs) -> None:
        self.prob_calls.append(np.asarray(probs, dtype=float))

    @staticmethod
    def aggregate_bool(mask):
        return mask


class _SegmentManager:
    def __init__(self) -> None:
        self.added: tuple[list, int] | None = None
        self.mutations: list = []

    def collect_mutations(self) -> list:
        return list(self.mutations)

    def add_mutations(self, mutations, n_mutations) -> None:
        self.added = (list(mutations), int(n_mutations))


def _init_global(name: str, value) -> None:
    setattr(global_config, f"_{name}", value)
    global_config._initialized_constants[name] = True


def test_mcmc_maps_homo_oligomer_chain_scores_back_to_single_protomer() -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer()
        segment_manager = _SegmentManager()
        _init_global("protomers", {"A": protomer_a})
        _init_global("segment_manager", segment_manager)
        global_config.set_variable("mutation_num", 1)

        mutator = RandomSubstitutionMutator(mutation_method="random", select_positions="random")
        mutator.mutation_weight = 1.0
        scores = StandardScores(
            names=["AAAA"],
            plddt={"AAAA": [0.9]},
            complex_paths={"AAAA": SimpleNamespace()},
            per_proto_scores={
                "A": np.array([0.1, 0.2], dtype=float),
                "B": np.array([0.3, 0.4], dtype=float),
                "C": np.array([0.5, 0.6], dtype=float),
                "D": np.array([0.7, 0.8], dtype=float),
            },
        )

        mutator.prepare_mutation_output(
            mutation_output={},
            protomer=None,
            evaluation_scores=scores,
            runtime_config={},
        )

        assert len(protomer_a.calls) == 1
        assert np.allclose(
            protomer_a.calls[0],
            np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
                dtype=float,
            ),
        )
        assert segment_manager.added is not None
        assert segment_manager.added[0] == []
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_requires_position_selector() -> None:
    try:
        try:
            PoreBlockingSubstitutionMutator(position_selector=None)
        except ValueError as exc:
            assert "requires a position_selector" in str(exc)
        else:
            raise AssertionError("Expected pore blocker mutator to require a selector")
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_samples_small_residues_only_on_selected_positions() -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer("YFLS")
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})
        _init_global("exclude_AA", [])

        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
            avoid_current_residue=True,
        )

        mutator.compute_mutation({"excluded_pos_per_proto": {"A": [0, 3]}})

        assert len(protomer_a.prob_calls) == 1
        probs = protomer_a.prob_calls[0]
        assert probs.shape == (1, 4, 20)

        allowed = {restype_str_to_int[aa] for aa in "GAST"}
        for pos in [1, 2]:
            nonzero = set(np.flatnonzero(probs[0, pos] > 0.0).tolist())
            assert nonzero <= allowed
            assert restype_str_to_int[protomer_a.sequence[pos]] not in nonzero

        assert probs[0, 0, restype_str_to_int["Y"]] == 1.0
        assert probs[0, 3, restype_str_to_int["S"]] == 1.0
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_queues_all_selected_blockers_despite_global_mutation_num() -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer("YFLS")
        segment_manager = _SegmentManager()
        segment_manager.mutations = [
            ("segA", (0, "Y", "G", 0.1, 0.3)),
            ("segA", (1, "F", "A", 0.2, 0.2)),
            ("segA", (2, "L", "S", 0.3, 0.1)),
        ]
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})
        _init_global("segment_manager", segment_manager)
        global_config.set_variable("mutation_num", 1)

        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
        )
        mutator.mutation_weight = 1.0
        mutator.prepare_mutation_output(
            mutation_output={},
            protomer=None,
            evaluation_scores=StandardScores(names=[], plddt={}, complex_paths={}),
            runtime_config={},
        )

        assert segment_manager.added is not None
        assert segment_manager.added[1] == 3
        assert [mutation[1][-1] for mutation in segment_manager.added[0]] == [
            0.1,
            0.2,
            0.3,
        ]
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_respects_explicit_max_mutations() -> None:
    global_config.reset()
    try:
        segment_manager = _SegmentManager()
        segment_manager.mutations = [
            ("segA", (0, "Y", "G", 0.1, 0.3)),
            ("segA", (1, "F", "A", 0.2, 0.2)),
            ("segA", (2, "L", "S", 0.3, 0.1)),
        ]
        _init_global("protomers", {"A": _RecordingProtomer("YFLS")})
        _init_global("segments", {})
        _init_global("segment_manager", segment_manager)
        global_config.set_variable("mutation_num", 10)

        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
            max_mutations=2,
        )
        mutator.prepare_mutation_output(
            mutation_output={},
            protomer=None,
            evaluation_scores=StandardScores(names=[], plddt={}, complex_paths={}),
            runtime_config={},
        )

        assert segment_manager.added is not None
        assert segment_manager.added[1] == 2
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_inactive_step_queues_no_mutations() -> None:
    global_config.reset()
    try:
        segment_manager = _SegmentManager()
        segment_manager.mutations = [
            ("segA", (0, "Y", "G", 0.1, 0.3)),
            ("segA", (1, "F", "A", 0.2, 0.2)),
        ]
        protomer_a = _RecordingProtomer("YF")
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})
        _init_global("segment_manager", segment_manager)
        _init_global("exclude_AA", [])
        global_config.set_variable("mutation_num", 10)
        global_config.set_variable("step", 2)

        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
            active_steps=[1, 3],
        )
        mutation_input = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )
        mutator.compute_mutation(mutation_input)
        mutator.prepare_mutation_output(
            mutation_output={},
            protomer=None,
            evaluation_scores=StandardScores(names=[], plddt={}, complex_paths={}),
            runtime_config={},
        )

        assert mutation_input["pore_blocker_active"] is False
        assert mutation_input["excluded_pos_per_proto"] == {"A": [0, 1]}
        assert segment_manager.added is not None
        assert segment_manager.added[1] == 0
        assert mutator.last_mutation_telemetry["metrics"]["pore_blocker_reason"] == (
            "inactive_step"
        )
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_active_interval_advances_activation_index(monkeypatch) -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer("YF")
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})
        _init_global("exclude_AA", [])

        def all_mutable_selector(_position_selector, _evaluation_result, protomers):
            return {
                proto: np.ones(protomer.length, dtype=bool)
                for proto, protomer in protomers.items()
            }

        monkeypatch.setattr(
            "astrolabe.mutators.utils.construct_position_selector",
            all_mutable_selector,
        )
        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
            active_step_interval=2,
            active_step_offset=1,
        )

        global_config.set_variable("step", 1)
        mutation_input_1 = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )
        activation_1 = global_config.get_variable("pore_bottleneck_activation_index")

        global_config.set_variable("step", 2)
        mutation_input_2 = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )

        global_config.set_variable("step", 3)
        mutation_input_3 = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )
        activation_3 = global_config.get_variable("pore_bottleneck_activation_index")

        assert mutation_input_1["pore_blocker_active"] is True
        assert mutation_input_2["pore_blocker_active"] is False
        assert mutation_input_3["pore_blocker_active"] is True
        assert activation_1 == 0
        assert activation_3 == 1
    finally:
        global_config.reset()


def test_pore_blocker_mcmc_does_not_fallback_when_selector_hits_only_fixed_positions(monkeypatch) -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer("YFLS")
        protomer_a.mutable_mask = np.array([False, True, True, True])
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})
        _init_global("exclude_AA", [])

        def fixed_only_selector(_position_selector, _evaluation_result, protomers):
            return {
                proto: np.array([True, False, False, False], dtype=bool)
                for proto in protomers
            }

        monkeypatch.setattr(
            "astrolabe.mutators.utils.construct_position_selector",
            fixed_only_selector,
        )

        mutator = PoreBlockingSubstitutionMutator(
            position_selector="dummy_selector()",
            target_residues="GAST",
        )
        mutation_input = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )

        assert mutation_input["excluded_pos_per_proto"] == {"A": [0, 1, 2, 3]}

        mutator.compute_mutation(mutation_input)
        probs = protomer_a.prob_calls[0]
        for pos, aa in enumerate(protomer_a.sequence):
            assert probs[0, pos, restype_str_to_int[aa]] == 1.0
            assert np.count_nonzero(probs[0, pos]) == 1
    finally:
        global_config.reset()


def test_mcmc_keeps_legacy_fallback_when_selector_hits_only_fixed_positions(monkeypatch) -> None:
    global_config.reset()
    try:
        protomer_a = _RecordingProtomer("YFLS")
        protomer_a.mutable_mask = np.array([False, True, True, True])
        _init_global("protomers", {"A": protomer_a})
        _init_global("segments", {})

        def fixed_only_selector(_position_selector, _evaluation_result, protomers):
            return {
                proto: np.array([True, False, False, False], dtype=bool)
                for proto in protomers
            }

        monkeypatch.setattr(
            "astrolabe.mutators.utils.construct_position_selector",
            fixed_only_selector,
        )

        mutator = RandomSubstitutionMutator(
            mutation_method="random",
            select_positions="random",
            position_selector="dummy_selector()",
        )
        mutation_input = mutator.prepare_mutation_input(
            {"A": protomer_a},
            evaluation_scores=SimpleNamespace(),
            runtime_config={},
        )

        assert mutation_input["excluded_pos_per_proto"] == {"A": [0]}
    finally:
        global_config.reset()


def test_aggregate_per_proto_scores_keeps_direct_protomer_scores_for_mixed_order_oligos() -> None:
    global_config.reset()
    try:
        _init_global(
            "protomers",
            {
                "A": _RecordingProtomer(),
                "B": _RecordingProtomer(),
            },
        )
        scores = StandardScores(
            names=["AB", "BA"],
            plddt={"AB": [0.9], "BA": [0.8]},
            complex_paths={"AB": SimpleNamespace(), "BA": SimpleNamespace()},
            per_proto_scores={
                "A": np.array([0.1, 0.2], dtype=float),
                "B": np.array([0.3, 0.4], dtype=float),
            },
        )

        aggregated = aggregate_per_proto_scores(scores, runtime_config={})

        assert set(aggregated) == {"A", "B"}
        assert np.allclose(aggregated["A"], np.array([[0.1, 0.2]], dtype=float))
        assert np.allclose(aggregated["B"], np.array([[0.3, 0.4]], dtype=float))
    finally:
        global_config.reset()


def test_aggregate_per_proto_scores_prefers_chain_mapping_for_homooligomers_even_when_chain_ids_overlap_protomers() -> None:
    global_config.reset()
    try:
        _init_global(
            "protomers",
            {
                "A": _RecordingProtomer(),
                "B": _RecordingProtomer(),
                "C": _RecordingProtomer(),
                "D": _RecordingProtomer(),
            },
        )
        scores = StandardScores(
            names=["AAAA"],
            plddt={"AAAA": [0.9]},
            complex_paths={"AAAA": SimpleNamespace()},
            per_proto_scores={
                "A": np.array([0.1, 0.2], dtype=float),
                "B": np.array([0.3, 0.4], dtype=float),
                "C": np.array([0.5, 0.6], dtype=float),
                "D": np.array([0.7, 0.8], dtype=float),
            },
        )

        aggregated = aggregate_per_proto_scores(scores, runtime_config={})

        assert set(aggregated) == {"A"}
        assert np.allclose(
            aggregated["A"],
            np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
                dtype=float,
            ),
        )
    finally:
        global_config.reset()


def test_aggregate_per_proto_scores_uses_configured_named_score_terms() -> None:
    global_config.reset()
    try:
        _init_global(
            "protomers",
            {
                "A": _RecordingProtomer(),
            },
        )
        scores = StandardScores(
            names=["A"],
            plddt={"A": [0.9]},
            complex_paths={"A": SimpleNamespace()},
            per_proto_scores={
                "A": np.array([0.9, 0.8, 0.7], dtype=float),
            },
            per_proto_score_terms={
                "A": {
                    "plddt": np.array([0.9, 0.8, 0.7], dtype=float),
                    "sap": np.array([1.0, 2.0, 3.0], dtype=float),
                    "buns": np.array([0.0, 1.0, 0.0], dtype=float),
                }
            },
        )

        aggregated = aggregate_per_proto_scores(
            scores,
            runtime_config={
                "per_residue_scores": {
                    "mutator_score_terms": {
                        "plddt": {"weight": 1.0},
                        "sap": {"weight": -0.1},
                        "buns": {"weight": -0.5},
                    }
                }
            },
        )

        assert np.allclose(
            aggregated["A"],
            np.array([0.8, 0.1, 0.4], dtype=float),
        )
    finally:
        global_config.reset()
