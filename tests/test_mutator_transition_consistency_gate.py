from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.nucleic_acid_design import NucleicAcidDesignMutator
from astrolabe.mutators import utils as mutator_utils


def _init_runtime() -> None:
    global_config.reset()
    global_config.init_const("segments", {})


def test_build_mutator_runtime_uses_context_presence_as_gate_activity() -> None:
    _init_runtime()
    context = object()
    stage_manager = SimpleNamespace(
        current_oligo=["AB"],
        transition_consistency_gate_context=context,
    )
    global_config.init_const("stage_manager", stage_manager)

    active_runtime = mutator_utils.build_mutator_runtime_config(step=1)
    assert active_runtime["transition_consistency_gate_context"] is context

    stage_manager.transition_consistency_gate_context = None
    inactive_runtime = mutator_utils.build_mutator_runtime_config(step=2)
    assert inactive_runtime["transition_consistency_gate_context"] is None


class _SamplingMutator(Mutator):
    def __init__(self) -> None:
        self.prepare_calls = []
        self.compute_calls = []
        self.runtime_seen = []

    def compute_mutation(self, input_feat):
        self.compute_calls.append(input_feat)
        return {
            "A": {
                "sampling_probs": np.ones((1, 1, 20), dtype=float) / 20.0,
            }
        }

    def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
        self.prepare_calls.append(evaluation_scores)
        return f"input:{evaluation_scores}"

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        self.runtime_seen.append(dict(runtime_config or {}))
        return mutation_output


class _PlainMutator(Mutator):
    def __init__(self) -> None:
        self.prepare_calls = []
        self.compute_calls = []
        self.runtime_seen = []

    def compute_mutation(self, input_feat):
        self.compute_calls.append(input_feat)
        return {"plain": input_feat}

    def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
        self.prepare_calls.append(evaluation_scores)
        return f"input:{evaluation_scores}"

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        self.runtime_seen.append(dict(runtime_config or {}))
        return mutation_output


def test_mutator_base_runs_transition_consistency_gate_hook_for_sampling_outputs() -> None:
    _init_runtime()
    mutator = _SamplingMutator()
    runtime_config = {
        "transition_consistency_gate_context": SimpleNamespace(
            weight=0.25, reference_scores="old-scores"
        ),
    }

    result = mutator.mutate(
        protomer={},
        evaluation_scores="current-scores",
        runtime_config=runtime_config,
    )

    assert mutator.prepare_calls == ["current-scores", "old-scores"]
    assert mutator.compute_calls == ["input:current-scores", "input:old-scores"]
    assert (
        mutator.runtime_seen[0]["transition_consistency_gate_output"] is not None
    )
    assert "transition_consistency_gate_output" not in runtime_config
    assert result["A"]["sampling_probs"].shape == (1, 1, 20)


def test_mutator_base_skips_transition_consistency_gate_hook_for_plain_outputs() -> None:
    _init_runtime()
    mutator = _PlainMutator()
    runtime_config = {
        "transition_consistency_gate_context": SimpleNamespace(
            weight=0.25, reference_scores="old-scores"
        ),
    }

    result = mutator.mutate(
        protomer={},
        evaluation_scores="current-scores",
        runtime_config=runtime_config,
    )

    assert mutator.prepare_calls == ["current-scores"]
    assert mutator.compute_calls == ["input:current-scores"]
    assert (
        mutator.runtime_seen[0]["transition_consistency_gate_output"] is None
    )
    assert "transition_consistency_gate_output" not in runtime_config
    assert result["plain"] == "input:current-scores"


def test_nucleic_acid_mutator_passes_transition_consistency_gate_runtime_into_common_sampling_helper(
    monkeypatch,
) -> None:
    _init_runtime()

    class _DummyNAMutator(NucleicAcidDesignMutator):
        target_type = ProtomerTypes.RNA

        def compute_mutation(self, input_feat):
            return input_feat

        def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
            return {}

    captured = {}
    monkeypatch.setattr(
        "astrolabe.mutators.nucleic_acid_design.prepare_exclude_sites",
        lambda *args, **kwargs: {"A": []},
    )
    monkeypatch.setattr(
        "astrolabe.mutators.nucleic_acid_design.sample_mutations_by_aa_probs",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.RNA,
                length=1,
                sequence="A",
                components=["segA"],
            )
        },
    )
    global_config.init_const(
        "segment_manager",
        SimpleNamespace(add_mutations=lambda *args, **kwargs: None),
    )
    global_config.segments["segA"] = SimpleNamespace(sequence="A")
    mutator = _DummyNAMutator()

    mutator.prepare_mutation_output(
        mutation_output={"A": {"sampling_probs": np.ones((1, 1, 4), dtype=float) / 4.0}},
        protomers=global_config.protomers,
        evaluation_scores=SimpleNamespace(names=["A"], per_proto_scores=None),
        runtime_config={
            "transition_consistency_gate_context": SimpleNamespace(weight=0.25)
        },
    )

    assert (
        captured["runtime_config"]["transition_consistency_gate_context"].weight == 0.25
    )


def test_collect_blended_proto_probabilities_handles_tensor_like_values_without_numpy_array_protocol(
    monkeypatch,
) -> None:
    _init_runtime()

    class _FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

        def __array__(self, dtype=None):
            raise TypeError("direct numpy conversion should not be used")

    monkeypatch.setattr(mutator_utils, "torch", SimpleNamespace(Tensor=_FakeTensor))
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                length=1,
                sequence="A",
            )
        },
    )

    current_scores = SimpleNamespace(names=["A"])

    def _fake_collect(evaluation_scores, mutation_output, *, protomer_type=None):
        if evaluation_scores is current_scores:
            return {"A": _FakeTensor([[1.0, 2.0]])}
        return {"A": _FakeTensor([[3.0, 4.0]])}

    monkeypatch.setattr(mutator_utils, "collect_proto_probabilities", _fake_collect)

    blended = mutator_utils.collect_blended_proto_probabilities(
        evaluation_scores=current_scores,
        mutation_output=object(),
        runtime_config={
            "transition_consistency_gate_context": SimpleNamespace(
                weight=0.25,
                reference_scores="reference-scores",
            ),
            "transition_consistency_gate_output": object(),
        },
    )

    assert np.allclose(blended["A"], np.asarray([[1.5, 2.5]], dtype=float))
