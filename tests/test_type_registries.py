from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.mutators.base import Mutator, build_mutators


def test_construct_evaluators_supports_custom_registry_types(tmp_path: Path) -> None:
    from astrolabe.evaluators.registry import register_evaluator_factory

    class _TestEvaluator(Evaluator):
        def run_function(self, input_feat):
            return input_feat

        def prepare_input(self, inputs, runtime_config):
            return inputs

        def prepare_output(self, output, inputs, runtime_config):
            return output

    @register_evaluator_factory("test_eval")
    def _factory(term_conf, *, alias, raw_model_results_dir, oligo_names, evaluator_conf):
        assert alias == "E1"
        assert raw_model_results_dir == tmp_path
        assert oligo_names == ["A"]
        assert evaluator_conf is not None
        assert term_conf is not None
        return _TestEvaluator()

    evaluator_conf = OmegaConf.create({"evaluator_terms": {"E1": {"_type": "test_eval"}}})
    evaluators = construct_evaluators(
        evaluator_conf,
        tmp_path,
        oligo_names=["A"],
        enabled_aliases=["E1"],
    )
    assert isinstance(evaluators["E1"], _TestEvaluator)


def test_build_mutators_supports_custom_registry_types(tmp_path: Path) -> None:
    from astrolabe.mutators.registry import register_mutator_factory

    class _TestMutator(Mutator):
        def compute_mutation(self, input_feat):
            return input_feat

        def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
            return {}

        def prepare_mutation_output(
            self, mutation_output, protomer, evaluation_scores, runtime_config
        ):
            return protomer

    @register_mutator_factory("test_mutator")
    def _factory(term_conf, *, alias, mutator_conf, raw_model_results_dir):
        assert alias == "M1"
        assert raw_model_results_dir == tmp_path
        assert mutator_conf is not None
        assert term_conf is not None
        return _TestMutator()

    mutator_conf = OmegaConf.create({"mutator_terms": {"M1": {"_type": "test_mutator"}}})
    mutators = build_mutators(mutator_conf, tmp_path)
    assert isinstance(mutators["M1"], _TestMutator)
