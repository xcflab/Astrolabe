from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import Mutator, build_mutators
from astrolabe.mutators.capabilities import (
    PROTEIN_ONLY,
    MutatorCapabilities,
    MutatorSelectionPolicy,
)
from astrolabe.mutators.dplm import DPLMMutator
from astrolabe.mutators.dynamicmpnn import DynamicMPNNMutator
from astrolabe.mutators.esm3 import ESM3Mutator
from astrolabe.mutators.ligandmpnn import LigandMPNNMutator
from astrolabe.mutators.random_substitution import RandomSubstitutionMutator
from astrolabe.mutators.na_mpnn import NAMPNNMutator
from astrolabe.mutators.registry import register_mutator_factory
from astrolabe.mutators.surfdesign import SurfDesignMutator
from astrolabe.mutators.surfpro import SurfProMutator


class _ProteinOnlyMutator(Mutator):
    capabilities = MutatorCapabilities(supported_types=PROTEIN_ONLY)

    def compute_mutation(self, input_feat):
        return input_feat

    def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
        return {}

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        return protomer


@register_mutator_factory("cap_test_protein_only")
def _factory(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    return _ProteinOnlyMutator()


def test_representative_mutators_declare_capabilities() -> None:
    assert RandomSubstitutionMutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert RandomSubstitutionMutator.capabilities.supports_type(ProtomerTypes.RNA)
    assert RandomSubstitutionMutator.capabilities.supports_type(ProtomerTypes.DNA)
    assert RandomSubstitutionMutator.capabilities.needs_structure is False

    assert LigandMPNNMutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert LigandMPNNMutator.capabilities.needs_structure is True
    assert LigandMPNNMutator.capabilities.outputs_probabilities is True
    assert LigandMPNNMutator.capabilities.supports_transition_blending is True
    assert (
        LigandMPNNMutator.capabilities.supports_initialization_from_structure
        is True
    )

    assert (
        DynamicMPNNMutator.capabilities.supports_initialization_from_structure
        is False
    )

    assert DPLMMutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert DPLMMutator.capabilities.needs_structure is False
    assert DPLMMutator.capabilities.outputs_scores is True
    assert DPLMMutator.capabilities.supports_shared_selection_policy is False

    assert NAMPNNMutator.capabilities.supports_type(ProtomerTypes.RNA)
    assert NAMPNNMutator.capabilities.supports_type(ProtomerTypes.DNA)
    assert NAMPNNMutator.capabilities.needs_structure is True

    assert SurfDesignMutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert SurfDesignMutator.capabilities.needs_structure is True
    assert SurfDesignMutator.capabilities.outputs_probabilities is True

    assert SurfProMutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert SurfProMutator.capabilities.needs_structure is True
    assert SurfProMutator.capabilities.outputs_probabilities is True

    assert ESM3Mutator.capabilities.supports_type(ProtomerTypes.PROTEIN)
    assert ESM3Mutator.capabilities.supports_type(ProtomerTypes.RNA) is False
    assert ESM3Mutator.capabilities.supports_transition_blending is True


def test_build_mutators_rejects_incompatible_protomer_types(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {"R": SimpleNamespace(type=ProtomerTypes.RNA)},
    )
    mutator_conf = OmegaConf.create(
        {
            "mutator_terms": {"M1": {"_type": "cap_test_protein_only"}},
        }
    )

    with pytest.raises(ValueError, match="does not support any configured protomer"):
        build_mutators(mutator_conf, tmp_path)

    global_config.reset()


def test_build_mutators_allows_mixed_configs_with_supported_intersection(
    tmp_path: Path,
) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "L": SimpleNamespace(type=ProtomerTypes.CCD),
        },
    )
    mutator_conf = OmegaConf.create(
        {
            "mutator_terms": {"M1": {"_type": "cap_test_protein_only"}},
        }
    )

    built = build_mutators(mutator_conf, tmp_path)

    assert "M1" in built
    global_config.reset()


def test_mutator_selection_policy_is_opt_in_and_isolated() -> None:
    candidates = [
        {"entity": "A", "entropy": 0.1, "score": 0.9},
        {"entity": "A", "entropy": 0.9, "score": 0.1},
        {"entity": "B", "entropy": 0.5, "score": 0.5},
    ]

    assert MutatorSelectionPolicy().apply(candidates) == candidates

    policy = MutatorSelectionPolicy(
        sort_by="entropy",
        descending=True,
        per_entity_limit={"A": 1, "B": 1},
        limit=2,
    )
    selected = policy.apply(candidates)

    assert selected == [candidates[1], candidates[2]]
