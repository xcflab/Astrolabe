import pathlib
from abc import abstractmethod
from typing import Any, Dict

from astrolabe.core import global_config
from astrolabe.mutators.capabilities import (
    MutatorCapabilities,
    validate_mutator_capabilities,
)
from astrolabe.mutators.registry import MutatorRegistry, register_mutator_factory
from astrolabe.mutators.utils import (
    get_transition_consistency_gate_context,
    mutation_output_supports_sampling_probs,
)
from omegaconf import OmegaConf


class Mutator:
    position_selector = None
    capabilities = MutatorCapabilities()

    def shutdown(self) -> None:
        """Release external resources held by the mutator."""
        return None

    @abstractmethod
    def compute_mutation(self, input_feat):
        pass

    def mutate(self, protomer, evaluation_scores=None, runtime_config=None):
        for name, segment in global_config.segments.items():
            segment.clean_up()
        sentinel = object()
        previous_alias = sentinel
        previous_record_flag = sentinel
        if runtime_config is not None:
            previous_alias = runtime_config.get("current_mutator_alias", sentinel)
            previous_record_flag = runtime_config.get(
                "record_mutator_selector_scores", sentinel
            )
            runtime_config["current_mutator_alias"] = (
                getattr(self, "alias", None)
                or getattr(self, "name", None)
                or type(self).__name__
            )
            runtime_config["record_mutator_selector_scores"] = True
        try:
            input_feat = self.prepare_mutation_input(
                protomer, evaluation_scores, runtime_config
            )
        finally:
            if runtime_config is not None:
                if previous_alias is sentinel:
                    runtime_config.pop("current_mutator_alias", None)
                else:
                    runtime_config["current_mutator_alias"] = previous_alias
                if previous_record_flag is sentinel:
                    runtime_config.pop("record_mutator_selector_scores", None)
                else:
                    runtime_config["record_mutator_selector_scores"] = previous_record_flag
        mutation_output = self.compute_mutation(input_feat)
        extra_output = None
        transition_consistency_gate_context = get_transition_consistency_gate_context(
            runtime_config
        )
        if (
            transition_consistency_gate_context is not None
            and float(
                getattr(transition_consistency_gate_context, "weight", 0.0) or 0.0
            )
            > 0.0
            and mutation_output_supports_sampling_probs(mutation_output)
        ):
            if runtime_config is not None:
                previous_record_flag = runtime_config.get(
                    "record_mutator_selector_scores", sentinel
                )
                runtime_config["record_mutator_selector_scores"] = False
            try:
                extra_input_feat = self.prepare_mutation_input(
                    protomer,
                    transition_consistency_gate_context.reference_scores,
                    runtime_config,
                )
            finally:
                if runtime_config is not None:
                    if previous_record_flag is sentinel:
                        runtime_config.pop("record_mutator_selector_scores", None)
                    else:
                        runtime_config["record_mutator_selector_scores"] = (
                            previous_record_flag
                        )
            extra_output = self.compute_mutation(extra_input_feat)

        sentinel = object()
        previous_extra_output = sentinel
        if runtime_config is not None:
            previous_extra_output = runtime_config.get(
                "transition_consistency_gate_output", sentinel
            )
            runtime_config["transition_consistency_gate_output"] = extra_output
        try:
            std_output = self.prepare_mutation_output(
                mutation_output, protomer, evaluation_scores, runtime_config
            )
        finally:
            if runtime_config is not None:
                if previous_extra_output is sentinel:
                    runtime_config.pop("transition_consistency_gate_output", None)
                else:
                    runtime_config["transition_consistency_gate_output"] = (
                        previous_extra_output
                    )
        return std_output

    @abstractmethod
    def prepare_mutation_input(
        self, protomer, evaluation_scores, runtime_config: Dict[str, Any]
    ):
        pass

    @abstractmethod
    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        """
        General workflow:
        1. iterate through the mutation_output["probs"] to get the mutated sites (sample or argmax)
        2. update the sequence with the mutated sites
        3. return the updated protomer
        """
        pass

    def n_mutations(self):
        base_value = getattr(global_config, "mutation_num", None)
        if base_value is None:
            base_value = getattr(global_config, "n_mutations", 0)
        if base_value is None:
            base_value = 0
        weight = getattr(self, "mutation_weight", 1.0)
        try:
            return max(0, int(round(float(base_value) * float(weight))))
        except (TypeError, ValueError):
            return 0


def _merge_runner_conf(base_conf, overrides):
    base_dict = (
        OmegaConf.to_container(base_conf, resolve=True) if base_conf is not None else {}
    )
    runner_dict = dict(base_dict)
    overrides = dict(overrides)
    leftover = {}
    for key, value in overrides.items():
        if key in runner_dict:
            runner_dict[key] = value
        else:
            leftover[key] = value
    return OmegaConf.create(runner_dict), leftover


def build_mutators(mutator_conf, raw_model_results_dir: "pathlib.Path"):
    enabled_aliases = getattr(mutator_conf, "enabled_aliases", None)
    if enabled_aliases is None:
        enabled_aliases = list((getattr(mutator_conf, "mutator_terms", {}) or {}).keys())
    enabled_aliases = [str(a) for a in enabled_aliases if str(a)]
    enabled_set = set(enabled_aliases)

    mutators = {}

    for alias, conf in mutator_conf.mutator_terms.items():
        # Allow turning off mutators via config overlays by setting them to null.
        if conf is None:
            continue
        # Build only mutators that are enabled for this run (derived from stages by
        # argparser, or defaulting to all mutator_terms keys).
        if enabled_set and str(alias) not in enabled_set:
            continue
        conf = dict(conf)
        name = conf.pop("_type", None)
        if not name:
            raise ValueError(f"Mutator name must be specified for {alias} in `_type`")

        selector_override = conf.pop("position_selector", None)

        position_selector = selector_override
        selectors = getattr(mutator_conf, "selectors", None)
        if position_selector is None and selectors:
            position_selector = selectors.get(alias, None)
            if position_selector is None:
                position_selector = selectors.get(name, None)
            if position_selector is None:
                position_selector = selectors.get(str(name).lower(), None)
        if position_selector is not None:
            conf["position_selector"] = position_selector

        try:
            factory = MutatorRegistry.get(name)
        except KeyError as exc:
            available = ", ".join(MutatorRegistry.list_available()) or "<none>"
            raise ValueError(
                f"Unknown mutator name {name!r}. Available mutator types: {available}"
            ) from exc

        mutator = factory(
            conf,
            alias=alias,
            mutator_conf=mutator_conf,
            raw_model_results_dir=raw_model_results_dir,
        )
        mutator.alias = str(alias)
        try:
            protomers = global_config.protomers
        except RuntimeError:
            protomers = {}
        validate_mutator_capabilities(mutator, alias=str(alias), protomers=protomers)
        mutators[alias] = mutator

    return mutators


@register_mutator_factory("mcmc")
def _build_mcmc_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.random_substitution import RandomSubstitutionMutator

    kwargs = {**OmegaConf.to_container(mutator_conf.MCMC, resolve=True), **term_conf}
    return RandomSubstitutionMutator(**kwargs)


@register_mutator_factory("pore_blocker_mcmc")
def _build_pore_blocker_mcmc_mutator(
    term_conf,
    *,
    alias,
    mutator_conf,
    raw_model_results_dir,
):
    from astrolabe.mutators.random_substitution import PoreBlockingSubstitutionMutator

    base_conf = OmegaConf.to_container(mutator_conf.MCMC, resolve=True)
    kwargs = {**base_conf, **term_conf}
    kwargs.setdefault("mutation_method", "frequency_adjusted")
    kwargs.setdefault("select_positions", "plddt")
    return PoreBlockingSubstitutionMutator(**kwargs)


def _build_mpnn_mutator(
    mutator_cls,
    term_conf,
    *,
    mutator_conf,
    raw_model_results_dir,
    output_name,
):
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN

    runner_conf, mutator_kwargs = _merge_runner_conf(mutator_conf.LigandMPNN, term_conf)
    runner_conf["out_folder"] = str(raw_model_results_dir / output_name)
    sequence_model_runner = LigandMPNN(runner_conf)
    return mutator_cls(
        sequence_model_runner=sequence_model_runner,
        **mutator_kwargs,
    )


@register_mutator_factory("ligandmpnn")
def _build_ligandmpnn_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.ligandmpnn import LigandMPNNMutator

    return _build_mpnn_mutator(
        LigandMPNNMutator,
        term_conf,
        mutator_conf=mutator_conf,
        raw_model_results_dir=raw_model_results_dir,
        output_name="LigandMPNN",
    )


@register_mutator_factory("grouped_ligandmpnn")
def _build_grouped_ligandmpnn_mutator(
    term_conf, *, alias, mutator_conf, raw_model_results_dir
):
    from astrolabe.mutators.ligandmpnn import GroupedLigandMPNNMutator

    return _build_mpnn_mutator(
        GroupedLigandMPNNMutator,
        term_conf,
        mutator_conf=mutator_conf,
        raw_model_results_dir=raw_model_results_dir,
        output_name="GroupedLigandMPNN",
    )


def _as_proteinmpnn_term_conf(term_conf):
    term_conf = dict(term_conf)
    model_type = str(term_conf.get("model_type") or "protein_mpnn").lower()
    if model_type != "protein_mpnn":
        raise ValueError(
            "ProteinMPNN mutators require model_type: protein_mpnn, "
            f"got {model_type!r}."
        )
    term_conf["model_type"] = "protein_mpnn"
    return term_conf


@register_mutator_factory("ProteinMPNN")
def _build_proteinmpnn_mutator(
    term_conf, *, alias, mutator_conf, raw_model_results_dir
):
    from astrolabe.mutators.proteinmpnn import ProteinMPNNMutator

    return _build_mpnn_mutator(
        ProteinMPNNMutator,
        _as_proteinmpnn_term_conf(term_conf),
        mutator_conf=mutator_conf,
        raw_model_results_dir=raw_model_results_dir,
        output_name=f"ProteinMPNN_{alias}",
    )


@register_mutator_factory("MultistateProteinMPNN")
@register_mutator_factory("multistate_proteinmpnn")
@register_mutator_factory("FarApartTiedMultitargetLigandMPNN")
@register_mutator_factory("far_apart_tied_multitarget_ligandmpnn")
def _build_multistate_proteinmpnn_mutator(
    term_conf, *, alias, mutator_conf, raw_model_results_dir
):
    from astrolabe.mutators.proteinmpnn import MultistateProteinMPNNMutator

    return _build_mpnn_mutator(
        MultistateProteinMPNNMutator,
        _as_proteinmpnn_term_conf(term_conf),
        mutator_conf=mutator_conf,
        raw_model_results_dir=raw_model_results_dir,
        output_name=f"MultistateProteinMPNN_{alias}",
    )


@register_mutator_factory("dynamicmpnn")
def _build_dynamicmpnn_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.dynamicmpnn import DynamicMPNNMutator
    from astrolabe.mutators.wrappers.dynamicmpnn_run import DynamicMPNNRunner

    runner_conf, mutator_kwargs = _merge_runner_conf(
        mutator_conf.DynamicMPNN, term_conf
    )
    runner_conf["output_dir"] = str(raw_model_results_dir / "DynamicMPNN")
    return DynamicMPNNMutator(
        sequence_model_runner=DynamicMPNNRunner(runner_conf),
        **mutator_kwargs,
    )


@register_mutator_factory("surfdesign")
def _build_surfdesign_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.surfdesign import SurfDesignMutator
    from astrolabe.mutators.wrappers.surfdesign_runner import SurfDesignRunner

    runner_conf, mutator_kwargs = _merge_runner_conf(
        mutator_conf.SurfDesign, term_conf
    )
    runner_conf["output_dir"] = str(raw_model_results_dir / "SurfDesign")
    return SurfDesignMutator(
        sequence_model_runner=SurfDesignRunner(runner_conf),
        **mutator_kwargs,
    )


@register_mutator_factory("surfpro")
def _build_surfpro_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.surfpro import SurfProMutator
    from astrolabe.mutators.wrappers.surfpro_runner import SurfProRunner

    runner_conf, mutator_kwargs = _merge_runner_conf(mutator_conf.SurfPro, term_conf)
    runner_conf["output_dir"] = str(raw_model_results_dir / "SurfPro")
    return SurfProMutator(
        sequence_model_runner=SurfProRunner(runner_conf),
        **mutator_kwargs,
    )


@register_mutator_factory("carbondesign")
def _build_carbondesign_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.carbondesign import CarbonDesignMutator
    from astrolabe.mutators.wrappers.carbondesign_runner import CarbonDesignRunner

    runner_conf, mutator_kwargs = _merge_runner_conf(mutator_conf.CarbonDesign, term_conf)
    runner_conf["output_dir"] = str(raw_model_results_dir / "CarbonDesign")
    sequence_model_runner = CarbonDesignRunner(runner_conf)
    return CarbonDesignMutator(
        sequence_model_runner=sequence_model_runner,
        **mutator_kwargs,
    )


@register_mutator_factory("esmif")
def _build_esmif_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.esm_if1 import ESMIF1Mutator

    return ESMIF1Mutator(**term_conf)


@register_mutator_factory("esm2")
def _build_esm2_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.esm2 import ESM2Mutator

    return ESM2Mutator(**term_conf)


@register_mutator_factory("esm3")
def _build_esm3_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.esm3 import ESM3Mutator

    return ESM3Mutator(**term_conf)


@register_mutator_factory("motifbreaker")
def _build_motifbreaker_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.motif_breaker import MotifBreakerMutator

    return MotifBreakerMutator(**term_conf)


@register_mutator_factory("dplm")
def _build_dplm_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.dplm import DPLMMutator

    runner_kwargs = {**OmegaConf.to_container(mutator_conf.DPLM, resolve=True), **term_conf}
    runner_kwargs["work_dir"] = str(raw_model_results_dir / "DPLM")
    return DPLMMutator(**runner_kwargs)


@register_mutator_factory("xmix")
def _build_xmix_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.x_mix import XMixMutator

    return XMixMutator(**term_conf)


@register_mutator_factory("nampnn")
def _build_nampnn_mutator(term_conf, *, alias, mutator_conf, raw_model_results_dir):
    from astrolabe.mutators.na_mpnn import NAMPNNMutator
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    runner_conf, mutator_kwargs = _merge_runner_conf(mutator_conf.NAMPNN, term_conf)
    sequence_model_runner = NAMPNNRunner(
        runner_conf,
        output_root=raw_model_results_dir / "NAMPNN",
    )
    return NAMPNNMutator(
        sequence_model_runner=sequence_model_runner,
        **mutator_kwargs,
    )
