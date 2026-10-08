import ast
import copy
import functools
import logging
import math
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None
from omegaconf import DictConfig

from astrolabe.core import global_config
from astrolabe.losses.utils import LossRegistry
from astrolabe.score_contributions import (
    score_contributions_from_runtime_config,
    score_contributions_to_snapshot,
)
from astrolabe.utils.log_events import event_span, summarize_event_items

# Global registry instance
loss_registry = LossRegistry()
logger = logging.getLogger(__name__)
INTRINSIC_PARAMS = {"oligomer_specific"}
DYNAMIC_BINDING_SITE_LOSS = "dynamic_binding_site_allowed_min_dist"


def extract_loss_terms(
    expressions: List[str], excluded_terms: None | List[str] = None
) -> List[str]:
    """Extract unique loss terms from expressions, including attribute accesses."""
    included_terms = set()
    for expr in expressions:
        if expr:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "best":
                    continue
                if isinstance(node, ast.Attribute):
                    current = node
                    while isinstance(current, ast.Attribute):
                        current = current.value
                    if isinstance(current, ast.Name) and current.id == "best":
                        continue
                # Handle both simple names and attribute accesses (e.g., ABC.cross_contact_lis)
                if isinstance(node, ast.Name):
                    included_terms.add(node.id)
                elif isinstance(node, ast.Attribute):
                    included_terms.add(node.attr)
    if excluded_terms:
        included_terms -= set(excluded_terms)
    return list(included_terms)


def extract_best_namespace_terms(expr):
    """Extract leaf metric names referenced through the `best` namespace."""
    if not expr:
        return set()
    tree = ast.parse(expr, mode="eval")
    chains: list[tuple[str, ...]] = []

    def _attribute_chain(node: ast.Attribute) -> tuple[str, ...] | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        parts.append(current.id)
        return tuple(reversed(parts))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if chain is not None and len(chain) >= 2 and chain[0] == "best":
            chains.append(chain)

    leaves: set[str] = set()
    for chain in chains:
        is_prefix = any(
            len(other) > len(chain) and other[: len(chain)] == chain
            for other in chains
        )
        if not is_prefix:
            leaves.add(chain[-1])
    return leaves


def build_loss_specs(
    names,
):
    return {
        name: (name, loss_registry.get_function(name), {})
        for name in names
        if name in loss_registry.valid_losses
    }


def _freeze_cache_value(value):
    if isinstance(value, DictConfig):
        return tuple(
            sorted(
                (
                    str(key),
                    _freeze_cache_value(value[key]),
                )
                for key in value.keys()
            )
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _freeze_cache_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


class LossEngine:
    """Enhanced loss computer using registry-based loss functions."""

    def __init__(
        self,
        loss_config: Dict[str, Any],
        runtime_context=None,
        validate_config: bool = True,
    ):
        # Ensure registry-backed loss modules are imported so their decorators run.
        # Best-effort: many modules depend on optional heavy deps (pyrosetta, rdkit, ...).
        try:  # pragma: no cover - import side effects
            import astrolabe.losses  # noqa: F401
        except Exception:
            pass
        self.registry = loss_registry
        self.runtime_context = runtime_context or global_config.build_runtime_context()
        self.defined_losses = loss_config.get("loss_terms", {}) or {}
        self.expression_terms = (
            loss_config.get("expression_terms", {}) or {}
        )

    def setup(
        self,
        weighted_score_expression,
        early_stopping,
        acceptance_threshold,
        acceptance_gate=None,
        validate_config: bool = True,
        prefix_early_stopping=None,
    ):
        # `loss_specs` can change across stages; invalidate any cached score extraction
        # results so the next call recomputes with the updated registry specs.
        self._clear_score_cache()

        # Store expressions
        (
            self.weighted_score_expression,
            self.early_stopping,
            self.acceptance_threshold,
            self.acceptance_gate,
            self.prefix_early_stopping,
        ) = (
            weighted_score_expression,
            early_stopping,
            acceptance_threshold,
            acceptance_gate,
            prefix_early_stopping,
        )
        required_expression_terms = self._ordered_expression_terms_for_expressions(
            [
                self.weighted_score_expression,
                self.early_stopping,
                self.acceptance_threshold,
                self.acceptance_gate,
                self.prefix_early_stopping,
            ]
        )
        included_loss_terms = extract_loss_terms(
            expressions=[
                self.early_stopping,
                self.acceptance_threshold,
                self.acceptance_gate,
                self.prefix_early_stopping,
                self.weighted_score_expression,
                *[
                    self.expression_terms[name]
                    for name in required_expression_terms
                ],
            ],
            excluded_terms=list(self.expression_terms.keys()),
        )
        included_loss_terms.extend(
            extract_best_namespace_terms(self.prefix_early_stopping)
        )
        # Treat explicit `null` loss entries as disabled (useful for overlays).
        defined_loss_terms = {
            k for k, v in (self.defined_losses or {}).items() if v is not None
        }
        # Explicitly defined loss terms should always be computed so they can be
        # referenced outside of the main weighted score expression (e.g. for
        # predocker conditions or debug logging).
        required_loss_terms = set(included_loss_terms) | defined_loss_terms
        # todo for those losses that have no required params, we can use loss name directly
        self.parsed_defined_losses = self._parse_loss_specs(self.defined_losses)

        self.loss_specs = build_loss_specs(
            required_loss_terms.intersection(set(self.registry.param_free_losses))
        )
        self.loss_specs.update(
            {
                loss_name: self.parsed_defined_losses.get(loss_name)
                for loss_name in required_loss_terms.intersection(
                    self.parsed_defined_losses.keys()
                )
            }
        )

        # intersect with defined losses and registered losses
        self.included_loss_terms = list(
            required_loss_terms.intersection(
                defined_loss_terms | set(self.registry.valid_losses)
            )
        )

        # Parse loss specifications
        self.loss_names = list(self.loss_specs.keys())

        if set(self.included_loss_terms) - set(self.loss_names):
            raise ValueError(
                f"Loss terms having required arguments {set(self.included_loss_terms) - set(self.loss_names)} shoule be defined in the `loss_terms`."
            )

        # Validate configuration
        if validate_config:
            self._validate_configuration()

    def _parse_loss_specs(
        self,
        loss_names: Dict[str, DictConfig],
    ) -> Dict[str, Dict[str, Any]]:
        """Parse loss specifications from configuration."""
        # todo not parsed from string, but from a dict in the config
        specs = {}

        for loss_name in loss_names:
            # Handle oligomer-specific format: <oligomer>.<loss>:<params>

            if loss_names[loss_name] is None:
                continue

            params = dict(copy.deepcopy(loss_names[loss_name]))
            if "_type" not in params:
                raise ValueError(
                    f"Loss specification '{loss_name}' must have a '_type' field."
                )
            spec_string = params.pop("_type")

            if "_for" in params:
                oligomer = params.pop("_for")
                if isinstance(oligomer, str):
                    oligomer = oligomer.split(",")

                params["oligomer_specific"] = oligomer

            metadata = self.registry.get_metadata(spec_string)
            if set(params.keys()) - (
                set(metadata.optional_params)
                | set(metadata.required_params)
                | INTRINSIC_PARAMS
            ):
                raise ValueError(
                    f"Invalid parameters for loss '{spec_string}': {set(params.keys()) - (set(metadata.optional_params) | set(metadata.required_params))}"
                )

            if set(metadata.required_params) - set(params.keys()):
                raise ValueError(
                    f"Missing required parameters for loss '{spec_string}': {set(metadata.required_params) - set(params.keys())}"
                )
            specs[loss_name] = (
                spec_string,
                self.registry.get_function(spec_string),
                params,
            )

        return specs

    def _validate_configuration(self) -> None:
        """Validate loss configuration against registry."""
        issues = []

        for loss_name, (spec, func, params) in self.loss_specs.items():
            # Check if loss function exists
            if spec not in self.registry.valid_losses:
                issues.append(f"Unknown loss function: {loss_name}")
                continue

            # Validate parameters
            all_params = list(params.keys() - INTRINSIC_PARAMS)

            param_issues = self.registry.validate_params(spec, all_params)
            issues.extend(param_issues)

        if issues:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )

    def _event_context(self) -> tuple[str | None, int]:
        stage_manager = self.runtime_context.stage_manager
        return getattr(stage_manager, "current_stage", None), self.runtime_context.step

    def compute_loss(self, eval_result, runtime_config):
        """Compute loss and record one aggregate execution unit."""
        stage, step = self._event_context()
        with event_span(
            logger=logger,
            component="loss_engine",
            alias=self.__class__.__name__,
            stage=stage,
            step=step,
            detail="compute_loss",
        ) as span:
            weighted_score, total_env = self._compute_loss(eval_result, runtime_config)
            span.finish(
                result="aggregate",
                weighted_score=weighted_score,
                score_terms=len(total_env),
            )
            return weighted_score, total_env

    def _compute_loss(self, eval_result, runtime_config):
        per_oligo_scores, per_terms_score = self._extract_scores_with_registry(
            eval_result
        )
        default_symbols = self._default_expression_symbols()
        reference_symbols = self._reference_gate_symbols(runtime_config)
        contribution_symbols = self._runtime_score_contribution_symbols(runtime_config)
        total_env = {
            **default_symbols,
            **per_oligo_scores,
            **per_terms_score,
        }
        self._merge_reference_symbols(total_env, reference_symbols)
        self._merge_score_contribution_symbols(total_env, contribution_symbols)
        required_expression_terms = self._ordered_expression_terms_for_expressions(
            [
                self.weighted_score_expression,
                self.early_stopping,
                self.acceptance_threshold,
                self.acceptance_gate,
                self.prefix_early_stopping,
            ]
        )
        expr_values = self._evaluate_expression_terms(
            total_env, self._basic_operations(), term_names=required_expression_terms
        )
        for key, value in expr_values.items():
            per_terms_score[key] = value
        total_env.update(expr_values)
        return self.weighted_score(total_env, self.weighted_score_expression), total_env

    def eval_score(self, eval_result, score_expression) -> float:
        per_oligo_scores, per_terms_score = self._extract_scores_with_registry(
            eval_result
        )
        default_symbols = self._default_expression_symbols()
        total_env = {**default_symbols, **per_oligo_scores, **per_terms_score}
        return self.weighted_score(total_env, score_expression)

    def eval_condition(self, total_env, condition_expression) -> bool:
        """Evaluate a condition expression against the evaluation result."""
        result = self.weighted_score(total_env, condition_expression)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        if torch is not None:
            tensor_type = getattr(torch, "Tensor", None)
            if tensor_type is not None and isinstance(result, tensor_type):
                if result.numel() == 1 and result.dtype == torch.bool:
                    return bool(result.item())

        raise ValueError(
            f"Condition expression '{condition_expression}' did not evaluate to a boolean value, got {result}."
        )

    def score_context(self, eval_result, runtime_config=None) -> dict:
        """Build a score context dict for evaluating expressions/conditions.

        This is the public, non-cached entrypoint used by stage logic (e.g. stage
        entrance conditions). It intentionally does not expose registry internals.
        """
        per_oligo_scores, per_terms_score = self._extract_scores_with_registry(eval_result)
        total_env = {**per_oligo_scores, **per_terms_score}
        self._merge_score_contribution_symbols(
            total_env,
            self._runtime_score_contribution_symbols(runtime_config),
        )
        return total_env

    def score_context_for_expressions(
        self,
        eval_result,
        expressions: List[str | None],
    ) -> dict:
        """Build a score context for arbitrary expressions without mutating stage setup."""
        loss_specs = self._build_loss_specs_for_expressions(expressions)
        per_oligo_scores, per_terms_score = self._extract_scores_from_specs(
            eval_result,
            loss_specs,
        )
        return {**per_oligo_scores, **per_terms_score}

    def weighted_score(self, total_env, expression):
        """Calculate weighted score based on the configured expression."""
        if not expression:
            raise ValueError("No weighted score expression provided.")

        basic_operations = self._basic_operations()

        default_symbols = self._default_expression_symbols()
        augmented_env = {**total_env, **default_symbols}

        expr_symbols = self._evaluate_expression_terms(
            augmented_env,
            basic_operations,
            term_names=self._ordered_expression_terms_for_expressions([expression]),
        )
        augmented_env.update(expr_symbols)

        # Extract variables from the expression and validate availability
        variables = extract_variables(expression)
        allowed_symbols = set(augmented_env.keys()) | set(__builtins__.keys()) | set(
            basic_operations.keys()
        )
        missing_vars = variables - allowed_symbols
        if missing_vars:
            raise ValueError(
                f"Variables in weighted_score_expression not found: {missing_vars}"
            )

        # Prepare context for evaluation
        context = {
            **augmented_env,
            **basic_operations,
        }

        try:
            return eval(expression, {}, context)
        except Exception as exc:
            raise RuntimeError(
                f"Error evaluating weighted score expression: {exc}"
            ) from exc

    def _default_expression_symbols(self):
        symbols = {}

        try:
            protomers = self.runtime_context.protomers or {}
        except RuntimeError:
            protomers = {}
        for name, protomer in protomers.items():
            try:
                length_value = int(getattr(protomer, "length", len(protomer.sequence)))
            except Exception:
                length_value = 0
            symbols[f"{name}_length"] = length_value

        try:
            stage_manager = self.runtime_context.stage_manager
        except RuntimeError:
            stage_manager = None
        if stage_manager is not None:
            all_oligos = getattr(stage_manager, "all_oligos", None)
            if callable(all_oligos):
                all_oligos = all_oligos()
            for oligo in all_oligos or []:
                length = 0
                for proto_name in oligo:
                    proto_obj = protomers.get(proto_name)
                    if proto_obj is None:
                        length = 0
                        break
                    try:
                        length += int(getattr(proto_obj, "length", len(proto_obj.sequence)))
                    except Exception:
                        length = 0
                        break
                symbols[f"{oligo}_length"] = length

        try:
            step_value = self.runtime_context.step or 0
        except RuntimeError:
            step_value = 0
        symbols.setdefault("step", int(step_value))
        symbols.setdefault("iteration", int(step_value))
        symbols.setdefault("prefix_steps", int(step_value) + 1)

        return symbols

    def _reference_gate_symbols(self, runtime_config):
        if runtime_config is None:
            return {}
        getter = runtime_config.get if hasattr(runtime_config, "get") else None
        result = (
            getter("reference_gate_result")
            if callable(getter)
            else getattr(runtime_config, "reference_gate_result", None)
        )
        if result is None:
            return {}
        from astrolabe.structure_start import reference_gate_score_context

        return reference_gate_score_context(result)

    def _runtime_score_contribution_symbols(self, runtime_config):
        contributions = score_contributions_from_runtime_config(runtime_config)
        if not contributions:
            return {}
        return score_contributions_to_snapshot(contributions)

    def _merge_reference_symbols(self, total_env: dict, reference_symbols: dict) -> None:
        for key, value in (reference_symbols or {}).items():
            if isinstance(value, dict) and isinstance(total_env.get(key), dict):
                total_env[key].update(value)
            else:
                total_env[key] = value

    def _merge_score_contribution_symbols(
        self,
        total_env: dict,
        contribution_symbols: dict,
    ) -> None:
        for key, value in (contribution_symbols or {}).items():
            if isinstance(value, dict):
                existing = total_env.get(key)
                if existing is None:
                    total_env[key] = LossTerms(
                        {
                            sub_key: self._as_context_value(sub_value)
                            for sub_key, sub_value in value.items()
                        }
                    )
                    continue
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"Score contribution key {key!r} conflicts with an existing score value"
                    )
                for sub_key, sub_value in value.items():
                    if sub_key in existing:
                        raise ValueError(
                            "Score contribution would overwrite existing score key: "
                            f"{key}.{sub_key}"
                        )
                    existing[sub_key] = self._as_context_value(sub_value)
                continue
            if key in total_env:
                raise ValueError(
                    f"Score contribution would overwrite existing score key: {key}"
                )
            total_env[key] = value

    def _as_context_value(self, value):
        if isinstance(value, dict):
            return LossTerms(
                {key: self._as_context_value(inner) for key, inner in value.items()}
            )
        return value

    def _ordered_expression_terms_for_expressions(self, expressions: List[str | None]) -> List[str]:
        if not self.expression_terms:
            return []

        requested = set()
        for expr in expressions:
            requested.update(extract_variables(expr).intersection(self.expression_terms.keys()))
            requested.update(extract_best_namespace_terms(expr).intersection(self.expression_terms.keys()))

        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise RuntimeError(f"Cyclic expression term dependency detected at '{name}'")
            visiting.add(name)
            deps = extract_variables(self.expression_terms[name]).intersection(self.expression_terms.keys())
            for dep in self.expression_terms.keys():
                if dep in deps:
                    visit(dep)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in self.expression_terms.keys():
            if name in requested:
                visit(name)

        return ordered

    def _evaluate_expression_terms(self, env, ops, term_names=None):
        results = {}
        if not self.expression_terms:
            return results

        context = {**env, **ops}
        if term_names is None:
            term_names = list(self.expression_terms.keys())

        for name in term_names:
            expr = self.expression_terms[name]
            try:
                value = eval(expr, {"__builtins__": {}}, context)
                results[name] = value
                context[name] = value
            except Exception as exc:
                raise RuntimeError(
                    f"Error evaluating expression term '{name}': {exc}"
                ) from exc
        return results

    def _basic_operations(self):
        return {
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "round": round,
            "np.mean": np.mean,
            "np.std": np.std,
            "np": np,
            "math": math,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "tanh": math.tanh,
            "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
            "np.exp": np.exp,
            "np.log": np.log,
            "np.sqrt": np.sqrt,
            "np.tanh": np.tanh,
        }

    def is_early_stopping(self, score_context):
        """Check if early stopping condition is met."""
        if not self.early_stopping:
            return False
        return self.eval_condition(score_context, self.early_stopping)

    def is_prefix_early_stopping(self, score_context, best_score_context):
        """Check if prefix-history early stopping condition is met."""
        if not self.prefix_early_stopping:
            return False
        context = dict(score_context or {})
        context["best"] = self._as_context_value(best_score_context or {})
        return self.eval_condition(context, self.prefix_early_stopping)

    def is_soft_accepted(self, score_context):
        """Check if soft acceptance condition is met."""
        if not self.acceptance_threshold:
            return False
        return self.weighted_score(score_context, self.acceptance_threshold)

    def passes_acceptance_gate(self, score_context):
        """Check if the hard acceptance gate allows this step to be accepted."""
        if not self.acceptance_gate:
            return True
        return self.eval_condition(score_context, self.acceptance_gate)

    def _extract_scores_with_registry(self, eval_result) -> tuple:
        """Extract scores using registered loss functions."""
        return self._extract_scores_with_registry_cached(
            eval_result,
            self._score_cache_token(),
        )

    def _clear_score_cache(self) -> None:
        self._extract_scores_with_registry_cached.cache_clear()

    def _score_cache_token(self):
        if not any(
            spec == DYNAMIC_BINDING_SITE_LOSS
            for spec, _loss_func, _params in getattr(self, "loss_specs", {}).values()
        ):
            return None
        try:
            dynamic_binding_site = getattr(global_config, "dynamic_binding_site", None)
        except RuntimeError:
            dynamic_binding_site = None
        return (
            DYNAMIC_BINDING_SITE_LOSS,
            _freeze_cache_value(dynamic_binding_site),
        )

    @functools.lru_cache(maxsize=4)
    def _extract_scores_with_registry_cached(self, eval_result, _cache_token) -> tuple:
        """Extract scores using registered loss functions."""
        return self._extract_scores_from_specs(eval_result, self.loss_specs)

    def _extract_scores_from_specs(self, eval_result, loss_specs) -> tuple:
        """Extract scores using an explicit set of loss specs."""
        loss_per_term_per_oligo = {}
        stage, step = self._event_context()

        for loss_name, (spec, loss_func, params) in loss_specs.items():
            # Never mutate `params`: `self.loss_specs` is reused across steps and
            # this method is cached.
            oligomer_specific = params.get("oligomer_specific", None)
            call_params = {k: v for k, v in params.items() if k not in INTRINSIC_PARAMS}
            this_eval_result = (
                eval_result.filter_by_oligomer(oligomer_specific)
                if oligomer_specific is not None
                else eval_result
            )
            oligo_names = list(getattr(this_eval_result, "names", None) or [])
            oligo = summarize_event_items(oligo_names)
            if oligo is None and oligomer_specific is not None:
                requested_oligomers = (
                    oligomer_specific
                    if isinstance(oligomer_specific, (list, tuple, set))
                    else [oligomer_specific]
                )
                oligo = summarize_event_items(requested_oligomers)
            with event_span(
                logger=logger,
                component="loss",
                alias=loss_name,
                stage=stage,
                step=step,
                oligo=oligo,
                detail=spec,
            ) as span:
                if oligomer_specific is not None and not oligo_names:
                    scores = {}
                    span.skip(result="no_matching_oligomer", result_count=0)
                else:
                    scores = loss_func(this_eval_result, **call_params)
                    if isinstance(scores, dict):
                        values = list(scores.values())
                        span.finish(
                            result="per_oligomer",
                            result_count=len(scores),
                            result_mean=float(np.mean(values)) if values else 0.0,
                        )
                    else:
                        span.finish(
                            result="scalar",
                            result_value=float(scores),
                        )
            loss_per_term_per_oligo[loss_name] = scores

        mean_scores = {}
        for loss in loss_specs.keys():
            scores = loss_per_term_per_oligo[loss]
            if isinstance(scores, dict):
                values = list(scores.values())
                mean_scores[loss] = float(np.mean(values)) if values else 0.0
            else:
                mean_scores[loss] = scores

        loss_per_oligo_per_term = defaultdict(dict)
        for loss_name, scores in loss_per_term_per_oligo.items():
            if isinstance(scores, dict):
                for oligo_name, score in scores.items():
                    loss_per_oligo_per_term[oligo_name][loss_name] = score
            else:
                # If the loss function returns a single value, assign it to all oligomers
                for oligo_name in eval_result.names:
                    loss_per_oligo_per_term[oligo_name][loss_name] = scores

        return (
            LossTerms({k: LossTerms(v) for k, v in loss_per_oligo_per_term.items()}),
            LossTerms(mean_scores),
        )

    def _build_loss_specs_for_expressions(
        self,
        expressions: List[str | None],
    ) -> Dict[str, tuple]:
        required_expression_terms = self._ordered_expression_terms_for_expressions(
            expressions
        )
        included_loss_terms = set(
            extract_loss_terms(
                expressions=[
                    *(expr for expr in expressions if expr),
                    *[
                        self.expression_terms[name]
                        for name in required_expression_terms
                    ],
                ],
                excluded_terms=list(self.expression_terms.keys()),
            )
        )
        parsed_defined_losses = getattr(
            self,
            "parsed_defined_losses",
            self._parse_loss_specs(self.defined_losses),
        )
        specs = build_loss_specs(
            included_loss_terms.intersection(set(self.registry.param_free_losses))
        )
        specs.update(
            {
                loss_name: parsed_defined_losses.get(loss_name)
                for loss_name in included_loss_terms.intersection(
                    parsed_defined_losses.keys()
                )
            }
        )
        return {k: v for k, v in specs.items() if v is not None}


def extract_variables(expr):
    if not expr:
        return set()
    tree = ast.parse(expr, mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


specific_keys = ["plddt", "ptm", "iptm", "pde"]
intra_chain_keys = [
    "intra_contact_lis",
    "intra_iptm",
    "intra_lis",
    "intra_pae_min",
]
cross_chain_keys = [
    "cross_contact_lis",
    "cross_iptm",
    "cross_lis",
    "cross_pae_min",
]
_VALID_LOSSES = (
    specific_keys
    + intra_chain_keys
    + cross_chain_keys
    + [
        "tmscore",
        "binder_target_distance",
        "hotspot_dsasa",
        "hotspot_buried_ratio",
        "ddg",
        "ipae",
        "ipae_min",
        "ipae_minimal",
        "ipae_max",
        "chain_buried_fraction",
        "chain_buried_fraction_bb",
        "chain_center_distance",
        "single_chain_rog",
        "calculate_buried_contribution_diff",
        "calculate_holes_diff",
        "circle_dual_cyclic",
        "min_frac_dssp",
        "cyclic_symmetry_composite",
        "separation_std",
        "repeat_unit_rmsd_raw",
        "repeat_unit_rmsd_norm",
        "pseudocycle_rotation_raw",
        "pseudocycle_rotation_norm",
        "pseudocycle_rise_raw",
        "pseudocycle_rise_norm",
        "pseudocycle_closure_raw",
        "pseudocycle_closure_norm",
        "point_group_orbit_rmsd_raw",
        "point_group_orbit_rmsd_ratio",
        "point_group_rotation_raw",
        "point_group_symmetry_norm",
        "ananas_rmsd_raw",
        "ananas_rmsd_norm",
        "dihedral_ring_sep_cv_raw",
        "dihedral_pair_sep_cv_raw",
        "dihedral_pair_contact_count_raw",
        "dihedral_pair_contact_count_norm",
        "dihedral_pair_contact_fraction_raw",
        "dihedral_pair_contact_fraction_norm",
        "dihedral_pair_min_distance_raw",
        "dihedral_pair_min_distance_norm",
        "protomer_rmsd_raw",
        "protomer_rmsd_norm",
    ]
)


class LossTerms(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value
