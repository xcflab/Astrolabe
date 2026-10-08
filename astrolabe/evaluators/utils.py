from __future__ import annotations

import dataclasses
import pathlib
import sys
import typing
from dataclasses import dataclass
from typing import List

import numpy as np
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.registry import EvaluatorRegistry, register_evaluator_factory
from astrolabe.score_contributions import (
    filter_score_contributions,
    normalize_score_contributions,
)
from astrolabe.utils.mmcif_fallback import (
    ensure_centered_pdb_fallback,
    load_pose_with_optional_mmcif_pdb_fallback,
)


def _resolve_alphafold2_import_root(data_dir: str | pathlib.Path | None) -> pathlib.Path | None:
    vendor_root = pathlib.Path(__file__).resolve().parents[2] / "vendor" / "alphafold2"
    if not data_dir:
        if (vendor_root / "alphafold" / "common" / "protein.py").exists():
            return vendor_root
        return None
    data_path = pathlib.Path(str(data_dir)).expanduser().resolve()
    candidates = [data_path, data_path.parent]
    for candidate in candidates:
        if (candidate / "alphafold" / "common" / "protein.py").exists():
            if (
                not _alphafold2_supports_initial_prev_pos(candidate)
                and _alphafold2_supports_initial_prev_pos(vendor_root)
            ):
                return vendor_root
            return candidate
    return None


def _alphafold2_supports_initial_prev_pos(import_root: pathlib.Path) -> bool:
    modules_path = import_root / "alphafold" / "model" / "modules.py"
    if not modules_path.exists():
        return (import_root / "alphafold" / "common" / "protein.py").exists()
    return "initial_prev_pos" in modules_path.read_text(encoding="utf-8")


def _ensure_alphafold2_import_path(data_dir: str | pathlib.Path | None) -> None:
    import_root = _resolve_alphafold2_import_root(data_dir)
    if import_root is None:
        return
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


def _merge_conf_with_defaults(
    evaluator_conf,
    evaluator_name: str,
    override_conf,
    *,
    alias: str,
):
    base_conf = getattr(evaluator_conf, evaluator_name, None)
    if isinstance(base_conf, dict):
        base_container = dict(base_conf)
        base_container.pop("_type", None)
        base_conf = OmegaConf.create(base_container)
    override_cfg = None

    if override_conf is not None:
        if isinstance(override_conf, dict):
            override_container: typing.Any = dict(override_conf)
        else:
            try:
                override_container = OmegaConf.to_container(override_conf, resolve=True)
            except Exception as exc:
                raise ValueError(
                    f"Failed to resolve evaluator config for alias {alias!r}: {exc}"
                ) from exc

        if isinstance(override_container, dict):
            override_container.pop("_type", None)
            if override_container:
                override_cfg = OmegaConf.create(override_container)

    configs = [cfg for cfg in (base_conf, override_cfg) if cfg is not None]
    if not configs:
        return None
    if len(configs) == 1:
        return configs[0]
    return OmegaConf.merge(*configs)


def construct_evaluators(
    evaluator_conf,
    raw_model_results_dir,
    oligo_names=None,
    *,
    enabled_aliases: list[str] | None = None,
):
    if isinstance(evaluator_conf, list):
        raise NotImplementedError("Multiple evaluators are not supported yet")
    selected_aliases = (
        {str(a) for a in enabled_aliases} if enabled_aliases is not None else None
    )

    evaluators = {}
    for alias, conf in evaluator_conf.evaluator_terms.items():
        # Allow turning off evaluators via config overlays by setting them to null.
        if conf is None:
            continue
        if selected_aliases is not None and str(alias) not in selected_aliases:
            continue
        name = conf.get("_type") if hasattr(conf, "get") else None
        if not name:
            continue

        # Factories expect a config without the `_type` discriminator.
        term_conf = conf
        if isinstance(conf, dict):
            term_container: typing.Any = dict(conf)
        else:
            try:
                term_container = OmegaConf.to_container(conf, resolve=True)
            except Exception as exc:
                raise ValueError(
                    f"Failed to resolve evaluator config for alias {alias!r}: {exc}"
                ) from exc
        if isinstance(term_container, dict):
            term_container.pop("_type", None)
            term_conf = OmegaConf.create(term_container)

        try:
            factory = EvaluatorRegistry.get(name)
        except KeyError as exc:
            available = ", ".join(EvaluatorRegistry.list_available()) or "<none>"
            raise ValueError(
                f"Unknown evaluator name {name!r}. Available evaluator types: {available}"
            ) from exc

        evaluator = factory(
            term_conf,
            alias=alias,
            raw_model_results_dir=raw_model_results_dir,
            oligo_names=oligo_names,
            evaluator_conf=evaluator_conf,
        )
        setattr(evaluator, "alias", str(alias))
        evaluators[alias] = evaluator
    return evaluators


@register_evaluator_factory("alphafold3")
def _build_alphafold3_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.wrappers.alphafold3 import alphafold3_setup

    af3_conf = _merge_conf_with_defaults(
        evaluator_conf, "alphafold3", term_conf, alias=alias
    )
    if af3_conf is None:
        raise ValueError("AlphaFold3 configuration is missing.")
    model_runner = alphafold3_setup(af3_conf)
    evaluator = AlphaFold3Evaluator(model_runner, raw_model_results_dir / "alphafold3", af3_conf)
    evaluator.pre_eval_hooks_config = getattr(evaluator_conf, "pre_eval_hooks", None)
    return evaluator


@register_evaluator_factory("rf3")
def _build_rf3_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.rf3 import RoseTTAFold3Evaluator
    from astrolabe.evaluators.wrappers.rf3 import rf3_setup

    rf3_conf = _merge_conf_with_defaults(evaluator_conf, "rf3", term_conf, alias=alias)
    if rf3_conf is None:
        raise ValueError("RF3 configuration is missing.")
    model_runner = rf3_setup(rf3_conf)
    return RoseTTAFold3Evaluator(model_runner, raw_model_results_dir / "rf3", rf3_conf)


@register_evaluator_factory("alphafold2")
def _build_alphafold2_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    if oligo_names is None:
        raise ValueError("AlphaFold2 evaluator requires the list of oligomers.")

    af2_conf = _merge_conf_with_defaults(
        evaluator_conf, "alphafold2", term_conf, alias=alias
    )
    af2_kwargs = {}
    if af2_conf is not None:
        af2_kwargs = OmegaConf.to_container(af2_conf, resolve=True)
    _ensure_alphafold2_import_path(af2_kwargs.get("data_dir"))

    from astrolabe.evaluators.alphafold2 import AlphaFold2Evaluator
    from astrolabe.evaluators.wrappers.alphafold import alphafold_setup

    model_runner = alphafold_setup(
        oligo_names=oligo_names,
        **af2_kwargs,
    )
    return AlphaFold2Evaluator(model_runner, raw_model_results_dir / "alphafold2")


def _esmfold_kwargs_from_config(config) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if config is None:
        return kwargs
    if isinstance(config, dict):
        container: typing.Any = dict(config)
    else:
        container = OmegaConf.to_container(config, resolve=True)
    if isinstance(container, dict):
        model_name = container.get("model_name")
        device = container.get("device")
        if model_name is not None:
            kwargs["model_name"] = str(model_name)
        if device is not None:
            kwargs["device"] = str(device)
    return kwargs


def _esmfold2_kwargs_from_config(config) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if config is None:
        return kwargs
    if isinstance(config, dict):
        container: typing.Any = dict(config)
    else:
        container = OmegaConf.to_container(config, resolve=True)
    allowed = {
        "python_bin",
        "esm_repo_path",
        "transformers_repo_path",
        "model_name",
        "cache_dir",
        "device",
        "esmc_precision",
        "num_loops",
        "num_sampling_steps",
        "num_diffusion_samples",
        "seed",
        "lm_dropout",
        "msa_max_depth",
        "msa_column_mask_rate",
        "chunk_size",
        "local_files_only",
        "timeout_seconds",
    }
    if isinstance(container, dict):
        kwargs.update({k: v for k, v in container.items() if k in allowed})
    return kwargs


@register_evaluator_factory("esmfold")
def _build_esmfold_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.esmfold import ESMFoldEvaluator

    esmfold_conf = _merge_conf_with_defaults(
        evaluator_conf, "esmfold", term_conf, alias=alias
    )
    return ESMFoldEvaluator(**_esmfold_kwargs_from_config(esmfold_conf))


@register_evaluator_factory("esmfold2")
def _build_esmfold2_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.esmfold2 import ESMFold2Evaluator

    esmfold2_conf = _merge_conf_with_defaults(
        evaluator_conf, "esmfold2", term_conf, alias=alias
    )
    return ESMFold2Evaluator(
        raw_model_results_dir=raw_model_results_dir / "esmfold2",
        **_esmfold2_kwargs_from_config(esmfold2_conf),
    )


@register_evaluator_factory("chai1")
def _build_chai1_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    chai_conf = _merge_conf_with_defaults(evaluator_conf, "chai1", term_conf, alias=alias)
    if chai_conf is None:
        raise ValueError("Chai1 configuration is missing.")
    return Chai1Evaluator(chai_conf, raw_model_results_dir / "chai1")


@register_evaluator_factory("boltz")
def _build_boltz_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.boltz import BoltzEvaluator
    from astrolabe.evaluators.wrappers.boltz import Boltz1Runner

    boltz_conf = _merge_conf_with_defaults(evaluator_conf, "boltz", term_conf, alias=alias)
    if boltz_conf is None:
        raise ValueError("Boltz configuration is missing.")
    model_runner = Boltz1Runner(boltz_conf)
    return BoltzEvaluator(model_runner, raw_model_results_dir / "boltz")


@register_evaluator_factory("opendde")
def _build_opendde_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.opendde import OpenDDEEvaluator
    from astrolabe.evaluators.wrappers.opendde import OpenDDERunner

    opendde_conf = _merge_conf_with_defaults(
        evaluator_conf, "opendde", term_conf, alias=alias
    )
    if opendde_conf is None:
        raise ValueError("OpenDDE configuration is missing.")
    model_runner = OpenDDERunner.from_config(opendde_conf)
    return OpenDDEEvaluator(model_runner, raw_model_results_dir / "opendde")


@register_evaluator_factory("protenix")
def _build_protenix_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.protenix import ProtenixEvaluator
    from astrolabe.evaluators.wrappers.protenix import ProtenixRunner

    protenix_conf = _merge_conf_with_defaults(
        evaluator_conf, "protenix", term_conf, alias=alias
    )
    if protenix_conf is None:
        raise ValueError("Protenix configuration is missing.")
    raw_use_constraints = protenix_conf.get("use_constraints", True)
    if not isinstance(raw_use_constraints, bool):
        raise ValueError(
            f"Protenix evaluator alias {alias!r} use_constraints must be boolean."
        )
    use_constraints = raw_use_constraints
    if "use_constraints" in protenix_conf:
        del protenix_conf["use_constraints"]

    # If binding_site constraints are requested, Protenix needs the constraint-capable
    # checkpoint + enabled constraint embedders. The default checkpoint ignores the
    # provided pocket/contact features, which makes binding_site appear ineffective.
    from astrolabe.core import global_config as _global_config
    from astrolabe.core.binding_site import has_binding_site_config as _has_binding_site_config

    try:
        raw_constraint = getattr(_global_config, "protenix_constraint", None)
    except RuntimeError:
        raw_constraint = None
    try:
        static_binding_site = getattr(_global_config, "binding_site", None)
    except RuntimeError:
        static_binding_site = None
    try:
        global_dynamic_binding_site_enabled = bool(
            getattr(_global_config, "dynamic_binding_site_enabled", False)
        )
    except RuntimeError:
        global_dynamic_binding_site_enabled = False
    dynamic_binding_site_enabled = global_dynamic_binding_site_enabled or bool(
        protenix_conf.get("dynamic_binding_site", False)
    )
    if "dynamic_binding_site" in protenix_conf:
        del protenix_conf["dynamic_binding_site"]
    constraint_guidance_enabled = _has_binding_site_config(
        static_binding_site
    ) or dynamic_binding_site_enabled or raw_constraint is not None

    if use_constraints and constraint_guidance_enabled:
        # Enable embedders if present in config tree.
        try:
            ce = protenix_conf.model.constraint_embedder
            for key in (
                "pocket_embedder",
                "contact_embedder",
                "substructure_embedder",
                "contact_atom_embedder",
            ):
                if hasattr(ce, key) and hasattr(getattr(ce, key), "enable"):
                    getattr(ce, key).enable = True
        except Exception:
            # If the config tree doesn't match, fail fast: binding_site should not
            # silently become a no-op.
            raise ValueError(
                "constraint guidance is set but Protenix constraint embedders are missing from config."
            )

        # Use the constraint-capable checkpoint (configured at the global Hydra root).
        protenix_conf.model_name = "protenix_base_constraint_v0.5.0"
        protenix_conf.load_checkpoint_path = "${weight_path.protenix_constraint}"
        try:
            resolved = OmegaConf.to_container(protenix_conf, resolve=True)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(
                "constraint guidance is set but failed to resolve `weight_path.protenix_constraint` "
                f"for Protenix: {exc}"
            ) from exc
        ckpt_path = None
        if isinstance(resolved, dict):
            ckpt_path = resolved.get("load_checkpoint_path")
        if not isinstance(ckpt_path, str) or not ckpt_path.strip():
            raise ValueError(
                "constraint guidance is set but resolved Protenix constraint checkpoint path is empty."
            )
        import os as _os

        if not _os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"constraint guidance is set but Protenix constraint checkpoint is missing: {ckpt_path}"
            )

    protenix_conf.dump_dir = raw_model_results_dir / "protenix"
    model_runner = ProtenixRunner(protenix_conf)
    return ProtenixEvaluator(model_runner, use_constraints=use_constraints)


@register_evaluator_factory("rhofold")
def _build_rhofold_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    import torch
    from rhofold.config import rhofold_config
    from rhofold.rhofold import RhoFold

    from astrolabe.evaluators.rhofold import RhoFoldEvaluator

    rhofold_conf = _merge_conf_with_defaults(
        evaluator_conf, "rhofold", term_conf, alias=alias
    )
    if rhofold_conf is None:
        raise ValueError("RhoFold configuration is missing.")

    ckpt = getattr(rhofold_conf, "checkpoint", None)
    if not ckpt:
        raise ValueError("RhoFold checkpoint is missing (`evaluator.rhofold.checkpoint`).")

    model_runner = RhoFold(rhofold_config)
    state = torch.load(str(ckpt), map_location=torch.device("cpu"))
    model_runner.load_state_dict(state["model"])
    model_runner.eval()

    return RhoFoldEvaluator(
        model_runner,
        raw_model_results_dir / "rhofold",
        rhofold_conf,
    )


@register_evaluator_factory("dummy")
def _build_dummy_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.dummy import DummyEvaluator

    kwargs: dict[str, typing.Any] = {}
    if term_conf is not None:
        try:
            container = OmegaConf.to_container(term_conf, resolve=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to resolve dummy evaluator config for alias {alias!r}: {exc}"
            ) from exc
        if isinstance(container, dict):
            kwargs.update(container)
    if kwargs.get("write_structure") and kwargs.get("structure_dir") is None:
        kwargs["structure_dir"] = raw_model_results_dir / "dummy"
    return DummyEvaluator(**kwargs)


@register_evaluator_factory("rosetta_crystal")
def _build_rosetta_crystal_evaluator(
    term_conf,
    *,
    alias,
    raw_model_results_dir,
    oligo_names,
    evaluator_conf,
):
    from astrolabe.evaluators.rosetta_crystal import RosettaCrystalEvaluator

    try:
        kwargs = OmegaConf.to_container(term_conf, resolve=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to resolve rosetta_crystal evaluator config for alias {alias!r}: {exc}"
        ) from exc
    if not isinstance(kwargs, dict):
        raise ValueError("rosetta_crystal evaluator configuration must be a mapping")
    return RosettaCrystalEvaluator(
        **kwargs,
        raw_model_results_dir=raw_model_results_dir / "rosetta_crystal",
    )


@dataclasses.dataclass
class StandardScores:
    names: list[str]
    # Oligomer-keyed metrics. Values are always lists after __post_init__ normalization.
    plddt: dict[str, list[np.ndarray] | list[float]]
    complex_paths: dict[str, pathlib.Path]
    # Protomer-keyed per-residue metrics (e.g., per-chain pLDDT).
    # Values are always lists after __post_init__ normalization.
    per_proto_scores: dict[str, list[np.ndarray]] | None = None
    # Protomer-keyed, named per-residue metrics. Example:
    # {"A": {"plddt": [array], "sap": [array], "buried_unsat": [array]}}.
    per_proto_score_terms: dict[str, dict[str, list[np.ndarray]]] | None = None
    ptm: dict[str, list[float]] | None = None
    pae: dict[str, list[np.ndarray]] | None = None
    pde: dict[str, list[np.ndarray]] | None = None
    iptm: dict[str, list[float]] | None = None
    # Oligomer-keyed chain-pair metrics, e.g. {"AB": {"chain_pair_iptm": [mat, ...]}}.
    # Values are always lists after __post_init__ normalization.
    chain_pair_scores: dict[str, dict[str, list[np.ndarray]]] | None = None
    complex_poses: dict[str, typing.Any] | None = None
    score_contributions: typing.Any | None = None

    @staticmethod
    def _wrap_dict_values_in_list(
        data: dict[str, typing.Any] | None,
    ) -> dict[str, list[typing.Any]] | None:
        if data is None:
            return None
        return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}

    @staticmethod
    def _wrap_nested_dict_values_in_list(
        data: dict[str, dict[str, typing.Any]] | dict[str, typing.Any] | None,
        *,
        list_metric_key: str,
    ) -> dict[str, dict[str, list[typing.Any]]] | None:
        if data is None:
            return None
        out: dict[str, dict[str, list[typing.Any]]] = {}
        for outer_key, outer_val in data.items():
            if outer_val is None:
                continue
            if isinstance(outer_val, dict):
                out[outer_key] = {
                    k: (v if isinstance(v, list) else [v])
                    for k, v in outer_val.items()
                }
                continue
            if isinstance(outer_val, list):
                out[outer_key] = {list_metric_key: outer_val}
                continue
            out[outer_key] = {list_metric_key: [outer_val]}
        return out

    @staticmethod
    def _maybe_scale_unit_interval_list(
        values: list[typing.Any],
        *,
        scale_factor: float = 100.0,
        max_unscaled_threshold: float = 2.0,
    ) -> list[typing.Any]:
        """Best-effort normalize [0,100]-style scores to [0,1].

        Many backends report pLDDT as either [0,1] or [0,100]. Downstream losses
        assume [0,1], so we normalise values that look like percentages.
        """
        if not values:
            return values

        scaled: list[typing.Any] = []
        for v in values:
            if v is None:
                scaled.append(v)
                continue

            try:
                if isinstance(v, np.ndarray):
                    max_val = float(np.nanmax(v))
                else:
                    max_val = float(np.nanmax(np.asarray(v, dtype=float)))
            except Exception:
                scaled.append(v)
                continue

            if max_val <= max_unscaled_threshold:
                scaled.append(v)
                continue

            if isinstance(v, np.ndarray):
                scaled.append(np.asarray(v, dtype=float) / float(scale_factor))
            elif isinstance(v, (int, float, np.number)):
                scaled.append(float(v) / float(scale_factor))
            else:
                scaled.append(np.asarray(v, dtype=float) / float(scale_factor))
        return scaled

    def __post_init__(self) -> None:
        # Normalise common fields so downstream code can treat them as list-valued.
        self.plddt = typing.cast(
            dict[str, list[np.ndarray] | list[float]],
            self._wrap_dict_values_in_list(self.plddt),
        )
        self.plddt = typing.cast(
            dict[str, list[np.ndarray] | list[float]],
            {
                k: self._maybe_scale_unit_interval_list(v)
                for k, v in (self.plddt or {}).items()
            },
        )
        self.ptm = typing.cast(
            dict[str, list[float]] | None,
            self._wrap_dict_values_in_list(self.ptm),
        )
        self.iptm = typing.cast(
            dict[str, list[float]] | None,
            self._wrap_dict_values_in_list(self.iptm),
        )
        self.pae = typing.cast(
            dict[str, list[np.ndarray]] | None,
            self._wrap_dict_values_in_list(self.pae),
        )
        self.pde = typing.cast(
            dict[str, list[np.ndarray]] | None,
            self._wrap_dict_values_in_list(self.pde),
        )
        self.per_proto_scores = typing.cast(
            dict[str, list[np.ndarray]] | None,
            self._wrap_dict_values_in_list(self.per_proto_scores),
        )
        if self.per_proto_scores is not None:
            self.per_proto_scores = typing.cast(
                dict[str, list[np.ndarray]],
                {
                    k: self._maybe_scale_unit_interval_list(v)
                    for k, v in (self.per_proto_scores or {}).items()
                },
            )
        self.per_proto_score_terms = typing.cast(
            dict[str, dict[str, list[np.ndarray]]] | None,
            self._wrap_nested_dict_values_in_list(
                self.per_proto_score_terms,
                list_metric_key="score",
            ),
        )
        if self.per_proto_score_terms is not None:
            normalized_terms: dict[str, dict[str, list[np.ndarray]]] = {}
            for proto, metric_map in self.per_proto_score_terms.items():
                normalized_terms[proto] = {}
                for metric_name, values in metric_map.items():
                    if str(metric_name) == "plddt":
                        values = self._maybe_scale_unit_interval_list(values)
                    normalized_terms[proto][metric_name] = values
            self.per_proto_score_terms = normalized_terms
        self.chain_pair_scores = typing.cast(
            dict[str, dict[str, list[np.ndarray]]] | None,
            self._wrap_nested_dict_values_in_list(
                self.chain_pair_scores,
                list_metric_key="chain_pair_iptm",
            ),
        )
        if self.score_contributions is not None:
            self.score_contributions = normalize_score_contributions(
                self.score_contributions
            )

    def __getitem__(self, item):
        return getattr(self, item)

    def __hash__(self):
        return id(self)

    def filter_by_oligomer(self, oligomers: List[str]) -> StandardScores:
        """Filter scores by specified oligomers.

        Notes:
            - Most fields in StandardScores are keyed by oligomer name (e.g. "AB").
            - `per_proto_scores` is keyed by protomer name (e.g. "A", "B") and must
              be filtered by the set of protomers present in the selected oligomers.
        """
        allowed_oligos = set(oligomers or [])
        filtered_names = [name for name in self.names if name in allowed_oligos]
        allowed_protos = set("".join(filtered_names))

        filtered_scores: dict[str, typing.Any] = {"names": filtered_names}

        for key in self.__dataclass_fields__.keys() - {"names"}:
            value = getattr(self, key, None)
            if value is None:
                filtered_scores[key] = None
                continue
            if key == "score_contributions":
                filtered_scores[key] = filter_score_contributions(
                    value,
                    allowed_oligos=allowed_oligos,
                    allowed_protomers=allowed_protos,
                )
                continue

            if key in {"per_proto_scores", "per_proto_score_terms"} and isinstance(value, dict):
                filtered_scores[key] = {
                    proto: scores
                    for proto, scores in value.items()
                    if proto in allowed_protos
                }
                continue

            if key == "complex_poses" and hasattr(value, "poses"):
                filtered_poses = {
                    oligo: pose
                    for oligo, pose in value.poses.items()
                    if oligo in allowed_oligos
                }
                if dataclasses.is_dataclass(value):
                    filtered_scores[key] = dataclasses.replace(
                        value,
                        poses=filtered_poses,
                    )
                else:
                    filtered_scores[key] = type(value)(poses=filtered_poses)
                continue

            if isinstance(value, dict):
                filtered_scores[key] = {
                    k: v for k, v in value.items() if k in allowed_oligos
                }
            elif isinstance(value, list):
                filtered_scores[key] = [
                    v for v, name in zip(value, self.names) if name in allowed_oligos
                ]
            else:
                filtered_scores[key] = value

        return StandardScores(**filtered_scores)

    def clone(self) -> StandardScores:
        """Create a new StandardScores instance that shares underlying data."""
        return dataclasses.replace(self)


# noinspection PyUnresolvedReferences
@dataclass
class ComplexPoses:
    poses: dict[str, "pyrosetta.Pose"]

    @classmethod
    def construct_complex_poses(cls, eval_result):
        import pyrosetta

        poses = {}
        try:
            extra_residue_params = sorted(
                pathlib.Path(global_config.out_dir).glob("raw_model_results/*.params")
            )
        except RuntimeError:
            extra_residue_params = []
        try:
            allow_mmcif_pdb_fallback = bool(global_config.allow_mmcif_pdb_fallback)
        except RuntimeError:
            allow_mmcif_pdb_fallback = False
        for oligo_name, complex_path in eval_result.complex_paths.items():
            complex_path_str = str(complex_path)
            pose = load_pose_with_optional_mmcif_pdb_fallback(
                pyrosetta,
                complex_path_str,
                allow_mmcif_pdb_fallback=allow_mmcif_pdb_fallback,
                fallback_builder=ensure_centered_pdb_fallback,
                extra_residue_params=extra_residue_params,
            )
            poses[oligo_name] = pose
        return cls(poses)
