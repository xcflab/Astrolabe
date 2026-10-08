from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Mapping as ABCMapping, MutableMapping
from contextlib import nullcontext
from copy import deepcopy
from os.path import join as opjoin
from pathlib import Path
from typing import Mapping, Any

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import convert as _pdbx_convert
from torch import distributed as dist

from astrolabe.core import global_config
from astrolabe.core.folding_input import DnaChain, Ligand, ProteinChain, RnaChain
from astrolabe.evaluators.x_masking import preserve_unknown_residue_sequence

# Prefer vendored Protenix code under `vendor/protenix` over any globally installed
# vendored checkout, to match repo policy and keep behavior stable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_PROTENIX = _REPO_ROOT / "vendor" / "protenix"
if _VENDORED_PROTENIX.exists():
    sys.path.insert(0, str(_VENDORED_PROTENIX))

# Protenix expects Biotite's pdbx.convert to expose a bond-type mapping constant
# with this name, but some Biotite builds ship no bond constants at all.
if not hasattr(_pdbx_convert, "PDBX_BOND_TYPE_ID_TO_TYPE"):
    _pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE = {}

_PROTENIX_IMPORTED = False


def _lazy_import_protenix() -> None:
    """Import Protenix modules only when the Protenix evaluator is enabled.

    This keeps `import astrolabe...wrappers.protenix` side-effect free: importing
    the module should not trigger network downloads or require Protenix CCD caches
    to exist on disk until Protenix is actually used.
    """

    global _PROTENIX_IMPORTED
    if _PROTENIX_IMPORTED:
        return

    if not _VENDORED_PROTENIX.exists():
        raise FileNotFoundError(
            "Vendored Protenix not found under vendor/protenix; "
            "ensure git submodules are initialized."
        )

    _ensure_protenix_ccd_cache_files_exist()
    # Protenix v2 defaults to compiling a fast CUDA layernorm extension at import
    # time. The shared GH env may not have a new enough GCC for that JIT path, so
    # default to Protenix's torch/OpenFold LayerNorm unless the user opts in.
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    try:
        try:
            from protenix.data.pipeline.data_pipeline import DataPipeline
            from protenix.data.inference.json_to_feature import SampleDictToFeatures
            from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
            from protenix.data.template.template_featurizer import (
                InferenceTemplateFeaturizer,
            )
            from protenix.data.template.template_parser import TemplateParser
            from protenix.data.template.template_utils import TemplateHitFeaturizer
        except ImportError:
            from protenix.data.data_pipeline import DataPipeline
            from protenix.data.json_to_feature import SampleDictToFeatures
            from protenix.data.msa_featurizer import InferenceMSAFeaturizer

            InferenceTemplateFeaturizer = None
            TemplateParser = None
            TemplateHitFeaturizer = None
        from protenix.data.utils import (
            make_dummy_feature,
            data_type_transform,
            save_structure_cif,
        )
        from protenix.model.protenix import Protenix
        from protenix.utils.distributed import DIST_WRAPPER
        from protenix.utils.file_io import save_json
        from protenix.utils.torch_utils import round_values, dict_to_tensor, to_device
    except Exception as exc:  # pragma: no cover - optional dependency import path
        raise RuntimeError(
            "Failed to import vendored Protenix. "
            "Ensure all Protenix runtime dependencies are available."
        ) from exc

    globals().update(
        {
            "DataPipeline": DataPipeline,
            "SampleDictToFeatures": SampleDictToFeatures,
            "InferenceMSAFeaturizer": InferenceMSAFeaturizer,
            "InferenceTemplateFeaturizer": InferenceTemplateFeaturizer,
            "TemplateParser": TemplateParser,
            "TemplateHitFeaturizer": TemplateHitFeaturizer,
            "make_dummy_feature": make_dummy_feature,
            "data_type_transform": data_type_transform,
            "save_structure_cif": save_structure_cif,
            "Protenix": Protenix,
            "DIST_WRAPPER": DIST_WRAPPER,
            "save_json": save_json,
            "round_values": round_values,
            "dict_to_tensor": dict_to_tensor,
            "to_device": to_device,
        }
    )

    _PROTENIX_IMPORTED = True


def _resolve_path(p: str | os.PathLike[str] | None) -> str | None:
    if not p:
        return None
    s = os.fspath(p)
    s = os.path.expanduser(s)
    s = os.path.expandvars(s)
    return os.path.abspath(s)


def _ensure_protenix_ccd_cache_files_exist() -> None:
    """
    Protenix imports `protenix.data.ccd` at module import time and resolves its
    CCD files from ``PROTENIX_ROOT_DIR/common``. Keep that root writable and
    link it to the explicitly configured GH assets when they are available.
    """
    try:
        from protenix.utils.distributed import DIST_WRAPPER as _dist_wrapper
    except Exception:
        class _FallbackDistWrapper:
            rank = 0
            world_size = 1
            local_rank = 0

        _dist_wrapper = _FallbackDistWrapper()

    try:
        from protenix.web_service.dependency_url import URL as _dep_urls
    except Exception:
        _dep_urls = {}

    protenix_root = Path(
        os.environ.get(
            "PROTENIX_ROOT_DIR",
            str(Path.home() / ".gh_envs" / "repos" / "Protenix"),
        )
    )
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(protenix_root))
    release_data = protenix_root / "release_data"
    common_data = protenix_root / "common"
    release_data.mkdir(parents=True, exist_ok=True)
    src_cif = Path(
        os.environ.get(
            "GH_PROTENIX_CCD",
            str(release_data / "components.v20240608.cif"),
        )
    )
    src_pkl = Path(
        os.environ.get(
            "GH_PROTENIX_CCD_RDKIT",
            str(release_data / "components.v20240608.cif.rdkit_mol.pkl"),
        )
    )
    common_cif = common_data / "components.cif"
    common_pkl = common_data / "components.cif.rdkit_mol.pkl"

    def _download_to(url: str, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
        urllib.request.urlretrieve(url, str(tmp))  # noqa: S310
        os.replace(str(tmp), str(dst_path))

    def _ensure_one(key: str, src_path: Path) -> None:
        if src_path.exists():
            return
        url = _dep_urls.get(key)
        if not url:
            return
        if getattr(_dist_wrapper, "rank", 0) == 0:
            logging.info("Downloading Protenix asset %s -> %s", key, str(src_path))
            _download_to(url, src_path)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    _ensure_one("ccd_components_file", src_cif)
    _ensure_one("ccd_components_rdkit_mol_file", src_pkl)

    def _link(dst: Path, src: Path) -> None:
        if dst.is_symlink() and (not dst.exists()):
            try:
                dst.unlink()
            except Exception:
                return
        if dst.exists() or (not src.exists()):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
        except Exception:
            try:
                # Fallback for filesystems that disallow symlinks.
                import shutil

                shutil.copy2(src, dst)
            except Exception:
                return

    _link(common_cif, src_cif)
    _link(common_pkl, src_pkl)


def _patch_protenix_cyclic_relative_position_encoding() -> None:
    from protenix.model.modules.embedders import RelativePositionEncoding

    if getattr(RelativePositionEncoding, "_GH_CYCLIC_RELATIVE_PATCHES", False):
        return

    def _generate_relp(self, input_feature_dict: dict[str, Any]) -> dict[str, Any]:
        """asym_id, residue_index, entity_id, token_index, sym_id"""
        with torch.no_grad():
            asym_id = input_feature_dict["asym_id"]
            residue_index = input_feature_dict["residue_index"]
            entity_id = input_feature_dict["entity_id"]
            token_index = input_feature_dict["token_index"]
            sym_id = input_feature_dict["sym_id"]

            b_same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).long()
            b_same_residue = (
                residue_index[..., :, None] == residue_index[..., None, :]
            ).long()
            b_same_entity = (entity_id[..., :, None] == entity_id[..., None, :]).long()
            d_residue = torch.clip(
                input=residue_index[..., :, None]
                - residue_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain + (1 - b_same_chain) * (2 * self.r_max + 1)
            cyclic_relative_patches = input_feature_dict.get("cyclic_relative_patches")
            if cyclic_relative_patches is not None and cyclic_relative_patches.numel():
                cyclic_relative_patches = cyclic_relative_patches.to(
                    device=residue_index.device, dtype=torch.long
                ).reshape(-1, 4)
                for asym_value, residue_i, residue_j, offset in cyclic_relative_patches:
                    source_mask = (asym_id == asym_value) & (
                        residue_index == residue_i
                    )
                    target_mask = (asym_id == asym_value) & (
                        residue_index == residue_j
                    )
                    patch_mask = source_mask[..., :, None] & target_mask[..., None, :]
                    patch_value = torch.clamp(
                        offset + self.r_max, min=0, max=2 * self.r_max
                    )
                    d_residue = torch.where(patch_mask, patch_value, d_residue)
            a_rel_pos = torch.nn.functional.one_hot(d_residue, 2 * (self.r_max + 1))
            d_token = torch.clip(
                input=token_index[..., :, None]
                - token_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain * b_same_residue + (
                1 - b_same_chain * b_same_residue
            ) * (2 * self.r_max + 1)
            a_rel_token = torch.nn.functional.one_hot(d_token, 2 * (self.r_max + 1))
            d_chain = torch.clip(
                input=sym_id[..., :, None] - sym_id[..., None, :] + self.s_max,
                min=0,
                max=2 * self.s_max,
            ) * b_same_entity + (1 - b_same_entity) * (2 * self.s_max + 1)
            a_rel_chain = torch.nn.functional.one_hot(d_chain, 2 * (self.s_max + 1))

            relp = torch.cat(
                [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                dim=-1,
            ).float()
            input_feature_dict["relp"] = relp
        return input_feature_dict

    RelativePositionEncoding.generate_relp = _generate_relp
    RelativePositionEncoding._GH_CYCLIC_RELATIVE_PATCHES = True


def _patch_protenix_fast_truncnorm_init() -> None:
    # Protenix imports a SciPy-based truncnorm sampler for weight init, which is
    # extremely slow for large tensors. Patch it to use torch's trunc_normal_.
    try:
        import protenix.model.modules.primitives as _ptx_prims
        try:
            import protenix.openfold_local.model.primitives as _extra_prims
        except Exception:
            import protenix.model.triangular.layers as _extra_prims
    except Exception:
        return

    if getattr(_ptx_prims, "_GH_FAST_TRUNCNORM_INIT", False):
        return

    std_trunc = 0.8796256610342398  # std of N(0,1) truncated to [-2,2]

    def _fast_trunc_normal_init_(weights, scale: float = 1.0, fan: str = "fan_in"):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        if fan == "fan_in":
            f = fan_in
        elif fan == "fan_out":
            f = fan_out
        elif fan == "fan_avg":
            f = (fan_in + fan_out) / 2
        else:
            raise ValueError("Invalid fan option")

        scale_eff = scale / max(1, f)
        std = (scale_eff**0.5) / std_trunc
        with torch.no_grad():
            torch.nn.init.trunc_normal_(
                weights, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
            )

    _extra_prims.trunc_normal_init_ = _fast_trunc_normal_init_
    _ptx_prims.trunc_normal_init_ = _fast_trunc_normal_init_
    _ptx_prims._GH_FAST_TRUNCNORM_INIT = True


def _as_plain_mapping(configs: Any) -> dict[str, Any]:
    if configs is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(configs):
            value = OmegaConf.to_container(configs, resolve=True)
            return value if isinstance(value, dict) else {}
    except Exception:
        pass
    if hasattr(configs, "to_dict"):
        value = configs.to_dict()
        return value if isinstance(value, dict) else {}
    if isinstance(configs, ABCMapping):
        return dict(configs)
    return {
        key: getattr(configs, key)
        for key in dir(configs)
        if not key.startswith("_") and not callable(getattr(configs, key))
    }


def _is_mapping_like(value: Any) -> bool:
    return isinstance(value, ABCMapping) or (
        hasattr(value, "items") and hasattr(value, "__getitem__")
    )


def _configdict_safe_value(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, ABCMapping):
        return {key: _configdict_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_configdict_safe_value(item) for item in value)
    if isinstance(value, list):
        return [_configdict_safe_value(item) for item in value]
    return value


def _deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        value = _configdict_safe_value(value)
        current = dst.get(key) if hasattr(dst, "get") else None
        if _is_mapping_like(value) and _is_mapping_like(current):
            _deep_update(current, value)
        else:
            dst[key] = value


def _select_protenix_model_name(raw: Mapping[str, Any]) -> str:
    model_name = raw.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name
    ckpt = raw.get("load_checkpoint_path")
    if isinstance(ckpt, str) and ckpt.strip():
        return Path(os.path.expandvars(os.path.expanduser(ckpt))).stem
    return "protenix-v2"


def _normalize_protenix_configs(configs: Any) -> Any:
    _lazy_import_protenix()
    try:
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from ml_collections import ConfigDict
        from protenix.config.config import parse_configs
    except Exception:
        return configs

    raw = _as_plain_mapping(configs)
    model_name = _select_protenix_model_name(raw)
    defaults = {**configs_base, **{"data": data_configs}, **inference_configs}
    defaults["model_name"] = model_name
    normalized = parse_configs(configs=defaults, fill_required_with_null=True)
    if model_name in model_configs:
        _deep_update(normalized, ConfigDict(model_configs[model_name]).to_dict())

    runtime_overrides: dict[str, Any] = {}
    for key in (
        "dump_dir",
        "need_atom_confidence",
        "sorted_by_ranking_score",
        "input_json_path",
        "load_checkpoint_path",
        "load_checkpoint_dir",
        "load_strict",
        "seed",
        "seeds",
        "num_workers",
        "use_msa",
        "dtype",
        "triangle_multiplicative",
        "triangle_attention",
        "enable_tf32",
        "enable_efficient_fusion",
        "enable_diffusion_shared_vars_cache",
        "msa_pair_as_unpair",
        "use_template",
        "use_rna_msa",
        "use_seeds_in_json",
        "skip_amp",
        "use_deepspeed_evo_attention",
        "infer_setting",
        "data",
    ):
        if key in raw:
            runtime_overrides[key] = raw[key]

    if "sample_diffusion" in raw and isinstance(raw["sample_diffusion"], ABCMapping):
        runtime_overrides["sample_diffusion"] = {
            k: v
            for k, v in raw["sample_diffusion"].items()
            if k
            in {
                "N_sample",
                "N_sample_mini_rollout",
                "N_step",
                "N_step_mini_rollout",
                "guidance",
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            }
        }
    if "model" in raw and isinstance(raw["model"], ABCMapping):
        model_overrides: dict[str, Any] = {}
        if "N_cycle" in raw["model"]:
            model_overrides["N_cycle"] = raw["model"]["N_cycle"]
        constraint_embedder = raw["model"].get("constraint_embedder")
        if isinstance(constraint_embedder, ABCMapping):
            model_overrides["constraint_embedder"] = constraint_embedder
        if model_overrides:
            runtime_overrides["model"] = model_overrides

    runtime_overrides["model_name"] = model_name
    _deep_update(normalized, runtime_overrides)
    return normalized


def _get_nested_config_bool(configs: Any, path: str, default: bool) -> bool:
    cur = configs
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, ABCMapping):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    if cur is None:
        return default
    return bool(cur)


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    _lazy_import_protenix()
    # Remove atom_coordinate
    full_confidence_dict.pop("atom_coordinate")
    # Remove atom_is_polymer
    full_confidence_dict.pop("atom_is_polymer")
    # Keep two decimal places
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


def protenix_from_dict_to_feature(
    single_sample_dict: Mapping[str, Any],
    configs: Any | None = None,
) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
    """
    Processes a single sample from the input JSON to generate features and statistics.

    Args:
        single_sample_dict: A dictionary containing the sample data.

    Returns:
        A tuple containing:
            - A dictionary of features.
            - An AtomArray object.
            - A dictionary of time tracking statistics.
    """
    _lazy_import_protenix()
    # general features
    t0 = time.time()
    sample_kwargs: dict[str, Any] = {}
    if "extract_features_for_tfg" in inspect.signature(SampleDictToFeatures).parameters:
        sample_kwargs["extract_features_for_tfg"] = _get_nested_config_bool(
            configs, "sample_diffusion.guidance.enable", False
        )
    sample2feat = SampleDictToFeatures(
        single_sample_dict,
        **sample_kwargs,
    )
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    entity_poly_type = sample2feat.entity_poly_type
    t1 = time.time()

    msa_features = {}
    if _get_nested_config_bool(configs, "use_msa", True):
        msa_params = inspect.signature(
            InferenceMSAFeaturizer.make_msa_feature
        ).parameters
        if "entity_to_asym_id" in msa_params:
            entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(
                atom_array
            )
            msa_features = InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict.get("sequences", []),
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
        else:
            msa_features = InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict.get("sequences", []),
                atom_array=atom_array,
                msa_pair_as_unpair=_get_nested_config_bool(
                    configs, "msa_pair_as_unpair", True
                ),
                use_rna_msa=_get_nested_config_bool(configs, "use_rna_msa", False),
            )

    use_template = _get_nested_config_bool(configs, "use_template", False)
    template_paths = [
        protein["proteinChain"]["templatesPath"]
        for protein in single_sample_dict.get("sequences", [])
        if "proteinChain" in protein
        and protein["proteinChain"].get("templatesPath")
    ]
    online_template_featurizer = None
    if use_template and template_paths:
        online_template_featurizer = _make_inline_template_featurizer(
            configs, template_paths=template_paths
        )

    template_features = {}
    if InferenceTemplateFeaturizer is not None:
        template_features = InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=single_sample_dict.get("sequences", []),
            atom_array=atom_array,
            use_template=use_template,
            online_template_featurizer=online_template_featurizer,
        )
    elif use_template and template_paths:
        raise RuntimeError(
            "Inline Protenix templates require InferenceTemplateFeaturizer, "
            "which is unavailable in the vendored Protenix installation."
        )
    if use_template and template_paths and not np.any(
        template_features.get("template_atom_mask", ())
    ):
        raise RuntimeError(
            "Inline Protenix templates produced no atom-masked template features; "
            "see the preceding template processing error."
        )

    # Make dummy features for not implemented features
    dummy_feats = []
    if not template_features:
        dummy_feats.append("template")
    else:
        template_features = dict_to_tensor(template_features)
        features_dict.update(template_features)
    if not msa_features:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_tensor(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(
        features_dict=features_dict,
        dummy_feats=dummy_feats,
    )

    # Transform to right data type
    feat = data_type_transform(feat_or_label_dict=features_dict)

    t2 = time.time()

    data = {}
    data["input_feature_dict"] = feat

    # Add dimension related items
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]
    N_msa = feat["msa"].shape[0]

    stats = {}
    for mol_type in ["ligand", "protein", "dna", "rna"]:
        mol_type_mask = feat[f"is_{mol_type}"].bool()
        stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
        stats[f"{mol_type}/token"] = len(
            torch.unique(feat["atom_to_token_idx"][mol_type_mask])
        )

    N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
    data.update(
        {
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        }
    )

    def formatted_key(key):
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        else:
            pass
        return f"N_{type_}_{unit}"

    data.update(
        {
            formatted_key(k): torch.tensor([stats[k]])
            for k in [
                "protein/atom",
                "ligand/atom",
                "dna/atom",
                "rna/atom",
                "protein/token",
                "ligand/token",
                "dna/token",
                "rna/token",
            ]
        }
    )
    data.update({"entity_poly_type": entity_poly_type})
    t3 = time.time()
    time_tracker = {
        "crop": t1 - t0,
        "featurizer": t2 - t1,
        "added_feature": t3 - t2,
    }

    return data, atom_array, time_tracker


def _msa_dir_for_protenix(*, cplx: str, entity_id: int) -> Path:
    return (
        Path(global_config.out_dir)
        / "sequences"
        / "protenix_msa"
        / str(cplx)
        / str(entity_id)
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _minimal_a3m(sequence: str, header: str = "query") -> str:
    return f">{header}\n{sequence}\n"


def _template_dir_for_protenix(*, cplx: str) -> Path:
    return Path(global_config.out_dir) / "sequences" / "protenix_templates" / str(cplx)


def _get_nested_config_value(configs: Any, path: str, default: Any = None) -> Any:
    cur = configs
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, ABCMapping):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def _inline_template_pdb_id(mmcif: str) -> str:
    match = re.search(r"(?m)^_entry\.id\s+['\"]?([A-Za-z0-9]{4})", mmcif)
    if match is None:
        raise ValueError(
            "Inline AF3 template mmCIF must define a four-character _entry.id "
            "for Protenix local template featurization."
        )
    return match.group(1).lower()


def _materialize_inline_templates(
    *, chain: ProteinChain, cplx: str, entity_id: int
) -> str | None:
    templates = getattr(chain, "templates", None) or ()
    if not templates:
        return None
    _lazy_import_protenix()
    if TemplateParser is None:
        raise RuntimeError(
            "Inline Protenix templates require TemplateParser, which is unavailable "
            "in the vendored Protenix installation."
        )

    root = _template_dir_for_protenix(cplx=cplx)
    mmcif_dir = root / "mmcif"
    hits: list[str] = []
    release_dates_path = root / "release_date_cache.json"
    release_dates = (
        json.loads(release_dates_path.read_text(encoding="utf-8"))
        if release_dates_path.exists()
        else {}
    )
    if not isinstance(release_dates, dict):
        raise ValueError(f"Invalid Protenix template release-date cache: {release_dates_path}")

    for template in templates:
        mmcif = getattr(template, "mmcif", None)
        if not isinstance(mmcif, str) or not mmcif.strip():
            raise ValueError("Inline AF3 template is missing its mmCIF content.")
        pdb_id = _inline_template_pdb_id(mmcif)
        parsed = TemplateParser.parse(file_id=pdb_id, mmcif_string=mmcif)
        if parsed.mmcif_object is None:
            raise ValueError(
                f"Failed to parse inline AF3 template {pdb_id}: {parsed.errors!r}"
            )
        release_date = parsed.mmcif_object.header.get("release_date")
        if not release_date:
            raise ValueError(
                f"Inline AF3 template {pdb_id} has no mmCIF release date."
            )
        template_map = getattr(template, "query_to_template_map", {}) or {}
        mapped_indices = [int(index) for index in template_map.values()]
        candidates = [
            (chain_id, sequence)
            for chain_id, sequence in parsed.mmcif_object.chain_to_seqres.items()
            if not mapped_indices or max(mapped_indices) < len(sequence)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Inline AF3 template {pdb_id} must resolve to exactly one chain "
                f"compatible with templateIndices; found {[chain_id for chain_id, _ in candidates]!r}."
            )
        chain_id, template_sequence = candidates[0]
        cif_path = mmcif_dir / f"{pdb_id}.cif"
        if cif_path.exists() and cif_path.read_text(encoding="utf-8") != mmcif:
            raise ValueError(
                f"Conflicting inline mmCIF content for Protenix template {pdb_id} in {cplx!r}."
            )
        _write_text_atomic(cif_path, mmcif)
        release_dates[pdb_id] = {"release_date": release_date}
        hits.append(
            f">{pdb_id}_{chain_id}/1-{len(template_sequence)} mol:protein "
            f"length:{len(template_sequence)}\n{template_sequence}\n"
        )

    _write_text_atomic(release_dates_path, json.dumps(release_dates, sort_keys=True))
    _write_text_atomic(root / "obsolete_to_successor.json", "{}")
    hit_path = root / str(entity_id) / "inline_templates.a3m"
    _write_text_atomic(hit_path, "".join(hits))
    return str(hit_path)


def _make_inline_template_featurizer(configs: Any, *, template_paths: list[str]):
    if TemplateHitFeaturizer is None:
        raise RuntimeError(
            "Inline Protenix templates require TemplateHitFeaturizer, which is unavailable "
            "in the vendored Protenix installation."
        )
    roots = {Path(path).parent.parent for path in template_paths}
    if len(roots) != 1:
        raise ValueError(
            f"Inline Protenix template paths must share one run directory, got {template_paths!r}."
        )
    template_root = roots.pop()
    kalign_path = _get_nested_config_value(
        configs, "data.template.kalign_binary_path"
    )
    if not isinstance(kalign_path, str) or not os.path.isfile(kalign_path) or not os.access(kalign_path, os.X_OK):
        raise RuntimeError(
            "Inline Protenix templates require an executable "
            f"data.template.kalign_binary_path, got {kalign_path!r}."
        )
    release_dates_path = template_root / "release_date_cache.json"
    obsolete_pdbs_path = template_root / "obsolete_to_successor.json"
    for path in (release_dates_path, obsolete_pdbs_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing materialized Protenix template asset: {path}")
    return TemplateHitFeaturizer(
        mmcif_dir=str(template_root / "mmcif"),
        max_hits=4,
        kalign_binary_path=kalign_path,
        max_template_date="9999-12-31",
        release_dates_path=str(release_dates_path),
        obsolete_pdbs_path=str(obsolete_pdbs_path),
        strict_error_check=True,
        fetch_remote=False,
    )


def from_af3_to_protenix(
    cplx_input,
    cplx,
    *,
    use_constraints: bool = True,
    use_templates: bool = False,
):
    protenix_inputs = {"name": cplx, "sequences": []}
    chain_id_to_entities: dict[str, list[int]] = defaultdict(list)

    def _single_entity_id(chain_id: str, *, purpose: str) -> int:
        entities = chain_id_to_entities.get(str(chain_id)) or []
        if len(entities) != 1:
            raise ValueError(
                f"Protenix {purpose} requires chain {chain_id!r} to map to exactly one entity, "
                f"got {len(entities)} entities {entities!r} in {cplx!r}."
            )
        return int(entities[0])

    for chain in cplx_input.chains:
        chain_input = {
            "count": 1,
        }
        if isinstance(chain, ProteinChain):
            protein_sequence = preserve_unknown_residue_sequence(
                chain.sequence, evaluator="protenix"
            )
            entity_id = len(protenix_inputs["sequences"]) + 1
            chain_id_to_entities[str(getattr(chain, "id"))].append(entity_id)
            chain_input["sequence"] = protein_sequence
            chain_input["modifications"] = [
                {"ptmType": f"CCD_{mod}", "ptmPosition": site}
                for mod, site in chain.ptms
            ]
            msa_dir = _msa_dir_for_protenix(cplx=cplx, entity_id=entity_id)
            unpaired = getattr(chain, "unpaired_msa", "") or ""
            paired = getattr(chain, "paired_msa", "") or ""
            if not unpaired.strip():
                unpaired = _minimal_a3m(
                    protein_sequence, header=f"protein|{entity_id}"
                )
            if not paired.strip():
                paired = _minimal_a3m(
                    protein_sequence, header=f"protein|{entity_id}"
                )
            _write_text_atomic(msa_dir / "non_pairing.a3m", unpaired)
            _write_text_atomic(msa_dir / "pairing.a3m", paired)
            chain_input["id"] = [str(getattr(chain, "id"))]
            chain_input["unpairedMsaPath"] = str(msa_dir / "non_pairing.a3m")
            chain_input["pairedMsaPath"] = str(msa_dir / "pairing.a3m")
            chain_input["msa"] = {
                "precomputed_msa_dir": str(msa_dir),
                "pairing_db": "uniref100",
            }
            if use_templates:
                templates_path = _materialize_inline_templates(
                    chain=chain, cplx=cplx, entity_id=entity_id
                )
                if templates_path is not None:
                    chain_input["templatesPath"] = templates_path
            chain_type = "proteinChain"
            protenix_inputs["sequences"].append({chain_type: chain_input})
        elif isinstance(chain, DnaChain):
            entity_id = len(protenix_inputs["sequences"]) + 1
            chain_id_to_entities[str(getattr(chain, "id"))].append(entity_id)
            chain_input["sequence"] = chain.sequence
            mods = getattr(chain, "modifications", [])
            if callable(mods):
                mods = mods()
            chain_input["modifications"] = [
                {"ptmType": f"CCD_{mod}", "ptmPosition": site}
                for mod, site in mods
            ]
            chain_input["id"] = [str(getattr(chain, "id"))]
            chain_type = "dnaSequence"
            protenix_inputs["sequences"].append({chain_type: chain_input})
        elif isinstance(chain, RnaChain):
            entity_id = len(protenix_inputs["sequences"]) + 1
            chain_id_to_entities[str(getattr(chain, "id"))].append(entity_id)
            chain_input["sequence"] = chain.sequence
            mods = getattr(chain, "modifications", [])
            if callable(mods):
                mods = mods()
            chain_input["modifications"] = [
                {"ptmType": f"CCD_{mod}", "ptmPosition": site}
                for mod, site in mods
            ]
            chain_input["id"] = [str(getattr(chain, "id"))]
            unpaired = getattr(chain, "unpaired_msa", "") or ""
            if unpaired.strip():
                msa_dir = _msa_dir_for_protenix(cplx=cplx, entity_id=entity_id)
                _write_text_atomic(msa_dir / "rna_unpaired.a3m", unpaired)
                chain_input["unpairedMsaPath"] = str(msa_dir / "rna_unpaired.a3m")
            chain_type = "rnaSequence"
            protenix_inputs["sequences"].append({chain_type: chain_input})
        elif isinstance(chain, Ligand):
            if chain.smiles is not None:
                entity_id = len(protenix_inputs["sequences"]) + 1
                chain_id_to_entities[str(getattr(chain, "id"))].append(entity_id)
                chain_input["ligand"] = chain.smiles
                chain_type = "ligand"
                protenix_inputs["sequences"].append({chain_type: chain_input})
            elif chain.ccd_ids is not None:
                for ccd_id in chain.ccd_ids:
                    entity_id = len(protenix_inputs["sequences"]) + 1
                    chain_id_to_entities[str(getattr(chain, "id"))].append(entity_id)
                    chain_input["ligand"] = f"CCD_{ccd_id}"
                    chain_type = "ligand"
                    protenix_inputs["sequences"].append(
                        {chain_type: deepcopy(chain_input)}
                    )
            else:
                raise ValueError("Ligand chain must have either smiles or ccd_ids")

    bonded = getattr(cplx_input, "bonded_atom_pairs", None) or []
    if bonded:
        covalent_bonds: list[dict[str, object]] = []
        for atom1, atom2 in bonded:
            c1, r1, a1 = atom1
            c2, r2, a2 = atom2
            e1 = _single_entity_id(str(c1), purpose="bond constraint")
            e2 = _single_entity_id(str(c2), purpose="bond constraint")
            covalent_bonds.append(
                {
                    "entity1": int(e1),
                    "copy1": 1,
                    "position1": int(r1),
                    "atom1": str(a1),
                    "entity2": int(e2),
                    "copy2": 1,
                    "position2": int(r2),
                    "atom2": str(a2),
                }
            )
        protenix_inputs["covalent_bonds"] = covalent_bonds

    if not use_constraints:
        return protenix_inputs

    # Backward-compat: allow passing raw Protenix constraint dict directly.
    raw_constraint = getattr(global_config, "protenix_constraint", None)
    if raw_constraint:
        protenix_inputs["constraint"] = raw_constraint
        return protenix_inputs

    from astrolabe.core.binding_site import parse_binding_site
    from astrolabe.evaluators.binding_site_constraints import (
        build_protenix_binding_site_constraint,
    )

    binding_site_cfg = getattr(global_config, "binding_site", None)
    spec = parse_binding_site(binding_site_cfg, oligo_name=cplx)
    if spec is None:
        try:
            dynamic_binding_site = getattr(global_config, "dynamic_binding_site", None)
        except RuntimeError:
            dynamic_binding_site = None
        spec = parse_binding_site(dynamic_binding_site, oligo_name=cplx)
    if spec is not None:
        constraint = build_protenix_binding_site_constraint(
            spec,
            chain_id_to_entities=chain_id_to_entities,
            context=cplx,
            oligo_name=cplx,
        )
        if constraint is not None:
            protenix_inputs["constraint"] = constraint

    # with open("/tmp/protenix_inputs.json", "w") as f:
    #     json.dump(protenix_inputs, f)
    return protenix_inputs


def protenix_data_to_feat(single_sample_dict):
    _lazy_import_protenix()
    sample2feat = SampleDictToFeatures(
        single_sample_dict,
    )
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    entity_poly_type = sample2feat.entity_poly_type

    # Msa features
    entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
    msa_features = {}

    # Make dummy features for not implemented features
    dummy_feats = ["template"]
    if len(msa_features) == 0:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_tensor(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(
        features_dict=features_dict,
        dummy_feats=dummy_feats,
    )

    # Transform to right data type
    feat = data_type_transform(feat_or_label_dict=features_dict)

    data = extract_input_feature(entity_poly_type, feat)

    return data


def extract_input_feature(entity_poly_type, feat):
    data = {}
    data["input_feature_dict"] = feat
    # Add dimension related items
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]
    N_msa = feat["msa"].shape[0]
    stats = {}
    for mol_type in ["ligand", "protein", "dna", "rna"]:
        mol_type_mask = feat[f"is_{mol_type}"].bool()
        stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
        stats[f"{mol_type}/token"] = len(
            torch.unique(feat["atom_to_token_idx"][mol_type_mask])
        )
    N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
    data.update(
        {
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        }
    )

    def formatted_key(key):
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        else:
            pass
        return f"N_{type_}_{unit}"

    data.update(
        {
            formatted_key(k): torch.tensor([stats[k]])
            for k in [
                "protein/atom",
                "ligand/atom",
                "dna/atom",
                "rna/atom",
                "protein/token",
                "ligand/token",
                "dna/token",
                "rna/token",
            ]
        }
    )
    data.update({"entity_poly_type": entity_poly_type})
    return data


class DataDumper:
    def __init__(
        self,
        base_dir,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
        )

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data based on the dataset name, sample name, and seed.
        """
        dump_dir = os.path.join(
            self.base_dir, dataset_name, sample_name, f"seed_{seed}"
        )
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
    ):
        """
        Dump raw predictions from the model:
            structure: Save the predicted coordinates as CIF files.
            confidence: Save the confidence data as JSON files.
        """
        _lazy_import_protenix()
        prediction_save_dir = os.path.join(dump_dir, "predictions")
        os.makedirs(prediction_save_dir, exist_ok=True)

        # Dump structure
        b_factor = None
        if "full_data" in pred_dict:
            all_atom_plddt = []
            # len(pred_dict["full_data"]) == N_sample
            for each_sample_dict in pred_dict["full_data"]:
                if "atom_plddt" in each_sample_dict:
                    # atom_plddt.shape == [N_atom]
                    atom_plddt = each_sample_dict["atom_plddt"]
                    if atom_plddt.dtype == torch.bfloat16:
                        atom_plddt = atom_plddt.to(torch.float32)
                    all_atom_plddt.append(atom_plddt.cpu().numpy() * 100.0)

            if len(all_atom_plddt) == len(pred_dict["full_data"]):
                b_factor = all_atom_plddt
        sorted_indices = self._get_ranker_indices(data=pred_dict)
        self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
            sorted_indices=sorted_indices,
            b_factor=b_factor,
        )
        # Dump confidence
        self._save_confidence(
            data=pred_dict,
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            seed=seed,
            sorted_indices=sorted_indices,
        )

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
        sorted_indices: None,
        b_factor: torch.Tensor = None,
    ):
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        if sorted_indices is None:
            sorted_indices = range(N_sample)  # do not rank the output file
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_seed_{seed}_sample_{rank}.cif",
            )
            if b_factor is not None:
                # b_factor.shape == [N_sample, N_atom]
                atom_array.set_annotation("b_factor", np.round(b_factor[idx], 2))

            save_structure_cif(
                atom_array=atom_array,
                pred_coordinate=pred_coordinates[idx],
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=sample_name,
            )

    def _get_ranker_indices(self, data: dict):
        N_sample = len(data["summary_confidence"])
        if self.sorted_by_ranking_score:
            value = torch.tensor(
                [
                    data["summary_confidence"][i]["ranking_score"]
                    for i in range(N_sample)
                ]
            )
            sorted_indices = [
                i for i in torch.argsort(torch.argsort(value, descending=True))
            ]
        else:
            sorted_indices = [i for i in range(N_sample)]
        return sorted_indices

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        seed: int,
        sorted_indices: None,
    ):
        N_sample = len(data["summary_confidence"])
        for idx in range(N_sample):
            if self.need_atom_confidence:
                data["full_data"][idx] = get_clean_full_confidence(
                    data["full_data"][idx]
                )
        if sorted_indices is None:
            sorted_indices = range(N_sample)
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_seed_{seed}_summary_confidence_sample_{rank}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{rank}.json",
                )
                save_json(data["full_data"][idx], output_fpath, indent=None)


def update_inference_configs(configs: Any, N_token: int):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


class ProtenixRunner(object):
    def __init__(self, configs: Any) -> None:
        _ensure_protenix_ccd_cache_files_exist()
        _lazy_import_protenix()
        self.configs = _normalize_protenix_configs(configs)
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=self.configs.need_atom_confidence,
            sorted_by_ranking_score=self.configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        # NOTE: torch.cuda.device_count() may be non-zero via NVML even when CUDA
        # driver init fails (e.g. cuInit error). Use is_available() as the source of truth.
        try:
            cuda_count = int(torch.cuda.device_count())
        except Exception:
            cuda_count = 0
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        self.use_cuda = cuda_available
        if cuda_count > 0 and (not cuda_available):
            raise RuntimeError(
                "Protenix requires CUDA for inference, but torch.cuda.is_available() is False "
                f"while CUDA devices are visible (count={cuda_count}). "
                "Refusing CPU fallback. Check your CUDA driver / device permissions."
            )

        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(cuda_count))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl" if self.use_cuda else "gloo")
        if getattr(self.configs, "use_deepspeed_evo_attention", False):
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        _patch_protenix_cyclic_relative_position_encoding()
        _patch_protenix_fast_truncnorm_init()
        self.model = Protenix(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = _resolve_path(getattr(self.configs, "load_checkpoint_path", None))
        if not checkpoint_path:
            checkpoint_dir = _resolve_path(
                getattr(self.configs, "load_checkpoint_dir", None)
            )
            model_name = getattr(self.configs, "model_name", None)
            if checkpoint_dir and model_name:
                checkpoint_path = str(Path(checkpoint_dir) / f"{model_name}.pt")
        if not checkpoint_path:
            raise ValueError("Protenix load_checkpoint_path is empty")

        try:
            from protenix.web_service.dependency_url import URL as _dep_urls
        except Exception:
            _dep_urls = {}

        def _download_to(url: str, dst: str) -> None:
            dst_resolved = _resolve_path(dst)
            if not dst_resolved:
                return
            dst_path = Path(dst_resolved)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
            urllib.request.urlretrieve(url, str(tmp))  # noqa: S310
            os.replace(str(tmp), str(dst_path))

        def _download_if_missing(key: str, dst: str) -> None:
            dst_resolved = _resolve_path(dst)
            if not dst_resolved:
                return
            if os.path.exists(dst_resolved):
                return
            url = _dep_urls.get(key)
            if not url:
                return
            if DIST_WRAPPER.rank == 0:
                logging.info("Downloading Protenix asset %s -> %s", key, dst_resolved)
                _download_to(url, dst_resolved)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        # Best-effort: download required data caches if missing.
        data_cfg = getattr(self.configs, "data", None)
        if data_cfg is not None:
            _download_if_missing(
                "ccd_components_file",
                getattr(data_cfg, "ccd_components_file", None),
            )
            _download_if_missing(
                "ccd_components_rdkit_mol_file",
                getattr(data_cfg, "ccd_components_rdkit_mol_file", None),
            )
            # Protenix internally reads CCD paths from `configs.configs_data` at import time.
            # On hosts without /af3-dev, it falls back to <protenix_code_dir>/release_data/ccd_cache.
            # Ensure those fallback paths exist by symlinking to our configured asset locations.
            try:
                from configs.configs_data import data_configs as _ptx_data_cfg  # type: ignore

                def _link_if_missing(dst: str | None, src: str | None) -> None:
                    dst_resolved = _resolve_path(dst)
                    src_resolved = _resolve_path(src)
                    if not dst_resolved or not src_resolved:
                        return
                    if os.path.exists(dst_resolved) or (not os.path.exists(src_resolved)):
                        return
                    try:
                        Path(dst_resolved).parent.mkdir(parents=True, exist_ok=True)
                        Path(dst_resolved).symlink_to(Path(src_resolved))
                    except Exception:
                        # Best-effort only; if we can't write here, downstream will surface a clear error.
                        return

                _link_if_missing(
                    _ptx_data_cfg.get("ccd_components_file"),
                    getattr(data_cfg, "ccd_components_file", None),
                )
                _link_if_missing(
                    _ptx_data_cfg.get("ccd_components_rdkit_mol_file"),
                    getattr(data_cfg, "ccd_components_rdkit_mol_file", None),
                )
            except Exception:
                pass

        if not os.path.exists(checkpoint_path):
            model_key = Path(checkpoint_path).stem
            _download_if_missing(model_key, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Protenix checkpoint missing: {checkpoint_path}")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=getattr(
                    self.configs, "mc_dropout_apply_rate", 0.4
                ),
            )

        return prediction

    def print(self, msg: str) -> None:
        logging.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs
