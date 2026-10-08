from collections import defaultdict
import importlib
from pathlib import Path
import sys
from typing import Any

import torch

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_esmfold_vendor_paths() -> None:
    for path in (
        _repo_root() / "vendor" / "openfold_esm2",
        _repo_root() / "vendor" / "esm2",
    ):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _load_renamed_esmfold_model_class():
    repo_vendor = str(_repo_root() / "vendor")
    original_sys_path = list(sys.path)
    namespace_module = sys.modules.get("esm2")
    if namespace_module is not None and getattr(namespace_module, "__file__", None) is None:
        sys.modules.pop("esm2", None)
    try:
        sys.path = [path for path in sys.path if path != repo_vendor]
        return importlib.import_module("esm2.esmfold.v1.esmfold").ESMFold
    finally:
        sys.path = original_sys_path


def _load_esmfold_model_class():
    try:
        return _load_renamed_esmfold_model_class()
    except ImportError as esm2_exc:
        _ensure_esmfold_vendor_paths()
        try:
            return _load_renamed_esmfold_model_class()
        except ImportError:
            try:
                return importlib.import_module("esm.esmfold.v1.esmfold").ESMFold
            except ImportError:
                raise RuntimeError(
                    "Failed to import ESMFold model implementation."
                ) from esm2_exc


def _adapt_esmfold_state_dict_for_openfold(
    model_state: dict[str, Any],
) -> dict[str, Any]:
    adapted = dict(model_state)
    for base_key in (
        "trunk.structure_module.ipa.linear_q_points",
        "trunk.structure_module.ipa.linear_kv_points",
    ):
        for suffix in ("weight", "bias"):
            old_key = f"{base_key}.{suffix}"
            new_key = f"{base_key}.linear.{suffix}"
            if old_key in adapted and new_key not in adapted:
                adapted[new_key] = adapted[old_key]
    return adapted


def renumber_each_chain(pdb_str):
    """Renumber each chain in a PDB string to start from 1."""
    output_lines = []
    current_chain = None
    residue_number = 0
    last_residue_id = None

    for line in pdb_str.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            residue_id = line[22:27].strip()
            if chain_id != current_chain:
                current_chain = chain_id
                residue_number = 1
                last_residue_id = residue_id
            elif residue_id != last_residue_id:
                residue_number += 1
                last_residue_id = residue_id
            new_line = line[:22] + f"{residue_number:>4}" + line[26:]
            output_lines.append(new_line)
        else:
            output_lines.append(line)

    return "\n".join(output_lines) + "\n"


class ESMFoldEvaluator(Evaluator):
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "esmfold_3B_v1",
        device: str | None = None,
    ):
        self._model = model
        self._model_name = model_name
        self._device = device

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        model_name = self._model_name
        if model_name.endswith(".pt"):  # local, treat as filepath
            model_path = Path(model_name).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(
                    "ESMFold weights not found at "
                    f"{model_path}. Provide a local .pt file path for `model_name`."
                )
        else:
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
            model_path = cache_dir / f"{model_name}.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    "ESMFold weights are missing and auto-download is disabled. "
                    f"Expected cached checkpoint at {model_path}. "
                    "Download it ahead of time or set `model_name` to an explicit local .pt path."
                )

        model_data = torch.load(str(model_path), map_location="cpu", weights_only=False)
        cfg = model_data["cfg"]["model"]
        model_state = _adapt_esmfold_state_dict_for_openfold(model_data["model"])

        ESMFold = _load_esmfold_model_class()
        model = ESMFold(esmfold_config=cfg)
        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())

        missing_essential_keys = []
        for missing_key in expected_keys - found_keys:
            if not missing_key.startswith("esm."):
                missing_essential_keys.append(missing_key)
        if missing_essential_keys:
            raise RuntimeError(
                f"Keys '{', '.join(missing_essential_keys)}' are missing."
            )

        model.load_state_dict(model_state, strict=False)

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

        self._model = model.eval().to(device)
        return self._model

    def prepare_input(self, inputs, runtime_config):
        input_feat = {}
        for k, v in inputs.items():
            if not all(
                global_config.protomers[c].type == ProtomerTypes.PROTEIN for c in k
            ):
                raise ValueError("ESMFold can only handle protein sequences")
            input_feat[k] = ":".join(c.sequence for c in v.protein_chains)
        return input_feat

    def prepare_output(self, output, inputs, runtime_config):
        return output

    def run_function(self, input_feat):
        with torch.no_grad():
            names = list(input_feat.keys())
            model = self._ensure_model()
            output = model.infer([input_feat[n] for n in names])
        structure_path = global_config.out_dir / "structures"
        results = defaultdict(dict)
        for i, oligo_name in enumerate(names):
            jobname = f"step_{global_config.step:03d}_{oligo_name}"
            with open(structure_path / f"{jobname}_model.pdb", "w") as f:
                f.write(renumber_each_chain(model.output_to_pdb(output)[i]))
            results["complex_paths"][oligo_name] = (
                structure_path / f"{jobname}_model.pdb"
            )
            mask = output["chain_index"][i] != -1
            results["per_proto_scores"][oligo_name] = (
                output["plddt"][i, mask].mean(-1).cpu().numpy() / 100
            )
            res_idx = 0
            results["per_proto_scores"] = defaultdict(list)
            for chain_id in oligo_name:
                chain_len = global_config.protomers[chain_id].length
                results["per_proto_scores"][chain_id].append(
                    (output["plddt"][i, mask].mean(-1).cpu().numpy() / 100)[
                        res_idx : res_idx + chain_len
                    ]
                )
                res_idx += chain_len
            results["plddt"][oligo_name] = output["mean_plddt"][i].cpu().numpy() / 100
            results["ptm"][oligo_name] = output["ptm"][i].cpu().numpy()
            results["pae"][oligo_name] = (
                output["predicted_aligned_error"][i][mask][:, mask].cpu().numpy()
            )

        return StandardScores(names=names, **results)
