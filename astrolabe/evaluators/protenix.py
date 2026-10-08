from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from astrolabe.core import global_config
from astrolabe.core.cyclization import flatten_cyclic_relative_patches
from astrolabe.core.folding_input import ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.protenix import (
    protenix_from_dict_to_feature,
    from_af3_to_protenix,
    update_inference_configs,
)


class ProtenixEvaluator(Evaluator):
    _SUMMARY_VECTOR_FIELDS = (
        "chain_ptm",
        "chain_iptm",
        "chain_plddt",
        "chain_gpde",
    )
    _SUMMARY_MATRIX_FIELDS = (
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_pair_plddt",
        "chain_pair_gpde",
        "chain_pair_pae_min",
    )

    def __init__(self, model_runner: "ProtenixRunner", *, use_constraints: bool = True):
        self.model_runner = model_runner
        self.use_constraints = bool(use_constraints)

    @classmethod
    def _normalize_summary_value(cls, value, *, field_name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"Protenix summary field {field_name!r} is None.")

        if torch.is_tensor(value):
            return np.asarray(value.detach().cpu().float().numpy(), dtype=float)

        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=float)

        if isinstance(value, (list, tuple)):
            normalized_items = [
                cls._normalize_summary_value(item, field_name=field_name)
                for item in value
            ]
            return np.asarray(normalized_items, dtype=float)

        if all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
            return np.asarray(value.detach().cpu().numpy(), dtype=float)

        try:
            return np.asarray(value, dtype=float)
        except Exception as exc:
            raise ValueError(
                f"Failed to normalize Protenix summary field {field_name!r} "
                f"from value of type {type(value).__name__}."
            ) from exc

    @classmethod
    def _normalize_summary_scalar(cls, value, *, field_name: str) -> float:
        arr = cls._normalize_summary_value(value, field_name=field_name)
        if arr.size != 1:
            raise ValueError(
                f"Expected scalar Protenix summary field {field_name!r}, got shape {arr.shape}."
            )
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _find_raw_prediction_cif(
        *,
        dump_dir: Path,
        dataset_name: str,
        oligo_name: str,
        seed: int,
        sample_idx: int = 0,
    ) -> Path | None:
        fname = f"{oligo_name}_seed_{seed}_sample_{sample_idx}.cif"
        candidates = [
            dump_dir
            / dataset_name
            / oligo_name
            / f"seed_{seed}"
            / "predictions"
            / fname,
            dump_dir / oligo_name / f"seed_{seed}" / "predictions" / fname,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Last resort: tolerate older/alternate layouts by searching under dump_dir.
        hits = list(dump_dir.glob(f"**/predictions/{fname}"))
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return hits[0]
        return None

    @staticmethod
    def _structure_start_soft_select_enabled() -> bool:
        try:
            state = global_config.structure_start
        except RuntimeError:
            return False
        config = getattr(state, "config", None)
        return bool(config is not None and getattr(config, "mode", None) == "soft_select")

    def prepare_input(self, inputs, runtime_config):
        result = {}
        for oligo_name, cplx_input in inputs.items():
            prepared = protenix_from_dict_to_feature(
                from_af3_to_protenix(
                    cplx_input,
                    oligo_name,
                    use_constraints=self.use_constraints,
                    use_templates=bool(
                        getattr(self.model_runner.configs, "use_template", False)
                    ),
                ),
                configs=self.model_runner.configs,
            )
            data, atom_array, stats = prepared
            cyclic_chain_ids = {
                str(chain_id)
                for chain_id in (getattr(cplx_input, "cyclic_chain_ids", None) or ())
            }
            if cyclic_chain_ids:
                cyclic_asym_ids = [
                    chain_idx
                    for chain_idx, chain in enumerate(cplx_input.chains)
                    if str(getattr(chain, "id", "")) in cyclic_chain_ids
                ]
                if cyclic_asym_ids:
                    data["input_feature_dict"]["cyclic_asym_ids"] = torch.tensor(
                        cyclic_asym_ids, dtype=torch.long
                    )
            chain_map = {
                str(getattr(chain, "id", "")): chain_idx
                for chain_idx, chain in enumerate(cplx_input.chains)
            }
            cyclic_relative_patches = tuple(
                (
                    int(chain_map[chain_id]),
                    int(residue_i),
                    int(residue_j),
                    int(offset),
                )
                for chain_id, residue_i, residue_j, offset in flatten_cyclic_relative_patches(
                    getattr(cplx_input, "cyclic_relative_patches", None)
                )
            )
            if cyclic_relative_patches:
                data["input_feature_dict"]["cyclic_relative_patches"] = torch.tensor(
                    cyclic_relative_patches, dtype=torch.long
                )
            result[oligo_name] = (data, atom_array, stats)
        return result

    def run_function(self, input_feat):
        result = {}
        step = int(getattr(global_config, "step", 0) or 0)
        dataset_name = f"step_{step:03d}"
        for oligo_name, cplx_input in input_feat.items():
            data, atom_array, data_error_message = cplx_input
            new_configs = update_inference_configs(
                self.model_runner.configs, data["N_token"].item()
            )
            self.model_runner.update_model_configs(new_configs)
            prediction = self.model_runner.predict(data)
            self.model_runner.dumper.dump(
                dataset_name=dataset_name,
                pdb_id=oligo_name,
                seed=self.model_runner.configs.seed,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
            )
            prediction["atom_to_residue"] = data["input_feature_dict"][
                "atom_to_token_idx"
            ]
            result[oligo_name] = prediction

        return result

    def prepare_output(self, output, inputs, runtime_config):
        """"""

        pae = {}
        pde = {}
        ptm = {}
        iptm = {}
        chain_pair_scores = {}
        complex_paths = {}
        plddt = {}
        per_proto_plddt = defaultdict(list)
        step = int(getattr(global_config, "step", 0) or 0)
        dataset_name = f"step_{step:03d}"
        structures_dir = global_config.out_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)
        soft_select_candidates = self._structure_start_soft_select_enabled()
        candidate_payloads: dict[int, dict[str, dict]] = defaultdict(
            lambda: {
                "plddt": {},
                "ptm": {},
                "pae": {},
                "pde": {},
                "iptm": {},
                "per_proto_scores": defaultdict(list),
                "chain_pair_scores": {},
                "complex_paths": {},
            }
        )
        sample_indices_by_oligo: dict[str, set[int]] = {}

        def sample_metrics(prediction: dict, sample_idx: int):
            full_data = prediction.get("full_data") or []
            if sample_idx >= len(full_data):
                return None
            sample_data = full_data[sample_idx]
            summary = prediction.get("summary_confidence") or []
            sample_summary = summary[sample_idx] if sample_idx < len(summary) else None
            sample_ptm = 0.0
            sample_iptm = 0.0
            sample_chain_pair_scores = {}
            if sample_summary:
                sample_ptm = self._normalize_summary_scalar(
                    sample_summary.get("ptm", 0.0), field_name="ptm"
                )
                sample_iptm = self._normalize_summary_scalar(
                    sample_summary.get("iptm", 0.0), field_name="iptm"
                )
                for field_name in self._SUMMARY_VECTOR_FIELDS + self._SUMMARY_MATRIX_FIELDS:
                    if field_name not in sample_summary or sample_summary.get(field_name) is None:
                        continue
                    sample_chain_pair_scores[field_name] = [
                        self._normalize_summary_value(
                            sample_summary.get(field_name), field_name=field_name
                        )
                    ]
            _unique_keys, indices = np.unique(
                prediction["atom_to_residue"].cpu().numpy(), return_index=True
            )
            groups = np.split(
                self._normalize_summary_value(
                    sample_data["atom_plddt"], field_name="atom_plddt"
                ),
                indices,
            )[1:]
            per_residue_plddt = np.array([group.mean() for group in groups])
            return (
                sample_ptm,
                sample_iptm,
                sample_chain_pair_scores,
                per_residue_plddt,
                sample_data["token_pair_pae"],
                sample_data["token_pair_pde"],
            )

        for oligo_name, prediction in output.items():
            seed = self.model_runner.configs.seed
            dump_dir = Path(self.model_runner.configs.dump_dir)
            jobname = f"step_{step:03d}_{oligo_name}"
            raw_cif = self._find_raw_prediction_cif(
                dump_dir=dump_dir,
                dataset_name=dataset_name,
                oligo_name=oligo_name,
                seed=seed,
                sample_idx=0,
            )
            out_cif = structures_dir / f"{jobname}_model.cif"
            try:
                if raw_cif is None:
                    raise FileNotFoundError(
                        f"Protenix prediction CIF not found under {dump_dir} for {oligo_name}"
                    )
                shutil.copy(str(raw_cif), str(out_cif))
                complex_paths[oligo_name] = out_cif
            except Exception:
                if raw_cif is not None and raw_cif.exists():
                    complex_paths[oligo_name] = raw_cif

            metrics0 = sample_metrics(prediction, 0)
            if metrics0 is None:
                ptm[oligo_name] = [0.0]
                iptm[oligo_name] = [0.0]
                per_residue_plddt = np.array([])
            else:
                (
                    sample_ptm,
                    sample_iptm,
                    sample_chain_pair_scores,
                    per_residue_plddt,
                    sample_pae,
                    sample_pde,
                ) = metrics0
                ptm[oligo_name] = [sample_ptm]
                iptm[oligo_name] = [sample_iptm]
                chain_pair_scores[oligo_name] = sample_chain_pair_scores
                pae[oligo_name] = sample_pae
                pde[oligo_name] = sample_pde
            plddt[oligo_name] = [per_residue_plddt]
            res_idx = 0
            for chain_id, chain in zip(
                inputs[oligo_name].name, inputs[oligo_name].chains
            ):
                if isinstance(chain, ProteinChain):
                    per_proto_plddt[chain_id].append(
                        per_residue_plddt[res_idx : res_idx + len(chain.sequence)]
                    )
                    res_idx += len(chain.sequence)

            if not soft_select_candidates:
                continue
            sample_indices_by_oligo[oligo_name] = set()
            sample_count = len(prediction.get("full_data") or [])
            summary = prediction.get("summary_confidence") or []
            sample_count = max(sample_count, len(summary))
            for sample_idx in range(sample_count):
                metrics = sample_metrics(prediction, sample_idx)
                if metrics is None:
                    continue
                raw_sample_cif = self._find_raw_prediction_cif(
                    dump_dir=dump_dir,
                    dataset_name=dataset_name,
                    oligo_name=oligo_name,
                    seed=seed,
                    sample_idx=sample_idx,
                )
                if raw_sample_cif is None:
                    continue
                sample_jobname = f"{jobname}_candidate_{sample_idx:03d}"
                sample_path = structures_dir / f"{sample_jobname}_model.cif"
                shutil.copy(str(raw_sample_cif), str(sample_path))
                (
                    sample_ptm,
                    sample_iptm,
                    sample_chain_pair_scores,
                    per_residue_plddt,
                    sample_pae,
                    sample_pde,
                ) = metrics
                payload = candidate_payloads[sample_idx]
                payload["plddt"][oligo_name] = [per_residue_plddt]
                payload["ptm"][oligo_name] = [sample_ptm]
                payload["iptm"][oligo_name] = [sample_iptm]
                payload["pae"][oligo_name] = sample_pae
                payload["pde"][oligo_name] = sample_pde
                payload["chain_pair_scores"][oligo_name] = sample_chain_pair_scores
                payload["complex_paths"][oligo_name] = sample_path
                res_idx = 0
                for chain_id, chain in zip(
                    inputs[oligo_name].name, inputs[oligo_name].chains
                ):
                    if isinstance(chain, ProteinChain):
                        payload["per_proto_scores"][chain_id].append(
                            per_residue_plddt[res_idx : res_idx + len(chain.sequence)]
                        )
                        res_idx += len(chain.sequence)
                sample_indices_by_oligo[oligo_name].add(sample_idx)

        scores = StandardScores(
            names=list(inputs.keys()),
            plddt=plddt,
            ptm=ptm,
            pae=pae,
            iptm=iptm,
            per_proto_scores=per_proto_plddt,
            pde=pde,
            chain_pair_scores=chain_pair_scores or None,
            complex_paths=complex_paths,
        )
        if soft_select_candidates and sample_indices_by_oligo:
            common_indices = set.intersection(*sample_indices_by_oligo.values())
            candidates = []
            for sample_idx in sorted(common_indices):
                payload = candidate_payloads[sample_idx]
                candidates.append(
                    StandardScores(
                        names=list(inputs.keys()),
                        plddt=payload["plddt"],
                        ptm=payload["ptm"],
                        pae=payload["pae"],
                        iptm=payload["iptm"],
                        per_proto_scores=dict(payload["per_proto_scores"]),
                        pde=payload["pde"],
                        chain_pair_scores=payload["chain_pair_scores"] or None,
                        complex_paths=payload["complex_paths"],
                    )
                )
            if candidates:
                scores.candidate_scores = candidates
        return scores
