from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.esmfold2 import ESMFold2Runner


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


class ESMFold2Evaluator(Evaluator):
    """Subprocess evaluator for the independent Biohub ESMFold2 model."""

    def __init__(
        self,
        *,
        runner: ESMFold2Runner | None = None,
        raw_model_results_dir: str | Path | None = None,
        model_name: str = "biohub/ESMFold2",
        python_bin: str | None = None,
        esm_repo_path: str | None = None,
        transformers_repo_path: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        esmc_precision: str = "bf16",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        lm_dropout: float | None = 0.0,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        chunk_size: int | None = 64,
        local_files_only: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self.raw_model_results_dir = Path(raw_model_results_dir) if raw_model_results_dir else None
        self.runner = runner or ESMFold2Runner(
            python_bin=python_bin,
            esm_repo_path=esm_repo_path,
            transformers_repo_path=transformers_repo_path,
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            esmc_precision=esmc_precision,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            lm_dropout=lm_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            chunk_size=chunk_size,
            local_files_only=local_files_only,
            timeout_seconds=timeout_seconds,
        )

    def prepare_input(self, inputs, runtime_config):
        jobs = []
        for oligo_name, folding_input in inputs.items():
            if not all(
                global_config.protomers[chain_id].type == ProtomerTypes.PROTEIN
                for chain_id in oligo_name
            ):
                raise ValueError("ESMFold2 currently supports protein chains only")
            chains = []
            for chain_id, chain in zip(oligo_name, folding_input.protein_chains):
                chains.append({"id": chain_id, "sequence": chain.sequence})
            jobs.append({"name": oligo_name, "chains": chains})
        return jobs

    def run_function(self, input_feat):
        raw_dir = self.raw_model_results_dir
        if raw_dir is None:
            raw_dir = global_config.out_dir / "raw_model_results" / "esmfold2"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return self.runner.run(input_feat, raw_dir)

    def prepare_output(self, output, inputs, runtime_config):
        names = [job["name"] for job in output["jobs"]]
        structure_dir = global_config.out_dir / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        complex_paths = {}
        plddt = {}
        ptm = {}
        iptm = {}
        pae = {}
        pde = {}
        per_proto_scores = defaultdict(list)

        for job in output["jobs"]:
            oligo_name = job["name"]
            jobname = f"step_{global_config.step:03d}_{oligo_name}"
            source_path = Path(job["structure_path"])
            target_path = structure_dir / f"{jobname}_model.cif"
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            complex_paths[oligo_name] = target_path

            residue_plddt = _as_array(job["plddt"])
            plddt[oligo_name] = float(job["plddt_mean"])
            if job.get("ptm") is not None:
                ptm[oligo_name] = float(job["ptm"])
            if job.get("iptm") is not None:
                iptm[oligo_name] = float(job["iptm"])
            if job.get("pae") is not None:
                pae[oligo_name] = _as_array(job["pae"])
            if job.get("pde") is not None:
                pde[oligo_name] = _as_array(job["pde"])

            offset = 0
            for chain_id in oligo_name:
                chain_len = global_config.protomers[chain_id].length
                per_proto_scores[chain_id].append(residue_plddt[offset : offset + chain_len])
                offset += chain_len

        return StandardScores(
            names=names,
            plddt=plddt,
            complex_paths=complex_paths,
            per_proto_scores=dict(per_proto_scores),
            ptm=ptm or None,
            iptm=iptm or None,
            pae=pae or None,
            pde=pde or None,
        )
