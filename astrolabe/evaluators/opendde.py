from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from astrolabe.core import global_config
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.opendde import (
    OpenDDEPreparedInput,
    OpenDDERunner,
    collect_opendde_outputs,
    write_opendde_input,
)


class OpenDDEEvaluator(Evaluator):
    def __init__(self, model_runner: OpenDDERunner, output_dir: Path):
        self.model_runner = model_runner
        self.output_dir = Path(output_dir)

    def prepare_input(self, inputs, runtime_config):
        step = int(getattr(global_config, "step", 0) or 0)
        sequence_dir = Path(global_config.out_dir) / "sequences" / "opendde"
        prepared: dict[str, OpenDDEPreparedInput] = {}
        include_msa_paths = bool(
            getattr(self.model_runner, "use_msa", False)
            or getattr(self.model_runner, "use_template", False)
            or getattr(self.model_runner, "use_rna_msa", False)
        )
        for oligo_name, cplx_input in inputs.items():
            job_name = f"step_{step:03d}_{oligo_name}"
            prepared[oligo_name] = write_opendde_input(
                cplx_input,
                oligo_name=oligo_name,
                job_name=job_name,
                input_dir=sequence_dir,
                output_dir=self.output_dir,
                include_msa_paths=include_msa_paths,
                use_template=bool(getattr(self.model_runner, "use_template", False)),
            )
        return prepared

    def run_function(self, input_feat):
        for prepared in input_feat.values():
            self.model_runner.run_inference(
                input_json=prepared.json_path,
                out_dir=prepared.output_dir,
            )
        return input_feat

    def prepare_output(self, output, inputs, runtime_config):
        structures_dir = Path(global_config.out_dir) / "structures"
        plddt: dict[str, list[Any]] = {}
        ptm: dict[str, list[float]] = {}
        iptm: dict[str, list[float]] = {}
        pde: dict[str, list[float]] = {}
        complex_paths: dict[str, Path] = {}
        per_proto_scores = defaultdict(list)
        chain_pair_scores: dict[str, dict[str, list[Any]]] = {}

        for oligo_name, prepared in output.items():
            collected = collect_opendde_outputs(prepared, structures_dir=structures_dir)
            plddt[oligo_name] = collected["plddt"]
            if collected["ptm"] is not None:
                ptm[oligo_name] = collected["ptm"]
            if collected["iptm"] is not None:
                iptm[oligo_name] = collected["iptm"]
            if collected["pde"] is not None:
                pde[oligo_name] = collected["pde"]
            complex_paths[oligo_name] = collected["complex_path"]
            for proto, values in (collected["per_proto_scores"] or {}).items():
                per_proto_scores[proto].extend(values)
            if collected["chain_pair_scores"]:
                chain_pair_scores[oligo_name] = collected["chain_pair_scores"]

        return StandardScores(
            names=list(output.keys()),
            plddt=plddt,
            ptm=ptm or None,
            pae=None,
            iptm=iptm or None,
            pde=pde or None,
            per_proto_scores=dict(per_proto_scores) or None,
            chain_pair_scores=chain_pair_scores or None,
            complex_paths=complex_paths,
        )
