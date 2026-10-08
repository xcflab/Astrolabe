from __future__ import annotations

import glob
import json
import logging
import pathlib
from collections import defaultdict

import numpy as np
import omegaconf
from Bio import PDB as PDB
from Bio.PDB import MMCIFParser

from astrolabe.core import global_config
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.boltz import cli, from_af3_to_boltz

if __name__ == "__main__":
    cli()


class BoltzEvaluator(Evaluator):
    def __init__(self, model_runner, output_dir):
        self.model_runner = model_runner
        self.output_dir = output_dir

    def run_function(self, input_feat):
        for oligo_name, cplx_input in input_feat.items():
            self.model_runner.run_inference(cplx_input["data"], cplx_input["out_dir"])
        return input_feat

    def prepare_input(self, inputs, runtime_config):

        step = global_config.step
        result = defaultdict(dict)

        sequence_dir = global_config.out_dir / "sequences"
        runtime_config["chain2input_chain"] = {}
        for oligo_name, cplx_input in inputs.items():
            out_dir = self.output_dir / f"step_{step:03d}_{oligo_name}"
            yaml_file = f"step_{step:03d}_{oligo_name}_input.yaml"
            data_input, chain2input_chain = from_af3_to_boltz(cplx_input)
            runtime_config["chain2input_chain"][oligo_name] = chain2input_chain
            with open(sequence_dir / yaml_file, "w") as f:
                omegaconf.OmegaConf.save(data_input, f)
            data = self.model_runner.featurize_data(sequence_dir / yaml_file, out_dir)
            result[oligo_name]["out_dir"] = out_dir
            result[oligo_name]["data"] = data
        input_feat = result
        return result

    def prepare_output(self, output, inputs, runtime_config):
        step = global_config.step
        chain2proto = runtime_config.get("chain2proto") or {}
        structure_path = global_config.out_dir / "structures"
        plddt = {}
        ptm = {}
        iptm = {}
        pde = {}
        complex_paths = {}
        chain_pair_scores = {}
        per_proto_scores = defaultdict(list)

        for oligo_name, data in output.items():
            chain2input_chain = runtime_config["chain2input_chain"][oligo_name]
            output_dir = data["out_dir"]
            best_model = glob.glob(f"{output_dir}/predictions/*/*_model_0.cif")[0]
            scores = glob.glob(f"{output_dir}/predictions/*/*_model_*.json")
            scores = [json.load(open(score, "r")) for score in sorted(scores)]
            complex_plddt, complex_paths[oligo_name] = self.rechain_structure(
                best_model, chain2input_chain, oligo_name, step, structure_path
            )
            input_obj = inputs[oligo_name]

            idx = 0
            for chain in input_obj.protein_chains:
                chain_slice = slice(idx, idx + len(chain.sequence))
                idx += len(chain.sequence)
                proto = chain2proto.get(chain.id, chain.id)
                per_proto_scores[proto].append(complex_plddt[chain_slice])

            plddt[oligo_name] = [score["complex_plddt"] for score in scores]
            ptm[oligo_name] = [score["ptm"] for score in scores]
            iptm[oligo_name] = [score["iptm"] for score in scores]
            pde[oligo_name] = [score["complex_pde"] for score in scores]
            chain_pair_scores[oligo_name] = {
                "chain_pair_iptm": [
                    np.array(
                        [
                            [v2 for v2 in v1.values()]
                            for v1 in score["pair_chains_iptm"].values()
                        ],
                        dtype=float,
                    )
                    for score in scores
                ],
                "chain_ptm": [
                    np.array(
                        [value for value in score["chains_ptm"].values()],
                        dtype=float,
                    )
                    for score in scores
                ],
            }
        return StandardScores(
            names=list(inputs.keys()),
            plddt=plddt,
            ptm=ptm,
            pae=None,
            iptm=iptm,
            per_proto_scores=per_proto_scores,
            pde=pde,
            chain_pair_scores=chain_pair_scores,
            complex_paths=complex_paths,
        )

    def rechain_structure(
        self, best_model, chain2input_chain, oligo_name, step, structure_path
    ):
        parser = MMCIFParser()
        structure = parser.get_structure("model", best_model)
        new_structure = PDB.Model.Model(0)
        for model in structure:
            for chain in model:
                new_chain = chain.copy()
                if chain.get_id() in chain2input_chain:
                    logging.info(
                        f"Rechain {chain.get_id()} to {chain2input_chain[chain.get_id()]}"
                    )
                    new_chain.id = chain2input_chain[chain.get_id()]

                    if new_chain.id in new_structure:
                        for residue in new_chain:
                            new_structure[new_chain.id].insert(0, residue)
                    else:
                        new_structure.add(new_chain)
        bfactor = []
        sequence = []
        for residue in new_structure.get_residues():
            if "CA" in residue:
                bfactor.append(residue["CA"].get_bfactor())
                sequence.append(residue.get_resname())

        complex_path = pathlib.Path(
            f"{structure_path}/step_{step:03d}_{oligo_name}_model.cif"
        )
        io = PDB.MMCIFIO()
        io.set_structure(new_structure)
        with open(complex_path, "w") as f:
            io.save(f)
            f.write('_dummy.title ""\n')

        return bfactor, complex_path
