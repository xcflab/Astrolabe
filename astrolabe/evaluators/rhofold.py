import atexit
import logging
import os
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from astrolabe.core.folding_input import Input, RnaChain

try:
    from alphafold3.common.folding_input import RnaChain as AF3RnaChain
except ModuleNotFoundError:  # pragma: no cover
    AF3RnaChain = None

RnaChainTypes = (RnaChain,) if AF3RnaChain is None else (RnaChain, AF3RnaChain)

from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.core import global_config
from rhofold.data.balstn import BLASTN
from rhofold.model.rna_fm.data import get_rna_fm_token
from rhofold.utils import save_ss2ct
from rhofold.utils.alphabet import get_features, get_msa_feature


class RhoFoldEvaluator(Evaluator):
    """
    Evaluator for RhoFold.
    """

    def __init__(self, model_runner, output_dir, config):
        self.config = config or {}
        self.model_runner = model_runner.to(config.device)
        self.output_dir = Path(output_dir)

    def prepare_output(self, output, inputs, runtime_config):
        step = runtime_config.get("step", 0)
        plddt = {}
        complex_paths = {}
        structure_dir = Path(global_config.out_dir) / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)
        for oligo_name, input in inputs.items():
            oligo_output = output[oligo_name]
            jobname = f"step_{step:03d}_{oligo_name}"
            raw_result_dir = self.output_dir / jobname
            os.makedirs(raw_result_dir, exist_ok=True)

            # Secondary structure, .ct format
            ss_prob_map = torch.sigmoid(oligo_output["ss"][0, 0]).data.cpu().numpy()
            ss_file = raw_result_dir / "ss.ct"
            save_ss2ct(ss_prob_map, input.chains[0].sequence, ss_file, threshold=0.5)

            # Dist prob map & Secondary structure prob map, .npz format
            npz_file = raw_result_dir / "results.npz"
            np.savez_compressed(
                npz_file,
                dist_n=torch.softmax(oligo_output["n"].squeeze(0), dim=0)
                .data.cpu()
                .numpy(),
                dist_p=torch.softmax(oligo_output["p"].squeeze(0), dim=0)
                .data.cpu()
                .numpy(),
                dist_c=torch.softmax(oligo_output["c4_"].squeeze(0), dim=0)
                .data.cpu()
                .numpy(),
                ss_prob_map=ss_prob_map,
                plddt=oligo_output["plddt"][0].data.cpu().numpy(),
            )
            plddt[oligo_name] = oligo_output["plddt"][0].data.cpu().numpy().tolist()
            # Save the prediction
            unrelaxed_model = raw_result_dir / f"{jobname}_model.pdb"

            # The last cords prediction
            node_cords_pred = oligo_output["cord_tns_pred"][-1].squeeze(0)
            self.model_runner.structure_module.converter.export_pdb_file(
                input.chains[0].sequence,
                node_cords_pred.data.cpu().numpy(),
                path=unrelaxed_model,
                chain_id=None,
                confidence=oligo_output["plddt"][0].data.cpu().numpy(),
            )
            final_model = structure_dir / f"{jobname}_model.pdb"

            # Amber relaxation
            if self.config.relax_steps is not None:
                relax_steps = int(self.config.relax_steps)
                if relax_steps > 0:
                    try:
                        from rhofold.relax.relax import AmberRelaxation
                    except ModuleNotFoundError as exc:
                        raise RuntimeError(
                            "RhoFold relaxation requires OpenMM; install the optional OpenMM "
                            "dependency or set evaluator.rhofold.relax_steps=0."
                        ) from exc
                    amber_relax = AmberRelaxation(
                        max_iterations=relax_steps, logger=logging.getLogger("rhofold")
                    )
                    relaxed_model = raw_result_dir / f"relaxed_{relax_steps}_model.pdb"
                    amber_relax.process(unrelaxed_model, relaxed_model)

                    shutil.copy2(relaxed_model, final_model)
                else:
                    shutil.copy2(unrelaxed_model, final_model)
            else:
                shutil.copy2(unrelaxed_model, final_model)
            complex_paths[oligo_name] = final_model
        return StandardScores(
            names=list(inputs.keys()),
            plddt=plddt,
            ptm=None,
            pae=None,
            iptm=None,
            per_proto_scores=plddt,
            pde=None,
            chain_pair_scores=None,
            complex_paths=complex_paths,
        )

    def prepare_input(self, inputs: dict[str, Input], runtime_config=None):
        result = {}
        step = runtime_config.get("step", 0)
        for oligo_name, input in inputs.items():
            if len(input.chains) > 1:
                raise ValueError(
                    "RhoFold currently only supports single RNA chains, got multiple chains."
                )
            sequence = [
                chain.sequence for chain in input.chains if isinstance(chain, RnaChainTypes)
            ]
            if not sequence:
                raise ValueError(f"RhoFold input {oligo_name!r} does not contain an RNA chain.")

            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".a3m", delete=False
            )
            tmp_file.write(f">{oligo_name}\n{sequence[0]}\n")
            tmp_file.flush()
            atexit.register(os.unlink, tmp_file.name)

            # self.get_blast(tmp_file)
            msa_tokens = get_msa_feature(msa_path=tmp_file.name, msa_depth=128)
            rna_fm_tokens = get_rna_fm_token(
                tmp_file.name,
            )

            result[oligo_name] = {
                "seq": sequence[0],
                "tokens": msa_tokens.unsqueeze(0),
                "rna_fm_tokens": rna_fm_tokens.unsqueeze(0),
            }
        return result

    def get_blast(self, input_fas, output_msa_path=None):
        databases = [
            f"{self.config.database_dpath}/rnacentral.fasta",
            f"{self.config.database_dpath}/nt",
        ]
        blast = BLASTN(binary_dpath=self.config.binary_dpath, databases=databases)
        blast.query(
            self.config.input_fas, output_msa_path, logging.getLogger("rhofold")
        )

    def run_function(self, input_feat):
        result = {}
        with torch.no_grad():
            for oligo_name, feat in input_feat.items():
                result[oligo_name] = self.model_runner(
                    tokens=feat["tokens"].to(self.config.device),
                    rna_fm_tokens=feat["rna_fm_tokens"].to(self.config.device),
                    seq=feat["seq"],
                )[-1]
        return result
