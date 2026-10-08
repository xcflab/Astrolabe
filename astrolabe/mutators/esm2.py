from typing import Dict, Any

import esm2
import torch
import numpy as np
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import (
    prepare_exclude_sites,
    sample_mutations_by_aa_probs,
)


def extract_esm2_probs(model, batch_tokens, temperature=0.3):
    with torch.no_grad():
        logits = model(batch_tokens.cuda())["logits"]
        logits /= temperature
        logits = logits[..., 4:24]
        probs = torch.softmax(logits, dim=-1)
        # remove start <cls> and end <eos> token
        probs = probs[..., 1:-1, :]
    return {"sampling_probs": probs}


class ESM2Mutator(Mutator):
    name = "ESM2"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=False,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
    )

    def __init__(self, position_selector=None, predocker=None):
        model, self.alphabet = esm2.pretrained.esm2_t33_650M_UR50D()
        # use eval mode for deterministic mutation_output e.g. without random dropout
        model.eval()
        self.runner = model.cuda()
        self.predocker = predocker
        self.position_selector = position_selector
        self.aa_order = self.alphabet.all_toks[4:24]  # ESM2 uses a specific token order

    def compute_mutation(self, mutation_input_features):
        mutation_output = {}
        for oligo_name, input_feat in mutation_input_features.items():
            oligo_output = extract_esm2_probs(self.runner, **input_feat)
            oligo_output.update({"name": oligo_name, "feature_dict": input_feat})
            mutation_output[oligo_name] = oligo_output

        return mutation_output

    def prepare_mutation_input(
        self,
        protomer,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):

        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomer,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )
        mutation_input_features = {}
        protein_seqs = {
            proto: global_config.protomers[proto].sequence
            for proto in global_config.protomers
            if global_config.protomers[proto].type == ProtomerTypes.PROTEIN
        }
        batch_converter = self.alphabet.get_batch_converter()
        for oligo_name in evaluation_scores.names:
            sequence = "".join(
                protein_seqs[proto] for proto in oligo_name if proto in protein_seqs
            )

            fixed_indices = []
            offset = 0
            for proto in oligo_name:
                if proto in excluded_positions and proto in protein_seqs:
                    fixed_indices += [i + offset for i in excluded_positions[proto]]
                    offset += len(global_config.protomers[proto].sequence)

            masked_sequence = ["<mask>"] * len(sequence)

            leaving_indices = list(
                set(range(len(masked_sequence))) - set(fixed_indices)
            )

            mutation_quota = min(self.n_mutations(), len(leaving_indices))
            preserve_count = max(0, len(leaving_indices) - mutation_quota)
            to_sampled_fixed_indices = (
                np.random.choice(leaving_indices, preserve_count, replace=False).tolist()
                if preserve_count > 0
                else []
            )
            for i in fixed_indices + to_sampled_fixed_indices:
                masked_sequence[i] = sequence[i]
            _, _, batch_tokens = batch_converter(
                [(oligo_name, "".join(masked_sequence))]
            )
            mutation_input_features[oligo_name] = {
                "batch_tokens": batch_tokens,
            }

        return mutation_input_features

    def prepare_mutation_output(
        self, mutation_output, protomer, evaluation_scores, runtime_config
    ):
        sample_mutations_by_aa_probs(
            evaluation_scores,
            mutation_output,
            self.aa_order,
            self.mutation_weight,
            runtime_config=runtime_config,
        )
