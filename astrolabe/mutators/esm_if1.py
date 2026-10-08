from typing import Tuple, Any, Dict

import esm2
import torch
import torch.nn.functional as F
from biotite.structure import filter_peptide_backbone
from biotite.structure import get_chains
from biotite.structure.io import pdbx, pdb
from esm2.inverse_folding.util import get_atom_coords_residuewise, CoordBatchConverter

from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import (
    prepare_exclude_sites,
    sample_mutations_by_aa_probs,
)
from astrolabe.structure_processors.base import run_nested_predocker_on_scores


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith("cif"):
        with open(fpath) as fin:
            pdbxf = pdbx.CIFFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith("pdb"):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_peptide_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def esm_extract_probs(
    model, coords, partial_seq=None, temperature=1.0, confidence=None
):
    with torch.no_grad():
        L = len(coords)
        # Convert to batch format
        device = next(model.parameters()).device
        batch_converter = CoordBatchConverter(model.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = batch_converter(
            [(coords, confidence, None)], device=device
        )
        # Start with prepend token
        mask_idx = model.decoder.dictionary.get_idx("<mask>")
        sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=int).to(device)
        sampled_tokens[0, 0] = model.decoder.dictionary.get_idx("<cath>")
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i + 1] = model.decoder.dictionary.get_idx(c)

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Run encoder only once
        encoder_out = model.encoder(batch_coords, padding_mask, confidence)
        complex_probs = torch.zeros((L, 20), dtype=torch.float32).to(device)
        # Decode one token at a time
        for i in range(1, L + 1):
            if sampled_tokens[0, i] != mask_idx:
                complex_probs[i - 1, -1] = 1
                continue
            logits, _ = model.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            complex_probs[i - 1] = probs[:, 4:24]
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]
    return {"sampling_probs": complex_probs.unsqueeze(0).cpu()}


class ESMIF1Mutator(Mutator):
    name = "ESMIF"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=True,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
    )

    def __init__(self, position_selector=None, predocker=None):
        self.predocker = None
        model, self.alphabet = esm2.pretrained.esm_if1_gvp4_t16_142M_UR50()
        # use eval mode for deterministic mutation_output e.g. without random dropout
        model.eval()
        self.predocker = predocker
        self.runner = model.cuda()
        self.position_selector = position_selector
        self.aa_order = self.alphabet.all_toks[4:24]

    def compute_mutation(self, mutation_input_features: Tuple[Any, Any]):
        mutation_output = {}
        for oligo_name, input_feat in mutation_input_features.items():
            oligo_output = esm_extract_probs(model=self.runner, **input_feat)
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
        pdb_paths = evaluation_scores.complex_paths

        if self.predocker is not None:
            predocked_scores = run_nested_predocker_on_scores(
                self.predocker,
                evaluation_scores,
            )
            pdb_paths = predocked_scores.complex_paths
        for oligo_name in evaluation_scores.names:
            fpath = pdb_paths[oligo_name]
            chains = [
                int_id_to_str_id(i + 1)
                for i, proto in enumerate(oligo_name)
                if proto in protein_seqs
            ]

            structure = load_structure(str(fpath), chains)
            coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
            sequence = "".join(
                protein_seqs[chain] for chain in oligo_name if chain in protein_seqs
            )

            fixed_indices = []
            offset = 0
            for proto in oligo_name:
                if proto in excluded_positions and proto in protein_seqs:
                    fixed_indices += [i + offset for i in excluded_positions[proto]]
                    offset += len(protein_seqs[proto])

            assert len(sequence) == len(coords)

            masked_sequence = ["<mask>"] * coords.shape[0]
            for i in fixed_indices:
                masked_sequence[i] = sequence[i]
            mutation_input_features[oligo_name] = {
                "coords": coords,
                "partial_seq": masked_sequence,
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
