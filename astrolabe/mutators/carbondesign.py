import logging
import pathlib
import re
import shutil
import uuid
from typing import Any, Dict, List

import torch
from Bio.PDB import PDBIO, PDBParser, Select

from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import prepare_exclude_sites, sample_mutations_by_aa_probs
from astrolabe.structure_processors.base import run_nested_predocker_on_scores
from astrolabe.utils.constant import aa_order

CARBON_AA_ORDER = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


class _ChainSelect(Select):
    def __init__(self, allowed_chain: str):
        super().__init__()
        self.allowed_chain = allowed_chain

    def accept_chain(self, chain):
        return chain.id == self.allowed_chain


class CarbonDesignMutator(Mutator):
    name = "CarbonDesign"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=True,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
    )

    def __init__(self, sequence_model_runner, position_selector=None, predocker=None):
        self.runner = sequence_model_runner
        self.position_selector = position_selector
        self.predocker = predocker
        self.aa_order = aa_order[:20]
        self._aa_to_column = {aa: idx for idx, aa in enumerate(self.aa_order)}
        self._torch_reorder_index = torch.tensor(
            [CARBON_AA_ORDER.index(aa) for aa in self.aa_order], dtype=torch.long
        )
        self._pdb_parser = PDBParser(QUIET=True)
        self._inputs_root = pathlib.Path(self.runner.output_dir) / "inputs"
        self._inputs_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------- helpers ------------------------------ #
    @staticmethod
    def _sanitize(label: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))
        return sanitized.strip("_") or "oligo"

    def _renumber_chain(self, chain):
        """Ensure residue numbering is contiguous starting at 0 for protein residues."""
        counter = 0
        for residue in chain:
            hetflag, _, icode = residue.id
            if hetflag != " ":
                continue
            residue.id = (" ", counter, icode)
            counter += 1

    def _write_single_chain(self, structure, chain_id: str, out_path: pathlib.Path):
        chain = structure[chain_id]
        self._renumber_chain(chain)
        io = PDBIO()
        io.set_structure(structure)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        io.save(str(out_path), _ChainSelect(chain_id))

    def _cleanup(self, feature_dict: Dict[str, Any]) -> None:
        cleanup_path = feature_dict.get("cleanup_path")
        if cleanup_path is None:
            return
        shutil.rmtree(cleanup_path, ignore_errors=True)

    # ---------------------------- mutator API ----------------------------- #
    def compute_mutation(self, mutation_input_features: Dict[str, Dict[str, Any]]):
        mutation_output = {}
        for oligo_name, input_feat in mutation_input_features.items():
            chain_outputs = self.runner.design_sequences(
                data_dir=input_feat["data_dir"],
                name_idx=input_feat["name_idx"],
            )
            chain_probs: List[torch.Tensor] = []
            for chain_meta in input_feat["chain_metadata"]:
                chain_result = chain_outputs.get(chain_meta["name"])
                if chain_result is None:
                    raise RuntimeError(
                        f"Missing CarbonDesign output for chain {chain_meta['name']}"
                    )
                probs = chain_result["sampling_probs"].clone()
                if probs.shape[-1] != len(CARBON_AA_ORDER):
                    raise ValueError(
                        f"Expected {len(CARBON_AA_ORDER)} amino acid logits, "
                        f"got {probs.shape[-1]}"
                    )
                if probs.shape[1] < chain_meta["length"]:
                    raise ValueError(
                        "CarbonDesign output shorter than protomer length for "
                        f"{chain_meta['proto']}: {probs.shape[1]} < {chain_meta['length']}"
                    )
                trimmed = probs[:, : chain_meta["length"], :]
                reordered = trimmed.index_select(-1, self._torch_reorder_index)
                chain_probs.append(reordered)

            if not chain_probs:
                continue
            combined = torch.cat(chain_probs, dim=1)
            mutation_output[oligo_name] = {
                "sampling_probs": combined,
                "name": oligo_name,
                "feature_dict": input_feat,
            }
            self._cleanup(input_feat)
        return mutation_output

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        pdb_paths = {k: pathlib.Path(v) for k, v in evaluation_scores.complex_paths.items()}
        if self.predocker is not None:
            predocked_scores = run_nested_predocker_on_scores(
                self.predocker,
                evaluation_scores,
            )
            pdb_paths = {
                k: pathlib.Path(v)
                for k, v in predocked_scores.complex_paths.items()
            }

        mutation_input_features: Dict[str, Dict[str, Any]] = {}

        for oligo_name in evaluation_scores.names:
            sanitized = self._sanitize(oligo_name)
            job_dir = self._inputs_root / f"{sanitized}_{uuid.uuid4().hex}"
            structure_path = pdb_paths.get(oligo_name)
            if structure_path is None or not structure_path.exists():
                logging.warning("Skipping %s due to missing structure", oligo_name)
                continue
            structure = self._pdb_parser.get_structure(
                sanitized, str(structure_path)
            )
            try:
                model = next(structure.get_models())
            except StopIteration:
                logging.warning("No models found in %s; skipping", structure_path)
                continue

            chain_metadata = []
            for chain_index, proto in enumerate(oligo_name):
                if proto not in protomers:
                    continue
                proto_obj = protomers[proto]
                if proto_obj.type != ProtomerTypes.PROTEIN:
                    continue
                chain_id = int_id_to_str_id(chain_index + 1)
                chain_name = f"{sanitized}_{proto}_{chain_id}"
                out_path = job_dir / f"{chain_name}.pdb"
                self._write_single_chain(model, chain_id, out_path)
                chain_metadata.append(
                    {
                        "name": chain_name,
                        "path": out_path,
                        "length": proto_obj.length,
                        "proto": proto,
                    }
                )

            if not chain_metadata:
                shutil.rmtree(job_dir, ignore_errors=True)
                continue

            mutation_input_features[oligo_name] = {
                "data_dir": str(job_dir),
                "name_idx": [meta["name"] for meta in chain_metadata],
                "chain_metadata": chain_metadata,
                "cleanup_path": job_dir,
            }

        return mutation_input_features

    def prepare_mutation_output(
        self,
        mutation_output,
        protomers,
        evaluation_scores,
        runtime_config,
    ):
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            evaluation_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
        )

        for oligo_name, oligo_output in mutation_output.items():
            probs = oligo_output["sampling_probs"]
            offset = 0
            for proto in oligo_name:
                if proto not in protomers:
                    continue
                proto_obj = protomers[proto]
                if proto_obj.type != ProtomerTypes.PROTEIN:
                    continue
                proto_len = proto_obj.length
                proto_probs = probs[:, offset : offset + proto_len, :]
                frozen_positions = excluded_positions.get(proto, [])
                for pos in frozen_positions:
                    if pos >= proto_len:
                        continue
                    aa = proto_obj.sequence[pos]
                    aa_idx = self._aa_to_column.get(aa, None)
                    if aa_idx is None:
                        continue
                    proto_probs[:, pos, :] = 0.0
                    proto_probs[:, pos, aa_idx] = 1.0
                offset += proto_len

        sample_mutations_by_aa_probs(
            evaluation_scores,
            mutation_output,
            self.aa_order,
            self.mutation_weight,
            runtime_config=runtime_config,
        )
