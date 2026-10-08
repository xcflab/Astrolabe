from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import (
    prepare_exclude_sites,
    sample_mutations_by_aa_probs,
)
from astrolabe.utils.constant import aa_order as default_aa_order


def _ensure_vendored_esm3_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / "vendor" / "esm3"
    if vendor_root.exists():
        vendor_str = str(vendor_root)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)


class ESM3Mutator(Mutator):
    name = "ESM3"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=False,
        outputs_probabilities=True,
        outputs_scores=False,
        supports_transition_blending=True,
        supports_shared_selection_policy=True,
    )

    def __init__(
        self,
        model_name: str = "esm3-open",
        device: str | None = None,
        temperature: float = 0.3,
        position_selector=None,
        predocker=None,
    ) -> None:
        try:
            from esm3.models.esm3 import ESM3
            from esm3.sdk.api import ESMProtein, LogitsConfig
            from esm3.tokenization import get_esm3_model_tokenizers
        except ImportError:  # pragma: no cover - env dependent
            _ensure_vendored_esm3_on_path()
            try:
                from esm3.models.esm3 import ESM3
                from esm3.sdk.api import ESMProtein, LogitsConfig
                from esm3.tokenization import get_esm3_model_tokenizers
            except ImportError as exc:
                raise ImportError(
                    "ESM3Mutator requires the vendored `esm3` package and its Python deps "
                    "(notably `huggingface_hub`, `zstd`, and `transformers`)."
                ) from exc

        self._esm3_cls = ESM3
        self._esm_protein_cls = ESMProtein
        self._logits_config = LogitsConfig(sequence=True)

        self.model_name = model_name
        self.temperature = temperature
        self.position_selector = position_selector
        self.predocker = predocker

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        pinned_snapshot = os.environ.get("GH_ESM3_MODEL_SNAPSHOT")
        if pinned_snapshot:
            snapshot = Path(pinned_snapshot).expanduser().resolve()
            if not snapshot.is_dir():
                raise FileNotFoundError(f"Pinned ESM3 snapshot does not exist: {snapshot}")
            import esm3.pretrained as esm3_pretrained
            import esm3.utils.constants.esm3 as esm3_constants

            original_data_root = esm3_constants.data_root

            def pinned_data_root(model: str):
                if model.startswith("esm3"):
                    return snapshot
                return original_data_root(model)

            esm3_constants.data_root = pinned_data_root
            esm3_pretrained.data_root = pinned_data_root

        self.runner = self._esm3_cls.from_pretrained(model_name, device=self.device)
        self.runner.eval()
        self.runner = self.runner.to(self.device)

        tokenizers = get_esm3_model_tokenizers(model_name)
        self.sequence_tokenizer = tokenizers.sequence
        self.aa_order = list(default_aa_order)
        self._aa_token_ids = torch.tensor(
            [self.sequence_tokenizer.convert_tokens_to_ids(aa) for aa in self.aa_order],
            device=self.device,
        )

    def _build_protein_prompt(self, sequence: str) -> "ESMProtein":
        return self._esm_protein_cls(sequence=sequence)

    def _mutation_quota(self, sequence_length: int) -> int:
        base_mutations = getattr(global_config, "mutation_num", 0) or 0
        if isinstance(base_mutations, str):
            value = base_mutations.strip()
            if value.endswith("%"):
                try:
                    fraction = float(value[:-1]) / 100.0
                except ValueError:
                    fraction = 0.0
                base_mutations = fraction * sequence_length
            else:
                try:
                    base_mutations = float(value)
                except ValueError:
                    base_mutations = 0
        weight = getattr(self, "mutation_weight", 1.0)
        try:
            quota = int(round(base_mutations * weight))
        except TypeError:
            quota = int(base_mutations)
        return max(0, min(quota, sequence_length))

    def _mask_sequence(
        self,
        sequence: str,
        fixed_indices: List[int],
        target_mutations: int,
    ) -> str:
        masked = ["_"] * len(sequence)
        fixed = set(fixed_indices)
        available = list(sorted(set(range(len(sequence))) - fixed))
        n_masks = min(target_mutations, len(available))
        additional = max(len(sequence) - len(fixed) - n_masks, 0)

        if additional > 0 and available:
            sampled = np.random.choice(available, additional, replace=False).tolist()
            fixed.update(sampled)

        for idx in fixed:
            masked[idx] = sequence[idx]

        return "".join(masked)

    def compute_mutation(self, mutation_input_features: Dict[str, Dict[str, Any]]):
        mutation_output = {}
        logits_config = self._logits_config

        for oligo_name, input_feat in mutation_input_features.items():
            protein_prompt = input_feat["protein_prompt"]
            protein_tensor = self.runner.encode(protein_prompt)
            logits_output = self.runner.logits(
                protein_tensor, config=logits_config
            )
            seq_logits = logits_output.logits.sequence
            if seq_logits.dim() == 2:
                seq_logits = seq_logits.unsqueeze(0)

            seq_logits = seq_logits[:, 1:-1, :]
            aa_logits = seq_logits.index_select(-1, self._aa_token_ids)
            probs = torch.softmax(aa_logits / self.temperature, dim=-1)

            mutation_output[oligo_name] = {
                "sampling_probs": probs,
                "name": oligo_name,
                "feature_dict": input_feat,
            }

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

        protein_seqs = {
            proto: global_config.protomers[proto].sequence
            for proto in global_config.protomers
            if global_config.protomers[proto].type == ProtomerTypes.PROTEIN
        }

        masked_protomer_sequences = {}
        for proto, sequence in protein_seqs.items():
            mutation_quota = self._mutation_quota(sequence_length=len(sequence))
            masked_protomer_sequences[proto] = self._mask_sequence(
                sequence,
                excluded_positions.get(proto, []),
                mutation_quota,
            )

        mutation_input_features = {}

        for oligo_name in evaluation_scores.names:
            masked_sequence_parts = [
                masked_protomer_sequences[proto]
                for proto in oligo_name
                if proto in masked_protomer_sequences
            ]
            if not masked_sequence_parts:
                continue

            masked_sequence = "".join(masked_sequence_parts)

            mutation_input_features[oligo_name] = {
                "protein_prompt": self._build_protein_prompt(masked_sequence),
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
