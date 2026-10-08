from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser
from jax.lib import xla_bridge

from alphafold.common import protein

from astrolabe.core import global_config
from astrolabe.core.cyclization import build_cyclic_offset_matrix
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.alphafold import (
    make_AF2_features,
    make_AFM_features,
)
from astrolabe.utils.constant import DEFAULT_SEEDS

logger = logging.getLogger(__name__)

try:
    backend = xla_bridge.get_backend().platform
    logger.info("JAX backend: %s", backend)
except Exception:  # pragma: no cover - logging only
    logger.debug("Unable to determine JAX backend platform", exc_info=True)


class AlphaFold2Evaluator(Evaluator):
    """AlphaFold2 evaluator using the same prepare/run/output interface as AF3."""

    def __init__(
        self,
        model_runners: Dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> None:
        self.model_runners = model_runners
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._raw_features: Dict[str, Dict[str, np.ndarray]] = {}
        self._seeds: Dict[str, List[int]] = {}
        self._is_multimer: Dict[str, bool] = {}
        self._inputs: Dict[str, Any] = {}

    @staticmethod
    def _resolve_initial_guess_cfg(
        runner: Any, oligo_name: str | None = None
    ) -> Dict[str, Any] | None:
        cfg = getattr(runner, "_gh_initial_guess_cfg", None)
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            cfg_dict = dict(cfg)
        elif hasattr(cfg, "items"):
            cfg_dict = dict(cfg.items())
        else:
            raise TypeError(
                "evaluator.alphafold2.initial_guess must be a mapping when provided"
            )

        if oligo_name and isinstance(cfg_dict.get(oligo_name), Mapping):
            base = {
                key: value
                for key, value in cfg_dict.items()
                if key not in {oligo_name, "references", "paths", "chain_maps"}
            }
            base.update(dict(cfg_dict[oligo_name]))
            return base
        if oligo_name and isinstance(cfg_dict.get("references"), Mapping):
            references = cfg_dict["references"]
            if isinstance(references.get(oligo_name), Mapping):
                base = {
                    key: value
                    for key, value in cfg_dict.items()
                    if key != "references"
                }
                base.update(dict(references[oligo_name]))
                return base
        if oligo_name and isinstance(cfg_dict.get("paths"), Mapping):
            paths = cfg_dict["paths"]
            if oligo_name in paths:
                cfg_dict = dict(cfg_dict)
                cfg_dict["path"] = paths[oligo_name]
        if oligo_name and isinstance(cfg_dict.get("chain_maps"), Mapping):
            chain_maps = cfg_dict["chain_maps"]
            if isinstance(chain_maps.get(oligo_name), Mapping):
                cfg_dict = dict(cfg_dict)
                cfg_dict["chain_map"] = chain_maps[oligo_name]
        return cfg_dict

    @staticmethod
    def _initial_guess_enabled(cfg: Dict[str, Any] | None) -> bool:
        if not cfg:
            return False
        enabled = cfg.get("enabled")
        if enabled is None:
            return bool(cfg.get("path"))
        return bool(enabled)

    @staticmethod
    def _initial_guess_use_initial_prev_pos(cfg: Dict[str, Any] | None) -> bool:
        if not cfg:
            return False
        return bool(cfg.get("use_initial_prev_pos", True))

    @staticmethod
    def _initial_guess_use_template_features(cfg: Dict[str, Any] | None) -> bool:
        if not cfg:
            return False
        return bool(cfg.get("use_template_features", True))

    @staticmethod
    def _load_initial_guess_structure(path: Path):
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if ".cif" in suffixes or ".mmcif" in suffixes:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        return parser.get_structure("af2_initial_guess", str(path))

    @classmethod
    def _initial_guess_source(
        cls,
        *,
        cfg: Dict[str, Any],
        oligo_name: str,
    ):
        path_value = cfg.get("path") or cfg.get("init_guess_path")
        if path_value is None:
            raise ValueError(
                f"AF2 initial_guess is enabled for '{oligo_name}' but no path was provided."
            )
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"AF2 initial_guess path not found: {path}")

        structure = cls._load_initial_guess_structure(path)
        models = list(structure.get_models())
        if len(models) != 1:
            raise ValueError(
                f"AF2 initial_guess requires a single-model structure, got {len(models)}: {path}"
            )
        chain_map = cfg.get("chain_map") or {}
        if not isinstance(chain_map, Mapping):
            raise TypeError("evaluator.alphafold2.initial_guess.chain_map must be a mapping")
        return (
            path,
            models[0],
            chain_map,
            bool(cfg.get("allow_missing_chains", False)),
            bool(cfg.get("allow_missing_residues", False)),
        )

    @staticmethod
    def _iter_protein_residues(chain):
        for residue in chain.get_residues():
            hetflag, _, icode = residue.id
            if hetflag.strip():
                continue
            if icode.strip():
                raise ValueError(
                    "AF2 initial_guess structures with insertion codes are not supported "
                    f"(chain={chain.id!r}, residue={residue.id!r})."
                )
            yield residue

    @staticmethod
    def _residue_sequence(residues: Sequence[Any]) -> str:
        seq = []
        for residue in residues:
            resname = str(residue.resname).upper()
            seq.append(protein.residue_constants.restype_3to1.get(resname, "X"))
        return "".join(seq)

    @staticmethod
    def _fill_positions_from_residues(
        positions: np.ndarray,
        residues: Sequence[Any],
        target_indices: Sequence[int],
        masks: np.ndarray | None = None,
    ) -> int:
        filled = 0
        for residue, target_index in zip(residues, target_indices):
            for atom_name, atom_index in protein.residue_constants.atom_order.items():
                if atom_name in residue:
                    positions[target_index, atom_index] = residue[atom_name].get_coord().astype(
                        np.float32
                    )
                    if masks is not None:
                        masks[target_index, atom_index] = 1.0
            filled += 1
        return filled

    @staticmethod
    def _fill_template_metadata_from_residues(
        *,
        residues: Sequence[Any],
        target_indices: Sequence[int],
        template_aatype: np.ndarray,
        template_sequence: list[str],
        template_confidence: np.ndarray,
    ) -> None:
        aa_to_idx = protein.residue_constants.HHBLITS_AA_TO_ID
        unknown_idx = aa_to_idx["X"]
        for residue, target_index in zip(residues, target_indices):
            aa = protein.residue_constants.restype_3to1.get(
                str(residue.resname).upper(),
                "X",
            )
            aa_idx = aa_to_idx.get(aa, unknown_idx)
            template_aatype[target_index, :] = 0.0
            template_aatype[target_index, aa_idx] = 1.0
            template_sequence[target_index] = aa
            template_confidence[target_index] = 1.0

    @staticmethod
    def _empty_initial_guess_template_features(sequence: str) -> dict[str, np.ndarray]:
        atom_type_num = protein.residue_constants.atom_type_num
        hhblits_aatype_count = (
            max(protein.residue_constants.HHBLITS_AA_TO_ID.values()) + 1
        )
        template_aatype = np.zeros(
            (1, len(sequence), hhblits_aatype_count),
            dtype=np.float32,
        )
        template_aatype[0, :, protein.residue_constants.HHBLITS_AA_TO_ID["-"]] = 1.0
        return {
            "template_all_atom_positions": np.zeros(
                (1, len(sequence), atom_type_num, 3),
                dtype=np.float32,
            ),
            "template_all_atom_masks": np.zeros(
                (1, len(sequence), atom_type_num),
                dtype=np.float32,
            ),
            "template_sequence": np.array(
                [("-" * len(sequence)).encode()],
                dtype=object,
            ),
            "template_aatype": template_aatype,
            "template_confidence_scores": -np.ones((1, len(sequence)), dtype=np.float32),
            "template_domain_names": np.array(["none".encode()], dtype=object),
            "template_release_date": np.array(["1970-01-01".encode()], dtype=object),
            "template_sum_probs": np.zeros((1,), dtype=np.float32),
        }

    @classmethod
    def _initial_guess_chain_atom_arrays(
        cls,
        *,
        residues: Sequence[Any],
        expected_sequence: str,
        atom_type_num: int,
        oligo_name: str,
        chain_id: str,
        source_chain_id: str,
        path: Path,
        allow_missing_residues: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        expected = len(expected_sequence)
        positions = np.zeros((expected, atom_type_num, 3), dtype=np.float32)
        masks = np.zeros((expected, atom_type_num), dtype=np.float32)
        hhblits_aatype_count = (
            max(protein.residue_constants.HHBLITS_AA_TO_ID.values()) + 1
        )
        template_aatype = np.zeros(
            (expected, hhblits_aatype_count),
            dtype=np.float32,
        )
        template_aatype[:, protein.residue_constants.HHBLITS_AA_TO_ID["-"]] = 1.0
        template_sequence = ["-"] * expected
        template_confidence = np.zeros(expected, dtype=np.float32)
        if len(residues) == expected:
            target_indices = range(expected)
            cls._fill_positions_from_residues(
                positions,
                residues,
                target_indices,
                masks=masks,
            )
            cls._fill_template_metadata_from_residues(
                residues=residues,
                target_indices=target_indices,
                template_aatype=template_aatype,
                template_sequence=template_sequence,
                template_confidence=template_confidence,
            )
            return (
                positions,
                masks,
                template_aatype,
                "".join(template_sequence),
                template_confidence,
            )

        if not allow_missing_residues:
            raise ValueError(
                "AF2 initial_guess residue count mismatch for "
                f"oligo '{oligo_name}' chain {chain_id!r}: expected {expected}, "
                f"got {len(residues)} from source chain {source_chain_id!r} in {path}. "
                "Set allow_missing_residues=true to sequence-align and zero-fill "
                "missing residues."
            )

        source_sequence = cls._residue_sequence(residues)
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -2.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5
        alignments = aligner.align(expected_sequence, source_sequence)
        if not alignments:
            raise ValueError(
                f"AF2 initial_guess could not align source chain {source_chain_id!r} "
                f"to oligo '{oligo_name}' chain {chain_id!r}."
            )
        alignment = alignments[0]
        target_indices: list[int] = []
        source_indices: list[int] = []
        for target_block, source_block in zip(alignment.aligned[0], alignment.aligned[1]):
            target_start, target_end = map(int, target_block)
            source_start, source_end = map(int, source_block)
            span = min(target_end - target_start, source_end - source_start)
            for offset in range(span):
                target_index = target_start + offset
                source_index = source_start + offset
                if expected_sequence[target_index] == source_sequence[source_index]:
                    target_indices.append(target_index)
                    source_indices.append(source_index)

        if not target_indices:
            raise ValueError(
                f"AF2 initial_guess alignment placed no matching residues for "
                f"oligo '{oligo_name}' chain {chain_id!r} from {path}."
            )
        mapped_residues = [residues[index] for index in source_indices]
        cls._fill_positions_from_residues(
            positions,
            mapped_residues,
            target_indices,
            masks=masks,
        )
        cls._fill_template_metadata_from_residues(
            residues=mapped_residues,
            target_indices=target_indices,
            template_aatype=template_aatype,
            template_sequence=template_sequence,
            template_confidence=template_confidence,
        )
        logger.info(
            "AF2 initial_guess sequence-aligned chain %s/%s from %s: filled=%s/%s",
            oligo_name,
            chain_id,
            path,
            len(target_indices),
            expected,
        )
        return (
            positions,
            masks,
            template_aatype,
            "".join(template_sequence),
            template_confidence,
        )

    @classmethod
    def _initial_guess_chain_positions(cls, **kwargs) -> np.ndarray:
        positions, _, _, _, _ = cls._initial_guess_chain_atom_arrays(**kwargs)
        return positions

    @staticmethod
    def _initial_guess_source_chain_id(
        *,
        chain: Any,
        proto_name: str,
        chain_map: Mapping,
    ) -> Any:
        if str(chain.id) in chain_map:
            return chain_map[str(chain.id)]
        if proto_name in chain_map:
            return chain_map[proto_name]
        return str(chain.id)

    @classmethod
    def _initial_guess_positions(
        cls,
        *,
        cfg: Dict[str, Any],
        oligo_name: str,
        input_obj: Any,
    ) -> np.ndarray:
        path, model, chain_map, allow_missing_chains, allow_missing_residues = (
            cls._initial_guess_source(cfg=cfg, oligo_name=oligo_name)
        )

        atom_type_num = protein.residue_constants.atom_type_num
        all_positions: list[np.ndarray] = []
        protein_chains = list(input_obj.protein_chains)
        proto_names = list(getattr(input_obj, "name", "") or "")
        for index, chain in enumerate(protein_chains):
            proto_name = proto_names[index] if index < len(proto_names) else str(chain.id)
            source_chain_id = cls._initial_guess_source_chain_id(
                chain=chain,
                proto_name=proto_name,
                chain_map=chain_map,
            )
            expected = len(chain.sequence)
            if source_chain_id in {None, "", "-"}:
                if not allow_missing_chains:
                    raise ValueError(
                        "AF2 initial_guess chain_map explicitly omitted source chain "
                        f"for oligo '{oligo_name}' chain {chain.id!r}/{proto_name!r}; "
                        "set allow_missing_chains=true to zero-fill omitted chains."
                    )
                all_positions.append(
                    np.zeros((expected, atom_type_num, 3), dtype=np.float32)
                )
                continue

            if source_chain_id not in model:
                raise ValueError(
                    "AF2 initial_guess chain_map could not resolve source chain "
                    f"{source_chain_id!r} for oligo '{oligo_name}' chain "
                    f"{chain.id!r}/{proto_name!r} in {path}."
                )
            residues = list(cls._iter_protein_residues(model[source_chain_id]))
            positions = cls._initial_guess_chain_positions(
                residues=residues,
                expected_sequence=str(chain.sequence),
                atom_type_num=atom_type_num,
                oligo_name=oligo_name,
                chain_id=str(chain.id),
                source_chain_id=str(source_chain_id),
                path=path,
                allow_missing_residues=allow_missing_residues,
            )
            all_positions.append(positions)

        if not all_positions:
            raise ValueError(f"AF2 initial_guess for '{oligo_name}' has no protein chains.")
        return np.concatenate(all_positions, axis=0).astype(np.float32, copy=False)

    @classmethod
    def _initial_guess_template_features_by_chain(
        cls,
        *,
        cfg: Dict[str, Any],
        oligo_name: str,
        input_obj: Any,
    ) -> dict[str, dict[str, np.ndarray]]:
        path, model, chain_map, allow_missing_chains, allow_missing_residues = (
            cls._initial_guess_source(cfg=cfg, oligo_name=oligo_name)
        )
        atom_type_num = protein.residue_constants.atom_type_num
        protein_chains = list(input_obj.protein_chains)
        proto_names = list(getattr(input_obj, "name", "") or "")
        feature_by_chain: dict[str, dict[str, np.ndarray]] = {}
        for index, chain in enumerate(protein_chains):
            proto_name = proto_names[index] if index < len(proto_names) else str(chain.id)
            source_chain_id = cls._initial_guess_source_chain_id(
                chain=chain,
                proto_name=proto_name,
                chain_map=chain_map,
            )
            if source_chain_id in {None, "", "-"}:
                if not allow_missing_chains:
                    raise ValueError(
                        "AF2 initial_guess chain_map explicitly omitted source chain "
                        f"for oligo '{oligo_name}' chain {chain.id!r}/{proto_name!r}; "
                        "set allow_missing_chains=true to zero-fill omitted chains."
                    )
                feature_by_chain[str(chain.id)] = cls._empty_initial_guess_template_features(
                    str(chain.sequence)
                )
                continue
            if source_chain_id not in model:
                raise ValueError(
                    "AF2 initial_guess chain_map could not resolve source chain "
                    f"{source_chain_id!r} for oligo '{oligo_name}' chain "
                    f"{chain.id!r}/{proto_name!r} in {path}."
                )
            residues = list(cls._iter_protein_residues(model[source_chain_id]))
            positions, masks, template_aatype, template_sequence, confidence = (
                cls._initial_guess_chain_atom_arrays(
                    residues=residues,
                    expected_sequence=str(chain.sequence),
                    atom_type_num=atom_type_num,
                    oligo_name=oligo_name,
                    chain_id=str(chain.id),
                    source_chain_id=str(source_chain_id),
                    path=path,
                    allow_missing_residues=allow_missing_residues,
                )
            )
            feature_by_chain[str(chain.id)] = {
                "template_all_atom_positions": positions[None],
                "template_all_atom_masks": masks[None],
                "template_sequence": np.array([template_sequence.encode()], dtype=object),
                "template_aatype": template_aatype[None],
                "template_confidence_scores": confidence[None],
                "template_domain_names": np.array(
                    [f"initial_guess_{source_chain_id}".encode()],
                    dtype=object,
                ),
                "template_release_date": np.array(["1970-01-01".encode()], dtype=object),
                "template_sum_probs": np.array([1.0], dtype=np.float32),
            }
        return feature_by_chain

    @classmethod
    def _maybe_add_initial_guess(
        cls,
        processed_features: Dict[str, np.ndarray],
        *,
        cfg: Dict[str, Any] | None,
        oligo_name: str,
        input_obj: Any,
    ) -> bool:
        if not cls._initial_guess_enabled(cfg):
            return False
        if not cls._initial_guess_use_initial_prev_pos(cfg):
            logger.info("AF2 initial_prev_pos injection disabled for %s", oligo_name)
            return False
        positions = cls._initial_guess_positions(
            cfg=cfg or {}, oligo_name=oligo_name, input_obj=input_obj
        )
        processed_features["initial_prev_pos"] = positions
        logger.info(
            "AF2 initial_guess injected for %s: path=%s residues=%s",
            oligo_name,
            (cfg or {}).get("path") or (cfg or {}).get("init_guess_path"),
            positions.shape[0],
        )
        return True

    @staticmethod
    def _maybe_add_multichain_residue_index_offsets(
        processed_features: Dict[str, np.ndarray],
        *,
        input_obj: Any,
        multimer_mode: bool,
        offset: int = 200,
        oligo_name: str | None = None,
    ) -> bool:
        if multimer_mode:
            return False
        protein_chains = [chain for chain in input_obj.protein_chains]
        chain_lengths = [len(chain.sequence) for chain in protein_chains]
        if len(chain_lengths) <= 1:
            return False
        if "residue_index" not in processed_features:
            raise KeyError("Cannot apply multi-chain residue_index offsets: feature missing")
        original = np.asarray(processed_features["residue_index"])
        total_length = sum(chain_lengths)
        if original.shape and original.shape[-1] == total_length:
            updated = original.copy()
        elif original.size == total_length:
            updated = original.reshape(total_length).copy()
        else:
            raise ValueError(
                "Cannot apply multi-chain residue_index offsets: residue_index shape "
                f"{original.shape} is incompatible with total protein length {total_length}"
            )
        starts = np.cumsum([0] + chain_lengths[:-1])
        for chain_idx, (start, length) in enumerate(zip(starts, chain_lengths)):
            if chain_idx == 0:
                continue
            updated[..., start : start + length] += chain_idx * offset
        processed_features["residue_index"] = updated.reshape(original.shape).astype(
            original.dtype, copy=False
        )
        logger.info(
            "AF2 automatic multi-chain residue_index offsets applied for %s: "
            "chain_lengths=%s offset=%s",
            oligo_name or "<unknown>",
            chain_lengths,
            offset,
        )
        return True

    @staticmethod
    def _select_query_residue_index_jumps(
        cfg: Dict[str, Any] | None,
        *,
        oligo_name: str,
    ) -> list[dict[str, int]]:
        if not cfg:
            return []
        raw = cfg.get("query_residue_index_jumps") or cfg.get("residue_index_jumps")
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            raw = raw.get(oligo_name, raw.get("*", raw.get("default", [])))
        if isinstance(raw, Mapping):
            raw = [raw]
        specs: list[dict[str, int]] = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                raise TypeError("query_residue_index_jumps entries must be mappings")
            after = item.get("after") or item.get("jump_after") or item.get("break_after")
            offset = item.get("offset") or item.get("jump") or item.get("gap")
            if after is None or offset is None:
                raise ValueError("query_residue_index_jumps entries require after and offset")
            specs.append({"after": int(after), "offset": int(offset)})
        return specs

    @classmethod
    def _maybe_add_query_residue_index_jumps(
        cls,
        processed_features: Dict[str, np.ndarray],
        *,
        cfg: Dict[str, Any] | None,
        oligo_name: str,
        input_obj: Any,
    ) -> bool:
        specs = cls._select_query_residue_index_jumps(cfg, oligo_name=oligo_name)
        if not specs:
            return False
        if "residue_index" not in processed_features:
            raise KeyError("Cannot apply query_residue_index_jumps: residue_index feature missing")
        original = np.asarray(processed_features["residue_index"])
        protein_chains = [chain for chain in input_obj.protein_chains]
        chain_lengths = [len(chain.sequence) for chain in protein_chains]
        total_length = sum(chain_lengths)
        if original.shape and original.shape[-1] == total_length:
            updated = original.copy()
        elif original.size == total_length:
            updated = original.reshape(total_length).copy()
        else:
            raise ValueError(
                "Cannot apply query_residue_index_jumps: residue_index shape "
                f"{original.shape} is incompatible with total protein length {total_length}"
            )
        starts = np.cumsum([0] + chain_lengths[:-1])
        for spec in specs:
            after = spec["after"]
            offset = spec["offset"]
            for start, length in zip(starts, chain_lengths):
                if after <= 0 or after >= length:
                    raise ValueError(
                        f"query_residue_index_jumps after={after} is outside chain length {length}"
                    )
                updated[..., start + after : start + length] += offset
        processed_features["residue_index"] = updated.reshape(original.shape).astype(original.dtype, copy=False)
        logger.info(
            "AF2 query residue_index jump applied for %s: specs=%s",
            oligo_name,
            specs,
        )
        return True

    @staticmethod
    def _normalise_seed_input(seed_value: Any) -> List[int]:
        """Convert a variety of seed specifications into a concrete list."""
        if seed_value is None:
            seed_value = DEFAULT_SEEDS
        if isinstance(seed_value, int):
            return [int(seed_value)]
        if isinstance(seed_value, Iterable):
            seeds = [int(seed) for seed in seed_value]
            if not seeds:
                raise ValueError("random_seed iterable must not be empty")
            return seeds
        raise TypeError("random_seed must be an int or an iterable of ints")

    @staticmethod
    def _build_feature_dict(
        input_obj,
        runner,
        initial_guess_cfg: Dict[str, Any] | None = None,
        oligo_name: str | None = None,
    ):
        """Construct AlphaFold2 features using the official helpers."""
        protein_chains = [chain for chain in input_obj.protein_chains]
        if not protein_chains:
            raise ValueError("AlphaFold evaluator requires at least one protein chain.")
        is_multimer = getattr(runner, "multimer_mode", False)
        initial_guess_templates = {}
        if (
            AlphaFold2Evaluator._initial_guess_enabled(initial_guess_cfg)
            and AlphaFold2Evaluator._initial_guess_use_template_features(initial_guess_cfg)
        ):
            if oligo_name is None:
                raise ValueError("oligo_name is required when AF2 initial_guess is enabled.")
            initial_guess_templates = (
                AlphaFold2Evaluator._initial_guess_template_features_by_chain(
                    cfg=initial_guess_cfg or {},
                    oligo_name=oligo_name,
                    input_obj=input_obj,
                )
            )
        if is_multimer:
            chain_id_map = {
                chain.id: SimpleNamespace(sequence=chain.sequence)
                for chain in protein_chains
            }
            unique_sequences = {chain.sequence for chain in protein_chains}
            chain_homology = {}
            for chain in protein_chains:
                templates = getattr(chain, "templates", None)
                chain_data = {
                    "unpaired_msa": getattr(chain, "unpaired_msa", None),
                    "paired_msa": getattr(chain, "paired_msa", None),
                    "templates": templates,
                }
                if not templates and str(chain.id) in initial_guess_templates:
                    chain_data["template_features"] = initial_guess_templates[str(chain.id)]
                chain_homology[chain.id] = chain_data
            feature_dict = make_AFM_features(
                chain_id_map=chain_id_map,
                is_homomer_or_monomer=len(unique_sequences) == 1,
                chain_homology=chain_homology,
            )
        else:
            query_sequence = "".join(chain.sequence for chain in protein_chains)
            feature_dict = make_AF2_features(query_sequence)
            if initial_guess_templates:
                merged_templates = [
                    initial_guess_templates[str(chain.id)]
                    for chain in protein_chains
                    if str(chain.id) in initial_guess_templates
                ]
                if len(merged_templates) == len(protein_chains):
                    for key in (
                        "template_all_atom_positions",
                        "template_all_atom_masks",
                        "template_aatype",
                        "template_confidence_scores",
                    ):
                        feature_dict[key] = np.concatenate(
                            [template[key] for template in merged_templates],
                            axis=1,
                        )
                    feature_dict["template_sequence"] = np.array(
                        [
                            "".join(
                                template["template_sequence"][0].decode()
                                for template in merged_templates
                            ).encode()
                        ],
                        dtype=object,
                    )
                    feature_dict["template_domain_names"] = np.array(
                        [f"initial_guess_{oligo_name}".encode()],
                        dtype=object,
                    )
                    feature_dict["template_release_date"] = np.array(
                        ["1970-01-01".encode()],
                        dtype=object,
                    )
                    feature_dict["template_sum_probs"] = np.array([1.0], dtype=np.float32)
        return feature_dict, is_multimer

    @staticmethod
    def _apply_plddt_to_b_factors(prot: protein.Protein, prediction: Dict[str, Any]) -> protein.Protein:
        """Return a protein whose B-factors encode per-residue pLDDT."""

        if prediction is None or "plddt" not in prediction:
            return prot

        plddt = prediction["plddt"]
        if plddt is None:
            return prot

        plddt = np.asarray(plddt, dtype=np.float32).reshape(-1)
        num_res = prot.aatype.shape[0]
        if plddt.shape[0] != num_res:
            logger.warning(
                "Cannot map pLDDT to B-factors: residue count mismatch (expected %s, got %s)",
                num_res,
                plddt.shape[0],
            )
            return prot

        tiled_plddt = np.repeat(
            plddt[:, None],
            protein.residue_constants.atom_type_num,
            axis=-1,
        ).astype(np.float32, copy=False)

        return protein.Protein(
            atom_positions=prot.atom_positions,
            atom_mask=prot.atom_mask,
            aatype=prot.aatype,
            residue_index=prot.residue_index,
            chain_index=prot.chain_index,
            b_factors=tiled_plddt,
        )

    def prepare_input(self, inputs, runtime_config=None):
        """Prepare AlphaFold2 features while keeping the AF3-style interface."""
        runtime_config = runtime_config or {}

        self._raw_features.clear()
        self._seeds.clear()
        self._is_multimer.clear()
        self._inputs = inputs

        global_seed = runtime_config.get("random_seed") if hasattr(runtime_config, "get") else None

        for oligo_name, input_obj in inputs.items():
            if any(
                global_config.protomers[proto].type != ProtomerTypes.PROTEIN
                for proto in oligo_name
            ):
                raise ValueError(
                    "AlphaFold evaluator currently supports protein-only oligomers."
            )

            runner = self.model_runners[oligo_name]
            initial_guess_cfg = self._resolve_initial_guess_cfg(runner, oligo_name)
            feature_dict, is_multimer = self._build_feature_dict(
                input_obj,
                runner,
                initial_guess_cfg=initial_guess_cfg,
                oligo_name=oligo_name,
            )
            self._raw_features[oligo_name] = feature_dict
            self._is_multimer[oligo_name] = is_multimer

            per_oligo_cfg = None
            if hasattr(runtime_config, "get"):
                per_oligo_cfg = runtime_config.get(oligo_name)
            seed_value = None
            if per_oligo_cfg and hasattr(per_oligo_cfg, "get"):
                seed_value = per_oligo_cfg.get("random_seed")
            if seed_value is None:
                if isinstance(global_seed, dict):
                    seed_value = global_seed.get(oligo_name)
                else:
                    seed_value = global_seed
            self._seeds[oligo_name] = self._normalise_seed_input(seed_value)

        return inputs

    @staticmethod
    def _maybe_add_cyclic_offset(processed_features, input_obj) -> None:
        cyclic_chain_ids = tuple(
            str(chain_id)
            for chain_id in (getattr(input_obj, "cyclic_chain_ids", None) or ())
        )
        cyclic_relative_patches = tuple(
            getattr(input_obj, "cyclic_relative_patches", None) or ()
        )
        if not cyclic_chain_ids and not cyclic_relative_patches:
            return

        protein_chains = [chain for chain in input_obj.protein_chains]
        if not protein_chains:
            return

        residue_index = np.asarray(processed_features.get("residue_index"))
        if residue_index.size == 0:
            return

        processed_features["cyclic_offset"] = build_cyclic_offset_matrix(
            residue_index.reshape(-1),
            chain_lengths=[len(chain.sequence) for chain in protein_chains],
            cyclic_chain_ids=cyclic_chain_ids,
            chain_ids=[str(chain.id) for chain in protein_chains],
            cyclic_relative_patches=getattr(
                input_obj, "cyclic_relative_patches", None
            ),
        ).astype(np.int32, copy=False)

    def run_function(self, input_feat):
        """Execute AlphaFold2 inference for each oligomer."""
        outputs = {}

        for oligo_name in input_feat.keys():
            runner = self.model_runners[oligo_name]
            initial_guess_cfg = self._resolve_initial_guess_cfg(runner, oligo_name)
            seeds = self._seeds[oligo_name]
            raw_features = self._raw_features[oligo_name]
            is_multimer = self._is_multimer[oligo_name]
            remove_leading_dim = not getattr(runner, "multimer_mode", False)

            per_seed_results = []
            for seed in seeds:
                processed_features = runner.process_features(
                    deepcopy(raw_features), random_seed=seed
                )
                self._maybe_add_multichain_residue_index_offsets(
                    processed_features,
                    input_obj=self._inputs[oligo_name],
                    multimer_mode=is_multimer,
                    oligo_name=oligo_name,
                )
                self._maybe_add_query_residue_index_jumps(
                    processed_features,
                    cfg=initial_guess_cfg,
                    oligo_name=oligo_name,
                    input_obj=self._inputs[oligo_name],
                )
                self._maybe_add_cyclic_offset(
                    processed_features, self._inputs[oligo_name]
                )
                self._maybe_add_initial_guess(
                    processed_features,
                    cfg=initial_guess_cfg,
                    oligo_name=oligo_name,
                    input_obj=self._inputs[oligo_name],
                )
                prediction = runner.predict(processed_features, random_seed=seed)
                b_factors = np.tile(
                    prediction["plddt"][:, None],
                    (1, protein.residue_constants.atom_type_num),
                )
                predicted_protein = protein.from_prediction(
                    features=processed_features,
                    result=prediction,
                    b_factors=b_factors,
                    remove_leading_feature_dimension=remove_leading_dim,
                )
                chain_index = (
                    np.asarray(predicted_protein.chain_index)
                    .reshape(-1)
                    .astype(np.int32, copy=False)
                )
                if chain_index.size > 0:
                    chain_index = chain_index - chain_index.min()
                predicted_protein = protein.Protein(
                    atom_positions=predicted_protein.atom_positions,
                    atom_mask=predicted_protein.atom_mask,
                    aatype=np.asarray(predicted_protein.aatype).reshape(-1),
                    residue_index=np.asarray(predicted_protein.residue_index)
                    .reshape(-1)
                    .astype(np.int32, copy=False),
                    chain_index=chain_index,
                    b_factors=predicted_protein.b_factors,
                )
                if (
                    not getattr(runner, "multimer_mode", False)
                    and self._inputs
                    and len(self._inputs[oligo_name].chains) > 1
                ):
                    seq_lengths = [
                        len(chain.sequence)
                        for chain in self._inputs[oligo_name].chains
                        if hasattr(chain, "sequence")
                    ]
                    total_len = sum(seq_lengths)
                    if total_len == predicted_protein.chain_index.shape[0]:
                        new_chain_index = np.zeros_like(
                            predicted_protein.chain_index, dtype=np.int32
                        )
                        new_residue_index = np.zeros_like(
                            predicted_protein.residue_index, dtype=np.int32
                        )
                        start = 0
                        for chain_idx, chain_len in enumerate(seq_lengths):
                            end = start + chain_len
                            new_chain_index[start:end] = chain_idx
                            new_residue_index[start:end] = np.arange(
                                1, chain_len + 1, dtype=np.int32
                            )
                            start = end
                        predicted_protein = protein.Protein(
                            atom_positions=predicted_protein.atom_positions,
                            atom_mask=predicted_protein.atom_mask,
                            aatype=predicted_protein.aatype,
                            residue_index=new_residue_index,
                            chain_index=new_chain_index,
                            b_factors=predicted_protein.b_factors,
                        )

                predicted_protein = self._apply_plddt_to_b_factors(
                    predicted_protein, prediction
                )
                per_seed_results.append(
                    {
                        "seed": seed,
                        "prediction": prediction,
                        "protein": predicted_protein,
                    }
                )
            outputs[oligo_name] = per_seed_results

        return outputs

    def prepare_output(self, outputs, inputs, runtime_config):
        """Aggregate AlphaFold2 outputs into StandardScores."""
        names = list(inputs.keys())
        structure_dir = global_config.out_dir / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        per_proto_plddt: defaultdict[str, List[np.ndarray]] = defaultdict(list)
        plddt: Dict[str, List[np.ndarray]] = {}
        ptm: Dict[str, List[float]] = {}
        iptm: Dict[str, List[float]] = {}
        pae: Dict[str, List[np.ndarray]] = {}
        chain_pair_scores: Dict[str, Dict[str, List[np.ndarray]]] = {}
        complex_paths: Dict[str, Path] = {}

        for oligo_name in names:
            if oligo_name not in outputs:
                raise ValueError(f"Missing prediction outputs for '{oligo_name}'.")

            per_seed_results = outputs[oligo_name]
            if not per_seed_results:
                raise ValueError(f"No predictions produced for '{oligo_name}'.")

            sorted_results = sorted(
                per_seed_results,
                key=lambda r: r["prediction"].get("ranking_confidence", 0.0),
                reverse=True,
            )

            best_result = sorted_results[0]
            best_result["protein"] = self._apply_plddt_to_b_factors(
                best_result["protein"], best_result["prediction"]
            )
            jobname = f"step_{global_config.step:03d}_{oligo_name}"
            pdb_path = structure_dir / f"{jobname}_model.pdb"
            with open(pdb_path, "w") as handle:
                handle.write(protein.to_pdb(best_result["protein"]))
            complex_paths[oligo_name] = pdb_path

            plddt[oligo_name] = [
                np.asarray(result["prediction"]["plddt"], dtype=np.float32) / 100.0
                for result in sorted_results
            ]

            ptm_scores = [
                float(result["prediction"]["ptm"])
                for result in sorted_results
                if "ptm" in result["prediction"]
            ]
            if ptm_scores:
                ptm[oligo_name] = ptm_scores

            iptm_scores = [
                float(result["prediction"]["iptm"])
                for result in sorted_results
                if "iptm" in result["prediction"]
            ]
            if iptm_scores:
                iptm[oligo_name] = iptm_scores

            chain_pair_iptm = [
                np.asarray(
                    result["prediction"]["chain_pair_iptm"], dtype=np.float32
                )
                for result in sorted_results
                if "chain_pair_iptm" in result["prediction"]
            ]
            if chain_pair_iptm:
                chain_pair_scores[oligo_name] = {
                    "chain_pair_iptm": chain_pair_iptm
                }

            chain_ptm = [
                np.asarray(result["prediction"]["chain_ptm"], dtype=np.float32)
                for result in sorted_results
                if "chain_ptm" in result["prediction"]
            ]
            if chain_ptm:
                chain_pair_scores.setdefault(oligo_name, {})["chain_ptm"] = chain_ptm

            pae_matrices = [
                np.asarray(
                    result["prediction"]["predicted_aligned_error"], dtype=np.float32
                )
                for result in sorted_results
                if "predicted_aligned_error" in result["prediction"]
            ]
            if pae_matrices:
                pae[oligo_name] = pae_matrices

            for result in sorted_results:
                chain_plddt = (
                    np.asarray(result["prediction"]["plddt"], dtype=np.float32) / 100.0
                )
                res_idx = 0
                for proto_name, chain in zip(
                    inputs[oligo_name].name, inputs[oligo_name].chains
                ):
                    protomer = global_config.protomers.get(proto_name)
                    if protomer is None or protomer.type != ProtomerTypes.PROTEIN:
                        continue
                    chain_length = len(chain.sequence)
                    per_proto_plddt[proto_name].append(
                        chain_plddt[res_idx : res_idx + chain_length]
                    )
                    res_idx += chain_length

        scores = StandardScores(
            names=names,
            plddt=plddt,
            ptm=ptm or None,
            pae=pae or None,
            iptm=iptm or None,
            per_proto_scores=dict(per_proto_plddt),
            pde=None,
            chain_pair_scores=chain_pair_scores or None,
            complex_paths=complex_paths,
        )
        return scores
