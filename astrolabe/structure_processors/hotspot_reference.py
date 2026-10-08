from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser

from astrolabe.core import global_config
from astrolabe.core.binding_site import (
    ChainPositions,
    parse_binding_site,
    parse_chain_positions,
)
from astrolabe.core.chain_resolution import resolve_single_chain_id
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.confidence import binder_ptm
from astrolabe.structure_processors import StructureProcessor, register_structure_processor


def _as_plain_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"hotspot_reference_align requires `{field}`.")
    if not isinstance(value, Mapping) and hasattr(value, "items"):
        value = {k: value[k] for k in value.keys()}
    if not isinstance(value, Mapping):
        raise ValueError(f"hotspot_reference_align `{field}` must be a mapping.")
    return {str(k): v for k, v in value.items()}


def _parse_chain_positions_value(value: Any, *, field: str, oligo: str) -> ChainPositions:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = [parse_chain_positions(str(item)) for item in value]
        chains = {item.chain for item in parsed}
        if len(chains) != 1:
            raise ValueError(
                f"hotspot_reference_align {field}.{oligo} must use one chain selector, "
                f"got {sorted(chains)}."
            )
        positions: list[int] = []
        seen: set[int] = set()
        for item in parsed:
            for pos in item.positions:
                if pos not in seen:
                    seen.add(pos)
                    positions.append(pos)
        return ChainPositions(chain=parsed[0].chain, positions=tuple(positions))

    parsed = parse_chain_positions(str(value))
    if not parsed.positions:
        raise ValueError(
            f"hotspot_reference_align {field}.{oligo} requires explicit residue positions."
        )
    return parsed


def _sanitize_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return token.strip("_") or "oligo"


@register_structure_processor("hotspot_reference_align")
class HotspotReferenceAlignmentProcessor(StructureProcessor):
    """Choose a reference complex and align other complexes into its hotspot frame."""

    def __init__(
        self,
        *,
        hotspot_scores,
        target_hotspots,
        binder_chain=None,
        tie_delta: float = 0.05,
        atom_name: str = "CA",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hotspot_scores = {
            oligo: str(term)
            for oligo, term in _as_plain_mapping(
                hotspot_scores,
                field="hotspot_scores",
            ).items()
        }
        self.target_hotspots = {
            oligo: _parse_chain_positions_value(
                value,
                field="target_hotspots",
                oligo=oligo,
            )
            for oligo, value in _as_plain_mapping(
                target_hotspots,
                field="target_hotspots",
            ).items()
        }
        if binder_chain is None:
            self.binder_chain = None
        elif isinstance(binder_chain, Mapping) or hasattr(binder_chain, "items"):
            self.binder_chain = {
                str(k): str(v)
                for k, v in _as_plain_mapping(
                    binder_chain,
                    field="binder_chain",
                ).items()
            }
        else:
            self.binder_chain = str(binder_chain)
        self.tie_delta = float(tie_delta)
        self.atom_name = str(atom_name)
        self.last_reference_oligo: str | None = None
        self.last_reference_reason: str | None = None

    def _run_impl(self, input_file) -> str:
        raise RuntimeError(
            "hotspot_reference_align needs full StandardScores; attach it to a "
            "mutator predocker chain and run through prepare_mutation_input."
        )

    def run_on_scores(self, evaluation_scores: StandardScores) -> StandardScores:
        if self._condition is not None and not self.eval_condition():
            logging.info(
                "Condition for %s not met, skipping predocker.",
                self.__class__.__name__,
            )
            return evaluation_scores

        oligos = [
            name
            for name in evaluation_scores.names
            if name in self.hotspot_scores and name in self.target_hotspots
        ]
        if len(oligos) < 2:
            raise ValueError(
                "hotspot_reference_align requires at least two configured oligomers "
                f"in the evaluator result; got {oligos!r}."
            )

        reference_oligo, reason = self._select_reference(evaluation_scores, oligos)
        self.last_reference_oligo = reference_oligo
        self.last_reference_reason = reason

        aligned_paths = dict(evaluation_scores.complex_paths)
        reference_path = Path(aligned_paths[reference_oligo])
        reference_structure = self._load_structure(reference_path)
        reference_model = next(reference_structure.get_models())
        ref_spec = self.target_hotspots[reference_oligo]
        ref_chain_id = self._resolve_structure_chain_id(
            reference_oligo,
            ref_spec.chain,
            reference_model,
        )
        ref_coords = self._hotspot_atom_coords(reference_model, ref_chain_id, ref_spec)

        for oligo in oligos:
            if oligo == reference_oligo:
                continue
            mobile_path = Path(aligned_paths[oligo])
            mobile_structure = self._load_structure(mobile_path)
            mobile_model = next(mobile_structure.get_models())
            mobile_spec = self.target_hotspots[oligo]
            mobile_chain_id = self._resolve_structure_chain_id(
                oligo,
                mobile_spec.chain,
                mobile_model,
            )
            mobile_coords = self._hotspot_atom_coords(
                mobile_model,
                mobile_chain_id,
                mobile_spec,
            )
            rotation, translation = self._superpose(mobile_coords, ref_coords)
            self._apply_transform(mobile_structure, rotation, translation)
            aligned_paths[oligo] = self._write_aligned_structure(
                mobile_structure,
                oligo=oligo,
            )

        updated = evaluation_scores.clone()
        updated.complex_paths = aligned_paths
        logging.info(
            "hotspot_reference_align selected %s by %s; aligned %d complex(es).",
            reference_oligo,
            reason,
            len(oligos) - 1,
        )
        return updated

    def _select_reference(
        self,
        evaluation_scores: StandardScores,
        oligos: list[str],
    ) -> tuple[str, str]:
        hotspot_values = self._hotspot_values(evaluation_scores, oligos)
        ordered = sorted(
            oligos,
            key=lambda name: hotspot_values[name],
            reverse=True,
        )
        best = ordered[0]
        second = ordered[1]
        delta = float(hotspot_values[best] - hotspot_values[second])
        if delta >= self.tie_delta:
            return best, "hotspot_score"

        tie_group = [
            oligo
            for oligo in ordered
            if float(hotspot_values[best] - hotspot_values[oligo]) < self.tie_delta
        ]
        binder_ptm_values = {
            oligo: self._binder_ptm_for_oligo(evaluation_scores, oligo)
            for oligo in tie_group
        }
        reference = max(tie_group, key=lambda oligo: binder_ptm_values[oligo])
        return reference, "binder_ptm_tiebreak"

    def _hotspot_values(
        self,
        evaluation_scores: StandardScores,
        oligos: list[str],
    ) -> dict[str, float]:
        score_context = self._score_context(evaluation_scores)
        values: dict[str, float] = {}
        for oligo in oligos:
            term = self.hotspot_scores[oligo]
            values[oligo] = self._lookup_score(score_context, oligo, term)
        return values

    def _score_context(self, evaluation_scores: StandardScores) -> Mapping[str, Any]:
        loss_computer = getattr(global_config, "loss_computer", None)
        if loss_computer is None:
            raise ValueError(
                "hotspot_reference_align requires global_config.loss_computer "
                "to resolve configured hotspot score terms."
            )
        score_context = getattr(loss_computer, "score_context", None)
        if callable(score_context):
            return score_context(evaluation_scores)
        per_oligo_scores, per_terms_score = loss_computer._extract_scores_with_registry(
            evaluation_scores
        )
        return {**per_oligo_scores, **per_terms_score}

    def _lookup_score(self, score_context: Mapping[str, Any], oligo: str, term: str) -> float:
        per_oligo = score_context.get(oligo)
        if isinstance(per_oligo, Mapping) and term in per_oligo:
            return float(per_oligo[term])
        if per_oligo is not None and hasattr(per_oligo, term):
            return float(getattr(per_oligo, term))
        dotted = f"{oligo}.{term}"
        if dotted in score_context:
            return float(score_context[dotted])
        raise KeyError(
            f"hotspot_reference_align could not find score term {term!r} for "
            f"oligo {oligo!r} in loss score context."
        )

    def _binder_ptm_for_oligo(
        self,
        evaluation_scores: StandardScores,
        oligo: str,
    ) -> float:
        selector = self._binder_chain_for_oligo(oligo)
        filtered = evaluation_scores.filter_by_oligomer([oligo])
        return float(binder_ptm(filtered, binder_chain=selector).get(oligo, 0.0))

    def _binder_chain_for_oligo(self, oligo: str) -> str | None:
        if isinstance(self.binder_chain, Mapping):
            if oligo not in self.binder_chain:
                raise ValueError(
                    f"hotspot_reference_align binder_chain mapping is missing {oligo!r}."
                )
            return str(self.binder_chain[oligo])
        if self.binder_chain is not None:
            return str(self.binder_chain)
        try:
            spec = parse_binding_site(
                getattr(global_config, "binding_site", None),
                oligo_name=oligo,
            )
        except Exception:
            spec = None
        if spec is not None:
            return str(spec.binder.chain)
        return None

    def _load_structure(self, path: Path):
        suffix = path.suffix.lower()
        parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
        return parser.get_structure(path.stem, str(path))

    def _resolve_structure_chain_id(self, oligo: str, selector: str, model) -> str:
        resolved = resolve_single_chain_id(
            oligo,
            selector,
            context="hotspot_reference_align.target_hotspots",
        )
        model_chains = {chain.id for chain in model.get_chains()}
        if resolved is not None and resolved in model_chains:
            return resolved

        raw = str(selector)
        for prefix in ("chain:", "protomer:"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix) :]
                break
        raw = raw.strip().upper()
        if raw in model_chains:
            return raw

        raise ValueError(
            f"hotspot_reference_align could not resolve selector {selector!r} "
            f"for oligo {oligo!r}; structure chains={sorted(model_chains)!r}."
        )

    def _hotspot_atom_coords(
        self,
        model,
        chain_id: str,
        spec: ChainPositions,
    ) -> np.ndarray:
        chain = model[chain_id]
        coords: list[np.ndarray] = []
        missing: list[int] = []
        for pos in spec.positions:
            residue = self._residue_by_position(chain, pos)
            if residue is None or self.atom_name not in residue:
                missing.append(int(pos))
                continue
            coords.append(np.asarray(residue[self.atom_name].coord, dtype=float))
        if missing:
            raise ValueError(
                f"hotspot_reference_align missing atom {self.atom_name!r} for "
                f"chain {chain_id!r} residues {missing!r}."
            )
        if len(coords) < 1:
            raise ValueError(
                "hotspot_reference_align requires at least one hotspot atom "
                f"for stable alignment, got {len(coords)}."
            )
        return np.stack(coords, axis=0)

    def _residue_by_position(self, chain, position: int):
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag == " " and int(resseq) == int(position) and str(icode).strip() == "":
                return residue
        for residue in chain:
            hetflag, resseq, _icode = residue.id
            if hetflag == " " and int(resseq) == int(position):
                return residue
        return None

    def _superpose(
        self,
        mobile_coords: np.ndarray,
        reference_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if mobile_coords.shape != reference_coords.shape:
            raise ValueError(
                "hotspot_reference_align corresponding hotspot lists must have "
                f"the same shape, got {mobile_coords.shape} and {reference_coords.shape}."
            )
        if mobile_coords.shape[0] == 1:
            rotation = np.eye(3, dtype=float)
            translation = reference_coords[0] - mobile_coords[0]
            return rotation, translation
        mobile_centroid = np.mean(mobile_coords, axis=0)
        reference_centroid = np.mean(reference_coords, axis=0)
        mobile_centered = mobile_coords - mobile_centroid
        reference_centered = reference_coords - reference_centroid
        covariance = mobile_centered.T @ reference_centered
        u, _s, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        translation = reference_centroid - mobile_centroid @ rotation
        return rotation, translation

    def _apply_transform(
        self,
        structure,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        for atom in structure.get_atoms():
            atom.coord = np.asarray(atom.coord, dtype=float) @ rotation + translation

    def _write_aligned_structure(self, structure, *, oligo: str) -> Path:
        step = self._current_step()
        out_path = self.debug_dir / (
            f"hotspot_reference_align_step_{step:03d}_{_sanitize_filename_token(oligo)}.pdb"
        )
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_path))
        return out_path

    def _current_step(self) -> int:
        try:
            return int(global_config.step)
        except Exception:
            return 0
