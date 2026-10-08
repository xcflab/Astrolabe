from __future__ import annotations

import hashlib
import gzip
import logging
import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Select, Structure
from Bio.SVDSuperimposer import SVDSuperimposer

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.structure_processors import StructureProcessor, register_structure_processor

logger = logging.getLogger(__name__)

NUCLEIC_RESIDUES = {
    "A",
    "C",
    "G",
    "U",
    "T",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "DI",
    "ADE",
    "CYT",
    "GUA",
    "URA",
    "THY",
}


class HADDOCK3DockingError(RuntimeError):
    """Raised when HADDOCK3 predocking cannot produce a usable complex."""


@dataclass(frozen=True)
class ContactMetrics:
    min_distance: float
    contact_count: int


@dataclass(frozen=True)
class ContactResiduePair:
    binder_chain: str
    binder_position: int
    target_chain: str
    target_position: int
    min_distance: float


class _ChainSelect(Select):
    def __init__(self, chain_ids: set[str]):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):  # noqa: ANN001 - Bio.PDB callback signature.
        return 1 if chain.id in self.chain_ids else 0


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "complex"


def _load_structure(path: Path):
    suffixes = [suffix.lower() for suffix in path.suffixes]
    structure_suffix = suffixes[-2] if suffixes[-1:] == [".gz"] and len(suffixes) >= 2 else path.suffix.lower()
    if suffixes[-1:] == [".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            if structure_suffix in {".cif", ".mmcif"}:
                return MMCIFParser(QUIET=True).get_structure(path.stem, handle)
            return PDBParser(QUIET=True).get_structure(path.stem, handle)
    if structure_suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _first_model(path: Path):
    structure = _load_structure(path)
    return next(structure.get_models())


def _is_nucleic_residue(residue) -> bool:  # noqa: ANN001 - Bio.PDB residue.
    return str(residue.get_resname()).strip().upper() in NUCLEIC_RESIDUES


def _chain_has_nucleic_residue(chain) -> bool:  # noqa: ANN001 - Bio.PDB chain.
    return any(_is_nucleic_residue(residue) for residue in chain.get_residues())


def _normalize_chain_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = re.split(r"[\s,]+", value.strip())
        return [token for token in tokens if token]
    return [str(item) for item in value]


def _residue_number(residue) -> int | None:  # noqa: ANN001 - Bio.PDB residue.
    hetflag, resseq, insertion = residue.id
    if insertion not in {" ", ""}:
        logger.warning("Ignoring insertion code %s for HADDOCK restraint residue %s.", insertion, residue)
    if str(hetflag).strip() and str(hetflag).strip() != "W":
        return None
    return int(resseq)


def _atom_coords_for_chains(path: Path, chain_ids: set[str]) -> np.ndarray:
    model = _first_model(path)
    coords: list[np.ndarray] = []
    for chain in model:
        if chain.id not in chain_ids:
            continue
        for atom in chain.get_atoms():
            element = str(getattr(atom, "element", "") or "").strip().upper()
            if element == "H":
                continue
            coords.append(np.asarray(atom.coord, dtype=float))
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.stack(coords, axis=0)


def _is_hydrogen(atom) -> bool:  # noqa: ANN001 - Bio.PDB atom.
    element = str(getattr(atom, "element", "") or "").strip().upper()
    return element == "H" or atom.get_name().strip().upper().startswith("H")


def _chain_by_id(model, chain_id: str):  # noqa: ANN001 - Bio.PDB model.
    for chain in model:
        if chain.id == chain_id:
            return chain
    raise HADDOCK3DockingError(f"Chain {chain_id!r} not found in {model!r}.")


def _residue_atom_key(atom) -> tuple[int, str, str] | None:  # noqa: ANN001 - Bio.PDB atom.
    residue = atom.get_parent()
    number = _residue_number(residue)
    if number is None:
        return None
    insertion = str(residue.id[2] or "").strip()
    return (number, insertion, atom.get_name().strip())


def _contact_metrics(
    path: Path,
    *,
    binder_chain: str,
    target_chains: set[str],
    contact_cutoff: float,
) -> ContactMetrics:
    binder = _atom_coords_for_chains(path, {binder_chain})
    target = _atom_coords_for_chains(path, target_chains)
    if binder.size == 0 or target.size == 0:
        return ContactMetrics(min_distance=float("inf"), contact_count=0)

    min_distance = float("inf")
    contact_count = 0
    chunk_size = 512
    cutoff_sq = float(contact_cutoff) ** 2
    for start in range(0, len(binder), chunk_size):
        diff = binder[start : start + chunk_size, None, :] - target[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_distance = min(min_distance, float(np.sqrt(np.min(dist_sq))))
        contact_count += int(np.count_nonzero(dist_sq <= cutoff_sq))
    return ContactMetrics(min_distance=min_distance, contact_count=contact_count)


def _format_positions(chain_id: str, positions: list[int]) -> str:
    if not positions:
        return str(chain_id)
    return f"{chain_id}:{','.join(str(int(pos)) for pos in positions)}"


def _contact_residue_pairs(
    path: Path,
    *,
    binder_chain: str,
    target_chains: set[str],
    contact_cutoff: float,
) -> list[ContactResiduePair]:
    model = _first_model(path)
    binder_atoms: list[tuple[np.ndarray, int]] = []
    target_atoms: list[tuple[np.ndarray, str, int]] = []
    for chain in model:
        if chain.id == binder_chain:
            for residue in chain.get_residues():
                residue_number = _residue_number(residue)
                if residue_number is None:
                    continue
                for atom in residue.get_atoms():
                    if not _is_hydrogen(atom):
                        binder_atoms.append((np.asarray(atom.coord, dtype=float), residue_number))
            continue
        if chain.id not in target_chains:
            continue
        for residue in chain.get_residues():
            residue_number = _residue_number(residue)
            if residue_number is None:
                continue
            for atom in residue.get_atoms():
                if not _is_hydrogen(atom):
                    target_atoms.append(
                        (np.asarray(atom.coord, dtype=float), chain.id, residue_number)
                    )

    if not binder_atoms or not target_atoms:
        return []

    cutoff_sq = float(contact_cutoff) ** 2
    best_by_pair: dict[tuple[int, str, int], float] = {}
    for binder_coord, binder_pos in binder_atoms:
        for target_coord, target_chain, target_pos in target_atoms:
            dist_sq = float(np.sum((binder_coord - target_coord) ** 2))
            if dist_sq > cutoff_sq:
                continue
            key = (int(binder_pos), str(target_chain), int(target_pos))
            distance = float(np.sqrt(dist_sq))
            previous = best_by_pair.get(key)
            if previous is None or distance < previous:
                best_by_pair[key] = distance

    pairs = [
        ContactResiduePair(
            binder_chain=str(binder_chain),
            binder_position=binder_pos,
            target_chain=target_chain,
            target_position=target_pos,
            min_distance=distance,
        )
        for (binder_pos, target_chain, target_pos), distance in best_by_pair.items()
    ]
    pairs.sort(key=lambda pair: (pair.min_distance, pair.target_chain, pair.target_position, pair.binder_position))
    return pairs


@register_structure_processor("HADDOCK3_nucleic")
@register_structure_processor("haddock3_nucleic")
class HADDOCK3NucleicDockingProcessor(StructureProcessor):
    """Rigid-body predocker for protein-nucleic acid complexes using HADDOCK3."""

    def __init__(
        self,
        *,
        haddock3_bin: str | None = None,
        binder_chain: str = "A",
        target_chains: Any = None,
        anchor_chains: Any = None,
        max_anchor_chains: int | None = 1,
        nucleic_only: bool = True,
        contact_cutoff: float = 4.0,
        skip_if_contact: bool = True,
        require_contact: bool = True,
        sampling: int = 100,
        ncores: int = 1,
        restraint_distance: Any = None,
        max_restraints: int = 64,
        update_dynamic_binding_site: bool = True,
        binding_site_mode: str = "both",
        binding_site_max_distance: float = 6.0,
        binding_site_min_distance: float = 0.0,
        max_binding_site_pairs: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.haddock3_bin = self._resolve_haddock3_bin(haddock3_bin)
        self.binder_chain = str(binder_chain)
        self.target_chains = _normalize_chain_list(target_chains)
        self.anchor_chains = _normalize_chain_list(anchor_chains)
        self.max_anchor_chains = None if max_anchor_chains is None else int(max_anchor_chains)
        self.nucleic_only = bool(nucleic_only)
        self.contact_cutoff = float(contact_cutoff)
        self.skip_if_contact = bool(skip_if_contact)
        self.require_contact = bool(require_contact)
        self.sampling = int(sampling)
        self.ncores = int(ncores)
        self.restraint_distance = self._normalize_restraint_distance(restraint_distance)
        self.max_restraints = int(max_restraints)
        self.update_dynamic_binding_site = bool(update_dynamic_binding_site)
        self.binding_site_mode = str(binding_site_mode).strip().lower()
        if self.binding_site_mode not in {"both", "pocket", "contact"}:
            raise ValueError(
                "HADDOCK3 binding_site_mode must be one of both/pocket/contact, "
                f"got {binding_site_mode!r}."
            )
        self.binding_site_max_distance = float(binding_site_max_distance)
        self.binding_site_min_distance = float(binding_site_min_distance)
        self.max_binding_site_pairs = int(max_binding_site_pairs)
        if self.max_binding_site_pairs <= 0:
            raise ValueError("HADDOCK3 max_binding_site_pairs must be > 0.")
        self._runtime_oligo_name: str | None = None
        self._last_binding_site_specs_by_path: dict[str, dict[str, Any]] = {}

    def _generate_cache_key(self, input_file) -> str:
        path = Path(input_file)
        digest = hashlib.md5()
        digest.update(str(path.resolve()).encode())
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(
            repr(
                {
                    "class": self.__class__.__name__,
                    "binder_chain": self.binder_chain,
                    "target_chains": self.target_chains,
                    "anchor_chains": self.anchor_chains,
                    "max_anchor_chains": self.max_anchor_chains,
                    "nucleic_only": self.nucleic_only,
                    "contact_cutoff": self.contact_cutoff,
                    "sampling": self.sampling,
                    "ncores": self.ncores,
                    "restraint_distance": self.restraint_distance,
                    "max_restraints": self.max_restraints,
                    "update_dynamic_binding_site": self.update_dynamic_binding_site,
                    "binding_site_mode": self.binding_site_mode,
                    "binding_site_max_distance": self.binding_site_max_distance,
                    "binding_site_min_distance": self.binding_site_min_distance,
                    "max_binding_site_pairs": self.max_binding_site_pairs,
                }
            ).encode()
        )
        return digest.hexdigest()

    def run_on_scores(self, evaluation_scores: StandardScores) -> StandardScores:
        if self._condition is not None and not self.eval_condition():
            logging.info(
                "Condition for %s not met, skipping predocker.",
                self.__class__.__name__,
            )
            return evaluation_scores

        updated_paths = dict(evaluation_scores.complex_paths)
        for oligo_name, complex_path in evaluation_scores.complex_paths.items():
            updated_paths[oligo_name] = self.run_for_oligo(str(complex_path), oligo_name)
        updated = evaluation_scores.clone()
        updated.complex_paths = updated_paths
        return updated

    def run_for_oligo(self, input_file: str, oligo_name: str) -> str:
        self._runtime_oligo_name = str(oligo_name)
        try:
            return self.run(input_file)
        finally:
            self._runtime_oligo_name = None

    def take_cache(self, cache_key):
        result = super().take_cache(cache_key)
        self._publish_dynamic_binding_site_for_output(result)
        return result

    def _run_impl(self, input_file: str) -> str:
        input_path = Path(input_file).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"HADDOCK3 predocker input not found: {input_path}")

        target_chains = self._resolve_target_chains(input_path)
        if self.skip_if_contact:
            current = _contact_metrics(
                input_path,
                binder_chain=self.binder_chain,
                target_chains=target_chains,
                contact_cutoff=self.contact_cutoff,
            )
            if current.contact_count > 0:
                logger.info(
                    "HADDOCK3 predocker skipped %s: existing %d contact(s) within %.1f A.",
                    input_path.name,
                    current.contact_count,
                    self.contact_cutoff,
                )
                final_existing = self._copy_existing_contact(input_path)
                self._record_and_publish_dynamic_binding_site(
                    Path(final_existing),
                    target_chains=target_chains,
                )
                return final_existing

        work_dir = self._prepare_work_dir(input_path)
        candidates: list[tuple[int, float, Path, str]] = []
        failures: list[str] = []
        for anchor_chain in self._resolve_anchor_chains(target_chains):
            anchor_dir = work_dir / f"anchor_{_safe_name(anchor_chain)}"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            binder_pdb = anchor_dir / "binder.pdb"
            target_pdb = anchor_dir / f"target_{_safe_name(anchor_chain)}.pdb"
            self._write_chain_subset(input_path, binder_pdb, {self.binder_chain})
            self._write_chain_subset(input_path, target_pdb, {anchor_chain})

            restraints = self._write_restraints(
                input_path,
                anchor_dir / "restraints.tbl",
                {anchor_chain},
            )
            cfg = self._write_config(anchor_dir, binder_pdb, target_pdb, restraints)
            try:
                self._run_haddock(cfg, anchor_dir)
                docked_anchor = self._select_best_candidate(
                    anchor_dir / "haddock3_run" / "1_rigidbody",
                    {anchor_chain},
                )
                docked_name = docked_anchor.name.removesuffix(".gz")
                reconstructed = work_dir / (
                    f"reconstructed_{_safe_name(anchor_chain)}_{docked_name}"
                )
                self._write_reconstructed_complex(
                    input_path,
                    docked_anchor,
                    reconstructed,
                    anchor_chain=anchor_chain,
                    target_chains=target_chains,
                )
                metrics = _contact_metrics(
                    reconstructed,
                    binder_chain=self.binder_chain,
                    target_chains=target_chains,
                    contact_cutoff=self.contact_cutoff,
                )
                if self.require_contact and metrics.contact_count <= 0:
                    failures.append(
                        f"{anchor_chain}: reconstructed complex has no "
                        f"{self.contact_cutoff:g} A contact; min distance "
                        f"{metrics.min_distance:.3f} A"
                    )
                    continue
                candidates.append(
                    (metrics.contact_count, metrics.min_distance, reconstructed, anchor_chain)
                )
            except Exception as exc:
                failures.append(f"{anchor_chain}: {exc}")
                logger.warning("HADDOCK3 anchor-chain run failed for %s: %s", anchor_chain, exc)

        if not candidates:
            details = "; ".join(failures) if failures else "no anchor chains attempted"
            raise HADDOCK3DockingError(
                f"HADDOCK3 produced no usable predocked complex for {input_path}. {details}"
            )

        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        _, _, best, best_anchor = candidates[0]
        logger.info("Selected HADDOCK3 anchor chain %s for %s.", best_anchor, input_path.name)
        final_path = Path(self._get_output_path(f"haddock3_{input_path.stem}.pdb"))
        if best.resolve() != final_path.resolve():
            shutil.copy2(best, final_path)
        self._record_and_publish_dynamic_binding_site(
            final_path,
            target_chains=target_chains,
        )
        return str(final_path)

    def _copy_existing_contact(self, input_path: Path) -> str:
        suffix = input_path.suffix or ".pdb"
        final_path = Path(self._get_output_path(f"haddock3_existing_{input_path.stem}{suffix}"))
        if input_path.resolve() != final_path.resolve():
            shutil.copy2(input_path, final_path)
        return str(final_path)

    def _record_and_publish_dynamic_binding_site(
        self,
        output_path: Path,
        *,
        target_chains: set[str],
    ) -> None:
        spec = self._build_dynamic_binding_site_spec(output_path, target_chains=target_chains)
        if spec is None:
            return
        self._last_binding_site_specs_by_path[str(output_path)] = spec
        try:
            resolved = str(output_path.resolve())
        except OSError:
            resolved = str(output_path)
        self._last_binding_site_specs_by_path[resolved] = spec
        self._publish_dynamic_binding_site_spec(spec)

    def _publish_dynamic_binding_site_for_output(self, output_path: Any) -> None:
        if output_path is None:
            return
        key = str(output_path)
        spec = self._last_binding_site_specs_by_path.get(key)
        if spec is None:
            try:
                spec = self._last_binding_site_specs_by_path.get(str(Path(key).resolve()))
            except OSError:
                spec = None
        if spec is not None:
            self._publish_dynamic_binding_site_spec(spec)

    def _publish_dynamic_binding_site_spec(self, spec: dict[str, Any]) -> None:
        if not self.update_dynamic_binding_site:
            return
        oligo_name = getattr(self, "_runtime_oligo_name", None)
        if not oligo_name:
            return
        try:
            current = getattr(global_config, "dynamic_binding_site", None)
        except RuntimeError:
            logger.debug(
                "Skipping dynamic binding_site publication because global_config.dynamic_binding_site is not initialized."
            )
            return
        if current is None:
            current_map: dict[str, Any] = {}
        elif hasattr(current, "items"):
            current_map = {str(k): current[k] for k in current.keys()} if hasattr(current, "keys") else dict(current)
        else:
            current_map = {}
        current_map[str(oligo_name)] = spec
        global_config.update_const("dynamic_binding_site", current_map)
        logger.info(
            "Updated dynamic binding_site for %s from HADDOCK3 predocked contacts: binder=%s target=%s mode=%s.",
            oligo_name,
            spec.get("binder"),
            spec.get("target"),
            spec.get("mode"),
        )

    def _build_dynamic_binding_site_spec(
        self,
        output_path: Path,
        *,
        target_chains: set[str],
    ) -> dict[str, Any] | None:
        pairs = _contact_residue_pairs(
            output_path,
            binder_chain=self.binder_chain,
            target_chains=target_chains,
            contact_cutoff=self.contact_cutoff,
        )
        if not pairs:
            if self.require_contact:
                raise HADDOCK3DockingError(
                    f"Cannot derive dynamic binding_site from {output_path}: no "
                    f"{self.contact_cutoff:g} A binder-target residue contacts."
                )
            return None
        selected = pairs[: self.max_binding_site_pairs]
        target_chain_counts: dict[str, int] = {}
        for pair in selected:
            target_chain_counts[pair.target_chain] = target_chain_counts.get(pair.target_chain, 0) + 1
        target_chain = sorted(
            target_chain_counts,
            key=lambda chain: (-target_chain_counts[chain], chain),
        )[0]
        selected_for_chain = [pair for pair in selected if pair.target_chain == target_chain]
        binder_positions = list(
            dict.fromkeys(pair.binder_position for pair in selected_for_chain)
        )
        target_positions = list(
            dict.fromkeys(pair.target_position for pair in selected_for_chain)
        )
        if not binder_positions or not target_positions:
            return None
        return {
            "binder": _format_positions(self.binder_chain, binder_positions),
            "target": _format_positions(target_chain, target_positions),
            "mode": self.binding_site_mode,
            "max_distance": self.binding_site_max_distance,
            "min_distance": self.binding_site_min_distance,
        }

    @staticmethod
    def _normalize_restraint_distance(value: Any) -> tuple[float, float, float]:
        if value is None:
            return (2.0, 4.0, 8.0)
        if isinstance(value, str):
            parts = [float(part) for part in re.split(r"[\s,]+", value.strip()) if part]
        else:
            parts = [float(part) for part in value]
        if len(parts) != 3:
            raise ValueError("HADDOCK3 restraint_distance must contain lower, target, upper.")
        return (parts[0], parts[1], parts[2])

    @staticmethod
    def _resolve_haddock3_bin(value: str | None) -> str:
        if value:
            expanded = os.path.expanduser(str(value))
            if os.path.sep in expanded:
                path = Path(expanded)
                if not path.exists():
                    raise FileNotFoundError(f"Configured haddock3_bin does not exist: {path}")
                return str(path)
            resolved = shutil.which(expanded)
            if resolved:
                return resolved
            raise FileNotFoundError(f"Configured haddock3_bin is not on PATH: {value}")

        resolved = shutil.which("haddock3")
        if resolved:
            return resolved
        raise FileNotFoundError(
            "HADDOCK3 predocker requires `predocker.haddock3.haddock3_bin` "
            "or a `haddock3` executable on PATH."
        )

    def _resolve_target_chains(self, input_path: Path) -> set[str]:
        model = _first_model(input_path)
        chain_ids = {chain.id for chain in model}
        if self.binder_chain not in chain_ids:
            raise HADDOCK3DockingError(
                f"Binder chain {self.binder_chain!r} not found in {input_path}."
            )

        if self.target_chains is not None:
            selected = set(self.target_chains)
            missing = selected - chain_ids
            if missing:
                raise HADDOCK3DockingError(
                    f"Target chain(s) {sorted(missing)} not found in {input_path}."
                )
            return selected

        selected = {
            chain.id
            for chain in model
            if chain.id != self.binder_chain
            and (not self.nucleic_only or _chain_has_nucleic_residue(chain))
        }
        if not selected:
            raise HADDOCK3DockingError(
                f"No target nucleic-acid chains found in {input_path}; set "
                "`target_chains` or `nucleic_only: false` if this is intended."
            )
        return selected

    def _resolve_anchor_chains(self, target_chains: set[str]) -> list[str]:
        if self.anchor_chains is None:
            selected = sorted(target_chains)
        else:
            selected = list(self.anchor_chains)
            missing = set(selected) - target_chains
            if missing:
                raise HADDOCK3DockingError(
                    "HADDOCK3 anchor_chain(s) must be part of target_chains. "
                    f"Missing from targets: {sorted(missing)}"
                )
        if self.max_anchor_chains is not None:
            selected = selected[: self.max_anchor_chains]
        if not selected:
            raise HADDOCK3DockingError("No HADDOCK3 anchor chains selected.")
        return selected

    def _prepare_work_dir(self, input_path: Path) -> Path:
        digest = hashlib.md5(str(input_path.resolve()).encode()).hexdigest()[:10]
        work_dir = self.debug_dir / f"haddock3_work_{_safe_name(input_path.stem)}_{digest}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        return work_dir

    @staticmethod
    def _write_chain_subset(input_path: Path, output_path: Path, chain_ids: set[str]) -> None:
        structure = _load_structure(input_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path), _ChainSelect(chain_ids))
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HADDOCK3DockingError(
                f"Failed to write chain subset {sorted(chain_ids)} from {input_path}."
            )

    def _target_restraint_residues(
        self,
        input_path: Path,
        target_chains: set[str],
    ) -> list[tuple[str, int]]:
        residues: list[tuple[str, int]] = []
        model = _first_model(input_path)
        for chain in model:
            if chain.id not in target_chains:
                continue
            for residue in chain.get_residues():
                if self.nucleic_only and not _is_nucleic_residue(residue):
                    continue
                number = _residue_number(residue)
                if number is not None:
                    residues.append((chain.id, number))
        if not residues:
            raise HADDOCK3DockingError(
                f"No target residues available for HADDOCK restraints in {input_path}."
            )
        if len(residues) <= self.max_restraints:
            return residues
        indices = np.linspace(0, len(residues) - 1, num=self.max_restraints, dtype=int)
        return [residues[int(index)] for index in indices]

    def _write_restraints(
        self,
        input_path: Path,
        output_path: Path,
        target_chains: set[str],
    ) -> Path:
        lower, target, upper = self.restraint_distance
        lines = [
            "! HADDOCK3 ambiguous restraints generated by GH haddock3_nucleic predocker",
            f"! binder chain: {self.binder_chain}; target chains: {','.join(sorted(target_chains))}",
        ]
        for chain_id, residue_number in self._target_restraint_residues(input_path, target_chains):
            lines.append(
                f"assign (segid {chain_id} and resid {residue_number}) "
                f"(segid {self.binder_chain}) {lower:g} {target:g} {upper:g}"
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def _write_reconstructed_complex(
        self,
        original_path: Path,
        docked_path: Path,
        output_path: Path,
        *,
        anchor_chain: str,
        target_chains: set[str],
    ) -> None:
        original_model = _first_model(original_path)
        docked_model = _first_model(docked_path)
        rotation, translation = self._fit_docked_anchor_to_original(
            original_model,
            docked_model,
            anchor_chain=anchor_chain,
        )

        output_structure = Structure.Structure(output_path.stem)
        output_model = Model.Model(0)
        output_structure.add(output_model)

        docked_binder = deepcopy(_chain_by_id(docked_model, self.binder_chain))
        docked_binder.id = self.binder_chain
        for atom in docked_binder.get_atoms():
            atom.coord = np.asarray(atom.coord, dtype=float) @ rotation + translation
        output_model.add(docked_binder)

        for chain in original_model:
            if chain.id == self.binder_chain:
                continue
            if chain.id in target_chains:
                output_model.add(deepcopy(chain))

        io = PDBIO()
        io.set_structure(output_structure)
        io.save(str(output_path))
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HADDOCK3DockingError(
                f"Failed to write reconstructed HADDOCK3 complex to {output_path}."
            )

    def _fit_docked_anchor_to_original(
        self,
        original_model,
        docked_model,
        *,
        anchor_chain: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_chain = _chain_by_id(original_model, anchor_chain)
        docked_chain = _chain_by_id(docked_model, anchor_chain)
        original_atoms = {}
        for atom in original_chain.get_atoms():
            if _is_hydrogen(atom):
                continue
            key = _residue_atom_key(atom)
            if key is not None:
                original_atoms[key] = np.asarray(atom.coord, dtype=float)

        ref_coords: list[np.ndarray] = []
        docked_coords: list[np.ndarray] = []
        for atom in docked_chain.get_atoms():
            if _is_hydrogen(atom):
                continue
            key = _residue_atom_key(atom)
            if key not in original_atoms:
                continue
            ref_coords.append(original_atoms[key])
            docked_coords.append(np.asarray(atom.coord, dtype=float))

        if len(ref_coords) < 3:
            raise HADDOCK3DockingError(
                f"Cannot reconstruct full complex from HADDOCK3 output {docked_model}: "
                f"only {len(ref_coords)} matched heavy atoms for anchor chain {anchor_chain!r}."
            )

        sup = SVDSuperimposer()
        sup.set(np.stack(ref_coords, axis=0), np.stack(docked_coords, axis=0))
        sup.run()
        return sup.get_rotran()

    def _write_config(
        self,
        work_dir: Path,
        binder_pdb: Path,
        target_pdb: Path,
        restraints: Path,
    ) -> Path:
        cfg = work_dir / "haddock3.cfg"
        cfg.write_text(
            "\n".join(
                [
                    'run_dir = "haddock3_run"',
                    f"ncores = {self.ncores}",
                    'mode = "local"',
                    "molecules = [",
                    f'    "{binder_pdb}",',
                    f'    "{target_pdb}",',
                    "]",
                    "",
                    "[topoaa]",
                    "",
                    "[rigidbody]",
                    f"sampling = {self.sampling}",
                    f'ambig_fname = "{restraints}"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return cfg

    def _run_haddock(self, cfg: Path, work_dir: Path) -> None:
        log_path = work_dir / "haddock3.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                [self.haddock3_bin, str(cfg)],
                cwd=str(work_dir),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            raise HADDOCK3DockingError(
                f"HADDOCK3 failed with exit code {proc.returncode}. Log: {log_path}"
            )

    def _select_best_candidate(self, rigidbody_dir: Path, target_chains: set[str]) -> Path:
        candidates = sorted(rigidbody_dir.glob("rigidbody_*.pdb"))
        candidates.extend(sorted(rigidbody_dir.glob("rigidbody_*.pdb.gz")))
        if not candidates:
            raise HADDOCK3DockingError(
                f"HADDOCK3 produced no rigidbody candidates under {rigidbody_dir}."
            )

        ranked: list[tuple[int, float, Path]] = []
        for candidate in candidates:
            metrics = _contact_metrics(
                candidate,
                binder_chain=self.binder_chain,
                target_chains=target_chains,
                contact_cutoff=self.contact_cutoff,
            )
            ranked.append((metrics.contact_count, metrics.min_distance, candidate))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2].name))
        best_contacts, best_distance, best_path = ranked[0]
        if self.require_contact and best_contacts <= 0:
            raise HADDOCK3DockingError(
                "HADDOCK3 completed but no candidate formed a "
                f"{self.contact_cutoff:g} A binder-target contact. "
                f"Best min distance was {best_distance:.3f} A in {best_path}."
            )
        logger.info(
            "Selected HADDOCK3 candidate %s with %d contact(s), min distance %.3f A.",
            best_path.name,
            best_contacts,
            best_distance,
        )
        return best_path
