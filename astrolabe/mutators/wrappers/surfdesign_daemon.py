#!/usr/bin/env python3
"""Long-lived SurfDesign inference worker using JSONL over stdin/stdout."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass
class Runtime:
    torch: Any
    model: Any
    alphabet: Any
    generator: Any
    data_processor: Any
    full_mask: Any
    surface_smooth: Any
    device: str
    logit_temperature: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SurfDesign JSONL daemon")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--strategy", default="denoise")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--surface-mode", default="pymol")
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def _write(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_runtime(args: argparse.Namespace) -> Runtime:
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))

    import torch
    from src import utils
    from src.datamodules.datasets import DataProcessor
    from src.datamodules.datasets.cath import surface_smooth
    from src.models.generator import IterativeRefinementGenerator
    from src.tasks.designer import _full_mask

    pl_task, _experiment_config = utils.load_from_experiment(
        str(Path(args.experiment_path).resolve()), ckpt=args.checkpoint
    )
    model = pl_task.model.to(args.device)
    model.eval()
    alphabet = pl_task.alphabet
    generator = IterativeRefinementGenerator(
        alphabet=alphabet,
        max_iter=args.max_iter,
        strategy=args.strategy,
        temperature=args.temperature,
    )
    return Runtime(
        torch=torch,
        model=model,
        alphabet=alphabet,
        generator=generator,
        data_processor=DataProcessor(),
        full_mask=_full_mask,
        surface_smooth=surface_smooth,
        device=args.device,
        logit_temperature=args.logit_temperature,
    )


class SurfaceBuilder:
    def __init__(self, work_dir: Path, surface_mode: str):
        if surface_mode != "pymol":
            raise ValueError("SurfDesign daemon requires --surface-mode pymol.")
        import pymol2
        from src.datamodules.datasets.cath import (
            computer_property,
            load_point_cloud_by_file_extension,
        )

        self.work_dir = work_dir / "surface_cache"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pymol = pymol2.PyMOL()
        self.pymol.start()
        self.cmd = self.pymol.cmd
        self.computer_property = computer_property
        self.load_point_cloud = load_point_cloud_by_file_extension

    def _as_pdb(self, structure_path: Path) -> Path:
        suffix = structure_path.suffix.lower()
        if suffix == ".pdb":
            return structure_path
        if suffix not in {".cif", ".mmcif"}:
            raise ValueError(
                "SurfDesign requires an evaluator structure in PDB or mmCIF format."
            )

        from Bio.PDB import MMCIFParser, PDBIO

        source_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
        converted_path = self.work_dir / f"{source_hash}.input.pdb"
        if converted_path.is_file():
            return converted_path
        structure = MMCIFParser(QUIET=True).get_structure(
            structure_path.stem, str(structure_path)
        )
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(converted_path))
        return converted_path

    @staticmethod
    def _backbone_lines(pdb_path: Path) -> tuple[list[str], list[str], dict[str, int]]:
        output: list[str] = []
        chain_order: list[str] = []
        residue_ordinals: dict[str, dict[tuple[str, str], int]] = {}
        for raw_line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line
            record = line[:6].strip()
            resname = line[17:20].strip().upper() if len(line) >= 20 else ""
            if record == "HETATM" and resname == "MSE":
                line = "ATOM  " + line[6:17] + "MET" + line[20:]
                resname = "MET"
                record = "ATOM"
            if record != "ATOM" or resname not in AA3 or len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue
            altloc = line[16:17]
            if altloc not in {"", " ", "A"}:
                continue
            chain = line[21:22]
            if not chain.strip():
                raise ValueError("SurfDesign requires non-empty one-character PDB chain IDs.")
            if chain not in chain_order:
                chain_order.append(chain)
                residue_ordinals[chain] = {}
            residue_key = (line[22:26], line[26:27])
            chain_residues = residue_ordinals[chain]
            if residue_key not in chain_residues:
                chain_residues[residue_key] = len(chain_residues) + 1
            ordinal = chain_residues[residue_key]
            normalized = line[:16] + " " + line[17:22] + f"{ordinal:4d}" + " " + line[27:]
            output.append(normalized)
        if not output:
            raise ValueError(f"SurfDesign found no standard protein backbone atoms in {pdb_path}.")
        return output, chain_order, {chain: len(residue_ordinals[chain]) for chain in chain_order}

    def build(self, pdb_path: Path):
        pdb_path = self._as_pdb(pdb_path)
        lines, chain_order, chain_lengths = self._backbone_lines(pdb_path)
        backbone_text = "\n".join(lines) + "\nEND\n"
        cache_key = hashlib.sha256(backbone_text.encode("utf-8")).hexdigest()
        cache_path = self.work_dir / f"{cache_key}.pkl"
        backbone_path = self.work_dir / f"{cache_key}.pdb"
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                surface = pickle.load(handle)
            return backbone_path, chain_order, chain_lengths, surface

        backbone_path.write_text(backbone_text, encoding="utf-8")
        object_path = self.work_dir / f"{cache_key}.obj"
        self.cmd.reinitialize()
        self.cmd.load(str(backbone_path), "protein")
        self.cmd.show_as("surface", "protein")
        self.cmd.set_view(
            (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1)
        )
        self.cmd.save(str(object_path), "protein")
        if not object_path.is_file() or object_path.stat().st_size == 0:
            raise RuntimeError(f"PyMOL did not produce a SurfDesign surface: {object_path}")
        points, normals = self.load_point_cloud(str(object_path))
        hydrophobicity, hbond = self.computer_property(
            str(backbone_path), points, "pymol"
        )
        surface = (points, normals, hydrophobicity, hbond)
        with cache_path.open("wb") as handle:
            pickle.dump(surface, handle)
        return backbone_path, chain_order, chain_lengths, surface

    def close(self) -> None:
        self.pymol.stop()


def _fixed_flat_indices(
    encoded_residues: str, chain_order: list[str], chain_lengths: dict[str, int]
) -> list[int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for chain in chain_order:
        offsets[chain] = cursor
        cursor += chain_lengths[chain]
    flat_indices: list[int] = []
    for encoded in encoded_residues.split():
        match = re.fullmatch(r"([A-Za-z0-9])(\d+)([A-Za-z]?)", encoded)
        if match is None:
            raise ValueError(f"Invalid SurfDesign fixed residue identifier: {encoded!r}")
        chain, position_text, _insertion = match.groups()
        if chain not in offsets:
            raise ValueError(f"SurfDesign fixed residue uses missing chain {chain!r}: {encoded!r}")
        position = int(position_text)
        if position < 1 or position > chain_lengths[chain]:
            raise ValueError(f"SurfDesign fixed residue is out of range: {encoded!r}")
        flat_indices.append(offsets[chain] + position - 1)
    return flat_indices


def _infer(
    runtime: Runtime,
    surface_builder: SurfaceBuilder,
    pdb_path: Path,
    fixed_residues: str,
) -> tuple[list[list[float]], str]:
    torch = runtime.torch
    backbone_path, chain_order, chain_lengths, surface = surface_builder.build(pdb_path)
    structure = runtime.data_processor.parse_PDB(
        str(backbone_path), input_chain_list=chain_order
    )
    structure["name"] = backbone_path.stem

    featurizer = runtime.alphabet.featurizer
    points, normals, hydrophobicity, hbond = surface

    def get_surface(_name: str, smooth: bool):
        used_points = runtime.surface_smooth(points) if smooth else points
        return used_points, normals, hydrophobicity, hbond

    featurizer.get_surface = get_surface
    batch = runtime.alphabet.featurize(raw_batch=[structure])
    batch = {
        key: value.to(runtime.device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    prev_tokens, _prev_mask = runtime.full_mask(
        batch["tokens"], batch["coord_mask"], runtime.alphabet
    )
    batch["prev_tokens"] = prev_tokens
    residue_token_indices = torch.where(batch["coord_mask"][0])[0].tolist()
    expected_length = sum(chain_lengths.values())
    if len(residue_token_indices) != expected_length:
        raise RuntimeError(
            "SurfDesign PDB/token length mismatch: "
            f"surface PDB has {expected_length} residues, featurizer has "
            f"{len(residue_token_indices)}."
        )
    fixed_indices = _fixed_flat_indices(fixed_residues, chain_order, chain_lengths)
    for flat_index in fixed_indices:
        token_index = residue_token_indices[flat_index]
        batch["prev_tokens"][0, token_index] = batch["tokens"][0, token_index]
    batch["prev_token_mask"] = batch["prev_tokens"].eq(runtime.alphabet.mask_idx)

    autocast = (
        torch.autocast(device_type="cuda")
        if runtime.device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        output_tokens, _output_scores = runtime.generator.generate(
            model=runtime.model,
            batch=batch,
            replace_visible_tokens=bool(fixed_indices),
        )
        encoder_out = runtime.model.forward_encoder(batch)
        logits = runtime.model.decoder(
            tokens=output_tokens, encoder_out=encoder_out
        )["logits"]
        if not getattr(runtime.model.cfg, "separate_loss", False):
            logits = logits + encoder_out["logits"]

    amino_acid_indices = [runtime.alphabet.get_idx(aa) for aa in AA_ORDER]
    residue_logits = logits[0, batch["coord_mask"][0]][:, amino_acid_indices]
    probabilities = torch.softmax(
        residue_logits.float() / runtime.logit_temperature, dim=-1
    )
    sequence = runtime.alphabet.decode(output_tokens, remove_special=True)[0]
    return probabilities.cpu().tolist(), sequence


def main() -> int:
    args = _parse_args()
    protocol_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        runtime = _load_runtime(args)
        surface_builder = SurfaceBuilder(Path(args.work_dir).resolve(), args.surface_mode)
    _write(
        protocol_out,
        {"type": "ready", "pid": os.getpid(), "model_load_count": 1},
    )

    request_count = 0
    for raw in sys.stdin:
        message = json.loads(raw)
        message_type = message.get("type")
        if message_type == "shutdown":
            with contextlib.redirect_stdout(sys.stderr):
                surface_builder.close()
            _write(protocol_out, {"type": "ok", "message": "bye"})
            return 0
        if message_type != "infer":
            raise ValueError(f"Unknown SurfDesign daemon message type: {message_type!r}")

        request_count += 1
        _seed_everything(int(message.get("seed", 0)))
        with contextlib.redirect_stdout(sys.stderr):
            probabilities, sequence = _infer(
                runtime,
                surface_builder,
                Path(message["pdb_path"]).resolve(),
                str(message.get("fixed_residues") or ""),
            )
        _write(
            protocol_out,
            {
                "type": "result",
                "id": message.get("id"),
                "pid": os.getpid(),
                "model_load_count": 1,
                "request_count": request_count,
                "probabilities": probabilities,
                "sequence": sequence,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
