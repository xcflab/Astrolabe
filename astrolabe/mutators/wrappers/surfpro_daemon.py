#!/usr/bin/env python3
"""Long-lived SurfPro inference worker using JSONL over stdin/stdout."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass
class Runtime:
    torch: Any
    model: Any
    dictionary: Any
    get_edges_batch: Any
    device: str
    logit_temperature: float


@dataclass
class ProteinInput:
    pdb_path: Path
    sequence: str
    chain_order: list[str]
    chain_lengths: dict[str, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SurfPro JSONL daemon")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--msms-bin", required=True)
    parser.add_argument("--pdb-to-xyzrn-bin", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--max-surface-vertices", type=int, default=5000)
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
    torch.cuda.manual_seed_all(seed)


def _load_runtime(args: argparse.Namespace) -> Runtime:
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))

    import torch

    if not args.device.startswith("cuda"):
        raise ValueError("SurfPro requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("SurfPro requires CUDA, but torch.cuda.is_available() is false.")
    torch_device = torch.device(args.device)
    torch.cuda.set_device(torch_device.index if torch_device.index is not None else 0)

    from fairseq import checkpoint_utils
    from fairseq.models import egnn
    from fairseq.models import geometric_protein_model
    from fairseq.models import surface_protein_model

    egnn.device = torch_device
    geometric_protein_model.device = torch_device
    surface_protein_model.device = torch_device

    models, _cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(Path(args.checkpoint).resolve())],
        arg_overrides={"data": str(repo_path)},
    )
    if len(models) != 1:
        raise RuntimeError(f"Expected one SurfPro model, loaded {len(models)}.")
    model = models[0].to(torch_device)
    model.eval()
    return Runtime(
        torch=torch,
        model=model,
        dictionary=task.tgt_dict,
        get_edges_batch=surface_protein_model.get_edges_batch,
        device=args.device,
        logit_temperature=args.logit_temperature,
    )


class SurfaceBuilder:
    def __init__(
        self,
        repo_path: Path,
        work_dir: Path,
        msms_bin: Path,
        pdb_to_xyzrn_bin: Path,
        max_surface_vertices: int,
    ):
        self.repo_path = repo_path.resolve()
        self.work_dir = work_dir.resolve() / "surface_cache"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.msms_bin = msms_bin.resolve()
        self.pdb_to_xyzrn_bin = pdb_to_xyzrn_bin.resolve()
        self.max_surface_vertices = max_surface_vertices

        sys.path.insert(0, str(self.repo_path))
        from preprocess import prepare_surface

        self.prepare_surface = prepare_surface

    def _as_pdb(self, structure_path: Path) -> Path:
        suffix = structure_path.suffix.lower()
        if suffix == ".pdb":
            return structure_path
        if suffix not in {".cif", ".mmcif"}:
            raise ValueError(
                "SurfPro requires an evaluator structure in PDB or mmCIF format."
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

    def _protein_input(self, structure_path: Path) -> ProteinInput:
        pdb_path = self._as_pdb(structure_path)
        output: list[str] = []
        chain_order: list[str] = []
        chain_residues: dict[str, dict[tuple[str, str], int]] = {}
        sequence: list[str] = []
        global_ordinal = 0

        for raw_line in pdb_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line
            record = line[:6].strip()
            resname = line[17:20].strip().upper() if len(line) >= 20 else ""
            if record == "HETATM" and resname == "MSE":
                line = "ATOM  " + line[6:17] + "MET" + line[20:]
                resname = "MET"
                record = "ATOM"
            if record != "ATOM" or resname not in AA3_TO_1 or len(line) < 54:
                continue
            altloc = line[16:17]
            if altloc not in {"", " ", "A"}:
                continue
            chain = line[21:22]
            if not chain.strip():
                raise ValueError("SurfPro requires non-empty one-character PDB chain IDs.")
            if chain not in chain_order:
                chain_order.append(chain)
                chain_residues[chain] = {}
            residue_key = (line[22:26], line[26:27])
            if residue_key not in chain_residues[chain]:
                global_ordinal += 1
                chain_residues[chain][residue_key] = global_ordinal
                sequence.append(AA3_TO_1[resname])
            ordinal = chain_residues[chain][residue_key]
            normalized = line[:16] + " " + line[17:22] + f"{ordinal:4d}" + " " + line[27:]
            output.append(normalized)

        if not output:
            raise ValueError(f"SurfPro found no standard protein atoms in {pdb_path}.")
        protein_text = "\n".join(output) + "\nEND\n"
        source_hash = hashlib.sha256(protein_text.encode("utf-8")).hexdigest()
        normalized_path = self.work_dir / f"{source_hash}.protein.pdb"
        if not normalized_path.is_file():
            normalized_path.write_text(protein_text, encoding="utf-8")
        return ProteinInput(
            pdb_path=normalized_path,
            sequence="".join(sequence),
            chain_order=chain_order,
            chain_lengths={chain: len(chain_residues[chain]) for chain in chain_order},
        )

    @staticmethod
    def _binary_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"

    def _surface_cache_key(self, protein: ProteinInput, seed: int) -> str:
        digest = hashlib.sha256()
        digest.update(protein.pdb_path.read_bytes())
        digest.update(self._binary_fingerprint(self.msms_bin).encode("utf-8"))
        digest.update(self._binary_fingerprint(self.pdb_to_xyzrn_bin).encode("utf-8"))
        digest.update(f"{self.max_surface_vertices}:{seed}".encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SurfPro preprocessing command failed ({result.returncode}): "
                f"{' '.join(command)}\n{result.stderr}"
            )
        return result

    def _generate_msms_surface(self, protein: ProteinInput, prefix: Path) -> Path:
        xyzrn_path = prefix.with_suffix(".xyzrn")
        with xyzrn_path.open("w", encoding="utf-8") as handle:
            self._run_checked(
                [str(self.pdb_to_xyzrn_bin), str(protein.pdb_path)],
                stdout=handle,
                cwd=str(self.pdb_to_xyzrn_bin.parent),
            )
        if not xyzrn_path.is_file() or xyzrn_path.stat().st_size == 0:
            raise RuntimeError(f"pdb_to_xyzrn produced an empty file: {xyzrn_path}")

        self._run_checked(
            [str(self.msms_bin), "-if", str(xyzrn_path), "-of", str(prefix)],
            stdout=subprocess.PIPE,
        )
        vert_path = prefix.with_suffix(".vert")
        if not vert_path.is_file() or vert_path.stat().st_size == 0:
            raise RuntimeError(f"MSMS did not produce a non-empty vert file: {vert_path}")
        return vert_path

    def _prepare_features(self, vert_path: Path):
        import numpy as np

        vertices, atom_labels = self.prepare_surface.extract_vertix(str(vert_path))
        if len(vertices) != len(atom_labels) or len(vertices) <= 1:
            raise RuntimeError(
                f"SurfPro surface preprocessing produced {len(vertices)} usable vertices."
            )
        coordinates = np.asarray(vertices, dtype=np.float64)
        center = np.mean(coordinates, axis=0)
        extent = float(np.max(np.max(coordinates, axis=0) - np.min(coordinates, axis=0)))
        if not np.isfinite(extent) or extent <= 0:
            raise RuntimeError(f"SurfPro surface has invalid coordinate extent: {extent}")
        coordinates = (coordinates - center) / extent

        selected = list(range(len(coordinates)))
        if len(selected) > self.max_surface_vertices:
            ids = self.prepare_surface.octree(coordinates)
            grouped: dict[int, list[int]] = {}
            for index, voxel_id in enumerate(ids.tolist()):
                grouped.setdefault(int(voxel_id), []).append(index)
            ratio = float(self.max_surface_vertices) / len(selected)
            selected = []
            for indices in grouped.values():
                count = int(len(indices) * ratio)
                selected.extend(random.sample(indices, count))
        if len(selected) <= 1:
            raise RuntimeError(
                "SurfPro octree downsampling left fewer than two surface vertices."
            )

        parsed_labels: list[tuple[int, int]] = []
        for index in selected:
            fields = atom_labels[index].split("_")
            if len(fields) < 3 or fields[1] not in AA3_TO_1:
                raise ValueError(
                    "SurfPro requires pdb_to_xyzrn/MSMS atom labels in "
                    f"ATOM_RESNAME_RESID format, got {atom_labels[index]!r}."
                )
            try:
                residue_index = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"SurfPro surface label has a non-integer residue index: {atom_labels[index]!r}."
                ) from exc
            parsed_labels.append((residue_index, AA_ORDER.index(AA3_TO_1[fields[1]])))

        order = sorted(range(len(selected)), key=lambda idx: parsed_labels[idx][0])
        selected_array = np.asarray([selected[index] for index in order], dtype=int)
        aa_indices = np.asarray(
            [parsed_labels[index][1] for index in order], dtype=np.int64
        )
        return coordinates[selected_array].astype(np.float32), aa_indices

    def build(self, structure_path: Path, seed: int):
        import numpy as np

        protein = self._protein_input(structure_path)
        cache_key = self._surface_cache_key(protein, seed)
        cache_path = self.work_dir / f"{cache_key}.surface.npz"
        if cache_path.is_file():
            with np.load(cache_path) as cached:
                coordinates = cached["coordinates"]
                aa_indices = cached["aa_indices"]
            return protein, coordinates, aa_indices

        prefix = self.work_dir / cache_key
        vert_path = self._generate_msms_surface(protein, prefix)
        coordinates, aa_indices = self._prepare_features(vert_path)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            coordinates=coordinates,
            aa_indices=aa_indices,
        )
        temporary.replace(cache_path)
        return protein, coordinates, aa_indices


def _fixed_flat_indices(
    encoded_residues: str, chain_order: list[str], chain_lengths: dict[str, int]
) -> set[int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for chain in chain_order:
        offsets[chain] = cursor
        cursor += chain_lengths[chain]
    flat_indices: set[int] = set()
    for encoded in encoded_residues.split():
        match = re.fullmatch(r"([A-Za-z0-9])(\d+)([A-Za-z]?)", encoded)
        if match is None:
            raise ValueError(f"Invalid SurfPro fixed residue identifier: {encoded!r}")
        chain, position_text, _insertion = match.groups()
        if chain not in offsets:
            raise ValueError(f"SurfPro fixed residue uses missing chain {chain!r}: {encoded!r}")
        position = int(position_text)
        if position < 1 or position > chain_lengths[chain]:
            raise ValueError(f"SurfPro fixed residue is out of range: {encoded!r}")
        flat_indices.add(offsets[chain] + position - 1)
    return flat_indices


def _encode_surface(runtime: Runtime, coordinates, aa_indices):
    torch = runtime.torch
    model = runtime.model
    coor_features = torch.as_tensor(
        coordinates, dtype=torch.float32, device=runtime.device
    ).unsqueeze(0)
    aa_features = torch.as_tensor(
        aa_indices, dtype=torch.long, device=runtime.device
    ).unsqueeze(0)
    src_lengths = torch.tensor(
        [len(aa_indices)], dtype=torch.long, device=runtime.device
    )

    batch_size = 1
    surf_aa_embs = torch.index_select(
        model.surf_aa_features, 0, aa_features.reshape(-1)
    ).reshape(batch_size, -1, 2)
    flattened_coordinates = coor_features.view(batch_size, -1, 3)
    k = min(model.k, int(src_lengths.min().item()) - 1)
    if k < 1:
        raise RuntimeError("SurfPro requires at least two surface vertices.")
    edges = runtime.get_edges_batch(
        flattened_coordinates.size(1),
        batch_size,
        flattened_coordinates.detach().cpu(),
        k,
    )
    hidden, encoded_coordinates = model.encoder(
        surf_aa_embs,
        flattened_coordinates,
        edges,
        None,
        k,
    )
    hidden = hidden.reshape(batch_size, -1, hidden.size(-1))
    encoded_coordinates = encoded_coordinates.reshape(batch_size, -1, 3)
    hidden = model.pooling_encoder(
        (None, encoded_coordinates, None, hidden)
    )
    encoder_out = {
        "encoder_out": [hidden.transpose(0, 1)],
        "encoder_padding_mask": [],
    }
    return encoder_out, src_lengths


def _decode_sequence(
    runtime: Runtime,
    encoder_out,
    src_lengths,
    native_sequence: str,
    fixed_indices: set[int],
) -> tuple[list[list[float]], str]:
    torch = runtime.torch
    dictionary = runtime.dictionary
    aa_token_indices = [dictionary.get_idx(aa) for aa in AA_ORDER]
    native_tokens = [dictionary.get_idx(aa) for aa in native_sequence]
    prev_tokens = torch.tensor(
        [[dictionary.bos()]], dtype=torch.long, device=runtime.device
    )
    probabilities: list[list[float]] = []
    generated: list[str] = []

    for position, native_token in enumerate(native_tokens):
        decoder_out, _extra = runtime.model.decoder.inference_forward(
            prev_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=True,
        )
        step_logits = decoder_out[:, -1, :]
        aa_logits = step_logits[:, aa_token_indices]
        aa_probabilities = torch.softmax(
            aa_logits.float() / runtime.logit_temperature,
            dim=-1,
        )
        probabilities.append(aa_probabilities[0].cpu().tolist())

        if position in fixed_indices:
            next_token = native_token
        else:
            aa_choice = int(torch.argmax(aa_logits, dim=-1).item())
            next_token = aa_token_indices[aa_choice]
        generated.append(dictionary.get_tok(next_token))
        next_tensor = torch.tensor(
            [[next_token]], dtype=torch.long, device=runtime.device
        )
        prev_tokens = torch.cat((prev_tokens, next_tensor), dim=1)
    return probabilities, "".join(generated)


def _infer(
    runtime: Runtime,
    surface_builder: SurfaceBuilder,
    pdb_path: Path,
    fixed_residues: str,
    seed: int,
) -> tuple[list[list[float]], str]:
    protein, coordinates, aa_indices = surface_builder.build(pdb_path, seed)
    if len(protein.sequence) > 1024:
        raise ValueError(
            f"SurfPro supports at most 1024 residues, got {len(protein.sequence)}."
        )
    fixed_indices = _fixed_flat_indices(
        fixed_residues, protein.chain_order, protein.chain_lengths
    )
    with runtime.torch.no_grad():
        encoder_out, src_lengths = _encode_surface(
            runtime, coordinates, aa_indices
        )
        return _decode_sequence(
            runtime,
            encoder_out,
            src_lengths,
            protein.sequence,
            fixed_indices,
        )


def main() -> int:
    args = _parse_args()
    protocol_out = sys.stdout
    repo_path = Path(args.repo_path).resolve()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = _load_runtime(args)
        surface_builder = SurfaceBuilder(
            repo_path=repo_path,
            work_dir=Path(args.work_dir),
            msms_bin=Path(args.msms_bin),
            pdb_to_xyzrn_bin=Path(args.pdb_to_xyzrn_bin),
            max_surface_vertices=args.max_surface_vertices,
        )
    _write(
        protocol_out,
        {"type": "ready", "pid": os.getpid(), "model_load_count": 1},
    )

    request_count = 0
    for raw in sys.stdin:
        message = json.loads(raw)
        message_type = message.get("type")
        if message_type == "shutdown":
            _write(protocol_out, {"type": "ok", "message": "bye"})
            return 0
        if message_type != "infer":
            raise ValueError(f"Unknown SurfPro daemon message type: {message_type!r}")

        request_count += 1
        seed = int(message.get("seed", 0))
        _seed_everything(seed)
        with contextlib.redirect_stdout(sys.stderr):
            probabilities, sequence = _infer(
                runtime,
                surface_builder,
                Path(message["pdb_path"]).resolve(),
                str(message.get("fixed_residues") or ""),
                seed,
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
