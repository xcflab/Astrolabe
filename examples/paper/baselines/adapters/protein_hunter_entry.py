#!/usr/bin/env python3
from __future__ import annotations

import importlib
import runpy
import sys
import tempfile
from pathlib import Path

import gemmi
import torch


def _safe_backbone_frame_indices(
    *,
    token_exists_mask,
    token_backbone_frame_mask,
    token_backbone_frame_index,
    coord_len: int,
):
    exists = torch.as_tensor(token_exists_mask, dtype=torch.bool).reshape(-1)
    frame_mask = torch.as_tensor(token_backbone_frame_mask, dtype=torch.bool).reshape(-1)
    frame_index = torch.as_tensor(token_backbone_frame_index, dtype=torch.long).reshape(-1, 3)
    keep_rows = exists & frame_mask
    filtered = frame_index[keep_rows]
    if filtered.numel() == 0:
        return filtered
    in_bounds = ((filtered >= 0) & (filtered < coord_len)).all(dim=-1)
    return filtered[in_bounds]


def _prepare_ligandmpnn_input_structure(source_path: Path, dest_dir: Path) -> Path:
    source_path = Path(source_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() in {".cif", ".mmcif"}:
        dest_path = dest_dir / (source_path.stem + ".pdb")
        structure = gemmi.read_structure(str(source_path))
        structure.write_pdb(str(dest_path))
        return dest_path
    dest_path = dest_dir / source_path.name
    if source_path.resolve() == dest_path.resolve():
        return source_path
    dest_path.write_bytes(source_path.read_bytes())
    return dest_path


def _patched_get_backbone_coords_from_result(state):
    if state.result is None or state.batch_inputs is None:
        return torch.empty(0, 3, 3)

    padded_coords = state.result["coords"].cpu()
    if padded_coords.dim() == 3:
        padded_coords = padded_coords.squeeze(0)

    bb_indices = _safe_backbone_frame_indices(
        token_exists_mask=state.batch_inputs["token_exists_mask"],
        token_backbone_frame_mask=state.batch_inputs["token_backbone_frame_mask"],
        token_backbone_frame_index=state.batch_inputs["token_backbone_frame_index"],
        coord_len=padded_coords.shape[0],
    )
    if bb_indices.numel() == 0:
        raise RuntimeError("No valid backbone frame indices remain after filtering")
    return padded_coords[bb_indices]


def _patch_modules(protein_hunter_root: Path) -> None:
    chai_dir = protein_hunter_root / "chai_ph"
    root_str = str(protein_hunter_root)
    chai_str = str(chai_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if chai_str not in sys.path:
        sys.path.insert(0, chai_str)

    pipeline = importlib.import_module("pipeline")
    helpers = importlib.import_module("chai_ph.helpers")
    wrapper_mod = importlib.import_module("LigandMPNN.wrapper")

    helpers.get_backbone_coords_from_result = _patched_get_backbone_coords_from_result
    pipeline.get_backbone_coords_from_result = _patched_get_backbone_coords_from_result

    original_run = wrapper_mod.LigandMPNNWrapper.run

    def patched_run(self, pdb_path, *args, **kwargs):
        pdb_path = Path(pdb_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = _prepare_ligandmpnn_input_structure(pdb_path, Path(tmpdir))
            return original_run(self, str(prepared), *args, **kwargs)

    wrapper_mod.LigandMPNNWrapper.run = patched_run


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: protein_hunter_entry.py <design.py> [args...]")
    design_path = Path(sys.argv[1]).resolve()
    protein_hunter_root = design_path.parent.parent
    _patch_modules(protein_hunter_root)
    sys.argv = [str(design_path)] + sys.argv[2:]
    runpy.run_path(str(design_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
