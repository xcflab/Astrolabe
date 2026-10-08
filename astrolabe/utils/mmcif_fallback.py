from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence

import numpy as np


def write_centered_pdb_from_mmcif(mmcif_path: pathlib.Path, pdb_path: pathlib.Path) -> None:
    from Bio.PDB import MMCIFParser, PDBIO  # type: ignore

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", str(mmcif_path))

    coords = []
    for atom in structure.get_atoms():
        c = atom.get_coord()
        if c is None:
            continue
        c = np.asarray(c, dtype=float)
        if np.isfinite(c).all():
            coords.append(c)

    centroid = None
    scale = 1.0
    if coords:
        coords_arr = np.stack(coords, axis=0)
        centroid = np.mean(coords_arr, axis=0)
        centered = coords_arr - centroid
        maxabs = float(np.abs(centered).max())
        if maxabs > 999.0:
            scale = 950.0 / maxabs

    if centroid is not None:
        for atom in structure.get_atoms():
            c = atom.get_coord()
            if c is None:
                continue
            c = np.asarray(c, dtype=float)
            if np.isfinite(c).all():
                atom.set_coord((c - centroid) * scale)

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(pdb_path.parent),
        prefix=pdb_path.name + ".tmp.",
        suffix=".pdb",
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        io.save(str(tmp_path))
        os.replace(str(tmp_path), str(pdb_path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def ensure_centered_pdb_fallback(mmcif_path: pathlib.Path, suffix: str = ".pyrosetta_fallback.pdb") -> pathlib.Path:
    pdb_path = mmcif_path.with_suffix(suffix)
    try:
        cif_mtime = mmcif_path.stat().st_mtime
    except FileNotFoundError:
        return pdb_path

    needs_write = True
    try:
        needs_write = (not pdb_path.exists()) or (pdb_path.stat().st_mtime < cif_mtime)
    except FileNotFoundError:
        needs_write = True

    if needs_write:
        write_centered_pdb_from_mmcif(mmcif_path, pdb_path)
    return pdb_path


def _is_already_loaded_residue_type_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "Attempting to add a new residue type" in text
        and "already exists in the cache" in text
    )


def _generate_nonstandard_residue_set(pyrosetta, pose, params: Sequence[str]) -> None:
    seen: set[str] = set()
    unique_params: list[str] = []
    for param in params:
        if param in seen:
            continue
        seen.add(param)
        unique_params.append(param)

    for param in unique_params:
        try:
            pyrosetta.generate_nonstandard_residue_set(pose, [param])
        except Exception as exc:
            if _is_already_loaded_residue_type_error(exc):
                continue
            raise


def load_pose_with_optional_mmcif_pdb_fallback(
    pyrosetta,
    structure_path: str | pathlib.Path,
    *,
    allow_mmcif_pdb_fallback: bool = False,
    fallback_builder=ensure_centered_pdb_fallback,
    unsupported_format_as_pdb: bool = False,
    extra_residue_params: Sequence[str | pathlib.Path] | None = None,
) -> object:
    path = pathlib.Path(str(structure_path))
    path_str = str(path)
    params = [str(p) for p in (extra_residue_params or [])]

    def _pose_from_file_with_params(filename: str, **kwargs):
        if not params:
            return pyrosetta.pose_from_file(filename, **kwargs)
        pose = pyrosetta.Pose()
        _generate_nonstandard_residue_set(pyrosetta, pose, params)
        pyrosetta.pose_from_file(pose, filename, **kwargs)
        return pose

    def _pose_from_pdb_with_params(filename: str):
        if not params:
            return pyrosetta.pose_from_pdb(filename)
        pose = pyrosetta.Pose()
        _generate_nonstandard_residue_set(pyrosetta, pose, params)
        pyrosetta.pose_from_file(pose, filename)
        return pose

    if path_str.endswith(".pdb"):
        return _pose_from_pdb_with_params(path_str)
    if path_str.endswith(".cif"):
        try:
            return _pose_from_file_with_params(
                path_str,
                type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file,
                read_fold_tree=True,
            )
        except Exception:
            try:
                return _pose_from_file_with_params(
                    path_str,
                    type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file,
                    read_fold_tree=False,
                )
            except Exception as exc:
                if allow_mmcif_pdb_fallback:
                    fallback_path = fallback_builder(path)
                    return _pose_from_pdb_with_params(str(fallback_path))
                raise RuntimeError(
                    "PyRosetta failed to load mmCIF structure: "
                    f"{path_str}. This run disallows mmCIF->PDB fallbacks."
                ) from exc
    if unsupported_format_as_pdb:
        return _pose_from_pdb_with_params(path_str)
    return _pose_from_file_with_params(path_str)
