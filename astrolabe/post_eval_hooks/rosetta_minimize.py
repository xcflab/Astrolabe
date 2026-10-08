from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized


_PYROSETTA_INITED = False


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return dict(value or {})


def _pyrosetta_init_options() -> str:
    init_opts = "-ignore_zero_occupancy false -mute all -beta"
    candidates: list[Path] = []
    dalpha = os.environ.get("GH_DALPHABALL", "").strip()
    if dalpha:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(dalpha))))
    candidates.append(Path(__file__).resolve().parents[1] / "assets" / "DAlphaBall.gcc")
    dalpha_path = next((p for p in candidates if p.exists()), None)
    if dalpha_path is None:
        return init_opts
    try:
        mode = dalpha_path.stat().st_mode
        if (mode & 0o100) == 0:
            dalpha_path.chmod(mode | 0o100)
    except Exception:
        pass
    return f"{init_opts} -holes:dalphaball {dalpha_path}"


def _ensure_pyrosetta_initialized():
    global _PYROSETTA_INITED
    import pyrosetta  # type: ignore

    if _PYROSETTA_INITED:
        return pyrosetta
    if pyrosetta_was_initialized(pyrosetta):
        _PYROSETTA_INITED = True
        return pyrosetta
    pyrosetta.init(_pyrosetta_init_options())
    _PYROSETTA_INITED = True
    return pyrosetta


def _pose_from_path(path: Path):
    pyrosetta = _ensure_pyrosetta_initialized()

    from astrolabe.core import global_config
    from astrolabe.utils.mmcif_fallback import (
        ensure_centered_pdb_fallback,
        load_pose_with_optional_mmcif_pdb_fallback,
    )

    try:
        allow_mmcif_pdb_fallback = bool(global_config.allow_mmcif_pdb_fallback)
    except RuntimeError:
        allow_mmcif_pdb_fallback = False
    return load_pose_with_optional_mmcif_pdb_fallback(
        pyrosetta,
        str(path),
        allow_mmcif_pdb_fallback=allow_mmcif_pdb_fallback,
        fallback_builder=ensure_centered_pdb_fallback,
    )


def _score_pose(pose, scorefxn) -> float:
    return float(scorefxn(pose))


def _pack_pose(pose, scorefxn) -> None:
    pyrosetta = _ensure_pyrosetta_initialized()

    task = pyrosetta.standard_packer_task(pose)
    task.restrict_to_repacking()
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(
        scorefxn,
        task,
    )
    packer.apply(pose)


def _relax_pose(pose, scorefxn, *, repeats: int, cartesian: bool) -> None:
    pyrosetta = _ensure_pyrosetta_initialized()

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.max_iter(max(1, int(repeats)))
    try:
        relax.cartesian(bool(cartesian))
    except Exception:
        if cartesian:
            raise
    relax.apply(pose)


def _minimize_path(
    source_path: Path,
    output_path: Path,
    *,
    mode: str,
    only_if_energy_above: float | None,
    relax_repeats: int,
    cartesian: bool,
) -> tuple[Path, dict[str, float | str | bool]]:
    pyrosetta = _ensure_pyrosetta_initialized()

    pose = _pose_from_path(source_path)
    scorefxn = pyrosetta.get_fa_scorefxn()
    before = _score_pose(pose, scorefxn)
    should_run = only_if_energy_above is None or before > float(only_if_energy_above)
    if should_run:
        if mode == "repack":
            _pack_pose(pose, scorefxn)
        elif mode == "relax":
            _relax_pose(
                pose,
                scorefxn,
                repeats=int(relax_repeats),
                cartesian=bool(cartesian),
            )
        elif mode == "repack_then_relax":
            _pack_pose(pose, scorefxn)
            _relax_pose(
                pose,
                scorefxn,
                repeats=int(relax_repeats),
                cartesian=bool(cartesian),
            )
        else:
            raise ValueError(
                "post_eval_hooks.rosetta_minimize.mode must be one of "
                "'repack', 'relax', or 'repack_then_relax'."
            )
        after = _score_pose(pose, scorefxn)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(output_path))
        chosen = output_path
    else:
        after = before
        chosen = source_path
    return chosen, {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "mode": mode,
        "ran": bool(should_run),
        "energy_before": float(before),
        "energy_after": float(after),
    }


def apply_rosetta_minimize_hook(
    evaluation_result,
    hook_config,
    *,
    out_dir: Path,
    step: int,
):
    """Run Rosetta repack/relax immediately after evaluator output is available."""
    cfg = _to_plain_mapping(hook_config)
    mode = str(cfg.get("mode", "repack")).strip().lower()
    only_if_energy_above = cfg.get("only_if_energy_above")
    only_if_energy_above = (
        None if only_if_energy_above in (None, "null", "None") else float(only_if_energy_above)
    )
    relax_repeats = int(cfg.get("relax_repeats", 5) or 5)
    cartesian = bool(cfg.get("cartesian", False))

    new_paths = dict(getattr(evaluation_result, "complex_paths", {}) or {})
    reports: dict[str, dict[str, float | str | bool]] = {}
    for oligo_name, complex_path in list(new_paths.items()):
        source_path = Path(complex_path)
        suffix = ".pdb"
        output_path = out_dir / f"step_{step:03d}_{oligo_name}_{mode}{suffix}"
        chosen, report = _minimize_path(
            source_path,
            output_path,
            mode=mode,
            only_if_energy_above=only_if_energy_above,
            relax_repeats=relax_repeats,
            cartesian=cartesian,
        )
        new_paths[oligo_name] = chosen
        reports[oligo_name] = report
        logging.info(
            "post_eval rosetta_minimize oligo=%s mode=%s ran=%s energy %.3f -> %.3f",
            oligo_name,
            report["mode"],
            report["ran"],
            report["energy_before"],
            report["energy_after"],
        )

    evaluation_result.complex_paths = new_paths
    evaluation_result.post_eval_hook_reports = reports
    return evaluation_result
