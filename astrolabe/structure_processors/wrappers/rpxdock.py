from __future__ import annotations

import os
import pickle
import subprocess
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from astrolabe.core.repeat_symmetry import kabsch_transform


class RpxDockDependencyError(FileNotFoundError):
    """Raised when vendored rpxdock is unavailable or incomplete."""


class RpxDockExecutionError(RuntimeError):
    """Raised when rpxdock command execution fails."""


class RpxDockOutputError(RuntimeError):
    """Raised when rpxdock did not produce parseable docking outputs."""


@dataclass(frozen=True)
class RpxDockRunConfig:
    """Runtime options for a local one-component rpxdock run."""

    nfold: int
    output_dir: Path
    python_bin: str = "python"
    vendor_repo: Path | None = None
    willutil_repo: Path | None = None
    docking_method: str = "hier"
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    dump_pdbs: bool = True
    timeout_s: int = 900
    env: Mapping[str, str] | None = None
    architecture: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    cart_bounds: tuple[float, float] | None = None
    max_longaxis_dot_z: float | None = None
    components_aligned_to_sym_axes: bool = False
    limit_rotation_to_z: bool = True
    hscore_data_dir: Path | None = None
    hscore_files: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RpxDockCandidate:
    """One ranked candidate returned by a local rpxdock run."""

    rank: int
    docked_structure_path: Path
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class RpxDockRunResult:
    """Docking result for predocker integration."""

    docked_structure_path: Path
    rotation: np.ndarray
    translation: np.ndarray
    command: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    candidates: tuple[RpxDockCandidate, ...] = field(default_factory=tuple)


def resolve_vendored_rpxdock(vendor_repo: str | Path | None = None) -> Path:
    """
    Resolve vendored rpxdock repository path.

    Raises:
        RpxDockDependencyError: when the vendored path is missing or incomplete.
    """

    if vendor_repo is None:
        repo_root = Path(__file__).resolve().parents[3]
        repo = repo_root / "vendor" / "rpxdock"
    else:
        repo = Path(vendor_repo).expanduser().resolve()

    if not repo.exists():
        raise RpxDockDependencyError(
            f"Vendored rpxdock not found at {repo}. "
            "Expected a git-submodule checkout under vendor/rpxdock."
        )

    cli_py = repo / "rpxdock" / "app" / "dock.py"
    if not cli_py.exists():
        raise RpxDockDependencyError(
            f"Vendored rpxdock CLI entrypoint missing: {cli_py}. "
            "The checkout appears incomplete."
        )
    return repo


def build_cyclic_dock_command(
    monomer_path: str | Path,
    config: RpxDockRunConfig,
) -> list[str]:
    """
    Build a CLI command for cyclic docking with rpxdock.

    The command uses the upstream `python -m rpxdock.app.dock` interface.
    """

    monomer = Path(monomer_path).expanduser().resolve()
    architecture = config.architecture or f"C{int(config.nfold)}"
    arch_norm = str(architecture).strip().upper()
    if arch_norm.startswith("C") and int(config.nfold) < 2:
        raise ValueError(f"nfold must be >=2 for cyclic docking, got {config.nfold}.")
    if not (arch_norm.startswith("C") or arch_norm.startswith("D")):
        raise ValueError(f"Only Cn and Dn one-component rpxdock architectures are supported, got {architecture!r}.")
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_prefix = output_dir / "rpxdock_cyclic"

    cmd = [
        str(config.python_bin),
        "-m",
        "rpxdock.app.dock",
        "--architecture",
        architecture,
        "--inputs1",
        str(monomer),
        "--docking_method",
        str(config.docking_method),
        "--output_prefix",
        str(output_prefix),
        "--save_results_as_tarball",
        "False",
    ]
    if bool(config.dump_pdbs):
        cmd.append("--dump_pdbs")
    if config.hscore_data_dir is not None:
        cmd.extend(["--hscore_data_dir", str(Path(config.hscore_data_dir).expanduser().resolve())])
    if config.hscore_files:
        cmd.append("--hscore_files")
        cmd.extend([str(x) for x in config.hscore_files])
    if config.cart_bounds is not None:
        lb, ub = config.cart_bounds
        cmd.extend(["--cart_bounds", str(float(lb)), str(float(ub))])
    if config.max_longaxis_dot_z is not None:
        cmd.extend(["--max_longaxis_dot_z", str(float(config.max_longaxis_dot_z))])
    if bool(config.components_aligned_to_sym_axes):
        cmd.append("--components_already_aligned_to_sym_axes")
    if bool(config.limit_rotation_to_z):
        cmd.append("--limit_rotation_to_z")
    cmd.extend([str(x) for x in config.extra_args])
    return cmd


def _build_run_env(
    vendored_rpxdock: Path,
    vendored_willutil: Path | None,
    extra_env: Mapping[str, str] | None,
    *,
    prepend_python_paths: Sequence[str | Path] = (),
) -> dict[str, str]:
    env = dict(os.environ)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    env.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    env.setdefault("PYTHONSAFEPATH", "1")

    # Ensure vendored checkout is imported first.
    py_paths = [str(Path(path).expanduser().resolve()) for path in prepend_python_paths]
    py_paths.append(str(vendored_rpxdock))
    if vendored_willutil is not None:
        py_paths.append(str(vendored_willutil))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    return env


def _write_rpxdock_compat_sitecustomize(output_dir: Path) -> Path:
    compat_dir = Path(output_dir).expanduser().resolve() / "_gh_rpxdock_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        r'''
from __future__ import annotations

import numpy as _np

import rpxdock as _rp
from rpxdock.body import body as _body_mod


def _gh_contiguous_xform(value, dtype):
    arr = _np.asarray(value)
    if arr.ndim in (2, 3) and arr.shape[-2:] == (4, 4):
        return _np.ascontiguousarray(arr, dtype=dtype)
    return value


def _gh_wrap_bvh_xform_function(name, dtype):
    if not hasattr(_rp.bvh, name):
        return
    original = getattr(_rp.bvh, name)
    if getattr(original, "__gh_xform_patch__", False):
        return

    def _wrapped(bvh1, bvh2, pos1, pos2, *args, **kwargs):
        pos1 = _gh_contiguous_xform(pos1, dtype)
        pos2 = _gh_contiguous_xform(pos2, dtype)
        if (
            hasattr(pos1, "ndim")
            and hasattr(pos2, "ndim")
            and pos1.ndim == 3
            and pos2.ndim == 3
            and (pos1.shape[0] == 0 or pos2.shape[0] == 0)
        ):
            if name in {"bvh_collect_pairs_vec", "bvh_collect_pairs_range_vec"}:
                return (
                    _np.zeros((0, 2), dtype=_np.int32),
                    _np.zeros((0, 2), dtype=_np.int32),
                )
            if name == "bvh_isect_fixed_range_vec":
                return (
                    _np.zeros((0,), dtype=bool),
                    _np.zeros((0, 2), dtype=_np.int32),
                )
            if name == "bvh_isect_vec":
                return _np.zeros((0,), dtype=bool)
            if name == "bvh_count_pairs_vec":
                return _np.zeros((0,), dtype=_np.int32)
            if name == "bvh_slide_vec":
                return _np.zeros((0,), dtype=_np.float64)
        return original(
            bvh1,
            bvh2,
            pos1,
            pos2,
            *args,
            **kwargs,
        )

    _wrapped.__gh_xform_patch__ = True
    setattr(_rp.bvh, name, _wrapped)


for _gh_name in (
    "bvh_isect",
    "bvh_isect_vec",
    "bvh_isect_fixed_range_vec",
    "bvh_min_dist",
    "bvh_min_dist_vec",
    "bvh_count_pairs",
    "bvh_count_pairs_vec",
    "bvh_slide",
    "bvh_slide_vec",
    "isect_range",
):
    _gh_wrap_bvh_xform_function(_gh_name, _np.float64)

for _gh_name in (
    "bvh_collect_pairs",
    "bvh_collect_pairs_vec",
    "bvh_collect_pairs_range_vec",
):
    _gh_wrap_bvh_xform_function(_gh_name, _np.float32)


def _gh_intersect_float64(self, other, xself=None, xother=None, mindis=2 * _body_mod._CLASHRAD, **kw):
    xself = self.pos if xself is None else xself
    xother = other.pos if xother is None else xother
    if xself.shape == xother.shape:
        origshape = xself.shape[:-2]
        xself_arr = _np.ascontiguousarray(xself.reshape(-1, 4, 4), dtype=_np.float64)
        xother_arr = _np.ascontiguousarray(xother.reshape(-1, 4, 4), dtype=_np.float64)
        isect = _rp.bvh.bvh_isect_vec(
            self.bvh_bb,
            other.bvh_bb,
            xself_arr,
            xother_arr,
            mindis,
        )
        return isect.reshape(origshape)
    return _rp.bvh.bvh_isect_vec(
        self.bvh_bb,
        other.bvh_bb,
        _np.ascontiguousarray(xself, dtype=_np.float64),
        _np.ascontiguousarray(xother, dtype=_np.float64),
        mindis,
    )


_gh_intersect_float64.__gh_float64_patch__ = True
if not getattr(_body_mod.Body.intersect, "__gh_float64_patch__", False):
    _body_mod.Body.intersect = _gh_intersect_float64
'''.lstrip(),
        encoding="utf-8",
    )
    return compat_dir


def _write_command_logs(
    *,
    output_dir: Path,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str] | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = stdout_path or (output_dir / "rpxdock.stdout.log")
    stderr_file = stderr_path or (output_dir / "rpxdock.stderr.log")
    cmd_txt = " ".join(command) + "\n"
    if completed is None:
        stdout_file.write_text(cmd_txt, encoding="utf-8")
        stderr_file.write_text("", encoding="utf-8")
    else:
        stdout_file.write_text(cmd_txt + (completed.stdout or ""), encoding="utf-8")
        stderr_file.write_text(completed.stderr or "", encoding="utf-8")
    return stdout_file, stderr_file


def _load_first_chain_ca_coords(path: Path) -> np.ndarray:
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("rpxdock", str(path))
    model = next(structure.get_models(), None)
    if model is None:
        raise RpxDockOutputError(f"Docked structure has no models: {path}")
    first_chain = next(iter(model), None)
    if first_chain is None:
        raise RpxDockOutputError(f"Docked structure has no chains: {path}")
    coords = []
    for residue in first_chain:
        if "CA" in residue:
            coords.append(np.asarray(residue["CA"].coord, dtype=np.float64))
    if not coords:
        raise RpxDockOutputError(f"No CA atoms found in first chain of: {path}")
    return np.asarray(coords, dtype=np.float64)


def _collect_docked_structure_paths(output_dir: Path, monomer_path: Path) -> list[Path]:
    candidates = []
    for ext in ("*.pdb", "*.cif", "*.mmcif"):
        candidates.extend(output_dir.rglob(ext))
    monomer_name = monomer_path.name
    filtered = [p.resolve() for p in candidates if p.is_file() and p.name != monomer_name]
    if not filtered:
        raise RpxDockOutputError(
            f"rpxdock produced no docked structure under {output_dir}."
        )
    return filtered


def _candidate_preference_key(path: Path) -> tuple[int, int, float]:
    name = path.name
    overall = re.search(r"__top(\d+)_", name)
    if overall is not None and "__job" not in name:
        return (0, int(overall.group(1)), -path.stat().st_mtime)
    per_job = re.search(r"__job\d+_top(\d+)_", name)
    if per_job is not None:
        return (1, int(per_job.group(1)), -path.stat().st_mtime)
    return (2, 0, -path.stat().st_mtime)


def _candidate_rank(path: Path) -> int | None:
    match = re.search(r"__top(\d+)_", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _list_overall_docked_candidates(output_dir: Path, monomer_path: Path) -> tuple[Path, ...]:
    filtered = [
        path
        for path in _collect_docked_structure_paths(output_dir, monomer_path)
        if "__job" not in path.name and re.search(r"__top\d+_", path.name)
    ]
    filtered.sort(key=_candidate_preference_key)
    return tuple(filtered)


def _pick_docked_structure(output_dir: Path, monomer_path: Path) -> Path:
    filtered = _collect_docked_structure_paths(output_dir, monomer_path)
    filtered.sort(key=_candidate_preference_key)
    return filtered[0]


def _find_result_pickle(output_dir: Path, architecture: str) -> Path | None:
    output_dir = Path(output_dir).expanduser().resolve()
    arch = str(architecture).strip()
    candidates = [
        output_dir / f"rpxdock_cyclic{arch}__Result.pickle",
        output_dir / f"rpxdock_cyclic{arch}_Result.pickle",
        output_dir / "rpxdock_cyclic_Result.pickle",
    ]
    for path in candidates:
        if path.exists():
            return path
    globbed = sorted(
        output_dir.glob("*_Result.pickle"),
        key=lambda path: (-path.stat().st_mtime, path.name),
    )
    if globbed:
        return globbed[0]
    return None


def _normalize_xforms_array(result_pickle: str | Path) -> np.ndarray:
    path = Path(result_pickle).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"rpxdock result pickle not found: {path}")
    with path.open("rb") as fh:
        result = pickle.load(fh)
    xforms = np.asarray(result["xforms"].data, dtype=np.float64)
    if xforms.ndim != 3 or xforms.shape[1:] != (4, 4) or xforms.shape[0] < 1:
        raise RpxDockOutputError(
            f"Unexpected xforms shape in {path}: {xforms.shape}"
        )
    return xforms


def _extract_transforms_from_result_pickle_all(
    result_pickle: str | Path,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    xforms = _normalize_xforms_array(result_pickle)
    out = []
    for xform in xforms:
        arr = np.asarray(xform, dtype=np.float64)
        out.append(
            (
                np.asarray(arr[:3, :3].T, dtype=np.float64),
                np.asarray(arr[:3, 3], dtype=np.float64),
            )
        )
    return tuple(out)


def _extract_transform_from_result_pickle(result_pickle: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return _extract_transforms_from_result_pickle_all(result_pickle)[0]


def _extract_transforms_from_result_pickle_all_external(
    *,
    python_bin: str,
    vendored_rpxdock: Path,
    vendored_willutil: Path | None,
    result_pickle: str | Path,
    env: Mapping[str, str] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    script = """
import json, pickle, numpy as np, sys
from pathlib import Path
p = Path(sys.argv[1]).expanduser().resolve()
with p.open('rb') as fh:
    result = pickle.load(fh)
xforms = np.asarray(result['xforms'].data, dtype=np.float64)
if xforms.ndim != 3 or xforms.shape[1:] != (4, 4) or xforms.shape[0] < 1:
    raise RuntimeError(f'unexpected xforms shape: {xforms.shape}')
payload = [
    {
        'rotation': np.asarray(xform[:3, :3].T, dtype=np.float64).tolist(),
        'translation': np.asarray(xform[:3, 3], dtype=np.float64).tolist(),
    }
    for xform in xforms
]
print(json.dumps(payload))
""".strip()
    run_env = _build_run_env(vendored_rpxdock, vendored_willutil, env)
    result_path = Path(result_pickle).expanduser().resolve()
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", script, str(result_path)],
            check=True,
            text=True,
            capture_output=True,
            env=run_env,
            cwd=str(result_path.parent),
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RpxDockOutputError(
            f"Could not extract transforms from rpxdock result pickle {result_path}: {detail}"
        ) from exc
    payload = json.loads(completed.stdout.strip())
    return tuple(
        (
            np.asarray(item["rotation"], dtype=np.float64),
            np.asarray(item["translation"], dtype=np.float64),
        )
        for item in payload
    )


def _extract_transform_from_result_pickle_external(
    *,
    python_bin: str,
    vendored_rpxdock: Path,
    vendored_willutil: Path | None,
    result_pickle: str | Path,
    env: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _extract_transforms_from_result_pickle_all_external(
        python_bin=python_bin,
        vendored_rpxdock=vendored_rpxdock,
        vendored_willutil=vendored_willutil,
        result_pickle=result_pickle,
        env=env,
    )[0]


def _collect_docked_candidates(
    output_dir: Path,
    monomer_path: Path,
    result_pickle: Path,
    *,
    python_bin: str,
    vendored_rpxdock: Path,
    vendored_willutil: Path | None,
    env: Mapping[str, str] | None,
) -> tuple[RpxDockCandidate, ...]:
    try:
        overall_candidates = _list_overall_docked_candidates(output_dir, monomer_path)
    except RpxDockOutputError:
        overall_candidates = ()
    try:
        transforms = _extract_transforms_from_result_pickle_all(result_pickle)
    except RpxDockOutputError:
        raise
    except Exception:
        transforms = _extract_transforms_from_result_pickle_all_external(
            python_bin=python_bin,
            vendored_rpxdock=vendored_rpxdock,
            vendored_willutil=vendored_willutil,
            result_pickle=result_pickle,
            env=env,
        )

    candidates: list[RpxDockCandidate] = []
    for path in overall_candidates:
        rank = _candidate_rank(path)
        if rank is None or rank >= len(transforms):
            continue
        rotation, translation = transforms[rank]
        candidates.append(
            RpxDockCandidate(
                rank=rank,
                docked_structure_path=path,
                rotation=rotation,
                translation=translation,
            )
        )
    if candidates:
        return tuple(candidates)

    if not overall_candidates:
        return tuple(
            RpxDockCandidate(
                rank=idx,
                docked_structure_path=Path(result_pickle).expanduser().resolve(),
                rotation=rotation,
                translation=translation,
            )
            for idx, (rotation, translation) in enumerate(transforms)
        )

    primary = _pick_docked_structure(output_dir, monomer_path)
    rotation, translation = transforms[0]
    return (
        RpxDockCandidate(
            rank=0,
            docked_structure_path=primary,
            rotation=rotation,
            translation=translation,
        ),
    )


def _estimate_rigid_transform(monomer_path: Path, docked_path: Path) -> tuple[np.ndarray, np.ndarray]:
    src = _load_first_chain_ca_coords(monomer_path)
    dst = _load_first_chain_ca_coords(docked_path)
    n = min(src.shape[0], dst.shape[0])
    if n < 3:
        raise RpxDockOutputError(
            f"Need >=3 CA points to estimate rigid transform, got {n}."
        )
    rotation, translation = kabsch_transform(src[:n], dst[:n])
    return rotation, translation


def run_local_cyclic_dock(
    monomer_path: str | Path,
    config: RpxDockRunConfig,
) -> RpxDockRunResult:
    """
    Run local cyclic redocking via vendored rpxdock and return parsed result.

    Raises:
        RpxDockDependencyError: vendored dependency missing.
        RpxDockExecutionError: command failed or timed out.
        RpxDockOutputError: no usable docked output produced.
    """

    monomer = Path(monomer_path).expanduser().resolve()
    if not monomer.exists():
        raise FileNotFoundError(f"Monomer structure not found: {monomer}")

    vendored = resolve_vendored_rpxdock(config.vendor_repo)
    vendored_willutil = (
        Path(config.willutil_repo).expanduser().resolve()
        if config.willutil_repo is not None
        else None
    )
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hscore_data_dir = config.hscore_data_dir
    hscore_files = config.hscore_files
    if hscore_data_dir is None:
        candidate = vendored / "rpxdock" / "data" / "hscore"
        if candidate.exists():
            hscore_data_dir = candidate
    if not hscore_files:
        candidate = Path(hscore_data_dir) / "small_ilv_h" if hscore_data_dir is not None else None
        if candidate is not None and candidate.exists():
            hscore_files = ("small_ilv_h",)
    config = RpxDockRunConfig(
        nfold=config.nfold,
        output_dir=config.output_dir,
        python_bin=config.python_bin,
        vendor_repo=config.vendor_repo,
        willutil_repo=config.willutil_repo,
        docking_method=config.docking_method,
        extra_args=config.extra_args,
        dump_pdbs=config.dump_pdbs,
        timeout_s=config.timeout_s,
        env=config.env,
        architecture=config.architecture,
        stdout_path=config.stdout_path,
        stderr_path=config.stderr_path,
        cart_bounds=config.cart_bounds,
        max_longaxis_dot_z=config.max_longaxis_dot_z,
        components_aligned_to_sym_axes=config.components_aligned_to_sym_axes,
        limit_rotation_to_z=config.limit_rotation_to_z,
        hscore_data_dir=hscore_data_dir,
        hscore_files=tuple(hscore_files),
    )

    command = build_cyclic_dock_command(monomer, config)
    compat_dir = _write_rpxdock_compat_sitecustomize(output_dir)
    env = _build_run_env(
        vendored,
        vendored_willutil,
        config.env,
        prepend_python_paths=(compat_dir,),
    )

    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(output_dir),
            env=env,
            timeout=int(config.timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_file, stderr_file = _write_command_logs(
            output_dir=output_dir,
            command=command,
            completed=None,
            stdout_path=config.stdout_path,
            stderr_path=config.stderr_path,
        )
        raise RpxDockExecutionError(
            f"rpxdock timed out after {config.timeout_s}s. "
            f"stdout={stdout_file} stderr={stderr_file}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        completed = subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=exc.returncode,
            stdout=exc.stdout,
            stderr=exc.stderr,
        )
        stdout_file, stderr_file = _write_command_logs(
            output_dir=output_dir,
            command=command,
            completed=completed,
            stdout_path=config.stdout_path,
            stderr_path=config.stderr_path,
        )
        raise RpxDockExecutionError(
            f"rpxdock command failed with exit code {exc.returncode}. "
            f"stdout={stdout_file} stderr={stderr_file}"
        ) from exc

    stdout_file, stderr_file = _write_command_logs(
        output_dir=output_dir,
        command=command,
        completed=completed,
        stdout_path=config.stdout_path,
        stderr_path=config.stderr_path,
    )

    result_pickle = _find_result_pickle(
        output_dir,
        config.architecture or f"C{int(config.nfold)}",
    )
    if result_pickle is not None:
        candidates = _collect_docked_candidates(
            output_dir,
            monomer,
            result_pickle,
            python_bin=config.python_bin,
            vendored_rpxdock=vendored,
            vendored_willutil=vendored_willutil,
            env=config.env,
        )
        primary = candidates[0]
        rotation, translation = primary.rotation, primary.translation
        docked = primary.docked_structure_path
    else:
        docked = _pick_docked_structure(output_dir, monomer)
        rotation, translation = _estimate_rigid_transform(monomer, docked)
        candidates = (
            RpxDockCandidate(
                rank=0,
                docked_structure_path=docked,
                rotation=rotation,
                translation=translation,
            ),
        )
    return RpxDockRunResult(
        docked_structure_path=docked,
        rotation=rotation,
        translation=translation,
        command=tuple(command),
        stdout_path=stdout_file,
        stderr_path=stderr_file,
        candidates=tuple(candidates),
    )
