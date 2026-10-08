from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_with_optional_src(path: str | Path | None) -> str | None:
    if path is None:
        return None
    root = Path(path).expanduser().resolve()
    src = root / "src"
    if src.exists():
        return str(src)
    return str(root)


def _prepend_path(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _resolve_model_source(model_name: str) -> tuple[str, bool]:
    pinned_snapshot = os.environ.get("GH_ESMFOLD2_MODEL_SNAPSHOT")
    if pinned_snapshot and model_name == "biohub/ESMFold2":
        return pinned_snapshot, True
    return model_name, False


def _resolve_esmc_snapshot_path() -> Path | None:
    esmc_snapshot = os.environ.get("GH_ESMFOLD2_ESMC_SNAPSHOT")
    if not esmc_snapshot:
        return None
    esmc_path = Path(esmc_snapshot).expanduser().resolve()
    for required in ("config.json", "model.safetensors.index.json"):
        if not (esmc_path / required).is_file():
            raise FileNotFoundError(
                f"Pinned ESMFold2 ESMC snapshot is missing {required}: {esmc_path}"
            )
    return esmc_path


def _model_load_kwargs(payload: dict[str, Any], *, pinned_snapshot: bool) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "esmc_precision": payload.get("esmc_precision") or "bf16",
    }
    cache_dir = payload.get("cache_dir")
    if cache_dir:
        model_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    offline = bool(payload.get("local_files_only", False)) or pinned_snapshot
    if offline:
        model_kwargs["local_files_only"] = True
    if _resolve_esmc_snapshot_path() is not None:
        model_kwargs["local_files_only"] = True
    elif offline:
        raise FileNotFoundError(
            "Offline ESMFold2 requires GH_ESMFOLD2_ESMC_SNAPSHOT; the nested "
            "ESMC-6B loader does not inherit local_files_only"
        )
    return model_kwargs


def _load_esmfold2_model(
    model_cls: Any,
    model_name: str,
    model_kwargs: dict[str, Any],
    *,
    esmc_snapshot_path: Path | None,
) -> Any:
    if esmc_snapshot_path is None:
        return model_cls.from_pretrained(model_name, **model_kwargs).eval()

    esmc_precision = model_kwargs.get("esmc_precision") or "bf16"
    model = model_cls.from_pretrained(
        model_name,
        load_esmc=False,
        **model_kwargs,
    )
    model.config.esmc_id = str(esmc_snapshot_path)
    model.load_esmc(str(esmc_snapshot_path), precision=esmc_precision)
    return model.eval()


def _ccd_cache_dir(
    model_source: str,
    *,
    pinned_snapshot: bool,
    local_files_only: bool = False,
) -> Path | None:
    """Return the local model directory that owns the pinned CCD pickle."""
    source = Path(model_source).expanduser()
    if not pinned_snapshot and not source.is_dir():
        if local_files_only:
            explicit_ccd = os.environ.get("ESMCFOLD_CCD_PATH")
            if not explicit_ccd:
                raise FileNotFoundError(
                    "Offline ESMFold2 requires GH_ESMFOLD2_MODEL_SNAPSHOT or "
                    "ESMCFOLD_CCD_PATH"
                )
            ccd = Path(explicit_ccd).expanduser().resolve()
            if not ccd.is_file() or ccd.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Explicit ESMFold2 CCD pickle is missing or empty: {ccd}"
                )
        return None
    source = source.resolve()
    ccd = source / "ccd.pkl"
    if not ccd.is_file() or ccd.stat().st_size == 0:
        raise FileNotFoundError(f"Pinned ESMFold2 CCD pickle is missing or empty: {ccd}")
    return source


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _mean_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _worker_run(payload: dict[str, Any]) -> dict[str, Any]:
    _prepend_path(_path_with_optional_src(payload.get("transformers_repo_path")))
    _prepend_path(_path_with_optional_src(payload.get("esm_repo_path")))

    import torch
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model_name, pinned_snapshot = _resolve_model_source(payload["model_name"])
    device = payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    esmc_snapshot_path = _resolve_esmc_snapshot_path()
    model_kwargs = _model_load_kwargs(payload, pinned_snapshot=pinned_snapshot)

    model = _load_esmfold2_model(
        ESMFold2Model,
        model_name,
        model_kwargs,
        esmc_snapshot_path=esmc_snapshot_path,
    )
    if device != "cpu":
        model = model.to(device)
    if payload.get("chunk_size") is not None and hasattr(model, "set_chunk_size"):
        model.set_chunk_size(payload["chunk_size"])

    output_dir = Path(payload["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = ESMFold2InputBuilder(
        ccd_cache=_ccd_cache_dir(
            model_name,
            pinned_snapshot=pinned_snapshot,
            local_files_only=bool(payload.get("local_files_only", False)),
        )
    )
    results: list[dict[str, Any]] = []

    fold_kwargs = {
        "num_loops": int(payload.get("num_loops", 3)),
        "num_sampling_steps": int(payload.get("num_sampling_steps", 50)),
        "num_diffusion_samples": int(payload.get("num_diffusion_samples", 1)),
        "seed": payload.get("seed"),
        "lm_dropout": payload.get("lm_dropout"),
        "msa_max_depth": payload.get("msa_max_depth"),
        "msa_column_mask_rate": float(payload.get("msa_column_mask_rate", 0.1)),
    }
    for key in ("noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct"):
        if payload.get(key) is not None:
            fold_kwargs[key] = payload[key]

    for job in payload["jobs"]:
        name = str(job["name"])
        chains = [
            ProteinInput(id=str(chain["id"]), sequence=str(chain["sequence"]))
            for chain in job["chains"]
        ]
        prediction_input = StructurePredictionInput(sequences=chains)
        result = builder.fold(
            model,
            prediction_input,
            complex_id=name,
            **fold_kwargs,
        )
        if isinstance(result, list):
            result = result[0]

        structure_path = output_dir / f"{name}_model.cif"
        structure_path.write_text(result.complex.to_mmcif(), encoding="utf-8")
        results.append(
            {
                "name": name,
                "structure_path": str(structure_path),
                "plddt": _to_python(result.plddt),
                "plddt_mean": _mean_float(result.plddt),
                "ptm": _mean_float(result.ptm),
                "iptm": _mean_float(result.iptm),
                "pae": _to_python(getattr(result, "pae", None)),
                "pde": _to_python(getattr(result, "pde", None)),
            }
        )

    return {
        "model_name": model_name,
        "device": device,
        "jobs": results,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Biohub ESMFold2 inference jobs.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = _worker_run(payload)
    Path(args.output_json).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


class ESMFold2Runner:
    def __init__(
        self,
        *,
        python_bin: str | None = None,
        esm_repo_path: str | None = None,
        transformers_repo_path: str | None = None,
        model_name: str = "biohub/ESMFold2",
        cache_dir: str | None = None,
        device: str | None = None,
        esmc_precision: str = "bf16",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        lm_dropout: float | None = 0.0,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        chunk_size: int | None = 64,
        local_files_only: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        self.python_bin = python_bin
        self.esm_repo_path = esm_repo_path
        self.transformers_repo_path = transformers_repo_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.esmc_precision = esmc_precision
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps
        self.num_diffusion_samples = num_diffusion_samples
        self.seed = seed
        self.lm_dropout = lm_dropout
        self.msa_max_depth = msa_max_depth
        self.msa_column_mask_rate = msa_column_mask_rate
        self.chunk_size = chunk_size
        self.local_files_only = local_files_only
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: Any) -> "ESMFold2Runner":
        if hasattr(config, "items"):
            container = dict(config)
        else:
            from omegaconf import OmegaConf

            container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("ESMFold2 config must resolve to a mapping")
        allowed = {
            "python_bin",
            "esm_repo_path",
            "transformers_repo_path",
            "model_name",
            "cache_dir",
            "device",
            "esmc_precision",
            "num_loops",
            "num_sampling_steps",
            "num_diffusion_samples",
            "seed",
            "lm_dropout",
            "msa_max_depth",
            "msa_column_mask_rate",
            "chunk_size",
            "local_files_only",
            "timeout_seconds",
        }
        return cls(**{k: v for k, v in container.items() if k in allowed})

    def run(self, jobs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "esm_repo_path": self.esm_repo_path,
            "transformers_repo_path": self.transformers_repo_path,
            "esmc_precision": self.esmc_precision,
            "num_loops": self.num_loops,
            "num_sampling_steps": self.num_sampling_steps,
            "num_diffusion_samples": self.num_diffusion_samples,
            "seed": self.seed,
            "lm_dropout": self.lm_dropout,
            "msa_max_depth": self.msa_max_depth,
            "msa_column_mask_rate": self.msa_column_mask_rate,
            "chunk_size": self.chunk_size,
            "output_dir": str(output_dir / "structures"),
            "jobs": jobs,
        }
        input_json = output_dir / "esmfold2_input.json"
        output_json = output_dir / "esmfold2_output.json"
        stdout_log = output_dir / "esmfold2.stdout.log"
        stderr_log = output_dir / "esmfold2.stderr.log"
        input_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        python_bin = self.python_bin or sys.executable
        if not shutil.which(python_bin) and not Path(python_bin).exists():
            raise FileNotFoundError(f"ESMFold2 python_bin not found: {python_bin}")
        env = os.environ.copy()
        pythonpath = [str(_repo_root())]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        cmd = [
            python_bin,
            "-m",
            "astrolabe.evaluators.wrappers.esmfold2",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-30:])
            raise RuntimeError(
                f"ESMFold2 subprocess failed with exit code {completed.returncode}. "
                f"See {stderr_log}. Stderr tail:\n{tail}"
            )
        if not output_json.exists():
            raise FileNotFoundError(f"ESMFold2 output JSON was not written: {output_json}")
        return json.loads(output_json.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
