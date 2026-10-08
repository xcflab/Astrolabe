from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


class SurfDesignRunner:
    """Persistent JSONL adapter for the vendored SurfDesign model."""

    def __init__(self, config: Mapping[str, Any] | Any):
        cfg = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else dict(config or {})
        )
        if not _as_bool(cfg.get("daemon", True)):
            raise ValueError(
                "SurfDesign only supports persistent daemon execution; "
                "per-call model loading is disabled."
            )

        self.python_bin = self._required_file(cfg.get("python_bin"), "python_bin")
        self.repo_path = self._required_dir(cfg.get("repo_path"), "repo_path")
        self.experiment_path = self._required_dir(
            cfg.get("experiment_path"), "experiment_path"
        )
        self.checkpoint = str(cfg.get("checkpoint") or "best.ckpt")
        checkpoint_path = Path(self.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.experiment_path / "checkpoints" / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SurfDesign checkpoint does not exist: {checkpoint_path}"
            )
        hydra_config = self.experiment_path / ".hydra" / "config.yaml"
        if not hydra_config.is_file():
            raise FileNotFoundError(
                f"SurfDesign experiment config does not exist: {hydra_config}"
            )

        self.device = str(cfg.get("device") or "cuda")
        self.max_iter = int(cfg.get("max_iter", 5))
        self.strategy = str(cfg.get("strategy") or "denoise")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.seed = int(cfg.get("seed", 0))
        self.logit_temperature = float(cfg.get("logit_temperature", 1.0))
        if self.logit_temperature <= 0:
            raise ValueError("SurfDesign logit_temperature must be greater than zero.")
        self.surface_mode = str(cfg.get("surface_mode") or "pymol").lower()
        if self.surface_mode != "pymol":
            raise ValueError("SurfDesign runtime inference currently requires surface_mode=pymol.")

        output_dir = cfg.get("output_dir") or "surfdesign_outputs"
        self.output_dir = self._resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = float(cfg.get("startup_timeout", 600.0))
        request_timeout = cfg.get("request_timeout")
        self.request_timeout = (
            None if request_timeout in (None, "", "None", "null") else float(request_timeout)
        )
        daemon_script = cfg.get("daemon_script_path")
        if daemon_script is None:
            daemon_script = Path(__file__).with_name("surfdesign_daemon.py")
        self.daemon_script_path = self._required_file(
            daemon_script, "daemon_script_path"
        )

        self._proc: subprocess.Popen[str] | None = None
        self._started = False
        self._closed = False
        self._request_index = 0
        self._lock = threading.RLock()

    @staticmethod
    def _resolve_path(value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @classmethod
    def _required_file(cls, value: Any, name: str) -> Path:
        if value in (None, "", "None", "null", "???"):
            raise ValueError(f"SurfDesign {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"SurfDesign {name} does not exist: {path}")
        return path

    @classmethod
    def _required_dir(cls, value: Any, name: str) -> Path:
        if value in (None, "", "None", "null", "???"):
            raise ValueError(f"SurfDesign {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_dir():
            raise FileNotFoundError(f"SurfDesign {name} does not exist: {path}")
        return path

    def _command(self) -> list[str]:
        return [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
            "--repo-path",
            str(self.repo_path),
            "--experiment-path",
            str(self.experiment_path),
            "--checkpoint",
            self.checkpoint,
            "--device",
            self.device,
            "--max-iter",
            str(self.max_iter),
            "--strategy",
            self.strategy,
            "--temperature",
            str(self.temperature),
            "--logit-temperature",
            str(self.logit_temperature),
            "--surface-mode",
            self.surface_mode,
            "--work-dir",
            str(self.output_dir),
        ]

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("APPTAINERENV_PYTHONPATH", None)
        return env

    def _ensure_daemon(self) -> subprocess.Popen[str]:
        if self._closed:
            raise RuntimeError("SurfDesignRunner is closed.")
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc
            raise RuntimeError(
                "SurfDesign daemon is no longer running; implicit model restart is disabled."
            )
        if self._started:
            raise RuntimeError(
                "SurfDesign daemon is no longer running; implicit model restart is disabled."
            )

        proc = subprocess.Popen(
            self._command(),
            cwd=str(self.repo_path),
            env=self._subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        self._started = True
        try:
            ready = self._read_response(proc, self.startup_timeout)
        except Exception:
            proc.terminate()
            proc.wait()
            raise
        if ready.get("type") != "ready" or ready.get("model_load_count") != 1:
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"Unexpected SurfDesign daemon startup response: {ready}")
        return proc

    @staticmethod
    def _read_response(
        proc: subprocess.Popen[str], timeout: float | None
    ) -> dict[str, Any]:
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout):
                raise TimeoutError("Timed out waiting for SurfDesign daemon response.")
        finally:
            selector.close()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"SurfDesign daemon exited unexpectedly with code {proc.poll()}."
            )
        return json.loads(line)

    @staticmethod
    def _write_request(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _validate_probabilities(value: Any) -> np.ndarray:
        probabilities = np.asarray(value, dtype=float)
        if probabilities.ndim == 2:
            probabilities = probabilities[None, ...]
        if probabilities.ndim != 3 or probabilities.shape[0] != 1 or probabilities.shape[-1] != 20:
            raise ValueError(
                "SurfDesign probabilities must have shape [residues, 20] or "
                f"[1, residues, 20], got {probabilities.shape}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("SurfDesign probabilities contain non-finite values.")
        if np.any(probabilities < 0):
            raise ValueError("SurfDesign probabilities contain negative values.")
        row_sums = probabilities.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                "SurfDesign probabilities must sum to 1 over amino acids; "
                f"observed min/max {row_sums.min():.6g}/{row_sums.max():.6g}."
            )
        return probabilities

    def design_sequences(self, **features: Any) -> dict[str, Any]:
        pdb_path = self._required_file(features.get("pdb_path"), "pdb_path")
        if pdb_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError(
                "SurfDesign mutator requires evaluator structures in PDB or mmCIF format."
            )
        request_id = uuid.uuid4().hex
        with self._lock:
            payload = {
                "type": "infer",
                "id": request_id,
                "pdb_path": str(pdb_path),
                "fixed_residues": str(features.get("fixed_residues") or ""),
                "seed": self.seed + self._request_index,
            }
            self._request_index += 1
            proc = self._ensure_daemon()
            self._write_request(proc, payload)
            response = self._read_response(proc, self.request_timeout)
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError(f"Unexpected SurfDesign daemon response: {response}")
        return {
            "sampling_probs": self._validate_probabilities(response.get("probabilities")),
            "surfdesign_sequence": response.get("sequence"),
            "surfdesign_daemon_pid": response.get("pid"),
            "surfdesign_model_load_count": response.get("model_load_count"),
            "surfdesign_request_count": response.get("request_count"),
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            proc = self._proc
            if proc is not None:
                if proc.poll() is not None:
                    self._closed = True
                    raise RuntimeError(
                        "SurfDesign daemon exited before shutdown; "
                        f"return code {proc.returncode}."
                    )
                self._write_request(proc, {"type": "shutdown"})
                response = self._read_response(proc, 30.0)
                if response.get("type") != "ok":
                    raise RuntimeError(
                        f"Unexpected SurfDesign daemon shutdown response: {response}"
                    )
                return_code = proc.wait(timeout=30.0)
                if return_code != 0:
                    raise RuntimeError(
                        f"SurfDesign daemon shutdown returned code {return_code}."
                    )
            self._closed = True

    shutdown = close
