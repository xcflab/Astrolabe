from __future__ import annotations

import json
import os
import re
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


class SurfProRunner:
    """Persistent JSONL adapter for the vendored SurfPro model."""

    def __init__(self, config: Mapping[str, Any] | Any):
        cfg = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else dict(config or {})
        )
        if not _as_bool(cfg.get("daemon", True)):
            raise ValueError(
                "SurfPro only supports persistent daemon execution; "
                "per-call model loading is disabled."
            )

        self.python_bin = self._required_executable(
            cfg.get("python_bin"), "python_bin"
        )
        self.repo_path = self._required_dir(cfg.get("repo_path"), "repo_path")
        model_source = self.repo_path / "fairseq" / "models" / "surface_protein_model.py"
        if not model_source.is_file():
            raise FileNotFoundError(
                f"SurfPro model source does not exist: {model_source}"
            )
        checkpoint_value = cfg.get("checkpoint") or "checkpoint_best.pt"
        checkpoint_path = Path(str(checkpoint_value)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_dir = self._required_dir(
                cfg.get("checkpoint_dir"), "checkpoint_dir"
            )
            checkpoint_path = checkpoint_dir / checkpoint_path
        self.checkpoint = self._required_file(checkpoint_path, "checkpoint")
        self.msms_bin = self._required_executable(cfg.get("msms_bin"), "msms_bin")
        self.pdb_to_xyzrn_bin = self._required_executable(
            cfg.get("pdb_to_xyzrn_bin"), "pdb_to_xyzrn_bin"
        )

        self.device = str(cfg.get("device") or "cuda")
        if re.fullmatch(r"cuda(?::\d+)?", self.device) is None:
            raise ValueError(
                "SurfPro upstream hardcodes CUDA in its SRU++ surface encoder; "
                "device must be cuda or cuda:<index>."
            )
        self.logit_temperature = float(cfg.get("logit_temperature", 1.0))
        if self.logit_temperature <= 0:
            raise ValueError("SurfPro logit_temperature must be greater than zero.")
        self.seed = int(cfg.get("seed", 0))
        self.max_surface_vertices = int(cfg.get("max_surface_vertices", 5000))
        if self.max_surface_vertices < 2:
            raise ValueError("SurfPro max_surface_vertices must be at least 2.")

        output_dir = cfg.get("output_dir") or "surfpro_outputs"
        self.output_dir = self._resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = float(cfg.get("startup_timeout", 600.0))
        request_timeout = cfg.get("request_timeout")
        self.request_timeout = (
            None
            if request_timeout in (None, "", "None", "null")
            else float(request_timeout)
        )
        daemon_script = cfg.get("daemon_script_path")
        if daemon_script is None:
            daemon_script = Path(__file__).with_name("surfpro_daemon.py")
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
            raise ValueError(f"SurfPro {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"SurfPro {name} does not exist: {path}")
        return path

    @classmethod
    def _required_executable(cls, value: Any, name: str) -> Path:
        path = cls._required_file(value, name)
        if not os.access(path, os.X_OK):
            raise PermissionError(f"SurfPro {name} is not executable: {path}")
        return path

    @classmethod
    def _required_dir(cls, value: Any, name: str) -> Path:
        if value in (None, "", "None", "null", "???"):
            raise ValueError(f"SurfPro {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_dir():
            raise FileNotFoundError(f"SurfPro {name} does not exist: {path}")
        return path

    def _command(self) -> list[str]:
        return [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
            "--repo-path",
            str(self.repo_path),
            "--checkpoint",
            str(self.checkpoint),
            "--msms-bin",
            str(self.msms_bin),
            "--pdb-to-xyzrn-bin",
            str(self.pdb_to_xyzrn_bin),
            "--device",
            self.device,
            "--logit-temperature",
            str(self.logit_temperature),
            "--max-surface-vertices",
            str(self.max_surface_vertices),
            "--work-dir",
            str(self.output_dir),
        ]

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("APPTAINERENV_PYTHONPATH", None)
        isolated_pythonpath = env.get("GH_SURFPRO_PYTHONPATH", "").strip()
        if isolated_pythonpath:
            env["PYTHONPATH"] = isolated_pythonpath
        return env

    def _ensure_daemon(self) -> subprocess.Popen[str]:
        if self._closed:
            raise RuntimeError("SurfProRunner is closed.")
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc
            raise RuntimeError(
                "SurfPro daemon is no longer running; implicit model restart is disabled."
            )
        if self._started:
            raise RuntimeError(
                "SurfPro daemon is no longer running; implicit model restart is disabled."
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
            raise RuntimeError(f"Unexpected SurfPro daemon startup response: {ready}")
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
                raise TimeoutError("Timed out waiting for SurfPro daemon response.")
        finally:
            selector.close()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"SurfPro daemon exited unexpectedly with code {proc.poll()}."
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
        if (
            probabilities.ndim != 3
            or probabilities.shape[0] != 1
            or probabilities.shape[-1] != 20
        ):
            raise ValueError(
                "SurfPro probabilities must have shape [residues, 20] or "
                f"[1, residues, 20], got {probabilities.shape}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("SurfPro probabilities contain non-finite values.")
        if np.any(probabilities < 0):
            raise ValueError("SurfPro probabilities contain negative values.")
        row_sums = probabilities.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                "SurfPro probabilities must sum to 1 over amino acids; "
                f"observed min/max {row_sums.min():.6g}/{row_sums.max():.6g}."
            )
        return probabilities

    def design_sequences(self, **features: Any) -> dict[str, Any]:
        pdb_path = self._required_file(features.get("pdb_path"), "pdb_path")
        if pdb_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError(
                "SurfPro mutator requires evaluator structures in PDB or mmCIF format."
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
            raise RuntimeError(f"Unexpected SurfPro daemon response: {response}")
        return {
            "sampling_probs": self._validate_probabilities(
                response.get("probabilities")
            ),
            "surfpro_sequence": response.get("sequence"),
            "surfpro_daemon_pid": response.get("pid"),
            "surfpro_model_load_count": response.get("model_load_count"),
            "surfpro_request_count": response.get("request_count"),
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
                        "SurfPro daemon exited before shutdown; "
                        f"return code {proc.returncode}."
                    )
                self._write_request(proc, {"type": "shutdown"})
                response = self._read_response(proc, 30.0)
                if response.get("type") != "ok":
                    raise RuntimeError(
                        f"Unexpected SurfPro daemon shutdown response: {response}"
                    )
                return_code = proc.wait(timeout=30.0)
                if return_code != 0:
                    raise RuntimeError(
                        f"SurfPro daemon shutdown returned code {return_code}."
                    )
            self._closed = True

    shutdown = close
