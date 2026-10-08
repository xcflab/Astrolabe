from __future__ import annotations

import atexit
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import uuid
from typing import Any, Dict, List


_CACHE_PAYLOAD_VERSION = 2


def default_rapidock_daemon_script_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "rapidock_daemon.py"


class RAPiDockRunner:
    """Thin subprocess wrapper around a long-lived RAPiDock daemon."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.python_bin = self._resolve_path(self._get("python_bin"), must_exist=True)
        self.repo_path = self._resolve_path(self._get("repo_path"), must_exist=True)
        self.model_dir = self._resolve_path(self._get("model_dir"), must_exist=True)
        self.ckpt = str(self._get("ckpt", "rapidock_global.pt"))
        ckpt_path = self.model_dir / self.ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Required RAPiDock checkpoint '{ckpt_path}' does not exist.")

        self.confidence_model_dir = self._get("confidence_model_dir")
        self.confidence_ckpt = str(self._get("confidence_ckpt", "best_model.pt"))
        self.device = self._get("device")
        self.daemon = bool(self._get("daemon", True))
        self.response_cache = bool(self._get("response_cache", True))
        self.num_samples_bank = int(self._get("num_samples_bank", self._get("num_samples", 5)))
        self.batch_size = int(self._get("batch_size", min(max(self.num_samples_bank, 1), 10)))
        self.inference_steps = int(self._get("inference_steps", 20))
        self.actual_steps = self._get("actual_steps")
        self.no_final_step_noise = bool(self._get("no_final_step_noise", False))
        self.scoring_function = self._get("scoring_function")
        self.fastrelax = bool(self._get("fastrelax", False))
        self.save_visualisation = bool(self._get("save_visualisation", False))
        self.cpu = int(self._get("cpu", os.cpu_count() or 1))
        self.conformation_type = str(self._get("conformation_type", "E"))
        self.conformation_partial = self._get("conformation_partial")

        response_cache_dir = self._get("response_cache_dir")
        if response_cache_dir in (None, "", "null", "None"):
            response_cache_dir = pathlib.Path("~/.cache/gh/rapidock_runner").expanduser()
        self.response_cache_dir = self._resolve_path(response_cache_dir, must_exist=False)
        self.response_cache_dir.mkdir(parents=True, exist_ok=True)

        work_dir = self._get("work_dir")
        if work_dir in (None, "", "null", "None"):
            work_dir = self.response_cache_dir / "banks"
        self.work_dir = self._resolve_path(work_dir, must_exist=False)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        daemon_script_path = self._get("daemon_script_path")
        if daemon_script_path in (None, "", "null", "None"):
            daemon_script_path = default_rapidock_daemon_script_path()
        self.daemon_script_path = self._resolve_path(daemon_script_path, must_exist=True)

        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._response_cache_mem: Dict[str, List[str]] = {}
        self._cursor_mem: Dict[str, int] = {}
        atexit.register(self.shutdown)

    def _get(self, key: str, default=None):
        if hasattr(self.args, key):
            return getattr(self.args, key)
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return default

    def _resolve_path(self, value, *, must_exist: bool) -> pathlib.Path:
        if value is None:
            raise ValueError("Required RAPiDock runner argument is missing.")
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path.cwd() / path).resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Required path '{path}' does not exist.")
        return path

    @staticmethod
    def _sha256_file(path: pathlib.Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _optional_file_sha256(self, path: pathlib.Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        return self._sha256_file(path)

    def _dock_params(self) -> Dict[str, Any]:
        params = {
            "num_samples_bank": int(self.num_samples_bank),
            "batch_size": int(self.batch_size),
            "inference_steps": int(self.inference_steps),
            "actual_steps": (
                None if self.actual_steps in (None, "", "null", "None") else int(self.actual_steps)
            ),
            "no_final_step_noise": bool(self.no_final_step_noise),
            "scoring_function": self.scoring_function,
            "fastrelax": bool(self.fastrelax),
            "save_visualisation": bool(self.save_visualisation),
            "cpu": int(self.cpu),
            "conformation_type": str(self.conformation_type),
            "conformation_partial": self.conformation_partial,
        }
        return params

    def _cache_key(self, *, protein_pdb: str, peptide_pdb: str) -> str:
        protein_path = pathlib.Path(protein_pdb).resolve()
        peptide_path = pathlib.Path(peptide_pdb).resolve()
        payload = {
            "version": _CACHE_PAYLOAD_VERSION,
            "protein_sha256": self._sha256_file(protein_path),
            "peptide_sha256": self._sha256_file(peptide_path),
            "repo_path": str(self.repo_path),
            "inference_sha256": self._optional_file_sha256(self.repo_path / "inference.py"),
            "daemon_script_path": str(self.daemon_script_path),
            "daemon_script_sha256": self._optional_file_sha256(self.daemon_script_path),
            "model_dir": str(self.model_dir),
            "ckpt": self.ckpt,
            "device": None if self.device in (None, "", "null", "None") else str(self.device),
            "confidence_model_dir": (
                None
                if self.confidence_model_dir in (None, "", "null", "None")
                else str(self.confidence_model_dir)
            ),
            "confidence_ckpt": self.confidence_ckpt,
            "params": self._dock_params(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> pathlib.Path:
        return self.response_cache_dir / f"{key}.json"

    @staticmethod
    def _existing_candidates(candidates: List[str]) -> List[str]:
        return [str(item) for item in candidates if pathlib.Path(item).exists()]

    def _load_cached_candidates(self, key: str) -> List[str]:
        if key in self._response_cache_mem:
            candidates = self._existing_candidates(self._response_cache_mem[key])
            if len(candidates) != len(self._response_cache_mem[key]):
                self._store_cached_candidates(key, candidates)
            return list(candidates)

        path = self._cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception:
            return []
        if payload.get("version") != _CACHE_PAYLOAD_VERSION:
            return []

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            return []
        candidates = self._existing_candidates(
            [str(item) for item in raw_candidates if isinstance(item, str)]
        )
        self._response_cache_mem[key] = list(candidates)
        return list(candidates)

    def _store_cached_candidates(self, key: str, candidates: List[str]) -> None:
        normalized = [str(path) for path in candidates if pathlib.Path(path).exists()]
        self._response_cache_mem[key] = list(normalized)
        payload = {"version": _CACHE_PAYLOAD_VERSION, "candidates": normalized}
        path = self._cache_path(key)
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    def _ensure_daemon(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        cmd = [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
            "--repo-path",
            str(self.repo_path),
            "--model-dir",
            str(self.model_dir),
            "--ckpt",
            str(self.ckpt),
        ]
        if self.confidence_model_dir not in (None, "", "null", "None"):
            cmd.extend(["--confidence-model-dir", str(self.confidence_model_dir)])
            cmd.extend(["--confidence-ckpt", str(self.confidence_ckpt)])
        if self.device not in (None, "", "null", "None"):
            cmd.extend(["--device", str(self.device)])

        proc = subprocess.Popen(
            args=cmd,
            cwd=str(self.repo_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        return proc

    def _daemon_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._ensure_daemon()
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("RAPiDock daemon exited unexpectedly.")
        response = json.loads(line)
        if response.get("type") == "error":
            error_type = response.get("error_type") or "RuntimeError"
            message = response.get("message") or "RAPiDock daemon returned an error."
            raise RuntimeError(f"{error_type}: {message}")
        return response

    def _request_bank(self, *, name: str, protein_pdb: str, peptide_pdb: str) -> List[str]:
        request_id = uuid.uuid4().hex
        output_dir = self.work_dir / request_id
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "dock",
            "id": request_id,
            "tasks": [
                {
                    "name": str(name),
                    "protein_pdb": str(protein_pdb),
                    "peptide_pdb": str(peptide_pdb),
                    "output_dir": str(output_dir),
                }
            ],
            "params": self._dock_params(),
        }
        response = self._daemon_request(payload)
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError(f"Unexpected RAPiDock daemon response: {response}")

        results = response.get("results") or []
        if not results:
            raise RuntimeError("RAPiDock daemon returned no results.")
        candidates = [
            str(item.get("peptide_pdb"))
            for item in (results[0].get("candidates") or [])
            if item.get("peptide_pdb")
        ]
        if not candidates:
            raise RuntimeError("RAPiDock daemon returned no candidate pdb paths.")
        return candidates

    def ping(self) -> str:
        with self._lock:
            request_id = uuid.uuid4().hex
            response = self._daemon_request({"type": "ping", "id": request_id})
            if response.get("type") != "pong" or response.get("id") != request_id:
                raise RuntimeError(f"Unexpected RAPiDock ping response: {response}")
            return "pong"

    def _request_bank_with_retry(
        self, *, name: str, protein_pdb: str, peptide_pdb: str
    ) -> List[str]:
        try:
            return self._request_bank(
                name=name,
                protein_pdb=protein_pdb,
                peptide_pdb=peptide_pdb,
            )
        except Exception:
            self.shutdown()
            return self._request_bank(
                name=name,
                protein_pdb=protein_pdb,
                peptide_pdb=peptide_pdb,
            )

    def _load_or_request_bank(
        self,
        *,
        key: str,
        name: str,
        protein_pdb: str,
        peptide_pdb: str,
        force_refresh: bool,
    ) -> List[str]:
        if self.response_cache and not force_refresh:
            candidates = self._load_cached_candidates(key)
            if candidates:
                return candidates

        candidates = self._request_bank_with_retry(
            name=name,
            protein_pdb=protein_pdb,
            peptide_pdb=peptide_pdb,
        )
        if self.response_cache:
            self._store_cached_candidates(key, candidates)
            self._cursor_mem[key] = 0
        return candidates

    def dock_candidates(self, *, name: str, protein_pdb: str, peptide_pdb: str) -> List[str]:
        protein_path = str(self._resolve_path(protein_pdb, must_exist=True))
        peptide_path = str(self._resolve_path(peptide_pdb, must_exist=True))

        with self._lock:
            key = self._cache_key(protein_pdb=protein_path, peptide_pdb=peptide_path)
            candidates = self._load_or_request_bank(
                key=key,
                name=name,
                protein_pdb=protein_path,
                peptide_pdb=peptide_path,
                force_refresh=False,
            )

            if not candidates:
                raise RuntimeError("RAPiDock candidate bank is empty.")

            return list(candidates)

    def dock_candidate(self, *, name: str, protein_pdb: str, peptide_pdb: str) -> str:
        protein_path = str(self._resolve_path(protein_pdb, must_exist=True))
        peptide_path = str(self._resolve_path(peptide_pdb, must_exist=True))

        if not self.response_cache:
            return self._request_bank_with_retry(
                name=name,
                protein_pdb=protein_path,
                peptide_pdb=peptide_path,
            )[0]

        with self._lock:
            key = self._cache_key(protein_pdb=protein_path, peptide_pdb=peptide_path)
            candidates = self._load_or_request_bank(
                key=key,
                name=name,
                protein_pdb=protein_path,
                peptide_pdb=peptide_path,
                force_refresh=False,
            )
            cursor = self._cursor_mem.get(key, 0)
            if cursor >= len(candidates):
                candidates = self._load_or_request_bank(
                    key=key,
                    name=name,
                    protein_pdb=protein_path,
                    peptide_pdb=peptide_path,
                    force_refresh=True,
                )
                cursor = 0

            existing_candidates = self._existing_candidates(candidates)
            if len(existing_candidates) != len(candidates):
                candidates = existing_candidates
                self._store_cached_candidates(key, candidates)
                if cursor >= len(candidates):
                    cursor = 0

            if not candidates:
                candidates = self._load_or_request_bank(
                    key=key,
                    name=name,
                    protein_pdb=protein_path,
                    peptide_pdb=peptide_path,
                    force_refresh=True,
                )
                cursor = 0
            if not candidates:
                raise RuntimeError("RAPiDock candidate bank is empty.")

            candidate = candidates[cursor]
            self._cursor_mem[key] = cursor + 1
            return candidate

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            return
        try:
            with self._lock:
                self._daemon_request({"type": "shutdown"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        self._proc = None
