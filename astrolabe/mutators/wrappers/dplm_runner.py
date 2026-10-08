from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import pathlib
from random import randint
import subprocess
import threading
import uuid
import weakref
from typing import Any, Dict, Iterable, Iterator, List, Sequence


def _terminate_process(proc: subprocess.Popen) -> None:
    """Best-effort cleanup for a DPLM daemon without retaining the runner."""
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    try:
        proc.terminate()
    except Exception:
        pass


class DPLMRunner:
    """Thin wrapper that launches the external DPLM interpreter for sequence completion."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.python_bin = self._resolve_path(self._get("python_bin"), must_exist=True)
        self.repo_path = self._resolve_path(self._get("repo_path"), must_exist=True)
        self.model_name = self._get("model_name", "airkingbd/dplm_150m")
        self.device = self._get("device", "auto")
        self.sampling_strategy = self._get("sampling_strategy", "gumbel_argmax")
        self.max_iter = int(self._get("max_iter", 500))
        self.num_samples = int(self._get("num_samples", 1))
        self.temperature = float(self._get("temperature", 1.0))
        self.daemon = bool(self._get("daemon", True))
        self.response_cache = bool(self._get("response_cache", True))
        work_dir = self._get("work_dir")
        if work_dir is None:
            work_dir = pathlib.Path.cwd() / "dplm_cache"
        self.work_dir = self._resolve_path(work_dir, must_exist=False)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        response_cache_dir = self._get("response_cache_dir")
        if response_cache_dir in (None, "", "null", "None"):
            response_cache_dir = pathlib.Path("~/.cache/gh/dplm_runner").expanduser()
        self.response_cache_dir = self._resolve_path(response_cache_dir, must_exist=False)
        self.response_cache_dir.mkdir(parents=True, exist_ok=True)
        triton_cache_dir = self._get("triton_cache_dir")
        if triton_cache_dir is None:
            triton_cache_dir = self.work_dir / "triton_cache"
        self.triton_cache_dir = self._resolve_path(triton_cache_dir, must_exist=False)
        self.triton_cache_dir.mkdir(parents=True, exist_ok=True)
        script_path = self._get("script_path")
        if script_path is None:
            script_path = pathlib.Path(__file__).resolve().parent / "dplm_infer.py"
        self.script_path = self._resolve_path(script_path, must_exist=True)
        self.daemon_script_path = pathlib.Path(__file__).resolve().parent / "dplm_daemon.py"
        self._proc: subprocess.Popen | None = None
        self._proc_finalizer: weakref.finalize | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._response_cache_mem: Dict[str, List[str]] = {}

    def __enter__(self) -> "DPLMRunner":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DPLMRunner is closed.")

    def _get(self, key: str, default=None):
        if hasattr(self.args, key):
            return getattr(self.args, key)
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return default

    def _resolve_path(self, value, must_exist: bool) -> pathlib.Path:
        if value is None:
            raise ValueError("Required DPLM runner argument is missing.")
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path.cwd() / path).resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Required path '{path}' does not exist.")
        return path

    def _build_payload(self, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "repo_path": str(self.repo_path),
            "model_name": self.model_name,
            "device": self.device,
            "sampling_strategy": self.sampling_strategy,
            "max_iter": self.max_iter,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "seed": randint(0, 1_000_000),
            "tasks": list(tasks),
        }

    def _write_json(self, path: pathlib.Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def _read_results(self, path: pathlib.Path) -> Dict[str, List[str]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = {}
        for item in payload.get("results", []):
            name = item.get("name")
            if name is None:
                continue
            results[name] = item.get("samples", [])
        return results

    def _cache_key(self, task: Dict[str, Any]) -> str:
        payload = {
            "version": 1,
            "name": str(task.get("name") or ""),
            "masked_sequence": str(task.get("masked_sequence") or ""),
            "model_name": str(self.model_name),
            "device": str(self.device),
            "sampling_strategy": str(self.sampling_strategy),
            "max_iter": int(self.max_iter),
            "temperature": float(self.temperature),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> pathlib.Path:
        return self.response_cache_dir / f"{key}.json"

    def _cache_lock_path(self, key: str) -> pathlib.Path:
        return self.response_cache_dir / f"{key}.lock"

    @contextmanager
    def _locked_cache(self, key: str) -> Iterator[None]:
        path = self._cache_lock_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _normalize_samples(samples: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for sample in samples:
            if not isinstance(sample, str):
                continue
            seq = "".join(sample.split()).upper()
            if seq:
                out.append(seq)
        return out

    def _load_cached_samples_unlocked(self, key: str) -> List[str]:
        path = self._cache_path(key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return []
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []
        samples_raw = payload.get("samples")
        if not isinstance(samples_raw, list):
            return []
        samples = self._normalize_samples(samples_raw)
        self._response_cache_mem[key] = list(samples)
        return list(samples)

    def _store_cached_samples_unlocked(self, key: str, samples: Sequence[str]) -> None:
        normalized = self._normalize_samples(samples)
        self._response_cache_mem[key] = list(normalized)

        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "samples": normalized}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(tmp_path), str(path))

    def _load_cached_samples(self, key: str) -> List[str]:
        with self._locked_cache(key):
            return self._load_cached_samples_unlocked(key)

    def _store_cached_samples(self, key: str, samples: Sequence[str]) -> None:
        with self._locked_cache(key):
            self._store_cached_samples_unlocked(key, samples)

    def _consume_cached_samples(self, key: str, request_num: int) -> List[str]:
        request_num = max(1, int(request_num))
        with self._locked_cache(key):
            cached = self._load_cached_samples_unlocked(key)
            consumed = cached[:request_num]
            if consumed:
                self._store_cached_samples_unlocked(key, cached[request_num:])
            return list(consumed)

    def _append_cached_samples(self, key: str, samples: Sequence[str]) -> None:
        normalized = self._normalize_samples(samples)
        if not normalized:
            return
        with self._locked_cache(key):
            cached = self._load_cached_samples_unlocked(key)
            self._store_cached_samples_unlocked(key, [*cached, *normalized])

    def _build_host_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # DPLM runs in a host-side Python, while the parent process may be a
        # container wrapper with py312-only overlays for AF3/Protenix.
        env.pop("PYTHONPATH", None)
        env.pop("APPTAINERENV_PYTHONPATH", None)
        env.pop("TRITON_CACHE_DIR", None)
        env["TRITON_CACHE_DIR"] = str(self.triton_cache_dir)
        return env

    def _ensure_daemon(self) -> subprocess.Popen:
        self._ensure_open()
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc_finalizer is not None:
            self._proc_finalizer.detach()
            self._proc_finalizer = None

        cmd = [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
            "--model-name",
            str(self.model_name),
            "--device",
            str(self.device),
            "--sampling-strategy",
            str(self.sampling_strategy),
            "--max-iter",
            str(self.max_iter),
            "--num-samples",
            str(self.num_samples),
            "--temperature",
            str(self.temperature),
        ]

        env = self._build_host_subprocess_env()

        proc = subprocess.Popen(
            args=cmd,
            cwd=str(self.repo_path),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        self._proc_finalizer = weakref.finalize(self, _terminate_process, proc)
        return proc

    def _daemon_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._ensure_daemon()
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("DPLM daemon exited unexpectedly.")
        return json.loads(line)

    def _daemon_infer(self, tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
        request_id = uuid.uuid4().hex
        req = {
            "type": "infer",
            "id": request_id,
            "seed": randint(0, 1_000_000),
            "tasks": list(tasks),
            "max_iter": self.max_iter,
            "sampling_strategy": self.sampling_strategy,
            "temperature": self.temperature,
        }
        resp = self._daemon_request(req)
        if resp.get("type") != "result" or resp.get("id") != request_id:
            raise RuntimeError(f"Unexpected DPLM daemon response: {resp}")
        results = {}
        for item in resp.get("results", []) or []:
            name = item.get("name")
            if name is None:
                continue
            results[name] = item.get("samples", []) or []
        return results

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            if self._proc_finalizer is not None:
                self._proc_finalizer.detach()
                self._proc_finalizer = None
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
        if self._proc_finalizer is not None:
            self._proc_finalizer.detach()
            self._proc_finalizer = None
        self._proc = None

    def close(self) -> None:
        if self._closed:
            return
        self.shutdown()
        self._closed = True

    def _infer_tasks_no_cache(self, task_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        self._ensure_open()
        if self.daemon:
            try:
                return self._daemon_infer(task_list)
            except Exception:
                # One restart attempt in case of a broken pipe / stale daemon.
                self.shutdown()
                return self._daemon_infer(task_list)

        payload = self._build_payload(task_list)
        session_id = uuid.uuid4().hex
        input_path = self.work_dir / f"dplm_input_{session_id}.json"
        output_path = self.work_dir / f"dplm_output_{session_id}.json"
        self._write_json(input_path, payload)

        cmd = [
            str(self.python_bin),
            str(self.script_path),
            "--config",
            str(input_path),
            "--output",
            str(output_path),
        ]

        env = self._build_host_subprocess_env()

        try:
            subprocess.run(
                args=cmd,
                check=True,
                cwd=str(self.repo_path),
                env=env,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(
                f"DPLM runner failed with exit code {exc.returncode}. Command: {' '.join(cmd)}"
            ) from exc

        try:
            return self._read_results(output_path)
        finally:
            # Cleanup intermediate files to avoid unbounded growth.
            for path in (input_path, output_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def design_sequences(self, tasks: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
        self._ensure_open()
        task_list = list(tasks)
        if not task_list:
            return {}

        with self._lock:
            if not self.response_cache:
                return self._infer_tasks_no_cache(task_list)

            results: Dict[str, List[str]] = {}
            misses: List[Dict[str, Any]] = []
            miss_specs: Dict[str, Dict[str, Any]] = {}

            for task in task_list:
                name = task.get("name")
                if name is None:
                    continue
                request_num = int(task.get("num_samples", self.num_samples) or self.num_samples)
                request_num = max(1, request_num)
                key = self._cache_key(task)
                cached = self._consume_cached_samples(key, request_num)
                if len(cached) >= request_num:
                    results[name] = list(cached)
                    continue

                infer_task = dict(task)
                missing_num = request_num - len(cached)
                infer_num_samples = max(missing_num, int(self.num_samples or 1))
                infer_task["num_samples"] = infer_num_samples
                misses.append(infer_task)
                miss_specs[str(name)] = {
                    "cache_key": key,
                    "cached": cached,
                    "missing_num": missing_num,
                    "request_num": request_num,
                }

            if misses:
                inferred = self._infer_tasks_no_cache(misses)
                for infer_task in misses:
                    name = infer_task.get("name")
                    if name is None:
                        continue
                    spec = miss_specs.get(str(name)) or {}
                    cache_key = str(spec.get("cache_key", ""))
                    cached = list(spec.get("cached", []) or [])
                    missing_num = int(spec.get("missing_num", 1) or 1)
                    request_num = int(spec.get("request_num", 1) or 1)
                    generated = self._normalize_samples(inferred.get(name) or [])
                    if cache_key and len(generated) > missing_num:
                        self._append_cached_samples(cache_key, generated[missing_num:])
                    results[name] = [*cached, *generated[:missing_num]][:request_num]

            return results
