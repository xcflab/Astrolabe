from __future__ import annotations

import atexit
import inspect
import json
import random
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

_REQUEST_ONLY_META_KEYS = frozenset(
    {
        "template_selection",
        "ground_truth_conformer_selection",
        "fixed_ligands",
        "add_missing_atoms",
    }
)


@dataclass(frozen=True, slots=True)
class _RNGStateSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: Any
    cuda_states: tuple[Any, ...] | None


def _capture_rng_state() -> _RNGStateSnapshot:
    import torch

    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return _RNGStateSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state().clone(),
        cuda_states=cuda_states,
    )


def _restore_rng_state(snapshot: _RNGStateSnapshot) -> None:
    import torch

    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_state.clone())
    if snapshot.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.clone() for state in snapshot.cuda_states])


def _normalize_config(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        container = dict(config)
    else:
        container = OmegaConf.to_container(config, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"RF3 config must resolve to a mapping, got {type(container).__name__}")
    container.pop("_type", None)
    return container


def _engine_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    for key in _REQUEST_ONLY_META_KEYS:
        cleaned.pop(key, None)
    if cleaned.get("fallback_conformer_to_input_coords") is None:
        cleaned.pop("fallback_conformer_to_input_coords", None)
    return cleaned


def _config_cache_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _load_rf3_inference_engine(repo_path: str | Path):
    repo_root = Path(repo_path).expanduser().resolve()
    import_roots = (
        repo_root / "src",
        repo_root / "models" / "rf3" / "src",
    )
    for import_root in import_roots:
        import_root_str = str(import_root)
        if import_root.exists() and import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

    from rf3.inference_engines.rf3 import RF3InferenceEngine

    return RF3InferenceEngine


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _resolve_existing_path(path_like: str | Path | None, *, label: str) -> Path | None:
    if path_like in (None, "", "null", "None"):
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"RF3 {label} does not exist: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RF3RunnerRequest:
    request_id: str
    prepared_input_path: Path
    output_dir: Path
    template_selection: list[str]
    ground_truth_conformer_selection: list[str]
    rf3_overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RF3RunnerResponse:
    status: str
    request_id: str
    output_dir: Path
    predicted_structure_path: Path | None
    summary_confidence_path: Path | None
    summary_confidences: dict[str, Any] | None
    confidences_path: Path | None
    ranking_scores_path: Path | None
    error: str | None = None


@dataclass(slots=True)
class _CachedEngine:
    engine: Any
    post_init_rng_state: _RNGStateSnapshot


class RF3RunnerService:
    def __init__(self, base_config: Any):
        self.base_config = _normalize_config(base_config)
        self.python_bin = _resolve_existing_path(
            self.base_config.get("python_bin"),
            label="python_bin",
        )
        self.daemon_script_path = Path(__file__).resolve().parent / "rf3_daemon.py"
        self.repo_root = Path(__file__).resolve().parents[3]
        self._engine_cache: dict[str, _CachedEngine] = {}
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        if self.python_bin is not None:
            atexit.register(self.shutdown)

    def _build_effective_config(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        effective = dict(self.base_config)
        effective.update(dict(overrides or {}))
        return effective

    def _get_engine(self, overrides: dict[str, Any] | None) -> _CachedEngine:
        effective = _engine_config(self._build_effective_config(overrides))
        effective.pop("python_bin", None)
        repo_path = effective.pop("repo_path", None)
        if not repo_path:
            raise ValueError("RF3 config requires `repo_path`.")

        cache_key = _config_cache_key({"repo_path": str(repo_path), **effective})
        cached = self._engine_cache.get(cache_key)
        if cached is not None:
            return cached
        engine_cls = _load_rf3_inference_engine(repo_path)
        engine = engine_cls(**effective)
        initialize = getattr(engine, "initialize", None)
        if callable(initialize):
            initialize()
        cached = _CachedEngine(
            engine=engine,
            post_init_rng_state=_capture_rng_state(),
        )
        self._engine_cache[cache_key] = cached
        return cached

    def _ensure_daemon(self) -> subprocess.Popen:
        if self.python_bin is None:
            raise ValueError("RF3 daemon requires `python_bin`.")
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        cmd = [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
        ]
        proc = subprocess.Popen(
            args=cmd,
            cwd=str(self.repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        return proc

    def _daemon_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._ensure_daemon()
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("RF3 daemon exited unexpectedly.")
        return json.loads(line)

    @staticmethod
    def _run_kwargs(engine: Any, request: RF3RunnerRequest) -> dict[str, Any]:
        kwargs = {
            "inputs": request.prepared_input_path,
            "out_dir": Path(request.output_dir),
            "annotate_b_factor_with_plddt": True,
            "template_selection": list(request.template_selection),
            "ground_truth_conformer_selection": list(
                request.ground_truth_conformer_selection
            ),
        }
        if _accepts_keyword(getattr(engine, "run"), "add_missing_atoms"):
            kwargs["add_missing_atoms"] = bool(
                request.rf3_overrides.get("add_missing_atoms", True)
            )
        return kwargs

    @staticmethod
    def _build_response(request: RF3RunnerRequest) -> RF3RunnerResponse:
        example_id = json.loads(request.prepared_input_path.read_text(encoding="utf-8"))["name"]
        output_dir = Path(request.output_dir)
        candidate_dirs = [output_dir, output_dir / example_id]

        def _first_existing(name: str) -> Path | None:
            for candidate_dir in candidate_dirs:
                candidate = candidate_dir / name
                if candidate.exists():
                    return candidate
            return None

        structure_path = _first_existing(f"{example_id}_model.cif")
        if structure_path is None:
            structure_path = _first_existing(f"{example_id}_model.pdb")

        summary_path = _first_existing(f"{example_id}_summary_confidences.json")
        confidences_path = _first_existing(f"{example_id}_confidences.json")
        ranking_scores_path = _first_existing(f"{example_id}_ranking_scores.csv")
        summary_confidences = None
        if summary_path is not None:
            summary_confidences = json.loads(summary_path.read_text(encoding="utf-8"))

        return RF3RunnerResponse(
            status="success",
            request_id=request.request_id,
            output_dir=output_dir,
            predicted_structure_path=structure_path,
            summary_confidence_path=summary_path,
            summary_confidences=summary_confidences,
            confidences_path=confidences_path,
            ranking_scores_path=ranking_scores_path,
            error=None,
        )

    def _submit_via_daemon(self, request: RF3RunnerRequest) -> RF3RunnerResponse:
        effective_config = _engine_config(self._build_effective_config(request.rf3_overrides))
        response = self._daemon_request(
            {
                "type": "infer",
                "id": request.request_id,
                "config": effective_config,
                "prepared_input_path": str(request.prepared_input_path),
                "output_dir": str(request.output_dir),
                "template_selection": list(request.template_selection),
                "ground_truth_conformer_selection": list(
                    request.ground_truth_conformer_selection
                ),
                "add_missing_atoms": bool(request.rf3_overrides.get("add_missing_atoms", True)),
            }
        )
        if response.get("type") != "result" or response.get("id") != request.request_id:
            raise RuntimeError(f"Unexpected RF3 daemon response: {response}")
        if response.get("status") != "success":
            error = str(response.get("error") or "unknown error")
            return RF3RunnerResponse(
                status="failed",
                request_id=request.request_id,
                output_dir=Path(request.output_dir),
                predicted_structure_path=None,
                summary_confidence_path=None,
                summary_confidences=None,
                confidences_path=None,
                ranking_scores_path=None,
                error=error,
            )
        return self._build_response(request)

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            return
        try:
            self._daemon_request({"type": "shutdown"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        self._proc = None

    def submit(self, request: RF3RunnerRequest) -> RF3RunnerResponse:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.python_bin is not None:
                with self._lock:
                    return self._submit_via_daemon(request)

            with self._lock:
                cached = self._get_engine(request.rf3_overrides)
                _restore_rng_state(cached.post_init_rng_state)
                cached.engine.run(**self._run_kwargs(cached.engine, request))
            return self._build_response(request)
        except Exception as exc:
            return RF3RunnerResponse(
                status="failed",
                request_id=request.request_id,
                output_dir=output_dir,
                predicted_structure_path=None,
                summary_confidence_path=None,
                summary_confidences=None,
                confidences_path=None,
                ranking_scores_path=None,
                error=f"{type(exc).__name__}: {exc}",
            )


_RUNNER_CACHE: dict[str, RF3RunnerService] = {}


def rf3_setup(config: Any) -> RF3RunnerService:
    normalized = _normalize_config(config)
    cache_key = _config_cache_key(normalized)
    runner = _RUNNER_CACHE.get(cache_key)
    if runner is not None:
        return runner
    runner = RF3RunnerService(normalized)
    _RUNNER_CACHE[cache_key] = runner
    return runner
