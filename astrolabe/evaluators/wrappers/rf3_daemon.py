#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REQUEST_ONLY_META_KEYS = frozenset(
    {
        "template_selection",
        "ground_truth_conformer_selection",
        "fixed_ligands",
        "add_missing_atoms",
    }
)


def _config_cache_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _load_rf3_inference_engine(repo_path: str | Path):
    script_dir = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != script_dir]
    sys.modules.pop("rf3", None)

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


def _write(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _capture_rng_state():
    import torch

    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return {
        "python_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state().clone(),
        "cuda_states": cuda_states,
    }


def _restore_rng_state(snapshot: dict[str, Any]) -> None:
    import torch

    random.setstate(snapshot["python_state"])
    np.random.set_state(snapshot["numpy_state"])
    torch.set_rng_state(snapshot["torch_state"].clone())
    cuda_states = snapshot.get("cuda_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.clone() for state in cuda_states])


def main() -> int:
    engine_cache: dict[str, dict[str, Any]] = {}

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"type": "error", "message": f"invalid json: {exc}"})
            continue

        msg_type = message.get("type")
        if msg_type == "shutdown":
            _write({"type": "ok", "message": "bye"})
            return 0
        if msg_type != "infer":
            _write({"type": "error", "message": f"unknown type: {msg_type}"})
            continue

        request_id = str(message.get("id") or "")
        config = dict(message.get("config") or {})
        config.pop("python_bin", None)
        for key in _REQUEST_ONLY_META_KEYS:
            config.pop(key, None)
        if config.get("fallback_conformer_to_input_coords") is None:
            config.pop("fallback_conformer_to_input_coords", None)
        repo_path = config.pop("repo_path", None)
        if not repo_path:
            _write(
                {
                    "type": "result",
                    "id": request_id,
                    "status": "failed",
                    "error": "ValueError: RF3 config requires `repo_path`.",
                }
            )
            continue

        try:
            cache_key = _config_cache_key({"repo_path": str(repo_path), **config})
            cached = engine_cache.get(cache_key)
            if cached is None:
                engine_cls = _load_rf3_inference_engine(repo_path)
                engine = engine_cls(**config)
                initialize = getattr(engine, "initialize", None)
                if callable(initialize):
                    initialize()
                cached = {
                    "engine": engine,
                    "post_init_rng_state": _capture_rng_state(),
                }
                engine_cache[cache_key] = cached

            _restore_rng_state(cached["post_init_rng_state"])
            run_kwargs = {
                "inputs": Path(str(message["prepared_input_path"])),
                "out_dir": Path(str(message["output_dir"])),
                "annotate_b_factor_with_plddt": True,
                "template_selection": list(message.get("template_selection") or []),
                "ground_truth_conformer_selection": list(
                    message.get("ground_truth_conformer_selection") or []
                ),
            }
            if _accepts_keyword(cached["engine"].run, "add_missing_atoms"):
                run_kwargs["add_missing_atoms"] = bool(
                    message.get("add_missing_atoms", True)
                )
            cached["engine"].run(**run_kwargs)
            _write({"type": "result", "id": request_id, "status": "success"})
        except Exception as exc:
            _write(
                {
                    "type": "result",
                    "id": request_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
