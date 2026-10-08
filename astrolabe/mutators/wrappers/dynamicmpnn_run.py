from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Mapping

import numpy as np
from omegaconf import OmegaConf


class DynamicMPNNRunner:
    """Subprocess adapter for Alex-Abrudan/DynamicMPNN.

    The real DynamicMPNN project is an external GVP-GNN inverse-folding model.  This
    adapter deliberately does not reuse LigandMPNN.  The preferred path calls a
    configured Python callable in-process and consumes its raw amino-acid
    probabilities directly.  A subprocess command remains available only for
    externally isolated DynamicMPNN installations.
    """

    def __init__(self, config: Mapping[str, Any] | Any):
        cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config or {})
        self.output_dir = Path(str(cfg.get("output_dir") or cfg.get("out_folder") or "dynamicmpnn_outputs"))
        self.backend_callable = cfg.get("backend_callable")
        self.command_template = cfg.get("command_template")
        self.repo_path = cfg.get("repo_path")
        self.checkpoint = cfg.get("checkpoint") or cfg.get("model_ref")
        self.temperature = float(cfg.get("temperature", 0.3))
        self.num_samples = int(cfg.get("num_samples", 1))
        self.probabilities_path = str(cfg.get("probabilities_path", "probabilities.npy"))
        self.probabilities_key = str(cfg.get("probabilities_key", "probabilities"))
        self.timeout = cfg.get("timeout", None)
        self.extra_env = dict(cfg.get("env") or {})
        self._run_index = 0

    def design_sequences(self, **features):
        pdb_paths = self._load_pdb_paths(features)
        if len(pdb_paths) < 2:
            raise ValueError(
                "DynamicMPNN requires multiple conformations for joint inference; "
                "got fewer than two PDB paths."
            )

        run_dir = self.output_dir / f"run_{self._run_index:04d}"
        self._run_index += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        probabilities_file = run_dir / self.probabilities_path
        manifest = self._build_manifest(features, pdb_paths, run_dir, probabilities_file)

        if self.backend_callable:
            callable_fn = self._resolve_callable(self.backend_callable)
            sampling_probs = self._validate_probabilities(
                callable_fn(manifest=manifest, features=features),
                Path("<in-process DynamicMPNN probabilities>"),
            )
            return {
                "sampling_probs": sampling_probs,
                "dynamicmpnn_backend": "Alex-Abrudan/DynamicMPNN",
                "dynamicmpnn_backend_mode": "in_process",
            }

        if not self.command_template:
            raise RuntimeError(
                "DynamicMPNN mutator is configured for the real Alex-Abrudan/DynamicMPNN "
                "backend, but mutator.DynamicMPNN.backend_callable is not set. "
                "Provide an in-process callable that calls DynamicMPNN.sample(..., "
                "return_logits=True), converts logits to raw amino-acid probabilities, "
                "and returns an array shaped [samples, residues, 20] or [residues, 20]."
            )

        probabilities_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "dynamicmpnn_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        command = self._format_command(manifest_path, run_dir, probabilities_file)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.extra_env.items()})
        subprocess.run(
            command,
            shell=True,
            cwd=str(self.repo_path) if self.repo_path else None,
            env=env,
            check=True,
            timeout=float(self.timeout) if self.timeout else None,
        )

        sampling_probs = self._read_probabilities(probabilities_file)
        return {
            "sampling_probs": sampling_probs,
            "dynamicmpnn_probabilities": str(probabilities_file),
            "dynamicmpnn_manifest": str(manifest_path),
            "dynamicmpnn_command": command,
            "dynamicmpnn_backend": "Alex-Abrudan/DynamicMPNN",
            "dynamicmpnn_backend_mode": "subprocess",
        }

    @staticmethod
    def _resolve_callable(value) -> Callable[..., Any]:
        if callable(value):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("DynamicMPNN backend_callable must be a callable or import path.")
        module_name, sep, attr_name = value.strip().partition(":")
        if not sep:
            module_name, _, attr_name = value.strip().rpartition(".")
        if not module_name or not attr_name:
            raise ValueError(
                "DynamicMPNN backend_callable must use 'module:function' or 'module.function'."
            )
        fn = getattr(importlib.import_module(module_name), attr_name)
        if not callable(fn):
            raise TypeError(f"DynamicMPNN backend_callable is not callable: {value!r}")
        return fn

    def _format_command(self, manifest_path: Path, run_dir: Path, probabilities_file: Path) -> str:
        values = {
            "manifest": shlex.quote(str(manifest_path)),
            "out_dir": shlex.quote(str(run_dir)),
            "probabilities": shlex.quote(str(probabilities_file)),
            "checkpoint": shlex.quote(str(self.checkpoint or "")),
            "temperature": str(self.temperature),
            "num_samples": str(self.num_samples),
        }
        fields = {field_name for _, field_name, _, _ in Formatter().parse(str(self.command_template)) if field_name}
        missing = sorted(fields - set(values))
        if missing:
            raise ValueError(f"Unsupported DynamicMPNN command_template fields: {missing}")
        return str(self.command_template).format(**values)

    def _build_manifest(
        self,
        features: Mapping[str, Any],
        pdb_paths: list[str],
        run_dir: Path,
        probabilities_file: Path,
    ) -> dict[str, Any]:
        return {
            "backend": "Alex-Abrudan/DynamicMPNN",
            "pdb_paths": pdb_paths,
            "fixed_residues_multi": self._load_mapping(features.get("fixed_residues_multi")),
            "redesigned_residues_multi": self._load_mapping(features.get("redesigned_residues_multi")),
            "chain_ids": features.get("parse_these_chains_only") or None,
            "temperature": self.temperature,
            "num_samples": self.num_samples,
            "checkpoint": self.checkpoint,
            "out_dir": str(run_dir),
            "probabilities_path": str(probabilities_file),
            "probabilities_key": self.probabilities_key,
        }

    @staticmethod
    def _load_pdb_paths(features: Mapping[str, Any]) -> list[str]:
        if features.get("pdb_path_multi"):
            with open(features["pdb_path_multi"], "r", encoding="utf-8") as fh:
                return [str(x) for x in json.load(fh)]
        if features.get("pdb_path"):
            return [str(features["pdb_path"])]
        raise ValueError("DynamicMPNN features require pdb_path_multi or pdb_path.")

    @staticmethod
    def _load_mapping(value):
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        with open(value, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _read_probabilities(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"DynamicMPNN raw probability file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if self.probabilities_key not in data:
                    raise KeyError(
                        f"DynamicMPNN probability archive {path} lacks key "
                        f"{self.probabilities_key!r}; available keys: {sorted(data.files)}"
                    )
                arr = data[self.probabilities_key]
        elif suffix == ".json":
            arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        else:
            raise ValueError(
                f"Unsupported DynamicMPNN probability file suffix {suffix!r}; "
                "write .npy, .npz, or .json probabilities."
            )
        return self._validate_probabilities(arr, path)

    @staticmethod
    def _validate_probabilities(arr, path: Path) -> np.ndarray:
        probs = np.asarray(arr, dtype=float)
        if probs.ndim == 2:
            probs = probs[None, ...]
        if probs.ndim != 3 or probs.shape[-1] != 20:
            raise ValueError(
                f"DynamicMPNN probabilities must have shape [samples, residues, 20] "
                f"or [residues, 20], got {probs.shape} from {path}."
            )
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"DynamicMPNN probabilities contain non-finite values: {path}")
        if np.any(probs < 0.0):
            raise ValueError(f"DynamicMPNN probabilities contain negative values: {path}")
        row_sums = probs.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError(
                f"DynamicMPNN probabilities must sum to 1 over amino acids; "
                f"observed min/max sums {row_sums.min():.6g}/{row_sums.max():.6g} in {path}."
            )
        return probs
