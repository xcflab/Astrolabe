from __future__ import annotations

import os
from pathlib import Path

import jax
from omegaconf import OmegaConf

from astrolabe.evaluators.wrappers.alphafold3 import alphafold3_setup


def test_alphafold3_setup_sets_refstructure_env_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jax, "local_devices", lambda **_kwargs: [jax.devices("cpu")[0]])

    monkeypatch.delenv("ALPHAFOLD3_REFSTRUCTURE_WORKERS", raising=False)
    monkeypatch.delenv("ALPHAFOLD3_REFSTRUCTURE_CACHE", raising=False)
    monkeypatch.delenv("ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES", raising=False)

    cfg = OmegaConf.create(
        {
            "jax_compilation_cache_dir": None,
            "flash_attention_implementation": "cudnn",
            "num_samples": 1,
            "num_recycles": 1,
            "save_embeddings": False,
            "model_dir": str(tmp_path),
            "refstructure_workers": 8,
            "refstructure_cache": True,
            "refstructure_cache_max_entries": 123,
        }
    )

    alphafold3_setup(cfg)

    assert os.environ["ALPHAFOLD3_REFSTRUCTURE_WORKERS"] == "8"
    assert os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE"] == "1"
    assert os.environ["ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES"] == "123"

