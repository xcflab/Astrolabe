#!/usr/bin/env python3
"""Helper script to run DPLM inference from an external interpreter."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional


def _ensure_byprot_on_path(repo_path: pathlib.Path) -> None:
    """Make `byprot` importable without requiring an installed package."""
    src_path = repo_path / "src"
    if src_path.exists():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPLM inference on masked sequences.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration with tasks.")
    parser.add_argument("--output", required=True, help="Destination JSON for generated sequences.")
    return parser.parse_args()


def _resolve_repo_path(raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DPLM repo path '{path}' does not exist.")
    return path


def _resolve_model_name(model_name: str) -> str:
    pinned_snapshot = os.environ.get("GH_DPLM_MODEL_SNAPSHOT")
    if pinned_snapshot and model_name == "airkingbd/dplm_150m":
        return pinned_snapshot
    return model_name


def _select_device(requested: str) -> str:
    import torch

    if requested in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_sequence(seq: str) -> str:
    """Remove whitespace between tokens emitted by the tokenizer."""
    return "".join(seq.split())


def _run_generation(
    model,
    tokenizer,
    masked_sequence: str,
    num_samples: int,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    device: str,
):
    import torch

    if num_samples <= 0:
        return []

    batch_sequences = [masked_sequence] * num_samples
    encoded = tokenizer.batch_encode_plus(
        batch_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    partial_masks = input_ids.ne(model.mask_id)

    with torch.no_grad():
        outputs = model.generate(
            input_tokens=input_ids,
            tokenizer=tokenizer,
            max_iter=max_iter,
            sampling_strategy=sampling_strategy,
            temperature=temperature,
            partial_masks=partial_masks,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [_clean_sequence(seq) for seq in decoded]


def main() -> None:
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    repo_path_raw = config.get("repo_path") or os.getcwd()
    repo_path = _resolve_repo_path(str(repo_path_raw))
    _ensure_byprot_on_path(repo_path)

    # Lazy imports after sys.path update
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

    import torch

    device = _select_device(config.get("device", "auto"))
    raw_seed = config.get("seed")
    try:
        seed = int(raw_seed) if raw_seed is not None else None
    except (TypeError, ValueError):
        seed = None
    _seed_everything(seed)

    model_name = _resolve_model_name(
        config.get("model_name", "airkingbd/dplm_150m")
    )
    model = DiffusionProteinLanguageModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    max_iter = int(config.get("max_iter", 400))
    sampling_strategy = config.get("sampling_strategy", "gumbel_argmax")
    temperature = float(config.get("temperature", 1.0))
    default_num_samples = int(config.get("num_samples", 1))

    tasks: List[Dict[str, Any]] = config.get("tasks", [])
    results: List[Dict[str, Any]] = []

    for task in tasks:
        name = task["name"]
        masked_sequence = task["masked_sequence"]
        num_samples = int(task.get("num_samples", default_num_samples))
        sequences = _run_generation(
            model=model,
            tokenizer=tokenizer,
            masked_sequence=masked_sequence,
            num_samples=num_samples,
            max_iter=max_iter,
            sampling_strategy=sampling_strategy,
            temperature=temperature,
            device=device,
        )
        results.append({"name": name, "samples": sequences})

    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle)


if __name__ == "__main__":
    main()
