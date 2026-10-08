#!/usr/bin/env python3
"""Long-lived DPLM inference daemon (JSONL stdin/stdout).

Protocol (one JSON object per line):
  - {"type":"infer","id":"<uuid>","seed":123,"tasks":[{"name":"X","masked_sequence":"...","num_samples":1}]}
    -> {"type":"result","id":"<uuid>","results":[{"name":"X","samples":["..."]}]}
  - {"type":"shutdown"} -> {"type":"ok","message":"bye"}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from astrolabe.mutators.wrappers.dplm_infer import _resolve_model_name


def _ensure_byprot_on_path() -> None:
    """Make `byprot` importable without requiring an installed package."""
    repo_path = pathlib.Path.cwd()
    src_path = repo_path / "src"
    if src_path.exists():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPLM JSONL daemon.")
    p.add_argument("--model-name", default="airkingbd/dplm_150m")
    p.add_argument("--device", default="auto")
    p.add_argument("--sampling-strategy", default="gumbel_argmax")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


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
    return "".join(str(seq).split())


def _run_generation(
    *,
    model,
    tokenizer,
    masked_sequence: str,
    num_samples: int,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    device: str,
) -> List[str]:
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


def _write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()

    # Lazy import (keeps `gh doctor` light if someone imports this module).
    _ensure_byprot_on_path()
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

    device = _select_device(args.device)
    _seed_everything(random.randint(0, 1_000_000))

    model = DiffusionProteinLanguageModel.from_pretrained(
        _resolve_model_name(args.model_name)
    )
    model = model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"type": "error", "message": f"invalid json: {exc}"})
            continue

        mtype = msg.get("type")
        if mtype == "shutdown":
            _write({"type": "ok", "message": "bye"})
            return 0
        if mtype != "infer":
            _write({"type": "error", "message": f"unknown type: {mtype}"})
            continue

        req_id = msg.get("id")
        _seed_everything(msg.get("seed"))

        tasks: List[Dict[str, Any]] = msg.get("tasks", []) or []
        results: List[Dict[str, Any]] = []
        for task in tasks:
            name = task.get("name")
            masked_sequence = task.get("masked_sequence")
            if name is None or masked_sequence is None:
                continue
            num_samples = int(task.get("num_samples", args.num_samples) or 0)
            samples = _run_generation(
                model=model,
                tokenizer=tokenizer,
                masked_sequence=str(masked_sequence),
                num_samples=num_samples,
                max_iter=int(msg.get("max_iter", args.max_iter)),
                sampling_strategy=str(msg.get("sampling_strategy", args.sampling_strategy)),
                temperature=float(msg.get("temperature", args.temperature)),
                device=device,
            )
            results.append({"name": name, "samples": samples})

        _write({"type": "result", "id": req_id, "results": results})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
