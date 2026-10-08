from __future__ import annotations

import json
from pathlib import Path

from astrolabe.core import global_config
from astrolabe.losses.base import loss_registry


def _autodock_penalty_metadata_path(*, out_dir: Path, step: int, oligo_name: str) -> Path:
    return out_dir / "predocker" / f"autodock_penalty_step_{int(step):03d}_{oligo_name}.json"


@loss_registry.register(
    optional_params=["penalty_value"],
    description="Fixed penalty when AutoDock predocker recorded a soft docking failure for the current step.",
)
def autodock_penalty(eval_result, penalty_value: float = 100.0):
    try:
        out_dir = Path(global_config.out_dir)
    except Exception:
        out_dir = Path(".")
    try:
        step_value = int(global_config.step or 0)
    except Exception:
        step_value = 0

    result = {}
    for oligo_name in eval_result.names:
        metadata_path = _autodock_penalty_metadata_path(
            out_dir=out_dir, step=step_value, oligo_name=str(oligo_name)
        )
        if not metadata_path.exists():
            result[oligo_name] = 0.0
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            result[oligo_name] = float(penalty_value)
            continue
        result[oligo_name] = (
            float(penalty_value) if bool(payload.get("penalty_active")) else 0.0
        )
    return result
