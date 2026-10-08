from __future__ import annotations

import json
from pathlib import Path

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores


def test_autodock_penalty_reads_current_step_metadata(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.set_variable("step", 1)

    penalty_dir = tmp_path / "predocker"
    penalty_dir.mkdir(parents=True, exist_ok=True)
    (penalty_dir / "autodock_penalty_step_001_AB.json").write_text(
        json.dumps(
            {
                "penalty_active": True,
                "reason": "binding_site_unavailable",
                "message": "Local pocket alignment RMSD exceeds threshold: 14.699 > 3.000",
            }
        ),
        encoding="utf-8",
    )

    from astrolabe.losses.predocker import autodock_penalty

    eval_result = StandardScores(
        names=["AB"],
        plddt={"AB": [0.9]},
        complex_paths={"AB": tmp_path / "ab.pdb"},
    )

    scores = autodock_penalty(eval_result, penalty_value=100.0)
    assert scores["AB"] == 100.0
