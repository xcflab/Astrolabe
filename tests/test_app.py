from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe import Astrolabe, app


def test_load_run_module_does_not_depend_on_cwd_or_repo_root_on_syspath(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    module = app._load_run_module()

    # The runner is resolved from the package location, not from the cwd or sys.path.
    assert Path(module.__file__).resolve() == app.run_script_path().resolve()
    assert Path(module.__file__).name == "run.py"
    assert hasattr(module, "_run_pipeline")


@pytest.mark.parametrize("public_api", [False, True])
def test_run_from_cfg_uses_canonical_repeat_lifecycle(
    monkeypatch, tmp_path: Path, capsys, public_api
) -> None:
    run = app._load_run_module()
    raw_root = tmp_path / "raw"
    monkeypatch.delenv("GH_WAVE1_UNUSED_MISSING", raising=False)
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "count": 4,
            "repeats": "${count}",
            "seed": 17,
            "raw_model_results_dir": str(raw_root),
            "unused": "${oc.env:GH_WAVE1_UNUSED_MISSING}",
        }
    )
    seen = []

    monkeypatch.setattr(run, "install_handlers", lambda: None)
    monkeypatch.setattr(run, "stop_requested", lambda: len(seen) == 3)
    monkeypatch.setattr(
        run,
        "_run_pipeline",
        lambda run_cfg: seen.append(
            (int(run_cfg.seed), Path(run_cfg.raw_model_results_dir))
        ),
    )

    if public_api:
        application = Astrolabe(cfg)
        assert application.config is cfg
        application.run()
    else:
        app.run_from_cfg(cfg)

    assert seen == [
        (17, raw_root / "repeat_001"),
        (18, raw_root / "repeat_002"),
        (19, raw_root / "repeat_003"),
    ]
    assert "exiting after repeat 3/4." in capsys.readouterr().err
