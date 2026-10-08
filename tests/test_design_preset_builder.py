from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from astrolabe.design_preset_builder import run_design_preset


def test_run_design_preset_composes_mutates_dumps_and_runs(tmp_path: Path) -> None:
    ns = argparse.Namespace(
        config_dir=None,
        dump_config=str(tmp_path / "config.yaml"),
        print_config=False,
        dry_run=False,
        autoselect_gpu=True,
    )
    calls: list[tuple[str, object]] = []

    def default_config_dir() -> Path:
        return tmp_path / "config"

    def compose_config(**kwargs):
        calls.append(("compose", kwargs))
        return OmegaConf.create({"task_name": "unset"})

    def mutate_config(cfg) -> None:
        cfg.task_name = "demo"
        calls.append(("mutate", cfg.task_name))

    def dump_config(cfg, out_path: str | None) -> None:
        calls.append(("dump", (cfg.task_name, out_path)))

    def run_config(cfg, **kwargs) -> None:
        calls.append(("run", (cfg.task_name, kwargs)))

    rc = run_design_preset(
        ns,
        config_name="benchmark_binder_v1_4",
        mutate_config=mutate_config,
        default_config_dir=default_config_dir,
        compose_config=compose_config,
        dump_config=dump_config,
        run_config=run_config,
    )

    assert rc == 0
    assert calls == [
        (
            "compose",
            {
                "config_dir": tmp_path / "config",
                "config_name": "benchmark_binder_v1_4",
                "overrides": [],
            },
        ),
        ("mutate", "demo"),
        ("dump", ("demo", str(tmp_path / "config.yaml"))),
        ("run", ("demo", {"autoselect_gpu": True})),
    ]


def test_run_design_preset_dry_run_dumps_without_running(tmp_path: Path) -> None:
    ns = argparse.Namespace(
        config_dir=None,
        dump_config=None,
        print_config=False,
        dry_run=True,
        autoselect_gpu=False,
    )
    calls: list[str] = []

    rc = run_design_preset(
        ns,
        config_name="benchmark_binder_v1_4",
        mutate_config=lambda cfg: cfg.update({"task_name": "demo"}),
        default_config_dir=lambda: tmp_path / "config",
        compose_config=lambda **_kwargs: OmegaConf.create({"task_name": "unset"}),
        dump_config=lambda _cfg, _out_path: calls.append("dump"),
        run_config=lambda _cfg, **_kwargs: calls.append("run"),
    )

    assert rc == 0
    assert calls == ["dump"]
