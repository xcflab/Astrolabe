from __future__ import annotations

import argparse
import pathlib
from collections.abc import Callable
from typing import Any

from omegaconf import OmegaConf


ComposeConfig = Callable[..., Any]
DumpConfig = Callable[[Any, str | None], None]
MutateConfig = Callable[[Any], None]
RunConfig = Callable[..., None]


def config_dir_from_namespace(
    ns: argparse.Namespace,
    *,
    default_config_dir: Callable[[], pathlib.Path],
) -> pathlib.Path:
    raw_config_dir = getattr(ns, "config_dir", None)
    return pathlib.Path(raw_config_dir).resolve() if raw_config_dir else default_config_dir()


def compose_config_from_namespace(
    ns: argparse.Namespace,
    *,
    config_name: str,
    default_config_dir: Callable[[], pathlib.Path],
    compose_config: ComposeConfig,
) -> Any:
    return compose_config(
        config_dir=config_dir_from_namespace(ns, default_config_dir=default_config_dir),
        config_name=config_name,
        overrides=[],
    )


def finish_config(
    ns: argparse.Namespace,
    cfg: Any,
    *,
    dump_config: DumpConfig,
    run_config: RunConfig,
    resolve_print: bool = False,
) -> int:
    if bool(getattr(ns, "print_config", False)) or bool(getattr(ns, "dry_run", False)):
        print(OmegaConf.to_yaml(cfg, resolve=resolve_print))
        if bool(getattr(ns, "dry_run", False)):
            dump_config(cfg, getattr(ns, "dump_config", None))
            return 0

    dump_config(cfg, getattr(ns, "dump_config", None))
    run_config(cfg, autoselect_gpu=bool(getattr(ns, "autoselect_gpu", False)))
    return 0


def run_design_preset(
    ns: argparse.Namespace,
    *,
    config_name: str,
    mutate_config: MutateConfig,
    default_config_dir: Callable[[], pathlib.Path],
    compose_config: ComposeConfig,
    dump_config: DumpConfig,
    run_config: RunConfig,
    resolve_print: bool = False,
) -> int:
    cfg = compose_config_from_namespace(
        ns,
        config_name=config_name,
        default_config_dir=default_config_dir,
        compose_config=compose_config,
    )
    mutate_config(cfg)
    return finish_config(
        ns,
        cfg,
        dump_config=dump_config,
        run_config=run_config,
        resolve_print=resolve_print,
    )
