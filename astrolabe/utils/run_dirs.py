from __future__ import annotations

import re
from pathlib import Path


RUN_DIR_MIN_WIDTH = 3
_RUN_DIR_RE = re.compile(r"^(?P<index>\d+)_run(?:_.+)?$")


def format_run_dir_name(index: int) -> str:
    index = int(index)
    if index < 1:
        raise ValueError(f"run index must be >= 1, got {index}")
    return f"{index:0{RUN_DIR_MIN_WIDTH}d}_run"


def parse_run_dir_index(name: str | Path) -> int | None:
    if isinstance(name, Path):
        name = name.name
    match = _RUN_DIR_RE.match(str(name))
    if match is None:
        return None
    return int(match.group("index"))


def run_dir_sort_key(path: str | Path) -> tuple[int, str]:
    if isinstance(path, Path):
        name = path.name
    else:
        name = Path(str(path)).name
    run_index = parse_run_dir_index(name)
    if run_index is None:
        raise ValueError(f"not a run directory name: {name}")
    return run_index, name
