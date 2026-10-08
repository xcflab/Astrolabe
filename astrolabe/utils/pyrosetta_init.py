from __future__ import annotations

from typing import Any


def pyrosetta_was_initialized(pyrosetta_module: Any) -> bool:
    """Return whether PyRosetta has already been initialized."""
    is_initialized = getattr(pyrosetta_module, "is_initialized", None)
    if callable(is_initialized):
        return bool(is_initialized())

    return bool(pyrosetta_module.rosetta.basic.was_init_called())
