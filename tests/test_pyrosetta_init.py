from __future__ import annotations

import sys
from types import SimpleNamespace

import run as run_mod
from astrolabe.post_eval_hooks import rosetta_minimize
from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized


class _FakeBasic:
    def __init__(self, initialized: bool) -> None:
        self.initialized = initialized

    def was_init_called(self) -> bool:
        return self.initialized


def _fake_pyrosetta(*, initialized: bool):
    calls: list[str] = []

    def init(opts: str) -> None:
        calls.append(opts)

    module = SimpleNamespace(
        init=init,
        rosetta=SimpleNamespace(basic=_FakeBasic(initialized)),
    )
    return module, calls


def test_pyrosetta_was_initialized_uses_rosetta_basic_fallback() -> None:
    module, _calls = _fake_pyrosetta(initialized=True)

    assert pyrosetta_was_initialized(module) is True


def test_rosetta_minimize_does_not_reinitialize_existing_pyrosetta(monkeypatch) -> None:
    module, calls = _fake_pyrosetta(initialized=True)
    monkeypatch.setitem(sys.modules, "pyrosetta", module)
    monkeypatch.setattr(rosetta_minimize, "_PYROSETTA_INITED", False)

    got = rosetta_minimize._ensure_pyrosetta_initialized()

    assert got is module
    assert calls == []


def test_run_pyrosetta_runtime_does_not_reinitialize_existing_pyrosetta(
    monkeypatch,
) -> None:
    module, calls = _fake_pyrosetta(initialized=True)
    monkeypatch.setitem(sys.modules, "pyrosetta", module)
    monkeypatch.setattr(run_mod, "setup_extra_residues", lambda structure_files=None: None)

    run_mod._init_pyrosetta_runtime(structure_files={})

    assert calls == []
