from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_protenix_load_checkpoint_does_not_delete_checkpoint_on_load_error(
    tmp_path, monkeypatch
) -> None:
    from astrolabe.evaluators.wrappers import protenix as protenix_mod

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"not-a-real-checkpoint")

    runner = protenix_mod.ProtenixRunner.__new__(protenix_mod.ProtenixRunner)
    runner.configs = SimpleNamespace(
        load_checkpoint_path=str(checkpoint_path),
        load_strict=True,
        data=None,
    )
    runner.device = "cpu"
    runner.print = lambda *args, **kwargs: None

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(protenix_mod.torch, "load", _boom, raising=True)

    with pytest.raises(RuntimeError, match="boom"):
        runner.load_checkpoint()

    assert checkpoint_path.exists(), "load_checkpoint must not delete checkpoints on error"

