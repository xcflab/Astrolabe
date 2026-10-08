from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_protenix_init_env_fails_when_cuda_visible_but_unavailable(monkeypatch) -> None:
    from astrolabe.evaluators.wrappers import protenix as protenix_mod

    runner = protenix_mod.ProtenixRunner.__new__(protenix_mod.ProtenixRunner)
    runner.configs = SimpleNamespace(use_deepspeed_evo_attention=False)
    runner.print = lambda *args, **kwargs: None

    monkeypatch.setattr(
        protenix_mod,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
        raising=False,
    )
    monkeypatch.setattr(protenix_mod.torch.cuda, "device_count", lambda: 1, raising=True)
    monkeypatch.setattr(protenix_mod.torch.cuda, "is_available", lambda: False, raising=True)

    with pytest.raises(RuntimeError, match="Protenix requires CUDA"):
        runner.init_env()
