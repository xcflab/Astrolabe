from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def _install_dummy_byprot() -> None:
    class _DummyModel:
        mask_id = 0

        def __init__(self) -> None:
            self.tokenizer = object()

        @classmethod
        def from_pretrained(cls, _name: str):
            return cls()

        def to(self, _device: str):
            return self

        def eval(self):
            return None

    byprot = types.ModuleType("byprot")
    models = types.ModuleType("byprot.models")
    dplm_pkg = types.ModuleType("byprot.models.dplm")
    dplm_mod = types.ModuleType("byprot.models.dplm.dplm")
    dplm_mod.DiffusionProteinLanguageModel = _DummyModel

    sys.modules["byprot"] = byprot
    sys.modules["byprot.models"] = models
    sys.modules["byprot.models.dplm"] = dplm_pkg
    sys.modules["byprot.models.dplm.dplm"] = dplm_mod


def test_dplm_infer_uses_seed_from_config(tmp_path: Path, monkeypatch) -> None:
    _install_dummy_byprot()

    from astrolabe.mutators.wrappers import dplm_infer

    calls: list[int | None] = []

    def _record(seed):
        calls.append(seed)

    monkeypatch.setattr(dplm_infer, "_seed_everything", _record)
    monkeypatch.setattr(dplm_infer.random, "randint", lambda *_args, **_kwargs: 999)

    cfg_path = tmp_path / "cfg.json"
    out_path = tmp_path / "out.json"
    cfg_path.write_text(
        json.dumps({"repo_path": str(tmp_path), "seed": 123, "tasks": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["dplm_infer.py", "--config", str(cfg_path), "--output", str(out_path)],
    )
    dplm_infer.main()

    assert calls == [123]


def test_dplm_default_model_uses_pinned_snapshot(monkeypatch, tmp_path: Path) -> None:
    from astrolabe.mutators.wrappers import dplm_infer

    snapshot = tmp_path / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("GH_DPLM_MODEL_SNAPSHOT", str(snapshot))

    assert dplm_infer._resolve_model_name("airkingbd/dplm_150m") == str(snapshot)
    assert dplm_infer._resolve_model_name("custom/model") == "custom/model"
