from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


def _install_dummy_esm3(monkeypatch):
    calls: dict[str, object] = {}

    class _DummyRunner:
        def __init__(self) -> None:
            self.eval_called = False
            self.to_device = None

        def eval(self) -> None:
            self.eval_called = True

        def to(self, device):
            self.to_device = device
            return self

    class _DummyESM3:
        @classmethod
        def from_pretrained(cls, model_name: str, device=None):
            calls["model_name"] = model_name
            calls["device"] = device
            return _DummyRunner()

    class _DummyESMProtein:
        def __init__(self, sequence: str) -> None:
            self.sequence = sequence

    class _DummyLogitsConfig:
        def __init__(self, *, sequence: bool) -> None:
            self.sequence = sequence

    class _DummyTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return ord(token[0])

    esm3_pkg = types.ModuleType("esm3")
    esm3_pkg.__path__ = []
    models_pkg = types.ModuleType("esm3.models")
    esm3_mod = types.ModuleType("esm3.models.esm3")
    sdk_pkg = types.ModuleType("esm3.sdk")
    api_mod = types.ModuleType("esm3.sdk.api")
    tokenization_mod = types.ModuleType("esm3.tokenization")
    pretrained_mod = types.ModuleType("esm3.pretrained")
    pretrained_mod.data_root = lambda model: Path("/unpinned") / model
    utils_pkg = types.ModuleType("esm3.utils")
    utils_pkg.__path__ = []
    constants_pkg = types.ModuleType("esm3.utils.constants")
    constants_pkg.__path__ = []
    constants_mod = types.ModuleType("esm3.utils.constants.esm3")
    constants_mod.data_root = lambda model: Path("/unpinned") / model

    esm3_mod.ESM3 = _DummyESM3
    api_mod.ESMProtein = _DummyESMProtein
    api_mod.LogitsConfig = _DummyLogitsConfig
    tokenization_mod.get_esm3_model_tokenizers = lambda _name: types.SimpleNamespace(
        sequence=_DummyTokenizer()
    )

    for name, module in {
        "esm3": esm3_pkg,
        "esm3.models": models_pkg,
        "esm3.models.esm3": esm3_mod,
        "esm3.sdk": sdk_pkg,
        "esm3.sdk.api": api_mod,
        "esm3.tokenization": tokenization_mod,
        "esm3.pretrained": pretrained_mod,
        "esm3.utils": utils_pkg,
        "esm3.utils.constants": constants_pkg,
        "esm3.utils.constants.esm3": constants_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return calls


def test_esm3_mutator_passes_requested_device_to_from_pretrained(monkeypatch) -> None:
    calls = _install_dummy_esm3(monkeypatch)

    from astrolabe.mutators.esm3 import ESM3Mutator

    mutator = ESM3Mutator(model_name="esm3-open", device="cpu")

    assert calls["model_name"] == "esm3-open"
    assert calls["device"] == torch.device("cpu")
    assert mutator.runner.eval_called is True
    assert mutator.runner.to_device == torch.device("cpu")


def test_esm3_uses_pinned_snapshot_data_root(monkeypatch, tmp_path: Path) -> None:
    _install_dummy_esm3(monkeypatch)
    snapshot = tmp_path / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("GH_ESM3_MODEL_SNAPSHOT", str(snapshot))

    from astrolabe.mutators.esm3 import ESM3Mutator

    ESM3Mutator(model_name="esm3-open", device="cpu")

    pretrained = sys.modules["esm3.pretrained"]
    constants = sys.modules["esm3.utils.constants.esm3"]
    assert pretrained.data_root("esm3") == snapshot.resolve()
    assert constants.data_root("esm3") == snapshot.resolve()
