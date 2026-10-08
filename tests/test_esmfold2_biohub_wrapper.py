from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.esmfold2 import ESMFold2Evaluator
from astrolabe.evaluators.wrappers.esmfold2 import (
    ESMFold2Runner,
    _ccd_cache_dir,
    _load_esmfold2_model,
    _model_load_kwargs,
    _resolve_model_source,
    _worker_run,
)


def _init_global(tmp_path: Path, *, step: int = 3) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN, length=4),
            "B": SimpleNamespace(type=ProtomerTypes.PROTEIN, length=3),
        },
    )
    global_config.set_variable("step", step)


def test_esmfold2_runner_writes_payload_and_reads_subprocess_output(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, *, cwd, env, text, stdout, stderr, timeout):
        output_json = Path(cmd[cmd.index("--output-json") + 1])
        payload = json.loads(Path(cmd[cmd.index("--input-json") + 1]).read_text(encoding="utf-8"))
        assert cmd[:3] == [sys.executable, "-m", "astrolabe.evaluators.wrappers.esmfold2"]
        assert Path(cwd) == Path(__file__).resolve().parents[1]
        assert payload["model_name"] == "biohub/ESMFold2"
        assert payload["esm_repo_path"] == "/vendor/biohub_esm"
        assert payload["transformers_repo_path"] == "/vendor/biohub_transformers"
        assert payload["jobs"][0]["chains"][0] == {"id": "A", "sequence": "ACDE"}
        output_json.write_text(
            json.dumps({"jobs": [{"name": "A", "structure_path": str(tmp_path / "a.cif"), "plddt": [0.5], "plddt_mean": 0.5}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="warn")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = ESMFold2Runner(
        esm_repo_path="/vendor/biohub_esm",
        transformers_repo_path="/vendor/biohub_transformers",
        timeout_seconds=12,
    )

    result = runner.run(
        [{"name": "A", "chains": [{"id": "A", "sequence": "ACDE"}]}],
        tmp_path / "raw",
    )

    assert result["jobs"][0]["plddt_mean"] == 0.5
    assert (tmp_path / "raw" / "esmfold2.stdout.log").read_text(encoding="utf-8") == "ok"
    assert (tmp_path / "raw" / "esmfold2.stderr.log").read_text(encoding="utf-8") == "warn"


def test_esmfold2_default_model_uses_pinned_snapshot(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("GH_ESMFOLD2_MODEL_SNAPSHOT", str(snapshot))

    assert _resolve_model_source("biohub/ESMFold2") == (str(snapshot), True)
    assert _resolve_model_source("biohub/ESMFold2-Fast") == (
        "biohub/ESMFold2-Fast",
        False,
    )


def test_esmfold2_loads_esmc_from_pinned_offline_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    esmc_snapshot = tmp_path / "esmc-6b"
    esmc_snapshot.mkdir()
    (esmc_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (esmc_snapshot / "model.safetensors.index.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setenv("GH_ESMFOLD2_ESMC_SNAPSHOT", str(esmc_snapshot))

    assert _model_load_kwargs(
        {"esmc_precision": "bf16", "cache_dir": str(tmp_path / "cache")},
        pinned_snapshot=True,
    ) == {
        "esmc_precision": "bf16",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }


def test_esmfold2_pinned_esmc_snapshot_uses_load_esmc_boundary(tmp_path: Path) -> None:
    esmc_snapshot = tmp_path / "esmc"
    esmc_snapshot.mkdir()
    captured = {}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(esmc_id="biohub/ESMC-6B")

        @classmethod
        def from_pretrained(cls, source, *args, load_esmc=True, **kwargs):
            if "esmc_id" in kwargs:
                raise TypeError(
                    "ESMFold2Model.__init__() got an unexpected keyword argument 'esmc_id'"
                )
            captured["source"] = source
            captured["load_esmc"] = load_esmc
            captured["model_kwargs"] = kwargs
            return cls()

        def load_esmc(self, esmc_model_path, *, precision="bf16"):
            captured["load_esmc_call"] = {
                "esmc_model_path": esmc_model_path,
                "precision": precision,
            }

        def eval(self):
            captured["eval"] = True
            return self

    model = _load_esmfold2_model(
        FakeModel,
        "/snapshots/esmfold2",
        {"esmc_precision": "fp32", "local_files_only": True},
        esmc_snapshot_path=esmc_snapshot.resolve(),
    )

    assert captured["source"] == "/snapshots/esmfold2"
    assert captured["load_esmc"] is False
    assert captured["model_kwargs"] == {
        "esmc_precision": "fp32",
        "local_files_only": True,
    }
    assert captured["load_esmc_call"] == {
        "esmc_model_path": str(esmc_snapshot.resolve()),
        "precision": "fp32",
    }
    assert model.config.esmc_id == str(esmc_snapshot.resolve())
    assert captured["eval"] is True


def test_esmfold2_uses_ccd_from_pinned_offline_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "esmfold2"
    snapshot.mkdir()
    ccd = snapshot / "ccd.pkl"
    ccd.write_bytes(b"pinned")

    assert _ccd_cache_dir(str(snapshot), pinned_snapshot=True) == snapshot.resolve()

    ccd.unlink()
    with pytest.raises(FileNotFoundError, match="CCD pickle"):
        _ccd_cache_dir(str(snapshot), pinned_snapshot=True)


def test_esmfold2_remote_model_keeps_upstream_ccd_resolution() -> None:
    assert _ccd_cache_dir("biohub/ESMFold2-Fast", pinned_snapshot=False) is None


def test_esmfold2_offline_remote_model_requires_explicit_ccd(monkeypatch) -> None:
    monkeypatch.delenv("ESMCFOLD_CCD_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="Offline ESMFold2"):
        _ccd_cache_dir(
            "biohub/ESMFold2-Fast",
            pinned_snapshot=False,
            local_files_only=True,
        )


def test_esmfold2_worker_passes_pinned_ccd_to_real_builder_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "ccd.pkl").write_bytes(b"pinned")
    monkeypatch.setenv("GH_ESMFOLD2_MODEL_SNAPSHOT", str(snapshot))
    esmc_snapshot = tmp_path / "esmc"
    esmc_snapshot.mkdir()
    (esmc_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (esmc_snapshot / "model.safetensors.index.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setenv("GH_ESMFOLD2_ESMC_SNAPSHOT", str(esmc_snapshot))

    captured = {}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(esmc_id="biohub/ESMC-6B")

        @classmethod
        def from_pretrained(cls, source, *args, load_esmc=True, **kwargs):
            if "esmc_id" in kwargs:
                raise TypeError(
                    "ESMFold2Model.__init__() got an unexpected keyword argument 'esmc_id'"
                )
            captured["source"] = source
            captured["load_esmc"] = load_esmc
            captured["model_kwargs"] = kwargs
            return cls()

        def load_esmc(self, esmc_model_path, *, precision="bf16"):
            captured["load_esmc_call"] = {
                "esmc_model_path": esmc_model_path,
                "precision": precision,
            }

        def eval(self):
            return self

    class FakeBuilder:
        def __init__(self, *, ccd_cache=None):
            captured["ccd_cache"] = ccd_cache

    esmfold2_module = ModuleType("esm.models.esmfold2")
    esmfold2_module.ESMFold2InputBuilder = FakeBuilder
    esmfold2_module.ProteinInput = object
    esmfold2_module.StructurePredictionInput = object
    transformers_module = ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    transformers_module.ESMFold2Model = FakeModel
    module_tree = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "esm": ModuleType("esm"),
        "esm.models": ModuleType("esm.models"),
        "esm.models.esmfold2": esmfold2_module,
        "transformers": ModuleType("transformers"),
        "transformers.models": ModuleType("transformers.models"),
        "transformers.models.esmfold2": ModuleType("transformers.models.esmfold2"),
        "transformers.models.esmfold2.modeling_esmfold2": transformers_module,
    }
    for name, module in module_tree.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = _worker_run(
        {
            "model_name": "biohub/ESMFold2",
            "output_dir": str(tmp_path / "output"),
            "jobs": [],
            "local_files_only": True,
        }
    )

    assert result["jobs"] == []
    assert captured["source"] == str(snapshot)
    assert captured["load_esmc"] is False
    assert captured["model_kwargs"]["local_files_only"] is True
    assert "esmc_id" not in captured["model_kwargs"]
    assert captured["load_esmc_call"] == {
        "esmc_model_path": str(esmc_snapshot.resolve()),
        "precision": "bf16",
    }
    assert captured["ccd_cache"] == snapshot.resolve()


def test_esmfold2_pinned_top_model_requires_pinned_esmc(monkeypatch) -> None:
    monkeypatch.delenv("GH_ESMFOLD2_ESMC_SNAPSHOT", raising=False)
    with pytest.raises(FileNotFoundError, match="requires GH_ESMFOLD2_ESMC_SNAPSHOT"):
        _model_load_kwargs({}, pinned_snapshot=True)


def test_esmfold2_evaluator_returns_standard_scores(tmp_path: Path) -> None:
    _init_global(tmp_path)
    raw_structure = tmp_path / "raw.cif"
    raw_structure.write_text("data_A\n#\n", encoding="utf-8")

    class FakeRunner:
        def run(self, jobs, output_dir):
            assert jobs == [{"name": "AB", "chains": [{"id": "A", "sequence": "ACDE"}, {"id": "B", "sequence": "FGH"}]}]
            return {
                "jobs": [
                    {
                        "name": "AB",
                        "structure_path": str(raw_structure),
                        "plddt": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        "plddt_mean": 0.4,
                        "ptm": 0.2,
                        "iptm": 0.1,
                        "pae": [[0.0]],
                        "pde": [[1.0]],
                    }
                ]
            }

    evaluator = ESMFold2Evaluator(runner=FakeRunner(), raw_model_results_dir=tmp_path / "raw")
    scores = evaluator.evaluate(
        {
            "AB": Input(
                name="AB",
                chains=(
                    ProteinChain(id="A", sequence="ACDE", ptms=()),
                    ProteinChain(id="B", sequence="FGH", ptms=()),
                ),
                rng_seeds=(0,),
            )
        },
        runtime_config={},
    )

    assert scores.names == ["AB"]
    assert scores.plddt["AB"] == [0.4]
    assert scores.ptm["AB"] == [0.2]
    assert scores.iptm["AB"] == [0.1]
    assert np.allclose(scores.per_proto_scores["A"][0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(scores.per_proto_scores["B"][0], [0.5, 0.6, 0.7])
    assert scores.complex_paths["AB"].name == "step_003_AB_model.cif"
    assert scores.complex_paths["AB"].read_text(encoding="utf-8") == "data_A\n#\n"
