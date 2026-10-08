from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.surfdesign import SurfDesignMutator
from astrolabe.mutators.wrappers.surfdesign_daemon import Runtime, SurfaceBuilder, _infer
from astrolabe.mutators.wrappers.surfdesign_runner import SurfDesignRunner
from astrolabe.runtime_requirements import task_requirements


def _fake_daemon(path: Path) -> Path:
    path.write_text(
        """\
import argparse
import json
import os
import sys

argparse.ArgumentParser(add_help=False).parse_known_args()
print(json.dumps({"type": "ready", "pid": os.getpid(), "model_load_count": 1}), flush=True)
count = 0
for raw in sys.stdin:
    request = json.loads(raw)
    if request.get("type") == "shutdown":
        print(json.dumps({"type": "ok", "message": "bye"}), flush=True)
        raise SystemExit(0)
    count += 1
    probabilities = [[0.05] * 20, [0.05] * 20]
    print(json.dumps({
        "type": "result",
        "id": request["id"],
        "pid": os.getpid(),
        "model_load_count": 1,
        "request_count": count,
        "probabilities": probabilities,
        "sequence": "AC",
    }), flush=True)
""",
        encoding="utf-8",
    )
    return path


def _runner_config(tmp_path: Path) -> tuple[dict, Path]:
    repo_path = tmp_path / "surfdesign"
    repo_path.mkdir()
    experiment_path = tmp_path / "experiment"
    (experiment_path / ".hydra").mkdir(parents=True)
    (experiment_path / ".hydra" / "config.yaml").write_text("model: stub\n")
    (experiment_path / "checkpoints").mkdir()
    (experiment_path / "checkpoints" / "best.ckpt").write_text("stub\n")
    daemon_script = _fake_daemon(tmp_path / "fake_surfdesign_daemon.py")
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    return (
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_path),
            "experiment_path": str(experiment_path),
            "checkpoint": "best.ckpt",
            "daemon_script_path": str(daemon_script),
            "output_dir": str(tmp_path / "output"),
            "startup_timeout": 5,
            "request_timeout": 5,
            "seed": 17,
        },
        pdb_path,
    )


def test_runner_reuses_one_daemon_and_model(tmp_path: Path) -> None:
    config, pdb_path = _runner_config(tmp_path)
    runner = SurfDesignRunner(config)
    cif_path = pdb_path.with_suffix(".cif")
    cif_path.write_text("data_stub\n", encoding="utf-8")

    first = runner.design_sequences(pdb_path=pdb_path, fixed_residues="A1")
    second = runner.design_sequences(pdb_path=cif_path)

    assert first["sampling_probs"].shape == (1, 2, 20)
    assert np.allclose(first["sampling_probs"].sum(axis=-1), 1.0)
    assert first["surfdesign_daemon_pid"] == second["surfdesign_daemon_pid"]
    assert first["surfdesign_model_load_count"] == 1
    assert second["surfdesign_model_load_count"] == 1
    assert [first["surfdesign_request_count"], second["surfdesign_request_count"]] == [1, 2]

    runner.close()


def test_runner_rejects_per_call_loading(tmp_path: Path) -> None:
    config, _pdb_path = _runner_config(tmp_path)
    config["daemon"] = False

    with pytest.raises(ValueError, match="persistent daemon"):
        SurfDesignRunner(config)


def test_runner_does_not_restart_dead_daemon(tmp_path: Path) -> None:
    config, pdb_path = _runner_config(tmp_path)
    runner = SurfDesignRunner(config)
    runner.design_sequences(pdb_path=pdb_path)
    assert runner._proc is not None
    runner._proc.terminate()
    runner._proc.wait()

    with pytest.raises(RuntimeError, match="implicit model restart is disabled"):
        runner.design_sequences(pdb_path=pdb_path)
    with pytest.raises(RuntimeError, match="exited before shutdown"):
        runner.close()


def test_mutator_shutdown_closes_runner() -> None:
    runner = SimpleNamespace(closed=False)
    runner.close = lambda: setattr(runner, "closed", True)
    mutator = SurfDesignMutator(sequence_model_runner=runner)

    mutator.shutdown()

    assert runner.closed is True


def test_registry_builds_surfdesign_mutator(tmp_path: Path) -> None:
    runner_config, _pdb_path = _runner_config(tmp_path)
    mutator_config = OmegaConf.create(
        {
            "SurfDesign": runner_config,
            "mutator_terms": {
                "surface": {
                    "_type": "SurfDesign",
                    "position_selector": "plddt < 80",
                }
            },
        }
    )
    global_config.reset()

    mutators = build_mutators(mutator_config, tmp_path / "raw")

    assert isinstance(mutators["surface"], SurfDesignMutator)
    assert mutators["surface"].position_selector == "plddt < 80"
    mutators["surface"].shutdown()
    global_config.reset()


def test_surface_backbone_normalization_preserves_chain_ordinals(tmp_path: Path) -> None:
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   ALA A  10      11.000  12.000  13.000  1.00 20.00           N
ATOM      2  CA  ALA A  10      12.000  12.000  13.000  1.00 20.00           C
ATOM      3  N   CYS B  42      21.000  22.000  23.000  1.00 20.00           N
ATOM      4  CA  CYS B  42      22.000  22.000  23.000  1.00 20.00           C
HETATM    5  C1  LIG C   1      31.000  32.000  33.000  1.00 20.00           C
""",
        encoding="utf-8",
    )

    lines, chain_order, chain_lengths = SurfaceBuilder._backbone_lines(pdb_path)

    assert chain_order == ["A", "B"]
    assert chain_lengths == {"A": 1, "B": 1}
    assert [line[22:26] for line in lines] == ["   1", "   1", "   1", "   1"]
    assert all(line.startswith("ATOM") for line in lines)


def test_open_dde_mmcif_can_be_converted_for_surfdesign(tmp_path: Path) -> None:
    builder = object.__new__(SurfaceBuilder)
    builder.work_dir = tmp_path
    source = Path(__file__).resolve().parents[1] / "examples" / "input" / "9TP_cid.cif"

    converted = builder._as_pdb(source)
    lines, chain_order, chain_lengths = SurfaceBuilder._backbone_lines(converted)

    assert converted.suffix == ".pdb"
    assert lines
    assert chain_order
    assert sum(chain_lengths.values()) > 0


def test_daemon_infer_returns_standard_amino_acid_probabilities(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    surface = tuple(torch.zeros((2, 3)) for _ in range(4))
    surface_builder = SimpleNamespace(
        build=lambda _path: (pdb_path, ["A"], {"A": 2}, surface)
    )
    featurizer = SimpleNamespace(get_surface=None)

    class Alphabet:
        mask_idx = 31

        @staticmethod
        def featurize(raw_batch):
            assert raw_batch[0]["name"] == "input"
            return {
                "tokens": torch.tensor([[30, 0, 1, 29]]),
                "coord_mask": torch.tensor([[False, True, True, False]]),
            }

        @staticmethod
        def get_idx(amino_acid):
            return "ACDEFGHIKLMNPQRSTVWY".index(amino_acid)

        @staticmethod
        def decode(_tokens, remove_special):
            assert remove_special is True
            return ["AC"]

    Alphabet.featurizer = featurizer

    class DataProcessor:
        @staticmethod
        def parse_PDB(_path, input_chain_list):
            assert input_chain_list == ["A"]
            return {"name": "unused"}

    class Generator:
        @staticmethod
        def generate(model, batch, replace_visible_tokens):
            assert model is runtime.model
            assert replace_visible_tokens is True
            assert batch["prev_tokens"][0, 1] == batch["tokens"][0, 1]
            return batch["tokens"].clone(), torch.zeros_like(batch["tokens"], dtype=torch.float)

    logits = torch.zeros((1, 4, 20))
    logits[0, 1, 0] = 5.0
    logits[0, 2, 1] = 5.0
    model = SimpleNamespace(
        cfg=SimpleNamespace(separate_loss=True),
        forward_encoder=lambda _batch: {"logits": torch.zeros_like(logits)},
        decoder=lambda tokens, encoder_out: {"logits": logits},
    )

    def full_mask(tokens, coord_mask, alphabet):
        masked = tokens.clone()
        masked[coord_mask] = alphabet.mask_idx
        return masked, coord_mask

    runtime = Runtime(
        torch=torch,
        model=model,
        alphabet=Alphabet(),
        generator=Generator(),
        data_processor=DataProcessor(),
        full_mask=full_mask,
        surface_smooth=lambda points: points,
        device="cpu",
        logit_temperature=1.0,
    )

    probabilities, sequence = _infer(runtime, surface_builder, pdb_path, "A1")

    probabilities_array = np.asarray(probabilities)
    assert probabilities_array.shape == (2, 20)
    assert np.allclose(probabilities_array.sum(axis=-1), 1.0)
    assert probabilities_array.argmax(axis=-1).tolist() == [0, 1]
    assert sequence == "AC"


def test_task_requirements_include_surfdesign_runtime_and_experiment() -> None:
    requirements = task_requirements(
        {
            "mutator": {
                "SurfDesign": {"checkpoint": "default.ckpt"},
                "mutator_terms": {
                    "surface": {
                        "_type": "SurfDesign",
                        "checkpoint": "trained.ckpt",
                    }
                },
            },
            "stages": {"default": {"mutators": ["surface"]}},
        }
    )

    assert {
        "weight_path.surfdesign",
        "subprocess_envs.surfdesign.python_bin",
        "subprocess_envs.surfdesign.repo_path",
    } <= set(requirements["required_env_keys"])
    assert {
        "base_env_key": "weight_path.surfdesign",
        "path": ".hydra/config.yaml",
        "label": "surfdesign_config",
    } in requirements["required_env_repo_files"]
    assert {
        "base_env_key": "weight_path.surfdesign",
        "path": "checkpoints/trained.ckpt",
        "label": "surfdesign_checkpoint",
    } in requirements["required_env_repo_files"]


def test_configs_expose_surfdesign_runtime_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base = OmegaConf.load(repo_root / "config" / "base.yaml")
    env = OmegaConf.load(repo_root / "config" / "env.yaml")
    container_env = OmegaConf.load(repo_root / "config" / "env_container.yaml")
    third_party = OmegaConf.load(repo_root / "config" / "third_party_envs.yaml")
    OmegaConf.resolve(third_party)

    assert base.mutator.SurfDesign.daemon is True
    assert base.mutator.SurfDesign.checkpoint == "best.ckpt"
    assert "surfdesign" in env.weight_path
    assert "surfdesign" in env.subprocess_envs
    assert "surfdesign" in container_env.weight_path
    assert "surfdesign" in container_env.subprocess_envs
    assert container_env.subprocess_envs.surfdesign.python_bin == (
        "/envs/surfdesign/bin/python"
    )
    assert container_env.subprocess_envs.surfdesign.repo_path == "/ext/surfdesign"
    assert "surfdesign" in [entry.name for entry in third_party.envs]
