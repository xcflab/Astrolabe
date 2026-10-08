from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.surfpro import SurfProMutator
from astrolabe.mutators.wrappers.surfpro_daemon import (
    Runtime,
    SurfaceBuilder,
    _decode_sequence,
    _encode_surface,
    _fixed_flat_indices,
)
from astrolabe.mutators.wrappers.surfpro_runner import SurfProRunner
from astrolabe.runtime_requirements import task_requirements


def _executable(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


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
    repo_path = tmp_path / "surfpro"
    (repo_path / "fairseq" / "models").mkdir(parents=True)
    (repo_path / "fairseq" / "models" / "surface_protein_model.py").write_text(
        "# stub\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_text("stub\n", encoding="utf-8")
    daemon_script = _fake_daemon(tmp_path / "fake_surfpro_daemon.py")
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    return (
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_path),
            "checkpoint": str(checkpoint),
            "msms_bin": str(_executable(tmp_path / "msms")),
            "pdb_to_xyzrn_bin": str(_executable(tmp_path / "pdb_to_xyzrn")),
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
    runner = SurfProRunner(config)
    cif_path = pdb_path.with_suffix(".cif")
    cif_path.write_text("data_stub\n", encoding="utf-8")

    first = runner.design_sequences(pdb_path=pdb_path, fixed_residues="A1")
    second = runner.design_sequences(pdb_path=cif_path)

    assert first["sampling_probs"].shape == (1, 2, 20)
    assert np.allclose(first["sampling_probs"].sum(axis=-1), 1.0)
    assert first["surfpro_daemon_pid"] == second["surfpro_daemon_pid"]
    assert first["surfpro_model_load_count"] == 1
    assert second["surfpro_model_load_count"] == 1
    assert [first["surfpro_request_count"], second["surfpro_request_count"]] == [1, 2]

    runner.close()


def test_runner_rejects_per_call_loading_and_cpu(tmp_path: Path) -> None:
    config, _pdb_path = _runner_config(tmp_path)
    config["daemon"] = False
    with pytest.raises(ValueError, match="persistent daemon"):
        SurfProRunner(config)

    config["daemon"] = True
    config["device"] = "cpu"
    with pytest.raises(ValueError, match="hardcodes CUDA"):
        SurfProRunner(config)


def test_runner_requires_msms_toolchain(tmp_path: Path) -> None:
    config, _pdb_path = _runner_config(tmp_path)
    Path(config["msms_bin"]).unlink()

    with pytest.raises(FileNotFoundError, match="msms_bin"):
        SurfProRunner(config)


def test_runner_uses_isolated_surface_dependency_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/ambient")
    monkeypatch.setenv("APPTAINERENV_PYTHONPATH", "/ambient-container")
    monkeypatch.setenv(
        "GH_SURFPRO_PYTHONPATH",
        "/ext/surfpro:/ext/sru:/ext/protein_sequence_models",
    )

    env = SurfProRunner._subprocess_env()

    assert env["PYTHONPATH"] == (
        "/ext/surfpro:/ext/sru:/ext/protein_sequence_models"
    )
    assert "APPTAINERENV_PYTHONPATH" not in env


def test_runner_resolves_relative_checkpoint_from_weight_root(tmp_path: Path) -> None:
    config, _pdb_path = _runner_config(tmp_path)
    checkpoint = Path(config["checkpoint"])
    config["checkpoint_dir"] = str(checkpoint.parent)
    config["checkpoint"] = checkpoint.name

    runner = SurfProRunner(config)

    assert runner.checkpoint == checkpoint
    runner.close()


def test_runner_does_not_restart_dead_daemon(tmp_path: Path) -> None:
    config, pdb_path = _runner_config(tmp_path)
    runner = SurfProRunner(config)
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
    mutator = SurfProMutator(sequence_model_runner=runner)

    mutator.shutdown()

    assert runner.closed is True


def test_registry_builds_surfpro_mutator(tmp_path: Path) -> None:
    runner_config, _pdb_path = _runner_config(tmp_path)
    mutator_config = OmegaConf.create(
        {
            "SurfPro": runner_config,
            "mutator_terms": {
                "surface": {
                    "_type": "SurfPro",
                    "position_selector": "plddt < 80",
                }
            },
        }
    )
    global_config.reset()

    mutators = build_mutators(mutator_config, tmp_path / "raw")

    assert isinstance(mutators["surface"], SurfProMutator)
    assert mutators["surface"].position_selector == "plddt < 80"
    mutators["surface"].shutdown()
    global_config.reset()


def test_protein_input_normalizes_global_residue_ordinals(tmp_path: Path) -> None:
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
    builder = object.__new__(SurfaceBuilder)
    builder.work_dir = tmp_path

    protein = builder._protein_input(pdb_path)
    atom_lines = protein.pdb_path.read_text(encoding="utf-8").splitlines()[:-1]

    assert protein.sequence == "AC"
    assert protein.chain_order == ["A", "B"]
    assert protein.chain_lengths == {"A": 1, "B": 1}
    assert [line[22:26] for line in atom_lines] == ["   1", "   1", "   2", "   2"]


def test_open_dde_mmcif_can_be_converted_for_surfpro(tmp_path: Path) -> None:
    builder = object.__new__(SurfaceBuilder)
    builder.work_dir = tmp_path
    source = Path(__file__).resolve().parents[1] / "examples" / "input" / "9TP_cid.cif"

    converted = builder._as_pdb(source)
    protein = builder._protein_input(converted)

    assert converted.suffix == ".pdb"
    assert protein.sequence
    assert protein.chain_order
    assert sum(protein.chain_lengths.values()) == len(protein.sequence)


def test_surface_features_are_normalized_and_sorted_by_residue(tmp_path: Path) -> None:
    builder = object.__new__(SurfaceBuilder)
    builder.max_surface_vertices = 5000
    vertices = [
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    labels = ["CA_CYS_2", "CA_ALA_1", "CA_CYS_2"]
    builder.prepare_surface = SimpleNamespace(
        extract_vertix=lambda _path: (vertices, labels)
    )

    coordinates, aa_indices = builder._prepare_features(tmp_path / "surface.vert")

    assert coordinates.shape == (3, 3)
    assert np.allclose(coordinates[:, 0], [-0.5, 0.5, 0.0])
    assert aa_indices.tolist() == [0, 1, 1]


def test_fixed_residue_indices_follow_chain_ordinals() -> None:
    assert _fixed_flat_indices("A2 B1", ["A", "B"], {"A": 2, "B": 3}) == {1, 2}
    with pytest.raises(ValueError, match="out of range"):
        _fixed_flat_indices("B4", ["A", "B"], {"A": 2, "B": 3})


def test_autoregressive_decode_returns_standard_aa_probabilities() -> None:
    torch = pytest.importorskip("torch")

    class Dictionary:
        tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["<special>", "<bos>"]

        @classmethod
        def get_idx(cls, token):
            return cls.tokens.index(token)

        @classmethod
        def get_tok(cls, index):
            return cls.tokens[index]

        @classmethod
        def bos(cls):
            return 21

    class Decoder:
        @staticmethod
        def inference_forward(prev_tokens, **_kwargs):
            logits = torch.zeros((1, prev_tokens.shape[1], 22))
            logits[0, -1, 1] = 5.0
            logits[0, -1, 20] = 10.0
            return logits, {}

    runtime = Runtime(
        torch=torch,
        model=SimpleNamespace(decoder=Decoder()),
        dictionary=Dictionary(),
        get_edges_batch=None,
        device="cpu",
        logit_temperature=1.0,
    )

    probabilities, sequence = _decode_sequence(
        runtime,
        encoder_out={},
        src_lengths=torch.tensor([2]),
        native_sequence="AC",
        fixed_indices={0},
    )

    probabilities_array = np.asarray(probabilities)
    assert probabilities_array.shape == (2, 20)
    assert np.allclose(probabilities_array.sum(axis=-1), 1.0)
    assert probabilities_array.argmax(axis=-1).tolist() == [1, 1]
    assert sequence == "AC"


def test_surface_encoder_uses_surfpro_four_argument_edge_builder() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def get_edges_batch(n_nodes, batch_size, coordinates, k):
        calls.append((n_nodes, batch_size, tuple(coordinates.shape), k))
        return [torch.tensor([0, 1]), torch.tensor([1, 0])]

    class Encoder:
        @staticmethod
        def __call__(aa_features, coordinates, edges, _edge_attr, k):
            assert aa_features.shape == (1, 3, 2)
            assert len(edges) == 2
            assert k == 2
            return torch.zeros((3, 4)), coordinates.reshape(3, 3)

    model = SimpleNamespace(
        surf_aa_features=torch.zeros((20, 2)),
        k=30,
        encoder=Encoder(),
        pooling_encoder=lambda values: values[-1],
    )
    runtime = Runtime(
        torch=torch,
        model=model,
        dictionary=None,
        get_edges_batch=get_edges_batch,
        device="cpu",
        logit_temperature=1.0,
    )

    encoder_out, src_lengths = _encode_surface(
        runtime,
        np.zeros((3, 3), dtype=np.float32),
        np.array([0, 1, 2], dtype=np.int64),
    )

    assert calls == [(3, 1, (1, 3, 3), 2)]
    assert encoder_out["encoder_out"][0].shape == (3, 1, 4)
    assert src_lengths.tolist() == [3]


def test_task_requirements_include_surfpro_runtime_and_checkpoint() -> None:
    requirements = task_requirements(
        {
            "mutator": {
                "SurfPro": {"checkpoint": "default.pt"},
                "mutator_terms": {
                    "surface": {
                        "_type": "SurfPro",
                        "checkpoint": "trained.pt",
                    }
                },
            },
            "stages": {"default": {"mutators": ["surface"]}},
        }
    )

    assert {
        "weight_path.surfpro",
        "subprocess_envs.surfpro.python_bin",
        "subprocess_envs.surfpro.repo_path",
        "subprocess_envs.surfpro.msms_bin",
        "subprocess_envs.surfpro.pdb_to_xyzrn_bin",
    } <= set(requirements["required_env_keys"])
    assert "vendor/sru/sru/__init__.py" in requirements["required_repo_files"]
    assert (
        "vendor/protein_sequence_models/sequence_models/structure.py"
        in requirements["required_repo_files"]
    )
    assert {
        "base_env_key": "subprocess_envs.surfpro.repo_path",
        "path": "fairseq/models/surface_protein_model.py",
        "label": "surfpro_source",
    } in requirements["required_env_repo_files"]
    assert {
        "base_env_key": "weight_path.surfpro",
        "path": "trained.pt",
        "label": "surfpro_checkpoint",
    } in requirements["required_env_repo_files"]


def test_configs_expose_surfpro_runtime_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base = OmegaConf.load(repo_root / "config" / "base.yaml")
    env = OmegaConf.load(repo_root / "config" / "env.yaml")
    container_env = OmegaConf.load(repo_root / "config" / "env_container.yaml")
    third_party = OmegaConf.load(repo_root / "config" / "third_party_envs.yaml")
    OmegaConf.resolve(third_party)

    assert base.mutator.SurfPro.daemon is True
    assert base.mutator.SurfPro.checkpoint == "checkpoint_best.pt"
    assert base.mutator.SurfPro.max_surface_vertices == 5000
    assert "surfpro" in env.weight_path
    assert "surfpro" in env.subprocess_envs
    assert "surfpro" in container_env.weight_path
    assert "surfpro" in container_env.subprocess_envs
    assert container_env.subprocess_envs.surfpro.python_bin == "/envs/surfpro/bin/python"
    assert container_env.subprocess_envs.surfpro.repo_path == "/ext/surfpro"
    assert container_env.subprocess_envs.surfpro.msms_bin == (
        "/tools/surfpro/msms/msms"
    )
    assert "surfpro" in [entry.name for entry in third_party.envs]
