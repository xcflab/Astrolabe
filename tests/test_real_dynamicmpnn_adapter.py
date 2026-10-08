from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.dynamicmpnn import DynamicMPNNMutator
from astrolabe.mutators.ligandmpnn import GroupedLigandMPNNMutator, LigandMPNNMutator
from astrolabe.mutators.registry import MutatorRegistry
from astrolabe.mutators.wrappers.dynamicmpnn_run import DynamicMPNNRunner


def _raw_probabilities_backend(*, manifest, features):
    assert manifest["pdb_paths"] == ["open.pdb", "closed.pdb"]
    assert features["pdb_path_multi"]
    probs = np.zeros((1, 2, 20), dtype=float)
    probs[0, 0, 0] = 0.25
    probs[0, 0, 1] = 0.75
    probs[0, 1, 2] = 1.0
    return probs


def test_dynamicmpnn_runner_requires_in_process_backend(tmp_path: Path) -> None:
    runner = DynamicMPNNRunner({"output_dir": str(tmp_path)})
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    try:
        runner.design_sequences(pdb_path_multi=str(pdbs))
    except RuntimeError as exc:
        assert "Alex-Abrudan/DynamicMPNN" in str(exc)
        assert "backend_callable" in str(exc)
    else:
        raise AssertionError("Expected missing DynamicMPNN backend command to fail visibly")


def test_dynamicmpnn_runner_reads_raw_probabilities_in_process(tmp_path: Path) -> None:
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path),
            "backend_callable": _raw_probabilities_backend,
        }
    )
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    output = runner.design_sequences(pdb_path_multi=str(pdbs))

    assert output["dynamicmpnn_backend"] == "Alex-Abrudan/DynamicMPNN"
    assert output["sampling_probs"].shape == (1, 2, 20)
    np.testing.assert_allclose(output["sampling_probs"][0, 0, :2], [0.25, 0.75])
    assert output["dynamicmpnn_backend_mode"] == "in_process"
    assert "dynamicmpnn_probabilities" not in output
    assert "dynamicmpnn_manifest" not in output


def test_dynamicmpnn_runner_subprocess_reads_raw_probabilities(tmp_path: Path) -> None:
    command = (
        "python -c \"import numpy as np; "
        "p=np.zeros((1, 2, 20)); "
        "p[0, 0, 0]=0.25; p[0, 0, 1]=0.75; "
        "p[0, 1, 2]=1.0; "
        "np.save(__import__('sys').argv[1], p)\" {probabilities}"
    )
    runner = DynamicMPNNRunner({"output_dir": str(tmp_path), "command_template": command})
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    output = runner.design_sequences(pdb_path_multi=str(pdbs))

    assert output["dynamicmpnn_backend_mode"] == "subprocess"
    np.testing.assert_allclose(output["sampling_probs"][0, 0, :2], [0.25, 0.75])
    assert Path(output["dynamicmpnn_probabilities"]).exists()
    assert Path(output["dynamicmpnn_manifest"]).exists()


def test_dynamicmpnn_runner_rejects_sample_only_output(tmp_path: Path) -> None:
    command = "mkdir -p {out_dir}/samples && printf 'sequence\\nAC\\n' > {out_dir}/samples/samples.csv"
    runner = DynamicMPNNRunner({"output_dir": str(tmp_path), "command_template": command})
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    try:
        runner.design_sequences(pdb_path_multi=str(pdbs))
    except FileNotFoundError as exc:
        assert "raw probability file" in str(exc)
    else:
        raise AssertionError("Expected sample-only DynamicMPNN output to be rejected")


def test_mutator_registry_separates_dynamicmpnn_from_grouped_ligandmpnn(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    mutator_conf = OmegaConf.create(
        {
            "DynamicMPNN": {
                "backend_callable": f"{__name__}:_raw_probabilities_backend"
            },
            "mutator_terms": {"D": {"_type": "dynamicmpnn"}},
        }
    )

    mutators = build_mutators(mutator_conf, tmp_path)

    assert isinstance(mutators["D"], DynamicMPNNMutator)
    assert not isinstance(mutators["D"], LigandMPNNMutator)
    assert not isinstance(mutators["D"], GroupedLigandMPNNMutator)
    assert isinstance(mutators["D"].runner, DynamicMPNNRunner)
    assert MutatorRegistry.get("grouped_ligandmpnn") is not MutatorRegistry.get(
        "dynamicmpnn"
    )


def test_dynamicmpnn_mutator_returns_one_joint_output_to_alias_group(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw")
    global_config.init_const("segments", {})
    shared_segment = SimpleNamespace(is_fixed=False)
    global_config.segments["A_1"] = shared_segment
    global_config.segments["B_1"] = shared_segment
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                mutable_length=1,
                length=1,
                components=["A_1"],
            ),
            "B": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                mutable_length=1,
                length=1,
                components=["B_1"],
            ),
        },
    )
    global_config.set_variable("step", 1)

    class _JointRunner:
        def __init__(self) -> None:
            self.calls = []

        def design_sequences(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["pdb_path_multi"]
            paths = json.loads(Path(kwargs["pdb_path_multi"]).read_text(encoding="utf-8"))
            assert paths == ["open.pdb", "closed.pdb"]
            return {"sampling_probs": np.asarray([[[0.25, 0.75]]], dtype=float)}

    runner = _JointRunner()
    mutator = DynamicMPNNMutator(sequence_model_runner=runner)
    output = mutator.compute_mutation(
        {
            "A": {"pdb_path": "open.pdb", "fixed_residues": ""},
            "B": {"pdb_path": "closed.pdb", "fixed_residues": ""},
        }
    )

    assert len(runner.calls) == 1
    np.testing.assert_allclose(output["A"]["sampling_probs"], output["B"]["sampling_probs"])
