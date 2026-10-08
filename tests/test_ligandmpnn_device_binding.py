from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN
from astrolabe.mutators.wrappers.ligandmpnn_utils import restype_str_to_int


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        omit_AA="",
        bias_AA="",
        temperature=0.1,
        verbose=False,
        ligand_mpnn_use_atom_context=True,
        ligand_mpnn_cutoff_for_score=8.0,
        parse_atoms_with_zero_occupancy=False,
        number_of_batches=1,
        zero_indexed=False,
        fasta_seq_separation=":",
        save_stats=False,
        file_ending="",
        model_type="ligand_mpnn",
        bias_mode="additive",
        ligand_mpnn_use_side_chain_context=True,
        seed=7,
        bias_AA_per_residue_multi="",
        bias_AA_per_residue="",
        omit_AA_per_residue_multi="",
        omit_AA_per_residue="",
        out_folder=str(tmp_path),
        checkpoint_protein_mpnn="unused.pt",
        checkpoint_ligand_mpnn="ligand.pt",
        checkpoint_per_residue_label_membrane_mpnn="unused.pt",
        checkpoint_global_label_membrane_mpnn="unused.pt",
        checkpoint_soluble_mpnn="unused.pt",
        design_pattern=None,
        design_pattern_range=None,
    )


def test_ligandmpnn_binds_explicit_cuda_ordinal(monkeypatch, tmp_path: Path) -> None:
    created: dict[str, object] = {}

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, device):
            created["to_device"] = device
            return self

        def eval(self):
            return self

    def _fake_model_ctor(**kwargs):
        created["ctor_device"] = kwargs["device"]
        return _FakeModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "atom_context_num": 16,
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        _fake_model_ctor,
    )

    model = LigandMPNN(_args(tmp_path))

    assert model.device == torch.device("cuda:1")
    assert created["ctor_device"] == torch.device("cuda:1")
    assert created["to_device"] == torch.device("cuda:1")


def test_soluble_mpnn_disables_saved_side_chain_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: dict[str, object] = {}

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, device):
            created["to_device"] = device
            return self

        def eval(self):
            return self

    def _fake_model_ctor(**kwargs):
        created["side_chain_context"] = kwargs["ligand_mpnn_use_side_chain_context"]
        return _FakeModel()

    args = _args(tmp_path)
    args.model_type = "soluble_mpnn"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        _fake_model_ctor,
    )

    model = LigandMPNN(args)

    assert model.ligand_mpnn_use_side_chain_context == 0
    assert created["side_chain_context"] == 0


def test_ligandmpnn_accepts_single_bias_file_for_single_pdb(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _StopAfterBias(RuntimeError):
        pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "atom_context_num": 16,
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        lambda **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.parse_PDB",
        lambda *_args, **_kwargs: (
            {
                "R_idx": torch.tensor([10]),
                "chain_letters": ["A"],
            },
            None,
            None,
            [""],
            None,
        ),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_StopAfterBias()),
    )

    pdb_path = tmp_path / "example.pdb"
    pdb_path.write_text("ATOM\nEND\n", encoding="utf-8")
    bias_path = tmp_path / "bias.json"
    bias_path.write_text(json.dumps({"A10": {"S": 1.5}}), encoding="utf-8")

    args = _args(tmp_path)
    args.bias_AA_per_residue = str(bias_path)
    model = LigandMPNN(args)

    with pytest.raises(_StopAfterBias):
        model.design_sequences(
            pdb_path=str(pdb_path),
            fixed_residues="",
            redesigned_residues="",
        )


def test_ligandmpnn_accepts_in_memory_bias_payload_for_single_pdb(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _StopAfterBias(RuntimeError):
        pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "atom_context_num": 16,
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        lambda **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.parse_PDB",
        lambda *_args, **_kwargs: (
            {
                "R_idx": torch.tensor([10]),
                "chain_letters": ["A"],
            },
            None,
            None,
            [""],
            None,
        ),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_StopAfterBias()),
    )

    pdb_path = tmp_path / "example.pdb"
    pdb_path.write_text("ATOM\nEND\n", encoding="utf-8")

    args = _args(tmp_path)
    args.bias_AA_per_residue = {"A10": {"S": 1.5}}
    model = LigandMPNN(args)

    with pytest.raises(_StopAfterBias):
        model.design_sequences(
            pdb_path=str(pdb_path),
            fixed_residues="",
            redesigned_residues="",
        )


def test_ligandmpnn_coerces_global_transmembrane_label_from_string(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _StopAfterMembrane(RuntimeError):
        pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "atom_context_num": 1,
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        lambda **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.parse_PDB",
        lambda *_args, **_kwargs: (
            {
                "R_idx": torch.tensor([10]),
                "chain_letters": ["A"],
            },
            None,
            None,
            [""],
            None,
        ),
    )

    def _fake_featurize(protein_dict, *args, **kwargs):
        captured["labels"] = protein_dict["membrane_per_residue_labels"].clone()
        raise _StopAfterMembrane()

    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.featurize",
        _fake_featurize,
    )

    pdb_path = tmp_path / "example.pdb"
    pdb_path.write_text("ATOM\nEND\n", encoding="utf-8")

    args = _args(tmp_path)
    args.model_type = "global_label_membrane_mpnn"
    model = LigandMPNN(args)

    with pytest.raises(_StopAfterMembrane):
        model.design_sequences(
            pdb_path=str(pdb_path),
            fixed_residues="",
            redesigned_residues="",
            global_transmembrane_label="1",
        )

    assert torch.equal(captured["labels"], torch.tensor([1]))


def test_ligandmpnn_converts_multiplicative_bias_per_residue_with_runtime_temperature(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    class _StopAfterBias(RuntimeError):
        pass

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def sample(self, feature_dict):
            captured["bias"] = feature_dict["bias"][0].detach().cpu()
            raise _StopAfterBias()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda *_args, **_kwargs: {
            "atom_context_num": 16,
            "num_edges": 30,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        lambda **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.parse_PDB",
        lambda *_args, **_kwargs: (
            {
                "R_idx": torch.tensor([10]),
                "chain_letters": ["A"],
            },
            None,
            None,
            [""],
            None,
        ),
    )

    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.featurize",
        lambda *_args, **_kwargs: {
            "X": torch.zeros((1, 1, 4, 3)),
            "mask": torch.ones((1, 1), dtype=torch.float32),
            "mask_XY": torch.ones((1, 1), dtype=torch.float32),
            "S": torch.zeros((1, 1), dtype=torch.int64),
        },
    )

    pdb_path = tmp_path / "example.pdb"
    pdb_path.write_text("ATOM\nEND\n", encoding="utf-8")

    args = _args(tmp_path)
    args.bias_mode = "multiplicative"
    args.temperature = 0.3
    args.bias_AA_per_residue = OmegaConf.create(
        {
            "A10": {
                "S": math.exp(1.0 / 0.3),
                "V": math.exp(-0.3 / 0.3),
                "K": math.exp(-0.6 / 0.3),
            }
        }
    )
    model = LigandMPNN(args)

    with pytest.raises(_StopAfterBias):
        model.design_sequences(
            pdb_path=str(pdb_path),
            fixed_residues="",
            redesigned_residues="",
        )

    bias = captured["bias"]
    assert bias.shape == (1, 21)
    assert bias[0, restype_str_to_int["S"]].item() == pytest.approx(1.0, abs=1e-6)
    assert bias[0, restype_str_to_int["V"]].item() == pytest.approx(-0.3, abs=1e-6)
    assert bias[0, restype_str_to_int["K"]].item() == pytest.approx(-0.6, abs=1e-6)


@pytest.mark.parametrize(
    ("model_type", "expected_checkpoint"),
    [
        ("per_residue_label_membrane_mpnn", "per_residue.pt"),
        ("global_label_membrane_mpnn", "global_label.pt"),
    ],
)
def test_ligandmpnn_membrane_model_type_selects_matching_checkpoint(
    monkeypatch,
    tmp_path: Path,
    model_type: str,
    expected_checkpoint: str,
) -> None:
    created: dict[str, object] = {}

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.torch.load",
        lambda checkpoint_path, **_kwargs: {
            "atom_context_num": 16,
            "num_edges": 30,
            "model_state_dict": {},
            "checkpoint_path": checkpoint_path,
        },
    )

    def _fake_model_ctor(**kwargs):
        created["ctor_kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(
        "astrolabe.mutators.wrappers.ligandmpnn_run.ProteinMPNN",
        _fake_model_ctor,
    )

    args = _args(tmp_path)
    args.model_type = model_type
    args.checkpoint_per_residue_label_membrane_mpnn = "per_residue.pt"
    args.checkpoint_global_label_membrane_mpnn = "global_label.pt"
    model = LigandMPNN(args)

    assert model.checkpoint_path == expected_checkpoint
    assert created["ctor_kwargs"]["model_type"] == model_type
