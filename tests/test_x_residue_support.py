from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.core.sequence_parsing import is_valid_protein_sequence
from astrolabe.utils.af3_compat import to_af3_input


def _monkeypatch_global_const(monkeypatch, name: str, value) -> None:
    monkeypatch.setattr(global_config, f"_{name}", value, raising=True)
    monkeypatch.setitem(global_config._initialized_constants, name, True)


def test_sequence_parsing_accepts_x_residue() -> None:
    assert is_valid_protein_sequence("ACDXEF") is True


def test_to_af3_input_accepts_x_residue() -> None:
    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDXEF",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    try:
        af3_inp = to_af3_input(inp)
    except RuntimeError as exc:
        raise pytest.skip(str(exc))
    assert len(getattr(af3_inp, "chains")) == 1
    assert getattr(getattr(af3_inp, "chains")[0], "sequence") == "ACDXEF"


def test_af3_featurization_accepts_all_x_chain_with_compat_patch() -> None:
    from alphafold3.common.folding_input import Input as Af3Input
    from alphafold3.common.folding_input import ProteinChain as Af3ProteinChain
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation

    from astrolabe.utils.af3_compat import allow_all_x_protein_chains_during_featurisation

    fold_input = Af3Input(
        chains=[
            Af3ProteinChain(
                id="A",
                sequence="XXXXX",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    with allow_all_x_protein_chains_during_featurisation(fold_input):
        features = featurisation.featurise_input(
            fold_input=fold_input,
            buckets=None,
            ccd=ccd,
            verbose=False,
        )
    assert len(features) == 1
    assert features[0]["aatype"].shape[0] == 5


def test_protenix_input_builder_writes_a3m_with_x(tmp_path: Path, monkeypatch) -> None:
    from astrolabe.evaluators.wrappers.protenix import from_af3_to_protenix

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(monkeypatch, "binding_site", None)
    _monkeypatch_global_const(monkeypatch, "protenix_constraint", None)
    monkeypatch.setattr(global_config, "_step", 0, raising=True)

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDXEF",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    data = from_af3_to_protenix(inp, "A")
    assert data["sequences"][0]["proteinChain"]["sequence"] == "ACDXEF"

    msa_dir = tmp_path / "sequences" / "protenix_msa" / "A" / "1"
    assert (msa_dir / "non_pairing.a3m").read_text(encoding="utf-8").strip().endswith(
        "ACDXEF"
    )
    assert (msa_dir / "pairing.a3m").read_text(encoding="utf-8").strip().endswith("ACDXEF")


def test_protenix_input_builder_writes_a3m_with_all_x(tmp_path: Path, monkeypatch) -> None:
    from astrolabe.evaluators.wrappers.protenix import from_af3_to_protenix

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(monkeypatch, "binding_site", None)
    _monkeypatch_global_const(monkeypatch, "protenix_constraint", None)
    monkeypatch.setattr(global_config, "_step", 0, raising=True)

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="XXXXX",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    data = from_af3_to_protenix(inp, "A")
    assert data["sequences"][0]["proteinChain"]["sequence"] == "XXXXX"

    msa_dir = tmp_path / "sequences" / "protenix_msa" / "A" / "1"
    assert (msa_dir / "non_pairing.a3m").read_text(encoding="utf-8").strip().endswith(
        "XXXXX"
    )
    assert (msa_dir / "pairing.a3m").read_text(encoding="utf-8").strip().endswith("XXXXX")


def test_protenix_msa_dir_reused_across_steps(tmp_path: Path, monkeypatch) -> None:
    from astrolabe.evaluators.wrappers.protenix import from_af3_to_protenix

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(monkeypatch, "binding_site", None)
    _monkeypatch_global_const(monkeypatch, "protenix_constraint", None)

    inp = Input(
        chains=[
            ProteinChain(
                id="A",
                sequence="ACDXEF",
                ptms=[],
                unpaired_msa=">query\nACDXEF\n",
                paired_msa=">query\nACDXEF\n",
                templates=[],
            )
        ],
        name="A",
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )

    monkeypatch.setattr(global_config, "_step", 0, raising=True)
    d0 = from_af3_to_protenix(inp, "A")
    p0 = d0["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"]

    monkeypatch.setattr(global_config, "_step", 1, raising=True)
    d1 = from_af3_to_protenix(inp, "A")
    p1 = d1["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"]

    assert p0 == p1
    assert Path(p0) == tmp_path / "sequences" / "protenix_msa" / "A" / "1"


def test_chai_prepare_input_writes_fasta_with_x(tmp_path: Path, monkeypatch) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(monkeypatch, "binding_site", None)
    monkeypatch.setattr(global_config, "_step", 0, raising=True)

    evaluator = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai1")
    inputs = {
        "A": Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDXEF",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                )
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }
    prepared = evaluator.prepare_input(inputs, runtime_config={})
    fasta_path = prepared["A"]["fasta_file"]
    assert fasta_path.read_text(encoding="utf-8").splitlines() == [
        ">protein|A",
        "ACDXEF",
    ]


def test_chai_prepare_input_reuses_msa_dir_across_steps(
    tmp_path: Path, monkeypatch
) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(monkeypatch, "binding_site", None)

    evaluator = Chai1Evaluator(model_config={"use_msa_server": False}, output_dir=tmp_path / "chai1")
    inputs = {
        "A": Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="ACDXEF",
                    ptms=[],
                    unpaired_msa=">query\nACDXEF\n>hit\nACD-EF\n",
                    paired_msa="",
                    templates=[],
                )
            ],
            name="A",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }

    monkeypatch.setattr(global_config, "_step", 0, raising=True)
    p0 = evaluator.prepare_input(inputs, runtime_config={})["A"]["msa_directory"]

    monkeypatch.setattr(global_config, "_step", 1, raising=True)
    p1 = evaluator.prepare_input(inputs, runtime_config={})["A"]["msa_directory"]

    assert p0 == p1
    assert p0 == tmp_path / "sequences" / "chai1_msas" / "A"
    assert len(list(p0.glob("*.aligned.pqt"))) == 1


def test_evaluator_x_adapter_preserves_length_and_masks_chai_esm() -> None:
    from astrolabe.evaluators.x_masking import (
        chai_esm_token_count,
        mask_unknown_residues_for_chai_esm,
        preserve_unknown_residue_sequence,
    )

    seq = "ACDXEF"
    for evaluator in ("alphafold3", "boltz", "protenix", "chai1"):
        assert preserve_unknown_residue_sequence(seq, evaluator=evaluator) == seq

    masked = mask_unknown_residues_for_chai_esm(seq)
    assert masked == "ACD<mask>EF"
    assert chai_esm_token_count(masked) == len(seq)


def test_chai_masked_esm_embedding_context_uses_mask_token_for_x(monkeypatch) -> None:
    from chai_lab.data.dataset.embeddings import esm as esm_mod
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from astrolabe.evaluators.x_masking import (
        build_chai_masked_esm_embedding_context,
    )

    captured: dict[str, torch.Tensor] = {}

    @contextmanager
    def fake_esm_model(device):
        class FakeModel:
            def __call__(self, *, tokens):
                captured["tokens"] = tokens.detach().cpu()
                return torch.zeros(
                    (1, tokens.shape[1], 4),
                    dtype=torch.float32,
                    device=tokens.device,
                )

        yield FakeModel()

    monkeypatch.setattr(esm_mod, "esm_model", fake_esm_model)

    chain = SimpleNamespace(
        entity_data=SimpleNamespace(entity_type=EntityType.PROTEIN, sequence="AXC"),
        structure_context=SimpleNamespace(
            num_tokens=3,
            token_residue_index=torch.arange(3, dtype=torch.long),
        ),
    )

    ctx = build_chai_masked_esm_embedding_context([chain], torch.device("cpu"))

    assert captured["tokens"].tolist() == [[0, 5, 32, 23, 2]]
    assert tuple(ctx.esm_embeddings.shape) == (3, 4)


def test_chai_masked_esm_embedding_context_masks_all_x_sequence(monkeypatch) -> None:
    from chai_lab.data.dataset.embeddings import esm as esm_mod
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from astrolabe.evaluators.x_masking import (
        build_chai_masked_esm_embedding_context,
    )

    captured: dict[str, torch.Tensor] = {}

    @contextmanager
    def fake_esm_model(device):
        class FakeModel:
            def __call__(self, *, tokens):
                captured["tokens"] = tokens.detach().cpu()
                return torch.zeros(
                    (1, tokens.shape[1], 2),
                    dtype=torch.float32,
                    device=tokens.device,
                )

        yield FakeModel()

    monkeypatch.setattr(esm_mod, "esm_model", fake_esm_model)

    chain = SimpleNamespace(
        entity_data=SimpleNamespace(entity_type=EntityType.PROTEIN, sequence="XXXXX"),
        structure_context=SimpleNamespace(
            num_tokens=5,
            token_residue_index=torch.arange(5, dtype=torch.long),
        ),
    )

    ctx = build_chai_masked_esm_embedding_context([chain], torch.device("cpu"))

    assert captured["tokens"].tolist() == [[0, 32, 32, 32, 32, 32, 2]]
    assert tuple(ctx.esm_embeddings.shape) == (5, 2)

def test_chai_prepare_input_preserves_x_with_binding_site_and_target_msa(
    tmp_path: Path, monkeypatch
) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    target_sequence = "MNNNST"

    _monkeypatch_global_const(monkeypatch, "out_dir", tmp_path)
    _monkeypatch_global_const(
        monkeypatch,
        "binding_site",
        {"AB": {"binder": "A:2-3", "target": "B:2"}},
    )
    monkeypatch.setattr(global_config, "_step", 0, raising=True)

    evaluator = Chai1Evaluator(
        model_config={"use_msa_server": False},
        output_dir=tmp_path / "chai1",
    )
    inputs = {
        "AB": Input(
            chains=[
                ProteinChain(
                    id="A",
                    sequence="AXXXAXXXAA",
                    ptms=[],
                    unpaired_msa="",
                    paired_msa="",
                    templates=[],
                ),
                ProteinChain(
                    id="B",
                    sequence=target_sequence,
                    ptms=[],
                    unpaired_msa=f">query\n{target_sequence}\n>hit\nMN-NST\n",
                    paired_msa="",
                    templates=[],
                ),
            ],
            name="AB",
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )
    }

    prepared = evaluator.prepare_input(inputs, runtime_config={})["AB"]

    assert prepared["fasta_file"].read_text(encoding="utf-8").splitlines() == [
        ">protein|A",
        "AXXXAXXXAA",
        ">protein|B",
        target_sequence,
    ]

    constraint_path = prepared["constraint_path"]
    assert constraint_path is not None
    constraint_text = constraint_path.read_text(encoding="utf-8")
    assert "connection_type" in constraint_text
    assert "pocket" in constraint_text
    assert "contact" in constraint_text
    assert "X2" in constraint_text
    assert "X3" in constraint_text

    msa_directory = prepared["msa_directory"]
    assert msa_directory == tmp_path / "sequences" / "chai1_msas" / "AB"
    assert len(list(msa_directory.glob("*.aligned.pqt"))) == 1
