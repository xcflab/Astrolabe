from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for vendor_path in (
    REPO_ROOT / "vendor" / "alphafold3" / "src",
    REPO_ROOT / "vendor" / "chai_lab",
    REPO_ROOT / "vendor" / "protenix",
    REPO_ROOT / "vendor" / "boltz" / "src",
):
    vendor_str = str(vendor_path)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _af3_relative_encoding_deps():
    import jax
    import jax.numpy as jnp
    from alphafold3.model.network.featurization import create_relative_encoding

    jax.config.update("jax_platform_name", "cpu")
    return jnp, create_relative_encoding


def _chai_relative_sep_deps():
    import torch
    from chai_lab.chai1 import _apply_cyclic_relative_separation
    from chai_lab.data.features.generators.relative_sep import (
        RelativeSequenceSeparation,
        SMALL_SEP_BINS,
    )
    from chai_lab.utils.tensor_utils import string_to_tensorcode

    return (
        torch,
        _apply_cyclic_relative_separation,
        RelativeSequenceSeparation,
        SMALL_SEP_BINS,
        string_to_tensorcode,
    )


def test_af3_relative_encoding_wraps_selected_cyclic_chain() -> None:
    try:
        jnp, create_relative_encoding = _af3_relative_encoding_deps()
        seq_features = SimpleNamespace(
            token_index=jnp.arange(4, dtype=jnp.int32),
            residue_index=jnp.arange(1, 5, dtype=jnp.int32),
            asym_id=jnp.ones((4,), dtype=jnp.int32),
            entity_id=jnp.ones((4,), dtype=jnp.int32),
            sym_id=jnp.zeros((4,), dtype=jnp.int32),
        )
        rel = create_relative_encoding(
            seq_features=seq_features,
            max_relative_idx=4,
            max_relative_chain=2,
            cyclic_asym_ids=(1,),
        )
    except RuntimeError as exc:
        pytest.skip(f"JAX backend unavailable for AF3 unit test: {exc}")

    rel_pos = rel[..., : 2 * 4 + 2].argmax(axis=-1)
    assert int(rel_pos[0, 3]) == 5
    assert int(rel_pos[3, 0]) == 3


def test_af3_relative_encoding_applies_sparse_patch_without_full_chain_wrap() -> None:
    try:
        jnp, create_relative_encoding = _af3_relative_encoding_deps()
        seq_features = SimpleNamespace(
            token_index=jnp.arange(8, dtype=jnp.int32),
            residue_index=jnp.arange(1, 9, dtype=jnp.int32),
            asym_id=jnp.ones((8,), dtype=jnp.int32),
            entity_id=jnp.ones((8,), dtype=jnp.int32),
            sym_id=jnp.zeros((8,), dtype=jnp.int32),
        )
        rel = create_relative_encoding(
            seq_features=seq_features,
            max_relative_idx=8,
            max_relative_chain=2,
            cyclic_relative_patches=((1, 1, 6, 1), (1, 6, 1, -1)),
        )
    except RuntimeError as exc:
        pytest.skip(f"JAX backend unavailable for AF3 unit test: {exc}")

    rel_pos = rel[..., : 2 * 8 + 2].argmax(axis=-1)
    assert int(rel_pos[0, 5]) == 9
    assert int(rel_pos[5, 0]) == 7
    assert int(rel_pos[0, 7]) == 1


def test_chai_relative_sequence_separation_wraps_selected_subchain() -> None:
    (
        torch,
        _apply_cyclic_relative_separation,
        RelativeSequenceSeparation,
        SMALL_SEP_BINS,
        string_to_tensorcode,
    ) = _chai_relative_sep_deps()
    code = string_to_tensorcode("A", pad_to_length=4)
    subchain_ids = torch.stack([code, code, code, code], dim=0).unsqueeze(0)
    batch = {
        "inputs": {
            "token_residue_index": torch.arange(4, dtype=torch.long).unsqueeze(0),
            "token_asym_id": torch.ones((1, 4), dtype=torch.long),
            "token_exists_mask": torch.ones((1, 4), dtype=torch.bool),
            "subchain_id": subchain_ids,
        },
        "features": {},
    }
    rel_sep = RelativeSequenceSeparation().generate(batch)
    before = rel_sep.clone()
    batch["features"]["RelativeSequenceSeparation"] = rel_sep

    _apply_cyclic_relative_separation(batch, ("A",))

    sep_bins = torch.tensor(SMALL_SEP_BINS, dtype=torch.float32)
    expected = torch.searchsorted(sep_bins, torch.tensor(1.0) + 1e-4).item()
    assert int(before[0, 0, 3, 0]) != int(rel_sep[0, 0, 3, 0])
    assert int(rel_sep[0, 0, 3, 0]) == int(expected)


def test_chai_relative_sequence_separation_is_unchanged_without_cyclization() -> None:
    (
        torch,
        _apply_cyclic_relative_separation,
        RelativeSequenceSeparation,
        _SMALL_SEP_BINS,
        string_to_tensorcode,
    ) = _chai_relative_sep_deps()
    code = string_to_tensorcode("A", pad_to_length=4)
    subchain_ids = torch.stack([code, code, code, code], dim=0).unsqueeze(0)
    batch = {
        "inputs": {
            "token_residue_index": torch.arange(4, dtype=torch.long).unsqueeze(0),
            "token_asym_id": torch.ones((1, 4), dtype=torch.long),
            "token_exists_mask": torch.ones((1, 4), dtype=torch.bool),
            "subchain_id": subchain_ids,
        },
        "features": {},
    }
    rel_sep = RelativeSequenceSeparation().generate(batch)
    before = rel_sep.clone()
    batch["features"]["RelativeSequenceSeparation"] = rel_sep

    _apply_cyclic_relative_separation(batch, None)

    assert torch.equal(before, rel_sep)


def test_chai_relative_sequence_separation_applies_sparse_patch() -> None:
    (
        torch,
        _apply_cyclic_relative_separation,
        RelativeSequenceSeparation,
        SMALL_SEP_BINS,
        string_to_tensorcode,
    ) = _chai_relative_sep_deps()
    code = string_to_tensorcode("A", pad_to_length=4)
    subchain_ids = torch.stack([code] * 8, dim=0).unsqueeze(0)
    batch = {
        "inputs": {
            "token_residue_index": torch.arange(8, dtype=torch.long).unsqueeze(0),
            "token_asym_id": torch.ones((1, 8), dtype=torch.long),
            "token_exists_mask": torch.ones((1, 8), dtype=torch.bool),
            "subchain_id": subchain_ids,
        },
        "features": {},
    }
    rel_sep = RelativeSequenceSeparation().generate(batch)
    batch["features"]["RelativeSequenceSeparation"] = rel_sep

    _apply_cyclic_relative_separation(
        batch,
        None,
        [{"chain_id": "A", "offsets": [(1, 6, 1), (6, 1, -1)]}],
    )

    sep_bins = torch.tensor(SMALL_SEP_BINS, dtype=torch.float32)
    expected_pos = torch.searchsorted(sep_bins, torch.tensor(1.0) + 1e-4).item()
    expected_neg = torch.searchsorted(sep_bins, torch.tensor(-1.0) + 1e-4).item()
    assert int(rel_sep[0, 0, 5, 0]) == int(expected_pos)
    assert int(rel_sep[0, 5, 0, 0]) == int(expected_neg)


def test_protenix_relative_position_encoding_applies_sparse_patch() -> None:
    import torch
    from astrolabe.evaluators.wrappers.protenix import (
        _patch_protenix_cyclic_relative_position_encoding,
    )
    from protenix.model.modules.embedders import RelativePositionEncoding

    _patch_protenix_cyclic_relative_position_encoding()
    encoder = RelativePositionEncoding(r_max=8, s_max=2, c_z=8)
    feats = {
        "asym_id": torch.zeros((1, 8), dtype=torch.long),
        "residue_index": torch.arange(1, 9, dtype=torch.long).unsqueeze(0),
        "entity_id": torch.ones((1, 8), dtype=torch.long),
        "token_index": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "sym_id": torch.zeros((1, 8), dtype=torch.long),
        "cyclic_relative_patches": torch.tensor(
            [[0, 1, 6, 1], [0, 6, 1, -1]], dtype=torch.long
        ),
    }
    out = encoder.generate_relp(feats)
    rel_pos = out["relp"][0, :, :, : 2 * (8 + 1)].argmax(dim=-1)
    assert int(rel_pos[0, 5]) == 9
    assert int(rel_pos[5, 0]) == 7


def test_boltz_relative_position_encoding_applies_sparse_patch() -> None:
    import torch
    from boltz.model.modules.encoders import RelativePositionEncoder

    encoder = RelativePositionEncoder(token_z=8, r_max=8, s_max=2)
    encoder.linear_layer = torch.nn.Identity()
    feats = {
        "asym_id": torch.zeros((1, 8), dtype=torch.long),
        "residue_index": torch.arange(1, 9, dtype=torch.long).unsqueeze(0),
        "entity_id": torch.ones((1, 8), dtype=torch.long),
        "token_index": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "sym_id": torch.zeros((1, 8), dtype=torch.long),
        "cyclic_period": torch.zeros((1, 8), dtype=torch.long),
        "cyclic_relative_patches": torch.tensor(
            [[0, 1, 6, 1], [0, 6, 1, -1]], dtype=torch.long
        ),
    }
    rel = encoder(feats)
    rel_pos = rel[0, :, :, : 2 * 8 + 2].argmax(dim=-1)
    assert int(rel_pos[0, 5]) == 9
    assert int(rel_pos[5, 0]) == 7
