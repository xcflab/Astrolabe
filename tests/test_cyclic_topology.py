from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for vendor_path in (
    REPO_ROOT / "vendor" / "alphafold2",
    REPO_ROOT / "vendor" / "alphafold3" / "src",
):
    vendor_str = str(vendor_path)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

from astrolabe.core import global_config
from astrolabe.core.cyclization import (
    CyclicRelativeOffset,
    CyclicRelativePatch,
    ResolvedCyclizationAtomRef,
    ResolvedCyclizationBond,
    build_cyclic_offset_matrix,
    compute_highfold_cycpoem_offset,
    derive_highfold_cycpoem_relative_patches,
    derive_global_cyclic_relative_patches,
    derive_cyclic_relative_patches,
    derive_cyclic_chain_ids,
)
from astrolabe.core.folding_input import Input, ProteinChain


def _protein_chain(*, chain_id: str, sequence: str) -> ProteinChain:
    return ProteinChain(
        id=chain_id,
        sequence=sequence,
        ptms=[],
        unpaired_msa="",
        paired_msa="",
        templates=[],
    )


def test_derive_cyclic_chain_ids_marks_all_matching_occurrences() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=1, atom="N"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=5, atom="C"),
        ),
    )
    assert derive_cyclic_chain_ids(
        resolved,
        oligo_name="ABA",
        protomer_lengths={"A": 5, "B": 7},
    ) == ("A", "C")


def test_derive_cyclic_chain_ids_ignores_noncanonical_bonds() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=2, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=7, atom="SG"),
        ),
    )
    assert derive_cyclic_chain_ids(
        resolved,
        oligo_name="A",
        protomer_lengths={"A": 7},
    ) == ()


def test_build_cyclic_offset_matrix_wraps_intra_chain_offsets() -> None:
    cyclic_offset = build_cyclic_offset_matrix(
        np.arange(4, dtype=np.int32),
        chain_lengths=[4],
        cyclic_chain_ids=("A",),
        chain_ids=("A",),
    )
    assert int(cyclic_offset[0, 3]) == 1
    assert int(cyclic_offset[3, 0]) == -1


def test_build_cyclic_offset_matrix_no_cyclic_is_identity_offset() -> None:
    offset = build_cyclic_offset_matrix(
        np.arange(4, dtype=np.int32),
        chain_lengths=[4],
        cyclic_chain_ids=(),
        chain_ids=("A",),
    )
    expected = np.arange(4, dtype=np.int32)[:, None] - np.arange(
        4, dtype=np.int32
    )[None, :]
    assert np.array_equal(offset, expected)


def test_build_cyclic_offset_matrix_applies_sparse_relative_patch_only_to_window_pairs() -> None:
    patches = (
        CyclicRelativePatch(
            chain_id="A",
            offsets=(
                CyclicRelativeOffset(residue_i=2, residue_j=7, offset=1),
                CyclicRelativeOffset(residue_i=7, residue_j=2, offset=-1),
                CyclicRelativeOffset(residue_i=3, residue_j=7, offset=2),
                CyclicRelativeOffset(residue_i=7, residue_j=3, offset=-2),
            ),
        ),
    )
    offset = build_cyclic_offset_matrix(
        np.arange(8, dtype=np.int32),
        chain_lengths=[8],
        cyclic_chain_ids=(),
        chain_ids=("A",),
        cyclic_relative_patches=patches,
    )
    assert int(offset[1, 6]) == 1
    assert int(offset[6, 1]) == -1
    assert int(offset[2, 6]) == 2
    assert int(offset[6, 2]) == -2
    assert int(offset[0, 7]) == -7
    assert int(offset[3, 4]) == -1


def test_build_cyclic_offset_matrix_rejects_conflicting_sparse_patch_cells() -> None:
    patches = (
        CyclicRelativePatch(
            chain_id="A",
            offsets=(CyclicRelativeOffset(residue_i=2, residue_j=7, offset=1),),
        ),
        CyclicRelativePatch(
            chain_id="A",
            offsets=(CyclicRelativeOffset(residue_i=2, residue_j=7, offset=2),),
        ),
    )
    with pytest.raises(ValueError, match="matrix cell conflict"):
        build_cyclic_offset_matrix(
            np.arange(8, dtype=np.int32),
            chain_lengths=[8],
            cyclic_chain_ids=(),
            chain_ids=("A",),
            cyclic_relative_patches=patches,
        )


def test_compute_highfold_cycpoem_offset_matches_reference_shortest_path_matrix() -> None:
    offset = compute_highfold_cycpoem_offset(8, [(2, 7), (4, 8)])
    expected = np.array(
        [
            [0, -1, -2, -2, -3, -3, -2, -1],
            [1, 0, -1, -2, -3, -2, -1, -2],
            [2, 1, 0, -1, -2, -3, -2, -2],
            [2, 2, 1, 0, -1, -2, -2, -1],
            [3, 3, 2, 1, 0, -1, -2, -2],
            [3, 2, 3, 2, 1, 0, -1, -2],
            [2, 1, 2, 2, 2, 1, 0, -1],
            [1, 2, 2, 1, 2, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(offset, expected)


def test_derive_highfold_cycpoem_relative_patches_emit_full_chain_matrix() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=2, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=7, atom="SG"),
        ),
    )
    patches = derive_highfold_cycpoem_relative_patches(
        resolved,
        oligo_name="A",
        protomer_lengths={"A": 8},
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch.chain_id == "A"
    assert len(patch.offsets) == 64
    by_pair = {
        (entry.residue_i, entry.residue_j): entry.offset for entry in patch.offsets
    }
    assert by_pair[(1, 8)] == -1
    assert by_pair[(8, 1)] == 1
    assert by_pair[(2, 7)] == -1
    assert by_pair[(7, 2)] == 1
    assert by_pair[(4, 5)] == -1


def test_derive_global_cyclic_relative_patches_emit_full_cyclized_interval() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=2, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=7, atom="SG"),
        ),
    )
    patches = derive_global_cyclic_relative_patches(
        resolved,
        oligo_name="A",
        protomer_lengths={"A": 8},
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch.chain_id == "A"
    assert len(patch.offsets) == 36
    by_pair = {
        (entry.residue_i, entry.residue_j): entry.offset for entry in patch.offsets
    }
    assert by_pair[(2, 7)] == 1
    assert by_pair[(7, 2)] == -1
    assert by_pair[(2, 5)] == -3
    assert by_pair[(5, 2)] == 3
    assert (1, 8) not in by_pair


def test_derive_cyclic_relative_patches_allows_overlapping_intervals_when_endpoint_groups_disjoint() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=1, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=6, atom="SG"),
            offset_window=2,
        ),
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=3, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=8, atom="SG"),
            offset_window=2,
        ),
    )
    patches = derive_cyclic_relative_patches(
        resolved,
        oligo_name="A",
        protomer_lengths={"A": 8},
    )
    assert len(patches) == 2


def test_derive_cyclic_relative_patches_allows_shared_endpoint_groups_without_cell_conflict() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=1, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=6, atom="SG"),
            offset_window=3,
        ),
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=3, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=8, atom="SG"),
            offset_window=2,
        ),
    )
    patches = derive_cyclic_relative_patches(
        resolved,
        oligo_name="A",
        protomer_lengths={"A": 8},
    )
    assert len(patches) == 2


def test_derive_cyclic_relative_patches_rejects_matrix_cell_conflict() -> None:
    resolved = (
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=1, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=4, atom="SG"),
            offset_window=3,
        ),
        ResolvedCyclizationBond(
            a=ResolvedCyclizationAtomRef(proto="A", pos=1, atom="SG"),
            b=ResolvedCyclizationAtomRef(proto="A", pos=5, atom="SG"),
            offset_window=3,
        ),
    )
    with pytest.raises(ValueError, match="matrix cell conflict"):
        derive_cyclic_relative_patches(
            resolved,
            oligo_name="A",
            protomer_lengths={"A": 8},
        )


def test_boltz_wrapper_marks_cyclic_chain() -> None:
    from astrolabe.evaluators.wrappers.boltz import from_af3_to_boltz

    global_config.reset()
    global_config.init_const("binding_site", None)
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ACDE")],
        rng_seeds=[0],
        cyclic_chain_ids=("A",),
        bonded_atom_pairs=[(("A", 1, "N"), ("A", 4, "C"))],
        user_ccd=None,
    )
    data, _chain_map = from_af3_to_boltz(inp)
    assert data["sequences"][0]["protein"]["cyclic"] is True


def test_boltz_wrapper_omits_cyclic_flag_without_cyclization() -> None:
    from astrolabe.evaluators.wrappers.boltz import from_af3_to_boltz

    global_config.reset()
    global_config.init_const("binding_site", None)
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ACDE")],
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    data, _chain_map = from_af3_to_boltz(inp)
    assert "cyclic" not in data["sequences"][0]["protein"]


def test_protenix_prepare_input_injects_cyclic_asym_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    import astrolabe.evaluators.protenix as protenix_eval
    from astrolabe.evaluators.protenix import ProtenixEvaluator

    monkeypatch.setattr(
        protenix_eval,
        "from_af3_to_protenix",
        lambda cplx_input, oligo_name, **_kwargs: {"mock": True},
    )
    monkeypatch.setattr(
        protenix_eval,
        "protenix_from_dict_to_feature",
        lambda _sample, configs=None: ({"input_feature_dict": {}}, object(), {}),
    )

    evaluator = ProtenixEvaluator(model_runner=SimpleNamespace(configs={}))
    inp = Input(
        name="AB",
        chains=[
            _protein_chain(chain_id="A", sequence="ACDE"),
            _protein_chain(chain_id="B", sequence="FGHI"),
        ],
        rng_seeds=[0],
        cyclic_chain_ids=("B",),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"AB": inp}, {})
    assert prepared["AB"][0]["input_feature_dict"]["cyclic_asym_ids"].tolist() == [1]


def test_protenix_prepare_input_skips_cyclic_asym_ids_without_cyclization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import astrolabe.evaluators.protenix as protenix_eval
    from astrolabe.evaluators.protenix import ProtenixEvaluator

    monkeypatch.setattr(
        protenix_eval,
        "from_af3_to_protenix",
        lambda cplx_input, oligo_name, **_kwargs: {"mock": True},
    )
    monkeypatch.setattr(
        protenix_eval,
        "protenix_from_dict_to_feature",
        lambda _sample, configs=None: ({"input_feature_dict": {}}, object(), {}),
    )

    evaluator = ProtenixEvaluator(model_runner=SimpleNamespace(configs={}))
    inp = Input(
        name="AB",
        chains=[
            _protein_chain(chain_id="A", sequence="ACDE"),
            _protein_chain(chain_id="B", sequence="FGHI"),
        ],
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"AB": inp}, {})
    assert "cyclic_asym_ids" not in prepared["AB"][0]["input_feature_dict"]


def test_protenix_prepare_input_injects_cyclic_relative_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import astrolabe.evaluators.protenix as protenix_eval
    from astrolabe.evaluators.protenix import ProtenixEvaluator

    monkeypatch.setattr(
        protenix_eval,
        "from_af3_to_protenix",
        lambda cplx_input, oligo_name, **_kwargs: {"mock": True},
    )
    monkeypatch.setattr(
        protenix_eval,
        "protenix_from_dict_to_feature",
        lambda _sample, configs=None: ({"input_feature_dict": {}}, object(), {}),
    )

    evaluator = ProtenixEvaluator(model_runner=SimpleNamespace(configs={}))
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ABCDEFGH")],
        rng_seeds=[0],
        cyclic_relative_patches=(
            CyclicRelativePatch(
                chain_id="A",
                offsets=(
                    CyclicRelativeOffset(residue_i=1, residue_j=6, offset=1),
                    CyclicRelativeOffset(residue_i=6, residue_j=1, offset=-1),
                ),
            ),
        ),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"A": inp}, {})
    assert prepared["A"][0]["input_feature_dict"]["cyclic_relative_patches"].tolist() == [
        [0, 1, 6, 1],
        [0, 6, 1, -1],
    ]


def test_chai_prepare_input_passes_cyclic_subchains(tmp_path: Path) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    evaluator = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai")
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ACDE")],
        rng_seeds=[0],
        cyclic_chain_ids=("A",),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"A": inp}, {})
    assert prepared["A"]["cyclic_subchains"] == ["A"]


def test_chai_prepare_input_keeps_cyclic_subchains_none_without_cyclization(
    tmp_path: Path,
) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    evaluator = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai")
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ACDE")],
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"A": inp}, {})
    assert prepared["A"]["cyclic_subchains"] is None


def test_chai_prepare_input_passes_cyclic_relative_patches(tmp_path: Path) -> None:
    from astrolabe.evaluators.chai1 import Chai1Evaluator

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("binding_site", None)
    global_config.set_variable("step", 0)

    evaluator = Chai1Evaluator(model_config={}, output_dir=tmp_path / "chai")
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ABCDEFGH")],
        rng_seeds=[0],
        cyclic_relative_patches=(
            CyclicRelativePatch(
                chain_id="A",
                offsets=(
                    CyclicRelativeOffset(residue_i=1, residue_j=6, offset=1),
                    CyclicRelativeOffset(residue_i=6, residue_j=1, offset=-1),
                ),
            ),
        ),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    prepared = evaluator.prepare_input({"A": inp}, {})
    assert prepared["A"]["cyclic_relative_patches"] == [
        {
            "chain_id": "A",
            "offsets": [(1, 6, 1), (6, 1, -1)],
        }
    ]


def test_af3_prepare_input_derives_cyclic_asym_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import astrolabe.utils.af3_compat as af3_compat
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator

    global_config.reset()
    global_config.init_const("num_tokens", {"AB": 8})

    monkeypatch.setattr(af3_compat, "to_af3_input", lambda inp: inp)

    evaluator = AlphaFold3Evaluator(model_runner=None, output_dir=tmp_path)
    inp = Input(
        name="AB",
        chains=[
            _protein_chain(chain_id="A", sequence="ACDE"),
            _protein_chain(chain_id="B", sequence="FGHI"),
        ],
        rng_seeds=[0],
        cyclic_chain_ids=("B",),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    evaluator.prepare_input({"AB": inp}, {})
    assert evaluator.kwargs["AB"]["cyclic_asym_ids"] == (2,)


def test_af3_prepare_input_omits_cyclic_asym_ids_without_cyclization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import astrolabe.utils.af3_compat as af3_compat
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator

    global_config.reset()
    global_config.init_const("num_tokens", {"AB": 8})

    monkeypatch.setattr(af3_compat, "to_af3_input", lambda inp: inp)

    evaluator = AlphaFold3Evaluator(model_runner=None, output_dir=tmp_path)
    inp = Input(
        name="AB",
        chains=[
            _protein_chain(chain_id="A", sequence="ACDE"),
            _protein_chain(chain_id="B", sequence="FGHI"),
        ],
        rng_seeds=[0],
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    evaluator.prepare_input({"AB": inp}, {})
    assert evaluator.kwargs["AB"] == {}


def test_af3_prepare_input_derives_cyclic_relative_patches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import astrolabe.utils.af3_compat as af3_compat
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator

    global_config.reset()
    global_config.init_const("num_tokens", {"A": 8})

    monkeypatch.setattr(af3_compat, "to_af3_input", lambda inp: inp)

    evaluator = AlphaFold3Evaluator(model_runner=None, output_dir=tmp_path)
    inp = Input(
        name="A",
        chains=[_protein_chain(chain_id="A", sequence="ABCDEFGH")],
        rng_seeds=[0],
        cyclic_relative_patches=(
            CyclicRelativePatch(
                chain_id="A",
                offsets=(
                    CyclicRelativeOffset(residue_i=1, residue_j=6, offset=1),
                    CyclicRelativeOffset(residue_i=6, residue_j=1, offset=-1),
                ),
            ),
        ),
        bonded_atom_pairs=None,
        user_ccd=None,
    )
    evaluator.prepare_input({"A": inp}, {})
    assert evaluator.kwargs["A"]["cyclic_relative_patches"] == (
        (1, 1, 6, 1),
        (1, 6, 1, -1),
    )
