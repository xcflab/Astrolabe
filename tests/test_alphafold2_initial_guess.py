from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import types

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
vendor_path = REPO_ROOT / "vendor" / "alphafold2"
vendor_str = str(vendor_path)
if vendor_str not in sys.path:
    sys.path.insert(0, vendor_str)

try:
    import jax  # noqa: F401
except ModuleNotFoundError:
    jax_mod = types.ModuleType("jax")
    jax_mod.tree = types.SimpleNamespace(map=lambda fn, value: value)
    jax_lib_mod = types.ModuleType("jax.lib")
    xla_bridge_mod = types.ModuleType("jax.lib.xla_bridge")
    xla_bridge_mod.get_backend = lambda: types.SimpleNamespace(platform="test")
    jax_lib_mod.xla_bridge = xla_bridge_mod
    sys.modules["jax"] = jax_mod
    sys.modules["jax.lib"] = jax_lib_mod
    sys.modules["jax.lib.xla_bridge"] = xla_bridge_mod

from alphafold.common import residue_constants

# The default local test interpreter is Python 3.9, while the lightweight
# folding_input dataclasses use Python 3.10 dataclass knobs. This test does not
# instantiate those classes, so make import-time parsing compatible.
_dataclass = dataclasses.dataclass


def _dataclass_py39_compat(*args, **kwargs):
    kwargs.pop("slots", None)
    kwargs.pop("kw_only", None)
    return _dataclass(*args, **kwargs)


dataclasses.dataclass = _dataclass_py39_compat

_STUBBED_MODULE_NAMES = (
    "astrolabe.core.protomer_config_parser",
    "astrolabe.evaluators.wrappers.alphafold",
    "astrolabe.utils.constant",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}

protomer_parser_mod = types.ModuleType("astrolabe.core.protomer_config_parser")
protomer_parser_mod.ProtomerTypes = types.SimpleNamespace(PROTEIN="protein")
sys.modules["astrolabe.core.protomer_config_parser"] = protomer_parser_mod
af2_wrapper_mod = types.ModuleType("astrolabe.evaluators.wrappers.alphafold")
af2_wrapper_mod.make_AF2_features = lambda *args, **kwargs: {}
af2_wrapper_mod.make_AFM_features = lambda *args, **kwargs: {}
sys.modules["astrolabe.evaluators.wrappers.alphafold"] = af2_wrapper_mod
constant_mod = types.ModuleType("astrolabe.utils.constant")
constant_mod.DEFAULT_SEEDS = [0]
sys.modules["astrolabe.utils.constant"] = constant_mod

from astrolabe.evaluators.alphafold2 import AlphaFold2Evaluator

dataclasses.dataclass = _dataclass
for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


class _Chain:
    def __init__(self, chain_id: str, sequence: str, templates=None):
        self.id = chain_id
        self.sequence = sequence
        self.templates = templates


class _Input:
    def __init__(self, name: str, chains: tuple[_Chain, ...]):
        self.name = name
        self.chains = chains
        self.rng_seeds = (1,)

    @property
    def protein_chains(self) -> list[_Chain]:
        return list(self.chains)


def _chain(chain_id: str, sequence: str, templates=None) -> _Chain:
    return _Chain(chain_id=chain_id, sequence=sequence, templates=templates)


def _atom_line(
    serial: int,
    atom: str,
    resname: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} {chain_id:1s}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}"
    )


def _write_initial_guess_pdb(path: Path) -> None:
    lines = [
        _atom_line(1, "N", "ALA", "X", 1, 1.0, 0.0, 0.0),
        _atom_line(2, "CA", "ALA", "X", 1, 2.0, 0.0, 0.0),
        _atom_line(3, "CB", "ALA", "X", 1, 3.0, 0.0, 0.0),
        _atom_line(4, "N", "GLY", "X", 2, 4.0, 0.0, 0.0),
        _atom_line(5, "CA", "GLY", "X", 2, 5.0, 0.0, 0.0),
        "TER",
        _atom_line(6, "N", "SER", "Y", 1, 6.0, 0.0, 0.0),
        _atom_line(7, "CA", "SER", "Y", 1, 7.0, 0.0, 0.0),
        _atom_line(8, "CB", "SER", "Y", 1, 8.0, 0.0, 0.0),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n")


def test_af2_initial_guess_disabled_by_default() -> None:
    assert AlphaFold2Evaluator._initial_guess_enabled(None) is False
    assert (
        AlphaFold2Evaluator._initial_guess_enabled(
            {"enabled": False, "path": "/tmp/x.pdb"}
        )
        is False
    )


def test_af2_initial_guess_resolves_per_oligo_paths_and_chain_maps() -> None:
    cfg = {
        "enabled": True,
        "paths": {"AAAAA": "/tmp/open.pdb", "BBBBB": "/tmp/closed.pdb"},
        "chain_maps": {
            "AAAAA": {"A": "X"},
            "BBBBB": {"B": "Y"},
        },
        "allow_missing_residues": True,
    }
    runner = types.SimpleNamespace(_gh_initial_guess_cfg=cfg)

    open_cfg = AlphaFold2Evaluator._resolve_initial_guess_cfg(runner, "AAAAA")
    closed_cfg = AlphaFold2Evaluator._resolve_initial_guess_cfg(runner, "BBBBB")

    assert open_cfg["path"] == "/tmp/open.pdb"
    assert open_cfg["chain_map"] == {"A": "X"}
    assert open_cfg["allow_missing_residues"] is True
    assert closed_cfg["path"] == "/tmp/closed.pdb"
    assert closed_cfg["chain_map"] == {"B": "Y"}
    assert closed_cfg["allow_missing_residues"] is True


def test_af2_initial_guess_resolves_per_oligo_reference_blocks() -> None:
    runner = types.SimpleNamespace(
        _gh_initial_guess_cfg={
            "enabled": True,
            "allow_missing_chains": False,
            "references": {
                "AAAAA": {
                    "path": "/tmp/open.pdb",
                    "chain_map": {"A": "X"},
                    "allow_missing_chains": True,
                }
            },
        }
    )

    resolved = AlphaFold2Evaluator._resolve_initial_guess_cfg(runner, "AAAAA")

    assert resolved == {
        "enabled": True,
        "allow_missing_chains": True,
        "path": "/tmp/open.pdb",
        "chain_map": {"A": "X"},
    }


def test_af2_setup_keeps_initial_guess_on_model_1_or_2_with_one_recycle(
    monkeypatch,
) -> None:
    dataclasses.dataclass = _dataclass_py39_compat
    try:
        from astrolabe.evaluators.wrappers import alphafold as af2_wrapper
    finally:
        dataclasses.dataclass = _dataclass

    created_configs = {}
    param_calls = []

    def _fake_model_config(model_name):
        cfg = types.SimpleNamespace(
            model=types.SimpleNamespace(
                num_recycle=None,
                global_config=types.SimpleNamespace(multimer_mode=False),
            ),
            data=types.SimpleNamespace(
                common=types.SimpleNamespace(
                    num_recycle=None,
                    max_extra_msa=None,
                ),
                eval=types.SimpleNamespace(
                    max_msa_clusters=None,
                    num_ensemble=None,
                ),
            ),
        )
        created_configs[model_name] = cfg
        return cfg

    def _fake_get_model_haiku_params(*, model_name, data_dir):
        param_calls.append((model_name, data_dir))
        return {"model_name": model_name}

    class _FakeRunModel:
        def __init__(self, model_config, model_params):
            self.model_config = model_config
            self.model_params = model_params
            self.multimer_mode = getattr(
                model_config.model.global_config,
                "multimer_mode",
                False,
            )

    monkeypatch.setattr(af2_wrapper.config, "model_config", _fake_model_config)
    monkeypatch.setattr(
        af2_wrapper.data,
        "get_model_haiku_params",
        _fake_get_model_haiku_params,
    )
    monkeypatch.setattr(af2_wrapper.model, "RunModel", _FakeRunModel)

    initial_guess = {
        "enabled": True,
        "references": {
            "AAAAA": {"path": "/tmp/open.pdb"},
            "BBBBB": {"path": "/tmp/closed.pdb"},
        },
    }

    for model_id in (1, 2):
        model_name = f"model_{model_id}_ptm"
        runners = af2_wrapper.alphafold_setup(
            oligo_names=["AAAAA", "BBBBB"],
            model_id=model_id,
            recycles=1,
            suffix="ptm",
            data_dir="/weights/af2",
            initial_guess=initial_guess,
        )

        assert set(runners) == {"AAAAA", "BBBBB"}
        assert runners["AAAAA"] is runners["BBBBB"]
        assert runners["AAAAA"]._gh_initial_guess_cfg is initial_guess
        assert created_configs[model_name].model.num_recycle == 1
        assert created_configs[model_name].data.common.num_recycle == 1
        assert created_configs[model_name].data.common.max_extra_msa == 5
        assert created_configs[model_name].data.eval.max_msa_clusters == 5
        assert created_configs[model_name].data.eval.num_ensemble == 1
        assert created_configs[model_name].model.global_config.multimer_mode is False
        assert (model_name, "/weights/af2") in param_calls


def test_af2_initial_guess_positions_follow_input_chain_order(tmp_path: Path) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )

    positions = AlphaFold2Evaluator._initial_guess_positions(
        cfg={"enabled": True, "path": str(pdb_path), "chain_map": {"A": "X", "B": "Y"}},
        oligo_name="AB",
        input_obj=input_obj,
    )

    ca_idx = residue_constants.atom_order["CA"]
    assert positions.shape == (3, residue_constants.atom_type_num, 3)
    np.testing.assert_allclose(positions[0, ca_idx], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[1, ca_idx], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[2, ca_idx], [7.0, 0.0, 0.0])


def test_af2_auto_adds_residue_index_offsets_between_non_multimer_chains() -> None:
    input_obj = _Input(
        name="ABC",
        chains=(_chain("A", "AG"), _chain("B", "S"), _chain("C", "TVK")),
    )
    processed_features = {"residue_index": np.arange(6, dtype=np.int32)}

    added = AlphaFold2Evaluator._maybe_add_multichain_residue_index_offsets(
        processed_features,
        input_obj=input_obj,
        multimer_mode=False,
        offset=200,
        oligo_name="ABC",
    )

    assert added is True
    np.testing.assert_array_equal(
        processed_features["residue_index"],
        np.array([0, 1, 202, 403, 404, 405], dtype=np.int32),
    )


def test_af2_auto_chain_offsets_skip_multimer_features() -> None:
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    processed_features = {"residue_index": np.arange(3, dtype=np.int32)}

    added = AlphaFold2Evaluator._maybe_add_multichain_residue_index_offsets(
        processed_features,
        input_obj=input_obj,
        multimer_mode=True,
        offset=200,
        oligo_name="AB",
    )

    assert added is False
    np.testing.assert_array_equal(
        processed_features["residue_index"],
        np.array([0, 1, 2], dtype=np.int32),
    )


def test_af2_initial_guess_is_added_as_initial_prev_pos(tmp_path: Path) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    processed_features: dict[str, np.ndarray] = {}

    added = AlphaFold2Evaluator._maybe_add_initial_guess(
        processed_features,
        cfg={"enabled": True, "path": str(pdb_path), "chain_map": {"A": "X", "B": "Y"}},
        oligo_name="AB",
        input_obj=input_obj,
    )

    assert added is True
    assert processed_features["initial_prev_pos"].shape == (
        3,
        residue_constants.atom_type_num,
        3,
    )


def test_af2_initial_guess_can_disable_initial_prev_pos(tmp_path: Path) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    processed_features: dict[str, np.ndarray] = {}

    added = AlphaFold2Evaluator._maybe_add_initial_guess(
        processed_features,
        cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": "X", "B": "Y"},
            "use_initial_prev_pos": False,
        },
        oligo_name="AB",
        input_obj=input_obj,
    )

    assert added is False
    assert "initial_prev_pos" not in processed_features


def test_af2_initial_guess_populates_multimer_template_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    captured: dict[str, dict] = {}

    def fake_make_afm_features(*, chain_id_map, is_homomer_or_monomer, chain_homology):
        captured.update(chain_homology)
        return {"feature_marker": np.array([1], dtype=np.int32)}

    monkeypatch.setattr(
        "astrolabe.evaluators.alphafold2.make_AFM_features",
        fake_make_afm_features,
    )

    feature_dict, is_multimer = AlphaFold2Evaluator._build_feature_dict(
        input_obj,
        types.SimpleNamespace(multimer_mode=True),
        initial_guess_cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": "X", "B": "Y"},
        },
        oligo_name="AB",
    )

    ca_idx = residue_constants.atom_order["CA"]
    assert is_multimer is True
    assert feature_dict["feature_marker"].tolist() == [1]
    assert captured["A"]["templates"] is None
    assert "template_features" in captured["A"]
    template_a = captured["A"]["template_features"]
    assert template_a["template_all_atom_masks"].sum() > 0
    assert template_a["template_aatype"].shape == (1, 2, 22)
    np.testing.assert_allclose(
        template_a["template_all_atom_positions"][0, 0, ca_idx],
        [2.0, 0.0, 0.0],
    )
    assert template_a["template_all_atom_masks"][0, 0, ca_idx] == 1.0


def test_af2_initial_guess_can_disable_template_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    captured: dict[str, dict] = {}

    def fake_make_afm_features(*, chain_id_map, is_homomer_or_monomer, chain_homology):
        captured.update(chain_homology)
        return {"feature_marker": np.array([1], dtype=np.int32)}

    monkeypatch.setattr(
        "astrolabe.evaluators.alphafold2.make_AFM_features",
        fake_make_afm_features,
    )

    feature_dict, is_multimer = AlphaFold2Evaluator._build_feature_dict(
        input_obj,
        types.SimpleNamespace(multimer_mode=True),
        initial_guess_cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": "X", "B": "Y"},
            "use_template_features": False,
        },
        oligo_name="AB",
    )

    assert is_multimer is True
    assert feature_dict["feature_marker"].tolist() == [1]
    assert "template_features" not in captured["A"]
    assert "template_features" not in captured["B"]


def test_af2_initial_guess_does_not_override_explicit_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    explicit_template = object()
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG", templates=[explicit_template]), _chain("B", "S")),
    )
    captured: dict[str, dict] = {}

    def fake_make_afm_features(*, chain_id_map, is_homomer_or_monomer, chain_homology):
        captured.update(chain_homology)
        return {"feature_marker": np.array([1], dtype=np.int32)}

    monkeypatch.setattr(
        "astrolabe.evaluators.alphafold2.make_AFM_features",
        fake_make_afm_features,
    )

    AlphaFold2Evaluator._build_feature_dict(
        input_obj,
        types.SimpleNamespace(multimer_mode=True),
        initial_guess_cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": "X", "B": "Y"},
        },
        oligo_name="AB",
    )

    assert captured["A"]["templates"] == [explicit_template]
    assert "template_features" not in captured["A"]
    assert "template_features" in captured["B"]


def test_af2_initial_guess_template_features_zero_fill_omitted_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "initial_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )
    captured: dict[str, dict] = {}

    def fake_make_afm_features(*, chain_id_map, is_homomer_or_monomer, chain_homology):
        captured.update(chain_homology)
        return {"feature_marker": np.array([1], dtype=np.int32)}

    monkeypatch.setattr(
        "astrolabe.evaluators.alphafold2.make_AFM_features",
        fake_make_afm_features,
    )

    AlphaFold2Evaluator._build_feature_dict(
        input_obj,
        types.SimpleNamespace(multimer_mode=True),
        initial_guess_cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": None, "B": "Y"},
            "allow_missing_chains": True,
        },
        oligo_name="AB",
    )

    assert captured["A"]["template_features"]["template_all_atom_masks"].sum() == 0
    assert captured["B"]["template_features"]["template_all_atom_masks"].sum() > 0


def test_af2_initial_guess_can_zero_fill_explicitly_omitted_chain(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "target_only_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )

    positions = AlphaFold2Evaluator._initial_guess_positions(
        cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": None, "B": "Y"},
            "allow_missing_chains": True,
        },
        oligo_name="AB",
        input_obj=input_obj,
    )

    ca_idx = residue_constants.atom_order["CA"]
    assert positions.shape == (3, residue_constants.atom_type_num, 3)
    np.testing.assert_allclose(positions[:2], 0.0)
    np.testing.assert_allclose(positions[2, ca_idx], [7.0, 0.0, 0.0])


def test_af2_initial_guess_rejects_omitted_chain_unless_allowed(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "target_only_guess.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "S")),
    )

    with pytest.raises(ValueError, match="allow_missing_chains=true"):
        AlphaFold2Evaluator._initial_guess_positions(
            cfg={
                "enabled": True,
                "path": str(pdb_path),
                "chain_map": {"A": None, "B": "Y"},
            },
            oligo_name="AB",
            input_obj=input_obj,
        )


def test_af2_initial_guess_can_zero_fill_missing_residues_by_sequence_alignment(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "target_with_missing_residues.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "TS")),
    )

    positions = AlphaFold2Evaluator._initial_guess_positions(
        cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": None, "B": "Y"},
            "allow_missing_chains": True,
            "allow_missing_residues": True,
        },
        oligo_name="AB",
        input_obj=input_obj,
    )

    ca_idx = residue_constants.atom_order["CA"]
    assert positions.shape == (4, residue_constants.atom_type_num, 3)
    np.testing.assert_allclose(positions[:3], 0.0)
    np.testing.assert_allclose(positions[3, ca_idx], [7.0, 0.0, 0.0])


def test_af2_initial_guess_template_features_mark_missing_residues_as_gap(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "target_with_missing_residues.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "TS")),
    )

    features_by_chain = AlphaFold2Evaluator._initial_guess_template_features_by_chain(
        cfg={
            "enabled": True,
            "path": str(pdb_path),
            "chain_map": {"A": None, "B": "Y"},
            "allow_missing_chains": True,
            "allow_missing_residues": True,
        },
        oligo_name="AB",
        input_obj=input_obj,
    )

    gap_idx = residue_constants.HHBLITS_AA_TO_ID["-"]
    ser_idx = residue_constants.HHBLITS_AA_TO_ID["S"]
    template_b = features_by_chain["B"]
    assert template_b["template_sequence"][0] == b"-S"
    assert template_b["template_aatype"][0, 0, gap_idx] == 1.0
    assert template_b["template_aatype"][0, 0].sum() == pytest.approx(1.0)
    assert template_b["template_aatype"][0, 1, ser_idx] == 1.0
    assert template_b["template_aatype"][0, 1, gap_idx] == 0.0


def test_af2_initial_guess_rejects_missing_residues_unless_allowed(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "target_with_missing_residues.pdb"
    _write_initial_guess_pdb(pdb_path)
    input_obj = _Input(
        name="AB",
        chains=(_chain("A", "AG"), _chain("B", "TS")),
    )

    with pytest.raises(ValueError, match="allow_missing_residues=true"):
        AlphaFold2Evaluator._initial_guess_positions(
            cfg={
                "enabled": True,
                "path": str(pdb_path),
                "chain_map": {"A": None, "B": "Y"},
                "allow_missing_chains": True,
            },
            oligo_name="AB",
            input_obj=input_obj,
        )
