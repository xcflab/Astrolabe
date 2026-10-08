from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alphafold3.common.folding_input import Input, Ligand, ProteinChain
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators import alphafold3 as af3_eval


def test_prepare_output_reuses_extracted_scores_per_oligo(monkeypatch, tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers", {"A": SimpleNamespace(type=ProtomerTypes.PROTEIN)}
    )
    global_config.set_variable("step", 0)

    evaluator = af3_eval.AlphaFold3Evaluator(
        model_runner=None,
        output_dir=tmp_path / "raw" / "alphafold3",
    )
    fold_input = Input(
        name="A",
        chains=[
            ProteinChain(
                id="A",
                sequence="AA",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        rng_seeds=[1],
    )
    outputs = {"A": [SimpleNamespace(seed=1, inference_results=[])]}

    def fake_write_outputs(*, output_dir, job_name, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{job_name}_model.cif").write_text("data_test\n", encoding="utf-8")

    scores = {
        "per_res_scores": {"per_res_plddt": [[90.0, 80.0]]},
        "per_str_scores": {
            "ptm": [0.1],
            "iptm": [0.2],
            "predicted_distance_error": [0.3],
        },
        "inter_o_cross_scores": {"pae_xchain": [np.array([[0.0]])]},
        "chain_pair_scores": {"chain_pair_lis": [np.array([[1.0]])]},
    }
    get_af3_scores_calls = []

    def fake_get_af3_scores(results):
        get_af3_scores_calls.append(results)
        return scores

    monkeypatch.setattr(af3_eval, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(af3_eval, "get_af3_scores", fake_get_af3_scores)

    try:
        prepared = evaluator.prepare_output(outputs, {"A": fold_input}, runtime_config={})
    finally:
        global_config.reset()

    assert len(get_af3_scores_calls) == 1
    assert get_af3_scores_calls[0] is outputs["A"]
    assert prepared.names == ["A"]
    assert np.allclose(prepared.plddt["A"][0], np.array([0.9, 0.8]))
    assert np.allclose(prepared.per_proto_scores["A"][0], np.array([0.9, 0.8]))


def test_prepare_output_passes_save_all_samples_from_af3_conf(
    monkeypatch, tmp_path: Path
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers", {"A": SimpleNamespace(type=ProtomerTypes.PROTEIN)}
    )
    global_config.set_variable("step", 0)

    evaluator = af3_eval.AlphaFold3Evaluator(
        model_runner=None,
        output_dir=tmp_path / "raw" / "alphafold3",
        af3_conf=SimpleNamespace(save_all_samples=True),
    )
    fold_input = Input(
        name="A",
        chains=[
            ProteinChain(
                id="A",
                sequence="AA",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            )
        ],
        rng_seeds=[1],
    )
    outputs = {"A": [SimpleNamespace(seed=1, inference_results=[])]}
    save_all_samples_values = []

    def fake_write_outputs(*, output_dir, job_name, save_all_samples, **_kwargs):
        save_all_samples_values.append(save_all_samples)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{job_name}_model.cif").write_text(
            "data_test\n", encoding="utf-8"
        )

    scores = {
        "per_res_scores": {"per_res_plddt": [[90.0, 80.0]]},
        "per_str_scores": {
            "ptm": [0.1],
            "iptm": [0.2],
            "predicted_distance_error": [0.3],
        },
        "inter_o_cross_scores": {"pae_xchain": [np.array([[0.0]])]},
        "chain_pair_scores": {"chain_pair_lis": [np.array([[1.0]])]},
    }

    monkeypatch.setattr(af3_eval, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(af3_eval, "get_af3_scores", lambda _results: scores)

    try:
        evaluator.prepare_output(outputs, {"A": fold_input}, runtime_config={})
    finally:
        global_config.reset()

    assert save_all_samples_values == [True]


def test_prepare_output_remaps_ccd_ligand_resname_in_canonical_structure(
    monkeypatch, tmp_path: Path
) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "B": SimpleNamespace(
                type=ProtomerTypes.CCD,
                sequence="B49",
                name="B49",
            ),
        },
    )
    global_config.init_const("ligand_resnames", {"B": "Z49", "B49": "Z49"})
    global_config.set_variable("step", 0)

    evaluator = af3_eval.AlphaFold3Evaluator(
        model_runner=None,
        output_dir=tmp_path / "raw" / "alphafold3",
    )
    fold_input = Input(
        name="AB",
        chains=[
            ProteinChain(
                id="A",
                sequence="AA",
                ptms=[],
                unpaired_msa="",
                paired_msa="",
                templates=[],
            ),
            Ligand(id="B", ccd_ids=["B49"]),
        ],
        rng_seeds=[1],
    )
    outputs = {"AB": [SimpleNamespace(seed=1, inference_results=[])]}

    def fake_write_outputs(*, output_dir, job_name, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{job_name}_model.cif").write_text(
            "\n".join(
                [
                    'data_step_000_AB',
                    'HETATM 1 C C1 . B49 B 1 . ? 0.0 0.0 0.0 1.00 1.00 1 B 1',
                    '_pdbx_nonpoly_scheme.mon_id B49',
                    '_chem_comp.id B49',
                    '_struct_conn.ptnr1_label_comp_id LIG_B',
                ]
            ),
            encoding="utf-8",
        )

    scores = {
        "per_res_scores": {"per_res_plddt": [[90.0, 80.0, 70.0]]},
        "per_str_scores": {
            "ptm": [0.1],
            "iptm": [0.2],
            "predicted_distance_error": [0.3],
        },
        "inter_o_cross_scores": {"pae_xchain": [np.array([[0.0]])]},
        "chain_pair_scores": {"chain_pair_lis": [np.array([[1.0]])]},
    }

    monkeypatch.setattr(af3_eval, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(af3_eval, "get_af3_scores", lambda _results: scores)

    try:
        evaluator.prepare_output(outputs, {"AB": fold_input}, runtime_config={})
    finally:
        global_config.reset()

    canonical = tmp_path / "structures" / "step_000_AB_model.cif"
    text = canonical.read_text(encoding="utf-8")
    assert "Z49" in text
    assert "B49" not in text
    assert "LIG_B" not in text
