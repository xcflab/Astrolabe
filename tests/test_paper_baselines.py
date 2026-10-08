from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import yaml

from astrolabe.paper_baselines import METHODS, build_baseline_plan, run_baseline


def _inputs(tmp_path, payload):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_bindcraft_dry_run_prepares_output_settings_without_writing(tmp_path):
    settings = tmp_path / "settings.json"
    original = {"design_path": "original", "starting_pdb": "target.pdb", "lengths": [80, 120]}
    settings.write_text(json.dumps(original))
    cfg = _inputs(tmp_path, {"settings": "settings.json", "filters": "filter.json", "advanced": "advanced.json"})
    output = tmp_path / "new output"
    plan = run_baseline("bindcraft", inputs=cfg, tool_dir=tmp_path / "BindCraft", out_dir=output, seed=19, dry_run=True)
    rendered = json.loads(plan["generated_files"][str(output / "native_settings.json")])
    assert rendered["design_path"] == str(output / "design")
    assert rendered["lengths"] == original["lengths"]
    assert json.loads(settings.read_text()) == original
    assert not output.exists()
    assert plan["executed"] is False
    assert plan["seed_support"] == "native_not_exposed"
    assert "--seed" not in plan["commands"][0]
    assert str(tmp_path / "filter.json") in plan["missing_assets"]


def test_rfdiffusion_then_mpnn_use_the_same_generated_backbone(tmp_path):
    cfg = _inputs(tmp_path, {
        "rfdiffusion": {"input_pdb": "target.pdb", "target_chain": "B", "target_length": 120,
                        "binder_length": 80, "hotspots_1_based": [12, 42], "checkpoint": "rf.pt"},
        "proteinmpnn": {"tool_dir": "ProteinMPNN", "designed_chains": "A", "num_seq_per_target": 8,
                        "sampling_temp": 0.1, "model_name": "v_48_020", "weights": "weights"},
    })
    output = tmp_path / "result"
    plan = build_baseline_plan("rfdiffusion-proteinmpnn", inputs=cfg, tool_dir=tmp_path / "RFdiffusion", out_dir=output, seed=31)
    rf, mpnn = plan["commands"]
    assert "contigmap.contigs=[80-80/0 B1-120]" in rf
    assert "ppi.hotspot_res=[B12,B42]" in rf
    assert "inference.design_startnum=31" in rf
    assert mpnn[mpnn.index("--pdb_path") + 1] == str(output / "rfdiffusion/design_31.pdb")
    assert mpnn[mpnn.index("--num_seq_per_target") + 1] == "8"
    assert mpnn[mpnn.index("--seed") + 1] == "31"


def test_shipped_protein_rf_recipe_matches_manuscript_sampling(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = build_baseline_plan(
        "rfdiffusion-proteinmpnn",
        inputs=root / "examples/paper/baselines/03_rfdiffusion_proteinmpnn.yaml",
        tool_dir=tmp_path / "RFdiffusion", out_dir=tmp_path / "out", seed=1,
    )
    rf, mpnn = plan["commands"]
    assert "ppi.hotspot_res=[]" in rf
    assert mpnn[mpnn.index("--num_seq_per_target") + 1] == "8"
    assert mpnn[mpnn.index("--sampling_temp") + 1] == "0.3"
    assert mpnn[mpnn.index("--pdb_path_chains") + 1] == "A"


@pytest.mark.parametrize("context", ["protein", "ligand"])
def test_halludesign_keeps_context_specific_proposal_and_input_fix(tmp_path, context):
    cfg = _inputs(tmp_path, {"model": "protenix", "context": context, "input_file": "initial.pdb",
                            "template_path": "template.json", "steps": 20, "cutlass_path": "cutlass",
                            "ligand_smiles": "C(C)O"})
    plan = build_baseline_plan("halludesign", inputs=cfg, tool_dir=tmp_path / "HalluDesign", out_dir=tmp_path / "out", seed=1)
    command = plan["commands"][0]
    assert Path(command[1]).name == f"halludesign_{context}_protenix_entry.py"
    assert Path(command[1]).is_file()
    assert ("--fix_chain_index" in command) == (context == "protein")
    assert ("--run_last_cycle" in command) == (context == "ligand")
    assert command[command.index("--mpnn") + 1] == (
        "protein_mpnn" if context == "protein" else "ligandmpnn_plus_proteinmpnn"
    )


def test_switchcraft_native_api_keeps_paper_sampling_boundary(tmp_path):
    cfg = _inputs(tmp_path, {"native_config": "switch.yaml", "length": 120})
    plan = build_baseline_plan("switchcraft", inputs=cfg, tool_dir=tmp_path / "switchcraft", out_dir=tmp_path / "out", seed=22)
    command = plan["commands"][0]
    assert command[1] == "-c"
    assert "switchcraft.run" in command[2]
    assert "ligandmpnn_seqs=0" in command[2]
    assert "recycles=0" in command[2]
    assert command[-2:] == ["120", "22"]
    assert plan["source_commit"] == "67ffc0d155112624b557ea14dc55dbb9ac703933"


@pytest.mark.parametrize("method,payload,script", [
    ("rf-msd", {"native_config": "msd.yaml", "pair": "il4_il13", "unit_id": "u1", "gpu": "0"}, "run_rfdiffusion_proteinmpnn_msd.py"),
    ("complexdesign", {"native_config": "complex.yaml", "pair": "il4_il13", "binder_length": 100, "target_pdb": "targets.pdb", "af_params_dir": "params"}, "complexdesign_afm.py"),
    ("bindcraft-dual", {"native_config": "bindcraft_alt.yaml"}, "run_bindcraft_alt.py"),
])
def test_study_adaptations_use_shipped_scientific_drivers(tmp_path, method, payload, script):
    cfg = _inputs(tmp_path, payload)
    plan = build_baseline_plan(method, inputs=cfg, tool_dir=tmp_path / "native", out_dir=tmp_path / "out", seed=1)
    driver = Path(plan["commands"][0][1])
    assert driver.name == script
    assert driver.is_file()
    assert "examples/paper/baselines/adapters" in str(driver)


def test_real_execution_stops_at_missing_native_assets(tmp_path):
    cfg = _inputs(tmp_path, {"native_config": "target.yaml", "num_designs": 1000, "budget": 80})
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Baseline inputs are incomplete"):
        run_baseline("boltzgen", inputs=cfg, tool_dir=tmp_path / "missing", out_dir=output, seed=1)
    assert not output.exists()


def test_rf_msd_does_not_silently_use_another_configured_checkout(tmp_path):
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump({"rfdiffusion": {"source": "other_checkout"}}))
    cfg = _inputs(tmp_path, {"native_config": "native.yaml", "pair": "il4_il13", "unit_id": "u1", "gpu": "0"})
    plan = build_baseline_plan("rf-msd", inputs=cfg, tool_dir=tmp_path / "chosen_checkout", out_dir=tmp_path / "out", seed=1)
    assert "Native config rfdiffusion.source must equal --tool-dir" in plan["blockers"]


def test_arbitrary_command_is_not_a_named_baseline(tmp_path):
    cfg = _inputs(tmp_path, {"command": ["echo", "wrong"]})
    with pytest.raises(ValueError, match="arbitrary commands"):
        build_baseline_plan("boltzgen", inputs=cfg, tool_dir=tmp_path, out_dir=tmp_path / "out", seed=1)
    assert len(METHODS) == 14


@pytest.mark.parametrize("method,script", [("rfaa", "run_inference.py"), ("rfdiffusion2", "rf_diffusion/run_inference.py")])
def test_ligand_rf_native_recipe_is_100_backbones_and_side_effect_free(tmp_path, method, script):
    cfg = _inputs(tmp_path, {
        "input_pdb": "ligand.pdb", "ligand_code": "FAD", "length": "140-180", "checkpoint": "rf.pt",
        "ligandmpnn": {"tool_dir": "LigandMPNN", "python": "/runtime/mpnn/python", "checkpoint": "mpnn.pt", "chains_to_design": "A"},
    })
    output = tmp_path / "output"
    plan = run_baseline(method, inputs=cfg, tool_dir=tmp_path / "native", out_dir=output, seed=7, dry_run=True)
    generation, sequences = plan["commands"]
    assert generation[1] == str(tmp_path / "native" / script)
    assert "inference.num_designs=100" in generation
    assert "inference.design_startnum=7" in generation
    manifest = json.loads(plan["generated_files"][str(output / "backbone_manifest.json")])
    assert len(manifest["outputs"]) == 100
    assert manifest["outputs"][0]["example_id"] == "design_7"
    assert manifest["outputs"][-1]["example_id"] == "design_106"
    if method == "rfdiffusion2":
        assert manifest["outputs"][0]["structure_path"].endswith("design_7-atomized-bb-False.pdb")
    assert Path(sequences[1]).name == "fit_ligand_sequences.py"
    assert plan["recipe_kind"] == "manuscript_native"
    assert not output.exists()


def test_rfd3_native_worker_preserves_exact_paper_backbone_count(tmp_path):
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"one_ligand": {"length": "140-180"}}))
    cfg = _inputs(tmp_path, {
        "native_config": "native.json", "checkpoint": "rfd3.ckpt",
        "ligandmpnn": {"tool_dir": "LigandMPNN", "python": "/runtime/mpnn/python", "checkpoint": "mpnn.pt", "chains_to_design": "A"},
    })
    output = tmp_path / "out"
    plan = run_baseline("rfdiffusion3", inputs=cfg, tool_dir=tmp_path / "foundry", out_dir=output, seed=11, dry_run=True)
    request = json.loads(plan["generated_files"][str(output / "rfd3_request.json")])
    assert request["n_batches"] * request["config"]["diffusion_batch_size"] == 100
    assert request["config"]["seed"] == 11
    assert "inference_sampler" not in request["config"]
    assert Path(plan["commands"][0][2]).name == "rfd3_worker.py"
    assert not output.exists()


def test_rf_sequence_fitting_preserves_five_candidates_per_backbone(tmp_path):
    root = Path(__file__).resolve().parents[1]
    helper = root / "examples/paper/baselines/adapters/fit_ligand_sequences.py"
    spec = importlib.util.spec_from_file_location("fit_paper_sequences", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = [{"structure_path": str(tmp_path / f"design_{i}.pdb")} for i in range(100)]
    commands = module.build_commands(outputs, python="python", tool_dir=tmp_path / "mpnn",
                                    checkpoint=tmp_path / "weights.pt", chains="A", out_dir=tmp_path / "out", seed=1)
    assert len(commands) == 100
    for command in commands:
        assert command[command.index("--batch_size") + 1] == "5"
        assert command[command.index("--number_of_batches") + 1] == "1"
        assert command[command.index("--model_type") + 1] == "ligand_mpnn"
    with pytest.raises(ValueError, match="requires 100 backbones"):
        module.build_commands(outputs[:-1], python="python", tool_dir=tmp_path, checkpoint="x", chains="A", out_dir=tmp_path, seed=1)


def test_boltzdesign1_keeps_distinct_design_and_redesign_counts(tmp_path):
    cfg = _inputs(tmp_path, {"target_name": "FAD", "ligand_code": "FAD", "input_pdb": "fad.pdb",
                            "design_samples": 100, "num_designs": 2, "checkpoint": "boltz1.ckpt", "ccd_path": "ccd.pkl"})
    plan = build_baseline_plan("boltzdesign1", inputs=cfg, tool_dir=tmp_path / "BoltzDesign1", out_dir=tmp_path / "out", seed=1)
    command = plan["commands"][0]
    assert command[command.index("--design_samples") + 1] == "100"
    assert command[command.index("--num_designs") + 1] == "2"
    assert command[command.index("--target_type") + 1] == "small_molecule"
    assert command[command.index("--run_alphafold") + 1] == "False"
    assert plan["seed_support"] == "native_not_exposed"
