"""Source-grounded external comparator launchers; no model is imported at planning time."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


METHODS = (
    "bindcraft", "rfdiffusion-proteinmpnn", "halludesign", "proteinhunter",
    "boltzgen", "switchcraft", "chamaileon", "bindcraft-dual",
    "rf-msd", "complexdesign",
    "boltzdesign1", "rfaa", "rfdiffusion2", "rfdiffusion3",
)

SOURCES = {
    "bindcraft": "013_paper_writing/config/001_manuscript_package/03_per_design_score_release/02_balanced_current_sources/03_bindcraft_missing5.py",
    "rfdiffusion-proteinmpnn": "016_flexible_target_binder/30_method_benchmark/scripts/run_rf_mpnn_canary.py; scripts/run_rfdiffusion_proteinmpnn_msd.py:build_rf_command",
    "halludesign": "004_protein_binder/scripts/binder_missing_methods/run_one.py; 007_halludesign_proteinhunter_bench/002_explore_ligand_binding_from_scratch/bench/ligand_benchmark.py",
    "proteinhunter": "004_protein_binder/scripts/binder_missing_methods/common.py:build_protein_hunter_command_parts; 007_halludesign_proteinhunter_bench/002_explore_ligand_binding_from_scratch/bench/ligand_benchmark.py",
    "boltzgen": "004_protein_binder/scripts/binder_missing_methods/run_one.py:_run_boltzgen",
    "switchcraft": "018_ligand_switch_benchmark/config/001_ligand_switch_method_benchmark/05_per_ligand_100unit_pooled_comparison/01_design_unit_completion/switchcraft/run_switchcraft_unit.py",
    "chamaileon": "016_flexible_target_binder/30_method_benchmark/scripts/prepare_chamaileon_pilot.py:build_command",
    "bindcraft-dual": "016_flexible_target_binder/30_method_benchmark/scripts/run_bindcraft_alt.py",
    "rf-msd": "016_flexible_target_binder/30_method_benchmark/scripts/run_rfdiffusion_proteinmpnn_msd.py",
    "complexdesign": "016_flexible_target_binder/30_method_benchmark/scripts/complexdesign_afm.py",
    "boltzdesign1": "001_benchmark/ligand_benchmark/run_benchmark.py:BoltzDesignRunner; BoltzDesign1/boltzdesign.py native parser",
    "rfaa": "001_benchmark/ligand_benchmark/run_benchmark.py:RFAllAtomRunner; baker-laboratory/rf_diffusion_all_atom/run_inference.py",
    "rfdiffusion2": "https://github.com/RosettaCommons/RFdiffusion2/blob/d365cbf4db3958814a9f8e4f6f94fa309dfebc2b/rf_diffusion/run_inference.py",
    "rfdiffusion3": "astrolabe/initializers/wrappers/rfd3.py; astrolabe/initializers/wrappers/rfd3_worker.py; Foundry/models/rfd3/src/rfd3/engine.py",
}

# Pins observed in the archived launch records; they are provenance, not a claim
# that the user's external checkout has been installed or qualified.
SOURCE_PINS = {
    "bindcraft": "d2d3cd0b5d6b02d12d24afa59e640717e36f552c",
    "boltzgen": "a3149cf18eeb58648d1abbb27539bd73f746cdda",
    "switchcraft": "67ffc0d155112624b557ea14dc55dbb9ac703933",
    "boltzdesign1": "627c0cc7bab41e56f544c5d15467b2dbeb490168",
    "rfaa": "f913a19e16f30858ce7a724fe028475b1871319c",
    "rfdiffusion2": "d365cbf4db3958814a9f8e4f6f94fa309dfebc2b",
    "rfdiffusion3": "e7601390a5b8a22002bccace9ac0fe240ef1371b",
}

BASELINE_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "examples/paper/baselines"

_SWITCHCRAFT_CALL = """import argparse, os, random, sys
from pathlib import Path
import numpy as np
import torch
repo, config, output, length, seed = sys.argv[1:]
seed = int(seed)
random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
os.chdir(repo); sys.path.insert(0, repo)
import switchcraft
switchcraft.run(argparse.Namespace(config=config, num_designs=1, outpath=output,
    debug=False, verbose=False, recycles=0, length=int(length), num_workers=1,
    worker_id=0, ligandmpnn_seqs=0, visualize=True))
"""


def build_baseline_plan(method, *, inputs, tool_dir, out_dir, seed):
    """Return commands, required assets, and planned config writes without executing.

    ``inputs`` is a launcher YAML referring to the method's native inputs. Paths
    in this YAML are relative to that file. Starred methods use the shipped
    research adapters with external model source and their native configs.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown baseline {method!r}; choose from {', '.join(METHODS)}")
    inputs = Path(inputs).expanduser().resolve()
    cfg = yaml.safe_load(inputs.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Baseline inputs must be a YAML mapping")
    if "command" in cfg or "argv" in cfg:
        raise ValueError("Baseline inputs describe native assets; arbitrary commands are not supported")
    tool = Path(tool_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    python = str(cfg.get("python", sys.executable))
    required = []
    generated = {}
    blockers = []
    notes = []
    env = {"PYTHONHASHSEED": str(seed)}
    cwd = tool
    commands = []

    def value(key, mapping=cfg):
        if key not in mapping or mapping[key] in (None, "???"):
            raise ValueError(f"{method} inputs require {key!r}")
        return mapping[key]

    def path(key, mapping=cfg):
        item = Path(str(value(key, mapping))).expanduser()
        if not item.is_absolute():
            item = inputs.parent / item
        item = item.resolve()
        required.append(item)
        return str(item)

    def script(relative):
        item = tool / relative
        required.append(item)
        return str(item)

    def adapter(name):
        item = BASELINE_RESOURCE_DIR / "adapters" / name
        required.append(item)
        return str(item)

    if method == "bindcraft":
        settings = Path(path("settings"))
        rendered = output / "native_settings.json"
        if settings.is_file():
            native = json.loads(settings.read_text())
            native["design_path"] = str(output / "design")
            if native.get("starting_pdb"):
                target = Path(native["starting_pdb"]).expanduser()
                target = target if target.is_absolute() else settings.parent / target
                native["starting_pdb"] = str(target.resolve())
                required.append(target.resolve())
            generated[str(rendered)] = json.dumps(native, indent=2) + "\n"
        commands = [[python, script("bindcraft.py"), "--settings", str(rendered),
                     "--filters", path("filters"), "--advanced", path("advanced")]]
        notes.append("Native BindCraft has no seed argument. Requested seed is recorded and sets PYTHONHASHSEED, not all model RNGs.")
        notes.append("MainPaper 105 / SI 19 require the four-stage AF2-multimer protocol: 75 logits, 45 soft, 5 one-hot, 15 greedy iterations in the supplied advanced settings. The benchmark representative is the hallucinated trajectory sequence before ProteinMPNN redesign and final filtering; this launcher runs the native workflow without selecting that representative.")
    elif method == "halludesign":
        model = value("model")
        context = value("context")
        if model not in {"af3", "protenix"} or context not in {"protein", "ligand"}:
            raise ValueError("HalluDesign model must be af3/protenix and context protein/ligand")
        native = script("HalluDesign_run.py")
        command = [python]
        if model == "protenix":
            command.append(adapter(f"halludesign_{context}_protenix_entry.py"))
            notes.append("Uses the shipped original Protenix MSA repair entrypoint; plain upstream invocation does not reproduce the repaired runs.")
        command += [native, "--HalluDesign_model", model, "--input_file", path("input_file"),
                    "--template_path", path("template_path"), "--output_dir", str(output),
                    "--mpnn", "protein_mpnn" if context == "protein" else "ligandmpnn_plus_proteinmpnn",
                    "--num_seqs", "1", "--num_recycles", str(value("steps")),
                    "--ref_time_steps", "50", "--random_init"]
        if context == "protein":
            command += ["--fix_chain_index", "B"]
        else:
            command += ["--design_epoch_begin", str(value("steps")), "--run_last_cycle"]
            command += (["--ccd", str(value("ligand_code")), "--af3_num_samples", "1"]
                        if model == "af3" else ["--sm", str(value("ligand_smiles"))])
        if model == "af3":
            env["AF3_MODEL_DIR"] = path("af3_model_dir")
            env["GH_WEIGHT_AF3"] = env["AF3_MODEL_DIR"]
        else:
            env["CUTLASS_PATH"] = path("cutlass_path")
        commands = [command]
        notes.append("The source launcher exposes only PYTHONHASHSEED, not a numerical model seed.")
        notes.append("MainPaper 109 / SI 20 use 50 protein-binder cycles with the target fixed; MainPaper 129 / SI 23 use 20 ligand-binder cycles. Both use one proposal per cycle and 50 reference time steps. steps is supplied explicitly; protein representatives are selected by the highest optimization ipTM in the separate evaluation workflow.")
    elif method == "proteinhunter":
        context = value("context")
        if context not in {"protein", "ligand"}:
            raise ValueError("ProteinHunter context must be protein/ligand")
        command = [python]
        if context == "protein":
            command.append(adapter("protein_hunter_entry.py"))
        command += [script("chai_ph/design.py"), "--jobname", "job", "--n_trials", "1",
                    "--n_cycles", str(value("steps")), "--min_protein_length", str(value("length")),
                    "--max_protein_length", str(value("length")), "--seq", "",
                    "--target_seq", str(value("target_sequence" if context == "protein" else "ligand_smiles")),
                    "--percent_X", "80" if context == "protein" else "0"]
        if context == "protein":
            command += ["--n_recycles", "3", "--n_diff_steps", "200", "--hysteresis_mode", "templates",
                        "--repredict", "--omit_aa", "", "--temperature", "0.1", "--scale_temp_by_plddt",
                        "--gpu_id", "0", "--use_msa_for_af3"]
            if cfg.get("target_pdb"):
                command += ["--target_pdb", path("target_pdb"), "--target_pdb_chain", str(value("target_pdb_chain"))]
        required.append(tool / "LigandMPNN")
        cwd = output
        commands = [command]
        notes.append("The native result directory is results_chai/job inside out_dir. The source CLI has no numerical seed flag.")
        notes.append("MainPaper 114 / SI 20 use one trial of 50 protein-binder cycles against a fixed target, exclude sequences containing X, and retain the highest Chai-1 ipTM for AF3 evaluation. MainPaper 129 / SI 23 use 20 ligand-binder cycles, the scheduled fixed length, an empty starting sequence and percent_X=0. steps is explicit; representative filtering is separate from this native launch.")
    elif method == "boltzgen":
        required.append(tool / "src/boltzgen/cli/boltzgen.py")
        env["PYTHONPATH"] = str(tool / "src")
        commands = [[python, "-m", "boltzgen.cli.boltzgen", "run", path("native_config"),
                     "--output", str(output), "--protocol", "protein-anything",
                     "--num_designs", str(value("num_designs")), "--budget", str(value("budget"))]]
        notes.append("Native target specification includes its own structures/MSAs; native model assets must already be available. This source command has no numerical seed flag.")
        notes.append("The protein benchmark uses 1,000 initial designs and a final budget of 80 per target (MainPaper 112 / SI 20). The ligand benchmark retains 100 default-setting designs per ligand (MainPaper 130); the shipped 02_boltzgen.yaml is the protein recipe.")
    elif method == "switchcraft":
        required.append(tool / "switchcraft.py")
        commands = [[python, "-c", _SWITCHCRAFT_CALL, str(tool), path("native_config"),
                     str(output), str(value("length")), str(seed)]]
        notes.append("Uses the archived native switchcraft.run API with random/NumPy/Torch seeds, one design, no extra recycles or LigandMPNN sequences. Supply the original complete native state/loss YAML and local assets.")
        notes.append("MainPaper 169 / SI 25 require a 240-step two-state protocol, LigandContactLoss weight 1.0, ConfChangeLoss weight 10.0, and no motif constraints. These settings belong in native_config; recycles=0 is the separate native API setting, not a zero-step optimization. Terminal sequences are selected for common evaluation separately.")
    elif method == "rfdiffusion-proteinmpnn":
        rf, mpnn = value("rfdiffusion"), value("proteinmpnn")
        target = f"{value('target_chain', rf)}1-{value('target_length', rf)}"
        length = value("binder_length", rf)
        prefix = output / "rfdiffusion/design"
        hotspots = ",".join(f"{rf['target_chain']}{int(x)}" for x in value("hotspots_1_based", rf))
        command = [python, script("scripts/run_inference.py"), f"inference.output_prefix={prefix}",
                   f"inference.input_pdb={path('input_pdb', rf)}",
                   f"contigmap.contigs=[{length}-{length}/0 {target}]", f"ppi.hotspot_res=[{hotspots}]",
                   "inference.num_designs=1", f"inference.ckpt_override_path={path('checkpoint', rf)}",
                   "inference.deterministic=True", f"inference.design_startnum={seed}"]
        mpnn_root = Path(path("tool_dir", mpnn))
        required.append(mpnn_root / "protein_mpnn_run.py")
        mpnn_command = [str(mpnn.get("python", python)), str(mpnn_root / "protein_mpnn_run.py"),
                        "--pdb_path", f"{prefix}_{seed}.pdb", "--pdb_path_chains", str(value("designed_chains", mpnn)),
                        "--out_folder", str(output / "proteinmpnn"), "--num_seq_per_target", str(value("num_seq_per_target", mpnn)),
                        "--batch_size", "1", "--sampling_temp", str(value("sampling_temp", mpnn)),
                        "--model_name", str(value("model_name", mpnn)), "--path_to_model_weights", path("weights", mpnn),
                        "--seed", str(seed)]
        commands = [command, mpnn_command]
        notes.append("One RFdiffusion backbone followed by native ProteinMPNN. This is a single-target pipeline; RF-MSD uses its distinct original multi-state adapter.")
        notes.append("MainPaper 107 uses no hotspots, eight sequences per backbone at temperature 0.3, and only the binder chain designed (RFdiffusion output chain A; target B fixed). Common AF3 evaluation of all eight sequences and highest-ipTM backbone representative selection are separate from these generation commands.")
    elif method == "chamaileon":
        inference = value("inference")
        dataset_json = path("dataset_json")
        if Path(dataset_json).is_file() and not isinstance(json.loads(Path(dataset_json).read_text()), list):
            blockers.append("Chamaileon dataset_json must be the native bare list, not the adapter metadata wrapper")
        command = [python, "-W", "ignore", script("multiflow/experiments/inference_multi_target_binder_eval.py"),
                   "-cn", "inference_multi_target_binder_eval",
                   f"inference.multi_target_binder_eval_ckpt_path={path('checkpoint')}",
                   f"inference.predict_dir={output}", "inference.inference_subdir=native",
                   f"inference.seed={seed}", "inference.num_gpus=1",
                   f"inference.also_fold_pmpnn_seq={bool(value('also_fold_pmpnn_seq', inference))}",
                   f"inference.samples.samples_per_length={value('samples_per_length', inference)}",
                   f"inference.samples.length_subset=[{value('binder_length')}]",
                   "inference.pdb_multi_target_binder_test_dataset.num_eval_lengths=1",
                   "inference.pdb_multi_target_binder_test_dataset.samples_per_eval_length=1",
                   f"inference.pdb_multi_target_binder_test_dataset.json_path={dataset_json}",
                   f"inference.pdb_multi_target_binder_test_dataset.cluster_path={path('cluster_path')}",
                   f"inference.pdb_multi_target_binder_test_dataset.processed_pdb_path_prefix={path('processed_pdb_path_prefix')}",
                   f"inference.pdb_multi_target_binder_test_dataset.use_hotspot={bool(value('use_hotspot', inference))}",
                   f"inference.interpolant.beam_search={bool(value('beam_search', inference))}",
                   f"inference.interpolant.sampling.num_timesteps={value('num_timesteps', inference)}",
                   f"inference.interpolant.sampling.switch_steps={value('switch_steps', inference)}",
                   f"inference.interpolant.alphafold2_params_dir={path('alphafold2_params_dir')}"]
        commands = [command]
        notes.append("Use the matching released Chamaileon checkpoint/config and preprocessed target PKLs; this launcher does not synthesize their features.")
        notes.append("MainPaper 148 uses both target structures and their hotspot masks, with 100 planned units per pair. Exact timestep/beam/switch settings are taken from the supplied native configuration; the manuscript does not specify their values.")
    elif method == "bindcraft-dual":
        native_path = Path(path("native_config"))
        if native_path.is_file():
            native = yaml.safe_load(native_path.read_text())
            expected = [int(seed) * 100 + offset for offset in (11, 12, 13, 14)]
            if native.get("trajectory_seeds") != expected:
                blockers.append(f"Native trajectory_seeds must equal source unit mapping {expected}")
            configured_output = Path(native.get("output", {}).get("root", "")).expanduser()
            if not configured_output.is_absolute():
                configured_output = native_path.parent / configured_output
            if configured_output.resolve() != output:
                blockers.append("Native config output.root must equal --out-dir; set its other output paths consistently")
        required.append(tool / "functions")
        env["ASTROLABE_BINDCRAFT_DIR"] = str(tool)
        commands = [[python, adapter("run_bindcraft_alt.py"), "--config", str(native_path), "--run"]]
        notes.append("Uses the shipped joint-gradient-mean / jointly accepted semi-greedy BindCraft adaptation specified by the latest MainPaper. It does not recreate the archived V1 alternating cohort; that historical driver was not recovered.")
    elif method == "rf-msd":
        required.append(tool / "scripts/run_inference.py")
        native_config = Path(path("native_config"))
        if native_config.is_file():
            native = yaml.safe_load(native_config.read_text())
            configured_source = Path(native["rfdiffusion"]["source"]).expanduser()
            if not configured_source.is_absolute():
                configured_source = native_config.parent / configured_source
            if configured_source.resolve() != tool:
                blockers.append("Native config rfdiffusion.source must equal --tool-dir")
        commands = [[python, adapter("run_rfdiffusion_proteinmpnn_msd.py"),
                     "--config", str(native_config), "--pair", str(value("pair")),
                     "--unit-id", str(value("unit_id")), "--seed", str(seed),
                     "--gpu", str(value("gpu")), "--output-root", str(output)]]
        notes.append("Uses the shipped RF-MSD adapter; native config rfdiffusion.source must name tool_dir. Requires ProteinMPNN-MSD run/generate_json.py and run/run_protein_mpnn.py. Vanilla tied-chain ProteinMPNN is not this method.")
        notes.append("MainPaper 146 requires both generation directions, 200 backbone units per pair, and 50 shared sequences per backbone. One invocation generates against the first configured target; run the second direction with targets and hotspot correspondences exchanged. Highest pair-minimum AF3 ipTM representative selection occurs after evaluating the 50 sequences and is not performed by this generation adapter.")
    elif method == "complexdesign":
        required.append(tool / "colabdesign")
        required.append(BASELINE_RESOURCE_DIR / "sources/colabdesign_v1_1_3_target_pair_mask.patch")
        env["PYTHONPATH"] = str(tool)
        commands = [[python, adapter("complexdesign_afm.py"), "--config", path("native_config"),
                     "--pair", str(value("pair")), "--seed", str(seed), "--binder-length", str(value("binder_length")),
                     "--target-pdb", path("target_pdb"), "--af-params-dir", path("af_params_dir"),
                     "--output-dir", str(output)]]
        notes.append("Uses the shipped ComplexDesign AF-Multimer adapter/helper/patch. Requires ColabDesign with that exact target-pair mask patch and marker, target preparation, and native contract YAML.")
    elif method == "boltzdesign1":
        required.extend([tool / "boltzdesign", tool / "LigandMPNN"])
        env["PYTHONPATH"] = str(tool)
        command = [python, script("boltzdesign.py"), "--target_name", str(value("target_name")),
                   "--target_type", "small_molecule", "--target_mols", str(value("ligand_code")),
                   "--pdb_path", path("input_pdb"), "--work_dir", str(output),
                   "--gpu_id", str(cfg.get("gpu", 0)), "--design_samples", str(value("design_samples")),
                   "--num_designs", str(value("num_designs")), "--suffix", str(seed),
                   "--boltz_checkpoint", path("checkpoint"), "--ccd_path", path("ccd_path"),
                   "--run_alphafold", "False", "--run_rosetta", "False"]
        for key in ("length_min", "length_max"):
            if key in cfg:
                command += ["--" + key, str(cfg[key])]
        commands = [command]
        notes.append("Native BoltzDesign1 generation and its default LigandMPNN refinement run in an isolated work_dir. design_samples and num_designs are explicit; remaining optimizer settings use this source's native defaults. AF3/Rosetta evaluation is a separate paper step.")
        notes.append("The source CLI has no numerical seed flag; suffix only names the run and PYTHONHASHSEED is not complete RNG control. This is a manuscript/native recipe, not recovered archived generation commands.")
    elif method in {"rfaa", "rfdiffusion2", "rfdiffusion3"}:
        backbone_dir = output / "backbones"
        manifest_path = output / "backbone_manifest.json"
        if method == "rfdiffusion3":
            native_input = path("native_config")
            if Path(native_input).is_file():
                specifications = json.loads(Path(native_input).read_text())
                if not isinstance(specifications, dict) or len(specifications) != 1:
                    blockers.append("RFD3 native_config must contain exactly one ligand design specification")
            required.extend([tool / "models/rfd3/src/rfd3/engine.py", tool / "src"])
            env["PYTHONPATH"] = os.pathsep.join([str(tool / "src"), str(tool / "models/rfd3/src")])
            engine_config = dict(cfg.get("engine_config", {}))
            engine_config.update(ckpt_path=path("checkpoint"), seed=int(seed), diffusion_batch_size=1)
            request_path = output / "rfd3_request.json"
            generated[str(request_path)] = json.dumps({
                "config": engine_config, "input_path": native_input, "n_batches": 100,
                "output_dir": str(backbone_dir), "response_path": str(manifest_path),
            }, indent=2) + "\n"
            worker = Path(__file__).resolve().parent / "initializers/wrappers/rfd3_worker.py"
            required.append(worker)
            generation = [python, "-u", str(worker), str(request_path)]
            notes.append("Uses the existing Foundry worker/API. One model instance samples 100 batches of one backbone; native sampler defaults are preserved unless explicitly set in engine_config.")
        else:
            is_rfd2 = method == "rfdiffusion2"
            native_script = "rf_diffusion/run_inference.py" if is_rfd2 else "run_inference.py"
            prefix = backbone_dir / "design"
            generation = [python, script(native_script), "--config-name", "aa",
                          f"inference.input_pdb={path('input_pdb')}",
                          f"inference.ligand={value('ligand_code')}",
                          f"contigmap.contigs=[{value('length')}]", f"inference.output_prefix={prefix}",
                          "inference.num_designs=100", f"inference.design_startnum={seed}",
                          "inference.deterministic=True", f"inference.ckpt_path={path('checkpoint')}"]
            suffix = "-atomized-bb-False" if is_rfd2 else ""
            if is_rfd2:
                env["PYTHONPATH"] = str(tool)
                generation += ["inference.guidepost_xyz_as_design=True",
                               "inference.guidepost_xyz_as_design_bb=[False]", "inference.silent_out=False"]
            generated[str(manifest_path)] = json.dumps({"outputs": [
                {"example_id": f"design_{index}", "structure_path": f"{prefix}_{index}{suffix}.pdb"}
                for index in range(int(seed), int(seed) + 100)
            ]}, indent=2) + "\n"
            if method == "rfaa":
                notes.append("The paper's RFAA/RoseTTAFold All-Atom label maps to the archived rf_diffusion_all_atom generator, not the separate RoseTTAFold-All-Atom prediction CLI.")
            else:
                notes.append("Uses official RFdiffusion2 aa configuration with the atomized-bb-False output branch identified in the retained candidate names. Other native sampling settings come from the pinned source.")
        mpnn = value("ligandmpnn")
        mpnn_tool = Path(path("tool_dir", mpnn))
        required.append(mpnn_tool / "run.py")
        sequence = [sys.executable, adapter("fit_ligand_sequences.py"),
                    "--backbone-manifest", str(manifest_path), "--python", str(value("python", mpnn)),
                    "--tool-dir", str(mpnn_tool), "--checkpoint", path("checkpoint", mpnn),
                    "--chains", str(value("chains_to_design", mpnn)),
                    "--out-dir", str(output / "sequences"), "--seed", str(seed)]
        commands = [generation, sequence]
        notes.append("Manuscript/native recipe: 100 backbones and five native LigandMPNN sequences per backbone. Historical generation scripts were not recovered; the older 001_benchmark eight-sequence recipe is not represented as the paper cohort.")

    for asset in cfg.get("assets", []):
        item = Path(str(asset)).expanduser()
        required.append((item if item.is_absolute() else inputs.parent / item).resolve())
    missing = sorted({str(item) for item in required if not item.exists()})
    if method not in SOURCE_PINS:
        notes.append("No complete upstream revision pin was recovered for this launcher; supply the source version used with your native inputs.")
    seed_support = ("native_not_exposed" if method in {
        "bindcraft", "halludesign", "proteinhunter", "boltzgen", "boltzdesign1"
    } else "native")
    return {"method": method, "commands": commands, "cwd": str(cwd), "environment": env,
            "requested_seed": int(seed), "out_dir": str(output), "generated_files": generated,
            "missing_assets": missing, "blockers": blockers, "notes": notes,
            "seed_support": seed_support, "source": SOURCES[method],
            "recipe_kind": "manuscript_native" if method in {"boltzdesign1", "rfaa", "rfdiffusion2", "rfdiffusion3"} else "source_adapter",
            "source_commit": SOURCE_PINS.get(method), "executed": False}


def build_baseline_command(method, *, inputs, tool_dir, out_dir, seed):
    """Return one argv or, for a sequential pipeline, a list of argv lists."""
    commands = build_baseline_plan(method, inputs=inputs, tool_dir=tool_dir, out_dir=out_dir, seed=seed)["commands"]
    return commands[0] if len(commands) == 1 else commands


def run_baseline(method, *, inputs, tool_dir, out_dir, seed, dry_run=False):
    """Run the source-grounded commands, or return their side-effect-free plan."""
    plan = build_baseline_plan(method, inputs=inputs, tool_dir=tool_dir, out_dir=out_dir, seed=seed)
    if dry_run:
        return plan
    if plan["missing_assets"] or plan["blockers"]:
        raise ValueError("Baseline inputs are incomplete: " + "; ".join(plan["missing_assets"] + plan["blockers"]))
    output = Path(plan["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if method in {"rfaa", "rfdiffusion2", "rfdiffusion3"}:
        (output / "backbones").mkdir(exist_ok=True)
    for name, content in plan["generated_files"].items():
        Path(name).write_text(content)
    if method == "proteinhunter":
        link = output / "LigandMPNN"
        source = Path(tool_dir).expanduser().resolve() / "LigandMPNN"
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        elif link.resolve() != source:
            raise ValueError(f"Existing {link} does not refer to the selected ProteinHunter LigandMPNN")
    if method == "rfdiffusion-proteinmpnn":
        (output / "rfdiffusion").mkdir(exist_ok=True)
    if method == "boltzdesign1":
        for name in ("boltzdesign", "LigandMPNN"):
            link = output / name
            source = Path(tool_dir).expanduser().resolve() / name
            if not link.exists():
                link.symlink_to(source, target_is_directory=True)
            elif link.resolve() != source:
                raise ValueError(f"Existing {link} does not refer to the selected BoltzDesign1 checkout")
    env = {**os.environ, **plan["environment"]}
    python_dir = str(Path(plan["commands"][0][0]).parent)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
    for command in plan["commands"]:
        subprocess.run(command, cwd=plan["cwd"], env=env, check=True)
    plan["executed"] = True
    return plan
