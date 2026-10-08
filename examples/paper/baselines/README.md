# External methods used in the paper

```bash
astrolabe paper baseline boltzgen \
  --inputs examples/paper/baselines/02_boltzgen.yaml \
  --tool-dir vendor/boltzgen --out-dir outputs/boltzgen --seed 1 --dry-run
```

Fill the input paths for the selected method, inspect the printed native
command and missing assets, then remove `--dry-run` to execute it. Dry-run
does not import a model, create the output directory, or run a subprocess.
It does not establish that an external environment or model is ready.

The launcher YAML names native configurations and assets; paths are relative
to the launcher YAML. An optional `python` selects the existing external
runtime. Optional `assets` lists additional files/directories that must be
present. Model weights, target inputs, and external tools are not downloaded
or installed by these commands. There is no arbitrary command passthrough.

| CLI method | Required launcher fields | `--tool-dir` |
|---|---|---|
| `bindcraft` | `settings`, `filters`, `advanced` | BindCraft checkout |
| `rfdiffusion-proteinmpnn` | `rfdiffusion` and `proteinmpnn` mappings; see `03_rfdiffusion_proteinmpnn.yaml` | RFdiffusion checkout |
| `halludesign` | `context` (`protein`/`ligand`), `model` (`af3`/`protenix`), `input_file`, `template_path`, `steps`; AF3: `af3_model_dir`; Protenix: `cutlass_path`; ligand: `ligand_code` or `ligand_smiles` for its backend | HalluDesign checkout containing `HalluDesign_run.py` |
| `proteinhunter` | `context`, `length`, `steps`; `target_sequence` for protein or `ligand_smiles` for ligand; optional protein `target_pdb` + `target_pdb_chain` | Protein-Hunter checkout |
| `boltzgen` | `native_config`, `num_designs`, `budget` | BoltzGen checkout |
| `switchcraft` | `native_config`, `length` | SwitchCraft checkout |
| `chamaileon` | `checkpoint`, `dataset_json`, `cluster_path`, `processed_pdb_path_prefix`, `alphafold2_params_dir`, `binder_length`, `inference` mapping | Chamaileon checkout |
| `bindcraft-dual` (BindCraft*) | `native_config` | BindCraft checkout |
| `rf-msd` (RF-MSD*) | `native_config`, `pair`, `unit_id`, `gpu` | RFdiffusion checkout |
| `complexdesign` (ComplexDesign*) | `native_config`, `pair`, `binder_length`, `target_pdb`, `af_params_dir` | Patched ColabDesign checkout |
| `boltzdesign1` | `target_name`, `ligand_code`, `input_pdb`, `design_samples`, `num_designs`, `checkpoint`, `ccd_path`; optional `length_min`/`length_max` | BoltzDesign1 checkout, including its LigandMPNN |
| `rfaa` | `input_pdb`, `ligand_code`, `length`, `checkpoint`, `ligandmpnn` mapping | `baker-laboratory/rf_diffusion_all_atom` checkout |
| `rfdiffusion2` | `input_pdb`, `ligand_code`, `length`, `checkpoint`, `ligandmpnn` mapping | Official `RosettaCommons/RFdiffusion2` checkout |
| `rfdiffusion3` | `native_config`, `checkpoint`, `ligandmpnn` mapping; optional `engine_config` | Foundry checkout |

The Chamaileon `inference` mapping requires `also_fold_pmpnn_seq`,
`samples_per_length`, `use_hotspot`, `beam_search`, `num_timesteps`, and
`switch_steps`, taken from the actual native configuration. Its dataset JSON
must be the native **bare list**, with the corresponding target feature PKLs
and cluster file. The adapter metadata wrapper is not a valid dataset.

The RFdiffusion/ProteinMPNN command generates one backbone and then runs
ProteinMPNN on it. The paper example samples eight binder sequences at
temperature 0.3 with the target fixed and no hotspot input. Evaluate all eight
with AF3 and select the highest-ipTM sequence for the backbone unit. RF-MSD* instead
uses the shipped original adapter: one target-1 backbone, the specified
target correspondence and rigid transformation, and the maintained
ProteinMPNN-MSD `run/generate_json.py` and `run/run_protein_mpnn.py` pipeline.
Its 50 shared sequences per backbone are not replaced by vanilla tied-chain
ProteinMPNN. The native RF-MSD YAML must point `rfdiffusion.source` to the
selected `--tool-dir` and name its separate ProteinMPNN-MSD source/weights.
Both RF-MSD generation directions must be requested. For the reverse native
configuration, exchange `pairs.<pair>.target_1` and `target_2` and exchange their
`paired_hotspots`; in an expanded configuration exchange the two `targets`
entries and the corresponding two hotspot entries. Run the two launcher
YAMLs with different `unit_id`, `--seed` and `--out-dir` values. Select the
highest pair-minimum AF3 ipTM among the 50 sequences for each backbone;
two directions contribute 200 planned backbone units per target pair.

BindCraft* and ComplexDesign* are this study's adaptations. `bindcraft-dual`
uses the joint gradient mean and evaluates its discrete proposal in both
target contexts. As confirmed by the author on 2026-10-09, it retains the
historical `tries=1` behavior: accept the sole proposal without comparing
the incumbent and record the mean full loss across the two contexts.
The older V1 alternating cohort is a separate historical protocol;
its frozen driver was not recovered and is not represented by this command.
The current scientific
drivers are shipped under `adapters/`, alongside the required local helper
and ColabDesign target-pair-mask patch. They retain the original method and
input checks. BindCraft* native YAML must supply four trajectory seeds
`[100*seed+11, 100*seed+12, 100*seed+13, 100*seed+14]` and output paths matching
`--out-dir`. ComplexDesign* requires the external ColabDesign checkout to have
the shipped patch applied and its original `.complexdesign_patch.json`
marker. The launcher does not patch an installed environment automatically.
The original native configurations still carry required target, AF parameter,
advanced-settings, and provenance inputs; supply those files explicitly.
Each BindCraft* native invocation produces four trajectories, so 100 planned
trajectories require 25 such invocations with their distinct seed sets.

The HalluDesign-Protenix and ProteinHunter protein entrypoints retain the
original input compatibility fixes and are also included in `adapters/`.
ProteinHunter runs in `--out-dir`, where its native results appear under
`results_chai/job`. BindCraft receives a generated copy of the native settings
with `design_path` under `--out-dir`; the user's settings file is preserved.

SwitchCraft invokes the same `switchcraft.run` API as the paper's unit runner:
one design, zero additional recycles, zero post-optimization LigandMPNN
sequences, and explicit Python/NumPy/Torch seeds. Supply the original complete
state/loss configuration and its matching assets. Source command flags for
BindCraft, HalluDesign, ProteinHunter, and BoltzGen expose **no numerical model
seed**; their plan reports `seed_support: native_not_exposed`. The requested
seed sets `PYTHONHASHSEED` and does not imply complete RNG reproducibility.

`01_adapter_sources.json` records original research paths, source hashes, and
the sole adapter change: allowing the BindCraft source root to be supplied by
`ASTROLABE_BINDCRAFT_DIR`. Headers and scientific code are retained. Upstream
model code and weights remain external and retain their own licenses. The
available source pins are reported in each plan; missing pins are explicit.
These launchers have command-construction validation, not a new GPU benchmark
or a claim that the historical paper cohorts have been regenerated.

## Additional ligand comparators

The four additional ligand methods are explicitly marked
`recipe_kind: manuscript_native`. Their historical generation scripts were
not recovered from the frozen AF3 endpoint archives. Native interfaces were
verified against the pinned source revisions in
`04_ligand_native_sources.json`; these recipes do not claim to recreate an
unavailable historical runtime/configuration snapshot.

`rfaa`, `rfdiffusion2`, and `rfdiffusion3` generate **100 backbones**, then fit
**five LigandMPNN sequences per backbone**. The `ligandmpnn` mapping requires
`tool_dir`, an absolute runtime `python`, `checkpoint`, and
`chains_to_design`. The included fitting adapter uses native
`--batch_size 5 --number_of_batches 1`, retains every backbone, and rejects
a missing or duplicate backbone rather than shrinking the denominator. The
older `001_benchmark` eight-sequence example is not used. Sequence sampling
uses seed plus backbone index; the historical RNG schedule was not recovered.

The paper's display label “RoseTTAFold All-Atom”/“RFAA” maps in the original
project to the **RFdiffusion All-Atom generator** (`rf_diffusion_all_atom`).
The `rfaa` command follows that source mapping, not the separate all-atom
structure-prediction program. RFdiffusion2 uses its native `aa` configuration
and the `atomized-bb-False` branch visible in the retained candidate names.
Its other sampler settings remain native defaults. Supply a ligand-containing
PDB, the desired contig length (for example a fixed length or range), and the
correct checkpoint; no missing ligand geometry is invented.

RFdiffusion3 reuses Astrolabe's existing Foundry worker. `native_config` must
contain exactly one complete native ligand design specification. One loaded
model executes 100 batches of one output, giving exactly 100 backbones without
requiring a large GPU batch. Sampler defaults remain unchanged unless the
caller explicitly supplies `engine_config` settings. Foundry and LigandMPNN
may use separate runtimes.

BoltzDesign1 keeps its native optimization and internal LigandMPNN refinement;
`design_samples` and `num_designs` are separate required counts. Its complete
historical optimization schedule was not recovered, so the specified source's
native defaults are retained. The launcher creates an isolated work directory
with links to the selected source assets, passes explicit Boltz-1/CCD paths,
and leaves common AF3/Rosetta evaluation to the paper evaluation commands.
It exposes no numerical seed flag. No method downloads weights or installs
source during planning; absent checkpoints and source checkouts are listed.
