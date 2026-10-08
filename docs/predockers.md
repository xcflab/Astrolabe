# Predocker Configuration

Predockers prepare evaluator output structures before a mutator consumes them.
They are configured by alias under `predocker.predocker_terms`; mutators refer
to those aliases with their `predocker` list.

The mainline registry is lazy. Adding a predocker type requires updating
`astrolabe/structure_processors/base.py` so the type can be imported on demand without
slowing CLI startup.

## Configuration Shape

```yaml
predocker:
  predocker_terms:
    vina_box:
      _type: autodock_vina
      _condition: target_ligand_in_target_pocket < 0.8
      binding_site_strategy: aligned_reference_pocket

  autodock_vina:
    vina_path: /path/to/vina
    obabel_path: /path/to/obabel

mutator:
  mutator_terms:
    ligand_mpnn_refine:
      _type: LigandMPNN
      predocker: [vina_box]
```

Type-level sections such as `predocker.autodock_vina` provide defaults.
Alias-level fields under `predocker.predocker_terms.<alias>` override those
defaults. Alias names are user-facing because mutators and stages refer to
them; do not rename aliases in generated configs.

## Shared Parameters

- `_type`: required in each alias term; selects the predocker class.
- `_condition`: optional loss expression evaluated through the loss score
  context. If false, the predocker returns the input unchanged and logs the
  skip.
- `cache_enabled`: optional `StructureProcessor` cache toggle. Cache keys are
  in-memory and scoped to the predocker instance.
- Any backend parameter may be placed either in the type-level defaults section
  or directly on the alias. Alias-level values win.

Artifacts are written under `<out_dir>/predocker`. Runtime failures should
raise or write explicit penalty metadata; hidden fallback is not allowed.

## Registered Types

### `all_alanine`

Mutates protein residues to alanine while preserving non-amino-acid atoms.

- Typical use: simplify side chains before sequence-model refinement.
- Required backend paths: none.
- Main artifacts: `allala_<input>.pdb`.
- Failure mode: structure parsing or writing failure raises from the underlying
  structure IO layer.

```yaml
predocker:
  predocker_terms:
    allala:
      _type: all_alanine
```

### `autodock_gpu`

Runs the AutoDock-GPU docking workflow for protein-ligand complexes.

- Env/path parameters: `autodock_gpu_path`, `autogrid4_path`, `obabel_path`,
  `mk_prepare_receptor`, `mk_prepare_ligand`, `mk_prepare_python_bin`,
  `mk_prepare_pythonpath`.
- Backend-private parameters: `num_modes`, `exhaustiveness`,
  binding-site strategy and reference-pocket fields.
- Artifacts: prepared receptor/ligand files, grid files, docking output, final
  merged structure, and binding-site metadata.
- Failure mode: missing executables or subprocess failure raises visibly.

```yaml
predocker:
  predocker_terms:
    gpu_dock:
      _type: autodock_gpu
      binding_site_strategy: interface
```

### `autodock_vina`

Runs the AutoDock Vina docking workflow for protein-ligand complexes.

- Env/path parameters: `vina_path`, `obabel_path`, `mk_prepare_receptor`,
  `mk_prepare_ligand`, `mk_prepare_python_bin`, `mk_prepare_pythonpath`.
- Backend-private parameters: `num_modes`, `exhaustiveness`,
  `binding_site_strategy`, `binding_site_coverage`, `binding_site_padding`,
  `reference_structure_path`, `reference_structure_chain`,
  `current_structure_chain`, `reference_pocket_positions`,
  `reference_pocket_positions_by_chain`, `reference_box_padding`,
  `reference_min_box_size`, `reference_box_geometry`, and
  `reference_alignment_mode`.
- Artifacts: docking pose candidates, selected pose, binding-site metadata, and
  optional penalty metadata.
- Fallback: docking penalty paths are explicit artifacts and score terms; they
  are not reported as normal docking success.

```yaml
predocker:
  predocker_terms:
    vina_box:
      _type: autodock_vina
      binding_site_strategy: residue
      binding_site_coverage: 80
```

### `ligand_fpocket_alignment`

Aligns ligand centroid to an fpocket pocket center before downstream docking or
design.

- Env/path parameters: `fpocket_path`.
- Backend-private parameters: `pocket_rank` plus shared AutoDock separation and
  ligand alignment parameters.
- Fallback: if fpocket extraction fails, the implementation logs the reason and
  uses the protein centroid.
- Artifacts: aligned complex under `<out_dir>/predocker`.

```yaml
predocker:
  predocker_terms:
    fpocket_align:
      _type: ligand_fpocket_alignment
      pocket_rank: 1
```

### `ligand_centroid_alignment`

Translates the ligand so its centroid matches the protein centroid.

- Required backend paths: shared structure parsing/conversion dependencies.
- Backend-private parameters: shared AutoDock separation and alignment
  parameters.
- Artifacts: centroid-aligned complex under `<out_dir>/predocker`.

```yaml
predocker:
  predocker_terms:
    centroid_align:
      _type: ligand_centroid_alignment
```

### `RAPiDock`

Runs peptide-protein docking through the RAPiDock runner.

- Required parameters: `model_dir` and `ckpt`.
- Backend-private parameters: `hotspot`, `max_peptide_length`, and runner
  parameters consumed by `RAPiDockRunner`.
- Artifacts: split protein/peptide structures, RAPiDock candidate materialized
  into the predocker directory, and merged selected complex.
- Failure mode: docking pipeline failures are raised as `PeptideDockingError`
  with the original exception chained.

```yaml
predocker:
  predocker_terms:
    peptide_dock:
      _type: RAPiDock
      model_dir: /path/to/rapidock/model
      ckpt: /path/to/rapidock.ckpt
```

### `dimer_interface_placer`

Moves a ligand-like chain toward a low-density dimer interface region.

- Backend-private parameters: `radius_cutoff`, `window_size`, `step_size`,
  `density_threshold`.
- Required input: structure with at least two protein chains and one target
  non-protein chain.
- Artifacts: placed complex under `<out_dir>/predocker`.
- Failure mode: invalid chain composition raises a visible `ValueError`.

```yaml
predocker:
  predocker_terms:
    dimer_place:
      _type: dimer_interface_placer
      radius_cutoff: 5.0
```

### `chemnet_placer`

Resamples a ligand chain with PLACER/ChemNet.

- Required dependency: importable `PLACER` module.
- Backend-private parameters: `num_samples` and PLACER runner options.
- Required input: structure with a suitable ligand-like target chain.
- Failure mode: missing PLACER raises at initialization; invalid inputs raise
  from the placer workflow.

```yaml
predocker:
  predocker_terms:
    chemnet:
      _type: chemnet_placer
      num_samples: 5
```

### `symmetry_canonicalizer`

Canonicalizes repeat, pseudocyclic, hetero-oligomer, or cyclic-oligomer
structures before refinement.

- Required parameters: `mode`; `oligo` is required except for runtime
  cyclic-oligomer clones.
- Backend-private parameters: `backend`, `order`, `reference_chain_index`,
  `included_protomer`, `repeat_canonicalizer_mode`,
  `rpxdock_failure_fallback`, and `rpxdock_extra_args`.
- Env/path parameters: `rpxdock_python_bin`, `rpxdock_vendor_repo`,
  `rpxdock_willutil_repo`, `rpxdock_timeout_s`.
- Fallback: rpxdock fallback must be one of `rfdiffusion`, `input`, or `raise`.
- Artifacts: symmetry case directories and manifest-like intermediate outputs
  under `<out_dir>/predocker/symmetry_cases`.

```yaml
predocker:
  predocker_terms:
    symmetry:
      _type: symmetry_canonicalizer
      mode: cyclic_oligomer
      backend: rfdiffusion
```

### `hotspot_reference_align`

Chooses a reference complex from evaluator `StandardScores` and aligns other
complexes into its hotspot frame.

- Required parameters: `hotspot_scores` and `target_hotspots`.
- Optional parameters: `binder_chain`, `tie_delta`, `atom_name`.
- Input mode: score-aware only; attach it to a mutator predocker chain so it is
  run through `run_nested_predocker_on_scores()`.
- Failure mode: calling the legacy path-only `_run_impl()` raises because full
  scores are required.

```yaml
predocker:
  predocker_terms:
    hotspot_reference:
      _type: hotspot_reference_align
      hotspot_scores:
        AB: pair_hotspot_support
        AC: pair_hotspot_support
      target_hotspots:
        AB: B:10,11,12
        AC: C:10,11,12
```

### `haddock3_nucleic`

Runs HADDOCK3 (Haddock) docking for nucleic-acid focused complexes, generating
ambiguous restraints from configured binder, target, and anchor chains.

- Env/path parameters: `haddock3_bin`.
- Backend-private parameters: `binder_chain`, `target_chains`, `anchor_chains`,
  `max_anchor_chains`, `nucleic_only`, `contact_cutoff`, `skip_if_contact`,
  `require_contact`, `sampling`, `ncores`, `restraint_distance`,
  `max_restraints`, `update_dynamic_binding_site`, `binding_site_mode`,
  `binding_site_max_distance`, `binding_site_min_distance`, and
  `max_binding_site_pairs`.
- Artifacts: HADDOCK work directory, generated restraint/config files, log file,
  and selected docked complex under `<out_dir>/predocker`.
- Failure mode: missing HADDOCK3 binary, invalid chains, or subprocess failure
  raises visibly.

```yaml
predocker:
  predocker_terms:
    haddock_nucleic:
      _type: haddock3_nucleic
      binder_chain: A
      target_chains: [B]
```

### `HADDOCK3_nucleic`

Compatibility spelling for `haddock3_nucleic`. New configs should use the
lowercase type; this alias stays registered for older overlays and generated
configs.

## Branch-Only Or Absent Types

Do not document a predocker as available until it is present in
`_PREDOCKER_TYPE_TO_MODULE`. Treat any unregistered experimental predocker as
branch-only and keep its config out of shared examples.

## Refactor Plan

- Keep the current config shape and merge order.
- Keep `_condition` on `StructureProcessor`, evaluated through loss score context.
- Keep env/path parameters backend-owned until an env contract migration is
  planned.
- Add new predocker types by updating the lazy registry map, adding a docs
  section, and adding focused tests for defaults, alias override, artifacts, and
  failure visibility.
