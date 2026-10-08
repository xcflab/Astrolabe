# Dn Symmetry Scoring

This note documents the Dn scoring terms used by the AF-guided dihedral
oligomer templates:

- `config/benchmark_dihedral_d2_dplm_protenix_af2_rpxdock_v1.yaml`
- `config/benchmark_dihedral_d3_dplm_protenix_af2_rpxdock_v1.yaml`

## Workflow Contract

The default stage is a Protenix-scored monomer stage. It must produce a usable
monomer before any oligomer prediction is attempted:

```yaml
default:
  oligo:
    - A
  evaluator: Protenix
  mutators:
    - dplm
  mutation_num: 100%
  acceptance_threshold: null
  weighted_score_expression: plddt + ptm - shape_penalty
  early_stopping: null
oligomer:
  _entrance: plddt > 0.70 and ((binder_loop_frac < 0.4) and (binder_rog <= globin_rog + 3))
```

The oligomer stage uses RPXDock only as a predocking proposal for
LigandMPNN. The score is computed from the evaluator output structure
(`eval_result.complex_paths[oligo_name]`), so AnAnaS and Dn metrics judge the
AF2-predicted oligomer, not the RPXDock projection by itself.

## Metrics

The Dn templates use these predicted-structure losses:

- `point_group_symmetry_norm`: fits a requested `D<n>` layout and scores orbit
  and protomer-orientation residuals.
- `ananas_rmsd_norm`: runs AnAnaS on the AF2-predicted oligomer. Missing
  symmetry is penalized instead of being treated as success.
- `dihedral_ring_sep_cv_raw`: checks consistency of adjacent-ring separation.
- `dihedral_pair_sep_cv_raw`: checks consistency of C2-paired protomer
  separation.
- `dihedral_pair_contact_count_norm`: rewards residue-pair contacts across the
  Dn C2-paired protomers.
- `dihedral_pair_min_distance_norm`: rewards compact C2-paired protomer
  geometry.

`separation_std` is not the Dn symmetry objective in these templates. It can
still exist for older cyclic oligomer workflows, but the D2/D3 dihedral
templates use the Dn terms above:

```yaml
dn_contact_term: d2_pair_contact + d2_pair_min_distance
dn_symmetry_term: d2_point_group_symmetry + d2_ananas + dn_ring_term + dn_pair_sep_term
```

For D3, the same pattern is used with `d3_*` terms.

## Acceptance Gate

The oligomer stage has an explicit AnAnaS gate:

```yaml
acceptance_gate: d2_ananas > 0.0
```

or:

```yaml
acceptance_gate: d3_ananas > 0.0
```

The first stage-entry row bypasses this target-stage gate so the run can leave
monomer optimization exactly once and start oligomer refinement. Later oligomer
mutation decisions apply the gate normally, so accepted refinement rows still
require detected Dn symmetry. Stage-entry rows are marked with
`acceptance_gate_skipped_for_stage_entry=1.0` in `scores.csv`.

For independent production screens, the D2/D3 configs use `steps: 41` because
the optimizer runs mutation steps `1..steps-1`; this yields 40 optimization
steps after the initial evaluation.

## Validation Evidence

The implementation is covered by targeted tests:

```bash
python -m pytest -q \
  tests/test_structure_losses.py \
  tests/test_repeat_symmetry.py \
  tests/test_run_stage_transition.py \
  tests/test_transition_consistency_gate.py
```

Task-level structural acceptance evidence for the D2/D3 runs is stored under:

```text
.codex-tasks/20260510-dn-protenix-af-guided/acceptance/
```

Those PyMOL sessions are intended for manual structural review of the
AF2-predicted oligomers, not just command success.
