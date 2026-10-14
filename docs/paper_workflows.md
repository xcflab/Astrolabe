# Manuscript workflows

The `astrolabe paper` commands connect the manuscript tasks to the existing
design engine, original RNA beam search, fixed-sequence evaluation and analysis.
Protocol configuration is in `config/paper/`; target and topology inputs are in
`examples/paper/inputs/`. `paper list` lists the design/evaluation protocols;
`paper analyze --help` lists the statistical and structural analyses.

The source manuscript is [MainPaper](https://docs.google.com/document/d/1Kq6pbxS8-CMkd34qNVdVih78L7LzSx4fKRgBnvmOvf4/edit),
with [Supplementary Information](https://docs.google.com/document/d/1Fs5F6eJsMFJEUKlRWSOuWw4K9a4Tn78mHHidtgN7yt0/edit).
The release also recovers configurations, templates and scientific routines
from the original run records. Their locations and checksums accompany the
bundled inputs. No preprint or DOI has been assigned.
The configuration alignment uses the 2026-10-08 16:12 UTC MainPaper and
16:03 UTC SI revisions, with the author's 2026-10-09 clarifications recorded
in [paper_sources.json](../resources/paper_sources.json).

The reviewer-facing data release is stored in the manuscript Drive folder and
contains only cleaned figure-linked data, a public manifest and the bundled
reviewer archive. Historical unredacted records are kept separately for internal
audit and are not part of the reviewer-facing release.

## Install and prepare inputs

Install Astrolabe using the commands in the [README](../README.md#install), then prepare
the required predictors and sequence models with the
[container installer](../container/README.md#install-af3). Select the models and
structural tools needed by the chosen protocol; `scripts/install.py --list-software`
shows the available installation names. Model parameters remain external.

```bash
astrolabe paper prepare-inputs --out-dir data/paper
astrolabe paper list
astrolabe paper show rna-aptamer
```

The input command checks the bundled archive/member SHA-256 hashes and extracts
the templates, MSAs and chemical definitions. Existing identical inputs can be
reused; different existing files are not overwritten. From a wheel installation,
the printed package location and `astrolabe.config_dir().parent / 'examples/paper'`
locate the supplied input overlays. In a source checkout, use the paths below.

## Invoke a workflow

```bash
astrolabe paper run rna-aptamer \
  --inputs examples/paper/inputs/nucleic/rna_aptamer_1urn_u1a.yaml \
  --data-dir data/paper --seed 1 --out-dir outputs/u1a-rna
```

The layering order is protocol, input YAML, named command options, then
`--overrides`/`--overrides-file`. `--dry-run` composes and prints this configuration;
`--dump-config filled.yaml` saves it. Neither preview nor successful composition
is evidence of a model trajectory. A normal invocation runs one trajectory;
request the paper cohort explicitly with `--num-runs` and retain its planned
unit manifest. `steps: N` includes the initialized state, so a completed run has
state indices 0 through N−1.

| Task | Example protocol | Input | Models / structural tools |
| --- | --- | --- | --- |
| Protein binder | `protein-binder-af3` | `protein/cd38_3dzi.yaml` | AF3, DPLM, LigandMPNN and Rosetta metrics |
| SAP-aware design | `protein-binder-sap` | Protein target or supplied parent input | AF3, interface LigandMPNN, SolubleMPNN, PyRosetta |
| IDR | `idr-binder` | `protein/idr_hif1a.yaml` | Predictor/mutators in the selected manuscript or archived recipe |
| Flexible multidomain Cre | `cre-binder-af3` | `protein/cre.yaml` | AF3, DPLM 650M, LigandMPNN; split-domain templates |
| Shared dual-target minibinder | `dual-target-af3` | `protein/dual_il4_il13.yaml` | AF3 and shared-sequence ProteinMPNN/SolubleMPNN; alternative Protenix v2/ESMFold2 recipes |
| VHH/scFv frameworks | `antibody-framework` | `protein/antibody_cd38_3dzi_vhh_7eow.yaml` | AF3, ProteinMPNN, framework MSA/template hook |
| Cross-reactive VHH | `cross-reactive-vhh` | `protein/vhh_il4_il13.yaml` | AF3 and ProteinMPNN probabilities averaged across the two contexts |
| Small-molecule binder | `ligand-binder-af3` | `--ligand FAD --ligand-type ccd` | AF3, DPLM, LigandMPNN, Vina and Rosetta metrics |
| Ligand-associated two states | `two-state-af3` | Ligand and exact historical Protenix checkpoint | Historical Protenix v0.5, AF3 or Boltz-2, LigandMPNN |
| Target-pocket retargeting | `pocket-rescue` | `interventions/pocket_catm.yaml` | AF3, LigandMPNN, mapped pocket geometry and conditional Vina |
| Chemically induced dimer | `cid-rescue` | Explicit two proteins and ligand | AF3, LigandMPNN, interface geometry and conditional Vina |
| Pseudo-C3 | `pseudo-c3` | An `assemblies/*_input.yaml` and its reference CIF | AF2, synchronized cores, independent patches and original proposal mixture |
| Tandem repeats | `tandem-repeat-3` | Recipe supplies synchronized repeat/linker pattern | AF2 and ProteinMPNN in both stages |
| Cyclic/dihedral assemblies | `symmetry-c3` / `symmetry-d2` | Recipe length/seed, or a supplied sequence | AF2; Protenix monomer stage for dihedral cases; RPXDock rescue |
| Bound cyclic peptide | `cyclic-peptide-bound` | A `cyclic/bound/` topology input | AF3, LigandMPNN, explicit atom-level chemistry |
| Peptide-only encoding / multiring | `cyclic-peptide-representation` / `cyclic-peptide-multiring` | A `cyclic/` topology input | AF3; manuscript and archived mutators are separately named |
| RNA/DNA aptamer | `rna-aptamer` / `dna-aptamer` | Four targets under `nucleic/` | AF3, NA-MPNN, fixed protein template |
| Protein binder to RNA/DNA | `protein-binder-rna` / `protein-binder-dna` | Six targets under `nucleic/` | DPLM, exact Protenix v1 checkpoint, NA-MPNN and conditional HADDOCK3 |
| RNA oligomer beam | `rna-oligomer-4` / `rna-oligomer-5` | Original 87-nt starts included in recipe | AF3, NA-MPNN; beam 8, four proposals/parent, budget 1,000 |
| Bivalent RNA beam | `bivalent-rna` | Random shared 20-nt repeats and bundled U1A template | AF3, NA-MPNN; 200 evaluations/start |

Input paths in the table are relative to `examples/paper/inputs/`. The catalogue
also includes matched no-rescue, single-target, alternative-predictor and
archived variants. Inspect the selected YAML for exact sampling settings and
its representative rule. Missing historical checkpoints are explicit required
inputs; a current Protenix v2 model is not substituted for v0.5 or v1.

RNA aptamer runs use 20 states; DNA uses 40, with both modalities summarized
from the maximum-objective state among indices 0–19. The oligomer beam ranks
the original five-sample mean pLDDT; bivalent scoring uses the selected structure
and its corresponding confidence arrays, preserving the original contact/PAE
objective. Beam outputs retain proposals, repeated-sequence exclusions, native
structures, final beam and whether the evaluation budget was fully used.

To run the three random synchronized starts specified in the Methods:

```bash
astrolabe paper run bivalent-rna --num-runs 3 --seed 1 --out-dir outputs/bivalent
```

Use `bivalent-rna-archived` to replay the three recovered starts and their
original seeds. Those starts were derived by contact-guided shortening.

For ligand-associated two-state design, supply the historical first-stage
checkpoint explicitly:

```bash
astrolabe paper run two-state-af3 --ligand SAM --seed 1 \
  --overrides 'paper_protocol.historical_protenix_checkpoint=/path/to/protenix_base_default_v0.5.0.pt' \
  --out-dir outputs/sam-two-state
```

The first stage predicts apo and holo; the second predicts only holo. Use
`two-state-boltz2` for the declared Boltz-2 second-stage replacement.

## SolubleMPNN post-design control

```bash
astrolabe paper run solublempnn-control \
  --inputs examples/paper/inputs/protein/cd38_3dzi.yaml --data-dir data/paper \
  --parent-structure parent_complex.cif --binder-chain A --target-chain B \
  --num-runs 20 --seed 1 --out-dir outputs/solublempnn-control
```

The command fixes the target and binder residues having a heavy atom strictly
within 4 Å of a target heavy atom. It samples the remaining binder positions
on the same parent backbone with SolubleMPNN and performs a fixed-sequence AF3
evaluation. The first 20 variants are nested within one parent design; they
are not 20 independent benchmark units.

## Fixed-sequence evaluation

Use `evaluate-af3`, `evaluate-protenix2`, `evaluate-esmfold2` or `evaluate-af2`
with literal sequences in the input YAML. These recipes evaluate state 0 and
make no sequence proposals. Supply the paper's target MSA/template conditions;
use empty MSAs/templates for the apo/holo ensemble experiment.

For apo/holo analysis, run each state with five evaluation seeds and five
samples per seed, and keep every native sample under `raw_model_results`:

```bash
astrolabe paper run evaluate-af3 --inputs holo.yaml --num-runs 5 --seed 1 --out-dir outputs/holo
astrolabe paper run evaluate-af3 --inputs apo.yaml --num-runs 5 --seed 1 --out-dir outputs/apo
astrolabe paper analyze two-state --manifest ensemble.csv --out-dir analysis/two-state
```

The structure manifest columns are documented in
[`paper_twostate.py`](../astrolabe/analysis/paper_twostate.py). Keep predictors
separate. The analysis preserves the seed/sample hierarchy, undefined geometry,
confidence and compactness gates, and state-selective ordering calculations.

## Collect and analyze

Create `planned.csv` before execution with one row per independent unit:

```csv
unit_id,method,group,run_dir,selection,stage
design_001,Astrolabe,target_A,outputs/target_A/001_run,last_accepted,complex
design_002,Astrolabe,target_A,outputs/target_A/002_run,last_accepted,complex
```

Use the actual run directories and stage names emitted by the chosen protocol.
Leave a failed unit in this manifest. Relative run paths are resolved from the
manifest directory. `selection` and `stage` are explicit per-unit choices.

```bash
astrolabe paper collect --planned planned.csv --output records.csv
astrolabe paper analyze scores --endpoint dual-target \
  --planned planned.csv --records records.csv --out-dir analysis/dual-target
```

The collector retains all logged states, including rejected proposals and
repeated sequences. Analysis selects representatives according to the declared
rule, evaluates conjunctions within the same state, keeps planned denominators
and averages target-specific fractions with equal target weights. Output files
are `units.csv`, `selected_records.csv`, `groups.csv`, `overall.csv` and
`overall.json`; paired-target analyses also produce `dual_target_ccdf.csv`.

Use `--column METRIC=CSV_COLUMN` for explicit metric names. Confidence and pLDDT
inputs use the 0–1 scale; compactness and SAP must already use the paper's
length normalization. For protein benchmark endpoints, provide common AF3
evaluation records after method-specific representative selection. Do not treat
native optimization scores as common AF3 re-evaluation.

For the SolubleMPNN control, one planned row is one parent, `selection` is
`all_variants`, and its records use zero-based `variant_index` 0–19 in generation
order. All cumulative thresholds must be satisfied by the same variant.
`parent_id` is the stable join key for the underlying initial design or run:
identical IDs indicate paired baseline, SAP-aware or post-design-control records.
Baseline and SAP-aware IDs match for 12 protein targets; ORM2 uses an independently
sampled baseline cohort, while its SAP-aware and post-design-control records are
paired.

For cyclic peptides, use the release's `closure_all_bonds` metric, which checks
absolute distance error ≤0.30 Å across every specified bond and fails missing
atoms. The optimization Gaussian score is a separate quantity. The frozen
target-bound cohort contains 2,720 designs across 13 topology–linkage categories,
including 160 side-chain–side-chain long-gap lactam designs.

Additional structural commands:

```bash
astrolabe paper analyze cre --reference cre_reference.cif --structure designed_complex.cif
astrolabe paper analyze rna --kind oligomer --structure rna_assembly.cif
astrolabe paper analyze rna --kind bivalent --structure bivalent_complex.cif
```

### Ranking, sensitivity and paired-interface statistics

```bash
astrolabe paper analyze ligand-rank --records ligand_representatives.csv --out-dir analysis/ligand-rank
astrolabe paper analyze sensitivity --endpoint protein-binder \
  --planned planned.csv --records records.csv --denominator planned --out-dir analysis/protein-sensitivity
astrolabe paper analyze sensitivity --endpoint ligand-binder \
  --planned planned.csv --records records.csv --denominator planned \
  --comparison-methods Astrolabe-AF3 ProteinHunter --seed 0 --out-dir analysis/ligand-sensitivity
astrolabe paper analyze ddg --records paired_interfaces.csv --seed 0 --out-dir analysis/ddg
```

The ligand-ranking CSV contains `unit_id,method,group` and the seven metrics
`iptm,ptm,plddt,ddg,dsasa,ligand_hbonds,sc`. Each row is one **fixed** design
representative. For RF-family backbones, select the candidate once using the
original common-AF3 ipTM and the native candidate-ID tie-break before this step.
Ranking uses metric medians within each method/ligand, average ranks for ties,
equal weights for FAD/IAI/OQO/SAM, then equal weights for the seven metrics.
Only ddG is minimized. Missing values are omitted from that metric's median;
counts and incomplete aggregate ranks remain explicit in the CSV/JSON outputs.

Sensitivity uses the same planned/record schema and fixed selection as `scores`.
The frozen manuscript protein- and ligand-benchmark success-rate panels use
`--denominator planned`, retaining missing units as non-passing. The `available`
option is reserved for explicitly labelled measured-cohort diagnostics and is
not the manuscript success-rate denominator. SolubleMPNN variants stay within
their parent and must pass all applicable gates together.
The original protein grids vary ipTM, normalized Rg and normalized SAP; the
ligand analysis supplies 20,000-resample pointwise intervals and the Rg
1.15–1.45 rate-difference envelope. `--comparison-methods` names exact CSV labels.
The envelope is a sensitivity range, separate from a confidence interval.

The ddG CSV additionally requires `condition,molecule_type,ddg_ab,ddg_ac`.
Use separate conditions for simultaneous design and each single-target control,
and `molecule_type` values `minibinder` or `VHH`. It reports signed AC−AB,
absolute imbalance, quartiles and the median's 95% percentile interval from
5,000 design-level bootstrap samples. Both bootstrap commands record the seed
and accept `--bootstrap-samples`; defaults match the manuscript analyses.
Use repeated `--column NAME=CSV_COLUMN` options for explicit column mappings.

### Cross-predictor confidence and binding-mode agreement

```bash
astrolabe paper analyze agreement --manifest paired_structures.csv \
  --astrolabe-method Astrolabe --seed 20260930 --out-dir analysis/predictor-agreement
```

Each manifest row describes one fixed design and target side. Required headers
are `design_id,method,group,side,target_sequence,binder_sequence,af3_structure,protenix_structure`;
the two sides share a binder sequence. Structure paths are relative to the CSV.
Default confidence headers are `af3_iptm,protenix_iptm`, on a 0–1 scale. To add
ESMFold2, supply all mappings explicitly with repeated
`--confidence-column AF3=af3_iptm --confidence-column Protenix=protenix_iptm --confidence-column ESMFold2=esmfold2_iptm`.
The structures and confidence must refer to the same frozen representatives.

The analysis matches exact sequences, aligns target Cα atoms within 10 Å of the
AF3 binder, and computes pose RMSD, independently aligned binder RMSD and
5 Å heavy-atom residue-contact recall. Each design contributes its larger
two-side RMSD and smaller recall. Target-pair Spearman correlations and method
means follow the original estimability conditions; intervals use 2,000
design-level bootstrap samples within target pairs. Outputs also include
confidence-stratum medians/counts, group-balanced win probabilities (ties count
one half), and strict `>0.8` cross-predictor overlaps.

Mark incompatible target constructs with `comparable=false` and an
`exclusion_reason`, as for the frozen Chamaileon IL12/IL23 geometry comparison.
Their confidence records remain available. Missing structures and sequence
mismatches are reported as incomplete geometry and never counted as
disagreement. Eight CSV/JSON files retain per-side, per-design, per-group and
method-level results and the calculation parameters.

## External comparators

The native comparator and study-adapter entry points are listed in the
[baseline guide](../examples/paper/baselines/README.md). They include BindCraft,
RFdiffusion/ProteinMPNN, HalluDesign, ProteinHunter, BoltzGen, SwitchCraft,
Chamaileon, joint BindCraft*, RF-MSD*, ComplexDesign*, BoltzDesign1, RFAA,
RFdiffusion2 and RFdiffusion3. The study-specific
adapters are shipped; external model code, weights and native inputs must be
supplied through the launcher YAML and `--tool-dir`.

```bash
astrolabe paper baseline boltzgen \
  --inputs examples/paper/baselines/02_boltzgen.yaml \
  --tool-dir vendor/boltzgen --out-dir outputs/boltzgen --seed 1 --dry-run
```

The preview prints exact commands and missing assets. Remove `--dry-run` only
in the appropriate model environment. Methods whose native CLI has no model
seed option report that limitation explicitly. The current joint BindCraft*
implementation is separate from the old alternating cohort, whose frozen
driver was not recovered.

The four added ligand-generation launchers are explicitly labelled
`manuscript_native`: their interfaces and source pins were verified, while
the historical generation drivers were not recovered. The RF-family recipes
generate 100 backbones and five LigandMPNN candidates per backbone; the fixed
50-unit comparison subset and representative selection are subsequent analysis
steps. The archived RFAA display label maps to the `rf_diffusion_all_atom`
generator. BoltzDesign1 uses its pinned native defaults and explicit sampling
counts; its original iteration schedule and historical weight identity remain
unconfirmed. Exact required assets and these distinctions are printed in the
launcher plan and recorded in the baseline provenance.

For antibody runs, configure `ASTROLABE_TURBOAB_DB` for the VHH/IgG database,
`ASTROLABE_AF3_DATA_PIPELINE` as the AF3 `run_alphafold.py` path, and optionally
`ASTROLABE_AF3_PYTHON` for that runtime. Framework templates/MSAs are included
in the input bundle; the external search database and model runtime are not.

## Methods-to-configuration mapping

The unsuffixed protocols implement the current Methods and the author's
clarifications. `-archived` protocols preserve recovered historical settings.
Use the archive for an existing cohort whose provenance names that setting;
changing a recipe does not retrospectively change its measured results.
Checks compose Hydra defaults together with the supplied input overlays.

| Methods section | Formal configuration contract | Historical variant |
| --- | --- | --- |
| Protein-binder benchmark | DPLM monomer stage; LigandMPNN complex stage; `ipTM + 0.4 pTM + pLDDT + 0.2 SC - P`, with the stated Rg/loop penalty | `protein-binder-af3-archived` / `protein-binder-protenix-archived` retain the additional ESM3 operator |
| SAP-aware / IDR design | Interface-restricted and non-interface proposals remain restricted even when their selection is empty; task-specific objectives and representative rules are retained | `protein-binder-sap-archived`, `idr-binder-archived` |
| Dual-target minibinder / VHH | Independent per-complex ProteinMPNN probabilities are averaged at shared positions; VHH temperature is 0.15 and only CDRs are mutable | `dual-target-af3-archived`, `cross-reactive-vhh-archived` retain far-apart tied decoding |
| Single-target VHH controls | All-X CDR initialization; DPLM monomer shaping followed by LigandMPNN complex optimization; fixed framework, 100 states | `single-target-vhh-control-archived` retains the former single-stage objective |
| Ligand-associated two states | Protenix v0.5: 4 cycles, 200 diffusion steps, 5 samples; apo-specific entrance/penalty; holo-only second stage; 25 states | `two-state-af3-archived`, `two-state-boltz2-archived` retain both predictions in stage two and mean-Rg gating |
| Protein binders to nucleic acids | DPLM initialization; NA-MPNN proposals use HADDOCK3 context when no contact is present | `protein-binder-rna-archived`, `protein-binder-dna-archived` retain uniform starts without HADDOCK |
| RNA/DNA aptamers / RNA oligomers | Aptamer objective `ipTM + pLDDT + 0.4 pTM`; 20/40 states with selection from states 0–19; oligomer beam 8, 4 proposals per parent, 1,000 evaluations | Recovered input and scoring provenance are retained |
| Bivalent RNA | Random shared 20-nt repeat initialization; three starts requested with `--num-runs 3`; 200 evaluations per start | `bivalent-rna-archived` retains the three contact-shortened starts |
| Single-chain repeats | Three/four/five synchronized units; ProteinMPNN proposals throughout; AF2, 100 states | `tandem-repeat-3-archived`, `-4-archived`, `-5-archived` retain MCMC mixtures |
| Bound cyclic / multiring peptides | LigandMPNN proposals; bound objective `ipTM + 0.3 min(Q)`; multiring `pLDDT + 0.4 pTM + 0.3 mean(Q)`; topology overlays preserve this route | `cyclic-peptide-bound-archived` retains the mixed standard-residue route; `cyclic-peptide-multiring-archived` retains MCMC |
| Pocket rescue | Occupancy below 0.8 triggers Vina; first-ranked pose, no occupancy filter or retry; failures remain failed planned runs | `pocket-iptm-archived`, `pocket-oriented-archived`, `pocket-rescue-archived` |
| RFdiffusion–ProteinMPNN baseline | Eight sequences per backbone at temperature 0.3, fixed target, no hotspots; select highest common-AF3 ipTM afterward | The corrected launcher example is `03_rfdiffusion_proteinmpnn.yaml` |

For two-state refinement the author confirmed the holo-only score
`(holo_pLDDT + holo_pTM + holo_ipTM)/3 + 0.35*burial - 0.45*sigmoid((r_holo-1.10)/0.08)`.
The first-stage apo Rg term uses `A.binder_rog`, not the apo/holo mean.
Optimization retains the all-atom Rg definition from the Methods; the final
ensemble analysis uses C-alpha Rg as separately specified. The author also
confirmed the two-stage VHH control above and retained BindCraft*'s historical
single-proposal acceptance rule, detailed in the baseline guide.

Values not specified by the manuscript remain explicit implementation
parameters with their recovered sources, including bivalent beam/objective
details and the symmetry search's `ananas > 0.25` early stop. Symmetry success
still requires the separate, stricter conjunction in the Methods. Native
baseline settings supplied in external YAML must follow the requirements
printed by `paper baseline --dry-run`; a launcher cannot certify an unseen
external configuration. RNA structural examples additionally require the
manual geometric selection described in the manuscript.

Incorrect historical ordinary-terminal lactone/thiolactone control inputs are
documented in the cyclic input provenance, with corrected post-bond chemistry
provided for executable examples. CID's generic methods are implemented; its archived input cohort is not
included in this software release and requires explicit inputs.

This repository provides callable software and inputs, with explicit scientific
provenance. A complete new GPU rerun, frozen figure reproduction and a software
DOI are separate release artifacts. The [validation record](release_acceptance.md)
states the checks actually performed.