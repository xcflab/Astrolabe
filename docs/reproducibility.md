# Reproducibility

This guide records the inputs, environment and execution boundaries for the
accompanying Astrolabe manuscript and the optional installation diagnostic. The current software version
is 0.1.0. Save the exact source commit, resolved configuration and run provenance
with any results.

## System requirements and installation

| Workflow | Requirements |
| --- | --- |
| CPU demo and result analysis | Python ≥3.11; Linux CPU workflow covers 3.11/3.12; no GPU or model weights |
| Model-backed design | Linux x86-64, selected predictor runtime, compatible NVIDIA driver/GPU, model parameters and task inputs |
| Container setup | Git, network access to selected pinned sources and assets, Apptainer/Singularity or a supported build backend |

Install Astrolabe using the commands in the [README](../README.md#install).
Python dependencies are declared in `setup.py`. The container runtime is recorded
in `container/requirements.runtime.lock` and `container/conda-linux-64.lock`;
third-party sources and patches are described in `config/third_party_sources.yaml`.
A full source export also supplies `SUBMODULE_PINS.json`.

For models, follow the [container guide](../container/README.md#install-af3).
Runtime, GPU memory, storage and setup time depend on the selected backend and
input size. The [software matrix](../needed_softwares.md) documents those components.

## AF3 installation example

After preparing the selected runtime and assets:

```bash
./scripts/astrolabe env doctor --task release_protein --strict
./scripts/astrolabe run --task release_protein
```

`config/release_protein.yaml` specifies one 80-residue sequence, seed 1729 and
`steps: 5`. A complete trajectory contains initial state 0 and at most four
proposal evaluations. Check scores, coordinates, resolved configuration and the
terminal provenance; retain the sequence from the native score records.
This example establishes a model installation only when it is actually run and
its outputs checked. The [validation record](release_acceptance.md) states which
checks have been completed.

## Manuscript workflows and sources

The [paper workflow guide](paper_workflows.md) provides callable generation,
fixed-sequence evaluation, structural analysis and planned-unit accounting.
The [input catalogue](../examples/paper/README.md) links the target, framework,
assembly and chemical-topology overlays. Bundled input archives have SHA-256
and original-source manifests in `resources/paper_inputs_*.json`.

```bash
astrolabe paper prepare-inputs --out-dir data/paper
astrolabe paper list
astrolabe paper show dual-target-af3
```

Protocol source references and manuscript/archived differences are preserved
with the configurations. The manuscript is *Astrolabe: Programmable all-atom
biomolecular design through cross-context coordination and structural intervention*,
by Zhengpeng Zeng and Chunfu Xu; no preprint or DOI is available yet.

The reviewer-facing manuscript data release has a separate
[Drive data entry](https://drive.google.com/drive/folders/1XxYyYHEdhOdf3sTbDRnuqBhaVj4FN3SX)
and [reader guide](https://drive.google.com/file/d/1tXbp78RxrsFxpZrsBOBkCdLw6ZJxmpmb/view).
The root contains cleaned figure-linked data, the planned-denominator protein
benchmark table, the complete 2,720-design target-bound cyclic-peptide cohort,
release checksums, the updated Figure 5 and a bundled reviewer ZIP. Historical
unredacted archives are isolated under `internal_audit_archive_20261009`; they
remain available for author-side traceability but are not reviewer-facing data.
Original model outputs and archived cohorts remain distinct from newly generated
runs; input availability and software checks do not constitute a new GPU rerun.

All benchmark success fractions use planned design units as denominators. A unit
without a source record or evaluable representative remains in its method–target
or method–ligand cell and counts as non-passing. Metric distributions and metric
ranks may use the observed values for the relevant metric, with their effective
counts reported explicitly. `parent_id` is the stable join key for the underlying
initial design or run: identical IDs indicate paired baseline, SAP-aware or
post-design-control records. Baseline and SAP-aware IDs match for 12 protein
targets; ORM2 uses an independently sampled baseline cohort and a paired
SAP-aware/post-design-control cohort.

Save the source commit, resolved configuration, exact model checkpoint, seeds,
input hashes, native predictions, planned units and declared selection rule
with each experiment. Final figure reproduction uses the matching frozen
cohort, including generation failures in its specified denominator.

## Optional installation diagnostic

The following small CPU calculation checks geometry and result handling. The
paper's scientific workflow entry points are described above.

### CPU coordinate check

```bash
python examples/01_cpu_reproduce.py --out-dir outputs/cpu-example
```

The output directory must not already exist. The input is
`examples/input/9TP_cid.cif`, a supplied AF3 prediction, not an experimental
structure. The calculation uses chain A residues 1–85 and chain B residues 1–87,
a distance cutoff of 8 Å, and no random sampling. All 85 × 87 residue pairs form
the contact denominator. The provenance records the input digest, Python/package
versions and calculation parameters.

### Expected numerical results

| Quantity | Expected value |
| --- | ---: |
| Residue-pair denominator | 7,395 |
| Cα contact pairs at ≤8 Å | 19 |
| All-atom residue contact pairs at ≤8 Å | 205 |
| Cα minimum distance | 5.622136021060697 Å |
| All-atom minimum distance | 2.3769934158267843 Å |
| Cα contact fraction | 19 / 7,395 |
| All-atom contact fraction | 205 / 7,395 |

The existing regression uses a 1e-6 Å absolute tolerance for the distances and
1e-12 for the contact fractions. It independently calculates Cα distances from
the coordinates. These values test geometry and result handling, not design
success, binding affinity or experimental activity.

### Reference runtime

On Linux x86-64 with an Intel Xeon Silver 4210R CPU and CPython 3.11.13, the
15 distribution, CLI and CPU-demo checks completed together in 16.87 seconds.
This measurement includes the complete demo with already-installed dependencies;
it excludes dependency downloads and model setup. GPU workflow runtimes depend
on the predictor, molecule sizes and hardware.

### Output files

All paths below are relative to `outputs/cpu-example/`.

| Path | Content |
| --- | --- |
| `raw/9tp_geometry/001_run/scores.csv` | Direct geometry measurements |
| `raw/9tp_geometry/001_run/config.yaml` | Analysis configuration |
| `raw/9tp_geometry/001_run/provenance.json` | Input identity, versions and parameters |
| `scores.sqlite` and `scores.csv` | Collected record and CSV export |
| `selected/filter_report.csv` and selected CIF | Selection using `ca_allowed_min_dist <= 8` |
| `report/posthoc_report.md` | Report with summary tables and plots |

The demo should produce one collected record and one selected structure. It
preserves existing output directories by refusing to overwrite them.