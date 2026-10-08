# Astrolabe release validation

Astrolabe 0.1.0 provides a Python package, a command-line interface, a complete
source archive, and separate installation of selected model backends.
This page records the validation scope; runnable instructions and expected
outputs are in [reproducibility](reproducibility.md).

## Methods alignment checks (2026-10-09)

The current release has 77 design/evaluation protocols, including explicitly
named historical variants, plus 14 comparator launchers, 195 input overlays
and four input archives. The [Methods mapping](paper_workflows.md#methods-to-configuration-mapping)
records the manuscript revisions, corrected defaults, effective input layering
and author-confirmed choices.

The full source regression ran 2,103 cases: 2,078 passed, 22 failed and three
were skipped. All 22 failures subsequently passed in a focused run of 108
tests (45.26 seconds), giving 2,100 cases with passing results. Two new formula
tests needed the score engine's normal attribute-based context; the other
failures were test-environment bindings: the host Zsh's PCRE library, a writable
Protenix cache root, and a test-only CUDA-preinitialization override. An earlier
collection attempt also exposed a NumPy precedence conflict; the scientific
container's dependencies now take precedence over the auxiliary test tools.
No production dependency versions or scientific algorithms were changed for
these environment corrections.

Checks cover the apo-specific transition gate, a holo-only objective requiring
no apo descriptors, VHH monomer-to-complex MSA caching, per-context sequence
probability averaging, manuscript mutators after input-overlay composition,
archived entrypoints, and real Bash/Zsh completion round trips. The three
skips remain the two CUDA cases and the opt-in PyRosetta integration case.
These checks establish the configuration/software contract; no new GPU paper
cohort was generated.

## Initial paper workflow expansion (2026-10-09)

The initial expansion supplied 64 design/evaluation protocols, 14 external
comparator launchers, 195 input overlays and four checked input archives.
Manuscript revisions are recorded in `resources/paper_sources.json`; method
differences and missing historical parameters are described in
[the workflow guide](paper_workflows.md).

| Boundary | Observed result |
| --- | --- |
| Full source regression and corrective rechecks | The complete pass ran 2,069 tests: 2,052 passed, 14 failed and three were skipped. All 14 failed cases subsequently passed after correcting integration contracts and completing the test environment. The focused pass accepted 210 cases; its remaining three environment failures passed in the final recheck. |
| CLI execution matrix | All 57 canonical leaf commands parse. The 14 paper commands execute against prepared files: analyses run calculations, input preparation extracts an archive, and model launchers validate real configurations through dry-run. Four focused matrix tests passed. |
| Completion | Parser-derived protocol, comparator and analysis choices pass real Bash and Zsh source round trips. The full completion file was rerun; the Zsh initialization test also passed with its complete shell runtime. |
| Paper numerical methods | RNA geometry/beam, two-state ensembles, cyclic bond checks, protein protocols, pocket transforms, planned-unit endpoints, seven-metric ranking, sensitivity and predictor agreement passed their tests. An existing bivalent RNA example also matched 11 original metrics to a maximum absolute difference of 2.43e-11. |
| Distribution payload | Wheel checks cover paper configurations, input resources and comparator adapters/patches. The wheel includes the paper examples and excludes unrelated example projects, model code, local overlays and generated outputs. |

The corrective checks covered the new CLI matrix, the curated wheel payload,
canonical framework/manifest fields, and explicit first-pose caching in paper
recipes while preserving the general processor's existing cache behavior.
The scientific CPU test environment used CPython 3.11, initialized pinned
dependency sources and the repository's vendor patches. It also needed writable
temporary/cache directories, the complete Zsh runtime, existing Protenix CCD
assets and the compatible Fortran libraries used by the bundled DAlphaBall.
These were test-environment bindings, with no new model jobs or SIF builds.

The three full-suite skips were two CUDA-only tests and the opt-in PyRosetta
ligand-unsatisfied-contact integration test. No new GPU manuscript cohort was
run. Launcher tests establish command/configuration behavior; their numerical
model results still require the declared model environment and inputs.

## Established validation

| Boundary | Observed result |
| --- | --- |
| CPU installation | Non-editable installation from an exported source tree and a built wheel; imports, packaged configurations and framework resources work outside the checkout; dependency check passes. |
| CPU analysis | The bundled coordinate example computes geometry, collects records into SQLite, filters a structure and writes CSV/Markdown reports. |
| CLI and completion | Canonical `astrolabe` entry point, configuration composition and 79 completion regressions, including real Bash and Zsh round trips, passed during the namespace migration. |
| Component refactor | The full source regression ran 1,920 cases: 1,906 passed, 12 failed because optional source dependencies were not initialized, and two CUDA cases were skipped. After preparing those dependencies, all 12 failed cases passed. Total: 1,918 passing cases and two CUDA skips. |
| Source distribution | A full source archive bootstraps the installer without the main repository history. It records 31 vendor pins; wheel and sdist support the CPU package. |
| Model dependency sources | AlphaFold2 and Protenix use retrievable upstream commits plus shipped patches. Applying those patches reproduced the previously used dependency trees. |

The table summarizes earlier executed checks on the Astrolabe publication
source. It is not a claim that every check was repeated after each documentation
edit. Current CPU builds and their downloadable artifacts are available through
[GitHub Actions](https://github.com/Molaison/Astrolabe/actions/workflows/release-cpu.yml).

## Fresh repository checks (2026-10-08)

The independent Astrolabe repository passed 15 distribution, CLI and real-coordinate
CPU-demo tests in 16.87 seconds on CPython 3.11.13. The checks cover wheel contents,
installed resource paths, a full source export, archive bootstrap and the demo's
numerical results. `CITATION.cff` also passed the official CFF JSON Schema.

The corresponding command is:

```bash
python -m pytest -q tests/test_distribution.py tests/test_release_cli.py tests/test_cpu_release_example.py
```

## Reproducible CPU reference

The included `examples/input/9TP_cid.cif` is an existing AF3 prediction.
For chains A (85 residues) and B (87 residues), the 8 Å contact calculation has
7,395 residue pairs, 19 Cα contacts and 205 all-atom contacts. The respective
minimum distances are 5.622136021060697 Å and 2.3769934158267843 Å.
The regression independently computes Cα distances from the input coordinates.

These results validate coordinate analysis and result handling. They are not
new designs, experimental activity measurements or a paper-level success rate.

## Remaining scientific release evidence

- A newly built image and an end-to-end trajectory must be validated for each
  model backend advertised as tested in that release. The AF3 installation
  example is documented in [the container guide](../container/README.md#install-af3).
- The two skipped CUDA-dependent numerical cases require a suitable GPU runtime.
- Figure-level reproduction requires frozen inputs, cohorts, configurations,
  model assets, representative-selection rules and expected results. See
  [the manuscript reproduction boundary](reproducibility.md).
- The manuscript has no preprint or DOI yet. Its title and author order are
  recorded in the README; `CITATION.cff` identifies the software and repository.
  A version tag and an archived software DOI should identify the final paper release.

The CPU workflow in `.github/workflows/release-cpu.yml` defines the portable
Python 3.11/3.12 checks. GPU inference and manuscript experiments have their own
inputs and acceptance conditions; they are not run by that workflow.
