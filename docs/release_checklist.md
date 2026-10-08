# Paper-code release checklist

Use the [README](../README.md) as the installation entry point and
[reproducibility guide](reproducibility.md) for inputs, commands and expected
outputs. The [validation record](release_acceptance.md) distinguishes completed
software checks from scientific experiments that still need evidence.

1. **Source and installation.** Select the release commit. Install a built wheel
   in a fresh Python environment and run the README commands outside the source
   tree. Keep the full source archive available for model installation; the CPU
   wheel does not contain optional model stacks.
2. **Executable example.** Run `examples/01_cpu_reproduce.py` into a new output
   directory. Check its numerical values, SQLite record, selected structure and
   reports against the reproducibility guide. Record the environment and elapsed time.
3. **Automated checks.** Require the Python 3.11/3.12 CPU workflow to pass. Download
   the source archive, wheel and sdist from the same workflow run. CLI changes
   also require real Bash/Zsh completion checks as defined in that workflow.
4. **Scientific reproduction.** Freeze the paper's inputs, seed/cohort manifest,
   exact configurations and model versions. Provide figure/table commands,
   representative-selection rules, failure accounting and expected outputs.
   Validate a real trajectory for every backend described as tested.
5. **Citation, data and licenses.** Match the README and `CITATION.cff` to the
   manuscript authors and title. Document accessible data locations and any
   justified access restrictions. Keep `LICENSE`, `THIRD_PARTY_LICENSES.md` and
   the shipped upstream notices with the code; model weights retain their own terms.
6. **Versioned archive.** Create a release tag on the tested commit, attach the
   distributions, and archive that exact version in a persistent repository such
   as Zenodo. Add the resulting software DOI and the paper/preprint DOI when they
   exist, and use those identifiers in the manuscript's Code availability statement.

`python scripts/release/export.py /path/to/new/Astrolabe --archive dist/astrolabe-source.tar.gz`
creates the full source archive. The destination must be new. The exporter
excludes local task history, site overlays, caches, model weights and run outputs,
and records the exact vendor references needed by the installer.

`just third-party-licenses` regenerates the dependency notice inventory.
`just release-audit` is a source-hygiene helper; it requires initialized selected
vendor sources and does not establish model inference or manuscript reproduction.

## Basis for the release structure

The [Nature Portfolio reporting standards](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards)
require a Code availability statement for code central to a study and recommend
versioned, citable deposition that allows readers to repeat the results. There
is no mandatory repository directory tree. The task-oriented installation and
usage sections in [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
and the included inputs/examples in [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
provide practical examples from the same research area. Astrolabe retains its
existing Python package and configuration layout while supplying those reader-facing entry points.
