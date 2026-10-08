# Astrolabe

[![CPU checks](https://github.com/Molaison/Astrolabe/actions/workflows/release-cpu.yml/badge.svg)](https://github.com/Molaison/Astrolabe/actions/workflows/release-cpu.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Astrolabe is a programmable framework for all-atom biomolecular design. It
coordinates structure prediction and sequence design across molecular contexts,
combines user-defined objectives, synchronizes shared sequences, and applies
triggered structural interventions before the next sequence proposal.

**Accompanying manuscript**  
*Astrolabe: Programmable all-atom biomolecular design through cross-context coordination and structural intervention*  
Zhengpeng Zeng and Chunfu Xu  
The manuscript is currently unpublished; no preprint or DOI is available yet.

[Paper workflows](docs/paper_workflows.md) · [Installation](#install) · [Examples](examples/README.md) · [Reproducibility](docs/reproducibility.md) · [中文文档](docs/index.md) · [Citation](#citation)

## Install

Astrolabe requires **Python 3.11 or newer**. Scientific design uses the selected
GPU predictors, sequence models and structural tools. The package also provides
configuration, input preparation and result analysis on a CPU.

```bash
git clone https://github.com/Molaison/Astrolabe.git
cd Astrolabe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
astrolabe --help
```

The repository may require access through your GitHub account while the
manuscript is in preparation. A supplied `astrolabe-source.tar.gz` can instead
be extracted with `tar -xzf astrolabe-source.tar.gz`; run the same installation
commands from its `Astrolabe/` directory. The wheel and Python sdist install the
CPU package. The full source archive also includes model installers and recipes.

To inspect a composed design configuration without running a predictor:

```bash
astrolabe run --task release_protein --dry-run env=env
```

The source-checkout entry point is `./scripts/astrolabe`.

## Run the paper workflows

Install the required model backends with [scripts/install.py](container/README.md#install-af3)
and activate the checkout. The [software matrix](needed_softwares.md) lists each
model's assets and runtime. Then use the common manuscript entry point:

```bash
./scripts/astrolabe paper list
./scripts/astrolabe paper prepare-inputs --out-dir data/paper
./scripts/astrolabe paper run rna-oligomer-4 --out-dir outputs/rna-c4
```

The last command runs NA-MPNN/AF3 beam refinement from the supplied 87-nt RNA
candidate. It uses the archived model settings and a budget of 1,000 AF3 calls.
It requires the configured AF3 and NA-MPNN runtime and weights. Use `paper show
rna-oligomer-4` to inspect the protocol, or append `--dry-run` to inspect a run.

| Scientific task | Command family / protocol |
| --- | --- |
| Protein binders, SAP-aware design, IDR and split-domain Cre | `protein-binder-*`, `idr-binder`, `cre-binder-*` |
| Shared minibinders and cross-reactive VHHs | `dual-target-*`, `cross-reactive-vhh`, single-target controls |
| Ligand binding, apo/holo design, pocket retargeting and CID | `ligand-binder-*`, `two-state-*`, `pocket-*`, `cid-*` |
| C3–C6 / D2–D3 assemblies and synchronized repeats | `symmetry-*`, `pseudo-c3`, `tandem-repeat-*` |
| Single-ring and multiring peptides | `cyclic-peptide-*` |
| RNA/DNA aptamers, nucleic-acid targets and RNA beam search | `rna-aptamer`, `dna-aptamer`, `protein-binder-rna/dna`, `rna-oligomer-*`, `bivalent-rna` |
| Fixed-sequence evaluation and paper endpoints | `evaluate-*`, `paper collect`, `paper analyze` |

All protocol names are arguments to `astrolabe paper run`. Target, topology and
framework inputs are under [examples/paper](examples/paper/README.md). Bundled
MSAs, templates and chemical definitions have source and SHA-256 manifests;
model weights are separate. Each normal invocation runs one trajectory. Use
`--num-runs` to request a cohort explicitly.

The [paper workflow guide](docs/paper_workflows.md) gives input paths, model
requirements, budgets, representative rules, outputs and analysis commands.
Archived recipes preserve settings recovered from the original runs. Where the
manuscript text and archived implementation differ, the guide identifies the
difference and names the corresponding variant.

## Python API

The public application accepts a composed Hydra `DictConfig`:

```python
import astrolabe
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=str(astrolabe.config_dir()), version_base=None):
    cfg = compose(config_name="release_protein", overrides=["env=env"])

astrolabe.Astrolabe(cfg).run()
```

This API example runs AF3 and requires its configured runtime and assets.
See the [user guide](docs/user_guide.md) for configuration and the
[developer guide](docs/developer_guide.md) for the design loop and component APIs.

## Repository layout

```text
astrolabe/       Python package and design components
config/          Portable task and environment configurations
examples/        Runnable demo, example inputs and benchmark helpers
scripts/         Installation, container, release and maintenance tools
container/       Container recipes and dependency locks
tests/           Software and numerical regressions
docs/            Usage, API, reproducibility and release documentation
vendor/          Third-party source and pinned submodules
```

Within `astrolabe/`, evaluators, mutators, initializers, selectors, losses and
structure processors have separate modules. Configuration type keys retain their
scientific meaning. [Migration notes](docs/index.md#从旧接口迁移) cover earlier
`hallucination` and `gh` interfaces.

## Reproducing the manuscript

The [reproducibility guide](docs/reproducibility.md) links the executable protocols,
input bundles, analysis commands and source records. The [paper workflow guide](docs/paper_workflows.md)
records differences between manuscript descriptions and archived runs. Software
checks and comparisons with existing outputs are reported in
[release validation](docs/release_acceptance.md); they do not represent a new GPU
rerun of every manuscript cohort.

For a short installation diagnostic, the existing coordinate-analysis example
remains available:

```bash
python examples/01_cpu_reproduce.py --out-dir outputs/cpu-example
```

It reads supplied coordinates and checks geometry, collection and reporting.
The reproducibility guide lists its expected numerical outputs.

## Citation

Use [CITATION.cff](CITATION.cff) for machine-readable software citation metadata:

```bibtex
@software{zeng_astrolabe,
  author  = {Zeng, Zhengpeng and Xu, Chunfu},
  title   = {Astrolabe},
  version = {0.1.0},
  url     = {https://github.com/Molaison/Astrolabe}
}
```

Record the commit or archived release used for an experiment. The paper and
archived-software DOI will be added when available.

## License and support

Astrolabe is distributed under [Apache-2.0](LICENSE). Third-party source and
model assets retain their own terms; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
For usage or installation problems, open a [GitHub issue](https://github.com/Molaison/Astrolabe/issues)
with the software version, command, environment and relevant error output.
