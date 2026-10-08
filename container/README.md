# Astrolabe container guide

Use this route for model inference. The [CPU installation](../README.md#install)
needs neither a container nor model weights. The container isolates CUDA/Python
dependencies; large model weights stay on the host and are mounted at runtime.

## Requirements

- Linux x86-64, Python 3.11+, Git, and network access to the pinned third-party
  repositories and package registries.
- A working Apptainer/Singularity installation. The installer can provision
  Apptainer when absent, but cannot supply administrator-controlled user
  namespaces, uid mappings, or device permissions.
- For inference: an allocated NVIDIA GPU and a compatible host driver.
- Sufficient local disk for the image, build cache, and selected assets. Model
  storage is separate; see the [software matrix](../needed_softwares.md).
- AF3 parameters obtained under their own terms. They are not redistributed.

Do not use an HPC login node for model inference. Build/run within the site's
approved allocation; the exact scheduler/account/partition is site-specific.

## Install AF3

Extract the supplied `astrolabe-source.tar.gz` and enter `Astrolabe/`. Unlike the
Python sdist/wheel, this full source archive includes the installer and recipes.
Then run:

```bash
python3 scripts/install.py \
  --root "$HOME/.astrolabe" \
  --profile local \
  --software alphafold3 \
  --af3-weights /absolute/path/to/alphafold3/parameters \
  --fakeroot \
  --activate-checkout
```

Replace the parameter path with your authorized local directory. This example
does not select PyRosetta or download unrelated model weights. The image recipe
still carries the common in-process runtime needed by the framework; selecting
one model limits selected assets and optional licensed features, not every base
Python package.

For a source archive, the installer verifies `SOURCE_EXPORT.json`, creates a
local Git index, and initializes exact `SUBMODULE_PINS.json` commits using
`.gitmodules`. Reviewed local changes are applied from `patches/vendor/`; the
source-owner inventory is `config/third_party_sources.yaml`. It does not fetch Astrolabe's main repository, add its remote,
or invent upstream commit history. Keep exported source files unchanged until
this first initialization; subsequent configuration edits are ordinary local
changes. A normal Git checkout uses its existing index instead.

The installer reuses its persistent build/download cache, builds or selects a
SIF, checks the image and selected assets, writes a site profile and runner,
and runs import/environment checks. `--activate-checkout` activates only after
those steps succeed. A failure is not a usable installation.

```bash
./scripts/astrolabe env doctor --task release_protein --strict
./scripts/astrolabe run --task release_protein
```

`release_protein` is a small AF3/random-substitution trajectory: one 80-residue
sequence, seed 1729, five steps. It is an installation example, not a validated
binder or reproduction of the paper's experimental results. Inspect the run's
`scores.csv`, structures, saved configuration and terminal status; merely
starting the GPU process is not acceptance.

## Installation modes and assets

```bash
python3 scripts/install.py --list-software
python3 scripts/install.py --help
```

| Need | Installer option |
|---|---|
| Select a backend | `--software <name>`; inspect the catalog for exact names |
| Reuse an image | `--skip-build --sif /absolute/path/image.sif` |
| Use a site container engine | `--apptainer-bin /absolute/path/apptainer` |
| Require an existing engine | `--bootstrap-apptainer never` |
| Reuse build downloads | `--container-build-cache /absolute/path/cache` |
| Download supported public model assets | `--download-public-assets` |
| Use existing AF2 / LigandMPNN / Protenix assets | `--af2-params`, `--ligandmpnn-params`, `--protenix-assets` |
| Expose another input/output directory | `--extra-bind /host/path:/container/path` |
| Pass explicit runtime settings | `--env-default KEY=VALUE`, `--prepend-env KEY=VALUE` |

Prebuilt SIF reuse requires matching build metadata, compatible Astrolabe
source/layout, and selected features. Keep adjacent build metadata with the
image; an arbitrary older image is not a substitute for a qualified release.

Restricted features require explicit opt-in. PyRosetta needs
`--accept-license pyrosetta-license` and the official pinned quarterly wheel;
use `--pyrosetta-wheel /path/to/authorized.whl` to provide it locally. Selected
AnAnaS and Rosetta tools have separate license gates. Keep images containing
restricted software private to the authorized site. See
[third-party environments](../docs/third_party_envs.md) and
[licenses](../THIRD_PARTY_LICENSES.md).

## Build or execution restrictions

| Failure | Action |
|---|---|
| `newuidmap` / `newgidmap`, user namespaces or fakeroot unavailable | Ask the site administrator for supported tooling, or use a compatible prebuilt SIF. Installing another engine alone does not grant kernel privileges. |
| Site provides Docker instead of usable host Apptainer | Use `--container-builder docker --runner-backend docker-apptainer`; Docker permission remains a prerequisite. |
| Direct OCI registry access is blocked | For Docker builds only, explicitly select `--docker-base-source docker-archive` to use a target-host Docker archive. |
| Remote Apptainer builds are configured | Use `--remote` instead of `--fakeroot`; a working client and remote service are still required. |
| No GPU is visible | Run inside the correct allocation and verify host driver/device access. Do not silently change the evaluator or precision. |
| Model assets or license acceptance missing | Supply the exact selected resources/consent. `--allow-missing-af3` is only for non-AF3 installation checks, not AF3 inference. |

For scheduler-based validation, `--srun 'srun ...'` prefixes validation commands
with the site's allocation command. Building a container and scheduling the
scientific trajectory are separate operations.

## Files, runners and source identity

With the installation root in the example above:

```text
~/.astrolabe/images/Astrolabe_af3.sif
~/.astrolabe/sites/local.json
~/.astrolabe/tools/gh_local_container_python.py
~/.astrolabe/cache/
```

The generated runner keeps its `gh_` filename for compatibility. Activation
writes `config/env/.astrolabe_python_path` and points `config/env/env_local.yaml`
to the portable container environment. Legacy `.gh_python_path` is read for
compatibility, not generated by the new installer.

- Calling the generated runner directly uses the source frozen inside the SIF
  at `/opt/Astrolabe`.
- Calling `./scripts/astrolabe` after activation binds the current source tree
  into the container. Record local changes with the run; a bind-mounted source
  is not evidence that the image itself was rebuilt.
- Without activation, select the runner explicitly:

```bash
GH_PY="$HOME/.astrolabe/tools/gh_local_container_python.py" \
  ./scripts/astrolabe env doctor --task release_protein --strict
```

## Maintainer builds

`scripts/install.py` is the supported complete entrypoint. The low-level
`scripts/containers/apptainer_build.py` consumes initialized, indexed source and
requires licensed/build inputs to be staged explicitly. `container/Astrolabe.def`
is the minimal skeleton; `container/Astrolabe.af3.def` is the AF3 runtime recipe.
Prefer the installer rather than invoking `apptainer build` on an unprepared
source directory, which may omit dependencies or copy unintended local data.

Keep vendor source revisions, image metadata, selected assets, configuration,
seed and native outputs with scientific execution records. Current validation
and outstanding release conditions are in
[release acceptance](../docs/release_acceptance.md).
