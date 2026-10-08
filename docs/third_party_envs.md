# Host-side third-party environments (resume + interactive)

For a new installation follow [README](../README.md): an isolated CPU package
for configuration/post-processing, and `scripts/install.py` for model
inference. This page also retains the host environment manager for existing
non-container deployments; it is not the default installation route.

Some third-party tools are best kept in **separate Python environments**, especially when they are invoked via `subprocess` and have conflicting dependency stacks.

This repo provides a resumable host-side env manager:

- `scripts/installation/env_setup.py`
- Spec file: `config/third_party_envs.yaml`

## Vendor submodules (required)

Third-party repos are vendored under `vendor/` as **git submodules**.
For the default `gh` runtime env, initialize the required runtime subset first:

```bash
git submodule update --init --recursive --depth 1 --jobs 4 \
  vendor/alphafold3 vendor/alphafold2 vendor/protenix vendor/rhofold \
  vendor/chai_lab vendor/boltz vendor/esm2 vendor/esm3
```

Use the repo checkout plus the reviewed submodule URLs recorded in
`.gitmodules`; do not upload a local source tree as the normal install source.
`scripts/installation/env_setup.py` fails early when a local vendored pip path is missing
or is not a Python project, which usually means submodules were not initialized
from their recorded remotes.

Optional/heavy backends such as OpenDDE, SurfDesign, SurfPro, NA-MPNN, OpenFold
ESMFold, RPXDock/WillUtil, and RhoDesign are not needed for the default runtime
bootstrap. Initialize them only for tasks that use those backends, or run the
full set explicitly:

```bash
git submodule update --init --recursive --depth 1 --jobs 4
```

AF2, Chai, and Boltz are fetched from upstream GitHub repositories, with compact
Astrolabe-specific behavior kept as explicit repo-owned patches under
`patches/vendor/` and applied by `scripts/maintenance/apply_vendor_patches.py` during
bootstrap/env setup. ESM2, ESM3, and RPXDock use Molaison forks because their
package-isolation or runtime-stability deltas are large enough to own in fork
history. Protenix uses public upstream v2 plus a compact repo-owned ESM2 patch.

Patch-managed submodules are marked `ignore = dirty` in `.gitmodules`. This
keeps the parent repository status readable after setup applies the expected
patches in-place. It does not skip patching or validation; use
`python scripts/maintenance/apply_vendor_patches.py --check` to verify that the current
vendor checkouts contain the required Astrolabe changes.

Environment specs must not clone or checkout directly into `vendor/<name>` when
that path is already a git submodule. Use `git submodule update --init` for the
source checkout, then let `config/third_party_envs.yaml` install from that
vendored path.

## Source modes and drift guard

Each third-party component should have one source owner. Pick one of these
modes before adding install logic:

| Mode | Owner | Use when | Drift guard |
|---|---|---|---|
| Official submodule plus patch | `.gitmodules` and `patches/vendor/<name>/` | The upstream repo is mostly usable and Astrolabe only needs small build or behavior changes. | Keep the patch in `scripts/maintenance/apply_vendor_patches.py` and validate with `python scripts/maintenance/apply_vendor_patches.py --check` after submodules are initialized. |
| Fork-backed submodule | `.gitmodules` points at an Astrolabe fork, with `branch` recorded when the reviewed fork delta is not on the fork default branch | The delta is large, long-lived, or mostly file/package rename work. | Record `expected_url` and any `expected_branch` in `config/third_party_sources.yaml`; do not also clone the same repo through `config/third_party_envs.yaml`. |
| In-tree vendored source | Direct files under `vendor/` | The dependency is small source code or a binary/tool subtree that is reviewed as repo content. | Track licenses through `scripts/release/generate_third_party_licenses.py`. |
| Path asset | `manager: path` or `manager: asset` in `config/third_party_envs.yaml` | The component is weights, cache, models, or external data rather than a Python package checkout. | Export only paths; do not encode repo state in the asset entry. |
| Isolated env | A separate env entry in `config/third_party_envs.yaml` | Dependency stacks conflict with the main `gh` env or a backend is subprocess-only. | Keep source paths explicit in `export:` and verify with `astrolabe env doctor --task <task> --strict`. |

`config/third_party_sources.yaml` is the source-mode inventory. It is checked
against `.gitmodules` and `scripts/maintenance/apply_vendor_patches.py` by tests and by
`scripts/installation/env_setup.py` before environment setup. When moving a dependency
from patch-managed to fork-backed source, update that inventory, then update
`.gitmodules`, and only then remove or shrink the local patch.
Every branch-backed entry also records `expected_commit`; validation compares it
to the committed superproject `HEAD` gitlink so a branch name or staged-only fix
cannot mask a stale published pin.

OpenDDE is integrated as an official submodule and subprocess evaluator. It can
use the main `gh` interpreter when compatible dependencies are already present;
`GH_OPENDDE_PYTHON` remains available for sites that need a separate env.
SurfDesign is an official submodule and persistent subprocess mutator; its
upstream dependency lock is isolated behind `GH_SURFDESIGN_PYTHON`.
SurfPro is also an official submodule and persistent subprocess mutator; its
legacy CUDA/SRU++ stack is isolated behind `GH_SURFPRO_PYTHON`.

Large rename, package-isolation, or runtime-stability patches belong in forks.
ESM2, ESM3, and RPXDock are now fork-backed submodules; the ESM forks mostly
rename packages so the main `gh` environment can import both backends without
collisions. Chai and Boltz should
remain patch-managed while the cyclic-relative changes stay compact; move them
to forks only if those patches keep growing. AF3 currently uses the
`Molaison/alphafold3` fork recorded in `.gitmodules`; its remaining local
third-party CMake patch is small enough to stay patch-managed on top of that
fork. AF2's recycle-state/cyclic-offset patch is small enough to remain
official submodule plus patch.

When a reviewed fork remote does not exist yet for a future patch-managed
component, materialize the current patch as a fork-ready worktree instead of
changing `.gitmodules` to a guessed URL:

```bash
just vendor-fork-worktree <patch-group> /tmp/gh-<patch-group>-fork
```

Each command creates a detached worktree with the Astrolabe patch committed on top of
the current vendored upstream checkout. Push that commit to the real reviewed
fork, then update `config/third_party_sources.yaml`, `.gitmodules`, and remove
or shrink the local patch group in the same change.

`THIRD_PARTY_LICENSES.md` is generated from the current `vendor/*` inventory.
Generate to a temporary path first when submodules are not initialized:

```bash
python scripts/release/generate_third_party_licenses.py --out /tmp/THIRD_PARTY_LICENSES.md
diff -u THIRD_PARTY_LICENSES.md /tmp/THIRD_PARTY_LICENSES.md
```

For release review, overwrite the committed file from a checkout whose intended
vendor components are initialized; otherwise `not_checked_out` rows can be
workspace state rather than release truth. A source-inventory-only refresh is
still acceptable from an uninitialized checkout when the diff is limited to
`.gitmodules` sources or missing vendor rows and the `not_checked_out` status is
intentional.

## Quickstart

Create the runtime environments under `~/.gh_envs`, merge their exported paths
into the ignored `config/env/env_local.yaml`, and verify those paths:

```bash
just env-setup
```

Then run with Hydra override:

```bash
astrolabe run --task test
```

## PyRosetta

PyRosetta is **not** installed by default via conda/pip deps because it is hosted on the PyRosetta4 archive (not PyPI).

This repo installs it via `scripts/installation/install_pyrosetta.py` (wired in `config/third_party_envs.yaml` under the `gh` env `commands:`):

```bash
# Install latest compatible wheel for your Python (inside the astrolabe env)
python3 scripts/installation/env_setup.py --only gh --rerun commands -y
```

Optional overrides:

- Exact wheel (local file or URL): `GH_PYROSETTA_WHEEL=/path/to/pyrosetta-...whl`

## OpenDDE subprocess evaluator

OpenDDE source lives in `vendor/opendde` and is invoked through a subprocess.
When `subprocess_envs.opendde.python_bin` / `GH_OPENDDE_PYTHON` is unset, the
wrapper uses the current `gh` interpreter. Initialize the source and assets before
using an `evaluator.evaluator_terms.*._type: OpenDDE` task:

```bash
git submodule update --init --recursive vendor/opendde
python3 scripts/installation/env_setup.py --only opendde_assets -y
```

If the main `gh` interpreter lacks OpenDDE-compatible dependencies, create the
optional isolated env as well:

```bash
python3 scripts/installation/env_setup.py --only opendde,opendde_assets -y
```

OpenDDE runtime data is rooted at `weight_path.opendde` / `OPENDDE_ROOT_DIR` and
should contain `checkpoint/opendde.pt`, `common/`, and optionally
`search_database/` for template or RNA-MSA preprocessing. The Astrolabe wrapper defaults
to `use_msa: false`, `use_template: false`, and `use_rna_msa: false` so mutation
loops do not accidentally call external feature searches. Enable those explicitly
when the OpenDDE data/tools are installed.

Example evaluator term:

```yaml
evaluator:
  evaluator_terms:
    OpenDDE:
      _type: OpenDDE
      num_samples: 1
      num_steps: 200
      num_cycles: 10
```

## SurfDesign persistent mutator

SurfDesign source lives in `vendor/surfdesign`. Initialize it, then point Astrolabe at
an environment prepared from the upstream `environment.yml` and at a trained
SurfDesign experiment directory:

```bash
git submodule update --init --recursive vendor/surfdesign
export GH_SURFDESIGN_PYTHON=/path/to/surfdesign-env/bin/python
export GH_WEIGHT_SURFDESIGN=/path/to/surf_design_esm2_650m
astrolabe env doctor --strict --task <task>
```

`GH_WEIGHT_SURFDESIGN` must contain `.hydra/config.yaml` and
`checkpoints/best.ckpt` (or the checkpoint filename selected under
`mutator.SurfDesign.checkpoint`). The runner keeps one model-loaded JSONL daemon
for the pipeline lifetime, generates the required PyMOL molecular surface for
each PDB backbone, and returns standard 20-amino-acid probabilities to Astrolabe.
PDB evaluator structures are consumed directly; mmCIF outputs such as OpenDDE's
are converted to PDB inside the isolated worker before surface generation. The
SurfDesign environment must include PyMOL plus the upstream surface-graph
dependencies.

```yaml
mutator:
  mutator_terms:
    surface_design:
      _type: SurfDesign
      position_selector: "surface()"

stages:
  default:
    mutators: [surface_design]
```

Upstream currently does not publish a trained SurfDesign checkpoint or a
top-level license file. Model assets must be supplied explicitly, and release
distribution remains blocked on license review as recorded in
`THIRD_PARTY_LICENSES.md`.

## SurfPro persistent mutator

SurfPro source lives in `vendor/surfpro`. Its upstream runtime pins Python 3.8,
PyTorch 1.10.2/CUDA 11.3, and SRU++ `3.0.0-dev`, so it runs in an explicitly
configured external environment rather than the main `gh` interpreter. Download
the published CATH 4.2 `checkpoint_best.pt` linked from the upstream README and
provide the MSMS toolchain used by upstream preprocessing:

```bash
git submodule update --init --recursive \
  vendor/surfpro vendor/sru vendor/protein_sequence_models
export GH_SURFPRO_PYTHON=/path/to/surfpro-env/bin/python
export GH_WEIGHT_SURFPRO=/path/to/surfpro-weights
export GH_SURFPRO_MSMS=/path/to/msms
export GH_SURFPRO_PDB_TO_XYZRN=/path/to/pdb_to_xyzrn
"$GH_SURFPRO_PYTHON" -m pip install "pip<24.1"
"$GH_SURFPRO_PYTHON" -m pip install -e vendor/sru
"$GH_SURFPRO_PYTHON" -m pip install -e vendor/protein_sequence_models
"$GH_SURFPRO_PYTHON" -m pip install -e vendor/surfpro
astrolabe env doctor --strict --task <task>
```

`GH_WEIGHT_SURFPRO` must contain `checkpoint_best.pt` unless
`mutator.SurfPro.checkpoint` selects another file. The worker runs annotated
`pdb_to_xyzrn` plus MSMS, then reuses SurfPro smoothing and
octree preprocessing. One model-loaded JSONL daemon serves the whole pipeline
and returns autoregressive probabilities in Astrolabe's standard 20-amino-acid order;
the autoregressive prefix is likewise constrained to those 20 residues.
PDB structures are consumed directly and mmCIF evaluator outputs are converted
inside the worker. SurfPro is CUDA-only because its upstream SRU++ encoder calls
CUDA directly. Use the annotated `pdb_to_xyzrn`, not `pdb_to_xyzr`; the official
script also requires `nawk` on `PATH` and `atmtypenumbers` beside the script.

```yaml
mutator:
  mutator_terms:
    surfpro_surface:
      _type: SurfPro
      position_selector: "surface()"

stages:
  default:
    mutators: [surfpro_surface]
```

The upstream repository has no top-level license file, so release distribution
remains blocked on license review as recorded in `THIRD_PARTY_LICENSES.md`.

## Chai precomputed MSAs (`.aligned.pqt`)

Chai can load precomputed MSAs from `msa_directory`, keyed by the query sequence hash (`*.aligned.pqt`).

This requires a Parquet engine in the `gh` env. We use `pyarrow` (installed via conda-forge in `config/third_party_envs.yaml`).

If you already created the `gh` env before `pyarrow` was added, re-run:

```bash
python3 scripts/installation/env_setup.py --only gh --rerun conda_install -y
```

## Interactive env.yaml fill + validation (task-aware)

To interactively fill missing `weight_path.*` / tool paths in `env_local.yaml` and validate immediately:

```bash
astrolabe env wizard --task benchmark_binder_v1_4
```

Then point your Hydra env config at the right interpreters for subprocess-only tools:

- Host config: `config/env.yaml` → `subprocess_envs.*`
- Container-friendly config: `config/env_container.yaml` → `${oc.env:...}` placeholders

Important: `subprocess_envs.*.python_bin` only affects subprocess runners. The
AF2 evaluator still imports its Python stack in-process, so the main `gh` env
prepared by `scripts/installation/env_setup.py` must contain TensorFlow. `astrolabe env doctor
--strict --task <af2-task> --python-deps` now checks that directly.

Example (host):

```yaml
subprocess_envs:
  meeko:
    python_bin: /home/user/.gh_envs/envs/meeko/bin/python
  dplm:
    python_bin: /home/user/.gh_envs/envs/dplm/bin/python
    repo_path: /home/user/repos/byprot_dplm
```

The values above can be generated automatically if your env spec includes `export:` mappings (see `config/third_party_envs.yaml`). `scripts/installation/env_setup.py` also writes `config/env/.gh_python_path` for `scripts/astrolabe` and the legacy `scripts/gh_dev` wrapper when the `gh` env is available.

Non-interactive (CI / batch):

```bash
python3 scripts/installation/env_setup.py -y --jobs 2 --profile runtime
```

Only a subset:

```bash
python3 scripts/installation/env_setup.py --only gh,boltz
```

## Resume behavior

Each environment keeps step state under:

```text
~/.gh_envs/.gh_env_setup/<env>/state.json
```

Re-running the script skips completed steps automatically.

### Incremental updates (recommended)

If you edit `config/third_party_envs.yaml` (e.g. add a new `pip:` spec or a new `commands:` entry),
you should **not** need `--rerun`.

`scripts/installation/env_setup.py` fingerprints the `pip:` and `commands:` lists and will:

- detect when those lists change
- execute only the *new/missing* items (itemized state under `pip_install::<hash>` / `commands::<hash>`)
- then re-mark the coarse step (`pip_install` / `commands`) as complete

If you really want to force re-run everything in a step, use:

```bash
python3 scripts/installation/env_setup.py --only gh --rerun pip_install -y
python3 scripts/installation/env_setup.py --only gh --rerun commands -y
```

## Existing envs

If an env already exists, the script asks what to do:
- use existing (continue/resume)
- mark done (skip all steps)
- remove and recreate
- skip

For fully non-interactive runs, pass `-y` (defaults to “use existing”).

If an entry uses `manager: external`, `prefix:` or `python_bin:` must point at an
existing Python env. Missing external envs are treated as setup errors, not as a
request to silently create a replacement env. Use `manager: micromamba` when the
spec should create the env itself.

If you want to pre-decide actions in an interactive session and then run setup non-interactively, use:

```bash
python3 scripts/installation/env_confirm.py --out decisions.json
python3 scripts/installation/env_setup.py --decisions decisions.json -y --jobs 2
```

## From-scratch runtime bootstrap

The default new-server path is the runtime profile. It skips optional/heavy
backends such as DPLM, RAPiDock, RPXDock/WillUtil, OpenFold ESMFold, NA-MPNN,
and RhoDesign unless you explicitly enable them for a task:

```bash
just env-setup
```

For the container-first path, use `scripts/install.py`. Container builds
reuse `<root>/cache/container-build` by default for target-server downloads
inside the SIF build. Do not replace that with local wheel/SIF uploads; override
the cache path with `--container-build-cache <path>` only to share a
server-side cache between official checkouts.

Select the exact runtime closure with repeatable/comma-separated `--software`:

```bash
python3 scripts/install.py \
  --software alphafold3,alphafold2,ligandmpnn,boltz \
  --af3-weights /approved/alphafold3/models \
  --download-assets
```

`python3 scripts/install.py --list-software` reports every public,
gated, restricted, bundled, and manual component. Public assets use
`<root>/assets`, `<root>/cache/torch`, and `<root>/cache/huggingface`; each
download retains `.part` bytes, uses a per-target lock, validates fixed
revision/SHA where available, and writes `<root>/assets/software-manifest.json`.
HTTP transfers verify `Content-Length`, `Content-Range`, and the final object
length before atomic promotion; a short `200`/`206` remains resumable, and a
`416` response cannot promote a short partial file. The retry budget scales with
asset size because every valid partial response makes reusable progress.
Fair-ESM ESM2 and ESM-IF checkpoints are pinned by exact size and SHA256;
ESMFold additionally pins and prefetches the ESM2-3B and contact-regression
files that its constructor loads, so model loading stays offline.
Deterministic missing/manual resource checks run before submodule initialization,
patching, environment installation, downloads, or SIF construction, so a bad path
does not waste a build attempt.
The manifest is a cumulative asset inventory; `selected`/`last_selection` records
the latest request, while previous verified software entries remain. `ready`
means resource validation passed, not that a weighted model smoke passed. The
implicit default is the same strict `core` plan as `--software core`: missing
assets, host envs, repos, or binaries fail instead of being silently omitted or
replaced by another version.

ESM3, DPLM, and ESMFold2 use their exact catalog snapshot directory at runtime
with Hugging Face/Transformers offline mode enabled. A cache containing another
revision does not satisfy the selected plan. Prefetch passes an exact
`allow_patterns` file set and writes `HF_HOME/gh-snapshots/<software>.json` with
the selected repo, commit, required-file sizes, and SHA256 values. Runtime uses a
curated exact snapshot under `HF_HOME/gh-snapshots/<software>/`; hardlinks reuse
the Hub blob inodes without duplicating model bytes and remain valid if Hub paths
are cleaned, while unrelated files remain outside the runtime view.

Use `--setup-host-envs` in the same command for selected subprocess backends. DPLM runs its
repo-owned `vendor/dplm/scripts/install.sh`; ESMFold2, OpenDDE, RAPiDock, and
RPXDock use the corresponding isolated entries in `config/third_party_envs.yaml`.
The core SIF remains the package owner for in-process backends, including the
default licensed PyRosetta feature. Exact software selections that omit PyRosetta
produce a reduced image. Selecting AnAnaS or Rosetta bakes the verified AnAnaS
binary or the 18-file `molfile_to_params` closure into the same private SIF; no
runtime host bind is used. The container-first path does not create a duplicate
host `gh` env.

For the legacy full spec, use `--profile all` and exclude only the backends you
do not want. Expect longer submodule/download/build times:

```bash
python3 scripts/installation/env_setup.py -y --profile all --exclude rapidock,rapidock_assets --write-env-yaml config/env/env_local.yaml --env-yaml-mode merge --verify-env-yaml
```

If a component has heavy install steps (e.g. `dplm`), you can do a dry bootstrap first:

```bash
python3 scripts/installation/env_setup.py -y --profile all --exclude rapidock,rapidock_assets --skip-commands --write-env-yaml config/env/env_local.yaml --env-yaml-mode merge --verify-env-yaml
```

## Repo/weights paths (manager=path)

`config/third_party_envs.yaml` can include non-env entries that only manage directories and repo checkouts:

- `manager: path` (or `manager: asset`) creates the directory (resumable).
- `pre_pip_commands:` run before editable `pip:` installs; this is where
  repo-owned vendor patches belong.
- `commands:` run after `pip:` installs, for generated data/build steps
  such as AF3 `build_data`.

For exports that may not exist yet (e.g., weights you must download manually), you can mark them optional:

```yaml
export:
  weight_path.alphafold3:
    value: "{{PREFIX}}/models"
    env_var: GH_WEIGHT_AF3
    optional: true
```

`export` mappings support three fields:

- `value`: generated default path.
- `env_var`: host environment variable to prefer when it is already set.
- `optional`: allow the generated default path to be absent during
  `--verify-env-yaml`.

When `env_var` is present and the host environment already defines it,
`scripts/installation/env_setup.py --write-env-yaml` writes that existing path instead of
the default `{{PREFIX}}`-derived path. This keeps downloaded weights and
preinstalled tool paths reusable across setup runs. Optional generated defaults
do not fail verification when absent, but an explicit env-var value is treated
as intentional and must exist when `--verify-env-yaml` is used.

`scripts/install.py` follows the same reuse rule for container asset inputs:
CLI flags still win, but when they are omitted the script reads host
`GH_WEIGHT_AF3`, `GH_WEIGHT_AF2`, `GH_WEIGHT_LIGANDMPNN`,
`GH_WEIGHT_PROTENIX`, `GH_WEIGHT_PROTENIX_CONSTRAINT`, `GH_PROTENIX_CCD`,
`GH_PROTENIX_CCD_RDKIT`, `GH_WEIGHT_BOLTZ`, `CHAI_DOWNLOADS_DIR`, `TORCH_HOME`,
`HF_HOME`, `GH_RHOFOLD_CHECKPOINT`, `OPENDDE_ROOT_DIR`, `GH_WEIGHT_RAPIDOCK`,
`GH_WEIGHT_RF3`, `GH_WEIGHT_RFD3`, `GH_WEIGHT_SURFPRO`, and `GH_USALIGN` before
falling back to managed defaults.

## Single-env mode

`config/third_party_envs.yaml` is configured for a **single** runtime env `gh` that installs all vendored evaluator deps.

Create it:

```bash
python3 scripts/installation/env_setup.py --only gh -y --jobs 2
```

ESM naming in single-env mode (avoid top-level `esm` collision):
- ESM2: `vendor/esm2` → `import esm2`
- ESM3: `vendor/esm3` → `import esm3`
- ESMFold: `from esm2.esmfold.v1.esmfold import ESMFold`

The unified SIF also carries the Python source from `vendor/openfold_esm2`, which
is required by ESMFold. It is path-loaded rather than compiled as a second
OpenFold package, avoiding another Torch/CUDA ABI stack.

Astrolabe exposes `esmfold` plus an independent `esmfold2` evaluator type. `esmfold2`
uses the official Biohub ESMFold2 model (`biohub/ESMFold2`) through a subprocess,
because it needs the Biohub ESM package and Biohub's newer/forked transformers
implementation. Configure it with an evaluator term such as `_type: esmfold2`;
defaults live under `evaluator.esmfold2` in `config/base.yaml`.

Initialize the source and runtime env:

```bash
git submodule update --init --recursive vendor/biohub_esm vendor/biohub_transformers
python3 scripts/installation/env_setup.py --only esmfold2 -y
```

For a site-managed runtime, export `GH_ESMFOLD2_PYTHON`,
`GH_ESMFOLD2_ESM_REPO`, and `GH_ESMFOLD2_TRANSFORMERS_REPO`. Override
`evaluator.esmfold2.model_name` to `biohub/ESMFold2-Fast` for the faster
single-sequence variant, or to a local Hugging Face model directory.

Run:

```bash
micromamba run -n gh --root-prefix ~/.gh_envs astrolabe run --task test
```

## RhoFold (RNA evaluator)

RhoFold is vendored under `vendor/rhofold` and installed into the `gh` env via `config/third_party_envs.yaml`.

The sequence-only Astrolabe route needs one runtime asset (env var or
`env_local.yaml` overlay):

- `GH_RHOFOLD_CHECKPOINT`: model checkpoint (`rhofold.checkpoint`)

`GH_RHOFOLD_BIN` and `GH_RHOFOLD_DB` are only needed if the currently disabled
BLAST/MSA preprocessing path is enabled. They are not downloaded or required by
the maintained sequence-only evaluator.

These keys live in `config/env.yaml` / `config/env_container.yaml`.

## RF3 and RFD3 (shared Foundry environment)

RF3 evaluation and RFD3 initialization use the same Foundry checkout and
subprocess Python. Configure the shared runtime plus the two model checkpoints:

```bash
export GH_RF3_REPO=/path/to/foundry
export GH_RF3_PYTHON=/path/to/foundry/.venv/bin/python
export GH_WEIGHT_RF3=/path/to/rf3_checkpoint.ckpt
export GH_WEIGHT_RFD3=/path/to/rfd3_latest.ckpt
```

The bootstrap preserves the lexical `.venv/bin/python` path. For uv-style
absolute interpreter symlinks it binds the venv, its original shebang path, and
the uv base interpreter prefix; resolving the symlink to a bare base Python
would lose Foundry's site-packages and is rejected by the runtime smoke.

`initializer.mode: rfd3` invokes RFD3 in that isolated environment, writes the
generated PDB under the run's `structures/rfd3_initializer/` directory, and then
passes it to `initializer.sequence_designer`. The sequence designer is a normal
mutator alias, typically `_type: LigandMPNN` with `model_type: protein_mpnn` or
`ligand_mpnn`. The nested `initializer.rfd3.specification` accepts native RFD3
fields. Astrolabe fills missing length/contig fields from `seqs`, maps fixed `+` motifs
through `motif_map` or an unambiguous input-sequence match, and preserves fields
such as `select_hotspots`, `select_fixed_atoms`, and `infer_ori_strategy`.

## NA-MPNN (protein/RNA/DNA mutator)

NA-MPNN is vendored under `vendor/na_mpnn` (the default checkpoint ships in the submodule).

To enable it in a task config, add a mutator term with `_type: nampnn`. Runner knobs are under `mutator.NAMPNN` (see `config/base.yaml`).

The Astrolabe integration is configured for **mixed polymer design** by default:

- `design_na_only: 0`
- `parse_na_only: 0`
- `na_shared_tokens: 0`

This lets one mutator design protein, RNA, or DNA chains in the same oligomer while still respecting Astrolabe-side fixed segments / locked positions.

Runtime notes:

- NA-MPNN imports `prody`, so the `gh` env spec includes `prody>=2.6.1`.
- `scripts/installation/env_setup.py --write-env-yaml ...` exports `subprocess_envs.nampnn.python_bin` and, when available, `subprocess_envs.nampnn.repo_path` into `env_local.yaml`. The repo path is optional for runtime-profile verification.
- The runner fails fast if `vendor/na_mpnn` is missing; initialize
  `vendor/na_mpnn` before running tasks with `_type: nampnn`.
- If you update `config/third_party_envs.yaml`, refresh the `gh` env with:

```bash
python3 scripts/installation/env_setup.py --only gh -y
```

Constraint:

- `na_shared_tokens=1` is not valid when the same design call includes both RNA and DNA chains. Keep the default `0` unless you know the target oligomer is nucleic-acid homogeneous.

## Smoke tasks

The maintained container smoke configs are:

- `config/container_smoke.yaml` (task: `container_smoke`)
- `config/container_af3_smoke.yaml` (task: `container_af3_smoke`)
- `config/container_af3_smiles_smoke.yaml` (task: `container_af3_smiles_smoke`)
- `config/container_rhofold_smoke.yaml` (task: `container_rhofold_smoke`)

Example:

```bash
GH_PY=~/.gh_envs/tools/gh_h100_container_python.py \
  scripts/astrolabe env doctor --task container_smoke --env config/env_container.yaml --resolve
```

Release validation separates command discovery from handler execution:

```bash
python scripts/validation/sif_cli_smoketest.py --sif /path/to/release.sif
python scripts/validation/sif_cli_execution_smoketest.py --sif /path/to/release.sif
```

The first command checks every parser leaf, alias, help path, fixed choice, and
completion surface. The second runs one non-help case for every canonical leaf,
including real SQLite/report fixtures and a tiny PyRosetta plus HOLE channel
screen, while keeping model-family commands on their explicit dry-run paths.

The maintained container path is task-scoped. It does not install HMMER for the
official AF3 MSA/data pipeline, and it does not build RAPiDock or AutoDock-GPU
unless a future task explicitly supplies those external tool paths.

Protenix is vendored from the public `bytedance/Protenix` upstream recorded in
`.gitmodules`, with a small repo-owned ESM2 import patch. The default checkpoint targets the v2 release model,
`protenix-v2.pt`. The in-process Astrolabe wrapper still accepts `weight_path.protenix`
as a checkpoint file path; for upstream Protenix cache compatibility, keep
assets under a Protenix root containing `checkpoint/` and `common/`, or let the
wrapper create compatibility links under `~/.gh_envs/repos/Protenix`.

## What can live in separate envs / external tools (subprocess)

- DPLM: invoked via `astrolabe/mutators/wrappers/dplm_runner.py` (defaults to a reusable daemon via `dplm_daemon.py`) using a configured `python_bin`.
- Meeko helper scripts: in AutoDock-GPU/Vina predockers we run `mk_prepare_*` via the configured `mk_prepare_python_bin` (wired from `subprocess_envs.meeko.python_bin`).
- Docking executables: OpenBabel / Vina are provided by the maintained
  container; AutoDock-GPU / AutoGrid are explicit external paths for
  `_type: autodock_gpu`. They are invoked via `subprocess` in
  `astrolabe/structure_processors/autodock.py:1` and configured from `config/env.yaml`.
- DSSP (mkdssp) + USalign: invoked via subprocess in `astrolabe/losses/structure.py` and are configured from `config/env.yaml` (`dssp_path`, `usalign_path`). The portable default points at the tracked Linux `vendor/mkdssp` convenience binary, and the repo also tracks DSSP/libcifpp source inventory for license/provenance review. Set `GH_DSSP` when you need a site-specific binary.
- AnAnaS: invoked via subprocess by the repeat-symmetry losses. Selecting
  `ananas` in the container bootstrap downloads the exact academic v1.1 archive
  after explicit license acceptance and installs `/usr/local/bin/AnAnaS` in the
  private SIF. Host-only setups may still set `GH_ANANAS`.
