# Third-party resources and container runtime

The default new-server path is `scripts/install.py`: build/reuse the SIF,
download public assets, write a site profile, and generate the `GH_PY` runner on
the target server. The AF3 SIF installs repo-required in-process Python
dependencies from vendored source; do not patch a finished bootstrap with manual
`pip install` unless you are debugging a new dependency.
SIF builds use a persistent target-server cache at
`<root>/cache/container-build` by default. That cache is mounted at `/cache`
during the build and is used for pip wheels and micromamba packages.

Large model weights and site tools stay outside the image and are mounted:

- AF3 weights: private, pass with `--af3-weights`.
- AF2 params: pass an existing data dir/`params/` dir with `--af2-params`, or let
  `--download-public-assets` place them under `~/.gh_envs/assets/alphafold2`.
- LigandMPNN and Protenix public assets: handled by `--download-public-assets`.
- Rosetta/libgfortran3/site tools: pass explicit host paths when that workflow is
  enabled.

## AlphaFold 3

Vendored source owner and upstream:

```text
https://github.com/Molaison/alphafold3
forked from https://github.com/google-deepmind/alphafold3
```

Official upstream reference:

```text
https://github.com/google-deepmind/alphafold3
```

Follow their install guide and Dockerfile (used as the reference base for container builds):

```text
https://github.com/google-deepmind/alphafold3/blob/main/docker/Dockerfile
```

Notes:
- AF3 model parameters must be obtained from Google per their terms.
- Vendored AF3 fork code and Python deps are installed in `Astrolabe.af3.def`.
- The GH container does not install HMMER. GH's maintained AF3 evaluator path
  constructs folding inputs directly and runs inference; it does not call the
  official AF3 MSA/data pipeline. HMMER/database paths are required only if a
  future task explicitly enables that external data-pipeline workflow.

## Full Backend Source Set

The default bootstrap keeps optional/heavy backends out of the critical path.
Use `scripts/install.py --needed-softwares full` only when the selected
tasks need these subprocess or source-backed tools. For host-side env creation,
use `scripts/installation/env_setup.py --profile needed_softwares` or `--profile all`.

| Backend | Source/env | Needed for |
|---|---|---|
| DPLM | `vendor/dplm` or `GH_DPLM_REPO`; runner Python from `GH_DPLM_PYTHON` | DPLM mutator subprocesses. |
| NA-MPNN | `vendor/na_mpnn` or `GH_NAMPNN_REPO`; runner Python from `GH_NAMPNN_PYTHON` | RNA/DNA/protein mixed mutator subprocesses. |
| RPXDock | `GH_RPXDOCK_REPO` plus `GH_WILLUTIL_REPO` and matching `GH_RPXDOCK_PYTHON` | `symmetry_canonicalizer` with `backend: rpxdock`. |
| Meeko | `GH_MEEKO_PYTHON` and optional `GH_MEEKO_PYTHONPATH` | AutoDock-GPU/Vina ligand preparation helpers. |
| HADDOCK3 | `GH_HADDOCK3_BIN` | HADDOCK3 nucleic-acid predocker. |

## LigandMPNN

Repository and instructions:

```text
https://github.com/dauparas/LigandMPNN
```

Bootstrap downloads the public checkpoint files used by `config/base.yaml` into
the host asset root and binds that directory at `/models/ligandmpnn`.

## RAPiDock

RAPiDock is intentionally not part of the maintained bootstrap path. Do not use
it for the current container setup unless a future task explicitly re-enables it.

## AutoDock-GPU

AutoDock-GPU is not built into the maintained image. Keep it as an explicit
external predocker dependency and set `GH_AUTODOCK_GPU`/`GH_AUTOGRID4` only for
tasks that select `_type: autodock_gpu`.

## Protenix

Source owner and upstream:

```text
https://github.com/bytedance/Protenix
pinned v2 source plus patches/vendor/protenix/gh-protenix-use-vendored-esm2.patch
```

The SIF installs vendored Protenix code. Bootstrap downloads the default and
constraint checkpoints plus CCD cache files, then binds them under
`/models/protenix` and exports the matching `GH_PROTENIX_*` variables.

## Boltz

Repository and instructions:

```text
https://github.com/jwohlwend/boltz
```

Vendored Boltz code and runtime deps are installed in the SIF. Boltz model cache
remains a host-mounted resource when enabled.
