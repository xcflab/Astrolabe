# Astrolabe software requirements

Astrolabe has a small CPU-facing installation path and optional scientific backends with separate requirements. Install only the components selected by the configuration.

## Requirement matrix

| Layer | Needed for | Installation policy |
|---|---|---|
| Python virtual environment | CLI, configuration, orchestration, existing-coordinate workflows | Start from a clean environment and install the repository editable |
| Repository package `astrolabe` | Public Python API and CLI | `python -m pip install -e .` |
| Vendored source under `vendor/` | Optional backend implementations pinned by the project | Use the repository-provided source; do not substitute unrelated system packages |
| Model weights and databases | Only predictors selected by the configuration | Obtain from the original provider under its license and record the exact revision |
| GPU/CUDA stack | GPU inference backends only | Prefer the documented container or the backend's qualified environment |
| Container runtime | Reproducible optional model environments | Optional for the CPU example; see `container/README.md` |

## Clean CPU setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
astrolabe --help
```

This path supports configuration inspection, orchestration that does not invoke GPU predictors, and examples that consume existing AF3 coordinate files. It does **not** provide AF3 inference on CPU.

## Optional backends

Concrete predictors retain their scientific names. Their weights, databases, accelerator requirements, and licenses are backend-specific; selecting a backend is what activates those requirements. Do not install every optional stack merely to read configs or use existing coordinates.

The canonical container root is `/opt/Astrolabe`. The maintained definition files are `container/Astrolabe.def` and `container/Astrolabe.af3.def`.

Some optional model paths still lack complete publication-release end-to-end acceptance. The public documentation therefore avoids claiming universal backend readiness.

## Release metadata still required

Before a public release is cited as published, maintainers must provide and verify:

- confirmed author names, order, affiliations if desired, and ORCIDs;
- the live public repository URL;
- the release version/tag and release date;
- the software DOI, if minted;
- the associated paper DOI or final citation, when available.
