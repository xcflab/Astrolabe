set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

gh_python := env_var_or_default("GH_PY", `python3 -c "from pathlib import Path; from astrolabe.dev_wrapper import resolve_default_gh_python; p=Path('.venv/bin/python'); print(str(p.absolute()) if p.is_file() else (resolve_default_gh_python(Path.cwd()) or 'python3'))"`)
mamba_exe := env_var_or_default("MAMBA_EXE", "~/.gh_envs/.bin/micromamba")

# Post-hoc score collection + structure filtering
#
# These are lightweight acceptance commands for the post-hoc CLI tools.

OUT_ROOT := env_var_or_default("GH_OUT_ROOT", "")
DB := "/tmp/gh_posthoc_accept.db"
CSV := "/tmp/gh_posthoc_accept.csv"
DST := "/tmp/gh_posthoc_accept_filtered"
PRESET := env_var_or_default("GH_FILTER_PRESET", "")

REFOLD_MANIFEST := "/tmp/gh_refold_accept.jsonl"
REFOLD_REPORT := "/tmp/gh_refold_accept_report.csv"
REFOLD_DRY_MANIFEST := "/tmp/gh_refold_dry_accept.jsonl"
REFOLD_OUT := "/tmp/gh_refold_dry_out"

# Defaults (override with `just key=value recipe`).
workers := "1"
cache := "0"
cache_max_entries := "20000"
feat_lengths := "500 1000"
feat_runs := "3"
feat_warmup := "1"

baseline_workers := "1"
opt_workers := "8"
opt_cache := "1"
opt_cache_max_entries := "20000"
infer_lengths := "500 1000"
infer_num_recycles := "1"
infer_num_samples := "1"

env-setup:
    python3 scripts/installation/env_setup.py -y --profile runtime --write-env-yaml config/env/env_local.yaml --verify-env-yaml

test:
    GH_HOST_REPO_ROOT="{{justfile_directory()}}" {{gh_python}} -m pytest -q

# Recompute contact geometry on the bundled prediction, then collect/filter/report.
reproduce-cpu out="outputs/cpu-example":
    {{gh_python}} examples/01_cpu_reproduce.py --out-dir '{{out}}'

container-lock:
    uv pip compile container/requirements.runtime.in --output-file container/requirements.runtime.lock --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes --custom-compile-command 'just container-lock'

container-conda-lock:
    {{mamba_exe}} create --dry-run --json --no-rc --override-channels --channel-alias https://conda.anaconda.org -y -p /tmp/gh_container_prefix_solve -c conda-forge -c bioconda python=3.11.14 pip=25.3 numpy=1.26.4 zlib=1.3.2 openbabel=3.1.1 vina=1.2.7 dssp=4.5.7 | jq -r '"@EXPLICIT", (.actions.LINK[] | .url + "#" + .md5)' | sed 's#https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge#https://conda.anaconda.org/conda-forge#g' > container/conda-linux-64.lock

clean-transients:
    PYTHONDONTWRITEBYTECODE=1 {{gh_python}} scripts/maintenance/clean_transients.py

third-party-licenses:
    PYTHONDONTWRITEBYTECODE=1 {{gh_python}} scripts/release/generate_third_party_licenses.py

release-audit:
    PYTHONDONTWRITEBYTECODE=1 {{gh_python}} scripts/release/audit.py

vendor-fork-worktree name out:
    PYTHONDONTWRITEBYTECODE=1 {{gh_python}} scripts/maintenance/apply_vendor_patches.py --only {{name}} --materialize-fork-worktree '{{out}}'

vendor-test-baseline:
    git submodule update --init --recursive --depth 1 --jobs 4 vendor/alphafold2 vendor/alphafold3 vendor/boltz vendor/chai_lab vendor/esm2 vendor/esm3 vendor/rpxdock
    PYTHONDONTWRITEBYTECODE=1 {{gh_python}} scripts/maintenance/apply_vendor_patches.py

af3_feat_profile:
    ALPHAFOLD3_REFSTRUCTURE_WORKERS={{workers}} ALPHAFOLD3_REFSTRUCTURE_CACHE={{cache}} ALPHAFOLD3_REFSTRUCTURE_CACHE_MAX_ENTRIES={{cache_max_entries}} {{gh_python}} examples/benchmarks/af3_featurization_profile.py --lengths {{feat_lengths}} --runs {{feat_runs}} --warmup {{feat_warmup}} --msa-crop-size 32 --max-templates 0 --pad-num-chains 1

af3_infer_bench:
    {{gh_python}} examples/benchmarks/af3_inference_benchmark.py --lengths {{infer_lengths}} --num-recycles {{infer_num_recycles}} --num-samples {{infer_num_samples}} --baseline-workers {{baseline_workers}} --opt-workers {{opt_workers}} --opt-cache {{opt_cache}} --opt-cache-max-entries {{opt_cache_max_entries}}

posthoc-collect-smoke:
  test -n '{{OUT_ROOT}}' || (echo 'ERR: set GH_OUT_ROOT (a run output root)'; exit 2)
  ./scripts/astrolabe posthoc collect --out-root '{{OUT_ROOT}}' --db '{{DB}}' --export-csv '{{CSV}}' --workers 4 --binder-chain B --target-chain A

posthoc-filter-smoke:
  test -n '{{PRESET}}' || (echo 'ERR: set GH_FILTER_PRESET (a design-metrics preset YAML)'; exit 2)
  ./scripts/astrolabe posthoc filter --db '{{DB}}' --preset '{{PRESET}}' --dst '{{DST}}' --prefer-oligo AB --allow-missing-required
  test -f '{{DST}}/filter_report.csv'
  test -f '{{DST}}/filtered.sqlite'

posthoc-refold-export-smoke:
  test -n '{{PRESET}}' || (echo 'ERR: set GH_FILTER_PRESET (a design-metrics preset YAML)'; exit 2)
  ./scripts/astrolabe posthoc refold-export --db '{{DB}}' --preset '{{PRESET}}' --out-manifest '{{REFOLD_MANIFEST}}' --report-csv '{{REFOLD_REPORT}}' --prefer-oligo AB --no-passed-only --limit 1
  test -f '{{REFOLD_REPORT}}'
  test "$({{gh_python}} -c 'import csv; print(sum(1 for _ in csv.DictReader(open("{{REFOLD_REPORT}}"))))')" -ge 1

posthoc-refold-run-dry-smoke:
  rm -rf '{{REFOLD_OUT}}' '{{REFOLD_DRY_MANIFEST}}' /tmp/gh_refold_run_dry_summary.json
  {{gh_python}} -c "import json; from pathlib import Path; p=Path('{{REFOLD_DRY_MANIFEST}}'); item={'task':'dummy_task','run':'001_run','step':1,'oligo':'AB','seqs':{'A':'ACDE','B':'FGHI'},'types':{'A':'protein','B':'protein'},'homologies':{'A':None,'B':None},'src_run_dir':''}; p.write_text(json.dumps(item)+'\\n', encoding='utf-8')"
  ./scripts/astrolabe posthoc refold-run --manifest '{{REFOLD_DRY_MANIFEST}}' --out-root '{{REFOLD_OUT}}' --dry-run --workers 3 --format json > /tmp/gh_refold_run_dry_summary.json
  {{gh_python}} -c "import json; s=json.load(open('/tmp/gh_refold_run_dry_summary.json')); assert s['groups'] == 1, s; assert s['resource_groups'] == 1, s; assert s['workers_requested'] == 3, s; assert s['workers'] == 1, s; assert s['evaluator_cache_hits'] == 0, s; assert s['evaluator_cache_misses'] == 0, s"
  test -f '{{REFOLD_OUT}}/dummy_task/001_run/scores.csv'

# Build Astrolabe from the current worktree export, never from HEAD only.
release-export archive="dist/astrolabe-source.tar.gz":
    #!/usr/bin/env bash
    set -euo pipefail
    stage="$(mktemp -d /tmp/astrolabe-release.XXXXXX)"
    trap 'rm -rf "$stage"' EXIT
    mkdir -p "$(dirname "{{archive}}")"
    {{gh_python}} scripts/release/export.py "$stage/Astrolabe" --archive "{{archive}}"

release-build dist="dist":
    #!/usr/bin/env bash
    set -euo pipefail
    stage="$(mktemp -d /tmp/astrolabe-build.XXXXXX)"
    trap 'rm -rf "$stage"' EXIT
    mkdir -p "{{dist}}"
    out="$(cd "{{dist}}" && pwd)"
    {{gh_python}} scripts/release/export.py "$stage/Astrolabe" --archive "$out/astrolabe-source.tar.gz"
    cd "$stage/Astrolabe"
    {{gh_python}} -m build --no-isolation --outdir "$out"
