#!/usr/bin/env bash
# Check one already-built pinned Anthropic kit through Apptainer.
# Software installation happens in the SIF definition's %post during build.
# The weight roots below are existing host assets; this driver never downloads
# them and never calls the kit's runtime `install` command.
set -uo pipefail

KIT=${1:?kit name is required}
SIF=${2:?SIF path is required}
OUT=${3:?persistent output directory is required}
APPTAINER=${APPTAINER_BIN:-$HOME/.gh_container/envs/_gh_apptainer/bin/apptainer}
REVISION=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
GH_CONTAINER_ROOT=${GH_CONTAINER_ROOT:-${HOME:?HOME must be set}/.gh_container}
GH_KITS_ROOT=${GH_KITS_ROOT:-${HOME:?HOME must be set}/.gh_anthropic_kits}
SHARED_GH_CONTAINER_ROOT=${GH_SHARED_CONTAINER_ROOT:-$GH_CONTAINER_ROOT}
SHARED_GH_KITS_ROOT=${GH_SHARED_KITS_ROOT:-$GH_KITS_ROOT}/$REVISION

mkdir -p "$OUT/logs" "$OUT/jit" "$OUT/weights"
exec > >(tee "$OUT/logs/container-install.log") 2>&1
echo "revision=$REVISION kit=$KIT sif=$SIF host=$(hostname) started_at=$(date --iso-8601=seconds)"

run_kit() {
  local nv=()
  if [ "${1:-}" = "--nv" ]; then
    nv=(--nv)
    shift
  fi
  "$APPTAINER" run --cleanenv "${nv[@]}" "${bind_args[@]}" "${env_args[@]}" "$SIF" "$@"
}

link_if_file() {
  local source=$1 target=$2
  if [ -f "$source" ]; then
    ln -f "$source" "$target"
    echo "reused=$source -> $target" >&2
  else
    echo "missing=$source" >&2
  fi
}

first_file() {
  local candidate
  for candidate in "$@"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

first_dir_with_file() {
  local marker=$1 candidate
  shift
  for candidate in "$@"; do
    if [ -n "$candidate" ] && [ -f "$candidate/$marker" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

link_first_file() {
  local target=$1 source
  shift
  source=$(first_file "$@" || true)
  if [ -n "$source" ]; then
    link_if_file "$source" "$target"
    return 0
  fi
  echo "missing=$target" >&2
  return 1
}

prepare_protenix_root() {
  local root="$OUT/weights/protenix"
  mkdir -p "$root/checkpoint" "$root/common"
  local protenix_assets=${PROTENIX_ASSETS_ROOT:-$GH_CONTAINER_ROOT/assets/Protenix}
  link_first_file "$root/checkpoint/protenix-v2.pt" \
    "${PROTENIX_CHECKPOINT:-}" \
    "$protenix_assets/release_model/protenix-v2.pt" \
    "$SHARED_GH_CONTAINER_ROOT/assets/Protenix/release_model/protenix-v2.pt"
  local common=${PROTENIX_COMMON_ROOT:-}
  if [ -z "$common" ]; then
    common=$(first_dir_with_file components.cif \
      "$GH_CONTAINER_ROOT/assets/Protenix/common" \
      "$SHARED_GH_CONTAINER_ROOT/assets/Protenix/common" \
      "$HOME/.cache/opendde/common" \
      "$HOME/.cache/opendde/common" || true)
  fi
  link_first_file "$root/common/components.cif" \
    "$common/components.cif" "$common/components.v20240608.cif"
  link_first_file "$root/common/components.cif.rdkit_mol.pkl" \
    "$common/components.cif.rdkit_mol.pkl" "$common/components.v20240608.cif.rdkit_mol.pkl"
  for name in obsolete_release_date.csv clusters-by-entity-40.txt release_date_cache.json obsolete_to_successor.json; do
    link_first_file "$root/common/$name" "$common/$name"
  done
  printf '%s\n' "$root"
}

bind_args=()
env_args=(--env PIP_NO_INDEX=1 --env PIP_DISABLE_PIP_VERSION_CHECK=1)
check_args=()
weight_source=missing

case "$KIT" in
  af3_jax)
    af3_root=${AF3_JAX_WEIGHTS:-}
    if [ -z "$af3_root" ]; then
      af3_root=$(first_dir_with_file p2/of3_ported_weights.bin.zst \
        "$GH_CONTAINER_ROOT/assets/af3_jax" \
        "$GH_CONTAINER_ROOT/assets/alphafold3_jax" \
        "$GH_CONTAINER_ROOT/assets/af3" \
        "$SHARED_GH_CONTAINER_ROOT/assets/af3_jax" \
        "$SHARED_GH_CONTAINER_ROOT/assets/alphafold3_jax" \
        "$SHARED_GH_KITS_ROOT/af3_jax/weights" || true)
    fi
    if [ -n "$af3_root" ] && [ -d "$af3_root" ]; then
      bind_args+=(--bind "$af3_root:/weights/af3_jax:ro")
      env_args+=(--env AF3_JAX_PARAMS_ROOT=/weights/af3_jax)
      weight_source=$af3_root
    else
      env_args+=(--env AF3_JAX_PARAMS_ROOT=/weights/af3_jax)
      weight_source=missing:af3-jax-p2
    fi
    check_args=(check --config h100 --variant p2)
    ;;
  protenix_v2)
    protenix_root=$(prepare_protenix_root)
    bind_args+=(--bind "$protenix_root:/weights/protenix:ro")
    env_args+=(--env PROTENIX_ROOT_DIR=/weights/protenix --env MODEL_OPT_JIT_ROOT=/jit)
    bind_args+=(--bind "$OUT/jit:/jit")
    weight_source=$protenix_root
    check_args=(check --config h100)
    ;;
  boltz2)
    boltz_root=${BOLTZ2_WEIGHTS:-}
    if [ -z "$boltz_root" ]; then
      boltz_root=$(first_dir_with_file boltz2_conf.ckpt \
        "$GH_CONTAINER_ROOT/assets/boltz2" \
        "$GH_CONTAINER_ROOT/assets/boltz" \
        "$HOME/.cache/boltz" \
        "$HOME/.cache/boltz2" \
        "$SHARED_GH_CONTAINER_ROOT/assets/boltz2" \
        "$SHARED_GH_CONTAINER_ROOT/assets/boltz" \
        "$SHARED_GH_KITS_ROOT/boltz2-agent/weights" || true)
    fi
    if [ -n "$boltz_root" ] && [ -d "$boltz_root" ]; then
      bind_args+=(--bind "$boltz_root:/weights/boltz2:ro")
      weight_source=$boltz_root
    else
      weight_source=missing:boltz2
    fi
    bind_args+=(--bind "$OUT/jit:/jit")
    env_args+=(--env BOLTZ_CACHE=/weights/boltz2 --env MODEL_OPT_JIT_ROOT=/jit)
    check_args=(check --config h100 --mode fast)
    ;;
  chai1)
    chai_root=${CHAI1_WEIGHTS:-}
    if [ -z "$chai_root" ]; then
      chai_root=$(first_dir_with_file conformers_v1.apkl \
        "$GH_CONTAINER_ROOT/assets/chai1" \
        "$GH_CONTAINER_ROOT/assets/chai" \
        "$HOME/.cache/chai1" \
        "$SHARED_GH_CONTAINER_ROOT/assets/chai1" \
        "$SHARED_GH_CONTAINER_ROOT/assets/chai" \
        "$SHARED_GH_KITS_ROOT/chai1-agent/weights" || true)
    fi
    if [ -n "$chai_root" ] && [ -d "$chai_root" ]; then
      bind_args+=(--bind "$chai_root:/weights/chai1:ro")
      weight_source=$chai_root
    else
      weight_source=missing:chai1
    fi
    bind_args+=(--bind "$OUT/jit:/cache")
    env_args+=(--env CHAI_DOWNLOADS_DIR=/weights/chai1 --env MODEL_OPT_JIT_ROOT=/cache)
    check_args=(check --config h100)
    ;;
  esm_if1)
    esm_if1_root="$OUT/weights/esm_if1"
    mkdir -p "$esm_if1_root"
    esm_if1_file=$(first_file \
      "${ESM_IF1_WEIGHTS:-}" \
      "${ESM_IF1_WEIGHTS_ROOT:-}/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$GH_CONTAINER_ROOT/assets/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$GH_CONTAINER_ROOT/cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$HOME/.cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$SHARED_GH_CONTAINER_ROOT/assets/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$SHARED_GH_CONTAINER_ROOT/cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt" \
      "$HOME/.cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt" || true)
    if [ -n "$esm_if1_file" ]; then
      link_if_file "$esm_if1_file" "$esm_if1_root/esm_if1_gvp4_t16_142M_UR50.pt"
      bind_args+=(--bind "$esm_if1_root:/weights/esm_if1:ro")
      weight_source=$esm_if1_file
      env_args+=(--env ESM_IF1_WEIGHTS=/weights/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt)
    else
      weight_source=missing:esm-if1
    fi
    bind_args+=(--bind "$OUT/jit:/jit")
    env_args+=(--env TORCH_HOME=/weights/esm_if1 --env MODEL_OPT_JIT_ROOT=/jit)
    check_args=(check --config h100 --mode fast)
    ;;
  esmfold2)
    hf_root=${ESMFOLD2_WEIGHTS_ROOT:-}
    if [ -z "$hf_root" ]; then
      hf_root=$(first_dir_with_file hub/models--biohub--ESMFold2/refs/main \
        "$GH_CONTAINER_ROOT/cache/huggingface" \
        "$HOME/.cache/huggingface" \
        "$SHARED_GH_CONTAINER_ROOT/cache/huggingface" \
        "$HOME/.cache/huggingface" || true)
    fi
    if [ -n "$hf_root" ] && [ -d "$hf_root" ]; then
      bind_args+=(--bind "$hf_root:/weights/esmfold2:ro")
      weight_source=$hf_root
    else
      weight_source=missing:esmfold2
    fi
    bind_args+=(--bind "$OUT/jit:/jit")
    env_args+=(--env HF_HOME=/weights/esmfold2 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MODEL_OPT_JIT_ROOT=/jit)
    esmfold_variant=${ESMFOLD2_VARIANT:-}
    if [ -z "$esmfold_variant" ]; then
      if [ -d "$hf_root/hub/models--biohub--ESMFold2-Fast" ]; then
        esmfold_variant=fast
      else
        esmfold_variant=full_nomsa
      fi
    fi
    check_args=(check --config h100 --variant "$esmfold_variant" --mode fast)
    ;;
  opendde)
    opendde_root=${OPENDDE_WEIGHTS_ROOT:-}
    if [ -z "$opendde_root" ]; then
      opendde_root=$(first_dir_with_file checkpoint/opendde.pt \
        "$GH_CONTAINER_ROOT/assets/opendde" \
        "$GH_CONTAINER_ROOT/repos/opendde" \
        "$HOME/.cache/opendde" \
        "$SHARED_GH_CONTAINER_ROOT/assets/opendde" \
        "$SHARED_GH_CONTAINER_ROOT/repos/opendde" \
        "$HOME/.cache/opendde" || true)
    fi
    if [ -n "$opendde_root" ] && [ -d "$opendde_root" ]; then
      bind_args+=(--bind "$opendde_root:/weights/opendde:ro")
      weight_source=$opendde_root
    else
      weight_source=missing:opendde
    fi
    bind_args+=(--bind "$OUT/jit:/jit")
    env_args+=(--env OPENDDE_ROOT_DIR=/weights/opendde --env MODEL_OPT_JIT_ROOT=/jit)
    check_args=(check --config h100 --mode fast)
    ;;
  *)
    echo "unsupported kit: $KIT" >&2
    exit 2
    ;;
esac

check_log="$OUT/logs/check.log"
run_kit --nv "${check_args[@]}" 2>&1 | tee "$check_log"
check_rc=${PIPESTATUS[0]}

status=FAILED
if [ "$check_rc" -eq 0 ]; then
  status=SUCCEEDED
fi
cat > "$OUT/container-install-status.txt" <<EOF
status=$status
revision=$REVISION
kit=$KIT
sif=$SIF
weight_source=$weight_source
install_rc=not_applicable_build_time
check_rc=$check_rc
ended_at=$(date --iso-8601=seconds)
EOF
echo "status=$status install_rc=not_applicable_build_time check_rc=$check_rc weight_source=$weight_source"
[ "$status" = SUCCEEDED ]
