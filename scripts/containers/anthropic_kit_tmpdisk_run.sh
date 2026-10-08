#!/usr/bin/env bash
# Run one localimage Apptainer build on node-local /tmp, then retain only the
# final SIF and terminal evidence in the supplied persistent directory.
set -uo pipefail

KIT=${1:?kit name is required}
FINAL=${2:?persistent output directory is required}
LOCAL_WHEEL=${3:-}
LOCAL_FILE=${4:-}
LOCAL_UV=${5:-}
REVISION=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
APPTAINER=${APPTAINER_BIN:-apptainer}
BASE=${BASE_SIF:?set BASE_SIF to an existing authorized runtime image}
TMP=/tmp/anthropic-${KIT}-$$
WHEEL_ARGS=()
if [ -n "$LOCAL_WHEEL" ]; then
  WHEEL_ARGS=(--local-wheel "$LOCAL_WHEEL")
fi
if [ -n "$LOCAL_FILE" ]; then
  WHEEL_ARGS+=(--local-file "$LOCAL_FILE")
fi
if [ -n "$LOCAL_UV" ]; then
  WHEEL_ARGS+=(--local-uv "$LOCAL_UV")
fi

mkdir -p "$FINAL/logs" "$FINAL/state" "$TMP"
exec > >(tee "$FINAL/logs/job.log") 2>&1
echo "revision=$REVISION kit=$KIT host=$(hostname) started_at=$(date --iso-8601=seconds)"

"${PYTHON:-python3}" "$REPO/scripts/containers/anthropic_kits_apptainer.py" \
  --kit "$KIT" --base-sif "$BASE" --output-root "$TMP" \
  --apptainer-bin "$APPTAINER" --jobs "${BUILD_JOBS:-8}" --force \
  "${WHEEL_ARGS[@]}"
build_rc=$?

mkdir -p "$FINAL/images"
cp -a "$TMP/images/." "$FINAL/images/" 2>/dev/null || true
cat > "$FINAL/terminal-status.txt" <<EOF
status=$(if [ "$build_rc" -eq 0 ]; then echo BUILD_SUCCEEDED; else echo BUILD_FAILED; fi)
revision=$REVISION
kit=$KIT
host=$(hostname)
build_rc=$build_rc
sif=$FINAL/images/$KIT.sif
ended_at=$(date --iso-8601=seconds)
EOF
exit "$build_rc"
