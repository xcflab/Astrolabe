#!/usr/bin/env bash
# Build one pinned Anthropic kit on a Slurm node. The definition's %post
# installs the software into the SIF; model assets stay outside the image and
# are reused later through read-only Apptainer binds.
# Required environment: KIT and ROOT. Optional: DEF, BUILD_JOBS, APPTAINER_BIN.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -uo pipefail
: "${KIT:?set KIT}"
: "${ROOT:?set ROOT to the persistent kit directory}"
APPTAINER_BIN=${APPTAINER_BIN:-$HOME/.gh_container/envs/_gh_apptainer/bin/apptainer}
DEF=${DEF:-$ROOT/defs/${KIT}.def}
BUILD_JOBS=${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}
IMAGE=$ROOT/images/${KIT}.sif

mkdir -p "$ROOT/images" "$ROOT/logs" "$ROOT/apptainer-cache" "$ROOT/apptainer-tmp"
printf 'state=building\nstage=sif\nkit=%s\nhost=%s\nrevision=f4f62fa6592ae4938d49b1757bea0cfeff9f468e\nslurm_job_id=%s\nstarted_at=%s\n' \
  "$KIT" "$(hostname -f)" "${SLURM_JOB_ID:-unknown}" "$(date --iso-8601=seconds)" > "$ROOT/terminal-status.txt"

export APPTAINER_CACHEDIR="$ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$ROOT/apptainer-tmp"
export BUILD_JOBS
export PATH="$HOME/.gh_container/tools/fakeroot/bin:$HOME/.gh_container/envs/_gh_apptainer/bin:$PATH"
exec > >(tee "$ROOT/logs/build-${SLURM_JOB_ID:-manual}.log") 2>&1
echo "kit=$KIT host=$(hostname -f) definition=$DEF image=$IMAGE build_jobs=$BUILD_JOBS"
"$APPTAINER_BIN" build --fakeroot --warn-unused-build-args "$IMAGE" "$DEF"
rc=$?

if [ "$rc" -eq 0 ]; then
  state=build_complete
else
  state=failed
fi
printf 'state=%s\nstage=sif\nkit=%s\nhost=%s\nrevision=f4f62fa6592ae4938d49b1757bea0cfeff9f468e\nslurm_job_id=%s\nimage=%s\nexit_code=%s\nended_at=%s\n' \
  "$state" "$KIT" "$(hostname -f)" "${SLURM_JOB_ID:-unknown}" "$IMAGE" "$rc" "$(date --iso-8601=seconds)" > "$ROOT/terminal-status.txt"
exit "$rc"
