#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${GH_PY:-${HOME}/.gh_envs/envs/gh/bin/python}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/hcar1_constraint_compliance_benchmark}"
HCAR1_HOMOLOGY="${HCAR1_HOMOLOGY:?set HCAR1_HOMOLOGY to AF3 JSON path plus chain, e.g. /path/to/af3_unique_chains_part_1_data.json:A}"
CONFIG_ENV="${CONFIG_ENV:-${GH_CONFIG_ENV:-env}}"
RUNNER_GPU="${RUNNER_GPU:-0}"
SESSION_TAG="${1:-run_$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-nitrogen}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
JOB_NAME="gh_cc_${SESSION_TAG}"
LOG_DIR="${OUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT=$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sbatch")
cat > "${SBATCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}.%j.err

set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=${RUNNER_GPU}
"${PYTHON_BIN}" examples/hcar1/run_constraint_compliance_benchmark.py \\
  --session-tag "${SESSION_TAG}" \\
  --config-env "${CONFIG_ENV}" \\
  --target-homology "${HCAR1_HOMOLOGY}" \\
  --gpu "${RUNNER_GPU}" \\
  --out-dir "${OUT_DIR}"
EOF

chmod +x "${SBATCH_SCRIPT}"
JOB_ID=$(sbatch "${SBATCH_SCRIPT}" | awk '{print $NF}')
echo "job_id=${JOB_ID}"
echo "session_tag=${SESSION_TAG}"
echo "sbatch_script=${SBATCH_SCRIPT}"
echo "stdout_log=${LOG_DIR}/${JOB_NAME}.${JOB_ID}.out"
echo "stderr_log=${LOG_DIR}/${JOB_NAME}.${JOB_ID}.err"
