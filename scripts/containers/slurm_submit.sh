#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/containers/slurm_submit.sh [slurm options] -- <command...>

Examples:
  scripts/containers/slurm_submit.sh --partition gpu --gpus 1 --cpus 8 --mem 32G --time 04:00:00 -- \
    ./scripts/astrolabe design binder --task-name demo --out-dir outputs --binder-len 80-120 --target-seq "+MSEQ..." --dry-run

Common options:
  --job-name NAME        Slurm job name (default: gh_<timestamp>)
  --partition PART       Slurm partition
  --account ACCT         Slurm account
  --qos QOS              Slurm qos
  --time HH:MM:SS        Wall time (default: 01:00:00)
  --cpus N               CPUs per task (default: 4)
  --mem MEM              Memory (default: 16G)
  --gpus N               GPUs (default: 1; set 0 for CPU jobs)
  --log-dir DIR          Log directory (default: logs/slurm)
  --chdir DIR            Working directory for the job (default: current dir)
  --script-out PATH      Write the generated sbatch script to PATH (default: <log-dir>/<job-name>.sbatch)
  --dry-run              Print the sbatch script and exit (no submission)
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERR: missing command in PATH: $1" >&2; exit 2; }
}

job_name=""
partition=""
account=""
qos=""
time_limit="01:00:00"
cpus="4"
mem="16G"
gpus="1"
log_dir="logs/slurm"
chdir="$(pwd)"
script_out=""
dry_run="0"

cmd=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --job-name)
      job_name="${2:-}"; shift 2 ;;
    --partition)
      partition="${2:-}"; shift 2 ;;
    --account)
      account="${2:-}"; shift 2 ;;
    --qos)
      qos="${2:-}"; shift 2 ;;
    --time)
      time_limit="${2:-}"; shift 2 ;;
    --cpus)
      cpus="${2:-}"; shift 2 ;;
    --mem)
      mem="${2:-}"; shift 2 ;;
    --gpus)
      gpus="${2:-}"; shift 2 ;;
    --log-dir)
      log_dir="${2:-}"; shift 2 ;;
    --chdir)
      chdir="${2:-}"; shift 2 ;;
    --script-out)
      script_out="${2:-}"; shift 2 ;;
    --dry-run)
      dry_run="1"; shift ;;
    --)
      shift
      cmd=( "$@" )
      break
      ;;
    *)
      echo "ERR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#cmd[@]} -eq 0 ]]; then
  echo "ERR: missing command (use -- <command...>)" >&2
  usage >&2
  exit 2
fi

if [[ "$dry_run" != "1" ]]; then
  require_cmd sbatch
fi

if [[ -z "$job_name" ]]; then
  job_name="gh_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$log_dir"

out_log="${log_dir}/${job_name}.%j.out"
err_log="${log_dir}/${job_name}.%j.err"

if [[ -z "$script_out" ]]; then
  script_out="${log_dir}/${job_name}.sbatch"
fi

tmp="${script_out}.tmp"

{
  echo '#!/usr/bin/env bash'
  echo "#SBATCH -J ${job_name}"
  [[ -n "$partition" ]] && echo "#SBATCH -p ${partition}"
  [[ -n "$account" ]] && echo "#SBATCH -A ${account}"
  [[ -n "$qos" ]] && echo "#SBATCH --qos=${qos}"
  echo "#SBATCH --time=${time_limit}"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks=1"
  echo "#SBATCH --cpus-per-task=${cpus}"
  echo "#SBATCH --mem=${mem}"
  if [[ "${gpus}" != "0" ]]; then
    echo "#SBATCH --gres=gpu:${gpus}"
  fi
  echo "#SBATCH -o ${out_log}"
  echo "#SBATCH -e ${err_log}"
  echo "#SBATCH --chdir=${chdir}"
  echo
  echo 'set -euo pipefail'
  echo
  echo 'echo "[slurm] host: $(hostname)"'
  echo 'echo "[slurm] date: $(date -Is)"'
  echo 'echo "[slurm] pwd:  $(pwd)"'
  echo 'echo -n "[slurm] cmd: "'
  echo -n "printf '%q ' "
  printf '%q ' "${cmd[@]}"
  echo
  echo 'echo'
  echo
  printf '%q ' "${cmd[@]}"
  echo
} >"$tmp"

mv -f "$tmp" "$script_out"

if [[ "$dry_run" == "1" ]]; then
  cat "$script_out"
  exit 0
fi

sbatch "$script_out"
