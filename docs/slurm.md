# Slurm (sbatch) 使用说明

本仓库提供一个极薄的提交包装脚本：`scripts/containers/slurm_submit.sh`。它不理解 `astrolabe` 的业务逻辑，只负责把任意命令包进 `sbatch` 并写好日志（符合“调度/业务解耦”）。

## 1) 最小示例（dry-run 先看脚本）

```bash
scripts/containers/slurm_submit.sh --partition gpu --gpus 1 --cpus 8 --mem 32G --time 04:00:00 --dry-run -- \
  ./scripts/astrolabe run --task test --dry-run
```

## 2) 提交一个 one-shot 设计任务

```bash
scripts/containers/slurm_submit.sh --partition gpu --gpus 1 --cpus 8 --mem 32G --time 04:00:00 -- \
  ./scripts/astrolabe design binder \
    --task-name demo_binder \
    --out-dir outputs \
    --binder-len 80-120 \
    --target-seq "+MSEQUENCE..." \
    --dry-run
```

已有初始链 A 时，可用 `--initial-binder-seq ACDE...` 或 `--initial-binder-expr "ACDE 40-60"` 替代对应长度参数，例如 `--binder-len`、`--peptide-len`、`--aptamer-len`、`--protein-len`。

## 3) 常用参数

- `--job-name`: job 名（默认 `gh_<timestamp>`）
- `--log-dir`: slurm stdout/stderr 与 sbatch 脚本输出目录（默认 `logs/slurm`）
- `--chdir`: job 的工作目录（默认当前目录；用于确保相对路径行为一致）

## 4) Python/环境选择

`./scripts/astrolabe` 默认用 `$HOME/.gh_envs/envs/gh/bin/python`，也可以在提交命令里显式指定：

```bash
scripts/containers/slurm_submit.sh --partition gpu --gpus 1 --cpus 8 --mem 32G --time 04:00:00 -- \
  GH_PY=/path/to/python ./scripts/astrolabe run --task benchmark_binder_v1_4 --dry-run
```
