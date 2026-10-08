# Changelog

本项目没有强制的语义化版本号；此处按日期记录对 CLI/配置系统/工具链的用户可见变更。

## 2026-07-05

- CLI: 增加 `gh experimental candidate-batch`，作为 process-isolated candidate-pool runner 的 canonical 入口；不再保留平行脚本别名。
- Docs: 模块 owner 与组合工作流最初以独立教程整理，现已并入 `docs/developer_guide.md`。
- Env/source: 增加 `config/third_party_sources.yaml`，用机器可检查清单记录 vendor submodule、fork-backed source、patch group 与 fork-candidate 状态。

## 2026-04-14

- Post-hoc: `gh posthoc filter` 增加 `--run-selection {all-passing,representative-per-run}`。`representative-per-run` 会先对每个 run 取“最后一个 stage 中 weighted_score 最高”的代表记录，再应用筛选。
- Post-hoc: `gh posthoc collect` / `gh posthoc filter` 增加更直观的参数别名：`--outputs`=`--out-root`、`--sqlite`=`--db`、`--csv`=`--export-csv`、`--expr`=`--filter-expr`、`--dest`=`--dst`、`--oligo`=`--prefer-oligo`。
- Runtime: 不再对整份运行配置做全量 `OmegaConf.resolve()`；未使用 evaluator / mutator / env 插值不会阻塞当前任务运行或 repeat 调度。
- Runtime: `scores.csv` 现在会默认带上常用置信度快照（如 `plddt` / `ptm` / `iptm`，以及可直接从当前 `StandardScores` 推出的 `ipsae` / `pae_min`），避免删除 `raw_model_results` 后丢失这些指标。

## 2026-01-23

- Post-hoc: 增加 refold 模块；当前规范入口为 `gh posthoc refold`，低层
  `refold-export` / `refold-run` 仅用于 manifest 调试或恢复。
- Post-hoc: `gh posthoc filter` 增加 `--allow-missing-required`（探索性运行时允许 DB 缺失 required 指标）。
- Docs: 增加 refold 设计；稳定用户流程现见 `docs/user_guide.md`。

## 2026-01-21

- Post-hoc: 增加增量/并行打分收集与筛选（`gh posthoc collect` / `gh posthoc filter`），输出 SQLite + CSV。
- Docs: 增加 post-hoc collect/filter 设计；稳定用户流程现见 `docs/user_guide.md`。
