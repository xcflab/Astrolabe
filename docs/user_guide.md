# Astrolabe 用户手册：从输入、优化到可审计结果

首次安装和完整入门流程以 [`README.md`](../README.md) 为准，本手册是进阶参考。
它描述当前 checkout 中已经由
parser、runtime 和真实运行记录验证的工作流；命令一律以
`./scripts/astrolabe` 为准。不要把历史文档中的任务名、参数名或输出计数当作当前
契约，先用本手册的 dry-run 和 `--help` 复核。项目总览与安装入口见
[`README.md`](../README.md)，实现与扩展见本手册末尾链接的开发者手册。


### 命名迁移与兼容边界

- Python 导入使用 `astrolabe`，安装后的 CLI 使用 `astrolabe`。
- checkout 内的 canonical launcher 是 `./scripts/astrolabe`。
- `./scripts/gh_dev` 只保留为旧 bootstrap/checkout owner 的委托兼容包装器，不是安装别名。
- `GH_*`、`~/.gh_envs`、`.gh_python_path` 仍是兼容合同；`scripts/install.py` 是 canonical bootstrap；不要把它们改写成尚未实现的新名字。

## 0. 名称边界与一页式路线

### 0.1 Astrolabe 与历史兼容名

- **Astrolabe** 是当前公开的软件名、Python namespace 和 CLI；安装后命令为 `astrolabe`。
- 旧版本可能仍出现 **GuidedHallucination** / **GH**；它们是迁移来源或兼容名，不是第二个已安装命令。
- checkout 内使用 `./scripts/astrolabe ...`；`./scripts/gh_dev` 只为既有 bootstrap owner 委托到同一入口。

### 0.2 推荐主流程

```text
确认 checkout 与环境
  -> 选择 design / refine / run
  -> compose 模板、standalone、overlay、overrides
  -> --print-config --dry-run 检查最终配置
  -> 最小真实 smoke（必要时）
  -> 正式 `astrolabe run`
  -> 检查输出合同和 Pipeline completed successfully
  -> collect -> filter -> refold -> report
```

单次优化轨迹的数据流是：

```text
初始化序列/结构
  -> step 0 evaluator 初始评估 -> post-eval -> loss/decision 基线
  -> 每个后续 step：mutator 提议候选
       -> evaluator 预测候选结构和指标
       -> post-eval -> loss/context
       -> decision 根据分数、threshold、gate 决定接受/拒绝
       -> 接受则提交序列，拒绝则回滚
  -> 直到 steps 上限或 early stopping
```

`repeats` 在这个循环外创建多个 run；它不改变单条轨迹内部的 step 语义。

不要把“配置成功”“进程退出 0”“Slurm COMPLETED”和“科学结果成功”混为一谈。
最终成功必须同时满足运行合同、产物合同和任务自己的科学筛选条件。

## 1. 运行前：确认代码、路径和环境

### 1.1 确认正在使用的 checkout

在修改配置或提交正式运行前，确认所在目录包含 `scripts/astrolabe`、`config/`
和 `astrolabe/`，而不是只存 outputs 的项目目录。完整源码包可以直接安装；
模型安装器会根据导出清单建立本地索引，不需要先克隆主仓库。

如果使用 Git checkout，可记录版本状态：

```bash
git rev-parse --show-toplevel
git status --short
git branch --show-current
```

正式批次应记录源码 commit（源码包则记录 `SOURCE_EXPORT.json`、依赖 pins 和本地改动）、配置文件和 env 文件。不同机器上
不要直接复用含远端绝对路径的 `config.yaml`；先做显式 path mapping，再 dry-run。

### 1.2 安装入口与历史 host 环境

首次使用按 README 安装：CPU 后处理使用隔离 Python 环境，模型运行使用
`python3 scripts/install.py --activate-checkout` 并明确选择模型和权重。


安装后可直接运行仓库保留的 CPU 示例；它只读取已有坐标，不执行 AF3 推理：

~~~bash
python examples/01_cpu_reproduce.py --out-dir outputs/cpu-example
~~~

下面的 host recipe 保留给已有非容器部署，不是新用户默认入口：

```bash
just env-setup
```

该 recipe 会调用仓库维护的环境脚本，生成/更新 site-local
`config/env/env_local.yaml` 并校验其内容。不要为普通 Astrolabe 运行临时执行一串
`pip install`，仓库依赖和第三方源遵循 `vendor/` 与既有 environment contract。

任务感知的检查：

```bash
./scripts/astrolabe env doctor --task benchmark_binder_v1_4 --strict
```

如果只想看缺失项，可省略 `--strict`；如果 env 文件不在默认位置：

```bash
./scripts/astrolabe env doctor \
  --env config/env/env_local.yaml \
  --task benchmark_binder_v1_4 \
  --resolve --python-deps
```

`--strict` 会解析 interpolation、检查任务涉及的路径并检查 Python imports；缺失
时应修复 env/资产，而不是更换 evaluator 或静默跳过后端。`--soft-fail` 只适合
规划或盘点，不是运行验收。

### 1.3 Env wizard

已有任务模板但本机路径尚未填写时：

```bash
./scripts/astrolabe env wizard \
  --task benchmark_binder_v1_4 \
  --env config/env/env_local.yaml
```

wizard 根据任务实际使用的 evaluator、mutator、predocker 和 loss 推断所需键。
默认写回指定 env 文件；需要另存时使用 `--out /abs/path/env.yaml`。写完仍必须
运行 `env doctor --strict`。

### 1.4 Container/bootstrap 入口

正式模型运行的容器安装器为 `scripts/install.py`。先查看可选组件：

```bash
python3 scripts/install.py --list-software
```

再按站点、许可、已有 SIF、模型资产和容器引擎选择 profile。bootstrap 不意味着
“无前提安装”：受限权重、许可证、网络、Apptainer/Docker、GPU 和远端路径仍需
先确认。完整资产、host/container 差异、PyRosetta 和第三方 source contract 见
[`docs/third_party_envs.md`](third_party_envs.md)。

容器 runner 的源码身份要特别注意：直接调用生成的 runner 时，默认使用 SIF 内
冻结源码。通过 bootstrap 的 `--activate-checkout` 激活当前 checkout 后，
`./scripts/astrolabe ...` 会自动绑定当前 checkout，无需再设置 `GH_PY`、
`GH_HOST_REPO_ROOT` 或模型路径。只有读取 checkout 外的输入/输出目录时才需要在
安装 profile 中配置 `--extra-bind`。容器 import smoke、doctor 和真实小任务都
通过后，才可把它当可运行环境。

## 2. 选择入口：design、refine 还是 run

### 2.1 `astrolabe design`：从序列/长度/框架生成新任务

适合没有现成复合物作为起点的 one-shot 设计。查看分组入口：

```bash
./scripts/astrolabe design --help
```

常用预设包括：

| 需求 | 当前入口 |
|---|---|
| 蛋白 binder | `astrolabe design binder`（`protein-binder` 是同一预设 alias） |
| peptide binder | `astrolabe design peptide-binder` |
| RNA/DNA aptamer | `astrolabe design aptamer` / `astrolabe design dna-aptamer` |
| antibody/scFv | `astrolabe design antibody` |
| protein + ligand | `astrolabe design ligand` |
| oligomer | `astrolabe design oligomer` |
| 多 target binder | `astrolabe design multitarget-binder` |
| chemical-induced dimer | `astrolabe design cid-generate` |

最小 binder dry-run：

```bash
./scripts/astrolabe design binder \
  --task-name demo_binder \
  --out-dir outputs \
  --binder-len 80-120 \
  --target-seq MKTAYIAKQRQISFVKSHFSRQDILDLWQ \
  --print-config --dry-run
```

三种 binder 初始输入互斥：`--binder-len`、`--initial-binder-seq` 或
`--initial-binder-expr`。已有固定片段、文件片段或多 segment 语句时，使用
`--initial-binder-expr`，不要把高级 statement 塞进长度参数。

目标可来自：

- `--target-seq`：直接序列，作为固定 target；
- `--target-homology /abs/input.json:B`：AF3 JSON 必须带链后缀，或 `.a3m`；
- `--target-structure complex.cif --target-chain B`：从结构取序列并注入 template。

若同时提供多个来源，Astrolabe 会校验序列一致性。homology 已有 templates 时不会被
结构来源无条件覆盖；没有 templates 时才合并 structure-derived template。

peptide 示例：

```bash
./scripts/astrolabe design peptide-binder \
  --task-name demo_peptide \
  --out-dir outputs \
  --peptide-len 8-16 \
  --target-seq MKTAYIAKQRQISFVKSHFSRQDILDLWQ \
  --binding-site-target B:4-8 \
  --dry-run
```

环化、非标准残基和跨链共价键属于专门输入契约，应使用对应字段说明；见
[`docs/nonstandard_residue_contracts.md`](nonstandard_residue_contracts.md)、
[`docs/cyclization_topology_examples.md`](cyclization_topology_examples.md)。

antibody 示例：

```bash
./scripts/astrolabe design antibody --list-frameworks
./scripts/astrolabe design antibody \
  --task-name demo_scfv \
  --out-dir outputs \
  --framework scfv4d5_vhvl \
  --cdr-lengths H1:7-11,H2:6-10,H3:9-20 \
  --target-seq MKTAYIAKQRQISFVKSHFSRQDILDLWQ \
  --dry-run
```

### 2.2 `astrolabe refine`：从现有结构/序列继续优化

有现成 complex、binder、ligand pocket 或 oligomer 时优先选择 refine：

```bash
./scripts/astrolabe refine --help
```

常用入口：

| 起点 | 入口 |
|---|---|
| complex 中的 binder 链 | `astrolabe refine binder` |
| protein + ligand 结构 | `astrolabe refine ligand` |
| 仅在 ligand pocket 周围 redesign | `astrolabe refine ligand-pocket` |
| channel/broaden 任务 | `astrolabe refine channel` |
| homo-oligomer | `astrolabe refine oligomer` |

最小结构 refine：

```bash
./scripts/astrolabe refine binder \
  --task-name demo_refine \
  --out-dir outputs \
  --structure input_complex.cif \
  --target-chain B \
  --binder-chains A \
  --target-template-from-structure \
  --print-config --dry-run
```

`--binder-chains` 是逗号分隔的输入结构 chain ID，顺序保留。它不同于配置表达式
中的 protomer 名；复杂输入先让 refine 生成规范化配置，再检查 `seqs`、`oligo`
和 chain mapping。

### 2.3 `astrolabe run`：已知模板/配置的正式入口

当任务配置已经存在，或要精确控制多 stage、evaluator、mutator 和 loss，直接用：

```bash
./scripts/astrolabe run --task <task> ...
```

`run` 是最接近 runtime 的 canonical 入口；one-shot `design/refine` 最终也会
生成同类配置并进入相同的 pipeline。

## 3. 配置来源与覆盖顺序

### 3.1 三类配置来源

1. `--task <name>`：选择 `config/<name>.yaml`，由该模板的 Hydra `defaults:`
   组合 base 与 config groups；
2. `--standalone /abs/config.yaml`：可单独使用，也可与 `--task` 同用；Astrolabe 先组合
   选定模板（未给 `--task` 时用 `base`），再把 standalone 当作外部配置合并，
   且明确删除 standalone 自己的 `defaults:`，不会据此重新走一遍 Hydra；
3. `--overlay /abs/overlay.yaml`：在前面所有结果之上做最后一次 OmegaConf merge。

常用示例：

```bash
./scripts/astrolabe run \
  --task benchmark_binder_v1_4 \
  --overlay /abs/path/d.binder.overlay.yaml \
  --print-config --dry-run
```

### 3.2 实际覆盖顺序（必须记住）

对只有 `--task` 的 `astrolabe run` 路径，实际顺序是：

```text
所选模板的 Hydra defaults/base/_self_
  -> --overrides-file
  -> --overrides 字符串
  -> 命令尾部 Hydra overrides
  -> --overlay（最后一次 merge）
```

前三类 CLI override 按上述顺序收集；没有 standalone 时，它们都交给 Hydra
compose，其中 config-group override（如 `env=env`）改变 defaults 选择，普通
dotlist 改字段。对含 `--standalone` 的路径，实际顺序是：

```text
所选模板的 Hydra defaults/base/_self_
  -> CLI 中识别出的 config-group overrides（compose 时）
  -> standalone 内容（standalone.defaults 被忽略）
  -> CLI 普通 dotlist overrides
  -> --overlay（最后一次 merge）
```

所以 **overlay 最后** 是硬规则。`--overrides-file`、`--overrides` 和命令尾部
override 若写同一字段，按它们的收集次序后者胜出；再之后 overlay 仍可覆盖它。
要让某个值来源唯一，就不要在多个层重复写。

`env=env` 这类 config-group 切换必须在 compose 阶段发生；standalone 模式下
Astrolabe 会把 group override 与普通 dotlist override 分开处理。普通字段如
`steps=10`、`seed=42` 在 standalone 之后应用。

`astrolabe design ...`、`astrolabe refine ...` 和旧的 `astrolabe bench-*` 是 **one-shot builder 路径**：
它们先组合各自固定模板，再由 builder 把 `--task-name`、`--out-dir`、输入和
`--steps/--seed/--repeats` 写入配置，最后直接进入同一 runtime。不要把这些参数
误写成 `astrolabe run` 的 flags；需要复用 builder 结果时用 `--dump-config` 保存它，再
审计 `--print-config --dry-run`。

### 3.3 overrides-file

复杂列表、表达式和带空格的字符串优先写文件：

```text
steps=10
repeats=5
stages.default.mutation_num=50%
stages.default.mutator_weights=[0.45,0.35,0.20]
```

执行：

```bash
./scripts/astrolabe run \
  --task benchmark_binder_omni_optimize_refine \
  --overrides-file overrides.txt \
  --print-config --dry-run
```

这样比把 YAML/列表塞进 shell 引号更容易审计，也可在正式运行时复用同一文件。

### 3.4 `config` 辅助命令

#### scaffold：从模板生成 overlay

模板中有 `???` 时：

```bash
./scripts/astrolabe config scaffold \
  --task benchmark_binder_v1_4 \
  --out d.binder.overlay.yaml
```

它只生成待填写的 overlay；生成后仍需 `astrolabe run --task ... --overlay ... --dry-run`。

#### compose：从 base/template 交互式搭草稿

```bash
./scripts/astrolabe config compose
./scripts/astrolabe config compose \
  --base template \
  --task benchmark_binder_v1_4 \
  --name demo
```

交互提示中的写盘命令含义：

- `:write`：只写 `<name>.yaml`；
- `:write-all`：同时写 `<name>.yaml` 和 `d.<name>.overlay.yaml`；
- `:write-overlay`：只写 `d.<name>.overlay.yaml`。

推荐把可复用任务模板和任务特定差异分开：模板/standalone 保存完整契约，
overlay 只放此次实验的输入、输出、steps、seed 或明确调参。

#### wizard：补全已有 standalone

```bash
./scripts/astrolabe config wizard \
  --standalone /abs/path/config.yaml
```

默认只补全并打印下一步命令。只有确认过最终配置后才使用 `--execute`；可配合
`--print-config --dry-run`，不要让交互命令绕过审计。

## 4. `--print-config --dry-run` 到真实 run

### 4.1 标准两步法

第一步只 compose，不启动 evaluator：

```bash
./scripts/astrolabe run \
  --task benchmark_binder_v1_4 \
  --overlay d.demo.overlay.yaml \
  --overrides-file overrides.txt \
  --print-config --dry-run \
  > composed.yaml
```

第二步使用完全相同的来源去真实运行：

```bash
./scripts/astrolabe run \
  --task benchmark_binder_v1_4 \
  --overlay d.demo.overlay.yaml \
  --overrides-file overrides.txt
```

不要把 dry-run 打印的 YAML 手工改一份再跑；如果要修改，改 overlay 或
overrides-file，然后重新 dry-run。使用 one-shot 命令时同样先加
`--print-config --dry-run`，再移除 `--dry-run`。

### 4.2 Dry-run 检查清单

在打印结果中至少确认：

- `task_name`、`out_dir` 非空且没有写到错误的项目目录；
- `seqs`、`types`、`named_files`、`homologies` 与 chain mapping 一致；
- 每个 stage 的 `oligo`、`evaluator`、`mutators` 都指向已定义 alias；
- `stages.*.mutation_num` 和 `stages.*.mutator_weights` 是本次意图；
- `weighted_score_expression`、`acceptance_threshold`、`acceptance_gate` 没有
  误用旧字段或未解析的插值；
- evaluator/mutator 的模型路径、python runner、backend 和资源设置正确；
- `use_metropolis_acceptance` 与 `initial_temperature` 是否有意开启；
- output 目录和 run 数量符合预算。

`--print-config` 显示的是 compose 后配置，不是模型已经可加载的证明。依赖模型、
GPU、外部 daemon 或网络的任务，在 expensive run 前做一个缩小 steps/repeats 的
真实 smoke；smoke 只验证执行链，不替代科学结果。

### 4.3 真实 run 的成功判定

至少保存并核对：

1. 进程退出状态为 0；
2. run 日志含 `Pipeline completed successfully`；
3. run 目录存在 `config.yaml`、`scores.csv`、`structures/`、`sequences/` 和
   `logs/`（具体 backend 可能有额外 raw artifacts）；
4. `scores.csv` 非空，step、stage、oligo 和 decision 记录与配置相符；
5. 结构/序列文件非空且数量与任务合同相符；
6. 没有把失败目录、部分重试目录或仅有日志的目录当作成功结果。

典型目录：

```text
<out_dir>/<task_name>/
├── 001_run_.../
│   ├── config.yaml
│   ├── scores.csv
│   ├── structures/
│   ├── sequences/
│   └── logs/
└── runs.lock
```

大模型临时中间物默认分配到 `/tmp/gh_<run_name>`，`/tmp` 不可用时才退回 run
目录内的 `raw_model_results/`；自动管理的目录在成功后清理、失败时保留。需要
稳定保留时在配置中显式设置 `raw_model_results_dir`。不要以 Slurm 的
`COMPLETED` 单独判定科学成功；调度器只证明进程退出。

### 4.4 `steps`、`repeats` 和 seed

- `steps` 是优化循环的配置上限；具体模板可能把 step 0 用作初始评估，因此不要
  将它机械解释为恰好 N 次 mutation。
- `repeats` 是同一配置请求创建的 run 数。CLI 与兼容入口共享同一个生命周期。
- 显式 seed 为 s 时依次使用 s、s+1、…；整组 seed 必须在 1..2**32-1 内。
  seed=null 时由 task_name 与从零开始的 repeat_index 稳定派生。
- 实际 seed 记录在各 run 的 `config.yaml` 与 `provenance.json` 中。同 seed 不保证
  跨硬件、依赖版本或模型版本逐位一致。中断后重跑是新轨迹，不是逐步恢复。

## 5. 输入配置的最小心智模型

### 5.1 Protomer statement

常见顶层字段：

```yaml
seqs:
  A: "80-120"
  B: "+MKTAYIAKQRQISFVKSHFSRQDILDLWQ"
types:
  A: protein
  B: protein
homologies:
  B: /abs/input/target.json:A
```

- `80-120` 生成随机长度范围；`100` 是固定长度；
- `+SEQUENCE` 固定整条序列；
- `named_files` + `alias:chain:start-end` 可从 PDB/mmCIF 抽取结构片段；
- `types` 支持 protein、RNA、DNA、SMILES、CCD；省略时默认 protein；
- PTM/CCD 与 ligand 输入需满足 backend 和 Rosetta 参数契约，遇到失败应保留
  原始错误，不把 ligand 当普通蛋白字符处理。

### 5.2 Chain 与 protomer

多数 loss/constraint 中的 `chain_*` token 默认按 protomer 解析：如果 protomer
在 `oligo` 中重复，会匹配其所有拷贝。需要精确指定复杂结构 chain 时写
`chain:<ID>`，例如 `chain:B`。这对 binding site、interface、ligand-contact
和 post-hoc 计算都很重要。

### 5.3 Stage

典型 stage：

```yaml
stages:
  default:
    oligo: [A]
    evaluator: Protenix
    mutators: [dplm]
    mutation_num: 100%
    mutator_weights: [1.0]
    weighted_score_expression: "..."
    acceptance_threshold: "..."
    acceptance_gate: null
    early_stopping: null
  complex:
    _entrance: "..."
    oligo: [AB]
    evaluator: Alphafold3
    mutators: [LigandMPNN_itf]
```

`evaluator`/`mutators` 是 alias，不是任意 Python 类名；它们必须在对应
`evaluator.evaluator_terms` / `mutator.mutator_terms` 中定义。`_entrance` 决定
阶段切换；阶段内 `acceptance_gate` 仍是接受路径的硬约束。

## 6. 参数 owner 表：改哪里，不要混调

调参前先在最终 compose 配置中找到 owner。下面的 owner 是运行时语义，而不是
“所有出现同名字段的地方都可改”。

| 问题 | 唯一主要 owner | 作用 | 不要误改成 |
|---|---|---|---|
| 任务名/输出 | `task_name`, `out_dir` | 输出目录和运行身份 | 临时 shell 名 |
| 重复与随机 | `repeats`, `seed` | run 数量与随机状态；独立 seed 需显式编排 | `temperature` |
| 优化长度 | `steps` | 循环步数上限 | evaluator recycle |
| stage 状态 | `stages.<s>.oligo` | 本阶段评估哪个复合体 | `chain_*` |
| evaluator 选择 | `stages.<s>.evaluator` + `evaluator_terms` | 结构预测后端 | mutator alias |
| evaluator 参数 | `evaluator.evaluator_terms.<alias>` | recycle、sampling、模型参数等 | loss 权重 |
| mutator 选择 | `stages.<s>.mutators` + `mutator_terms` | 候选提议器 | evaluator 后端 |
| mutator 比例 | `stages.<s>.mutator_weights` | 各 mutator 分到的 mutation budget 份额 | `mutation_num` |
| 候选预算 | `stages.<s>.mutation_num` | 本 step 总 mutation budget | backend batch size |
| mutator 模型采样 | `mutator.mutator_terms.<alias>.temperature` | LigandMPNN/DPLM/ESM3 等采样 | `initial_temperature` |
| score 目标 | `stages.<s>.weighted_score_expression` | loss context 加权目标 | hard gate |
| score 阈值 | `stages.<s>.acceptance_threshold` | 达标时的 soft accept 条件 | score 权重 |
| 阶段进入 | `stages.<s>._entrance` | 从前一 stage 进入下一 stage | accept/reject 温度 |
| 硬接受门 | `stages.<s>.acceptance_gate` | 候选必须满足的 gate | soft score penalty |
| 早停 | `stages.<s>.early_stopping` | 达标后结束优化 | `_entrance` |
| 决策退火 | `use_metropolis_acceptance`, `initial_temperature` | worse-score 候选的决策层接受 | backend sampling |
| 参考结构 | `structure_start` | reference snapshot、TM-score/gate | weighted score 自动 bonus |
| 外部工具 | `env.yaml` / `env_local.yaml` | binary、weight、python 路径 | 任务科学参数 |
| 并发/资源 | Slurm wrapper、backend resource/env 字段 | CPU/GPU、内存、worker | repeats |
| post-hoc 指标 | `collect` 的 chain/preset 参数 | 从输出推导指标 | 重新修改原 run 分数 |

旧配置字段 `evaluator.name`、`mutator.names`、`mutator.mutation_rate`、
`mutator.mutator_weights` 不是当前新配置入口。迁移时把实现放入 terms alias，
再由 stage 引用。

## 7. 证据化调参闭环

### 7.1 一次只改一个 owner 家族

完整闭环按下列顺序走：

1. **冻结问题与验收**：先写成功指标、threshold/gate、分母和失败/缺失处理；
2. **冻结基线**：记录输入、源码 commit、最终 compose 配置、backend/model、显式
   seed 集合、steps、资源和预计 wall time；
3. **跑出基线分布**：从完整输出统计每个 loss term 的范围、方差、weighted
   contribution、gate 命中率、accepted/rejected 数和失败类型，不只看最好结果；
4. **定位 owner**：饱和/量纲问题改 score transform 或权重，硬条件误杀改 gate，
   proposal 过大改 `mutation_num`，mutator 预算失衡改 `mutator_weights`，模型候选
   多样性改 sampling temperature，worse-score 接受行为才改 decision temperature；
5. **做单变量 A/B**：相同输入、seed、backend、steps 和资源下只改一个 owner
   家族。先 dry-run；仅在执行风险不清楚时用缩小 steps/repeats 的 disposable canary；
6. **重放固定 candidate**：只改变 score 公式/阈值时，对同一个 before/after
   candidate replay，分开报告每项 delta 与 gate，避免把候选变化误当作公式效果；
7. **多 seed 确认**：显式 seed 清单、每条 `repeats=1`，报告全部启动 unit、分布、
   accepted endpoint、失败和异常；资源变化另做执行 A/B；
8. **按预先标准决策**：采纳、回退或继续下一轮，并把负结果、公式版本和唯一下一
   假设写回磁盘证据。

每轮至少保存原始 config/overlay、最终 `--print-config`、命令与环境/资源身份、
`scores.csv`/日志/产物，以及前后指标和 decision reason。不要在同一轮同时改 loss
权重、threshold、gate、`mutation_num` 和 temperature；那样无法归因。

### 7.2 区分 loss 权重、penalty、threshold 与 gate

- **loss 权重**进入 `weighted_score_expression`，改变连续目标的相对贡献；
- **penalty** 也是 weighted score 的一部分，可以只在越过某条线后连续增大；它
  不会像 gate 一样无条件拒绝候选；
- **acceptance threshold** 是 soft accept 条件：即使 weighted score 没变好，
  达标且 hard gate 通过时仍可接受；`early_stopping` 则决定是否结束循环；
- **gate** 是硬可行性条件，通常在 `acceptance_gate` 或 stage transition gate，
  即使 weighted score 变好也可以拒绝候选；
- `structure_start.mode: soft_select` 负责 reference candidate 选择；显式
  `reference_penalty` 和 hard `repredict_gate` 又是不同机制。不要把“保留参考
  结构”自动理解为正向加分或硬拒绝。

先统计每个 loss term 在真实样本中的范围、方差和 weighted contribution，再决定
权重。绝对数值大不等于应该给小权重：如果 term 已饱和，实际 delta 可能几乎为零；
如果某个 gate 在边界附近，微小变化就可能改变接受率。

### 7.3 区分 stage mutation_num 与 mutator_weights

`mutation_num` 是本 step 的总 mutation budget，owner 在
`stages.<s>.mutation_num`。百分比会按当前 mutable protomer 的总可变长度解析为
整数。`mutator_weights` 随后归一化，并把这个整数预算分给各 mutator。例如：

```yaml
stages:
  default:
    mutation_num: 50%
    mutators: [quality, surface, interface]
    mutator_weights: [0.45, 0.35, 0.20]
```

- 改 `mutation_num`：改变本 step 所有 mutator 合计的提议预算；
- 改 `mutator_weights`：改变各 mutator 分到的预算份额；
- 两者不是同一旋钮；不要为弥补一个过大的 mutation budget 而同时重排权重；
- stage 中列出的 mutator 会按顺序逐个执行，`mutator_weights` **不是**随机选择某个
  alias 的概率。position selector、固定残基、取整和模型输出仍会使实际变更位点数
  少于预算；
- `100%` 按所有 mutable protomer 的可变长度求预算，不等于每个位点最终都改变；
- 当前运行以 stage 字段为准，根层 `mutation_num` 仅作旧配置兼容镜像。

### 7.4 区分 backend sampling temperature 与 decision initial_temperature

```yaml
use_metropolis_acceptance: true
initial_temperature: 0.02      # decision 层：是否接受 worse score

mutator:
  mutator_terms:
    LigandMPNN_interface:
      _type: LigandMPNN
      temperature: 0.30         # backend：候选氨基酸采样
```

`initial_temperature` 只有在显式开启 `use_metropolis_acceptance` 时才走决策层
Metropolis worse-score 路径。LigandMPNN、DPLM、ESM3 等 alias 下的 `temperature`
改变模型采样分布。调高一个不会替代另一个；调参记录必须写明是哪一层。

### 7.5 区分 steps 与 repeats

- 想看单条轨迹是否继续改善，改 `steps`，保持 seed、repeats 和其他参数不变；
- 想看跨随机起点稳定性，显式编排 seed 集合并保持 `steps` 不变；不要把增加
  `repeats` 本身当成已经获得独立 seed；
- 报告平均值、分布、accepted count 和失败 count，不只报最好的一条轨迹；
- 分母应包含所有启动的独立 unit，超时、无模型、无输出应按预先冻结的科学合同
  计为失败或单独标注，不能运行后才排除。

### 7.6 区分 backend 与 resources

更换 AF3、Protenix、Boltz 或 Chai 是科学方法变化，不是资源调优。修改 GPU、
CPU、内存、worker、Slurm partition、container runner 是执行条件变化。二者必须
分开记录；若要公平比较，冻结模型/backend 和 score contract，只改变资源并确认
没有 OOM、超时或路径差异。

## 8. 调参结果的可审计记录

接受决策属于运行时使用的公式和配置。修改权重、归一化或 gate 后，应重新计算
分项贡献，不能直接沿用旧轨迹的“改善”结论。加权分数增加也不意味着每个指标
都改善；报告中保留各指标的方向、范围和异常值。

批量运行应保留逐轨迹 identity、输入、seed、失败项和原生输出。执行完成与科学
成功分开判断：缺失指标、无效结构或环境错误不能补成通过。使用下面的记录格式：

```text
公式版本/overlay:
固定项（backend、seed、steps、repeats、资源）:
唯一改变的 owner:
候选 mutation:
weighted score before/after:
每个 loss term before/after 与贡献份额:
gate/threshold 是否触发:
accepted/rejected 及 reason:
输出合同是否完整:
下一轮假设:
```

没有这些信息时，只能说“运行过”或“候选被写出”，不能说参数已校准。

## 9. 正式后处理：collect → filter → refold → report

后处理应在运行输出合同已核对后执行。默认正式筛选不建议使用
`--allow-missing-required`；缺 required 指标时优先 refold 补齐，或修复 collect
输入，而不是把缺失当作通过。

### 9.1 collect：输出树到 SQLite/CSV

```bash
./scripts/astrolabe posthoc collect \
  --out-root outputs \
  --db analysis/posthoc.sqlite \
  --export-csv analysis/posthoc.csv \
  --workers 8 \
  --binder-chain B \
  --target-chain A
```

多个 outputs 根可以重复 `--out-root`。有正式指标合同时显式指定 preset 或
required metrics：

```bash
./scripts/astrolabe posthoc collect \
  --out-root outputs \
  --db analysis/posthoc.sqlite \
  --required-metrics-preset analysis/metrics.yaml
```

collect 是增量的：同一输出不变时重跑应报告 `skipped>0`，而不是重新计算全部
记录。`--binder-chain`/`--target-chain` 必须与实际结构 chain 语义匹配；不要把
默认 A/B 机械套到所有任务。

### 9.2 filter：表达式/preset 筛结构

先预览 preset：

```bash
./scripts/astrolabe posthoc filter \
  --preset analysis/metrics.yaml \
  --preview-preset
```

表达式筛选并写 hardlink 结果：

```bash
./scripts/astrolabe posthoc filter \
  --db analysis/posthoc.sqlite \
  --filter-expr 'weighted_score > 0 and AB.iptm >= 0.7' \
  --prefer-oligo AB \
  --dst analysis/selected \
  --report-csv analysis/selected/filter_report.csv
```

默认是 hardlink；只有明确需要复制时才加 `--copy`。正式筛选时选择
`--run-selection representative-per-run` 还是 `all-passing` 必须在分析计划中
预先冻结：前者每个 logical run 取最后 stage 中 weighted score 最高代表，后者
保留全部通过行。`--dry-run` 只计算数量，不写结构。

如果 preset-required 指标缺失/无效，filter 应失败并提示 refold；正式结果不要用
`--allow-missing-required`。该开关只用于探索缺失字段的影响，且报告中必须显式
标注。

### 9.3 refold：补齐或重算预测指标

规范的高层命令是：

```bash
./scripts/astrolabe posthoc refold \
  --db analysis/posthoc.sqlite \
  --preset analysis/metrics.yaml \
  --dst analysis/refold \
  --evaluators af3,protenix \
  --workers 2
```

也可用 `--filter-expr` 选择待 refold 记录。正式运行前：

```bash
./scripts/astrolabe posthoc refold \
  --db analysis/posthoc.sqlite \
  --filter-expr 'weighted_score > 0' \
  --dst analysis/refold \
  --dry-run
```

dry-run 会准备 manifest、验证输入并统计工作量，但不会启动模型或 collect-back。
如果保存的 config 使用远端相对路径，使用经审计的 `--path-map FROM=TO` 或
`--relpath-base /abs/base`；不要静默换模型或输入。

低层 `refold-export`/`refold-run` 只适合调试或已有 manifest 的恢复场景；普通用户
优先用 `posthoc refold` 以保留 selector、collect-back 和 provenance。

### 9.4 report：从 DB 生成趋势和成功率

```bash
./scripts/astrolabe posthoc report \
  --db analysis/posthoc.sqlite \
  --preset analysis/metrics.yaml \
  --dst analysis/report
```

`report` 与 `filter/refold` 一样，必须且只能给 `--preset` 或 `--filter-expr` 之一；
这个 selector 定义成功率分子。报告必须写清 denominator、selector、required
metrics、refold 是否参与以及失败/缺失输出如何处理。`runs_processed` 等 summary
counter 不能替代对 SQLite、CSV、结构文件和日志的事实检查。

## 10. 失败排查：按四类分流

### 10.1 配置类

**症状**：Hydra `ConfigKeyError`、MissingConfigException、字段被忽略、dry-run
和真实 run 读到不同值。

**顺序**：

1. 重跑 `--print-config --dry-run`，保存完整 YAML；
2. 检查 `--task`、`--standalone`、`--overlay` 和 overrides 顺序；
3. 确认 overlay 是否在最后，config-group `env=...` 是否在 compose 阶段；
4. 搜索旧入口字段，迁移到 `evaluator_terms`、`mutator_terms` 和 stage alias；
5. 对 process substitution 传入的 YAML 改用真实临时文件，不使用 `/dev/fd`；
6. 检查 standalone 相对路径是否因 `--chdir` 指向了预期目录。

不要因为配置错误就换 backend、降低 gate 或加 fallback；先修 owner。

### 10.2 环境/资产类

**症状**：import 失败、模型文件找不到、外部 binary 缺失、CUDA/daemon 启动失败、
节点无网络、container 和 host 导入了不同源码。

**顺序**：

1. `./scripts/astrolabe env doctor --task <task> --strict`；
2. 检查最终配置中的绝对路径、Python runner 和 `CUDA_VISIBLE_DEVICES`；
3. 检查模型 cache、受限资产、许可证和容器/SIF provenance；
4. 对目标节点做真实小 import/evaluator smoke；
5. 记录原始错误和节点/环境，不以 `--soft-fail` 当作通过。

典型边界：dry-run 不加载所有模型；Slurm COMPLETED 不证明模型运行；远端绝对
路径不能直接用于本地 replay；AF3/DPLM 等后端的无网失败不是 loss 失败。

### 10.3 科学失败类

**症状**：evaluator 已完成并有结构，但 gate 不通过、weighted score 下降、结构
偏离 reference、interface/几何/稳定性指标恶化。

**顺序**：

1. 确认 run 到达 evaluator/loss/decision，而不是中途异常；
2. 比较 accepted endpoint 与被拒候选，不只看最后一行；
3. 把连续 loss、threshold 和 hard gate 分列；
4. 检查 score 量纲是否饱和、是否重复归一化、reference penalty 是否触发；
5. 只改变一个 owner 家族后做短 replay，保留负结果；
6. 在新公式下重新计算旧 candidate，不沿用历史 accepted 标签。

科学失败不是故障，也不是应该静默换模型的理由。若重平衡后 delta 改变，应以
新配置结果为准，并保留旧结果作为历史版本。

### 10.4 输出合同类

**症状**：进程退出 0 但没有 `scores.csv`/结构；collect 处理 0 个 run；筛选只
生成表没有结构；run 目录落在 sibling 时间戳路径；重试重复计算。

**顺序**：

1. 以日志 `Pipeline completed successfully`、run `config.yaml`、scores、结构和
   sequences 逐项核对；
2. 按实际命名同时检查 `001_run/` 和带 job/array suffix 的目录；
3. 确认 evaluator 是否本来就有不同输出合同，不能机械要求所有任务同样文件；
4. 检查 hardlink/copy 目标和 `filter_report.csv`；
5. 只对缺失合同的 run 重试，保留已完整 sibling 并记录 canonical 归属；
6. collect 增量重跑，确认 unchanged records 被 skip。

## 11. 长任务、Slurm 与恢复

Slurm wrapper 只负责提交任意命令，不理解 Astrolabe 业务逻辑：

```bash
scripts/containers/slurm_submit.sh \
  --partition gpu --gpus 1 --cpus 8 --mem 32G --time 04:00:00 -- \
  ./scripts/astrolabe run --task benchmark_binder_v1_4
```

提交前先让命令在当前 checkout dry-run；提交后同时看 job state、每行日志、run
目录和科学产物。task file/queue wrapper 的每行格式必须遵循对应 scheduler contract，
不要添加未经支持的 `label | command`。资源并发以 live capacity 为准，不把持久分片
数量当作即时 worker 数。

如果作业失败，先区分：

- 科学命令未启动：配置/环境/队列 wrapper；
- evaluator 启动后崩溃：模型/节点/后端环境；
- evaluator 完成但 candidate 被拒：科学结果；
- 进程退出但产物不全：输出合同。

恢复时按 manifest、exitcode、PID/Slurm、日志和实际输出重算状态；不要只看 stale
claim、旧 agent 消息或一次 `squeue`。已有完整科学输出不应因 controller 状态过时
而重跑。

更多提交选项和资源边界见 [`docs/slurm.md`](slurm.md)。

## 12. 特殊任务参考与边界

本手册只保留主流程；需要专门契约时进入下列维护中的参考页：

- [`docs/command_design_rules.md`](command_design_rules.md)：新增/改名 CLI、配置和
  completion 的公共契约；
- [`docs/third_party_envs.md`](third_party_envs.md)：host/container、vendor、权重、
  外部工具和隔离环境；
- [`docs/slurm.md`](slurm.md)：Slurm wrapper 与提交参数；
- [`docs/predockers.md`](predockers.md)：ligand/docking 预处理；
- [`docs/nonstandard_residue_contracts.md`](nonstandard_residue_contracts.md)：PTM、
  CCD、SMILES、Rosetta params；
- [`docs/cyclization_topology_examples.md`](cyclization_topology_examples.md)：环化、
  sparse relative patch 和真实结构检查；
- [`docs/d_n_symmetry.md`](d_n_symmetry.md)：D2/D3 对称性指标与 gate；
- [`docs/release_checklist.md`](release_checklist.md)：发布前许可证、transient 和
  release audit；
- [`docs/changelog.md`](changelog.md)：按日期记录用户可见变更。

代码 owner、运行时 score context、扩展和开发验证见
[`docs/developer_guide.md`](developer_guide.md)。本手册不复制完整 `--help`；命令
参数发生变化时，以当前 parser 的 `./scripts/astrolabe <command> --help` 和本手册的
覆盖顺序/owner 规则共同复核。

## 13. 最终验收清单

在把一批结果交给下游前，逐项打勾：

```text
[ ] checkout、commit、env 文件和 runner 身份已记录
[ ] env doctor --strict 通过，受限资产/许可证边界已确认
[ ] 入口是 design/refine/run 中有意选择的一种
[ ] template/standalone/overrides/overlay 的最终顺序已审计
[ ] --print-config --dry-run 输出已保存
[ ] 真实 smoke（如需要）通过，且没有静默更换 backend/resource
[ ] 正式 run 日志和 output contract 完整
[ ] scores.csv 里的 accepted endpoint 与最后一行已区分
[ ] 每轮调参只改一个 owner 家族，保留 delta 和失败原因
[ ] loss 权重、threshold、gate、mutation_num、mutator_weights、两类
    temperature、steps/repeats、backend/resources 未混为一类
[ ] collect 已写 DB/CSV，并确认增量 skip 行为
[ ] filter preset/denominator/run-selection 已冻结；正式筛选未使用
    --allow-missing-required，或缺失原因已明确标注
[ ] 必要时 refold 已 collect-back，再运行正式 filter
[ ] report 写清分母、缺失/失败处理和公式版本
[ ] 历史 candidate（包括 Cre 041）已按当前 score 公式 replay，未沿用旧接受标签
```

只有完成这份清单，才应把结果称为“当前 Astrolabe 流程下可复现、可审计的成功运行”。
