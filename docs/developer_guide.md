# Astrolabe 开发者手册：Owner、数据流与扩展合同

本文是 Astrolabe 在本仓库中的 canonical
开发者手册。它回答四个问题：行为应由谁负责、数据如何流动、公共参数怎样演进、
以及用什么最小证据证明修改完成。用户如何准备输入、运行、调参和验收，见
[用户手册](user_guide.md)。

开发环境可使用 `python -m pip install '.[dev]'` 安装测试与打包工具。
模型相关测试另需对应的 vendored 运行时；验收命令见 `justfile`，当前覆盖范围见
[发布验收](release_acceptance.md)。公开源码包不依赖内部项目管理文件。

## 0. 名称、入口与基本原则

### 0.1 Astrolabe 与历史兼容名的边界

- **Astrolabe** 是当前公开的软件名、Python namespace 和唯一安装 CLI。
- **GuidedHallucination** / **GH** 只作为历史名称或环境兼容合同保留；不安装 `gh` alias，以避免与 GitHub CLI 冲突。
- 安装后的 console entry point 是 `astrolabe=astrolabe.cli:main`。
- 开发 checkout 使用 `./scripts/astrolabe`；`scripts/gh_dev` 仅委托到该入口。
- 内部 `GH_*`、环境组 `gh`、`~/.gh_envs`、`.gh_python_path` 不随公共 CLI 重命名；bootstrap 固定为 `scripts/install.py`。

本文中的“Astrolabe pipeline”指：

```text
输入/配置
  -> initializer
  -> initial evaluate -> post-eval -> initial score/loss snapshot
  -> 每个后续 step：mutator（可先经过 predocker）
       -> evaluate -> post-eval -> loss/context -> decision/gate
       -> accept/rollback -> stage transaction（如满足入口）
  -> 持久化输出
  -> posthoc collect/filter/refold/report
```

### 0.2 命名迁移边界

- 公共 Python namespace 与 CLI 分别是 `astrolabe` 和 `astrolabe`。
- `scripts/gh_dev` 仅为委托到 `scripts/astrolabe` 的兼容包装器。
- 内部兼容名不随公共 CLI 一起重命名：保留 `GH_*`、`~/.gh_envs`、`.gh_python_path`、环境组 `gh`；bootstrap 固定为 `scripts/install.py`。

### 0.3 开发的五条硬规则

1. **先找 owner，再写代码。** 不把领域行为塞进 CLI、`run.py` 或任务脚本。
2. **先找 consumer，再加 knob。** 没有已验证 consumer 和 acceptance criterion 的参数
   不进入公共配置。
3. **失败可见。** 缺配置、缺指标、缺工具和 backend 失败应在 owning path 明确报错；
   fallback 必须显式、可配置、可审计。
4. **输出是合同。** `config.yaml`、`scores.csv`、结构路径、manifest 和 posthoc DB/CSV
   都有下游 consumer，不能把“文件存在”当作内容正确。
5. **最小充分验证。** 先运行覆盖 owner 的现有测试和真实可观察边界；不同风险边界才
   增加验证层，不重复堆叠 lint、全套测试、reviewer 和人工复核证明同一件事。

### 0.4 功能增加前必须写清楚

每个非平凡修改先回答：

1. 用户可观察行为是什么？
2. 公共表面是 CLI、配置、Python API、运行输出还是 posthoc schema？
3. 哪个模块是单一 owner？
4. 当前哪个实现/registry/parser 可复用？
5. 失败应在哪一层暴露？是否真的需要 fallback？
6. 哪个最小命令能证明目标行为？哪个直接受影响的共享边界还需验证？

推荐形状：

```text
小而明确的用户表面
  -> 显式 config/data model
  -> 共享 builder / registry / helper
  -> backend 实现或 wrapper
  -> owner-boundary test / realistic smoke
```

避免形状：

```text
新 flag
  -> CLI special case
  -> compose special case
  -> run.py private runtime key
  -> posthoc 再猜一次语义
```

## 1. Repository map 与单一 owner

```text
setup.py                         astrolabe console entry point
scripts/astrolabe                当前 checkout 的 canonical 开发入口
scripts/install.py               canonical server/container bootstrap
scripts/gh_dev                   唯一保留的 launcher 兼容入口
scripts/installation/            host env、asset 与 software catalog
scripts/containers/              image build、container kit 与 SLURM wrapper
scripts/release/                 export、license inventory 与 release audit
scripts/maintenance/             transient cleanup 与 vendor patch maintenance
scripts/validation/              显式 smoke/runtime validation
examples/benchmarks/             可重放性能 benchmark
astrolabe/cli.py             argparse command/handler 与 Hydra compose
astrolabe/cli_command_definitions.py
                                 共享命令定义
astrolabe/cli_interactive.py config compose/scaffold 等交互 builder
astrolabe/shell_completion.py Bash/Zsh 共用 completion engine
astrolabe/app.py             repeat lifecycle；延迟加载 run.py
run.py                           单次 run 的薄 orchestration
astrolabe/argparser.py       runtime normalization、schema validation、run dir
astrolabe/run_config_lifecycle.py
                                 persisted config normalization/pruning
astrolabe/core/              输入、protomer/chain/structure 等 domain model
astrolabe/initializers/       第一个序列/结构状态
astrolabe/structure_start.py 输入结构的 reference/select/gate 语义
astrolabe/stage_manager.py   当前 stage 的解析和可见设置
astrolabe/stage_transaction.py
                                 stage entry 的 plan/attempt/decision/commit/rollback
astrolabe/stage_transition.py
                                 transition gate 数据、score context 与 artifact
astrolabe/evaluators/         evaluator registry、StandardScores、wrapper
astrolabe/post_eval_hooks/   evaluate 后、loss 前的结构处理
astrolabe/losses/            loss registry、expression engine、gate expression
astrolabe/decision_context.py
                                 所有 acceptance 路径共享的 score context
astrolabe/decision_maker.py  accept/reject、history、scores.csv
astrolabe/mutators/           proposal engine、registry、backend wrapper
astrolabe/selectors/         可复用位点选择语义
astrolabe/structure_processors/         mutator 前的结构变换
astrolabe/runtime_requirements.py
                                 task-aware env/asset requirement inference
astrolabe/posthoc/           collect/filter/refold/report 与 SQLite owner
astrolabe/analysis/          跨项目复用的分析逻辑
astrolabe/experimental/      尚未稳定的实验性公共流程
astrolabe/utils/             无独立领域语义的共享底层工具
config/                          Hydra templates、component aliases、env specs
examples/                        可复用例子和 example-specific helper
vendor/                          仓库拥有的第三方源码/submodule/wrapper
tests/                           行为与回归测试
```

### 1.1 Component 类名与配置身份

| Python package | 责任 | 公开类名示例 | 配置合同 |
|---|---|---|---|
| `astrolabe/evaluators/` | 运行 predictor/scorer 并产生标准分数 | `AlphaFold2Evaluator`、`AlphaFold3Evaluator`、`Chai1Evaluator` | 既有 `_type` key 保持稳定 |
| `astrolabe/mutators/` | 提出 sequence/structure proposal | `RandomSubstitutionMutator` | 既有 `_type` key 和运行时 `.name` 保持稳定 |
| `astrolabe/initializers/` | 构造初始 sequence/structure state | role-specific initializer classes | 既有 mode/`_type` key 保持稳定 |
| `astrolabe/structure_processors/` | mutator 前的通用结构处理 | `AutoDockVinaLigandPoseRescue`、`SymmetryRescueProcessor` | 既有 predocker/processor `_type` key 保持稳定 |

Python 类名现在直接表达角色，但配置 registry identity 不随类名迁移。`StructureProcessor` 是比 rescue 更广的通用抽象；rescue 是其中一种 role-specific implementation，并非所有 structure processor 都是 rescue。

### 1.2 Script 文件布局

| Path | Owner |
|---|---|
| `scripts/astrolabe` | source checkout 的 canonical launcher |
| `scripts/install.py` | canonical bootstrap |
| `scripts/gh_dev` | 唯一保留的真实兼容入口 |
| `scripts/installation/` | host environment、asset prefetch、software catalog |
| `scripts/containers/` | Apptainer/container build、kit 与 scheduler wrapper |
| `scripts/release/` | source export、license generation、release audit |
| `scripts/maintenance/` | repo cleanup 与 vendor patch maintenance |
| `scripts/validation/` | 独立 smoke/validation program |
| `examples/benchmarks/` | benchmark driver，不属于安装脚本 |

Post-hoc collect/filter/refold 与 experimental candidate-batch 由 canonical `astrolabe` subcommand 拥有，不再保留平行 legacy script alias。

判断 owner 的方法不是“哪个文件最方便改”，而是“哪个模块已经拥有该状态的
lifecycle”。例如：

- 新命令只在 CLI 解析 intent；设计语义进入 builder/config/core。
- stage switch 的副作用进入 transaction/transition，不进入 evaluator。
- gate 可见字段进入 shared decision context，不在 `run.py` 拼临时 dict。
- repeated post-run selection 进入 `posthoc/`，不新建任务分析脚本。
- mutator 前的结构变换进入 predocker，不把 docking 分支写进 mutator 主逻辑。

## 2. 从 `astrolabe` 到输出的真实数据流

### 2.1 CLI 与配置组合

正式入口链是：

```text
./scripts/astrolabe 或已安装 astrolabe
  -> astrolabe.cli:main
  -> argparse command handler
  -> _compose_hydra_config()
  -> _run_from_config()
  -> astrolabe.app.run_from_cfg()
```

`astrolabe.cli` 应保持轻量。只有真实运行被请求时，
`astrolabe.app._load_run_module()` 才动态加载仓库根 `run.py`。因此
`astrolabe --help`、completion、`astrolabe env doctor` 和 config inspection 不应被 AF3、
PyRosetta、Torch/JAX 或其他可选 backend 的 import 拖垮。

`astrolabe run` 先按固定次序收集 CLI overrides：`--overrides-file` 中的有效行、
`--overrides` 字符串、命令尾部 `hydra_overrides`。之后分两条路径：

**没有 `--standalone`：**

1. 从 `config_dir` 选择 `--task <name>`；未指定时以 `base` 为基底。
2. 将上述全部 overrides 交给 Hydra compose；config-group override（例如
   `env=env`）改变 defaults 选择，普通 dotlist 改字段，同一字段按收集次序后者胜出。
3. `--overlay` 最后由 `OmegaConf.merge` 合并。

**有 `--standalone`：**

1. 仍先选择 repo template；未指定 `--task` 时以 `base` 为基底。
2. 只把左侧名称对应 `config_dir` 子目录的 config-group overrides 交给 Hydra
   compose；其余 overrides 暂存。
3. 剥离 standalone 自己的 `defaults:`，再把其内容合并到 repo template 之上。
4. 将暂存的普通 dotlist overrides 按原收集次序合并到 standalone 之后。
5. `--overlay` 仍是最后一次 merge。

两条路径中，`--print-config --dry-run` 都打印 overlay 之后的最终组合视图，并在
进入 `app` 前退出。

不要从参数在命令行中的书写顺序猜优先级，也不要在 backend 中重新读取原始 YAML。
所有 runtime consumer 只读组合后的 config。

### 2.2 公共 Python API

`astrolabe.Astrolabe` 的构造参数是已经组合完成的 Hydra `DictConfig`，不是 YAML 路径。公共示例应使用 `initialize_config_dir(config_dir=str(astrolabe.config_dir()), version_base=None)` 和 `compose(...)`，再调用 `Astrolabe(cfg).run()`。`release_protein` 会执行 AF3 推理，因此需要 GPU、AF3 参数/资产和匹配的环境配置；CPU 示例必须走 `examples/01_cpu_reproduce.py`，不能把配置路径字符串传给应用类。

### 2.3 `astrolabe config compose` 的边界

`astrolabe config compose` 是交互式 draft builder，不是第二套 schema：

- 默认 `:write` 只写 `<name>.yaml`。
- `:write-all` 同时写 `<name>.yaml` 与 `d.<name>.overlay.yaml`。
- `:write-overlay` 只写 overlay。
- builder 的 section 是 `basic/evaluator/mutator/loss/predocker/stages`；它必须生成
  正常 `astrolabe run` 能组合的 config，不能发明 builder-only runtime 字段。
- 非 TTY 必须早期返回清楚错误和 exit code `2`，不能抛交互库 traceback。

task-private overlay 不放 repo root；可复用 overlay/example 放 `examples/`，项目私有值
留在 owning project/task workspace。

### 2.4 App 与 repeat lifecycle

`astrolabe.app.run_from_cfg()`：

1. 将组合 config 序列化为未 resolve 的 YAML，作为每次 repeat 的干净基线。
2. 校验 `repeats > 0`。
3. 每个 repeat 重建 config、设置 `repeat_index` 并隔离显式
   `raw_model_results_dir` 子目录；当前 runtime 会在显式 `seed` 非零时逐 repeat
   复用它，只有 seed 缺失/为零时才为该 repeat 生成随机值。
4. `global_config.reset()` 后调用 `run._run_pipeline()`。
5. graceful stop 只在一个 repeat 完成后停止后续 repeat。

不要让 repeat 之间共享可变 singleton、backend handle 或未显式拥有的 cache。
每次 run 的资源由 `_shutdown_pipeline_resources()` 收尾。

### 2.5 单次 pipeline

`run._run_pipeline()` 的主路径是：

```text
parse_arguments(cfg)
  -> 创建 run dir，规范化配置，写 persisted config.yaml
initialize_pipeline_components(args)
  -> global config / protomers / cyclization
  -> initializer
  -> mutators / LossEngine / StageManager / evaluators / predockers
  -> structure_start state
load_or_evaluate_initial()
  -> StandardScores -> post-eval hooks -> final score snapshot
run_optimization_loop()
  -> mutate -> evaluate -> post-eval -> loss/context -> decide
  -> stage entry transaction（如满足入口）
  -> incremental scores.csv / structures / sequences
shutdown + owned raw-output cleanup
```

`run.py` 只拥有步骤顺序和 module wiring。任何可复用规则都应下沉到 initializer、
stage transaction、decision context、component registry 或 output helper。

### 2.6 运行输出与 posthoc

典型 run contract：

```text
<out_dir>/<task_name>/<NNN_run...>/
  config.yaml
  scores.csv
  logs/
  structures/
  sequences/
  raw_model_results/   # 仅当显式/回退到 run dir 时
```

关键 consumer：

- `decision_maker.Decisions` 维护 accepted/rejected history，并增量重写
  `scores.csv`。
- `posthoc.collect` 读取 `scores.csv`、`structures/` 和 `config.yaml`，计算必要派生
  指标并 upsert 到 SQLite；文件 signature 支持增量跳过未变 run。
- `posthoc.filtering` 从 DB 选行，写 `filter_report.csv` 和筛选后的结构/DB。
- `posthoc.refold` 从 DB 的 sequence metrics 与 source `config.yaml` 生成或执行
  refold plan/manifest。
- `posthoc.report` 写 summary/per-step CSV 和报告。

改变 column、step/stage 语义、structure naming 或 persisted config 时，必须同时审查
这些 consumer。成功判定应检查行数、列、accepted/stage 关系、结构映射或 manifest
字段，而不只是路径存在。

## 3. 配置 schema、alias 与 lifecycle

### 3.1 配置是公共 API

可复用默认值放 `config/base.yaml`，task 工作流放 `config/<task>.yaml`，机器路径和
权重放 env spec 或环境变量。公共 schema 应表达领域意图，而不是 backend 调用栈：

```yaml
evaluator:
  evaluator_terms:
    Fold:
      _type: protenix
mutator:
  mutator_terms:
    Propose:
      _type: LigandMPNN
predocker:
  predocker_terms:
    Align:
      _type: hotspot_reference_align
stages:
  default:
    evaluator: Fold
    mutators: [Propose]
```

原则：

- `*_terms.<alias>._type` 定义实现；stage 和其他 component 引用 alias。
- alias 是任务级可读身份，`_type` 是 registry identity；不要互换。
- stage 引用不存在的 evaluator/mutator alias 必须在 runtime 初始化前失败，并列出
  available aliases。
- 只在跨 stage 不变时使用 global field；随 stage/oligo 改变的行为由
  `stages.<name>.*` 拥有。
- 新输入语法先进入 `astrolabe/core` parser/model，不在 evaluator/mutator 各解析
  一遍。
- 新 schema 默认严格拒绝 unknown/deprecated 字段；兼容读取必须有真实 consumer、
  明确 migration 和删除条件。

### 3.2 Stage-owned 字段

新的 runtime 配置应以 stage 为 owner：

- `evaluator`
- `oligo`
- `mutators`
- `mutator_weights`
- `mutation_num`
- `weighted_score_expression`
- `acceptance_threshold`
- `acceptance_gate`
- `early_stopping` / `prefix_early_stopping`
- 非 default stage 的 `_entrance`
- `transition_consistency_gate`

根级 `mutation_num` 和 `initial_temperature` 等历史字段仍可能存在于 base/config，
但不应成为新 feature 的 owner。`argparser._validate_schema_compatibility()` 已拒绝
`evaluator.name`、`mutator.mutation_num`、非默认 `mutator.mutation_rate` 和
`mutator.mutator_weights` 等过时写法。

`mutation_num` 可是整数、按 protomer mapping 或百分比字符串；它最终针对 mutable
length 解析并校验不超过可变位点数。不要在 mutator wrapper 中再次解释这个预算。

### 3.3 Enabled alias 与 persisted cleanup

`parse_arguments()` 根据所有 stage 计算真正使用的 aliases：

- mutator aliases 来自 `stages.*.mutators`；structure/RFD3 initializer 的
  `initializer.sequence_designer` 也强制加入。
- evaluator aliases 来自 stage evaluator、default fallback，以及 transition gate
  自己指定的 evaluator。
- predocker aliases 由实际 runtime usage 分析得到。

这些结果写入 derived `enabled_aliases`。component factory 与 persisted config 都应
复用它，不要各自推导一个近似集合。

写入 run 的 `config.yaml` 不是原始输入副本，而是
`run_config_lifecycle.build_persisted_run_config()` 生成的可重放、最小必要视图：

- 归一化兼容 initializer 字段，移除可由 owner 表达的 legacy root mirror。
- evaluator/mutator/predocker 只保留 enabled aliases 及必要 backend config。
- loss 只保留 stage expressions 实际引用的 `loss_terms`，并递归保留依赖的
  `expression_terms`。
- env 只保留 task requirement 或 refold replay 需要的路径。
- AF3/Protenix refold replay 所需 backend config 例外保留。
- derived selectors、enabled aliases 等在 pruning 前注入，因此 persisted view 代表
  实际执行语义。

扩展 lifecycle 时必须同时回答：新字段是否需要 replay？谁推导它？是否应写入
`config.yaml`？posthoc/refold 是否消费？不能为了“完整”把整份 env 和所有未启用
backend 都持久化。

### 3.4 临时 raw outputs 的 ownership

- 用户显式设置 `raw_model_results_dir`：Astrolabe 使用它，绝不自动删除。
- 未设置：Astrolabe 尝试创建 `/tmp/gh_<run>`，写 `.gh_auto_managed` marker；不可用时回退
  到 `<run>/raw_model_results`。
- 仅成功完成且 marker 可验证时自动删除 owned 目录。
- run 失败时保留 raw outputs 用于诊断。
- marker 缺失或删除失败必须报错，不能扩大删除范围。

任何新 scratch/cache 都应同样定义 owner、marker、成功/失败 retention 和 replay
需求。

## 4. 核心 component 的 owner 与扩展清单

### 4.1 输入与 domain model

Owner：`astrolabe/core/`，包括 protomer statement、sequence parsing、folding
input、chain resolution 和 structure parsing。

扩展清单：

1. 定义显式数据字段或 typed model，不靠文件名/字符串位置暗示语义。
2. 在 core 一次解析，向 evaluator、initializer、mutator 提供已解析对象。
3. chain identity 需要无歧义时使用明确 chain/residue mapping，不猜 protomer 与
   structure chain 的对应。
4. PDB/mmCIF、非标准残基、covalent/cyclization 等边界沿用现有 domain helper。
5. invalid state 在 backend 之前失败，并说明字段路径与期待格式。

非标准残基与拓扑的稳定专题合同见
[非标准残基合同](nonstandard_residue_contracts.md)、
[环化拓扑示例](cyclization_topology_examples.md) 和
[D-n 对称性](d_n_symmetry.md)。

### 4.2 Initializer

Owner：`astrolabe/initializers/`；`run.build_initializer()` 只负责选择和 wiring。

当前 mode 包括 `random`、`dplm`、`aa_set`、`mask_mix`、`pssm`、`all_a`、
`x_mix`、`structure`、`rfd3`。每一种“第一条序列如何产生”的策略都是
`Initializer`，返回 `InitializerResult`。

结构型 initializer 的合同：

- `initializer.sequence_designer` 必须指向具备 structure-based design capability 的
  mutator alias，即使该 alias 不在 stage mutators 中也会被构建和持久化。
- `initializer.structure` / `initializer.rfd3` 持有 source、`chain_map`、必要时
  `motif_map` 与 partial mapping 语义。
- 只替换 mutable protein positions；fixed segment、locked position 和 covalent
  endpoint 不因初始化被偷偷解锁。
- RFD3 先生成结构，再复用同一个 structure-to-sequence 路径，不创建第二套 inverse
  folding pipeline。

扩展 initializer：增加实现类、在 `build_initializer()` 接线、声明 runtime
requirements、确保 initializer-only mutator 不被 pruning，并验证 fixed/mapping/
manifest/output contract。

### 4.3 `structure_start`

Owner：`astrolabe/structure_start.py`。它与 initializer 不同：initializer 决定
第一个设计 sequence/structure state；`structure_start` 决定现有结构怎样作为
reference、candidate selection 或 step-0/pre-accept gate。

其配置和结果必须通过 `StructureStartConfig/State` 与 `ReferenceGateResult` 流动，
reference score 字段由 shared decision context 注入。不要把 reference gate 偷塞进
initializer 或任务 CLI。

### 4.4 Stage manager、transaction 与 transition

Owner 分工：

- `stage_manager.py`：解析 stage/default inheritance，暴露 current evaluator、oligo、
  mutators、budget 和 expressions。
- `stage_transaction.py`：stage entry 的 plan、attempt、decision，以及 commit/rollback
  的有序副作用。
- `stage_transition.py`：transition consistency config/context/result、reference
  comparison、artifact/manifest 和 gate score fields。

进入新 stage 可能需要换 evaluator/oligo 并重新评估，所以它不是把
`current_stage` 字符串直接改掉。修改时必须覆盖：source snapshot、target
re-evaluation、seed-pass、active attempt、失败 rollback、成功 commit，以及
`scores.csv` 中 stage-change 行的语义。

**强制合同：** transition-consistency fields 必须经
`stage_transition.transition_consistency_gate_score_context` 产生。seed-pass 与
active-gate acceptance 必须看到同一组字段；不得在其中一个路径手写近似 dict。
default stage 不能定义 transition-consistency gate。

### 4.5 Evaluator 与 `StandardScores`

Owner：`astrolabe/evaluators/registry.py`、具体 evaluator 和需要外部进程的
`astrolabe/evaluators/wrappers/`。

Evaluator 输入标准 folding inputs，输出 `StandardScores`：

- `names` 与 oligomer-keyed `plddt/ptm/iptm/pae/pde/complex_paths`；
- protomer-keyed per-residue scores；
- chain-pair metrics；
- optional `ComplexPoses`；
- opt-in `ScoreContribution`。

公共 normalized score 在这里统一尺度与 shape。例如 pLDDT 的 `[0,100]` backend
输出会归一到下游假设的 `[0,1]`。不要让每个 loss 猜 backend 单位。

增加 evaluator：

1. 注册轻量 factory；registry 模块不能 import 重 backend。
2. wrapper 负责 subprocess、模型调用、raw parse 和 backend-specific output。
3. 返回 normalized `StandardScores`；定义每个 metric 的 scope、shape、单位和方向。
4. raw artifacts 留在 owned raw output；稳定结构路径进入 `complex_paths`。
5. 缺 env/weights/tool 在该 evaluator 被实际选择时清楚失败。
6. 验证 registry/alias、score parse、结构映射、seed determinism（若承诺）和真实
   backend smoke（若行为依赖 backend）。

### 4.6 Score snapshot 与 contribution

Owner：`astrolabe/evaluators/score_snapshot.py`、
`astrolabe/score_contributions.py` 和 evaluator score helpers。

`ScoreContribution` 是 runtime component 向统一 snapshot 增加指标的显式合同，包含
`source`、`metric_name`、`value`、`scope`、`entity`、`shape`、`aggregation`、
`flatten_as` 和 provenance。它不能覆盖已有 key；非 global scope 必须指明 entity。

稳定 score field 必须先进入 snapshot/contribution owner，再被 loss、gate、decision
或 posthoc 引用。不要先在 expression 中依赖一个只在某 backend detail dict 偶然存在
的字段。

### 4.7 Post-evaluation hooks

Owner：`astrolabe/post_eval_hooks/`；`run.py` 只调用现有 hook point。Hook 位于
evaluator 输出后、需要结构的 loss 和 final score snapshot 前。

扩展时明确：

- 是否改变 structure、pose 或只增加 report；
- 哪些 loss 需要它，未配置时是否允许自动启用；
- hook 输出怎样合并多个 oligomer；
- backend/env 需求和失败是否影响主输出合同；
- hook 字段不能泄漏到未启用的决策路径。

### 4.8 Loss 与 expression

Owner：`astrolabe/losses/base.py`、loss registry 和具体 loss module。

`LossEngine` 将 evaluator snapshot、registered loss terms 和 `expression_terms` 组合成
`weighted_score` 及 detail environment。Stage 使用 expression 定义 objective、
acceptance gate/threshold、early stopping 和入口条件。

增加 loss/metric：

1. 注册 term 及参数 metadata；不要用 custom gate dict 代替 loss/expression。
2. 定义缺失输入是 error、N/A 还是显式 exploratory opt-in。
3. 说明 metric 的 scope、聚合方式、单位和优化方向。
4. oligomer/protomer-specific term 必须使用共享 chain/protomer resolution。
5. explicit loss detail 优先于 default snapshot；禁止无声覆盖同名 key。
6. 验证 expression variable、依赖 term、missing input、flattened output 和 persisted
   pruning。

### 4.9 Decision context 与 decision maker

Owner 分工：

- `astrolabe.decision_context` 构造一个共享 `DecisionScoreContext`：
  weighted/current/previous/delta scores、stage-entry、reference gate、transition gate、
  early stop、soft acceptance 与 hard-gate result。
- `astrolabe.decision_maker` 拥有 accept/reject policy、Metropolis compatibility、
  sequence/score/stage history、mutation apply/clear 和 `scores.csv`。

**强制合同：** `run.py` 和 `decision_maker` 的 acceptance-gate shared context 必须经
`astrolabe.decision_context`。stage-entry acceptance、普通 mutation decision、
reference gate 和 transition gate 不能各自维护私有字段语义。

`force_accept` 不能绕过 active reference 或 transition-consistency gate。Hard gate
失败必须留下 rejected row；mutation state 必须 clear，不能只改 log 文案。

### 4.10 Mutator 与 selector

Owner：`astrolabe/mutators/registry.py`、具体 mutator、wrapper，以及
`astrolabe/selectors/`。

Mutator 提出下一 sequence/structure-backed proposal；它不决定最终是否接受。
`build_mutator_runtime_config()` 将当前 step/stage/oligo/selector 等共享 runtime
上下文传入 mutator。不要增加只能从 `global_config` 猜出的隐式条件。

增加 mutator：

1. 通过 `_type` 注册并声明 capability；backend runner 放 wrapper。
2. 复用 selector expression/normalized selector，不建第二套位点语言。
3. 尊重 stage-owned `mutation_num`、alias weights、locked/fixed/covalent positions。
4. 若组合多个 state/oligo，显式定义 ignored/negative/shared protomer 语义和权重
   归一化，不从列表顺序猜。
5. 记录 proposal telemetry；proposal 失败不能伪装为“无变化成功”。
6. 验证 alias wiring、selector、fixed positions、seed contract、missing resources
   和 mutation apply/rollback 边界。

### 4.11 Predocker

Owner：`astrolabe/structure_processors/base.py`、具体 predocker 和 wrappers。配置通过
`predocker.predocker_terms.<alias>` 定义，再由 mutator alias 的 `predocker` 列表
选择。详细稳定合同见 [Predocker reference](predockers.md)。

扩展时：

- 定义输入/输出结构、cache key、manifest 和是否允许覆盖 input；
- 外部 tool 调用放 wrapper；
- fallback policy 必须是显式 enum/field，并记录 backend、reason 和实际输出来源；
- geometry 改变需用小型真实结构验证，而不只 mock subprocess；
- 部分失败不能留下看似 final 的结构。

### 4.12 Runtime requirements 与 env

Owner：`astrolabe/runtime_requirements.py`、`config/third_party_envs.yaml`、
`scripts/installation/env_setup.py` 和 env configs。

- `astrolabe env doctor` / wizard 必须 task-aware，说明哪个 evaluator/mutator/predocker/loss
  需要哪个 key。
- executable、weight、repo path 和 subprocess Python 进入 env config，不硬编码在
  component。
- 可选 backend 的 Python import 保持 lazy；只有路径实际执行时才因缺依赖失败。
- 如果 backend 可在独立 subprocess env 运行，保持 CLI 主环境轻量，仅传显式
  path/env contract。
- 运行第三方依赖位于 `vendor/`（vendored source/submodule），不要用文档中的
  `pip install` 或 runtime 自动安装替代仓库声明。
- `config/third_party_sources.yaml`、`.gitmodules`、patch helper 和 env spec 的 source
  ownership 必须一致；已有 submodule path 不能被普通 clone 覆盖。

环境与资产细节见 [第三方环境与模型资产](third_party_envs.md)；调度行为见
[Slurm reference](slurm.md)。

### 4.13 Posthoc

Owner：`astrolabe/posthoc/` 与对应 `astrolabe posthoc ...` handler。稳定链路是：

```text
run dirs
  -> collect: scores/config/structures -> SQLite (+ optional CSV)
  -> filter: expression/preset -> selected rows + report + structures/DB
  -> refold: selector + source config + sequences -> manifest/new run outputs
       -> collect back into the same DB
  -> filter/report
```

普通工作流使用高层 `astrolabe posthoc refold`；`refold-export` / `refold-run` 是已有
manifest 的低层调试与恢复入口，不是第二套规范流程。

扩展 posthoc：

1. 优先扩展 DB/schema/command，而不是新增一次性分析脚本。
2. collect 必须增量、安全处理 unchanged run，并区分 missing 与 invalid metric。
3. preset-required metric 缺失默认失败；仅显式 exploratory option 可放宽。
4. file writes 用临时文件/原子替换，结构 materialization 的 copy/hardlink 语义需可见。
5. refold 必须保存 source run/config/sequence/backend 与 manifest input hash，保证 replay
   可追溯。
6. CLI、completion、DB/CSV output、smoke acceptance 同步更新。

## 5. Score 与参数的公共合同

### 5.1 每个新 knob 的七项声明

任何进入 YAML、CLI、persisted config 或 stable Python API 的参数，都必须在实现和
面向使用者的文档中能回答：

| 项 | 必须说明 |
|---|---|
| Consumer | 哪个具体 owner/function 读取；没有 consumer 就不加 |
| Meaning | 改变搜索、采样、评估、接受、停止还是资源行为 |
| Units | Å、比例、概率、步数、样本数、无量纲等 |
| Direction | 更大更好、更小更好、仅类别选择，或不是质量指标 |
| Valid range | 允许区间、边界是否包含、`null` 的含义 |
| Default/owner | 默认放哪里，是否 stage/backend/env-specific |
| Acceptance | 什么观测证明该值生效且未破坏共享边界 |

同一个物理概念只保留一个 canonical 名称。兼容 alias 必须路由同一 consumer，不能
形成两套实现。

### 5.2 实际调参流程

历史项目中可靠的调参不是“看到一个低分就改权重”，而是以下闭环：

1. **冻结比较条件。** 固定 code revision、template/overlay、backend/model、input、
   seeds、steps/repeats 和 output root；先保存最终 dry-run config。
2. **从最小真实样本建立观测。** 当执行不确定性高时用 disposable canary 缩小数量，
   不修改 production config；执行路径已知则直接 full run，并尽早检查第一批相互独立
   的完整输出。
3. **按 stage/oligo/accepted 状态看分布。** 从 `scores.csv` 或 posthoc DB 比较 count、
   missing rate、median/quantile、极端值、accepted/rejected delta，而不是只看 best row。
4. **检查贡献而非只看总分。** 对 objective 的每一项计算典型 `weight * value`，确认
   它在代表性分布上达到预期相对贡献；单位或尺度不同先规范化/变换，再调 coefficient。
5. **一次只改一个决策维度。** 搜索预算、proposal diversity、objective ranking、
   hard validity gate、early stop 分开修改；保持 paired seeds/input。
6. **用下游 acceptance 回验。** Astrolabe 内部分数改善不等于科学结果改善；必要时通过
   refold、结构检查或目标 benchmark 检验，同时保留 denominator 与失败行。
7. **冻结采用值和证据。** 记录 overlay、run dirs、summary/DB、选择理由和未解决风险；
   不从聊天记忆重建参数。

参数按 owner 分为四类：

| 目的 | 典型参数 | 观察量 |
|---|---|---|
| 搜索覆盖 | `steps`、`repeats`、stage `mutation_num`、mutator weights/selectors | unique proposals、acceptance rate、coverage、runtime |
| Backend proposal/sampling | mutator/evaluator 自己的 `temperature`、`num_samples`、recycles 等 | diversity、determinism、backend quality/runtime |
| 排序与接受 | score transforms/weights、`acceptance_threshold`、Metropolis decision setting | contribution、delta、accepted/rejected 分布 |
| 停止/有效性 | hard `acceptance_gate`、transition/reference gate、early stopping | invalid rejection、false reject、completed steps |

### 5.3 Contribution calibration

权重只在 metric 的单位、方向和有效范围已确定后才有意义。推荐在代表性固定样本上
保存以下表，而不是凭量级猜系数：

```text
term | scope | direction | missing% | q10 | median | q90 | transform | weight
     | median(weight * transformed_value) | intended role
```

校准准则：

- 先检查 score snapshot 是否已统一尺度；不要用 weight 修补 backend 单位 bug。
- 以 median/IQR 或具有明确物理意义的 reference delta 估算贡献，不被单个 extreme
  candidate 支配。
- 分 stage、oligo 和 dataset family 看分布；不同 denominator 不混成一个命中率。
- 若某项只为打破 tie，其典型贡献应明显小于主目标；若是主目标，应在常见变动范围内
  能改变排序。
- 输出 raw term 与 contribution，确保总分变化可解释。
- 在采用前做 paired comparison，报告改善、退化、tie、missing 和失败，而非只报告
  选中结果。

### 5.4 Hard gate、soft objective 与 temperature

Hard gate 只用于明确的无效态或不可继续态，例如缺 required metric、违反不可协商
几何/拓扑合同、reference/transition consistency 不合格。它不应用来把连续质量指标
粗暴二值化，以掩盖 objective 校准不足。连续偏好优先进入 score/loss；探索性阈值
使用 soft acceptance 或 posthoc filter，并保留失败分布。

必须区分两类 temperature：

- **Backend sampling temperature**：mutator/evaluator 生成 proposal 的随机性，由该
  backend config 拥有。
- **Decision temperature**：`decision_maker` 在 Metropolis acceptance 中接受较差
  proposal 的概率；只有 `use_metropolis_acceptance` 启用才生效，历史
  `initial_temperature` 属于该兼容路径。

新参数必须用能看出 owner 的名字，不能用一个全局 `temperature` 同时影响 proposal
与 accept/reject。比较实验中也不能一边改 sampling、一边改 decision temperature。

## 6. CLI 与 shell completion 合同

所有新增/重命名 `gh` subcommand、alias、option、positional 或 fixed-choice value 都
必须遵守 [CLI 设计规则](command_design_rules.md)。最低要求：

1. parser/handler 只表达 intent，并复用 shared option helper。
2. `--help` 说明选择规则、path 语义、layering、derived default 和 failure mode。
3. 空调用不能意外进入 pipeline；交互命令非 TTY 返回 `2`。
4. preview/`--dry-run` 是真实 CLI 路径，不要求 execution-only assets。
5. completion 优先从 argparse parser introspection 派生。
6. parser 不足时在 `astrolabe/shell_completion.py` 增加 shared dynamic provider；
   Bash/Zsh 不维护分叉 command tree。
7. dynamic completion 必须 lazy 且 parameter-gated；只有当前 argument 需要时才访问
   SQLite、filesystem 或 config。
8. 当前 `./scripts/astrolabe completion complete ...` 约 0.5 s median 是上限预算；新
   provider 不应显著变慢。
9. completion 修改必须更新 `tests/test_cli_completion.py`，并运行真实 Bash、Zsh
   source round-trip。

真实 CLI check 至少包含：

```bash
./scripts/astrolabe <command> --help
./scripts/astrolabe <command> ... --dry-run   # 若该命令支持
```

只 import parser、只调用 handler function 或只 snapshot help 都不足以证明 shell
contract。

## 7. Dependency、failure 与 output discipline

### 7.1 Vendor 和 environment

- runtime 第三方源码归 `vendor/`；不要新增系统/global `pip install` 步骤。
- 新 source/submodule 先确认 path owner，更新 license/source inventory 与 env spec。
- backend 权重、二进制和独立 Python 环境由 env config + runtime requirement 管理。
- optional backend import 必须 late；CLI startup 不为未选择 backend 付费或失败。
- 可复用 setup flow 进入 `scripts/installation/env_setup.py`/`just env-setup`，不新增平行 bootstrap。

### 7.2 Visible failure 与 fallback

可接受：

```text
primary backend failed
  -> explicit fallback_policy=input
  -> manifest/log records backend, policy, reason, actual output
```

不可接受：

```text
primary backend failed
  -> catch Exception
  -> return input/empty score
  -> downstream looks successful
```

修 bug 时先复现 observed failure，再在 owner 修。只处理会影响输出合同或 repair/rerun
决策的失败；不要借机增加 speculative fallback、feature flag 或第二实现。

### 7.3 Output contract checklist

改变输出前逐项检查：

- 谁写、何时写、失败时是否有 partial artifact？
- schema/version、units、row identity、stage/step/accepted semantics 是什么？
- write 是否 atomic，重跑是否 idempotent/incremental？
- `config.yaml` 是否保留 replay 所需 consumer fields？
- posthoc collect/filter/refold/report 是否仍能读？
- 用户能否区分 primary success、fallback、skipped、partial 和 failure？
- 大型/临时 artifact 的 owner 和 cleanup marker 是否明确？

## 8. 文档治理

仓库 overview 只保留三个 canonical 入口：

1. `README.md`：名称、最短路径、文档索引。
2. `docs/user_guide.md`：用户流程、配置、调参、运行、验收和 posthoc。
3. `docs/developer_guide.md`：owner、数据流、扩展与验证合同（本文）。

独立文档只保留范围窄、跨任务稳定且会长期独立维护的 reference：

- [CLI 设计规则](command_design_rules.md)
- [第三方环境与模型资产](third_party_envs.md)
- [Slurm](slurm.md)
- [Predocker](predockers.md)
- [非标准残基合同](nonstandard_residue_contracts.md)
- [环化拓扑示例](cyclization_topology_examples.md)
- [D-n 对称性](d_n_symmetry.md)
- [发布检查](release_checklist.md)
- [变更记录](changelog.md)

新功能的用户说明进入 user guide，owner/extension 说明进入本文；不要为每个 feature
再建 quickstart、tutorial、notes 或 utility inventory。一次性计划、评审、状态、
执行证据和交接保存在独立的项目记录中，不混入公开文档。
项目私有科学分析和 run evidence 留在项目 workspace，
不是 repo docs。

文档修改必须：

- 以当前 CLI/config/runtime 为真，不复制已过时的命令；
- 保留 unresolved field、backend 限制与 blocker，不用流畅文字掩盖；
- 不把测试名或 commit message 当作唯一用户说明；
- 删除文档时同步修正 canonical 入口和 repo 内链接，不留下替代索引表维护负担。

## 9. 最小验证阶梯

### V0：静态/文档边界

适用于纯文档、小型 schema 文案或机械路径修正：

```bash
git diff --check
git diff -- docs/developer_guide.md
```

再直接检查：保留链接存在、无待删除文档引用、命令/字段与当前 parser/config 一致。

### V1：Owner-focused behavior

可复用代码修改先运行最窄的现有测试：

```bash
python -m pytest -q <existing-focused-test.py>
```

同时运行请求的可观察边界，例如 CLI help/dry-run、一个 score row、一个结构变换或
一个 posthoc report。一次性脚本不创建测试文件、fixture、framework 或 validation
helper；直接检查目标输出。

### V2：直接受影响的 shared boundary

只有修改跨模块 wiring 时增加一层，例如：

- CLI surface：completion test + Bash/Zsh round-trip。
- config/lifecycle：representative `--print-config --dry-run` + persisted config test。
- evaluator/score/loss：score snapshot/expression + `scores.csv` row contract。
- stage/decision：transaction + stage-entry/active-gate + row history。
- posthoc：对应 smoke，检查 DB/CSV/structure contract。
- backend wrapper：真实最小 backend smoke/canary，而非只 mock。

### V3：Repository/专项 acceptance

核心 evaluator/mutator/decision/loss/`StandardScores` 共享抽象发生改变时，在
具有相应模型依赖的开发环境运行：

```bash
python -m pytest -q
# 等价快捷入口
just test
```

专项命令见 `justfile`，例如 AF3 determinism/profile/inference benchmark、
posthoc collect/filter/refold smoke。只有修改覆盖该边界时才运行；不要把所有昂贵
验收都堆到无关修改上。

如果 full suite 因未改动的明确缺依赖被阻塞，可以报告 blocker，但必须已有 focused
验证覆盖本次行为。不能因为 suite 慢、预算不足或接近结束而把未验证说成通过。

### 停止规则

当请求的 observable behavior 和一个直接受影响的 shared boundary 已证明，就停止
验证。只有证据冲突或新增独立风险边界时才增加下一层。真实运行则负责到下一有意义
决策点：早期完整输出不正确就停、修、重跑；输出正确且 durable supervisor/磁盘
handoff 已接管，才可交接。

## 10. Dirty worktree 与提交保护

本仓库常同时存在 runtime artifacts、vendor/submodule drift 和其他人的修改。开始和
结束都执行只读检查：

```bash
git status --short
git diff -- <owned-paths>
git diff --check
```

规则：

- 不 reset、checkout、clean 或覆盖无关修改。
- 只编辑明确 owned files；发现同一文件并发修改时先整合，不以 HEAD 覆盖工作树。
- stage 时列出具体路径；有 runtime artifacts 时不用 `git add -A`。
- 本地任务管理记录不进入公开源码包；科学运行保留独立的可恢复执行记录。
- logs、outputs、datasets、raw model results、生成 overlay/PSE 不混入 feature commit，
  除非任务明确要求这些 artifact。
- reusable examples 放 `examples/`；task-private overlay 不放 repo root。
- 新 linked worktree 放在主 checkout 外；删除前保留其独有修改和运行记录。

提交前逐项确认：

1. diff 只含目标行为所需文件；
2. public surface、completion、docs、config 和 test 按实际影响同步；
3. 新参数有 consumer/units/direction/range/default/owner/acceptance；
4. failure/fallback/output/replay contract 可见；
5. 选择的验证覆盖真实路径，不只是 import；
6. 没有 task-private/runtime/generated artifact；
7. 没有未经请求的依赖、抽象、flag、兼容分支或第二实现。

## 11. 常见扩展模板

### 11.1 新 CLI command/option

1. 在 parser/handler 定义 intent，复用 shared option。
2. 遵守 command design rules，写清 layering/path/default/failure。
3. 让 completion 从 parser 派生；必要 dynamic provider 保持 lazy。
4. 更新 `tests/test_cli_completion.py` 和最窄 command test。
5. 跑真实 help/dry-run 与 Bash/Zsh round-trip。
6. 用户行为写入 user guide；owner 变化才更新本文。

### 11.2 新 config template 或 knob

1. 先标 consumer 和 owner；决定 global、stage、backend 或 env scope。
2. 使用 alias，不把 class name 写进 stage。
3. 声明 units/direction/range/default/null semantics。
4. 在 `--print-config --dry-run` 检查最终值。
5. 若需 replay，扩展 persisted lifecycle；否则证明会被安全 pruning。
6. 用代表性 output 证明 knob 生效，不只 assert 配置里有 key。

### 11.3 新 evaluator/mutator/predocker

1. 在对应 registry 注册轻量 factory；heavy import 留到实际调用。
2. wrapper 拥有 subprocess/backend parse；component 拥有领域合同。
3. 接入 runtime requirements/env/vendor ownership。
4. 定义 output/score/manifest、failure/fallback 和 deterministic seed contract。
5. 运行 registry/alias/缺资源测试及一个最小真实 smoke。
6. 若改变 CLI/config public choice，同步 completion 和 user guide。

### 11.4 新 score/loss/gate

1. 先把 stable metric 加入 `StandardScores` snapshot 或 `ScoreContribution`。
2. 明确 scope、aggregation、units、direction、missing policy。
3. 注册 loss/term，验证 expression 与 flattening。
4. 用观测分布校准 transform/weight，记录贡献而不只总分。
5. hard gate 仅用于明确无效态；连续质量留在 objective/soft policy。
6. shared acceptance fields 进入 `decision_context`；transition fields 经
   `transition_consistency_gate_score_context`。
7. 验证普通 decision、stage entry、reference/transition path 与 `scores.csv`。

### 11.5 新 stage/transition behavior

1. 在 `StageUnit/StageManager` 表达 config/state。
2. 在 `stage_transaction` 定义 plan/attempt/decision/commit/rollback。
3. 在 `stage_transition` 定义 gate result、artifact 和 shared score fields。
4. 不在 evaluator/mutator 或 `run.py` 私有 key 中复制 stage semantics。
5. 覆盖 default inheritance、entrance、target re-evaluation、seed/active gate、失败
   rollback、成功 commit 和 row-history。

### 11.6 新 posthoc behavior

1. 先定义 DB/row/metric identity 和 missing policy。
2. 扩展 collect/filter/refold/report 的 owning module 与 CLI。
3. 保持 incremental/idempotent/atomic write。
4. 更新 completion 与 command help。
5. 跑对应 `just posthoc-...-smoke`，检查 DB/CSV/结构/manifest 内容。

## 12. Definition of done

一个修改只有同时满足以下条件才完成：

- 请求行为通过预期 `gh`/config/Python/output surface 真正可用；
- 复杂度位于正确 owner，`run.py`/CLI 没有新增可复用领域 special case；
- 参数、score、stage 和 output schema 的公共语义已说明并可审计；
- invalid state 和 backend failure 清楚可见，fallback 不伪装 primary success；
- persisted config 和 posthoc/replay consumer 没被遗漏；
- 最小充分验证覆盖 owner 和直接受影响的共享边界；
- canonical 用户/开发者文档同步，未新增一次性 docs；
- dirty worktree 中无关改动和 runtime artifacts 得到保护。

测试通过只是证据，不是定义本身；只有测试覆盖真实 requirement，输出合同也满足，
修改才算完成。
