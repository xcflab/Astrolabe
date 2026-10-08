# Astrolabe 中文文档

Astrolabe 将结构预测、序列提案、评分与接受决策组成闭环，并支持跨分子状态的
序列同步和按条件触发的结构干预。论文标题为
*Astrolabe: Programmable all-atom biomolecular design through cross-context coordination and structural intervention*，
作者依次为 Zhengpeng Zeng、Chunfu Xu。目前尚无预印本或 DOI。

## 安装与论文工作流

需要 Python 3.11 或更新版本；CPU 持续集成覆盖 Python 3.11 和 3.12。

```bash
git clone https://github.com/Molaison/Astrolabe.git
cd Astrolabe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
astrolabe --help
astrolabe paper list
astrolabe paper prepare-inputs --out-dir data/paper
```

仓库在论文准备期间可能需要 GitHub 访问权限。也可解压随附的完整源码包
`astrolabe-source.tar.gz`，进入 `Astrolabe/` 后执行相同安装命令。

配置所选 GPU 模型环境后，可直接调用论文中的设计任务：

```bash
./scripts/astrolabe paper run rna-aptamer \
  --inputs examples/paper/inputs/nucleic/rna_aptamer_1urn_u1a.yaml \
  --data-dir data/paper --seed 1 --out-dir outputs/u1a-rna
```

蛋白、配体、多靶点、抗体、对称装配、环肽、DNA/RNA 和 beam search 均在统一的
`paper` 命令下。具体输入、模型需求、代表选择及分析命令见[论文工作流](paper_workflows.md)。
一次默认运行一个轨迹；`--num-runs` 指定重复次数，`--dry-run` 只打印配置。

## 使用路线

| 目标 | 入口 |
| --- | --- |
| 安装、论文任务入口、软件引用 | [README](../README.md) |
| 论文协议、原始输入和分析 | [论文工作流](paper_workflows.md) |
| 查看配置、调用命令、组织设计任务 | [用户手册](user_guide.md) |
| Python API 与组件扩展 | [开发者手册](developer_guide.md) |
| GPU、模型资产与 AF3 安装轨迹 | [容器指南](../container/README.md#install-af3) |
| 示例选择与适用范围 | [examples](../examples/README.md) |
| 输入、环境、预期结果与论文复现范围 | [复现说明](reproducibility.md) |
| 软件检查与科学验收状态 | [发布验证](release_acceptance.md) |

原有 CPU 坐标例子保留为安装诊断，数值及输出见[复现说明](reproducibility.md)。

公开 Python 入口为 `astrolabe.Astrolabe(cfg).run()`，其中 `cfg` 是已经组合完成的
Hydra `DictConfig`。模型推理还需要相应 GPU 环境与资产。`release_protein` 是
80 残基、seed 1729、五个评估状态的 AF3 安装例子，初始状态之后最多有四次提案。

## 从旧接口迁移

1. Python 包使用 `astrolabe`；安装命令使用 `astrolabe`；源码入口为 `./scripts/astrolabe`。
2. 组件目录为 `evaluators`、`mutators`、`initializers`、`selectors`、`losses` 和
   `structure_processors`。模型名称、已有 YAML 类型标识和算法含义保持一致。
3. `scripts/gh_dev` 委托给规范命令。已有 `GH_*` 环境变量和 `.gh_python_path`
   仍可读取；新安装器写出 `.astrolabe_python_path`。
4. 容器规范路径为 `/opt/Astrolabe`，安装器为 `scripts/install.py`。

序列同步共享已接受的序列，不要求不同评估器产生相同结构。Rescue 在配置条件
命中时修改后续提案的结构上下文；阶段转换后按新的配置重新评估。

## 其他参考

- [第三方软件要求](../needed_softwares.md)与[第三方环境](third_party_envs.md)
- [结构处理器配置](predockers.md)与[Slurm 使用](slurm.md)
- [命令设计约定](command_design_rules.md)与[发布清单](release_checklist.md)
- [软件引用](../CITATION.cff)与[许可证](../LICENSE)
