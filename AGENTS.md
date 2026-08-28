# 仓库贡献指南

## 项目结构与模块组织

本项目采用 `src/` 布局，可安装包位于 `src/carbon_transfer/`。配置加载、数据构建、数据划分、模型、训练、指标、评估和任务编排逻辑均放在包内。

主要目录职责如下：

- `src/carbon_transfer/experiments/`：任务定义、发现、运行和协议级聚合。
- `src/carbon_transfer/models/`：模型实现、模型规格和稳定模型 ID 注册。
- `scripts/`：面向用户的独立实验入口；每个实验协议拥有自己的 argparse、默认配置和校验。
- `configs/experiments/`：自包含实验配置。
- `tests/`：CLI、脚本入口、数据管线、模型、进度输出和实验协议测试。
- `data/features/`：预处理特征；`data/splits/`：确定性划分 manifest。
- `outputs/<experiment>/runs/`：模型、预测、指标和完成标记；`outputs/<experiment>/reports/`：聚合报告。

`carbon` CLI 只负责模型列表、数据构建、划分构建和验证，不用于训练或评估。训练和评估必须通过 `python scripts/run_*.py` 执行。旧的 `main_experiment.py`、旧通用训练脚本和 `artifacts*/` 目录不得恢复。

## 构建、测试与开发命令

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python -m pip install -e .
python -m pytest
```

数据与划分命令：

```bash
carbon models list
carbon data build --config configs/data.yaml
carbon splits build --config configs/experiments.yaml
carbon validate data --config configs/data.yaml
carbon validate experiment --config configs/experiments/stage1_cross_city.yaml
```

实验入口：

```bash
python scripts/run_stage1_cross_city.py --dry-run
python scripts/run_stage1_cross_region.py --dry-run
python scripts/run_single_month_cross_region.py --dry-run
python scripts/run_single_month_cross_region_multiseed.py --dry-run
python scripts/run_annual_cross_region.py --dry-run
```

大型实验必须先执行 `--dry-run`。脚本统一支持城市、模型、seed、epoch、patience、batch size 和 device 覆盖；命令行参数优先于 YAML。使用 `--evaluate-only` 只重建报告，不启动训练。

## 新增实验与数据划分

新增协议时应同时提供 `configs/experiments/<experiment>.yaml` 和语义明确的 `scripts/run_<experiment>.py`。脚本自行定义 argparse、协议固定值、fold 到城市的映射和参数校验，但必须调用包内的任务发现、训练和聚合逻辑，不得复制模型训练实现。

配置必须明确包含 `split_experiment`、输入文件、模型 ID、seed、训练参数、`artifact_dir` 和 `report_dir`。不同协议不得依赖配置文件名推断，也不得在一次报告中混合扫描不同协议的结果。

若需要新的时间范围、区域定义或划分规则，应先在包内实现确定性 manifest 生成。不要在训练循环或入口脚本中临时切分数据。新增实验至少验证任务数量、城市筛选、模型与 seed 筛选、split 互斥和数据泄漏约束。

## 新增模型与 Baseline

新模型或 baseline 放在 `src/carbon_transfer/models/`，使用稳定、唯一的 `snake_case` 模型 ID，并在模型注册表中登记。脚本通过注册表获得可选模型，不得为单个模型硬编码新的训练入口。

新增模型应同步检查模型发现、训练分派、预测、保存和评估格式，并测试未知 ID、固定 seed 输出路径和必要产物。除非明确迁移，不要修改已有模型 ID。

## 进度输出与训练日志

BPNN、CarbonGCN 和 OpenCarbon 每完成一个 epoch 只输出一条永久日志；TTY 和重定向日志必须一致。禁止 `tqdm`、动态覆盖行和 batch 级刷新。epoch 日志至少包含 fold、模型、seed、epoch、训练损失、验证 MAE、整体验证集 log 空间 R²、最佳验证 MAE、stale 计数和耗时。R²仅用于监控，模型选择和 early stopping 仍以验证 MAE 为准。

LightGBM 记录 fit start、fit end 和 best iteration，不伪造 epoch。修改进度输出时必须更新 `tests/test_progress.py`，并保持 `training_log.json` 的 `events` 与 `epochs` 结构。

## 编码风格与测试

Python 使用四空格缩进并符合 PEP 8。公开及复杂函数添加类型标注；模块、函数和变量使用 `snake_case`，类使用 `PascalCase`。路径优先使用 `pathlib.Path`，文本明确使用 UTF-8。

Pytest 测试命名为 `tests/test_*.py` 和 `test_<行为>`。修改模型、划分、任务发现、实验脚本或聚合时，应增加相应回归测试。

提交前运行：

```bash
python -m pytest
python -m compileall -q src scripts tests
carbon models list
carbon validate experiment --config configs/experiments/stage1_cross_city.yaml
python scripts/run_stage1_cross_city.py --dry-run
python scripts/run_annual_cross_region.py --dry-run
```

若环境缺少依赖或数据，应明确列出未执行检查及原因，不得声称测试通过。

## 数据、输出与实验安全

不得无提示覆盖正式任务。包含 `COMPLETE` 的任务默认跳过；只有用户明确要求替换结果时才可使用 `--force`。执行前确认协议、城市、fold、模型和 seed 的作用范围。

`outputs/smoke/` 仅用于诊断。除非明确要求，不要提交原始数据、预处理数据、输出、报告、权重、预测或日志。不得删除、移动或重写已有正式输出；代码重构应保持已有 checkpoint、预测和指标可读取。

## 提交与合并请求要求

提交标题应简短并使用祈使语气，例如 `experiments: 添加年度跨区域入口`。合并请求应说明代码或实验变更、实际验证命令、受影响配置与输出。结果变化时附上精简指标；仅可视化变更需要截图。
