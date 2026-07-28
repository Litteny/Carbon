# 仓库贡献指南

## 项目结构与模块组织

`carbon_transfer/` 是可安装的 Python 包，包含配置加载、数据集访问、数据划分、模型定义、训练、指标计算和评估逻辑。命令行流程位于 `scripts/`，实验及数据配置位于 `configs/`，测试位于 `tests/`。`raw_data/` 保存原始输入，`data/features/` 和 `data/splits/` 保存预处理数据，`artifacts*/` 与 `reports/` 保存生成结果。实验范围参见 `PROJECT_TASK.md`，第一阶段流程参见 `README_GPU.md`。

## 构建、测试与开发命令

使用 Python 3.9 或更高版本，并创建虚拟环境：

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python -m pip install -e .
pytest
```

运行 `python scripts/build_dataset.py --config configs/data.yaml` 构建特征数据，再运行 `python scripts/build_splits.py --config configs/experiments.yaml` 创建数据划分清单。使用 `python scripts/run_stage1.py --config configs/stage1_gpu.yaml --dry-run` 预览正式任务矩阵。使用 `python scripts/validate_stage1.py` 检查预处理数据、划分完整性和冒烟测试产物。

## 编码风格与命名约定

Python 代码使用四个空格缩进。公开函数及复杂函数应添加类型标注。模块、函数和变量使用 `snake_case`，类使用 `PascalCase`。命令行入口应保持精简，可复用逻辑应放入 `carbon_transfer/`。文件路径优先使用 `pathlib.Path`，文本读写明确指定 UTF-8，并使用相对项目根目录的配置路径。项目暂未配置自动格式化或检查工具，提交的代码应符合 PEP 8，并保持导入分组清晰。

## 测试要求

Pytest 会发现 `tests/test_*.py`。测试函数命名为 `test_<行为>`，每个测试聚焦一个可观察约束。修改特征、模型、数据划分或指标时，应补充针对特征泄漏、张量形状、划分互斥及指标边界情况的回归测试。提交前运行 `pytest`。部分测试依赖仓库中的预处理数据文件。

## 提交与合并请求要求

当前工作副本不包含 Git 历史，无法归纳既有提交格式。提交标题应简短、使用祈使语气，并可添加范围，例如 `training: 修复未完成任务的恢复逻辑`。除非属于预期交付物，否则不要提交生成产物。合并请求应说明代码或实验变更、列出已执行的验证命令、标明受影响的配置与输出，并关联相关问题。结果发生变化时附上精简的指标或报告片段；仅在涉及可视化产物时提供截图。

## 数据与实验安全

不要无提示地覆盖已完成的正式任务。运行器会跳过包含 `COMPLETE` 标记的目录；只有明确需要替换结果时才使用 `--force`。不得把 `artifacts_smoke/` 中的结果当作正式实验结果。
