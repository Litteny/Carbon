---
task_id: c
role: 时空机器学习与实验设计研究员
status: complete
sources_found: 6
---

## Sources

[1] Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment | https://arxiv.org/abs/2309.13378 | Source-Type: academic | Accessibility: public | As Of: 2023-09 | Authority: 9/10
[2] Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization | https://arxiv.org/abs/2311.11114 | Source-Type: academic | Accessibility: public | As Of: 2023-11 | Authority: 9/10
[3] Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank | https://arxiv.org/abs/2308.09727 | Source-Type: academic | Accessibility: public | As Of: 2023-08 | Authority: 8/10
[4] Unveiling the Inflexibility of Adaptive Embedding in Traffic Forecasting | https://arxiv.org/abs/2411.11448 | Source-Type: academic | Accessibility: public | As Of: 2024-11 | Authority: 7/10
[5] Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies | https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408/ | Source-Type: academic | Accessibility: public | As Of: 2010-06 | Authority: 9/10
[6] Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects | https://op.europa.eu/en/publication-detail/-/publication/6e30553e-76f7-11e9-9f05-01aa75ed71a1/language-en | Source-Type: official | Accessibility: public | As Of: 2019-05 | Authority: 9/10

## Findings

- CaST 将时空图生成写成含时间环境、空间上下文、历史观测与未来结果的 SCM，以 back-door 分离不变实体表示与时间环境，并以 front-door 代理和 Hodge-Laplacian 边卷积建模空间因果传播，但其三数据集预测实验本身并不能识别真实政策效应。 [1]
- EAGLE 把动态图的潜在环境显式解耦、实例化新的环境并以节点级干预学习时空不变模式，说明跨城市模型可把“城市×季节/制度”当作环境来训练，但不变性目标仍依赖环境划分及潜在机制共享假设。 [2]
- TPB 的严格跨城设置以四个城市轮流作为目标城市、其余三个作为源城市，并仅用目标城市少量数据微调；这支持将碳排放研究设计为 leave-one-city-out 外层循环，同时必须另报 zero-shot 与 few-shot，不能把目标城微调后的结果称为纯 domain generalization。 [3]
- TPB 在 PEMS-BAY、METR-LA、成都和深圳上比较 10 个基线，并报告相对其他跨城方法平均降低 RMSE 6.52%、MAE 7.71%，但四城规模且都属于交通速度领域限制了对城市碳排放的外推。 [3]
- 2024 年的扩展交通基准发现现有 STGNN 随时间推移明显退化，并把固定节点自适应嵌入识别为跨时间、跨图结构泛化瓶颈；因此碳排放评估应采用按日历向前滚动的时间外推，而不是随机切分，并应测试新增/缺失 POI 与图结构变化。 [4]
- 负对照应复现与主分析相近的偏差来源但排除假设中的因果机制；对城市政策可使用政策实施前的伪处理日期、理论上不受政策影响的排放部门/时段，以及相似但未实施政策的暴露作为负对照，显著“效应”应视为残余混杂或分析错误警报。 [5]
- 合成控制用未干预城市的加权组合构造反事实，欧盟方法指南强调可行性、数据需求与方法选择；在城市碳政策研究中应将 STGNN 预测器与 synthetic control / event-study 分开：前者负责高维结果建模，后者负责政策效应识别及政策前拟合、安慰剂城市和安慰剂日期检验。 [6]
- 严格主协议应嵌套为“外层留一城市完全不可见、内层按城市分组调参、目标城按时间只向前测试”，且所有标准化、图构建、环境聚类和表征预训练只在源城训练窗拟合，以阻断目标城市统计量和未来信息泄漏。 [1][2][3][4]
- 最小消融应分别移除环境解耦/不变性惩罚、因果或功能邻接边、干预模块、静态协变量、时间特征和目标城微调，并与 ERM-STGNN、非图时序模型、仅源城训练、单城训练及简单气候/能源强度基线比较；只有在多种城市与时间移位下稳定获益，才能把收益归因于框架组件。 [1][2][3]
- 因果预测与因果效应是两个不同估计目标：较低的 leave-one-city-out MAE 只证明预测迁移，只有在明确处理、时间顺序、无干扰/可建模溢出、可交换性或可信准实验设计成立并通过负对照后，才可报告城市政策的 ATT/动态效应。 [1][5][6]

## Deep Read Notes

### Source [1]: CaST
Key data: NeurIPS 2023；三个真实数据集；SCM 含时间环境 E、空间上下文 C、历史 X 与未来 Y，并分别实例化 back-door 与 front-door 调整。
Key insight: “因果”模块可用于学不变预测机制，但论文的预测指标并不自动赋予政策效应可识别性。
Useful for: 因果时空图架构、环境解耦、因果传播消融。

### Source [2]: EAGLE
Key data: NeurIPS 2023；多通道环境解耦、环境实例化、节点级干预和不变模式识别，并在真实与合成动态图分布偏移上评估。
Key insight: 城市差异可作为多环境证据，但需要显式检验环境定义错误与机制不共享。
Useful for: domain generalization、OOD 压力测试、环境消融。

### Source [3]: TPB
Key data: 四个城市/区域数据集轮流作目标域，其余三个作源域；比较 10 个基线；目标域允许 few-shot 微调。
Key insight: 留城评估必须区分 zero-shot、few-shot 和 full-target supervision，避免协议混名。
Useful for: leave-one-city-out 协议、源/目标隔离、跨城基线。

### Source [4]: Adaptive Embedding Study
Key data: 扩展时间基准显示 STGNN 长期性能退化；PCA 嵌入允许训练与测试图结构不同并支持跨城 zero-shot。
Key insight: 节点 ID 嵌入会把模型锁死在训练城市，结构可变表示是跨城碳迁移的必要压力测试。
Useful for: 时间外推、图结构变化、zero-shot 设计。

### Source [5]: Negative Controls
Key data: 区分负对照暴露与负对照结果；理想负对照与主变量共享未观测共同原因但不存在目标因果路径。
Key insight: 负对照失败只能提示偏差，不一定定位偏差类型；负对照选择错误也会误报。
Useful for: 伪政策日期、伪结果、残余混杂诊断。

### Source [6]: EU Synthetic Control Guide
Key data: 官方方法指南系统讨论 synthetic control 的可行性、数据需求和方法学事项。
Key insight: 单城或少数城市政策冲击需要显式反事实，而预测器准确不等于反事实可信。
Useful for: 政策评估、安慰剂城市/时间、干预前拟合检查。

## Gaps

- 未找到经多城市真实碳排放政策干预验证、同时完成因果识别与跨城 STGNN 迁移的成熟统一基准；交通论文只能提供架构与评估类比，不能直接证明碳领域有效。
- CaST、EAGLE 等方法主要在预测或合成干预上验证，潜在环境的语义、back-door/front-door 条件及节点间干扰在真实城市政策中可能不成立。
- 反方解释一：跨城收益可能来自更强正则化、预训练规模或周期模式复用，而非恢复了稳定因果机制；需以容量匹配基线和随机/错误因果图负对照区分。
- 反方解释二：留一城市仍可能因相邻城市共享电网、天气、政策扩散或供应链而违反独立域和无干扰假设，导致测试城市并非真正“不可见环境”。
- 失败模式包括目标城标准化泄漏、用全时段建图、随机时间切分、用目标城选择超参数、把 few-shot 称为 zero-shot、政策前拟合差仍报告 synthetic-control 效应、负对照不具 U-comparability，以及多次安慰剂检验未校正。
- Source [4] 为尚未注明正式同行评审版本的预印本，相关结论应降为中等置信度并在采用前复现实验。
