---
task_id: a
role: 因果迁移学习研究员
status: complete
sources_found: 6
---

## Sources

[1] Causal inference using invariant prediction: identification and confidence intervals | https://arxiv.org/abs/1501.01332 | Source-Type: academic | Accessibility: public | As Of: 2016-11 | Authority: 9/10
[2] Invariant Risk Minimization | https://arxiv.org/abs/1907.02893 | Source-Type: academic | Accessibility: public | As Of: 2020-03 | Authority: 9/10
[3] Anchor regression: heterogeneous data meet causality | https://arxiv.org/abs/1801.06229 | Source-Type: academic | Accessibility: public | As Of: 2021-08 | Authority: 9/10
[4] Domain Adaptation by Using Causal Inference to Predict Invariant Conditional Distributions | https://proceedings.neurips.cc/paper/2018/hash/39e98420b5e98bfbdc8a619bef7b8f61-Abstract.html | Source-Type: academic | Accessibility: public | As Of: 2018-12 | Authority: 9/10
[5] Transportability of Causal and Statistical Relations: A Formal Approach | https://ftp.cs.ucla.edu/pub/stat_ser/r372-a.pdf | Source-Type: academic | Accessibility: public | As Of: 2011-08 | Authority: 9/10
[6] Toward Causal Representation Learning | https://arxiv.org/abs/2102.11107 | Source-Type: academic | Accessibility: public | As Of: 2021-02 | Authority: 9/10

## Findings

- 跨城市碳预测可先把“城市/年份/政策期”定义为环境，并用 SCM 表示人口、经济、能源结构、气象、土地利用、交通与排放之间的机制，同时用环境或 selection node 标记哪些机制会随城市改变；这一做法的价值是把“可迁移”变成对结构和变化机制的显式假设，而非把城市 ID 当普通特征。[5][6]
- ICP 通过检验候选特征集合是否使目标条件分布或残差分布跨环境不变，并取所有未拒绝集合的交集来保守识别目标变量的直接原因；用于城市任务时，可先做可解释的稳定驱动筛选，但原始方法的主要保证依赖结构方程与正确环境异质性，环境变化不足时可能只返回空集。[1]
- IRM 学习一个表示，使同一个顶层预测器在所有源城市环境中同时最优，因而适合把 GNN/MLP 的城市表征学习与跨城市不变机制结合；其理论外推依赖训练环境具有足够多样性和共同的不变关系，城市划分不应只是同分布数据的随机切片。[2]
- Anchor regression 在平方损失中惩罚残差对外生 anchor 可解释的部分，并以超参数在 OLS 与更强的 shift-robust 解之间连续权衡；城市、区域、政策期或可观测制度冲击可作为 anchor，但它主要保证线性结构中一类 shift interventions 的分布鲁棒性，不能自动等同于识别真实因果效应。[3]
- 因果领域适应可以在因果图上枚举预测特征集合，选择那些在允许的干预集合下具有不变条件分布且目标域风险最低的预测器；这比无条件对齐所有城市表征更有针对性，因为若边际分布变化本身承载排放机制信息，强行对齐可能损害预测。[4]
- Transportability 用 selection diagrams 表示源城与目标城哪些机制不同，并用 do-calculus 判断目标因果量是否可由源城实验/观测数据与目标城观测数据识别；因此当任务不仅是预测而是回答能源政策反事实时，应将其作为最严格的上层框架，而不是仅靠 IRM 或域对齐。[5]
- 因果表征学习的模块化 ICM/Sparse Mechanism Shift 观点建议把城市模型拆成可复用模块，例如活动水平、能源强度、排放因子和气象响应，并只对目标城发生变化的少数模块适配；但从遥感、POI 或路网等低层观测中恢复真正因果变量仍是开放问题，需领域知识或干预变化辅助。[6]
- 推荐的组合架构是“SCM/selection diagram 明确假设 → ICP 筛稳定驱动并审计环境 → IRM 或因果条件不变性训练非线性预测器 → anchor penalty 覆盖已知城市偏移 → 少量目标城数据校准变化模块 → transportability 判断政策反事实是否可识别”。[1][2][3][4][5][6]
- 评估应使用 leave-one-city-out 和预先定义的跨区域/跨时间 manifest，同时报告平均误差、最差城市误差、跨环境残差不变性与机制稳定性；随机混合城市样本只能评估插值，不能证明迁移或因果不变性。[1][2][3]
- 最可行的第一阶段不是直接宣称学到完整因果图，而是把已知城市属性作为环境与 anchor，在现有预测网络上加入 IRM/anchor 目标并用 ICP 做后验审计，再与 ERM、标准 domain adaptation 和目标城微调进行消融比较。[1][2][3][4]

## Deep Read Notes

### Source [1]: Causal inference using invariant prediction
Key data: 论文把多个实验条件视为 environments，逐一检验候选集合的条件模型/残差是否跨环境相同，并以未拒绝集合的交集构造保守因果变量集合及覆盖率保证。
Key insight: 不变性不仅是正则项，也可作为带统计检验的“稳定驱动审计器”；但识别强度由环境干预到哪些变量决定。
Useful for: 城市排放驱动筛选、残差不变性检验、解释为何需要有意义的跨城/跨期环境。

### Source [2]: Invariant Risk Minimization
Key data: IRM 的原则是寻找表示，使其上的同一预测器对每个训练环境都最优；论文线性结果要求足够多且处于 general position 的环境，明确指出随机拆分同一分布并不提供所需环境多样性。
Key insight: 可将深度特征学习与因果不变性合并，但泛化来自环境和结构假设，不是仅加入一个 penalty 就自然获得。
Useful for: 在 CarbonGCN/BPNN 等模型上实现城市级 domain generalization 目标及设计 leave-one-city-out 协议。

### Source [3]: Anchor regression: heterogeneous data meet causality
Key data: anchor loss 在 OLS 和近似 IV/强鲁棒解之间插值，并对线性 SCM 中由 anchor 方向诱发、强度受限的一类 shift perturbations 给出最坏风险保证，即使标准 IV 假设不完全成立仍可有鲁棒性解释。
Key insight: 它把“已知会变化的城市因素”直接转为可调鲁棒半径，适合误差—稳定性权衡，但其鲁棒参数通常不是因果效应。
Useful for: 城市/区域/年份/政策批次作为 anchor 的可解释鲁棒 baseline 与敏感性曲线。

### Source [4]: Domain Adaptation by Using Causal Inference to Predict Invariant Conditional Distributions
Key data: 方法在给定因果图和可能干预目标时寻找 invariant conditional distributions，并分别讨论目标域有标签或无标签时的预测器选择。
Key insight: 应对齐被图结构支持的不变条件机制，而不是默认域不变表示一定正确。
Useful for: 源城—目标城因果 domain adaptation、特征集合选择和少量目标标签校准。

### Source [5]: Transportability of Causal and Statistical Relations
Key data: selection diagram 用额外节点编码总体差异；论文给出 transportability 的形式定义并以 do-calculus 推导可迁移公式或判定不可识别。
Key insight: “跨城预测准确”与“源城政策效应可迁到目标城”是不同问题，后者需要明确哪些结构机制跨城改变及目标城可获得哪些观测量。
Useful for: 政策干预/反事实层、源城实验与目标城观测数据融合、不可识别性声明。

### Source [6]: Toward Causal Representation Learning
Key data: 综述用 Independent Causal Mechanisms 与 Sparse Mechanism Shift 解释为何因果模块可能跨任务复用，并强调从高维低层数据学习因果变量本身仍是核心开放问题。
Key insight: 模块化表征比单一共享 latent 更贴合城市机制只局部变化的现实，但必须以领域变量、环境变化或干预信号约束。
Useful for: 融合遥感/POI/路网的模块化编码器、目标城少样本适配与研究边界说明。

## Gaps

- 未找到直接把上述完整组合框架用于“跨城市碳排放空间预测”的成熟原始论文，因此组合方案是由通用因果迁移理论推导出的研究设计，应用有效性必须由本仓库数据的跨城实验验证。
- 环境不变可能源于变量变化不足、测量误差共同或城市样本量太小，而不一定意味着真实因果机制；应同时做环境可分性、干预覆盖和伪环境检验。
- ICP、IRM 与 anchor regression 的经典保证分别受结构方程、环境多样性、线性 shift 类等假设限制；深度非线性实现通常只保留归纳偏置而不保留全部有限样本保证。
- 反方解释：足够大的 ERM、预训练或普通域适应模型若覆盖了目标城变化，也可能在预测上超过因果方法；因果框架的优势应以最差城市风险、极端政策/气候偏移及少样本适配效率检验，而不能由方法名称预设。
- 城市是复合环境，城市 ID 同时编码大量未观测因素；把它直接作为 anchor 或 domain 可能产生不可解释的鲁棒方向，最好进一步拆成气候带、能源结构、产业结构、政策与数据采集机制。
- Transportability 需要足够可信的 selection diagram；若跨城机制差异位置无法由知识或数据约束，do-calculus 可能判定不可识别，不能用表示学习掩盖这一缺口。
