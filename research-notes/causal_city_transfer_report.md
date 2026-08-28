# 城市迁移碳排放预测的因果框架与构建方案

> 研究日期：2026-08-03｜模式：Standard｜AS_OF：2026-08-03｜批准来源：12｜官方源占比：33.3%

## 摘要

最合适的方案不是单押某一种“因果神经网络”，而是建立一个分层框架：以结构因果模型（SCM）和 selection diagram 规定城市间哪些机制稳定、哪些机制变化；以 Invariant Causal Prediction（ICP）审计稳定驱动；以 IRM、风险方差或 anchor regression 训练跨城市稳健预测器；以模块化的活动量—能源强度—排放因子结构约束输出；只有在研究政策效应时，才进一步使用 transportability、事件研究或 synthetic control。原因是“预测未见城市”与“估计政策干预效应”是不同的估计任务，低 MAE 不等于政策效应可识别。[1][2][3][4][12]

对当前 Carbon 项目，近期可落地的是“预测型因果迁移”：保留既有 POI、MODIS、VIIRS、气象、月份和 3×3 邻域编码器，把城市、气候带、月份/年份视为 environment，在训练损失中加入跨环境风险稳定项，并做 leave-one-city-out、时间外推和夜光移除消融。当前数据缺少能源结构、产业、交通、人口经济和政策面板，因此暂不能声称识别完整因果图或政策效应。

## 1. 先界定两个任务

### 1.1 跨城市稳健预测

目标是估计未见目标城市的 `P(Y | X)`，例如网格月度 `log1p_emission`。因果框架在这里是一种归纳偏置：寻找跨环境更稳定的机制，减少模型依赖城市特有的伪相关。ICP 检验给定特征集合后目标条件分布或残差是否跨环境不变；IRM 寻找一个表示，使同一预测头在所有源环境中同时最优；anchor regression 则惩罚残差中可被已知环境偏移解释的部分。[1][2][3]

### 1.2 政策因果效应

目标是估计诸如“低排放区使交通碳排放降低多少”的 ATT、动态效应或反事实。这要求明确 treatment、时间顺序、调整集、干扰/溢出和识别假设。Selection diagram 和 transportability 判断源城市的实验或准实验结论能否迁到目标城市；单城政策冲击可以用 synthetic control 或 event study，并检查政策前拟合和安慰剂效应。[4][12]

**置信度：High。** 两类 estimand 在方法论上明确不同，来源包括因果迁移理论与官方政策评估指南。[4][12]

**反方解释：** 因果正则化也许只表现为普通正则化；若 ERM 已覆盖目标分布，它可能更准。因此优势必须由未见城市、极端气候/制度偏移和最差城市风险证明，不能由方法名称预设。[2][3]

## 2. 建议的城市结构因果图

建议以城市 `c`、网格 `i`、月份 `t` 为索引：

```text
E: 地理/气候带/历史/统计口径/城市制度
├──> S: 人口、收入、发展阶段、产业基础
├──> U: 城市形态、土地利用、路网、建筑形态
├──> R: 能源禀赋、电网碳强度、燃料价格
└──> P: 政策与治理

S,U,R,P,W(气象) ──> I(产业活动), B(建筑负荷), T(交通活动), M(能源结构)
I,B,T,M ──> A(分部门×燃料活动量) ──> Y(碳排放)

POI/夜光/MODIS = 对 U、A、植被和活动强度的有噪代理观测
```

IPCC 指出人口、收入、城市化阶段、城市形态、基础设施和供能共同塑造城市排放；城市之间存在显著的能源使用类型和基础设施锁定差异。[5][8][9] 排放方程最好遵循清单物理结构：部门—燃料活动量乘排放因子，再聚合到边界明确的 Scope 1、Scope 2 或消费口径结果。[6][7]

变量角色必须随研究问题确定，而不是永久固定：

- 环境/效应修饰因子：气候带、资源禀赋、区域电网、统计口径、历史基础设施。
- 常见混杂：基期人口、收入、产业、财政与治理能力。
- 常见介体：能源消费、交通活动、建筑冷热负荷、产业产出和能源结构。
- 代理变量：POI、夜光、遥感建成区；它们不是天然的根因。
- 潜在碰撞点：既受排放压力又受治理能力影响的政策采用，或同时受经济活动和供电可得性影响的夜光。估计政策效应时随意控制这些变量可能制造偏差。

标签必须先做核算边界协调。GPC 的核心就是城市清单的一致、透明和可聚合；不同边界或不同 Scope 的标签混训会把核算差异误认成机制差异。[6]

**置信度：Medium-High。** 上游骨架与核算方程有官方和学术依据，但完整 DAG 是跨领域综合设计，尚无单篇多城市研究完全验证。[5][6][7][8][9]

**反方解释：** 城市形态与排放的关联可能来自收入、能源价格、公共交通投资或居住自选择；高密度也可能降低交通排放却增加建材、热岛和制冷排放。因此要分别建模部门路径，不能将 POI 或密度系数直接解释为总效应。[5][9]

## 3. 因果框架的选择

### 3.1 顶层：SCM + selection diagram（必选）

它负责声明：从源城市到目标城市，哪些结构方程变化。建议把以下节点标为可能变化：`S→A` 的消费/生产关系、`U→T` 的出行关系、`W→B` 的建筑气象响应、`M→Y` 的电网/燃料排放因子，以及测量过程 `latent→POI/VIIRS`。物理核算模块相对稳定，社会经济生成和代理测量模块允许城市特异。[4][7]

### 3.2 审计：ICP（推荐）

用城市、月份、年份或气候带定义环境，检验候选驱动或学习表示在给定后残差是否仍随环境变化。它适合回答“哪些变量在现有环境变化下表现稳定”，但环境变化不足、城市数量太少或测量误差相同时可能返回空集或虚假稳定。[1]

### 3.3 训练：IRM / 风险方差惩罚（推荐首版）

在现有 BPNN、CarbonGCN 或 OpenCarbon 编码器后保持共享预测头，最小化：

`L = mean_e R_e + λ_var Var_e(R_e) + λ_inv Ω_IRM + λ_contrast L_contrastive`

其中 `e` 是源城市，后续可细化为城市×季节或城市×气候带。严格 IRM 优化不稳定时，可先用风险方差（VREx 风格）或 GroupDRO 作为工程代理，再把 IRM 作为对照。IRM 的理论依赖足够多样的真实环境；将同城数据随机分组不提供这种证据。[2]

### 3.4 已知偏移：anchor regression（强 baseline）

把气候带、年份、能源结构、区域电网或政策期作为 anchor，对“残差仍可由 anchor 预测”施加惩罚，并绘制误差—鲁棒性随惩罚强度的曲线。城市 ID 可以做粗粒度 anchor，但它混合过多机制，解释性较弱。[3]

### 3.5 时空结构：因果启发的动态图模块（第二阶段）

把边拆成三类：地理邻接、功能相似和可能的传播边；共享稳定边模块，对城市特异边或环境门控单独适配。CaST 表明可以用含时间环境、空间上下文、历史观测和未来结果的 SCM 组织时空图预测，但这类预测实验本身不能证明政策效应。[10]

### 3.6 政策层：transportability + 准实验（有政策问题时才启用）

先用 selection diagram 判断源城效应是否能由源城数据和目标城观测量迁移；对具体政策使用 synthetic control、event study、阈值或分期推广设计。若图和数据不能识别目标量，应明确报告“不可识别”，而不是让表示学习输出一个貌似精确的反事实。[4][12]

**置信度：Medium-High。** 方法本身证据强，组合用于跨城市碳排放仍属于待实证验证的研究设计。[1][2][3][4][10]

**反方解释：** 端到端预训练、普通 domain adaptation 或容量匹配的 ERM 可能胜过因果方法；必须以错误 DAG、随机环境和等参数量模型作为负对照，区分因果结构收益与额外容量/正则化收益。

## 4. 面向当前 Carbon 仓库的可实施架构

当前仓库已有 `cross_city` 留城 manifest，输入包括 17 类 POI、MODIS、VIIRS、5 类气象和月份，模型含 BPNN、CarbonGCN 和 OpenCarbon。建议按四步实施：

### Phase A：不改数据的最小可行版本

1. 新增模型 ID，例如 `opencarbon_invariant`，不改现有模型 ID。
2. 保留 POI、remote、environment 三编码器；增加机制头：`activity_proxy`、`weather_response`、`land_use`，最后由共享排放头融合。
3. 训练 batch 必须包含多个源城市，并按城市计算 `R_e`；加入风险方差或 GroupDRO，随后比较 IRM penalty。
4. 禁止把目标城用于标准化、早停、环境聚类、图构建或超参数选择。
5. 输出每城风险、最差城市风险、风险方差和残差—环境可预测性，而不仅是宏平均 MAE。

### Phase B：处理代理捷径

同期 VIIRS 夜光可能是排放活动代理，也可能与排放标签共同生成，导致模型学到难迁移的近标签捷径。至少运行：全特征、去 VIIRS、只用滞后 VIIRS、去月份、去 POI、去邻域、随机邻域六组消融。POI 与 MODIS 也要检查平台覆盖和传感器测量机制是否跨城改变。[5][9]

### Phase C：补充真正的机制变量

按城市×月/年加入人口、GDP/收入、产业产出、建筑面积、道路与公共交通、发电结构、电网碳强度、燃料价格、政策实施时间。将结果拆成工业、建筑、交通和电力相关排放，或至少分别保存 Scope 1/2。这样才能把 `Activity × EF` 物理模块和代理观测模块分开。[6][7]

### Phase D：目标城适配

分别报告：zero-shot（目标城完全无标签）、few-shot（少量早期月份校准）、full-target supervision。Few-shot 只更新城市特异门控、归一化或排放因子模块，共享机制冻结。三种协议不可混称。

**置信度：High（Phase A/B 可实现），Medium（Phase C/D 取决于新增数据）。** 当前结构和数据字段已在本地仓库中核对；因果解释边界由官方核算框架支持。[6][7]

**反方解释：** 当前只有少量城市时，IRM/ICP 的环境多样性可能不足；更现实的首个收益可能来自按城市分组训练和严格防泄漏，而非恢复了因果机制。[1][2]

## 5. 实验协议与判定标准

采用嵌套协议：外层每次完全留出一个城市；内层只用源城市按城市分组调参；时间上训练过去、测试未来。所有归一化和图构建只拟合训练窗。至少比较 ERM-LightGBM/BPNN/CarbonGCN/OpenCarbon、GroupDRO/风险方差、IRM、anchor 以及目标城 few-shot。

除 `log_mae/log_rmse/log_r2` 外，增加：

- 城市宏平均与 worst-city MAE；
- 跨城市风险标准差；
- 残差预测城市/气候带的准确率（越接近不可分越好，但不能单独证明因果）；
- 极端气象、产业型/服务型城市、不同电网区域的分层误差；
- zero-shot 与 few-shot 的样本效率曲线；
- 错误 DAG、随机环境、随机邻接的负对照。

若研究政策效应，另建协议而非复用预测指标：检查政策前趋势与预拟合，使用伪政策日期、伪处理城市和理论上不受政策影响的部门/时段作负对照。负对照显著意味着残余混杂或分析错误警报，但负对照不显著也不能证明无偏。[11][12]

建议将“成功”预注册为：相对容量匹配 ERM，在多数留城 fold 上改善、worst-city 风险下降、未来时段不退化，同时错误环境/错误图不产生同等收益。若仅平均 MAE 小幅提升但最差城市或时间外推恶化，不应宣称因果迁移成功。

**置信度：High。** 这是可证伪且防泄漏的预测评估；政策部分则依赖具体干预是否满足识别条件。[11][12]

**反方解释：** 城市之间可能共享电网、天气系统、政策扩散或供应链，留一城市并不保证独立环境；需要按区域联留、时间封锁和溢出敏感性分析进一步压力测试。

## 6. 核心争议与反向审查

1. **稳定不等于因果。** 某变量可能因环境变化不足而稳定；ICP/IRM 结果必须结合领域图、伪环境和干预测试解释。[1][2]
2. **预测不等于政策识别。** 时空因果模块降低 MAE，仍不能取代 treatment、调整集和准实验反事实。[4][10][12]
3. **代理可能是捷径。** 夜光和 POI 可提高当期预测，却可能随城市测量系统变化而失效；滞后和移除消融是必要条件。[5][9]
4. **物理机制也非完全稳定。** `Activity × EF` 结构稳定，但排放因子、电网边界和燃料质量会变化；这些应是可校准模块而非全局常数。[6][7]
5. **小城市样本限制因果泛化。** 城市数量少时，最强的结论只能是“在给定城市集合中更稳健”，不能外推到任意新城市。[1][2]

## 7. 推荐路线图

优先顺序如下：

1. 先实现按城市分组的 ERM、风险方差/GroupDRO 和 anchor baseline。
2. 加入严格 zero-shot 留城与时间外推报告，修复所有目标城统计泄漏。
3. 做 VIIRS/POI/月份/邻域及错误环境负对照。
4. 用 ICP 风格残差不变性作为审计，而非先当作因果发现真值。
5. 补齐能源、产业、交通、人口和政策数据后，再实现部门化 SCM 与模块适配。
6. 只有遇到明确政策问题，才单独建立 transportability + synthetic control/event-study 分支。

一句话的模型定义是：**“以 SCM 规定模块，以城市环境风险稳定训练共享机制，以少量目标数据校准变化模块，以独立准实验识别政策效应。”**

## 局限性

本研究未发现一个已经在多城市碳排放上同时验证完整 SCM、跨城时空图迁移和政策因果识别的成熟统一基准；部分架构证据来自通用因果迁移或时空预测领域。较早的核算与因果理论来源并非因过时而失效，但具体城市数据、能源结构和政策状态仍需按 2026 年可用数据更新。报告提出的是可检验研究设计，不是已证明优于当前 OpenCarbon 的结果。

## 参考文献

[1] Peters, Bühlmann, Meinshausen. “Causal inference using invariant prediction: identification and confidence intervals.” Source-Type: academic. As Of: 2016-11. https://arxiv.org/abs/1501.01332

[2] Arjovsky et al. “Invariant Risk Minimization.” Source-Type: academic. As Of: 2020-03. https://arxiv.org/abs/1907.02893

[3] Rothenhäusler et al. “Anchor regression: heterogeneous data meet causality.” Source-Type: academic. As Of: 2021-08. https://arxiv.org/abs/1801.06229

[4] Pearl and Bareinboim. “Transportability of Causal and Statistical Relations: A Formal Approach.” Source-Type: academic. As Of: 2011-08. https://ftp.cs.ucla.edu/pub/stat_ser/r372-a.pdf

[5] IPCC. “Climate Change 2022: Mitigation of Climate Change, Chapter 8.” Source-Type: official. As Of: 2022. https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-8/

[6] GHG Protocol, C40, ICLEI. “Global Protocol for Community-Scale Greenhouse Gas Emission Inventories (GPC 1.1).” Source-Type: official. As Of: 2021. https://ghgprotocol.org/ghg-protocol-cities

[7] IPCC. “2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 2: Energy.” Source-Type: official. As Of: 2006. https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol2.html

[8] Baiocchi et al. “Global typology of urban energy use and potentials for an urbanization mitigation wedge.” Source-Type: academic. As Of: 2015-01. https://doi.org/10.1073/pnas.1315545112

[9] Creutzig et al. “Infrastructure Shapes Differences in the Carbon Intensities of Chinese Cities.” Source-Type: academic. As Of: 2018-04. https://doi.org/10.1021/acs.est.7b05654

[10] Wang et al. “Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment.” Source-Type: academic. As Of: 2023-09. https://arxiv.org/abs/2309.13378

[11] Lipsitch, Tchetgen Tchetgen, Cohen. “Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies.” Source-Type: academic. As Of: 2010-06. https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408/

[12] European Commission. “Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects.” Source-Type: official. As Of: 2019-05. https://op.europa.eu/en/publication-detail/-/publication/6e30553e-76f7-11e9-9f05-01aa75ed71a1/language-en
