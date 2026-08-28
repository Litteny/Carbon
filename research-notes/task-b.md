---
task_id: b
role: 城市碳排放与因果图研究员
status: complete
sources_found: 6
---

## Sources

[1] Climate Change 2022: Mitigation of Climate Change, Chapter 8 — Urban Systems and Other Settlements | https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-8/ | Source-Type: official | Accessibility: public | As Of: 2022 | Authority: 10/10
[2] Global Protocol for Community-Scale Greenhouse Gas Emission Inventories (GPC 1.1) | https://ghgprotocol.org/ghg-protocol-cities | Source-Type: official | Accessibility: public | As Of: 2021 | Authority: 9/10
[3] 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 2: Energy | https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol2.html | Source-Type: official | Accessibility: public | As Of: 2006 | Authority: 10/10
[4] Global typology of urban energy use and potentials for an urbanization mitigation wedge | https://doi.org/10.1073/pnas.1315545112 | Source-Type: academic | Accessibility: public | As Of: 2015-01 | Authority: 9/10
[5] Infrastructure Shapes Differences in the Carbon Intensities of Chinese Cities | https://doi.org/10.1021/acs.est.7b05654 | Source-Type: academic | Accessibility: public | As Of: 2018-04 | Authority: 9/10
[6] Causal effects of built environment characteristics on travel behaviour: a longitudinal approach | https://journals.open.tudelft.nl/ejtir/article/view/3165 | Source-Type: academic | Accessibility: public | As Of: 2016-09 | Authority: 8/10

## Findings

- IPCC 将人口规模、收入、城市化阶段和城市形态的相互作用列为城市温室气体排放驱动因素，并指出城市设计、建设、管理与供能会锁定行为、生活方式和未来排放，因此它们应构成城市迁移模型的上游因果骨架，而非无方向的特征集合。 [1]
- 推荐的结构因果图为 `环境/历史 E -> {人口经济 S, 城市形态 U, 能源禀赋 R, 政策 P}`，`{S,U,R,P,气象 W} -> {产业活动 I, 建筑能源 B, 交通活动 T, 电力/燃料结构 M}`，`{I,B,T,M} -> 部门能源活动 A -> 排放 Y`，并加入 `U -> T`、`W -> B`、`P -> {U,M,I,T}` 及必要的城市—年份滞后边。 [1][3][4][5][6]
- 变量角色应预先声明：地理区位、气候带、资源禀赋、历史基础设施和基期产业是环境变量/效应修饰因子；人口、收入、发展阶段、财政能力和基期政策偏好通常是暴露—排放关系的混杂因素；能源消费、出行行为、建筑负荷、产业产出和燃料/电力结构通常是介体。 [1][3][4][5][6]
- 遥感夜光、建成区、路网和 POI 密度应主要作为城市形态、活动强度与功能混合度的代理观测而非天然“原因”，因为把由经济活动和基础设施共同造成的夜光或 POI 纳入调整集，可能阻断中介路径或引入测量偏差。 [1][4][5]
- GPC 1.1 要求一致识别、计算和报告城市排放并与 2019 Refinement 对齐，因此结果 `Y` 必须先固定为清单边界明确的总量、人均量或强度，以及地域/购电/价值链范围；不同核算边界不能作为同一个标签混训。 [2]
- IPCC 能源清单方法以部门活动数据与排放因子组织核算，因而最可解释的结果方程是 `Y_{sector,fuel}=Activity_{sector,fuel} × EF_{fuel}` 后跨部门聚合，并把排放因子、供电碳强度和燃料结构显式建模，而不是仅从 POI 或夜光端到端回归总排放。 [3]
- 城市形态对排放的主要路径可拆成 `密度/混合用地/可达性 -> 出行距离与方式 -> 交通能源 -> Y` 和 `紧凑度/建筑形态 -> 建筑面积与冷热负荷 -> 建筑能源 -> Y`，而全球城市能源类型研究与中国城市研究均支持气候、经济、城市形态及基础设施共同解释跨城市能源或碳强度差异。 [1][4][5]
- 碰撞点候选包括同时受治理能力与排放压力影响的“是否实施低碳政策”、同时受城市吸引力与住房供给影响的“净迁入人口”、以及同时受经济活动与电力可得性影响的“夜光”；若分别估计政策、人口或形态效应，不应在缺乏时间顺序的情况下条件化这些变量。 [1][4][5]
- 跨城市迁移时应把城市身份、区域电网、气候带、统计口径和政策制度作为环境变量 `E`，采用共享机制与环境特异机制分层；优先迁移较稳定的物理模块 `活动×排放因子`，对 `E -> S/U/I/A` 的社会经济生成机制允许城市随机效应、域特异参数或不变性检验。 [1][2][3][4][5]
- 纵向研究发现住宅建成环境对汽车使用和出行态度的影响小但显著，同时明确指出横截面设计只能提供有限因果证据；因此城市迁移框架应使用滞后暴露、城市/年份固定效应、政策事件或工具变量识别，并用留一城市外推检验而不能把跨城市相关性直接解释为干预效应。 [6]

## Deep Read Notes

### Source [1]: IPCC AR6 WGIII Chapter 8
Key data: IPCC 估计消费口径城市排放由 2015 年约 25 GtCO2-eq（全球约 62%）升至 2020 年约 29 GtCO2-eq（67–72%），并明确列出人口、收入、城市化阶段和城市形态的交互驱动。
Key insight: 相同城市化水平可有很不同的人均排放，说明“城市化率”不是可跨环境稳定外推的单一原因，必须展开为形态、基础设施、能源和生活方式路径。
Useful for: DAG 上游变量、跨城市环境变量、核算边界与反方解释。

### Source [2]: GPC 1.1
Key data: GPC 由 WRI、C40 与 ICLEI 共同建立，目标包括形成基年清单、设定减排目标、跟踪绩效以及保证城市间一致透明和可聚合的报告。
Key insight: 因果预测首先要定义 estimand 的清单范围；边界变化本身会造成标签漂移，不能被误认为城市机制变化。
Useful for: 结果变量定义、标签协调、城市迁移前的数据审计。

### Source [4]: Global typology of urban energy use
Key data: 论文用全球城市数据建立城市能源使用类型，研究对象明确连接城市化特征、气候、经济和城市形态与能源需求及减排潜力。
Key insight: 城市间存在机制异质性，适合用环境分层或 mixture-of-mechanisms，而非假设所有城市共享单一回归系数。
Useful for: 城市分组、环境效应修饰、机制迁移设计。

### Source [5]: Infrastructure Shapes Differences in Chinese Cities
Key data: 论文在中国城市尺度研究基础设施如何塑造碳强度差异，发表于 Environmental Science & Technology，DOI 10.1021/acs.est.7b05654。
Key insight: 基础设施既是历史发展路径的结果又会锁定后续活动和能源使用，因此在不同因果问题下可能是暴露、介体或基期混杂，必须按时间索引处理。
Useful for: 基础设施/产业/能源子图以及中国城市迁移设定。

### Source [6]: Longitudinal built environment and travel behaviour
Key data: 两期纵向数据的 cross-lagged panel SEM 显示住宅建成环境对汽车使用和出行态度有小但显著影响，并观察到态度会适应建成环境。
Key insight: 社会人口、出行态度、居住自选择和反向因果使横截面建成环境系数不等于政策干预效应。
Useful for: `城市形态 -> 出行 -> 排放` 子图、混杂控制与识别策略。

## Gaps

- 未找到一个经多城市外部验证、同时覆盖形态—产业—能源—气象—政策—遥感/POI全链条的公认结构因果图；上述 DAG 是依据核算物理关系和分领域证据形成的可检验综合框架，而不是已被单篇论文完全识别的真图。
- 城市清单普遍存在统计口径、行政边界、能源平衡表、跨境电力和供应链排放不一致；POI、夜光和土地覆盖还存在平台覆盖、传感器饱和、尺度错配与时间漂移，迁移评估需把测量过程单独建模。
- 反方解释一：紧凑城市与低交通排放的关系可能主要来自收入、燃油价格、公共交通投资和居住自选择，而非密度本身；Source [6] 的纵向结果也表明效应较小。
- 反方解释二：高密度可降低交通能耗，却可能通过高层建筑材料、拥堵、热岛和制冷需求增加其他排放，因此总效应依赖气候、建筑技术和核算范围，不能从单一交通路径推断。
- 反方解释三：政策变量常由既有高排放或治理能力触发；仅控制“是否有政策”会产生碰撞偏差，政策效果应利用实施时间、资格阈值、分期推广或可信工具变量识别。
- 数据设计建议但尚缺本任务直接实证验证：至少构建城市×年份面板，所有暴露用 `t-1` 或更早值，分别保存地域 Scope 1、购电 Scope 2 与消费/价值链结果，并报告 DAG 假设、最小调整集、负对照、城市留出和机制不变性检验。
