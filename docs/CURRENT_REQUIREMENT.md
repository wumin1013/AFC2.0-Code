# 当前需求：初步规则 Semi-Markov 全行程六类区间划分 alpha（直接替换旧区间）

日期：2026-07-15

## 1. 目标

在不使用神经网络和实测加工信号的条件下，仅根据当前工艺信息表完成覆盖原始全行程的六类连续区间划分，为后续 TCN、Patch Transformer 和 Neural Semi-Markov 训练保留稳定接口。

新的六类 Semi-Markov 结果直接替换现有运行时稳态区间划分，成为 `current_interval_records`、`pred_power_intervals` 和 `pit_records` 的唯一权威区间来源。不得保留旧稳态算法与新算法并行运行、相互覆盖或由界面切换的双轨状态；旧 profile 中的历史区间边界可以继续作为兼容数据读取，但不得覆盖本次重新计算得到的六类边界。

同时精简主界面：移除顶层“数据分析表”Tab 页，只保留“工艺信息分析”和“PIT / SMIF”等当前主流程入口。这里的“移除”仅指不再创建和注册该 Tab，不删除底层数据分析函数、历史导出逻辑或已有数据结构。

六类标签固定为：

- `idle`：空载；
- `entry`：进刀；
- `steady`：稳态候选；
- `transition`：过渡；
- `nonsteady`：非稳态；
- `exit`：退刀。

六类数字编码固定为：

- `0 = idle`；
- `1 = entry`；
- `2 = steady`；
- `3 = transition`；
- `4 = nonsteady`；
- `5 = exit`。

每个连续区间必须同时保存字符串标签 `segment_type` 和整数编码 `state_code`。这里的“六类”表示允许产生任意数量的连续区间，每个区间属于六类之一，不要求最终结果恰好只有六条记录。

## 2. 当前输入

初步 alpha 只允许使用：

- 累计行程 s；
- 指令行号 `line_id`；
- 切深 `ap`；
- 切宽 `ae`；
- 原始编程进给 `F_program`。

现有代码字段映射：

- 使用 `path_start` / `path_end` 或已解析累计行程生成统一 s；
- `line_no_aligned` 用于内部连续行分组，同时保留 `line_no_raw` / `N_str` 用于显示和导出；
- `ap`、`ae` 直接使用现有字段；
- `feed_effective` 在当前版本中映射为 `F_program`；
- 原表没有 `point_id` 时按原始行顺序生成。

累计行程必须先进行有效性判断，再决定是否使用：

1. 输入累计行程有限、单调不减且具有有效正跨度时，优先使用输入累计行程；
2. 输入存在 `s(mm)` 列但全零、无有效跨度、严重回退或无法形成物理行程时，不得因为列存在就将其视为有效行程；
3. 无有效输入累计行程时，优先使用现有 G 代码直线/圆弧几何计算结果或已绑定 NC profile 的 `path_start` / `path_end`；
4. 同一指令包含多个工艺点时，将该指令物理长度确定性地分配到各点；`line_no_raw` 缺失时使用 `line_no_aligned` 进行连续指令分组；
5. 如果输入行程、G 代码几何和 NC profile 均无法提供有效物理行程，允许使用集中配置中的确定性顺序回退，但必须在诊断中标记为非物理行程，所有相关区间强制 `is_optimizable=False`、`review_required=True`。

硬性规则：如果输入表中存在累计行程列，但该列的有效数值全部为 `0`，必须忽略这组全零累计行程，采用程序根据 G 代码/NC 几何计算得到的行程；不得因为输入列存在而使最终 `s`、`path_start` 和 `path_end` 保持全零。只有程序也无法计算出有效物理行程时，才允许进入上述非物理顺序回退。

区间边界继续采用现有一基点号格式：

```text
起点行号.点号-终点行号.点号
```

内部没有原始行号时，可使用连续内部 `line_id` 生成边界标签，但必须同时保留原始点序号，保证逐点可追溯。

初步版本明确不使用：

- 主轴功率、切削功率或预测功率；
- 实际进给 `F_actual`；
- 尚未接入的指令进给 `F_cmd`；
- 振动、报警或其他机床采样。

## 3. 特征与原子段

需要计算：

- `MRR_program = ap * ae * F_program / 60`；
- `ap`、`ae`、`F_program`、`MRR_program` 对 s 的变化率；
- 局部均值、标准差、趋势和有效切削标志；
- 指令行切换、指令行物理长度、行内相对位置；
- 短指令行标志和单位行程换行密度。

在保留以下边界的前提下压缩连续相似点形成原子段：

- 指令行变化；
- 空载/有效切削变化；
- `ap`、`ae` 或 `F_program` 的显著变化；
- MRR趋势或风险标志变化。

所有阈值必须集中在可序列化配置中，不得散落硬编码在界面或导出函数里。持续长度使用毫米行程，不使用简单点数作为唯一依据。

## 4. 算法结构

新增独立模块 `src/project/segmentation/`，至少包含：

- `schemas.py`：标签、输入版本、配置和结果结构；
- `features.py`：标准输入、派生特征和原子段；
- `scorers.py`：`SegmentScorer` 接口与 `RuleSegmentScorer`；
- `semi_markov.py`：转移语法、持续长度、Viterbi和回溯；
- `pipeline.py`：完整运行流程、全行程展开和诊断；
- `__init__.py`：稳定公开接口。

核心调用形式：

```python
result = SegmentationPipeline(config).run(
    input_frame,
    scorer=RuleSegmentScorer(config),
)
```

`SegmentScorer` 与 `SemiMarkovDecoder` 必须分离。后续 TCN 或 Patch Transformer 只替换区间评分器，不得重写原子段、语法、持续长度、Viterbi、回溯和输出结构。

解码完成后，六类区间必须适配为现有运行时区间记录结构并通过统一状态写入入口更新 `current_interval_records`；`pred_power_intervals` 和 `pit_records` 仅作为兼容别名同步读取同一批记录。旧规则划分函数不得在随后绘图、profile 加载、参数刷新或导出过程中再次覆盖六类结果。

初步版本采用固定工艺语法：

```text
空载 → 进刀 → 过渡 / 稳态 / 非稳态
稳态 → 过渡 / 非稳态 / 退刀
过渡 → 稳态 / 非稳态 / 退刀
非稳态 → 过渡 / 稳态 / 退刀
退刀 → 空载
```

## 5. 输出

新的结构化输出统一放入 `output/segmentation/`。六类结果按本需求更新当前 PIT/区间运行状态，但不得覆盖现有 `SampleData.rg` 文件、目标值结果或其他旧导出文件；只有用户执行既有保存操作时，才允许继续按原格式写入 `SampleData.rg`。

### 5.1 逐点全行程表 `point_labels.csv`

至少包含：

- `point_id`；
- s、`line_id`、原始行号；
- `ap`、`ae`、`F_program`、`MRR_program`；
- `interval_id`、`segment_type`、`state_code`；
- 当前点对应的一基“行号.点号”标签；
- `is_optimizable`、`review_required`。

原始输入的每个点必须恰好对应一行结果。

### 5.2 连续区间表 `intervals.csv`

至少包含：

- 区间编号；
- 起止点、起止行程、起止指令行；
- 一基“起点行号.点号-终点行号.点号”兼容边界；
- 六类区间类型、`state_code`、物理长度、点数；
- `ap`、`ae`、F、MRR统计量；
- 最优规则分数、第二高分、`score_margin`；
- `confidence_type=rule_margin`、高/中/低规则置信等级；
- `is_optimizable`、`review_required`、`decision_reason`。

为后续兼容，预留但当前允许为空：

- `input_schema_version`、`scorer_type`、`model_version`；
- `class_confidence`；
- `boundary_confidence`；
- `optimization_confidence`。

### 5.3 可视化 `overview.png`

必须展示：

- 全行程 `ap`、`ae`、`F_program` 和 `MRR_program`；
- 六类区间彩色背景；
- 区间边界；
- 可优化稳态候选的单独标识。

### 5.4 诊断 `diagnostics.json`

至少包含：

- 输入点数、原子段数、最终区间数；
- 全行程覆盖率；
- 空洞数和重叠数；
- 非法状态转移数；
- 过短稳态数量；
- 使用的输入版本和完整配置；
- 行程来源、行程有效性和是否使用非物理顺序回退；
- 重复运行一致性摘要。

### 5.5 现有行号.点号及 `.rg` 兼容

- 新的六类全覆盖区间记录、`point_labels.csv` 和 `intervals.csv` 必须写入 `state_code=0..5`；
- 现有区间边界字段继续采用“行号.点号-行号.点号”，点号从 1 开始；
- `SampleData.rg` 的文件结构、分隔符、程序名、理想值和冒号后实测平均值语义保持不变，不得把冒号后的平均值静默改成 `0..5` 类型码；
- `SampleData.rg` 只导出满足安全条件、可优化的 `steady` 区间，其他五类虽然不写入优化范围，但仍必须存在于六类全覆盖结果中；
- 不改变 `ProcessDataPath.txt`、`ProcessInfo.csv` 和 i 代码的既有格式。

## 6. 安全规则

- 只有达到最短物理长度、远离进退刀保护边界且规则置信等级足够的 `steady` 才可设置 `is_optimizable=True`；
- 使用非物理顺序行程回退的 `steady` 不得设置为可优化；
- `idle`、`entry`、`transition`、`nonsteady` 和 `exit` 默认不优化；
- 低置信区间仍必须获得六类中的一个标签，以保证全行程覆盖，但强制 `is_optimizable=False` 和 `review_required=True`；
- 若解码无合法路径，必须安全回退到完整覆盖的保守结果，不得输出空洞。

## 7. 现有代码接入边界

允许修改：

- `src/project/academic_workbench.py`：增加标准输入适配、运行入口和最近一次完整 `SegmentationResult` 诊断/导出对象；该对象不得形成与 `current_interval_records` 并行的第二套业务区间状态；保留现有数据分析底层函数，但不再由顶层“数据分析表”Tab 调用其界面构建函数；
- `src/project/ui_bootstrap.py`：增加“全行程六类划分”入口和状态显示；删除顶层 `data_analysis_tab` 的创建、Notebook 注册以及 `create_data_analysis_tab()` 启动调用；
- `src/project/plot_support.py`：增加六类颜色和全行程绘图支持；
- `src/project/analysis_export.py`：增加独立六类结果导出；
- `src/project/processing_core.py`：增加输入累计行程有效性判断，并在全零或无有效跨度时回退到 G 代码几何或 NC profile 行程；
- `src/project/pit_model.py`：将六类全覆盖结果接入统一当前区间状态，禁止旧划分或历史 profile 边界覆盖新结果，并使下游只对安全稳态执行优化相关逻辑；
- `src/project/config_state.py`：将现有区间详情和计数显示适配为六类全行程结果；
- 新增 `src/project/segmentation/` 模块。

`academic_workbench.py` 可以保留最近一次完整 `SegmentationResult` 作为诊断和导出对象，但它不是与旧区间并行的第二套业务状态；业务区间唯一来源仍为写入 `current_interval_records` 的六类结果。

初步版本不得覆盖或改变：

- `src/project/academic_analysis.py` 中现有稳态/非稳态分析和目标值逻辑；
- `SampleData.rg` 导出结构及数值语义；
- 负载预测、目标功率、参考功率和 i 代码生成；
- 现有 profile、数据库、依赖和配置文件格式。
- `academic_workbench.py` 中的数据分析计算、导入和导出函数；本需求只移除界面 Tab，不做底层功能清理或重构。

现有依赖稳态区间的 Kc、目标值、优化和 `.rg` 导出消费者，必须从六类记录中显式筛选 `segment_type == "steady"` 且满足相应安全条件的区间，不得把 `entry`、`transition`、`nonsteady`、`exit` 或 `idle` 当作可优化稳态。

## 8. 初步版本不做

- 不增加表格分类模型；
- 不增加 PyTorch；
- 不实现 TCN、Transformer或Patch Transformer；
- 不实现 Neural Semi-Markov 损失；
- 不训练任何模型；
- 不使用实测主轴功率或实际进给；
- 不实现概率置信度校准；
- 不修改 `tests/` 或 `scripts/verification/`。

## 9. 验收标准

1. 在 Conda 环境 `AFC` 中可以导入并运行新分割模块。
2. 使用 `data/sample/ProcessInfo.csv` 时，逐点结果行数与有效原始输入点数一致。
3. 全行程覆盖率为100%，空洞数为0，重叠数为0。
4. 非法状态转移数为0。
5. 每个区间均可追溯到原始点、行程、指令行和一基“行号.点号”边界。
6. 只有符合安全条件的稳态候选能够标记为可优化。
7. 相同输入和配置重复运行得到完全一致的区间边界和标签。
8. 成功生成逐点表、区间表、全行程图和诊断文件。
9. 现有目标值、profile 和旧导出接口及格式不受影响；现有稳态/PIT 区间边界按本需求由六类全覆盖结果替换，其公共读取入口继续可用。
10. 实际运行导入检查、源码编译检查和当前样例端到端检查；无法执行或失败的检查必须如实报告。
11. 主 Notebook 中不再显示“数据分析表”Tab，应用启动和切换剩余 Tab 时无属性错误；底层数据分析代码及其历史数据保持不变。
12. `current_interval_records`、`pred_power_intervals` 和 `pit_records` 反映同一批六类全覆盖结果，旧稳态划分不会在刷新、绘图、profile 加载或导出时重新覆盖它们。
13. 所有区间同时包含 `segment_type` 和固定的 `state_code=0..5`，内部点号及导出点号从 1 开始。
14. 对 `data/sample/ProcessInfo.csv`，不能因为 `s(mm)` 全零而将全行程保持为零；应使用可用的 G 代码/NC 几何行程，并在诊断中记录实际行程来源。只有所有物理来源均不可用时才允许非物理顺序回退。
15. `SampleData.rg` 格式及冒号后实测平均值语义保持不变，且其中不得出现五类不可优化区间。

## 10. 后续接口

后续按以下顺序扩展，不属于初步版本范围：

1. 增加 `F_cmd` 输入版本；
2. 建立规则预标注 + 人工区间修正数据；
3. 训练 TCN 点级编码器；
4. 实现神经区间评分和 Neural Semi-Markov 损失；
5. 输出并校准类别、边界和优化置信度；
6. 完成 4 类受控轨迹、每类 3 个独立 NC 程序、每程序 3 轮加工，共 36 条开发数据；按完整程序进行三折交叉验证，每折使用 8 个程序的 24 条数据训练、4 个程序的 12 条数据验证；另用 1 个完全独立的实际零件程序加工 3 轮作为冻结测试集；
7. 训练 Patch Transformer 作为对比；
8. 实测主轴功率用于标签、测试和实际有效性验证；上一轮功率仅作为可选重复加工增强实验。
