# BiLSTM实际进给模块

更新：2026-10-09。独立BiLSTM、共享九折与全量训练编排、网页/CLI任务和版本化推理能力已实现。本次没有用不完整的真实90轮切削数据训练，没有由合成验收产生正式加工权重。完整契约见[AFC2.0负载预测统一实施与使用规范](AFC2.0负载预测统一实施与使用规范.md)。

## 输入与输出

BiLSTM从NC、明确机床几何配置及加工前已知条件的空间序列预测实际进给，单位mm/min。39维特征由FEED_FEATURE_COLUMNS固定，包含运动类型、当前位置与段终点XYZ、圆弧几何、编程F、倍率、段长、段内比例、方向、进出夹角、曲率及相邻段存在性/几何/进给。机床与设置版本、加工模式和采样间距参与适用性检查；程序ID、轮次、行号、执行实例及行程索引用于对齐和追溯，不进入学习特征。

默认空间间距0.5 mm，各NC段等分且保留端点。核心窗口256点，前后各64点上下文，核心损失不重复；短段使用变长遮罩。G0、暂停、缺失/无效区间和执行连续性变化形成断点，不跨缺口拼窗。训练和预测复用同一NC几何网格，不能用未来实测记录补齐边界。

输出保留程序物理行号、执行实例、全局及段内行程，是进给随路径的曲线。停留、换刀、主轴等待等缺少完整时间映射时，不虚构完整加工时间或总主轴功率。原feed.py自主规划保留为独立对照，结果不能标成BiLSTM预测或实测标签。编程、控制器指令、实测实际及预测实际进给分别保存，inc不属于必需输入。

## 标签、清单与冻结快照

标签优先使用明确实际进给通道，其次使用实际XYZ轴速度矢量模，均沿用mm/min。只有显式确认位置与时间同步且差分校验通过，才能使用实际XYZ位置差分乘60；指令位置不能生成实际进给标签。

按物理行、执行实例及段内真实行程对齐。无效位置、时间缺口、重复访问不明及驻留同位置多值保持无效，不跨空缺插值。平移使用已核对的position_offset_mm；未提供时须由完整连续访问通过段长/几何审核确定，不进行旋转或形状拟合。质量缺失保持未知。

正式清单使用afc.dataset_manifest.1，登记record_id、round_id、feature、dataset_role、NC/机床/实测路径、倍率/加工模式/设置版本、采集时间或审核事件顺序、刀具会话、质量证据及measurement_content_sha256。共享服务检查后准备固定snapshot_id，运行中新文件不改变快照。

开发数据development为90轮五类共450个有效位置。memory_validation、independent_test、target_adaptation、adaptation_validation和interval_calibration不进入开发归一化或早停。相同NC的独立重复实测允许跨开发折；同一采样内容复制、更名或改文件头仍只算一个来源。

## 正式九折与训练顺序

划分唯一来源为experiment_protocol.py的明确轮次表：九种S×T组合各留出10轮，其余80轮训练。同轮五类全部文件和窗口同折，不能按程序族重划正式协议。旧assignment程序留出只允许显式诊断，不能冒充正式结果。

每折仅用训练记录拟合归一化，依次训练并冻结BiLSTM、主趋势和路径残差。负载训练用实测实际进给，正式端到端验证用该折冻结BiLSTM预测值。进给使用独立MSE、梯度裁剪、验证早停和最佳权重；负载误差不回传进给网络，长记录不因点数更多获得更高权重。

候选按轮次/类别等权RMSE→MAE→高负载低估率依次选择；全90轮重新初始化，各阶段训练轮数取所选配置九折最佳轮数中位数向上取整。独立3代表程序×5轮记忆验证及3新零件程序×5轮冻结测试都不参与选型。独立BiLSTM准备不依赖基础功率，负载准备另需已验证的实测转速标签接口及基础模型/报告哈希。

## 运行入口

在项目根目录执行：

```powershell
conda activate AFC
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python scripts/load_prediction.py protocol
python scripts/load_prediction.py inspect config/inspect_request.json
python scripts/load_prediction.py prepare-feed config/prepare_feed_request.json
python scripts/load_prediction.py preview config/feed_job_request.json
```

检查与准备请求指定manifest。训练预览请求填写返回的snapshot_id、mode为feed及可选candidates。资料齐备、预览通过且本次任务要求正式训练后，用同一请求调用train；导入和预览不会自动优化网络。完整JSON字段见统一规范。

网页共用TrainingService/JobService：

```powershell
$env:PYTHONPATH = 'src'
python -m project.load_prediction.web_service --port 8765
```

访问http://127.0.0.1:8765，按清单、快照、候选、预览、启动、模型和预测操作。阶段、损失、进度、失败及取消状态持久保存；浏览器关闭不取消任务，服务重启时中断任务不能冒充成功。

独立兼容入口同样使用共享九折快照：

```powershell
python scripts/actual_feed.py check config/feed_manifest.json
python scripts/actual_feed.py train config/feed_manifest.json --snapshot SNAPSHOT_ID --execute
python scripts/actual_feed.py predict config/new_case.json --model data/paper1/service/models/MODEL_ID/feed.pt
```

check和不带--execute的train仅检查/预览。正式训练需共享冻结快照；--diagnostic-program-holdout才启用旧程序留出。新案例只含一个record，需nc、machine、program_id、feed_override、machining_mode和machine_settings_version，不读取未来measurement。输出predicted_feed.csv及prediction.json；未训练、损坏、版本或已知条件不兼容时拒绝预测。

## 与负载模型连接

负载物理输入统一为feed_input_mm_min：

| 用途 | 来源 | 标记 |
|---|---|---|
| 负载网络训练 | 对齐的实测实际进给 | measured_actual_feed |
| 正式九折验证及加工前预测 | 冻结BiLSTM输出 | bilstm_actual_feed |
| 误差传播诊断 | 编程/实测/预测进给分别运行同一冻结负载模型 | 逐项标明，记忆关闭或固定 |

prepare_predicted_load_features复用BiLSTM路径网格；prepare_load_samples使用实测标签，缺少标签时训练掩码关闭。predict_cutting_power默认要求预测进给，实测诊断需显式feed_mode="measured"。三种进给诊断与因果记忆对比是分别实施的实验。

负载包为afc.load.3，保存物理工况编码、输入定义、冻结版本和直接有序q0.05/q0.50/q0.95。旧规划进给或点预测权重不能直接解释为新模型，须重新准备样本和训练。BiLSTM输出进给点预测，下游区间来自负载分位数网络；不能把进给误差当作已经完成传播的风险结论。

## 验证与证据边界

软件验收覆盖特征、窗口掩码、分组/内容排重、冻结梯度、最佳权重、保存加载、版本及未训练拦截。合成CPU短程优化只验证链路，权重标记synthetic_test_only并隔离于正式模型列表。本次实际测试数量以交付报告为准。

旧独立模块曾记录95项测试和28个Python文件检查通过，早于统一实施，不能充当本次全部能力的验收数量。原始采样和NC只读，派生文本UTF-8无BOM。共用读取、几何和服务体现DRY；数据、进给、负载、任务各司其职体现SOLID；不增加无证据时间线或假区间体现KISS/YAGNI。

下一步补齐90轮、真实配置与标签审核后执行冻结协议。五轮空载事件指标通过不能替代实测转速扣除接口验收；这影响负载资格，但不阻止独立进给检查或其具备真实数据后的独立训练。
