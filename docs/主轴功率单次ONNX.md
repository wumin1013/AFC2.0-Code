# 主轴功率单次 ONNX 推理

整段模型使用 `afc.spindle_sequence_onnx.1` 张量接口。一次 ONNX Runtime 调用完成分窗、接缝补推、基础功率和切削功率计算，并在指令序列中插入 M03 启动点。软件从工况、G 代码和工艺表生成请求张量，按返回顺序读取结果。

## 生成模型

在代码仓库根目录执行。已核对的转换环境为 Python 3.10、PyTorch 2.9.0、ONNX 1.17.0 和 ONNX Runtime 1.20.1 CPU。

```powershell
python -m pip install torch==2.9.0 numpy onnx==1.17.0 onnxruntime==1.20.1
python scripts/export_spindle_sequence.py build --training-pt "<原训练PT>" --output "<新路径>/SpindlePowerSequence_V1.0.pt"
python scripts/export_spindle_sequence.py export --pt "<整段PT>" --output "<新路径>/SpindlePowerSequence_V1.0.onnx"
```

`build` 接收带动态基础功率组件的 AFC 训练 PT。`export` 从整段 TorchScript PT 导出单文件 ONNX，检查 ONNX 图并用 CPU ONNX Runtime 对照 PT 的七项输出。需要用原训练模板继续训练时，可以恢复包内原始 PT。

```powershell
python scripts/export_spindle_sequence.py recover-training --pt "<整段PT>" --output "<新路径>/SpindlePowerPredictionModel_V1.0.pt"
```

## ONNX 张量接口

| 输入 | 类型与形状 | 含义 |
|---|---|---|
| `features` | float64 `[N,20]` | 原工艺点特征，沿用模型的 20 列顺序 |
| `feature_valid` | bool `[N]` | 原点是否具备有效的切削特征 |
| `source_geometry` | int64 `[N,5]` | 原点号、NC 物理行、执行次数、连续段号、运动类型 |
| `segment_mm` | float64 `[N]` | 原工艺点段内位置 |
| `query_path_mm` | float64 `[N]` | 用于推理的原点路径位置 |
| `command_rpm` | float64 `[N]` | 每个原工艺点的指令转速 |
| `ap_ae_mm` | float64 `[N,2]` | 原工艺点的切深、切宽 |
| `event_anchor` | int64 `[E]` | 每个 M03 所在的原点顺序索引，无锚点为 -1 |
| `event_line` | int64 `[E]` | M03 的 NC 物理行 |
| `event_rpm` | float64 `[E]` | M03 的指令转速 |
| `extrapolate` | bool `[1]` | 是否允许稳态基础功率外推 |

`N` 为原工艺点数。`E` 为 M03 事件数，无 M03 时三个事件输入长度为零。`T` 为模型返回的整段采样数。

| 输出 | 类型与形状 | 含义 |
|---|---|---|
| `source_power` | float64 `[N,5]` | 按原工艺点排列的五列功率 |
| `source_valid` | bool `[N]` | 原工艺点五列功率是否完整 |
| `trace_power` | float64 `[T,5]` | 模型已排序的完整指令域功率 |
| `trace_valid` | bool `[T]` | 指令域采样的五列功率是否完整 |
| `trace_positions` | int64 `[T,4]` | 类型 0 原点、1 启动，随后为原点顺序索引、NC 行、事件索引。无对应索引填 -1 |
| `trace_event_elapsed_s` | float64 `[T]` | 启动采样的事件内时间，原工艺点为 NaN |
| `event_status` | int64 `[E]` | 0 已接入，1 缺少锚点，2 超出标定域 |

功率列依次为总功率、下界、上界、基础功率、切削功率，单位为 W。曲线直接读取 `trace_power`，逐点结果读取 `source_power`。
