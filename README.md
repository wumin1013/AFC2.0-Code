# 工艺信息提取与稳态区间划分

本项目读取铣削工艺信息和实测数据，完成过程域区间划分、负载分析、图形显示和结果导出。桌面界面基于 Tkinter 与 Matplotlib。

## 两种运行模式

### 源码研究模式

源码研究模式以主页面承载机理辨识、反解、预测、profile 和手动数据导入；独立 PIT 页面仅只读显示 `ap / ae / F / MRR` 的行程域与指令域图。项目使用 Conda 环境 `AFC`：

```powershell
conda activate AFC
python src/interval_analysis/main.py
```

运行依赖记录在 `requirements.txt`。本机 `AFC` 环境可能与该文件的历史版本锁定存在差异，发布构建会在产物中记录实际参与构建的版本，不会自动安装或升级依赖。

主代码点击“导入 SampleData”时，应在同一个多选窗口内同时选择一对同目录、同主文件名的 CSV 和 TXT，例如 `SampleData_1.csv` 与 `SampleData_1.txt`；只选一个文件或文件名不配对时不会开始加载。

### 当前发布模式（AFC2.0.2alpha）

发布模式使用独立入口，启动时优先自动读取文件名逐字符完全匹配的 `SampleData.csv` 与 `SampleData.txt`；CSV/TXT 支持 UTF-8/BOM、UTF-16/32、GB18030/GBK、Big5、Shift-JIS、EUC-JP/KR 和常见 Windows ANSI 编码，并避免把可被 EUC-KR 严格解码的 GB18030 中文程序名误判为韩文。文件对缺失或无效时，回退识别 EXE 同目录唯一的实验实测通道导出 CSV。错误大小写、增加后缀或改名后的文件不作为 SampleData。导入工艺信息并完成六态划分后，以固定 `P_idle=250 W` 和编程 F 反解全局 `Kc/Ke`，再优先使用 EXE 同目录 `iipinc.txt` 的指令进给计算预测负载；缺失、无效或未覆盖的指令进给逐行回退工艺信息中的编程进给。反解始终执行，预测曲线默认隐藏，可通过图形工具栏的“预测负载”勾选框显示。右侧“运行结果”只显示处理、实际负载和预测负载三项通俗状态，启动与导入过程中合并重复绘图刷新。六态解码为单原子候选使用等价的标量快速路径，长工艺文件不再为每个状态创建大量单元素数组。发布包仍不包含 PIT / SMIF、profile、scikit-learn 或 SciPy。

实测数据与工艺信息完成六态划分及采样映射后，按当前程序/刀具的切削稳态区间显示有限实测点平均功率；理想功率始终等于该稳态平均功率乘以当前优化倍率。没有可用切削稳态区间时，两项功率显示为 `-`。`SampleData.rg` 保持原有字段、顺序和 UTF-8 编码，区间集合同时包含六态划分中的空载和切削稳态区间。

主研究版和发布版启动时均沿用旧版实际负载工具的上方布局：以 Afo 主窗口 `1080×630+0+0` 为基准取 90%，初始窗口为 `972×567+0+0`，不自动最大化；仍可通过系统窗口按钮手动最大化或调整大小。

默认紧凑布局统一优先把至少 68% 的主体高度留给图表。主研究版启动时整块机理辨识详情默认折叠，“展开详情”入口保留在公共操作栏；发布版沿用相同的高度预算。当前 `972×567` 隐藏实测中，两版图表区域分别约占 69.5% 和 72.3%，不会再由上方控件挤压主要绘图区。

图表嵌入较矮窗口时，横轴标题和顶部程序行号轴按最终文字边界动态补足边距，避免固定百分比边距在缩放后把“时间 (ms)”或长程序行号裁到画布外。导入工艺信息后直接在同一实际负载 Figure、Canvas 和曲线上叠加区间，映射范围变化时同步刷新横轴、顶部行号轴和“功率 (W)”语义，不再重建画布或闪烁。

实验实测 CSV 与 `iipinc.txt` 均使用批量/轻量解析路径；`iipinc.txt` 不再为每个物理行创建小型 NumPy 数组。真实 53 MB、641,094 行文件的冷解析约由 5.1 秒降至 1.9 秒，同时继续保持“预测曲线默认隐藏、后台预测已经生成”的行为。

## 发布产物与版本规则

当前发布版本由 `src/interval_analysis/release_metadata.py` 中的 `RELEASE_VERSION` 唯一定义，当前值为 `AFC2.0.2alpha`。以后升级为 `AFC2.0.2beta`、`AFC2.0.3alpha` 等版本时，只修改这个常量；EXE 文件名、窗口标题、发布说明、构建清单、版本专属调试目录和最小 ZIP 名都会随之更新。调试目录中的阶段名首字母大写，例如 `alpha` 对应 `Alpha-debug`、`beta` 对应 `Beta-debug`。

每个版本生成一套独立且长期保留的调试目录和最小发布包。当前 Alpha 版本为：

```text
../历史归档/发布版本/
├─ AFC2.0.2Alpha-debug/
│  ├─ AFC2.0.2alpha.exe
│  ├─ _internal/
│  ├─ 发布说明.txt
│  ├─ build-manifest.json
│  ├─ SHA256SUMS.txt
│  └─ 调试输入、运行结果、config/、data/ 等调试现场
└─ AFC2.0.2alpha.zip
```

版本专属调试目录采用 PyInstaller 6.12.0 的 `onedir`、`windowed` 模式。该目录专门用于持续调试，允许并需要保留其中的输入、结果、本机配置和运行状态；重新构建同一版本时，脚本只替换 EXE、`_internal/`、发布说明、构建清单和校验清单，其余调试现场会完整备份、恢复并逐文件复核 SHA-256。不同版本的调试目录和 ZIP 彼此独立且长期保留，例如以后构建 `AFC2.0.2Beta-debug/` 与 `AFC2.0.2beta.zip` 时，不会迁移、覆盖或删除 Alpha 的两个发布产物。

最小 ZIP 是唯一用于交付的干净发布包。它只包含以下可运行载荷，不包含调试目录中的 CSV/TXT 输入、结果文件、`config/`、`data/`、源码、测试或缓存：

```text
AFC2.0.2alpha.zip
├─ AFC2.0.2alpha.exe
├─ _internal/
├─ 发布说明.txt
├─ build-manifest.json
└─ SHA256SUMS.txt
```

必须交付完整 ZIP，不能只发送 EXE。解压后，外部软件应在程序启动前将以下运行输入放到 EXE 同目录：

```text
SampleData.csv
SampleData.txt
# 或仅放置一个实验实测通道导出 CSV（SampleData 文件对优先）
iipinc.txt        # 可选；缺失或无效时使用工艺信息中的 F
```

`iipinc.txt` 第一列是一基物理行号；程序自动识别单行格式和相邻双行重复格式：单行逐行使用且第八列乘以 `0.6`，严格双行重复时每对只保留一行且第八列乘以 `0.6×10^-4`，并允许末尾存在与数据行数一致的 `Total periods <数量>` 汇总行。文件有效但个别行缺失时只回退该行的编程 F。保存结果时，发布版 `ProcessInfo.csv` 的 F 会按同一行内中点规则替换为指令进给，MRR 随写出的 F 重新计算；未覆盖点保留原编程 F，并在保存提示中报告。程序不轮询或热重载，绘图切换会复用按文件大小和修改时间建立的内存缓存。发布目录必须可写，不建议放入 `Program Files`。发布 EXE 生成的 `SampleData.rg`、`ProcessInfo.csv`、图片、表格和错误日志直接写到 EXE 同目录。源码研究模式的桌面默认导出和启动日志写入 `data/runtime/`。

工艺信息通常导入 `AfoMilling_origin_trace_*.txt`；程序保存出的 `ProcessInfo.csv` 也可重新导入。解析器按表头区分两种布局：AfoMilling 可带首列“序号”，ProcessInfo 不带该列；两者的 N 都按一基程序物理行号读取，内部统一转换为零基行号。

## 构建发布候选包

构建工具版本记录在 `requirements-build.txt`。脚本要求使用 `AFC` 环境中的 Python，并验证 PyInstaller 版本为 6.12.0。

在项目根目录运行：

```powershell
& .\scripts\build_release.ps1
```

也可以从任意当前目录调用，并显式指定解释器：

```powershell
& 'C:\path\to\project\scripts\build_release.ps1' `
  -PythonPath 'C:\path\to\AFC\python.exe'
```

默认生成 `../历史归档/发布版本/AFC2.0.2Alpha-debug/` 和 `../历史归档/发布版本/AFC2.0.2alpha.zip`。脚本先构建并验证纯净候选目录，再从该候选目录生成 ZIP、解压 ZIP 并再次完整验证；两者均成功后才替换当前版本的调试目录、恢复该版本的调试现场并替换同版本 ZIP。其他版本的 `*-debug/` 与 ZIP 保持独立并长期保留。构建验证通过后会自动删除 `.build-AFC2.0.2alpha` 中间目录；如需在构建前清理失败构建留下的缓存，可增加 `-CleanBuild`。脚本的清理范围限于当前版本的发布产物和专用构建中间目录。

```powershell
& .\scripts\build_release.ps1 -CleanBuild
```

如需在 `../历史归档/发布版本/` 之外额外留存正式归档，必须显式提供完整目标目录，且要求 Git 工作区干净；脚本会从已经验证的最小 ZIP 解压归档，不会带入任何版本调试目录的数据：

```powershell
& .\scripts\build_release.ps1 `
  -ArchiveDestination 'C:\path\to\历史归档\AFC2.0.2alpha'
```

归档流程先解压到目标父目录中的临时目录，重新校验 SHA-256 和 denylist，再在同一父目录内改名。最终目标已存在时立即停止，绝不覆盖。二进制发布包不提交 Git。

## 验证现有发布目录

构建脚本会自动运行验证。也可单独执行：

```powershell
& .\scripts\verification\verify_release_build.ps1 `
  -ReleaseDirectory 'C:\temp\AFC2.0.2alpha-clean' `
  -ExpectedVersion 'AFC2.0.2alpha' `
  -PythonPath 'C:\path\to\AFC\python.exe'
```

其中 `ReleaseDirectory` 应指向最小 ZIP 解压得到的空白发布目录。版本专属调试目录包含额外调试现场，不用于纯净性验证；对它重点验证 EXE 启动和该版本调试数据是否保留。构建脚本会自动完成候选目录与最小 ZIP 解压目录的两轮完整验证。

验证内容包括：

- EXE、`_internal`、说明、构建清单和 SHA-256 清单完整；
- 每个发布文件均纳入 SHA-256 校验；
- NumPy/Pandas、ctypes、Pillow 和 Tk 所需的 Conda 原生运行库完整，Tcl/Tk DLL 与数据脚本版本一致，并在临时副本及最终中文目录中分别执行无窗口发布应用初始化；
- PyInstaller CArchive/PYZ 包含轻量 `prediction_support`、`release_prediction`，同时不含研究应用、PIT、scikit-learn、SciPy、profile、项目源码或用户数据；
- 不含当前开发机项目路径、用户目录和 Python 环境绝对路径。

## 测试与检查

在 `AFC` 环境中运行：

```powershell
python -m compileall -q src
python -m unittest discover -s tests -p 'test_*.py'
python scripts/verification/verify_segmentation_cases.py --help
```

正式发布还需要在无 Python/Conda 的 Windows 10/11 x64 环境完成启动、读取、划分、绘图与保存冒烟测试，并检查 1366×768、1920×1080 及 100%/125%/150% DPI 布局。未完成这些人工验收时，只能视为候选包。

## 项目目录

```text
src/                   按职责组织的源码与应用入口
packaging/             PyInstaller spec 与发布说明模板
config/                本机应用配置
data/sample/           静态样例工艺数据
data/runtime/          本机运行状态
profiles/              源码研究模式案例配置
scripts/verification/  自动与人工验证脚本
tests/                 自动化回归测试
docs/                  需求与技术文档
archive/               不参与当前运行的历史源码
output/                源码研究模式的程序输出和临时结果
../历史归档/发布版本/      各版本独立调试目录与当前最小 ZIP（不纳入 Git）
```

`config/app_config.json`、`data/runtime/`、`profiles/cache/`、`output/` 和 `../历史归档/发布版本/` 属于本机可变内容，均不纳入版本管理。发布构建不会复制这些目录，也不会复制 `data/sample/` 或 `profiles/`。

## 负载预测与主轴功率

负载预测使用 NumPy 2 系列，依赖文件按已验收的研究环境固定版本。在 Windows 的 Anaconda Prompt 中启用供负载预测独立使用的 AFC 环境。

```bat
conda activate AFC
set MKL_THREADING_LAYER=SEQUENTIAL
python -m pip install -r requirements-load-prediction.txt
python scripts\load_prediction.py --help
set PYTHONPATH=src
python -m spindle_power.load_prediction.unified_cli --help
```

主轴功率预测集中在 `src/spindle_power/`，包含训练与研究入口 `load_prediction/` 和单文件模型加载与全行程推理 `spindle_runtime/`。论文 1 的实验批次保存在本地 `data/paper1/`，训练服务默认写入 `data/paper1/service/`。论文采用的研究初始权重及回执见独立论文仓库的 `models/`。

### 源码分工

| 功能 | 代码入口 |
| --- | --- |
| 区间划分、理想值和 RG | src/interval_analysis/ 保存区间应用、理想值、传统机理模型和 RG 导出。src/interval_analysis/segmentation/ 提供区间划分算法。 |
| 负载预测研究模型 | `src/spindle_power/load_prediction/` 保存物理主趋势、路径残差、训练、九折评价和研究预测入口。 |
| 主轴功率部署推理 | `src/spindle_power/spindle_runtime/` 加载单文件模型包并完成全工艺位置的主轴功率推理。 |

源码直接按功能放在 src/ 下。后续论文使用独立实现时，新增相应功能包。共用轨迹处理在 src/afc_common/。
