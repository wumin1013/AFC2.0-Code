# 工艺信息提取与稳态区间划分

本项目读取铣削工艺信息和实测数据，完成过程域区间划分、负载分析、图形显示和结果导出。桌面界面基于 Tkinter 与 Matplotlib。

## 两种运行模式

### 源码研究模式

源码研究模式以主页面承载机理辨识、反解、预测、profile 和手动数据导入；独立 PIT 页面仅只读显示 `ap / ae / F / MRR` 的行程域与指令域图。项目使用 Conda 环境 `AFC`：

```powershell
conda activate AFC
python src/project/main.py
```

运行依赖记录在 `requirements.txt`。本机 `AFC` 环境可能与该文件的历史版本锁定存在差异，发布构建会在产物中记录实际参与构建的版本，不会自动安装或升级依赖。

### AFC2.0.2alpha 发布模式

发布模式使用独立入口，启动时优先自动读取完整的 SampleData 文件对；文件对缺失或无效时，回退识别 EXE 同目录唯一的实验实测通道导出 CSV。导入工艺信息并完成六态划分后，以固定 `P_idle=250 W` 和编程 F 反解全局 `Kc/Ke`，再优先使用 EXE 同目录 `iipinc.txt` 的指令进给计算预测负载；缺失、无效或未覆盖的指令进给逐行回退工艺信息中的编程进给。反解始终执行，预测曲线默认隐藏，可通过图形工具栏的“预测负载”勾选框显示。右侧“运行结果”只显示处理、实际负载和预测负载三项通俗状态，启动与导入过程中合并重复绘图刷新。发布包仍不包含 PIT / SMIF、profile、scikit-learn 或 SciPy。

发布目录采用 PyInstaller 6.12.0 的 `onedir`、`windowed` 模式：

```text
AFC2.0.2alpha/
├─ AFC2.0.2alpha.exe
├─ _internal/
├─ 发布说明.txt
├─ build-manifest.json
└─ SHA256SUMS.txt
```

必须交付完整目录，不能只发送 EXE。外部软件应在程序启动前将以下运行输入放到 EXE 同目录：

```text
SampleData.csv
SampleData.txt
# 或仅放置一个实验实测通道导出 CSV（SampleData 文件对优先）
iipinc.txt        # 可选；缺失或无效时使用工艺信息中的 F
```

`iipinc.txt` 第一列是一基物理行号；程序自动识别单行格式和相邻双行重复格式：单行逐行使用且第八列乘以 `0.6`，严格双行重复时每对只保留一行且第八列乘以 `0.6×10^-4`，并允许末尾存在与数据行数一致的 `Total periods <数量>` 汇总行。文件有效但个别行缺失时只回退该行的编程 F。保存结果时，发布版 `ProcessInfo.csv` 的 F 会按同一行内中点规则替换为指令进给，MRR 随写出的 F 重新计算；未覆盖点保留原编程 F，并在保存提示中报告。程序不轮询或热重载，绘图切换会复用按文件大小和修改时间建立的内存缓存。发布目录必须可写，不建议放入 `Program Files`。发布 EXE 生成的 `SampleData.rg`、`ProcessInfo.csv`、图片、表格和错误日志直接写到 EXE 同目录，不创建 `output` 子目录；源码研究模式仍写项目 `output/`。

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

默认可交付候选包位于 `release_versions/AFC2.0.2alpha/`，不会执行正式归档。替换旧目录前，脚本会备份根目录现存的 CSV、`SampleData*.txt` 和 `iipinc.txt`，新包就位后按 SHA-256 原样恢复；这些运行文件不属于发布载荷。构建验证通过后会自动删除 `.build-AFC2.0.2alpha` 中间目录；如需在构建前清理失败构建留下的缓存，可增加 `-CleanBuild`。脚本只操作项目 `release_versions/` 下的专用目录，不会清理源码研究模式的 `output/`。

```powershell
& .\scripts\build_release.ps1 -CleanBuild
```

正式归档必须显式提供完整目标目录，且要求 Git 工作区干净；脚本会自动使用全量干净构建，不复用候选缓存：

```powershell
& .\scripts\build_release.ps1 `
  -ArchiveDestination 'C:\path\to\历史归档\AFC2.0.2alpha'
```

归档流程先复制到目标父目录中的临时目录，重新校验 SHA-256 和 denylist，再在同一父目录内改名。最终目标已存在时立即停止，绝不覆盖。二进制发布包不提交 Git。

## 验证现有发布目录

构建脚本会自动运行验证。也可单独执行：

```powershell
& .\scripts\verification\verify_release_build.ps1 `
  -ReleaseDirectory '.\release_versions\AFC2.0.2alpha' `
  -PythonPath 'C:\path\to\AFC\python.exe'
```

验证内容包括：

- EXE、`_internal`、说明、构建清单和 SHA-256 清单完整；
- 每个发布文件均纳入 SHA-256 校验；
- NumPy/Pandas 所需 MKL 原生运行库完整，并在临时副本中执行一次无窗口发布应用初始化；
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
src/project/           主程序源码与入口
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
release_versions/      本机多文件发布版本（不纳入 Git）
```

`config/app_config.json`、`data/runtime/`、`profiles/cache/`、`output/` 和 `release_versions/` 属于本机可变内容，均不纳入版本管理。发布构建不会复制这些目录，也不会复制 `data/sample/` 或 `profiles/`。
