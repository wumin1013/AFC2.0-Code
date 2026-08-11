# 工艺信息提取与稳态区间划分

本项目读取铣削工艺信息和实测数据，完成过程域区间划分、负载分析、图形显示和结果导出。桌面界面基于 Tkinter 与 Matplotlib。

## 两种运行模式

### 源码研究模式

源码研究模式保留机理辨识、PIT / SMIF、profile 和手动数据导入等研究能力。项目使用 Conda 环境 `AFC`：

```powershell
conda activate AFC
python src/project/main.py
```

运行依赖记录在 `requirements.txt`。本机 `AFC` 环境可能与该文件的历史版本锁定存在差异，发布构建会在产物中记录实际参与构建的版本，不会自动安装或升级依赖。

### AFC2.0.2alpha 发布模式

发布模式使用独立入口，只保留自动读取 SampleData、自动区间划分、实际负载与区间背景显示和结果保存。MRR 只负责区间划分；发布版不计算或显示预测负载，也不包含机理辨识、PIT / SMIF、profile、预测支持及其专用依赖。

发布目录采用 PyInstaller 6.12.0 的 `onedir`、`windowed` 模式：

```text
AFC2.0.2alpha/
├─ AFC2.0.2alpha.exe
├─ _internal/
├─ 发布说明.txt
├─ build-manifest.json
└─ SHA256SUMS.txt
```

必须交付完整目录，不能只发送 EXE。外部软件应在程序启动前将以下两个文件放到 EXE 同目录：

```text
SampleData.csv
SampleData.txt
```

程序只在启动阶段读取一次，不轮询或热重载。发布目录必须可写，不建议放入 `Program Files`。发布 EXE 生成的 `SampleData.rg`、`ProcessInfo.csv`、图片、表格和错误日志直接写到 EXE 同目录，不创建 `output` 子目录；源码研究模式仍写项目 `output/`。

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

默认可交付候选包位于 `release_versions/AFC2.0.2alpha/`，不会执行正式归档。构建验证通过后会自动删除 `.build-AFC2.0.2alpha` 中间目录；如需在构建前清理失败构建留下的缓存，可增加 `-CleanBuild`。脚本只操作项目 `release_versions/` 下的专用目录，不会清理源码研究模式的 `output/`。

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
- 文件树、ZIP、PyInstaller CArchive/PYZ 中不含研究应用、PIT、scikit-learn、SciPy、profile、项目源码或用户数据；
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
