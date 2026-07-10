# 工艺信息提取与稳态区间划分

本项目用于读取铣削工艺信息和实测数据，完成空载功率处理、参数辨识、预测负载计算、稳态区间划分及结果导出。桌面界面基于 Tkinter 和 Matplotlib。

## 开发环境

项目使用 Conda 虚拟环境 `AFC`，当前环境路径为：

```text
D:\Enviroment\Anaconda3\envs\AFC
```

推荐在 PowerShell 中启动：

```powershell
conda activate AFC
python src/project/main.py
```

如果当前终端无法激活 Conda，可以直接调用环境解释器：

```powershell
& 'D:\Enviroment\Anaconda3\envs\AFC\python.exe' src/project/main.py
```

运行依赖记录在 `requirements.txt`。当前仓库不包含自动化测试或人工验证代码。

## 目录说明

```text
src/project/           主程序源码和入口
config/                本机应用配置
data/sample/           随项目保留的样例工艺数据
data/runtime/          程序运行状态
profiles/              正式 Kc/Ke 案例配置
scripts/verification/  预留的人工验证脚本目录（当前为空）
tests/                 预留的自动化测试目录（当前为空）
docs/                  技术和阶段文档
archive/               不参与运行的历史源码备份
output/                程序输出和调试结果
```

`config/app_config.json`、`data/runtime/`、`profiles/cache/` 和 `output/` 都是本机可变内容，已由 `.gitignore` 忽略。正式案例配置（例如 `profiles/OBQX.kcke`）不属于缓存。

## 配置、数据与输出

- 应用配置：`config/app_config.json`
- 运行状态：`data/runtime/ideal_store.json`
- 样例工艺数据：`data/sample/ProcessInfo.csv`
- 案例配置：`profiles/`
- 默认导出目录：`output/`

`app_config.json` 中保存的外部数据路径仍由用户选择，程序不会把它们改写为固定仓库路径。

## 常用检查

```powershell
& 'D:\Enviroment\Anaconda3\envs\AFC\python.exe' -c "import sys; sys.path.insert(0, 'src'); import project"
& 'D:\Enviroment\Anaconda3\envs\AFC\python.exe' -m compileall -q src
```

当前仓库没有 PyInstaller spec 或其他正式构建脚本；源码启动是现阶段受支持的启动方式。
